#include "metal_engine.h"
#include "stream_format.h"
#include "../tokenizer.h"
#include "../../third_party/httplib.h"
#include "../../third_party/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

uint32_t parse_u32(const std::string& text, const char* option) {
    if (text.empty() || text[0]=='-') throw std::runtime_error(std::string("invalid ")+option);
    size_t used=0; unsigned long long value=std::stoull(text,&used,10);
    if(used!=text.size() || value>UINT32_MAX) throw std::runtime_error(std::string("invalid ")+option);
    return (uint32_t)value;
}

std::vector<uint32_t> to_u32(const std::vector<int>& ids) {
    std::vector<uint32_t> result; result.reserve(ids.size());
    for(int id:ids) { if(id<0) throw std::runtime_error("tokenizer returned a negative id"); result.push_back((uint32_t)id); }
    return result;
}

std::string text_content(const json& content) {
    if(content.is_string()) return content.get<std::string>();
    std::string out;
    if(content.is_array()) for(const auto& part:content) {
        if(!part.is_object()) continue;
        const std::string type=part.value("type","");
        if(type=="text" || type=="input_text" || type=="output_text") out+=part.value("text","");
    }
    return out;
}

std::vector<std::pair<std::string,std::string>> messages_from(const json& body) {
    std::vector<std::pair<std::string,std::string>> messages;
    if(body.contains("system")) {
        std::string system=text_content(body["system"]);
        if(!system.empty()) messages.push_back({"system",system});
    }
    if(!body.contains("messages") || !body["messages"].is_array())
        throw std::runtime_error("messages must be an array");
    for(const auto& message:body["messages"]) {
        if(!message.is_object()) continue;
        std::string role=message.value("role","");
        std::string content=message.contains("content")?text_content(message["content"]):"";
        if(!role.empty()) messages.push_back({role,content});
    }
    if(messages.empty()) throw std::runtime_error("messages are empty");
    return messages;
}

// OpenAI `stop` (string or array of strings) / Anthropic `stop_sequences`
// (array). Empty strings are dropped -- they would match at every position.
std::vector<std::string> parse_stops(const json& body, const char* key) {
    std::vector<std::string> out;
    if(!body.contains(key)) return out;
    const json& s=body[key];
    if(s.is_string()) { auto v=s.get<std::string>(); if(!v.empty()) out.push_back(std::move(v)); }
    else if(s.is_array()) for(const auto& e:s)
        if(e.is_string()) { auto v=e.get<std::string>(); if(!v.empty()) out.push_back(std::move(v)); }
    return out;
}

class PrefixCache {
  public:
    explicit PrefixCache(size_t capacity):capacity_(capacity){}

    bool restore(q27::MetalEngine& engine,const std::vector<uint32_t>& prompt,bool mtp,
                 size_t& matched,uint32_t& pending) {
        auto best=entries_.end(); size_t best_len=0;
        for(auto it=entries_.begin();it!=entries_.end();++it) {
            if(it->mtp!=mtp || it->tokens.size()>prompt.size() || it->tokens.size()<=best_len) continue;
            if(std::equal(it->tokens.begin(),it->tokens.end(),prompt.begin())) { best=it; best_len=it->tokens.size(); }
        }
        if(best==entries_.end()) return false;
        engine.restore_state(*best->snapshot); pending=best->pending; matched=best_len;
        entries_.splice(entries_.begin(),entries_,best);
        return true;
    }

    bool prepare_insert(const std::vector<uint32_t>& tokens,bool mtp) {
        if(!capacity_) return false;
        for(auto it=entries_.begin();it!=entries_.end();) {
            if(it->mtp==mtp && it->tokens==tokens) it=entries_.erase(it); else ++it;
        }
        // Release the LRU snapshot before allocating its replacement so peak
        // memory never exceeds the configured number of entries.
        while(entries_.size()>=capacity_) entries_.pop_back();
        return true;
    }

    void insert(std::vector<uint32_t> tokens,bool mtp,uint32_t pending,
                std::shared_ptr<q27::MetalEngine::Snapshot> snapshot) {
        entries_.push_front({std::move(tokens),mtp,pending,std::move(snapshot)});
    }
  private:
    struct Entry { std::vector<uint32_t> tokens; bool mtp; uint32_t pending; std::shared_ptr<q27::MetalEngine::Snapshot> snapshot; };
    size_t capacity_; std::list<Entry> entries_;
};

struct Runtime {
    q27::Tokenizer tokenizer;
    q27::MetalEngine engine;
    PrefixCache cache;
    uint32_t mtp_width;
    std::mutex mutex;

    Runtime(const std::string& model,const std::string& tok,uint32_t context,bool turbo3,
            uint32_t width,size_t cache_entries)
        :tokenizer(tok),engine(check_vocab(model),context,turbo3),cache(cache_entries),mtp_width(width){}

    std::string check_vocab(const std::string& model) {
        if(tokenizer.vocab_size()!=q27::MetalEngine::vocabulary_size())
            throw std::runtime_error("tokenizer/model vocabulary mismatch");
        return model;
    }

    // How generation ended. Stop == the model emitted EOS (finish_reason
    // "stop" / stop_reason "end_turn"); StopSequence == a requested stop
    // string matched ("stop" / "stop_sequence"); Length == max_tokens hit
    // ("length" / "max_tokens"); Cancelled == the client disconnected.
    enum class Finish { Length, Stop, StopSequence, Cancelled };
    struct Outcome {
        uint32_t prompt_tokens=0, output_tokens=0;
        size_t prefix_hit=0;
        Finish finish=Finish::Length;
        std::string stop_sequence; // set when finish==StopSequence
    };

    // Single generation core shared by streaming and non-streaming paths.
    // `emit(piece)` receives UTF-8-safe, stop-sequence-trimmed text as it is
    // produced and returns false when the client has gone away. The Metal
    // engine is serialized, so the mutex is held for the whole generation
    // exactly as the CUDA server holds its per-slot lease.
    Outcome run(const std::vector<uint32_t>& prompt,uint32_t count,
                const q27::SamplingParams& sampling,
                const std::vector<std::string>& stops,
                const std::function<bool(const std::string&)>& emit) {
        if(prompt.empty()) throw std::runtime_error("prompt is empty");
        std::lock_guard<std::mutex> lock(mutex);
        q27::validate_sampling(sampling);
        const bool mtp=mtp_width!=0 && sampling.temperature==0.0f;
        size_t hit=0; uint32_t pending=0;
        if(cache.restore(engine,prompt,mtp,hit,pending)) {
            if(hit<prompt.size()) {
                std::vector<uint32_t> suffix(prompt.begin()+hit,prompt.end());
                pending=engine.ingest_prompt(suffix,mtp,false);
            }
        } else pending=engine.ingest_prompt(prompt,mtp,true);
        // Cache the prompt state before generation mutates it. One entry costs
        // about 151 MiB for GDN state, so the default capacity is deliberately 1.
        if(cache.prepare_insert(prompt,mtp))
            cache.insert(prompt,mtp,pending,engine.capture_state());

        // token -> decode -> UTF-8 boundary gate -> stop-sequence holdback ->
        // emit. Returning false from the engine sink stops generation, whether
        // because the client left (client_gone) or a stop sequence completed
        // (stop_hit). The two are distinguished for finish-reason reporting.
        q27::Utf8Gate ugate;
        q27::StopBuffer stopbuf(stops);
        const uint32_t eos_id=(uint32_t)tokenizer.eos();
        bool client_gone=false, stop_hit=false;
        auto sink=[&](uint32_t token)->bool {
            bool stopped=false;
            std::string safe=stopbuf.feed(ugate.feed(tokenizer.decode_one((int)token)),stopped);
            if(!emit(safe)) { client_gone=true; return false; }
            if(stopped) { stop_hit=true; return false; }
            return true;
        };
        q27::MetalEngine::StopCause cause=q27::MetalEngine::StopCause::MaxTokens;
        uint32_t produced = sampling.temperature>0.0f
            ? engine.stream_sampled_from_logits(count,eos_id,sampling,sink,cause)
            : engine.stream_from_pending(pending,count,eos_id,mtp?mtp_width:0,sink,cause);
        // Flush the boundary gates: a dangling multi-byte tail becomes U+FFFD,
        // and any text held back as a possible stop-sequence prefix is real
        // output once the stream ends without matching.
        if(!client_gone && !stop_hit) {
            bool stopped=false;
            std::string tail=stopbuf.feed(ugate.flush(),stopped);
            tail+=stopbuf.flush();
            if(!tail.empty()) emit(tail);
            if(stopped) stop_hit=true;
        }

        Outcome out;
        out.prompt_tokens=(uint32_t)prompt.size();
        out.output_tokens=produced;
        out.prefix_hit=hit;
        if(client_gone) out.finish=Finish::Cancelled;
        else if(stop_hit) {
            out.finish=Finish::StopSequence;
            if(stopbuf.matched>=0 && stopbuf.matched<(int)stops.size())
                out.stop_sequence=stops[stopbuf.matched];
        } else if(cause==q27::MetalEngine::StopCause::Eos) out.finish=Finish::Stop;
        else out.finish=Finish::Length;
        return out;
    }
};

const char* openai_finish(Runtime::Finish f) {
    switch(f) {
        case Runtime::Finish::Length: return "length";
        default: return "stop"; // Stop (eos), StopSequence, and Cancelled
    }
}
const char* anthropic_stop(Runtime::Finish f) {
    switch(f) {
        case Runtime::Finish::Length: return "max_tokens";
        case Runtime::Finish::StopSequence: return "stop_sequence";
        default: return "end_turn"; // Stop (eos) and Cancelled
    }
}

q27::SamplingParams sampling_params(const json& body) {
    q27::SamplingParams result;
    result.temperature=body.value("temperature",0.0f);
    result.top_p=body.value("top_p",1.0f);
    result.top_k=body.value("top_k",0u);
    result.seed=body.value("seed",0ull);
    q27::validate_sampling(result);
    return result;
}

uint32_t max_tokens(const json& body) {
    long long value=body.value("max_tokens",body.value("max_output_tokens",128ll));
    if(value<0 || value>UINT32_MAX) throw std::runtime_error("invalid max_tokens");
    return (uint32_t)value;
}

bool wants_stream(const json& body) { return body.value("stream",false); }

long unix_now() { return (long)std::time(nullptr); }

void json_response(httplib::Response& response,const json& value,int status=200) {
    response.status=status;
    response.set_content(value.dump(-1,' ',false,json::error_handler_t::replace),"application/json");
}

} // namespace

int main(int argc,char** argv) {
    if(argc<3) {
        fprintf(stderr,"usage: %s model.q27 tokenizer.tok [--host 127.0.0.1] [--port 8080] [--ctx 8192] [--mtp 2..12] [--kv fp16|turbo3] [--prefix-entries N]\n",argv[0]);
        return 1;
    }
    try {
        std::string model=argv[1],tok=argv[2],host="127.0.0.1";
        uint32_t port=8080,context=8192,width=0,prefix_entries=1; bool turbo3=false;
        for(int i=3;i<argc;i++) {
            std::string arg=argv[i];
            if(arg=="--host" && i+1<argc) host=argv[++i];
            else if(arg=="--port" && i+1<argc) port=parse_u32(argv[++i],"--port");
            else if(arg=="--ctx" && i+1<argc) context=parse_u32(argv[++i],"--ctx");
            else if(arg=="--mtp" && i+1<argc) width=parse_u32(argv[++i],"--mtp");
            else if(arg=="--prefix-entries" && i+1<argc) prefix_entries=parse_u32(argv[++i],"--prefix-entries");
            else if(arg=="--kv" && i+1<argc) { std::string mode=argv[++i]; if(mode=="turbo3")turbo3=true; else if(mode!="fp16")throw std::runtime_error("invalid --kv"); }
            else throw std::runtime_error("unknown/incomplete argument: "+arg);
        }
        if(port>65535) throw std::runtime_error("port out of range");
        if(width && (width<2 || width>12)) throw std::runtime_error("MTP width must be 2..12");
        Runtime runtime(model,tok,context,turbo3,width,prefix_entries);
        httplib::Server server;
        server.Get("/health",[](const httplib::Request&,httplib::Response& r){json_response(r,{{"status","ok"}});});
        server.Get("/v1/models",[](const httplib::Request&,httplib::Response& r){json_response(r,{{"object","list"},{"data",json::array({{{"id","q27-metal"},{"object","model"}}})}});});

        auto guarded=[&](auto handler) {
            return [&,handler](const httplib::Request& request,httplib::Response& response) {
                try { handler(json::parse(request.body),response); }
                catch(const std::exception& e) { json_response(response,{{"error",{{"message",e.what()},{"type","invalid_request_error"}}}},400); }
            };
        };

        // ---- OpenAI /v1/completions and /v1/chat/completions ----
        // Shared responder: builds the non-streaming JSON body, or an SSE
        // stream of chat.completion.chunk / text_completion deltas terminated
        // by "data: [DONE]" -- matching src/server.cu's OpenAI event shapes.
        auto openai_respond=[&](const json& body,httplib::Response& r,bool chat,
                                std::vector<uint32_t> ids) {
            const uint32_t n=max_tokens(body);
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop");
            const bool stream=wants_stream(body);
            const char* obj=chat?"chat.completion":"text_completion";
            const char* objd=chat?"chat.completion.chunk":"text_completion";
            const std::string id=chat?"chatcmpl-metal":"cmpl-metal";
            const long created=unix_now();
            if(ids.empty()) throw std::runtime_error("prompt is empty");
            if(!stream) {
                std::string text;
                auto outcome=runtime.run(ids,n,sampling,stops,
                    [&](const std::string& piece){ text+=piece; return true; });
                json choice = chat
                    ? json{{"index",0},{"message",{{"role","assistant"},{"content",text}}},{"finish_reason",openai_finish(outcome.finish)}}
                    : json{{"index",0},{"text",text},{"finish_reason",openai_finish(outcome.finish)}};
                json_response(r,{{"id",id},{"object",obj},{"created",created},{"model","q27-metal"},
                    {"choices",json::array({choice})},
                    {"usage",{{"prompt_tokens",outcome.prompt_tokens},{"completion_tokens",outcome.output_tokens},
                              {"total_tokens",outcome.prompt_tokens+outcome.output_tokens}}},
                    {"q27_prefix_hit",outcome.prefix_hit}});
                return;
            }
            r.set_header("Content-Type","text/event-stream");
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,chat,objd,id,created](size_t,httplib::DataSink& sink)->bool {
                    try {
                        auto emit=[&](const std::string& piece)->bool {
                            std::string s=q27::sse_data(
                                q27::openai_stream_chunk(chat,id,objd,created,"q27-metal",piece));
                            return sink.write(s.data(),s.size());
                        };
                        auto outcome=runtime.run(ids,n,sampling,stops,emit);
                        // Terminal chunk with a real finish_reason before [DONE]
                        // (parity with server.cu security-review fix #7).
                        std::string fin=q27::sse_data(q27::openai_stream_final_chunk(
                            chat,id,objd,created,"q27-metal",openai_finish(outcome.finish)));
                        sink.write(fin.data(),fin.size());
                        std::string done=q27::sse_done(); sink.write(done.data(),done.size());
                    } catch(const std::exception& e) {
                        std::string s=q27::sse_data({{"error",{{"message",e.what()},{"type","invalid_request_error"}}}});
                        sink.write(s.data(),s.size());
                    }
                    sink.done();
                    return true;
                });
        };

        server.Post("/v1/completions",guarded([&](const json& body,httplib::Response& r){
            openai_respond(body,r,false,to_u32(runtime.tokenizer.encode(body.value("prompt",""))));
        }));
        server.Post("/v1/chat/completions",guarded([&](const json& body,httplib::Response& r){
            openai_respond(body,r,true,
                to_u32(runtime.tokenizer.apply_chat_template(messages_from(body),body.value("enable_thinking",true))));
        }));

        // ---- Anthropic /v1/messages ----
        // Non-streaming: a single text content block. Streaming: message_start,
        // content_block_start(text), content_block_delta(text_delta) per piece,
        // content_block_stop, message_delta(stop_reason,stop_sequence,usage),
        // message_stop -- the CUDA server's event sequence for a text-only reply.
        server.Post("/v1/messages",guarded([&](const json& body,httplib::Response& r){
            auto ids=to_u32(runtime.tokenizer.apply_chat_template(messages_from(body),true));
            const uint32_t n=max_tokens(body);
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop_sequences");
            const std::string mid="msg_metal";
            if(ids.empty()) throw std::runtime_error("prompt is empty");
            if(!wants_stream(body)) {
                std::string text;
                auto outcome=runtime.run(ids,n,sampling,stops,
                    [&](const std::string& piece){ text+=piece; return true; });
                json out={{"id",mid},{"type","message"},{"role","assistant"},{"model","q27-metal"},
                    {"content",json::array({{{"type","text"},{"text",text}}})},
                    {"stop_reason",anthropic_stop(outcome.finish)},
                    {"stop_sequence",outcome.finish==Runtime::Finish::StopSequence?json(outcome.stop_sequence):json(nullptr)},
                    {"usage",{{"input_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens}}},
                    {"q27_prefix_hit",outcome.prefix_hit}};
                json_response(r,out);
                return;
            }
            r.set_header("Content-Type","text/event-stream");
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,mid](size_t,httplib::DataSink& sink)->bool {
                    auto ev=[&](const char* name,const json& j){ std::string s=q27::sse_event(name,j); return sink.write(s.data(),s.size()); };
                    try {
                        json msg={{"id",mid},{"type","message"},{"role","assistant"},{"model","q27-metal"},
                            {"content",json::array()},{"stop_reason",nullptr},{"stop_sequence",nullptr},
                            {"usage",{{"input_tokens",(int)ids.size()},{"output_tokens",0}}}};
                        ev("message_start",{{"type","message_start"},{"message",msg}});
                        ev("content_block_start",{{"type","content_block_start"},{"index",0},
                            {"content_block",{{"type","text"},{"text",""}}}});
                        auto emit=[&](const std::string& piece)->bool {
                            if(piece.empty()) return true; // no empty text_delta (parity w/ server.cu)
                            return ev("content_block_delta",{{"type","content_block_delta"},{"index",0},
                                {"delta",{{"type","text_delta"},{"text",piece}}}});
                        };
                        auto outcome=runtime.run(ids,n,sampling,stops,emit);
                        ev("content_block_stop",{{"type","content_block_stop"},{"index",0}});
                        ev("message_delta",{{"type","message_delta"},
                            {"delta",{{"stop_reason",anthropic_stop(outcome.finish)},
                                      {"stop_sequence",outcome.finish==Runtime::Finish::StopSequence?json(outcome.stop_sequence):json(nullptr)}}},
                            {"usage",{{"output_tokens",outcome.output_tokens}}}});
                        ev("message_stop",{{"type","message_stop"}});
                    } catch(const std::exception& e) {
                        ev("error",{{"type","error"},{"error",{{"type","invalid_request_error"},{"message",e.what()}}}});
                    }
                    sink.done();
                    return true;
                });
        }));

        // ---- OpenAI Responses API (/v1/responses, Codex CLI) ----
        // Non-streaming: output_text plus a message/output_text item. Streaming
        // follows src/server.cu: response.created, output_item.added(message),
        // content_part.added, response.output_text.delta per piece,
        // output_text.done, content_part.done, output_item.done,
        // response.completed. Codex keys off the JSON `type` field.
        server.Post("/v1/responses",guarded([&](const json& body,httplib::Response& r){
            std::string input=body.contains("input")?text_content(body["input"]):"";
            auto ids=to_u32(runtime.tokenizer.encode(input));
            const uint32_t n=max_tokens(body);
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop");
            const std::string rid="resp_metal", mid="msg_metal";
            if(ids.empty()) throw std::runtime_error("input is empty");
            if(!wants_stream(body)) {
                std::string text;
                auto outcome=runtime.run(ids,n,sampling,stops,
                    [&](const std::string& piece){ text+=piece; return true; });
                json_response(r,{{"id",rid},{"object","response"},{"model","q27-metal"},{"status","completed"},
                    {"output_text",text},
                    {"output",json::array({{{"type","message"},{"id",mid},{"role","assistant"},{"status","completed"},
                        {"content",json::array({{{"type","output_text"},{"text",text},{"annotations",json::array()}}})}}})},
                    {"usage",{{"input_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                              {"total_tokens",outcome.prompt_tokens+outcome.output_tokens}}},
                    {"q27_prefix_hit",outcome.prefix_hit}});
                return;
            }
            r.set_header("Content-Type","text/event-stream");
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,rid,mid](size_t,httplib::DataSink& sink)->bool {
                    auto ev=[&](const json& j){ std::string s=q27::sse_event(j.value("type",std::string("x")),j); return sink.write(s.data(),s.size()); };
                    try {
                        ev({{"type","response.created"},{"response",{{"id",rid},{"object","response"},{"status","in_progress"}}}});
                        ev({{"type","response.output_item.added"},{"output_index",0},
                            {"item",{{"type","message"},{"id",mid},{"role","assistant"},{"status","in_progress"},{"content",json::array()}}}});
                        ev({{"type","response.content_part.added"},{"item_id",mid},{"output_index",0},{"content_index",0},
                            {"part",{{"type","output_text"},{"text",""},{"annotations",json::array()}}}});
                        std::string text;
                        auto emit=[&](const std::string& piece)->bool {
                            text+=piece;
                            if(piece.empty()) return true;
                            return ev({{"type","response.output_text.delta"},{"item_id",mid},
                                {"output_index",0},{"content_index",0},{"delta",piece}});
                        };
                        auto outcome=runtime.run(ids,n,sampling,stops,emit);
                        ev({{"type","response.output_text.done"},{"item_id",mid},{"output_index",0},{"content_index",0},{"text",text}});
                        ev({{"type","response.content_part.done"},{"item_id",mid},{"output_index",0},{"content_index",0},
                            {"part",{{"type","output_text"},{"text",text},{"annotations",json::array()}}}});
                        json item={{"type","message"},{"id",mid},{"role","assistant"},{"status","completed"},
                            {"content",json::array({{{"type","output_text"},{"text",text},{"annotations",json::array()}}})}};
                        ev({{"type","response.output_item.done"},{"output_index",0},{"item",item}});
                        ev({{"type","response.completed"},{"response",{{"id",rid},{"object","response"},{"status","completed"},
                            {"output",json::array({item})},
                            {"usage",{{"input_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                                      {"total_tokens",outcome.prompt_tokens+outcome.output_tokens}}}}}});
                    } catch(const std::exception& e) {
                        ev({{"type","error"},{"error",{{"type","invalid_request_error"},{"message",e.what()}}}});
                    }
                    sink.done();
                    return true;
                });
        }));

        fprintf(stderr,"q27 Metal server listening on http://%s:%u (ctx=%u, kv=%s, mtp=%u)\n",host.c_str(),port,context,turbo3?"turbo3":"fp16",width);
        if(!server.listen(host.c_str(),(int)port)) throw std::runtime_error("server listen failed");
        return 0;
    } catch(const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}
