#include "metal_engine.h"
#include "../tokenizer.h"
#include "../../third_party/httplib.h"
#include "../../third_party/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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

std::vector<int> to_int(const std::vector<uint32_t>& ids) {
    return std::vector<int>(ids.begin(),ids.end());
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

    struct Result { std::string text; uint32_t prompt_tokens; uint32_t output_tokens; size_t prefix_hit; };
    Result complete(const std::vector<uint32_t>& prompt,uint32_t count) {
        if(prompt.empty()) throw std::runtime_error("prompt is empty");
        std::lock_guard<std::mutex> lock(mutex);
        const bool mtp=mtp_width!=0;
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
        std::vector<uint32_t> output=engine.generate_from_pending(pending,count,mtp_width);
        auto eos=std::find(output.begin(),output.end(),(uint32_t)tokenizer.eos());
        if(eos!=output.end()) output.erase(eos+1,output.end());
        return {tokenizer.decode(to_int(output)),(uint32_t)prompt.size(),(uint32_t)output.size(),hit};
    }
};

uint32_t max_tokens(const json& body) {
    long long value=body.value("max_tokens",body.value("max_output_tokens",128ll));
    if(value<0 || value>UINT32_MAX) throw std::runtime_error("invalid max_tokens");
    return (uint32_t)value;
}

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
        server.Post("/v1/completions",guarded([&](const json& body,httplib::Response& r){
            std::string prompt=body.value("prompt",""); auto ids=to_u32(runtime.tokenizer.encode(prompt));
            auto result=runtime.complete(ids,max_tokens(body));
            json_response(r,{{"id","cmpl-metal"},{"object","text_completion"},{"model","q27-metal"},{"choices",json::array({{{"index",0},{"text",result.text},{"finish_reason","length"}}})},{"usage",{{"prompt_tokens",result.prompt_tokens},{"completion_tokens",result.output_tokens},{"total_tokens",result.prompt_tokens+result.output_tokens}}},{"q27_prefix_hit",result.prefix_hit}});
        }));
        server.Post("/v1/chat/completions",guarded([&](const json& body,httplib::Response& r){
            auto ids=to_u32(runtime.tokenizer.apply_chat_template(messages_from(body),body.value("enable_thinking",true)));
            auto result=runtime.complete(ids,max_tokens(body));
            json_response(r,{{"id","chatcmpl-metal"},{"object","chat.completion"},{"model","q27-metal"},{"choices",json::array({{{"index",0},{"message",{{"role","assistant"},{"content",result.text}}},{"finish_reason","length"}}})},{"usage",{{"prompt_tokens",result.prompt_tokens},{"completion_tokens",result.output_tokens},{"total_tokens",result.prompt_tokens+result.output_tokens}}},{"q27_prefix_hit",result.prefix_hit}});
        }));
        server.Post("/v1/messages",guarded([&](const json& body,httplib::Response& r){
            auto ids=to_u32(runtime.tokenizer.apply_chat_template(messages_from(body),true));
            auto result=runtime.complete(ids,max_tokens(body));
            json_response(r,{{"id","msg_metal"},{"type","message"},{"role","assistant"},{"model","q27-metal"},{"content",json::array({{{"type","text"},{"text",result.text}}})},{"stop_reason","max_tokens"},{"usage",{{"input_tokens",result.prompt_tokens},{"output_tokens",result.output_tokens}}},{"q27_prefix_hit",result.prefix_hit}});
        }));
        server.Post("/v1/responses",guarded([&](const json& body,httplib::Response& r){
            std::string input=body.contains("input")?text_content(body["input"]):"";
            auto ids=to_u32(runtime.tokenizer.encode(input)); auto result=runtime.complete(ids,max_tokens(body));
            json_response(r,{{"id","resp_metal"},{"object","response"},{"model","q27-metal"},{"output_text",result.text},{"output",json::array({{{"type","message"},{"role","assistant"},{"content",json::array({{{"type","output_text"},{"text",result.text}}})}}})},{"usage",{{"input_tokens",result.prompt_tokens},{"output_tokens",result.output_tokens},{"total_tokens",result.prompt_tokens+result.output_tokens}}},{"q27_prefix_hit",result.prefix_hit}});
        }));
        fprintf(stderr,"q27 Metal server listening on http://%s:%u (ctx=%u, kv=%s, mtp=%u)\n",host.c_str(),port,context,turbo3?"turbo3":"fp16",width);
        if(!server.listen(host.c_str(),(int)port)) throw std::runtime_error("server listen failed");
        return 0;
    } catch(const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}
