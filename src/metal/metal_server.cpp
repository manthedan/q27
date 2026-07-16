#include "metal_engine.h"
#include "stream_format.h"
#include "../tokenizer.h"
#include "../toolconstrain.h"
#include <cerrno>
#include "../../third_party/httplib.h"
#include "../../third_party/json.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
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

// Tool names for constrained decoding: OpenAI shape (tools[].function.name)
// and Anthropic shape (tools[].name) both accepted.
std::vector<std::string> tool_names_from(const json& body) {
    std::vector<std::string> names;
    if(!body.contains("tools") || !body["tools"].is_array()) return names;
    for(const auto& t:body["tools"]) {
        if(!t.is_object()) continue;
        if(t.contains("function") && t["function"].is_object() && t["function"].contains("name"))
            names.push_back(t["function"].value("name",""));
        else if(t.contains("name")) names.push_back(t.value("name",""));
    }
    names.erase(std::remove(names.begin(),names.end(),std::string()),names.end());
    return names;
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
    std::shared_ptr<q27::MetalEngine::Shared> shared;
    // One request slot = one engine on the shared mapping plus its private
    // prefix cache, scheduling phase, and constraint device-pool map (each
    // engine owns its own mask pool, so host mask id -> pool slot is
    // per-slot state; the host-side mask cache stays shared).
    struct Slot {
        q27::MetalEngine engine;
        PrefixCache cache;
        std::vector<int> host2dev;
        bool busy=false;
        enum class Phase { Idle, Prefill, Decode, Verify } phase=Phase::Idle;
        Slot(std::shared_ptr<q27::MetalEngine::Shared> s,uint32_t ctx,bool turbo3,size_t entries)
            :engine(std::move(s),ctx,turbo3),cache(entries) {}
    };
    std::vector<std::unique_ptr<Slot>> slots;
    uint32_t mtp_width;
    uint32_t context;
    // Lock order (multislot Phase 1 contract, docs/plans/2026-07-15-
    // multislot-phase1.md): route_ before lease_ — and in fact the two are
    // never held together. route_ guards slot assignment, phases, waiter
    // count, and wait stats, and is never held across GPU work; lease_
    // serializes every engine call (slots alias one Shared command queue)
    // and is held for one scheduling quantum at a time.
    //
    // The lease is a FIFO ticket lock, not a plain mutex: std::mutex makes
    // no fairness promise, and a tight decode loop (release, deliver,
    // reacquire) starves the other slot for a whole generation under it
    // (measured 5.5 s gate wait on the first two-slot run). Ticket order
    // caps the wait at one active quantum, which is the Phase 1 guarantee.
    struct Lease {
        std::mutex m;
        std::condition_variable cv;
        uint64_t next=0, serving=0;
        struct Guard {
            Lease* l=nullptr;
            Guard()=default;
            explicit Guard(Lease& lease):l(&lease) {
                std::unique_lock<std::mutex> lk(l->m);
                const uint64_t ticket=l->next++;
                l->cv.wait(lk,[&]{ return l->serving==ticket; });
            }
            Guard(Guard&& o) noexcept :l(o.l) { o.l=nullptr; }
            Guard& operator=(Guard&&)=delete;
            Guard(const Guard&)=delete;
            Guard& operator=(const Guard&)=delete;
            ~Guard() {
                if(!l) return;
                { std::lock_guard<std::mutex> lk(l->m); l->serving++; }
                l->cv.notify_all();
            }
        };
    };
    std::mutex route_;
    std::condition_variable slot_free_;
    Lease lease_;
    // Slot admission is ticketed too: a bare condition_variable lets a
    // newly arriving handler barge past an awakened waiter and starve it
    // (codex P2 on d243f92); tickets hand slots out in arrival order, and
    // the ticket spread doubles as the queue bound.
    uint64_t slot_next_=0, slot_serving_=0;
    uint32_t queue_waiters=0;
    static constexpr uint32_t QUEUE_MAX=8;
    // Queue overflow gets its own type so the HTTP layer can answer 503
    // with the documented overloaded_error body (G6) instead of the shared
    // 400 path. Streaming requests that overflow after headers are sent
    // keep the SSE error-event path (status already committed).
    struct ServerOverloaded : std::runtime_error { using std::runtime_error::runtime_error; };
    // Innermost lock: guards the shared host-side ToolMaskCache (mask
    // construction simulates the whole vocabulary on a miss). Order:
    // route_ | lease_ -> mask_mutex_; never the reverse.
    std::mutex mask_mutex_;
    // Wait accounting bucketed by what the competing traffic was doing at
    // arrival (idle/prefill/decode/verify), guarded by route_. Two distinct
    // quantities: queue wait (arrival -> slot admission; bounded only by
    // QUEUE_MAX generations) and gate wait (admission -> first GPU lease;
    // the Phase 1 one-active-quantum guarantee applies to THIS one).
    struct WaitStats { uint64_t n=0; double sum_ms=0, max_ms=0; };
    std::map<std::string,WaitStats> queue_wait_stats, gate_wait_stats;
    bool constrain_tools=false;
    std::vector<std::string> vocab_bytes_v;
    q27::ToolMaskCache mask_cache;

    Runtime(const std::string& model,const std::string& tok,uint32_t ctx,bool turbo3,
            uint32_t width,size_t cache_entries,bool constrain,uint32_t slot_count)
        :tokenizer(tok),mtp_width(width),context(ctx),constrain_tools(constrain) {
        if(tokenizer.vocab_size()!=q27::MetalEngine::vocabulary_size())
            throw std::runtime_error("tokenizer/model vocabulary mismatch");
        shared=q27::MetalEngine::open_shared(model);
        slots.push_back(std::make_unique<Slot>(shared,ctx,turbo3,cache_entries));
        // G6 admission (docs/plans/2026-07-16-g6-admission.md): additional
        // slots must fit the FULL per-slot footprint — KV + fixed engine
        // state + snapshot capacity x snapshot bytes — plus the backend-
        // shared GQA partial peak, against the device budget
        // (Q27_METAL_BUDGET_MB test/override hook; default = half the
        // recommended working set, the engine KV check's convention). The
        // engine's own KV check stays underneath as defense in depth; a
        // budget below even one slot still serves one (never zero).
        const char* budget_env=getenv("Q27_METAL_BUDGET_MB");
        uint64_t budget=shared->backend.recommended_working_set_size()/2;
        if(budget_env) {
            // Fail loud on a malformed override: "-1" through strtoull would
            // wrap to an effectively unlimited budget and bypass the gate.
            char* end=nullptr; errno=0;
            const unsigned long long mb=strtoull(budget_env,&end,10);
            if(errno || end==budget_env || *end || !mb || mb>(1ull<<24))
                throw std::runtime_error("Q27_METAL_BUDGET_MB must be an integer 1..16777216");
            budget=(uint64_t)mb*1024ull*1024ull;
        }
        const q27::MetalEngine& e0=slots[0]->engine;
        const uint64_t per_slot=e0.kv_reserved_bytes()
                               +q27::MetalEngine::fixed_state_bytes(e0.chunked_prefill())
                               +(uint64_t)cache_entries*e0.snapshot_bytes();
        const uint64_t shared_term=q27::MetalEngine::gqa_partial_peak(
                ctx,shared->backend.gqa_block_size());
        for(uint32_t s=1;s<slot_count;s++) {
            const uint64_t need=(uint64_t)(slots.size()+1)*per_slot+shared_term;
            if(need>budget) {
                fprintf(stderr,"multislot: slot %u admission rejected: %.0f MB needed "
                        "(%zu+1 slots x %.0f MB/slot + %.0f MB partials) > %.0f MB budget%s; "
                        "serving with %zu slot(s)\n",
                        s,need/1048576.0,slots.size(),per_slot/1048576.0,
                        shared_term/1048576.0,budget/1048576.0,
                        budget_env?" (Q27_METAL_BUDGET_MB)":"",slots.size());
                break;
            }
            try { slots.push_back(std::make_unique<Slot>(shared,ctx,turbo3,cache_entries)); }
            catch(const std::exception& e) {
                fprintf(stderr,"multislot: slot %u admission failed (%s); serving with %zu slot(s)\n",
                        s,e.what(),slots.size());
                break;
            }
        }
        if(constrain_tools) {
            vocab_bytes_v=tokenizer.vocab_bytes();
            mask_cache.init(&vocab_bytes_v,tokenizer.token_id("</tool_call>"));
            fprintf(stderr,"constrain-tools: grammar-locked <tool_call> bodies (open=%d close=%d)\n",
                    tokenizer.token_id("<tool_call>"),tokenizer.token_id("</tool_call>"));
        }
    }

    static const char* phase_name(Slot::Phase p) {
        switch(p) {
            case Slot::Phase::Prefill: return "prefill";
            case Slot::Phase::Decode: return "decode";
            case Slot::Phase::Verify: return "verify";
            default: return "idle";
        }
    }

    // Prefill width policy (Phase 1 contract): the width is a runtime
    // policy, not an engine constant. 96 when nothing competes, 48 when the
    // competing traffic is itself prefilling, 12 when a latency-sensitive
    // stream (decode/MTP verify) or a queued request is waiting for the GPU.
    uint32_t quantum_width(const Slot* self) {
        std::lock_guard<std::mutex> lk(route_);
        bool other_busy=false, other_latency=false;
        for(const auto& s:slots) {
            if(s.get()==self || !s->busy) continue;
            other_busy=true;
            if(s->phase!=Slot::Phase::Prefill) other_latency=true;
        }
        if(other_latency || queue_waiters>0) return 12;
        if(other_busy) return 48;
        return 96;
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
        double queue_wait_ms=0;    // arrival to slot admission
        double gate_wait_ms=0;     // slot admission to first GPU lease
        const char* arrival="idle"; // competing slot's phase at arrival
    };

    // Single generation core shared by streaming and non-streaming paths.
    // `emit(piece)` receives UTF-8-safe, stop-sequence-trimmed text as it is
    // produced and returns false when the client has gone away.
    //
    // Multislot Phase 1: a request claims an idle slot (engine + prefix
    // cache), then makes progress one scheduling quantum at a time — one
    // prefill chunk at the policy width, one decode step, or one MTP
    // draft/verify/commit round per GPU lease — so a concurrent request on
    // the other slot waits at most one active quantum, never a whole
    // generation. Text delivery (decode/UTF-8/stop gates/emit) runs outside
    // the lease: a slow client can stall its own stream, not the GPU.
    Outcome run(const std::vector<uint32_t>& prompt,uint32_t count,
                const q27::SamplingParams& sampling,
                const std::vector<std::string>& stops,
                const std::function<bool(const std::string&)>& emit,
                const std::vector<std::string>& tool_names={}) {
        if(prompt.empty()) throw std::runtime_error("prompt is empty");
        q27::validate_sampling(sampling);
        const bool mtp=mtp_width!=0 && sampling.temperature==0.0f;
        const auto arrive=std::chrono::steady_clock::now();

        // ---- slot acquisition (route_ only; never held across GPU work) ----
        Slot* slot=nullptr;
        const char* arrival="idle";
        {
            std::unique_lock<std::mutex> lk(route_);
            for(const auto& s:slots) if(s->busy) arrival=phase_name(s->phase);
            if(slot_next_-slot_serving_>=QUEUE_MAX)
                throw ServerOverloaded("server overloaded: request queue is full");
            const uint64_t ticket=slot_next_++;
            // Pass the turn on every exit path, or a thrown acquisition
            // would wedge every later ticket.
            struct TurnPass {
                Runtime& rt;
                ~TurnPass() { rt.slot_serving_++; rt.slot_free_.notify_all(); }
            } turn{*this};
            queue_waiters++;
            slot_free_.wait(lk,[&]{
                if(slot_serving_!=ticket) return false;
                for(const auto& s:slots) if(!s->busy) return true;
                return false;
            });
            queue_waiters--;
            for(const auto& s:slots) if(!s->busy) { slot=s.get(); break; }
            slot->busy=true;
            slot->phase=Slot::Phase::Prefill;
        }
        struct SlotRelease {
            Runtime& rt; Slot& s;
            ~SlotRelease() {
                { std::lock_guard<std::mutex> lk(rt.route_); s.busy=false; s.phase=Slot::Phase::Idle; }
                // notify_all, not notify_one: only the serving ticket's
                // waiter can proceed, and notify_one may wake a different
                // ticket that just re-sleeps — wedging the queue while a
                // slot sits idle (codex P1 on cdf85b2).
                rt.slot_free_.notify_all();
            }
        } slot_release{*this,*slot};
        q27::MetalEngine& engine=slot->engine;

        // First lease acquisition stamps the gate wait; every engine call
        // below runs under one of these scoped leases.
        const auto admitted=std::chrono::steady_clock::now();
        const double queue_wait_ms=
            std::chrono::duration<double,std::milli>(admitted-arrive).count();
        double gate_wait_ms=-1.0;
        auto lease_now=[&]()->Lease::Guard {
            Lease::Guard gpu(lease_);
            if(gate_wait_ms<0)
                gate_wait_ms=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-admitted).count();
            return gpu;
        };

        // ---- prompt ingestion, one quantum per chunk ----
        size_t hit=0; uint32_t pending=0;
        bool restored=false;
        {
            auto gpu=lease_now();
            // Round-2 expert P0 #1 (leak across requests): defensive entry
            // reset; the scope-exit guard below covers every later exit path.
            engine.set_tool_constraint(-1);
            restored=slot->cache.restore(engine,prompt,mtp,hit,pending);
            if(!restored) { engine.reset(); hit=0; }
            if((uint64_t)engine.position()+(prompt.size()-hit)>context)
                throw std::runtime_error("prompt exceeds context");
        }
        std::vector<uint32_t> suffix(prompt.begin()+hit,prompt.end());
        if(!suffix.empty()) {
            if(mtp || !engine.chunked_prefill()) {
                // MTP warming is token-serial inside the engine (each token
                // needs its final hidden state), so this path stays one
                // coarse quantum — a known Phase 1 limitation, documented in
                // the plan; the wait metrics expose it honestly.
                auto gpu=lease_now();
                pending=engine.ingest_prompt(suffix,mtp,false);
            } else {
                size_t i=0;
                const size_t chunkable=suffix.size()-1;
                while(chunkable-i>=2) {
                    const uint32_t width=quantum_width(slot);
                    const uint32_t take=(uint32_t)std::min<size_t>(width,chunkable-i);
                    auto gpu=lease_now();
                    engine.prefill_chunk(suffix.data()+i,take);
                    i+=take;
                }
                // Serial tail: at most one leftover chunkable token plus the
                // final token, which produces the logits and pending id —
                // mirrors MetalEngine::prefill()'s tail exactly.
                for(;i<suffix.size();i++) {
                    auto gpu=lease_now();
                    pending=engine.step(suffix[i]);
                }
            }
        }
        // Fail oversize generations before emitting anything, exactly like
        // the whole-generation streaming calls used to.
        if((uint64_t)engine.position()+(count?count-1:0)>context)
            throw std::runtime_error("generation exceeds context");
        // Cache the prompt state before generation mutates it. One entry costs
        // about 151 MiB for GDN state, so the default capacity is deliberately 1.
        // Cancelled requests never reach another insert, so post-cancel MTP
        // lane state is structurally non-cacheable (Phase 1 cancel invariant).
        // prepare_insert can release an evicted snapshot's GPU buffers, so it
        // stays under the lease alongside capture_state.
        {
            auto gpu=lease_now();
            if(slot->cache.prepare_insert(prompt,mtp))
                slot->cache.insert(prompt,mtp,pending,engine.capture_state());
        }
        {
            std::lock_guard<std::mutex> lk(route_);
            slot->phase = mtp ? Slot::Phase::Verify : Slot::Phase::Decode;
            WaitStats& qs=queue_wait_stats[arrival];
            qs.n++; qs.sum_ms+=queue_wait_ms; qs.max_ms=std::max(qs.max_ms,queue_wait_ms);
            WaitStats& gs=gate_wait_stats[arrival];
            gs.n++; gs.sum_ms+=std::max(gate_wait_ms,0.0); gs.max_ms=std::max(gs.max_ms,gate_wait_ms);
        }

        // token -> decode -> UTF-8 boundary gate -> stop-sequence holdback ->
        // emit, all outside the GPU lease. deliver() returns false to stop
        // generation, either because the client left (client_gone) or a stop
        // sequence completed (stop_hit); the two are distinguished for
        // finish-reason reporting.
        q27::Utf8Gate ugate;
        q27::StopBuffer stopbuf(stops);
        const uint32_t eos_id=(uint32_t)tokenizer.eos();
        bool client_gone=false, stop_hit=false;
        uint32_t produced=0;
        q27::MetalEngine::StopCause cause=q27::MetalEngine::StopCause::MaxTokens;
        auto deliver=[&](uint32_t token)->bool {
            bool stopped=false;
            std::string safe=stopbuf.feed(ugate.feed(tokenizer.decode_one((int)token)),stopped);
            if(!emit(safe)) { client_gone=true; cause=q27::MetalEngine::StopCause::Cancelled; return false; }
            if(stopped) { stop_hit=true; cause=q27::MetalEngine::StopCause::Cancelled; return false; }
            produced++;
            return true;
        };
        // Constrained tool decoding: trigger detection + grammar feeding on
        // the serial token stream (rounds are single tokens on this path, so
        // the CUDA engage-lag truncation degenerates to plain sequencing: the
        // constraint set here masks the NEXT token's logits inside step()).
        q27::BasicToolConstrainer<q27::MetalEngine,q27::Tokenizer> tc;
        tc.eng=&engine; tc.tok=&tokenizer; tc.cache=&mask_cache; tc.host2dev=&slot->host2dev;
        tc.enabled=constrain_tools && !tool_names.empty() && sampling.temperature==0.0f && !mtp_width;
        {
            auto gpu=lease_now();
            tc.begin(tool_names);
        }
        // Scope-exit constraint cleanup: runs on normal return, client
        // disconnect, and engine exceptions alike, and never throws (a
        // cleanup failure must not mask the original exception). Takes its
        // own lease — the per-quantum leases are all released by then.
        struct ConstraintCleanup {
            Runtime& rt;
            q27::BasicToolConstrainer<q27::MetalEngine,q27::Tokenizer>& tc;
            q27::MetalEngine& engine;
            ~ConstraintCleanup() {
                try {
                    Lease::Guard gpu(rt.lease_);
                    tc.end();
                    engine.set_tool_constraint(-1);
                } catch(...) {}
            }
        } constraint_cleanup{*this,tc,engine};

        // ---- generation, one quantum per lease ----
        if(sampling.temperature>0.0f) {
            std::mt19937_64 rng(sampling.seed);
            while(produced<count) {
                uint32_t token;
                {
                    auto gpu=lease_now();
                    token=engine.sample_from_logits(sampling,rng);
                }
                if(token==eos_id) { cause=q27::MetalEngine::StopCause::Eos; break; }
                if(!deliver(token)) break;
                if(produced==count) break;
                auto gpu=lease_now();
                engine.step(token);
            }
        } else if(mtp && engine.chunked_prefill()) {
            uint32_t live_width=std::min(mtp_width,4u);
            std::vector<uint32_t> committed;
            bool stopped=false;
            while(!stopped && produced<count) {
                if(produced+1==count) {
                    if(pending!=eos_id) deliver(pending);
                    else cause=q27::MetalEngine::StopCause::Eos;
                    break;
                }
                committed.clear();
                {
                    auto gpu=lease_now();
                    pending=engine.mtp_round(pending,count-produced,eos_id,mtp_width,
                                             live_width,committed);
                }
                for(uint32_t token:committed) {
                    if(token==eos_id) { cause=q27::MetalEngine::StopCause::Eos; stopped=true; break; }
                    if(!deliver(token)) { stopped=true; break; }
                }
            }
        } else {
            // Serial greedy walk (also the constrained-decode path): emit the
            // pending token, then each step yields the next. The constraint
            // ops for the token just emitted run under the same lease as the
            // step they mask, exactly as the old in-sink sequencing did.
            uint32_t cur=pending;
            while(produced<count) {
                if(cur==eos_id) { cause=q27::MetalEngine::StopCause::Eos; break; }
                if(!deliver(cur)) break;
                if(produced==count) break;
                if(tc.enabled && tc.active) {
                    // Pre-materialize the advanced state's mask OUTSIDE the
                    // lease: ToolMaskCache::get simulates the whole vocabulary
                    // on a miss (codex P2 on d243f92), which must not extend
                    // the other slot's wait. Same peek-advance the CUDA flow's
                    // on_pending uses; apply() below then hits the cache. The
                    // engage path (scan_round) still builds its entry mask
                    // under the lease — once per tool call, bounded.
                    q27::ToolGrammar peek=tc.tg;
                    bool ok=true;
                    for(char c:tokenizer.decode_one((int)cur))
                        if(!peek.advance(c)) { ok=false; break; }
                    if(ok && !peek.closed()) {
                        std::lock_guard<std::mutex> mk(mask_mutex_);
                        mask_cache.get(peek);
                    }
                }
                auto gpu=lease_now();
                if(tc.enabled) {
                    // mask_mutex_ inside the lease guards the shared host
                    // cache against a concurrent slot's prewarm above.
                    std::lock_guard<std::mutex> mk(mask_mutex_);
                    const int tid=(int)cur;
                    tc.scan_round(&tid,1);
                    tc.on_id(tid);
                    // Restage the ADVANCED grammar state's mask (codex P1):
                    // on_id moves tc.tg but stages nothing, so without this
                    // every step after the first constrained token decodes
                    // under the previous state's legal set.
                    if(tc.active) tc.apply(tc.tg);
                }
                cur=engine.step(cur);
            }
        }
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
        out.queue_wait_ms=queue_wait_ms;
        out.gate_wait_ms=std::max(gate_wait_ms,0.0);
        out.arrival=arrival;
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
        fprintf(stderr,"usage: %s model.q27 tokenizer.tok [--host 127.0.0.1] [--port 8080] [--ctx 8192] [--mtp 2..12] [--kv fp16|turbo3] [--prefix-entries N] [--constrain-tools] [--slots N]\n",argv[0]);
        return 1;
    }
    try {
        std::string model=argv[1],tok=argv[2],host="127.0.0.1";
        uint32_t port=8080,context=8192,width=0,prefix_entries=1,slot_count=2;
        bool turbo3=false; bool constrain_tools=false;
        for(int i=3;i<argc;i++) {
            std::string arg=argv[i];
            if(arg=="--host" && i+1<argc) host=argv[++i];
            else if(arg=="--port" && i+1<argc) port=parse_u32(argv[++i],"--port");
            else if(arg=="--ctx" && i+1<argc) context=parse_u32(argv[++i],"--ctx");
            else if(arg=="--mtp" && i+1<argc) width=parse_u32(argv[++i],"--mtp");
            else if(arg=="--prefix-entries" && i+1<argc) prefix_entries=parse_u32(argv[++i],"--prefix-entries");
            else if(arg=="--slots" && i+1<argc) slot_count=parse_u32(argv[++i],"--slots");
            else if(arg=="--kv" && i+1<argc) { std::string mode=argv[++i]; if(mode=="turbo3")turbo3=true; else if(mode!="fp16")throw std::runtime_error("invalid --kv"); }
            else if(arg=="--constrain-tools") constrain_tools=true;
            else throw std::runtime_error("unknown/incomplete argument: "+arg);
        }
        if(port>65535) throw std::runtime_error("port out of range");
        if(width && (width<2 || width>12)) throw std::runtime_error("MTP width must be 2..12");
        if(constrain_tools && width) throw std::runtime_error("--constrain-tools requires serial decode; drop --mtp (verify-lane masks are not wired on Metal)");
        // Phase 1's latency guarantee (wait <= one active quantum) only
        // holds with one competing slot; >2 needs the scheduler and the
        // width/stats model extended first (codex P2 on d243f92).
        if(slot_count<1 || slot_count>2) throw std::runtime_error("--slots must be 1..2 in multislot Phase 1");
        Runtime runtime(model,tok,context,turbo3,width,prefix_entries,constrain_tools,slot_count);
        httplib::Server server;
        // Bound the accept-side queue (codex P1 on d243f92): the default
        // task queue holds accepted connections without limit, so the
        // in-run admission bound alone could never engage — excess requests
        // would pile up behind the workers instead of being rejected.
        // 16 workers > QUEUE_MAX + slots: with only 8 workers the ticket-queue
        // overflow (503) was UNREACHABLE dead code — at most 7 requests could
        // wait while one generated (G6 found this). The 32-connection accept
        // queue stays the outer bound.
        server.new_task_queue=[]{ return new httplib::ThreadPool(16,32); };
        server.Get("/health",[](const httplib::Request&,httplib::Response& r){json_response(r,{{"status","ok"}});});
        // Wait honesty (Phase 1 contract): per-arrival-phase stats. Gate
        // wait (admission -> first lease) carries the one-quantum bound;
        // queue wait (arrival -> admission) is bounded only by QUEUE_MAX
        // generations and is reported so nobody mistakes one for the other.
        server.Get("/stats",[&runtime](const httplib::Request&,httplib::Response& r){
            auto bucket=[](const std::map<std::string,Runtime::WaitStats>& stats){
                json out=json::object();
                for(const auto& [phase,ws]:stats)
                    out[phase]={{"requests",ws.n},
                                {"mean_ms",ws.n?ws.sum_ms/ws.n:0.0},
                                {"max_ms",ws.max_ms}};
                return out;
            };
            json gate,queue;
            {
                std::lock_guard<std::mutex> lk(runtime.route_);
                gate=bucket(runtime.gate_wait_stats);
                queue=bucket(runtime.queue_wait_stats);
            }
            json_response(r,{{"slots",runtime.slots.size()},
                             {"gate_wait_by_arrival",gate},
                             {"queue_wait_by_arrival",queue}});
        });
        server.Get("/v1/models",[](const httplib::Request&,httplib::Response& r){json_response(r,{{"object","list"},{"data",json::array({{{"id","q27-metal"},{"object","model"}}})}});});

        auto guarded=[&](auto handler) {
            return [&,handler](const httplib::Request& request,httplib::Response& response) {
                try { handler(json::parse(request.body),response); }
                catch(const Runtime::ServerOverloaded& e) { json_response(response,{{"error",{{"message",e.what()},{"type","overloaded_error"}}}},503); }
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
                    [&](const std::string& piece){ text+=piece; return true; },
                    tool_names_from(body));
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
            const std::vector<std::string> tnames=tool_names_from(body);
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,chat,objd,id,created,tnames](size_t,httplib::DataSink& sink)->bool {
                    try {
                        auto emit=[&](const std::string& piece)->bool {
                            std::string s=q27::sse_data(
                                q27::openai_stream_chunk(chat,id,objd,created,"q27-metal",piece));
                            return sink.write(s.data(),s.size());
                        };
                        auto outcome=runtime.run(ids,n,sampling,stops,emit,tnames);
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
                    [&](const std::string& piece){ text+=piece; return true; },
                    tool_names_from(body));
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
            const std::vector<std::string> tnames=tool_names_from(body);
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,mid,tnames](size_t,httplib::DataSink& sink)->bool {
                    auto ev=[&](const char* name,const json& j){ std::string s=q27::sse_event(name,j); return sink.write(s.data(),s.size()); };
                    try {
                        json msg={{"id",mid},{"type","message"},{"role","assistant"},{"model","q27-metal"},
                            {"content",json::array()},{"stop_reason",nullptr},{"stop_sequence",nullptr},
                            {"usage",{{"input_tokens",(int)ids.size()},{"output_tokens",0}}}};
                        // A client gone before generation starts must not hold
                        // the engine through the token cap: gate the run on the
                        // opening writes, and probe the socket on empty pieces
                        // (still no empty text_delta — parity w/ server.cu).
                        if(!ev("message_start",{{"type","message_start"},{"message",msg}}) ||
                           !ev("content_block_start",{{"type","content_block_start"},{"index",0},
                               {"content_block",{{"type","text"},{"text",""}}}})) {
                            sink.done();
                            return true;
                        }
                        auto emit=[&](const std::string& piece)->bool {
                            if(piece.empty()) return sink.is_writable();
                            return ev("content_block_delta",{{"type","content_block_delta"},{"index",0},
                                {"delta",{{"type","text_delta"},{"text",piece}}}});
                        };
                        auto outcome=runtime.run(ids,n,sampling,stops,emit,tnames);
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
                    [&](const std::string& piece){ text+=piece; return true; },
                    tool_names_from(body));
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
            const std::vector<std::string> tnames=tool_names_from(body);
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,rid,mid,tnames](size_t,httplib::DataSink& sink)->bool {
                    auto ev=[&](const json& j){ std::string s=q27::sse_event(j.value("type",std::string("x")),j); return sink.write(s.data(),s.size()); };
                    try {
                        // Gate the run on the opening writes and probe the
                        // socket on empty pieces, so a disconnected client
                        // cannot hold the engine through the token cap.
                        if(!ev({{"type","response.created"},{"response",{{"id",rid},{"object","response"},{"status","in_progress"}}}}) ||
                           !ev({{"type","response.output_item.added"},{"output_index",0},
                               {"item",{{"type","message"},{"id",mid},{"role","assistant"},{"status","in_progress"},{"content",json::array()}}}}) ||
                           !ev({{"type","response.content_part.added"},{"item_id",mid},{"output_index",0},{"content_index",0},
                               {"part",{{"type","output_text"},{"text",""},{"annotations",json::array()}}}})) {
                            sink.done();
                            return true;
                        }
                        std::string text;
                        auto emit=[&](const std::string& piece)->bool {
                            text+=piece;
                            if(piece.empty()) return sink.is_writable();
                            return ev({{"type","response.output_text.delta"},{"item_id",mid},
                                {"output_index",0},{"content_index",0},{"delta",piece}});
                        };
                        auto outcome=runtime.run(ids,n,sampling,stops,emit,tnames);
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

        fprintf(stderr,"q27 Metal server listening on http://%s:%u (ctx=%u, kv=%s, mtp=%u, slots=%zu)\n",
                host.c_str(),port,context,turbo3?"turbo3":"fp16",width,runtime.slots.size());
        if(!server.listen(host.c_str(),(int)port)) throw std::runtime_error("server listen failed");
        return 0;
    } catch(const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}
