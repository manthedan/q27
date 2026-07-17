#include "metal_engine.h"
#include "stream_format.h"
#include "../suffixdraft.h"
#include "../tool_preamble.h"
#include "../tokenizer.h"
#include "../toolconstrain.h"
#include <cerrno>
#include "../../third_party/httplib.h"
#include "../../third_party/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>

#include <CommonCrypto/CommonDigest.h>
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

// OpenAI chat messages -> Msg list for chatml_prompt (which merges the tools
// preamble into the system message, replacing the manual merge that lived
// here). Beyond-CUDA: src/server.cu's chat endpoint is text-only by design
// (its structured tool paths are /v1/messages and /v1/responses), but pi.dev
// speaks openai-completions, so this endpoint must round-trip tool traffic —
// assistant.tool_calls arrays and role:"tool" results are reconstructed to
// the model's <tool_call>/<tool_response> markers (agentic-parity round,
// docs/plans/2026-07-17-metal-agentic-parity.md).
std::vector<q27::Msg> openai_msgs(const json& body) {
    std::vector<q27::Msg> msgs;
    if(body.contains("system")) {
        std::string system=text_content(body["system"]);
        if(!system.empty()) msgs.push_back({"system",system});
    }
    if(!body.contains("messages") || !body["messages"].is_array())
        throw std::runtime_error("messages must be an array");
    for(const auto& message:body["messages"]) {
        if(!message.is_object()) continue;
        std::string role=message.value("role","");
        if(role=="developer") role="system";
        std::string content=message.contains("content")?text_content(message["content"]):"";
        if(role.empty()) continue;
        if(role=="tool") {
            msgs.push_back({"user",q27::tool_response_text(content)});
            continue;
        }
        if(role=="assistant" && message.contains("tool_calls") && message["tool_calls"].is_array()) {
            for(const auto& c:message["tool_calls"]) {
                if(!c.is_object() || !c.contains("function") || !c["function"].is_object()) continue;
                const json& fn=c["function"];
                // OpenAI carries arguments as a JSON-encoded STRING; tolerate
                // an object too (some clients send it pre-parsed).
                json args=json::object();
                if(fn.contains("arguments")) {
                    if(fn["arguments"].is_string()) {
                        try { args=json::parse(fn["arguments"].get<std::string>()); }
                        catch(...) { args=fn["arguments"]; }
                    } else args=fn["arguments"];
                }
                if(!content.empty() && content.back()!='\n') content+="\n";
                content+=q27::tool_call_text(fn.value("name",""),args);
            }
        }
        msgs.push_back({role,content});
    }
    // Merge consecutive same-role messages (a run of tool results becomes one
    // user block), matching the CUDA responses handler's merge.
    std::vector<q27::Msg> merged;
    for(auto& m:msgs) {
        if(!merged.empty() && merged.back().role==m.role) merged.back().content+="\n"+m.content;
        else merged.push_back(std::move(m));
    }
    if(merged.empty()) throw std::runtime_error("messages are empty");
    if(merged[0].role=="system") q27::normalize_cc_billing_header(merged[0].content);
    return merged;
}

// Unique-ish ids for tool_use/tool_calls blocks: agent clients key results
// to call ids, so a fixed string would collide across turns.
std::atomic<long> req_counter{0};

// Context preflight ceiling: the run()-level throws stay as backstops, but
// agent clients need the refusal BEFORE slot claim / SSE commit, in each
// API's native error shape ("prompt is too long" is Claude Code's
// compact-now signal). Reserve mirrors run()'s generation entry check
// (position + count-1 <= context) plus the speculation lookahead.
uint32_t max_prompt_tokens(uint32_t context,uint32_t mtp_width,uint32_t suffix_width) {
    const uint32_t reserve=1+std::max(mtp_width,suffix_width);
    return context>reserve?context-reserve:1;
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

// Disk snapshot store (prefix snapshots Phase 2, docs/plans/2026-07-16-
// prefix-snapshots.md): token-prefix keyed — the server tokenizes every
// prompt itself, so "the snapshot's stored token ids are a prefix of this
// request's tokens" is exact and has no BPE-boundary hazard (recorded
// design deviation from ds4's byte-SHA1, which exists for stateless
// clients that retokenize). Files are named by the SHA1 of the token
// bytes, so an identical prefix overwrites rather than duplicates. LRU by
// mtime; hits touch the file. All file I/O runs OUTSIDE the GPU lease;
// only save_state/load_state (which read/write GPU buffers) go under it.
class DiskSnapshotStore {
  public:
    // tag = artifact/KV identity prefix baked into every filename, so one
    // directory shared by different artifacts or fp16/turbo3 servers never
    // cross-matches or overwrites incompatible snapshots (codex P2 on
    // 607160e); the deep header identity check at load stays underneath.
    void init(std::string dir, uint64_t max_bytes, std::string tag) {
        dir_=std::move(dir); max_bytes_=max_bytes; tag_=std::move(tag);
    }
    bool enabled() const { return !dir_.empty(); }

    // Longest stored token prefix of `prompt`. Full-length matches whose
    // logits are stale (mid-prefill saves) are skipped: state would be
    // exact but no pending token could be derived.
    bool best_match(const std::vector<uint32_t>& prompt,std::string& path_out,uint32_t& len_out) {
        if(!enabled()) return false;
        std::lock_guard<std::mutex> lk(m_);
        std::string best; uint32_t best_len=0;
        std::error_code ec;
        for(const auto& e:std::filesystem::directory_iterator(dir_,ec)) {
            if(!e.is_regular_file() || e.path().extension()!=".q27snap") continue;
            if(e.path().filename().string().rfind(tag_,0)!=0) continue;
            q27::MetalEngine::SnapshotInfo info;
            try { info=q27::MetalEngine::peek_snapshot(e.path().string()); }
            catch(...) { continue; }   // corrupt/foreign file: never a hit
            // Saves record exactly the encoded prefix; anything else is not
            // resumable by token matching.
            if(info.position!=info.tokens.size()) continue;
            if(info.tokens.empty() || info.tokens.size()>prompt.size()) continue;
            if(info.tokens.size()==prompt.size() && !info.logits_resident) continue;
            if(info.tokens.size()<=best_len) continue;
            if(std::equal(info.tokens.begin(),info.tokens.end(),prompt.begin())) {
                best=e.path().string(); best_len=(uint32_t)info.tokens.size();
            }
        }
        if(best.empty()) return false;
        std::filesystem::last_write_time(best,std::filesystem::file_time_type::clock::now(),ec);
        path_out=std::move(best); len_out=best_len;
        return true;
    }

    std::string path_for(const uint32_t* tokens,uint32_t count) const {
        unsigned char sha[20];
        CC_SHA1(tokens,(CC_LONG)(count*4),sha);
        char hex[41];
        for(int i=0;i<20;i++) snprintf(hex+2*i,3,"%02x",sha[i]);
        return dir_+"/"+tag_+hex+".q27snap";
    }

    // Budget enforcement: oldest-first until the directory fits. The
    // just-written file is deletable too — the budget is a hard cap, and
    // the gate asserts the total never exceeds it. Returns {files, bytes}
    // removed so the --trace stream can record the eviction decision.
    std::pair<size_t,uint64_t> evict_past_budget() {
        if(!enabled() || !max_bytes_) return {0,0};
        std::lock_guard<std::mutex> lk(m_);
        struct F { std::string path; uint64_t size; std::filesystem::file_time_type mtime; };
        std::vector<F> files; uint64_t total=0;
        std::error_code ec;
        for(const auto& e:std::filesystem::directory_iterator(dir_,ec)) {
            if(!e.is_regular_file() || e.path().extension()!=".q27snap") continue;
            const uint64_t sz=(uint64_t)e.file_size(ec);
            files.push_back({e.path().string(),sz,e.last_write_time(ec)});
            total+=sz;
        }
        std::sort(files.begin(),files.end(),[](const F& a,const F& b){ return a.mtime<b.mtime; });
        size_t n=0; uint64_t freed=0;
        for(const auto& f:files) {
            if(total<=max_bytes_) break;
            if(std::filesystem::remove(f.path,ec)) { total-=f.size; n++; freed+=f.size; }
        }
        return {n,freed};
    }

    std::atomic<uint64_t> hits{0}, saves{0};
  private:
    std::string dir_; uint64_t max_bytes_=0; std::string tag_; std::mutex m_;
};

// Whole-session trace stream (triage I2, docs/plans/2026-07-17-ds4-product-
// triage.md): one JSONL stream of the events the parity rounds kept having
// to reconstruct by hand from scattered logs — rendered prompts, snapshot /
// prefix-cache decisions, tool-parser recoveries, cancellations, error
// answers. Diagnostic switch only (ds4 flag rule): off by default, zero
// semantic effect when on. Every line carries wall `ts` plus monotonic
// `tms` (ms since open) so two-slot interleavings reconstruct exactly.
struct TraceLog {
    void open(const std::string& path) {
        // Path only (codex P2): "-"/stderr is not offered — the server's
        // fprintf logging shares stderr, so the JSONL stream would not be
        // clean. /dev/stderr remains available for anyone who wants the mix.
        f_=fopen(path.c_str(),"a");
        if(!f_) throw std::runtime_error("cannot open --trace path: "+path);
    }
    bool enabled() const { return f_!=nullptr; }
    // noexcept: a diagnostic stream must never throw through a handler or
    // mask a cancellation (codex P1). json dump/copy/fwrite are contained
    // here; call-site initializer lists are scalar-only, so their
    // construction can only throw on OOM — accepted and recorded.
    void event(json j) noexcept {
        if(!f_) return;
        try {
            j["ts"]=(long)std::time(nullptr);
            std::lock_guard<std::mutex> lk(m_);
            // tms stamped inside the lock: file order == tms order (codex P2).
            j["tms"]=(long)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now()-t0_).count();
            const std::string s=j.dump();
            fwrite(s.data(),1,s.size(),f_);
            fputc('\n',f_);
            fflush(f_);
        } catch(...) {}
    }
  private:
    FILE* f_=nullptr; std::mutex m_;
    const std::chrono::steady_clock::time_point t0_=std::chrono::steady_clock::now();
};

// Trace prompt payloads cap at 64 KB (ctx-limit test prompts run ~1 MB);
// truncation is recorded, never silent.
inline json trace_text(const std::string& s) {
    if(s.size()<=65536) return json(s);
    return json({{"truncated",true},{"bytes",(uint64_t)s.size()},{"head",s.substr(0,65536)}});
}

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
    // Suffix-burst decode width (--suffix, 2..VERIFY_CHUNK_MAX; 0 = off).
    // Mutually exclusive with --mtp: the burst path is the no-MTP tiers'
    // speculation lever (2026-07-16-suffix-burst-verify.md, server phase).
    uint32_t suffix_width;
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
    // Thrown when a liveness probe finds the client gone during queue wait
    // or prefill (2026-07-17-abandoned-request-cancellation.md) —
    // deliberately NOT a std::exception: the generic handler catches would
    // otherwise try to write an error body to a dead socket. Handlers and
    // stream providers catch it by name and return without writing.
    struct ClientGone {};
    std::atomic<uint64_t> cancelled_queue{0}, cancelled_prefill{0};
    // Engine failures during generation are server bugs, not request bugs:
    // Anthropic defines api_error (500) for them, and 400 is fatal-class to
    // codex while 500 retries (residue round 2026-07-17-responses-parity-
    // residue.md). Handlers wrap ONLY the run() call — parse/validation/
    // overflow all throw before it. Known coarseness, recorded: the rare
    // post-restore "prompt exceeds context" inside run() rides this class.
    struct EngineError : std::runtime_error { using std::runtime_error::runtime_error; };
    // Mid-queue cancelled tickets awaiting their in-order skip (route_).
    std::set<uint64_t> cancelled_tickets_;
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
    // Speculation ground truth for the multislot MTP gates: a quantum round
    // committing >1 token proves accepted drafts, so committed > rounds is
    // the nonzero-speculation assert's unfakeable signal (vacuous-gate rule).
    std::atomic<uint64_t> spec_rounds_total{0}, spec_committed_total{0};
    // Suffix-path round attribution: bursts actually dispatched vs serial
    // fallbacks, so /stats shows whether traffic rides the batched path at
    // all (a suffix server whose every round falls back is misconfigured
    // or serving burst-hostile traffic — either way it should be visible).
    std::atomic<uint64_t> suffix_burst_rounds_total{0}, suffix_fallback_rounds_total{0};
    bool constrain_tools=false;
    std::vector<std::string> vocab_bytes_v;
    q27::ToolMaskCache mask_cache;
    DiskSnapshotStore snapstore;
    TraceLog trace;
    std::string model_name;
    // Auto-snapshot threshold in prompt tokens (0 = hint-only); set with
    // the snapshot store, meaningful only when snapstore.enabled().
    size_t snap_auto_min=0;

    Runtime(const std::string& model,const std::string& tok,uint32_t ctx,bool turbo3,
            uint32_t width,uint32_t sfx_width,size_t cache_entries,bool constrain,
            uint32_t slot_count)
        :tokenizer(tok),mtp_width(width),suffix_width(sfx_width),context(ctx),
         constrain_tools(constrain) {
        // Server identity (homebrew plan Q2): /health and the boot trace name
        // the resident artifact so wrapper/clients can tell what's loaded.
        model_name=std::filesystem::path(model).filename().string();
        if(tokenizer.vocab_size()!=q27::MetalEngine::vocabulary_size())
            throw std::runtime_error("tokenizer/model vocabulary mismatch");
        shared=q27::MetalEngine::open_shared(model);
        slots.push_back(std::make_unique<Slot>(shared,ctx,turbo3,cache_entries));
        // Snapshot v2 (2026-07-17-kv-except-snapshot-v2.md): exception
        // engines snapshot like any other — side rows ride every surface
        // and snapshot_bytes() prices them, so no capacity override exists
        // anymore. Informational note only.
        if(slots.front()->engine.kv_fp16_except())
            fprintf(stderr,"q27 Metal server: KV exception cells active (Q27_METAL_KV_FP16_CELLS, side codec %s); side caches ride prefix/disk snapshots (v2)\n",
                    slots.front()->engine.kv_side_codec()?"e4m3":"fp16");
        // G6 admission (docs/plans/2026-07-16-g6-admission.md): additional
        // slots must fit the FULL per-slot footprint — KV + this slot's own
        // GQA partials (per-engine since audit E2, charged inside
        // kv_reserved_bytes) + fixed engine state + snapshot capacity x
        // snapshot bytes — against the device budget (Q27_METAL_BUDGET_MB
        // test/override hook; default = half the recommended working set,
        // the engine KV check's convention). The engine's own KV check
        // stays underneath as defense in depth; a budget below even one
        // slot still serves one (never zero).
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
        for(uint32_t s=1;s<slot_count;s++) {
            const uint64_t need=(uint64_t)(slots.size()+1)*per_slot;
            if(need>budget) {
                fprintf(stderr,"multislot: slot %u admission rejected: %.0f MB needed "
                        "(%zu+1 slots x %.0f MB/slot incl partials) > %.0f MB budget%s; "
                        "serving with %zu slot(s)\n",
                        s,need/1048576.0,slots.size(),per_slot/1048576.0,
                        budget/1048576.0,
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
        // Prefix snapshots Phase 2: opt-in via Q27_METAL_SNAPSHOT_DIR;
        // budget via Q27_METAL_SNAPSHOT_MAX_MB (validated, fail-loud, same
        // class as Q27_METAL_BUDGET_MB; default 8192 MB). The artifact
        // identity hash (~3 s over the 7 GB mapping) is primed HERE, at
        // startup, so the first hinted request never stalls the lease on it.
        if(const char* sdir=getenv("Q27_METAL_SNAPSHOT_DIR"); sdir && *sdir) {
            uint64_t snap_mb=8192;
            if(const char* smax=getenv("Q27_METAL_SNAPSHOT_MAX_MB"); smax && *smax) {
                char* end=nullptr; errno=0;
                const unsigned long long mb=strtoull(smax,&end,10);
                if(errno || end==smax || *end || !mb || mb>(1ull<<24))
                    throw std::runtime_error("Q27_METAL_SNAPSHOT_MAX_MB must be an integer 1..16777216");
                snap_mb=(uint64_t)mb;
            }
            std::error_code ec;
            std::filesystem::create_directories(sdir,ec);
            if(ec || !std::filesystem::is_directory(sdir))
                throw std::runtime_error(std::string("Q27_METAL_SNAPSHOT_DIR is not a usable directory: ")+sdir);
            const unsigned char* sha=slots[0]->engine.snapshot_identity();
            // Full 160-bit identity in the tag: a truncated prefix could
            // collide across artifacts sharing a directory and let one
            // server overwrite another's snapshots (codex P2 on f05ef2d).
            // Exception engines append their cell config (v2): different
            // cell lists sharing a directory must MISS each other's files
            // (load_state would loudly reject them, costing a fallback per
            // request); env-unset tags are unchanged so pre-v2 snapshot
            // files stay live.
            std::string tag;
            tag.reserve(64);
            char hex[3];
            for(int i=0;i<20;i++) { snprintf(hex,3,"%02x",sha[i]); tag+=hex; }
            tag+=turbo3?'t':'f';
            if(slots[0]->engine.kv_fp16_except()) {
                // 'x' = fp16 sides, 'y' = e4m3 sides: codec is config
                // identity, so the two must miss each other's files.
                tag+=slots[0]->engine.kv_side_codec()?'y':'x';
                const uint8_t* masks=slots[0]->engine.kv_fp16_head_masks();
                for(int i=0;i<16;i++) tag+="0123456789abcdef"[masks[i]&15];
            }
            tag+='-';
            snapstore.init(sdir,snap_mb*1024ull*1024ull,tag.c_str());
            // A restart over an oversized directory must come back under
            // budget without waiting for the next save (codex P2 on 607160e).
            snapstore.evict_past_budget();
            if(!slots[0]->engine.chunked_prefill())
                fprintf(stderr,"prefix-snapshots: WARNING — no chunked prefill on this device; "
                        "\"snapshot\" hints are ignored (loads still served)\n");
            // Auto-snapshot threshold (2026-07-17 T2 prefill finding,
            // docs/plans/2026-07-17-t2-prefill-throughput.md): chunked
            // prefill is compute-mature at ~28-40 tok/s on the M4, so a
            // large agentic prompt (pi ~8.4K tokens, CC larger) costs
            // minutes of TTFT — and real agent clients never send the
            // "snapshot" hint. With the snapshot dir already opted in,
            // prompts at/above the threshold behave as hinted; the existing
            // covered-prefix skip and LRU budget bound the write traffic.
            // Q27_METAL_SNAPSHOT_AUTO overrides (tokens; 0 disables auto).
            snap_auto_min=4096;
            if(const char* sauto=getenv("Q27_METAL_SNAPSHOT_AUTO"); sauto && *sauto) {
                char* end=nullptr; errno=0;
                const unsigned long long v=strtoull(sauto,&end,10);
                if(errno || end==sauto || *end || v>(1ull<<24))
                    throw std::runtime_error("Q27_METAL_SNAPSHOT_AUTO must be an integer 0..16777216");
                snap_auto_min=(size_t)v;
            }
            fprintf(stderr,"prefix-snapshots: dir %s, budget %llu MB, auto>=%zu tokens, tag %s\n",
                    sdir,(unsigned long long)snap_mb,snap_auto_min,tag.c_str());
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
                const std::vector<std::string>& tool_names={},
                bool snapshot_hint=false,
                const std::function<bool()>& live={},
                const std::string& trace_id="") {
        if(prompt.empty()) throw std::runtime_error("prompt is empty");
        q27::validate_sampling(sampling);
        const bool mtp=mtp_width!=0 && sampling.temperature==0.0f;
        const bool sfx=suffix_width!=0 && sampling.temperature==0.0f;
        const auto arrive=std::chrono::steady_clock::now();

        // ---- slot acquisition (route_ only; never held across GPU work) ----
        Slot* slot=nullptr;
        const char* arrival="idle";
        {
            std::unique_lock<std::mutex> lk(route_);
            for(const auto& s:slots) if(s->busy) arrival=phase_name(s->phase);
            // A cancelled ticket may sit at the front with no waiter left to
            // skip it: fast-forward before the capacity check (and again in
            // every wait predicate), or stale cancels would falsely 503 new
            // arrivals. Always called under route_.
            auto drain_cancelled=[&]{
                while(cancelled_tickets_.erase(slot_serving_)) slot_serving_++;
            };
            drain_cancelled();
            if(slot_next_-slot_serving_>=QUEUE_MAX)
                throw ServerOverloaded("server overloaded: request queue is full");
            const uint64_t ticket=slot_next_++;
            // Pass the turn on every exit path, or a thrown acquisition
            // would wedge every later ticket. A mid-queue cancel must NOT
            // pass the turn out of order (codex P1 on this round: an
            // unconditional increment from ticket k while ticket k-2 is
            // still serving skips a live waiter and wedges the FIFO) — it
            // registers in cancelled_tickets_ instead and disarms, and
            // whoever holds route_ when serving reaches it skips past.
            struct TurnPass {
                Runtime& rt; bool armed=true;
                ~TurnPass() { if(armed) { rt.slot_serving_++; rt.slot_free_.notify_all(); } }
            } turn{*this};
            queue_waiters++;
            // Timed wait so a dead client's ticket can self-evacuate: the
            // 250 ms tick probes liveness (zero-timeout select + MSG_PEEK,
            // negligible next to any quantum) — the pile-up class from the
            // 2026-07-17 incident (dead requests holding queue positions
            // for minutes, then running to completion for nobody) drains
            // without ever touching the GPU.
            for(;;) {
                const bool admitted_now=slot_free_.wait_for(lk,
                    std::chrono::milliseconds(250),[&]{
                        drain_cancelled();
                        if(slot_serving_!=ticket) return false;
                        for(const auto& s:slots) if(!s->busy) return true;
                        return false;
                    });
                if(admitted_now) break;
                if(live && !live()) {
                    queue_waiters--;
                    cancelled_queue++;
                    trace.event({{"kind","cancel"},{"phase","queue"},{"id",trace_id}});
                    if(slot_serving_==ticket) {
                        // Front of the queue: the normal TurnPass increment
                        // is in order.
                    } else {
                        cancelled_tickets_.insert(ticket);
                        turn.armed=false;
                    }
                    throw ClientGone{};
                }
            }
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
        bool restored=false, saved_snapshot=false;
        // Disk lookup runs before taking the lease (pure file I/O); only
        // the resulting load_state goes under it. MTP requests stay on the
        // cold path: lane warming state is not part of the snapshot
        // contract in this phase.
        std::string disk_path; uint32_t disk_len=0; bool disk_loaded=false;
        const bool disk_ok=!mtp && snapstore.enabled() &&
                           snapstore.best_match(prompt,disk_path,disk_len);
        {
            auto gpu=lease_now();
            // Round-2 expert P0 #1 (leak across requests): defensive entry
            // reset; the scope-exit guard below covers every later exit path.
            engine.set_tool_constraint(-1);
            restored=slot->cache.restore(engine,prompt,mtp,hit,pending);
            if(!restored) { engine.reset(); hit=0; }
            // The deeper prefix wins across tiers: a short in-memory entry
            // must not mask a much longer persisted one (codex P2 on
            // 607160e). load_state validates fully before its first GPU
            // write, so a rejected file leaves a memory-restored state
            // intact; only a mid-restore I/O error falls all the way cold.
            if(disk_ok && disk_len>hit) {
                try {
                    engine.load_state(disk_path);
                    disk_loaded=true;
                    hit=disk_len;
                    if(hit==prompt.size()) { pending=engine.pending_from_logits(); restored=true; }
                    snapstore.hits++;
                } catch(const std::exception&) {
                    // A bad disk candidate must not cost more than it
                    // offered: fall back to the in-memory tier before going
                    // cold (codex P2 on f05ef2d).
                    engine.reset(); hit=0; pending=0;
                    restored=slot->cache.restore(engine,prompt,mtp,hit,pending);
                    if(!restored) hit=0;
                }
            }
            if((uint64_t)engine.position()+(prompt.size()-hit)>context)
                throw std::runtime_error("prompt exceeds context");
        }
        trace.event({{"kind","prefix"},{"id",trace_id},
                     {"tier",hit==0?"cold":(disk_loaded?"disk":"memory")},
                     {"hit",(uint64_t)hit},{"prompt_tokens",(uint64_t)prompt.size()}});
        // Hinted save target: a stable boundary — trim a 32-token tail
        // (the question-specific suffix) and align down to a 96-token
        // prefill-chunk boundary. Reached exactly by capping one chunk's
        // width; skipped when the restored prefix already covers it.
        //
        // Auto-save (snap_auto_min) rides the same machinery. Two codex P2s
        // on this, recorded: (a) same-path save collisions are lease-
        // serialized — save_state only ever runs under a Lease::Guard, so
        // the worst case is a redundant rewrite of an identical file, not a
        // torn .tmp; (b) a non-repeating large-prompt workload pays one
        // ~1.3 GB write (+~15 s under the lease) per unique prefix — the
        // LRU budget bounds retention, not write churn. This box serves
        // repeated agent prefixes, where the trade wins; churn-sensitive
        // deployments set Q27_METAL_SNAPSHOT_AUTO=0 (hint-only).
        size_t save_at=0;
        const bool snap_wanted=snapshot_hint ||
                               (snap_auto_min && prompt.size()>=snap_auto_min);
        if(snap_wanted && !mtp && snapstore.enabled() && engine.chunked_prefill() &&
           prompt.size()>32+96) {
            const size_t target=(prompt.size()-32)/96*96;
            if(target>hit) save_at=target;
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
                    // Per-chunk liveness probe: a chunk is seconds of GPU
                    // work, the probe is a zero-timeout select. On death
                    // with an armed snapshot target (the big-prompt cases),
                    // bank the finished chunks at the current position
                    // first — a client-timeout retry then restores from
                    // disk instead of re-paying the whole prefill cold
                    // (the timeout-retry livelock class).
                    if(live && !live()) {
                        // Best-effort: a failed save must not turn a dead-
                        // client cancel into the generic error path
                        // (codex P3 on this round).
                        if(save_at && i>0) try {
                            {
                                auto gpu=lease_now();
                                engine.save_state(snapstore.path_for(prompt.data(),(uint32_t)(hit+i)),
                                                  prompt.data(),(uint32_t)(hit+i),false);
                            }
                            snapstore.saves++;
                            const auto ev=snapstore.evict_past_budget();
                            trace.event({{"kind","snapshot_bank"},{"id",trace_id},
                                         {"len",(uint64_t)(hit+i)},
                                         {"evicted_files",ev.first},{"evicted_bytes",ev.second}});
                        } catch(const std::exception& e) {
                            fprintf(stderr,"[cancel-save] skipped: %s\n",e.what());
                        }
                        cancelled_prefill++;
                        trace.event({{"kind","cancel"},{"phase","prefill"},{"id",trace_id}});
                        throw ClientGone{};
                    }
                    const uint32_t width=quantum_width(slot);
                    uint32_t take=(uint32_t)std::min<size_t>(width,chunkable-i);
                    if(save_at && hit+i<save_at) {
                        // prefill_chunk takes 2..96 tokens: a one-token gap
                        // to the boundary cannot be reached by capping, so
                        // the (best-effort) save is skipped rather than the
                        // request failing (codex P1 on 607160e).
                        if(save_at-(hit+i)==1) save_at=0;
                        else take=(uint32_t)std::min<size_t>(take,save_at-(hit+i));
                    }
                    auto gpu=lease_now();
                    engine.prefill_chunk(suffix.data()+i,take);
                    i+=take;
                    if(save_at && hit+i==save_at) {
                        // Mid-prefill state is exact but the logits row is
                        // stale — recorded in the file so a same-length
                        // request can never derive a pending token from it.
                        engine.save_state(snapstore.path_for(prompt.data(),(uint32_t)save_at),
                                          prompt.data(),(uint32_t)save_at,false);
                        snapstore.saves++;
                        trace.event({{"kind","snapshot_save"},{"id",trace_id},
                                     {"len",(uint64_t)save_at},
                                     {"mode",snapshot_hint?"hint":"auto"}});
                        save_at=0; saved_snapshot=true;
                    }
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
        // LRU enforcement is pure file I/O — outside the lease.
        if(saved_snapshot) {
            const auto ev=snapstore.evict_past_budget();
            if(ev.first) trace.event({{"kind","snapshot_evict"},{"id",trace_id},
                                      {"files",ev.first},{"bytes",ev.second}});
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
        tc.enabled=constrain_tools && !tool_names.empty() && sampling.temperature==0.0f && !mtp_width && !suffix_width;
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
                spec_rounds_total.fetch_add(1,std::memory_order_relaxed);
                spec_committed_total.fetch_add(committed.size(),std::memory_order_relaxed);
                for(uint32_t token:committed) {
                    if(token==eos_id) { cause=q27::MetalEngine::StopCause::Eos; stopped=true; break; }
                    if(!deliver(token)) { stopped=true; break; }
                }
            }
        } else if(sfx && engine.chunked_prefill()) {
            // Suffix-burst decode (2026-07-16-suffix-burst-verify.md, server
            // integration): drafter state is CPU-side, seeded from the FULL
            // prompt — including any restored prefix, which never reached
            // this engine's step loop — then fed every committed token by
            // suffix_step. One suffix_step per lease keeps the MTP branch's
            // quantum discipline; the real eos id rides into the round's
            // lane clamp, so an eos inside a burst commits and stops without
            // encoding past it (gate 5, live EOS).
            q27::SuffixDraft drafter;
            {
                std::vector<int> history(prompt.begin(),prompt.end());
                drafter.reset(history);
            }
            std::vector<uint32_t> committed;
            bool stopped=false;
            while(!stopped && produced<count) {
                if(produced+1==count) {
                    if(pending!=eos_id) deliver(pending);
                    else cause=q27::MetalEngine::StopCause::Eos;
                    break;
                }
                bool burst=false;
                {
                    // The drafter's propose/append inside suffix_step is
                    // host work under the lease — accepted deliberately
                    // (codex P2 on this change): it is bounded integer
                    // compares, microseconds against a multi-ms GPU round,
                    // nothing like the whole-vocab mask simulation that
                    // forced the constrained path's pre-lease prewarm.
                    // Splitting the round across the lease boundary would
                    // move burst policy back out of the engine.
                    auto gpu=lease_now();
                    pending=engine.suffix_step(drafter,pending,count-produced,eos_id,
                                               suffix_width,q27::MetalEngine::SUFFIX_MIN_MATCH,
                                               committed,&burst);
                }
                spec_rounds_total.fetch_add(1,std::memory_order_relaxed);
                spec_committed_total.fetch_add(committed.size(),std::memory_order_relaxed);
                (burst?suffix_burst_rounds_total:suffix_fallback_rounds_total)
                    .fetch_add(1,std::memory_order_relaxed);
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
        if(client_gone) {
            out.finish=Finish::Cancelled;
            trace.event({{"kind","cancel"},{"phase","generate"},{"id",trace_id}});
        }
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

// Per-endpoint defaults mirror the CUDA server (codex P3 on this round):
// /v1/messages 1024, OpenAI completions/chat 256, responses 4096.
// Q27_METAL_MAX_TOKENS_DEFAULT overrides all of them for requests that
// omit max_tokens (or send null, which json::value also defaults): pi.dev
// sends max_tokens:null, and the CUDA-parity 256 truncates real agent
// turns mid-answer ("maximum output token limit", 2026-07-17). Explicit
// client values always win; the context preflight still clamps to the
// remaining window.
uint32_t max_tokens(const json& body,long long dflt) {
    static const long long env_dflt=[]{
        const char* e=getenv("Q27_METAL_MAX_TOKENS_DEFAULT");
        return e&&*e?atoll(e):0ll;
    }();
    if(env_dflt>0) dflt=env_dflt;
    long long value=body.value("max_tokens",body.value("max_output_tokens",dflt));
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
        fprintf(stderr,"usage: %s model.q27 tokenizer.tok [--host 127.0.0.1] [--port 8080] [--ctx 8192] [--mtp 2..12 | --suffix 2..48] [--kv fp16|turbo3] [--prefix-entries N] [--constrain-tools] [--slots N] [--trace path]\n",argv[0]);
        return 1;
    }
    try {
        std::string model=argv[1],tok=argv[2],host="127.0.0.1";
        std::string trace_path;
        uint32_t port=8080,context=8192,width=0,suffix_width=0,prefix_entries=1,slot_count=2;
        bool turbo3=false; bool constrain_tools=false;
        for(int i=3;i<argc;i++) {
            std::string arg=argv[i];
            if(arg=="--host" && i+1<argc) host=argv[++i];
            else if(arg=="--port" && i+1<argc) port=parse_u32(argv[++i],"--port");
            else if(arg=="--ctx" && i+1<argc) context=parse_u32(argv[++i],"--ctx");
            else if(arg=="--mtp" && i+1<argc) width=parse_u32(argv[++i],"--mtp");
            else if(arg=="--suffix" && i+1<argc) suffix_width=parse_u32(argv[++i],"--suffix");
            else if(arg=="--prefix-entries" && i+1<argc) prefix_entries=parse_u32(argv[++i],"--prefix-entries");
            else if(arg=="--slots" && i+1<argc) slot_count=parse_u32(argv[++i],"--slots");
            else if(arg=="--kv" && i+1<argc) { std::string mode=argv[++i]; if(mode=="turbo3")turbo3=true; else if(mode!="fp16")throw std::runtime_error("invalid --kv"); }
            else if(arg=="--constrain-tools") constrain_tools=true;
            else if(arg=="--trace" && i+1<argc) trace_path=argv[++i];
            else throw std::runtime_error("unknown/incomplete argument: "+arg);
        }
        if(port>65535) throw std::runtime_error("port out of range");
        if(width && (width<2 || width>12)) throw std::runtime_error("MTP width must be 2..12");
        if(suffix_width && (suffix_width<2 || suffix_width>q27::MetalEngine::VERIFY_CHUNK_MAX))
            throw std::runtime_error("suffix width must be 2..48");
        // Bounded so entries x snapshot_bytes() can never wrap uint64 in the
        // G6 per-slot charge — side-inclusive snapshots (~4.6 GB at max ctx
        // under L7-full) put the wrap within uint32 entry range (codex P2 on
        // 4415c53); 4096 entries is already far beyond any real deployment.
        if(prefix_entries>4096) throw std::runtime_error("--prefix-entries must be 0..4096");
        if(width && suffix_width) throw std::runtime_error("--mtp and --suffix are mutually exclusive (one speculation lever per server)");
        if(constrain_tools && width) throw std::runtime_error("--constrain-tools requires serial decode; drop --mtp (verify-lane masks are not wired on Metal)");
        if(constrain_tools && suffix_width) throw std::runtime_error("--constrain-tools requires serial decode; drop --suffix (burst rounds argmax unmasked logits)");
        // Phase 1's latency guarantee (wait <= one active quantum) only
        // holds with one competing slot; >2 needs the scheduler and the
        // width/stats model extended first (codex P2 on d243f92).
        if(slot_count<1 || slot_count>2) throw std::runtime_error("--slots must be 1..2 in multislot Phase 1");
        Runtime runtime(model,tok,context,turbo3,width,suffix_width,prefix_entries,constrain_tools,slot_count);
        if(!trace_path.empty()) {
            runtime.trace.open(trace_path);
            runtime.trace.event({{"kind","boot"},{"ctx",context},{"kv",turbo3?"turbo3":"fp16"},
                                 {"mtp",width},{"suffix",suffix_width},{"slots",slot_count},
                                 {"model",runtime.model_name}});
        }
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
        server.Get("/health",[&runtime](const httplib::Request&,httplib::Response& r){json_response(r,{{"status","ok"},{"model",runtime.model_name}});});
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
                             {"queue_wait_by_arrival",queue},
                             {"speculation",{{"rounds",(uint64_t)runtime.spec_rounds_total},
                                             {"committed",(uint64_t)runtime.spec_committed_total},
                                             {"suffix_bursts",(uint64_t)runtime.suffix_burst_rounds_total},
                                             {"suffix_fallbacks",(uint64_t)runtime.suffix_fallback_rounds_total}}},
                             {"snapshots",{{"enabled",runtime.snapstore.enabled()},
                                           {"disk_hits",(uint64_t)runtime.snapstore.hits},
                                           {"disk_saves",(uint64_t)runtime.snapstore.saves}}},
                             {"cancellations",{{"queued",(uint64_t)runtime.cancelled_queue},
                                               {"prefill",(uint64_t)runtime.cancelled_prefill}}}});
        });
        server.Get("/v1/models",[](const httplib::Request&,httplib::Response& r){json_response(r,{{"object","list"},{"data",json::array({{{"id","q27-metal"},{"object","model"}}})}});});

        auto guarded=[&](auto handler) {
            return [&,handler](const httplib::Request& request,httplib::Response& response) {
                try { handler(json::parse(request.body),response,request.sock); }
                // 499 (client closed request): on a genuinely dead socket
                // the write fails harmlessly; on an is_socket_alive false
                // negative the client gets a parseable error instead of an
                // empty 200 (codex P2 on this round).
                catch(const Runtime::ClientGone&) { response.status=499; }
                catch(const Runtime::ServerOverloaded& e) { runtime.trace.event({{"kind","error"},{"status",503},{"type","overloaded_error"},{"message",e.what()}}); json_response(response,{{"error",{{"message",e.what()},{"type","overloaded_error"}}}},503); }
                catch(const Runtime::EngineError& e) { runtime.trace.event({{"kind","error"},{"status",500},{"type","api_error"},{"message",e.what()}}); json_response(response,{{"error",{{"message",e.what()},{"type","api_error"}}}},500); }
                catch(const std::exception& e) { runtime.trace.event({{"kind","error"},{"status",400},{"type","invalid_request_error"},{"message",e.what()}}); json_response(response,{{"error",{{"message",e.what()},{"type","invalid_request_error"}}}},400); }
            };
        };
        // Liveness probe for phases with no response writes yet (queue wait,
        // prefill) and for non-streaming generation. The socket fd rides
        // the q27 httplib patch (Request::sock).
        auto socket_live=[](socket_t sock){
            return [sock]{ return httplib::detail::is_socket_alive(sock); };
        };
        // Wraps ONLY a handler's run() call: engine failures reclassify as
        // EngineError (api_error 500); the cancellation and overload types
        // pass through untouched.
        auto engine_guard=[](auto&& fn)->decltype(fn()) {
            try { return fn(); }
            catch(const Runtime::ClientGone&) { throw; }
            catch(const Runtime::ServerOverloaded&) { throw; }
            catch(const Runtime::EngineError&) { throw; }
            catch(const std::exception& e) { throw Runtime::EngineError(e.what()); }
        };
        // Anthropic endpoints answer in Anthropic's error envelope
        // ({"type":"error","error":{...}} — the SDK inside Claude Code reads
        // error.message from it), not the OpenAI shape (codex P2 on this
        // round). Streaming errors after SSE commit use the error event.
        auto anthropic_guarded=[&](auto handler) {
            return [&,handler](const httplib::Request& request,httplib::Response& response) {
                json body;
                try { body=json::parse(request.body); }
                catch(...) {
                    response.status=400;
                    response.set_content(q27::anthropic_error_json("invalid_request_error","invalid JSON body"),"application/json");
                    return;
                }
                try { handler(body,response,request.sock); }
                // 499 as in `guarded` (codex P2): never an empty 200.
                catch(const Runtime::ClientGone&) { response.status=499; }
                catch(const Runtime::EngineError& e) {
                    runtime.trace.event({{"kind","error"},{"status",500},{"type","api_error"},{"message",e.what()}});
                    response.status=500;
                    response.set_content(q27::anthropic_error_json("api_error",e.what()),"application/json");
                }
                catch(const Runtime::ServerOverloaded& e) {
                    runtime.trace.event({{"kind","error"},{"status",503},{"type","overloaded_error"},{"message",e.what()}});
                    response.status=503;
                    response.set_content(q27::anthropic_error_json("overloaded_error",e.what()),"application/json");
                }
                catch(const std::exception& e) {
                    runtime.trace.event({{"kind","error"},{"status",400},{"type","invalid_request_error"},{"message",e.what()}});
                    response.status=400;
                    response.set_content(q27::anthropic_error_json("invalid_request_error",e.what()),"application/json");
                }
            };
        };

        // Shared context preflight: each endpoint refuses an oversized prompt
        // in its API's native 400 shape before slot claim / SSE commit.
        auto prompt_overflow=[&](size_t prompt_tokens,uint32_t& n,uint32_t& maxp)->bool {
            maxp=max_prompt_tokens(runtime.context,runtime.mtp_width,runtime.suffix_width);
            if(prompt_tokens>maxp) return true;
            if(prompt_tokens+n>runtime.context) n=runtime.context-(uint32_t)prompt_tokens;
            return false;
        };

        // ---- OpenAI /v1/completions (raw continuation; no template, no
        // tool protocol) ----
        server.Post("/v1/completions",guarded([&](const json& body,httplib::Response& r,socket_t sock){
            auto ids=to_u32(runtime.tokenizer.encode(body.value("prompt","")));
            uint32_t n=max_tokens(body,256);
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop");
            const std::string id="cmpl-metal-"+std::to_string((long)req_counter++);
            const long created=unix_now();
            if(ids.empty()) throw std::runtime_error("prompt is empty");
            uint32_t maxp=0;
            if(prompt_overflow(ids.size(),n,maxp)) {
                runtime.trace.event({{"kind","error"},{"status",400},{"type","context_length_exceeded"},
                    {"id",id},{"prompt_tokens",(uint64_t)ids.size()},{"max",maxp}});
                json_response(r,{{"error",{{"message",q27::ctx_limit_error_message((int)ids.size(),(int)maxp)},
                    {"type","invalid_request_error"},{"code","context_length_exceeded"}}}},400);
                return;
            }
            if(runtime.trace.enabled())
                // body.value("prompt","") repeats the encode line's identical
                // accessor verbatim — a non-string prompt throws THERE first
                // (guarded 400), so this event adds no new throw path (codex
                // P2 on the trace round, rejected with this evidence).
                runtime.trace.event({{"kind","request"},{"api","completions"},{"id",id},
                    {"stream",wants_stream(body)},{"prompt_tokens",(uint64_t)ids.size()},
                    {"max_tokens",n},{"rendered",trace_text(body.value("prompt",""))}});
            if(!wants_stream(body)) {
                std::string text; size_t probe=0;
                auto outcome=engine_guard([&]{
                    return runtime.run(ids,n,sampling,stops,
                        [&](const std::string& piece){ text+=piece;
                            return (++probe&15)?true:httplib::detail::is_socket_alive(sock); },
                        tool_names_from(body),body.value("snapshot",false),socket_live(sock),id); });
                runtime.trace.event({{"kind","outcome"},{"api","completions"},{"id",id},
                    {"finish",openai_finish(outcome.finish)},{"prompt_tokens",outcome.prompt_tokens},
                    {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                json_response(r,{{"id",id},{"object","text_completion"},{"created",created},{"model","q27-metal"},
                    {"choices",json::array({{{"index",0},{"text",text},{"finish_reason",openai_finish(outcome.finish)}}})},
                    {"usage",{{"prompt_tokens",outcome.prompt_tokens},{"completion_tokens",outcome.output_tokens},
                              {"total_tokens",outcome.prompt_tokens+outcome.output_tokens}}},
                    {"q27_prefix_hit",outcome.prefix_hit}});
                return;
            }
            r.set_header("Content-Type","text/event-stream");
            const std::vector<std::string> tnames=tool_names_from(body);
            const bool snap_hint=body.value("snapshot",false);
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,id,created,tnames,snap_hint,sock](size_t,httplib::DataSink& sink)->bool {
                    try {
                        auto emit=[&](const std::string& piece)->bool {
                            std::string s=q27::sse_data(
                                q27::openai_stream_chunk(false,id,"text_completion",created,"q27-metal",piece));
                            return sink.write(s.data(),s.size());
                        };
                        auto outcome=runtime.run(ids,n,sampling,stops,emit,tnames,snap_hint,
                            [sock]{ return httplib::detail::is_socket_alive(sock); },id);
                        // Terminal chunk with a real finish_reason before [DONE]
                        // (parity with server.cu security-review fix #7).
                        std::string fin=q27::sse_data(q27::openai_stream_final_chunk(
                            false,id,"text_completion",created,"q27-metal",openai_finish(outcome.finish)));
                        sink.write(fin.data(),fin.size());
                        std::string done=q27::sse_done(); sink.write(done.data(),done.size());
                        runtime.trace.event({{"kind","outcome"},{"api","completions"},{"id",id},
                            {"finish",openai_finish(outcome.finish)},{"prompt_tokens",outcome.prompt_tokens},
                            {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                    } catch(const Runtime::ClientGone&) {
                        return false;
                    } catch(const std::exception& e) {
                        std::string s=q27::sse_data({{"error",{{"message",e.what()},{"type","invalid_request_error"}}}});
                        sink.write(s.data(),s.size());
                    }
                    sink.done();
                    return true;
                });
        }));

        // ---- OpenAI /v1/chat/completions ----
        // Structured tool traffic both directions (agentic-parity round,
        // docs/plans/2026-07-17-metal-agentic-parity.md): incoming
        // assistant.tool_calls / role:"tool" via openai_msgs above; outgoing
        // <tool_call> segments become message.tool_calls (non-streaming) or
        // one delta.tool_calls chunk per call (streaming) with finish_reason
        // "tool_calls"; <think> segments go to reasoning_content (llama.cpp
        // convention) instead of leaking raw into content.
        server.Post("/v1/chat/completions",guarded([&](const json& body,httplib::Response& r,socket_t sock){
            bool think=body.value("enable_thinking",true);
            if(body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object())
                think=body["chat_template_kwargs"].value("enable_thinking",think);
            const json tools=body.contains("tools") && body["tools"].is_array()
                                 ?body["tools"]:json::array();
            const std::string rendered=q27::chatml_prompt(openai_msgs(body),tools,think);
            auto ids=to_u32(runtime.tokenizer.encode(rendered));
            uint32_t n=max_tokens(body,256);
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop");
            const long rid=req_counter++;
            const std::string id="chatcmpl-metal-"+std::to_string(rid);
            const long created=unix_now();
            if(ids.empty()) throw std::runtime_error("prompt is empty");
            uint32_t maxp=0;
            if(prompt_overflow(ids.size(),n,maxp)) {
                runtime.trace.event({{"kind","error"},{"status",400},{"type","context_length_exceeded"},
                    {"id",id},{"prompt_tokens",(uint64_t)ids.size()},{"max",maxp}});
                json_response(r,{{"error",{{"message",q27::ctx_limit_error_message((int)ids.size(),(int)maxp)},
                    {"type","invalid_request_error"},{"code","context_length_exceeded"}}}},400);
                return;
            }
            const bool has_tools=!tools.empty();
            const std::vector<std::string> tnames=tool_names_from(body);
            const bool snap_hint=body.value("snapshot",false);
            if(runtime.trace.enabled())
                runtime.trace.event({{"kind","request"},{"api","chat"},{"id",id},
                    {"stream",wants_stream(body)},{"prompt_tokens",(uint64_t)ids.size()},
                    {"max_tokens",n},{"tools",(uint64_t)tools.size()},
                    {"rendered",trace_text(rendered)}});
            if(!wants_stream(body)) {
                q27::StreamSplitter sp;
                std::string think_buf,text,tool_buf;
                std::vector<q27::ToolCall> calls;
                auto route=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                    if(ch==q27::StreamSplitter::TOOL) { tool_buf+=t; return; }
                    if(!tool_buf.empty()) {
                        calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));
                        tool_buf.clear();
                    }
                    (ch==q27::StreamSplitter::THINK?think_buf:text)+=t;
                };
                size_t probe=0;
                auto outcome=engine_guard([&]{
                    return runtime.run(ids,n,sampling,stops,
                        [&](const std::string& piece){ for(auto& [ch,t]:sp.feed(piece)) route(ch,t);
                            return (++probe&15)?true:httplib::detail::is_socket_alive(sock); },
                        tnames,snap_hint,socket_live(sock),id); });
                for(auto& [ch,t]:sp.flush()) route(ch,t);
                if(!tool_buf.empty()) calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));
                std::string th=q27::strip_ws2(think_buf),tx=q27::strip_ws2(text);
                // Malformed wrapped calls surface as text so nothing is lost;
                // then the wrapper-less recovery chain runs over the text.
                std::vector<q27::ToolCall> good;
                for(auto& c:calls) {
                    if(c.ok) good.push_back(std::move(c));
                    else tx+=(tx.empty()?"":"\n")+c.raw;
                }
                if(has_tools) {
                    std::string pre;
                    auto bcs=q27::parse_bare_tool_calls(tx,&pre,&tools);
                    if(!bcs.empty()) {
                        fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (chat nonstream)\n",bcs.size());
                        runtime.trace.event({{"kind","tool_recovery"},{"api","chat"},{"stream",false},{"count",bcs.size()}});
                        tx=pre;
                        for(auto& bc:bcs) good.push_back(std::move(bc));
                    }
                }
                json tcs=json::array();
                int ci=0;
                for(auto& c:good)
                    tcs.push_back({{"id","call_metal_"+std::to_string(rid)+"_"+std::to_string(ci++)},
                                   {"type","function"},
                                   {"function",{{"name",c.name},{"arguments",c.arguments.dump()}}}});
                json message={{"role","assistant"},
                              {"content",(!tcs.empty() && tx.empty())?json(nullptr):json(tx)}};
                if(!th.empty()) message["reasoning_content"]=th;
                if(!tcs.empty()) message["tool_calls"]=tcs;
                runtime.trace.event({{"kind","outcome"},{"api","chat"},{"id",id},
                    {"finish",!tcs.empty()?"tool_calls":openai_finish(outcome.finish)},
                    {"prompt_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                    {"prefix_hit",outcome.prefix_hit}});
                json_response(r,{{"id",id},{"object","chat.completion"},{"created",created},{"model","q27-metal"},
                    {"choices",json::array({{{"index",0},{"message",message},
                        {"finish_reason",!tcs.empty()?"tool_calls":openai_finish(outcome.finish)}}})},
                    {"usage",{{"prompt_tokens",outcome.prompt_tokens},{"completion_tokens",outcome.output_tokens},
                              {"total_tokens",outcome.prompt_tokens+outcome.output_tokens}}},
                    {"q27_prefix_hit",outcome.prefix_hit}});
                return;
            }
            r.set_header("Content-Type","text/event-stream");
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,id,rid,created,tools,has_tools,tnames,snap_hint,sock](size_t,httplib::DataSink& sink)->bool {
                    bool alive=true;
                    auto chunk=[&](const json& delta,const json& finish){
                        std::string s=q27::sse_data({{"id",id},{"object","chat.completion.chunk"},
                            {"created",created},{"model","q27-metal"},
                            {"choices",json::array({{{"index",0},{"delta",delta},{"finish_reason",finish}}})}});
                        if(!sink.write(s.data(),s.size())) alive=false;
                        return alive;
                    };
                    try {
                        // Opening role delta (OpenAI streaming convention);
                        // also the client-gone probe before generation starts.
                        if(!chunk({{"role","assistant"},{"content",""}},nullptr)) { sink.done(); return true; }
                        q27::StreamSplitter sp;
                        std::string tool_buf,text_accum;
                        int tool_counter=0;
                        bool any_call=false;
                        auto emit_tool=[&](){
                            auto c=q27::parse_tool_call(q27::strip_ws2(tool_buf));
                            tool_buf.clear();
                            if(!c.ok) { // malformed: surface as text so nothing is lost
                                text_accum+=c.raw;
                                chunk({{"content",c.raw}},nullptr);
                                return;
                            }
                            any_call=true;
                            chunk({{"tool_calls",json::array({{{"index",tool_counter},
                                {"id","call_metal_"+std::to_string(rid)+"_"+std::to_string(tool_counter)},
                                {"type","function"},
                                {"function",{{"name",c.name},{"arguments",c.arguments.dump()}}}}})}},nullptr);
                            tool_counter++;
                        };
                        auto emit_seg=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                            if(ch==q27::StreamSplitter::TOOL) { tool_buf+=t; return; }
                            if(!tool_buf.empty()) emit_tool();
                            if(t.empty()) return;
                            if(ch==q27::StreamSplitter::THINK) chunk({{"reasoning_content",t}},nullptr);
                            else { text_accum+=t; chunk({{"content",t}},nullptr); }
                        };
                        auto outcome=runtime.run(ids,n,sampling,stops,
                            [&](const std::string& piece)->bool {
                                for(auto& [ch,t]:sp.feed(piece)) emit_seg(ch,t);
                                return alive && sink.is_writable();
                            },tnames,snap_hint,
                            [sock]{ return httplib::detail::is_socket_alive(sock); },id);
                        for(auto& [ch,t]:sp.flush()) emit_seg(ch,t);
                        if(!tool_buf.empty()) emit_tool();
                        if(has_tools) {
                            // Wrapper-less recovery: the text already streamed
                            // as content deltas (cosmetic); the tool_calls
                            // chunks still fire so the client can execute.
                            std::string pre;
                            auto bcs=q27::parse_bare_tool_calls(text_accum,&pre,&tools);
                            if(!bcs.empty()) {
                                fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (chat stream)\n",bcs.size());
                                runtime.trace.event({{"kind","tool_recovery"},{"api","chat"},{"stream",true},{"count",bcs.size()}});
                            }
                            for(auto& bc:bcs) {
                                any_call=true;
                                chunk({{"tool_calls",json::array({{{"index",tool_counter},
                                    {"id","call_metal_"+std::to_string(rid)+"_"+std::to_string(tool_counter)},
                                    {"type","function"},
                                    {"function",{{"name",bc.name},{"arguments",bc.arguments.dump()}}}}})}},nullptr);
                                tool_counter++;
                            }
                        }
                        chunk(json::object(),any_call?"tool_calls":openai_finish(outcome.finish));
                        std::string done=q27::sse_done(); sink.write(done.data(),done.size());
                        runtime.trace.event({{"kind","outcome"},{"api","chat"},{"id",id},
                            {"finish",any_call?"tool_calls":openai_finish(outcome.finish)},
                            {"prompt_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                            {"prefix_hit",outcome.prefix_hit}});
                    } catch(const Runtime::ClientGone&) {
                        return false;
                    } catch(const std::exception& e) {
                        std::string s=q27::sse_data({{"error",{{"message",e.what()},{"type","invalid_request_error"}}}});
                        sink.write(s.data(),s.size());
                    }
                    sink.done();
                    return true;
                });
        }));

        // ---- Anthropic /v1/messages ----
        // Full agentic parity with src/server.cu:1087-1439 (2026-07-17 round):
        // request mapping via anthropic_msgs/anthropic_tools_json (incoming
        // tool_use/tool_result/thinking reconstructed, billing header
        // normalized), StreamSplitter output routing into thinking / text /
        // tool_use content blocks with input_json_delta streaming, bare-call
        // recovery, stop_reason "tool_use", and the "prompt is too long"
        // context refusal Claude Code keys compaction off.

        // CC calls count_tokens before compaction decisions; a 404 means it
        // estimates blind. Count = exactly what /v1/messages prefills for the
        // same body. CPU-only: no slot, no GPU lease.
        server.Post("/v1/messages/count_tokens",anthropic_guarded([&](const json& body,httplib::Response& r,socket_t){
            if(!body.contains("messages") || !body["messages"].is_array()) {
                r.status=400;
                r.set_content(q27::anthropic_error_json("invalid_request_error","messages: Field required"),
                              "application/json");
                return;
            }
            const std::string rendered=q27::chatml_prompt(
                q27::anthropic_msgs(body),q27::anthropic_tools_json(body),true);
            const long input_tokens=(long)runtime.tokenizer.encode(rendered).size();
            if(runtime.trace.enabled())
                runtime.trace.event({{"kind","request"},{"api","count_tokens"},{"id",""},
                    {"prompt_tokens",(uint64_t)input_tokens},{"rendered",trace_text(rendered)}});
            json_response(r,{{"input_tokens",input_tokens}});
        }));

        server.Post("/v1/messages",anthropic_guarded([&](const json& body,httplib::Response& r,socket_t sock){
            const json tools=q27::anthropic_tools_json(body);
            const std::string rendered=q27::chatml_prompt(q27::anthropic_msgs(body),tools,true);
            auto ids=to_u32(runtime.tokenizer.encode(rendered));
            uint32_t n=max_tokens(body,1024);
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop_sequences");
            const long rid=req_counter++;
            const std::string mid="msg_metal_"+std::to_string(rid);
            if(ids.empty()) throw std::runtime_error("prompt is empty");
            uint32_t maxp=0;
            if(prompt_overflow(ids.size(),n,maxp)) {
                fprintf(stderr,"[ctx-limit] prompt=%zu max=%u -> 400\n",ids.size(),maxp);
                runtime.trace.event({{"kind","error"},{"status",400},{"type","context_length_exceeded"},
                    {"id",mid},{"prompt_tokens",(uint64_t)ids.size()},{"max",maxp}});
                r.status=400;
                r.set_content(q27::anthropic_error_json("invalid_request_error",
                    q27::ctx_limit_error_message((int)ids.size(),(int)maxp)),"application/json");
                return;
            }
            const bool has_tools=tools.is_array() && !tools.empty();
            const std::vector<std::string> tnames=tool_names_from(body);
            const bool snap_hint=body.value("snapshot",false);
            if(runtime.trace.enabled())
                runtime.trace.event({{"kind","request"},{"api","messages"},{"id",mid},
                    {"stream",wants_stream(body)},{"prompt_tokens",(uint64_t)ids.size()},
                    {"max_tokens",n},{"tools",(uint64_t)(has_tools?tools.size():0)},
                    {"rendered",trace_text(rendered)}});
            if(!wants_stream(body)) {
                q27::StreamSplitter sp;
                std::string think,text,tool_buf;
                std::vector<q27::ToolCall> calls;
                auto route=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                    if(ch==q27::StreamSplitter::TOOL) { tool_buf+=t; return; }
                    if(!tool_buf.empty()) {
                        calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));
                        tool_buf.clear();
                    }
                    (ch==q27::StreamSplitter::THINK?think:text)+=t;
                };
                size_t probe=0;
                auto outcome=engine_guard([&]{
                    return runtime.run(ids,n,sampling,stops,
                        [&](const std::string& piece){ for(auto& [ch,t]:sp.feed(piece)) route(ch,t);
                            return (++probe&15)?true:httplib::detail::is_socket_alive(sock); },
                        tnames,snap_hint,socket_live(sock),mid); });
                for(auto& [ch,t]:sp.flush()) route(ch,t);
                if(!tool_buf.empty()) calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));
                json content=json::array();
                std::string th=q27::strip_ws2(think),tx=q27::strip_ws2(text);
                if(!th.empty())
                    content.push_back({{"type","thinking"},{"thinking",th},{"signature","q27-local"}});
                bool any_call=false;
                for(auto& c:calls) {
                    if(!c.ok) tx+=(tx.empty()?"":"\n")+c.raw; // malformed: keep as text
                    else any_call=true;
                }
                if(has_tools) {
                    // wrapper-less call recovery (see parse_bare_tool_calls)
                    std::string pre;
                    auto bcs=q27::parse_bare_tool_calls(tx,&pre,&tools);
                    if(!bcs.empty()) {
                        fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (nonstream)\n",bcs.size());
                        runtime.trace.event({{"kind","tool_recovery"},{"api","messages"},{"stream",false},{"count",bcs.size()}});
                        tx=pre;
                        for(auto& bc:bcs) calls.push_back(bc);
                        any_call=true;
                    }
                }
                if(!tx.empty() || (!any_call && th.empty()))
                    content.push_back({{"type","text"},{"text",tx}});
                int ci=0;
                for(auto& c:calls)
                    if(c.ok)
                        content.push_back({{"type","tool_use"},
                            {"id","toolu_metal_"+std::to_string(rid)+"_"+std::to_string(ci++)},
                            {"name",c.name},{"input",c.arguments}});
                json out={{"id",mid},{"type","message"},{"role","assistant"},{"model","q27-metal"},
                    {"content",content},
                    {"stop_reason",any_call?"tool_use":anthropic_stop(outcome.finish)},
                    {"stop_sequence",outcome.finish==Runtime::Finish::StopSequence?json(outcome.stop_sequence):json(nullptr)},
                    {"usage",{{"input_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens}}},
                    {"q27_prefix_hit",outcome.prefix_hit}};
                runtime.trace.event({{"kind","outcome"},{"api","messages"},{"id",mid},
                    {"finish",any_call?"tool_use":anthropic_stop(outcome.finish)},
                    {"prompt_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                    {"prefix_hit",outcome.prefix_hit}});
                json_response(r,out);
                return;
            }
            r.set_header("Content-Type","text/event-stream");
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,mid,rid,tools,has_tools,tnames,snap_hint,sock](size_t,httplib::DataSink& sink)->bool {
                    bool alive=true;
                    auto ev=[&](const char* name,const json& j){
                        std::string s=q27::sse_event(name,j);
                        if(!sink.write(s.data(),s.size())) alive=false;
                        return alive;
                    };
                    // Block bookkeeping mirrors server.cu's streaming handler:
                    // lazily opened think/text blocks, tool_use blocks emitted
                    // whole (start + one input_json_delta + stop) when a tool
                    // segment closes.
                    int block_counter=0,tool_counter=0,idx=-1,chan_open=-1;
                    bool any=false,any_call=false;
                    q27::StreamSplitter sp;
                    std::string tool_buf,text_accum;
                    auto close_block=[&](){
                        if(idx<0) return;
                        if(chan_open==1)
                            ev("content_block_delta",{{"type","content_block_delta"},{"index",idx},
                                {"delta",{{"type","signature_delta"},{"signature","q27-local"}}}});
                        ev("content_block_stop",{{"type","content_block_stop"},{"index",idx}});
                        idx=-1;
                    };
                    auto open_block=[&](int chan){
                        if(idx>=0 && chan_open!=chan) close_block();
                        if(idx<0) {
                            idx=block_counter++;
                            json cb=chan==1?json{{"type","thinking"},{"thinking",""}}
                                           :json{{"type","text"},{"text",""}};
                            ev("content_block_start",{{"type","content_block_start"},
                                {"index",idx},{"content_block",cb}});
                            chan_open=chan;
                            any=true;
                        }
                    };
                    auto emit_tool_block=[&](const std::string& name,const json& args){
                        any_call=true;
                        close_block();
                        const int ti=block_counter++;
                        const std::string tid="toolu_metal_"+std::to_string(rid)+"_"+
                                              std::to_string(tool_counter++);
                        ev("content_block_start",{{"type","content_block_start"},{"index",ti},
                            {"content_block",{{"type","tool_use"},{"id",tid},{"name",name},
                                              {"input",json::object()}}}});
                        ev("content_block_delta",{{"type","content_block_delta"},{"index",ti},
                            {"delta",{{"type","input_json_delta"},
                                      {"partial_json",q27::sse_dump(args)}}}});
                        ev("content_block_stop",{{"type","content_block_stop"},{"index",ti}});
                    };
                    auto emit_tool=[&](){
                        auto c=q27::parse_tool_call(q27::strip_ws2(tool_buf));
                        tool_buf.clear();
                        if(!c.ok) { // malformed: surface as text so nothing is lost
                            open_block(0);
                            text_accum+=c.raw;
                            ev("content_block_delta",{{"type","content_block_delta"},{"index",idx},
                                {"delta",{{"type","text_delta"},{"text",c.raw}}}});
                            return;
                        }
                        emit_tool_block(c.name,c.arguments);
                    };
                    auto emit_seg=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                        if(ch==q27::StreamSplitter::TOOL) { tool_buf+=t; return; }
                        if(!tool_buf.empty()) emit_tool();
                        if(t.empty()) return;
                        const int chan=ch==q27::StreamSplitter::THINK?1:0;
                        // suppress pure-whitespace text before/between blocks
                        if(chan==0 && idx<0 && q27::strip_ws2(t).empty()) return;
                        open_block(chan);
                        if(chan==0) text_accum+=t;
                        ev("content_block_delta",{{"type","content_block_delta"},{"index",idx},
                            {"delta",chan==1?json{{"type","thinking_delta"},{"thinking",t}}
                                            :json{{"type","text_delta"},{"text",t}}}});
                    };
                    try {
                        json msg={{"id",mid},{"type","message"},{"role","assistant"},{"model","q27-metal"},
                            {"content",json::array()},{"stop_reason",nullptr},{"stop_sequence",nullptr},
                            {"usage",{{"input_tokens",(int)ids.size()},{"output_tokens",0}}}};
                        // A client gone before generation starts must not hold
                        // the engine through the token cap: gate the run on the
                        // opening write, and probe the socket on quiet pieces.
                        if(!ev("message_start",{{"type","message_start"},{"message",msg}})) {
                            sink.done();
                            return true;
                        }
                        auto outcome=runtime.run(ids,n,sampling,stops,
                            [&](const std::string& piece)->bool {
                                for(auto& [ch,t]:sp.feed(piece)) emit_seg(ch,t);
                                return alive && sink.is_writable();
                            },tnames,snap_hint,
                            [sock]{ return httplib::detail::is_socket_alive(sock); },mid);
                        for(auto& [ch,t]:sp.flush()) emit_seg(ch,t);
                        if(!tool_buf.empty()) emit_tool();
                        if(has_tools) {
                            // wrapper-less recovery: text already streamed as
                            // text_delta (cosmetic); tool_use blocks still fire
                            std::string pre;
                            auto bcs=q27::parse_bare_tool_calls(text_accum,&pre,&tools);
                            if(!bcs.empty()) {
                                fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (stream)\n",bcs.size());
                                runtime.trace.event({{"kind","tool_recovery"},{"api","messages"},{"stream",true},{"count",bcs.size()}});
                                any=true;
                                for(auto& bc:bcs) emit_tool_block(bc.name,bc.arguments);
                            }
                        }
                        if(idx<0 && !any) { // nothing at all: empty text block for validity
                            idx=block_counter++;
                            chan_open=0;
                            ev("content_block_start",{{"type","content_block_start"},{"index",idx},
                                {"content_block",{{"type","text"},{"text",""}}}});
                        }
                        close_block();
                        ev("message_delta",{{"type","message_delta"},
                            {"delta",{{"stop_reason",any_call?"tool_use":anthropic_stop(outcome.finish)},
                                      {"stop_sequence",outcome.finish==Runtime::Finish::StopSequence?json(outcome.stop_sequence):json(nullptr)}}},
                            {"usage",{{"output_tokens",outcome.output_tokens}}}});
                        ev("message_stop",{{"type","message_stop"}});
                        runtime.trace.event({{"kind","outcome"},{"api","messages"},{"id",mid},
                            {"finish",any_call?"tool_use":anthropic_stop(outcome.finish)},
                            {"prompt_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                            {"prefix_hit",outcome.prefix_hit}});
                    } catch(const Runtime::ClientGone&) {
                        return false;
                    } catch(const std::exception& e) {
                        // First-class error event; message_stop still follows so
                        // naive clients get a well-formed stream (server.cu's
                        // batch-error convention).
                        ev("error",{{"type","error"},{"error",{{"type","api_error"},{"message",e.what()}}}});
                        ev("message_stop",{{"type","message_stop"}});
                    }
                    sink.done();
                    return true;
                });
        }));

        // ---- OpenAI Responses API (/v1/responses, Codex CLI) ----
        // Full CUDA port (src/server.cu:1441-1882, residue round
        // 2026-07-17-responses-parity-residue.md): instructions/input-item
        // mapping through the chat template (replacing the raw-text
        // preamble hack), custom freeform tools bridged to one-string-param
        // functions, hosted tool types skipped never rejected, and
        // structured function_call / custom_tool_call output items with the
        // codex 0.143 item lifecycle on the stream (an output_text.delta
        // without an open item aborts the codex turn). Wire facts from
        // codex-rs: the client keys off the JSON `type` field; the agent
        // loop consumes only response.output_item.done items;
        // response.completed{response:{id}} is the required terminator;
        // function_call.arguments is a JSON-encoded STRING. 400 is fatal
        // to codex, 500 retries — tolerate quirks, 500 on bugs.
        server.Post("/v1/responses",guarded([&](const json& body,httplib::Response& r,socket_t sock){
            const long rn=req_counter++;
            const std::string resp_id="resp_metal_"+std::to_string(rn);
            const std::string msg_id="msg_metal_"+std::to_string(rn);
            // Tools: flat function entries normalized to the nested shape
            // chatml_prompt renders; `custom` freeform tools (apply_patch)
            // bridged; hosted types (web_search etc.) skipped.
            json tools=json::array();
            std::set<std::string> custom_names;
            if(body.contains("tools") && body["tools"].is_array())
                for(const auto& t:body["tools"]) {
                    if(!t.is_object()) continue;
                    const std::string ty=t.value("type","");
                    if(t.contains("function") && t["function"].is_object()) tools.push_back(t);
                    else if(ty=="function" && t.contains("name"))
                        tools.push_back({{"type","function"},
                            {"function",{{"name",t.value("name","")},
                                         {"description",t.value("description","")},
                                         {"parameters",t.contains("parameters")?t["parameters"]
                                                                               :json::object()}}}});
                    else if(ty=="custom") {
                        const std::string cn=t.value("name","");
                        custom_names.insert(cn);
                        tools.push_back({{"type","function"},
                            {"function",{{"name",cn},
                                         {"description",t.value("description","")},
                                         {"parameters",{{"type","object"},
                                             {"properties",{{"input",{{"type","string"},
                                                 {"description","The complete raw input text for this tool."}}}}},
                                             {"required",json::array({"input"})}}}}}});
                    }
                }
            // input -> messages; instructions is the system prompt.
            std::vector<q27::Msg> msgs;
            if(body.contains("instructions") && body["instructions"].is_string())
                msgs.push_back({"system",body["instructions"]});
            if(body.contains("input")) {
                if(body["input"].is_string()) msgs.push_back({"user",body["input"]});
                else if(body["input"].is_array())
                    for(const auto& it:body["input"]) {
                        if(!it.is_object()) continue;
                        const std::string ty=it.value("type","message");
                        if(ty=="message") {
                            std::string role=it.value("role","user");
                            if(role=="developer") role="system";
                            msgs.push_back({role,it.contains("content")?text_content(it["content"]):""});
                        } else if(ty=="function_call" || ty=="custom_tool_call") {
                            json args;
                            if(ty=="function_call") {
                                try { args=json::parse(it.value("arguments","{}")); }
                                catch(...) { args=it.value("arguments",""); }
                            } else args={{"input",it.value("input","")}};
                            msgs.push_back({"assistant",q27::tool_call_text(it.value("name",""),args)});
                        } else if(ty=="function_call_output" || ty=="custom_tool_call_output") {
                            std::string out;
                            if(it.contains("output"))
                                out=it["output"].is_string()?it["output"].get<std::string>()
                                                            :text_content(it["output"]);
                            msgs.push_back({"user",q27::tool_response_text(out)});
                        }
                        // reasoning items in history are dropped (template behavior)
                    }
            }
            // Fail-loud before the template renders (codex P1 on the old
            // endpoint, preserved): tool declarations alone must not
            // generate.
            if(msgs.empty()) throw std::runtime_error("input is empty");
            std::vector<q27::Msg> merged;
            for(auto& m:msgs) {
                if(!merged.empty() && merged.back().role==m.role) merged.back().content+="\n"+m.content;
                else merged.push_back(m);
            }
            const std::string rendered=q27::chatml_prompt(merged,tools,true);
            auto ids=to_u32(runtime.tokenizer.encode(rendered));
            uint32_t n=max_tokens(body,4096);
            const q27::SamplingParams sampling=sampling_params(body);
            const std::vector<std::string> stops=parse_stops(body,"stop");
            if(ids.empty()) throw std::runtime_error("input is empty");
            uint32_t maxp=0;
            if(prompt_overflow(ids.size(),n,maxp)) {
                // context_length_exceeded is fatal-class for codex, correctly
                runtime.trace.event({{"kind","error"},{"status",400},{"type","context_length_exceeded"},
                    {"id",resp_id},{"prompt_tokens",(uint64_t)ids.size()},{"max",maxp}});
                json_response(r,{{"error",{{"code","context_length_exceeded"}}}},400);
                return;
            }
            const std::vector<std::string> tnames=tool_names_from(body);
            const bool snap_hint=body.value("snapshot",false);
            if(runtime.trace.enabled())
                runtime.trace.event({{"kind","request"},{"api","responses"},{"id",resp_id},
                    {"stream",wants_stream(body)},{"prompt_tokens",(uint64_t)ids.size()},
                    {"max_tokens",n},{"tools",(uint64_t)tools.size()},
                    {"rendered",trace_text(rendered)}});
            if(!wants_stream(body)) {
                json items=json::array();
                int tool_counter=0;
                std::string think,text,tool_buf,text_accum;
                auto flush_think=[&]{
                    std::string th=q27::strip_ws2(think); think.clear();
                    if(th.empty()) return;
                    items.push_back({{"type","reasoning"},{"id","rs_metal_"+std::to_string(rn)},
                        {"summary",json::array({{{"type","summary_text"},{"text",th}}})},
                        {"encrypted_content",nullptr}});
                };
                auto push_call=[&](const std::string& name,const json& args){
                    const std::string cid="call_metal_"+std::to_string(rn)+"_"+std::to_string(tool_counter++);
                    if(custom_names.count(name)) {
                        std::string input=args.is_object() && args.contains("input") && args["input"].is_string()
                                              ?args["input"].get<std::string>():args.dump();
                        items.push_back({{"type","custom_tool_call"},{"call_id",cid},{"name",name},{"input",input}});
                    } else
                        items.push_back({{"type","function_call"},{"call_id",cid},{"name",name},
                                         {"arguments",args.dump()}});
                };
                auto flush_text=[&](bool final_turn){
                    std::string tx=q27::strip_ws2(text); text.clear();
                    if(tx.empty()) return;
                    // Wrapper-less recovery rides EVERY text commit (codex P2
                    // round 2): a bare call completes within one segment
                    // before any think/tool transition, so per-segment
                    // recovery keeps coverage exact. Runs even with empty
                    // tools: codex registers its shell tool as a hosted type
                    // this handler skips, yet the model still emits bare
                    // calls for it. tx=pre convention (chat/Anthropic
                    // handlers): the pre-call prose message precedes the
                    // recovered calls, in model output order. Truncation
                    // repair is gated to the final end-of-turn flush (codex
                    // P2 round 3): a mid-turn segment boundary is not a
                    // truncation, so an incomplete call-shaped fragment
                    // before <think>/<tool_call> stays surfaced as text
                    // instead of being repaired into an invented call.
                    // Recorded coarseness (house convention, identical in
                    // the chat/Anthropic handlers): prose AFTER a recovered
                    // call within the same natural segment is trimmed by
                    // tx=pre — parse_bare_tool_calls reports no suffix.
                    std::string pre;
                    auto bcs=q27::parse_bare_tool_calls(tx,&pre,
                                                        tools.empty()?nullptr:&tools,
                                                        final_turn);
                    if(!bcs.empty()) {
                        fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (resp nonstream)\n",bcs.size());
                        runtime.trace.event({{"kind","tool_recovery"},{"api","responses"},{"stream",false},{"count",bcs.size()}});
                        tx=pre;
                    }
                    if(!tx.empty())
                        items.push_back({{"type","message"},{"id",msg_id},{"role","assistant"},
                            {"status","completed"},
                            {"content",json::array({{{"type","output_text"},{"text",tx},
                                                     {"annotations",json::array()}}})}});
                    for(auto& bc:bcs) push_call(bc.name,bc.arguments);
                };
                auto flush_tool=[&](bool final_turn){
                    auto c=q27::parse_tool_call(q27::strip_ws2(tool_buf)); tool_buf.clear();
                    if(!c.ok) { // malformed: commit the raw as its OWN text
                        // segment right now (chat/Anthropic recovery
                        // convention): recovery runs over it, so a
                        // recoverable nested call is not lost as raw text
                        // (codex P2 round 3) — and committing it alone means
                        // the tx=pre trim can never drop prose that followed
                        // the wrapper in the model's output (codex P2 round 4).
                        text+=(text.empty()?"":"\n")+c.raw;
                        flush_text(final_turn); return;
                    }
                    push_call(c.name,c.arguments);
                };
                q27::StreamSplitter sp;
                auto route=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                    if(ch==q27::StreamSplitter::TOOL) {
                        if(!think.empty()) flush_think();
                        if(!text.empty()) flush_text(false);
                        tool_buf+=t; return;
                    }
                    if(!tool_buf.empty()) flush_tool(false);
                    // codex P2: close pending text BEFORE think accumulates —
                    // same transition rule as TOOL, else a text→think→text
                    // turn interleaves items (reasoning pushed ahead of the
                    // message it followed) and the stream twin corrupts
                    // output_index.
                    if(ch==q27::StreamSplitter::THINK) { if(!text.empty()) flush_text(false); think+=t; return; }
                    if(!think.empty()) flush_think();
                    text+=t; text_accum+=t;
                };
                size_t probe=0;
                auto outcome=engine_guard([&]{
                    return runtime.run(ids,n,sampling,stops,
                        [&](const std::string& piece){ for(auto& [ch,t]:sp.feed(piece)) route(ch,t);
                            return (++probe&15)?true:httplib::detail::is_socket_alive(sock); },
                        tnames,snap_hint,socket_live(sock),resp_id); });
                for(auto& [ch,t]:sp.flush()) route(ch,t);
                if(!tool_buf.empty()) flush_tool(true);
                flush_think();
                flush_text(true);
                std::string all_text;
                for(const auto& it:items)
                    if(it.value("type","")=="message" && it.contains("content"))
                        for(const auto& c:it["content"])
                            if(c.value("type","")=="output_text") all_text+=c.value("text","");
                runtime.trace.event({{"kind","outcome"},{"api","responses"},{"id",resp_id},
                    {"finish","completed"},{"prompt_tokens",outcome.prompt_tokens},
                    {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                json_response(r,{{"id",resp_id},{"object","response"},{"model","q27-metal"},{"status","completed"},
                    {"output_text",all_text}, // Metal convenience field, pre-port consumers
                    {"output",items},
                    {"usage",{{"input_tokens",outcome.prompt_tokens},{"output_tokens",outcome.output_tokens},
                              {"total_tokens",outcome.prompt_tokens+outcome.output_tokens}}},
                    {"q27_prefix_hit",outcome.prefix_hit}});
                return;
            }
            r.set_header("Content-Type","text/event-stream");
            r.set_chunked_content_provider("text/event-stream",
                [&runtime,ids,n,sampling,stops,rn,resp_id,msg_id,tools,custom_names,tnames,snap_hint,sock](size_t,httplib::DataSink& sink)->bool {
                    bool alive=true;
                    auto ev=[&](const json& j){
                        std::string s=q27::sse_event(j.value("type",std::string("x")),j);
                        if(!sink.write(s.data(),s.size())) alive=false;
                        return alive;
                    };
                    // codex P3: item-lifecycle state + machinery hoisted
                    // above the try so the engine-failure path below can
                    // still close an open item and terminate the turn.
                    json items=json::array();
                    int tool_counter=0,out_index=0,msg_index=-1;
                    std::string think,text,tool_buf,text_accum;
                        auto item_done=[&](const json& it){
                            ev({{"type","response.output_item.done"},{"output_index",out_index++},{"item",it}});
                            items.push_back(it);
                        };
                        auto flush_think=[&]{
                            std::string th=q27::strip_ws2(think); think.clear();
                            if(th.empty()) return;
                            item_done({{"type","reasoning"},{"id","rs_metal_"+std::to_string(rn)},
                                {"summary",json::array({{{"type","summary_text"},{"text",th}}})},
                                {"encrypted_content",nullptr}});
                        };
                        // codex 0.143 item lifecycle: a delta needs an OPEN item
                        // (added + content_part.added), else the turn aborts.
                        auto open_text=[&]{
                            if(msg_index>=0) return;
                            msg_index=out_index;
                            ev({{"type","response.output_item.added"},{"output_index",msg_index},
                                {"item",{{"type","message"},{"id",msg_id},{"role","assistant"},
                                         {"status","in_progress"},{"content",json::array()}}}});
                            ev({{"type","response.content_part.added"},{"item_id",msg_id},
                                {"output_index",msg_index},{"content_index",0},
                                {"part",{{"type","output_text"},{"text",""},{"annotations",json::array()}}}});
                        };
                        auto flush_text=[&]{
                            if(msg_index<0) { text.clear(); return; }
                            std::string tx=q27::strip_ws2(text); text.clear();
                            ev({{"type","response.output_text.done"},{"item_id",msg_id},
                                {"output_index",msg_index},{"content_index",0},{"text",tx}});
                            ev({{"type","response.content_part.done"},{"item_id",msg_id},
                                {"output_index",msg_index},{"content_index",0},
                                {"part",{{"type","output_text"},{"text",tx},{"annotations",json::array()}}}});
                            json it={{"type","message"},{"id",msg_id},{"role","assistant"},{"status","completed"},
                                {"content",json::array({{{"type","output_text"},{"text",tx},
                                                         {"annotations",json::array()}}})}};
                            ev({{"type","response.output_item.done"},{"output_index",msg_index},{"item",it}});
                            items.push_back(it);
                            out_index=msg_index+1; msg_index=-1;
                        };
                        auto push_call=[&](const std::string& name,const json& args){
                            const std::string cid="call_metal_"+std::to_string(rn)+"_"+std::to_string(tool_counter++);
                            if(custom_names.count(name)) {
                                std::string input=args.is_object() && args.contains("input") && args["input"].is_string()
                                                      ?args["input"].get<std::string>():args.dump();
                                item_done({{"type","custom_tool_call"},{"call_id",cid},{"name",name},{"input",input}});
                            } else
                                item_done({{"type","function_call"},{"call_id",cid},{"name",name},
                                           {"arguments",args.dump()}});
                        };
                        auto flush_tool=[&](bool final_turn){
                            auto c=q27::parse_tool_call(q27::strip_ws2(tool_buf)); tool_buf.clear();
                            if(!c.ok) {
                                // A max_tokens-truncated FINAL wrapper never
                                // reached text_accum (TOOL content is not
                                // text), so the end-of-turn recovery below
                                // cannot see it — rescue it here with the
                                // truncation-repair path (codex P2 round 5).
                                // Goes beyond the CUDA reference, which
                                // commits the fragment as a message; matches
                                // the non-stream handler's rescue. Non-final
                                // malformed raw keeps the reference behavior.
                                if(final_turn) {
                                    std::string pre;
                                    auto bcs=q27::parse_bare_tool_calls(c.raw,&pre,
                                                                        tools.empty()?nullptr:&tools,true);
                                    if(!bcs.empty()) {
                                        fprintf(stderr,"[tool-fallback] %zu truncated wrapped call(s) recovered (resp stream)\n",bcs.size());
                                        runtime.trace.event({{"kind","tool_recovery"},{"api","responses"},{"stream",true},{"truncated_wrapper",true},{"count",bcs.size()}});
                                        if(!pre.empty())
                                            item_done({{"type","message"},{"role","assistant"},{"status","completed"},
                                                {"content",json::array({{{"type","output_text"},{"text",pre},
                                                                         {"annotations",json::array()}}})}});
                                        for(auto& bc:bcs) push_call(bc.name,bc.arguments);
                                        return;
                                    }
                                }
                                item_done({{"type","message"},{"role","assistant"},{"status","completed"},
                                    {"content",json::array({{{"type","output_text"},{"text",c.raw},
                                                             {"annotations",json::array()}}})}});
                                return;
                            }
                            push_call(c.name,c.arguments);
                        };
                        auto route=[&](q27::StreamSplitter::Chan ch,const std::string& t){
                            if(ch==q27::StreamSplitter::TOOL) {
                                if(!think.empty()) flush_think();
                                if(!text.empty()) flush_text();
                                tool_buf+=t; return;
                            }
                            if(!tool_buf.empty()) flush_tool(false);
                            // codex P2: a THINK transition must close an open
                            // text item first (same rule as TOOL) — else
                            // flush_think's item_done consumes the still-open
                            // message's output_index and the done events
                            // duplicate/reorder indices.
                            if(ch==q27::StreamSplitter::THINK) { if(!text.empty()) flush_text(); think+=t; return; }
                            if(!think.empty()) flush_think();
                            if(msg_index<0 && text.empty() && q27::strip_ws2(t).empty()) return;
                            open_text();
                            text+=t; text_accum+=t;
                            ev({{"type","response.output_text.delta"},{"item_id",msg_id},
                                {"output_index",msg_index},{"content_index",0},{"delta",t}});
                        };
                        try {
                        if(!ev({{"type","response.created"},
                                {"response",{{"id",resp_id},{"object","response"},{"status","in_progress"}}}})) {
                            sink.done(); return true;
                        }
                        q27::StreamSplitter sp;
                        auto outcome=runtime.run(ids,n,sampling,stops,
                            [&](const std::string& piece)->bool {
                                for(auto& [ch,t]:sp.feed(piece)) route(ch,t);
                                return alive && sink.is_writable();
                            },tnames,snap_hint,
                            [sock]{ return httplib::detail::is_socket_alive(sock); },resp_id);
                        for(auto& [ch,t]:sp.flush()) route(ch,t);
                        if(!tool_buf.empty()) flush_tool(true);
                        flush_think();
                        flush_text();
                        { // bare-call recovery even with empty tools (CUDA comment)
                            std::string pre;
                            auto bcs=q27::parse_bare_tool_calls(text_accum,&pre,
                                                                tools.empty()?nullptr:&tools);
                            if(!bcs.empty()) {
                                fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (resp stream)\n",bcs.size());
                                runtime.trace.event({{"kind","tool_recovery"},{"api","responses"},{"stream",true},{"count",bcs.size()}});
                            }
                            for(auto& bc:bcs) push_call(bc.name,bc.arguments);
                        }
                        runtime.trace.event({{"kind","outcome"},{"api","responses"},{"id",resp_id},
                            {"finish","completed"},{"prompt_tokens",outcome.prompt_tokens},
                            {"output_tokens",outcome.output_tokens},{"prefix_hit",outcome.prefix_hit}});
                        ev({{"type","response.completed"},
                            {"response",{{"id",resp_id},{"object","response"},{"status","completed"},
                                {"output",items},
                                {"usage",{{"input_tokens",outcome.prompt_tokens},
                                          {"input_tokens_details",{{"cached_tokens",0}}},
                                          {"output_tokens",outcome.output_tokens},
                                          {"output_tokens_details",{{"reasoning_tokens",0}}},
                                          {"total_tokens",outcome.prompt_tokens+outcome.output_tokens}}}}}});
                    } catch(const Runtime::ClientGone&) {
                        return false;
                    } catch(const std::exception& e) {
                        // codex P3: an engine failure mid-stream must not
                        // leave codex holding an unterminated item lifecycle
                        // over a 200 stream. Close any open item (the
                        // flushers emit the done triplet; no-ops when
                        // nothing is open — a pending tool buffer is dropped
                        // as unreliable), keep the first-class api_error
                        // event (Anthropic-stream precedent), then end the
                        // turn with the Responses-spec failure terminator
                        // carrying the partial output. Weak gate, recorded in
                        // the plan: no forced-failure failpoint on this path.
                        try { if(!think.empty()) flush_think(); flush_text(); } catch(...) {}
                        ev({{"type","error"},{"error",{{"type","api_error"},{"message",e.what()}}}});
                        ev({{"type","response.failed"},
                            {"response",{{"id",resp_id},{"object","response"},{"status","failed"},
                                {"last_error",{{"code","server_error"},{"message",e.what()}}},
                                {"output",items}}}});
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
