#pragma once
// Disk snapshot store (prefix snapshots Phase 2, docs/plans/2026-07-16-
// prefix-snapshots.md). Extracted from metal_server.cpp into its own header
// so the T1 eviction gate (tools/test_snapshot_evict_store.cpp) can drive
// the REAL store offline — this box holds exactly one resident model, so
// the eviction path must be testable without a server.
//
// token-prefix keyed: the server tokenizes every prompt itself, so "the
// snapshot's stored token ids are a prefix of this request's tokens" is
// exact and has no BPE-boundary hazard (recorded design deviation from
// ds4's byte-SHA1, which exists for stateless clients that retokenize).
// Files are named by the SHA1 of the token bytes, so an identical prefix
// overwrites rather than duplicates. LRU by mtime; hits touch the file.
// All file I/O runs OUTSIDE the GPU lease; only save_state/load_state
// (which read/write GPU buffers) go under it.
//
// The store reads snapshot metadata through a Peek functor injected at
// construction: production passes q27::MetalEngine::peek_snapshot; the
// offline gate passes a tiny stub that parses fabricated Q27SNAP1 fixtures.
// SnapshotInfo mirrors q27::MetalEngine::SnapshotInfo's two fields the
// store uses (position, logits_resident, tokens).

#include "snapshot_evict.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

struct SnapPeekInfo {
    uint32_t position = 0;
    bool logits_resident = true;
    std::vector<uint32_t> tokens;
};

class DiskSnapshotStore {
  public:
    // peek: path -> SnapPeekInfo (throws on unreadable/foreign file).
    using PeekFn = SnapPeekInfo (*)(const std::string&);
    // hash: token bytes -> 40-hex-char key (production: SHA1; the offline
    // gate: any deterministic hex). Injected so this header carries no
    // platform crypto dependency (autoreview P1: CommonCrypto is Apple-only,
    // and test-cpu builds this header on Linux/CUDA hosts too).
    using HashFn = void (*)(const uint32_t* tokens, uint32_t count, char out_hex[41]);

    explicit DiskSnapshotStore(PeekFn peek, HashFn hash) : peek_(peek), hash_(hash) {}

    // tag = artifact/KV identity prefix baked into every filename, so one
    // directory shared by different artifacts or fp16/turbo3 servers never
    // cross-matches or overwrites incompatible snapshots (codex P2 on
    // 607160e); the deep header identity check at load stays underneath.
    void init(std::string dir, uint64_t max_bytes, std::string tag, bool spine_pin=false) {
        dir_=std::move(dir); max_bytes_=max_bytes; tag_=std::move(tag); spine_pin_=spine_pin;
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
            const std::string pstr=e.path().string();
            SnapPeekInfo info;
            auto tc=meta_.find(pstr);
            if(tc!=meta_.end()) { info=tc->second; }
            else {
                try { info=peek_(pstr); }
                catch(...) { continue; }   // corrupt/foreign file: never a hit
                // Cache the FULL peek (position + logits_resident + tokens):
                // caching only tokens would fabricate logits_resident=true
                // for a mid-prefill (stale-logits) banked snapshot, letting a
                // later exact-length request resume from non-existent pending
                // logits (autoreview P2). The metadata checks below depend on
                // the real values.
                meta_[pstr]=info;
            }
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

    std::string path_for(const uint32_t* tokens,uint32_t count) {
        char hex[41];
        hash_(tokens,count,hex); hex[40]='\0';
        const std::string p=dir_+"/"+tag_+hex+".q27snap";
        // Register the key's tokens so the eviction spine check never
        // re-reads this file. NOT registered in meta_: path_for runs before
        // save_state writes the blob, so logits_resident is not yet known —
        // best_match must peek the finished file fresh (autoreview P2).
        std::lock_guard<std::mutex> lk(m_);
        if(tokens_.find(p)==tokens_.end()) tokens_[p]=std::vector<uint32_t>(tokens,tokens+count);
        return p;
    }

    // True only when the exact token-key pathname currently contains a
    // logits-resident snapshot with matching position/tokens. Callers use
    // this immediately before a lease-serialized stale-logits save so an
    // older save decision cannot downgrade a newly installed exact prefix.
    bool exact_resident(const uint32_t* tokens,uint32_t count) {
        if(!enabled() || !tokens || !count) return false;
        char hex[41]; hash_(tokens,count,hex); hex[40]='\0';
        const std::string path=dir_+"/"+tag_+hex+".q27snap";
        std::lock_guard<std::mutex> lk(m_);
        std::error_code ec;
        if(!std::filesystem::is_regular_file(path,ec)) return false;
        SnapPeekInfo info;
        auto cached=meta_.find(path);
        if(cached!=meta_.end()) info=cached->second;
        else {
            try { info=peek_(path); }
            catch(...) { return false; }
            meta_[path]=info;
        }
        return info.logits_resident && info.position==count &&
               info.tokens.size()==count &&
               std::equal(info.tokens.begin(),info.tokens.end(),tokens);
    }

    // A candidate that passed the shallow header peek but failed the engine's
    // full structural/configuration load is not usable. Remove it so future
    // requests do not repeatedly pay the failure and so a later save can
    // repair the same token key.
    void reject(const std::string& path) {
        std::lock_guard<std::mutex> lk(m_);
        std::error_code ec;
        std::filesystem::remove(path,ec);
        meta_.erase(path);
        tokens_.erase(path);
    }

    // Call after save_state atomically publishes/replaces a path. A prior
    // best_match may have cached metadata for the displaced inode (notably a
    // stale-logits mid-prefill bank); force the next lookup to inspect the
    // newly published file rather than retaining that verdict indefinitely.
    void published(const std::string& path) {
        std::lock_guard<std::mutex> lk(m_);
        meta_.erase(path);
    }

    // Budget enforcement until the directory fits. The just-written file is
    // deletable too — the budget is a hard cap, and the gate asserts the
    // total never exceeds it. Returns {files, bytes} removed so the --trace
    // stream can record the eviction decision.
    //
    // T1 (2026-07-18-t1-snapshot-eviction-classes.md): with spine_pin_, a
    // snapshot that is a strict token-prefix of another stored snapshot (a
    // growing conversation's spine) is evicted only AFTER every non-spine
    // (leaf) snapshot. Recency is already persisted by best_match()'s
    // touch-on-hit mtime, so within each class the order stays mtime-oldest
    // first — the pin only reorders ACROSS the spine/leaf classes. Eviction
    // only ever deletes files: a wrong victim costs a re-prefill, never a
    // wrong result (the restore path is untouched).
    std::pair<size_t,uint64_t> evict_past_budget() {
        if(!enabled() || !max_bytes_) return {0,0};
        std::lock_guard<std::mutex> lk(m_);
        // mtime folds to an opaque ordering key for the shared ordering
        // function (snapshot_evict.h — header-only so the T1 ordering is
        // unit-testable offline without a resident model).
        std::vector<q27::EvictCandidate> files; uint64_t total=0;
        std::error_code ec;
        for(const auto& e:std::filesystem::directory_iterator(dir_,ec)) {
            if(!e.is_regular_file() || e.path().extension()!=".q27snap") continue;
            const uint64_t sz=(uint64_t)e.file_size(ec);
            const auto mt=std::chrono::duration_cast<std::chrono::nanoseconds>(
                              e.last_write_time(ec).time_since_epoch()).count();
            files.push_back({e.path().string(),sz,(uint64_t)mt,false});
            total+=sz;
        }
        if(spine_pin_) {
            // Lazily read each file's stored token ids once (small next to
            // the state blobs), then mark spine = strict prefix of another.
            for(const auto& f:files)
                if(tokens_.find(f.path)==tokens_.end()) {
                    try { tokens_[f.path]=peek_(f.path).tokens; }
                    catch(...) { /* unreadable: stays a leaf, evictable */ }
                }
            q27::mark_spine(files,tokens_);
        }
        q27::eviction_order(files,spine_pin_);
        size_t n=0; uint64_t freed=0;
        for(const auto& f:files) {
            if(total<=max_bytes_) break;
            if(std::filesystem::remove(f.path,ec)) {
                total-=f.size; n++; freed+=f.size;
                tokens_.erase(f.path);
                meta_.erase(f.path);
                if(f.spine) evicted_spine++; else evicted_leaf++;
            }
        }
        return {n,freed};
    }

    std::atomic<uint64_t> hits{0}, saves{0}, evicted_spine{0}, evicted_leaf{0};
  private:
    PeekFn peek_;
    HashFn hash_;
    std::string dir_; uint64_t max_bytes_=0; std::string tag_; std::mutex m_;
    bool spine_pin_=false;
    std::map<std::string,std::vector<uint32_t>> tokens_;   // path -> token ids (eviction spine check)
    std::map<std::string,SnapPeekInfo> meta_;              // path -> full peek (best_match; real values only)
};
