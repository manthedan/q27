#pragma once
// T1 spine-vs-leaf eviction ordering (docs/plans/2026-07-18-t1-snapshot-
// eviction-classes.md). Header-only and free of GPU/engine dependencies so
// the ordering is unit-testable offline (tools/test_snapshot_evict.cpp)
// without a resident model — this box holds exactly one.
//
// Semantics: a snapshot is SPINE iff its stored token ids are a STRICT
// prefix of another stored snapshot's (the conversation grew past it). With
// the pin on, eviction removes every leaf (mtime-oldest first) before any
// spine (mtime-oldest first). Recency is already persisted by the caller's
// touch-on-hit mtime, so within a class the order stays mtime-oldest first
// — the pin only reorders ACROSS the spine/leaf classes. Eviction only ever
// deletes files: a wrong victim costs a re-prefill, never a wrong result.

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace q27 {

struct EvictCandidate {
    std::string path;
    uint64_t size = 0;
    uint64_t mtime = 0;      // opaque; only ordering matters
    bool spine = false;
};

// Mark spine bits: candidate i is spine iff tokens[i] is a strict prefix of
// tokens[j] for some j != i. Entries missing from `tokens` stay leaves.
inline void mark_spine(std::vector<EvictCandidate>& files,
                       const std::map<std::string, std::vector<uint32_t>>& tokens) {
    for (size_t i = 0; i < files.size(); i++) {
        auto ai = tokens.find(files[i].path);
        if (ai == tokens.end()) { files[i].spine = false; continue; }
        const std::vector<uint32_t>& a = ai->second;
        bool spine = false;
        for (size_t j = 0; j < files.size(); j++) {
            if (i == j) continue;
            auto bi = tokens.find(files[j].path);
            if (bi == tokens.end()) continue;
            const std::vector<uint32_t>& b = bi->second;
            if (a.size() < b.size() && std::equal(a.begin(), a.end(), b.begin())) { spine = true; break; }
        }
        files[i].spine = spine;
    }
}

// Eviction victim order: with pin, leaves before spine (each mtime-oldest
// first); without pin, flat mtime-oldest first. stable_sort keeps mtime
// order within a class.
inline void eviction_order(std::vector<EvictCandidate>& files, bool spine_pin) {
    if (spine_pin) {
        std::stable_sort(files.begin(), files.end(), [](const EvictCandidate& x, const EvictCandidate& y) {
            if (x.spine != y.spine) return !x.spine;   // leaves first
            return x.mtime < y.mtime;
        });
    } else {
        std::stable_sort(files.begin(), files.end(), [](const EvictCandidate& x, const EvictCandidate& y) {
            return x.mtime < y.mtime;
        });
    }
}

} // namespace q27
