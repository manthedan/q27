// T1 gate G1 (docs/metal/plans/2026-07-18-t1-snapshot-eviction-classes.md):
// spine-vs-leaf eviction ORDERING, offline — no model, no GPU, no server.
// This box holds exactly one resident model, so the ordering logic lives in
// header-only snapshot_evict.h and is tested here against synthetic inputs.
//
// Fixture: a growing conversation A < AB < ABC (spine chain) plus a large
// unrelated leaf D that is mtime-NEWER than the spine members.
//   pin off: flat mtime-oldest-first -> oldest spine A is the first victim;
//            newer leaf D is last.
//   pin on:  leaves before spine     -> leaf D is evicted before any spine
//            member despite being newer; spine A survives to the end.
// The gate asserts BOTH directions (the pin must change the victim) and
// exits non-zero on any failure.
#include "snapshot_evict.h"
#include <cstdio>
#include <string>
#include <vector>

using q27::EvictCandidate;

static int fails = 0;
#define CHECK(cond, msg) do { if(!(cond)) { std::printf("FAIL: %s\n", msg); fails++; } \
                              else std::printf("ok: %s\n", msg); } while(0)

// Build the fixture. Paths are labels only — no files are touched.
// mtime is an epoch-like ordering key: SMALLER = OLDER (evicted first under
// flat LRU). A=100 (oldest), AB=200, ABC=300, D=400 (newest).
static void fixture(std::vector<EvictCandidate>& files,
                    std::map<std::string, std::vector<uint32_t>>& tokens) {
    files = {{"A", 100, 100}, {"AB", 100, 200}, {"ABC", 100, 300}, {"D", 100, 400}};
    tokens["A"]   = {1};
    tokens["AB"]  = {1, 2};
    tokens["ABC"] = {1, 2, 3};
    tokens["D"]   = {9, 9, 9, 9};
}

static std::string join_order(const std::vector<EvictCandidate>& files) {
    std::string s;
    for (const auto& f : files) { if (!s.empty()) s += ","; s += f.path; }
    return s;
}

int main() {
    // --- classification ---
    {
        std::vector<EvictCandidate> files;
        std::map<std::string, std::vector<uint32_t>> tokens;
        fixture(files, tokens);
        q27::mark_spine(files, tokens);
        bool a=false, ab=false, abc=true, d=true;  // expected spine bits
        for (const auto& f : files) {
            if (f.path=="A") a=f.spine; else if (f.path=="AB") ab=f.spine;
            else if (f.path=="ABC") abc=f.spine; else if (f.path=="D") d=f.spine;
        }
        CHECK(a,   "A is spine (strict prefix of AB/ABC)");
        CHECK(ab,  "AB is spine (strict prefix of ABC)");
        CHECK(!abc,"ABC is leaf (nothing extends the chain tip)");
        CHECK(!d,  "D is leaf (unrelated)");
    }

    // --- pin off: flat mtime-oldest-first ---
    {
        std::vector<EvictCandidate> files;
        std::map<std::string, std::vector<uint32_t>> tokens;
        fixture(files, tokens);
        q27::mark_spine(files, tokens);
        q27::eviction_order(files, false);
        const std::string ord = join_order(files);
        std::printf("pin off order: %s\n", ord.c_str());
        CHECK(ord == "A,AB,ABC,D", "pin off: flat mtime -> A(oldest=100) first victim, D(newest=400) last");
    }

    // --- pin on: leaves before spine ---
    {
        std::vector<EvictCandidate> files;
        std::map<std::string, std::vector<uint32_t>> tokens;
        fixture(files, tokens);
        q27::mark_spine(files, tokens);
        q27::eviction_order(files, true);
        const std::string ord = join_order(files);
        std::printf("pin on order:  %s\n", ord.c_str());
        // Leaves are ABC(tip,300) and D(400); mtime-oldest leaf first: ABC then D;
        // then spine members mtime-oldest first: A(100) then AB(200).
        CHECK(ord == "ABC,D,A,AB", "pin on: leaves (ABC,D) evicted before spine (A,AB); spine A survives to the end");
        CHECK(files[0].path != "A", "pin on: spine A is NOT the first victim (the whole point)");
    }

    // --- the decisive contrast: pin must change the first victim ---
    {
        std::vector<EvictCandidate> off, on;
        std::map<std::string, std::vector<uint32_t>> t1, t2;
        fixture(off, t1); fixture(on, t2);
        q27::mark_spine(off, t1); q27::mark_spine(on, t2);
        q27::eviction_order(off, false);
        q27::eviction_order(on, true);
        CHECK(off.front().path == "A" && on.front().path != "A",
              "pin changes the first victim away from spine A");
    }

    // --- unreadable/missing tokens stay leaves (evictable) ---
    {
        std::vector<EvictCandidate> files = {{"X",100,150},{"A",100,100},{"AB",100,200}};
        std::map<std::string, std::vector<uint32_t>> tokens;  // X absent -> leaf
        tokens["A"]={1}; tokens["AB"]={1,2};
        q27::mark_spine(files, tokens);
        q27::eviction_order(files, true);
        std::string ord = join_order(files);
        std::printf("missing-token order: %s\n", ord.c_str());
        // X(leaf,150) before spine A(100) even though A is older: pin protects spine.
        CHECK(files[0].path=="X", "file with unreadable tokens treated as leaf, evicted before spine");
    }

    if (fails) { std::printf("G1 FAIL (%d)\n", fails); return 1; }
    std::printf("G1 PASS\n");
    return 0;
}
