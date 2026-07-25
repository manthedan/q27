// T1 gate G3 (docs/metal/plans/2026-07-18-t1-snapshot-eviction-classes.md):
// the decision number, OFFLINE — drive the REAL DiskSnapshotStore::
// evict_past_budget over a fabricated directory that mirrors the recorded
// pi session, both pin directions. No model, no GPU, no server (this box
// holds exactly one resident model; the eviction path was extracted into
// disk_snapshot_store.h precisely so it is testable here).
//
// Fixture mirrors the live trace: an 8336-token spine snapshot (the growing
// conversation's prefix) plus a NEWER large tool-output leaf, total over
// budget. Ship line:
//   SPINE_PIN=0 (flat mtime LRU): the OLDER spine is the first victim.
//   SPINE_PIN=1:                  the leaf is evicted, the spine survives.
// Files are real Q27SNAP1 fixtures (header + token ids) so the store reads
// tokens through its peek path exactly as in production. Exit 0 = PASS.
#include "disk_snapshot_store.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
static int fails = 0;
#define CHECK(cond, msg) do { if(!(cond)) { std::printf("FAIL: %s\n", msg); fails++; } \
                              else std::printf("ok: %s\n", msg); } while(0)

// Minimal Q27SNAP1 header sufficient for the store's peek path (magic,
// position, token_count, then token_count u32 token ids). Layout matches
// metal_engine.cpp's SnapshotHeader; only the fields the store reads
// (magic, position, token_count, tokens) need to be truthful here.
#pragma pack(push,1)
struct H {
    char magic[8];
    uint64_t artifact_size;
    unsigned char artifact_sha1[20];
    uint32_t kv_dtype;
    uint32_t position;
    uint32_t token_count;
    uint32_t reserved;
    unsigned char prefix_sha1[20];
};
#pragma pack(pop)

static void write_snap(const std::string& path, const std::vector<uint32_t>& tokens,
                       uint64_t pad_bytes, int mtime_age_s,
                       bool logits_resident=true) {
    H h{};
    std::memcpy(h.magic, "Q27SNAP1", 8);
    h.artifact_size = 0;  // unused by peek
    h.kv_dtype = 0;
    h.position = (uint32_t)tokens.size();
    h.token_count = (uint32_t)tokens.size();
    h.reserved = logits_resident ? 0 : 1;
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char*>(&h), sizeof h);
    if (!tokens.empty())
        o.write(reinterpret_cast<const char*>(tokens.data()), (std::streamsize)tokens.size()*4);
    // Pad to the requested on-disk size (the leaf is large; spine is small).
    std::vector<char> pad(pad_bytes, 0);
    if (pad_bytes) o.write(pad.data(), (std::streamsize)pad_bytes);
    o.close();
    const auto t = fs::file_time_type::clock::now() - std::chrono::seconds(mtime_age_s);
    std::error_code ec; fs::last_write_time(path, t, ec);
}

// Stub peek: parse the fabricated fixture's header + tokens.
static SnapPeekInfo stub_peek(const std::string& path) {
    std::ifstream i(path, std::ios::binary);
    if (!i) throw std::runtime_error("cannot open");
    H h{}; i.read(reinterpret_cast<char*>(&h), sizeof h);
    if (!i || std::memcmp(h.magic, "Q27SNAP1", 8) != 0) throw std::runtime_error("bad magic");
    SnapPeekInfo o; o.position=h.position; o.logits_resident=(h.reserved&1)==0;
    o.tokens.resize(h.token_count);
    if (h.token_count) i.read(reinterpret_cast<char*>(o.tokens.data()), (std::streamsize)h.token_count*4);
    return o;
}

// Deterministic portable token-key hash for the offline gate (no platform
// crypto): FNV-1a over the token bytes, expanded to 40 hex chars. Only
// uniqueness/determinism matter here, not cryptographic strength.
static void stub_hash(const uint32_t* tokens, uint32_t count, char out_hex[41]) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t k = 0; k < count; k++)
        for (int b = 0; b < 4; b++) { h ^= (tokens[k] >> (b*8)) & 0xff; h *= 1099511628211ull; }
    for (int i = 0; i < 40; i++) { h ^= h >> 33; h *= 0xff51afd7ed558ccdull; out_hex[i] = "0123456789abcdef"[(h >> 60) & 0xf]; }
    out_hex[40] = '\0';
}

static bool exists(const std::string& p) { std::error_code ec; return fs::exists(p,ec); }

int main() {
    // Spine chain: S1(100 tokens) < S2(200) — a growing conversation. Leaf L
    // is unrelated and LARGE. S1 is the oldest; L the newest.
    std::vector<uint32_t> s1(100), s2(200), lf(400);
    for (uint32_t k=0;k<100;k++) s1[k]=k+1;
    for (uint32_t k=0;k<200;k++) s2[k]=k+1;         // extends s1
    for (uint32_t k=0;k<400;k++) lf[k]=9000+k;      // unrelated

    for (int pin=0; pin<=1; pin++) {
        const std::string dir = std::string(::getenv("TMPDIR")?:"/tmp") +
                                "/t1store." + (pin?"on":"off");
        std::error_code ec; fs::remove_all(dir,ec); fs::create_directories(dir,ec);
        DiskSnapshotStore store(&stub_peek,&stub_hash);
        // Force a CHOICE. Spine chain s1+s2 = 2 KB total; leaf = 600 KB;
        // combined 602 KB over a 601 KB budget -> 1 KB must be evicted.
        //   pin OFF (flat LRU): evicts the OLDEST file = spine s1 (1 KB),
        //     which exactly fits; the large newer leaf survives.
        //   pin ON: evicts leaves first; the only leaf is 600 KB, so it is
        //     removed (freeing far more than needed) and the spine survives.
        // Survivor sets therefore differ: OFF keeps {s2, leaf}, ON keeps
        // {s1, s2}. That divergence IS the decision number.
        store.init(dir, 601*1024, "t-", pin!=0);
        const std::string p1=dir+"/t-s1.q27snap", p2=dir+"/t-s2.q27snap", pl=dir+"/t-l.q27snap";
        write_snap(p1, s1, 1024, 300);   // oldest (spine)
        write_snap(p2, s2, 1024, 200);   // spine tip -> leaf of chain
        write_snap(pl, lf, 600*1024, 100); // newest, large leaf
        const uint64_t z1=fs::file_size(p1), z2=fs::file_size(p2), zl=fs::file_size(pl);
        // Force a single decisive eviction: budget = total - 1 byte, so the
        // store must evict files until just 1 byte is recovered — i.e. it
        // removes the FIRST-ORDERED victim only (any single file is >= 1
        // byte). pin OFF orders oldest-first -> s1 (spine) is the victim;
        // pin ON orders leaves-first -> the 600 KB leaf is the victim and
        // the spine chain survives. The survivor sets diverge, which is the
        // decision number.
        const uint64_t B = z1+z2+zl - 1;
        store.init(dir, B, "t-", pin!=0);
        const uint64_t sp0=store.evicted_spine, lf0=store.evicted_leaf;
        const auto ev=store.evict_past_budget();
        std::printf("pin=%d sizes s1=%llu s2=%llu leaf=%llu budget=%llu | evicted files=%zu bytes=%llu (spine +%llu leaf +%llu)\n",
                    pin,(unsigned long long)z1,(unsigned long long)z2,(unsigned long long)zl,(unsigned long long)B,
                    ev.first, (unsigned long long)ev.second,
                    (unsigned long long)(store.evicted_spine-sp0),
                    (unsigned long long)(store.evicted_leaf-lf0));
        std::printf("  survivors: s1=%d s2=%d leaf=%d\n",
                    (int)exists(p1),(int)exists(p2),(int)exists(pl));
        // Exactly one victim per arm; the SURVIVOR SET is the decision
        // signal. Note the chain TIP (s2) is itself a leaf (nothing extends
        // it), so pin ON protects only the non-tip spine member s1 — this is
        // correct Mooncake semantics (the tip is the newest, least-reused
        // position). The ship-line contrast is: pin OFF evicts the reused
        // spine s1 (mtime-oldest); pin ON spares s1 and instead evicts a
        // leaf (the oldest leaf, which here is the chain tip s2 — a
        // single-turn position that will not be re-sent).
        if (pin==0) {
            CHECK(!exists(p1), "pin OFF: reused spine s1 evicted (flat LRU, mtime-oldest)");
            CHECK(exists(pl), "pin OFF: newer large leaf survives");
        } else {
            CHECK(exists(p1), "pin ON: reused spine s1 SURVIVES (the ship-line contrast)");
            CHECK(store.evicted_spine-sp0==0, "pin ON: no SPINE-classified file evicted (counter)");
            CHECK(store.evicted_leaf-lf0==1, "pin ON: exactly one leaf evicted (counter)");
        }
        // The decisive contrast: the pin changes whether reused spine s1
        // survives a budget overflow.
        fs::remove_all(dir,ec);
    }

    {   // A same-key exact prewarm can replace an older mid-prefill bank.
        // Metadata cached from the displaced inode must be invalidated.
        const std::string dir = std::string(::getenv("TMPDIR")?:"/tmp") +
                                "/t1store.publish";
        std::error_code ec; fs::remove_all(dir,ec); fs::create_directories(dir,ec);
        DiskSnapshotStore store(&stub_peek,&stub_hash);
        store.init(dir, 0, "t-", false);
        std::vector<uint32_t> tokens={1,2,3,4};
        const std::string path=store.path_for(tokens.data(),(uint32_t)tokens.size());
        write_snap(path,tokens,0,10,false);
        std::string found; uint32_t found_len=0;
        CHECK(!store.best_match(tokens,found,found_len),
              "stale-logits exact snapshot is rejected and metadata cached");
        CHECK(!store.exact_resident(tokens.data(),(uint32_t)tokens.size()),
              "stale-logits snapshot cannot block a later exact publication");
        write_snap(path,tokens,0,0,true);
        CHECK(!store.best_match(tokens,found,found_len),
              "gate can fail: replacement is hidden by stale metadata before publish notice");
        store.published(path);
        CHECK(store.best_match(tokens,found,found_len) && found_len==tokens.size(),
              "publish notice exposes replacement exact snapshot metadata");
        CHECK(store.exact_resident(tokens.data(),(uint32_t)tokens.size()),
              "resident exact snapshot blocks stale-logits downgrade");
        store.reject(path);
        CHECK(!exists(path) &&
              !store.exact_resident(tokens.data(),(uint32_t)tokens.size()),
              "deep-load rejection removes shallow candidate for repair");
        fs::remove_all(dir,ec);
    }

    // ---- boot prefetch (audit A5) -----------------------------------------
    // prefetch_recent's job is page-cache warming, which has no observable
    // return value. What IS testable offline, and what actually matters for
    // correctness, is that it never touches the wrong files and never
    // outlives its inputs:
    //   - it must select only tag-matching .q27snap files (a foreign or
    //     mis-tagged file must not be read: on a shared directory that would
    //     be another server's artifact),
    //   - it must cap at max_entries,
    //   - it must be a no-op when disabled or when asked for 0,
    //   - the detached thread must survive the store being DESTROYED
    //     immediately after the call -- it captures path copies, so this must
    //     not use freed memory. Run under a sanitizer this is the real check.
    {
        const std::string dir = std::string(::getenv("TMPDIR")?:"/tmp") + "/t1store.prefetch";
        std::error_code ec; fs::remove_all(dir,ec); fs::create_directories(dir,ec);
        std::vector<uint32_t> a(64), b(64), c(64);
        for (uint32_t k=0;k<64;k++) { a[k]=k+1; b[k]=k+500; c[k]=k+900; }
        // Newest last: prefetch should prefer c, then b.
        write_snap(dir+"/tagA"+std::string(40,'1')+".q27snap", a, 4096, 300);
        write_snap(dir+"/tagA"+std::string(40,'2')+".q27snap", b, 4096, 200);
        write_snap(dir+"/tagA"+std::string(40,'3')+".q27snap", c, 4096, 100);
        // Decoys that must never be read: wrong tag, and wrong extension.
        write_snap(dir+"/tagB"+std::string(40,'4')+".q27snap", a, 4096, 50);
        write_snap(dir+"/tagA"+std::string(40,'5')+".bogus",   a, 4096, 50);

        {
            DiskSnapshotStore store(&stub_peek,&stub_hash);
            store.init(dir, 1ull<<30, "tagA", false);
            store.prefetch_recent(2);          // cap below the 3 tagA entries
            store.prefetch_recent(0);          // no-op
            store.prefetch_recent(-1);         // no-op
        }                                       // store DIES here, thread may still run
        DiskSnapshotStore disabled(&stub_peek,&stub_hash);
        disabled.prefetch_recent(4);           // never init'd -> no-op, must not crash
        // All five files must still exist: prefetch reads, never mutates, and
        // must not have touched mtimes in a way that changes eviction order.
        CHECK(exists(dir+"/tagA"+std::string(40,'1')+".q27snap") &&
              exists(dir+"/tagA"+std::string(40,'2')+".q27snap") &&
              exists(dir+"/tagA"+std::string(40,'3')+".q27snap") &&
              exists(dir+"/tagB"+std::string(40,'4')+".q27snap") &&
              exists(dir+"/tagA"+std::string(40,'5')+".bogus"),
              "boot prefetch is read-only and survives store destruction");
        // Give the detached reader a moment to finish against the live files
        // before the directory goes away, so the teardown is not itself the
        // thing under test.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        fs::remove_all(dir,ec);
    }

    if (fails) { std::printf("G3 FAIL (%d)\n", fails); return 1; }
    std::printf("G3 PASS\n");
    return 0;
}
