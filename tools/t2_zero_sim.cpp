// t2_zero_sim: Phase-0 probe for k3 roadmap item #3 (group-skip over ternary
// zeros; adoption recorded in d5a5674). CPU-only pass over a .q27 pack: for
// every T2_G128 matrix tensor, count zero codes and all-zero 128-column
// groups, then price the group-skip design. A skipped group saves 32 data +
// 2 scale bytes; the bitmap costs 1 bit per group over ALL groups.
// Pre-registered kill line: net byte saving < 8% of total T2 data+scale
// bytes -> KILL before writing a kernel.
//
// T2_G128 decode (FORMAT.md): element i of a row -> byte i/4, 2-bit field at
// bit offset (i%4)*2, LSB-first. code 1 = zero; 0/2 = -/+scale; 3 forbidden.
// cols % 128 == 0 is a loader invariant, so the data blob tiles exactly into
// consecutive 32-byte groups; an all-zero group is 32 bytes of 0x55.
#include "loader.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <string>

namespace {

struct Stats {
    uint64_t tensors = 0, elems = 0, zeros = 0, code3 = 0;
    uint64_t groups = 0, zgroups = 0;
    uint64_t hist[6] = {0, 0, 0, 0, 0, 0}; // 0 | 1-31 | 32-63 | 64-95 | 96-127 | 128
    void add(const Stats& o) {
        tensors += o.tensors; elems += o.elems; zeros += o.zeros; code3 += o.code3;
        groups += o.groups; zgroups += o.zgroups;
        for (int i = 0; i < 6; i++) hist[i] += o.hist[i];
    }
    // data + scale bytes this class contributes to the pack
    double t2_bytes() const { return elems / 4.0 + groups * 2.0; }
    // group-skip: +34 B per skipped group, -1 bit per group (all groups)
    double net_saved_bytes() const { return zgroups * 34.0 - groups / 8.0; }
};

// Class from name substrings, checked in order (attn_qkv. is the GDN
// in-projection despite the attn_ prefix, so gdn must match first):
//   gdn   : "attn_qkv." or "ssm_"
//   attn  : "attn_"      (attn_q/attn_k/attn_v/attn_output matmuls)
//   ffn   : "ffn_"
//   other : rest (token_embd, output, ...)
int classify(const std::string& n) {
    if (n.find("attn_qkv.") != std::string::npos || n.find("ssm_") != std::string::npos) return 2;
    if (n.find("attn_") != std::string::npos) return 0;
    if (n.find("ffn_") != std::string::npos) return 1;
    return 3;
}
const char* kClassName[4] = {"attn", "ffn", "gdn", "other"};

// histogram bucket for a per-group zero count z in [0,128]
int bucket(int z) { return z == 0 ? 0 : (z == 128 ? 5 : 1 + z / 32); }

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.q27\n", argv[0]);
        return 1;
    }
    q27::Model m = q27::Model::open(argv[1]);

    // per-byte lookup: zero-code (01) and forbidden-code (11) counts in 4 fields
    uint8_t zt[256], ft[256];
    for (int b = 0; b < 256; b++) {
        int z = 0, f = 0;
        for (int s = 0; s < 8; s += 2) {
            int c = (b >> s) & 3;
            z += (c == 1);
            f += (c == 3);
        }
        zt[b] = (uint8_t)z;
        ft[b] = (uint8_t)f;
    }

    Stats cls[4], all;
    uint64_t skipped_1d = 0;
    for (const auto& t : m.tensors) {
        if (t.dtype != q27::DType::T2_G128) continue;
        if (t.shape.size() < 2) { skipped_1d++; continue; } // matrix tensors only

        Stats& s = cls[classify(t.name)];
        s.tensors++;
        s.elems += t.n_elements();
        const uint8_t* p = t.data;
        const uint64_t ngroups = t.data_size / 32; // rows*cols/128, contiguous tiling
        s.groups += ngroups;
        for (uint64_t g = 0; g < ngroups; g++, p += 32) {
            int z = 0, f = 0;
            for (int i = 0; i < 32; i++) { z += zt[p[i]]; f += ft[p[i]]; }
            s.zeros += z;
            s.code3 += f;
            s.hist[bucket(z)]++;
            s.zgroups += (z == 128); // z==128 <=> 32 bytes of 0x55
        }
    }
    for (int i = 0; i < 4; i++) all.add(cls[i]);
    if (all.elems == 0) {
        fprintf(stderr, "no T2_G128 matrix tensors in pack (not a ternary artifact?)\n");
        return 1;
    }
    if (skipped_1d) printf("note: skipped %" PRIu64 " 1-D T2 tensors\n", skipped_1d);
    if (all.code3)
        printf("*** %" PRIu64 " FORBIDDEN code-3 fields — corrupt pack or LAYOUT-READ BUG ***\n",
               all.code3);

    printf("class patterns: gdn=attn_qkv./ssm_*  attn=attn_*  ffn=ffn_*  other=rest\n\n");
    printf("%-7s %7s %15s %8s %13s %13s %10s\n", "class", "tensors", "elements", "zero%",
           "in-zgrp%all", "in-zgrp%zero", "net-save%");
    for (int i = 0; i <= 4; i++) {
        const Stats& s = (i < 4) ? cls[i] : all;
        if (s.elems == 0) continue;
        double zgel = s.zgroups * 128.0; // elements sitting in all-zero groups
        printf("%-7s %7" PRIu64 " %15" PRIu64 " %7.2f%% %12.2f%% %12.2f%% %9.2f%%\n",
               (i < 4) ? kClassName[i] : "OVERALL", s.tensors, s.elems,
               100.0 * s.zeros / s.elems, 100.0 * zgel / s.elems,
               s.zeros ? 100.0 * zgel / s.zeros : 0.0,
               100.0 * s.net_saved_bytes() / s.t2_bytes());
    }

    // sanity vs the recorded 29.7% headroom number: a big miss means we are
    // decoding the layout wrong, not discovering anything
    double zf = 100.0 * all.zeros / all.elems;
    if (fabs(zf - 29.7) > 2.0)
        printf("\n*** WARNING: overall zero-code fraction %.2f%% is far from the recorded 29.7%%"
               " — LAYOUT-READ BUG, not a discovery ***\n", zf);

    printf("\nper-group zero-code histogram (group = 128 cols; distance from contiguity):\n");
    const char* kb[6] = {"0", "1-31", "32-63", "64-95", "96-127", "128"};
    for (int i = 0; i < 6; i++)
        printf("  %-7s %14" PRIu64 "  %6.2f%%\n", kb[i], all.hist[i],
               100.0 * all.hist[i] / all.groups);

    printf("\nnet saving %.2f%% — >= 8%% funds a kernel look / < 8%% KILL (pre-registered)\n",
           100.0 * all.net_saved_bytes() / all.t2_bytes());
    return 0;
}
