// vgemm -- flat-in-W verify weight path. See vgemm.cuh for the why.
// The kernel is the design spike's, promoted: the arithmetic is NOT re-derived
// here (it was validated against gemv_q4_n/gemv_q8_n at rel 1e-6), only wrapped.
#include <algorithm>
#include <cstdlib>

#include "device_model.h"
#include "loader.h"
#include "vgemm.cuh"

namespace q27k {

static __device__ __forceinline__ void mma_s8(int& d0, int& d1, int& d2, int& d3, uint32_t a0,
                                              uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                              uint32_t b1) {
    const int z = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(z), "r"(z), "r"(z), "r"(z));
}

// MODE 0: store straight to the per-lane outputs (z == 1 -- no workspace, no
//         reduce node, deterministic by construction; this is the vocab head).
// MODE 1: store this z-slice's partial; k_reduce_z sums them in index order.
template <int MR, int DT, int MODE>
__global__ __launch_bounds__(256, 4) void k_vgemm(const uint8_t* __restrict__ W,
                                                  const __half* __restrict__ S,
                                                  __grid_constant__ const XLanes X,
                                                  __grid_constant__ const YLanes Y,
                                                  float* __restrict__ ws, int64_t rows,
                                                  int64_t cols, int T, int stages_per_z) {
    constexpr int NT = W_PLUMB;
    constexpr int KS = VG_KS;
    constexpr int WM = MR / 16;
    constexpr int KG = 8 / (WM * 2);
    constexpr int KB = KG * KS;
    constexpr int LDW = KB + 16, LDX = KB + 16;
    constexpr int XSC = KS / 32;      // x-scales per row per stage (group-32 activations)
    constexpr bool Q4IN = DT == 1, T2IN = DT == 2; // 0 = Q8, 1 = Q4, 2 = T2 (Bonsai 2)
    constexpr int NWS = Q4IN ? 2 : 1; // w-scales per row per stage (g64 vs g128)
    static_assert(KG >= 1 && KG * WM * 2 == 8, "warp split");

    extern __shared__ unsigned char smem_raw[];
    int8_t* s_w = (int8_t*)smem_raw;
    int8_t* s_x = (int8_t*)(s_w + MR * LDW);
    float* s_ws = (float*)(s_x + NT * LDX);
    float* s_xs = (float*)(s_ws + MR * KG * NWS);

    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid & 31;
    const int wm = warp % WM;
    const int wn = (warp / WM) % 2;
    const int kg = warp / (WM * 2);
    const int gid = lane >> 2, tg = lane & 3;

    const int64_t r0 = (int64_t)blockIdx.y * MR;
    const int n_stages = (int)(cols / KS);
    const int s_begin = blockIdx.z * stages_per_z;
    const int s_end = min(n_stages, s_begin + stages_per_z);
    if (s_begin >= s_end) return;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    constexpr int WBY = Q4IN ? (MR * KB / 2) : T2IN ? (MR * KB / 4) : (MR * KB);
    constexpr int SLD = (MR * KG * NWS + 255) / 256;
    constexpr int XSL = (NT * KG * XSC + 255) / 256;

    // E6a: stage W and X 16 BYTES AT A TIME, not 4. The bytes staged, the smem
    // contents and therefore every accumulation are unchanged -- only the width
    // of each global request and each smem store moves. The gain is memory-level
    // parallelism: an SM has a bounded number of outstanding load slots, so a
    // 4-byte LDG buys a quarter of the in-flight bytes a 16-byte one does, and
    // this tile is bandwidth-bound with nothing else to hide the shortfall.
    // Measured against a same-grid pure-read floor (tools/vgemm_e6).
    // T2 stages 8-byte chunks (32 codes) so the pass geometry stays Q4's:
    // half the bytes per request, same number of requests per row.
    constexpr int WCB = T2IN ? 8 : 16;              // bytes per weight chunk
    constexpr int WCPR = (Q4IN ? KB / 2 : T2IN ? KB / 4 : KB) / WCB; // chunks per row (Q4 8, Q8 16, T2 8)
    constexpr int WRPP = 256 / WCPR;                // rows covered per 256-thread pass
    constexpr int WV = MR / WRPP;                   // passes = chunks per thread (Q4 1, Q8 2, T2 1)
    constexpr int XCPR = KB / 16;                   // 16B chunks per lane row (16)
    static_assert(WV * 256 * WCB == WBY, "W stage is not an exact chunk per thread per pass");
    static_assert(256 / XCPR == NT && NT * KB == 256 * 16, "X stage is not one uint4 per thread");

    const int wrr = tid / WCPR, wch = tid % WCPR;
    const int xtt = tid / XCPR, xch = tid % XCPR;
    const int64_t wrstride = Q4IN ? (cols / 2) : T2IN ? (cols / 4) : cols;

    uint4 rw[WV], rx;
    float rws[SLD], rxs[XSL];

    auto load_stage = [&](int sst) {
        const int64_t k0 = (int64_t)sst * KS;
        const int64_t wk0 = Q4IN ? (k0 / 2) : T2IN ? (k0 / 4) : k0;
#pragma unroll
        for (int p = 0; p < WV; p++) {
            const int rr = wrr + p * WRPP;
            // 0x88 = the q4 zero point pre-bias (0x55 = T2's code 1), so padded
            // rows contribute 0.
            if constexpr (T2IN) {
                const uint2 v = (r0 + rr < rows)
                                    ? __ldg((const uint2*)(W + (r0 + rr) * wrstride + wk0 + wch * 8))
                                    : make_uint2(0x55555555u, 0x55555555u);
                rw[p] = make_uint4(v.x, v.y, 0u, 0u);
            } else {
                rw[p] = (r0 + rr < rows)
                            ? __ldg((const uint4*)(W + (r0 + rr) * wrstride + wk0 + wch * 16))
                            : (Q4IN ? make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u)
                                    : make_uint4(0u, 0u, 0u, 0u));
            }
        }
        // dead lanes are never dereferenced: X.nat[tt >= T] may be null.
        rx = xtt < T ? __ldg((const uint4*)(X.nat[xtt] + k0 + xch * 16))
                     : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int i = 0; i < SLD; i++) {
            int idx = i * 256 + tid;
            int rr = idx / (KG * NWS), g = idx % (KG * NWS);
            rws[i] = (idx < MR * KG * NWS && r0 + rr < rows)
                         ? __half2float(__ldg(S + (r0 + rr) * (cols / (Q4IN ? 64 : 128)) +
                                              k0 / (Q4IN ? 64 : 128) + g))
                         : 0.f;
        }
#pragma unroll
        for (int i = 0; i < XSL; i++) {
            int idx = i * 256 + tid;
            int tt = idx / (KG * XSC), cc = idx % (KG * XSC);
            rxs[i] = (idx < NT * KG * XSC && tt < T) ? __ldg(X.xs[tt] + k0 / 32 + cc) : 0.f;
        }
    };
    // one packed u32 (8 nibbles) -> 8 unpacked int8, with the GEMV's isum "-8"
    // bias folded in. Identical arithmetic to the pre-E6a per-u32 unpack.
    auto unpack8 = [](uint32_t p, uint32_t& a, uint32_t& b) {
        const uint32_t lo = p & 0x0F0F0F0Fu, hi = (p >> 4) & 0x0F0F0F0Fu;
        a = __vsub4(__byte_perm(lo, hi, 0x5140), 0x08080808u);
        b = __vsub4(__byte_perm(lo, hi, 0x7362), 0x08080808u);
    };
    // one interleaved T2 word (16 codes, kernels.cuh order) -> 16 sequential
    // int8 with the "-1" bias folded in: field 4b+{0,1,2,3} = e[2b], e[2b+1],
    // e[8+2b], e[9+2b], so the four masked extractions are the even/odd halves
    // of each 8-element run and __byte_perm re-interleaves them in order.
    auto unpack_t2 = [](uint32_t w, uint32_t& o0, uint32_t& o1, uint32_t& o2, uint32_t& o3) {
        const uint32_t M = 0x03030303u;
        const uint32_t A = w & M, B = (w >> 2) & M, C = (w >> 4) & M, D = (w >> 6) & M;
        o0 = __vsub4(__byte_perm(A, B, 0x5140), 0x01010101u); // e0..e3
        o1 = __vsub4(__byte_perm(A, B, 0x7362), 0x01010101u); // e4..e7
        o2 = __vsub4(__byte_perm(C, D, 0x5140), 0x01010101u); // e8..e11
        o3 = __vsub4(__byte_perm(C, D, 0x7362), 0x01010101u); // e12..e15
    };
    auto store_stage = [&]() {
#pragma unroll
        for (int p = 0; p < WV; p++) {
            const int rr = wrr + p * WRPP;
            if constexpr (T2IN) {
                // 8 packed bytes -> 32 unpacked, two STS.128 (wch*32 stays 16B-aligned)
                int8_t* dst = s_w + rr * LDW + wch * 32;
                uint4 o;
                unpack_t2(rw[p].x, o.x, o.y, o.z, o.w);
                *(uint4*)dst = o;
                unpack_t2(rw[p].y, o.x, o.y, o.z, o.w);
                *(uint4*)(dst + 16) = o;
            } else if constexpr (Q4IN) {
                // 16 packed bytes -> 32 unpacked, written as two STS.128. LDW is
                // a multiple of 16 and wch*32 is too, so both stay 16B-aligned.
                int8_t* dst = s_w + rr * LDW + wch * 32;
                uint4 o;
                unpack8(rw[p].x, o.x, o.y);
                unpack8(rw[p].y, o.z, o.w);
                *(uint4*)dst = o;
                unpack8(rw[p].z, o.x, o.y);
                unpack8(rw[p].w, o.z, o.w);
                *(uint4*)(dst + 16) = o;
            } else {
                *(uint4*)(s_w + rr * LDW + wch * 16) = rw[p];
            }
        }
        *(uint4*)(s_x + xtt * LDX + xch * 16) = rx;
#pragma unroll
        for (int i = 0; i < SLD; i++) {
            int idx = i * 256 + tid;
            if (idx < MR * KG * NWS) s_ws[idx] = rws[i];
        }
#pragma unroll
        for (int i = 0; i < XSL; i++) {
            int idx = i * 256 + tid;
            if (idx < NT * KG * XSC) s_xs[idx] = rxs[i];
        }
    };

    // Factored out so a -DQ27_VGEMM_NOMMA build can compile it away: that build
    // keeps the ENTIRE staging path (loads, unpack, smem stores, both barriers,
    // the intra-CTA reduce, the epilogue) and drops only the tensor work, which
    // is what splits the tile's cost into memory pipeline vs compute phase.
    // Measured 2026-08-19: 9.49 ms staging-only vs 10.21 ms full, against a
    // 9.24 ms same-grid floor -- 0.25 ms of pipeline, 0.72 ms of compute phase.
    // A -DQ27_VGEMM_NOTC build splits that 0.72 again: the tensor-core issue is
    // only 0.10 ms of it, the rest is these smem reads and the scaling chain.
    auto do_mma = [&]() {
        {
            const int kbase = kg * KS; // this warp-group's slice of the staged K
            const int tb = wn * 8;

            // E6c: hoist the SCALE loads out of the cc loop. SASS showed 40 LDS
            // per super-step, 16 of them scales -- for 6 distinct values that
            // ptxas never CSEd, because kb/64 and kb/32 hide the invariance
            // behind an integer divide it will not fold across the unroll.
            // Spelled out here it is 4 LDS: the two w-scales a Q4 row needs over
            // the whole cc range are ADJACENT (kb/64 = kg*2 + cc/2, so one
            // LDS.64), Q8 needs exactly one (kb/128 == kg for every cc), and the
            // four x-scales are consecutive (kb/32 = kg*4 + cc, so one LDS.128).
            // Same values, same multiply order, so bit-identical by inspection.
            constexpr int NWC = Q4IN ? 2 : 1; // distinct w-scales over the cc range
            float wsA[NWC], wsB[NWC];
            const int wr0 = wm * 16 + gid, wr1 = wm * 16 + gid + 8;
            if constexpr (Q4IN) {
                const float2 v0 = *(const float2*)&s_ws[wr0 * (KG * 2) + kg * 2];
                const float2 v1 = *(const float2*)&s_ws[wr1 * (KG * 2) + kg * 2];
                wsA[0] = v0.x; wsA[1] = v0.y;
                wsB[0] = v1.x; wsB[1] = v1.y;
            } else {
                wsA[0] = s_ws[wr0 * KG + kg];
                wsB[0] = s_ws[wr1 * KG + kg];
            }
            const float4 x0v = *(const float4*)&s_xs[(tb + tg * 2) * (KG * XSC) + kg * XSC];
            const float4 x1v = *(const float4*)&s_xs[(tb + tg * 2 + 1) * (KG * XSC) + kg * XSC];
            const float xs0v[4] = {x0v.x, x0v.y, x0v.z, x0v.w};
            const float xs1v[4] = {x1v.x, x1v.y, x1v.z, x1v.w};

#pragma unroll
            for (int cc = 0; cc < 4; cc++) {
                const int kb = kbase + cc * 32;
                // ldmatrix was tried here and REVERTED (BUILDLOG 08-19 (o)):
                // consistently -0.02 ms, but SASS instruction count was flat --
                // it re-encoded these six loads rather than removing work.
                const int8_t* wrow0 = s_w + wr0 * LDW + kb;
                const uint32_t a0 = *(const uint32_t*)(wrow0 + tg * 4);
                const uint32_t a1 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4);
                const uint32_t a2 = *(const uint32_t*)(wrow0 + tg * 4 + 16);
                const uint32_t a3 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4 + 16);
                const int8_t* xcol = s_x + (tb + gid) * LDX + kb;
                const uint32_t b0 = *(const uint32_t*)(xcol + tg * 4);
                const uint32_t b1 = *(const uint32_t*)(xcol + tg * 4 + 16);
                const float wsc0 = wsA[Q4IN ? cc / 2 : 0];
                const float wsc1 = wsB[Q4IN ? cc / 2 : 0];
                int d0, d1, d2, d3;
#ifndef Q27_VGEMM_NOTC
                mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
#else
                // attribution build (BUILDLOG 08-19 (o)): keep every fragment
                // load and the scaling chain live, drop ONLY the tensor-core
                // issue. 10.11 ms vs 10.21 full vs 9.49 staging-only -- i.e. the
                // tensor cores are 0.10 ms of the compute phase, not 0.72.
                d0 = (int)(a0 ^ b0); d1 = (int)(a1 ^ b1);
                d2 = (int)(a2 ^ b0); d3 = (int)(a3 ^ b1);
#endif
                const float xs0 = xs0v[cc], xs1 = xs1v[cc];
                acc[0] += wsc0 * xs0 * (float)d0;
                acc[1] += wsc0 * xs1 * (float)d1;
                acc[2] += wsc1 * xs0 * (float)d2;
                acc[3] += wsc1 * xs1 * (float)d3;
            }
        }
    };

    load_stage(s_begin);
    for (int sst = s_begin; sst < s_end; sst += KG) {
        __syncthreads();
        store_stage();
        if (sst + KG < s_end) load_stage(sst + KG);
        __syncthreads();
#ifndef Q27_VGEMM_NOMMA
        if (sst + kg < s_end) do_mma();
#else
        (void)do_mma; // attribution build: staging only, see the note above
#endif
    }

    // intra-CTA K-split reduce, FIXED WARP ORDER (no atomics, no shuffle races).
    // Aliases s_w -- dead by here, and 4KB <= MR*LDW. Nothing syncs after the
    // early return, so the divergent exit is legal.
    if constexpr (KG > 1) {
        float* red = (float*)smem_raw;
        __syncthreads();
        for (int e = 0; e < 4; e++) red[(warp * 32 + lane) * 4 + e] = acc[e];
        __syncthreads();
        if (kg != 0) return;
        const int b = wm + WM * wn;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            float s = 0.f;
            for (int g = 0; g < KG; g++) s += red[((b + g * WM * 2) * 32 + lane) * 4 + e];
            acc[e] = s;
        }
    }

    const int64_t row0 = r0 + wm * 16 + gid;
    const int tok0 = wn * 8 + tg * 2;
#pragma unroll
    for (int e = 0; e < 4; e++) {
        int64_t row = row0 + (e >= 2 ? 8 : 0);
        int tok = tok0 + (e & 1);
        if (row < rows && tok < T) {
            if constexpr (MODE == 1)
                ws[((size_t)blockIdx.z * T + tok) * rows + row] = acc[e];
            else
                Y.y[tok][row] = acc[e];
        }
    }
}

// Deterministic fixed-order reduce of the grid.z partials. i ascending, always.
__global__ void k_reduce_z(const float* __restrict__ ws, __grid_constant__ const YLanes Y,
                           int64_t rows, int T, int z) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (r >= rows || t >= T) return;
    float s = 0.f;
    for (int i = 0; i < z; i++) s += ws[((size_t)i * T + t) * rows + r];
    Y.y[t][r] = s;
}

// ---- host ----

// One wave is 170 SMs x 4 CTAs = 680 co-resident CTAs; aim at ~2 waves so the
// tail is cheap. Bounded by 8 (the deterministic partials must stay small) and by
// the K axis (never fewer than 4 stages per slice, or the prologue dominates).
static constexpr int VG_CTA_TARGET = 1400;
static constexpr int VG_ZMAX = 8;

// Bench-only override of the CTA target (Q27_VG_CTA_TARGET). Read ONCE, so every
// caller -- including vgemm_ws_bytes*, which must size the workspace for exactly
// the z the launch will use -- sees the same value for the life of the process.
// The default is the shipped 1400; setting it changes the number of grid.z
// partials and therefore the fp32 reduce order, so a non-default value is NOT
// bit-compatible with the recorded anchors.
static int vg_cta_target() {
    static const int v = [] {
        const char* e = getenv("Q27_VG_CTA_TARGET");
        int x = e ? atoi(e) : VG_CTA_TARGET;
        return x > 0 ? x : VG_CTA_TARGET;
    }();
    return v;
}

int vgemm_z(int64_t rows, int64_t cols) {
    const int VG_CTA_TARGET = vg_cta_target();
    const int row_ctas = (int)((rows + VG_MR - 1) / VG_MR);
    const int n_stages = (int)(cols / VG_KS);
    int z = (VG_CTA_TARGET + row_ctas - 1) / row_ctas;
    if (z < 1) z = 1;
    if (z > VG_ZMAX) z = VG_ZMAX;
    const int zk = n_stages / 4;
    if (z > zk) z = zk > 0 ? zk : 1;
    // Mirror the launcher's KB-align + trim so vgemm_ws_bytes() sizes the SAME z
    // the launch will actually use (an under-sized workspace is a heap overrun).
    int spz = (n_stages + z - 1) / z;
    spz = (spz + VG_KG - 1) / VG_KG * VG_KG;
    z = (n_stages + spz - 1) / spz;
    return z < 1 ? 1 : z;
}

size_t vgemm_ws_bytes_model(const q27::Model& m, int64_t min_rows) {
    size_t mx = 0;
    for (const q27::Tensor& t : m.tensors) {
        // T3_G128 never runs the vgemm (vgemm_verify refuses it), but the
        // workspace must exist: the conductor's union view asserts on it
        // (build_union_view), and a T3 pack's only other matmuls are the
        // Q8 MTP block -- a T3 body without one sized the workspace to zero
        // and every fused round died at the assert (2026-09-21, the first
        // 8 GB-card install without Q27_BATCH=0).
        if (t.dtype != q27::DType::Q4_G64 && t.dtype != q27::DType::Q8_G128 &&
            t.dtype != q27::DType::T2_G128 && t.dtype != q27::DType::T3_G128) continue;
        const int64_t rows = (int64_t)t.rows(), cols = (int64_t)t.cols();
        if (rows < min_rows) continue;          // stays on the GEMV
        if (cols % VG_KB_MULT != 0) continue;   // ineligible; vgemm_verify refuses it
        int z = vgemm_z(rows, cols);
        if (z <= 1) continue;                   // MODE 0 needs no workspace
        size_t b = (size_t)z * W_PLUMB * (size_t)rows * 4;
        if (b > mx) mx = b;
    }
    return mx;
}

size_t vgemm_ws_bytes(const q27::DevTensor* const* wl, int n) {
    size_t mx = 0;
    for (int i = 0; i < n; i++) {
        const q27::DevTensor& w = *wl[i];
        int z = vgemm_z(w.rows, w.cols);
        if (z <= 1) continue; // MODE 0 needs no workspace
        size_t b = (size_t)z * W_PLUMB * (size_t)w.rows * 4;
        if (b > mx) mx = b;
    }
    return mx;
}

template <int DT, int MODE>
static void set_attr_once() {
    static bool done = false;
    if (done) return;
    // 13.75 KB is under the 48 KB static default, so this is a no-op today. Keep
    // it (and the assert) so a future tile bump cannot silently reintroduce the
    // lazy-cudaFuncSetAttribute-during-graph-capture hazard.
    static_assert(vgemm_smem_bytes(DT == 1) < 48 * 1024,
                  "smem over the 48KB default -- setattr must move out of capture");
    CUDA_CHECK(cudaFuncSetAttribute(k_vgemm<VG_MR, DT, MODE>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)vgemm_smem_bytes(DT == 1)));
    done = true;
}

template <int DT>
static void launch(const q27::DevTensor& w, const XLanes& X, const YLanes& Y, float* ws, int T,
                   cudaStream_t st) {
    const int n_stages = (int)(w.cols / VG_KS);
    int z = vgemm_z(w.rows, w.cols);
    int spz = (n_stages + z - 1) / z;
    spz = (spz + VG_KG - 1) / VG_KG * VG_KG; // KB-align: a slice can never straddle n_stages
    z = (n_stages + spz - 1) / spz;          // trim empty slices (k_reduce_z would sum garbage)
    dim3 grid(1, (unsigned)((w.rows + VG_MR - 1) / VG_MR), (unsigned)z);
    const size_t sm = vgemm_smem_bytes(DT == 1);
    if (z == 1) {
        set_attr_once<DT, 0>();
        k_vgemm<VG_MR, DT, 0><<<grid, 256, sm, st>>>((const uint8_t*)w.data,
                                                       (const __half*)w.scales, X, Y, nullptr,
                                                       w.rows, w.cols, T, spz);
    } else {
        set_attr_once<DT, 1>();
        k_vgemm<VG_MR, DT, 1><<<grid, 256, sm, st>>>((const uint8_t*)w.data,
                                                       (const __half*)w.scales, X, Y, ws, w.rows,
                                                       w.cols, T, spz);
        dim3 g2((unsigned)((w.rows + 255) / 256), (unsigned)T);
        k_reduce_z<<<g2, 256, 0, st>>>(ws, Y, w.rows, T, z);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <int DT, int MODE>
static VgemmAttrs attrs_of() {
    cudaFuncAttributes a{};
    CUDA_CHECK(cudaFuncGetAttributes(&a, k_vgemm<VG_MR, DT, MODE>));
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k_vgemm<VG_MR, DT, MODE>,
                                                             256, vgemm_smem_bytes(DT == 1)));
    return VgemmAttrs{a.numRegs, (size_t)a.localSizeBytes, vgemm_smem_bytes(DT == 1), blocks};
}

VgemmAttrs vgemm_attrs(bool q4in, int mode) {
    if (q4in) return mode ? attrs_of<1, 1>() : attrs_of<1, 0>();
    return mode ? attrs_of<0, 1>() : attrs_of<0, 0>();
}
VgemmAttrs vgemm_attrs_dt(int dt, int mode) {
    if (dt == 2) return mode ? attrs_of<2, 1>() : attrs_of<2, 0>();
    return vgemm_attrs(dt == 1, mode);
}

bool vgemm_verify(const q27::DevTensor& w, const XLanes& X, const YLanes& Y, float* ws, int T,
                  cudaStream_t st) {
    if (T < 2 || T > W_PLUMB) return false;
    // A cols that is not a multiple of KB would silently drop its K tail. Every
    // decode shape (5120/6144/17408) complies; abort rather than corrupt.
    if (w.cols % VG_KB_MULT != 0) {
        fprintf(stderr, "vgemm: cols %ld not a multiple of %d\n", (long)w.cols, VG_KB_MULT);
        abort();
    }
    if (w.dtype == q27::DType::Q4_G64) launch<1>(w, X, Y, ws, T, st);
    else if (w.dtype == q27::DType::Q8_G128) launch<0>(w, X, Y, ws, T, st);
    else if (w.dtype == q27::DType::T2_G128) launch<2>(w, X, Y, ws, T, st);
    else return false;
    return true;
}

} // namespace q27k
