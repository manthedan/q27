#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x)                                                                  \
    do {                                                                               \
        cudaError_t err_ = (x);                                                        \
        if (err_ != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error: %s\n  at %s:%d: %s\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__, #x);                                           \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

// LANE PLUMBING WIDTH -- the FIXED number of lane slots the round kernels
// address. Q27_W_MAX (engine.cuh) only caps how many go LIVE and sizes the
// per-width memory (record-arena rows, captured-graph zoo); W_PLUMB sizes the
// structs, the outcome layout, and every "list every lane" array. It is 16
// because the lane-pointer structs (P3/CP3/IP3/XQ3/WIP3) carry p[16] and the
// fdmma verify kernel asserts 6*W <= 96 rows -- 16 is the hard ceiling of the
// current kernel family, so there is no reason to plumb anything narrower.
//
// It lives HERE, not in engine.cuh, because spec3.cu (k_prep_round /
// k_finish_round -- the kernels that walk every lane slot) does not include
// engine.cuh. Before W16 those kernels carried hardcoded 11/12 literals that
// no compiler could tie back to the plumbing width; a widening had to find
// them by grep. Now they derive from this constant.
#ifndef Q27_W_PLUMB
#define Q27_W_PLUMB 16
#endif
static constexpr int W_PLUMB = Q27_W_PLUMB;
// Q27_W_MAX (the live-width cap; engine.cuh documents the knob) is needed
// here too: translation units that never include engine.cuh (spec3.cu) gate
// kernels on it, and the default build passes no -D for it.
#ifndef Q27_W_MAX
#define Q27_W_MAX 12
#endif
static_assert(W_PLUMB <= 16, "lane-pointer structs are p[16]; k_attn_fdmma asserts 6*W <= 96");
// Round outcome layout: {n, t1, dr1..dr(W_PLUMB-1), pending}.
static constexpr int OUTCOME_INTS = W_PLUMB + 2;

// KV-cache format kind (Q27_KV): scalar fp16 (default) / fp8 E4M3 ("fp8") /
// turbo3 3-bit blocks ("turbo3", src/turbo3.cuh) / turbo3 V with plain fp16 K
// ("turbo3v" -- the GQA=6 escape hatch if turbo3-K craters, port spec risk
// section) / turbo5 5-bit K with turbo3 V ("turbo5k", src/turbo5.cuh -- the
// missing 5-6 bit rung, plan docs/plans/2026-08-01-5bit-k.md).
// Values 0/1 keep the old `bool fp8` call sites meaning-compatible
// (false->KV_F16, true->KV_FP8) where the parameter widened to int.
//
// ORDERING IS LOAD-BEARING: everything from KV_T3 up stores V as turbo3
// blocks, so the engine's `kv_kind >= KV_T3` tests (turbo store, inverse-WHT
// on the pooled output) pick up new turbo kinds automatically. Only K varies
// across them, which is exactly what Engine::kv_bytes(is_v) is shaped to
// express. Append new turbo kinds at the END; anything that is NOT a turbo
// kind must go below KV_T3.
enum KvKind : int { KV_F16 = 0, KV_FP8 = 1, KV_T3 = 2, KV_T3V = 3, KV_T5K = 4, KV_I8G64 = 5 };

// Does K carry the WHT rotation (so Q must be forward-rotated to match)?
// True for every turbo kind except turbo3v (plain fp16 K) and int8g64
// (ninfer's per-64-absmax int8 K, deliberately rotation-free -- the study
// arm measures that exact error profile; src/i8g64.cuh).
static inline bool kv_k_rotated(int kvk) { return kvk == KV_T3 || kvk == KV_T5K; }

// ---- M2a paged KV addressing (plan docs/plans/2026-08-16-m2-paged-kv.md) --
// KV rows live in 64-row pages reached through a per-pair block table: entry
// j = base pointer of the page backing rows [64j, 64j+64). The table POINTER
// is init-fixed (baked into captured graphs, the mask-pool discipline); the
// ENTRIES are rewritable between rounds (identity mapping in M2a; the M2b
// pool remaps them). Kernels receive the pair's table slice (void* const*)
// instead of a raw base pointer; every access stays single-row arithmetic:
//   row base  = (char*)tab[p >> 6] + (size_t)(p & 63) * row_bytes
//   block idx = (Block*)tab[p >> 6] + (p & 63) * n_kv_heads * 2 + h*2 + g
static constexpr int KV_PAGE = 64;         // rows per page
static constexpr int KV_PAGE_SHIFT = 6;
static constexpr int KV_PAGE_MASK = KV_PAGE - 1;
static_assert((1 << KV_PAGE_SHIFT) == KV_PAGE,
              "KV_PAGE_SHIFT must stay coupled to KV_PAGE -- every kernel indexes "
              "tab[p >> SHIFT] while the host builds pages of KV_PAGE rows");
// The MMA-family kernels hoist ONE table lookup per 32-position tile; that is
// sound only while a 32-aligned tile can never cross a page boundary.
static_assert(KV_PAGE % 32 == 0 && KV_PAGE >= 32,
              "per-tile page hoists (fdmma/H16/prefill) assume 32-aligned tiles "
              "never cross a KV page");

#ifdef __CUDACC__
// Row/block resolvers -- the ONLY sanctioned way a kernel touches KV memory
// under M2a. Addressing-only by construction: same bytes, same fp order.
template <typename T>
__device__ __forceinline__ T* kv_row(void* const* __restrict__ tab, int p,
                                     int row_elems) {
    return (T*)tab[p >> KV_PAGE_SHIFT] + (size_t)(p & KV_PAGE_MASK) * row_elems;
}
template <typename T>
__device__ __forceinline__ const T* kv_row_c(const void* const* __restrict__ tab, int p,
                                             int row_elems) {
    return (const T*)tab[p >> KV_PAGE_SHIFT] + (size_t)(p & KV_PAGE_MASK) * row_elems;
}
// Block-addressed kinds (turbo3/turbo5/i8g64): row = n_kv_heads*2 blocks.
template <typename B>
__device__ __forceinline__ B* kv_blk(void* const* __restrict__ tab, int p,
                                     int blocks_per_row) {
    return (B*)tab[p >> KV_PAGE_SHIFT] + (size_t)(p & KV_PAGE_MASK) * blocks_per_row;
}
template <typename B>
__device__ __forceinline__ const B* kv_blk_c(const void* const* __restrict__ tab, int p,
                                             int blocks_per_row) {
    return (const B*)tab[p >> KV_PAGE_SHIFT] + (size_t)(p & KV_PAGE_MASK) * blocks_per_row;
}
#endif

#ifdef __CUDACC__
#include <cuda_fp8.h>
// KV-cache element conversions (P2): fp16 default, fp8 E4M3 opt-in (Q27_KV=fp8).
// E4M3 store saturates to +-448; kvstats probe (2026-07-02, 8K wikitext tokens)
// measured K amax <= 21.8, V amax <= 118.6 across all 17 attention layers, so
// scale-free E4M3 has >=3.8x headroom and per-row scales buy nothing.
__device__ __forceinline__ float kv2f(__half x) { return __half2float(x); }
__device__ __forceinline__ float kv2f(__nv_fp8_e4m3 x) { return float(x); }
__device__ __forceinline__ __half kv2h(__half x) { return x; }
__device__ __forceinline__ __half kv2h(__nv_fp8_e4m3 x) {
    return __half(__nv_cvt_fp8_to_halfraw(x.__x, __NV_E4M3));
}
__device__ __forceinline__ void kv_set(__half& d, float x) { d = __float2half_rn(x); }
__device__ __forceinline__ void kv_set(__nv_fp8_e4m3& d, float x) { d = __nv_fp8_e4m3(x); }
#endif
