// q27 reference kernels: correct first, fast later (M2 replaces the GEMVs).
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace q27k {

// Dequantize an entire tensor to f32 (validation / small tensors only).
void dequant_q4(const uint8_t* W, const __half* S, float* out, int64_t rows, int64_t cols,
                cudaStream_t st = 0);
void dequant_q8(const int8_t* W, const __half* S, float* out, int64_t rows, int64_t cols,
                cudaStream_t st = 0);

// Quantized activation vector (mmvq-style): per 32-element block, int8 values with
// f32 scale and integer block sum. Two byte orders are kept: natural (for Q8 weights)
// and even/odd split per 8 (matching Q4 nibble unpack for dp4a).
struct XQuant {
    int8_t* nat = nullptr;   // [cols]
    uint2* eo = nullptr;     // [cols/8]: .x = bytes {x0,x2,x4,x6}, .y = {x1,x3,x5,x7}
    float* scale = nullptr;  // [cols/32]
    int* isum = nullptr;     // [cols/32] sum of quantized values per block
    // group-64 requantization (prefill MMA GEMM only; matches the Q4_G64
    // weight group so two K=32 mma steps share one fp dequant step). Distinct
    // VALUES from nat (amax over 64), not a re-scaling -- decode lanes leave
    // these null and are untouched.
    int8_t* nat64 = nullptr; // [cols]
    float* s64 = nullptr;    // [cols/64]
};
XQuant xquant_alloc(int64_t max_cols, bool g64 = false);
void quantize_x(const float* x, int64_t cols, const XQuant& xq, cudaStream_t st = 0);
void quantize_x_g64(const float* x, int64_t cols, const XQuant& xq, cudaStream_t st = 0);

// y[r] = sum_c W[r,c] * x[c].  W quantized row-major, reduction along contiguous axis.
// Q4/Q8 use dp4a against the pre-quantized activation vector.
void gemv_q4(const uint8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st = 0);
void gemv_q8(const int8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st = 0);

// Batched: one weight pass, N quantized activation columns -> y[n][rows]
// (y column-major by batch: y + n*rows). N in 2..5. The speculative-verify core.
// ys: per-column output pointers (ys[n][row]); no post-split copies needed.
void gemv_q4_n(const uint8_t* W, const __half* S, const XQuant* xqs, int nbatch,
               float* const* ys, int64_t rows, int64_t cols, cudaStream_t st = 0);
void gemv_q8_n(const int8_t* W, const __half* S, const XQuant* xqs, int nbatch,
               float* const* ys, int64_t rows, int64_t cols, cudaStream_t st = 0);
void gemv_f16(const __half* W, const float* x, float* y, int64_t rows, int64_t cols,
              cudaStream_t st = 0);

// ---- T2_G128 (ternary, Bonsai 2 Phase 2) ----
// On disk (FORMAT.md): element i of a row in 2-bit field (i%4)*2 of byte i/4,
// code c -> (c-1)*scale, fp16 scale per 128. The device copy is INTERLEAVED
// per 16-element word so that ((w >> 2k) & 0x03030303) yields the same four
// elements the Q4 kernels' even/odd activation words carry: field 4b+0 holds
// e[2b], 4b+1 e[2b+1], 4b+2 e[8+2b], 4b+3 e[9+2b]. t2_interleave_device does
// that in place right after upload (each word independent, idempotent-free:
// apply exactly once). The dot is dp4a(codes, x) - sum(x) per 32-block.
void t2_interleave_device(uint8_t* W, uint64_t bytes, cudaStream_t st = 0);
void gemv_t2(const uint8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st = 0);
void gemv_t2_n(const uint8_t* W, const __half* S, const XQuant* xqs, int nbatch,
               float* const* ys, int64_t rows, int64_t cols, cudaStream_t st = 0);

// ---- T3_G128 (ternary, five trits per byte; Bonsai 2 8 GB packs, 2026-09-20) ----
// On disk (FORMAT.md): 26 bytes per 128-group, base-3 c0 + 3c1 + 9c2 + 27c3
// + 81c4 (code c = trit + 1), byte 25 = columns 125..127 plus two code-1 pads;
// the Metal matvec reads that directly. The CUDA copy is a DIFFERENT layout,
// built once at upload (t3_relayout_device), chosen so the decode GEMV sums
// exactly the chunks gemv_t2 sums, in its order -- so gemv_t3 is bitwise
// gemv_t2 on the same ternary matrix:
//   a chunk is 32 elements (the activation quant block); lane L of the warp
//   owns chunks L, L+32, L+64, ... (k_gemv_q4's map). A WINDOW is 160 chunks
//   (5120 elements). Lane L's five chunks of window m (160m + 32i + L,
//   i = 0..4) form one 32-byte UNIT = 8 u32; a window stores u32 0..3 of all
//   32 units lane-major in its first 512 bytes (row + 1024m + 16L) and u32
//   4..7 in the second (row + 1024m + 512 + 16L), so each of the lane's two
//   16-byte loads is 512 contiguous bytes across the warp (a 32-byte lane
//   stride measured 25% slower on the 3090). u32 k of a unit holds dp4a
//   words 5k..5k+4 of the unit's 40 (word
//   qu = 8i + q: chunk i, activation word q in the XQuant.eo order, byte lane
//   b = element 32c + 8(q/2) + 2b + (q%2)). Byte b of the u32 packs lane b's
//   five codes across those words, V = sum_r code_r * 3^(4-r), stored SCALED
//   as ceil(V*256/243): round r then pops the next code as (q*3)>>8 with
//   q = (q*3)&255 (the TQ1_0 top-digit trick), two 16-bit lanes per u32.
//   A tail window (cols/32 % 160 != 0; 1 or 2 chunks per lane on this model)
//   packs ceil(8*ni/5) u32 per lane rounded up to an even count. Pad words
//   are code 1 (zero). t3_row_bytes: 5120 -> 1024, 6144 -> 1280, 17408 -> 3584
//   (1.60 / 1.67 / 1.65 bits per weight before the fp16 scale per 128).
// The prefill MMA GEMM stays gemm_t2_T: t3_to_t2_device rewrites one T3 matrix
// into T2's device-interleaved words (a ~22 MB scratch) right before it.
uint64_t t3_row_bytes(uint64_t cols);
uint64_t t3_device_bytes(uint64_t rows, uint64_t cols);
// src26: the FORMAT.md bytes (device memory), dst: t3_device_bytes(rows, cols)
void t3_relayout_device(const uint8_t* src26, uint8_t* dst, int64_t rows, int64_t cols,
                        cudaStream_t st = 0);
void gemv_t3(const uint8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st = 0);
void gemv_t3_n(const uint8_t* W, const __half* S, const XQuant* xqs, int nbatch,
               float* const* ys, int64_t rows, int64_t cols, cudaStream_t st = 0);
// W3: t3 device layout; W2: rows * cols/4 bytes in the t2_interleave_device order
void t3_to_t2_device(const uint8_t* W3, uint8_t* W2, int64_t rows, int64_t cols,
                     cudaStream_t st = 0);

// y = x * rsqrt(mean(x^2) + eps) * w      (single vector, n elements)
void rmsnorm(const float* x, const float* w, float* y, int n, float eps, cudaStream_t st = 0);

// out[i] = silu(gate[i]) * up[i]
void silu_mul(const float* gate, const float* up, float* out, int n, cudaStream_t st = 0);

// out[0..cols) = dequantized row *d_token of a Q8_G128 matrix (embedding lookup)
void embed_row_q8(const int8_t* W, const __half* S, const int* d_token, int64_t cols, float* out,
                  cudaStream_t st = 0);
// same over a T2_G128 embedding (device-interleaved words; bitwise the Q8 row
// for an exact ternary table -- Bonsai 2 slim packs)
void embed_row_t2(const uint8_t* W, const __half* S, const int* d_token, int64_t cols, float* out,
                  cudaStream_t st = 0);

// Grid-merged multi-token variants for the speculative round: identical
// per-token work distribution, tokens mapped to an extra grid dimension
// (1 launch vs ntok). ntok selects how many lanes are live (brace inits
// with fewer entries leave the rest null, unread); shallower widths fill a
// prefix. width-12 2026-07-10: 16 slots (verify lanes a..l = 12 live max;
// slots sized 16 so a future W=16 revisit is struct-free). Structs ride
// kernel params by value -- slots beyond ntok are never read.
struct P3 { float* p[16]; };
struct CP3 { const float* p[16]; };
struct IP3 { const int* p[16]; }; // int twin (draft-token lanes); lived in spec3.cuh until the sampled tail (blocks.cuh) needed it too
struct XQ3 { XQuant q[16]; };

void rmsnorm3(CP3 x, const float* w, P3 y, int n, float eps, cudaStream_t st = 0, int ntok = 3);
void add3(P3 x, CP3 y, int n, cudaStream_t st = 0, int ntok = 3);
void silu_mul3(P3 g, CP3 u, int n, cudaStream_t st = 0, int ntok = 3);
void quantize3(CP3 x, int64_t cols, const XQ3& xq, cudaStream_t st = 0, int ntok = 3);
// Fused rmsnorm3 + quantize3 of the result (bitwise those two launches; the
// verify forward quantizes every normed activation it produces).
void rmsnorm3q(CP3 x, const float* w, P3 y, const XQ3& xq, int n, float eps, cudaStream_t st = 0,
               int ntok = 3);

// ---- Bonsai 2 activation rotation (docs/plans/2026-09-18-bonsai2-ternary.md)
// The pack stores every projection in a rotated basis W' = W R^T with
// R = (1/32) H_1024 S per contiguous 1024-block of the INPUT dimension
// (H natural-order Sylvester Walsh-Hadamard, S a fixed +-1 diagonal, one
// sign vector per input width). Before such a matmul the activation gets
// fwd: y = (1/32) H (s * x); after an embedding lookup of a rotated row it
// gets inv: x = s * ((1/32) H y). In place, fp32, fixed operand order
// (bitwise deterministic, CPU-equal). width % 1024 == 0; signs[width].
void hadamard1024(float* x, const float* signs, int width, bool inv, cudaStream_t st = 0);
// T rows at row_stride floats apart (prefill).
void hadamard1024_rows(float* x, const float* signs, int width, int rows, long row_stride,
                       bool inv, cudaStream_t st = 0);
// Up to 16 lane vectors (speculative verify / multi-lane paths).
void hadamard1024_lanes(P3 x, const float* signs, int width, int nlanes, bool inv,
                        cudaStream_t st = 0);
// GDN value-head order: the engine's og is tiled [rep][nk][hd] (the GGUF
// convention); a Hadamard-folded ssm_out was folded in the training (grouped)
// order [nk][rep][hd], so permute before the rotation:
//   out[k*rep*hd + r*hd + h] = in[r*nk*hd + k*hd + h].
void gdn_v_tiled_to_grouped(const float* in, float* out, int hd, int nk, int rep,
                            cudaStream_t st = 0);
void gdn_v_tiled_to_grouped_rows(const float* in, float* out, int hd, int nk, int rep, int rows,
                                 long stride, cudaStream_t st = 0);
void gdn_v_tiled_to_grouped_lanes(CP3 in, P3 out, int hd, int nk, int rep, int nlanes,
                                  cudaStream_t st = 0);
// Fused rotate + quantize (decode): the int8 activation set (nat/eo/scale/
// isum, group 32) of the ROTATED vector, computed from the raw x without
// writing the rotated floats anywhere. Bitwise those of hadamard1024 on a
// copy followed by quantize_x (same butterfly, same quantize body). With
// perm, the input is gathered through the GDN tiled->grouped order first
// (hd/nk/rep as in gdn_v_tiled_to_grouped). One block per 1024-chunk.
void rotq(const float* x, const float* signs, int width, const XQuant& xq, cudaStream_t st = 0,
          bool perm = false, int hd = 0, int nk = 0, int rep = 0);
void rotq3(CP3 x, const float* signs, int width, const XQ3& xq, int ntok, cudaStream_t st = 0,
           bool perm = false, int hd = 0, int nk = 0, int rep = 0);
// rmsnorm3 (y = x * rsqrt(mean(x^2)+eps) * w, written UNROTATED to y) fused
// with the rotate+quantize of y: one block per lane. y stays available for
// the unfolded F16 projections (GDN alpha/beta). Norm part is k_rmsnorm3q's
// verbatim (bitwise rmsnorm3); quantize part is rotq's.
void rmsnorm3_rotq(CP3 x, const float* w, P3 y, const float* signs, const XQ3& xq, int n,
                   float eps, cudaStream_t st = 0, int ntok = 3);

} // namespace q27k
