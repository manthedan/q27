// Q27_SHADER_ABI 7
//
// Shaders compile from this file at RUNTIME, so a host binary built before a
// buffer-binding change silently misbinds against a newer file (this exact
// mismatch produced coherent-but-wrong decode output on 2026-07-14). The tag
// above names the binding layout; metal_backend.mm refuses to load a source
// whose tag differs from the one it was compiled against. Bump both on any
// change to kernel buffer indices or argument structs.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct MatvecArgs {
    uint rows;
    uint cols;
};
struct MatvecPairArgs { uint rows_a; uint rows_b; uint cols; };
struct MatmulArgs { uint rows; uint cols; uint x_rows; uint simdgroups; };

// Goldberg's exact-correction log1p — tracks CUDA log1pf to ~1 ulp; MSL has
// no log1p (k3 audit D2).
inline float log1p_f(float t) {
    const float u = 1.0f + t;
    return u == 1.0f ? t : log(u) * (t / (u - 1.0f));
}

kernel void q27_matvec_f32(
        device const float *weights [[buffer(0)]],
        device const float *x       [[buffer(1)]],
        device       float *out     [[buffer(2)]],
        constant MatvecArgs &args   [[buffer(3)]],
        uint group                   [[threadgroup_position_in_grid]],
        ushort lane                  [[thread_index_in_simdgroup]],
        ushort simdgroup             [[simdgroup_index_in_threadgroup]]) {
    const uint row = group * 8 + simdgroup;
    if (row >= args.rows) return;
    float sum = 0.0f;
    const ulong base = (ulong)row * args.cols;
    for (uint col = lane; col < args.cols; col += 32) sum += weights[base + col] * x[col];
    sum = simd_sum(sum);
    if (lane == 0) out[row] = sum;
}

kernel void q27_matvec_f16(
        device const half  *weights [[buffer(0)]],
        device const float *x       [[buffer(1)]],
        device       float *out     [[buffer(2)]],
        constant MatvecArgs &args   [[buffer(3)]],
        uint group                   [[threadgroup_position_in_grid]],
        ushort lane                  [[thread_index_in_simdgroup]],
        ushort simdgroup             [[simdgroup_index_in_threadgroup]]) {
    const uint row = group * 8 + simdgroup;
    if (row >= args.rows) return;
    float sum = 0.0f;
    const ulong base = (ulong)row * args.cols;
    for (uint col = lane; col < args.cols; col += 32) sum += float(weights[base + col]) * x[col];
    sum = simd_sum(sum);
    if (lane == 0) out[row] = sum;
}

// One 256-thread threadgroup per row pair. The production users are the
// 48-row GDN alpha/beta projections; a simdgroup-per-row layout put only
// 1,536 threads on the whole GPU and measured ~11 GB/s. packed_half4 keeps
// 2-byte alignment so mmap-aliased tensor offsets stay valid.
kernel void q27_matvec_f16_pair(
        device const half *weights_a [[buffer(0)]], device float *out_a [[buffer(1)]],
        device const half *weights_b [[buffer(2)]], device float *out_b [[buffer(3)]],
        device const float *x [[buffer(4)]], constant MatvecPairArgs &args [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]], ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const bool row_a = row < args.rows_a, row_b = row < args.rows_b;
    const ulong base = (ulong)row * args.cols;
    device const packed_half4 *wa4 = (device const packed_half4 *)(weights_a + base);
    device const packed_half4 *wb4 = (device const packed_half4 *)(weights_b + base);
    const uint vectors = args.cols / 4;
    float sa = 0.0f, sb = 0.0f;
    for (uint v = tid; v < vectors; v += 256) {
        const float4 xv = float4(x[v * 4], x[v * 4 + 1], x[v * 4 + 2], x[v * 4 + 3]);
        if (row_a) sa += dot(float4(wa4[v]), xv);
        if (row_b) sb += dot(float4(wb4[v]), xv);
    }
    for (uint col = vectors * 4 + tid; col < args.cols; col += 256) {
        const float xv = x[col];
        if (row_a) sa += float(weights_a[base + col]) * xv;
        if (row_b) sb += float(weights_b[base + col]) * xv;
    }
    threadgroup float partial[8];
    sa = simd_sum(sa);
    if (lane == 0) partial[simdgroup] = sa;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        float total = lane < 8 ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0 && row_a) out_a[row] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sb = simd_sum(sb);
    if (lane == 0) partial[simdgroup] = sb;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        float total = lane < 8 ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0 && row_b) out_b[row] = total;
    }
}

kernel void q27_matvec_q8_g128(
        device const char  *weights [[buffer(0)]],
        device const half  *scales  [[buffer(1)]],
        device const float *x       [[buffer(2)]],
        device       float *out     [[buffer(3)]],
        constant MatvecArgs &args   [[buffer(4)]],
        uint group                   [[threadgroup_position_in_grid]],
        ushort lane                  [[thread_index_in_simdgroup]],
        ushort simdgroup             [[simdgroup_index_in_threadgroup]]) {
    const uint row = group * 8 + simdgroup;
    if (row >= args.rows) return;
    float sum = 0.0f;
    const ulong base = (ulong)row * args.cols;
    const ulong scale_base = (ulong)row * (args.cols / 128);
    for (uint col = lane; col < args.cols; col += 32) {
        const float scale = float(scales[scale_base + col / 128]);
        sum += float(weights[base + col]) * scale * x[col];
    }
    sum = simd_sum(sum);
    if (lane == 0) out[row] = sum;
}

kernel void q27_matvec_q4_g64(
        device const uchar *weights [[buffer(0)]],
        device const half  *scales  [[buffer(1)]],
        device const float *x       [[buffer(2)]],
        device       float *out     [[buffer(3)]],
        constant MatvecArgs &args   [[buffer(4)]],
        uint group                   [[threadgroup_position_in_grid]],
        ushort lane                  [[thread_index_in_simdgroup]],
        ushort simdgroup             [[simdgroup_index_in_threadgroup]]) {
    const uint row = group * 8 + simdgroup;
    if (row >= args.rows) return;
    float sum = 0.0f;
    const ulong packed_base = (ulong)row * (args.cols / 2);
    const ulong scale_base = (ulong)row * (args.cols / 64);
    for (uint col = lane; col < args.cols; col += 32) {
        const uchar packed = weights[packed_base + col / 2];
        const int quant = int((col & 1) ? (packed >> 4) : (packed & 0x0f)) - 8;
        const float scale = float(scales[scale_base + col / 64]);
        sum += float(quant) * scale * x[col];
    }
    sum = simd_sum(sum);
    if (lane == 0) out[row] = sum;
}

// T2_G128 ternary weights (FORMAT.md): element i of a row -> byte i/4, 2-bit
// field at bit offset (i%4)*2 (sequential, LSB-first); code c in {0,1,2}
// decodes to (c-1)*scale, one fp16 scale per 128 columns. Code 3 never
// appears in valid artifacts (repack.py hard-fails on it).
//
// This is the intended production decode path for ternary weights: FLOAT
// activations, so the math is exact ternary (+x/-x/0, no activation
// quantization error). Two structural choices, both taken from measuring
// against the PrismML fork's kernel (theirs sustains ~60 GB/s effective on
// M4; a Q4-style shift/mask/imad loop and a byte->float4 LUT+fma variant
// both saturated ~125 Gelem/s ≈ 33 GB/s at 2.27 bpw):
//   1. select-form dot, no multiplies and no decode:
//        sum((c-1)*y) = sum_lo(y) + 2*sum_hi(y) - sum(y)
//      where sum_lo/sum_hi conditionally add y where the code's low/high
//      bit is set, and sum(y) is computed once per block slice.
//   2. four rows per simdgroup with the y-slice held in registers, so
//      activation reads amortize across rows.
// Lane layout: lane/8 picks one of four 128-column blocks in flight,
// (lane%8)*16 the 16-element slice within it; each lane's slice sits inside
// one scale group by construction.
kernel void q27_matvec_t2_g128(
        device const uchar *weights [[buffer(0)]],
        device const half  *scales  [[buffer(1)]],
        device const float *x       [[buffer(2)]],
        device       float *out     [[buffer(3)]],
        constant MatvecArgs &args   [[buffer(4)]],
        uint group                   [[threadgroup_position_in_grid]],
        ushort lane                  [[thread_index_in_simdgroup]],
        ushort simdgroup             [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = group * 32 + (uint)simdgroup * 4;   // 32 rows per threadgroup
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint nb = args.cols / 128;
    const uint ix = lane / 8;              // block in flight (4 per simdgroup)
    const uint il = (lane % 8) * 16;       // element offset within the block
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float *yb = x + ix * 128 + il;
    for (uint ib = ix; ib < nb; ib += 4) {
        float yl[16];
        float sumy = 0.0f;
        for (uint i = 0; i < 16; i++) { yl[i] = yb[i]; sumy += yb[i]; }
        for (uint r = 0; r < 4; r++) {
            const uint row = min(row0 + r, rlast);   // clamped rows compute, don't store
            device const uchar *qs = weights + (ulong)row * (args.cols / 4) + ib * 32 + il / 4;
            const float d = float(scales[(ulong)row * nb + ib]);
            const uchar4 b = *(device const uchar4 *)qs;
            float acc_lo = 0.0f, acc_hi = 0.0f;
            for (uint i = 0; i < 16; i++) {
                acc_lo += select(0.0f, yl[i], bool(b[i / 4] & (1u << (2 * (i % 4)))));
                acc_hi += select(0.0f, yl[i], bool(b[i / 4] & (2u << (2 * (i % 4)))));
            }
            sumf[r] += d * (acc_lo + 2.0f * acc_hi - sumy);
        }
        yb += 512;
    }
    for (uint r = 0; r < 4; r++) {
        const float tot = simd_sum(sumf[r]);
        if (lane == 0 && row0 + r < args.rows) out[row0 + r] = tot;
    }
}

// N=2 slot-batched select-form ternary GEMV (multislot Phase 2 probe): the
// production serial-decode kernel (q27_matvec_t2_g128) computing two
// activation vectors in one weight pass. The bit-extraction booleans — the
// issue-heavy part of the select-form dot — are shared across the rows;
// activations and outputs stay in separate buffers (no layout coupling).
// Per-row op order matches the single kernel exactly, so each output is
// bit-identical to two single dispatches.
// PARKED by measurement (2026-07-16, multislot Phase-2 probe): aggregate
// s_k 1.093 vs the 1.31 decision line — kept as the probe's reference
// surface, never engine-routed.
kernel void q27_matvec_t2_g128_x2(
        device const uchar *weights [[buffer(0)]],
        device const half  *scales  [[buffer(1)]],
        device const float *xa      [[buffer(2)]],
        device const float *xb      [[buffer(3)]],
        device       float *out_a   [[buffer(4)]],
        device       float *out_b   [[buffer(5)]],
        constant MatvecArgs &args   [[buffer(6)]],
        uint group                   [[threadgroup_position_in_grid]],
        ushort lane                  [[thread_index_in_simdgroup]],
        ushort simdgroup             [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = group * 32 + (uint)simdgroup * 4;   // 32 rows per threadgroup
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint nb = args.cols / 128;
    const uint ix = lane / 8;              // block in flight (4 per simdgroup)
    const uint il = (lane % 8) * 16;       // element offset within the block
    float sumfa[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sumfb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float *ya = xa + ix * 128 + il;
    device const float *yb = xb + ix * 128 + il;
    for (uint ib = ix; ib < nb; ib += 4) {
        float yla[16], ylb[16];
        float sumya = 0.0f, sumyb = 0.0f;
        for (uint i = 0; i < 16; i++) { yla[i] = ya[i]; sumya += ya[i]; }
        for (uint i = 0; i < 16; i++) { ylb[i] = yb[i]; sumyb += yb[i]; }
        for (uint r = 0; r < 4; r++) {
            const uint row = min(row0 + r, rlast);   // clamped rows compute, don't store
            device const uchar *qs = weights + (ulong)row * (args.cols / 4) + ib * 32 + il / 4;
            const float d = float(scales[(ulong)row * nb + ib]);
            const uchar4 b = *(device const uchar4 *)qs;
            float alo = 0.0f, ahi = 0.0f, blo = 0.0f, bhi = 0.0f;
            for (uint i = 0; i < 16; i++) {
                const bool lo = bool(b[i / 4] & (1u << (2 * (i % 4))));
                const bool hi = bool(b[i / 4] & (2u << (2 * (i % 4))));
                alo += select(0.0f, yla[i], lo);
                ahi += select(0.0f, yla[i], hi);
                blo += select(0.0f, ylb[i], lo);
                bhi += select(0.0f, ylb[i], hi);
            }
            sumfa[r] += d * (alo + 2.0f * ahi - sumya);
            sumfb[r] += d * (blo + 2.0f * bhi - sumyb);
        }
        ya += 512;
        yb += 512;
    }
    for (uint r = 0; r < 4; r++) {
        const float tota = simd_sum(sumfa[r]);
        const float totb = simd_sum(sumfb[r]);
        if (lane == 0 && row0 + r < args.rows) {
            out_a[row0 + r] = tota;
            out_b[row0 + r] = totb;
        }
    }
}

// T3_G128: five ternary codes per byte, base-3 (243 values), 26 bytes per
// 128-column group, scales as T2. The LUT expands a byte into five 2-bit
// codes (bits 2t..2t+1 = code for slot t); the select-form dot then runs
// the same math as the T2 kernel: scale * (sum(code_i * y_i) - sum(y_i)).
// Entries 243..255 decode to all-code-1 (trit 0), so a corrupt byte is
// inert rather than out-of-bounds.
constant ushort t3_codes[256] = {
    0x000, 0x001, 0x002, 0x004, 0x005, 0x006, 0x008, 0x009, 0x00a, 0x010, 0x011, 0x012, 0x014, 0x015, 0x016, 0x018,
    0x019, 0x01a, 0x020, 0x021, 0x022, 0x024, 0x025, 0x026, 0x028, 0x029, 0x02a, 0x040, 0x041, 0x042, 0x044, 0x045,
    0x046, 0x048, 0x049, 0x04a, 0x050, 0x051, 0x052, 0x054, 0x055, 0x056, 0x058, 0x059, 0x05a, 0x060, 0x061, 0x062,
    0x064, 0x065, 0x066, 0x068, 0x069, 0x06a, 0x080, 0x081, 0x082, 0x084, 0x085, 0x086, 0x088, 0x089, 0x08a, 0x090,
    0x091, 0x092, 0x094, 0x095, 0x096, 0x098, 0x099, 0x09a, 0x0a0, 0x0a1, 0x0a2, 0x0a4, 0x0a5, 0x0a6, 0x0a8, 0x0a9,
    0x0aa, 0x100, 0x101, 0x102, 0x104, 0x105, 0x106, 0x108, 0x109, 0x10a, 0x110, 0x111, 0x112, 0x114, 0x115, 0x116,
    0x118, 0x119, 0x11a, 0x120, 0x121, 0x122, 0x124, 0x125, 0x126, 0x128, 0x129, 0x12a, 0x140, 0x141, 0x142, 0x144,
    0x145, 0x146, 0x148, 0x149, 0x14a, 0x150, 0x151, 0x152, 0x154, 0x155, 0x156, 0x158, 0x159, 0x15a, 0x160, 0x161,
    0x162, 0x164, 0x165, 0x166, 0x168, 0x169, 0x16a, 0x180, 0x181, 0x182, 0x184, 0x185, 0x186, 0x188, 0x189, 0x18a,
    0x190, 0x191, 0x192, 0x194, 0x195, 0x196, 0x198, 0x199, 0x19a, 0x1a0, 0x1a1, 0x1a2, 0x1a4, 0x1a5, 0x1a6, 0x1a8,
    0x1a9, 0x1aa, 0x200, 0x201, 0x202, 0x204, 0x205, 0x206, 0x208, 0x209, 0x20a, 0x210, 0x211, 0x212, 0x214, 0x215,
    0x216, 0x218, 0x219, 0x21a, 0x220, 0x221, 0x222, 0x224, 0x225, 0x226, 0x228, 0x229, 0x22a, 0x240, 0x241, 0x242,
    0x244, 0x245, 0x246, 0x248, 0x249, 0x24a, 0x250, 0x251, 0x252, 0x254, 0x255, 0x256, 0x258, 0x259, 0x25a, 0x260,
    0x261, 0x262, 0x264, 0x265, 0x266, 0x268, 0x269, 0x26a, 0x280, 0x281, 0x282, 0x284, 0x285, 0x286, 0x288, 0x289,
    0x28a, 0x290, 0x291, 0x292, 0x294, 0x295, 0x296, 0x298, 0x299, 0x29a, 0x2a0, 0x2a1, 0x2a2, 0x2a4, 0x2a5, 0x2a6,
    0x2a8, 0x2a9, 0x2aa, 0x155, 0x155, 0x155, 0x155, 0x155, 0x155, 0x155, 0x155, 0x155, 0x155, 0x155, 0x155, 0x155,
};

// Pure-ALU base-3 decode: five 2-bit codes from one byte, no memory access.
inline ushort t3_decode(uint v) {
    const uint c0 = v % 3u, v1 = v / 3u;
    const uint c1 = v1 % 3u, v2 = v1 / 3u;
    const uint c2 = v2 % 3u, v3 = v2 / 3u;
    const uint c3 = v3 % 3u, c4 = v3 / 3u;
    return (ushort)(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6) | (c4 << 8));
}

kernel void q27_matvec_t3_g128(
        device const uchar *weights [[buffer(0)]],
        device const half  *scales  [[buffer(1)]],
        device const float *x       [[buffer(2)]],
        device       float *out     [[buffer(3)]],
        constant MatvecArgs &args   [[buffer(4)]],
        uint group                   [[threadgroup_position_in_grid]],
        ushort tid                   [[thread_index_in_threadgroup]],
        ushort lane                  [[thread_index_in_simdgroup]],
        ushort simdgroup             [[simdgroup_index_in_threadgroup]]) {
    // Divergent per-lane byte values make a constant-memory LUT serialize;
    // stage it in banked threadgroup memory once per threadgroup.
    threadgroup ushort lut[256];
    for (uint i = tid; i < 256; i += 256) lut[i] = t3_codes[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = group * 32 + (uint)simdgroup * 4;   // 32 rows per threadgroup
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint nb = args.cols / 128;
    const uint row_bytes = nb * 26;
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // uint-aligned weight walk: one 4-byte load per row covers 20 columns.
    // Odd group counts leave a 1..3-byte row tail (row_bytes = nb*26); the
    // first lanes pick those up as scalar bytes after the word loop. Each
    // byte keeps its own group scale (a word can span a group boundary);
    // byte b with b%26 == 25 is the 3-column group tail (slots 3-4 are
    // canonical code-1 padding).
    // Odd nb (cols % 256 != 0) disables the word walk entirely; every byte
    // then takes the scalar tail below. That path is slower, not wrong —
    // test_t3_wide gates cols=1152 (nb=9) against the CPU reference.
    const uint words_per_row = (row_bytes & 3) ? 0 : row_bytes / 4;
    for (uint wi = lane; wi < words_per_row; wi += 32) {
        uint wv[4];
        for (uint r = 0; r < 4; r++) {
            const uint row = min(row0 + r, rlast);   // clamped rows compute, don't store
            wv[r] = *(device const uint *)(weights + (ulong)row * row_bytes + wi * 4);
        }
        for (uint i = 0; i < 4; i++) {
            const uint b = wi * 4 + i;
            const uint g = b / 26;
            const uint p = b - g * 26;
            const uint col = g * 128 + p * 5;
            float yl[5];
            float sumy;
            if (p != 25) {
                const packed_float4 y4 = *(device const packed_float4 *)(x + col);
                yl[0] = y4.x; yl[1] = y4.y; yl[2] = y4.z; yl[3] = y4.w; yl[4] = x[col + 4];
                sumy = yl[0] + yl[1] + yl[2] + yl[3] + yl[4];
            } else {   // group tail: three columns
                yl[0] = x[col]; yl[1] = x[col + 1]; yl[2] = x[col + 2];
                yl[3] = 0.0f; yl[4] = 0.0f;
                sumy = yl[0] + yl[1] + yl[2];
            }
            for (uint r = 0; r < 4; r++) {
                const uint row = min(row0 + r, rlast);
                const ushort codes = lut[(wv[r] >> (8 * i)) & 0xff];
                const float d = float(scales[(ulong)row * nb + g]);
                float acc_lo = 0.0f, acc_hi = 0.0f;
                for (uint t = 0; t < 5; t++) {
                    acc_lo += select(0.0f, yl[t], bool(codes & (1u << (2 * t))));
                    acc_hi += select(0.0f, yl[t], bool(codes & (2u << (2 * t))));
                }
                sumf[r] += d * (acc_lo + 2.0f * acc_hi - sumy);
            }
        }
    }
    // Odd group counts make row_bytes odd, so odd rows are not word-aligned
    // and the word walk is skipped entirely; every byte then takes this
    // scalar lane-strided path (model shapes all have even nb and never do).
    for (uint b = words_per_row * 4 + lane; b < row_bytes; b += 32) {
        const uint g = b / 26;
        const uint p = b - g * 26;
        const uint col = g * 128 + p * 5;
        const uint valid = min(5u, 128u - p * 5);
        float yl[5];
        float sumy = 0.0f;
        for (uint t = 0; t < 5; t++) {
            yl[t] = t < valid ? x[col + t] : 0.0f;
            sumy += yl[t];
        }
        for (uint r = 0; r < 4; r++) {
            const uint row = min(row0 + r, rlast);
            const ushort codes = lut[weights[(ulong)row * row_bytes + b]];
            const float d = float(scales[(ulong)row * nb + g]);
            float acc_lo = 0.0f, acc_hi = 0.0f;
            for (uint t = 0; t < 5; t++) {
                acc_lo += select(0.0f, yl[t], bool(codes & (1u << (2 * t))));
                acc_hi += select(0.0f, yl[t], bool(codes & (2u << (2 * t))));
            }
            sumf[r] += d * (acc_lo + 2.0f * acc_hi - sumy);
        }
    }
    for (uint r = 0; r < 4; r++) {
        const float tot = simd_sum(sumf[r]);
        if (lane == 0 && row0 + r < args.rows) out[row0 + r] = tot;
    }
}

// B1_G128 binary weights (FORMAT.md): element i of a row -> byte i/8, 1 bit
// at offset i%8 (sequential LSB-first, same convention as T2); bit b decodes
// to (2b-1)*scale, one fp16 scale per 128 columns. This is the production
// decode path for binary weights: FLOAT activations, select-form dot —
//   sum((2b-1)*y) = 2*sum_{b=1}(y) - sum(y)
// one conditional add per element plus one shared sum(y) per block slice
// (T2 needs two conditional adds). Structure chosen by the Phase 0B bench
// (docs/plans/2026-07-15-binary-tier.md: c1 select 0.504 wall ratio vs T2,
// STRONG GO; sign-XOR measured 2.6x worse and was killed). Same shape as
// the T2 kernel: 4 rows per simdgroup (32 per threadgroup), 4 blocks in
// flight, the 16-float activation slice held in registers across rows; a
// lane's 16 elements are 2 weight bytes, ushort-aligned by construction
// (row stride cols/8 is a multiple of 16, il/8 is even).
kernel void q27_matvec_b1_g128(
        device const uchar *weights [[buffer(0)]],
        device const half  *scales  [[buffer(1)]],
        device const float *x       [[buffer(2)]],
        device       float *out     [[buffer(3)]],
        constant MatvecArgs &args   [[buffer(4)]],
        uint group                   [[threadgroup_position_in_grid]],
        ushort lane                  [[thread_index_in_simdgroup]],
        ushort simdgroup             [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = group * 32 + (uint)simdgroup * 4;   // 32 rows per threadgroup
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint nb = args.cols / 128;
    const uint ix = lane / 8;              // block in flight (4 per simdgroup)
    const uint il = (lane % 8) * 16;       // element offset within the block
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float *yb = x + ix * 128 + il;
    for (uint ib = ix; ib < nb; ib += 4) {
        float yl[16];
        float sumy = 0.0f;
        for (uint i = 0; i < 16; i++) { yl[i] = yb[i]; sumy += yb[i]; }
        for (uint r = 0; r < 4; r++) {
            const uint row = min(row0 + r, rlast);   // clamped rows compute, don't store
            const ushort b = *(device const ushort *)(weights + (ulong)row * (args.cols / 8) +
                                                      ib * 16 + il / 8);
            const float d = float(scales[(ulong)row * nb + ib]);
            float acc_pos = 0.0f;
            for (uint i = 0; i < 16; i++)
                acc_pos += select(0.0f, yl[i], bool(b & (1u << i)));
            sumf[r] += d * (2.0f * acc_pos - sumy);
        }
        yb += 512;
    }
    for (uint r = 0; r < 4; r++) {
        const float tot = simd_sum(sumf[r]);
        if (lane == 0 && row0 + r < args.rows) out[row0 + r] = tot;
    }
}

// ---- B1 Phase 0B probe kernels (bench-only, never engine-routed) ----
// B1_G128 (FORMAT.md): 1 bit per element, LSB-first sequential within the
// row (element i -> byte i/8, bit i%8); bit ? +d : -d, one fp16 scale d per
// 128 columns. Three dot structures under test, pre-registered bands in
// docs/plans/2026-07-15-binary-tier.md §Phase 0B. Candidates 1-2 reuse the
// T2 kernel's shape: 4 rows per simdgroup, 4 blocks in flight, the 16-float
// activation slice held in registers across rows; a lane's 16 elements are
// 2 weight bytes, ushort-aligned by construction (row stride cols/8 is a
// multiple of 16, il/8 is even).

// Candidate 1 — select-form: sum(±y) = 2*sum_bit1(y) - sum(y). One
// conditional accumulate per element, same select idiom as production T2.
kernel void q27_matvec_b1_select(
        device const uchar *weights [[buffer(0)]],
        device const half  *scales  [[buffer(1)]],
        device const float *x       [[buffer(2)]],
        device       float *out     [[buffer(3)]],
        constant MatvecArgs &args   [[buffer(4)]],
        uint group                   [[threadgroup_position_in_grid]],
        ushort lane                  [[thread_index_in_simdgroup]],
        ushort simdgroup             [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = group * 32 + (uint)simdgroup * 4;   // 32 rows per threadgroup
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint nb = args.cols / 128;
    const uint ix = lane / 8;              // block in flight (4 per simdgroup)
    const uint il = (lane % 8) * 16;       // element offset within the block
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float *yb = x + ix * 128 + il;
    for (uint ib = ix; ib < nb; ib += 4) {
        float yl[16];
        float sumy = 0.0f;
        for (uint i = 0; i < 16; i++) { yl[i] = yb[i]; sumy += yb[i]; }
        for (uint r = 0; r < 4; r++) {
            const uint row = min(row0 + r, rlast);   // clamped rows compute, don't store
            const ushort b = *(device const ushort *)(weights + (ulong)row * (args.cols / 8) +
                                                      ib * 16 + il / 8);
            const float d = float(scales[(ulong)row * nb + ib]);
            float acc_pos = 0.0f;
            for (uint i = 0; i < 16; i++)
                acc_pos += select(0.0f, yl[i], bool(b & (1u << i)));
            sumf[r] += d * (2.0f * acc_pos - sumy);
        }
        yb += 512;
    }
    for (uint r = 0; r < 4; r++) {
        const float tot = simd_sum(sumf[r]);
        if (lane == 0 && row0 + r < args.rows) out[row0 + r] = tot;
    }
}

// Candidate 2 — IEEE sign-bit XOR: ±y materialized by flipping y's sign bit
// from the weight bit (bit==1 -> +y, bit==0 -> -y). Unconditional
// accumulate, no sum(y) correction term. Exact for all finite y (a flipped
// -0.0 contributes +0.0; float addition of signed zeros never perturbs a
// finite accumulator).
kernel void q27_matvec_b1_signxor(
        device const uchar *weights [[buffer(0)]],
        device const half  *scales  [[buffer(1)]],
        device const float *x       [[buffer(2)]],
        device       float *out     [[buffer(3)]],
        constant MatvecArgs &args   [[buffer(4)]],
        uint group                   [[threadgroup_position_in_grid]],
        ushort lane                  [[thread_index_in_simdgroup]],
        ushort simdgroup             [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = group * 32 + (uint)simdgroup * 4;   // 32 rows per threadgroup
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint nb = args.cols / 128;
    const uint ix = lane / 8;              // block in flight (4 per simdgroup)
    const uint il = (lane % 8) * 16;       // element offset within the block
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float *yb = x + ix * 128 + il;
    for (uint ib = ix; ib < nb; ib += 4) {
        uint yl[16];   // activation bits, sign flip applied bitwise below
        for (uint i = 0; i < 16; i++) yl[i] = as_type<uint>(yb[i]);
        for (uint r = 0; r < 4; r++) {
            const uint row = min(row0 + r, rlast);   // clamped rows compute, don't store
            const uint b = *(device const ushort *)(weights + (ulong)row * (args.cols / 8) +
                                                    ib * 16 + il / 8);
            const float d = float(scales[(ulong)row * nb + ib]);
            float acc = 0.0f;
            for (uint i = 0; i < 16; i++)
                acc += as_type<float>(yl[i] ^ ((~(b >> i) & 1u) << 31));
            sumf[r] += d * acc;
        }
        yb += 512;
    }
    for (uint r = 0; r < 4; r++) {
        const float tot = simd_sum(sumf[r]);
        if (lane == 0 && row0 + r < args.rows) out[row0 + r] = tot;
    }
}

// Candidate 3 activation preprocessing — int8-domain offset quantization,
// bitplane transpose, per-group sums. One 128-thread threadgroup per
// 128-column group g:
//   s_g   = max|x|/127 (0 if the group is all zero)
//   u_i   = clamp(round(x_i/s_g) + 128, 0, 255)   (128 when s_g == 0)
//   planes[g*32 + p*4 + w] = bit p of u over word w (columns w*32..w*32+31,
//                            bit position = column%32, via simd_ballot)
//   aux[g] = { s_g, sum(u) }
// The dot kernel then reconstructs x_i ~= s_g*(u_i - 128) exactly in the
// int8 model; the quantization error vs float activations is the candidate's
// accuracy cost and is gated separately in the bench.
kernel void q27_b1_x_prep(
        device const float  *x      [[buffer(0)]],
        device       uint   *planes [[buffer(1)]],
        device       float2 *aux    [[buffer(2)]],
        constant MatvecArgs &args   [[buffer(3)]],
        uint group                   [[threadgroup_position_in_grid]],
        ushort tid                   [[thread_index_in_threadgroup]],
        ushort lane                  [[thread_index_in_simdgroup]],
        ushort simdgroup             [[simdgroup_index_in_threadgroup]]) {
    const float xv = x[group * 128 + tid];
    // The max/sum phases overlap safely ONLY because pmax and psum are
    // distinct arrays (a thread may write psum while another still reads
    // pmax). The barrier below the pmax reads is insurance: it keeps this
    // correct if the two arrays are ever consolidated into one.
    threadgroup float pmax[4], psum[4];
    float amax = simd_max(fabs(xv));
    if (lane == 0) pmax[simdgroup] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = max(max(pmax[0], pmax[1]), max(pmax[2], pmax[3]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = amax / 127.0f;
    // k3 audit D1 scope: B1 keeps round(xv/s) — Metal-native (no CUDA
    // twin); matches its metal_gemv_bench CPU model and certified battery.
    const uint u = s > 0.0f ? uint(clamp(round(xv / s) + 128.0f, 0.0f, 255.0f)) : 128u;
    for (uint p = 0; p < 8; p++) {
        const uint m = uint(uint64_t(simd_ballot(bool((u >> p) & 1u))));
        if (lane == 0) planes[group * 32 + p * 4 + simdgroup] = m;
    }
    float sumu = simd_sum(float(u));
    if (lane == 0) psum[simdgroup] = sumu;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0)
        aux[group] = float2(s, psum[0] + psum[1] + psum[2] + psum[3]);
}

// Candidate 3 — int8 bitplane + popcount dot. Per 32-column word: 8
// AND+popcounts of the weight-bit word against the 8 activation bitplanes,
// plane p weighted 2^p. With w_i in {-1,+1} = 2b_i - 1 and
// x_i ~= s*(u_i - 128):
//   sum_i w_i*x_i = 2*sum_{b_i=1} x_i - sum_i x_i
//                 = s*( 2*(sum_p 2^p*popc(b & u_p) - 128*popc(b))
//                       - (sum(u) - 128*128) )
// One simdgroup per row (8 rows per threadgroup), lanes stride the groups;
// all popcount math stays in uint until the one float scale per group.
kernel void q27_matvec_b1_popcount(
        device const uint   *weights [[buffer(0)]],
        device const half   *scales  [[buffer(1)]],
        device const uint   *planes  [[buffer(2)]],
        device const float2 *aux     [[buffer(3)]],
        device       float  *out     [[buffer(4)]],
        constant MatvecArgs &args    [[buffer(5)]],
        uint group                    [[threadgroup_position_in_grid]],
        ushort lane                   [[thread_index_in_simdgroup]],
        ushort simdgroup              [[simdgroup_index_in_threadgroup]]) {
    const uint row = group * 8 + simdgroup;
    if (row >= args.rows) return;
    const uint nb = args.cols / 128;
    const ulong wbase = (ulong)row * (args.cols / 32);
    float sum = 0.0f;
    for (uint g = lane; g < nb; g += 32) {
        uint planesum = 0, wpop = 0;
        for (uint w = 0; w < 4; w++) {
            const uint wb = weights[wbase + g * 4 + w];
            wpop += popcount(wb);
            for (uint p = 0; p < 8; p++)
                planesum += popcount(wb & planes[g * 32 + p * 4 + w]) << p;
        }
        const float2 sa = aux[g];
        const float d = float(scales[(ulong)row * nb + g]);
        sum += d * sa.x * (2.0f * (float(int(planesum) - int(128 * wpop))) -
                           (sa.y - 16384.0f));
    }
    sum = simd_sum(sum);
    if (lane == 0) out[row] = sum;
}

struct VectorArgs {
    uint n;
    uint groups;
    float eps;
};

struct HeadArgs {
    uint heads;
    uint head_dim;
    uint stride;
    uint groups;
    float eps;
};

inline float reduce_sum(float value, threadgroup float *partial,
                        ushort lane, ushort simdgroup, uint groups) {
    value = simd_sum(value);
    if (lane == 0) partial[simdgroup] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        float total = lane < groups ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

kernel void q27_embedding_q8(
        device const char *weights [[buffer(0)]],
        device const half *scales  [[buffer(1)]],
        device float *out          [[buffer(2)]],
        constant uint &token       [[buffer(3)]],
        constant uint &cols        [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) return;
    const ulong wi = (ulong)token * cols + gid;
    const ulong si = (ulong)token * (cols / 128) + gid / 128;
    out[gid] = float(weights[wi]) * float(scales[si]);
}

kernel void q27_embedding_t2(
        device const uchar *weights [[buffer(0)]],
        device const half *scales   [[buffer(1)]],
        device float *out           [[buffer(2)]],
        constant uint &token        [[buffer(3)]],
        constant uint &cols         [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) return;
    const ulong wi = (ulong)token * cols + gid;   // cols % 128 == 0, so wi/4 is exact bytes
    const uint code = (weights[wi >> 2] >> ((wi & 3) * 2)) & 3;
    const ulong si = (ulong)token * (cols / 128) + gid / 128;
    out[gid] = float(int(code) - 1) * float(scales[si]);
}

kernel void q27_embedding_b1(
        device const uchar *weights [[buffer(0)]],
        device const half *scales   [[buffer(1)]],
        device float *out           [[buffer(2)]],
        constant uint &token        [[buffer(3)]],
        constant uint &cols         [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) return;
    const ulong wi = (ulong)token * cols + gid;   // cols % 128 == 0, so wi/8 is exact bytes
    const uint bit = (weights[wi >> 3] >> (wi & 7)) & 1;
    const ulong si = (ulong)token * (cols / 128) + gid / 128;
    out[gid] = float(2 * int(bit) - 1) * float(scales[si]);
}

// GPU-resident greedy decode: identical embedding lookups, but the token id
// comes from the device buffer the previous step's argmax wrote, so chained
// steps need no CPU sync. There is NO clamp on the id: a corrupt id WOULD
// read out of bounds. Safety rests entirely on the argmax invariant —
// argmax only writes ids < n, and mask+argmax degrades to id 0.
kernel void q27_embedding_q8_dev(
        device const char *weights [[buffer(0)]],
        device const half *scales  [[buffer(1)]],
        device float *out          [[buffer(2)]],
        device const uint *token   [[buffer(3)]],
        constant uint &cols        [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) return;
    const ulong wi = (ulong)token[0] * cols + gid;
    const ulong si = (ulong)token[0] * (cols / 128) + gid / 128;
    out[gid] = float(weights[wi]) * float(scales[si]);
}

kernel void q27_embedding_t2_dev(
        device const uchar *weights [[buffer(0)]],
        device const half *scales   [[buffer(1)]],
        device float *out           [[buffer(2)]],
        device const uint *token    [[buffer(3)]],
        constant uint &cols         [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) return;
    const ulong wi = (ulong)token[0] * cols + gid;
    const uint code = (weights[wi >> 2] >> ((wi & 3) * 2)) & 3;
    const ulong si = (ulong)token[0] * (cols / 128) + gid / 128;
    out[gid] = float(int(code) - 1) * float(scales[si]);
}

kernel void q27_embedding_b1_dev(
        device const uchar *weights [[buffer(0)]],
        device const half *scales   [[buffer(1)]],
        device float *out           [[buffer(2)]],
        device const uint *token    [[buffer(3)]],
        constant uint &cols         [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= cols) return;
    const ulong wi = (ulong)token[0] * cols + gid;
    const uint bit = (weights[wi >> 3] >> (wi & 7)) & 1;
    const ulong si = (ulong)token[0] * (cols / 128) + gid / 128;
    out[gid] = float(2 * int(bit) - 1) * float(scales[si]);
}

kernel void q27_rmsnorm(
        device const float *x [[buffer(0)]],
        device const float *w [[buffer(1)]],
        device float *out     [[buffer(2)]],
        constant VectorArgs &args [[buffer(3)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    float sum = 0.0f;
    for (uint i = tid; i < args.n; i += 256) sum += x[i] * x[i];
    threadgroup float partial[32];
    sum = reduce_sum(sum, partial, lane, simdgroup, args.groups);
    const float inv = rsqrt(sum / float(args.n) + args.eps);
    for (uint i = tid; i < args.n; i += 256) out[i] = x[i] * inv * w[i];
}

kernel void q27_rmsnorm_quantized(
        device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
        device float *out [[buffer(2)]], device char *values [[buffer(3)]],
        device float *scales [[buffer(4)]], constant VectorArgs &args [[buffer(5)]],
        uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    float sum=0.0f;
    for(uint i=tid;i<args.n;i+=256) sum+=x[i]*x[i];
    threadgroup float partial[32];
    sum=reduce_sum(sum,partial,lane,simdgroup,args.groups);
    const float inv=rsqrt(sum/float(args.n)+args.eps);
    for(uint i=tid;i<args.n;i+=256) out[i]=x[i]*inv*w[i];
    threadgroup_barrier(mem_flags::mem_device);
    const uint blocks=args.n/32;
    for(uint block=simdgroup;block<blocks;block+=8) {
        const uint i=block*32+lane; const float v=out[i];
        const float amax=simd_max(abs(v)); const float scale=amax/127.0f;
        // k3 audit D1: reciprocal-multiply matches CUDA k_quantize_x — one
        // rounding rule across backends (rint == __float2int_rn under RNE).
        const float qinv=scale>0.0f?1.0f/scale:0.0f;
        int q=int(rint(v*qinv)); q=clamp(q,-127,127);
        values[i]=char(q); if(lane==0) scales[block]=scale;
    }
}

kernel void q27_rmsnorm_heads(
        device float *x         [[buffer(0)]],
        device const float *w   [[buffer(1)]],
        constant HeadArgs &args [[buffer(2)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (head >= args.heads) return;
    device float *xh = x + (ulong)head * args.stride;
    float sum = 0.0f;
    for (uint i = tid; i < args.head_dim; i += 256) sum += xh[i] * xh[i];
    threadgroup float partial[32];
    sum = reduce_sum(sum, partial, lane, simdgroup, args.groups);
    const float inv = rsqrt(sum / float(args.head_dim) + args.eps);
    for (uint i = tid; i < args.head_dim; i += 256) xh[i] = xh[i] * inv * w[i];
}

kernel void q27_l2norm_heads(
        device float *x         [[buffer(0)]],
        constant HeadArgs &args [[buffer(1)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (head >= args.heads) return;
    device float *xh = x + (ulong)head * args.head_dim;
    float sum = 0.0f;
    for (uint i = tid; i < args.head_dim; i += 256) sum += xh[i] * xh[i];
    threadgroup float partial[32];
    sum = reduce_sum(sum, partial, lane, simdgroup, args.groups);
    const float inv = rsqrt(max(sum, args.eps * args.eps));
    for (uint i = tid; i < args.head_dim; i += 256) xh[i] *= inv;
}

kernel void q27_silu_mul(device const float *gate [[buffer(0)]],
                         device const float *up   [[buffer(1)]],
                         device float *out        [[buffer(2)]],
                         constant uint &n         [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = (gate[gid] / (1.0f + exp(-gate[gid]))) * up[gid];
}

kernel void q27_add_inplace(device float *x       [[buffer(0)]],
                            device const float *y [[buffer(1)]],
                            constant uint &n       [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid < n) x[gid] += y[gid];
}

struct GateArgs { uint heads; uint head_dim; };
kernel void q27_sigmoid_gate_mul(device float *out     [[buffer(0)]],
                                  device const float *qg [[buffer(1)]],
                                  constant GateArgs &args [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
    const uint n = args.heads * args.head_dim;
    if (gid >= n) return;
    const uint h = gid / args.head_dim;
    const uint d = gid % args.head_dim;
    const float gate = qg[(ulong)h * (2 * args.head_dim) + args.head_dim + d];
    out[gid] *= 1.0f / (1.0f + exp(-gate));
}

struct RopeArgs {
    uint heads;
    uint head_dim;
    uint n_rot;
    uint stride;
    uint position;
    float freq_base;
};
kernel void q27_rope_neox(device float *x [[buffer(0)]],
                           constant RopeArgs &args [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, head = gid.y;
    if (head >= args.heads || d >= args.n_rot / 2) return;
    device float *xh = x + (ulong)head * args.stride;
    const float theta = float(args.position) * pow(args.freq_base, -2.0f * float(d) / float(args.n_rot));
    const float cs = cos(theta), sn = sin(theta);
    const float x0 = xh[d], x1 = xh[d + args.n_rot / 2];
    xh[d] = x0 * cs - x1 * sn;
    xh[d + args.n_rot / 2] = x0 * sn + x1 * cs;
}

kernel void q27_argmax(device const float *x [[buffer(0)]],
                        device uint *out       [[buffer(1)]],
                        constant uint &n       [[buffer(2)]],
                        uint tid [[thread_index_in_threadgroup]]) {
    float best = -INFINITY;
    uint best_i = 0;
    for (uint i = tid; i < n; i += 256) {
        const float value = x[i];
        if (value > best || (value == best && i < best_i)) { best = value; best_i = i; }
    }
    threadgroup float values[256];
    threadgroup uint indices[256];
    values[tid] = best; indices[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) {
        if (tid < step) {
            const float other = values[tid + step];
            const uint other_i = indices[tid + step];
            if (other > values[tid] || (other == values[tid] && other_i < indices[tid])) {
                values[tid] = other; indices[tid] = other_i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[0] = indices[0];
}

struct KvStoreArgs { uint position; uint row_length; };
kernel void q27_kv_store_f16(device const float *k [[buffer(0)]],
                              device const float *v [[buffer(1)]],
                              device half *kc       [[buffer(2)]],
                              device half *vc       [[buffer(3)]],
                              constant KvStoreArgs &args [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= args.row_length) return;
    const ulong off = (ulong)args.position * args.row_length + gid;
    kc[off] = half(k[gid]); vc[off] = half(v[gid]);
}

struct AttentionArgs {
    uint q_stride;
    uint seq_len;
    uint q_heads;
    uint kv_heads;
    uint head_dim;
    float scale;
};
// Online-softmax decode attention: one threadgroup per query head, eight
// simdgroups striping the sequence. Each simdgroup keeps a running
// (max, denominator, weighted-value) triple entirely in registers — one
// lane holds head_dim/32 output dims — and the eight partials merge through
// threadgroup memory in log2 rounds. No probability scratch is materialized,
// so the cost is one streaming pass over the head's K and V rows.
kernel void q27_attention_f16(device const float *q [[buffer(0)]],
                               device const half *kc [[buffer(1)]],
                               device const half *vc [[buffer(2)]],
                               device float *out      [[buffer(3)]],
                               constant AttentionArgs &args [[buffer(4)]],
                               uint qh [[threadgroup_position_in_grid]],
                               ushort lane [[thread_index_in_simdgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (qh >= args.q_heads) return;
    const uint gqa = args.q_heads / args.kv_heads;
    const uint kvh = qh / gqa;
    device const float *qh_ptr = q + (ulong)qh * args.q_stride;

    float acc[8];                       // head_dim <= 256 -> at most 8 dims per lane
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint p = sg; p < args.seq_len; p += 8) {
        device const half *kh = kc + ((ulong)p * args.kv_heads + kvh) * args.head_dim;
        float partial = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32) partial += qh_ptr[d] * float(kh[d]);
        const float score = simd_sum(partial) * args.scale;
        const float m_new = max(m, score);
        const float correction = exp(m - m_new);    // first iteration: exp(-inf) = 0
        const float weight = exp(score - m_new);
        l = l * correction + weight;
        device const half *vh = vc + ((ulong)p * args.kv_heads + kvh) * args.head_dim;
        for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
            acc[i] = acc[i] * correction + weight * float(vh[d]);
        m = m_new;
    }

    // Merge the eight simdgroup partials: rounds of 4, 2, 1. A simdgroup that
    // saw no positions carries m = -inf, l = 0 and merges as a no-op.
    threadgroup float tg_m[4], tg_l[4], tg_acc[4][256];
    for (uint offset = 4; offset >= 1; offset /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg >= offset && sg < 2 * offset) {
            if (lane == 0) { tg_m[sg - offset] = m; tg_l[sg - offset] = l; }
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
                tg_acc[sg - offset][d] = acc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < offset) {
            const float m_other = tg_m[sg], l_other = tg_l[sg];
            const float m_new = max(m, m_other);
            if (m_new == -INFINITY) continue;       // both stripes empty
            const float c_mine = exp(m - m_new), c_other = exp(m_other - m_new);
            l = l * c_mine + l_other * c_other;
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
                acc[i] = acc[i] * c_mine + tg_acc[sg][d] * c_other;
            m = m_new;
        }
    }
    if (sg == 0) {
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
            out[(ulong)qh * args.head_dim + d] = acc[i] * inv;
    }
}

kernel void q27_gdn_gates(device const float *alpha [[buffer(0)]],
                           device const float *beta_raw [[buffer(1)]],
                           device const float *ssm_a [[buffer(2)]],
                           device const float *ssm_dt [[buffer(3)]],
                           device float *g [[buffer(4)]],
                           device float *beta [[buffer(5)]],
                           constant uint &heads [[buffer(6)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= heads) return;
    const float value = alpha[gid] + ssm_dt[gid];
    const float softplus = value > 20.0f ? value
                         : (value < -16.0f ? exp(value) : log1p_f(exp(value)));
    g[gid] = ssm_a[gid] * softplus;
    beta[gid] = 1.0f / (1.0f + exp(-beta_raw[gid]));
}

kernel void q27_conv_step(device const float *ring_src [[buffer(0)]],
                           device float *ring_dst       [[buffer(1)]],
                           device const float *qkv      [[buffer(2)]],
                           device const float *weight   [[buffer(3)]],
                           device float *out            [[buffer(4)]],
                           constant uint &channels      [[buffer(5)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= channels) return;
    const float value = ring_src[gid] * weight[(ulong)gid * 4] +
                        ring_src[channels + gid] * weight[(ulong)gid * 4 + 1] +
                        ring_src[(ulong)2 * channels + gid] * weight[(ulong)gid * 4 + 2] +
                        qkv[gid] * weight[(ulong)gid * 4 + 3];
    out[gid] = value / (1.0f + exp(-value));
    ring_dst[gid] = ring_src[channels + gid];
    ring_dst[channels + gid] = ring_src[(ulong)2 * channels + gid];
    ring_dst[(ulong)2 * channels + gid] = qkv[gid];
}

struct DeltaArgs { uint value_heads; uint qk_heads; uint head_dim; };
kernel void q27_delta_step(device const float *state_src [[buffer(0)]],
                            device float *state_dst       [[buffer(1)]],
                            device const float *conv      [[buffer(2)]],
                            device const float *g         [[buffer(3)]],
                            device const float *beta      [[buffer(4)]],
                            device float *out             [[buffer(5)]],
                            constant DeltaArgs &args      [[buffer(6)]],
                            uint head [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    if (head >= args.value_heads || args.head_dim != 128 || args.qk_heads != 16) return;
    const uint j = tid & 127;
    const uint tile = tid >> 7;
    const uint i0 = tile * 32;
    const uint qk = head % args.qk_heads;
    threadgroup float q[128], k[128], part[4][128], delta[128];
    if (tile == 0) { q[j] = conv[(ulong)qk * 128 + j] * rsqrt(128.0f); k[j] = conv[2048 + (ulong)qk * 128 + j]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float decay = exp(g[head]);
    device const float *source = state_src + (ulong)head * 128 * 128;
    device float *dest = state_dst + (ulong)head * 128 * 128;
    float saved[32];
    float prediction = 0.0f;
    for (uint n = 0; n < 32; n++) {
        const uint i = i0 + n;
        const float value = source[(ulong)i * 128 + j] * decay;
        saved[n] = value;
        prediction += k[i] * value;
    }
    part[tile][j] = prediction;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tile == 0) {
        const float predicted = part[0][j] + part[1][j] + part[2][j] + part[3][j];
        delta[j] = beta[head] * (conv[4096 + (ulong)head * 128 + j] - predicted);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result = 0.0f;
    for (uint n = 0; n < 32; n++) {
        const uint i = i0 + n;
        const float value = saved[n] + k[i] * delta[j];
        dest[(ulong)i * 128 + j] = value;
        result += q[i] * value;
    }
    part[tile][j] = result;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tile == 0) out[(ulong)head * 128 + j] = part[0][j] + part[1][j] + part[2][j] + part[3][j];
}

kernel void q27_gated_norm_gdn(device const float *x [[buffer(0)]],
                                device const float *weight [[buffer(1)]],
                                device const float *gate [[buffer(2)]],
                                device float *out [[buffer(3)]],
                                constant HeadArgs &args [[buffer(4)]],
                                uint head [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]],
                                ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (head >= args.heads) return;
    device const float *xh = x + (ulong)head * args.head_dim;
    device const float *gh = gate + (ulong)head * args.head_dim;
    float sum = 0.0f;
    for (uint i = tid; i < args.head_dim; i += 256) sum += xh[i] * xh[i];
    threadgroup float partial[32];
    sum = reduce_sum(sum, partial, lane, simdgroup, args.groups);
    const float inv = rsqrt(sum / float(args.head_dim) + args.eps);
    for (uint i = tid; i < args.head_dim; i += 256) {
        const float gv = gh[i];
        out[(ulong)head * args.head_dim + i] = xh[i] * inv * weight[i] * (gv / (1.0f + exp(-gv)));
    }
}

struct ConcatArgs { uint a_count; uint b_count; };
kernel void q27_concat(device const float *a [[buffer(0)]],
                        device const float *b [[buffer(1)]],
                        device float *out [[buffer(2)]],
                        constant ConcatArgs &args [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < args.a_count) out[gid] = a[gid];
    else if (gid < args.a_count + args.b_count) out[gid] = b[gid - args.a_count];
}

kernel void q27_quantize_x(device const float *x [[buffer(0)]],
                            device char *values [[buffer(1)]],
                            device float *scales [[buffer(2)]],
                            constant uint &count [[buffer(3)]],
                            uint group [[threadgroup_position_in_grid]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    const uint i = group * 32 + lane;
    float v = i < count ? x[i] : 0.0f;
    float amax = simd_max(abs(v));
    float scale = amax / 127.0f;
    // k3 audit D1: reciprocal-multiply matches CUDA k_quantize_x — one
    // rounding rule across backends (rint == __float2int_rn under RNE).
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    int q = int(rint(v * inv));
    q = clamp(q, -127, 127);
    if (i < count) values[i] = char(q);
    if (lane == 0) scales[group] = scale;
}

// Packed-dot GEMV: each lane loads 16 weight values at once, computes an
// exact integer dot against 16 int8 activations (max magnitude 16*127*127,
// exactly representable in float), applies the combined scale per lane, and
// reduces once per row. Tensor blobs are 256-byte aligned (loader ALIGN), so
// the vector loads are safe. A lane's 16 columns never straddle a weight or
// activation scale group. The 128-column tail loop covers synthetic shapes;
// production columns are all multiples of 512.
inline int q27_dot4(char4 a, char4 b) {
    return int(a.x) * int(b.x) + int(a.y) * int(b.y) +
           int(a.z) * int(b.z) + int(a.w) * int(b.w);
}

// Even columns sit in low nibbles, odd columns in high nibbles (even=low
// order). Scalar shift-extract measures faster on M4 than the vectorized
// mask/shuffle decode (54-64 vs 39-44 GB/s).
inline int q27_dot8_q4(uint packed, char4 x0, char4 x1) {
    int sum = (int(packed         & 15u) - 8) * x0.x;
    sum += (int((packed >>  4) & 15u) - 8) * x0.y;
    sum += (int((packed >>  8) & 15u) - 8) * x0.z;
    sum += (int((packed >> 12) & 15u) - 8) * x0.w;
    sum += (int((packed >> 16) & 15u) - 8) * x1.x;
    sum += (int((packed >> 20) & 15u) - 8) * x1.y;
    sum += (int((packed >> 24) & 15u) - 8) * x1.z;
    sum += (int((packed >> 28)      ) - 8) * x1.w;
    return sum;
}

kernel void q27_matvec_q8_quantized(device const char *weights [[buffer(0)]],
                                     device const half *weight_scales [[buffer(1)]],
                                     device const char *x [[buffer(2)]],
                                     device const float *x_scales [[buffer(3)]],
                                     device float *out [[buffer(4)]],
                                     constant MatvecArgs &args [[buffer(5)]],
                                     uint group [[threadgroup_position_in_grid]],
                                     ushort lane [[thread_index_in_simdgroup]],
                                     ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint row = group * 8 + simdgroup;
    if (row >= args.rows) return;
    device const int4 *w16 = (device const int4 *)(weights + (ulong)row * args.cols);
    device const int4 *x16 = (device const int4 *)x;
    const ulong scale_base = (ulong)row * (args.cols / 128);
    float acc = 0.0f;
    const uint chunks = args.cols / 1024;
    for (uint chunk = 0; chunk < chunks; chunk++) {
        const uint idx = chunk * 64 + lane * 2;
        const int4 wp0 = w16[idx], wp1 = w16[idx + 1];
        const int4 xp0 = x16[idx], xp1 = x16[idx + 1];
        const int dot0 = q27_dot4(as_type<char4>(wp0.x), as_type<char4>(xp0.x)) +
                         q27_dot4(as_type<char4>(wp0.y), as_type<char4>(xp0.y)) +
                         q27_dot4(as_type<char4>(wp0.z), as_type<char4>(xp0.z)) +
                         q27_dot4(as_type<char4>(wp0.w), as_type<char4>(xp0.w));
        const int dot1 = q27_dot4(as_type<char4>(wp1.x), as_type<char4>(xp1.x)) +
                         q27_dot4(as_type<char4>(wp1.y), as_type<char4>(xp1.y)) +
                         q27_dot4(as_type<char4>(wp1.z), as_type<char4>(xp1.z)) +
                         q27_dot4(as_type<char4>(wp1.w), as_type<char4>(xp1.w));
        // A lane's 32 columns are 32-aligned: both 16-column halves share one
        // activation-scale block and one weight-scale group, and the summed
        // integer dot (<= 32*127*127) stays exactly representable in float.
        const uint c = chunk * 1024 + lane * 32;
        acc += float(dot0 + dot1) *
               float(weight_scales[scale_base + c / 128]) * x_scales[c / 32];
    }
    for (uint c = chunks * 1024 + lane * 4; c < args.cols; c += 128) {
        const char4 wp = *(device const char4 *)(weights + (ulong)row * args.cols + c);
        const char4 xp = *(device const char4 *)(x + c);
        acc += float(q27_dot4(wp, xp)) *
               float(weight_scales[scale_base + c / 128]) * x_scales[c / 32];
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}

kernel void q27_matvec_q4_quantized(device const uchar *weights [[buffer(0)]],
                                     device const half *weight_scales [[buffer(1)]],
                                     device const char *x [[buffer(2)]],
                                     device const float *x_scales [[buffer(3)]],
                                     device float *out [[buffer(4)]],
                                     constant MatvecArgs &args [[buffer(5)]],
                                     uint group [[threadgroup_position_in_grid]],
                                     ushort lane [[thread_index_in_simdgroup]],
                                     ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint row = group * 8 + simdgroup;
    if (row >= args.rows) return;
    device const uint4 *w4x8 = (device const uint4 *)(weights + (ulong)row * (args.cols / 2));
    device const int4 *x16 = (device const int4 *)x;
    const ulong scale_base = (ulong)row * (args.cols / 64);
    float acc = 0.0f;
    const uint chunks = args.cols / 1024;
    for (uint chunk = 0; chunk < chunks; chunk++) {
        const uint idx = chunk * 32 + lane;
        const uint4 wp = w4x8[idx];
        const int4 xp0 = x16[idx * 2];
        const int4 xp1 = x16[idx * 2 + 1];
        const int dot0 = q27_dot8_q4(wp.x, as_type<char4>(xp0.x), as_type<char4>(xp0.y)) +
                         q27_dot8_q4(wp.y, as_type<char4>(xp0.z), as_type<char4>(xp0.w));
        const int dot1 = q27_dot8_q4(wp.z, as_type<char4>(xp1.x), as_type<char4>(xp1.y)) +
                         q27_dot8_q4(wp.w, as_type<char4>(xp1.z), as_type<char4>(xp1.w));
        // Same 32-aligned lane layout as the Q8 kernel: one activation-scale
        // block and one weight-scale group cover the lane's 32 columns.
        const uint c = chunk * 1024 + lane * 32;
        acc += float(dot0 + dot1) *
               float(weight_scales[scale_base + c / 64]) * x_scales[c / 32];
    }
    for (uint c = chunks * 1024 + lane * 4; c < args.cols; c += 128) {
        const uchar2 wp = *(device const uchar2 *)(weights + (ulong)row * (args.cols / 2) + c / 2);
        const char4 xp = *(device const char4 *)(x + c);
        const int dot = (int(wp.x & 15u) - 8) * xp.x + (int(wp.x >> 4) - 8) * xp.y +
                        (int(wp.y & 15u) - 8) * xp.z + (int(wp.y >> 4) - 8) * xp.w;
        acc += float(dot) * float(weight_scales[scale_base + c / 64]) * x_scales[c / 32];
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}

// 16 sequential LSB-first 2-bit codes per uint; code c contributes (c-1)*x.
// The integer dot is exact: |dot| <= 16*1*127 = 2032 per uint, 4064 per
// lane-chunk, well inside float's exact-integer range.
inline int q27_dot16_t2(uint packed, int4 xp) {
    const char4 x0 = as_type<char4>(xp.x), x1 = as_type<char4>(xp.y);
    const char4 x2 = as_type<char4>(xp.z), x3 = as_type<char4>(xp.w);
    int sum = (int(packed         & 3u) - 1) * x0.x;
    sum += (int((packed >>  2) & 3u) - 1) * x0.y;
    sum += (int((packed >>  4) & 3u) - 1) * x0.z;
    sum += (int((packed >>  6) & 3u) - 1) * x0.w;
    sum += (int((packed >>  8) & 3u) - 1) * x1.x;
    sum += (int((packed >> 10) & 3u) - 1) * x1.y;
    sum += (int((packed >> 12) & 3u) - 1) * x1.z;
    sum += (int((packed >> 14) & 3u) - 1) * x1.w;
    sum += (int((packed >> 16) & 3u) - 1) * x2.x;
    sum += (int((packed >> 18) & 3u) - 1) * x2.y;
    sum += (int((packed >> 20) & 3u) - 1) * x2.z;
    sum += (int((packed >> 22) & 3u) - 1) * x2.w;
    sum += (int((packed >> 24) & 3u) - 1) * x3.x;
    sum += (int((packed >> 26) & 3u) - 1) * x3.y;
    sum += (int((packed >> 28) & 3u) - 1) * x3.z;
    sum += (int((packed >> 30)      ) - 1) * x3.w;
    return sum;
}

// Packed-dot ternary GEMV, same skeleton as the Q4/Q8 kernels: a lane's 32
// columns (one uint2 of codes) are 32-aligned, so they share one activation-
// scale block and sit inside one 128-column weight-scale group.
kernel void q27_matvec_t2_quantized(device const uchar *weights [[buffer(0)]],
                                     device const half *weight_scales [[buffer(1)]],
                                     device const char *x [[buffer(2)]],
                                     device const float *x_scales [[buffer(3)]],
                                     device float *out [[buffer(4)]],
                                     constant MatvecArgs &args [[buffer(5)]],
                                     uint group [[threadgroup_position_in_grid]],
                                     ushort lane [[thread_index_in_simdgroup]],
                                     ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint row = group * 8 + simdgroup;
    if (row >= args.rows) return;
    device const uint2 *w2 = (device const uint2 *)(weights + (ulong)row * (args.cols / 4));
    device const int4 *x16 = (device const int4 *)x;
    const ulong scale_base = (ulong)row * (args.cols / 128);
    float acc = 0.0f;
    const uint chunks = args.cols / 1024;
    for (uint chunk = 0; chunk < chunks; chunk++) {
        const uint idx = chunk * 32 + lane;      // one uint2 = 32 columns
        const uint2 wp = w2[idx];
        const int dot0 = q27_dot16_t2(wp.x, x16[idx * 2]);
        const int dot1 = q27_dot16_t2(wp.y, x16[idx * 2 + 1]);
        const uint c = chunk * 1024 + lane * 32;
        acc += float(dot0 + dot1) *
               float(weight_scales[scale_base + c / 128]) * x_scales[c / 32];
    }
    for (uint c = chunks * 1024 + lane * 4; c < args.cols; c += 128) {
        const uchar wp = weights[(ulong)row * (args.cols / 4) + c / 4];
        const char4 xp = *(device const char4 *)(x + c);
        const int dot = (int(wp         & 3u) - 1) * xp.x + (int((wp >> 2) & 3u) - 1) * xp.y +
                        (int((wp >> 4) & 3u) - 1) * xp.z + (int((wp >> 6)      ) - 1) * xp.w;
        acc += float(dot) * float(weight_scales[scale_base + c / 128]) * x_scales[c / 32];
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}

// 16 sequential LSB-first 1-bit codes from the low 16 bits; bit b
// contributes (2b-1)*x. Select form in the int domain: 2*sum_{bit=1} x -
// sum(x) — one conditional add per element plus one shared total (the T2
// unpack pays a shift/mask/sub per code). Exact: |dot| <= 16*127 per call,
// well inside float's exact-integer range.
inline int q27_dot16_b1(uint bits, int4 xp) {
    const char4 x0 = as_type<char4>(xp.x), x1 = as_type<char4>(xp.y);
    const char4 x2 = as_type<char4>(xp.z), x3 = as_type<char4>(xp.w);
    int pos = 0, tot = 0, v;
    v = int(x0.x); tot += v; pos += select(0, v, bool(bits & 0x0001u));
    v = int(x0.y); tot += v; pos += select(0, v, bool(bits & 0x0002u));
    v = int(x0.z); tot += v; pos += select(0, v, bool(bits & 0x0004u));
    v = int(x0.w); tot += v; pos += select(0, v, bool(bits & 0x0008u));
    v = int(x1.x); tot += v; pos += select(0, v, bool(bits & 0x0010u));
    v = int(x1.y); tot += v; pos += select(0, v, bool(bits & 0x0020u));
    v = int(x1.z); tot += v; pos += select(0, v, bool(bits & 0x0040u));
    v = int(x1.w); tot += v; pos += select(0, v, bool(bits & 0x0080u));
    v = int(x2.x); tot += v; pos += select(0, v, bool(bits & 0x0100u));
    v = int(x2.y); tot += v; pos += select(0, v, bool(bits & 0x0200u));
    v = int(x2.z); tot += v; pos += select(0, v, bool(bits & 0x0400u));
    v = int(x2.w); tot += v; pos += select(0, v, bool(bits & 0x0800u));
    v = int(x3.x); tot += v; pos += select(0, v, bool(bits & 0x1000u));
    v = int(x3.y); tot += v; pos += select(0, v, bool(bits & 0x2000u));
    v = int(x3.z); tot += v; pos += select(0, v, bool(bits & 0x4000u));
    v = int(x3.w); tot += v; pos += select(0, v, bool(bits & 0x8000u));
    return 2 * pos - tot;
}

// Packed-dot binary GEMV, same skeleton as the T2 kernel: a lane's 32
// columns (one uint of code bits) are 32-aligned, so they share one
// activation-scale block and sit inside one 128-column weight-scale group.
kernel void q27_matvec_b1_quantized(device const uchar *weights [[buffer(0)]],
                                     device const half *weight_scales [[buffer(1)]],
                                     device const char *x [[buffer(2)]],
                                     device const float *x_scales [[buffer(3)]],
                                     device float *out [[buffer(4)]],
                                     constant MatvecArgs &args [[buffer(5)]],
                                     uint group [[threadgroup_position_in_grid]],
                                     ushort lane [[thread_index_in_simdgroup]],
                                     ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint row = group * 8 + simdgroup;
    if (row >= args.rows) return;
    device const uint *w1 = (device const uint *)(weights + (ulong)row * (args.cols / 8));
    device const int4 *x16 = (device const int4 *)x;
    const ulong scale_base = (ulong)row * (args.cols / 128);
    float acc = 0.0f;
    const uint chunks = args.cols / 1024;
    for (uint chunk = 0; chunk < chunks; chunk++) {
        const uint idx = chunk * 32 + lane;      // one uint = 32 columns
        const uint wp = w1[idx];
        const int dot0 = q27_dot16_b1(wp, x16[idx * 2]);
        const int dot1 = q27_dot16_b1(wp >> 16, x16[idx * 2 + 1]);
        const uint c = chunk * 1024 + lane * 32;
        acc += float(dot0 + dot1) *
               float(weight_scales[scale_base + c / 128]) * x_scales[c / 32];
    }
    for (uint c = chunks * 1024 + lane * 4; c < args.cols; c += 128) {
        const uint wp = uint(weights[(ulong)row * (args.cols / 8) + c / 8]) >> (c % 8);
        const char4 xp = *(device const char4 *)(x + c);
        const int tot = int(xp.x) + int(xp.y) + int(xp.z) + int(xp.w);
        const int pos = select(0, int(xp.x), bool(wp & 1u)) +
                        select(0, int(xp.y), bool(wp & 2u)) +
                        select(0, int(xp.z), bool(wp & 4u)) +
                        select(0, int(xp.w), bool(wp & 8u));
        acc += float(2 * pos - tot) *
               float(weight_scales[scale_base + c / 128]) * x_scales[c / 32];
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}

// Dual-row ternary dot: unpack each 2-bit code ONCE, MAC into both rows.
// The packed-dot GEMV is issue-bound on the unpack (shift/mask/sub per
// code), not on the weight stream, so sharing the unpack — not the bytes —
// is where an N=2 batch can win. Integer sums are exact and order-
// independent, so each row's dot equals the single-row kernel's bit for bit.
inline void q27_dot16_t2_dual(uint packed, int4 xpa, int4 xpb,
                              thread int& sa, thread int& sb) {
    const char4 a0 = as_type<char4>(xpa.x), a1 = as_type<char4>(xpa.y);
    const char4 a2 = as_type<char4>(xpa.z), a3 = as_type<char4>(xpa.w);
    const char4 b0 = as_type<char4>(xpb.x), b1 = as_type<char4>(xpb.y);
    const char4 b2 = as_type<char4>(xpb.z), b3 = as_type<char4>(xpb.w);
    int w;
    w = int(packed         & 3u) - 1; sa += w * a0.x; sb += w * b0.x;
    w = int((packed >>  2) & 3u) - 1; sa += w * a0.y; sb += w * b0.y;
    w = int((packed >>  4) & 3u) - 1; sa += w * a0.z; sb += w * b0.z;
    w = int((packed >>  6) & 3u) - 1; sa += w * a0.w; sb += w * b0.w;
    w = int((packed >>  8) & 3u) - 1; sa += w * a1.x; sb += w * b1.x;
    w = int((packed >> 10) & 3u) - 1; sa += w * a1.y; sb += w * b1.y;
    w = int((packed >> 12) & 3u) - 1; sa += w * a1.z; sb += w * b1.z;
    w = int((packed >> 14) & 3u) - 1; sa += w * a1.w; sb += w * b1.w;
    w = int((packed >> 16) & 3u) - 1; sa += w * a2.x; sb += w * b2.x;
    w = int((packed >> 18) & 3u) - 1; sa += w * a2.y; sb += w * b2.y;
    w = int((packed >> 20) & 3u) - 1; sa += w * a2.z; sb += w * b2.z;
    w = int((packed >> 22) & 3u) - 1; sa += w * a2.w; sb += w * b2.w;
    w = int((packed >> 24) & 3u) - 1; sa += w * a3.x; sb += w * b3.x;
    w = int((packed >> 26) & 3u) - 1; sa += w * a3.y; sb += w * b3.y;
    w = int((packed >> 28) & 3u) - 1; sa += w * a3.z; sb += w * b3.z;
    w = int((packed >> 30)      ) - 1; sa += w * a3.w; sb += w * b3.w;
}

// N=2 slot-batched ternary GEMV (multislot Phase 2 probe): one pass over
// the weight stream computes two activation rows, sharing the per-code
// unpack via q27_dot16_t2_dual. Layouts match the MM kernels: x row-major
// [2, cols], x_scales [2, cols/32], out token-major [2, rows]. Per-row
// K-striping, scale-multiply order, and simd_sum are IDENTICAL to
// q27_matvec_t2_quantized, and the integer dots are exact, so each row's
// output is bit-identical to the single-row kernel. (The fused-pair
// precedent — two weights, one x — lost on register pressure; here the
// shared work is the unpack, the actual issue-bound resource.)
// PARKED by measurement (2026-07-16, multislot Phase-2 probe): aggregate
// s_k 1.093 vs the 1.31 decision line — kept as the probe's reference
// surface, never engine-routed.
kernel void q27_matvec_t2_quantized_x2(device const uchar *weights [[buffer(0)]],
                                        device const half *weight_scales [[buffer(1)]],
                                        device const char *x [[buffer(2)]],
                                        device const float *x_scales [[buffer(3)]],
                                        device float *out [[buffer(4)]],
                                        constant MatvecArgs &args [[buffer(5)]],
                                        uint group [[threadgroup_position_in_grid]],
                                        ushort lane [[thread_index_in_simdgroup]],
                                        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint row = group * 8 + simdgroup;
    if (row >= args.rows) return;
    device const uint2 *w2 = (device const uint2 *)(weights + (ulong)row * (args.cols / 4));
    device const int4 *xa16 = (device const int4 *)x;
    device const int4 *xb16 = (device const int4 *)(x + args.cols);
    device const float *xas = x_scales;
    device const float *xbs = x_scales + args.cols / 32;
    const ulong scale_base = (ulong)row * (args.cols / 128);
    float acc_a = 0.0f, acc_b = 0.0f;
    const uint chunks = args.cols / 1024;
    for (uint chunk = 0; chunk < chunks; chunk++) {
        const uint idx = chunk * 32 + lane;      // one uint2 = 32 columns
        const uint2 wp = w2[idx];
        const uint c = chunk * 1024 + lane * 32;
        const float ws = float(weight_scales[scale_base + c / 128]);
        int da = 0, db = 0;
        q27_dot16_t2_dual(wp.x, xa16[idx * 2],     xb16[idx * 2],     da, db);
        q27_dot16_t2_dual(wp.y, xa16[idx * 2 + 1], xb16[idx * 2 + 1], da, db);
        acc_a += float(da) * ws * xas[c / 32];
        acc_b += float(db) * ws * xbs[c / 32];
    }
    for (uint c = chunks * 1024 + lane * 4; c < args.cols; c += 128) {
        const uchar wp = weights[(ulong)row * (args.cols / 4) + c / 4];
        const float ws = float(weight_scales[scale_base + c / 128]);
        const char4 xa = *(device const char4 *)(x + c);
        const char4 xb = *(device const char4 *)(x + args.cols + c);
        const int da = (int(wp         & 3u) - 1) * xa.x + (int((wp >> 2) & 3u) - 1) * xa.y +
                       (int((wp >> 4) & 3u) - 1) * xa.z + (int((wp >> 6)      ) - 1) * xa.w;
        const int db = (int(wp         & 3u) - 1) * xb.x + (int((wp >> 2) & 3u) - 1) * xb.y +
                       (int((wp >> 4) & 3u) - 1) * xb.z + (int((wp >> 6)      ) - 1) * xb.w;
        acc_a += float(da) * ws * xas[c / 32];
        acc_b += float(db) * ws * xbs[c / 32];
    }
    acc_a = simd_sum(acc_a);
    acc_b = simd_sum(acc_b);
    if (lane == 0) { out[row] = acc_a; out[args.rows + row] = acc_b; }
}

// Tiled simdgroup-matrix GEMM (x_rows 1..12) for chunked prefill and
// batched MTP verification. A 128-thread threadgroup (4 simdgroups) owns a
// 32-row x 16-token output tile and walks K in 64-column tiles: weights are
// staged in threadgroup memory as raw dequantized integers (exact in f32),
// activations as their int8 value times the per-32-column activation scale,
// and each simdgroup accumulates its 8-row stripe with 8x8x8
// multiply-accumulates. Every K-tile coincides with (or subdivides) one
// weight-scale group, so accumulators flush through per-simdgroup scratch
// once per tile, scaled by the row's weight scale, into per-lane running
// totals. This replaces a packed-dot scalar variant that was issue-bound at
// ~4 ops per multiply-add (2 char->int converts, mul, add): the matrix unit
// converts operands once per staged tile, not once per MAC. Numerics: the
// weight side stays integer-exact; the activation side rounds once per
// value at staging; cross-tile accumulation is sequential in K per output
// element. Not bit-identical to the serial GEMV (whose per-lane K-striping
// plus simd_sum tree orders float additions differently), but the committed-
// token A/B gates cover the difference, as with the earlier tile kernel.
// Row slots past rows-1 clamp their loads and drop their stores; token
// slots past x_rows-1 stage zeros and skip their stores.
kernel void q27_matmul_q4_mm(
        device const uchar *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const char *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Wt[32 * 64];   // [row within tile][col within K-tile]
    threadgroup float Xt[64 * 16];   // [col within K-tile][token], prescaled
    threadgroup float Sc[4 * 128];   // per-simdgroup flush scratch
    const uint row0 = group.x * 32;
    const uint tok0 = group.y * 16;   // 16-token tile (wide-chunk grid)
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    // Staging assignments: one weight row / 16 columns and one token /
    // 8 transposed columns per thread.
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 2);
    const uint xloc = tid % 16, xcb = (tid / 16) * 8;   // Xt column is tile-local
    const uint xtok = tok0 + xloc;                       // device rows are global
    const bool xvalid = xtok < args.x_rows;
    device const char *xsrc = x + (ulong)min(xtok, args.x_rows - 1) * args.cols;
    const uint xsbase = min(xtok, args.x_rows - 1) * (args.cols / 32);
    // The flush scratch holds both 8x8 tiles row-major: element i covers
    // tile i/64, row (i%64)/8, token (i/64)*8 + i%8. Each lane owns
    // elements lane, lane+32, lane+64, lane+96 as its running totals:
    // components x/z sit in row lane/8, components y/w in row lane/8 + 4.
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const ulong wsrowA = (ulong)min(rowA, rlast) * (args.cols / 64);
    const ulong wsrowB = (ulong)min(rowB, rlast) * (args.cols / 64);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float4 racc = 0.0f;
    threadgroup float *sc = Sc + sg * 128;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        {
            const uint2 wp = *(device const uint2 *)(wsrc + (c0 + wcb) / 2);
            threadgroup float *dst = Wt + wrow * 64 + wcb;
            dst[0]  = float(int(wp.x         & 15u) - 8);
            dst[1]  = float(int((wp.x >>  4) & 15u) - 8);
            dst[2]  = float(int((wp.x >>  8) & 15u) - 8);
            dst[3]  = float(int((wp.x >> 12) & 15u) - 8);
            dst[4]  = float(int((wp.x >> 16) & 15u) - 8);
            dst[5]  = float(int((wp.x >> 20) & 15u) - 8);
            dst[6]  = float(int((wp.x >> 24) & 15u) - 8);
            dst[7]  = float(int((wp.x >> 28)      ) - 8);
            dst[8]  = float(int(wp.y         & 15u) - 8);
            dst[9]  = float(int((wp.y >>  4) & 15u) - 8);
            dst[10] = float(int((wp.y >>  8) & 15u) - 8);
            dst[11] = float(int((wp.y >> 12) & 15u) - 8);
            dst[12] = float(int((wp.y >> 16) & 15u) - 8);
            dst[13] = float(int((wp.y >> 20) & 15u) - 8);
            dst[14] = float(int((wp.y >> 24) & 15u) - 8);
            dst[15] = float(int((wp.y >> 28)      ) - 8);
        }
        {
            const float xs = xvalid ? x_scales[xsbase + (c0 + xcb) / 32] : 0.0f;
            const char4 xa = *(device const char4 *)(xsrc + c0 + xcb);
            const char4 xb = *(device const char4 *)(xsrc + c0 + xcb + 4);
            threadgroup float *dst = Xt + xcb * 16 + xloc;
            dst[0 * 16] = float(xa.x) * xs; dst[1 * 16] = float(xa.y) * xs;
            dst[2 * 16] = float(xa.z) * xs; dst[3 * 16] = float(xa.w) * xs;
            dst[4 * 16] = float(xb.x) * xs; dst[5 * 16] = float(xb.y) * xs;
            dst[6 * 16] = float(xb.z) * xs; dst[7 * 16] = float(xb.w) * xs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < 64; k8 += 8) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc0, sc, 8);
        simdgroup_store(acc1, sc + 64, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const float wsA = float(weight_scales[wsrowA + c0 / 64]);
        const float wsB = float(weight_scales[wsrowB + c0 / 64]);
        racc += float4(sc[lane], sc[lane + 32], sc[lane + 64], sc[lane + 96]) *
                float4(wsA, wsB, wsA, wsB);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const uint tokA = tok0 + lane % 8, tokB = tok0 + 8 + lane % 8;
    if (rowA < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowA] = racc.x;
    if (rowB < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowB] = racc.y;
    if (rowA < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowA] = racc.z;
    if (rowB < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowB] = racc.w;
}

kernel void q27_matmul_q8_mm(
        device const char *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const char *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Wt[32 * 64];
    threadgroup float Xt[64 * 16];
    threadgroup float Sc[4 * 128];
    const uint row0 = group.x * 32;
    const uint tok0 = group.y * 16;   // 16-token tile (wide-chunk grid)
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const char *wsrc = weights + (ulong)min(row0 + wrow, rlast) * args.cols;
    const uint xloc = tid % 16, xcb = (tid / 16) * 8;   // Xt column is tile-local
    const uint xtok = tok0 + xloc;                       // device rows are global
    const bool xvalid = xtok < args.x_rows;
    device const char *xsrc = x + (ulong)min(xtok, args.x_rows - 1) * args.cols;
    const uint xsbase = min(xtok, args.x_rows - 1) * (args.cols / 32);
    // Lane element rows as in the Q4 kernel; the 64-column K-tile subdivides
    // the 128-column Q8 scale group, so the per-tile flush scale stays
    // constant within the tile as required.
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const ulong wsrowA = (ulong)min(rowA, rlast) * (args.cols / 128);
    const ulong wsrowB = (ulong)min(rowB, rlast) * (args.cols / 128);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float4 racc = 0.0f;
    threadgroup float *sc = Sc + sg * 128;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        {
            const int4 wp = *(device const int4 *)(wsrc + c0 + wcb);
            const char4 w0 = as_type<char4>(wp.x), w1 = as_type<char4>(wp.y);
            const char4 w2 = as_type<char4>(wp.z), w3 = as_type<char4>(wp.w);
            threadgroup float *dst = Wt + wrow * 64 + wcb;
            dst[0]  = float(w0.x); dst[1]  = float(w0.y); dst[2]  = float(w0.z); dst[3]  = float(w0.w);
            dst[4]  = float(w1.x); dst[5]  = float(w1.y); dst[6]  = float(w1.z); dst[7]  = float(w1.w);
            dst[8]  = float(w2.x); dst[9]  = float(w2.y); dst[10] = float(w2.z); dst[11] = float(w2.w);
            dst[12] = float(w3.x); dst[13] = float(w3.y); dst[14] = float(w3.z); dst[15] = float(w3.w);
        }
        {
            const float xs = xvalid ? x_scales[xsbase + (c0 + xcb) / 32] : 0.0f;
            const char4 xa = *(device const char4 *)(xsrc + c0 + xcb);
            const char4 xb = *(device const char4 *)(xsrc + c0 + xcb + 4);
            threadgroup float *dst = Xt + xcb * 16 + xloc;
            dst[0 * 16] = float(xa.x) * xs; dst[1 * 16] = float(xa.y) * xs;
            dst[2 * 16] = float(xa.z) * xs; dst[3 * 16] = float(xa.w) * xs;
            dst[4 * 16] = float(xb.x) * xs; dst[5 * 16] = float(xb.y) * xs;
            dst[6 * 16] = float(xb.z) * xs; dst[7 * 16] = float(xb.w) * xs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < 64; k8 += 8) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc0, sc, 8);
        simdgroup_store(acc1, sc + 64, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const float wsA = float(weight_scales[wsrowA + c0 / 128]);
        const float wsB = float(weight_scales[wsrowB + c0 / 128]);
        racc += float4(sc[lane], sc[lane + 32], sc[lane + 64], sc[lane + 96]) *
                float4(wsA, wsB, wsA, wsB);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const uint tokA = tok0 + lane % 8, tokB = tok0 + 8 + lane % 8;
    if (rowA < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowA] = racc.x;
    if (rowB < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowB] = racc.y;
    if (rowA < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowA] = racc.z;
    if (rowB < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowB] = racc.w;
}

// Ternary tiled chunk GEMM. Identical structure to q27_matmul_q8_mm — the
// 64-column K-tile subdivides the 128-column T2 scale group, so the per-tile
// flush scale stays constant within the tile — with the staging block
// decoding 16 sequential LSB-first 2-bit codes per uint into exact integers.
kernel void q27_matmul_t2_mm(
        device const uchar *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const char *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Wt[32 * 64];
    threadgroup float Xt[64 * 16];
    threadgroup float Sc[4 * 128];
    const uint row0 = group.x * 32;
    const uint tok0 = group.y * 16;   // 16-token tile (wide-chunk grid)
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 4);
    const uint xloc = tid % 16, xcb = (tid / 16) * 8;   // Xt column is tile-local
    const uint xtok = tok0 + xloc;                       // device rows are global
    const bool xvalid = xtok < args.x_rows;
    device const char *xsrc = x + (ulong)min(xtok, args.x_rows - 1) * args.cols;
    const uint xsbase = min(xtok, args.x_rows - 1) * (args.cols / 32);
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const ulong wsrowA = (ulong)min(rowA, rlast) * (args.cols / 128);
    const ulong wsrowB = (ulong)min(rowB, rlast) * (args.cols / 128);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float4 racc = 0.0f;
    threadgroup float *sc = Sc + sg * 128;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        {
            const uint wp = *(device const uint *)(wsrc + (c0 + wcb) / 4);
            threadgroup float *dst = Wt + wrow * 64 + wcb;
            dst[0]  = float(int(wp         & 3u) - 1);
            dst[1]  = float(int((wp >>  2) & 3u) - 1);
            dst[2]  = float(int((wp >>  4) & 3u) - 1);
            dst[3]  = float(int((wp >>  6) & 3u) - 1);
            dst[4]  = float(int((wp >>  8) & 3u) - 1);
            dst[5]  = float(int((wp >> 10) & 3u) - 1);
            dst[6]  = float(int((wp >> 12) & 3u) - 1);
            dst[7]  = float(int((wp >> 14) & 3u) - 1);
            dst[8]  = float(int((wp >> 16) & 3u) - 1);
            dst[9]  = float(int((wp >> 18) & 3u) - 1);
            dst[10] = float(int((wp >> 20) & 3u) - 1);
            dst[11] = float(int((wp >> 22) & 3u) - 1);
            dst[12] = float(int((wp >> 24) & 3u) - 1);
            dst[13] = float(int((wp >> 26) & 3u) - 1);
            dst[14] = float(int((wp >> 28) & 3u) - 1);
            dst[15] = float(int((wp >> 30)      ) - 1);
        }
        {
            const float xs = xvalid ? x_scales[xsbase + (c0 + xcb) / 32] : 0.0f;
            const char4 xa = *(device const char4 *)(xsrc + c0 + xcb);
            const char4 xb = *(device const char4 *)(xsrc + c0 + xcb + 4);
            threadgroup float *dst = Xt + xcb * 16 + xloc;
            dst[0 * 16] = float(xa.x) * xs; dst[1 * 16] = float(xa.y) * xs;
            dst[2 * 16] = float(xa.z) * xs; dst[3 * 16] = float(xa.w) * xs;
            dst[4 * 16] = float(xb.x) * xs; dst[5 * 16] = float(xb.y) * xs;
            dst[6 * 16] = float(xb.z) * xs; dst[7 * 16] = float(xb.w) * xs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < 64; k8 += 8) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc0, sc, 8);
        simdgroup_store(acc1, sc + 64, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const float wsA = float(weight_scales[wsrowA + c0 / 128]);
        const float wsB = float(weight_scales[wsrowB + c0 / 128]);
        racc += float4(sc[lane], sc[lane + 32], sc[lane + 64], sc[lane + 96]) *
                float4(wsA, wsB, wsA, wsB);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const uint tokA = tok0 + lane % 8, tokB = tok0 + 8 + lane % 8;
    if (rowA < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowA] = racc.x;
    if (rowB < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowB] = racc.y;
    if (rowA < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowA] = racc.z;
    if (rowB < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowB] = racc.w;
}

// Half-staging variant of the T2 chunk GEMM (default path,
// Q27_METAL_GEMM_HALF=0 opts out; docs/plans/2026-07-15-gemm-half-staging.md
// variant G′): tiles are half — doubled simdgroup-MMA rate, halved
// threadgroup traffic — and BOTH operands stay integer-exact in half: trits
// on the weight side, raw int8 on the activation side. Accumulators are
// FLOAT (mixed-precision MMA), keeping int8 x trit sums exact to 2^24.
// Scales fold at the flush instead of at staging: each 64-K staged tile
// accumulates its two 32-K sub-slabs (the x-scale groups) into separate
// accumulator pairs, folded into racc in ONE barrier region per tile with
// component (row, token) scaled by ws(row) * xs(token, sub-slab). (Variant
// A staged prescaled activations and failed the shape suite; half
// ACCUMULATION rounds past 2048 — variant B's 32-K half flush moved the
// 2K NLL +0.4%.)
// Trit code -> half bit pattern: (c-1) as f16 is one of three constants
// (code 3 decodes to +2, preserving the arithmetic unpack's behavior for
// a corrupt pack byte). The MMA roofline measured the unpack/convert
// chain at 13.5% of the production GEMM (C/Beq, docs/plans/2026-07-16-
// mma-roofline.md); constructing the bit pattern directly deletes the
// integer subtract and int->half convert per element — the roofline's
// ONE authorized targeted round. The staged halves are identical values,
// so kernel output is bit-identical to the arithmetic unpack (pre/post
// artifact A/B gates the change).
constant ushort q27_t2_half_lut[4] = {0xbc00, 0x0000, 0x3c00, 0x4000};

// Byte -> 4 trit halves (little-endian codes): one constant-memory gather
// replaces four shift/mask/select chains. 2 KB, generated from the 2-bit
// code map (code 3 -> +2.0h, matching the arithmetic unpack).
constant half4 q27_t2_half4_lut[256] = {
    half4(-1.0h, -1.0h, -1.0h, -1.0h),
    half4(0.0h, -1.0h, -1.0h, -1.0h),
    half4(1.0h, -1.0h, -1.0h, -1.0h),
    half4(2.0h, -1.0h, -1.0h, -1.0h),
    half4(-1.0h, 0.0h, -1.0h, -1.0h),
    half4(0.0h, 0.0h, -1.0h, -1.0h),
    half4(1.0h, 0.0h, -1.0h, -1.0h),
    half4(2.0h, 0.0h, -1.0h, -1.0h),
    half4(-1.0h, 1.0h, -1.0h, -1.0h),
    half4(0.0h, 1.0h, -1.0h, -1.0h),
    half4(1.0h, 1.0h, -1.0h, -1.0h),
    half4(2.0h, 1.0h, -1.0h, -1.0h),
    half4(-1.0h, 2.0h, -1.0h, -1.0h),
    half4(0.0h, 2.0h, -1.0h, -1.0h),
    half4(1.0h, 2.0h, -1.0h, -1.0h),
    half4(2.0h, 2.0h, -1.0h, -1.0h),
    half4(-1.0h, -1.0h, 0.0h, -1.0h),
    half4(0.0h, -1.0h, 0.0h, -1.0h),
    half4(1.0h, -1.0h, 0.0h, -1.0h),
    half4(2.0h, -1.0h, 0.0h, -1.0h),
    half4(-1.0h, 0.0h, 0.0h, -1.0h),
    half4(0.0h, 0.0h, 0.0h, -1.0h),
    half4(1.0h, 0.0h, 0.0h, -1.0h),
    half4(2.0h, 0.0h, 0.0h, -1.0h),
    half4(-1.0h, 1.0h, 0.0h, -1.0h),
    half4(0.0h, 1.0h, 0.0h, -1.0h),
    half4(1.0h, 1.0h, 0.0h, -1.0h),
    half4(2.0h, 1.0h, 0.0h, -1.0h),
    half4(-1.0h, 2.0h, 0.0h, -1.0h),
    half4(0.0h, 2.0h, 0.0h, -1.0h),
    half4(1.0h, 2.0h, 0.0h, -1.0h),
    half4(2.0h, 2.0h, 0.0h, -1.0h),
    half4(-1.0h, -1.0h, 1.0h, -1.0h),
    half4(0.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(2.0h, -1.0h, 1.0h, -1.0h),
    half4(-1.0h, 0.0h, 1.0h, -1.0h),
    half4(0.0h, 0.0h, 1.0h, -1.0h),
    half4(1.0h, 0.0h, 1.0h, -1.0h),
    half4(2.0h, 0.0h, 1.0h, -1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h),
    half4(0.0h, 1.0h, 1.0h, -1.0h),
    half4(1.0h, 1.0h, 1.0h, -1.0h),
    half4(2.0h, 1.0h, 1.0h, -1.0h),
    half4(-1.0h, 2.0h, 1.0h, -1.0h),
    half4(0.0h, 2.0h, 1.0h, -1.0h),
    half4(1.0h, 2.0h, 1.0h, -1.0h),
    half4(2.0h, 2.0h, 1.0h, -1.0h),
    half4(-1.0h, -1.0h, 2.0h, -1.0h),
    half4(0.0h, -1.0h, 2.0h, -1.0h),
    half4(1.0h, -1.0h, 2.0h, -1.0h),
    half4(2.0h, -1.0h, 2.0h, -1.0h),
    half4(-1.0h, 0.0h, 2.0h, -1.0h),
    half4(0.0h, 0.0h, 2.0h, -1.0h),
    half4(1.0h, 0.0h, 2.0h, -1.0h),
    half4(2.0h, 0.0h, 2.0h, -1.0h),
    half4(-1.0h, 1.0h, 2.0h, -1.0h),
    half4(0.0h, 1.0h, 2.0h, -1.0h),
    half4(1.0h, 1.0h, 2.0h, -1.0h),
    half4(2.0h, 1.0h, 2.0h, -1.0h),
    half4(-1.0h, 2.0h, 2.0h, -1.0h),
    half4(0.0h, 2.0h, 2.0h, -1.0h),
    half4(1.0h, 2.0h, 2.0h, -1.0h),
    half4(2.0h, 2.0h, 2.0h, -1.0h),
    half4(-1.0h, -1.0h, -1.0h, 0.0h),
    half4(0.0h, -1.0h, -1.0h, 0.0h),
    half4(1.0h, -1.0h, -1.0h, 0.0h),
    half4(2.0h, -1.0h, -1.0h, 0.0h),
    half4(-1.0h, 0.0h, -1.0h, 0.0h),
    half4(0.0h, 0.0h, -1.0h, 0.0h),
    half4(1.0h, 0.0h, -1.0h, 0.0h),
    half4(2.0h, 0.0h, -1.0h, 0.0h),
    half4(-1.0h, 1.0h, -1.0h, 0.0h),
    half4(0.0h, 1.0h, -1.0h, 0.0h),
    half4(1.0h, 1.0h, -1.0h, 0.0h),
    half4(2.0h, 1.0h, -1.0h, 0.0h),
    half4(-1.0h, 2.0h, -1.0h, 0.0h),
    half4(0.0h, 2.0h, -1.0h, 0.0h),
    half4(1.0h, 2.0h, -1.0h, 0.0h),
    half4(2.0h, 2.0h, -1.0h, 0.0h),
    half4(-1.0h, -1.0h, 0.0h, 0.0h),
    half4(0.0h, -1.0h, 0.0h, 0.0h),
    half4(1.0h, -1.0h, 0.0h, 0.0h),
    half4(2.0h, -1.0h, 0.0h, 0.0h),
    half4(-1.0h, 0.0h, 0.0h, 0.0h),
    half4(0.0h, 0.0h, 0.0h, 0.0h),
    half4(1.0h, 0.0h, 0.0h, 0.0h),
    half4(2.0h, 0.0h, 0.0h, 0.0h),
    half4(-1.0h, 1.0h, 0.0h, 0.0h),
    half4(0.0h, 1.0h, 0.0h, 0.0h),
    half4(1.0h, 1.0h, 0.0h, 0.0h),
    half4(2.0h, 1.0h, 0.0h, 0.0h),
    half4(-1.0h, 2.0h, 0.0h, 0.0h),
    half4(0.0h, 2.0h, 0.0h, 0.0h),
    half4(1.0h, 2.0h, 0.0h, 0.0h),
    half4(2.0h, 2.0h, 0.0h, 0.0h),
    half4(-1.0h, -1.0h, 1.0h, 0.0h),
    half4(0.0h, -1.0h, 1.0h, 0.0h),
    half4(1.0h, -1.0h, 1.0h, 0.0h),
    half4(2.0h, -1.0h, 1.0h, 0.0h),
    half4(-1.0h, 0.0h, 1.0h, 0.0h),
    half4(0.0h, 0.0h, 1.0h, 0.0h),
    half4(1.0h, 0.0h, 1.0h, 0.0h),
    half4(2.0h, 0.0h, 1.0h, 0.0h),
    half4(-1.0h, 1.0h, 1.0h, 0.0h),
    half4(0.0h, 1.0h, 1.0h, 0.0h),
    half4(1.0h, 1.0h, 1.0h, 0.0h),
    half4(2.0h, 1.0h, 1.0h, 0.0h),
    half4(-1.0h, 2.0h, 1.0h, 0.0h),
    half4(0.0h, 2.0h, 1.0h, 0.0h),
    half4(1.0h, 2.0h, 1.0h, 0.0h),
    half4(2.0h, 2.0h, 1.0h, 0.0h),
    half4(-1.0h, -1.0h, 2.0h, 0.0h),
    half4(0.0h, -1.0h, 2.0h, 0.0h),
    half4(1.0h, -1.0h, 2.0h, 0.0h),
    half4(2.0h, -1.0h, 2.0h, 0.0h),
    half4(-1.0h, 0.0h, 2.0h, 0.0h),
    half4(0.0h, 0.0h, 2.0h, 0.0h),
    half4(1.0h, 0.0h, 2.0h, 0.0h),
    half4(2.0h, 0.0h, 2.0h, 0.0h),
    half4(-1.0h, 1.0h, 2.0h, 0.0h),
    half4(0.0h, 1.0h, 2.0h, 0.0h),
    half4(1.0h, 1.0h, 2.0h, 0.0h),
    half4(2.0h, 1.0h, 2.0h, 0.0h),
    half4(-1.0h, 2.0h, 2.0h, 0.0h),
    half4(0.0h, 2.0h, 2.0h, 0.0h),
    half4(1.0h, 2.0h, 2.0h, 0.0h),
    half4(2.0h, 2.0h, 2.0h, 0.0h),
    half4(-1.0h, -1.0h, -1.0h, 1.0h),
    half4(0.0h, -1.0h, -1.0h, 1.0h),
    half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(2.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, 0.0h, -1.0h, 1.0h),
    half4(0.0h, 0.0h, -1.0h, 1.0h),
    half4(1.0h, 0.0h, -1.0h, 1.0h),
    half4(2.0h, 0.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h),
    half4(0.0h, 1.0h, -1.0h, 1.0h),
    half4(1.0h, 1.0h, -1.0h, 1.0h),
    half4(2.0h, 1.0h, -1.0h, 1.0h),
    half4(-1.0h, 2.0h, -1.0h, 1.0h),
    half4(0.0h, 2.0h, -1.0h, 1.0h),
    half4(1.0h, 2.0h, -1.0h, 1.0h),
    half4(2.0h, 2.0h, -1.0h, 1.0h),
    half4(-1.0h, -1.0h, 0.0h, 1.0h),
    half4(0.0h, -1.0h, 0.0h, 1.0h),
    half4(1.0h, -1.0h, 0.0h, 1.0h),
    half4(2.0h, -1.0h, 0.0h, 1.0h),
    half4(-1.0h, 0.0h, 0.0h, 1.0h),
    half4(0.0h, 0.0h, 0.0h, 1.0h),
    half4(1.0h, 0.0h, 0.0h, 1.0h),
    half4(2.0h, 0.0h, 0.0h, 1.0h),
    half4(-1.0h, 1.0h, 0.0h, 1.0h),
    half4(0.0h, 1.0h, 0.0h, 1.0h),
    half4(1.0h, 1.0h, 0.0h, 1.0h),
    half4(2.0h, 1.0h, 0.0h, 1.0h),
    half4(-1.0h, 2.0h, 0.0h, 1.0h),
    half4(0.0h, 2.0h, 0.0h, 1.0h),
    half4(1.0h, 2.0h, 0.0h, 1.0h),
    half4(2.0h, 2.0h, 0.0h, 1.0h),
    half4(-1.0h, -1.0h, 1.0h, 1.0h),
    half4(0.0h, -1.0h, 1.0h, 1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h),
    half4(2.0h, -1.0h, 1.0h, 1.0h),
    half4(-1.0h, 0.0h, 1.0h, 1.0h),
    half4(0.0h, 0.0h, 1.0h, 1.0h),
    half4(1.0h, 0.0h, 1.0h, 1.0h),
    half4(2.0h, 0.0h, 1.0h, 1.0h),
    half4(-1.0h, 1.0h, 1.0h, 1.0h),
    half4(0.0h, 1.0h, 1.0h, 1.0h),
    half4(1.0h, 1.0h, 1.0h, 1.0h),
    half4(2.0h, 1.0h, 1.0h, 1.0h),
    half4(-1.0h, 2.0h, 1.0h, 1.0h),
    half4(0.0h, 2.0h, 1.0h, 1.0h),
    half4(1.0h, 2.0h, 1.0h, 1.0h),
    half4(2.0h, 2.0h, 1.0h, 1.0h),
    half4(-1.0h, -1.0h, 2.0h, 1.0h),
    half4(0.0h, -1.0h, 2.0h, 1.0h),
    half4(1.0h, -1.0h, 2.0h, 1.0h),
    half4(2.0h, -1.0h, 2.0h, 1.0h),
    half4(-1.0h, 0.0h, 2.0h, 1.0h),
    half4(0.0h, 0.0h, 2.0h, 1.0h),
    half4(1.0h, 0.0h, 2.0h, 1.0h),
    half4(2.0h, 0.0h, 2.0h, 1.0h),
    half4(-1.0h, 1.0h, 2.0h, 1.0h),
    half4(0.0h, 1.0h, 2.0h, 1.0h),
    half4(1.0h, 1.0h, 2.0h, 1.0h),
    half4(2.0h, 1.0h, 2.0h, 1.0h),
    half4(-1.0h, 2.0h, 2.0h, 1.0h),
    half4(0.0h, 2.0h, 2.0h, 1.0h),
    half4(1.0h, 2.0h, 2.0h, 1.0h),
    half4(2.0h, 2.0h, 2.0h, 1.0h),
    half4(-1.0h, -1.0h, -1.0h, 2.0h),
    half4(0.0h, -1.0h, -1.0h, 2.0h),
    half4(1.0h, -1.0h, -1.0h, 2.0h),
    half4(2.0h, -1.0h, -1.0h, 2.0h),
    half4(-1.0h, 0.0h, -1.0h, 2.0h),
    half4(0.0h, 0.0h, -1.0h, 2.0h),
    half4(1.0h, 0.0h, -1.0h, 2.0h),
    half4(2.0h, 0.0h, -1.0h, 2.0h),
    half4(-1.0h, 1.0h, -1.0h, 2.0h),
    half4(0.0h, 1.0h, -1.0h, 2.0h),
    half4(1.0h, 1.0h, -1.0h, 2.0h),
    half4(2.0h, 1.0h, -1.0h, 2.0h),
    half4(-1.0h, 2.0h, -1.0h, 2.0h),
    half4(0.0h, 2.0h, -1.0h, 2.0h),
    half4(1.0h, 2.0h, -1.0h, 2.0h),
    half4(2.0h, 2.0h, -1.0h, 2.0h),
    half4(-1.0h, -1.0h, 0.0h, 2.0h),
    half4(0.0h, -1.0h, 0.0h, 2.0h),
    half4(1.0h, -1.0h, 0.0h, 2.0h),
    half4(2.0h, -1.0h, 0.0h, 2.0h),
    half4(-1.0h, 0.0h, 0.0h, 2.0h),
    half4(0.0h, 0.0h, 0.0h, 2.0h),
    half4(1.0h, 0.0h, 0.0h, 2.0h),
    half4(2.0h, 0.0h, 0.0h, 2.0h),
    half4(-1.0h, 1.0h, 0.0h, 2.0h),
    half4(0.0h, 1.0h, 0.0h, 2.0h),
    half4(1.0h, 1.0h, 0.0h, 2.0h),
    half4(2.0h, 1.0h, 0.0h, 2.0h),
    half4(-1.0h, 2.0h, 0.0h, 2.0h),
    half4(0.0h, 2.0h, 0.0h, 2.0h),
    half4(1.0h, 2.0h, 0.0h, 2.0h),
    half4(2.0h, 2.0h, 0.0h, 2.0h),
    half4(-1.0h, -1.0h, 1.0h, 2.0h),
    half4(0.0h, -1.0h, 1.0h, 2.0h),
    half4(1.0h, -1.0h, 1.0h, 2.0h),
    half4(2.0h, -1.0h, 1.0h, 2.0h),
    half4(-1.0h, 0.0h, 1.0h, 2.0h),
    half4(0.0h, 0.0h, 1.0h, 2.0h),
    half4(1.0h, 0.0h, 1.0h, 2.0h),
    half4(2.0h, 0.0h, 1.0h, 2.0h),
    half4(-1.0h, 1.0h, 1.0h, 2.0h),
    half4(0.0h, 1.0h, 1.0h, 2.0h),
    half4(1.0h, 1.0h, 1.0h, 2.0h),
    half4(2.0h, 1.0h, 1.0h, 2.0h),
    half4(-1.0h, 2.0h, 1.0h, 2.0h),
    half4(0.0h, 2.0h, 1.0h, 2.0h),
    half4(1.0h, 2.0h, 1.0h, 2.0h),
    half4(2.0h, 2.0h, 1.0h, 2.0h),
    half4(-1.0h, -1.0h, 2.0h, 2.0h),
    half4(0.0h, -1.0h, 2.0h, 2.0h),
    half4(1.0h, -1.0h, 2.0h, 2.0h),
    half4(2.0h, -1.0h, 2.0h, 2.0h),
    half4(-1.0h, 0.0h, 2.0h, 2.0h),
    half4(0.0h, 0.0h, 2.0h, 2.0h),
    half4(1.0h, 0.0h, 2.0h, 2.0h),
    half4(2.0h, 0.0h, 2.0h, 2.0h),
    half4(-1.0h, 1.0h, 2.0h, 2.0h),
    half4(0.0h, 1.0h, 2.0h, 2.0h),
    half4(1.0h, 1.0h, 2.0h, 2.0h),
    half4(2.0h, 1.0h, 2.0h, 2.0h),
    half4(-1.0h, 2.0h, 2.0h, 2.0h),
    half4(0.0h, 2.0h, 2.0h, 2.0h),
    half4(1.0h, 2.0h, 2.0h, 2.0h),
    half4(2.0h, 2.0h, 2.0h, 2.0h),
};

kernel void q27_matmul_t2_mm_h(
        device const uchar *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const char *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Wt[32 * 64];
    threadgroup half Xt[64 * 16];
    threadgroup float Sc[4 * 256];
    const uint row0 = group.x * 32;
    const uint tok0 = group.y * 16;   // 16-token tile (wide-chunk grid)
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 4);
    const uint xloc = tid % 16, xcb = (tid / 16) * 8;   // Xt column is tile-local
    const uint xtok = tok0 + xloc;                       // device rows are global
    device const char *xsrc = x + (ulong)min(xtok, args.x_rows - 1) * args.cols;
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const ulong wsrowA = (ulong)min(rowA, rlast) * (args.cols / 128);
    const ulong wsrowB = (ulong)min(rowB, rlast) * (args.cols / 128);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float4 racc = 0.0f;
    threadgroup float *sc = Sc + sg * 256;
    const uint tokA = tok0 + lane % 8, tokB = tok0 + 8 + lane % 8;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        {
            const uint wp = *(device const uint *)(wsrc + (c0 + wcb) / 4);
            // Roofline round (docs/plans/2026-07-16-mma-roofline.md): one
            // byte-LUT gather per 4 trits + 4 vector stores replaces the
            // 16 shift/mask/int-sub/convert chains and 16 scalar stores —
            // staged halves are identical values, so output is
            // bit-identical to the arithmetic unpack. (K=128 staging was
            // also tried in the round and REGRESSED — 16 KB threadgroup
            // footprint cost more occupancy than the halved barrier
            // cadence saved; K=64 stands, as K=32 already showed from the
            // other side. wcb is a multiple of 16 -> half4-aligned.)
            threadgroup half4 *dst = (threadgroup half4 *)(Wt + wrow * 64 + wcb);
            dst[0] = q27_t2_half4_lut[wp         & 0xffu];
            dst[1] = q27_t2_half4_lut[(wp >>  8) & 0xffu];
            dst[2] = q27_t2_half4_lut[(wp >> 16) & 0xffu];
            dst[3] = q27_t2_half4_lut[wp >> 24         ];
        }
        {
            const char4 xa = *(device const char4 *)(xsrc + c0 + xcb);
            const char4 xb = *(device const char4 *)(xsrc + c0 + xcb + 4);
            threadgroup half *dst = Xt + xcb * 16 + xloc;
            // Raw int8 values: exact in half. The per-token 32-group scale
            // folds at the flush below; invalid token slots stage clamped
            // real values whose outputs are never stored. (Stores are
            // 16-strided by tile layout — not vectorizable; a device-side
            // int8->half pre-pass measured C/Cx = 1.006, not worth a
            // kernel + ABI bump.)
            dst[0 * 16] = half(xa.x); dst[1 * 16] = half(xa.y);
            dst[2 * 16] = half(xa.z); dst[3 * 16] = half(xa.w);
            dst[4 * 16] = half(xb.x); dst[5 * 16] = half(xb.y);
            dst[6 * 16] = half(xb.z); dst[7 * 16] = half(xb.w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float wsA = float(weight_scales[wsrowA + c0 / 128]);
        const float wsB = float(weight_scales[wsrowB + c0 / 128]);
        // Float accumulators keep int8 x trit sums exact to 2^24. The two
        // 32-K sub-slabs (activation-scale groups) accumulate into separate
        // tile pairs so both fold in ONE barrier region per staged 64.
        for (uint k8 = 0; k8 < 32; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        for (uint k8 = 32; k8 < 64; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
        simdgroup_store(acc0, sc, 8);
        simdgroup_store(acc1, sc + 64, 8);
        simdgroup_store(acc2, sc + 128, 8);
        simdgroup_store(acc3, sc + 192, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            const ulong xrow_a = (ulong)min(tokA, args.x_rows - 1) * (args.cols / 32);
            const ulong xrow_b = (ulong)min(tokB, args.x_rows - 1) * (args.cols / 32);
            const float xsA0 = x_scales[xrow_a + c0 / 32],     xsB0 = x_scales[xrow_b + c0 / 32];
            const float xsA1 = x_scales[xrow_a + c0 / 32 + 1], xsB1 = x_scales[xrow_b + c0 / 32 + 1];
            racc += float4(sc[lane], sc[lane + 32], sc[lane + 64], sc[lane + 96]) *
                    float4(wsA * xsA0, wsB * xsA0, wsA * xsB0, wsB * xsB0);
            racc += float4(sc[lane + 128], sc[lane + 160], sc[lane + 192], sc[lane + 224]) *
                    float4(wsA * xsA1, wsB * xsA1, wsA * xsB1, wsB * xsB1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (rowA < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowA] = racc.x;
    if (rowB < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowB] = racc.y;
    if (rowA < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowA] = racc.z;
    if (rowB < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowB] = racc.w;
}


// Nibble -> 4 binary-code halves (bit j = element j, LSB-first): one
// constant-memory gather replaces four shift/mask/select chains. Bit b
// stages (2b-1) as f16 — exact, so the mm kernel's math mirrors mm_h's.
constant half4 q27_b1_half4_lut[16] = {
    half4(-1.0h, -1.0h, -1.0h, -1.0h),
    half4( 1.0h, -1.0h, -1.0h, -1.0h),
    half4(-1.0h,  1.0h, -1.0h, -1.0h),
    half4( 1.0h,  1.0h, -1.0h, -1.0h),
    half4(-1.0h, -1.0h,  1.0h, -1.0h),
    half4( 1.0h, -1.0h,  1.0h, -1.0h),
    half4(-1.0h,  1.0h,  1.0h, -1.0h),
    half4( 1.0h,  1.0h,  1.0h, -1.0h),
    half4(-1.0h, -1.0h, -1.0h,  1.0h),
    half4( 1.0h, -1.0h, -1.0h,  1.0h),
    half4(-1.0h,  1.0h, -1.0h,  1.0h),
    half4( 1.0h,  1.0h, -1.0h,  1.0h),
    half4(-1.0h, -1.0h,  1.0h,  1.0h),
    half4( 1.0h, -1.0h,  1.0h,  1.0h),
    half4(-1.0h,  1.0h,  1.0h,  1.0h),
    half4( 1.0h,  1.0h,  1.0h,  1.0h),
};

// Binary tiled chunk GEMM on the half-staging pattern of q27_matmul_t2_mm_h
// (the production T2 route): half tiles, {-1,+1} weights and raw int8
// activations both integer-exact in half, activation scale folded at the
// per-32-K flush. The staging block reads one ushort (16 code bits) per
// thread — (c0 + wcb)/8 is even because wcb is a multiple of 16, so the
// load is ushort-aligned.
kernel void q27_matmul_b1_mm(
        device const uchar *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const char *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Wt[32 * 64];
    threadgroup half Xt[64 * 16];
    threadgroup float Sc[4 * 256];
    const uint row0 = group.x * 32;
    const uint tok0 = group.y * 16;   // 16-token tile (wide-chunk grid)
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 8);
    const uint xloc = tid % 16, xcb = (tid / 16) * 8;   // Xt column is tile-local
    const uint xtok = tok0 + xloc;                       // device rows are global
    device const char *xsrc = x + (ulong)min(xtok, args.x_rows - 1) * args.cols;
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const ulong wsrowA = (ulong)min(rowA, rlast) * (args.cols / 128);
    const ulong wsrowB = (ulong)min(rowB, rlast) * (args.cols / 128);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float4 racc = 0.0f;
    threadgroup float *sc = Sc + sg * 256;
    const uint tokA = tok0 + lane % 8, tokB = tok0 + 8 + lane % 8;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        {
            const ushort wp = *(device const ushort *)(wsrc + (c0 + wcb) / 8);
            threadgroup half4 *dst = (threadgroup half4 *)(Wt + wrow * 64 + wcb);
            dst[0] = q27_b1_half4_lut[wp         & 0xfu];
            dst[1] = q27_b1_half4_lut[(wp >>  4) & 0xfu];
            dst[2] = q27_b1_half4_lut[(wp >>  8) & 0xfu];
            dst[3] = q27_b1_half4_lut[wp >> 12        ];
        }
        {
            const char4 xa = *(device const char4 *)(xsrc + c0 + xcb);
            const char4 xb = *(device const char4 *)(xsrc + c0 + xcb + 4);
            threadgroup half *dst = Xt + xcb * 16 + xloc;
            // Raw int8 values: exact in half. The per-token 32-group scale
            // folds at the flush below; invalid token slots stage clamped
            // real values whose outputs are never stored.
            dst[0 * 16] = half(xa.x); dst[1 * 16] = half(xa.y);
            dst[2 * 16] = half(xa.z); dst[3 * 16] = half(xa.w);
            dst[4 * 16] = half(xb.x); dst[5 * 16] = half(xb.y);
            dst[6 * 16] = half(xb.z); dst[7 * 16] = half(xb.w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float wsA = float(weight_scales[wsrowA + c0 / 128]);
        const float wsB = float(weight_scales[wsrowB + c0 / 128]);
        // Float accumulators keep int8 x {-1,+1} sums exact. The two 32-K
        // sub-slabs (activation-scale groups) accumulate into separate tile
        // pairs so both fold in ONE barrier region per staged 64.
        for (uint k8 = 0; k8 < 32; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        for (uint k8 = 32; k8 < 64; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
        simdgroup_store(acc0, sc, 8);
        simdgroup_store(acc1, sc + 64, 8);
        simdgroup_store(acc2, sc + 128, 8);
        simdgroup_store(acc3, sc + 192, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            const ulong xrow_a = (ulong)min(tokA, args.x_rows - 1) * (args.cols / 32);
            const ulong xrow_b = (ulong)min(tokB, args.x_rows - 1) * (args.cols / 32);
            const float xsA0 = x_scales[xrow_a + c0 / 32],     xsB0 = x_scales[xrow_b + c0 / 32];
            const float xsA1 = x_scales[xrow_a + c0 / 32 + 1], xsB1 = x_scales[xrow_b + c0 / 32 + 1];
            racc += float4(sc[lane], sc[lane + 32], sc[lane + 64], sc[lane + 96]) *
                    float4(wsA * xsA0, wsB * xsA0, wsA * xsB0, wsB * xsB0);
            racc += float4(sc[lane + 128], sc[lane + 160], sc[lane + 192], sc[lane + 224]) *
                    float4(wsA * xsA1, wsB * xsA1, wsA * xsB1, wsB * xsB1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (rowA < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowA] = racc.x;
    if (rowB < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowB] = racc.y;
    if (rowA < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowA] = racc.z;
    if (rowB < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowB] = racc.w;
}


// ---- A/B/C MMA roofline probes (bench-only, docs/plans/2026-07-16-mma-
// roofline.md). Same dispatch grid, tile geometry, edge clamps, and
// physical MMA count as q27_matmul_t2_mm_h (arm C); never engine-routed.

// Arm K — function-constant specialization probe (BaseRT survey probe 2):
// the production mm_h body with the K dimension baked as a function
// constant, so the 64-K walk's trip count and every cols-derived address
// expression are compile-time literals (per-shape specialized PSO). Same
// math in the same order — output must be bit-identical to arm C.
constant uint FC_COLS [[function_constant(0)]];
kernel void q27_mma_roofline_k(
        device const uchar *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const char *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Wt[32 * 64];
    threadgroup half Xt[64 * 16];
    threadgroup float Sc[4 * 256];
    const uint row0 = group.x * 32;
    const uint tok0 = group.y * 16;
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (FC_COLS / 4);
    const uint xloc = tid % 16, xcb = (tid / 16) * 8;
    const uint xtok = tok0 + xloc;
    device const char *xsrc = x + (ulong)min(xtok, args.x_rows - 1) * FC_COLS;
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const ulong wsrowA = (ulong)min(rowA, rlast) * (FC_COLS / 128);
    const ulong wsrowB = (ulong)min(rowB, rlast) * (FC_COLS / 128);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float4 racc = 0.0f;
    threadgroup float *sc = Sc + sg * 256;
    const uint tokA = tok0 + lane % 8, tokB = tok0 + 8 + lane % 8;
    for (uint c0 = 0; c0 < FC_COLS; c0 += 64) {
        {
            const uint wp = *(device const uint *)(wsrc + (c0 + wcb) / 4);
            threadgroup half4 *dst = (threadgroup half4 *)(Wt + wrow * 64 + wcb);
            dst[0] = q27_t2_half4_lut[wp         & 0xffu];
            dst[1] = q27_t2_half4_lut[(wp >>  8) & 0xffu];
            dst[2] = q27_t2_half4_lut[(wp >> 16) & 0xffu];
            dst[3] = q27_t2_half4_lut[wp >> 24         ];
        }
        {
            const char4 xa = *(device const char4 *)(xsrc + c0 + xcb);
            const char4 xb = *(device const char4 *)(xsrc + c0 + xcb + 4);
            threadgroup half *dst = Xt + xcb * 16 + xloc;
            dst[0 * 16] = half(xa.x); dst[1 * 16] = half(xa.y);
            dst[2 * 16] = half(xa.z); dst[3 * 16] = half(xa.w);
            dst[4 * 16] = half(xb.x); dst[5 * 16] = half(xb.y);
            dst[6 * 16] = half(xb.z); dst[7 * 16] = half(xb.w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float wsA = float(weight_scales[wsrowA + c0 / 128]);
        const float wsB = float(weight_scales[wsrowB + c0 / 128]);
        for (uint k8 = 0; k8 < 32; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        for (uint k8 = 32; k8 < 64; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
        simdgroup_store(acc0, sc, 8);
        simdgroup_store(acc1, sc + 64, 8);
        simdgroup_store(acc2, sc + 128, 8);
        simdgroup_store(acc3, sc + 192, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            const ulong xrow_a = (ulong)min(tokA, args.x_rows - 1) * (FC_COLS / 32);
            const ulong xrow_b = (ulong)min(tokB, args.x_rows - 1) * (FC_COLS / 32);
            const float xsA0 = x_scales[xrow_a + c0 / 32],     xsB0 = x_scales[xrow_b + c0 / 32];
            const float xsA1 = x_scales[xrow_a + c0 / 32 + 1], xsB1 = x_scales[xrow_b + c0 / 32 + 1];
            racc += float4(sc[lane], sc[lane + 32], sc[lane + 64], sc[lane + 96]) *
                    float4(wsA * xsA0, wsB * xsA0, wsA * xsB0, wsB * xsB0);
            racc += float4(sc[lane + 128], sc[lane + 160], sc[lane + 192], sc[lane + 224]) *
                    float4(wsA * xsA1, wsB * xsA1, wsA * xsB1, wsB * xsB1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (rowA < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowA] = racc.x;
    if (rowB < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowB] = racc.y;
    if (rowA < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowA] = racc.z;
    if (rowB < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowB] = racc.w;
}

// Arm F — f16-accumulate probe (docs/plans/2026-07-16-f16acc-probe.md,
// BaseRT survey import #1): the production mm_h body with half
// accumulators and a half flush tile. The f16 sums live only within one
// 64-K slab (|sum| <= 64*127*2, inside half range); the scale-fold and
// cross-slab accumulation stay f32 in racc, unchanged. Bench-only.
kernel void q27_mma_roofline_f(
        device const uchar *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const char *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Wt[32 * 64];
    threadgroup half Xt[64 * 16];
    threadgroup half Sc[4 * 256];
    const uint row0 = group.x * 32;
    const uint tok0 = group.y * 16;
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 4);
    const uint xloc = tid % 16, xcb = (tid / 16) * 8;
    const uint xtok = tok0 + xloc;
    device const char *xsrc = x + (ulong)min(xtok, args.x_rows - 1) * args.cols;
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const ulong wsrowA = (ulong)min(rowA, rlast) * (args.cols / 128);
    const ulong wsrowB = (ulong)min(rowB, rlast) * (args.cols / 128);
    simdgroup_half8x8 acc0 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    simdgroup_half8x8 acc1 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    simdgroup_half8x8 acc2 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    simdgroup_half8x8 acc3 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    float4 racc = 0.0f;
    threadgroup half *sc = Sc + sg * 256;
    const uint tokA = tok0 + lane % 8, tokB = tok0 + 8 + lane % 8;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        {
            const uint wp = *(device const uint *)(wsrc + (c0 + wcb) / 4);
            threadgroup half4 *dst = (threadgroup half4 *)(Wt + wrow * 64 + wcb);
            dst[0] = q27_t2_half4_lut[wp         & 0xffu];
            dst[1] = q27_t2_half4_lut[(wp >>  8) & 0xffu];
            dst[2] = q27_t2_half4_lut[(wp >> 16) & 0xffu];
            dst[3] = q27_t2_half4_lut[wp >> 24         ];
        }
        {
            const char4 xa = *(device const char4 *)(xsrc + c0 + xcb);
            const char4 xb = *(device const char4 *)(xsrc + c0 + xcb + 4);
            threadgroup half *dst = Xt + xcb * 16 + xloc;
            dst[0 * 16] = half(xa.x); dst[1 * 16] = half(xa.y);
            dst[2 * 16] = half(xa.z); dst[3 * 16] = half(xa.w);
            dst[4 * 16] = half(xb.x); dst[5 * 16] = half(xb.y);
            dst[6 * 16] = half(xb.z); dst[7 * 16] = half(xb.w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float wsA = float(weight_scales[wsrowA + c0 / 128]);
        const float wsB = float(weight_scales[wsrowB + c0 / 128]);
        for (uint k8 = 0; k8 < 32; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        for (uint k8 = 32; k8 < 64; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
        simdgroup_store(acc0, sc, 8);
        simdgroup_store(acc1, sc + 64, 8);
        simdgroup_store(acc2, sc + 128, 8);
        simdgroup_store(acc3, sc + 192, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            const ulong xrow_a = (ulong)min(tokA, args.x_rows - 1) * (args.cols / 32);
            const ulong xrow_b = (ulong)min(tokB, args.x_rows - 1) * (args.cols / 32);
            const float xsA0 = x_scales[xrow_a + c0 / 32],     xsB0 = x_scales[xrow_b + c0 / 32];
            const float xsA1 = x_scales[xrow_a + c0 / 32 + 1], xsB1 = x_scales[xrow_b + c0 / 32 + 1];
            racc += float4(sc[lane], sc[lane + 32], sc[lane + 64], sc[lane + 96]) *
                    float4(wsA * xsA0, wsB * xsA0, wsA * xsB0, wsB * xsB0);
            racc += float4(sc[lane + 128], sc[lane + 160], sc[lane + 192], sc[lane + 224]) *
                    float4(wsA * xsA1, wsB * xsA1, wsA * xsB1, wsB * xsB1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        acc0 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        acc1 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        acc2 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        acc3 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (rowA < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowA] = racc.x;
    if (rowB < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowB] = racc.y;
    if (rowA < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowA] = racc.z;
    if (rowB < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowB] = racc.w;
}

// Arm B — half-plumbing roofline: the mm_h body with the packed-trit
// unpack and int8->half conversion deleted. Operands are already half in
// device memory and stage into the same threadgroup tiles with the same
// store pattern, barriers, MMA loops, and per-64-K ws*xs scale-fold flush.
// B - C isolates unpack/conversion machinery only.
kernel void q27_mma_roofline_b(
        device const half *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const half *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Wt[32 * 64];
    threadgroup half Xt[64 * 16];
    threadgroup float Sc[4 * 256];
    const uint row0 = group.x * 32;
    const uint tok0 = group.y * 16;
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const half *wsrc = weights + (ulong)min(row0 + wrow, rlast) * args.cols;
    const uint xloc = tid % 16, xcb = (tid / 16) * 8;
    const uint xtok = tok0 + xloc;
    device const half *xsrc = x + (ulong)min(xtok, args.x_rows - 1) * args.cols;
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const ulong wsrowA = (ulong)min(rowA, rlast) * (args.cols / 128);
    const ulong wsrowB = (ulong)min(rowB, rlast) * (args.cols / 128);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float4 racc = 0.0f;
    threadgroup float *sc = Sc + sg * 256;
    const uint tokA = tok0 + lane % 8, tokB = tok0 + 8 + lane % 8;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        {
            device const half4 *wp4 = (device const half4 *)(wsrc + c0 + wcb);
            threadgroup half *dst = Wt + wrow * 64 + wcb;
            const half4 w0 = wp4[0], w1 = wp4[1], w2 = wp4[2], w3 = wp4[3];
            dst[0]  = w0.x; dst[1]  = w0.y; dst[2]  = w0.z; dst[3]  = w0.w;
            dst[4]  = w1.x; dst[5]  = w1.y; dst[6]  = w1.z; dst[7]  = w1.w;
            dst[8]  = w2.x; dst[9]  = w2.y; dst[10] = w2.z; dst[11] = w2.w;
            dst[12] = w3.x; dst[13] = w3.y; dst[14] = w3.z; dst[15] = w3.w;
        }
        {
            device const half4 *xp4 = (device const half4 *)(xsrc + c0 + xcb);
            const half4 xa = xp4[0], xb = xp4[1];
            threadgroup half *dst = Xt + xcb * 16 + xloc;
            dst[0 * 16] = xa.x; dst[1 * 16] = xa.y;
            dst[2 * 16] = xa.z; dst[3 * 16] = xa.w;
            dst[4 * 16] = xb.x; dst[5 * 16] = xb.y;
            dst[6 * 16] = xb.z; dst[7 * 16] = xb.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float wsA = float(weight_scales[wsrowA + c0 / 128]);
        const float wsB = float(weight_scales[wsrowB + c0 / 128]);
        for (uint k8 = 0; k8 < 32; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        for (uint k8 = 32; k8 < 64; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
        simdgroup_store(acc0, sc, 8);
        simdgroup_store(acc1, sc + 64, 8);
        simdgroup_store(acc2, sc + 128, 8);
        simdgroup_store(acc3, sc + 192, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            const ulong xrow_a = (ulong)min(tokA, args.x_rows - 1) * (args.cols / 32);
            const ulong xrow_b = (ulong)min(tokB, args.x_rows - 1) * (args.cols / 32);
            const float xsA0 = x_scales[xrow_a + c0 / 32],     xsB0 = x_scales[xrow_b + c0 / 32];
            const float xsA1 = x_scales[xrow_a + c0 / 32 + 1], xsB1 = x_scales[xrow_b + c0 / 32 + 1];
            racc += float4(sc[lane], sc[lane + 32], sc[lane + 64], sc[lane + 96]) *
                    float4(wsA * xsA0, wsB * xsA0, wsA * xsB0, wsB * xsB0);
            racc += float4(sc[lane + 128], sc[lane + 160], sc[lane + 192], sc[lane + 224]) *
                    float4(wsA * xsA1, wsB * xsA1, wsA * xsB1, wsB * xsB1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (rowA < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowA] = racc.x;
    if (rowB < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowB] = racc.y;
    if (rowA < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowA] = racc.z;
    if (rowB < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowB] = racc.w;
}

// Arm B_eq — equal-traffic no-unpack roofline (codex P1 on the landing
// commit: literal arm B streams 8x the weight bytes of C, confounding the
// unpack cost with bandwidth). This variant loads EXACTLY C's device bytes
// per thread per tile (one half2 = 4 B for the weight, one half4 = 8 B for
// the activation), replicates the values into the same 16/8 staging
// stores, and keeps barriers, MMA loops, and the scale-fold flush
// identical. C/B_eq isolates unpack + conversion ALU at equal traffic;
// MMA timing is data-independent, so the replicated values don't matter.
kernel void q27_mma_roofline_b_eq(
        device const half *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const half *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Wt[32 * 64];
    threadgroup half Xt[64 * 16];
    threadgroup float Sc[4 * 256];
    const uint row0 = group.x * 32;
    const uint tok0 = group.y * 16;
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    // Packed-equivalent layout: cols/8 halves per row (= cols/4 bytes, C's
    // packed weight row size).
    device const half *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 8);
    const uint xloc = tid % 16, xcb = (tid / 16) * 8;
    const uint xtok = tok0 + xloc;
    device const half *xsrc = x + (ulong)min(xtok, args.x_rows - 1) * (args.cols / 2);
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const ulong wsrowA = (ulong)min(rowA, rlast) * (args.cols / 128);
    const ulong wsrowB = (ulong)min(rowB, rlast) * (args.cols / 128);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float4 racc = 0.0f;
    threadgroup float *sc = Sc + sg * 256;
    const uint tokA = tok0 + lane % 8, tokB = tok0 + 8 + lane % 8;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        {
            const half2 w2 = *(device const half2 *)(wsrc + (c0 + wcb) / 8);
            // Vector stores mirroring the production staging (codex P1: the
            // control must not fall behind the kernel it isolates).
            threadgroup half4 *dst = (threadgroup half4 *)(Wt + wrow * 64 + wcb);
            dst[0] = half4(w2.x, w2.y, w2.x, w2.y);
            dst[1] = half4(w2.x, w2.y, w2.x, w2.y);
            dst[2] = half4(w2.y, w2.x, w2.y, w2.x);
            dst[3] = half4(w2.y, w2.x, w2.y, w2.x);
        }
        {
            // Two 4-byte loads, matching C's two char4 loads per slot
            // (codex P2: one 8-byte load would understate load-issue cost).
            const half2 xa2 = *(device const half2 *)(xsrc + (c0 + xcb) / 2);
            const half2 xb2 = *(device const half2 *)(xsrc + (c0 + xcb) / 2 + 2);
            threadgroup half *dst = Xt + xcb * 16 + xloc;
            dst[0 * 16] = xa2.x; dst[1 * 16] = xa2.y;
            dst[2 * 16] = xb2.x; dst[3 * 16] = xb2.y;
            dst[4 * 16] = xb2.y; dst[5 * 16] = xb2.x;
            dst[6 * 16] = xa2.y; dst[7 * 16] = xa2.x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float wsA = float(weight_scales[wsrowA + c0 / 128]);
        const float wsB = float(weight_scales[wsrowB + c0 / 128]);
        for (uint k8 = 0; k8 < 32; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        for (uint k8 = 32; k8 < 64; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
        simdgroup_store(acc0, sc, 8);
        simdgroup_store(acc1, sc + 64, 8);
        simdgroup_store(acc2, sc + 128, 8);
        simdgroup_store(acc3, sc + 192, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            const ulong xrow_a = (ulong)min(tokA, args.x_rows - 1) * (args.cols / 32);
            const ulong xrow_b = (ulong)min(tokB, args.x_rows - 1) * (args.cols / 32);
            const float xsA0 = x_scales[xrow_a + c0 / 32],     xsB0 = x_scales[xrow_b + c0 / 32];
            const float xsA1 = x_scales[xrow_a + c0 / 32 + 1], xsB1 = x_scales[xrow_b + c0 / 32 + 1];
            racc += float4(sc[lane], sc[lane + 32], sc[lane + 64], sc[lane + 96]) *
                    float4(wsA * xsA0, wsB * xsA0, wsA * xsB0, wsB * xsB0);
            racc += float4(sc[lane + 128], sc[lane + 160], sc[lane + 192], sc[lane + 224]) *
                    float4(wsA * xsA1, wsB * xsA1, wsA * xsB1, wsB * xsB1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (rowA < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowA] = racc.x;
    if (rowB < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowB] = racc.y;
    if (rowA < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowA] = racc.z;
    if (rowB < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowB] = racc.w;
}

// Arm Cx — the pre-converted-activation candidate: production packed-T2
// weight staging (LUT unpack) with activations already half in device
// memory (raw int8 values converted once device-side are exact in half,
// so this candidate would be bit-identical in production; scale still
// folds at the flush). Cx measures the ceiling of deleting the per-tile
// char->half converts before building the production pre-pass.
kernel void q27_mma_roofline_cx(
        device const uchar *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const half *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Wt[32 * 64];
    threadgroup half Xt[64 * 16];
    threadgroup float Sc[4 * 256];
    const uint row0 = group.x * 32;
    const uint tok0 = group.y * 16;
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 4);
    const uint xloc = tid % 16, xcb = (tid / 16) * 8;
    const uint xtok = tok0 + xloc;
    device const half *xsrc = x + (ulong)min(xtok, args.x_rows - 1) * args.cols;
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const ulong wsrowA = (ulong)min(rowA, rlast) * (args.cols / 128);
    const ulong wsrowB = (ulong)min(rowB, rlast) * (args.cols / 128);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float4 racc = 0.0f;
    threadgroup float *sc = Sc + sg * 256;
    const uint tokA = tok0 + lane % 8, tokB = tok0 + 8 + lane % 8;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        {
            const uint wp = *(device const uint *)(wsrc + (c0 + wcb) / 4);
            // Exact copy of the production weight staging (codex P2: Cx must
            // differ from C only on the activation side).
            threadgroup half4 *dst = (threadgroup half4 *)(Wt + wrow * 64 + wcb);
            dst[0] = q27_t2_half4_lut[wp         & 0xffu];
            dst[1] = q27_t2_half4_lut[(wp >>  8) & 0xffu];
            dst[2] = q27_t2_half4_lut[(wp >> 16) & 0xffu];
            dst[3] = q27_t2_half4_lut[wp >> 24         ];
        }
        {
            const half4 xa = *(device const half4 *)(xsrc + c0 + xcb);
            const half4 xb = *(device const half4 *)(xsrc + c0 + xcb + 4);
            threadgroup half *dst = Xt + xcb * 16 + xloc;
            dst[0 * 16] = xa.x; dst[1 * 16] = xa.y;
            dst[2 * 16] = xa.z; dst[3 * 16] = xa.w;
            dst[4 * 16] = xb.x; dst[5 * 16] = xb.y;
            dst[6 * 16] = xb.z; dst[7 * 16] = xb.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float wsA = float(weight_scales[wsrowA + c0 / 128]);
        const float wsB = float(weight_scales[wsrowB + c0 / 128]);
        for (uint k8 = 0; k8 < 32; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        for (uint k8 = 32; k8 < 64; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
        simdgroup_store(acc0, sc, 8);
        simdgroup_store(acc1, sc + 64, 8);
        simdgroup_store(acc2, sc + 128, 8);
        simdgroup_store(acc3, sc + 192, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            const ulong xrow_a = (ulong)min(tokA, args.x_rows - 1) * (args.cols / 32);
            const ulong xrow_b = (ulong)min(tokB, args.x_rows - 1) * (args.cols / 32);
            const float xsA0 = x_scales[xrow_a + c0 / 32],     xsB0 = x_scales[xrow_b + c0 / 32];
            const float xsA1 = x_scales[xrow_a + c0 / 32 + 1], xsB1 = x_scales[xrow_b + c0 / 32 + 1];
            racc += float4(sc[lane], sc[lane + 32], sc[lane + 64], sc[lane + 96]) *
                    float4(wsA * xsA0, wsB * xsA0, wsA * xsB0, wsB * xsB0);
            racc += float4(sc[lane + 128], sc[lane + 160], sc[lane + 192], sc[lane + 224]) *
                    float4(wsA * xsA1, wsB * xsA1, wsA * xsB1, wsB * xsB1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (rowA < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowA] = racc.x;
    if (rowB < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowB] = racc.y;
    if (rowA < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowA] = racc.z;
    if (rowB < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowB] = racc.w;
}

// ---- Lever 1: direct-RHS chunk GEMM (docs/plans/2026-07-16-lever1-
// direct-rhs.md; bench-first, never engine-routed until it clears its
// pre-registered line). Structure per ds4 item 2/3 phase B: stage ONLY the
// weight tile (64 rows x 32 K, half, byte-LUT unpack), read the RHS
// directly from device via simdgroup_load, spend threadgroup memory on
// weights. K-tile = 32 = exactly one activation-scale group, so the
// per-K-tile flush folds ws(row) * xs(token) onto exact integer partial
// sums (int8 x trit, same exactness class as mm_h; fold order differs
// from mm_h so outputs are tolerance-gated, not bit-identical).

// RHS pre-pass: int8 [x_rows, cols] -> half K-major [cols, tokens_pad],
// raw int8 values (exact in half), zero-padded token slots. args reuse:
// simdgroups field carries tokens_pad.
kernel void q27_x_int8_to_half_t(device const char *x [[buffer(0)]],
                                  device half *xt [[buffer(1)]],
                                  constant MatmulArgs &args [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
    const uint tokens_pad = args.simdgroups;
    const uint c = gid / tokens_pad, t = gid % tokens_pad;
    if (c >= args.cols) return;
    xt[gid] = t < args.x_rows ? half(x[(ulong)t * args.cols + c]) : half(0.0h);
}

kernel void q27_matmul_t2_mm_dr(
        device const uchar *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const half *xt [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Wt[64 * 32];
    threadgroup float Sc[8 * 64];      // per-simdgroup 8x8 flush scratch
    const uint row0 = group.x * 64;
    const uint tok0 = group.y * 32;
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint tokens_pad = args.simdgroups;   // K-major RHS row stride
    const uint live = min(32u, args.x_rows - tok0);
    const uint ncol = (live + 7) / 8;          // live 8-token column fragments
    const uint wrow = tid / 4, wkb = (tid % 4) * 8;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 4);
    const uint sgrow0 = (uint)sg * 8;          // this simdgroup's row stripe
    // Flush mapping: lane owns elements (r = lane/8, t = lane%8) and
    // (r + 4, t) of each 8x8 tile; racc[col*2 + {0,1}] accumulate them.
    const uint rA = row0 + sgrow0 + lane / 8, rB = rA + 4;
    const ulong wsrowA = (ulong)min(rA, rlast) * (args.cols / 128);
    const ulong wsrowB = (ulong)min(rB, rlast) * (args.cols / 128);
    float racc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    threadgroup float *sc = Sc + (uint)sg * 64;
    for (uint c0 = 0; c0 < args.cols; c0 += 32) {
        {
            // 8 trits (one ushort) per thread -> two byte-LUT half4 stores;
            // 256 threads cover the 64x32 tile.
            const ushort wp = *(device const ushort *)(wsrc + (c0 + wkb) / 4);
            threadgroup half4 *dst = (threadgroup half4 *)(Wt + wrow * 32 + wkb);
            dst[0] = q27_t2_half4_lut[wp & 0xffu];
            dst[1] = q27_t2_half4_lut[wp >> 8];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint k8 = 0; k8 < 32; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + sgrow0 * 32 + k8, 32);
            device const half *bsrc = xt + (ulong)(c0 + k8) * tokens_pad + tok0;
            simdgroup_load(b, bsrc, tokens_pad);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            if (ncol > 1) {
                simdgroup_load(b, bsrc + 8, tokens_pad);
                simdgroup_multiply_accumulate(acc1, a, b, acc1);
            }
            if (ncol > 2) {
                simdgroup_load(b, bsrc + 16, tokens_pad);
                simdgroup_multiply_accumulate(acc2, a, b, acc2);
            }
            if (ncol > 3) {
                simdgroup_load(b, bsrc + 24, tokens_pad);
                simdgroup_multiply_accumulate(acc3, a, b, acc3);
            }
        }
        // Per-K-tile flush: this K-tile is one x-scale group (c0/32) and
        // sits inside one weight-scale group (c0/128).
        const float wsA = float(weight_scales[wsrowA + c0 / 128]);
        const float wsB = float(weight_scales[wsrowB + c0 / 128]);
        for (uint col = 0; col < ncol; col++) {
            switch (col) {
                case 0: simdgroup_store(acc0, sc, 8); break;
                case 1: simdgroup_store(acc1, sc, 8); break;
                case 2: simdgroup_store(acc2, sc, 8); break;
                default: simdgroup_store(acc3, sc, 8); break;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint tok = min(tok0 + col * 8 + lane % 8, args.x_rows - 1);
            const float xs = x_scales[(ulong)tok * (args.cols / 32) + c0 / 32];
            racc[col * 2]     += sc[lane]      * wsA * xs;
            racc[col * 2 + 1] += sc[lane + 32] * wsB * xs;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint col = 0; col < ncol; col++) {
        const uint tok = tok0 + col * 8 + lane % 8;
        if (tok >= args.x_rows) continue;
        if (rA < args.rows) out[(ulong)tok * args.rows + rA] = racc[col * 2];
        if (rB < args.rows) out[(ulong)tok * args.rows + rB] = racc[col * 2 + 1];
    }
}

// Lever-1 variant D2: same direct-RHS contract, remapped to cut redundant
// device B loads. 128 threads = 4 simdgroups; the 64-row x 16-token tile
// splits into (row-half, token-column) quadrants, so each B fragment is
// loaded by 2 simdgroups (vs 8 in the row-stripe mapping) and every
// simdgroup has live work at w = 12. Same K-tile-32 flush and exactness
// argument as q27_matmul_t2_mm_dr.
kernel void q27_matmul_t2_mm_dr2(
        device const uchar *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const half *xt [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Wt[64 * 32];
    threadgroup float Sc[4 * 64];
    const uint row0 = group.x * 64;
    const uint tok0 = group.y * 16;
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint tokens_pad = args.simdgroups;
    const uint live = min(16u, args.x_rows - tok0);
    const uint sgrow0 = ((uint)sg / 2) * 32;      // row-half within the tile
    const uint sgtok = ((uint)sg % 2) * 8;        // token column (8 wide)
    const bool tok_live = sgtok < live;
    const uint wrow = tid / 2, wkb = (tid % 2) * 16;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 4);
    const uint rA = row0 + sgrow0 + lane / 8;     // + 8*stripe below
    float racc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    threadgroup float *sc = Sc + (uint)sg * 64;
    for (uint c0 = 0; c0 < args.cols; c0 += 32) {
        {
            // One uint (16 trits) per thread -> 4 byte-LUT half4 stores;
            // 128 threads cover the 64x32 tile.
            const uint wp = *(device const uint *)(wsrc + (c0 + wkb) / 4);
            threadgroup half4 *dst = (threadgroup half4 *)(Wt + wrow * 32 + wkb);
            dst[0] = q27_t2_half4_lut[wp         & 0xffu];
            dst[1] = q27_t2_half4_lut[(wp >>  8) & 0xffu];
            dst[2] = q27_t2_half4_lut[(wp >> 16) & 0xffu];
            dst[3] = q27_t2_half4_lut[wp >> 24         ];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tok_live) {
            simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint k8 = 0; k8 < 32; k8 += 8) {
                simdgroup_half8x8 a, b;
                simdgroup_load(b, xt + (ulong)(c0 + k8) * tokens_pad + tok0 + sgtok, tokens_pad);
                simdgroup_load(a, Wt + sgrow0 * 32 + k8, 32);
                simdgroup_multiply_accumulate(acc0, a, b, acc0);
                simdgroup_load(a, Wt + (sgrow0 + 8) * 32 + k8, 32);
                simdgroup_multiply_accumulate(acc1, a, b, acc1);
                simdgroup_load(a, Wt + (sgrow0 + 16) * 32 + k8, 32);
                simdgroup_multiply_accumulate(acc2, a, b, acc2);
                simdgroup_load(a, Wt + (sgrow0 + 24) * 32 + k8, 32);
                simdgroup_multiply_accumulate(acc3, a, b, acc3);
            }
            const uint tok = min(tok0 + sgtok + lane % 8, args.x_rows - 1);
            const float xs = x_scales[(ulong)tok * (args.cols / 32) + c0 / 32];
            for (uint stripe = 0; stripe < 4; stripe++) {
                switch (stripe) {
                    case 0: simdgroup_store(acc0, sc, 8); break;
                    case 1: simdgroup_store(acc1, sc, 8); break;
                    case 2: simdgroup_store(acc2, sc, 8); break;
                    default: simdgroup_store(acc3, sc, 8); break;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                const uint r0 = rA + stripe * 8, r1 = r0 + 4;
                const float wsA = float(weight_scales[(ulong)min(r0, rlast) * (args.cols / 128) + c0 / 128]);
                const float wsB = float(weight_scales[(ulong)min(r1, rlast) * (args.cols / 128) + c0 / 128]);
                racc[stripe * 2]     += sc[lane]      * wsA * xs;
                racc[stripe * 2 + 1] += sc[lane + 32] * wsB * xs;
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint tok = tok0 + sgtok + lane % 8;
    if (tok >= args.x_rows) return;
    for (uint stripe = 0; stripe < 4; stripe++) {
        const uint r0 = rA + stripe * 8, r1 = r0 + 4;
        if (r0 < args.rows) out[(ulong)tok * args.rows + r0] = racc[stripe * 2];
        if (r1 < args.rows) out[(ulong)tok * args.rows + r1] = racc[stripe * 2 + 1];
    }
}

// Arm A — MMA-core roofline: tiles filled once from an opaque device seed
// (defeats constant folding), then the SAME per-64-K MMA loop count with
// accumulators carried across the whole K walk (the acc dependency chain
// defeats dead-code elimination), one final fold + store with the same
// row/token edge guards. No per-tile staging, no barriers, no flushes:
// A - B isolates staging/barrier/flush cadence at equal MMA count.
kernel void q27_mma_roofline_a(
        device const half *seed [[buffer(0)]],
        device float *out [[buffer(1)]], constant MatmulArgs &args [[buffer(2)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Wt[32 * 64];
    threadgroup half Xt[64 * 16];
    threadgroup float Sc[4 * 64];
    const uint row0 = group.x * 32;
    const uint tok0 = group.y * 16;
    if (row0 >= args.rows) return;
    for (uint i = tid; i < 32 * 64; i += 128) Wt[i] = seed[i];
    for (uint i = tid; i < 64 * 16; i += 128) Xt[i] = seed[32 * 64 + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);   // one-time fill
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const uint tokA = tok0 + lane % 8, tokB = tok0 + 8 + lane % 8;
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    threadgroup float *sc = Sc + sg * 64;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        for (uint k8 = 0; k8 < 32; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        for (uint k8 = 32; k8 < 64; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
    }
    // Single fold: acc0's tile through per-simdgroup scratch, plus a direct
    // dependency on acc1..acc3 via their stored diagonals so no accumulator
    // chain is dead.
    simdgroup_store(acc0, sc, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float r = sc[lane];
    simdgroup_store(acc1, sc, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    r += sc[lane + 32];
    simdgroup_store(acc2, sc, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    r += sc[lane];
    simdgroup_store(acc3, sc, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    r += sc[lane + 32];
    if (rowA < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowA] = r;
    if (rowB < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowB] = r + 1.0f;
}

// Constrained tool decoding: clear grammar-illegal logits to -inf under a
// uint32 bitset mask (bit set = legal), so GPU argmax, top-k extraction,
// and the CPU sampling fallback all see only legal tokens.
kernel void q27_mask_logits(device float *logits [[buffer(0)]],
                            device const uint *mask [[buffer(1)]],
                            constant uint &n [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    if (!((mask[gid >> 5] >> (gid & 31)) & 1u)) logits[gid] = -INFINITY;
}

// ---- Chunked layer-major prefill (2..96 tokens per dispatch; per-path
// caps: MTP rounds and NLL/KL teacher forcing at CHUNK_MAX=12, prompt
// ingestion at PREFILL_CHUNK_MAX=96, suffix-burst/oracle verify chunks at
// VERIFY_CHUNK_MAX=48 — wide-chunk phase A + lever 2) ----
//
// These kernels advance a whole token chunk through one operation so the
// engine can execute prompts layer-major and route projections through the
// simdgroup GEMM. Recurrent operators (convolution ring, DeltaNet state)
// stay sequential across the chunk inside a single dispatch and commit
// their state once per chunk.

struct EmbedRowsArgs { uint cols; uint count; uint tokens[96]; };
kernel void q27_embedding_q8_rows(
        device const char *weights [[buffer(0)]],
        device const half *scales  [[buffer(1)]],
        device float *out          [[buffer(2)]],
        constant EmbedRowsArgs &args [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.cols || gid.y >= args.count) return;
    const uint token = args.tokens[gid.y];
    const ulong wi = (ulong)token * args.cols + gid.x;
    const ulong si = (ulong)token * (args.cols / 128) + gid.x / 128;
    out[(ulong)gid.y * args.cols + gid.x] = float(weights[wi]) * float(scales[si]);
}

kernel void q27_embedding_t2_rows(
        device const uchar *weights [[buffer(0)]],
        device const half *scales   [[buffer(1)]],
        device float *out           [[buffer(2)]],
        constant EmbedRowsArgs &args [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.cols || gid.y >= args.count) return;
    const uint token = args.tokens[gid.y];
    const ulong wi = (ulong)token * args.cols + gid.x;
    const uint code = (weights[wi >> 2] >> ((wi & 3) * 2)) & 3;
    const ulong si = (ulong)token * (args.cols / 128) + gid.x / 128;
    out[(ulong)gid.y * args.cols + gid.x] = float(int(code) - 1) * float(scales[si]);
}

kernel void q27_embedding_b1_rows(
        device const uchar *weights [[buffer(0)]],
        device const half *scales   [[buffer(1)]],
        device float *out           [[buffer(2)]],
        constant EmbedRowsArgs &args [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.cols || gid.y >= args.count) return;
    const uint token = args.tokens[gid.y];
    const ulong wi = (ulong)token * args.cols + gid.x;
    const uint bit = (weights[wi >> 3] >> (wi & 7)) & 1;
    const ulong si = (ulong)token * (args.cols / 128) + gid.x / 128;
    out[(ulong)gid.y * args.cols + gid.x] = float(2 * int(bit) - 1) * float(scales[si]);
}

struct RowsNormArgs { uint n; uint rows; uint groups; float eps; };
kernel void q27_rmsnorm_rows_quantized(
        device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
        device float *out [[buffer(2)]], device char *values [[buffer(3)]],
        device float *scales [[buffer(4)]], constant RowsNormArgs &args [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (row >= args.rows) return;
    device const float *xr = x + (ulong)row * args.n;
    device float *or_ = out + (ulong)row * args.n;
    device char *vr = values + (ulong)row * args.n;
    device float *sr = scales + (ulong)row * (args.n / 32);
    float sum = 0.0f;
    for (uint i = tid; i < args.n; i += 256) sum += xr[i] * xr[i];
    threadgroup float partial[32];
    sum = reduce_sum(sum, partial, lane, simdgroup, args.groups);
    const float inv = rsqrt(sum / float(args.n) + args.eps);
    for (uint i = tid; i < args.n; i += 256) or_[i] = xr[i] * inv * w[i];
    threadgroup_barrier(mem_flags::mem_device);
    const uint blocks = args.n / 32;
    for (uint block = simdgroup; block < blocks; block += 8) {
        const uint i = block * 32 + lane; const float v = or_[i];
        const float amax = simd_max(abs(v)); const float scale = amax / 127.0f;
        // k3 audit D1: reciprocal-multiply matches CUDA k_quantize_x — one
        // rounding rule across backends (rint == __float2int_rn under RNE).
        const float qinv = scale > 0.0f ? 1.0f / scale : 0.0f;
        int q = int(rint(v * qinv)); q = clamp(q, -127, 127);
        vr[i] = char(q); if (lane == 0) sr[block] = scale;
    }
}

// Mirrors q27_matvec_f16_pair's per-row structure exactly (threadgroup per
// row, packed_half4 dots, 8-partial tree) so chunked and serial results stay
// bit-identical; the grid adds a token dimension.
struct MatvecPairRowsArgs { uint rows_a; uint rows_b; uint cols; uint tokens; };
kernel void q27_matvec_f16_pair_rows(
        device const half *weights_a [[buffer(0)]], device float *out_a [[buffer(1)]],
        device const half *weights_b [[buffer(2)]], device float *out_b [[buffer(3)]],
        device const float *x [[buffer(4)]], constant MatvecPairRowsArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]], ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.x, token = group.y;
    if (token >= args.tokens) return;
    device const float *xt = x + (ulong)token * args.cols;
    const bool row_a = row < args.rows_a, row_b = row < args.rows_b;
    const ulong base = (ulong)row * args.cols;
    device const packed_half4 *wa4 = (device const packed_half4 *)(weights_a + base);
    device const packed_half4 *wb4 = (device const packed_half4 *)(weights_b + base);
    const uint vectors = args.cols / 4;
    float sa = 0.0f, sb = 0.0f;
    for (uint v = tid; v < vectors; v += 256) {
        const float4 xv = float4(xt[v * 4], xt[v * 4 + 1], xt[v * 4 + 2], xt[v * 4 + 3]);
        if (row_a) sa += dot(float4(wa4[v]), xv);
        if (row_b) sb += dot(float4(wb4[v]), xv);
    }
    for (uint col = vectors * 4 + tid; col < args.cols; col += 256) {
        const float xv = xt[col];
        if (row_a) sa += float(weights_a[base + col]) * xv;
        if (row_b) sb += float(weights_b[base + col]) * xv;
    }
    threadgroup float partial[8];
    sa = simd_sum(sa);
    if (lane == 0) partial[simdgroup] = sa;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        float total = lane < 8 ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0 && row_a) out_a[(ulong)token * args.rows_a + row] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sb = simd_sum(sb);
    if (lane == 0) partial[simdgroup] = sb;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        float total = lane < 8 ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0 && row_b) out_b[(ulong)token * args.rows_b + row] = total;
    }
}

struct GatesRowsArgs { uint heads; uint tokens; };
kernel void q27_gdn_gates_rows(device const float *alpha [[buffer(0)]],
                                device const float *beta_raw [[buffer(1)]],
                                device const float *ssm_a [[buffer(2)]],
                                device const float *ssm_dt [[buffer(3)]],
                                device float *g [[buffer(4)]],
                                device float *beta [[buffer(5)]],
                                constant GatesRowsArgs &args [[buffer(6)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= args.heads * args.tokens) return;
    const uint head = gid % args.heads;
    const float value = alpha[gid] + ssm_dt[head];
    const float softplus = value > 20.0f ? value
                         : (value < -16.0f ? exp(value) : log1p_f(exp(value)));
    g[gid] = ssm_a[head] * softplus;
    beta[gid] = 1.0f / (1.0f + exp(-beta_raw[gid]));
}

// Separate src/dst ring bindings mirror q27_conv_step: MTP verification
// points dst at a discard slot so the optimistic chunk never commits state,
// and the acceptance replay commits it from parked inputs (src == dst).
struct ConvChunkArgs { uint channels; uint tokens; };
kernel void q27_conv_chunk(device const float *ring_src [[buffer(0)]],
                            device float *ring_dst [[buffer(1)]],
                            device const float *qkv [[buffer(2)]],
                            device const float *weight [[buffer(3)]],
                            device float *out [[buffer(4)]],
                            constant ConvChunkArgs &args [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= args.channels) return;
    float r0 = ring_src[gid], r1 = ring_src[args.channels + gid], r2 = ring_src[(ulong)2 * args.channels + gid];
    const float w0 = weight[(ulong)gid * 4], w1 = weight[(ulong)gid * 4 + 1];
    const float w2 = weight[(ulong)gid * 4 + 2], w3 = weight[(ulong)gid * 4 + 3];
    for (uint t = 0; t < args.tokens; t++) {
        const float xv = qkv[(ulong)t * args.channels + gid];
        const float value = r0 * w0 + r1 * w1 + r2 * w2 + xv * w3;
        out[(ulong)t * args.channels + gid] = value / (1.0f + exp(-value));
        r0 = r1; r1 = r2; r2 = xv;
    }
    ring_dst[gid] = r0; ring_dst[args.channels + gid] = r1; ring_dst[(ulong)2 * args.channels + gid] = r2;
}

struct DeltaChunkArgs { uint value_heads; uint qk_heads; uint head_dim; uint tokens; };
kernel void q27_delta_chunk(device const float *state_src [[buffer(0)]],
                             device float *state_dst [[buffer(1)]],
                             device const float *conv [[buffer(2)]],
                             device const float *g [[buffer(3)]],
                             device const float *beta [[buffer(4)]],
                             device float *out [[buffer(5)]],
                             constant DeltaChunkArgs &args [[buffer(6)]],
                             uint head [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    if (head >= args.value_heads || args.head_dim != 128 || args.qk_heads != 16) return;
    const uint j = tid & 127;
    const uint tile = tid >> 7;
    const uint i0 = tile * 32;
    const uint qk = head % args.qk_heads;
    const ulong conv_row = (ulong)(2 * args.qk_heads + args.value_heads) * 128;
    const ulong out_row = (ulong)args.value_heads * 128;
    threadgroup float q[128], k[128], part[4][128], delta[128];
    device const float *sh = state_src + (ulong)head * 128 * 128;
    device float *sd = state_dst + (ulong)head * 128 * 128;
    // The chunk's whole state slice lives in registers; it is written back
    // exactly once, at the chunk boundary, to state_dst (a discard slot for
    // MTP verification, the live state for prefill and acceptance replay).
    float saved[32];
    for (uint n = 0; n < 32; n++) saved[n] = sh[(ulong)(i0 + n) * 128 + j];
    for (uint t = 0; t < args.tokens; t++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const float *cv = conv + (ulong)t * conv_row;
        if (tile == 0) { q[j] = cv[(ulong)qk * 128 + j] * rsqrt(128.0f); k[j] = cv[2048 + (ulong)qk * 128 + j]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float decay = exp(g[(ulong)t * args.value_heads + head]);
        float prediction = 0.0f;
        for (uint n = 0; n < 32; n++) {
            saved[n] *= decay;
            prediction += k[i0 + n] * saved[n];
        }
        part[tile][j] = prediction;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tile == 0) {
            const float predicted = part[0][j] + part[1][j] + part[2][j] + part[3][j];
            delta[j] = beta[(ulong)t * args.value_heads + head] * (cv[4096 + (ulong)head * 128 + j] - predicted);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float result = 0.0f;
        for (uint n = 0; n < 32; n++) {
            saved[n] += k[i0 + n] * delta[j];
            result += q[i0 + n] * saved[n];
        }
        part[tile][j] = result;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tile == 0) out[(ulong)t * out_row + (ulong)head * 128 + j] =
            part[0][j] + part[1][j] + part[2][j] + part[3][j];
    }
    for (uint n = 0; n < 32; n++) sd[(ulong)(i0 + n) * 128 + j] = saved[n];
}

struct L2RowsArgs { uint heads; uint head_dim; uint row_stride; uint tokens; float eps; };
kernel void q27_l2norm_rows(device float *x [[buffer(0)]],
                             constant L2RowsArgs &args [[buffer(1)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]],
                             ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (group.x >= args.heads || group.y >= args.tokens) return;
    device float *xh = x + (ulong)group.y * args.row_stride + (ulong)group.x * args.head_dim;
    float sum = 0.0f;
    for (uint i = tid; i < args.head_dim; i += 256) sum += xh[i] * xh[i];
    threadgroup float partial[32];
    sum = reduce_sum(sum, partial, lane, simdgroup, 8);
    const float inv = rsqrt(max(sum, args.eps * args.eps));
    for (uint i = tid; i < args.head_dim; i += 256) xh[i] *= inv;
}

struct RopeRowsArgs {
    uint heads; uint head_dim; uint n_rot; uint stride;
    uint row_stride; uint position; uint tokens; float freq_base;
};
kernel void q27_rope_neox_rows(device float *x [[buffer(0)]],
                                constant RopeRowsArgs &args [[buffer(1)]],
                                uint3 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, head = gid.y, token = gid.z;
    if (head >= args.heads || d >= args.n_rot / 2 || token >= args.tokens) return;
    device float *xh = x + (ulong)token * args.row_stride + (ulong)head * args.stride;
    const float theta = float(args.position + token) *
                        pow(args.freq_base, -2.0f * float(d) / float(args.n_rot));
    const float cs = cos(theta), sn = sin(theta);
    const float x0 = xh[d], x1 = xh[d + args.n_rot / 2];
    xh[d] = x0 * cs - x1 * sn;
    xh[d + args.n_rot / 2] = x0 * sn + x1 * cs;
}

struct KvStoreRowsArgs { uint position; uint row_length; uint tokens; };
kernel void q27_kv_store_f16_rows(device const float *k [[buffer(0)]],
                                   device const float *v [[buffer(1)]],
                                   device half *kc       [[buffer(2)]],
                                   device half *vc       [[buffer(3)]],
                                   constant KvStoreRowsArgs &args [[buffer(4)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.row_length || gid.y >= args.tokens) return;
    const ulong src = (ulong)gid.y * args.row_length + gid.x;
    const ulong dst = (ulong)(args.position + gid.y) * args.row_length + gid.x;
    kc[dst] = half(k[src]); vc[dst] = half(v[src]);
}

struct GateRowsArgs { uint heads; uint head_dim; uint tokens; };
kernel void q27_sigmoid_gate_mul_rows(device float *out [[buffer(0)]],
                                       device const float *qg [[buffer(1)]],
                                       constant GateRowsArgs &args [[buffer(2)]],
                                       uint gid [[thread_position_in_grid]]) {
    const uint row = args.heads * args.head_dim;
    if (gid >= row * args.tokens) return;
    const uint t = gid / row;
    const uint h = (gid % row) / args.head_dim;
    const uint d = gid % args.head_dim;
    const float gate = qg[(ulong)t * row * 2 + (ulong)h * (2 * args.head_dim) + args.head_dim + d];
    out[gid] *= 1.0f / (1.0f + exp(-gate));
}

struct ArgmaxRowsArgs { uint n; uint rows; };
kernel void q27_argmax_rows(device const float *x [[buffer(0)]],
                             device uint *out       [[buffer(1)]],
                             constant ArgmaxRowsArgs &args [[buffer(2)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    if (row >= args.rows) return;
    device const float *xr = x + (ulong)row * args.n;
    float best = -INFINITY;
    uint best_i = 0;
    for (uint i = tid; i < args.n; i += 256) {
        const float value = xr[i];
        if (value > best || (value == best && i < best_i)) { best = value; best_i = i; }
    }
    threadgroup float values[256];
    threadgroup uint indices[256];
    values[tid] = best; indices[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) {
        if (tid < step) {
            const float other = values[tid + step];
            const uint other_i = indices[tid + step];
            if (other > values[tid] || (other == values[tid] && other_i < indices[tid])) {
                values[tid] = other; indices[tid] = other_i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[row] = indices[0];
}

// nll[r] = logsumexp(logits[r,:]) - logits[r, tgt[r]]. One threadgroup per
// row, 256 threads: max pass then sum-exp, matching the CUDA quality-gate
// protocol so Metal --nll-long is comparable to the CUDA buckets.
struct NllRowsArgs { uint n; uint rows; };
kernel void q27_nll_rows(device const float *logits [[buffer(0)]],
                          device const uint *tgt     [[buffer(1)]],
                          device float *nll          [[buffer(2)]],
                          constant NllRowsArgs &args [[buffer(3)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    if (row >= args.rows) return;
    device const float *xr = logits + (ulong)row * args.n;
    float mx = -INFINITY;
    for (uint i = tid; i < args.n; i += 256) mx = max(mx, xr[i]);
    threadgroup float values[256];
    values[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) {
        if (tid < step) values[tid] = max(values[tid], values[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mx = values[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float se = 0.0f;
    for (uint i = tid; i < args.n; i += 256) se += exp(xr[i] - mx);
    values[tid] = se;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) {
        if (tid < step) values[tid] += values[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        const uint target = tgt[row];
        nll[row] = log(values[0]) + mx - xr[target];
    }
}

struct AttentionCausalArgs {
    uint q_stride; uint q_row_stride; uint base_len;
    uint q_heads; uint kv_heads; uint head_dim; uint tokens; float scale;
};
// Chunk-causal FP16 attention, online-softmax. Mirrors the decode kernel
// (q27_attention_f16) exactly — one threadgroup per (query head, chunk
// token), eight simdgroups striping the token's visible sequence
// (base_len + token positions) with running max/denominator/weighted-value
// in registers and a log2 merge through threadgroup memory — so a chunk
// token's output is bit-identical to the serial decode kernel at the same
// sequence length, and no probability scratch is materialized (the old
// kernel serialized scores on thread 0 through a tokens x heads x context
// device buffer).
kernel void q27_attention_f16_causal(device const float *q [[buffer(0)]],
                                      device const half *kc [[buffer(1)]],
                                      device const half *vc [[buffer(2)]],
                                      device float *out      [[buffer(3)]],
                                      constant AttentionCausalArgs &args [[buffer(4)]],
                                      uint2 group [[threadgroup_position_in_grid]],
                                      ushort lane [[thread_index_in_simdgroup]],
                                      ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint qh = group.x, token = group.y;
    if (qh >= args.q_heads || token >= args.tokens) return;
    const uint seq_len = args.base_len + token;
    const uint gqa = args.q_heads / args.kv_heads;
    const uint kvh = qh / gqa;
    device const float *qh_ptr = q + (ulong)token * args.q_row_stride + (ulong)qh * args.q_stride;

    float acc[8];                       // head_dim <= 256 -> at most 8 dims per lane
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint p = sg; p < seq_len; p += 8) {
        device const half *kh = kc + ((ulong)p * args.kv_heads + kvh) * args.head_dim;
        float partial = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32) partial += qh_ptr[d] * float(kh[d]);
        const float score = simd_sum(partial) * args.scale;
        const float m_new = max(m, score);
        const float correction = exp(m - m_new);    // first iteration: exp(-inf) = 0
        const float weight = exp(score - m_new);
        l = l * correction + weight;
        device const half *vh = vc + ((ulong)p * args.kv_heads + kvh) * args.head_dim;
        for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
            acc[i] = acc[i] * correction + weight * float(vh[d]);
        m = m_new;
    }

    // Merge the eight simdgroup partials: rounds of 4, 2, 1. A simdgroup that
    // saw no positions carries m = -inf, l = 0 and merges as a no-op.
    threadgroup float tg_m[4], tg_l[4], tg_acc[4][256];
    for (uint offset = 4; offset >= 1; offset /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg >= offset && sg < 2 * offset) {
            if (lane == 0) { tg_m[sg - offset] = m; tg_l[sg - offset] = l; }
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
                tg_acc[sg - offset][d] = acc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < offset) {
            const float m_other = tg_m[sg], l_other = tg_l[sg];
            const float m_new = max(m, m_other);
            if (m_new == -INFINITY) continue;       // both stripes empty
            const float c_mine = exp(m - m_new), c_other = exp(m_other - m_new);
            l = l * c_mine + l_other * c_other;
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
                acc[i] = acc[i] * c_mine + tg_acc[sg][d] * c_other;
            m = m_new;
        }
    }
    if (sg == 0) {
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
            out[((ulong)token * args.q_heads + qh) * args.head_dim + d] = acc[i] * inv;
    }
}

kernel void q27_copy_bytes(device const uchar *src [[buffer(0)]],
                            device uchar *dst [[buffer(1)]],
                            constant ulong &bytes [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid < bytes) dst[gid] = src[gid];
}

constant float turbo_centroids[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f };
constant char turbo_s1[128] = { -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1 };
constant char turbo_s2[128] = { 1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1 };
constant float turbo_inv_sqrt_128 = 0.08838834764831845f;

inline uint turbo_nearest(float v) {
    if (v < -0.154496f) return 0; if (v < -0.092804f) return 1;
    if (v < -0.044243f) return 2; if (v < 0.0f) return 3;
    if (v < 0.044243f) return 4; if (v < 0.092804f) return 5;
    if (v < 0.154496f) return 6; return 7;
}

inline void turbo_butterfly(threadgroup float *xs, uint j) {
    for (uint h = 1; h < 128; h <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float a = xs[j], b = xs[j ^ h];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        xs[j] = (j & h) ? (b - a) : (a + b);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

struct TurboWhtArgs { uint heads; uint stride; uint inverse; };
kernel void q27_turbo_wht(device float *x [[buffer(0)]],
                           constant TurboWhtArgs &args [[buffer(1)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint j [[thread_index_in_threadgroup]]) {
    const uint head = group >> 1, g = group & 1;
    // Guards here must be threadgroup-uniform (they derive from
    // threadgroup_position only): a thread-index guard before the butterfly
    // barriers is divergent and would under-populate them if a dispatch ever
    // exceeded 128 threads. The host dispatches exactly 128; width guards
    // land host-side (k3 audit A2/E6).
    if (head >= args.heads) return;
    device float *xh = x + (ulong)head * args.stride + g * 128;
    threadgroup float xs[128];
    xs[j] = xh[j] * float(args.inverse ? turbo_s2[j] : turbo_s1[j]);
    turbo_butterfly(xs, j);
    xh[j] = xs[j] * turbo_inv_sqrt_128 * float(args.inverse ? turbo_s1[j] : turbo_s2[j]);
}

struct TurboStoreArgs { uint position; uint kv_heads; };
kernel void q27_kv_store_turbo3(device const float *k [[buffer(0)]],
                                 device const float *v [[buffer(1)]],
                                 device uchar *kc [[buffer(2)]],
                                 device uchar *vc [[buffer(3)]],
                                 constant TurboStoreArgs &args [[buffer(4)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint j [[thread_index_in_threadgroup]]) {
    const uint h = group.x >> 1, g = group.x & 1;
    if (h >= args.kv_heads || group.y >= 2) return;
    device const float *src = (group.y ? v : k) + (ulong)h * 256 + g * 128;
    device uchar *cache = group.y ? vc : kc;
    device uchar *block = cache + ((ulong)args.position * args.kv_heads * 2 + h * 2 + g) * 50;
    threadgroup float xs[128], red[128];
    threadgroup uchar indices[128];
    xs[j] = src[j]; red[j] = src[j] * src[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 64; s; s >>= 1) {
        if (j < s) red[j] += red[j + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float norm = sqrt(red[0]);
    xs[j] = xs[j] * (norm > 1e-10f ? 1.0f / norm : 0.0f) * float(turbo_s1[j]);
    turbo_butterfly(xs, j);
    const uint index = turbo_nearest(xs[j] * turbo_inv_sqrt_128 * float(turbo_s2[j]));
    indices[j] = uchar(index); red[j] = turbo_centroids[index] * turbo_centroids[index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 64; s; s >>= 1) {
        if (j < s) red[j] += red[j + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (j == 0) *(device half *)(block) = half(sqrt(red[0]) > 1e-10f ? norm / sqrt(red[0]) : norm);
    if ((j & 3) == 0) block[2 + j / 4] = (indices[j] & 3) | ((indices[j+1] & 3) << 2) |
                                                   ((indices[j+2] & 3) << 4) | ((indices[j+3] & 3) << 6);
    if ((j & 7) == 0) {
        uchar bits = 0;
        for (uint i = 0; i < 8; i++) bits |= uchar((indices[j+i] >> 2) << i);
        block[34 + j / 8] = bits;
    }
}

inline float turbo_dequant(device const uchar *block, uint j) {
    const uint low = (block[2 + (j >> 2)] >> ((j & 3) * 2)) & 3;
    const uint high = (block[34 + (j >> 3)] >> (j & 7)) & 1;
    return turbo_centroids[low | (high << 2)] * float(*(device const half *)block);
}

// Online-softmax turbo3 decode attention. Mirrors q27_attention_f16 exactly
// — one threadgroup per query head, eight simdgroups striping the sequence
// with running max/denominator/weighted-value in registers and a log2 merge
// through threadgroup memory — with the 50-byte turbo3 blocks dequantized
// on the fly. No probability scratch is materialized (the old kernel
// serialized every score on thread 0 through a heads x context buffer).
kernel void q27_attention_turbo3(device const float *q [[buffer(0)]],
                                  device const uchar *kc [[buffer(1)]],
                                  device const uchar *vc [[buffer(2)]],
                                  device float *out [[buffer(3)]],
                                  constant AttentionArgs &args [[buffer(4)]],
                                  uint qh [[threadgroup_position_in_grid]],
                                  ushort lane [[thread_index_in_simdgroup]],
                                  ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (qh >= args.q_heads) return;
    const uint gqa = args.q_heads / args.kv_heads;
    const uint kvh = qh / gqa;
    device const float *qh_ptr = q + (ulong)qh * args.q_stride;

    float acc[8];                       // head_dim <= 256 -> at most 8 dims per lane
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint p = sg; p < args.seq_len; p += 8) {
        device const uchar *kb = kc + ((ulong)p * args.kv_heads + kvh) * 2 * 50;
        float partial = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32)
            partial += qh_ptr[d] * turbo_dequant(kb + (d >> 7) * 50, d & 127);
        const float score = simd_sum(partial) * args.scale;
        const float m_new = max(m, score);
        const float correction = exp(m - m_new);    // first iteration: exp(-inf) = 0
        const float weight = exp(score - m_new);
        l = l * correction + weight;
        device const uchar *vb = vc + ((ulong)p * args.kv_heads + kvh) * 2 * 50;
        for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
            acc[i] = acc[i] * correction + weight * turbo_dequant(vb + (d >> 7) * 50, d & 127);
        m = m_new;
    }

    // Merge the eight simdgroup partials: rounds of 4, 2, 1. A simdgroup that
    // saw no positions carries m = -inf, l = 0 and merges as a no-op.
    threadgroup float tg_m[4], tg_l[4], tg_acc[4][256];
    for (uint offset = 4; offset >= 1; offset /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg >= offset && sg < 2 * offset) {
            if (lane == 0) { tg_m[sg - offset] = m; tg_l[sg - offset] = l; }
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
                tg_acc[sg - offset][d] = acc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < offset) {
            const float m_other = tg_m[sg], l_other = tg_l[sg];
            const float m_new = max(m, m_other);
            if (m_new == -INFINITY) continue;       // both stripes empty
            const float c_mine = exp(m - m_new), c_other = exp(m_other - m_new);
            l = l * c_mine + l_other * c_other;
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
                acc[i] = acc[i] * c_mine + tg_acc[sg][d] * c_other;
            m = m_new;
        }
    }
    if (sg == 0) {
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
            out[(ulong)qh * args.head_dim + d] = acc[i] * inv;
    }
}

struct TurboStoreRowsArgs { uint position; uint kv_heads; uint tokens; };
kernel void q27_kv_store_turbo3_rows(device const float *k [[buffer(0)]],
                                      device const float *v [[buffer(1)]],
                                      device uchar *kc [[buffer(2)]],
                                      device uchar *vc [[buffer(3)]],
                                      constant TurboStoreRowsArgs &args [[buffer(4)]],
                                      uint3 group [[threadgroup_position_in_grid]],
                                      uint j [[thread_index_in_threadgroup]]) {
    const uint h = group.x >> 1, g = group.x & 1, token = group.z;
    if (h >= args.kv_heads || group.y >= 2 || token >= args.tokens) return;
    device const float *src = (group.y ? v : k) +
        (ulong)token * args.kv_heads * 256 + (ulong)h * 256 + g * 128;
    device uchar *cache = group.y ? vc : kc;
    device uchar *block = cache +
        ((ulong)(args.position + token) * args.kv_heads * 2 + h * 2 + g) * 50;
    threadgroup float xs[128], red[128];
    threadgroup uchar indices[128];
    xs[j] = src[j]; red[j] = src[j] * src[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 64; s; s >>= 1) {
        if (j < s) red[j] += red[j + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float norm = sqrt(red[0]);
    xs[j] = xs[j] * (norm > 1e-10f ? 1.0f / norm : 0.0f) * float(turbo_s1[j]);
    turbo_butterfly(xs, j);
    const uint index = turbo_nearest(xs[j] * turbo_inv_sqrt_128 * float(turbo_s2[j]));
    indices[j] = uchar(index); red[j] = turbo_centroids[index] * turbo_centroids[index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 64; s; s >>= 1) {
        if (j < s) red[j] += red[j + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (j == 0) *(device half *)(block) = half(sqrt(red[0]) > 1e-10f ? norm / sqrt(red[0]) : norm);
    if ((j & 3) == 0) block[2 + j / 4] = (indices[j] & 3) | ((indices[j+1] & 3) << 2) |
                                                   ((indices[j+2] & 3) << 4) | ((indices[j+3] & 3) << 6);
    if ((j & 7) == 0) {
        uchar bits = 0;
        for (uint i = 0; i < 8; i++) bits |= uchar((indices[j+i] >> 2) << i);
        block[34 + j / 8] = bits;
    }
}

// KV-codec attribution store (kl-kv instrument): writes the fp16 KV cache,
// but routes one side (mode 1 = K, mode 2 = V) through the exact turbo3
// quantizer — normalize, sign flips, butterfly, 8-centroid nearest,
// half-rounded norm-correction scale — and back through the inverse
// transform. The engine stays on the fp16 attention kernels throughout, so
// a KL delta against the fp16 baseline is attributable to that one side's
// quantization alone.
// head selects a single KV head for cell-granular attribution (census);
// ~0u round-trips every head of the selected side. flags: step-2 scaling
// arms (docs/plans/2026-07-16-kv-codec-step2.md) — SCALE32 keeps the
// group scale in f32 through the round-trip, FEATURE descales each
// dimension by aux[scale_off + side/head/dim] before the quantizer and
// rescales after the inverse transform, STATS stores clean fp16 while
// accumulating per-feature sum-of-squares for BOTH sides into aux
// (atomic f32). aux layout: [side(K=0,V=1)][attn_idx][head][256 dims].
constant uint Q27_ATTRIB_SCALE32 = 1;
constant uint Q27_ATTRIB_FEATURE = 2;
constant uint Q27_ATTRIB_STATS   = 4;
struct TurboAttribArgs { uint position; uint kv_heads; uint tokens; uint mode; uint head;
                         uint flags; uint scale_off; };
kernel void q27_kv_store_f16_attrib_rows(device const float *k [[buffer(0)]],
                                          device const float *v [[buffer(1)]],
                                          device half *kc [[buffer(2)]],
                                          device half *vc [[buffer(3)]],
                                          constant TurboAttribArgs &args [[buffer(4)]],
                                          device float *aux [[buffer(5)]],
                                          uint3 group [[threadgroup_position_in_grid]],
                                          uint j [[thread_index_in_threadgroup]]) {
    const uint h = group.x >> 1, g = group.x & 1, token = group.z;
    if (h >= args.kv_heads || group.y >= 2 || token >= args.tokens) return;
    device const float *src = (group.y ? v : k) +
        (ulong)token * args.kv_heads * 256 + (ulong)h * 256 + g * 128;
    device half *dst = (group.y ? vc : kc) +
        (ulong)(args.position + token) * args.kv_heads * 256 + (ulong)h * 256 + g * 128;
    const uint aux_at = args.scale_off + group.y * 16384u + h * 256u + g * 128u + j;
    if (args.flags & Q27_ATTRIB_STATS) {
        // Stats pass: the stored cache stays clean fp16 (the subject IS the
        // baseline; its KL rides along as a zero canary) while per-feature
        // sum-of-squares accumulates for BOTH sides.
        atomic_fetch_add_explicit((device atomic_float *)&aux[aux_at],
                                  src[j] * src[j], memory_order_relaxed);
        dst[j] = half(src[j]);
        return;
    }
    // group.y and h are uniform across the threadgroup, so this early exit
    // and the barriers below never diverge within a threadgroup.
    if (args.mode != (group.y ? 2u : 1u) ||
        (args.head != ~0u && h != args.head)) { dst[j] = half(src[j]); return; }
    const float sj = (args.flags & Q27_ATTRIB_FEATURE) ? aux[aux_at] : 1.0f;
    threadgroup float xs[128], red[128];
    const float x0 = src[j] / sj;
    xs[j] = x0; red[j] = x0 * x0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 64; s; s >>= 1) {
        if (j < s) red[j] += red[j + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float norm = sqrt(red[0]);
    xs[j] = xs[j] * (norm > 1e-10f ? 1.0f / norm : 0.0f) * float(turbo_s1[j]);
    turbo_butterfly(xs, j);
    const uint index = turbo_nearest(xs[j] * turbo_inv_sqrt_128 * float(turbo_s2[j]));
    red[j] = turbo_centroids[index] * turbo_centroids[index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 64; s; s >>= 1) {
        if (j < s) red[j] += red[j + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The group scale passes through half exactly as the packed block header
    // does — unless the SCALE32 arm keeps it f32 to isolate scale precision
    // from scale structure.
    const float cn = sqrt(red[0]);
    const float raw_scale = cn > 1e-10f ? norm / cn : norm;
    const float scale = (args.flags & Q27_ATTRIB_SCALE32) ? raw_scale : float(half(raw_scale));
    xs[j] = turbo_centroids[index] * scale * float(turbo_s2[j]);
    turbo_butterfly(xs, j);
    dst[j] = half(xs[j] * turbo_inv_sqrt_128 * float(turbo_s1[j]) * sj);
}

// Chunk-causal turbo3 attention, online-softmax. Mirrors the turbo3 decode
// kernel exactly — one threadgroup per (query head, chunk token), eight
// simdgroups striping the token's visible sequence — so a chunk token's
// output is bit-identical to the decode kernel at the same sequence length,
// and the last probability-scratch user on the prefill path is gone.
kernel void q27_attention_turbo3_causal(device const float *q [[buffer(0)]],
                                         device const uchar *kc [[buffer(1)]],
                                         device const uchar *vc [[buffer(2)]],
                                         device float *out [[buffer(3)]],
                                         constant AttentionCausalArgs &args [[buffer(4)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         ushort lane [[thread_index_in_simdgroup]],
                                         ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint qh = group.x, token = group.y;
    if (qh >= args.q_heads || token >= args.tokens) return;
    const uint seq_len = args.base_len + token;
    const uint gqa = args.q_heads / args.kv_heads;
    const uint kvh = qh / gqa;
    device const float *qh_ptr = q + (ulong)token * args.q_row_stride + (ulong)qh * args.q_stride;

    float acc[8];                       // head_dim <= 256 -> at most 8 dims per lane
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint p = sg; p < seq_len; p += 8) {
        device const uchar *kb = kc + ((ulong)p * args.kv_heads + kvh) * 2 * 50;
        float partial = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32)
            partial += qh_ptr[d] * turbo_dequant(kb + (d >> 7) * 50, d & 127);
        const float score = simd_sum(partial) * args.scale;
        const float m_new = max(m, score);
        const float correction = exp(m - m_new);    // first iteration: exp(-inf) = 0
        const float weight = exp(score - m_new);
        l = l * correction + weight;
        device const uchar *vb = vc + ((ulong)p * args.kv_heads + kvh) * 2 * 50;
        for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
            acc[i] = acc[i] * correction + weight * turbo_dequant(vb + (d >> 7) * 50, d & 127);
        m = m_new;
    }

    // Merge the eight simdgroup partials: rounds of 4, 2, 1. A simdgroup that
    // saw no positions carries m = -inf, l = 0 and merges as a no-op.
    threadgroup float tg_m[4], tg_l[4], tg_acc[4][256];
    for (uint offset = 4; offset >= 1; offset /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg >= offset && sg < 2 * offset) {
            if (lane == 0) { tg_m[sg - offset] = m; tg_l[sg - offset] = l; }
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
                tg_acc[sg - offset][d] = acc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < offset) {
            const float m_other = tg_m[sg], l_other = tg_l[sg];
            const float m_new = max(m, m_other);
            if (m_new == -INFINITY) continue;       // both stripes empty
            const float c_mine = exp(m - m_new), c_other = exp(m_other - m_new);
            l = l * c_mine + l_other * c_other;
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
                acc[i] = acc[i] * c_mine + tg_acc[sg][d] * c_other;
            m = m_new;
        }
    }
    if (sg == 0) {
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
            out[((ulong)token * args.q_heads + qh) * args.head_dim + d] = acc[i] * inv;
    }
}

// GQA KV-reuse decode attention. The per-query-head kernels above stream
// each KV row q_heads/kv_heads (= 6) times; here one threadgroup serves ALL
// query heads of one KV head over one sequence block: K/V rows are staged
// (turbo3: dequantized once, not per query head) into threadgroup memory,
// and one simdgroup per query head runs its own online softmax over the
// staged tile. Blocks write {m, l, acc[head_dim]} partials (258 floats);
// q27_attention_gqa_merge folds the blocks in index order — deterministic
// run-to-run, but NOT bit-equal to the per-query-head kernels (different
// summation order). The host routes only long sequences here; short
// contexts keep the proven kernels, and the causal chunk kernels are
// untouched.
struct AttentionGqaArgs {
    uint q_stride;
    uint seq_len;
    uint q_heads;
    uint kv_heads;
    uint head_dim;
    uint block;
    uint n_blocks;
    float scale;
};

kernel void q27_attention_f16_gqa(device const float *q [[buffer(0)]],
                                   device const half *kc [[buffer(1)]],
                                   device const half *vc [[buffer(2)]],
                                   device float *partials [[buffer(3)]],
                                   constant AttentionGqaArgs &args [[buffer(4)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   ushort lane [[thread_index_in_simdgroup]],
                                   ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || blk >= args.n_blocks || sg >= gqa) return;
    const uint p0 = blk * args.block;
    const uint p1 = min(p0 + args.block, args.seq_len);
    const uint qh = kvh * gqa + sg;
    device const float *qh_ptr = q + (ulong)qh * args.q_stride;
    const uint tid = (uint)sg * 32 + lane, threads = gqa * 32;

    threadgroup float Kt[8][256], Vt[8][256];
    float acc[8];                       // head_dim <= 256 -> at most 8 dims per lane
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint t0 = p0; t0 < p1; t0 += 8) {
        const uint rows = min(8u, p1 - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * args.head_dim; idx += threads) {
            const uint r = idx / args.head_dim, d = idx % args.head_dim;
            const ulong row = ((ulong)(t0 + r) * args.kv_heads + kvh) * args.head_dim;
            Kt[r][d] = float(kc[row + d]);
            Vt[r][d] = float(vc[row + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < rows; r++) {
            float partial = 0.0f;
            for (uint d = lane; d < args.head_dim; d += 32) partial += qh_ptr[d] * Kt[r][d];
            const float score = simd_sum(partial) * args.scale;
            const float m_new = max(m, score);
            const float correction = exp(m - m_new);    // first iteration: exp(-inf) = 0
            const float weight = exp(score - m_new);
            l = l * correction + weight;
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
                acc[i] = acc[i] * correction + weight * Vt[r][d];
            m = m_new;
        }
    }
    device float *ph = partials + ((ulong)qh * args.n_blocks + blk) * 258;
    if (lane == 0) { ph[0] = m; ph[1] = l; }
    for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++) ph[2 + d] = acc[i];
}

kernel void q27_attention_turbo3_gqa(device const float *q [[buffer(0)]],
                                      device const uchar *kc [[buffer(1)]],
                                      device const uchar *vc [[buffer(2)]],
                                      device float *partials [[buffer(3)]],
                                      constant AttentionGqaArgs &args [[buffer(4)]],
                                      uint2 group [[threadgroup_position_in_grid]],
                                      ushort lane [[thread_index_in_simdgroup]],
                                      ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || blk >= args.n_blocks || sg >= gqa) return;
    const uint p0 = blk * args.block;
    const uint p1 = min(p0 + args.block, args.seq_len);
    const uint qh = kvh * gqa + sg;
    device const float *qh_ptr = q + (ulong)qh * args.q_stride;
    const uint tid = (uint)sg * 32 + lane, threads = gqa * 32;

    threadgroup float Kt[8][256], Vt[8][256];
    float acc[8];                       // head_dim == 256 -> 8 dims per lane
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint t0 = p0; t0 < p1; t0 += 8) {
        const uint rows = min(8u, p1 - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * 256; idx += threads) {
            const uint r = idx >> 8, d = idx & 255;
            device const uchar *kb = kc + ((ulong)(t0 + r) * args.kv_heads + kvh) * 2 * 50;
            device const uchar *vb = vc + ((ulong)(t0 + r) * args.kv_heads + kvh) * 2 * 50;
            Kt[r][d] = turbo_dequant(kb + (d >> 7) * 50, d & 127);
            Vt[r][d] = turbo_dequant(vb + (d >> 7) * 50, d & 127);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < rows; r++) {
            float partial = 0.0f;
            for (uint d = lane; d < 256; d += 32) partial += qh_ptr[d] * Kt[r][d];
            const float score = simd_sum(partial) * args.scale;
            const float m_new = max(m, score);
            const float correction = exp(m - m_new);    // first iteration: exp(-inf) = 0
            const float weight = exp(score - m_new);
            l = l * correction + weight;
            for (uint d = lane, i = 0; d < 256; d += 32, i++)
                acc[i] = acc[i] * correction + weight * Vt[r][d];
            m = m_new;
        }
    }
    device float *ph = partials + ((ulong)qh * args.n_blocks + blk) * 258;
    if (lane == 0) { ph[0] = m; ph[1] = l; }
    for (uint d = lane, i = 0; d < 256; d += 32, i++) ph[2 + d] = acc[i];
}

// Folds the {m, l, acc} block partials for one query head in block-index
// order and writes the normalized output row. One simdgroup per query head;
// a single-block dispatch reduces to normalize-and-store.
kernel void q27_attention_gqa_merge(device const float *partials [[buffer(0)]],
                                     device float *out [[buffer(1)]],
                                     constant AttentionGqaArgs &args [[buffer(2)]],
                                     uint qh [[threadgroup_position_in_grid]],
                                     ushort lane [[thread_index_in_simdgroup]]) {
    if (qh >= args.q_heads) return;
    float acc[8];
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint b = 0; b < args.n_blocks; b++) {
        device const float *ph = partials + ((ulong)qh * args.n_blocks + b) * 258;
        const float m_other = ph[0], l_other = ph[1];
        const float m_new = max(m, m_other);
        if (m_new == -INFINITY) continue;       // both partials empty
        const float c_mine = exp(m - m_new), c_other = exp(m_other - m_new);
        l = l * c_mine + l_other * c_other;
        for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
            acc[i] = acc[i] * c_mine + ph[2 + d] * c_other;
        m = m_new;
    }
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
        out[(ulong)qh * args.head_dim + d] = acc[i] * inv;
}

// GQA KV-reuse causal chunk attention. Same blocked structure as the decode
// GQA kernels above — identical tile staging, per-simdgroup online softmax,
// and block-order merge — so a chunk token's output is bit-identical to the
// GQA decode kernel at the same sequence length. Grid adds the chunk-token
// dimension; a token whose visible sequence ends before this block returns
// before any barrier, and the merge derives each token's live block count
// from base_len + token, so dead partials are never read.
struct AttentionGqaCausalArgs {
    uint q_stride;
    uint q_row_stride;
    uint base_len;
    uint q_heads;
    uint kv_heads;
    uint head_dim;
    uint block;
    uint n_blocks_max;
    uint tokens;
    float scale;
};

kernel void q27_attention_f16_causal_gqa(device const float *q [[buffer(0)]],
                                          device const half *kc [[buffer(1)]],
                                          device const half *vc [[buffer(2)]],
                                          device float *partials [[buffer(3)]],
                                          constant AttentionGqaCausalArgs &args [[buffer(4)]],
                                          uint3 group [[threadgroup_position_in_grid]],
                                          ushort lane [[thread_index_in_simdgroup]],
                                          ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y, token = group.z;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || token >= args.tokens || sg >= gqa) return;
    const uint seq_len = args.base_len + token;
    const uint p0 = blk * args.block;
    if (p0 >= seq_len) return;
    const uint p1 = min(p0 + args.block, seq_len);
    const uint qh = kvh * gqa + sg;
    device const float *qh_ptr = q + (ulong)token * args.q_row_stride + (ulong)qh * args.q_stride;
    const uint tid = (uint)sg * 32 + lane, threads = gqa * 32;

    threadgroup float Kt[8][256], Vt[8][256];
    float acc[8];                       // head_dim <= 256 -> at most 8 dims per lane
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint t0 = p0; t0 < p1; t0 += 8) {
        const uint rows = min(8u, p1 - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * args.head_dim; idx += threads) {
            const uint r = idx / args.head_dim, d = idx % args.head_dim;
            const ulong row = ((ulong)(t0 + r) * args.kv_heads + kvh) * args.head_dim;
            Kt[r][d] = float(kc[row + d]);
            Vt[r][d] = float(vc[row + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < rows; r++) {
            float partial = 0.0f;
            for (uint d = lane; d < args.head_dim; d += 32) partial += qh_ptr[d] * Kt[r][d];
            const float score = simd_sum(partial) * args.scale;
            const float m_new = max(m, score);
            const float correction = exp(m - m_new);    // first iteration: exp(-inf) = 0
            const float weight = exp(score - m_new);
            l = l * correction + weight;
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
                acc[i] = acc[i] * correction + weight * Vt[r][d];
            m = m_new;
        }
    }
    device float *ph = partials +
        (((ulong)token * args.q_heads + qh) * args.n_blocks_max + blk) * 258;
    if (lane == 0) { ph[0] = m; ph[1] = l; }
    for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++) ph[2 + d] = acc[i];
}

kernel void q27_attention_turbo3_causal_gqa(device const float *q [[buffer(0)]],
                                             device const uchar *kc [[buffer(1)]],
                                             device const uchar *vc [[buffer(2)]],
                                             device float *partials [[buffer(3)]],
                                             constant AttentionGqaCausalArgs &args [[buffer(4)]],
                                             uint3 group [[threadgroup_position_in_grid]],
                                             ushort lane [[thread_index_in_simdgroup]],
                                             ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y, token = group.z;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || token >= args.tokens || sg >= gqa) return;
    const uint seq_len = args.base_len + token;
    const uint p0 = blk * args.block;
    if (p0 >= seq_len) return;
    const uint p1 = min(p0 + args.block, seq_len);
    const uint qh = kvh * gqa + sg;
    device const float *qh_ptr = q + (ulong)token * args.q_row_stride + (ulong)qh * args.q_stride;
    const uint tid = (uint)sg * 32 + lane, threads = gqa * 32;

    threadgroup float Kt[8][256], Vt[8][256];
    float acc[8];                       // head_dim == 256 -> 8 dims per lane
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint t0 = p0; t0 < p1; t0 += 8) {
        const uint rows = min(8u, p1 - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * 256; idx += threads) {
            const uint r = idx >> 8, d = idx & 255;
            device const uchar *kb = kc + ((ulong)(t0 + r) * args.kv_heads + kvh) * 2 * 50;
            device const uchar *vb = vc + ((ulong)(t0 + r) * args.kv_heads + kvh) * 2 * 50;
            Kt[r][d] = turbo_dequant(kb + (d >> 7) * 50, d & 127);
            Vt[r][d] = turbo_dequant(vb + (d >> 7) * 50, d & 127);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < rows; r++) {
            float partial = 0.0f;
            for (uint d = lane; d < 256; d += 32) partial += qh_ptr[d] * Kt[r][d];
            const float score = simd_sum(partial) * args.scale;
            const float m_new = max(m, score);
            const float correction = exp(m - m_new);    // first iteration: exp(-inf) = 0
            const float weight = exp(score - m_new);
            l = l * correction + weight;
            for (uint d = lane, i = 0; d < 256; d += 32, i++)
                acc[i] = acc[i] * correction + weight * Vt[r][d];
            m = m_new;
        }
    }
    device float *ph = partials +
        (((ulong)token * args.q_heads + qh) * args.n_blocks_max + blk) * 258;
    if (lane == 0) { ph[0] = m; ph[1] = l; }
    for (uint d = lane, i = 0; d < 256; d += 32, i++) ph[2 + d] = acc[i];
}

// Causal-chunk merge: folds each (token, query head)'s live blocks in index
// order. The live block count comes from base_len + token, so partials of
// blocks past a token's visible sequence are never touched.
kernel void q27_attention_gqa_merge_rows(device const float *partials [[buffer(0)]],
                                          device float *out [[buffer(1)]],
                                          constant AttentionGqaCausalArgs &args [[buffer(2)]],
                                          uint2 group [[threadgroup_position_in_grid]],
                                          ushort lane [[thread_index_in_simdgroup]]) {
    const uint qh = group.x, token = group.y;
    if (qh >= args.q_heads || token >= args.tokens) return;
    const uint seq_len = args.base_len + token;
    const uint live_blocks = 1 + (seq_len - 1) / args.block;    // base_len >= 1 host-checked
    float acc[8];
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint b = 0; b < live_blocks; b++) {
        device const float *ph = partials +
            (((ulong)token * args.q_heads + qh) * args.n_blocks_max + b) * 258;
        const float m_other = ph[0], l_other = ph[1];
        const float m_new = max(m, m_other);
        if (m_new == -INFINITY) continue;       // both partials empty
        const float c_mine = exp(m - m_new), c_other = exp(m_other - m_new);
        l = l * c_mine + l_other * c_other;
        for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
            acc[i] = acc[i] * c_mine + ph[2 + d] * c_other;
        m = m_new;
    }
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
        out[((ulong)token * args.q_heads + qh) * args.head_dim + d] = acc[i] * inv;
}

// ---- Cache-block scheduling R1/R1b kernels ----
// (docs/plans/2026-07-15-cache-block-scheduling.md, Phase 0 results.)
// R1 (head-major layout) measured 1.00x and is PARKED — its probe kernel
// stays bench-only (build/metal_attn_bench), never engine-routed. R1b
// (token-tiled causal, factor 2) measured 2.0x at 32K+ and is production:
// the causal GQA dispatch routes through the _t2 kernels.

// R1 probe (parked): head-major turbo3 KV layout. Identical math to
// q27_attention_turbo3_gqa — only the cache addressing changes. Rows of one
// KV head are contiguous ((kvh * seq_cap + pos) * 100 bytes), so a
// (kvh, blk) threadgroup streams one ~102 KB run instead of 100 B picks at
// 400 B stride. Output must be bit-identical to the interleaved kernel on
// relaid-out data (the bench memcmps before timing).
struct AttentionGqaHmArgs {
    uint q_stride; uint seq_len; uint seq_cap; uint q_heads; uint kv_heads;
    uint head_dim; uint block; uint n_blocks; float scale;
};

kernel void q27_attention_turbo3_gqa_hm(device const float *q [[buffer(0)]],
                                         device const uchar *kc [[buffer(1)]],
                                         device const uchar *vc [[buffer(2)]],
                                         device float *partials [[buffer(3)]],
                                         constant AttentionGqaHmArgs &args [[buffer(4)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         ushort lane [[thread_index_in_simdgroup]],
                                         ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || blk >= args.n_blocks || sg >= gqa) return;
    const uint p0 = blk * args.block;
    const uint p1 = min(p0 + args.block, args.seq_len);
    const uint qh = kvh * gqa + sg;
    device const float *qh_ptr = q + (ulong)qh * args.q_stride;
    const uint tid = (uint)sg * 32 + lane, threads = gqa * 32;

    threadgroup float Kt[8][256], Vt[8][256];
    float acc[8];
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    device const uchar *khead = kc + (ulong)kvh * args.seq_cap * 100;
    device const uchar *vhead = vc + (ulong)kvh * args.seq_cap * 100;
    for (uint t0 = p0; t0 < p1; t0 += 8) {
        const uint rows = min(8u, p1 - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * 256; idx += threads) {
            const uint r = idx >> 8, d = idx & 255;
            device const uchar *kb = khead + (ulong)(t0 + r) * 100;
            device const uchar *vb = vhead + (ulong)(t0 + r) * 100;
            Kt[r][d] = turbo_dequant(kb + (d >> 7) * 50, d & 127);
            Vt[r][d] = turbo_dequant(vb + (d >> 7) * 50, d & 127);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < rows; r++) {
            float partial = 0.0f;
            for (uint d = lane; d < 256; d += 32) partial += qh_ptr[d] * Kt[r][d];
            const float score = simd_sum(partial) * args.scale;
            const float m_new = max(m, score);
            const float correction = exp(m - m_new);
            const float weight = exp(score - m_new);
            l = l * correction + weight;
            for (uint d = lane, i = 0; d < 256; d += 32, i++)
                acc[i] = acc[i] * correction + weight * Vt[r][d];
            m = m_new;
        }
    }
    device float *ph = partials + ((ulong)qh * args.n_blocks + blk) * 258;
    if (lane == 0) { ph[0] = m; ph[1] = l; }
    for (uint d = lane, i = 0; d < 256; d += 32, i++) ph[2 + d] = acc[i];
}

// R1b: token-tiled causal GQA. TF chunk tokens share one (kvh, blk)
// threadgroup: each staged 8-row tile is dequantized once and every resident
// token's per-simdgroup online softmax walks it in token order — the KV
// device stream and dequant ALU divide by TF while each token's arithmetic
// sequence is unchanged, so per-token output is bit-identical to the
// untiled causal GQA kernels (Phase-0 memcmp plus the chunk↔decode parity
// suite). Factor 2 is production (2.0× at 32K+); factor 4 measured strictly
// worse (register pressure) and exists only for the bench comparison.
// Causality: position rows at or past a token's visible sequence are skipped
// per token; only the tile containing a token's own boundary diverges across
// the TF tokens. Partials land at the same per-token slots, so the existing
// merge kernel is reused unchanged.
template <uint TF>
inline void turbo3_causal_gqa_tiled_body(device const float *q,
                                          device const uchar *kc,
                                          device const uchar *vc,
                                          device float *partials,
                                          constant AttentionGqaCausalArgs &args,
                                          uint3 group, ushort lane, ushort sg,
                                          threadgroup float (*Kt)[256],
                                          threadgroup float (*Vt)[256]) {
    const uint kvh = group.x, blk = group.y, tile0 = group.z * TF;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || tile0 >= args.tokens || sg >= gqa) return;
    const uint live = min(TF, args.tokens - tile0);
    const uint p0 = blk * args.block;
    const uint seq_last = args.base_len + tile0 + live - 1;   // deepest token's visible length
    if (p0 >= seq_last) return;
    const uint p1 = min(p0 + args.block, seq_last);
    const uint qh = kvh * gqa + sg;
    const uint tid = (uint)sg * 32 + lane, threads = gqa * 32;
    device const float *qp[TF];
    for (uint f = 0; f < TF; f++)
        qp[f] = q + (ulong)(tile0 + min(f, live - 1)) * args.q_row_stride + (ulong)qh * args.q_stride;

    float acc[TF][8];
    float m[TF], l[TF];
    for (uint f = 0; f < TF; f++) {
        m[f] = -INFINITY; l[f] = 0.0f;
        for (uint i = 0; i < 8; i++) acc[f][i] = 0.0f;
    }
    for (uint t0 = p0; t0 < p1; t0 += 8) {
        const uint rows = min(8u, p1 - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * 256; idx += threads) {
            const uint r = idx >> 8, d = idx & 255;
            device const uchar *kb = kc + ((ulong)(t0 + r) * args.kv_heads + kvh) * 2 * 50;
            device const uchar *vb = vc + ((ulong)(t0 + r) * args.kv_heads + kvh) * 2 * 50;
            Kt[r][d] = turbo_dequant(kb + (d >> 7) * 50, d & 127);
            Vt[r][d] = turbo_dequant(vb + (d >> 7) * 50, d & 127);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < rows; r++) {
            const uint pos = t0 + r;
            float partial[TF];
            for (uint f = 0; f < TF; f++) partial[f] = 0.0f;
            for (uint d = lane; d < 256; d += 32) {
                const float k = Kt[r][d];
                for (uint f = 0; f < TF; f++) partial[f] += qp[f][d] * k;
            }
            float corr[TF], w[TF];
            bool vis[TF];
            for (uint f = 0; f < TF; f++) {
                vis[f] = f < live && pos < args.base_len + tile0 + f;
                corr[f] = 1.0f; w[f] = 0.0f;
                if (vis[f]) {
                    const float score = simd_sum(partial[f]) * args.scale;
                    const float m_new = max(m[f], score);
                    corr[f] = exp(m[f] - m_new);            // first iteration: exp(-inf) = 0
                    w[f] = exp(score - m_new);
                    l[f] = l[f] * corr[f] + w[f];
                    m[f] = m_new;
                }
            }
            for (uint d = lane, i = 0; d < 256; d += 32, i++) {
                const float v = Vt[r][d];
                for (uint f = 0; f < TF; f++)
                    if (vis[f]) acc[f][i] = acc[f][i] * corr[f] + w[f] * v;
            }
        }
    }
    for (uint f = 0; f < live; f++) {
        if (p0 >= args.base_len + tile0 + f) continue;    // merge never reads this slot
        device float *ph = partials +
            (((ulong)(tile0 + f) * args.q_heads + qh) * args.n_blocks_max + blk) * 258;
        if (lane == 0) { ph[0] = m[f]; ph[1] = l[f]; }
        for (uint d = lane, i = 0; d < 256; d += 32, i++) ph[2 + d] = acc[f][i];
    }
}

// R3 probe: barrier-free direct-read block-partial causal GQA attention
// (docs/plans/2026-07-16-r3-barrier-free-attention.md). The tiled body with
// the threadgroup staging and both barriers deleted: each lane dequantizes
// its 8 K dims and 8 V dims per position directly from device into
// registers, in the same element order (d = lane + 32·i) as the staged
// kernels — so per-token arithmetic is unchanged and the output is
// bit-identical to the t2 kernel at equal block size (the bench memcmps
// before timing). Six simdgroups per threadgroup, one per GQA query head,
// share nothing; each KV row is read gqa times (~600 B vs 100 B staged),
// inside the byte range the fp16 A/B proved latency-tolerant. Block size
// arrives via args.block (host override on the probe entry) — the sweep is
// the experiment.
template <uint TF>
inline void turbo3_causal_gqa_bf_body(device const float *q,
                                       device const uchar *kc,
                                       device const uchar *vc,
                                       device float *partials,
                                       constant AttentionGqaCausalArgs &args,
                                       uint3 group, ushort lane, ushort sg) {
    const uint kvh = group.x, blk = group.y, tile0 = group.z * TF;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || tile0 >= args.tokens || sg >= gqa) return;
    const uint live = min(TF, args.tokens - tile0);
    const uint p0 = blk * args.block;
    const uint seq_last = args.base_len + tile0 + live - 1;   // deepest token's visible length
    if (p0 >= seq_last) return;
    const uint p1 = min(p0 + args.block, seq_last);
    const uint qh = kvh * gqa + sg;
    device const float *qp[TF];
    for (uint f = 0; f < TF; f++)
        qp[f] = q + (ulong)(tile0 + min(f, live - 1)) * args.q_row_stride + (ulong)qh * args.q_stride;

    float acc[TF][8];
    float m[TF], l[TF];
    for (uint f = 0; f < TF; f++) {
        m[f] = -INFINITY; l[f] = 0.0f;
        for (uint i = 0; i < 8; i++) acc[f][i] = 0.0f;
    }
    for (uint pos = p0; pos < p1; pos++) {
        device const uchar *kb = kc + ((ulong)pos * args.kv_heads + kvh) * 2 * 50;
        device const uchar *vb = vc + ((ulong)pos * args.kv_heads + kvh) * 2 * 50;
        float kv[8], vv[8];
        for (uint d = lane, i = 0; d < 256; d += 32, i++)
            kv[i] = turbo_dequant(kb + (d >> 7) * 50, d & 127);
        for (uint d = lane, i = 0; d < 256; d += 32, i++)
            vv[i] = turbo_dequant(vb + (d >> 7) * 50, d & 127);
        float partial[TF];
        for (uint f = 0; f < TF; f++) partial[f] = 0.0f;
        for (uint d = lane, i = 0; d < 256; d += 32, i++) {
            const float k = kv[i];
            for (uint f = 0; f < TF; f++) partial[f] += qp[f][d] * k;
        }
        float corr[TF], w[TF];
        bool vis[TF];
        for (uint f = 0; f < TF; f++) {
            vis[f] = f < live && pos < args.base_len + tile0 + f;
            corr[f] = 1.0f; w[f] = 0.0f;
            if (vis[f]) {
                const float score = simd_sum(partial[f]) * args.scale;
                const float m_new = max(m[f], score);
                corr[f] = exp(m[f] - m_new);            // first iteration: exp(-inf) = 0
                w[f] = exp(score - m_new);
                l[f] = l[f] * corr[f] + w[f];
                m[f] = m_new;
            }
        }
        for (uint i = 0; i < 8; i++) {
            const float v = vv[i];
            for (uint f = 0; f < TF; f++)
                if (vis[f]) acc[f][i] = acc[f][i] * corr[f] + w[f] * v;
        }
    }
    for (uint f = 0; f < live; f++) {
        if (p0 >= args.base_len + tile0 + f) continue;    // merge never reads this slot
        device float *ph = partials +
            (((ulong)(tile0 + f) * args.q_heads + qh) * args.n_blocks_max + blk) * 258;
        if (lane == 0) { ph[0] = m[f]; ph[1] = l[f]; }
        for (uint d = lane, i = 0; d < 256; d += 32, i++) ph[2 + d] = acc[f][i];
    }
}

kernel void q27_attention_turbo3_causal_gqa_bf2(device const float *q [[buffer(0)]],
        device const uchar *kc [[buffer(1)]], device const uchar *vc [[buffer(2)]],
        device float *partials [[buffer(3)]],
        constant AttentionGqaCausalArgs &args [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    turbo3_causal_gqa_bf_body<2>(q, kc, vc, partials, args, group, lane, sg);
}

kernel void q27_attention_turbo3_causal_gqa_t2(device const float *q [[buffer(0)]],
        device const uchar *kc [[buffer(1)]], device const uchar *vc [[buffer(2)]],
        device float *partials [[buffer(3)]],
        constant AttentionGqaCausalArgs &args [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Kt[8][256], Vt[8][256];
    turbo3_causal_gqa_tiled_body<2>(q, kc, vc, partials, args, group, lane, sg, Kt, Vt);
}

kernel void q27_attention_turbo3_causal_gqa_t4(device const float *q [[buffer(0)]],
        device const uchar *kc [[buffer(1)]], device const uchar *vc [[buffer(2)]],
        device float *partials [[buffer(3)]],
        constant AttentionGqaCausalArgs &args [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Kt[8][256], Vt[8][256];
    turbo3_causal_gqa_tiled_body<4>(q, kc, vc, partials, args, group, lane, sg, Kt, Vt);
}

// fp16 token-tiled causal GQA: the same tiling transform applied to
// q27_attention_f16_causal_gqa — head_dim-parametric staging and d-loops
// mirror the untiled kernel expression-for-expression so per-token output
// stays bit-identical.
template <uint TF>
inline void f16_causal_gqa_tiled_body(device const float *q,
                                       device const half *kc,
                                       device const half *vc,
                                       device float *partials,
                                       constant AttentionGqaCausalArgs &args,
                                       uint3 group, ushort lane, ushort sg,
                                       threadgroup float (*Kt)[256],
                                       threadgroup float (*Vt)[256]) {
    const uint kvh = group.x, blk = group.y, tile0 = group.z * TF;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || tile0 >= args.tokens || sg >= gqa) return;
    const uint live = min(TF, args.tokens - tile0);
    const uint p0 = blk * args.block;
    const uint seq_last = args.base_len + tile0 + live - 1;
    if (p0 >= seq_last) return;
    const uint p1 = min(p0 + args.block, seq_last);
    const uint qh = kvh * gqa + sg;
    const uint tid = (uint)sg * 32 + lane, threads = gqa * 32;
    device const float *qp[TF];
    for (uint f = 0; f < TF; f++)
        qp[f] = q + (ulong)(tile0 + min(f, live - 1)) * args.q_row_stride + (ulong)qh * args.q_stride;

    float acc[TF][8];
    float m[TF], l[TF];
    for (uint f = 0; f < TF; f++) {
        m[f] = -INFINITY; l[f] = 0.0f;
        for (uint i = 0; i < 8; i++) acc[f][i] = 0.0f;
    }
    for (uint t0 = p0; t0 < p1; t0 += 8) {
        const uint rows = min(8u, p1 - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * args.head_dim; idx += threads) {
            const uint r = idx / args.head_dim, d = idx % args.head_dim;
            const ulong row = ((ulong)(t0 + r) * args.kv_heads + kvh) * args.head_dim;
            Kt[r][d] = float(kc[row + d]);
            Vt[r][d] = float(vc[row + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < rows; r++) {
            const uint pos = t0 + r;
            float partial[TF];
            for (uint f = 0; f < TF; f++) partial[f] = 0.0f;
            for (uint d = lane; d < args.head_dim; d += 32) {
                const float k = Kt[r][d];
                for (uint f = 0; f < TF; f++) partial[f] += qp[f][d] * k;
            }
            float corr[TF], w[TF];
            bool vis[TF];
            for (uint f = 0; f < TF; f++) {
                vis[f] = f < live && pos < args.base_len + tile0 + f;
                corr[f] = 1.0f; w[f] = 0.0f;
                if (vis[f]) {
                    const float score = simd_sum(partial[f]) * args.scale;
                    const float m_new = max(m[f], score);
                    corr[f] = exp(m[f] - m_new);            // first iteration: exp(-inf) = 0
                    w[f] = exp(score - m_new);
                    l[f] = l[f] * corr[f] + w[f];
                    m[f] = m_new;
                }
            }
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++) {
                const float v = Vt[r][d];
                for (uint f = 0; f < TF; f++)
                    if (vis[f]) acc[f][i] = acc[f][i] * corr[f] + w[f] * v;
            }
        }
    }
    for (uint f = 0; f < live; f++) {
        if (p0 >= args.base_len + tile0 + f) continue;
        device float *ph = partials +
            (((ulong)(tile0 + f) * args.q_heads + qh) * args.n_blocks_max + blk) * 258;
        if (lane == 0) { ph[0] = m[f]; ph[1] = l[f]; }
        for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++) ph[2 + d] = acc[f][i];
    }
}

kernel void q27_attention_f16_causal_gqa_t2(device const float *q [[buffer(0)]],
        device const half *kc [[buffer(1)]], device const half *vc [[buffer(2)]],
        device float *partials [[buffer(3)]],
        constant AttentionGqaCausalArgs &args [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Kt[8][256], Vt[8][256];
    f16_causal_gqa_tiled_body<2>(q, kc, vc, partials, args, group, lane, sg, Kt, Vt);
}

// Top-k candidate extraction for GPU-assisted sampling: a 16-bit radix
// select finds a threshold whose over-set contains the true top-k, then
// compacts (value, index) pairs so the host reads ~k candidates instead of
// the whole vocabulary. The over-set can exceed k on 16-bit-key ties; the
// host truncates after an exact sort, and falls back to a full logits
// readback when out_count exceeds the capacity (degenerate tie storms).
struct TopkArgs { uint n; uint k; uint capacity; };

inline uint topk_sortable(float value) {
    // Monotonic float -> uint map: larger float == larger key. NaN maps
    // high or low deterministically; logits are finite in practice.
    uint u = as_type<uint>(value);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

kernel void q27_topk_logits(device const float *logits [[buffer(0)]],
                             device float *out_values [[buffer(1)]],
                             device uint *out_indices [[buffer(2)]],
                             device atomic_uint *out_count [[buffer(3)]],
                             constant TopkArgs &args [[buffer(4)]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint threads [[threads_per_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint tg_b1, tg_above, tg_threshold;

    // Pass 1: histogram of the top key byte; find the bin where the
    // descending cumulative count crosses k, and the count strictly above it.
    for (uint b = tid; b < 256; b += threads) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < args.n; i += threads)
        atomic_fetch_add_explicit(&hist[topk_sortable(logits[i]) >> 24], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint cumulative = 0, bin = 255;
        for (;; bin--) {
            const uint count = atomic_load_explicit(&hist[bin], memory_order_relaxed);
            if (cumulative + count >= args.k || bin == 0) { tg_above = cumulative; break; }
            cumulative += count;
        }
        tg_b1 = bin;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint b1 = tg_b1;
    const uint above = tg_above;

    // Pass 2: histogram of the second byte within the boundary bin; walk it
    // for the k - above candidates the boundary bin must still supply.
    for (uint b = tid; b < 256; b += threads) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < args.n; i += threads) {
        const uint key = topk_sortable(logits[i]);
        if ((key >> 24) == b1)
            atomic_fetch_add_explicit(&hist[(key >> 16) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const uint remaining = args.k > above ? args.k - above : 1;
        uint cumulative = 0, bin = 255;
        for (;; bin--) {
            cumulative += atomic_load_explicit(&hist[bin], memory_order_relaxed);
            if (cumulative >= remaining || bin == 0) break;
        }
        tg_threshold = (b1 << 8) | bin;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint threshold = tg_threshold;

    // Pass 3: compact every candidate at or above the 16-bit threshold.
    for (uint i = tid; i < args.n; i += threads) {
        const uint key16 = topk_sortable(logits[i]) >> 16;
        if (key16 >= threshold) {
            const uint slot = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
            if (slot < args.capacity) { out_values[slot] = logits[i]; out_indices[slot] = i; }
        }
    }
}
