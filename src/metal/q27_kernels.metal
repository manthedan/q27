// Q27_SHADER_ABI 3
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
    uint simdgroups;
};
struct MatvecPairArgs { uint rows_a; uint rows_b; uint cols; uint simdgroups; };
struct MatmulArgs { uint rows; uint cols; uint x_rows; uint simdgroups; };

inline void reduce_row(float sum, device float *out, uint row,
                       threadgroup float *partial, ushort lane, ushort simdgroup,
                       uint simdgroups) {
    sum = simd_sum(sum);
    if (lane == 0) partial[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        float total = lane < simdgroups ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) out[row] = total;
    }
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
        int q=scale>0.0f?int(rint(v/scale)):0; q=clamp(q,-127,127);
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
    const float softplus = value > 20.0f ? value : log(1.0f + exp(value));
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
    int q = scale > 0.0f ? int(rint(v / scale)) : 0;
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

// Multi-token packed-dot GEMM (x_rows 1..12) for chunked prefill and batched
// MTP verification. Same streaming structure as the packed-dot GEMV above:
// one simdgroup per output row, each lane owns a 32-column slice per
// iteration, per-lane partials reduce once per row via simd_sum. Each
// 32-column weight packet is loaded (and for Q4, nibble-decoded) once and
// dotted against every token row, so the weight stream costs the same as a
// single GEMV while producing up to 12 outputs. Accumulation order is
// exactly the GEMV's — integer dot per 32 columns, (float(dot) * weight
// scale) * activation scale, summed in column order, one simd_sum — so a
// chunked projection is bit-identical to running the serial GEMV per token.
// (A 4-rows-per-simdgroup variant that amortized activation conversion
// measured 1.6x slower: 32 live char4 weight packets plus 48 accumulators
// spill. The kernel is scalar-issue-bound near 4 ops per multiply-add; the
// planned simdgroup_matrix tile kernel is the structural fix.)
// Tokens process in unrolled groups of four; whole groups drop out for
// narrow chunks, and loads for token slots past x_rows-1 clamp to the last
// valid row (their results are discarded) so partial chunks never read out
// of bounds.
inline int q27_dot32(thread const char4 *w, device const int4 *x16, uint idx) {
    const int4 xp0 = x16[idx], xp1 = x16[idx + 1];
    return q27_dot4(w[0], as_type<char4>(xp0.x)) + q27_dot4(w[1], as_type<char4>(xp0.y)) +
           q27_dot4(w[2], as_type<char4>(xp0.z)) + q27_dot4(w[3], as_type<char4>(xp0.w)) +
           q27_dot4(w[4], as_type<char4>(xp1.x)) + q27_dot4(w[5], as_type<char4>(xp1.y)) +
           q27_dot4(w[6], as_type<char4>(xp1.z)) + q27_dot4(w[7], as_type<char4>(xp1.w));
}

inline void q27_decode_q4x8(uint packed, thread char4 *w) {
    w[0] = char4(char(int(packed         & 15u) - 8), char(int((packed >>  4) & 15u) - 8),
                 char(int((packed >>  8) & 15u) - 8), char(int((packed >> 12) & 15u) - 8));
    w[1] = char4(char(int((packed >> 16) & 15u) - 8), char(int((packed >> 20) & 15u) - 8),
                 char(int((packed >> 24) & 15u) - 8), char(int((packed >> 28)      ) - 8));
}

// One token's contribution for the vectorized main loop (32 columns) and the
// scalar tail (4 columns). `xoff16`/`xsoff` are the token row's base offsets
// in int4 / scale-group units, hoisted by the caller.
#define Q27_MM_MAIN(acc, xoff16, xsoff)                                              \
    acc += float(q27_dot32(w, x16, (xoff16) + idx2)) * wscale * x_scales[(xsoff) + cs];
#define Q27_MM_TAIL(acc, xoff16, xsoff)                                              \
    acc += float(q27_dot4(wt, *(device const char4 *)(x + (ulong)(xoff16) * 16 + c))) * \
           wscale * x_scales[(xsoff) + c / 32];
#define Q27_MM_TOKENS(MACRO)                                                         \
    MACRO(acc0.x, xoff16[0], xsoff[0]);                                              \
    MACRO(acc0.y, xoff16[1], xsoff[1]);                                              \
    MACRO(acc0.z, xoff16[2], xsoff[2]);                                              \
    MACRO(acc0.w, xoff16[3], xsoff[3]);                                              \
    if (args.x_rows > 4) {                                                           \
        MACRO(acc1.x, xoff16[4], xsoff[4]);                                          \
        MACRO(acc1.y, xoff16[5], xsoff[5]);                                          \
        MACRO(acc1.z, xoff16[6], xsoff[6]);                                          \
        MACRO(acc1.w, xoff16[7], xsoff[7]);                                          \
    }                                                                                \
    if (args.x_rows > 8) {                                                           \
        MACRO(acc2.x, xoff16[8], xsoff[8]);                                          \
        MACRO(acc2.y, xoff16[9], xsoff[9]);                                          \
        MACRO(acc2.z, xoff16[10], xsoff[10]);                                        \
        MACRO(acc2.w, xoff16[11], xsoff[11]);                                        \
    }

// Shared prologue (clamped token-row base offsets; x.count fits in 32 bits,
// checked at dispatch) and epilogue (simd_sum reduction, executed by every
// lane, then a lane-0 store of the first x_rows results).
#define Q27_MM_OFFSETS                                                               \
    const uint t_last = args.x_rows - 1;                                             \
    uint xoff16[12], xsoff[12];                                                      \
    for (uint t = 0; t < 12; t++) {                                                  \
        const uint tc = min(t, t_last);                                              \
        xoff16[t] = tc * (args.cols / 16);                                           \
        xsoff[t] = tc * (args.cols / 32);                                            \
    }                                                                                \
    float4 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
#define Q27_MM_STORE                                                                 \
    const float sums[12] = {                                                         \
        simd_sum(acc0.x), simd_sum(acc0.y), simd_sum(acc0.z), simd_sum(acc0.w),      \
        simd_sum(acc1.x), simd_sum(acc1.y), simd_sum(acc1.z), simd_sum(acc1.w),      \
        simd_sum(acc2.x), simd_sum(acc2.y), simd_sum(acc2.z), simd_sum(acc2.w)};     \
    if (lane == 0)                                                                   \
        for (uint t = 0; t < args.x_rows; t++) out[(ulong)t * args.rows + row] = sums[t];

kernel void q27_matmul_q4_rows(
        device const uchar *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const char *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint row = group * 8 + simdgroup;
    if (row >= args.rows) return;
    device const uint4 *w4x8 = (device const uint4 *)(weights + (ulong)row * (args.cols / 2));
    device const int4 *x16 = (device const int4 *)x;
    const ulong scale_base = (ulong)row * (args.cols / 64);
    Q27_MM_OFFSETS
    const uint chunks = args.cols / 1024;
    for (uint chunk = 0; chunk < chunks; chunk++) {
        const uint idx = chunk * 32 + lane;
        const uint4 wp = w4x8[idx];
        char4 w[8];
        q27_decode_q4x8(wp.x, w);
        q27_decode_q4x8(wp.y, w + 2);
        q27_decode_q4x8(wp.z, w + 4);
        q27_decode_q4x8(wp.w, w + 6);
        const uint c = chunk * 1024 + lane * 32;
        const uint idx2 = idx * 2, cs = c / 32;
        const float wscale = float(weight_scales[scale_base + c / 64]);
        Q27_MM_TOKENS(Q27_MM_MAIN)
    }
    for (uint c = chunks * 1024 + lane * 4; c < args.cols; c += 128) {
        const uchar2 wp = *(device const uchar2 *)(weights + (ulong)row * (args.cols / 2) + c / 2);
        const char4 wt = char4(char(int(wp.x & 15u) - 8), char(int(wp.x >> 4) - 8),
                               char(int(wp.y & 15u) - 8), char(int(wp.y >> 4) - 8));
        const float wscale = float(weight_scales[scale_base + c / 64]);
        Q27_MM_TOKENS(Q27_MM_TAIL)
    }
    Q27_MM_STORE
}

kernel void q27_matmul_q8_rows(
        device const char *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const char *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint row = group * 8 + simdgroup;
    if (row >= args.rows) return;
    device const int4 *w16 = (device const int4 *)(weights + (ulong)row * args.cols);
    device const int4 *x16 = (device const int4 *)x;
    const ulong scale_base = (ulong)row * (args.cols / 128);
    Q27_MM_OFFSETS
    const uint chunks = args.cols / 1024;
    for (uint chunk = 0; chunk < chunks; chunk++) {
        const uint idx = chunk * 64 + lane * 2;
        const int4 wp0 = w16[idx], wp1 = w16[idx + 1];
        const char4 w[8] = {
            as_type<char4>(wp0.x), as_type<char4>(wp0.y), as_type<char4>(wp0.z),
            as_type<char4>(wp0.w), as_type<char4>(wp1.x), as_type<char4>(wp1.y),
            as_type<char4>(wp1.z), as_type<char4>(wp1.w)};
        const uint c = chunk * 1024 + lane * 32;
        const uint idx2 = idx, cs = c / 32;
        const float wscale = float(weight_scales[scale_base + c / 128]);
        Q27_MM_TOKENS(Q27_MM_MAIN)
    }
    for (uint c = chunks * 1024 + lane * 4; c < args.cols; c += 128) {
        const char4 wt = *(device const char4 *)(weights + (ulong)row * args.cols + c);
        const float wscale = float(weight_scales[scale_base + c / 128]);
        Q27_MM_TOKENS(Q27_MM_TAIL)
    }
    Q27_MM_STORE
}

// ---- Chunked layer-major prefill (2..12 tokens per dispatch) ----
//
// These kernels advance a whole token chunk through one operation so the
// engine can execute prompts layer-major and route projections through the
// simdgroup GEMM. Recurrent operators (convolution ring, DeltaNet state)
// stay sequential across the chunk inside a single dispatch and commit
// their state once per chunk.

struct EmbedRowsArgs { uint cols; uint count; uint tokens[12]; };
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
        int q = scale > 0.0f ? int(rint(v / scale)) : 0; q = clamp(q, -127, 127);
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
    const float softplus = value > 20.0f ? value : log(1.0f + exp(value));
    g[gid] = ssm_a[head] * softplus;
    beta[gid] = 1.0f / (1.0f + exp(-beta_raw[gid]));
}

struct ConvChunkArgs { uint channels; uint tokens; };
kernel void q27_conv_chunk(device float *ring [[buffer(0)]],
                            device const float *qkv [[buffer(1)]],
                            device const float *weight [[buffer(2)]],
                            device float *out [[buffer(3)]],
                            constant ConvChunkArgs &args [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= args.channels) return;
    float r0 = ring[gid], r1 = ring[args.channels + gid], r2 = ring[(ulong)2 * args.channels + gid];
    const float w0 = weight[(ulong)gid * 4], w1 = weight[(ulong)gid * 4 + 1];
    const float w2 = weight[(ulong)gid * 4 + 2], w3 = weight[(ulong)gid * 4 + 3];
    for (uint t = 0; t < args.tokens; t++) {
        const float xv = qkv[(ulong)t * args.channels + gid];
        const float value = r0 * w0 + r1 * w1 + r2 * w2 + xv * w3;
        out[(ulong)t * args.channels + gid] = value / (1.0f + exp(-value));
        r0 = r1; r1 = r2; r2 = xv;
    }
    ring[gid] = r0; ring[args.channels + gid] = r1; ring[(ulong)2 * args.channels + gid] = r2;
}

struct DeltaChunkArgs { uint value_heads; uint qk_heads; uint head_dim; uint tokens; };
kernel void q27_delta_chunk(device float *state [[buffer(0)]],
                             device const float *conv [[buffer(1)]],
                             device const float *g [[buffer(2)]],
                             device const float *beta [[buffer(3)]],
                             device float *out [[buffer(4)]],
                             constant DeltaChunkArgs &args [[buffer(5)]],
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
    device float *sh = state + (ulong)head * 128 * 128;
    // The chunk's whole state slice lives in registers; it is written back
    // to device memory exactly once, at the chunk boundary.
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
    for (uint n = 0; n < 32; n++) sh[(ulong)(i0 + n) * 128 + j] = saved[n];
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

struct AttentionCausalArgs {
    uint q_stride; uint q_row_stride; uint base_len;
    uint q_heads; uint kv_heads; uint head_dim; uint tokens; float scale;
};
kernel void q27_attention_f16_causal(device const float *q [[buffer(0)]],
                                      device const half *kc [[buffer(1)]],
                                      device const half *vc [[buffer(2)]],
                                      device float *prob     [[buffer(3)]],
                                      device float *out      [[buffer(4)]],
                                      constant AttentionCausalArgs &args [[buffer(5)]],
                                      uint2 group [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]]) {
    const uint qh = group.x, token = group.y;
    if (qh >= args.q_heads || token >= args.tokens) return;
    const uint seq_len = args.base_len + token;
    const uint max_seq = args.base_len + args.tokens - 1;
    const uint gqa = args.q_heads / args.kv_heads;
    const uint kvh = qh / gqa;
    device const float *qh_ptr = q + (ulong)token * args.q_row_stride + (ulong)qh * args.q_stride;
    device float *ph = prob + ((ulong)token * args.q_heads + qh) * max_seq;
    if (tid == 0) {
        float maximum = -INFINITY;
        for (uint p = 0; p < seq_len; p++) {
            device const half *kh = kc + ((ulong)p * args.kv_heads + kvh) * args.head_dim;
            float score = 0.0f;
            for (uint d = 0; d < args.head_dim; d++) score += qh_ptr[d] * float(kh[d]);
            score *= args.scale;
            ph[p] = score;
            maximum = max(maximum, score);
        }
        float denominator = 0.0f;
        for (uint p = 0; p < seq_len; p++) { ph[p] = exp(ph[p] - maximum); denominator += ph[p]; }
        const float inv = 1.0f / denominator;
        for (uint p = 0; p < seq_len; p++) ph[p] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint d = tid; d < args.head_dim; d += 256) {
        float value = 0.0f;
        for (uint p = 0; p < seq_len; p++) {
            device const half *vh = vc + ((ulong)p * args.kv_heads + kvh) * args.head_dim;
            value += ph[p] * float(vh[d]);
        }
        out[((ulong)token * args.q_heads + qh) * args.head_dim + d] = value;
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
    if (head >= args.heads || j >= 128) return;
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
    if (h >= args.kv_heads || group.y >= 2 || j >= 128) return;
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

kernel void q27_attention_turbo3(device const float *q [[buffer(0)]],
                                  device const uchar *kc [[buffer(1)]],
                                  device const uchar *vc [[buffer(2)]],
                                  device float *prob [[buffer(3)]],
                                  device float *out [[buffer(4)]],
                                  constant AttentionArgs &args [[buffer(5)]],
                                  uint qh [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    if (qh >= args.q_heads) return;
    const uint gqa = args.q_heads / args.kv_heads, kvh = qh / gqa;
    device const float *qh_ptr = q + (ulong)qh * args.q_stride;
    device float *ph = prob + (ulong)qh * args.seq_len;
    if (tid == 0) {
        float maximum = -INFINITY;
        for (uint p = 0; p < args.seq_len; p++) {
            float score = 0.0f;
            for (uint d = 0; d < args.head_dim; d++) {
                device const uchar *block = kc + ((ulong)p * args.kv_heads * 2 + kvh * 2 + (d >> 7)) * 50;
                score += qh_ptr[d] * turbo_dequant(block, d & 127);
            }
            score *= args.scale; ph[p] = score; maximum = max(maximum, score);
        }
        float denominator = 0.0f;
        for (uint p = 0; p < args.seq_len; p++) { ph[p] = exp(ph[p] - maximum); denominator += ph[p]; }
        for (uint p = 0; p < args.seq_len; p++) ph[p] /= denominator;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint d = tid; d < args.head_dim; d += 256) {
        float value = 0.0f;
        for (uint p = 0; p < args.seq_len; p++) {
            device const uchar *block = vc + ((ulong)p * args.kv_heads * 2 + kvh * 2 + (d >> 7)) * 50;
            value += ph[p] * turbo_dequant(block, d & 127);
        }
        out[(ulong)qh * args.head_dim + d] = value;
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
    if (h >= args.kv_heads || group.y >= 2 || token >= args.tokens || j >= 128) return;
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

kernel void q27_attention_turbo3_causal(device const float *q [[buffer(0)]],
                                         device const uchar *kc [[buffer(1)]],
                                         device const uchar *vc [[buffer(2)]],
                                         device float *prob [[buffer(3)]],
                                         device float *out [[buffer(4)]],
                                         constant AttentionCausalArgs &args [[buffer(5)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]]) {
    const uint qh = group.x, token = group.y;
    if (qh >= args.q_heads || token >= args.tokens) return;
    const uint seq_len = args.base_len + token;
    const uint max_seq = args.base_len + args.tokens - 1;
    const uint gqa = args.q_heads / args.kv_heads, kvh = qh / gqa;
    device const float *qh_ptr = q + (ulong)token * args.q_row_stride + (ulong)qh * args.q_stride;
    device float *ph = prob + ((ulong)token * args.q_heads + qh) * max_seq;
    if (tid == 0) {
        float maximum = -INFINITY;
        for (uint p = 0; p < seq_len; p++) {
            float score = 0.0f;
            for (uint d = 0; d < args.head_dim; d++) {
                device const uchar *block = kc + ((ulong)p * args.kv_heads * 2 + kvh * 2 + (d >> 7)) * 50;
                score += qh_ptr[d] * turbo_dequant(block, d & 127);
            }
            score *= args.scale; ph[p] = score; maximum = max(maximum, score);
        }
        float denominator = 0.0f;
        for (uint p = 0; p < seq_len; p++) { ph[p] = exp(ph[p] - maximum); denominator += ph[p]; }
        for (uint p = 0; p < seq_len; p++) ph[p] /= denominator;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint d = tid; d < args.head_dim; d += 256) {
        float value = 0.0f;
        for (uint p = 0; p < seq_len; p++) {
            device const uchar *block = vc + ((ulong)p * args.kv_heads * 2 + kvh * 2 + (d >> 7)) * 50;
            value += ph[p] * turbo_dequant(block, d & 127);
        }
        out[((ulong)token * args.q_heads + qh) * args.head_dim + d] = value;
    }
}
