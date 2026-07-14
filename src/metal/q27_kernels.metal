#include <metal_stdlib>
using namespace metal;

struct MatvecArgs {
    uint rows;
    uint cols;
    uint simdgroups;
};

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
kernel void q27_attention_f16(device const float *q [[buffer(0)]],
                               device const half *kc [[buffer(1)]],
                               device const half *vc [[buffer(2)]],
                               device float *prob     [[buffer(3)]],
                               device float *out      [[buffer(4)]],
                               constant AttentionArgs &args [[buffer(5)]],
                               uint qh [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    if (qh >= args.q_heads) return;
    const uint gqa = args.q_heads / args.kv_heads;
    const uint kvh = qh / gqa;
    device const float *qh_ptr = q + (ulong)qh * args.q_stride;
    device float *ph = prob + (ulong)qh * args.seq_len;
    if (tid == 0) {
        float maximum = -INFINITY;
        for (uint p = 0; p < args.seq_len; p++) {
            device const half *kh = kc + ((ulong)p * args.kv_heads + kvh) * args.head_dim;
            float score = 0.0f;
            for (uint d = 0; d < args.head_dim; d++) score += qh_ptr[d] * float(kh[d]);
            score *= args.scale;
            ph[p] = score;
            maximum = max(maximum, score);
        }
        float denominator = 0.0f;
        for (uint p = 0; p < args.seq_len; p++) { ph[p] = exp(ph[p] - maximum); denominator += ph[p]; }
        const float inv = 1.0f / denominator;
        for (uint p = 0; p < args.seq_len; p++) ph[p] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint d = tid; d < args.head_dim; d += 256) {
        float value = 0.0f;
        for (uint p = 0; p < args.seq_len; p++) {
            device const half *vh = vc + ((ulong)p * args.kv_heads + kvh) * args.head_dim;
            value += ph[p] * float(vh[d]);
        }
        out[(ulong)qh * args.head_dim + d] = value;
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
    float result = 0.0f;
    const ulong base = (ulong)row * args.cols;
    for (uint b = 0; b < args.cols / 32; b++) {
        const uint c = b * 32 + lane;
        int subtotal = int(weights[base + c]) * int(x[c]);
        subtotal = simd_sum(subtotal);
        if (lane == 0) result += float(subtotal) * float(weight_scales[(ulong)row * (args.cols / 128) + b / 4]) * x_scales[b];
    }
    if (lane == 0) out[row] = result;
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
    float result = 0.0f;
    const ulong base = (ulong)row * (args.cols / 2);
    for (uint b = 0; b < args.cols / 32; b++) {
        const uint c = b * 32 + lane;
        const uchar packed = weights[base + c / 2];
        const int w = int((c & 1) ? (packed >> 4) : (packed & 15)) - 8;
        int subtotal = w * int(x[c]);
        subtotal = simd_sum(subtotal);
        if (lane == 0) result += float(subtotal) * float(weight_scales[(ulong)row * (args.cols / 64) + b / 2]) * x_scales[b];
    }
    if (lane == 0) out[row] = result;
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
