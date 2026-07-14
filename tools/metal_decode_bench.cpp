// Synthetic decode-step benchmark for the Metal backend.
//
// Replays the exact per-token dispatch sequence of MetalEngine::decode —
// 48 GDN blocks, 16 attention blocks, 64 FFN blocks, output head, argmax —
// against synthetic weights at the production shapes, one command buffer per
// token, one CPU readback per token. Because the synthetic weights are
// resident (no 17 GiB mmap), this measures the compute/bandwidth-limited
// warm-decode ceiling; run with Q27_METAL_PROFILE=1 for the per-kernel
// share table. This is the attribution tool for the critical-path item-3
// residual: the gap between this ceiling and the observed artifact tok/s is
// weight paging, and the profile shares split the rest.
//
// Memory-safe: synthetic buffers only (~1.6 GiB peak), no model artifact.

#include "metal_backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using q27::DType;

namespace {

// Mirrors MetalEngine's validated Qwen3.6-27B architecture constants.
constexpr uint32_t N_LAYER = 64;
constexpr uint32_t N_EMBD = 5120;
constexpr uint32_t N_FFN = 17408;
constexpr uint32_t N_HEAD = 24;
constexpr uint32_t N_KV = 4;
constexpr uint32_t HEAD_DIM = 256;
constexpr uint32_t N_ROT = 64;
constexpr uint32_t GDN_CH = 10240;
constexpr uint32_t GDN_V = 6144;
constexpr uint32_t GDN_HEADS = 48;
constexpr uint32_t GDN_QK_HEADS = 16;
constexpr uint32_t GDN_DIM = 128;
constexpr uint32_t VOCAB = 248320;
constexpr float EPS = 1e-6f;
constexpr float FREQ_BASE = 1e7f;

bool attention_layer(uint32_t layer) { return layer % 4 == 3; }

struct Synthetic {
    std::vector<uint8_t> data;
    std::vector<uint16_t> scales;
    q27::BackendTensor tensor;
};

uint64_t tensor_bytes(const q27::BackendTensor& t) {
    uint64_t data = t.rows * t.cols;
    if (t.dtype == DType::F32) data *= 4;
    if (t.dtype == DType::F16) data *= 2;
    if (t.dtype == DType::Q4_G64) data /= 2;
    const uint64_t group = t.dtype == DType::Q8_G128 ? 128 : t.dtype == DType::Q4_G64 ? 64 : 0;
    return data + (group ? t.rows * (t.cols / group) * 2 : 0);
}

Synthetic make_quant(q27::MetalBackend& backend, uint32_t rows, uint32_t cols, DType dtype) {
    Synthetic s;
    const uint64_t group = dtype == DType::Q8_G128 ? 128 : 64;
    s.data.resize((uint64_t)rows * cols / (dtype == DType::Q4_G64 ? 2 : 1));
    for (size_t i = 0; i < s.data.size(); i++) s.data[i] = (uint8_t)(i * 2654435761u >> 24);
    s.scales.assign((uint64_t)rows * (cols / group), 0x3c00 /* f16 1.0 */);
    q27::Tensor t;
    t.name = "synthetic";
    t.dtype = dtype;
    t.shape = {rows, cols};
    t.data = s.data.data();
    t.data_size = s.data.size();
    t.scales = reinterpret_cast<const uint8_t*>(s.scales.data());
    t.scales_size = s.scales.size() * sizeof(uint16_t);
    s.tensor = backend.upload(t);
    return s;
}

Synthetic make_f16(q27::MetalBackend& backend, uint32_t rows, uint32_t cols) {
    Synthetic s;
    s.data.resize((uint64_t)rows * cols * 2);
    for (size_t i = 0; i < s.data.size(); i += 2) {
        // Varied but always-finite small half-precision values.
        const uint16_t h = (uint16_t)(0x3000 | ((i * 2654435761u >> 20) & 0x03ff));
        std::memcpy(&s.data[i], &h, 2);
    }
    q27::Tensor t;
    t.name = "synthetic";
    t.dtype = DType::F16;
    t.shape = {rows, cols};
    t.data = s.data.data();
    t.data_size = s.data.size();
    s.tensor = backend.upload(t);
    return s;
}

Synthetic make_f32(q27::MetalBackend& backend, const std::vector<uint64_t>& shape) {
    Synthetic s;
    uint64_t count = 1;
    for (uint64_t d : shape) count *= d;
    s.data.resize(count * 4);
    for (uint64_t i = 0; i < count; i++) {
        const float v = 0.5f + float(i % 13) * 0.01f;
        std::memcpy(&s.data[i * 4], &v, 4);
    }
    q27::Tensor t;
    t.name = "synthetic";
    t.dtype = DType::F32;
    t.shape = shape;
    t.data = s.data.data();
    t.data_size = s.data.size();
    s.tensor = backend.upload(t);
    return s;
}

} // namespace

int main(int argc, char** argv) {
    uint32_t tokens = 16, seq = 128;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--tokens" && i + 1 < argc) tokens = (uint32_t)atoi(argv[++i]);
        else if (arg == "--seq" && i + 1 < argc) seq = (uint32_t)atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--tokens N] [--seq LEN]\n", argv[0]); return 1; }
    }
    if (!tokens || !seq) { fprintf(stderr, "invalid --tokens/--seq\n"); return 1; }

    q27::MetalBackend backend;
    printf("backend: %s, %u simulated tokens at context %u\n", backend.name().c_str(), tokens, seq);

    // One synthetic weight set per layer type; a block streams far more bytes
    // than any cache level, so reuse across the 48/16/64 repeats is
    // bandwidth-equivalent to distinct per-layer tensors.
    Synthetic gdn_qkv = make_quant(backend, GDN_CH, N_EMBD, DType::Q4_G64);
    Synthetic gdn_gate = make_quant(backend, GDN_V, N_EMBD, DType::Q4_G64);
    Synthetic gdn_out = make_quant(backend, N_EMBD, GDN_V, DType::Q4_G64);
    Synthetic ssm_alpha = make_f16(backend, GDN_HEADS, N_EMBD);
    Synthetic ssm_beta = make_f16(backend, GDN_HEADS, N_EMBD);
    Synthetic ssm_a = make_f32(backend, {GDN_HEADS});
    Synthetic ssm_dt = make_f32(backend, {GDN_HEADS});
    Synthetic ssm_conv = make_f32(backend, {GDN_CH, 4});
    Synthetic ssm_norm = make_f32(backend, {GDN_DIM});
    Synthetic attn_q = make_quant(backend, 2 * N_HEAD * HEAD_DIM, N_EMBD, DType::Q4_G64);
    Synthetic attn_k = make_quant(backend, N_KV * HEAD_DIM, N_EMBD, DType::Q4_G64);
    Synthetic attn_v = make_quant(backend, N_KV * HEAD_DIM, N_EMBD, DType::Q4_G64);
    Synthetic attn_out_w = make_quant(backend, N_EMBD, N_HEAD * HEAD_DIM, DType::Q4_G64);
    Synthetic q_norm = make_f32(backend, {HEAD_DIM});
    Synthetic k_norm = make_f32(backend, {HEAD_DIM});
    Synthetic ffn_gate_w = make_quant(backend, N_FFN, N_EMBD, DType::Q4_G64);
    Synthetic ffn_up_w = make_quant(backend, N_FFN, N_EMBD, DType::Q4_G64);
    Synthetic ffn_down_w = make_quant(backend, N_EMBD, N_FFN, DType::Q4_G64);
    Synthetic norm_w = make_f32(backend, {N_EMBD});
    // The head weight doubles as the embedding table, as in the artifact both
    // are VOCAB x N_EMBD Q8; embedding reads one row, the head streams all.
    Synthetic head = make_quant(backend, VOCAB, N_EMBD, DType::Q8_G128);

    const uint64_t gdn_bytes = tensor_bytes(gdn_qkv.tensor) + tensor_bytes(gdn_gate.tensor) +
                               tensor_bytes(gdn_out.tensor) + tensor_bytes(ssm_alpha.tensor) +
                               tensor_bytes(ssm_beta.tensor);
    const uint64_t attn_bytes = tensor_bytes(attn_q.tensor) + tensor_bytes(attn_k.tensor) +
                                tensor_bytes(attn_v.tensor) + tensor_bytes(attn_out_w.tensor);
    const uint64_t ffn_bytes = tensor_bytes(ffn_gate_w.tensor) + tensor_bytes(ffn_up_w.tensor) +
                               tensor_bytes(ffn_down_w.tensor);
    const double token_weight_bytes = 48.0 * gdn_bytes + 16.0 * attn_bytes +
                                      64.0 * ffn_bytes + tensor_bytes(head.tensor);
    printf("weight stream per token: %.2f GiB\n", token_weight_bytes / (1024.0 * 1024.0 * 1024.0));

    // Activation/state buffers mirroring MetalEngine.
    auto alloc_f32 = [&](uint64_t count) { return backend.allocate(count * sizeof(float)); };
    auto h = alloc_f32(N_EMBD), x1 = alloc_f32(N_EMBD), y = alloc_f32(N_EMBD);
    auto qg = alloc_f32(2 * N_HEAD * HEAD_DIM), kbuf = alloc_f32(N_KV * HEAD_DIM);
    auto vbuf = alloc_f32(N_KV * HEAD_DIM), attn_out = alloc_f32(N_HEAD * HEAD_DIM);
    auto attn_scratch = alloc_f32((uint64_t)N_HEAD * (seq + 1));
    auto qkv = alloc_f32(GDN_CH), z = alloc_f32(GDN_V), alpha = alloc_f32(GDN_HEADS);
    auto beta_raw = alloc_f32(GDN_HEADS), g = alloc_f32(GDN_HEADS), beta = alloc_f32(GDN_HEADS);
    auto conv_out = alloc_f32(GDN_CH), delta_out = alloc_f32(GDN_V), gated_out = alloc_f32(GDN_V);
    auto ffn_gate = alloc_f32(N_FFN), ffn_up = alloc_f32(N_FFN);
    auto logits = alloc_f32(VOCAB);
    auto token_out = backend.allocate(sizeof(uint32_t));
    auto recurrent = alloc_f32((uint64_t)GDN_HEADS * GDN_DIM * GDN_DIM);
    auto ring = alloc_f32((uint64_t)3 * GDN_CH);
    auto k_cache = backend.allocate((uint64_t)(seq + 1) * N_KV * HEAD_DIM * 2);
    auto v_cache = backend.allocate((uint64_t)(seq + 1) * N_KV * HEAD_DIM * 2);
    backend.zero(*recurrent); backend.zero(*ring);
    backend.zero(*k_cache); backend.zero(*v_cache);
    q27::BackendQuantized q5120 = backend.allocate_quantized(N_EMBD);
    q27::BackendQuantized q6144 = backend.allocate_quantized(GDN_V);
    q27::BackendQuantized q17408 = backend.allocate_quantized(N_FFN);

    {
        std::vector<float> x(N_EMBD);
        for (uint32_t i = 0; i < N_EMBD; i++) x[i] = (float)((i % 19) - 9) / 9.0f;
        backend.write(*h, 0, x.data(), x.size() * sizeof(float));
    }

    const float scale = 1.0f / 16.0f; // rsqrt(HEAD_DIM)
    const uint32_t position = seq - 1;

    auto gdn_block = [&]() {
        backend.matvec_quantized_pair(gdn_qkv.tensor, *qkv, gdn_gate.tensor, *z, q5120);
        backend.matvec_pair(ssm_alpha.tensor, *alpha, ssm_beta.tensor, *beta_raw, *x1);
        backend.gdn_gates(*alpha, *beta_raw, ssm_a.tensor, ssm_dt.tensor, *g, *beta, GDN_HEADS);
        backend.conv_step(*ring, *ring, *qkv, ssm_conv.tensor, *conv_out, GDN_CH);
        backend.l2norm_heads(*conv_out, 2 * GDN_QK_HEADS, GDN_DIM, EPS);
        backend.delta_step(*recurrent, *recurrent, *conv_out, *g, *beta, *delta_out,
                           GDN_HEADS, GDN_QK_HEADS, GDN_DIM);
        backend.gated_norm_gdn(*delta_out, ssm_norm.tensor, *z, *gated_out, GDN_HEADS, GDN_DIM, EPS);
        backend.quantize(*gated_out, q6144);
        backend.matvec_quantized(gdn_out.tensor, q6144, *y);
    };
    auto attention_block = [&]() {
        backend.matvec_quantized(attn_q.tensor, q5120, *qg);
        backend.rmsnorm_heads(*qg, q_norm.tensor, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS);
        backend.matvec_quantized_pair(attn_k.tensor, *kbuf, attn_v.tensor, *vbuf, q5120);
        backend.rmsnorm_heads(*kbuf, k_norm.tensor, N_KV, HEAD_DIM, HEAD_DIM, EPS);
        backend.rope_neox(*qg, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, position, FREQ_BASE);
        backend.rope_neox(*kbuf, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, position, FREQ_BASE);
        backend.kv_store_f16(*kbuf, *vbuf, *k_cache, *v_cache, position, N_KV * HEAD_DIM);
        backend.attention_f16(*qg, 2 * HEAD_DIM, *k_cache, *v_cache, *attn_scratch, *attn_out,
                              position + 1, N_HEAD, N_KV, HEAD_DIM, scale);
        backend.sigmoid_gate_mul(*attn_out, *qg, N_HEAD, HEAD_DIM);
        backend.quantize(*attn_out, q6144);
        backend.matvec_quantized(attn_out_w.tensor, q6144, *y);
    };
    auto ffn_block = [&]() {
        backend.matvec_quantized_pair(ffn_gate_w.tensor, *ffn_gate, ffn_up_w.tensor, *ffn_up, q5120);
        backend.silu_mul(*ffn_gate, *ffn_up, *ffn_gate, N_FFN);
        backend.quantize(*ffn_gate, q17408);
        backend.matvec_quantized(ffn_down_w.tensor, q17408, *y);
    };
    auto token_step = [&](uint32_t token) {
        backend.begin_commands();
        backend.embedding_q8(head.tensor, token % VOCAB, *h);
        for (uint32_t layer = 0; layer < N_LAYER; layer++) {
            backend.rmsnorm_quantized(*h, norm_w.tensor, *x1, N_EMBD, EPS, q5120);
            if (attention_layer(layer)) attention_block(); else gdn_block();
            backend.add_inplace(*h, *y, N_EMBD);
            backend.rmsnorm_quantized(*h, norm_w.tensor, *x1, N_EMBD, EPS, q5120);
            ffn_block();
            backend.add_inplace(*h, *y, N_EMBD);
        }
        backend.rmsnorm_quantized(*h, norm_w.tensor, *x1, N_EMBD, EPS, q5120);
        backend.matvec_quantized(head.tensor, q5120, *logits);
        backend.argmax(*logits, VOCAB, *token_out);
        backend.end_commands();
        uint32_t next = 0;
        backend.read(*token_out, 0, &next, sizeof(next)); // per-token CPU sync, as decode does
        return next;
    };

    token_step(1); token_step(2); // warmup: clock ramp + first-touch paging
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < tokens; i++) token_step(i + 3);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    printf("decode step: %.1f ms/token, %.2f tok/s, effective weight stream %.1f GB/s\n",
           seconds / tokens * 1e3, tokens / seconds,
           token_weight_bytes * tokens / seconds / 1e9);
    return 0;
}
