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

#include <algorithm>
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
    if (t.dtype == DType::T2_G128) data /= 4;
    if (t.dtype == DType::B1_G128) data /= 8;
    const uint64_t group = t.dtype == DType::Q4_G64 ? 64 :
        (t.dtype == DType::Q8_G128 || t.dtype == DType::T2_G128 ||
         t.dtype == DType::B1_G128) ? 128 : 0;
    return data + (group ? t.rows * (t.cols / group) * 2 : 0);
}

Synthetic make_quant(q27::MetalBackend& backend, uint32_t rows, uint32_t cols, DType dtype) {
    // Fill the Metal buffer through a small staging slice instead of a full
    // host copy, so even the 1.3 GiB head tensor never exists twice.
    Synthetic s;
    const uint64_t group = dtype == DType::Q4_G64 ? 64 : 128;
    const uint64_t data_bytes = (uint64_t)rows * cols /
        (dtype == DType::Q4_G64 ? 2 : dtype == DType::T2_G128 ? 4
                                    : dtype == DType::B1_G128 ? 8 : 1);
    s.tensor.dtype = dtype;
    s.tensor.rows = rows;
    s.tensor.cols = cols;
    s.tensor.data = backend.allocate(data_bytes);
    std::vector<uint8_t> slice(std::min<uint64_t>(data_bytes, 64ull << 20));
    for (uint64_t offset = 0; offset < data_bytes; offset += slice.size()) {
        const uint64_t chunk = std::min<uint64_t>(slice.size(), data_bytes - offset);
        for (uint64_t i = 0; i < chunk; i++) slice[i] = (uint8_t)((offset + i) * 2654435761u >> 24);
        backend.write(*s.tensor.data, offset, slice.data(), chunk);
    }
    s.scales.assign((uint64_t)rows * (cols / group), 0x3c00 /* f16 1.0 */);
    s.tensor.scales = backend.allocate(s.scales.size() * sizeof(uint16_t));
    backend.write(*s.tensor.scales, 0, s.scales.data(), s.scales.size() * sizeof(uint16_t));
    s.scales.clear(); s.scales.shrink_to_fit();
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
    // upload() copied the bytes into a Metal buffer; drop the host copy so the
    // tool's footprint stays at one copy of the ~1.4 GiB weight set.
    s.data.clear(); s.data.shrink_to_fit();
    s.scales.clear(); s.scales.shrink_to_fit();
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
    // upload() copied the bytes into a Metal buffer; drop the host copy so the
    // tool's footprint stays at one copy of the ~1.4 GiB weight set.
    s.data.clear(); s.data.shrink_to_fit();
    s.scales.clear(); s.scales.shrink_to_fit();
    return s;
}

} // namespace

int main(int argc, char** argv) {
    uint32_t tokens = 16, seq = 128;
    bool turbo3 = false, t2 = false, b1 = false;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--tokens" && i + 1 < argc) tokens = (uint32_t)atoi(argv[++i]);
        else if (arg == "--seq" && i + 1 < argc) seq = (uint32_t)atoi(argv[++i]);
        else if (arg == "--kv" && i + 1 < argc) {
            const std::string kv = argv[++i];
            if (kv == "turbo3") turbo3 = true;
            else if (kv != "fp16") { fprintf(stderr, "invalid --kv (fp16|turbo3)\n"); return 1; }
        }
        else if (arg == "--dtype" && i + 1 < argc) {
            const std::string d = argv[++i];
            if (d == "t2") t2 = true;
            else if (d == "b1") b1 = true;
            else if (d != "q4q8") { fprintf(stderr, "invalid --dtype (q4q8|t2|b1)\n"); return 1; }
        }
        else { fprintf(stderr, "usage: %s [--tokens N] [--seq LEN] [--kv fp16|turbo3] [--dtype q4q8|t2|b1]\n", argv[0]); return 1; }
    }
    if (!tokens || !seq) { fprintf(stderr, "invalid --tokens/--seq\n"); return 1; }

    q27::MetalBackend backend;
    printf("backend: %s, %u simulated tokens at context %u, %s KV%s\n",
           backend.name().c_str(), tokens, seq, turbo3 ? "turbo3" : "fp16",
           t2 ? ", ternary weights" : b1 ? ", binary weights" : "");

    // --dtype t2 replays the ternary-tier decode: every projection and the
    // head/embedding table T2, projections dispatched through the
    // float-activation GEMV exactly as MetalEngine::project routes them.
    // ssm_alpha/beta stay F16 in the q4q8 mix and T2 in the ternary mix, as
    // the artifacts pack them.
    // --dtype b1 mirrors t2 with the binary dtype end-to-end (alpha/beta
    // included, exactly as the bonsai-b1-v1 pack routes).
    const DType bulk = t2 ? DType::T2_G128 : b1 ? DType::B1_G128 : DType::Q4_G64;
    const DType vocab_dtype = t2 ? DType::T2_G128 : b1 ? DType::B1_G128 : DType::Q8_G128;

    // One synthetic weight set per layer type; a block streams far more bytes
    // than any cache level, so reuse across the 48/16/64 repeats is
    // bandwidth-equivalent to distinct per-layer tensors.
    Synthetic gdn_qkv = make_quant(backend, GDN_CH, N_EMBD, bulk);
    Synthetic gdn_gate = make_quant(backend, GDN_V, N_EMBD, bulk);
    Synthetic gdn_out = make_quant(backend, N_EMBD, GDN_V, bulk);
    Synthetic ssm_alpha = (t2 || b1) ? make_quant(backend, GDN_HEADS, N_EMBD, bulk)
                                     : make_f16(backend, GDN_HEADS, N_EMBD);
    Synthetic ssm_beta = (t2 || b1) ? make_quant(backend, GDN_HEADS, N_EMBD, bulk)
                                    : make_f16(backend, GDN_HEADS, N_EMBD);
    Synthetic ssm_a = make_f32(backend, {GDN_HEADS});
    Synthetic ssm_dt = make_f32(backend, {GDN_HEADS});
    Synthetic ssm_conv = make_f32(backend, {GDN_CH, 4});
    Synthetic ssm_norm = make_f32(backend, {GDN_DIM});
    Synthetic attn_q = make_quant(backend, 2 * N_HEAD * HEAD_DIM, N_EMBD, bulk);
    Synthetic attn_k = make_quant(backend, N_KV * HEAD_DIM, N_EMBD, bulk);
    Synthetic attn_v = make_quant(backend, N_KV * HEAD_DIM, N_EMBD, bulk);
    Synthetic attn_out_w = make_quant(backend, N_EMBD, N_HEAD * HEAD_DIM, bulk);
    Synthetic q_norm = make_f32(backend, {HEAD_DIM});
    Synthetic k_norm = make_f32(backend, {HEAD_DIM});
    Synthetic ffn_gate_w = make_quant(backend, N_FFN, N_EMBD, bulk);
    Synthetic ffn_up_w = make_quant(backend, N_FFN, N_EMBD, bulk);
    Synthetic ffn_down_w = make_quant(backend, N_EMBD, N_FFN, bulk);
    Synthetic norm_w = make_f32(backend, {N_EMBD});
    // The head weight doubles as the embedding table, as in the artifact both
    // are VOCAB x N_EMBD Q8; embedding reads one row, the head streams all.
    Synthetic head = make_quant(backend, VOCAB, N_EMBD, vocab_dtype);

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
    auto qkv = alloc_f32(GDN_CH), z = alloc_f32(GDN_V), alpha = alloc_f32(GDN_HEADS);
    auto beta_raw = alloc_f32(GDN_HEADS), g = alloc_f32(GDN_HEADS), beta = alloc_f32(GDN_HEADS);
    auto conv_out = alloc_f32(GDN_CH), delta_out = alloc_f32(GDN_V), gated_out = alloc_f32(GDN_V);
    auto ffn_gate = alloc_f32(N_FFN), ffn_up = alloc_f32(N_FFN);
    auto logits = alloc_f32(VOCAB);
    auto token_out = backend.allocate(sizeof(uint32_t));
    auto recurrent = alloc_f32((uint64_t)GDN_HEADS * GDN_DIM * GDN_DIM);
    auto ring = alloc_f32((uint64_t)3 * GDN_CH);
    const uint64_t cache_row_bytes = turbo3 ? (uint64_t)N_KV * 2 * 50
                                            : (uint64_t)N_KV * HEAD_DIM * 2;
    // Blocked-GQA partials scratch (caller-owned since audit E2), decode
    // width, sized for the deepest position this bench reaches.
    auto partials = backend.allocate_private(
        (uint64_t)N_HEAD * (1 + ((uint64_t)seq - 1) / 128) * 258 * 4);
    auto k_cache = backend.allocate((uint64_t)(seq + 1) * cache_row_bytes);
    auto v_cache = backend.allocate((uint64_t)(seq + 1) * cache_row_bytes);
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

    auto proj = [&](const q27::BackendTensor& w, q27::BackendBuffer& xf,
                    const q27::BackendQuantized& xq, q27::BackendBuffer& out) {
        if (w.dtype == DType::T2_G128 || w.dtype == DType::B1_G128) backend.matvec(w, xf, out);
        else backend.matvec_quantized(w, xq, out);
    };
    auto proj_pair = [&](const q27::BackendTensor& a, q27::BackendBuffer& a_out,
                         const q27::BackendTensor& b, q27::BackendBuffer& b_out,
                         q27::BackendBuffer& xf, const q27::BackendQuantized& xq) {
        if (a.dtype == DType::T2_G128 || a.dtype == DType::B1_G128) { proj(a, xf, xq, a_out); proj(b, xf, xq, b_out); }
        else backend.matvec_quantized_pair(a, a_out, b, b_out, xq);
    };
    auto gdn_block = [&]() {
        proj_pair(gdn_qkv.tensor, *qkv, gdn_gate.tensor, *z, *x1, q5120);
        backend.matvec_pair(ssm_alpha.tensor, *alpha, ssm_beta.tensor, *beta_raw, *x1);
        backend.gdn_gates(*alpha, *beta_raw, ssm_a.tensor, ssm_dt.tensor, *g, *beta, GDN_HEADS);
        backend.conv_step(*ring, *ring, *qkv, ssm_conv.tensor, *conv_out, GDN_CH);
        backend.l2norm_heads(*conv_out, 2 * GDN_QK_HEADS, GDN_DIM, EPS);
        backend.delta_step(*recurrent, *recurrent, *conv_out, *g, *beta, *delta_out,
                           GDN_HEADS, GDN_QK_HEADS, GDN_DIM);
        backend.gated_norm_gdn(*delta_out, ssm_norm.tensor, *z, *gated_out, GDN_HEADS, GDN_DIM, EPS);
        if (!t2 && !b1) backend.quantize(*gated_out, q6144);
        proj(gdn_out.tensor, *gated_out, q6144, *y);
    };
    auto attention_block = [&]() {
        proj(attn_q.tensor, *x1, q5120, *qg);
        backend.rmsnorm_heads(*qg, q_norm.tensor, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS);
        proj_pair(attn_k.tensor, *kbuf, attn_v.tensor, *vbuf, *x1, q5120);
        backend.rmsnorm_heads(*kbuf, k_norm.tensor, N_KV, HEAD_DIM, HEAD_DIM, EPS);
        backend.rope_neox(*qg, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, position, FREQ_BASE);
        backend.rope_neox(*kbuf, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, position, FREQ_BASE);
        if (turbo3) {
            backend.turbo_wht(*qg, N_HEAD, 2 * HEAD_DIM, false);
            backend.kv_store_turbo3(*kbuf, *vbuf, *k_cache, *v_cache, position, N_KV);
            backend.attention_turbo3(*qg, 2 * HEAD_DIM, *k_cache, *v_cache,
                                     *attn_out, position + 1, N_HEAD, N_KV, HEAD_DIM, scale,
                                     partials.get());
            backend.turbo_wht(*attn_out, N_HEAD, HEAD_DIM, true);
        } else {
            backend.kv_store_f16(*kbuf, *vbuf, *k_cache, *v_cache, position, N_KV * HEAD_DIM);
            backend.attention_f16(*qg, 2 * HEAD_DIM, *k_cache, *v_cache, *attn_out,
                                  position + 1, N_HEAD, N_KV, HEAD_DIM, scale, partials.get());
        }
        backend.sigmoid_gate_mul(*attn_out, *qg, N_HEAD, HEAD_DIM);
        if (!t2 && !b1) backend.quantize(*attn_out, q6144);
        proj(attn_out_w.tensor, *attn_out, q6144, *y);
    };
    auto ffn_block = [&]() {
        proj_pair(ffn_gate_w.tensor, *ffn_gate, ffn_up_w.tensor, *ffn_up, *x1, q5120);
        backend.silu_mul(*ffn_gate, *ffn_up, *ffn_gate, N_FFN);
        if (!t2 && !b1) backend.quantize(*ffn_gate, q17408);
        proj(ffn_down_w.tensor, *ffn_gate, q17408, *y);
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
        proj(head.tensor, *x1, q5120, *logits);
        backend.argmax(*logits, VOCAB, *token_out);
        backend.end_commands();
        uint32_t next = 0;
        backend.read(*token_out, 0, &next, sizeof(next)); // per-token CPU sync, as decode does
        return next;
    };

    token_step(1); token_step(2); // warmup: clock ramp + first-touch paging
    backend.profile_reset();      // keep the cold dispatches out of the attribution
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < tokens; i++) token_step(i + 3);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    printf("decode step: %.1f ms/token, %.2f tok/s, effective weight stream %.1f GB/s\n",
           seconds / tokens * 1e3, tokens / seconds,
           token_weight_bytes * tokens / seconds / 1e9);
    return 0;
}
