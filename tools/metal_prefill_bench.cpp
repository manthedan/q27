// Synthetic chunked-prefill benchmark for the Metal backend.
//
// Replays the exact per-chunk dispatch sequence of MetalEngine::chunk_forward
// — batched embedding, fused row RMSNorm+quantization, simdgroup projection
// GEMMs, chunk conv/DeltaNet, causal chunk attention, chunked FFN — against
// synthetic resident weights at the production shapes, one command buffer per
// chunk, exactly as prompt ingestion schedules it. Because the weights are
// resident (no 17 GiB mmap), this measures the compute/bandwidth-limited
// prefill ceiling; run with Q27_METAL_PROFILE=1 for the per-kernel share
// table. This is the attribution tool for the critical-path prefill item:
// the ~15 GB/s effective chunk bandwidth splits into per-kernel shares here.
//
// Memory-safe: synthetic buffers only (~1.6 GiB peak), no model artifact.

#include "metal_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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
constexpr uint32_t CHUNK_MAX = 12;
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
    // Fill the Metal buffer through a small staging slice instead of a full
    // host copy, so even the 1.3 GiB embedding table never exists twice.
    Synthetic s;
    const uint64_t group = dtype == DType::Q8_G128 ? 128 : 64;
    const uint64_t data_bytes = (uint64_t)rows * cols / (dtype == DType::Q4_G64 ? 2 : 1);
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
    s.data.clear(); s.data.shrink_to_fit();
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
    s.data.clear(); s.data.shrink_to_fit();
    return s;
}

q27::BackendQuantized quantized_view(const q27::BackendQuantized& full, uint32_t count) {
    q27::BackendQuantized view;
    view.count = count; view.values = full.values; view.scales = full.scales;
    return view;
}

} // namespace

int main(int argc, char** argv) {
    uint32_t prompt = 120, chunk_size = CHUNK_MAX;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) prompt = (uint32_t)atoi(argv[++i]);
        else if (arg == "--chunk" && i + 1 < argc) chunk_size = (uint32_t)atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--prompt N] [--chunk 2..12]\n", argv[0]); return 1; }
    }
    if (!prompt || chunk_size < 2 || chunk_size > CHUNK_MAX) {
        fprintf(stderr, "invalid --prompt/--chunk\n");
        return 1;
    }

    q27::MetalBackend backend;
    if (!backend.supports_quantized_matmul()) {
        fprintf(stderr, "device lacks simdgroup matmul support\n");
        return 1;
    }
    printf("backend: %s, %u-token synthetic prompt in chunks of %u\n",
           backend.name().c_str(), prompt, chunk_size);

    // One synthetic weight set per layer type, as in metal_decode_bench: a
    // chunk streams far more bytes than any cache level, so reuse across the
    // 48/16/64 layer repeats is bandwidth-equivalent to distinct tensors.
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
    Synthetic embed = make_quant(backend, VOCAB, N_EMBD, DType::Q8_G128);

    const uint64_t gdn_bytes = tensor_bytes(gdn_qkv.tensor) + tensor_bytes(gdn_gate.tensor) +
                               tensor_bytes(gdn_out.tensor) + tensor_bytes(ssm_alpha.tensor) +
                               tensor_bytes(ssm_beta.tensor);
    const uint64_t attn_bytes = tensor_bytes(attn_q.tensor) + tensor_bytes(attn_k.tensor) +
                                tensor_bytes(attn_v.tensor) + tensor_bytes(attn_out_w.tensor);
    const uint64_t ffn_bytes = tensor_bytes(ffn_gate_w.tensor) + tensor_bytes(ffn_up_w.tensor) +
                               tensor_bytes(ffn_down_w.tensor);
    const double chunk_weight_bytes = 48.0 * gdn_bytes + 16.0 * attn_bytes + 64.0 * ffn_bytes;
    printf("weight stream per chunk: %.2f GiB\n",
           chunk_weight_bytes / (1024.0 * 1024.0 * 1024.0));

    // Chunk-capacity activation/state buffers mirroring MetalEngine.
    const uint32_t ctx = prompt + 1;
    auto alloc_f32 = [&](uint64_t count) { return backend.allocate(count * sizeof(float)); };
    auto ch = alloc_f32((uint64_t)CHUNK_MAX * N_EMBD);
    auto cx1 = alloc_f32((uint64_t)CHUNK_MAX * N_EMBD);
    auto cy = alloc_f32((uint64_t)CHUNK_MAX * N_EMBD);
    auto cqg = alloc_f32((uint64_t)CHUNK_MAX * 2 * N_HEAD * HEAD_DIM);
    auto ckbuf = alloc_f32((uint64_t)CHUNK_MAX * N_KV * HEAD_DIM);
    auto cvbuf = alloc_f32((uint64_t)CHUNK_MAX * N_KV * HEAD_DIM);
    auto cattn_out = alloc_f32((uint64_t)CHUNK_MAX * N_HEAD * HEAD_DIM);
    auto cattn_scratch = alloc_f32((uint64_t)CHUNK_MAX * N_HEAD * ctx);
    auto cqkv = alloc_f32((uint64_t)CHUNK_MAX * GDN_CH);
    auto cz = alloc_f32((uint64_t)CHUNK_MAX * GDN_V);
    auto calpha = alloc_f32((uint64_t)CHUNK_MAX * GDN_HEADS);
    auto cbeta_raw = alloc_f32((uint64_t)CHUNK_MAX * GDN_HEADS);
    auto cg = alloc_f32((uint64_t)CHUNK_MAX * GDN_HEADS);
    auto cbeta = alloc_f32((uint64_t)CHUNK_MAX * GDN_HEADS);
    auto cconv_out = alloc_f32((uint64_t)CHUNK_MAX * GDN_CH);
    auto cdelta_out = alloc_f32((uint64_t)CHUNK_MAX * GDN_V);
    auto cgated_out = alloc_f32((uint64_t)CHUNK_MAX * GDN_V);
    auto cffn_gate = alloc_f32((uint64_t)CHUNK_MAX * N_FFN);
    auto cffn_up = alloc_f32((uint64_t)CHUNK_MAX * N_FFN);
    q27::BackendQuantized cq5120 = backend.allocate_quantized(CHUNK_MAX * N_EMBD);
    q27::BackendQuantized cq6144 = backend.allocate_quantized(CHUNK_MAX * GDN_V);
    q27::BackendQuantized cq17408 = backend.allocate_quantized(CHUNK_MAX * N_FFN);
    auto recurrent = alloc_f32((uint64_t)GDN_HEADS * GDN_DIM * GDN_DIM);
    auto ring = alloc_f32((uint64_t)3 * GDN_CH);
    auto k_cache = backend.allocate((uint64_t)ctx * N_KV * HEAD_DIM * 2);
    auto v_cache = backend.allocate((uint64_t)ctx * N_KV * HEAD_DIM * 2);
    backend.zero(*recurrent); backend.zero(*ring);
    backend.zero(*k_cache); backend.zero(*v_cache);

    const float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    uint32_t position = 0;

    auto gdn_chunk = [&](uint32_t count) {
        q27::BackendQuantized x5 = quantized_view(cq5120, count * N_EMBD);
        backend.matmul_quantized(gdn_qkv.tensor, x5, count, *cqkv);
        backend.matmul_quantized(gdn_gate.tensor, x5, count, *cz);
        backend.matvec_f16_pair_rows(ssm_alpha.tensor, *calpha, ssm_beta.tensor, *cbeta_raw,
                                     *cx1, count);
        backend.gdn_gates_rows(*calpha, *cbeta_raw, ssm_a.tensor, ssm_dt.tensor, *cg, *cbeta,
                               GDN_HEADS, count);
        backend.conv_chunk(*ring, *cqkv, ssm_conv.tensor, *cconv_out, GDN_CH, count);
        backend.l2norm_rows(*cconv_out, 2 * GDN_QK_HEADS, GDN_DIM, GDN_CH, count, EPS);
        backend.delta_chunk(*recurrent, *cconv_out, *cg, *cbeta, *cdelta_out,
                            GDN_HEADS, GDN_QK_HEADS, GDN_DIM, count);
        backend.gated_norm_gdn(*cdelta_out, ssm_norm.tensor, *cz, *cgated_out,
                               count * GDN_HEADS, GDN_DIM, EPS);
        q27::BackendQuantized x6 = quantized_view(cq6144, count * GDN_V);
        backend.quantize(*cgated_out, x6);
        backend.matmul_quantized(gdn_out.tensor, x6, count, *cy);
    };
    auto attention_chunk = [&](uint32_t count) {
        q27::BackendQuantized x5 = quantized_view(cq5120, count * N_EMBD);
        backend.matmul_quantized(attn_q.tensor, x5, count, *cqg);
        backend.rmsnorm_heads(*cqg, q_norm.tensor, count * N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS);
        backend.matmul_quantized(attn_k.tensor, x5, count, *ckbuf);
        backend.matmul_quantized(attn_v.tensor, x5, count, *cvbuf);
        backend.rmsnorm_heads(*ckbuf, k_norm.tensor, count * N_KV, HEAD_DIM, HEAD_DIM, EPS);
        backend.rope_neox_rows(*cqg, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM,
                               2 * N_HEAD * HEAD_DIM, position, count, FREQ_BASE);
        backend.rope_neox_rows(*ckbuf, N_KV, HEAD_DIM, N_ROT, HEAD_DIM,
                               N_KV * HEAD_DIM, position, count, FREQ_BASE);
        backend.kv_store_f16_rows(*ckbuf, *cvbuf, *k_cache, *v_cache, position,
                                  N_KV * HEAD_DIM, count);
        backend.attention_f16_causal(*cqg, 2 * HEAD_DIM, 2 * N_HEAD * HEAD_DIM,
                                     *k_cache, *v_cache, *cattn_scratch, *cattn_out,
                                     position + 1, N_HEAD, N_KV, HEAD_DIM, count, scale);
        backend.sigmoid_gate_mul_rows(*cattn_out, *cqg, N_HEAD, HEAD_DIM, count);
        q27::BackendQuantized x6 = quantized_view(cq6144, count * N_HEAD * HEAD_DIM);
        backend.quantize(*cattn_out, x6);
        backend.matmul_quantized(attn_out_w.tensor, x6, count, *cy);
    };
    auto ffn_chunk = [&](uint32_t count) {
        q27::BackendQuantized x5 = quantized_view(cq5120, count * N_EMBD);
        backend.matmul_quantized(ffn_gate_w.tensor, x5, count, *cffn_gate);
        backend.matmul_quantized(ffn_up_w.tensor, x5, count, *cffn_up);
        backend.silu_mul(*cffn_gate, *cffn_up, *cffn_gate, count * N_FFN);
        q27::BackendQuantized x17 = quantized_view(cq17408, count * N_FFN);
        backend.quantize(*cffn_gate, x17);
        backend.matmul_quantized(ffn_down_w.tensor, x17, count, *cy);
    };
    auto chunk_step = [&](const uint32_t* tokens, uint32_t count) {
        backend.begin_commands();
        backend.embedding_q8_rows(embed.tensor, tokens, count, *ch);
        q27::BackendQuantized x5 = quantized_view(cq5120, count * N_EMBD);
        for (uint32_t layer = 0; layer < N_LAYER; layer++) {
            backend.rmsnorm_rows_quantized(*ch, norm_w.tensor, *cx1, N_EMBD, count, EPS, x5);
            if (attention_layer(layer)) attention_chunk(count); else gdn_chunk(count);
            backend.add_inplace(*ch, *cy, count * N_EMBD);
            backend.rmsnorm_rows_quantized(*ch, norm_w.tensor, *cx1, N_EMBD, count, EPS, x5);
            ffn_chunk(count);
            backend.add_inplace(*ch, *cy, count * N_EMBD);
        }
        backend.end_commands();
        position += count;
    };

    std::vector<uint32_t> tokens(prompt);
    for (uint32_t i = 0; i < prompt; i++) tokens[i] = (i * 2654435761u) % VOCAB;

    // Warmup: clock ramp plus first-touch paging of the synthetic weights,
    // then restart the sequence so the measured run ingests the whole prompt.
    chunk_step(tokens.data(), chunk_size);
    backend.synchronize();
    position = 0;

    const auto start = std::chrono::steady_clock::now();
    uint32_t chunks = 0;
    for (uint32_t begin = 0; begin < prompt; begin += chunk_size, chunks++) {
        const uint32_t count = std::min(chunk_size, prompt - begin);
        chunk_step(tokens.data() + begin, count);
    }
    backend.synchronize();
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    printf("prefill: %.1f ms/chunk, %.2f tok/s, effective weight stream %.1f GB/s\n",
           seconds / chunks * 1e3, prompt / seconds,
           chunk_weight_bytes * chunks / seconds / 1e9);
    return 0;
}
