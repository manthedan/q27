#include "metal_engine.h"

#include "../../third_party/json.hpp"
#include "../suffixdraft.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace q27 {
namespace {

bool is_matrix_dtype(DType dtype) {
    return dtype == DType::Q4_G64 || dtype == DType::Q8_G128;
}

class CommandBatch {
  public:
    explicit CommandBatch(MetalBackend& backend) : backend_(backend) { backend_.begin_commands(); }
    ~CommandBatch() { if (active_) backend_.abort_commands(); }
    void finish() { backend_.end_commands(); active_ = false; }
  private:
    MetalBackend& backend_;
    bool active_ = true;
};

// A shorter view over a chunk-capacity quantized activation, so partial
// chunks quantize and multiply exactly `count` values without reallocating.
BackendQuantized quantized_view(const BackendQuantized& full, uint32_t count) {
    if (count > full.count) throw std::runtime_error("q27 Metal: quantized view exceeds capacity");
    BackendQuantized view;
    view.count = count; view.values = full.values; view.scales = full.scales;
    return view;
}

} // namespace

struct MetalEngine::Snapshot {
    struct StoredLayer {
        std::shared_ptr<BackendBuffer> recurrent, ring, k_cache, v_cache;
    };
    const MetalEngine* owner = nullptr;
    uint32_t position = 0;
    std::vector<StoredLayer> layers;
    std::shared_ptr<BackendBuffer> mtp_k_cache, mtp_v_cache, hidden, logits;
};

std::shared_ptr<BackendBuffer> MetalEngine::alloc_f32(uint64_t count) {
    return backend_.allocate(count * sizeof(float));
}

const BackendTensor& MetalEngine::weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) throw std::runtime_error("q27 Metal: weight is not mapped: " + name);
    return it->second;
}

const BackendTensor& MetalEngine::layer_weight(uint32_t layer, const char* leaf) const {
    return weight("blk." + std::to_string(layer) + "." + leaf);
}

void MetalEngine::validate_architecture() const {
    nlohmann::json meta;
    try { meta = nlohmann::json::parse(model_.meta_json); }
    catch (const std::exception& e) { throw std::runtime_error(std::string("q27 Metal: invalid metadata JSON: ") + e.what()); }

    auto exact = [&](const char* key, uint64_t expected) {
        if (!meta.contains(key) || !meta[key].is_number_unsigned() || meta[key].get<uint64_t>() != expected)
            throw std::runtime_error(std::string("q27 Metal: architecture mismatch: ") + key);
    };
    if (meta.value("general.architecture", std::string()) != "qwen35")
        throw std::runtime_error("q27 Metal: expected qwen35 architecture");
    exact("qwen35.block_count", 65); exact("qwen35.embedding_length", N_EMBD);
    exact("qwen35.feed_forward_length", N_FFN); exact("qwen35.attention.head_count", N_HEAD);
    exact("qwen35.attention.head_count_kv", N_KV); exact("qwen35.attention.key_length", HEAD_DIM);
    exact("qwen35.attention.value_length", HEAD_DIM); exact("qwen35.ssm.state_size", GDN_DIM);
    exact("qwen35.ssm.group_count", GDN_QK_HEADS); exact("qwen35.ssm.inner_size", GDN_V);
    exact("qwen35.context_length", 262144); exact("qwen35.rope.dimension_count", N_ROT);
    exact("qwen35.ssm.conv_kernel", 4); exact("qwen35.ssm.time_step_rank", GDN_HEADS);
    exact("qwen35.full_attention_interval", 4); exact("qwen35.nextn_predict_layers", 1);
    exact("group_q4", 64); exact("group_q8", 128);
    if (meta.value("nibble_order", std::string()) != "even=low")
        throw std::runtime_error("q27 Metal: incompatible Q4 nibble order");
    auto exact_float = [&](const char* key, double expected, double tolerance) {
        if (!meta.contains(key) || !meta[key].is_number() ||
            std::fabs(meta[key].get<double>() - expected) > tolerance)
            throw std::runtime_error(std::string("q27 Metal: architecture mismatch: ") + key);
    };
    exact_float("qwen35.rope.freq_base", FREQ_BASE, 0.0);
    exact_float("qwen35.attention.layer_norm_rms_epsilon", EPS, 1e-12);
    const std::vector<uint32_t> rope_sections{11,11,10,0};
    if (!meta.contains("qwen35.rope.dimension_sections") ||
        meta["qwen35.rope.dimension_sections"].get<std::vector<uint32_t>>() != rope_sections)
        throw std::runtime_error("q27 Metal: architecture mismatch: rope dimension sections");

    std::vector<uint32_t> expected_attention;
    for (uint32_t i = 3; i < N_LAYER; i += 4) expected_attention.push_back(i);
    if (!meta.contains("attn_layers") || meta["attn_layers"].size() != expected_attention.size() + 1)
        throw std::runtime_error("q27 Metal: invalid attention layer map");
    for (size_t i = 0; i < expected_attention.size(); i++)
        if (meta["attn_layers"][i].get<uint32_t>() != expected_attention[i])
            throw std::runtime_error("q27 Metal: unexpected attention layer map");
    if (meta["attn_layers"].back().get<uint32_t>() != 64)
        throw std::runtime_error("q27 Metal: missing MTP attention layer");

    auto require = [&](const std::string& name, DType dtype, std::initializer_list<uint64_t> shape) {
        const Tensor* tensor = model_.find(name);
        if (!tensor || tensor->dtype != dtype || tensor->shape != std::vector<uint64_t>(shape))
            throw std::runtime_error("q27 Metal: required tensor mismatch: " + name);
    };
    auto matrix = [&](const std::string& name, uint64_t rows, uint64_t cols) {
        const Tensor* tensor = model_.find(name);
        if (!tensor || !is_matrix_dtype(tensor->dtype) || tensor->shape != std::vector<uint64_t>{rows, cols})
            throw std::runtime_error("q27 Metal: required matrix mismatch: " + name);
    };

    require("token_embd.weight", DType::Q8_G128, {VOCAB, N_EMBD});
    require("output.weight", DType::Q8_G128, {VOCAB, N_EMBD});
    require("output_norm.weight", DType::F32, {N_EMBD});
    for (uint32_t layer = 0; layer < N_LAYER; layer++) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        require(p + "attn_norm.weight", DType::F32, {N_EMBD});
        require(p + "post_attention_norm.weight", DType::F32, {N_EMBD});
        matrix(p + "ffn_gate.weight", N_FFN, N_EMBD);
        matrix(p + "ffn_up.weight", N_FFN, N_EMBD);
        matrix(p + "ffn_down.weight", N_EMBD, N_FFN);
        if (attention_layer(layer)) {
            matrix(p + "attn_q.weight", 2 * N_HEAD * HEAD_DIM, N_EMBD);
            matrix(p + "attn_k.weight", N_KV * HEAD_DIM, N_EMBD);
            matrix(p + "attn_v.weight", N_KV * HEAD_DIM, N_EMBD);
            matrix(p + "attn_output.weight", N_EMBD, N_HEAD * HEAD_DIM);
            require(p + "attn_q_norm.weight", DType::F32, {HEAD_DIM});
            require(p + "attn_k_norm.weight", DType::F32, {HEAD_DIM});
        } else {
            matrix(p + "attn_qkv.weight", GDN_CH, N_EMBD);
            matrix(p + "attn_gate.weight", GDN_V, N_EMBD);
            require(p + "ssm_alpha.weight", DType::F16, {GDN_HEADS, N_EMBD});
            require(p + "ssm_beta.weight", DType::F16, {GDN_HEADS, N_EMBD});
            require(p + "ssm_a", DType::F32, {GDN_HEADS});
            require(p + "ssm_dt.bias", DType::F32, {GDN_HEADS});
            require(p + "ssm_conv1d.weight", DType::F32, {GDN_CH, 4});
            require(p + "ssm_norm.weight", DType::F32, {GDN_DIM});
            matrix(p + "ssm_out.weight", N_EMBD, GDN_V);
        }
    }
    const std::string p = "blk.64.";
    require(p + "nextn.enorm.weight", DType::F32, {N_EMBD});
    require(p + "nextn.hnorm.weight", DType::F32, {N_EMBD});
    require(p + "nextn.shared_head_norm.weight", DType::F32, {N_EMBD});
    matrix(p + "nextn.eh_proj.weight", N_EMBD, 2 * N_EMBD);
    require(p + "attn_norm.weight", DType::F32, {N_EMBD});
    require(p + "post_attention_norm.weight", DType::F32, {N_EMBD});
    require(p + "attn_q_norm.weight", DType::F32, {HEAD_DIM});
    require(p + "attn_k_norm.weight", DType::F32, {HEAD_DIM});
    matrix(p + "attn_q.weight", 2 * N_HEAD * HEAD_DIM, N_EMBD);
    matrix(p + "attn_k.weight", N_KV * HEAD_DIM, N_EMBD);
    matrix(p + "attn_v.weight", N_KV * HEAD_DIM, N_EMBD);
    matrix(p + "attn_output.weight", N_EMBD, N_HEAD * HEAD_DIM);
    matrix(p + "ffn_gate.weight", N_FFN, N_EMBD);
    matrix(p + "ffn_up.weight", N_FFN, N_EMBD);
    matrix(p + "ffn_down.weight", N_EMBD, N_FFN);
    if (const Tensor* draft_head = model_.find("output_q4.weight"))
        if (draft_head->dtype != DType::Q4_G64 || draft_head->shape != std::vector<uint64_t>{VOCAB, N_EMBD})
            throw std::runtime_error("q27 Metal: output_q4.weight mismatch");
}

MetalEngine::MetalEngine(const std::string& model_path, uint32_t context, bool turbo3_kv)
    : model_(Model::open(model_path)), max_context_(context), turbo3_kv_(turbo3_kv) {
    if (!context || context > 262144) throw std::runtime_error("q27 Metal: context must be 1..262144");
    validate_architecture();
    const uint64_t cache_row_bytes = turbo3_kv_ ? (uint64_t)N_KV * 2 * 50
                                                : (uint64_t)N_KV * HEAD_DIM * 2;
    const uint64_t total_cache_bytes = 17ull * 2 * max_context_ * cache_row_bytes;
    if (total_cache_bytes > backend_.recommended_working_set_size() / 2)
        throw std::runtime_error("q27 Metal: requested KV cache is too large for this device; use --kv turbo3 or reduce --ctx");

    // All wrappers alias the mmap. No weight-sized copy is created.
    weights_.reserve(model_.tensors.size());
    for (const Tensor& tensor : model_.tensors)
        weights_.emplace(tensor.name, backend_.upload(model_, tensor));

    layers_.resize(N_LAYER);
    for (uint32_t layer = 0; layer < N_LAYER; layer++) {
        if (attention_layer(layer)) {
            const uint64_t cache_bytes = (uint64_t)max_context_ * cache_row_bytes;
            layers_[layer].k_cache = backend_.allocate(cache_bytes);
            layers_[layer].v_cache = backend_.allocate(cache_bytes);
        } else {
            layers_[layer].recurrent = alloc_f32((uint64_t)GDN_HEADS * GDN_DIM * GDN_DIM);
            layers_[layer].ring = alloc_f32((uint64_t)3 * GDN_CH);
        }
    }

    h_ = alloc_f32(N_EMBD); x1_ = alloc_f32(N_EMBD); y_ = alloc_f32(N_EMBD);
    qg_ = alloc_f32(2 * N_HEAD * HEAD_DIM); kbuf_ = alloc_f32(N_KV * HEAD_DIM);
    vbuf_ = alloc_f32(N_KV * HEAD_DIM); attn_out_ = alloc_f32(N_HEAD * HEAD_DIM);
    qkv_ = alloc_f32(GDN_CH); z_ = alloc_f32(GDN_V); alpha_ = alloc_f32(GDN_HEADS);
    beta_raw_ = alloc_f32(GDN_HEADS); g_ = alloc_f32(GDN_HEADS); beta_ = alloc_f32(GDN_HEADS);
    conv_out_ = alloc_f32(GDN_CH); delta_out_ = alloc_f32(GDN_V); gated_out_ = alloc_f32(GDN_V);
    ffn_gate_ = alloc_f32(N_FFN); ffn_up_ = alloc_f32(N_FFN);
    logits_ = alloc_f32(VOCAB); token_out_ = backend_.allocate(sizeof(uint32_t));
    mtp_embed_norm_ = alloc_f32(N_EMBD); mtp_hidden_norm_ = alloc_f32(N_EMBD);
    mtp_concat_ = alloc_f32(2 * N_EMBD); mtp_x_ = alloc_f32(N_EMBD);
    mtp_hidden_out_ = alloc_f32(N_EMBD);
    const uint64_t mtp_cache_bytes = (uint64_t)max_context_ * cache_row_bytes;
    mtp_k_cache_ = backend_.allocate(mtp_cache_bytes); mtp_v_cache_ = backend_.allocate(mtp_cache_bytes);
    q5120_=backend_.allocate_quantized(N_EMBD); q6144_=backend_.allocate_quantized(GDN_V);
    q10240_=backend_.allocate_quantized(GDN_CH); q17408_=backend_.allocate_quantized(N_FFN);

    // Layer-major chunked prefill routes projections through the simdgroup
    // GEMM, so it requires the same device family. The per-chunk activation
    // buffers total a few MiB. Every attention kernel is online-softmax now,
    // so no probability scratch exists on any path at any context length.
    chunked_prefill_ = backend_.supports_quantized_matmul();
    if (chunked_prefill_) {
        ch_ = alloc_f32((uint64_t)CHUNK_MAX * N_EMBD);
        cx1_ = alloc_f32((uint64_t)CHUNK_MAX * N_EMBD);
        cy_ = alloc_f32((uint64_t)CHUNK_MAX * N_EMBD);
        cqg_ = alloc_f32((uint64_t)CHUNK_MAX * 2 * N_HEAD * HEAD_DIM);
        ckbuf_ = alloc_f32((uint64_t)CHUNK_MAX * N_KV * HEAD_DIM);
        cvbuf_ = alloc_f32((uint64_t)CHUNK_MAX * N_KV * HEAD_DIM);
        cattn_out_ = alloc_f32((uint64_t)CHUNK_MAX * N_HEAD * HEAD_DIM);
        cqkv_ = alloc_f32((uint64_t)CHUNK_MAX * GDN_CH);
        cz_ = alloc_f32((uint64_t)CHUNK_MAX * GDN_V);
        calpha_ = alloc_f32((uint64_t)CHUNK_MAX * GDN_HEADS);
        cbeta_raw_ = alloc_f32((uint64_t)CHUNK_MAX * GDN_HEADS);
        cg_ = alloc_f32((uint64_t)CHUNK_MAX * GDN_HEADS);
        cbeta_ = alloc_f32((uint64_t)CHUNK_MAX * GDN_HEADS);
        cconv_out_ = alloc_f32((uint64_t)CHUNK_MAX * GDN_CH);
        cdelta_out_ = alloc_f32((uint64_t)CHUNK_MAX * GDN_V);
        cgated_out_ = alloc_f32((uint64_t)CHUNK_MAX * GDN_V);
        cffn_gate_ = alloc_f32((uint64_t)CHUNK_MAX * N_FFN);
        cffn_up_ = alloc_f32((uint64_t)CHUNK_MAX * N_FFN);
        cq5120_ = backend_.allocate_quantized(CHUNK_MAX * N_EMBD);
        cq6144_ = backend_.allocate_quantized(CHUNK_MAX * GDN_V);
        cq17408_ = backend_.allocate_quantized(CHUNK_MAX * N_FFN);
        cfinal_ = alloc_f32((uint64_t)CHUNK_MAX * N_EMBD);
        clogits_ = alloc_f32((uint64_t)CHUNK_MAX * VOCAB);
        cpred_ = backend_.allocate((uint64_t)CHUNK_MAX * sizeof(uint32_t));
        ctargets_ = backend_.allocate((uint64_t)CHUNK_MAX * sizeof(uint32_t));
        cnll_ = alloc_f32(CHUNK_MAX);
        // One recurrent + ring slot per GDN layer; both stay physically lazy
        // until the first batched MTP round touches them.
        ckpt_recurrent_ = alloc_f32((uint64_t)(N_LAYER - N_LAYER / 4) * GDN_HEADS * GDN_DIM * GDN_DIM);
        ckpt_ring_ = alloc_f32((uint64_t)(N_LAYER - N_LAYER / 4) * 3 * GDN_CH);
    }
    reset();
}

void MetalEngine::set_chunked_prefill(bool enabled) {
    if (enabled && !ch_)
        throw std::runtime_error("q27 Metal: chunked prefill requires quantized matmul support");
    chunked_prefill_ = enabled;
}

void MetalEngine::reset() {
    position_ = 0;
    for (LayerState& layer : layers_) {
        if (layer.recurrent) backend_.zero(*layer.recurrent);
        if (layer.ring) backend_.zero(*layer.ring);
        // KV rows are written before they become visible through position_;
        // clearing the full reserved context would make long-context reset O(ctx).
    }
}

std::shared_ptr<MetalEngine::Snapshot> MetalEngine::capture_state() {
    backend_.synchronize();
    auto snapshot = std::make_shared<Snapshot>();
    snapshot->owner = this; snapshot->position = position_; snapshot->layers.resize(N_LAYER);
    const uint64_t cache_row = turbo3_kv_ ? (uint64_t)N_KV * 2 * 50
                                          : (uint64_t)N_KV * HEAD_DIM * 2;
    const uint64_t active_cache = (uint64_t)position_ * cache_row;
    CommandBatch batch(backend_);
    for (uint32_t i=0;i<N_LAYER;i++) {
        const LayerState& source=layers_[i]; auto& dest=snapshot->layers[i];
        if(source.recurrent) { dest.recurrent=backend_.allocate(source.recurrent->size()); backend_.copy(*source.recurrent,0,*dest.recurrent,0,source.recurrent->size()); }
        if(source.ring) { dest.ring=backend_.allocate(source.ring->size()); backend_.copy(*source.ring,0,*dest.ring,0,source.ring->size()); }
        if(source.k_cache && active_cache) { dest.k_cache=backend_.allocate(active_cache); dest.v_cache=backend_.allocate(active_cache); backend_.copy(*source.k_cache,0,*dest.k_cache,0,active_cache); backend_.copy(*source.v_cache,0,*dest.v_cache,0,active_cache); }
    }
    if(active_cache) {
        snapshot->mtp_k_cache=backend_.allocate(active_cache); snapshot->mtp_v_cache=backend_.allocate(active_cache);
        backend_.copy(*mtp_k_cache_,0,*snapshot->mtp_k_cache,0,active_cache);
        backend_.copy(*mtp_v_cache_,0,*snapshot->mtp_v_cache,0,active_cache);
    }
    snapshot->hidden=backend_.allocate(x1_->size()); backend_.copy(*x1_,0,*snapshot->hidden,0,x1_->size());
    snapshot->logits=backend_.allocate(logits_->size()); backend_.copy(*logits_,0,*snapshot->logits,0,logits_->size());
    batch.finish();
    return snapshot;
}

void MetalEngine::restore_state(const Snapshot& snapshot) {
    if(snapshot.owner!=this || snapshot.layers.size()!=N_LAYER || snapshot.position>max_context_)
        throw std::runtime_error("q27 Metal: incompatible state snapshot");
    backend_.synchronize();
    const uint64_t cache_row = turbo3_kv_ ? (uint64_t)N_KV * 2 * 50
                                          : (uint64_t)N_KV * HEAD_DIM * 2;
    const uint64_t active_cache=(uint64_t)snapshot.position*cache_row;
    CommandBatch batch(backend_);
    for(uint32_t i=0;i<N_LAYER;i++) {
        const auto& source=snapshot.layers[i]; LayerState& dest=layers_[i];
        if(source.recurrent) backend_.copy(*source.recurrent,0,*dest.recurrent,0,source.recurrent->size());
        if(source.ring) backend_.copy(*source.ring,0,*dest.ring,0,source.ring->size());
        if(source.k_cache) { backend_.copy(*source.k_cache,0,*dest.k_cache,0,active_cache); backend_.copy(*source.v_cache,0,*dest.v_cache,0,active_cache); }
    }
    if(snapshot.mtp_k_cache) { backend_.copy(*snapshot.mtp_k_cache,0,*mtp_k_cache_,0,active_cache); backend_.copy(*snapshot.mtp_v_cache,0,*mtp_v_cache_,0,active_cache); }
    backend_.copy(*snapshot.hidden,0,*x1_,0,x1_->size());
    if(snapshot.logits) backend_.copy(*snapshot.logits,0,*logits_,0,logits_->size());
    batch.finish();
    position_=snapshot.position;
}

void MetalEngine::gdn_block(uint32_t layer) {
    backend_.matvec_quantized_pair(layer_weight(layer,"attn_qkv.weight"),*qkv_,
                                   layer_weight(layer,"attn_gate.weight"),*z_,q5120_);
    backend_.matvec_pair(layer_weight(layer,"ssm_alpha.weight"),*alpha_,
                         layer_weight(layer,"ssm_beta.weight"),*beta_raw_,*x1_);
    backend_.gdn_gates(*alpha_, *beta_raw_, layer_weight(layer, "ssm_a"),
                       layer_weight(layer, "ssm_dt.bias"), *g_, *beta_, GDN_HEADS);
    LayerState& state = layers_[layer];
    backend_.conv_step(*state.ring, *state.ring, *qkv_,
                       layer_weight(layer, "ssm_conv1d.weight"), *conv_out_, GDN_CH);
    backend_.l2norm_heads(*conv_out_, 2 * GDN_QK_HEADS, GDN_DIM, EPS);
    backend_.delta_step(*state.recurrent, *state.recurrent, *conv_out_, *g_, *beta_,
                        *delta_out_, GDN_HEADS, GDN_QK_HEADS, GDN_DIM);
    backend_.gated_norm_gdn(*delta_out_, layer_weight(layer, "ssm_norm.weight"), *z_,
                            *gated_out_, GDN_HEADS, GDN_DIM, EPS);
    backend_.quantize(*gated_out_, q6144_);
    backend_.matvec_quantized(layer_weight(layer, "ssm_out.weight"), q6144_, *y_);
}

void MetalEngine::attention_block(uint32_t layer) {
    backend_.matvec_quantized(layer_weight(layer, "attn_q.weight"), q5120_, *qg_);
    backend_.rmsnorm_heads(*qg_, layer_weight(layer, "attn_q_norm.weight"),
                           N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS);
    backend_.matvec_quantized_pair(layer_weight(layer,"attn_k.weight"),*kbuf_,
                                   layer_weight(layer,"attn_v.weight"),*vbuf_,q5120_);
    backend_.rmsnorm_heads(*kbuf_, layer_weight(layer, "attn_k_norm.weight"),
                           N_KV, HEAD_DIM, HEAD_DIM, EPS);
    backend_.rope_neox(*qg_, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, position_, FREQ_BASE);
    backend_.rope_neox(*kbuf_, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, position_, FREQ_BASE);
    LayerState& state = layers_[layer];
    if (turbo3_kv_) {
        backend_.turbo_wht(*qg_, N_HEAD, 2 * HEAD_DIM, false);
        backend_.kv_store_turbo3(*kbuf_, *vbuf_, *state.k_cache, *state.v_cache, position_, N_KV);
        backend_.attention_turbo3(*qg_, 2 * HEAD_DIM, *state.k_cache, *state.v_cache,
                                  *attn_out_, position_ + 1, N_HEAD, N_KV,
                                  HEAD_DIM, 1.0f / std::sqrt((float)HEAD_DIM));
        backend_.turbo_wht(*attn_out_, N_HEAD, HEAD_DIM, true);
    } else {
        backend_.kv_store_f16(*kbuf_, *vbuf_, *state.k_cache, *state.v_cache, position_, N_KV * HEAD_DIM);
        backend_.attention_f16(*qg_, 2 * HEAD_DIM, *state.k_cache, *state.v_cache,
                               *attn_out_, position_ + 1, N_HEAD, N_KV,
                               HEAD_DIM, 1.0f / std::sqrt((float)HEAD_DIM));
    }
    backend_.sigmoid_gate_mul(*attn_out_, *qg_, N_HEAD, HEAD_DIM);
    backend_.quantize(*attn_out_, q6144_);
    backend_.matvec_quantized(layer_weight(layer, "attn_output.weight"), q6144_, *y_);
}

void MetalEngine::ffn(uint32_t layer) {
    backend_.matvec_quantized_pair(layer_weight(layer,"ffn_gate.weight"),*ffn_gate_,
                                   layer_weight(layer,"ffn_up.weight"),*ffn_up_,q5120_);
    backend_.silu_mul(*ffn_gate_, *ffn_up_, *ffn_gate_, N_FFN);
    backend_.quantize(*ffn_gate_, q17408_);
    backend_.matvec_quantized(layer_weight(layer, "ffn_down.weight"), q17408_, *y_);
}

void MetalEngine::encode_token(uint32_t token, bool produce_logits) {
    if (token >= VOCAB) throw std::runtime_error("q27 Metal: token out of range");
    if (position_ >= max_context_) throw std::runtime_error("q27 Metal: context exhausted");
    backend_.embedding_q8(weight("token_embd.weight"), token, *h_);
    for (uint32_t layer = 0; layer < N_LAYER; layer++) {
        backend_.rmsnorm_quantized(*h_,layer_weight(layer,"attn_norm.weight"),*x1_,N_EMBD,EPS,q5120_);
        if (attention_layer(layer)) attention_block(layer); else gdn_block(layer);
        backend_.add_inplace(*h_, *y_, N_EMBD);
        backend_.rmsnorm_quantized(*h_,layer_weight(layer,"post_attention_norm.weight"),*x1_,N_EMBD,EPS,q5120_);
        ffn(layer);
        backend_.add_inplace(*h_, *y_, N_EMBD);
    }
    if(produce_logits)
        backend_.rmsnorm_quantized(*h_,weight("output_norm.weight"),*x1_,N_EMBD,EPS,q5120_);
    else
        backend_.rmsnorm(*h_,weight("output_norm.weight"),*x1_,N_EMBD,EPS);
    if (produce_logits) {
        backend_.matvec_quantized(weight("output.weight"), q5120_, *logits_);
        backend_.argmax(*logits_, VOCAB, *token_out_);
    }
    position_++;
}

void MetalEngine::gdn_chunk(uint32_t layer, uint32_t count) {
    BackendQuantized x5 = quantized_view(cq5120_, count * N_EMBD);
    backend_.matmul_quantized(layer_weight(layer, "attn_qkv.weight"), x5, count, *cqkv_);
    backend_.matmul_quantized(layer_weight(layer, "attn_gate.weight"), x5, count, *cz_);
    backend_.matvec_f16_pair_rows(layer_weight(layer,"ssm_alpha.weight"),*calpha_,
                                  layer_weight(layer,"ssm_beta.weight"),*cbeta_raw_,*cx1_,count);
    backend_.gdn_gates_rows(*calpha_, *cbeta_raw_, layer_weight(layer, "ssm_a"),
                            layer_weight(layer, "ssm_dt.bias"), *cg_, *cbeta_, GDN_HEADS, count);
    LayerState& state = layers_[layer];
    backend_.conv_chunk(*state.ring, *cqkv_, layer_weight(layer, "ssm_conv1d.weight"),
                        *cconv_out_, GDN_CH, count);
    backend_.l2norm_rows(*cconv_out_, 2 * GDN_QK_HEADS, GDN_DIM, GDN_CH, count, EPS);
    backend_.delta_chunk(*state.recurrent, *cconv_out_, *cg_, *cbeta_, *cdelta_out_,
                         GDN_HEADS, GDN_QK_HEADS, GDN_DIM, count);
    // Token rows are contiguous, so the per-head gated norm batches by
    // flattening the chunk into count*GDN_HEADS heads.
    backend_.gated_norm_gdn(*cdelta_out_, layer_weight(layer, "ssm_norm.weight"), *cz_,
                            *cgated_out_, count * GDN_HEADS, GDN_DIM, EPS);
    BackendQuantized x6 = quantized_view(cq6144_, count * GDN_V);
    backend_.quantize(*cgated_out_, x6);
    backend_.matmul_quantized(layer_weight(layer, "ssm_out.weight"), x6, count, *cy_);
}

void MetalEngine::attention_chunk(uint32_t layer, uint32_t count) {
    BackendQuantized x5 = quantized_view(cq5120_, count * N_EMBD);
    backend_.matmul_quantized(layer_weight(layer, "attn_q.weight"), x5, count, *cqg_);
    backend_.rmsnorm_heads(*cqg_, layer_weight(layer, "attn_q_norm.weight"),
                           count * N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS);
    backend_.matmul_quantized(layer_weight(layer, "attn_k.weight"), x5, count, *ckbuf_);
    backend_.matmul_quantized(layer_weight(layer, "attn_v.weight"), x5, count, *cvbuf_);
    backend_.rmsnorm_heads(*ckbuf_, layer_weight(layer, "attn_k_norm.weight"),
                           count * N_KV, HEAD_DIM, HEAD_DIM, EPS);
    backend_.rope_neox_rows(*cqg_, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM,
                            2 * N_HEAD * HEAD_DIM, position_, count, FREQ_BASE);
    backend_.rope_neox_rows(*ckbuf_, N_KV, HEAD_DIM, N_ROT, HEAD_DIM,
                            N_KV * HEAD_DIM, position_, count, FREQ_BASE);
    LayerState& state = layers_[layer];
    const float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    if (turbo3_kv_) {
        backend_.turbo_wht(*cqg_, count * N_HEAD, 2 * HEAD_DIM, false);
        backend_.kv_store_turbo3_rows(*ckbuf_, *cvbuf_, *state.k_cache, *state.v_cache,
                                      position_, N_KV, count);
        backend_.attention_turbo3_causal(*cqg_, 2 * HEAD_DIM, 2 * N_HEAD * HEAD_DIM,
                                         *state.k_cache, *state.v_cache,
                                         *cattn_out_, position_ + 1, N_HEAD, N_KV,
                                         HEAD_DIM, count, scale);
        backend_.turbo_wht(*cattn_out_, count * N_HEAD, HEAD_DIM, true);
    } else {
        backend_.kv_store_f16_rows(*ckbuf_, *cvbuf_, *state.k_cache, *state.v_cache,
                                   position_, N_KV * HEAD_DIM, count);
        backend_.attention_f16_causal(*cqg_, 2 * HEAD_DIM, 2 * N_HEAD * HEAD_DIM,
                                      *state.k_cache, *state.v_cache,
                                      *cattn_out_, position_ + 1, N_HEAD, N_KV,
                                      HEAD_DIM, count, scale);
    }
    backend_.sigmoid_gate_mul_rows(*cattn_out_, *cqg_, N_HEAD, HEAD_DIM, count);
    BackendQuantized x6 = quantized_view(cq6144_, count * N_HEAD * HEAD_DIM);
    backend_.quantize(*cattn_out_, x6);
    backend_.matmul_quantized(layer_weight(layer, "attn_output.weight"), x6, count, *cy_);
}

void MetalEngine::ffn_chunk(uint32_t layer, uint32_t count) {
    BackendQuantized x5 = quantized_view(cq5120_, count * N_EMBD);
    backend_.matmul_quantized(layer_weight(layer, "ffn_gate.weight"), x5, count, *cffn_gate_);
    backend_.matmul_quantized(layer_weight(layer, "ffn_up.weight"), x5, count, *cffn_up_);
    backend_.silu_mul(*cffn_gate_, *cffn_up_, *cffn_gate_, count * N_FFN);
    BackendQuantized x17 = quantized_view(cq17408_, count * N_FFN);
    backend_.quantize(*cffn_gate_, x17);
    backend_.matmul_quantized(layer_weight(layer, "ffn_down.weight"), x17, count, *cy_);
}

void MetalEngine::chunk_forward(const uint32_t* tokens, uint32_t count) {
    if (!ch_) throw std::runtime_error("q27 Metal: chunked prefill is unavailable");
    if (!count || count > CHUNK_MAX) throw std::runtime_error("q27 Metal: invalid chunk size");
    if ((uint64_t)position_ + count > max_context_)
        throw std::runtime_error("q27 Metal: context exhausted");
    for (uint32_t i = 0; i < count; i++)
        if (tokens[i] >= VOCAB) throw std::runtime_error("q27 Metal: token out of range");
    backend_.embedding_q8_rows(weight("token_embd.weight"), tokens, count, *ch_);
    BackendQuantized x5 = quantized_view(cq5120_, count * N_EMBD);
    for (uint32_t layer = 0; layer < N_LAYER; layer++) {
        backend_.rmsnorm_rows_quantized(*ch_, layer_weight(layer, "attn_norm.weight"),
                                        *cx1_, N_EMBD, count, EPS, x5);
        if (attention_layer(layer)) attention_chunk(layer, count); else gdn_chunk(layer, count);
        backend_.add_inplace(*ch_, *cy_, count * N_EMBD);
        backend_.rmsnorm_rows_quantized(*ch_, layer_weight(layer, "post_attention_norm.weight"),
                                        *cx1_, N_EMBD, count, EPS, x5);
        ffn_chunk(layer, count);
        backend_.add_inplace(*ch_, *cy_, count * N_EMBD);
    }
}

void MetalEngine::encode_chunk(const uint32_t* tokens, uint32_t count) {
    chunk_forward(tokens, count);
    position_ += count;
}

// Copies all persistent GDN state (recurrent + convolution ring) to or from
// the checkpoint slots. Roughly 157 MiB of device traffic: negligible next
// to one chunk's full weight stream, and it makes the optimistic committing
// verification round reversible.
void MetalEngine::gdn_state_copy(bool restore) {
    uint64_t slot = 0;
    for (uint32_t layer = 0; layer < N_LAYER; layer++) {
        if (attention_layer(layer)) continue;
        LayerState& state = layers_[layer];
        const uint64_t recurrent_bytes = state.recurrent->size();
        const uint64_t ring_bytes = state.ring->size();
        if (restore) {
            backend_.copy(*ckpt_recurrent_, slot * recurrent_bytes, *state.recurrent, 0, recurrent_bytes);
            backend_.copy(*ckpt_ring_, slot * ring_bytes, *state.ring, 0, ring_bytes);
        } else {
            backend_.copy(*state.recurrent, 0, *ckpt_recurrent_, slot * recurrent_bytes, recurrent_bytes);
            backend_.copy(*state.ring, 0, *ckpt_ring_, slot * ring_bytes, ring_bytes);
        }
        slot++;
    }
}

uint32_t MetalEngine::step(uint32_t token) {
    if (token >= VOCAB) throw std::runtime_error("q27 Metal: token out of range");
    if (position_ >= max_context_) throw std::runtime_error("q27 Metal: context exhausted");
    CommandBatch batch(backend_);
    encode_token(token, true);
    batch.finish();
    uint32_t next = 0;
    backend_.read(*token_out_, 0, &next, sizeof(next));
    return next;
}

void MetalEngine::mtp_warm(const BackendBuffer& hidden, uint32_t token, uint32_t position) {
    constexpr uint32_t layer = 64;
    backend_.embedding_q8(weight("token_embd.weight"), token, *h_);
    backend_.rmsnorm(*h_, layer_weight(layer, "nextn.enorm.weight"), *mtp_embed_norm_, N_EMBD, EPS);
    backend_.rmsnorm(hidden, layer_weight(layer, "nextn.hnorm.weight"), *mtp_hidden_norm_, N_EMBD, EPS);
    backend_.concat(*mtp_embed_norm_, N_EMBD, *mtp_hidden_norm_, N_EMBD, *mtp_concat_);
    backend_.quantize(*mtp_concat_, q10240_);
    backend_.matvec_quantized(layer_weight(layer, "nextn.eh_proj.weight"), q10240_, *mtp_x_);
    backend_.rmsnorm_quantized(*mtp_x_,layer_weight(layer,"attn_norm.weight"),*x1_,N_EMBD,EPS,q5120_);
    backend_.matvec_quantized_pair(layer_weight(layer,"attn_k.weight"),*kbuf_,
                                   layer_weight(layer,"attn_v.weight"),*vbuf_,q5120_);
    backend_.rmsnorm_heads(*kbuf_, layer_weight(layer, "attn_k_norm.weight"),
                           N_KV, HEAD_DIM, HEAD_DIM, EPS);
    backend_.rope_neox(*kbuf_, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, position, FREQ_BASE);
    if (turbo3_kv_)
        backend_.kv_store_turbo3(*kbuf_, *vbuf_, *mtp_k_cache_, *mtp_v_cache_, position, N_KV);
    else
        backend_.kv_store_f16(*kbuf_, *vbuf_, *mtp_k_cache_, *mtp_v_cache_, position, N_KV * HEAD_DIM);
}

uint32_t MetalEngine::prefill(const std::vector<uint32_t>& prompt, bool warm_mtp) {
    if (prompt.empty()) throw std::runtime_error("q27 Metal: prompt is empty");
    if ((uint64_t)position_ + prompt.size() > max_context_)
        throw std::runtime_error("q27 Metal: prompt exceeds context");
    for (uint32_t token : prompt)
        if (token >= VOCAB) throw std::runtime_error("q27 Metal: token out of range");
    // Layer-major chunked ingestion. MTP warming needs each token's final
    // normalized hidden state, which the chunked path does not produce, so
    // MTP prompts stay on the token-serial path. The final prompt token is
    // always serial: it produces logits and leaves the last hidden state in
    // x1_ for MTP drafting and prefix snapshots.
    size_t serial_begin = 0;
    if (chunked_prefill_ && !warm_mtp && prompt.size() >= 3) {
        const size_t chunkable = prompt.size() - 1;
        while (chunkable - serial_begin >= 2) {
            const uint32_t count =
                (uint32_t)std::min<size_t>(CHUNK_MAX, chunkable - serial_begin);
            CommandBatch batch(backend_);
            encode_chunk(prompt.data() + serial_begin, count);
            batch.finish();
            serial_begin += count;
        }
    }
    // Bound encoder growth for long serial prompts: this avoids recording
    // millions of dispatches into one command buffer.
    constexpr size_t COMMAND_CHUNK=8;
    for(size_t begin=serial_begin;begin<prompt.size();begin+=COMMAND_CHUNK) {
        CommandBatch batch(backend_);
        if(begin==serial_begin && warm_mtp && position_>0) mtp_warm(*x1_,prompt.front(),position_);
        size_t end=std::min(prompt.size(),begin+COMMAND_CHUNK);
        for(size_t i=begin;i<end;i++) {
            encode_token(prompt[i],i+1==prompt.size());
            if(warm_mtp && i+1<prompt.size()) mtp_warm(*x1_,prompt[i+1],position_);
        }
        batch.finish();
    }
    uint32_t next = 0;
    backend_.read(*token_out_, 0, &next, sizeof(next));
    return next;
}

uint32_t MetalEngine::mtp_forward(const BackendBuffer& hidden, uint32_t token,
                                  uint32_t position) {
    if (position >= max_context_) throw std::runtime_error("q27 Metal: MTP context exhausted");
    if (token >= VOCAB) throw std::runtime_error("q27 Metal: MTP token out of range");
    constexpr uint32_t layer = 64;
    CommandBatch batch(backend_);
    backend_.embedding_q8(weight("token_embd.weight"), token, *h_);
    backend_.rmsnorm(*h_, layer_weight(layer, "nextn.enorm.weight"), *mtp_embed_norm_, N_EMBD, EPS);
    backend_.rmsnorm(hidden, layer_weight(layer, "nextn.hnorm.weight"), *mtp_hidden_norm_, N_EMBD, EPS);
    backend_.concat(*mtp_embed_norm_, N_EMBD, *mtp_hidden_norm_, N_EMBD, *mtp_concat_);
    backend_.quantize(*mtp_concat_, q10240_);
    backend_.matvec_quantized(layer_weight(layer, "nextn.eh_proj.weight"), q10240_, *mtp_x_);

    backend_.rmsnorm_quantized(*mtp_x_,layer_weight(layer,"attn_norm.weight"),*x1_,N_EMBD,EPS,q5120_);
    backend_.matvec_quantized(layer_weight(layer, "attn_q.weight"), q5120_, *qg_);
    backend_.rmsnorm_heads(*qg_, layer_weight(layer, "attn_q_norm.weight"),
                           N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS);
    backend_.matvec_quantized_pair(layer_weight(layer,"attn_k.weight"),*kbuf_,
                                   layer_weight(layer,"attn_v.weight"),*vbuf_,q5120_);
    backend_.rmsnorm_heads(*kbuf_, layer_weight(layer, "attn_k_norm.weight"),
                           N_KV, HEAD_DIM, HEAD_DIM, EPS);
    backend_.rope_neox(*qg_, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, position, FREQ_BASE);
    backend_.rope_neox(*kbuf_, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, position, FREQ_BASE);
    if (turbo3_kv_) {
        backend_.turbo_wht(*qg_, N_HEAD, 2 * HEAD_DIM, false);
        backend_.kv_store_turbo3(*kbuf_, *vbuf_, *mtp_k_cache_, *mtp_v_cache_, position, N_KV);
        backend_.attention_turbo3(*qg_, 2 * HEAD_DIM, *mtp_k_cache_, *mtp_v_cache_,
                                  *attn_out_, position + 1, N_HEAD, N_KV,
                                  HEAD_DIM, 1.0f / std::sqrt((float)HEAD_DIM));
        backend_.turbo_wht(*attn_out_, N_HEAD, HEAD_DIM, true);
    } else {
        backend_.kv_store_f16(*kbuf_, *vbuf_, *mtp_k_cache_, *mtp_v_cache_, position, N_KV * HEAD_DIM);
        backend_.attention_f16(*qg_, 2 * HEAD_DIM, *mtp_k_cache_, *mtp_v_cache_,
                               *attn_out_, position + 1, N_HEAD, N_KV,
                               HEAD_DIM, 1.0f / std::sqrt((float)HEAD_DIM));
    }
    backend_.sigmoid_gate_mul(*attn_out_, *qg_, N_HEAD, HEAD_DIM);
    backend_.quantize(*attn_out_, q6144_);
    backend_.matvec_quantized(layer_weight(layer, "attn_output.weight"), q6144_, *y_);
    backend_.add_inplace(*mtp_x_, *y_, N_EMBD);

    backend_.rmsnorm_quantized(*mtp_x_,layer_weight(layer,"post_attention_norm.weight"),*x1_,N_EMBD,EPS,q5120_);
    backend_.matvec_quantized_pair(layer_weight(layer,"ffn_gate.weight"),*ffn_gate_,
                                   layer_weight(layer,"ffn_up.weight"),*ffn_up_,q5120_);
    backend_.silu_mul(*ffn_gate_, *ffn_up_, *ffn_gate_, N_FFN);
    backend_.quantize(*ffn_gate_, q17408_);
    backend_.matvec_quantized(layer_weight(layer, "ffn_down.weight"), q17408_, *y_);
    backend_.add_inplace(*mtp_x_, *y_, N_EMBD);
    backend_.rmsnorm(*mtp_x_, layer_weight(layer, "nextn.shared_head_norm.weight"),
                     *mtp_hidden_out_, N_EMBD, EPS);
    const BackendTensor& head = model_.find("output_q4.weight") ? weight("output_q4.weight")
                                                                : weight("output.weight");
    backend_.quantize(*mtp_hidden_out_, q5120_);
    backend_.matvec_quantized(head, q5120_, *logits_);
    backend_.argmax(*logits_, VOCAB, *token_out_);
    batch.finish();
    uint32_t result = 0; backend_.read(*token_out_, 0, &result, sizeof(result));
    return result;
}

// One batched MTP round: draft serially through layer 64, then verify every
// lane in a single optimistic committing layer-major pass with a batched
// output head and per-lane argmax — one CPU synchronization per round
// instead of one per committed token. The GDN checkpoint makes partial
// acceptance reversible; committed tokens follow the exact serial-walk
// semantics, including never encoding the final output token.
std::vector<uint32_t> MetalEngine::generate_mtp_batched(uint32_t pending, uint32_t count,
                                                        uint32_t width) {
    std::vector<uint32_t> output;
    output.reserve(count);
    // Start narrow and let acceptance widen the window: committed tokens are
    // width-invariant, and a wide first round pays for many serial drafts
    // through a cold draft head before acceptance has been measured once.
    uint32_t live_width = std::min(width, 4u);
    while (output.size() < count) {
        if (output.size() + 1 == count) { output.push_back(pending); break; }
        const uint32_t remaining = (uint32_t)(count - output.size());
        uint32_t live = std::min(live_width, remaining);
        // The verify chunk stores a KV row for every lane, so it must stay
        // inside the reserved context even before acceptance is known.
        if ((uint64_t)position_ + live > max_context_)
            live = (uint32_t)(max_context_ - position_);
        if (live < 2) {
            output.push_back(pending);
            if (output.size() == count) break;
            pending = step(pending);
            continue;
        }
        static const bool trace = getenv("Q27_MTP_TRACE") != nullptr;
        auto clock = [] { return std::chrono::steady_clock::now(); };
        auto since = [](std::chrono::steady_clock::time_point start) {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        };
        auto draft_start = clock();
        std::vector<uint32_t> lanes(live);
        lanes[0] = pending;
        const BackendBuffer* hidden = x1_.get();
        for (uint32_t lane = 1; lane < live; lane++) {
            lanes[lane] = mtp_forward(*hidden, lanes[lane - 1], position_ + lane - 1);
            hidden = mtp_hidden_out_.get();
        }
        last_spec_stats_.rounds++;
        last_spec_stats_.drafted += live - 1;
        auto verify_start = clock();
        {
            CommandBatch batch(backend_);
            gdn_state_copy(false);
            chunk_forward(lanes.data(), live);
            BackendQuantized x5 = quantized_view(cq5120_, live * N_EMBD);
            backend_.rmsnorm_rows_quantized(*ch_, weight("output_norm.weight"), *cfinal_,
                                            N_EMBD, live, EPS, x5);
            backend_.matmul_quantized(weight("output.weight"), x5, live, *clogits_);
            backend_.argmax_rows(*clogits_, VOCAB, live, *cpred_);
            batch.finish();
        }
        std::vector<uint32_t> predictions(live);
        backend_.read(*cpred_, 0, predictions.data(), live * sizeof(uint32_t));
        uint32_t accepted = 0;
        while (accepted + 1 < live && predictions[accepted] == lanes[accepted + 1]) accepted++;
        uint32_t committed = std::min(accepted + 1, remaining);
        // The final output token is pushed but never encoded, exactly like
        // the serial walk, so snapshots and continuations stay compatible.
        const uint32_t encoded = committed == remaining ? committed - 1 : committed;
        last_spec_stats_.accepted += committed - 1;
        auto commit_start = clock();
        {
            CommandBatch batch(backend_);
            if (encoded != live) {
                gdn_state_copy(true);
                if (encoded) chunk_forward(lanes.data(), encoded);
            }
            if (encoded) {
                backend_.copy(*cfinal_, (uint64_t)(encoded - 1) * N_EMBD * sizeof(float),
                              *x1_, 0, (uint64_t)N_EMBD * sizeof(float));
                backend_.copy(*clogits_, (uint64_t)(encoded - 1) * VOCAB * sizeof(float),
                              *logits_, 0, (uint64_t)VOCAB * sizeof(float));
            }
            batch.finish();
        }
        position_ += encoded;
        if (trace)
            fprintf(stderr, "mtp round: live %u accepted %u | draft %.2fs verify %.2fs commit %.2fs\n",
                    live, accepted, std::chrono::duration<double>(verify_start - draft_start).count(),
                    std::chrono::duration<double>(commit_start - verify_start).count(), since(commit_start));
        for (uint32_t i = 0; i < committed; i++) output.push_back(lanes[i]);
        pending = predictions[committed - 1];
        // Width adaptation is a pure performance control: committed tokens
        // are width-invariant, matching the recorded 2/4/8/12 gate.
        live_width = accepted + 1 == live ? std::min(width, live_width + 2)
                                          : std::max(2u, accepted + 2);
    }
    return output;
}

uint32_t MetalEngine::ingest_prompt(const std::vector<uint32_t>& tokens, bool warm_mtp,
                                    bool reset_first) {
    if (reset_first) reset();
    return prefill(tokens, warm_mtp);
}

std::vector<float> MetalEngine::read_logits() {
    std::vector<float> result(VOCAB);
    backend_.synchronize();
    backend_.read(*logits_,0,result.data(),result.size()*sizeof(float));
    return result;
}

std::vector<float> MetalEngine::teacher_force_nll(const std::vector<uint32_t>& tokens) {
    if (tokens.size() < 2) throw std::runtime_error("q27 Metal: NLL needs at least two tokens");
    if (tokens.size() - 1 > max_context_)
        throw std::runtime_error("q27 Metal: NLL sequence exceeds context");
    for (uint32_t token : tokens)
        if (token >= VOCAB) throw std::runtime_error("q27 Metal: token out of range");
    reset();
    const uint32_t n_encode = (uint32_t)tokens.size() - 1;
    std::vector<float> result;
    result.reserve(n_encode);

    auto nll_cpu = [](const float* logits, uint32_t target, uint32_t vocab) -> float {
        double mx = -1e300;
        for (uint32_t v = 0; v < vocab; v++) mx = std::max(mx, (double)logits[v]);
        double se = 0.0;
        for (uint32_t v = 0; v < vocab; v++) se += std::exp((double)logits[v] - mx);
        return (float)(std::log(se) + mx - (double)logits[target]);
    };

    uint32_t done = 0;
    // Prefer the layer-major chunk path: one command buffer per up-to-12
    // tokens, batched output head, and a GPU logsumexp so only `count`
    // floats cross back to the CPU per chunk.
    if (chunked_prefill_ && n_encode >= 2) {
        while (n_encode - done >= 2) {
            const uint32_t count = std::min(CHUNK_MAX, n_encode - done);
            std::vector<uint32_t> targets(count);
            for (uint32_t r = 0; r < count; r++) targets[r] = tokens[done + r + 1];
            // Host write before the command batch: targets are shared-memory
            // and must be visible before the NLL kernel is encoded.
            backend_.write(*ctargets_, 0, targets.data(), count * sizeof(uint32_t));
            {
                CommandBatch batch(backend_);
                chunk_forward(tokens.data() + done, count);
                BackendQuantized x5 = quantized_view(cq5120_, count * N_EMBD);
                backend_.rmsnorm_rows_quantized(*ch_, weight("output_norm.weight"), *cfinal_,
                                                N_EMBD, count, EPS, x5);
                backend_.matmul_quantized(weight("output.weight"), x5, count, *clogits_);
                backend_.nll_rows(*clogits_, *ctargets_, *cnll_, VOCAB, count);
                batch.finish();
            }
            position_ += count;
            std::vector<float> chunk_nll(count);
            backend_.read(*cnll_, 0, chunk_nll.data(), count * sizeof(float));
            result.insert(result.end(), chunk_nll.begin(), chunk_nll.end());
            done += count;
            if ((done / CHUNK_MAX) % 32 == 0)
                fprintf(stderr, "  nll pos %u/%u\r", done, n_encode);
        }
    }
    while (done < n_encode) {
        {
            CommandBatch batch(backend_);
            encode_token(tokens[done], true);
            batch.finish();
        }
        std::vector<float> logits = read_logits();
        result.push_back(nll_cpu(logits.data(), tokens[done + 1], VOCAB));
        done++;
    }
    if (n_encode >= CHUNK_MAX) fprintf(stderr, "\n");
    return result;
}

std::vector<uint32_t> MetalEngine::generate_sampled_from_logits(uint32_t count,
                                                                 const SamplingParams& params) {
    validate_sampling(params); last_spec_stats_={};
    if((uint64_t)position_+(count?count-1:0)>max_context_)
        throw std::runtime_error("q27 Metal: generation exceeds context");
    std::mt19937_64 random(params.seed); std::vector<uint32_t> output; output.reserve(count);
    for(uint32_t i=0;i<count;i++) {
        uint32_t token=sample_logits_cpu(read_logits(),params,random); output.push_back(token);
        if(i+1<count) step(token);
    }
    return output;
}

std::vector<uint32_t> MetalEngine::generate_sampled(const std::vector<uint32_t>& prompt,
                                                     uint32_t count,const SamplingParams& params) {
    if(prompt.empty()) throw std::runtime_error("q27 Metal: prompt is empty");
    if((uint64_t)prompt.size()+count>max_context_+1)
        throw std::runtime_error("q27 Metal: prompt/generation exceeds context");
    ingest_prompt(prompt,false,true);
    return generate_sampled_from_logits(count,params);
}

std::vector<uint32_t> MetalEngine::generate_from_pending(uint32_t pending, uint32_t count,
                                                          uint32_t mtp_width) {
    last_spec_stats_={};
    if (pending >= VOCAB) throw std::runtime_error("q27 Metal: pending token out of range");
    if (mtp_width && (mtp_width < 2 || mtp_width > 12))
        throw std::runtime_error("q27 Metal: MTP width must be 2..12");
    if ((uint64_t)position_ + (count ? count - 1 : 0) > max_context_)
        throw std::runtime_error("q27 Metal: generation exceeds context");
    if (mtp_width && chunked_prefill_)
        return generate_mtp_batched(pending, count, mtp_width);
    std::vector<uint32_t> output;
    output.reserve(count);
    if (!mtp_width) {
        if (!count) return output;
        output.push_back(pending);
        for (uint32_t i=1;i<count;i++) { pending=step(pending); output.push_back(pending); }
        return output;
    }
    while (output.size() < count) {
        if (output.size() + 1 == count) { output.push_back(pending); break; }
        const uint32_t live_width = std::min<uint32_t>(mtp_width, (uint32_t)(count - output.size()));
        std::vector<uint32_t> drafts;
        drafts.reserve(live_width - 1);
        const BackendBuffer* hidden = x1_.get();
        uint32_t draft_token = pending;
        for (uint32_t lane = 1; lane < live_width; lane++) {
            draft_token = mtp_forward(*hidden, draft_token, position_ + lane - 1);
            drafts.push_back(draft_token);
            hidden = mtp_hidden_out_.get();
        }
        last_spec_stats_.rounds++; last_spec_stats_.drafted+=drafts.size();
        output.push_back(pending);
        uint32_t prediction = step(pending);
        for (uint32_t draft : drafts) {
            if (prediction != draft) break;
            last_spec_stats_.accepted++;
            output.push_back(draft);
            if (output.size() == count) return output;
            prediction = step(draft);
        }
        pending = prediction;
    }
    return output;
}

std::vector<uint32_t> MetalEngine::generate(const std::vector<uint32_t>& prompt, uint32_t count) {
    if (prompt.empty()) throw std::runtime_error("q27 Metal: prompt is empty");
    if ((uint64_t)prompt.size() + count > max_context_ + 1)
        throw std::runtime_error("q27 Metal: prompt/generation exceeds context");
    uint32_t pending = ingest_prompt(prompt, false, true);
    return generate_from_pending(pending, count);
}

std::vector<uint32_t> MetalEngine::generate_mtp(const std::vector<uint32_t>& prompt,
                                                 uint32_t count, uint32_t width) {
    if (prompt.empty()) throw std::runtime_error("q27 Metal: prompt is empty");
    if ((uint64_t)prompt.size() + count > max_context_ + 1)
        throw std::runtime_error("q27 Metal: prompt/generation exceeds context");
    uint32_t pending = ingest_prompt(prompt, true, true);
    return generate_from_pending(pending, count, width);
}

std::vector<uint32_t> MetalEngine::generate_suffix(const std::vector<uint32_t>& prompt,
                                                    uint32_t count, uint32_t width,
                                                    uint32_t minimum_match) {
    if(prompt.empty()) throw std::runtime_error("q27 Metal: prompt is empty");
    if(width<2 || width>12) throw std::runtime_error("q27 Metal: suffix width must be 2..12");
    if((uint64_t)prompt.size()+count>max_context_+1)
        throw std::runtime_error("q27 Metal: prompt/generation exceeds context");
    uint32_t pending=ingest_prompt(prompt,false,true); last_spec_stats_={};
    std::vector<int> history(prompt.begin(),prompt.end()); SuffixDraft drafter; drafter.reset(history);
    std::vector<uint32_t> output; output.reserve(count);
    while(output.size()<count) {
        if(output.size()+1==count) { output.push_back(pending); break; }
        const uint32_t lanes=std::min<uint32_t>(width-1,(uint32_t)(count-output.size()-1));
        std::vector<int> proposals(lanes);
        int match=drafter.propose_with((int)pending,(int)lanes,proposals.data());
        last_spec_stats_.rounds++;
        if(match>=(int)minimum_match) last_spec_stats_.drafted+=proposals.size();
        output.push_back(pending); drafter.append((int)pending);
        uint32_t prediction=step(pending);
        if(match>=(int)minimum_match) for(int proposal:proposals) {
            if(prediction!=(uint32_t)proposal) break;
            last_spec_stats_.accepted++;
            output.push_back((uint32_t)proposal); drafter.append(proposal);
            if(output.size()==count) return output;
            prediction=step((uint32_t)proposal);
        }
        pending=prediction;
    }
    return output;
}

} // namespace q27
