#include "metal_engine.h"

#include "../../third_party/json.hpp"
#include "../suffixdraft.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <CommonCrypto/CommonDigest.h>

namespace q27 {
namespace {

// Bonsai matrix tiers (T2 ternary / B1 binary): exact select-form math on
// float activations, no activation quantization — one dispatch policy for
// both (binary-tier plan, Phase 3). Q4/Q8 keep the packed-dot quantized path.
bool is_bonsai_dtype(DType dtype) {
    return dtype == DType::T2_G128 || dtype == DType::B1_G128;
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
    bool logits_resident = false;
    std::vector<StoredLayer> layers;
    std::shared_ptr<BackendBuffer> mtp_k_cache, mtp_v_cache, hidden, logits;
    // KV fp16 exception side rows (snapshot v2): flat, in kv_fp16_side_
    // traversal order (attn_idx asc, head asc), K then V per masked head.
    // Empty when the engine has no exception cells or position was 0. The
    // owner check pins the config: a Snapshot never crosses engines, so
    // the side layout always matches.
    std::vector<std::shared_ptr<BackendBuffer>> kv_side;
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
    // Bonsai artifacts (T2 ternary / B1 binary repacks): 64 blocks, no MTP
    // layer, bonsai-dtype embeddings/head/alpha/beta. The sibling packs
    // share this architecture even though their trained tensor bytes differ.
    const std::string policy = meta.value("quant_policy", std::string());
    const bool ternary = policy == "bonsai-t2-v1";
    const bool binary = policy == "bonsai-b1-v1";
    // Mixed-tier census packs (docs/metal/plans/2026-07-17-mixed-tier-census.md,
    // tools/q27_mix.py): per-tensor T2/B1 routing over the same bonsai
    // shape. Both tiers' layout declarations are demanded below.
    const bool mixed = policy == "bonsai-mixed-v1";
    const bool bonsai = ternary || binary || mixed;
    exact("qwen35.block_count", bonsai ? 64 : 65); exact("qwen35.embedding_length", N_EMBD);
    exact("qwen35.feed_forward_length", N_FFN); exact("qwen35.attention.head_count", N_HEAD);
    exact("qwen35.attention.head_count_kv", N_KV); exact("qwen35.attention.key_length", HEAD_DIM);
    exact("qwen35.attention.value_length", HEAD_DIM); exact("qwen35.ssm.state_size", GDN_DIM);
    exact("qwen35.ssm.group_count", GDN_QK_HEADS); exact("qwen35.ssm.inner_size", GDN_V);
    exact("qwen35.context_length", 262144); exact("qwen35.rope.dimension_count", N_ROT);
    exact("qwen35.ssm.conv_kernel", 4); exact("qwen35.ssm.time_step_rank", GDN_HEADS);
    exact("qwen35.full_attention_interval", 4);
    if (!bonsai) exact("qwen35.nextn_predict_layers", 1);
    exact("group_q4", 64); exact("group_q8", 128);
    auto exact_str = [&](const char* key, const char* expected) {
        if (meta.value(key, std::string()) != expected)
            throw std::runtime_error(std::string("q27 Metal: architecture mismatch: ") + key);
    };
    // The kernels hardcode the pack encodings; the meta strings are the
    // repack's declaration of what it wrote (codex P3 on the B1 wiring —
    // the T2 twin closes the same pre-existing gap).
    if (ternary || mixed) {
        exact("group_t2", 128);
        exact_str("t2_codes", "0=-1,1=0,2=+1;3 forbidden");
        exact_str("t2_slot_order", "seq-lsb-first");
    }
    if (binary || mixed) {
        exact("group_b1", 128);
        exact_str("b1_codes", "1=+d,0=-d");
        exact_str("b1_bit_order", "seq-lsb-first");
    }
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
    const size_t expected_map = expected_attention.size() + (bonsai ? 0 : 1);
    if (!meta.contains("attn_layers") || meta["attn_layers"].size() != expected_map)
        throw std::runtime_error("q27 Metal: invalid attention layer map");
    for (size_t i = 0; i < expected_attention.size(); i++)
        if (meta["attn_layers"][i].get<uint32_t>() != expected_attention[i])
            throw std::runtime_error("q27 Metal: unexpected attention layer map");
    if (!bonsai && meta["attn_layers"].back().get<uint32_t>() != 64)
        throw std::runtime_error("q27 Metal: missing MTP attention layer");

    auto require = [&](const std::string& name, DType dtype, std::initializer_list<uint64_t> shape) {
        const Tensor* tensor = model_.find(name);
        if (!tensor || tensor->dtype != dtype || tensor->shape != std::vector<uint64_t>(shape))
            throw std::runtime_error("q27 Metal: required tensor mismatch: " + name);
    };
    // The allowed matrix dtype set follows the pack policy (codex P2 on
    // the B1 wiring): per-tensor routing would happily run a mixed-tier
    // artifact, so a stray wrong-tier matrix means a broken repack and
    // must fail here, not compute silently.
    auto matrix_dtype_ok = [&](DType dtype) {
        if (ternary) return dtype == DType::T2_G128;
        if (binary) return dtype == DType::B1_G128;
        if (mixed) return dtype == DType::T2_G128 || dtype == DType::B1_G128;
        return dtype == DType::Q4_G64 || dtype == DType::Q8_G128;
    };
    auto matrix = [&](const std::string& name, uint64_t rows, uint64_t cols) {
        const Tensor* tensor = model_.find(name);
        if (!tensor || !matrix_dtype_ok(tensor->dtype) || tensor->shape != std::vector<uint64_t>{rows, cols})
            throw std::runtime_error("q27 Metal: required matrix mismatch: " + name);
    };

    // Tensors whose tier follows the pack policy: mixed admits either
    // bonsai dtype (per-tensor routing decides at dispatch), the pure
    // packs stay pinned exactly.
    auto require_tier = [&](const std::string& name, DType pure,
                            std::initializer_list<uint64_t> shape) {
        if (!mixed) { require(name, pure, shape); return; }
        const Tensor* tensor = model_.find(name);
        if (!tensor || (tensor->dtype != DType::T2_G128 && tensor->dtype != DType::B1_G128) ||
            tensor->shape != std::vector<uint64_t>(shape))
            throw std::runtime_error("q27 Metal: required tensor mismatch: " + name);
    };
    const DType vocab_dtype = ternary ? DType::T2_G128
                            : binary ? DType::B1_G128 : DType::Q8_G128;   // unused under mixed
    require_tier("token_embd.weight", vocab_dtype, {VOCAB, N_EMBD});
    // output.weight (the lm_head) is the ONE vocab tensor whose dtype varies
    // across the official Q-tier ladder, so it cannot be pinned like
    // token_embd. repack.py's --q4-head emits it at Q4_G64 and DROPS the
    // output_q4.weight dupe (one head then serves draft/verify/plain); that
    // is the defining move of q4s, and q5f/q6f inherit it and promote FFN
    // tensors on top. Pinning it to Q8 rejected all three tiers -- including
    // q5f, upstream's best-quality pack that fits a 24 GB card, i.e. this box.
    //
    // Widened only for the Q-tier family, and only for this tensor:
    //   - ternary/binary/mixed packs stay pinned exactly (require_tier), so
    //     a bonsai head cannot drift in;
    //   - token_embd stays pinned Q8 -- no shipped tier moves it (it is a
    //     row lookup, not a GEMV read), so it keeps its broken-repack check;
    //   - Q4_G64 and Q8_G128 are the only accepted values, matching what
    //     matrix() already allows for every layer weight.
    // Both head consumers handle either dtype: project() routes Q4/Q8 to
    // matvec_quantized (serial decode + MTP draft), and matmul_quantized
    // accepts Q4_G64 with its own q4 pipeline and group/divisor (batched
    // verify + chunked prefill). The MTP draft head already REQUIRES Q4_G64
    // at this exact shape when output_q4.weight is present, so a Q4
    // vocab-sized head is a path this engine has always exercised.
    if (ternary || binary || mixed) {
        require_tier("output.weight", vocab_dtype, {VOCAB, N_EMBD});
    } else {
        const Tensor* head = model_.find("output.weight");
        if (!head || (head->dtype != DType::Q4_G64 && head->dtype != DType::Q8_G128) ||
            head->shape != std::vector<uint64_t>{VOCAB, N_EMBD})
            throw std::runtime_error("q27 Metal: required tensor mismatch: output.weight");
    }
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
            if (bonsai) {
                require_tier(p + "ssm_alpha.weight", ternary ? DType::T2_G128 : DType::B1_G128,
                             {GDN_HEADS, N_EMBD});
                require_tier(p + "ssm_beta.weight", ternary ? DType::T2_G128 : DType::B1_G128,
                             {GDN_HEADS, N_EMBD});
            } else {
                require(p + "ssm_alpha.weight", DType::F16, {GDN_HEADS, N_EMBD});
                require(p + "ssm_beta.weight", DType::F16, {GDN_HEADS, N_EMBD});
            }
            require(p + "ssm_a", DType::F32, {GDN_HEADS});
            require(p + "ssm_dt.bias", DType::F32, {GDN_HEADS});
            require(p + "ssm_conv1d.weight", DType::F32, {GDN_CH, 4});
            require(p + "ssm_norm.weight", DType::F32, {GDN_DIM});
            matrix(p + "ssm_out.weight", N_EMBD, GDN_V);
        }
    }
    if (bonsai) {
        // No MTP layer in bonsai packs; a partial blk.64 would mean a broken
        // repack, so its absence is asserted rather than tolerated silently.
        if (model_.find("blk.64.attn_norm.weight") || model_.find("output_q4.weight"))
            throw std::runtime_error("q27 Metal: unexpected MTP tensors in a bonsai artifact");
        return;
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

std::shared_ptr<MetalEngine::Shared> MetalEngine::open_shared(const std::string& model_path) {
    auto shared = std::make_shared<Shared>(Model::open(model_path));
    shared->path = model_path;
    return shared;
}

// Return this engine's KV budget to the mapping so later engines on a
// still-live Shared are not falsely rejected (codex sweep finding). The
// assert guards double-return/underflow (round-2 expert P0 #2 companion).
MetalEngine::~MetalEngine() {
    assert(shared_->cache_bytes >= engine_cache_bytes_ && "KV reservation underflow");
    // Can't throw from a dtor; clamp+log beats silent wrap (k3 audit B8).
    if (shared_->cache_bytes < engine_cache_bytes_) {
        fprintf(stderr, "q27 Metal: KV reservation underflow — double return?\n");
        shared_->cache_bytes = 0;
    } else {
        shared_->cache_bytes -= engine_cache_bytes_;
    }
}

int MetalEngine::mask_pool_add(const void* bits) {
    constexpr uint64_t words = ((uint64_t)VOCAB + 31) / 32;
    if (!bits) throw std::runtime_error("q27 Metal: null constraint mask");
    if (mask_pool_used >= MASK_POOL_CAP) return -1;
    if (!mask_pool_) mask_pool_ = backend_.allocate(words * 4 * MASK_POOL_CAP);
    backend_.write(*mask_pool_, (uint64_t)mask_pool_used * words * 4, bits, words * 4);
    return mask_pool_used++;
}

void MetalEngine::set_tool_constraint(int mask_id) {
    if (mask_id >= mask_pool_used)
        throw std::runtime_error("q27 Metal: constraint mask id out of range");
    active_mask_ = mask_id < 0 ? -1 : mask_id;
}

namespace {
std::shared_ptr<MetalEngine::Shared> require_shared(std::shared_ptr<MetalEngine::Shared> shared) {
    if (!shared) throw std::runtime_error("q27 Metal: null shared context");
    return shared;
}
} // namespace

MetalEngine::MetalEngine(const std::string& model_path, uint32_t context, bool turbo3_kv)
    : MetalEngine(open_shared(model_path), context, turbo3_kv) {}

MetalEngine::MetalEngine(std::shared_ptr<Shared> shared, uint32_t context, bool turbo3_kv)
    : shared_(require_shared(std::move(shared))), model_(shared_->model),
      backend_(shared_->backend), max_context_(context), turbo3_kv_(turbo3_kv),
      weights_(shared_->weights) {
    if (!context || context > 262144) throw std::runtime_error("q27 Metal: context must be 1..262144");
    validate_architecture();
    has_mtp_ = model_.find("blk.64.attn_norm.weight") != nullptr;
    const uint64_t cache_row_bytes = turbo3_kv_ ? (uint64_t)N_KV * 2 * 50
                                                : (uint64_t)N_KV * HEAD_DIM * 2;
    // Per-engine blocked-GQA partials (audit E2): sized once here for this
    // engine's own context at the widest attention width this device can
    // dispatch, and reserved alongside the caches — it is the other
    // ctx-scaled allocation. Sizing deliberately ignores the GQA threshold:
    // the envelope instrument flips it at runtime, which must only change
    // routing, never invalidate the buffer.
    const bool chunk_capable = backend_.supports_quantized_matmul() && has_mtp_;
    const uint64_t partial_bytes =
        gqa_partial_peak(max_context_, backend_.gqa_block_size(), chunk_capable);
    // Production KV fp16 exception cells
    // (docs/metal/plans/2026-07-17-kv-except-production.md): parse the env once
    // here so the side-cache bytes join the same reservation. Cells use
    // census numbering (attn_idx*8 + head*2 + side); v1 requires a head's K
    // and V cells together (step 4b: K alone retains nothing, V alone
    // amplifies — only the pair is meaningful). fp16-KV engines ignore the
    // env (their cells are already fp16), so the kl-kv baseline coexists.
    uint64_t kv_side_bytes = 0;
    if (const char* cells_env = getenv("Q27_METAL_KV_FP16_CELLS")) {
        if (turbo3_kv_) {
            uint8_t side_masks[16][4] = {};
            const std::string list(cells_env);
            size_t at = 0;
            while (at < list.size()) {
                size_t comma = list.find(',', at);
                if (comma == std::string::npos) comma = list.size();
                const std::string field = list.substr(at, comma - at);
                size_t used = 0;
                const unsigned long cell = std::stoul(field, &used);
                if (used != field.size())
                    throw std::runtime_error("q27 Metal: malformed Q27_METAL_KV_FP16_CELLS entry: " + field);
                if (cell >= 128)
                    throw std::runtime_error("q27 Metal: Q27_METAL_KV_FP16_CELLS cells must be 0..127");
                side_masks[cell >> 3][(cell >> 1) & 3] |= uint8_t(1u << (cell & 1u));
                at = comma + 1;
            }
            for (uint32_t li = 0; li < 16; li++)
                for (uint32_t h = 0; h < 4; h++) {
                    if (side_masks[li][h] == 0) continue;
                    if (side_masks[li][h] != 3)
                        throw std::runtime_error("q27 Metal: Q27_METAL_KV_FP16_CELLS v1 needs a head's K and V cells together (step 4b: only the pair is protective)");
                    kv_fp16_head_masks_[li] |= uint8_t(1u << h);
                    kv_side_bytes += 2ull * max_context_ * HEAD_DIM * 2;   // K + V, fp16
                    kv_fp16_except_ = true;
                }
        } else {
            fprintf(stderr, "q27 Metal: Q27_METAL_KV_FP16_CELLS ignored on an fp16-KV engine (cells already fp16)\n");
        }
    }
    // Side-cache codec (hot-cells arm): e4m3 rounds the excepted cells'
    // rows onto the fp8 grid at store time — partial fidelity instead of
    // full fp16, at the real e4m3 side format's byte price. Meaningless
    // without an exception list, so that combination is rejected loudly
    // rather than silently ignored.
    if (const char* codec_env = getenv("Q27_METAL_KV_CELLS_CODEC")) {
        const std::string codec(codec_env);
        if (codec != "fp16" && codec != "e4m3")
            throw std::runtime_error("q27 Metal: Q27_METAL_KV_CELLS_CODEC must be fp16 or e4m3");
        if (codec == "e4m3") {
            if (kv_fp16_except_)
                kv_fp16_side_codec_ = 1;
            else if (turbo3_kv_ || !getenv("Q27_METAL_KV_FP16_CELLS"))
                // Covers the EMPTY cells list on turbo3 too: an explicitly
                // requested e4m3 arm must never silently degrade to plain
                // turbo3 (vacuous-instrument class; codex P2 on d9ee75a).
                throw std::runtime_error("q27 Metal: Q27_METAL_KV_CELLS_CODEC=e4m3 needs a non-empty Q27_METAL_KV_FP16_CELLS (nothing to encode)");
            // else: fp16-KV engine — the cells env was ignored above (with
            // its note), so the codec rides along ignored too; the kl-kv
            // baseline engine shares the subject's process environment.
        }
    }
    const uint64_t total_cache_bytes =
        (16ull + (has_mtp_ ? 1 : 0)) * 2 * max_context_ * cache_row_bytes
        + partial_bytes + kv_side_bytes;
    // Budget the combined caches of every engine on this mapping, not just
    // this one — two engines can each pass a per-engine check while jointly
    // overcommitting the device.
    if (shared_->cache_bytes + total_cache_bytes > backend_.recommended_working_set_size() / 2)
        throw std::runtime_error("q27 Metal: requested KV cache (across engines on this mapping) is too large for this device; use --kv turbo3 or reduce --ctx");
    // Constructor-exception safety (round-2 expert P0): the destructor only
    // runs for fully-constructed engines, so a throw in any allocation below
    // would otherwise strand this reservation and falsely reject later
    // engines on a still-live Shared. Roll back unless the constructor
    // completes; Shared is single-thread by contract, plain arithmetic.
    struct ReservationGuard {
        Shared& shared;
        uint64_t bytes;
        bool committed = false;
        ~ReservationGuard() { if (!committed) shared.cache_bytes -= bytes; }
    } reservation{*shared_, total_cache_bytes};
    shared_->cache_bytes += total_cache_bytes;
    engine_cache_bytes_ = total_cache_bytes;

    // All wrappers alias the mmap. No weight-sized copy is created. A second
    // engine on the same Shared reuses the wrap — never a second mapping.
    if (weights_.empty()) {
        weights_.reserve(model_.tensors.size());
        for (const Tensor& tensor : model_.tensors)
            weights_.emplace(tensor.name, backend_.upload(model_, tensor));
    }

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
    topk_values_ = alloc_f32(TOPK_CAPACITY);
    topk_indices_ = backend_.allocate(TOPK_CAPACITY * sizeof(uint32_t));
    topk_count_ = backend_.allocate(sizeof(uint32_t));
    token_ring_ = backend_.allocate(RESIDENT_MAX * sizeof(uint32_t));
    if (const char* env = getenv("Q27_METAL_GPU_SAMPLE"); env && *env)
        gpu_sample_ = strtoul(env, nullptr, 10) != 0;
    if (const char* env = getenv("Q27_METAL_RESIDENT"); env && *env)
        resident_ = strtoul(env, nullptr, 10) != 0;
    if (has_mtp_) {
        mtp_embed_norm_ = alloc_f32(N_EMBD); mtp_hidden_norm_ = alloc_f32(N_EMBD);
        mtp_concat_ = alloc_f32(2 * N_EMBD); mtp_x_ = alloc_f32(N_EMBD);
        mtp_hidden_out_ = alloc_f32(N_EMBD);
        const uint64_t mtp_cache_bytes = (uint64_t)max_context_ * cache_row_bytes;
        mtp_k_cache_ = backend_.allocate(mtp_cache_bytes); mtp_v_cache_ = backend_.allocate(mtp_cache_bytes);
    }
    q5120_=backend_.allocate_quantized(N_EMBD); q6144_=backend_.allocate_quantized(GDN_V);
    q10240_=backend_.allocate_quantized(GDN_CH); q17408_=backend_.allocate_quantized(N_FFN);

    gqa_partials_ = backend_.allocate_private(partial_bytes);

    if (kv_fp16_except_)
        for (uint32_t li = 0; li < 16; li++)
            for (uint32_t h = 0; h < 4; h++)
                if (kv_fp16_head_masks_[li] & (1u << h))
                    kv_fp16_side_[li].push_back(KvFp16Side{
                        h,
                        backend_.allocate_private((uint64_t)max_context_ * HEAD_DIM * 2),
                        backend_.allocate_private((uint64_t)max_context_ * HEAD_DIM * 2)});

    // Layer-major chunked prefill routes every projection through
    // activation-quantized simdgroup GEMM. Official Q4/Q8 models use that
    // contract; Bonsai T2/B1/mixed serial projection deliberately keeps
    // float activations, so it must remain serial until an equivalent
    // batched float-activation path exists.
    chunked_prefill_ = chunk_capable;
    if (chunked_prefill_) {
        ch_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * N_EMBD);
        cx1_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * N_EMBD);
        cy_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * N_EMBD);
        cqg_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * 2 * N_HEAD * HEAD_DIM);
        ckbuf_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * N_KV * HEAD_DIM);
        cvbuf_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * N_KV * HEAD_DIM);
        cattn_out_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * N_HEAD * HEAD_DIM);
        cqkv_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * GDN_CH);
        cz_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * GDN_V);
        calpha_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * GDN_HEADS);
        cbeta_raw_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * GDN_HEADS);
        cg_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * GDN_HEADS);
        cbeta_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * GDN_HEADS);
        cconv_out_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * GDN_CH);
        cdelta_out_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * GDN_V);
        cgated_out_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * GDN_V);
        cffn_gate_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * N_FFN);
        cffn_up_ = alloc_f32((uint64_t)PREFILL_CHUNK_MAX * N_FFN);
        cq5120_ = backend_.allocate_quantized(PREFILL_CHUNK_MAX * N_EMBD);
        cq6144_ = backend_.allocate_quantized(PREFILL_CHUNK_MAX * GDN_V);
        cq17408_ = backend_.allocate_quantized(PREFILL_CHUNK_MAX * N_FFN);
        // Verify-width surfaces (lever 2): sized for VERIFY_CHUNK_MAX so
        // oracle/verify rounds can run past the width-12 NLL/KL contract;
        // the teacher-force paths keep slicing at CHUNK_MAX regardless.
        cfinal_ = alloc_f32((uint64_t)VERIFY_CHUNK_MAX * N_EMBD);
        clogits_ = alloc_f32((uint64_t)VERIFY_CHUNK_MAX * VOCAB);
        cpred_ = backend_.allocate((uint64_t)VERIFY_CHUNK_MAX * sizeof(uint32_t));
        ctargets_ = backend_.allocate((uint64_t)CHUNK_MAX * sizeof(uint32_t));
        cnll_ = alloc_f32(CHUNK_MAX);
        // Batched MTP verification parks each GDN layer's chunk inputs
        // (~24 MiB total) so acceptance can replay the recurrence over the
        // accepted prefix, and discards speculative state commits into two
        // slots shared by every layer. All of it stays physically lazy until
        // the first batched MTP round.
        const uint32_t gdn_layers = N_LAYER - N_LAYER / 4;
        park_qkv_.reserve(gdn_layers); park_g_.reserve(gdn_layers); park_beta_.reserve(gdn_layers);
        for (uint32_t i = 0; i < gdn_layers; i++) {
            park_qkv_.push_back(alloc_f32((uint64_t)VERIFY_CHUNK_MAX * GDN_CH));
            park_g_.push_back(alloc_f32((uint64_t)VERIFY_CHUNK_MAX * GDN_HEADS));
            park_beta_.push_back(alloc_f32((uint64_t)VERIFY_CHUNK_MAX * GDN_HEADS));
        }
        discard_recurrent_ = alloc_f32((uint64_t)GDN_HEADS * GDN_DIM * GDN_DIM);
        discard_ring_ = alloc_f32((uint64_t)3 * GDN_CH);
    }
    reset();
    reservation.committed = true;
}

void MetalEngine::set_chunked_prefill(bool enabled) {
    if (enabled && !has_mtp_)
        throw std::runtime_error("q27 Metal: Bonsai models require serial float-activation prefill");
    if (enabled && !ch_)
        throw std::runtime_error("q27 Metal: chunked prefill requires quantized matmul support");
    chunked_prefill_ = enabled;
}

void MetalEngine::set_kv_attrib(uint32_t mode) {
    // Mode 4 (fp8-KV control arm): e4m3 round-trip of BOTH sides, every
    // head — production-exact for a transform-free codec. Mode 3 is set
    // via set_kv_attrib_except only (it needs the cell masks).
    if (mode > 2 && mode != 4)
        throw std::runtime_error("q27 Metal: KV attribution mode must be 0 (off), 1 (K), 2 (V), or 4 (e4m3 both sides)");
    if (mode && turbo3_kv_)
        throw std::runtime_error("q27 Metal: KV attribution requires an fp16-KV engine (drop --kv turbo3)");
    // Any change after rows are cached — including turning attribution off
    // or widening a cell back to all layers/heads — would leave a mixed
    // cache behind position_ (codex P2).
    if (position_ && (mode != kv_attrib_ ||
                      kv_attrib_layer_ != UINT32_MAX || kv_attrib_head_ != UINT32_MAX))
        throw std::runtime_error("q27 Metal: set KV attribution before encoding any tokens");
    kv_attrib_ = mode;
    kv_attrib_layer_ = UINT32_MAX;
    kv_attrib_head_ = UINT32_MAX;
    kv_attrib_flags_ = 0;
    kv_attrib_aux_.reset();
}

void MetalEngine::set_kv_attrib_rt(bool scale32, const float* feature_scales) {
    // Side arms (1/2) only: modes 3/4 have no turbo3 scale to modify.
    if (!kv_attrib_ || (kv_attrib_flags_ & 4u) || kv_attrib_ >= 3)
        throw std::runtime_error("q27 Metal: KV round-trip modifiers need a side arm (set_kv_attrib first)");
    if (position_)
        throw std::runtime_error("q27 Metal: set KV attribution before encoding any tokens");
    kv_attrib_flags_ = (scale32 ? 1u : 0u) | (feature_scales ? 2u : 0u);
    if (feature_scales) {
        kv_attrib_aux_ = backend_.allocate(2ull * 16 * 4 * 256 * 4);
        backend_.write(*kv_attrib_aux_, 0, feature_scales, 2ull * 16 * 4 * 256 * 4);
    }
}

void MetalEngine::set_kv_attrib_stats() {
    if (turbo3_kv_)
        throw std::runtime_error("q27 Metal: KV attribution requires an fp16-KV engine (drop --kv turbo3)");
    if (position_)
        throw std::runtime_error("q27 Metal: set KV attribution before encoding any tokens");
    kv_attrib_ = 1;              // ignored by the STATS branch; enables routing
    kv_attrib_layer_ = UINT32_MAX;
    kv_attrib_head_ = UINT32_MAX;
    kv_attrib_flags_ = 4u;
    kv_attrib_aux_ = backend_.allocate(2ull * 16 * 4 * 256 * 4);
    backend_.zero(*kv_attrib_aux_);
}

void MetalEngine::read_kv_attrib_stats(std::vector<float>& out) {
    if (!(kv_attrib_flags_ & 4u) || !kv_attrib_aux_)
        throw std::runtime_error("q27 Metal: no KV attribution stats pass is active");
    backend_.synchronize();
    out.resize(2ull * 16 * 4 * 256);
    backend_.read(*kv_attrib_aux_, 0, out.data(), out.size() * 4);
}

void MetalEngine::set_kv_attrib_cell(uint32_t mode, uint32_t layer, uint32_t head) {
    if (mode != 1 && mode != 2)
        throw std::runtime_error("q27 Metal: KV attribution cell mode must be 1 (K) or 2 (V)");
    if (turbo3_kv_)
        throw std::runtime_error("q27 Metal: KV attribution requires an fp16-KV engine (drop --kv turbo3)");
    if (layer != UINT32_MAX && (layer >= 64 || layer % 4 != 3))
        throw std::runtime_error("q27 Metal: KV attribution layer must be an attention layer (layer%4==3)");
    if (head != UINT32_MAX && head >= N_KV)
        throw std::runtime_error("q27 Metal: KV attribution head out of range");
    if (position_ && (mode != kv_attrib_ || layer != kv_attrib_layer_ || head != kv_attrib_head_))
        throw std::runtime_error("q27 Metal: set KV attribution before encoding any tokens");
    kv_attrib_ = mode;
    kv_attrib_layer_ = layer;
    kv_attrib_head_ = head;
    kv_attrib_flags_ = 0;
    kv_attrib_aux_.reset();
}

void MetalEngine::set_kv_attrib_except(const uint32_t* cells, size_t n) {
    if (turbo3_kv_)
        throw std::runtime_error("q27 Metal: KV attribution requires an fp16-KV engine (drop --kv turbo3)");
    if (position_)
        throw std::runtime_error("q27 Metal: set KV attribution before encoding any tokens");
    kv_attrib_ = 3;
    kv_attrib_layer_ = UINT32_MAX;
    kv_attrib_head_ = UINT32_MAX;
    kv_attrib_flags_ = 0;
    kv_attrib_aux_.reset();
    std::memset(kv_attrib_masks_, 0, sizeof kv_attrib_masks_);
    for (size_t i = 0; i < n; i++) {
        if (cells[i] >= 128)
            throw std::runtime_error("q27 Metal: KV exception cell must be 0..127");
        kv_attrib_masks_[cells[i] >> 3] |= uint8_t(1u << (cells[i] & 7u));
    }
}

void MetalEngine::reset() {
    position_ = 0;
    logits_resident_ = false;
    for (LayerState& layer : layers_) {
        if (layer.recurrent) backend_.zero(*layer.recurrent);
        if (layer.ring) backend_.zero(*layer.ring);
        // KV rows are written before they become visible through position_;
        // clearing the full reserved context would make long-context reset O(ctx).
    }
}

// G6 admission accounting. snapshot_bytes mirrors capture_state()'s
// allocations at worst case (position_ == max_context_); fixed_state_bytes
// mirrors the constructor's non-KV buffers (the >= 1 MB class; scalar-sized
// allocations omitted), and lazy worst-case buffers (mask pool, wide head
// stage, top-k staging) are charged as if populated — conservative
// admission (k3 audit B6/E7); gqa_partial_peak mirrors the constructor's own
// per-engine partials allocation (audit E2). Keep paired with those sites.
uint64_t MetalEngine::snapshot_bytes() const {
    const uint64_t cache_row = turbo3_kv_ ? (uint64_t)N_KV * 2 * 50
                                          : (uint64_t)N_KV * HEAD_DIM * 2;
    const uint64_t active = (uint64_t)max_context_ * cache_row;
    const uint64_t attn_layers = N_LAYER / 4, gdn_layers = N_LAYER - attn_layers;
    uint64_t bytes = gdn_layers * ((uint64_t)GDN_HEADS * GDN_DIM * GDN_DIM + 3ull * GDN_CH) * 4;
    bytes += attn_layers * 2 * active;
    if (has_mtp_) bytes += 2 * active;
    bytes += (uint64_t)N_EMBD * 4 + (uint64_t)VOCAB * 4;   // hidden + logits
    // Exception side rows (snapshot v2): K + V fp16 per masked head.
    uint64_t side_entries = 0;
    for (const auto& sides : kv_fp16_side_) side_entries += sides.size();
    bytes += side_entries * 2ull * max_context_ * HEAD_DIM * 2;
    return bytes;
}

uint64_t MetalEngine::fixed_state_bytes(bool chunked) {
    const uint64_t attn_layers = N_LAYER / 4, gdn_layers = N_LAYER - attn_layers;
    // Chunked-prefill f32 rows (ch/cx1/cy, cqg, ckbuf/cvbuf, cattn_out,
    // cqkv, cz, alpha/beta_raw/g/beta, cconv_out, cdelta_out, cgated_out,
    // ffn gate+up).
    const uint64_t chunk_row = (uint64_t)N_EMBD * 3 + 2ull * N_HEAD * HEAD_DIM +
                               2ull * N_KV * HEAD_DIM + (uint64_t)N_HEAD * HEAD_DIM +
                               GDN_CH + GDN_V + 4ull * GDN_HEADS + GDN_CH + GDN_V + GDN_V +
                               2ull * N_FFN;
    // Live GDN recurrence state (recurrent + conv ring per GDN layer) +
    // serial-path logits + hidden/scratch rows: allocated on every device.
    uint64_t bytes = gdn_layers * ((uint64_t)GDN_HEADS * GDN_DIM * GDN_DIM + 3ull * GDN_CH) * 4;
    bytes += (uint64_t)VOCAB * 4 + (uint64_t)N_EMBD * 4 * 4;
    // Constraint-mask pool at capacity (lazy in mask_pool_add; ~2.0 MB full)
    // + top-k sampling staging: allocated regardless of chunked (k3 audit B6/E7).
    bytes += (((uint64_t)VOCAB + 31) / 32) * 4 * MASK_POOL_CAP;
    bytes += (uint64_t)TOPK_CAPACITY * (4 + 4) + 4;
    if (!chunked) return bytes;   // pre-Apple7: no chunk/verify/replay buffers
    bytes += (uint64_t)PREFILL_CHUNK_MAX * chunk_row * 4;
    // Quantized activation copies (int8 values + f32 scales per 32).
    bytes += (uint64_t)PREFILL_CHUNK_MAX * (N_EMBD + GDN_V + N_FFN) * 9 / 8;
    // Verify-width surfaces (lever 2): cfinal_ + clogits_.
    bytes += (uint64_t)VERIFY_CHUNK_MAX * ((uint64_t)N_EMBD + VOCAB) * 4;
    // gdn_replay parks (qkv + g + beta per GDN layer).
    bytes += gdn_layers * (uint64_t)VERIFY_CHUNK_MAX * (GDN_CH + 2ull * GDN_HEADS) * 4;
    // Verify-chunk discard state slots (one shared pair per engine).
    bytes += ((uint64_t)GDN_HEADS * GDN_DIM * GDN_DIM + 3ull * GDN_CH) * 4;
    // Wide-head staging (lazy in teacher_force_logits_wide; k3 audit B6/E7).
    bytes += (uint64_t)CHUNK_MAX * N_EMBD * 4;
    return bytes;
}

uint64_t MetalEngine::gqa_partial_peak(uint32_t context, uint32_t block, bool chunked) {
    const uint64_t b = std::max(block, 1u);
    const uint64_t blocks = 1 + ((uint64_t)std::max(context, 1u) - 1) / b;
    // Without chunked prefill the causal-GQA path only ever sees one query
    // token (serial decode), so the widest partial buffer is one row.
    // Allocated eagerly per engine (audit E2), so there is no transient
    // allocate-then-replace coexistence to double-charge anymore.
    const uint64_t tokens = chunked ? PREFILL_CHUNK_MAX : 1;
    return tokens * N_HEAD * blocks * 258 * 4;
}

std::shared_ptr<MetalEngine::Snapshot> MetalEngine::capture_state() {
    backend_.synchronize();
    auto snapshot = std::make_shared<Snapshot>();
    snapshot->owner = this; snapshot->position = position_;
    snapshot->logits_resident = logits_resident_; snapshot->layers.resize(N_LAYER);
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
    if(active_cache && mtp_k_cache_) {
        snapshot->mtp_k_cache=backend_.allocate(active_cache); snapshot->mtp_v_cache=backend_.allocate(active_cache);
        backend_.copy(*mtp_k_cache_,0,*snapshot->mtp_k_cache,0,active_cache);
        backend_.copy(*mtp_v_cache_,0,*snapshot->mtp_v_cache,0,active_cache);
    }
    snapshot->hidden=backend_.allocate(x1_->size()); backend_.copy(*x1_,0,*snapshot->hidden,0,x1_->size());
    snapshot->logits=backend_.allocate(logits_->size()); backend_.copy(*logits_,0,*snapshot->logits,0,logits_->size());
    // Exception side rows ride the snapshot (v2): copy() is a GPU kernel,
    // so the private side caches are reachable; destinations are ordinary
    // shared buffers like every other snapshot blob.
    const uint64_t side_active=(uint64_t)position_*HEAD_DIM*2;
    if(kv_fp16_except_ && side_active)
        for(uint32_t li=0;li<16;li++)
            for(const KvFp16Side& side : kv_fp16_side_[li]) {
                auto k=backend_.allocate(side_active), v=backend_.allocate(side_active);
                backend_.copy(*side.k,0,*k,0,side_active);
                backend_.copy(*side.v,0,*v,0,side_active);
                snapshot->kv_side.push_back(std::move(k));
                snapshot->kv_side.push_back(std::move(v));
            }
    batch.finish();
    return snapshot;
}

void MetalEngine::restore_state(const Snapshot& snapshot) {
    if(snapshot.owner!=this || snapshot.layers.size()!=N_LAYER || snapshot.position>max_context_)
        throw std::runtime_error("q27 Metal: incompatible state snapshot");
    // Side-row bookkeeping must agree with the engine's exception config
    // before any GPU write: a snapshot without side rows cannot serve an
    // exception engine at position > 0 (the masked heads' fp16 history
    // would be stale — the exact 41705bb P2 recombination hazard).
    uint64_t side_entries=0;
    for(const auto& sides : kv_fp16_side_) side_entries+=sides.size();
    const uint64_t side_expected=snapshot.position?2*side_entries:0;
    if(snapshot.kv_side.size()!=side_expected)
        throw std::runtime_error("q27 Metal: incompatible state snapshot (KV fp16 exception side rows)");
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
    if(!snapshot.kv_side.empty()) {
        const uint64_t side_active=(uint64_t)snapshot.position*HEAD_DIM*2;
        size_t si=0;
        for(uint32_t li=0;li<16;li++)
            for(KvFp16Side& side : kv_fp16_side_[li]) {
                backend_.copy(*snapshot.kv_side[si++],0,*side.k,0,side_active);
                backend_.copy(*snapshot.kv_side[si++],0,*side.v,0,side_active);
            }
    }
    batch.finish();
    position_=snapshot.position;
    logits_resident_ = snapshot.logits_resident;
}

// ---- Prefix snapshots to disk (docs/metal/plans/2026-07-16-prefix-snapshots.md).
// Format Q27SNAP1 (LE): magic, artifact identity (file size + SHA1 of the
// first 64 KB), kv dtype, position, token metadata, then length-prefixed
// blobs in capture_state() order. Plain read/write, never mmap.
//
// KV fp16 exception extension (snapshot v2, docs/plans/2026-07-17-kv-
// except-snapshot-v2.md): header reserved bit 1 marks its presence
// (bit 0 remains !logits_resident). After the standard blobs: one
// length-prefixed 16-byte head-mask blob (the snapshot-config identity —
// side blob LENGTHS alone cannot distinguish cell lists of equal size),
// then per masked head the K and V side blobs in capture_state() side
// order. Presence and mask content must both match the loading engine
// exactly; either mismatch is a loud pass-1 reject. Pre-v2 binaries
// reject extended files via the trailing-bytes check.

namespace {

struct SnapshotHeader {
    char magic[8];
    uint64_t artifact_size;
    unsigned char artifact_sha1[20];
    uint32_t kv_dtype;      // 0 fp16, 1 turbo3
    uint32_t position;
    uint32_t token_count;
    uint32_t reserved;
    unsigned char prefix_sha1[20];   // Phase 2 server keying; zeros in Phase 1
};
constexpr char SNAP_MAGIC[8] = {'Q','2','7','S','N','A','P','1'};

void snap_write(FILE* f, const void* data, size_t bytes, const std::string& path) {
    if (fwrite(data, 1, bytes, f) != bytes)
        throw std::runtime_error("q27 Metal: short write to snapshot: " + path);
}

void snap_read(FILE* f, void* data, size_t bytes, const std::string& path) {
    if (fread(data, 1, bytes, f) != bytes)
        throw std::runtime_error("q27 Metal: truncated snapshot: " + path);
}

} // namespace

// SHA1 over the whole mapped artifact — the bytes this engine actually
// computes with. Chunked updates (CC_LONG is 32-bit); cached per Shared.
const unsigned char* MetalEngine::snapshot_identity() {
    if (!shared_->snap_sha_ready) {
        CC_SHA1_CTX ctx;
        CC_SHA1_Init(&ctx);
        const unsigned char* base = (const unsigned char*)model_.mapping_base();
        const uint64_t total = model_.mapping_size();
        if (!base || !total)
            throw std::runtime_error("q27 Metal: artifact mapping unavailable for snapshot identity");
        for (uint64_t off = 0; off < total; off += 256u << 20)
            CC_SHA1_Update(&ctx, base + off, (CC_LONG)std::min<uint64_t>(256u << 20, total - off));
        CC_SHA1_Final(shared_->snap_sha1, &ctx);
        shared_->snap_sha_ready = true;
    }
    return shared_->snap_sha1;
}

void MetalEngine::save_state(const std::string& path, const uint32_t* tokens,
                             uint32_t token_count, bool logits_resident) {
    if (token_count && !tokens)
        throw std::runtime_error("q27 Metal: snapshot token metadata is null");
    if (logits_resident && !logits_resident_)
        throw std::runtime_error("q27 Metal: cannot save snapshot with stale logits marked resident");
    backend_.synchronize();
    const uint64_t cache_row = turbo3_kv_ ? (uint64_t)N_KV * 2 * 50
                                          : (uint64_t)N_KV * HEAD_DIM * 2;
    const uint64_t active_cache = (uint64_t)position_ * cache_row;
    std::string tmp_pattern = path + ".tmp.XXXXXX";
    std::vector<char> tmp_name(tmp_pattern.begin(), tmp_pattern.end());
    tmp_name.push_back('\0');
    // Each writer gets a private same-directory inode. mkstemp's O_EXCL
    // creation prevents symlink redirection and avoids the old shared `.tmp`
    // unlink race when two server processes publish the same snapshot key.
    const int tmp_fd = mkstemp(tmp_name.data());
    const std::string tmp = tmp_fd >= 0 ? tmp_name.data() : tmp_pattern;
    if (tmp_fd < 0 || fcntl(tmp_fd, F_SETFD, FD_CLOEXEC) != 0 ||
        fchmod(tmp_fd, 0600) != 0) {
        if (tmp_fd >= 0) close(tmp_fd);
        (void)unlink(tmp.c_str());
        throw std::runtime_error("q27 Metal: cannot create private snapshot: " + tmp);
    }
    int lock_fd = -1;
    if (flock(tmp_fd, LOCK_EX | LOCK_NB) != 0 ||
        (lock_fd = dup(tmp_fd)) < 0 ||
        fcntl(lock_fd, F_SETFD, FD_CLOEXEC) != 0) {
        if (lock_fd >= 0) close(lock_fd);
        close(tmp_fd);
        (void)unlink(tmp.c_str());
        throw std::runtime_error("q27 Metal: cannot lock private snapshot: " + tmp);
    }
    FILE* f = fdopen(tmp_fd, "wb");
    if (!f) {
        close(lock_fd);
        close(tmp_fd);
        (void)unlink(tmp.c_str());
        throw std::runtime_error("q27 Metal: cannot create snapshot: " + tmp);
    }
    std::vector<unsigned char> stage(16u << 20);
    try {
        SnapshotHeader h{};
        memcpy(h.magic, SNAP_MAGIC, sizeof h.magic);
        h.artifact_size = model_.mapping_size();
        memcpy(h.artifact_sha1, snapshot_identity(), sizeof h.artifact_sha1);
        h.kv_dtype = turbo3_kv_ ? 1 : 0;
        h.position = position_;
        h.token_count = token_count;
        h.reserved = (logits_resident ? 0u : 1u) | (kv_fp16_except_ ? 2u : 0u)
                   | (kv_fp16_side_codec_ ? 4u : 0u);
        snap_write(f, &h, sizeof h, tmp);
        if (token_count) snap_write(f, tokens, (size_t)token_count * 4, tmp);
        auto put_blob = [&](const BackendBuffer* src, uint64_t bytes) {
            snap_write(f, &bytes, sizeof bytes, tmp);
            for (uint64_t off = 0; off < bytes; off += stage.size()) {
                const uint64_t n = std::min<uint64_t>(stage.size(), bytes - off);
                backend_.read(*src, off, stage.data(), n);
                snap_write(f, stage.data(), n, tmp);
            }
        };
        for (uint32_t i = 0; i < N_LAYER; i++) {
            const LayerState& s = layers_[i];
            put_blob(s.recurrent.get(), s.recurrent ? s.recurrent->size() : 0);
            put_blob(s.ring.get(), s.ring ? s.ring->size() : 0);
            put_blob(s.k_cache.get(), s.k_cache ? active_cache : 0);
            put_blob(s.v_cache.get(), s.v_cache ? active_cache : 0);
        }
        put_blob(mtp_k_cache_.get(), mtp_k_cache_ ? active_cache : 0);
        put_blob(mtp_v_cache_.get(), mtp_v_cache_ ? active_cache : 0);
        put_blob(x1_.get(), x1_->size());
        put_blob(logits_.get(), logits_->size());
        if (kv_fp16_except_) {
            // Head-mask blob, then the private side caches bounced through
            // one shared staging buffer (copy() is the only host path that
            // can source a StorageModePrivate buffer).
            const uint64_t mask_bytes = sizeof kv_fp16_head_masks_;
            snap_write(f, &mask_bytes, sizeof mask_bytes, tmp);
            snap_write(f, kv_fp16_head_masks_, mask_bytes, tmp);
            auto staging = backend_.allocate(stage.size());
            auto put_side_blob = [&](const BackendBuffer& src, uint64_t bytes) {
                snap_write(f, &bytes, sizeof bytes, tmp);
                for (uint64_t off = 0; off < bytes; off += stage.size()) {
                    const uint64_t n = std::min<uint64_t>(stage.size(), bytes - off);
                    backend_.copy(src, off, *staging, 0, n);
                    backend_.read(*staging, 0, stage.data(), n);
                    snap_write(f, stage.data(), n, tmp);
                }
            };
            const uint64_t side_active = (uint64_t)position_ * HEAD_DIM * 2;
            for (uint32_t li = 0; li < 16; li++)
                for (const KvFp16Side& side : kv_fp16_side_[li]) {
                    put_side_blob(*side.k, side_active);
                    put_side_blob(*side.v, side_active);
                }
            // Codec trailer, e4m3 sides only (codex P2 on d9ee75a): a
            // reserved bit alone cannot stop a PRE-codec binary from
            // silently continuing an e4m3 history with fp16 stores — this
            // extra blob trips that binary's own trailing-bytes check
            // loudly. fp16-side files carry no trailer, so they stay
            // loadable across the version boundary.
            if (kv_fp16_side_codec_) {
                const uint64_t codec_bytes = sizeof kv_fp16_side_codec_;
                snap_write(f, &codec_bytes, sizeof codec_bytes, tmp);
                snap_write(f, &kv_fp16_side_codec_, codec_bytes, tmp);
            }
        }
        // Test-only failpoints for the snapshot crash gate
        // (tools/snapshot_gate.sh); read fresh each save.
        const char* snap_crash = getenv("Q27_METAL_SNAP_CRASH");
        if (snap_crash && strcmp(snap_crash, "before-fsync") == 0) _exit(42);
        // rename gives atomicity, fsync gives content durability: a crash
        // without it can leave a structurally-valid file whose data pages
        // read back as zeroes — every blob length still matches, so pass-1
        // validation cannot catch it (k3 audit B1).
        // fflush/fsync failures must still reach fclose — a short-circuit
        // chain leaks the descriptor on every failed save (codex P3).
        if (fflush(f) != 0 || fsync(fileno(f)) != 0) {
            fclose(f);
            f = nullptr;
            throw std::runtime_error("q27 Metal: cannot finish snapshot: " + tmp);
        }
        if (fclose(f) != 0) {
            f = nullptr;
            throw std::runtime_error("q27 Metal: cannot finish snapshot: " + tmp);
        }
        f = nullptr;
        // The rename lives in the directory entry: without a directory fsync
        // a crash can drop it after this call reported success. The snapshot
        // is a correctness-bearing artifact, so failure here is fatal, never
        // advisory (k3 audit B1). The directory opens BEFORE the rename so an
        // open failure aborts while the previous snapshot is still in place
        // (codex P3); a post-rename fsync failure leaves the new file in
        // place — its content is already durable, only the rename's
        // durability is uncertain, and either name resolves to a valid
        // snapshot after a crash.
        const std::string::size_type slash = path.find_last_of('/');
        const std::string dir = slash == std::string::npos ? "."
                              : slash == 0 ? "/" : path.substr(0, slash);
        const int dfd = open(dir.c_str(), O_RDONLY);
        if (dfd < 0)
            throw std::runtime_error("q27 Metal: cannot open snapshot directory: " + dir);
        if (rename(tmp.c_str(), path.c_str()) != 0) {
            close(dfd);
            throw std::runtime_error("q27 Metal: cannot move snapshot into place: " + path);
        }
        if (fsync(dfd) != 0) {
            close(dfd);
            throw std::runtime_error("q27 Metal: cannot sync snapshot directory: " + dir);
        }
        close(dfd);
        close(lock_fd);
        lock_fd = -1;
        if (snap_crash && strcmp(snap_crash, "after-rename") == 0) _exit(42);
    } catch (...) {
        if (f) fclose(f);
        if (lock_fd >= 0) close(lock_fd);
        remove(tmp.c_str());
        throw;
    }
}

uint32_t MetalEngine::load_state(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0)
        throw std::runtime_error("q27 Metal: cannot open snapshot: " + path);
    try {
        const uint32_t position = load_state_fd(fd, path);
        close(fd);
        return position;
    } catch (...) {
        close(fd);
        throw;
    }
}

uint32_t MetalEngine::load_state_fd(int source_fd, const std::string& path) {
    const int snap_fd = fcntl(source_fd, F_DUPFD_CLOEXEC, 0);
    if (snap_fd < 0 || lseek(snap_fd, 0, SEEK_SET) < 0) {
        if (snap_fd >= 0) close(snap_fd);
        throw std::runtime_error("q27 Metal: cannot pin snapshot: " + path);
    }
    FILE* f = fdopen(snap_fd, "rb");
    if (!f) {
        close(snap_fd);
        throw std::runtime_error("q27 Metal: cannot read snapshot: " + path);
    }
    try {
        SnapshotHeader h{};
        snap_read(f, &h, sizeof h, path);
        if (memcmp(h.magic, SNAP_MAGIC, sizeof h.magic) != 0)
            throw std::runtime_error("q27 Metal: not a q27 snapshot: " + path);
        if (h.artifact_size != model_.mapping_size() ||
            memcmp(h.artifact_sha1, snapshot_identity(), 20) != 0)
            throw std::runtime_error("q27 Metal: snapshot was taken against a different artifact: " + path);
        if (h.kv_dtype != (turbo3_kv_ ? 1u : 0u))
            throw std::runtime_error("q27 Metal: snapshot KV dtype does not match this engine: " + path);
        if (h.position > max_context_)
            throw std::runtime_error("q27 Metal: snapshot position exceeds this engine's context: " + path);
        // KV fp16 exception extension presence must match this engine's
        // config exactly (v2): an env-unset snapshot has no fp16 history
        // for the masked heads, and an exception snapshot's continuation
        // assumed fp16 where a plain engine would have quantized.
        if (bool(h.reserved & 2) != kv_fp16_except_)
            throw std::runtime_error(kv_fp16_except_
                ? "q27 Metal: snapshot carries no KV fp16 exception side rows but this engine needs them (Q27_METAL_KV_FP16_CELLS): " + path
                : "q27 Metal: snapshot carries KV fp16 exception side rows but this engine has none: " + path);
        // Side codec is config identity too (bit 2): an fp16-side snapshot
        // continued under e4m3 stores (or vice versa) would mix codec
        // histories silently. Binaries older than the hot-cells arm ignore
        // this bit — recorded cross-version caveat.
        if (bool(h.reserved & 4) != (kv_fp16_side_codec_ != 0))
            throw std::runtime_error("q27 Metal: snapshot side-cache codec does not match this engine (Q27_METAL_KV_CELLS_CODEC): " + path);
        if (fseeko(f, (off_t)h.token_count * 4, SEEK_CUR) != 0)
            throw std::runtime_error("q27 Metal: truncated snapshot: " + path);
        const uint64_t cache_row = turbo3_kv_ ? (uint64_t)N_KV * 2 * 50
                                              : (uint64_t)N_KV * HEAD_DIM * 2;
        const uint64_t active_cache = (uint64_t)h.position * cache_row;
        // The expected blob sequence, mirrored from save_state. Validating
        // every length (pass 1) before the first GPU write (pass 2) means a
        // rejected file never leaves partially-restored state. Contract
        // (unchanged since Phase 1, restated at the codex round on 4415c53):
        // pass-2 failures — mid-stream I/O errors or same-inode mutation
        // caught by the TOCTOU re-checks — DO leave indeterminate buffer
        // state behind the old position_; callers must reset() or restore a
        // known state before reuse, as the server's disk-tier fallback does.
        // Kinds: Std
        // streams into a shared buffer; Mask is host data whose CONTENT is
        // validated in pass 1 (equal-length cell lists differ only there);
        // Side bounces through staging into a private buffer.
        struct BlobRef { BackendBuffer* buf; uint64_t bytes; enum Kind { Std, Mask, Side, Codec } kind; };
        std::vector<BlobRef> blobs;
        for (uint32_t i = 0; i < N_LAYER; i++) {
            LayerState& d = layers_[i];
            blobs.push_back({d.recurrent.get(), d.recurrent ? d.recurrent->size() : 0, BlobRef::Std});
            blobs.push_back({d.ring.get(), d.ring ? d.ring->size() : 0, BlobRef::Std});
            blobs.push_back({d.k_cache.get(), d.k_cache ? active_cache : 0, BlobRef::Std});
            blobs.push_back({d.v_cache.get(), d.v_cache ? active_cache : 0, BlobRef::Std});
        }
        blobs.push_back({mtp_k_cache_.get(), mtp_k_cache_ ? active_cache : 0, BlobRef::Std});
        blobs.push_back({mtp_v_cache_.get(), mtp_v_cache_ ? active_cache : 0, BlobRef::Std});
        blobs.push_back({x1_.get(), x1_->size(), BlobRef::Std});
        blobs.push_back({logits_.get(), logits_->size(), BlobRef::Std});
        if (kv_fp16_except_) {
            blobs.push_back({nullptr, sizeof kv_fp16_head_masks_, BlobRef::Mask});
            const uint64_t side_active = (uint64_t)h.position * HEAD_DIM * 2;
            for (uint32_t li = 0; li < 16; li++)
                for (KvFp16Side& side : kv_fp16_side_[li]) {
                    blobs.push_back({side.k.get(), side_active, BlobRef::Side});
                    blobs.push_back({side.v.get(), side_active, BlobRef::Side});
                }
            // e4m3 sides carry a codec trailer (see save_state); its
            // absence/presence is already pinned by reserved bit 2, its
            // CONTENT is checked like the mask blob's.
            if (kv_fp16_side_codec_)
                blobs.push_back({nullptr, sizeof kv_fp16_side_codec_, BlobRef::Codec});
        }
        auto check_mask = [&]() {
            uint8_t stored_masks[sizeof kv_fp16_head_masks_];
            snap_read(f, stored_masks, sizeof stored_masks, path);
            if (memcmp(stored_masks, kv_fp16_head_masks_, sizeof stored_masks) != 0)
                throw std::runtime_error("q27 Metal: snapshot KV fp16 exception cells do not match this engine (Q27_METAL_KV_FP16_CELLS): " + path);
        };
        auto check_codec = [&]() {
            uint32_t stored_codec = 0;
            snap_read(f, &stored_codec, sizeof stored_codec, path);
            if (stored_codec != kv_fp16_side_codec_)
                throw std::runtime_error("q27 Metal: snapshot side-cache codec does not match this engine (Q27_METAL_KV_CELLS_CODEC): " + path);
        };
        const off_t blob_start = ftello(f);
        // Real file size up front: fseeko past EOF succeeds silently, so the
        // walk below could otherwise bless a file truncated inside its FINAL
        // blob and pass 2 would partially restore (codex P2 on 39d74a0).
        if (fseeko(f, 0, SEEK_END) != 0)
            throw std::runtime_error("q27 Metal: cannot read snapshot: " + path);
        const off_t file_size = ftello(f);
        if (fseeko(f, blob_start, SEEK_SET) != 0)
            throw std::runtime_error("q27 Metal: cannot rewind snapshot: " + path);
        uint64_t expected_end = (uint64_t)blob_start;
        for (const auto& blob : blobs) {
            uint64_t stored = 0;
            snap_read(f, &stored, sizeof stored, path);
            if (stored != blob.bytes)
                throw std::runtime_error("q27 Metal: snapshot blob layout does not match this engine: " + path);
            expected_end += sizeof stored + stored;
            if (expected_end > (uint64_t)file_size)
                throw std::runtime_error("q27 Metal: truncated snapshot: " + path);
            // Mask/codec content is part of pass-1 validation: a mismatch
            // must reject BEFORE pass 2 writes any standard blob.
            if (blob.kind == BlobRef::Mask) check_mask();
            else if (blob.kind == BlobRef::Codec) check_codec();
            else if (fseeko(f, (off_t)stored, SEEK_CUR) != 0)
                throw std::runtime_error("q27 Metal: truncated snapshot: " + path);
        }
        if (expected_end != (uint64_t)file_size)
            throw std::runtime_error("q27 Metal: trailing bytes after snapshot blobs: " + path);
        // Pass 2: stream the validated blobs into the live buffers.
        backend_.synchronize();
        if (fseeko(f, blob_start, SEEK_SET) != 0)
            throw std::runtime_error("q27 Metal: cannot rewind snapshot: " + path);
        std::vector<unsigned char> stage(16u << 20);
        std::shared_ptr<BackendBuffer> staging;
        if (kv_fp16_except_) staging = backend_.allocate(stage.size());
        for (const auto& blob : blobs) {
            uint64_t stored = 0;
            snap_read(f, &stored, sizeof stored, path);
            // TOCTOU: pass 1 validated a file that could have been swapped
            // since (k3 audit B3).
            if (stored != blob.bytes)
                throw std::runtime_error("q27 Metal: snapshot changed during load: " + path);
            if (blob.kind == BlobRef::Mask) { check_mask(); continue; }
            if (blob.kind == BlobRef::Codec) { check_codec(); continue; }
            for (uint64_t off = 0; off < blob.bytes; off += stage.size()) {
                const uint64_t n = std::min<uint64_t>(stage.size(), blob.bytes - off);
                snap_read(f, stage.data(), n, path);
                if (blob.kind == BlobRef::Side) {
                    // Private destination: write() cannot reach it — bounce
                    // through the shared staging buffer with copy().
                    backend_.write(*staging, 0, stage.data(), n);
                    backend_.copy(*staging, 0, *blob.buf, off, n);
                } else {
                    backend_.write(*blob.buf, off, stage.data(), n);
                }
            }
        }
        fclose(f);
        position_ = h.position;
        logits_resident_ = !(h.reserved & 1u);
        return position_;
    } catch (...) {
        fclose(f);
        throw;
    }
}

MetalEngine::SnapshotInfo MetalEngine::peek_snapshot(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0)
        throw std::runtime_error("q27 Metal: cannot open snapshot: " + path);
    try {
        SnapshotInfo info = peek_snapshot_fd(fd, path);
        close(fd);
        return info;
    } catch (...) {
        close(fd);
        throw;
    }
}

MetalEngine::SnapshotInfo MetalEngine::peek_snapshot_fd(
    int source_fd, const std::string& path) {
    const int snap_fd = fcntl(source_fd, F_DUPFD_CLOEXEC, 0);
    if (snap_fd < 0 || lseek(snap_fd, 0, SEEK_SET) < 0) {
        if (snap_fd >= 0) close(snap_fd);
        throw std::runtime_error("q27 Metal: cannot pin snapshot: " + path);
    }
    FILE* f = fdopen(snap_fd, "rb");
    if (!f) {
        close(snap_fd);
        throw std::runtime_error("q27 Metal: cannot read snapshot: " + path);
    }
    try {
        SnapshotHeader h{};
        snap_read(f, &h, sizeof h, path);
        if (memcmp(h.magic, SNAP_MAGIC, sizeof h.magic) != 0)
            throw std::runtime_error("q27 Metal: not a q27 snapshot: " + path);
        SnapshotInfo info;
        info.position = h.position;
        // Bit test, not equality: bit 1 is the v2 exception extension.
        info.logits_resident = (h.reserved & 1) == 0;
        // Bound the metadata before allocating: max context plus the one
        // legal pending emitted token used by resident agent sessions. A
        // corrupt header must not drive a multi-GB scan-path allocation.
        if (h.token_count > 262145)
            throw std::runtime_error("q27 Metal: snapshot token count exceeds context plus pending token: " + path);
        if (fseeko(f, 0, SEEK_END) != 0)
            throw std::runtime_error("q27 Metal: cannot read snapshot: " + path);
        if ((uint64_t)ftello(f) < sizeof h + (uint64_t)h.token_count * 4)
            throw std::runtime_error("q27 Metal: truncated snapshot: " + path);
        if (fseeko(f, sizeof h, SEEK_SET) != 0)
            throw std::runtime_error("q27 Metal: cannot rewind snapshot: " + path);
        info.tokens.resize(h.token_count);
        if (h.token_count)
            snap_read(f, info.tokens.data(), (size_t)h.token_count * 4, path);
        fclose(f);
        return info;
    } catch (...) {
        fclose(f);
        throw;
    }
}

uint32_t MetalEngine::pending_from_logits() {
    if (!logits_resident_)
        throw std::runtime_error("q27 Metal: snapshot has no resident logits; continue prefill before generation");
    CommandBatch batch(backend_);
    backend_.argmax(*logits_, VOCAB, *token_out_);
    batch.finish();
    uint32_t pending = 0;
    backend_.read(*token_out_, 0, &pending, sizeof pending);
    return pending;
}

// Serial-decode projection dispatch: bonsai (T2/B1) weights route to the
// float-activation select-form GEMV (exact math, no activation quantization
// — ternary/binary-tier plans, Phase 2); Q4/Q8 keep the packed-dot quantized path. Both
// operand sets are always live at the call sites: the fused rmsnorm/quantize
// kernels produce the float output and the int8 copy together.
void MetalEngine::project(const BackendTensor& w, const BackendBuffer& x_float,
                          const BackendQuantized& xq, BackendBuffer& out) {
    if (is_bonsai_dtype(w.dtype)) backend_.matvec(w, x_float, out);
    else backend_.matvec_quantized(w, xq, out);
}

void MetalEngine::project_pair(const BackendTensor& a, BackendBuffer& a_out,
                               const BackendTensor& b, BackendBuffer& b_out,
                               const BackendBuffer& x_float, const BackendQuantized& xq) {
    if (is_bonsai_dtype(a.dtype) || is_bonsai_dtype(b.dtype)) {
        project(a, x_float, xq, a_out);
        project(b, x_float, xq, b_out);
    } else {
        backend_.matvec_quantized_pair(a, a_out, b, b_out, xq);
    }
}

void MetalEngine::gdn_block(uint32_t layer) {
    project_pair(layer_weight(layer,"attn_qkv.weight"),*qkv_,
                 layer_weight(layer,"attn_gate.weight"),*z_,*x1_,q5120_);
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
    const BackendTensor& ssm_out_w = layer_weight(layer, "ssm_out.weight");
    if (!is_bonsai_dtype(ssm_out_w.dtype)) backend_.quantize(*gated_out_, q6144_);
    project(ssm_out_w, *gated_out_, q6144_, *y_);
}

void MetalEngine::attention_block(uint32_t layer, uint32_t pos) {
    project(layer_weight(layer, "attn_q.weight"), *x1_, q5120_, *qg_);
    backend_.rmsnorm_heads(*qg_, layer_weight(layer, "attn_q_norm.weight"),
                           N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS);
    project_pair(layer_weight(layer,"attn_k.weight"),*kbuf_,
                 layer_weight(layer,"attn_v.weight"),*vbuf_,*x1_,q5120_);
    backend_.rmsnorm_heads(*kbuf_, layer_weight(layer, "attn_k_norm.weight"),
                           N_KV, HEAD_DIM, HEAD_DIM, EPS);
    backend_.rope_neox(*qg_, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, pos, FREQ_BASE);
    backend_.rope_neox(*kbuf_, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, pos, FREQ_BASE);
    LayerState& state = layers_[layer];
    if (turbo3_kv_) {
        backend_.turbo_wht(*qg_, N_HEAD, 2 * HEAD_DIM, false);
        backend_.kv_store_turbo3(*kbuf_, *vbuf_, *state.k_cache, *state.v_cache, pos, N_KV);
        // fp16 exception cells: side-store the masked heads' rows in the
        // turbo3 WHT domain (kbuf/vbuf are dead after the store, so the
        // in-place transform is safe) — the window re-attention below then
        // sees exactly what the turbo3 kernel's dequant approximates, and
        // the shared inverse WHT on attn_out_ fixes its rows with the rest.
        const auto& side = kv_fp16_side_[layer / 4];
        if (!side.empty()) {
            backend_.turbo_wht(*kbuf_, N_KV, HEAD_DIM, false);
            backend_.turbo_wht(*vbuf_, N_KV, HEAD_DIM, false);
            for (const KvFp16Side& s : side)
                backend_.kv_store_f16_head_rows_side(*kbuf_, *vbuf_, s.head * HEAD_DIM,
                                                     N_KV * HEAD_DIM, *s.k, *s.v,
                                                     pos, HEAD_DIM, 1, kv_fp16_side_codec_);
        }
        backend_.attention_turbo3(*qg_, 2 * HEAD_DIM, *state.k_cache, *state.v_cache,
                                  *attn_out_, pos + 1, N_HEAD, N_KV,
                                  HEAD_DIM, 1.0f / std::sqrt((float)HEAD_DIM),
                                  gqa_partials_.get());
        for (const KvFp16Side& s : side)
            backend_.attention_f16_window(*qg_, 2 * HEAD_DIM, s.head * (N_HEAD / N_KV),
                                          *s.k, *s.v, *attn_out_, pos + 1,
                                          N_HEAD / N_KV, HEAD_DIM,
                                          1.0f / std::sqrt((float)HEAD_DIM));
        backend_.turbo_wht(*attn_out_, N_HEAD, HEAD_DIM, true);
    } else {
        if (kv_attrib_ && (kv_attrib_layer_ == UINT32_MAX || kv_attrib_layer_ == layer))
            backend_.kv_store_f16_attrib_rows(*kbuf_, *vbuf_, *state.k_cache, *state.v_cache,
                                              pos, N_KV, 1, kv_attrib_,
                                              kv_attrib_ == 3 ? kv_attrib_masks_[layer / 4]
                                                              : kv_attrib_head_,
                                              kv_attrib_flags_, (layer / 4) * 1024,
                                              kv_attrib_aux_.get());
        else
            backend_.kv_store_f16(*kbuf_, *vbuf_, *state.k_cache, *state.v_cache, pos, N_KV * HEAD_DIM);
        backend_.attention_f16(*qg_, 2 * HEAD_DIM, *state.k_cache, *state.v_cache,
                               *attn_out_, pos + 1, N_HEAD, N_KV,
                               HEAD_DIM, 1.0f / std::sqrt((float)HEAD_DIM),
                               gqa_partials_.get());
    }
    backend_.sigmoid_gate_mul(*attn_out_, *qg_, N_HEAD, HEAD_DIM);
    const BackendTensor& attn_out_w = layer_weight(layer, "attn_output.weight");
    if (!is_bonsai_dtype(attn_out_w.dtype)) backend_.quantize(*attn_out_, q6144_);
    project(attn_out_w, *attn_out_, q6144_, *y_);
}

void MetalEngine::ffn(uint32_t layer) {
    project_pair(layer_weight(layer,"ffn_gate.weight"),*ffn_gate_,
                 layer_weight(layer,"ffn_up.weight"),*ffn_up_,*x1_,q5120_);
    backend_.silu_mul(*ffn_gate_, *ffn_up_, *ffn_gate_, N_FFN);
    const BackendTensor& ffn_down_w = layer_weight(layer, "ffn_down.weight");
    if (!is_bonsai_dtype(ffn_down_w.dtype)) backend_.quantize(*ffn_gate_, q17408_);
    project(ffn_down_w, *ffn_gate_, q17408_, *y_);
}

// position_ advances at the CALL SITE after successful finish — a backend
// throw must leave the engine's host state describing only work that
// completed (k3 audit B2/E4); mtp_round/suffix_round already follow this.
// pos_offset places the row for multi-token command batches (prefill,
// resident decode): the token encodes at position_ + pos_offset.
void MetalEngine::encode_token(uint32_t token, bool produce_logits, bool token_from_device,
                               uint32_t pos_offset) {
    if (!token_from_device && token >= VOCAB) throw std::runtime_error("q27 Metal: token out of range");
    const uint32_t pos = position_ + pos_offset;
    if (pos >= max_context_) throw std::runtime_error("q27 Metal: context exhausted");
    if (token_from_device)
        backend_.embedding_from_device(weight("token_embd.weight"), *token_out_, *h_);
    else
        backend_.embedding_q8(weight("token_embd.weight"), token, *h_);
    for (uint32_t layer = 0; layer < N_LAYER; layer++) {
        backend_.rmsnorm_quantized(*h_,layer_weight(layer,"attn_norm.weight"),*x1_,N_EMBD,EPS,q5120_);
        if (attention_layer(layer)) attention_block(layer, pos); else gdn_block(layer);
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
        project(weight("output.weight"), *x1_, q5120_, *logits_);
        if (active_mask_ >= 0)
            backend_.mask_logits(*logits_, *mask_pool_,
                                 (uint64_t)active_mask_ * (((uint64_t)VOCAB + 31) / 32) * 4, VOCAB);
        backend_.argmax(*logits_, VOCAB, *token_out_);
    }
}

void MetalEngine::gdn_chunk(uint32_t layer, uint32_t count, bool verify) {
    BackendQuantized x5 = quantized_view(cq5120_, count * N_EMBD);
    backend_.matmul_quantized(layer_weight(layer, "attn_qkv.weight"), x5, count, *cqkv_);
    backend_.matmul_quantized(layer_weight(layer, "attn_gate.weight"), x5, count, *cz_);
    // Official tier: fused F16 pair-rows kernel. Bonsai tiers: alpha/beta
    // are T2/B1 matrices; their chunk GEMMs write the same [token][row] layout.
    const BackendTensor& alpha_w = layer_weight(layer, "ssm_alpha.weight");
    const BackendTensor& beta_w = layer_weight(layer, "ssm_beta.weight");
    if (is_bonsai_dtype(alpha_w.dtype)) {
        backend_.matmul_quantized(alpha_w, x5, count, *calpha_);
        backend_.matmul_quantized(beta_w, x5, count, *cbeta_raw_);
    } else {
        backend_.matvec_f16_pair_rows(alpha_w, *calpha_, beta_w, *cbeta_raw_, *cx1_, count);
    }
    backend_.gdn_gates_rows(*calpha_, *cbeta_raw_, layer_weight(layer, "ssm_a"),
                            layer_weight(layer, "ssm_dt.bias"), *cg_, *cbeta_, GDN_HEADS, count);
    LayerState& state = layers_[layer];
    // Verification parks the recurrence inputs and discards the speculative
    // state commit; gdn_replay later commits real state for the accepted
    // prefix from the parked copies — bit-identical inputs, no re-encode.
    if (verify) {
        const uint32_t slot = gdn_slot(layer);
        backend_.copy(*cqkv_, 0, *park_qkv_[slot], 0, (uint64_t)count * GDN_CH * sizeof(float));
        backend_.copy(*cg_, 0, *park_g_[slot], 0, (uint64_t)count * GDN_HEADS * sizeof(float));
        backend_.copy(*cbeta_, 0, *park_beta_[slot], 0, (uint64_t)count * GDN_HEADS * sizeof(float));
    }
    BackendBuffer& ring_dst = verify ? *discard_ring_ : *state.ring;
    BackendBuffer& recurrent_dst = verify ? *discard_recurrent_ : *state.recurrent;
    backend_.conv_chunk(*state.ring, ring_dst, *cqkv_, layer_weight(layer, "ssm_conv1d.weight"),
                        *cconv_out_, GDN_CH, count);
    backend_.l2norm_rows(*cconv_out_, 2 * GDN_QK_HEADS, GDN_DIM, GDN_CH, count, EPS);
    backend_.delta_chunk(*state.recurrent, recurrent_dst, *cconv_out_, *cg_, *cbeta_, *cdelta_out_,
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
        // fp16 exception cells: WHT-domain side store + window re-attention
        // (see the serial branch for the domain argument). ckbuf/cvbuf are
        // dead after the turbo3 store.
        const auto& side = kv_fp16_side_[layer / 4];
        if (!side.empty()) {
            backend_.turbo_wht(*ckbuf_, count * N_KV, HEAD_DIM, false);
            backend_.turbo_wht(*cvbuf_, count * N_KV, HEAD_DIM, false);
            for (const KvFp16Side& s : side)
                backend_.kv_store_f16_head_rows_side(*ckbuf_, *cvbuf_, s.head * HEAD_DIM,
                                                     N_KV * HEAD_DIM, *s.k, *s.v,
                                                     position_, HEAD_DIM, count, kv_fp16_side_codec_);
        }
        backend_.attention_turbo3_causal(*cqg_, 2 * HEAD_DIM, 2 * N_HEAD * HEAD_DIM,
                                         *state.k_cache, *state.v_cache,
                                         *cattn_out_, position_ + 1, N_HEAD, N_KV,
                                         HEAD_DIM, count, scale, gqa_partials_.get());
        for (const KvFp16Side& s : side)
            backend_.attention_f16_causal_window(*cqg_, 2 * HEAD_DIM, 2 * N_HEAD * HEAD_DIM,
                                                 s.head * (N_HEAD / N_KV), *s.k, *s.v,
                                                 *cattn_out_, N_HEAD * HEAD_DIM,
                                                 position_ + 1, N_HEAD / N_KV,
                                                 HEAD_DIM, count, scale);
        backend_.turbo_wht(*cattn_out_, count * N_HEAD, HEAD_DIM, true);
    } else {
        if (kv_attrib_ && (kv_attrib_layer_ == UINT32_MAX || kv_attrib_layer_ == layer))
            backend_.kv_store_f16_attrib_rows(*ckbuf_, *cvbuf_, *state.k_cache, *state.v_cache,
                                              position_, N_KV, count, kv_attrib_,
                                              kv_attrib_ == 3 ? kv_attrib_masks_[layer / 4]
                                                              : kv_attrib_head_,
                                              kv_attrib_flags_, (layer / 4) * 1024,
                                              kv_attrib_aux_.get());
        else
            backend_.kv_store_f16_rows(*ckbuf_, *cvbuf_, *state.k_cache, *state.v_cache,
                                       position_, N_KV * HEAD_DIM, count);
        backend_.attention_f16_causal(*cqg_, 2 * HEAD_DIM, 2 * N_HEAD * HEAD_DIM,
                                      *state.k_cache, *state.v_cache,
                                      *cattn_out_, position_ + 1, N_HEAD, N_KV,
                                      HEAD_DIM, count, scale, gqa_partials_.get());
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

void MetalEngine::chunk_forward(const uint32_t* tokens, uint32_t count, bool verify) {
    if (!ch_) throw std::runtime_error("q27 Metal: chunked prefill is unavailable");
    // Verify chunks park per-layer inputs in CHUNK_MAX-sized buffers; plain
    // prefill chunks only need the (wider) layer-stack activations.
    if (!count || count > (verify ? VERIFY_CHUNK_MAX : PREFILL_CHUNK_MAX))
        throw std::runtime_error("q27 Metal: invalid chunk size");
    if ((uint64_t)position_ + count > max_context_)
        throw std::runtime_error("q27 Metal: context exhausted");
    for (uint32_t i = 0; i < count; i++)
        if (tokens[i] >= VOCAB) throw std::runtime_error("q27 Metal: token out of range");
    backend_.embedding_q8_rows(weight("token_embd.weight"), tokens, count, *ch_);
    BackendQuantized x5 = quantized_view(cq5120_, count * N_EMBD);
    for (uint32_t layer = 0; layer < N_LAYER; layer++) {
        backend_.rmsnorm_rows_quantized(*ch_, layer_weight(layer, "attn_norm.weight"),
                                        *cx1_, N_EMBD, count, EPS, x5);
        if (attention_layer(layer)) attention_chunk(layer, count); else gdn_chunk(layer, count, verify);
        backend_.add_inplace(*ch_, *cy_, count * N_EMBD);
        backend_.rmsnorm_rows_quantized(*ch_, layer_weight(layer, "post_attention_norm.weight"),
                                        *cx1_, N_EMBD, count, EPS, x5);
        ffn_chunk(layer, count);
        backend_.add_inplace(*ch_, *cy_, count * N_EMBD);
    }
}

// Commits GDN state (recurrent + convolution ring) for the first `count`
// verified lanes by replaying only the conv/DeltaNet recurrence from the
// inputs parked during the verify chunk. Both chunk kernels are sequential
// in-kernel, so the replayed state is bit-identical to the state the verify
// chunk would have committed after `count` lanes — the full-stack commit
// re-encode this replaces streamed every weight a second time (~0.9 s).
void MetalEngine::gdn_replay(uint32_t count) {
    for (uint32_t layer = 0; layer < N_LAYER; layer++) {
        if (attention_layer(layer)) continue;
        LayerState& state = layers_[layer];
        const uint32_t slot = gdn_slot(layer);
        backend_.conv_chunk(*state.ring, *state.ring, *park_qkv_[slot],
                            layer_weight(layer, "ssm_conv1d.weight"), *cconv_out_, GDN_CH, count);
        backend_.l2norm_rows(*cconv_out_, 2 * GDN_QK_HEADS, GDN_DIM, GDN_CH, count, EPS);
        backend_.delta_chunk(*state.recurrent, *state.recurrent, *cconv_out_,
                             *park_g_[slot], *park_beta_[slot], *cdelta_out_,
                             GDN_HEADS, GDN_QK_HEADS, GDN_DIM, count);
    }
}

// K chained greedy steps in one command buffer: each step's embedding reads
// the token id the previous argmax wrote (docs/metal/plans/2026-07-15-resident-greedy.md),
// and an in-batch copy archives every id into token_ring_ for one readback.
// Refuses to run under an active tool constraint — grammar feeding is a
// host-per-token loop by construction.
uint32_t MetalEngine::decode_resident(uint32_t pending, uint32_t* out, uint32_t k) {
    if (pending >= VOCAB) throw std::runtime_error("q27 Metal: token out of range");
    if (!k || k > RESIDENT_MAX) throw std::runtime_error("q27 Metal: resident slice must be 1..8");
    if (active_mask_ >= 0) throw std::runtime_error("q27 Metal: resident decode under tool constraint");
    backend_.write(*token_out_, 0, &pending, sizeof(pending));
    {
        CommandBatch batch(backend_);
        for (uint32_t i = 0; i < k; i++) {
            encode_token(0, true, true, i);
            backend_.copy(*token_out_, 0, *token_ring_, (uint64_t)i * sizeof(uint32_t),
                          sizeof(uint32_t));
        }
        batch.finish();
        position_ += k;
        logits_resident_ = true;
    }
    backend_.read(*token_ring_, 0, out, (uint64_t)k * sizeof(uint32_t));
    return out[k - 1];
}

uint32_t MetalEngine::step(uint32_t token) {
    if (token >= VOCAB) throw std::runtime_error("q27 Metal: token out of range");
    if (position_ >= max_context_) throw std::runtime_error("q27 Metal: context exhausted");
    CommandBatch batch(backend_);
    encode_token(token, true);
    batch.finish();
    position_++;
    logits_resident_ = true;
    uint32_t next = 0;
    backend_.read(*token_out_, 0, &next, sizeof(next));
    return next;
}

void MetalEngine::mtp_warm(const BackendBuffer& hidden, uint32_t token, uint32_t position) {
    if (!has_mtp_)
        throw std::runtime_error("q27 Metal: artifact has no MTP layer; use --suffix drafting");
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
    else if (kv_attrib_ && kv_attrib_layer_ == UINT32_MAX)
        // Plain round-trip only: step-2 flags index per-attn-layer slots
        // that do not exist for the MTP layer (instrument never runs MTP).
        // Mode 3: MTP KV is not a census cell, so it is never excepted
        // (mask 0 = both sides quantized), keeping the empty-mask control
        // arm comparable to the modes-1/2 full-side treatment.
        backend_.kv_store_f16_attrib_rows(*kbuf_, *vbuf_, *mtp_k_cache_, *mtp_v_cache_,
                                          position, N_KV, 1, kv_attrib_,
                                          kv_attrib_ == 3 ? 0u : kv_attrib_head_,
                                          0, 0, nullptr);
    else
        backend_.kv_store_f16(*kbuf_, *vbuf_, *mtp_k_cache_, *mtp_v_cache_, position, N_KV * HEAD_DIM);
}

// Scheduling-quantum prefill (multislot Phase 1): one bounded chunk encode
// per call. prefill() below drives its chunked loop through this, so the
// serving scheduler's per-quantum ingestion and whole-prompt ingestion are
// the same code path by construction.
void MetalEngine::prefill_chunk(const uint32_t* tokens, uint32_t count) {
    if (!chunked_prefill_)
        throw std::runtime_error("q27 Metal: prefill_chunk requires chunked prefill");
    if (count < 2 || count > PREFILL_CHUNK_MAX)
        throw std::runtime_error("q27 Metal: prefill chunk must be 2..96 tokens");
    if ((uint64_t)position_ + count > max_context_)
        throw std::runtime_error("q27 Metal: prompt exceeds context");
    for (uint32_t i = 0; i < count; i++)
        if (tokens[i] >= VOCAB) throw std::runtime_error("q27 Metal: token out of range");
    CommandBatch batch(backend_);
    chunk_forward(tokens, count);
    batch.finish();
    position_ += count;
    logits_resident_ = false;
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
                (uint32_t)std::min<size_t>(PREFILL_CHUNK_MAX, chunkable - serial_begin);
            prefill_chunk(prompt.data() + serial_begin, count);
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
            encode_token(prompt[i],i+1==prompt.size(),false,(uint32_t)(i-begin));
            if(warm_mtp && i+1<prompt.size()) mtp_warm(*x1_,prompt[i+1],position_+(uint32_t)(i-begin)+1);
        }
        batch.finish();
        position_ += (uint32_t)(end - begin);
    }
    logits_resident_ = true;
    uint32_t next = 0;
    backend_.read(*token_out_, 0, &next, sizeof(next));
    return next;
}

uint32_t MetalEngine::mtp_forward(const BackendBuffer& hidden, uint32_t token,
                                  uint32_t position) {
    if (!has_mtp_)
        throw std::runtime_error("q27 Metal: artifact has no MTP layer; use --suffix drafting");
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
                                  HEAD_DIM, 1.0f / std::sqrt((float)HEAD_DIM),
                                  gqa_partials_.get());
        backend_.turbo_wht(*attn_out_, N_HEAD, HEAD_DIM, true);
    } else {
        if (kv_attrib_ && kv_attrib_layer_ == UINT32_MAX)
            backend_.kv_store_f16_attrib_rows(*kbuf_, *vbuf_, *mtp_k_cache_, *mtp_v_cache_,
                                              position, N_KV, 1, kv_attrib_,
                                              kv_attrib_ == 3 ? 0u : kv_attrib_head_,
                                              0, 0, nullptr);
        else
            backend_.kv_store_f16(*kbuf_, *vbuf_, *mtp_k_cache_, *mtp_v_cache_, position, N_KV * HEAD_DIM);
        backend_.attention_f16(*qg_, 2 * HEAD_DIM, *mtp_k_cache_, *mtp_v_cache_,
                               *attn_out_, position + 1, N_HEAD, N_KV,
                               HEAD_DIM, 1.0f / std::sqrt((float)HEAD_DIM),
                               gqa_partials_.get());
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
// lane in a single state-free layer-major pass with a batched output head
// and per-lane argmax — one CPU synchronization per round instead of one
// per committed token. The verify chunk parks each GDN layer's recurrence
// inputs and discards its speculative state commits; acceptance then
// replays only the GDN recurrence over the accepted prefix (gdn_replay),
// so no state checkpoint, restore, or full-stack commit re-encode exists.
// KV rows written for rejected lanes stay invisible behind position_.
// Committed tokens follow the exact serial-walk semantics, including never
// encoding the final output token.
std::vector<uint32_t> MetalEngine::generate_mtp_batched(uint32_t pending, uint32_t count,
                                                        uint32_t width) {
    // Vector convenience wrapper over the streaming core: a sink that never
    // cancels and an EOS sentinel that never matches (tokens are < VOCAB <
    // UINT32_MAX) reproduce the previous whole-completion behaviour exactly.
    std::vector<uint32_t> output;
    output.reserve(count);
    StopCause cause;
    stream_mtp_batched(pending, count, width, UINT32_MAX,
                       [&](uint32_t token) { output.push_back(token); return true; }, cause);
    return output;
}

// One MTP draft/verify/commit round — the scheduling quantum for MTP
// generation (multislot Phase 1). Extracted verbatim from the streaming
// loop below, which now drives it, so the CLI/server whole-generation path
// and the per-quantum scheduler path cannot drift.
uint32_t MetalEngine::mtp_round(uint32_t pending, uint32_t remaining, uint32_t eos,
                                uint32_t width, uint32_t& live_width,
                                std::vector<uint32_t>& committed) {
    if (pending >= VOCAB) throw std::runtime_error("q27 Metal: pending token out of range");
    if (!has_mtp_)
        throw std::runtime_error("q27 Metal: artifact has no MTP layer; use --suffix drafting");
    if (width < 2 || width > CHUNK_MAX)
        throw std::runtime_error("q27 Metal: MTP width must be 2..12");
    if (!chunked_prefill_)
        throw std::runtime_error("q27 Metal: batched MTP requires chunked prefill");
    if (remaining < 2)
        throw std::runtime_error("q27 Metal: MTP round needs remaining >= 2 (emit the last token directly)");
    // A finished stream must not draft: EOS can arrive as the previous
    // round's bonus prediction (the server quantum loop hands it straight
    // back), and drafting past it would waste a full round and pollute the
    // speculation stats (codex P2). Mirrors the live<2 fallback's EOS skip.
    if (pending == eos) {
        committed.push_back(pending);
        return pending;
    }
    uint32_t live = std::min(std::min(live_width, width), remaining);
    // The verify chunk stores a KV row for every lane, so it must stay
    // inside the reserved context even before acceptance is known.
    if ((uint64_t)position_ + live > max_context_)
        live = (uint32_t)(max_context_ - position_);
    if (live < 2) {
        committed.push_back(pending);
        if (pending == eos) return pending;
        return step(pending);
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
        chunk_forward(lanes.data(), live, /*verify=*/true);
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
    uint32_t commit_n = std::min(accepted + 1, remaining);
    // The final output token is pushed but never encoded, exactly like
    // the serial walk, so snapshots and continuations stay compatible.
    uint32_t encoded = commit_n == remaining ? commit_n - 1 : commit_n;
    // EOS inside the committed prefix: the stream stops there, so state
    // must too (codex finding, 2026-07-15; eos-gate 2026-07-16 measured
    // position advanced past the last emitted token). Hand the caller a
    // committed slice ending AT the EOS token and encode only the tokens
    // before it — KV rows past it stay invisible behind position_, GDN
    // replays only the emitted prefix, exactly like the serial walk. The
    // CLI's never-matching EOS sentinel leaves this loop inert, so
    // sentinel-driven runs are bit-identical by construction.
    for (uint32_t i = 0; i < commit_n; i++)
        if (lanes[i] == eos) {
            commit_n = i + 1;
            encoded = i;
            break;
        }
    last_spec_stats_.accepted += commit_n - 1;
    auto commit_start = clock();
    if (encoded) {
        CommandBatch batch(backend_);
        gdn_replay(encoded);
        backend_.copy(*cfinal_, (uint64_t)(encoded - 1) * N_EMBD * sizeof(float),
                      *x1_, 0, (uint64_t)N_EMBD * sizeof(float));
        backend_.copy(*clogits_, (uint64_t)(encoded - 1) * VOCAB * sizeof(float),
                      *logits_, 0, (uint64_t)VOCAB * sizeof(float));
        batch.finish();
    }
    position_ += encoded;
    if (trace)
        fprintf(stderr, "mtp round: live %u accepted %u | draft %.2fs verify %.2fs commit %.2fs\n",
                live, accepted, std::chrono::duration<double>(verify_start - draft_start).count(),
                std::chrono::duration<double>(commit_start - verify_start).count(), since(commit_start));
    committed.insert(committed.end(), lanes.begin(), lanes.begin() + commit_n);
    // Width adaptation is a pure performance control: committed tokens
    // are width-invariant, matching the recorded 2/4/8/12 gate.
    live_width = accepted + 1 == live ? std::min(width, live_width + 2)
                                      : std::max(2u, accepted + 2);
    return predictions[commit_n - 1];
}

// Sampled MTP: greedy drafts, rejection-sample accept (Phase 0 host walk).
// Draft + verify match mtp_round; only the accept/pending tail differs.
uint32_t MetalEngine::mtp_sample_round(uint32_t pending, uint32_t remaining, uint32_t eos,
                                       uint32_t width, uint32_t& live_width,
                                       const SamplingParams& params, std::mt19937_64& rng,
                                       std::vector<uint32_t>& committed) {
    validate_sampling(params);
    if (pending >= VOCAB) throw std::runtime_error("q27 Metal: pending token out of range");
    if (!has_mtp_)
        throw std::runtime_error("q27 Metal: artifact has no MTP layer; use plain sampling");
    if (width < 2 || width > CHUNK_MAX)
        throw std::runtime_error("q27 Metal: MTP width must be 2..12");
    if (!chunked_prefill_)
        throw std::runtime_error("q27 Metal: batched MTP requires chunked prefill");
    if (remaining < 2)
        throw std::runtime_error("q27 Metal: MTP sample round needs remaining >= 2");
    if (pending == eos) {
        committed.push_back(pending);
        return pending;
    }
    uint32_t live = std::min(std::min(live_width, width), remaining);
    if ((uint64_t)position_ + live > max_context_)
        live = (uint32_t)(max_context_ - position_);
    if (live < 2) {
        // Serial sample fallback: emit pending, encode it, sample next.
        committed.push_back(pending);
        if (pending == eos) return pending;
        (void)step(pending);
        return sample_from_logits(params, rng);
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
        chunk_forward(lanes.data(), live, /*verify=*/true);
        BackendQuantized x5 = quantized_view(cq5120_, live * N_EMBD);
        backend_.rmsnorm_rows_quantized(*ch_, weight("output_norm.weight"), *cfinal_,
                                        N_EMBD, live, EPS, x5);
        backend_.matmul_quantized(weight("output.weight"), x5, live, *clogits_);
        // No argmax_rows — acceptance is rejection sampling on the served dist.
        batch.finish();
    }
    std::vector<uint32_t> drafts(live - 1);
    for (uint32_t i = 0; i + 1 < live; i++) drafts[i] = lanes[i + 1];

    // Prefer per-lane GPU top-k when top_k is set (card recipe uses 20):
    // ~k floats/ids per lane instead of full VOCAB readback + partial_sort.
    // Fall back to full logits on opt-out, top_k==0, or degenerate over-set.
    std::vector<ServedDistribution> lane_dists(live);
    bool used_topk = false;
    if (gpu_sample_ && params.temperature > 0.0f &&
        params.top_k >= 1 && params.top_k <= 256) {
        used_topk = true;
        for (uint32_t lane = 0; lane < live; lane++) {
            // Bind clogits_ row via byte offset — no full-row copy into logits_.
            // topk requires its own command (CPU-clears count); not batchable.
            const uint64_t row_off = (uint64_t)lane * VOCAB * sizeof(float);
            backend_.topk(*clogits_, VOCAB, params.top_k, *topk_values_,
                          *topk_indices_, *topk_count_, row_off);
            uint32_t count = 0;
            backend_.read(*topk_count_, 0, &count, sizeof(count));
            if (count < params.top_k || count > TOPK_CAPACITY) {
                used_topk = false;
                break;
            }
            std::vector<float> values(count);
            std::vector<uint32_t> indices(count);
            backend_.read(*topk_values_, 0, values.data(), count * sizeof(float));
            backend_.read(*topk_indices_, 0, indices.data(), count * sizeof(uint32_t));
            lane_dists[lane] = build_served_from_candidates(
                values.data(), indices.data(), count, params);
        }
    }
    if (!used_topk) {
        std::vector<float> lane_logits((size_t)live * VOCAB);
        backend_.read(*clogits_, 0, lane_logits.data(),
                      (uint64_t)live * VOCAB * sizeof(float));
        for (uint32_t lane = 0; lane < live; lane++)
            lane_dists[lane] = build_served_distribution(
                lane_logits.data() + (size_t)lane * VOCAB, VOCAB, params);
    }

    SpecRejectResult accept =
        spec_rejection_accept(lane_dists.data(), live, drafts.data(), rng);
    // accepted drafts before first reject (or all), for width adaptation.
    const uint32_t accepted = accept.n - 1;
    uint32_t commit_n = std::min(accept.n, remaining);
    uint32_t encoded = commit_n == remaining ? commit_n - 1 : commit_n;
    for (uint32_t i = 0; i < commit_n; i++)
        if (lanes[i] == eos) {
            commit_n = i + 1;
            encoded = i;
            break;
        }
    last_spec_stats_.accepted += commit_n > 0 ? commit_n - 1 : 0;
    auto commit_start = clock();
    if (encoded) {
        CommandBatch batch(backend_);
        gdn_replay(encoded);
        backend_.copy(*cfinal_, (uint64_t)(encoded - 1) * N_EMBD * sizeof(float),
                      *x1_, 0, (uint64_t)N_EMBD * sizeof(float));
        backend_.copy(*clogits_, (uint64_t)(encoded - 1) * VOCAB * sizeof(float),
                      *logits_, 0, (uint64_t)VOCAB * sizeof(float));
        batch.finish();
    }
    position_ += encoded;
    if (trace)
        fprintf(stderr,
                "mtp sample round: live %u n %u stop %u exclude %d topk %d | draft %.2fs verify %.2fs commit %.2fs\n",
                live, accept.n, accept.stop_lane, (int)accept.exclude, used_topk ? 1 : 0,
                std::chrono::duration<double>(verify_start - draft_start).count(),
                std::chrono::duration<double>(commit_start - verify_start).count(),
                since(commit_start));
    committed.insert(committed.end(), lanes.begin(), lanes.begin() + commit_n);
    live_width = accepted + 1 == live ? std::min(width, live_width + 2)
                                      : std::max(2u, accepted + 2);
    // Full walk used → pending already sampled. Remaining/EOS clamp → sample
    // from the last committed lane (mirrors greedy predictions[c-1]).
    if (commit_n == accept.n) return accept.pending;
    return sample_served(lane_dists[commit_n - 1], rng, /*exclude=*/-1);
}

// Gate 0 oracle round: mtp_round with the layer-64 draft stage replaced by
// caller-supplied reference lanes and the acceptance walk replaced by a
// teacher-forced full commit. The verify chunk, batched output head,
// per-lane argmax, and gdn_replay commit are shared verbatim, so the round
// cost is exactly V(w)+O(w) — what a perfect drafter would pay. Because the
// committed tokens ARE the lanes, KV rows written during the verify chunk
// are all real and the replayed GDN state matches a serial walk over the
// same tokens bit-exactly (both chunk kernels are sequential in-kernel).
void MetalEngine::oracle_round(const uint32_t* lanes, uint32_t live, bool last,
                               uint32_t* predictions) {
    if (live < 2 || live > VERIFY_CHUNK_MAX)
        throw std::runtime_error("q27 Metal: oracle width must be 2..VERIFY_CHUNK_MAX");
    if (!chunked_prefill_)
        throw std::runtime_error("q27 Metal: oracle round requires chunked prefill");
    // The verify chunk stores a KV row for every lane, so all `live` rows
    // must fit the reserved context even on a `last` round whose final
    // token never advances position_ — unlike the serial walk, which can
    // end at max_context_+1 because it never encodes the final token.
    // --oracle therefore needs --ctx >= prompt+count, one more than serial.
    if ((uint64_t)position_ + live > max_context_)
        throw std::runtime_error("q27 Metal: oracle verify rows exceed context; --oracle needs --ctx >= prompt+count");
    for (uint32_t lane = 0; lane < live; lane++)
        if (lanes[lane] >= VOCAB)
            throw std::runtime_error("q27 Metal: oracle lane token out of range");
    last_spec_stats_.rounds++;
    last_spec_stats_.drafted += live - 1;
    // Round-anatomy trace (verify-round-cost plan P0): verify batch vs
    // prediction readback vs commit batch, per round.
    static const bool trace = getenv("Q27_ORACLE_TRACE") != nullptr;
    auto clock = [] { return std::chrono::steady_clock::now(); };
    auto ms_since = [](std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    auto verify_start = clock();
    {
        CommandBatch batch(backend_);
        chunk_forward(lanes, live, /*verify=*/true);
        BackendQuantized x5 = quantized_view(cq5120_, live * N_EMBD);
        backend_.rmsnorm_rows_quantized(*ch_, weight("output_norm.weight"), *cfinal_,
                                        N_EMBD, live, EPS, x5);
        backend_.matmul_quantized(weight("output.weight"), x5, live, *clogits_);
        backend_.argmax_rows(*clogits_, VOCAB, live, *cpred_);
        batch.finish();
    }
    const double verify_ms = trace ? ms_since(verify_start) : 0.0;
    auto read_start = clock();
    backend_.read(*cpred_, 0, predictions, live * sizeof(uint32_t));
    const double read_ms = trace ? ms_since(read_start) : 0.0;
    auto commit_start = clock();
    // Teacher-forced commit of every lane; the final output token of a
    // generation is never encoded, exactly like the serial walk.
    const uint32_t encoded = last ? live - 1 : live;
    last_spec_stats_.accepted += live - 1;
    if (encoded) {
        CommandBatch batch(backend_);
        gdn_replay(encoded);
        backend_.copy(*cfinal_, (uint64_t)(encoded - 1) * N_EMBD * sizeof(float),
                      *x1_, 0, (uint64_t)N_EMBD * sizeof(float));
        backend_.copy(*clogits_, (uint64_t)(encoded - 1) * VOCAB * sizeof(float),
                      *logits_, 0, (uint64_t)VOCAB * sizeof(float));
        batch.finish();
    }
    position_ += encoded;
    if (trace)
        fprintf(stderr, "oracle round: live %u | verify %.2f ms read %.2f ms commit %.2f ms\n",
                live, verify_ms, read_ms, ms_since(commit_start));
}

uint32_t MetalEngine::stream_mtp_batched(uint32_t pending, uint32_t count, uint32_t width,
                                         uint32_t eos, const TokenSink& sink, StopCause& cause) {
    uint32_t emitted = 0;
    cause = StopCause::MaxTokens;
    // commit() streams one token through the sink, stopping on EOS or cancel.
    auto commit = [&](uint32_t token) -> bool {
        if (token == eos) { cause = StopCause::Eos; return true; }
        if (!sink(token)) {
            // mtp_round commits a whole accepted prefix before delivery. A
            // cancelled sink owns none of the current token and may own only
            // part of that prefix, so the speculative engine state cannot be
            // reused. Reset explicitly; the server already cached prompt
            // state before generation and cancelled requests never cache
            // post-generation state.
            reset();
            cause = StopCause::Cancelled;
            return true;
        }
        emitted++;
        return false;
    };
    // Start narrow and let acceptance widen the window: committed tokens are
    // width-invariant, and a wide first round pays for many serial drafts
    // through a cold draft head before acceptance has been measured once.
    uint32_t live_width = std::min(width, 4u);
    std::vector<uint32_t> committed;
    while (emitted < count) {
        if (emitted + 1 == count) { commit(pending); return emitted; }
        committed.clear();
        pending = mtp_round(pending, count - emitted, eos, width, live_width, committed);
        for (uint32_t token : committed)
            if (commit(token)) return emitted;
    }
    return emitted;
}

uint32_t MetalEngine::ingest_prompt(const std::vector<uint32_t>& tokens, bool warm_mtp,
                                    bool reset_first) {
    if (reset_first) reset();
    return prefill(tokens, warm_mtp);
}

std::vector<float> MetalEngine::read_logits() {
    if (!logits_resident_)
        throw std::runtime_error("q27 Metal: snapshot has no resident logits; continue prefill before reading logits");
    std::vector<float> result(VOCAB);
    backend_.synchronize();
    backend_.read(*logits_,0,result.data(),result.size()*sizeof(float));
    return result;
}

void MetalEngine::read_hidden(std::vector<float>& out) {
    out.resize(N_EMBD);
    backend_.synchronize();
    backend_.read(*x1_,0,out.data(),out.size()*sizeof(float));
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
            // Early readout: absolute PPL stabilizes long before a deep pass
            // finishes; print the running mean so a long run yields its
            // verdict in the first minutes and the tail only refines buckets.
            if (done / 2048 != (done - count) / 2048) {
                double sum = 0.0;
                for (float v : result) sum += v;
                fprintf(stderr, "  nll pos %u: running mean %.4f (ppl %.3f)\n",
                        done, sum / result.size(), std::exp(sum / result.size()));
            }
        }
    }
    while (done < n_encode) {
        {
            CommandBatch batch(backend_);
            encode_token(tokens[done], true);
            batch.finish();
        }
        position_++;
        logits_resident_ = true;
        if (chunked_prefill_) {
            // The leftover row rides the same float GPU reduction as the
            // chunked rows — a CPU double tail would be a third regime
            // inside one pass, and CUDA runs every row float on the GPU
            // (k3 audit D4/E5). logits_ is one VOCAB row, valid at rows=1.
            backend_.write(*ctargets_, 0, &tokens[done + 1], sizeof(uint32_t));
            {
                CommandBatch batch(backend_);
                backend_.nll_rows(*logits_, *ctargets_, *cnll_, VOCAB, 1);
                batch.finish();
            }
            float row_nll = 0.0f;
            backend_.read(*cnll_, 0, &row_nll, sizeof row_nll);
            result.push_back(row_nll);
        } else {
            // Pre-Apple7 all-serial fallback: no ctargets_/cnll_ exist and
            // the whole pass is one (CPU) regime already.
            std::vector<float> logits = read_logits();
            result.push_back(nll_cpu(logits.data(), tokens[done + 1], VOCAB));
        }
        done++;
    }
    if (n_encode >= CHUNK_MAX) fprintf(stderr, "\n");
    return result;
}

void MetalEngine::teacher_force_logits(const uint32_t* tokens, uint32_t count,
                                       std::vector<float>& out) {
    if (!count || count > CHUNK_MAX)
        throw std::runtime_error("q27 Metal: teacher_force_logits takes 1..12 tokens");
    if (active_mask_ >= 0)
        throw std::runtime_error("q27 Metal: teacher forcing refuses active tool constraints");
    if ((uint64_t)position_ + count > max_context_)
        throw std::runtime_error("q27 Metal: teacher-forced chunk exceeds context");
    for (uint32_t i = 0; i < count; i++)
        if (tokens[i] >= VOCAB) throw std::runtime_error("q27 Metal: token out of range");
    out.resize((size_t)count * VOCAB);
    if (chunked_prefill_ && count >= 2) {
        {
            CommandBatch batch(backend_);
            chunk_forward(tokens, count);
            BackendQuantized x5 = quantized_view(cq5120_, count * N_EMBD);
            backend_.rmsnorm_rows_quantized(*ch_, weight("output_norm.weight"), *cfinal_,
                                            N_EMBD, count, EPS, x5);
            backend_.matmul_quantized(weight("output.weight"), x5, count, *clogits_);
            // Commit recurrent/KV state and both serial coherence buffers as
            // one unit. A command failure poisons the backend in finish().
            backend_.copy(*clogits_, (uint64_t)(count - 1) * VOCAB * sizeof(float),
                          *logits_, 0, (uint64_t)VOCAB * sizeof(float));
            backend_.copy(*cfinal_, (uint64_t)(count - 1) * N_EMBD * sizeof(float),
                          *x1_, 0, (uint64_t)N_EMBD * sizeof(float));
            batch.finish();
        }
        position_ += count;
        logits_resident_ = true;
        backend_.read(*clogits_, 0, out.data(), out.size() * sizeof(float));
        return;
    }
    for (uint32_t i = 0; i < count; i++) {
        {
            CommandBatch batch(backend_);
            encode_token(tokens[i], true);
            batch.finish();
        }
        position_++;
        logits_resident_ = true;
        backend_.read(*logits_, 0, out.data() + (size_t)i * VOCAB,
                      (uint64_t)VOCAB * sizeof(float));
    }
}

void MetalEngine::teacher_force_logits_wide(const uint32_t* tokens, uint32_t count,
                                            std::vector<float>& out) {
    if (!count || count > PREFILL_CHUNK_MAX)
        throw std::runtime_error("q27 Metal: teacher_force_logits_wide takes 1..96 tokens");
    if (active_mask_ >= 0)
        throw std::runtime_error("q27 Metal: teacher forcing refuses active tool constraints");
    if (count <= CHUNK_MAX) { teacher_force_logits(tokens, count, out); return; }
    if (!chunked_prefill_ || !ch_)
        throw std::runtime_error("q27 Metal: wide teacher forcing requires chunked prefill");
    if ((uint64_t)position_ + count > max_context_)
        throw std::runtime_error("q27 Metal: teacher-forced chunk exceeds context");
    for (uint32_t i = 0; i < count; i++)
        if (tokens[i] >= VOCAB) throw std::runtime_error("q27 Metal: token out of range");
    out.resize((size_t)count * VOCAB);
    if (!wide_head_stage_)
        wide_head_stage_ = backend_.allocate((uint64_t)CHUNK_MAX * N_EMBD * sizeof(float));
    {
        CommandBatch batch(backend_);
        chunk_forward(tokens, count);
        batch.finish();
    }
    // chunk_forward has committed recurrent/KV state. Any later host-side or
    // encoding failure must poison the backend: position_ cannot describe a
    // reusable engine after a partial wide-head pass.
    try {
        // Head in CHUNK_MAX-row slices: cfinal_/clogits_ are CHUNK_MAX-sized,
        // so each slice's hidden rows are staged to offset 0 first.
        for (uint32_t s0 = 0; s0 < count; s0 += CHUNK_MAX) {
            const uint32_t slice = std::min(CHUNK_MAX, count - s0);
            const bool final_slice = s0 + slice == count;
            {
                CommandBatch batch(backend_);
                backend_.copy(*ch_, (uint64_t)s0 * N_EMBD * sizeof(float),
                              *wide_head_stage_, 0, (uint64_t)slice * N_EMBD * sizeof(float));
                BackendQuantized x5 = quantized_view(cq5120_, slice * N_EMBD);
                backend_.rmsnorm_rows_quantized(*wide_head_stage_, weight("output_norm.weight"),
                                                *cfinal_, N_EMBD, slice, EPS, x5);
                backend_.matmul_quantized(weight("output.weight"), x5, slice, *clogits_);
                if (final_slice) {
                    const uint32_t last_row = slice - 1;
                    backend_.copy(*clogits_, (uint64_t)last_row * VOCAB * sizeof(float),
                                  *logits_, 0, (uint64_t)VOCAB * sizeof(float));
                    backend_.copy(*cfinal_, (uint64_t)last_row * N_EMBD * sizeof(float),
                                  *x1_, 0, (uint64_t)N_EMBD * sizeof(float));
                }
                batch.finish();
            }
            backend_.read(*clogits_, 0, out.data() + (size_t)s0 * VOCAB,
                          (uint64_t)slice * VOCAB * sizeof(float));
        }
    } catch (...) {
        backend_.poison();
        throw;
    }
    position_ += count;
    logits_resident_ = true;
}

// GPU-assisted sampling: when top-k is active and within the radix-select
// range, extract the candidate over-set on the GPU and read back ~k pairs
// A candidate count above capacity signals degenerate ties or a NaN
// sentinel — fall back to the exact full-readback path, which validates
// every logit and is also the Q27_METAL_GPU_SAMPLE=0 opt-out and the
// temperature-0 / no-top-k route. Same-seed token sequences match the
// full path exactly (one uniform draw either way; real logits don't tie).
uint32_t MetalEngine::sample_next(const SamplingParams& params, std::mt19937_64& random) {
    if (!logits_resident_)
        throw std::runtime_error("q27 Metal: snapshot has no resident logits; continue prefill before sampling");
    if (gpu_sample_ && params.temperature != 0.0f && params.top_k >= 1 && params.top_k <= 256) {
        backend_.topk(*logits_, VOCAB, params.top_k, *topk_values_, *topk_indices_, *topk_count_);
        uint32_t count = 0;
        backend_.read(*topk_count_, 0, &count, sizeof(count));
        // count >= k is provable from the kernel's two-pass construction
        // (2026-07-17 triage doc); the lower bound here guards future
        // kernel edits — an under-set must fall back, never silently
        // sample from a truncated candidate list.
        if (count >= (uint32_t)params.top_k && count <= TOPK_CAPACITY) {
            std::vector<float> values(count);
            std::vector<uint32_t> indices(count);
            backend_.read(*topk_values_, 0, values.data(), count * sizeof(float));
            backend_.read(*topk_indices_, 0, indices.data(), count * sizeof(uint32_t));
            return sample_candidates_cpu(values, indices, count, params, random);
        }
    }
    return sample_logits_cpu(read_logits(), params, random);
}

uint32_t MetalEngine::sample_from_logits(const SamplingParams& params, std::mt19937_64& rng) {
    validate_sampling(params);
    return sample_next(params, rng);
}

std::vector<uint32_t> MetalEngine::generate_sampled_from_logits(uint32_t count,
                                                                 const SamplingParams& params) {
    validate_sampling(params); last_spec_stats_={};
    if((uint64_t)position_+(count?count-1:0)>max_context_)
        throw std::runtime_error("q27 Metal: generation exceeds context");
    std::mt19937_64 random(params.seed); std::vector<uint32_t> output; output.reserve(count);
    for(uint32_t i=0;i<count;i++) {
        uint32_t token=sample_next(params,random); output.push_back(token);
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

uint32_t MetalEngine::stream_sampled_from_logits(uint32_t count, uint32_t eos,
                                                 const SamplingParams& params,
                                                 const TokenSink& sink, StopCause& cause) {
    validate_sampling(params); last_spec_stats_={};
    if((uint64_t)position_+(count?count-1:0)>max_context_)
        throw std::runtime_error("q27 Metal: generation exceeds context");
    std::mt19937_64 random(params.seed);
    cause = StopCause::MaxTokens;
    uint32_t emitted=0;
    while(emitted<count) {
        uint32_t token=sample_next(params,random);
        if(token==eos) { cause=StopCause::Eos; return emitted; }
        if(!sink(token)) { cause=StopCause::Cancelled; return emitted; }
        if(++emitted==count) return emitted;
        step(token);
    }
    return emitted;
}

uint32_t MetalEngine::stream_from_pending(uint32_t pending, uint32_t count, uint32_t eos,
                                          uint32_t mtp_width, const TokenSink& sink,
                                          StopCause& cause) {
    last_spec_stats_={};
    if (pending >= VOCAB) throw std::runtime_error("q27 Metal: pending token out of range");
    if (mtp_width && !has_mtp_)
        throw std::runtime_error("q27 Metal: artifact has no MTP layer; use --suffix drafting");
    if (mtp_width && (mtp_width < 2 || mtp_width > 12))
        throw std::runtime_error("q27 Metal: MTP width must be 2..12");
    if ((uint64_t)position_ + (count ? count - 1 : 0) > max_context_)
        throw std::runtime_error("q27 Metal: generation exceeds context");
    cause = StopCause::MaxTokens;
    if (mtp_width && chunked_prefill_)
        return stream_mtp_batched(pending, count, mtp_width, eos, sink, cause);
    // Serial greedy walk: the pending token is emitted, then each step()
    // yields the next. EOS stops without emitting it; the sink returning
    // false is a client cancel (mirrors the CUDA engine's on_token contract).
    uint32_t emitted=0;
    uint32_t cur=pending;
    while(emitted<count) {
        if(cur==eos) { cause=StopCause::Eos; return emitted; }
        if(!sink(cur)) { cause=StopCause::Cancelled; return emitted; }
        if(++emitted==count) return emitted;
        cur=step(cur);
    }
    return emitted;
}

std::vector<uint32_t> MetalEngine::generate_from_pending(uint32_t pending, uint32_t count,
                                                          uint32_t mtp_width) {
    last_spec_stats_={};
    if (pending >= VOCAB) throw std::runtime_error("q27 Metal: pending token out of range");
    if (mtp_width && !has_mtp_)
        throw std::runtime_error("q27 Metal: artifact has no MTP layer; use --suffix drafting");
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
        if (resident_ && active_mask_ < 0) {
            uint32_t ids[RESIDENT_MAX];
            while (output.size() < count) {
                const uint32_t take = std::min<uint32_t>(RESIDENT_MAX,
                                                         (uint32_t)(count - output.size()));
                pending = decode_resident(pending, ids, take);
                output.insert(output.end(), ids, ids + take);
            }
            return output;
        }
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
    if (active_mask_ >= 0)
        throw std::runtime_error("q27 Metal: MTP is unmasked; tool constraints require serial decode");
    if (!has_mtp_)
        throw std::runtime_error("q27 Metal: artifact has no MTP layer; use --suffix drafting");
    if ((uint64_t)prompt.size() + count > max_context_ + 1)
        throw std::runtime_error("q27 Metal: prompt/generation exceeds context");
    uint32_t pending = ingest_prompt(prompt, true, true);
    return generate_from_pending(pending, count, width);
}

std::vector<uint32_t> MetalEngine::generate_mtp_sampled(const std::vector<uint32_t>& prompt,
                                                          uint32_t count, uint32_t width,
                                                          const SamplingParams& params) {
    validate_sampling(params);
    if (prompt.empty()) throw std::runtime_error("q27 Metal: prompt is empty");
    if (active_mask_ >= 0)
        throw std::runtime_error("q27 Metal: sampled MTP is unmasked; tool constraints require serial decode");
    if (!has_mtp_)
        throw std::runtime_error("q27 Metal: artifact has no MTP layer; use plain sampling");
    if (!chunked_prefill_)
        throw std::runtime_error("q27 Metal: sampled MTP requires chunked prefill");
    if (width < 2 || width > CHUNK_MAX)
        throw std::runtime_error("q27 Metal: MTP width must be 2..12");
    if ((uint64_t)prompt.size() + count > max_context_ + 1)
        throw std::runtime_error("q27 Metal: prompt/generation exceeds context");
    last_spec_stats_ = {};
    // Prefill leaves logits for the first gen token (always — including count==0
    // so --dump-logits / reuse see prompt-conditioned state). Sample pending
    // only when generating (not the greedy argmax from ingest_prompt's return).
    (void)ingest_prompt(prompt, true, true);
    if (!count) return {};
    std::mt19937_64 rng(params.seed);
    uint32_t pending = sample_from_logits(params, rng);
    std::vector<uint32_t> generated;
    generated.reserve(count);
    uint32_t live_width = std::min(width, 4u);
    std::vector<uint32_t> committed;
    while (generated.size() < count) {
        if (generated.size() + 1 == count) {
            generated.push_back(pending);
            break;
        }
        committed.clear();
        pending = mtp_sample_round(pending, (uint32_t)(count - generated.size()),
                                   UINT32_MAX, width, live_width, params, rng, committed);
        generated.insert(generated.end(), committed.begin(), committed.end());
    }
    return generated;
}

// Suffix-burst round (2026-07-16-suffix-burst-verify.md): oracle_round's
// caller-lane verify chunk + batched head + argmax, then mtp_round's REAL
// acceptance walk, commit_n/encoded rules, and early-EOS clamp (e765dde) —
// verbatim semantics, lanes from the CPU-side SuffixDraft instead of the
// layer-64 draft head. Committed tokens are greedy-identical to the serial
// walk by the same argument as mtp_round (modulo the documented
// tolerance-gated chunk-GEMM class).
uint32_t MetalEngine::suffix_round(uint32_t remaining, uint32_t eos, const uint32_t* lanes,
                                   uint32_t live, std::vector<uint32_t>& committed) {
    if (remaining < 2)
        throw std::runtime_error("q27 Metal: suffix round needs remaining >= 2 (emit the last token directly)");
    if (active_mask_ >= 0)
        throw std::runtime_error("q27 Metal: suffix rounds are unmasked; tool constraints require serial decode");
    if (live < 2 || live > VERIFY_CHUNK_MAX)
        throw std::runtime_error("q27 Metal: suffix round live width must be 2..VERIFY_CHUNK_MAX");
    if (!chunked_prefill_)
        throw std::runtime_error("q27 Metal: suffix round requires chunked prefill");
    if ((uint64_t)position_ + live > max_context_)
        throw std::runtime_error("q27 Metal: suffix verify rows exceed context");
    for (uint32_t lane = 0; lane < live; lane++)
        if (lanes[lane] >= VOCAB)
            throw std::runtime_error("q27 Metal: suffix lane token out of range");
    last_spec_stats_.rounds++;
    last_spec_stats_.drafted += live - 1;
    {
        CommandBatch batch(backend_);
        chunk_forward(lanes, live, /*verify=*/true);
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
    uint32_t commit_n = std::min(accepted + 1, remaining);
    uint32_t encoded = commit_n == remaining ? commit_n - 1 : commit_n;
    for (uint32_t i = 0; i < commit_n; i++)
        if (lanes[i] == eos) {
            commit_n = i + 1;
            encoded = i;
            break;
        }
    last_spec_stats_.accepted += commit_n - 1;
    if (encoded) {
        CommandBatch batch(backend_);
        gdn_replay(encoded);
        backend_.copy(*cfinal_, (uint64_t)(encoded - 1) * N_EMBD * sizeof(float),
                      *x1_, 0, (uint64_t)N_EMBD * sizeof(float));
        backend_.copy(*clogits_, (uint64_t)(encoded - 1) * VOCAB * sizeof(float),
                      *logits_, 0, (uint64_t)VOCAB * sizeof(float));
        batch.finish();
    }
    position_ += encoded;
    committed.insert(committed.end(), lanes, lanes + commit_n);
    // Dispatch evidence for the width gates (vacuous-gate lesson): printed
    // at the dispatch site, not the driver's bookkeeping.
    static const bool trace = getenv("Q27_SUFFIX_TRACE") != nullptr;
    if (trace)
        fprintf(stderr, "suffix round: live %u accepted %u committed %u\n", live, accepted, commit_n);
    return predictions[commit_n - 1];
}

uint32_t MetalEngine::suffix_step(SuffixDraft& drafter, uint32_t pending, uint32_t remaining,
                                  uint32_t eos, uint32_t width, uint32_t minimum_match,
                                  std::vector<uint32_t>& committed, bool* burst) {
    if (remaining < 2)
        throw std::runtime_error("q27 Metal: suffix step needs remaining >= 2 (emit the last token directly)");
    if (width < 2 || width > VERIFY_CHUNK_MAX)
        throw std::runtime_error("q27 Metal: suffix width must be 2..VERIFY_CHUNK_MAX");
    committed.clear();
    if (burst) *burst = false;
    // A pending eos commits without encoding, mirroring suffix_round's lane
    // clamp — the serial fallback below must never step() the eos token.
    // Unreachable from generate_suffix (its sentinel never matches).
    if (pending == eos) { committed.push_back(eos); return eos; }
    // Propose up to width-1 continuation tokens; the verify chunk also
    // needs one KV row per lane inside the reserved context.
    uint32_t max_lanes = std::min<uint32_t>(width, remaining);
    if ((uint64_t)position_ + max_lanes > max_context_)
        max_lanes = (uint32_t)(max_context_ - position_);
    int proposals[VERIFY_CHUNK_MAX];
    int match = 0;
    if (max_lanes >= 2)
        match = drafter.propose_with((int)pending, (int)(max_lanes - 1), proposals);
    // Match-capped width (plan contract, codex P2): forward lanes are
    // bounded by the matched suffix length — proposals past the match
    // evidence are lag-copy extrapolation, and dispatching them would
    // make the drafted stats and burst economics measure speculation
    // beyond what the match justifies.
    uint32_t live = match >= (int)minimum_match
                        ? std::min<uint32_t>(max_lanes, (uint32_t)match + 1) : 0;
    // Full-tile snap-down (lever 2: round cost steps one full weight
    // stream per 16-token tile): a partial second/third tile pays a
    // whole stream for < 16 possible tokens — never worth it. 17..31
    // lanes snap to 16, 33..47 snap to 32.
    if (live > 16 && live < 32) live = 16;
    else if (live > 32 && live < 48) live = 32;
    if (live >= 2) {
        last_suffix_stats_.burst_rounds++;
        if (live <= 16) last_suffix_stats_.lanes_le16++;
        else if (live == 32) last_suffix_stats_.lanes_32++;
        else last_suffix_stats_.lanes_48++;
        if (burst) *burst = true;
        uint32_t lanes[VERIFY_CHUNK_MAX];
        lanes[0] = pending;
        for (uint32_t i = 1; i < live; i++) lanes[i] = (uint32_t)proposals[i - 1];
        pending = suffix_round(remaining, eos, lanes, live, committed);
        for (uint32_t tok : committed) drafter.append((int)tok);
        return pending;
    }
    last_suffix_stats_.fallback_rounds++;
    last_spec_stats_.rounds++;
    committed.push_back(pending);
    drafter.append((int)pending);
    return step(pending);
}

std::vector<uint32_t> MetalEngine::generate_suffix(const std::vector<uint32_t>& prompt,
                                                   uint32_t count, uint32_t width,
                                                   uint32_t minimum_match) {
    if (prompt.empty()) throw std::runtime_error("q27 Metal: prompt is empty");
    if (width < 2 || width > VERIFY_CHUNK_MAX)
        throw std::runtime_error("q27 Metal: suffix width must be 2..VERIFY_CHUNK_MAX");
    if (!chunked_prefill_)
        throw std::runtime_error("q27 Metal: batched suffix requires chunked prefill; use --suffix-serial");
    // Reject constraints at entry (codex P2): burst rounds argmax unmasked
    // logits, and a serial-fallback round WOULD mask — a mixed stream is
    // worse than a loud error. Same contract as GPU-resident greedy.
    if (active_mask_ >= 0)
        throw std::runtime_error("q27 Metal: batched suffix refuses active tool constraints; use serial decode");
    if ((uint64_t)prompt.size() + count > max_context_ + 1)
        throw std::runtime_error("q27 Metal: prompt/generation exceeds context");
    uint32_t pending = ingest_prompt(prompt, false, true);
    last_spec_stats_ = {};
    last_suffix_stats_ = {};
    std::vector<int> history(prompt.begin(), prompt.end());
    SuffixDraft drafter;
    drafter.reset(history);
    std::vector<uint32_t> output;
    output.reserve(count);
    std::vector<uint32_t> committed;
    // EOS sentinel: the CLI drives this path with a never-matching token
    // when it wants a fixed count; a real eos clamps commits mid-burst.
    const uint32_t eos = VOCAB; // never matches: lanes are validated < VOCAB
    while (output.size() < count) {
        if (output.size() + 1 == count) { output.push_back(pending); break; }
        pending = suffix_step(drafter, pending, (uint32_t)(count - output.size()),
                              eos, width, minimum_match, committed);
        output.insert(output.end(), committed.begin(), committed.end());
    }
    return output;
}

std::vector<uint32_t> MetalEngine::generate_suffix_serial(const std::vector<uint32_t>& prompt,
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
