#pragma once

#include "metal_backend.h"
#include "../sampling.h"
#include "../loader.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace q27 {

class MetalEngine {
  public:
    struct Snapshot;
    struct SpecStats { uint64_t rounds=0,drafted=0,accepted=0; };
    // One-per-artifact state shared by engines in the same process: the
    // artifact mapping, the Metal queue/pipelines (the whole-mapping buffer
    // and residency set live behind the backend), and the weight wrap.
    // Engines sharing a Shared never map or wire the artifact twice, so the
    // one-model-load memory policy sees a single load however many engines
    // (e.g. an fp16-KV baseline and a turbo3-KV subject) attach to it.
    struct Shared {
        Model model;
        MetalBackend backend;
        std::unordered_map<std::string, BackendTensor> weights;
        // Combined KV-cache footprint of every engine on this mapping, so a
        // second engine cannot pass the per-engine budget check while the
        // pair overcommits the device (codex review finding, 2026-07-15).
        uint64_t cache_bytes = 0;
        explicit Shared(Model&& opened) : model(std::move(opened)) {}
    };
    static std::shared_ptr<Shared> open_shared(const std::string& model_path);
    explicit MetalEngine(const std::string& model_path, uint32_t context = 128,
                         bool turbo3_kv = false);
    MetalEngine(std::shared_ptr<Shared> shared, uint32_t context, bool turbo3_kv);
    // Reference members alias shared GPU state; a copy with an independent
    // position_ would corrupt its sibling. Engines are pinned to their spot.
    MetalEngine(const MetalEngine&) = delete;
    MetalEngine& operator=(const MetalEngine&) = delete;

    void reset();
    uint32_t step(uint32_t token);
    std::vector<uint32_t> generate(const std::vector<uint32_t>& prompt, uint32_t count);
    std::vector<uint32_t> generate_mtp(const std::vector<uint32_t>& prompt,
                                       uint32_t count, uint32_t width);
    std::vector<uint32_t> generate_suffix(const std::vector<uint32_t>& prompt,
                                          uint32_t count, uint32_t width,
                                          uint32_t minimum_match = 12);
    uint32_t ingest_prompt(const std::vector<uint32_t>& tokens, bool warm_mtp,
                           bool reset_first = true);
    std::vector<uint32_t> generate_from_pending(uint32_t pending, uint32_t count,
                                                uint32_t mtp_width = 0);
    std::vector<float> read_logits();
    // Teacher-forced NLL for tokens[0..N): returns N-1 values where
    // result[i] = -log P(tokens[i+1] | tokens[0..i]). Uses layer-major
    // chunked encode + batched output head when available.
    std::vector<float> teacher_force_nll(const std::vector<uint32_t>& tokens);
    // Teacher-forced chunk logits for cross-engine comparison gates (e.g.
    // the KL KV-tolerance gate): encode tokens[0..count) at the engine's
    // current position — count 1..12; a single token takes the serial path —
    // and fill `out` with count x vocab logits rows. The caller loops over
    // the stream and interleaves engines; both must advance in lockstep.
    void teacher_force_logits(const uint32_t* tokens, uint32_t count,
                              std::vector<float>& out);
    std::vector<uint32_t> generate_sampled(const std::vector<uint32_t>& prompt,
                                           uint32_t count,const SamplingParams& params);
    std::vector<uint32_t> generate_sampled_from_logits(uint32_t count,
                                                       const SamplingParams& params);

    // Streaming generation. `sink(token)` is called for each committed,
    // non-EOS token in order; return false from the sink to cancel (client
    // disconnect). Generation stops at EOS (StopCause::Eos, the EOS token is
    // not passed to the sink), when `count` tokens have been emitted
    // (StopCause::MaxTokens), or when the sink returns false
    // (StopCause::Cancelled). Returns the number of tokens passed to the sink.
    // These mirror the CUDA server's generate(prompt, n_max, eos, on_token)
    // contract so the two servers report finish_reason identically.
    enum class StopCause { MaxTokens, Eos, Cancelled };
    using TokenSink = std::function<bool(uint32_t)>;
    uint32_t stream_from_pending(uint32_t pending, uint32_t count, uint32_t eos,
                                 uint32_t mtp_width, const TokenSink& sink, StopCause& cause);
    uint32_t stream_sampled_from_logits(uint32_t count, uint32_t eos,
                                        const SamplingParams& params,
                                        const TokenSink& sink, StopCause& cause);

    std::shared_ptr<Snapshot> capture_state();
    void restore_state(const Snapshot& snapshot);
    static constexpr uint32_t vocabulary_size() { return 248320; }
    uint32_t position() const { return position_; }
    SpecStats last_spec_stats() const { return last_spec_stats_; }
    MetalBackend& backend() { return backend_; }
    bool chunked_prefill() const { return chunked_prefill_; }
    void set_chunked_prefill(bool enabled);

  private:
    static constexpr uint32_t N_LAYER = 64;
    static constexpr uint32_t N_EMBD = 5120;
    static constexpr uint32_t N_FFN = 17408;
    static constexpr uint32_t N_HEAD = 24;
    static constexpr uint32_t N_KV = 4;
    static constexpr uint32_t HEAD_DIM = 256;
    static constexpr uint32_t N_ROT = 64;
    static constexpr uint32_t GDN_CH = 10240;
    static constexpr uint32_t GDN_V = 6144;
    static constexpr uint32_t GDN_HEADS = 48;
    static constexpr uint32_t GDN_QK_HEADS = 16;
    static constexpr uint32_t GDN_DIM = 128;
    static constexpr uint32_t VOCAB = 248320;
    static constexpr uint32_t CHUNK_MAX = 12;
    static constexpr uint32_t TOPK_CAPACITY = 1024;
    static constexpr float EPS = 1e-6f;
    static constexpr float FREQ_BASE = 1e7f;

    struct LayerState {
        std::shared_ptr<BackendBuffer> recurrent;
        std::shared_ptr<BackendBuffer> ring;
        std::shared_ptr<BackendBuffer> k_cache;
        std::shared_ptr<BackendBuffer> v_cache;
    };

    std::shared_ptr<Shared> shared_;
    Model& model_;
    MetalBackend& backend_;
    uint32_t max_context_;
    bool turbo3_kv_;
    uint32_t position_ = 0;
    SpecStats last_spec_stats_;
    std::unordered_map<std::string, BackendTensor>& weights_;
    std::vector<LayerState> layers_;

    std::shared_ptr<BackendBuffer> h_, x1_, y_;
    std::shared_ptr<BackendBuffer> qg_, kbuf_, vbuf_, attn_out_;
    std::shared_ptr<BackendBuffer> qkv_, z_, alpha_, beta_raw_, g_, beta_, conv_out_;
    std::shared_ptr<BackendBuffer> delta_out_, gated_out_;
    std::shared_ptr<BackendBuffer> ffn_gate_, ffn_up_, logits_, token_out_;
    // GPU-assisted sampling: top-k candidate over-set staging
    // (Q27_METAL_GPU_SAMPLE=0 forces the full-logits readback path).
    std::shared_ptr<BackendBuffer> topk_values_, topk_indices_, topk_count_;
    bool gpu_sample_ = true;
    std::shared_ptr<BackendBuffer> mtp_embed_norm_, mtp_hidden_norm_, mtp_concat_;
    std::shared_ptr<BackendBuffer> mtp_x_, mtp_hidden_out_, mtp_k_cache_, mtp_v_cache_;
    BackendQuantized q5120_, q6144_, q10240_, q17408_;

    // Layer-major chunked prefill state (CHUNK_MAX token rows per buffer).
    bool chunked_prefill_ = false;
    std::shared_ptr<BackendBuffer> ch_, cx1_, cy_;
    std::shared_ptr<BackendBuffer> cqg_, ckbuf_, cvbuf_, cattn_out_;
    std::shared_ptr<BackendBuffer> cqkv_, cz_, calpha_, cbeta_raw_, cg_, cbeta_, cconv_out_;
    std::shared_ptr<BackendBuffer> cdelta_out_, cgated_out_, cffn_gate_, cffn_up_;
    BackendQuantized cq5120_, cq6144_, cq17408_;

    // Batched MTP verification: per-lane logits/predictions plus the parked
    // GDN inputs and shared discard slots that make the verify chunk
    // state-free — acceptance replays only the GDN recurrence over the
    // accepted prefix, so no checkpoint, restore, or commit re-encode exists.
    // ctargets_/cnll_ also serve teacher-forced NLL quality gates.
    std::shared_ptr<BackendBuffer> cfinal_, clogits_, cpred_;
    std::shared_ptr<BackendBuffer> ctargets_, cnll_;
    std::vector<std::shared_ptr<BackendBuffer>> park_qkv_, park_g_, park_beta_;
    std::shared_ptr<BackendBuffer> discard_recurrent_, discard_ring_;

    // Ternary artifacts (quant_policy bonsai-t2-v1) carry no MTP layer.
    bool has_mtp_ = true;

    std::shared_ptr<BackendBuffer> alloc_f32(uint64_t count);
    const BackendTensor& weight(const std::string& name) const;
    const BackendTensor& layer_weight(uint32_t layer, const char* leaf) const;
    static bool attention_layer(uint32_t layer) { return layer % 4 == 3; }

    void validate_architecture() const;
    uint32_t sample_next(const SamplingParams& params, std::mt19937_64& random);
    void project(const BackendTensor& w, const BackendBuffer& x_float,
                 const BackendQuantized& xq, BackendBuffer& out);
    void project_pair(const BackendTensor& a, BackendBuffer& a_out,
                      const BackendTensor& b, BackendBuffer& b_out,
                      const BackendBuffer& x_float, const BackendQuantized& xq);
    void gdn_block(uint32_t layer);
    void attention_block(uint32_t layer);
    void ffn(uint32_t layer);
    void encode_token(uint32_t token, bool produce_logits);
    void gdn_chunk(uint32_t layer, uint32_t count, bool verify);
    void attention_chunk(uint32_t layer, uint32_t count);
    void ffn_chunk(uint32_t layer, uint32_t count);
    void chunk_forward(const uint32_t* tokens, uint32_t count, bool verify = false);
    void encode_chunk(const uint32_t* tokens, uint32_t count);
    static uint32_t gdn_slot(uint32_t layer) { return layer - (layer + 1) / 4; }
    void gdn_replay(uint32_t count);
    std::vector<uint32_t> generate_mtp_batched(uint32_t pending, uint32_t count,
                                               uint32_t width);
    uint32_t stream_mtp_batched(uint32_t pending, uint32_t count, uint32_t width,
                                uint32_t eos, const TokenSink& sink, StopCause& cause);
    uint32_t prefill(const std::vector<uint32_t>& prompt, bool warm_mtp);
    void mtp_warm(const BackendBuffer& hidden, uint32_t token, uint32_t position);
    uint32_t mtp_forward(const BackendBuffer& hidden, uint32_t token, uint32_t position);
};

} // namespace q27
