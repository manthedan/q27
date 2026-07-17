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

class SuffixDraft;

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
    // Contract: engines on one Shared must be constructed and driven from a
    // single thread (or externally serialized) — they alias one command
    // queue and one batching state, and nothing here locks.
    struct Shared {
        Model model;
        MetalBackend backend;
        std::unordered_map<std::string, BackendTensor> weights;
        // Combined KV-cache footprint of every engine on this mapping, so a
        // second engine cannot pass the per-engine budget check while the
        // pair overcommits the device (codex review finding, 2026-07-15).
        uint64_t cache_bytes = 0;
        // Artifact path, kept for diagnostics, plus the disk-snapshot
        // identity: SHA1 over the WHOLE mapped artifact — the bytes this
        // process actually opened, immune to pathname swaps — computed
        // lazily on first snapshot use and cached (~2 s for 7 GB), never
        // on ordinary engine startup (docs/plans/2026-07-16-prefix-
        // snapshots.md; codex P1+P2 on 39d74a0).
        std::string path;
        unsigned char snap_sha1[20] = {};
        bool snap_sha_ready = false;
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
    ~MetalEngine();

    // G6 admission accounting (docs/plans/2026-07-16-g6-admission.md).
    // These mirror the constructor's allocations and capture_state()'s
    // snapshot composition — keep them paired with those sites.
    bool has_mtp() const { return has_mtp_; }
    // KV caches plus this engine's blocked-GQA partials scratch — every
    // ctx-scaled reservation the constructor charges against the shared
    // cross-engine budget (audit E2: partials are per-engine now).
    uint64_t kv_reserved_bytes() const { return engine_cache_bytes_; }
    uint64_t snapshot_bytes() const;               // worst case at max_context_
    // Per-engine non-KV buffers; the chunk/verify/replay terms exist only
    // when chunked prefill is available (pre-Apple7 devices skip them).
    // The GQA partials are charged inside kv_reserved_bytes, not here.
    static uint64_t fixed_state_bytes(bool chunked);
    // Per-engine causal-GQA partials buffer at the widest available
    // attention width (one token when chunked prefill is unavailable),
    // sized from the backend's EFFECTIVE block (Q27_METAL_GQA_BLOCK-aware).
    // Allocated eagerly by the constructor; the dispatch hot path only
    // bounds-checks (audit C3). Sizing never depends on the GQA threshold,
    // so runtime threshold flips (envelope instrument) only change routing.
    static uint64_t gqa_partial_peak(uint32_t context, uint32_t block, bool chunked);

    void reset();
    uint32_t step(uint32_t token);
    std::vector<uint32_t> generate(const std::vector<uint32_t>& prompt, uint32_t count);
    std::vector<uint32_t> generate_mtp(const std::vector<uint32_t>& prompt,
                                       uint32_t count, uint32_t width);
    // Verify/oracle width ceiling, decoupled from the width-12 NLL/KL
    // contract exactly as PREFILL_CHUNK_MAX decoupled prompt ingestion
    // (docs/plans/2026-07-16-lever2-verify-width.md). Sizes cfinal_/
    // clogits_/cpred_ and the gdn_replay parks; mtp_round stays capped at
    // CHUNK_MAX until the MTP lane machinery is testable (24 GB rig).
    static constexpr uint32_t VERIFY_CHUNK_MAX = 48;
    // Default drafter engage threshold: rounds whose longest suffix match is
    // shorter fall back to one serial step (Phase-0 sim: shorter thresholds
    // trade acceptance for round overhead on burst-hostile text).
    static constexpr uint32_t SUFFIX_MIN_MATCH = 12;
    // Batched suffix-burst verification (2026-07-16-suffix-burst-verify.md):
    // SuffixDraft proposals through the VERIFY_CHUNK_MAX-wide verify chunk
    // with mtp_round's acceptance/commit semantics. width 2..VERIFY_CHUNK_MAX;
    // rounds with match < minimum_match fall back to one serial step.
    std::vector<uint32_t> generate_suffix(const std::vector<uint32_t>& prompt,
                                          uint32_t count, uint32_t width,
                                          uint32_t minimum_match = SUFFIX_MIN_MATCH);
    // The pre-lever-2 serial walk (one step() per proposal): the batched
    // path's A/B control and byte-level reference. width 2..12.
    std::vector<uint32_t> generate_suffix_serial(const std::vector<uint32_t>& prompt,
                                                 uint32_t count, uint32_t width,
                                                 uint32_t minimum_match = SUFFIX_MIN_MATCH);
    // One suffix-burst driver round (the generate_suffix loop body, extracted
    // for the server's quantum loop): propose from the drafter, match-cap and
    // snap-down the width, then either a suffix_round burst or one serial
    // fallback step. Fills committed with the round's tokens (>= 1, clamped at
    // eos exactly like mtp_round — pass the REAL eos id here; generate_suffix
    // passes a never-matching sentinel to run to a fixed count) and returns
    // the next pending token. remaining must be >= 2 (the caller emits the
    // final token directly). burst, when non-null, reports whether this
    // round dispatched a batched verify (server /stats attribution).
    uint32_t suffix_step(SuffixDraft& drafter, uint32_t pending, uint32_t remaining,
                         uint32_t eos, uint32_t width, uint32_t minimum_match,
                         std::vector<uint32_t>& committed, bool* burst = nullptr);
    // Suffix-burst diagnostics for the last generate_suffix run: rounds that
    // fell back to serial, and fired-burst lane counts by full-tile bucket.
    struct SuffixStats {
        uint64_t fallback_rounds = 0;
        uint64_t burst_rounds = 0;
        uint64_t lanes_le16 = 0, lanes_32 = 0, lanes_48 = 0; // dispatched live widths
    };
    SuffixStats last_suffix_stats() const { return last_suffix_stats_; }
    uint32_t ingest_prompt(const std::vector<uint32_t>& tokens, bool warm_mtp,
                           bool reset_first = true);
    std::vector<uint32_t> generate_from_pending(uint32_t pending, uint32_t count,
                                                uint32_t mtp_width = 0);
    std::vector<float> read_logits();
    // x1_ readback — the hidden row capture_state/save_state persist — for
    // the --chunk-parity hidden-row leg (k3 audit A1/E3).
    void read_hidden(std::vector<float>& out);
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
    // Wide-path variant (round-2 expert P0 #3 gate): encode tokens[0..count)
    // through the PROMPT-INGESTION chunk width (count 1..PREFILL_CHUNK_MAX;
    // <= CHUNK_MAX delegates to teacher_force_logits) with the output head
    // applied in CHUNK_MAX-row slices, exposing every row's logits so
    // distribution-level gates (NLL/KL/top-k/margin) can cover widths
    // 17/48/96 — previously only committed-token A/Bs saw the wide path.
    void teacher_force_logits_wide(const uint32_t* tokens, uint32_t count,
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

    // Scheduling-quantum surface (multislot Phase 1,
    // docs/plans/2026-07-15-multislot-phase1.md): each call submits bounded
    // GPU work so a serving scheduler can interleave engines on one Shared.
    //
    // prefill_chunk encodes 2..PREFILL_CHUNK_MAX prompt tokens through the
    // layer-major chunk path without producing logits. Chunk-boundary
    // placement is quality-neutral (--chunk-parity gate: widths 17/48/96
    // bit-identical to 12), so the caller picks any width per call; the
    // final prompt token still goes through step() to produce logits and
    // the pending token, exactly like prefill()'s serial tail.
    void prefill_chunk(const uint32_t* tokens, uint32_t count);
    // One MTP draft/verify/commit round (one scheduling quantum). Appends
    // the committed tokens (always starting with `pending`) to `committed`;
    // the caller emits them and stops at `eos` itself — tokens after an EOS
    // were already encoded when the verify chunk ran, exactly as in the
    // streaming commit loop. Adapts live_width in place (callers initialize
    // it to min(width, 4)) and returns the next pending token. When context
    // or `remaining` (>= 2, bounds committed tokens) leaves no room to
    // verify, falls back to one serial step — skipped when `pending` is EOS
    // so a finished stream never encodes past its end.
    uint32_t mtp_round(uint32_t pending, uint32_t remaining, uint32_t eos, uint32_t width,
                       uint32_t& live_width, std::vector<uint32_t>& committed);
    // Sample one token from the current logits; the sampled-decode quantum
    // is sample_from_logits + step under one GPU lease. The RNG belongs to
    // the request, not the engine, so interleaved slots stay reproducible.
    uint32_t sample_from_logits(const SamplingParams& params, std::mt19937_64& rng);
    // Gate 0 oracle round (sibling-drafter-probe doc): one batched verify
    // round with the draft stage removed — `lanes` are caller-supplied
    // reference tokens (lanes[0] = pending), commit is teacher-forced to all
    // `live` lanes so verifier economics are measured at perfect acceptance
    // (D=0) with no drafter in the loop. Works on artifacts without an MTP
    // layer. Writes the per-lane argmax verdicts to `predictions[live]`
    // (observational agreement only — never acted on). `last` marks the
    // final round of a generation: the final token is committed but never
    // encoded, exactly like mtp_round/serial semantics.
    void oracle_round(const uint32_t* lanes, uint32_t live, bool last, uint32_t* predictions);
    // Suffix-burst round: oracle_round's caller-lane verify machinery with
    // mtp_round's real acceptance walk and early-EOS clamp. lanes[0] must be
    // the pending token; live = 2..VERIFY_CHUNK_MAX. Returns the next pending.
    uint32_t suffix_round(uint32_t remaining, uint32_t eos, const uint32_t* lanes,
                          uint32_t live, std::vector<uint32_t>& committed);

    std::shared_ptr<Snapshot> capture_state();
    void restore_state(const Snapshot& snapshot);
    // Prefix snapshots to disk (docs/plans/2026-07-16-prefix-snapshots.md,
    // Phase 1): the capture/restore composition streamed through host
    // memory with plain file I/O (no mmap — ds4's lesson). save writes
    // path.tmp then renames; load validates the whole file structure
    // (magic, artifact identity, kv dtype, position, every blob length)
    // before the first GPU write, so a rejected file never leaves mixed
    // state. tokens are prefix metadata for the Phase-2 server keying.
    // logits_resident=false marks a snapshot taken mid-prefill (chunk path):
    // its state is exact but the resident logits row is stale, so a resume
    // may continue ingestion from token[position] onward but must never
    // derive a pending token from the stored logits.
    void save_state(const std::string& path, const uint32_t* tokens, uint32_t token_count,
                    bool logits_resident = true);
    uint32_t load_state(const std::string& path);   // returns restored position
    // Header-only inspection for prefix keying (server): position, stored
    // token ids, and the logits-resident flag. Validates magic only — the
    // full structural validation happens on load_state.
    struct SnapshotInfo { uint32_t position = 0; bool logits_resident = true;
                          std::vector<uint32_t> tokens; };
    static SnapshotInfo peek_snapshot(const std::string& path);
    // Re-run the resident-logits argmax (same GPU kernel as prompt
    // ingestion) so generation after load_state resumes byte-identically.
    uint32_t pending_from_logits();
    const unsigned char* snapshot_identity();   // 20-byte SHA1, whole mapping, cached

    // Constrained tool decoding (BasicToolConstrainer engine surface): an
    // append-only device pool of uint32 legal-token bitsets. mask_pool_add
    // uploads one mask and returns its slot (-1 when the pool is full);
    // set_tool_constraint selects the active slot (-1 disengages). The
    // active mask is applied to the logits inside encode_token, before the
    // argmax, so greedy decode, top-k extraction, and CPU sampling all see
    // only legal tokens. Serial decode only; the MTP verify lanes
    // (set_tool_masks5) are not wired on Metal.
    static constexpr int MASK_POOL_CAP = 64;
    int mask_pool_used = 0;
    int mask_pool_add(const void* bits);
    void set_tool_constraint(int mask_id);
    static constexpr uint32_t vocabulary_size() { return 248320; }
    uint32_t position() const { return position_; }
    SpecStats last_spec_stats() const { return last_spec_stats_; }
    MetalBackend& backend() { return backend_; }
    bool chunked_prefill() const { return chunked_prefill_; }
    void set_chunked_prefill(bool enabled);
    // KV-codec attribution knob (kl-kv instrument): 0 = off, 1 = round-trip
    // K through the turbo3 quantizer at store time, 2 = V. The engine keeps
    // its fp16 cache and attention kernels, so the resulting KL against an
    // fp16 baseline isolates one side's quantization error. fp16-KV engines
    // only; set before any tokens are encoded.
    void set_kv_attrib(uint32_t mode);
    // Cell-granular variant (sensitivity census): restrict the round-trip
    // to one attention layer (absolute index, layer%4==3) and one KV head.
    // UINT32_MAX for layer/head widens that axis back to "all".
    void set_kv_attrib_cell(uint32_t mode, uint32_t layer, uint32_t head);
    // Step-2 scaling arms (docs/plans/2026-07-16-kv-codec-step2.md), on top
    // of an already-selected side arm: scale32 keeps the group scale in f32
    // through the round-trip; feature_scales (2*16*4*256 floats,
    // [side][attn_idx][head][dim]) descale each dimension before the
    // quantizer and rescale after the inverse transform. Instrument-only,
    // main-stream attention layers (never the MTP layer).
    void set_kv_attrib_rt(bool scale32, const float* feature_scales);
    // Stats pass: subject stores clean fp16 (KL vs baseline is exactly 0 —
    // a rode-along canary) while per-feature sum-of-squares accumulates for
    // both sides. read_kv_attrib_stats returns the 2*16*4*256 accumulator.
    void set_kv_attrib_stats();
    void read_kv_attrib_stats(std::vector<float>& out);
    // Step-4 exception probe (docs/plans/2026-07-17-kv-codec-step4-probe.md):
    // round-trip BOTH sides of every attention layer through the turbo3
    // quantizer EXCEPT the listed census cells (attn_idx*8 + head*2 + side),
    // which stay clean fp16. n == 0 is the control arm (everything
    // quantized). fp16-KV engines only; set before any tokens are encoded;
    // not combinable with the step-2 rt flags.
    void set_kv_attrib_except(const uint32_t* cells, size_t n);
    // True when Q27_METAL_KV_FP16_CELLS armed production fp16 exception
    // side caches on this engine (snapshots are refused in v1; the server
    // disables its prefix cache off this bit).
    bool kv_fp16_except() const { return kv_fp16_except_; }

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
    static constexpr uint32_t CHUNK_MAX = 12;          // MTP verify / NLL / KL width
    static constexpr uint32_t PREFILL_CHUNK_MAX = 96;  // prompt-ingestion width (8x12)
    static constexpr uint32_t TOPK_CAPACITY = 1024;
    static constexpr uint32_t RESIDENT_MAX = 8;
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
    uint32_t kv_attrib_ = 0;
    uint32_t kv_attrib_layer_ = UINT32_MAX;
    uint32_t kv_attrib_head_ = UINT32_MAX;
    uint32_t kv_attrib_flags_ = 0;
    // Mode-3 per-attn-layer exception masks (bit = head*2 + side); rides
    // the kernel's head argument at the attrib store call sites.
    uint8_t kv_attrib_masks_[16] = {};
    std::shared_ptr<BackendBuffer> kv_attrib_aux_;
    // Production KV fp16 exception cells
    // (docs/plans/2026-07-17-kv-except-production.md): per masked head, a
    // kv_heads=1 fp16 side cache in the turbo3 WHT domain; the head's
    // query-head window is re-attended f16 against it, overwriting the
    // production dispatch's output rows. Parsed from
    // Q27_METAL_KV_FP16_CELLS (census cell numbers, per-head K+V pairs
    // required) in the constructor; turbo3 engines only (ignored with a
    // note on fp16 engines — those cells are already fp16). Indexed by
    // attn_idx = layer/4. Bytes ride engine_cache_bytes_.
    struct KvFp16Side { uint32_t head; std::shared_ptr<BackendBuffer> k, v; };
    std::array<std::vector<KvFp16Side>, 16> kv_fp16_side_;
    bool kv_fp16_except_ = false;
    uint64_t engine_cache_bytes_ = 0;
    // Blocked-GQA softmax partials, engine-owned (audit E2): allocated once
    // in the constructor at gqa_partial_peak, GPU-private, freed with the
    // engine. Passed into every backend attention call.
    std::shared_ptr<BackendBuffer> gqa_partials_;
    uint32_t position_ = 0;
    SpecStats last_spec_stats_;
    SuffixStats last_suffix_stats_;
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
    std::shared_ptr<BackendBuffer> mask_pool_;   // lazy: MASK_POOL_CAP bitsets
    int active_mask_ = -1;
    // GPU-resident greedy decode: K chained steps per command buffer, token
    // ids archived device-side (Q27_METAL_RESIDENT=0 opts out).
    std::shared_ptr<BackendBuffer> token_ring_;
    bool resident_ = true;
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
    std::shared_ptr<BackendBuffer> wide_head_stage_;   // lazy, CHUNK_MAX x N_EMBD f32
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
    uint32_t decode_resident(uint32_t pending, uint32_t* out, uint32_t k);
    void project(const BackendTensor& w, const BackendBuffer& x_float,
                 const BackendQuantized& xq, BackendBuffer& out);
    void project_pair(const BackendTensor& a, BackendBuffer& a_out,
                      const BackendTensor& b, BackendBuffer& b_out,
                      const BackendBuffer& x_float, const BackendQuantized& xq);
    void gdn_block(uint32_t layer);
    void attention_block(uint32_t layer, uint32_t pos);
    void ffn(uint32_t layer);
    void encode_token(uint32_t token, bool produce_logits, bool token_from_device = false,
                      uint32_t pos_offset = 0);
    void gdn_chunk(uint32_t layer, uint32_t count, bool verify);
    void attention_chunk(uint32_t layer, uint32_t count);
    void ffn_chunk(uint32_t layer, uint32_t count);
    void chunk_forward(const uint32_t* tokens, uint32_t count, bool verify = false);
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
