#pragma once

#include "../backend.h"

#include <memory>
#include <string>

namespace q27 {

class MetalBackend final : public ComputeBackend {
  public:
    MetalBackend();
    ~MetalBackend() override;
    MetalBackend(const MetalBackend&) = delete;
    MetalBackend& operator=(const MetalBackend&) = delete;

    std::string name() const override;
    std::shared_ptr<BackendBuffer> allocate(uint64_t bytes) override;
    void write(BackendBuffer& dst, uint64_t offset, const void* src,
               uint64_t bytes) override;
    void read(const BackendBuffer& src, uint64_t offset, void* dst,
              uint64_t bytes) override;
    void zero(BackendBuffer& dst) override;
    void copy(const BackendBuffer& src, uint64_t src_offset,
              BackendBuffer& dst, uint64_t dst_offset, uint64_t bytes) override;
    BackendTensor upload(const Tensor& tensor) override;
    BackendTensor upload(const Model& model, const Tensor& tensor) override;
    void begin_commands() override;
    void end_commands() override;
    void abort_commands() noexcept override;
    void matvec(const BackendTensor& weight, const BackendBuffer& x,
                BackendBuffer& y) override;
    void matvec_pair(const BackendTensor& a, BackendBuffer& a_out,
                     const BackendTensor& b, BackendBuffer& b_out,
                     const BackendBuffer& x) override;
    BackendQuantized allocate_quantized(uint32_t count) override;
    void quantize(const BackendBuffer& x, BackendQuantized& out) override;
    void matvec_quantized(const BackendTensor& weight,
                          const BackendQuantized& x, BackendBuffer& y) override;
    void matvec_quantized_pair(const BackendTensor& a, BackendBuffer& a_out,
                               const BackendTensor& b, BackendBuffer& b_out,
                               const BackendQuantized& x) override;
    // N=2 slot-batched T2 GEMV (Phase 2 probe): x = 2 activation rows
    // ([2,cols] values / [2,cols/32] scales), y = [2,rows] token-major.
    // PARKED by measurement (2026-07-16): aggregate s_k 1.093 vs the 1.31
    // decision line — bench-only reference surface, never engine-routed.
    void matvec_quantized_x2(const BackendTensor& weight,
                             const BackendQuantized& x, BackendBuffer& y);
    // Select-form float-activation variant (the production serial-decode
    // path), two independent x/y buffer pairs. Same PARKED status.
    void matvec_x2(const BackendTensor& weight,
                   const BackendBuffer& x_a, const BackendBuffer& x_b,
                   BackendBuffer& y_a, BackendBuffer& y_b);
    // A/B/C MMA roofline probe (bench-only): arm 'a' = MMA-core ceiling
    // (w_or_seed = opaque tile seed, other pointers null), arm 'b' =
    // half-plumbing (half weights/scales/activations, float x scales).
    // Same dispatch grid and tile geometry as the production T2 GEMM.
    void mma_roofline(char arm, uint32_t rows, uint32_t cols, uint32_t x_rows,
                      const BackendBuffer& w_or_seed, const BackendBuffer* w_scales,
                      const BackendBuffer* x, const BackendBuffer* x_scales,
                      BackendBuffer& y);
    // B1 Phase 0B probe (bench-only, docs/plans/2026-07-15-binary-tier.md):
    // candidate 1 select / 2 sign-XOR / 3 int8 bitplane+popcount, raw
    // buffers, no DType. Candidate 3 dispatches its activation preprocess
    // (int8 quantize + bitplane transpose + group sums) before the dot, so
    // both land inside any timed region; scratch holds its planes + aux
    // ((cols/128)*136 bytes) and is ignored by candidates 1-2. Never
    // engine-routed.
    void matvec_b1_probe(int candidate, uint32_t rows, uint32_t cols,
                         const BackendBuffer& bits, const BackendBuffer& scales,
                         const BackendBuffer& x, BackendBuffer* scratch,
                         BackendBuffer& y);
    void matmul_quantized(const BackendTensor& weight,const BackendQuantized& x,
                          uint32_t x_rows,BackendBuffer& y) override;
    void embedding_q8(const BackendTensor& weight, uint32_t token,
                      BackendBuffer& out) override;
    void rmsnorm(const BackendBuffer& x, const BackendTensor& weight,
                 BackendBuffer& out, uint32_t n, float eps) override;
    void rmsnorm_quantized(const BackendBuffer& x, const BackendTensor& weight,
                           BackendBuffer& out, uint32_t n, float eps,
                           BackendQuantized& quantized) override;
    void rmsnorm_heads(BackendBuffer& x, const BackendTensor& weight,
                       uint32_t heads, uint32_t head_dim, uint32_t stride,
                       float eps) override;
    void l2norm_heads(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                      float eps) override;
    void silu_mul(const BackendBuffer& gate, const BackendBuffer& up,
                  BackendBuffer& out, uint32_t n) override;
    void add_inplace(BackendBuffer& x, const BackendBuffer& y, uint32_t n) override;
    void concat(const BackendBuffer& a, uint32_t a_count,
                const BackendBuffer& b, uint32_t b_count,
                BackendBuffer& out) override;
    void sigmoid_gate_mul(BackendBuffer& out, const BackendBuffer& qg,
                          uint32_t heads, uint32_t head_dim) override;
    void rope_neox(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                   uint32_t n_rot, uint32_t stride, uint32_t position,
                   float freq_base) override;
    void argmax(const BackendBuffer& x, uint32_t n, BackendBuffer& out_index) override;
    void topk(const BackendBuffer& x, uint32_t n, uint32_t k,
              BackendBuffer& values, BackendBuffer& indices, BackendBuffer& count) override;
    // Constrained decoding: -inf every logit whose bit is clear in the
    // uint32 bitset at mask_offset (word-aligned) inside masks.
    void mask_logits(BackendBuffer& logits, const BackendBuffer& masks,
                     uint64_t mask_offset, uint32_t n);
    // GPU-resident greedy decode: embedding row selected by a device-side
    // token id (the previous step's argmax output) — no CPU sync between
    // chained decode steps.
    void embedding_from_device(const BackendTensor& weight, const BackendBuffer& token,
                               BackendBuffer& out);
    void kv_store_f16(const BackendBuffer& k, const BackendBuffer& v,
                      BackendBuffer& k_cache, BackendBuffer& v_cache,
                      uint32_t position, uint32_t row_length) override;
    void turbo_wht(BackendBuffer& x, uint32_t heads, uint32_t stride,
                   bool inverse) override;
    void kv_store_turbo3(const BackendBuffer& k, const BackendBuffer& v,
                         BackendBuffer& k_cache, BackendBuffer& v_cache,
                         uint32_t position, uint32_t kv_heads) override;
    void attention_turbo3(const BackendBuffer& q, uint32_t q_stride,
                          const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                          BackendBuffer& out,
                          uint32_t seq_len, uint32_t q_heads, uint32_t kv_heads,
                          uint32_t head_dim, float scale) override;
    void attention_f16(const BackendBuffer& q, uint32_t q_stride,
                       const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                       BackendBuffer& out, uint32_t seq_len,
                       uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                       float scale) override;
    void gdn_gates(const BackendBuffer& alpha, const BackendBuffer& beta_raw,
                   const BackendTensor& ssm_a, const BackendTensor& ssm_dt,
                   BackendBuffer& g, BackendBuffer& beta, uint32_t heads) override;
    void conv_step(const BackendBuffer& ring_src, BackendBuffer& ring_dst,
                   const BackendBuffer& qkv, const BackendTensor& conv_weight,
                   BackendBuffer& out, uint32_t channels) override;
    void delta_step(const BackendBuffer& state_src, BackendBuffer& state_dst,
                    const BackendBuffer& conv, const BackendBuffer& g,
                    const BackendBuffer& beta, BackendBuffer& out,
                    uint32_t value_heads, uint32_t qk_heads,
                    uint32_t head_dim) override;
    void gated_norm_gdn(const BackendBuffer& x, const BackendTensor& weight,
                        const BackendBuffer& gate, BackendBuffer& out,
                        uint32_t heads, uint32_t head_dim, float eps) override;
    void embedding_q8_rows(const BackendTensor& weight, const uint32_t* tokens,
                           uint32_t count, BackendBuffer& out) override;
    void rmsnorm_rows_quantized(const BackendBuffer& x, const BackendTensor& weight,
                                BackendBuffer& out, uint32_t n, uint32_t rows,
                                float eps, BackendQuantized& quantized) override;
    void matvec_f16_pair_rows(const BackendTensor& a, BackendBuffer& a_out,
                              const BackendTensor& b, BackendBuffer& b_out,
                              const BackendBuffer& x, uint32_t rows) override;
    void gdn_gates_rows(const BackendBuffer& alpha, const BackendBuffer& beta_raw,
                        const BackendTensor& ssm_a, const BackendTensor& ssm_dt,
                        BackendBuffer& g, BackendBuffer& beta,
                        uint32_t heads, uint32_t tokens) override;
    void conv_chunk(const BackendBuffer& ring_src, BackendBuffer& ring_dst,
                    const BackendBuffer& qkv,
                    const BackendTensor& conv_weight, BackendBuffer& out,
                    uint32_t channels, uint32_t tokens) override;
    void delta_chunk(const BackendBuffer& state_src, BackendBuffer& state_dst,
                     const BackendBuffer& conv,
                     const BackendBuffer& g, const BackendBuffer& beta,
                     BackendBuffer& out, uint32_t value_heads, uint32_t qk_heads,
                     uint32_t head_dim, uint32_t tokens) override;
    void l2norm_rows(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                     uint32_t row_stride, uint32_t tokens, float eps) override;
    void rope_neox_rows(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                        uint32_t n_rot, uint32_t stride, uint32_t row_stride,
                        uint32_t position, uint32_t tokens, float freq_base) override;
    void kv_store_f16_rows(const BackendBuffer& k, const BackendBuffer& v,
                           BackendBuffer& k_cache, BackendBuffer& v_cache,
                           uint32_t position, uint32_t row_length, uint32_t tokens) override;
    void kv_store_turbo3_rows(const BackendBuffer& k, const BackendBuffer& v,
                              BackendBuffer& k_cache, BackendBuffer& v_cache,
                              uint32_t position, uint32_t kv_heads, uint32_t tokens) override;
    void kv_store_f16_attrib_rows(const BackendBuffer& k, const BackendBuffer& v,
                                  BackendBuffer& k_cache, BackendBuffer& v_cache,
                                  uint32_t position, uint32_t kv_heads, uint32_t tokens,
                                  uint32_t mode, uint32_t head, uint32_t flags,
                                  uint32_t scale_off, BackendBuffer* aux) override;
    void attention_f16_causal(const BackendBuffer& q, uint32_t q_stride,
                              uint32_t q_row_stride, const BackendBuffer& k_cache,
                              const BackendBuffer& v_cache,
                              BackendBuffer& out, uint32_t base_len, uint32_t q_heads,
                              uint32_t kv_heads, uint32_t head_dim, uint32_t tokens,
                              float scale) override;
    void attention_turbo3_causal(const BackendBuffer& q, uint32_t q_stride,
                                 uint32_t q_row_stride, const BackendBuffer& k_cache,
                                 const BackendBuffer& v_cache,
                                 BackendBuffer& out, uint32_t base_len, uint32_t q_heads,
                                 uint32_t kv_heads, uint32_t head_dim, uint32_t tokens,
                                 float scale) override;
    // Phase-0 probes for cache-block scheduling R1/R1b — bench-only entry
    // points (build/metal_attn_bench), never engine-routed; see
    // docs/plans/2026-07-15-cache-block-scheduling.md. k/v caches hold rows
    // head-major: (kvh * seq_cap + pos) * 100 bytes.
    void attention_turbo3_gqa_headmajor(const BackendBuffer& q, uint32_t q_stride,
                                        const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                                        BackendBuffer& out, uint32_t seq_len, uint32_t seq_cap,
                                        uint32_t q_heads, uint32_t kv_heads,
                                        uint32_t head_dim, float scale);
    // tile must be 2 or 4; interleaved (production-layout) caches.
    // R3 probe (bench-only): barrier-free direct-read block-partial causal
    // GQA at token factor 2 with an explicit block-size override.
    void attention_turbo3_causal_gqa_bf(const BackendBuffer& q, uint32_t q_stride,
                                        uint32_t q_row_stride,
                                        const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                                        BackendBuffer& out, uint32_t base_len,
                                        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                                        uint32_t tokens, uint32_t block, float scale);
    void attention_turbo3_causal_gqa_tiled(const BackendBuffer& q, uint32_t q_stride,
                                           uint32_t q_row_stride,
                                           const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                                           BackendBuffer& out, uint32_t base_len,
                                           uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                                           uint32_t tokens, uint32_t tile, float scale);
    void sigmoid_gate_mul_rows(BackendBuffer& out, const BackendBuffer& qg,
                               uint32_t heads, uint32_t head_dim, uint32_t tokens) override;
    void argmax_rows(const BackendBuffer& x, uint32_t n, uint32_t rows,
                     BackendBuffer& out_indices) override;
    void nll_rows(const BackendBuffer& logits, const BackendBuffer& targets,
                  BackendBuffer& nll, uint32_t n, uint32_t rows) override;
    void synchronize() override;

    // Clears Q27_METAL_PROFILE accumulation (stats, command-buffer and
    // busy/wait counters) so benches can exclude their warmup dispatches
    // from the attribution table. No-op when profiling is disabled.
    void profile_reset();

    uint64_t recommended_working_set_size() const;
    // Effective causal-GQA block size (Q27_METAL_GQA_BLOCK or 1024): the
    // admission accounting sizes the shared partial buffer from it.
    uint32_t gqa_block_size() const;
    // Whether the blocked (partials-allocating) GQA route is reachable at
    // this context: threshold nonzero and context deep enough to route it.
    bool gqa_blocked_reachable(uint32_t context) const;
    // Envelope-instrument hooks (docs/plans/2026-07-16-envelope-instrument.md):
    // flip the backend-global reduction-order knobs between two engines'
    // lockstep passes. Instrument use only — production reads the env once.
    // CONTRACT: these are BACKEND-scoped, not per-engine. Engines sharing
    // one backend (Shared mapping) see every flip; only one engine may
    // mutate them, and never while another engine's pass is in flight —
    // the envelope instrument flips them sequentially by design. Concurrent
    // flips would corrupt any A/B attribution riding on them.
    void set_gemm_half(bool enabled);
    void set_gqa_threshold(uint32_t threshold);
    uint64_t max_buffer_length() const;
    uint64_t max_threadgroup_memory_length() const;
    bool supports_quantized_matmul() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace q27
