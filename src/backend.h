// Backend-neutral device ownership and primitive execution boundary.
#pragma once

#include "loader.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace q27 {

class BackendBuffer {
  public:
    virtual ~BackendBuffer() = default;
    virtual uint64_t size() const = 0;
};

struct BackendQuantized {
    uint32_t count = 0;
    std::shared_ptr<BackendBuffer> values;
    std::shared_ptr<BackendBuffer> scales;
};

struct BackendTensor {
    DType dtype = DType::F32;
    uint64_t rows = 0;
    uint64_t cols = 0;
    std::shared_ptr<BackendBuffer> data;
    std::shared_ptr<BackendBuffer> scales;
    uint64_t data_offset = 0;
    uint64_t scales_offset = 0;
    // Logical tensor extents within the buffers. Buffers may be shared views
    // of a whole model mapping, so buffer size alone cannot bound a tensor;
    // 0 means unknown and falls back to the buffer size.
    uint64_t data_size = 0;
    uint64_t scales_size = 0;
};

// This intentionally starts small. New primitives should describe model
// operations, not CUDA/Metal launch mechanics. The CUDA engine remains the
// reference while its call sites are moved behind this interface incrementally.
class ComputeBackend {
  public:
    virtual ~ComputeBackend() = default;

    virtual std::string name() const = 0;
    virtual std::shared_ptr<BackendBuffer> allocate(uint64_t bytes) = 0;
    virtual void write(BackendBuffer& dst, uint64_t offset, const void* src,
                       uint64_t bytes) = 0;
    virtual void read(const BackendBuffer& src, uint64_t offset, void* dst,
                      uint64_t bytes) = 0;
    virtual void zero(BackendBuffer& dst) = 0;
    virtual void copy(const BackendBuffer& src, uint64_t src_offset,
                      BackendBuffer& dst, uint64_t dst_offset, uint64_t bytes) = 0;
    virtual BackendTensor upload(const Tensor& tensor) = 0;
    // Default may copy. Unified-memory backends can return mmap-backed views;
    // the Model must then outlive the returned tensor.
    virtual BackendTensor upload(const Model& model, const Tensor& tensor) {
        (void)model;
        return upload(tensor);
    }

    virtual void begin_commands() = 0;
    virtual void end_commands() = 0;
    virtual void abort_commands() noexcept = 0;

    // y[rows] = weight[rows, cols] * x[cols], accumulating into F32.
    virtual void matvec(const BackendTensor& weight, const BackendBuffer& x,
                        BackendBuffer& y) = 0;
    // Shared-input projection pair. Backends may fuse dispatch; the default
    // preserves semantics for reference implementations.
    virtual void matvec_pair(const BackendTensor& a, BackendBuffer& a_out,
                             const BackendTensor& b, BackendBuffer& b_out,
                             const BackendBuffer& x) {
        matvec(a, x, a_out); matvec(b, x, b_out);
    }
    virtual BackendQuantized allocate_quantized(uint32_t count) = 0;
    virtual void quantize(const BackendBuffer& x, BackendQuantized& out) = 0;
    virtual void matvec_quantized(const BackendTensor& weight,
                                  const BackendQuantized& x, BackendBuffer& y) = 0;
    virtual void matvec_quantized_pair(const BackendTensor& a, BackendBuffer& a_out,
                                       const BackendTensor& b, BackendBuffer& b_out,
                                       const BackendQuantized& x) {
        matvec_quantized(a, x, a_out); matvec_quantized(b, x, b_out);
    }
    virtual void matmul_quantized(const BackendTensor& weight,const BackendQuantized& x,
                                  uint32_t x_rows,BackendBuffer& y) {
        if(x_rows!=1) throw std::runtime_error("q27: backend has no quantized matmul");
        matvec_quantized(weight,x,y);
    }
    virtual void embedding_q8(const BackendTensor& weight, uint32_t token,
                              BackendBuffer& out) = 0;
    virtual void rmsnorm(const BackendBuffer& x, const BackendTensor& weight,
                         BackendBuffer& out, uint32_t n, float eps) = 0;
    virtual void rmsnorm_quantized(const BackendBuffer& x, const BackendTensor& weight,
                                   BackendBuffer& out, uint32_t n, float eps,
                                   BackendQuantized& quantized) {
        rmsnorm(x,weight,out,n,eps); quantize(out,quantized);
    }
    virtual void rmsnorm_heads(BackendBuffer& x, const BackendTensor& weight,
                               uint32_t heads, uint32_t head_dim, uint32_t stride,
                               float eps) = 0;
    virtual void l2norm_heads(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                              float eps) = 0;
    virtual void silu_mul(const BackendBuffer& gate, const BackendBuffer& up,
                          BackendBuffer& out, uint32_t n) = 0;
    virtual void add_inplace(BackendBuffer& x, const BackendBuffer& y, uint32_t n) = 0;
    virtual void concat(const BackendBuffer& a, uint32_t a_count,
                        const BackendBuffer& b, uint32_t b_count,
                        BackendBuffer& out) = 0;
    virtual void sigmoid_gate_mul(BackendBuffer& out, const BackendBuffer& qg,
                                  uint32_t heads, uint32_t head_dim) = 0;
    virtual void rope_neox(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                           uint32_t n_rot, uint32_t stride, uint32_t position,
                           float freq_base) = 0;
    virtual void argmax(const BackendBuffer& x, uint32_t n, BackendBuffer& out_index) = 0;
    // Top-k candidate extraction for GPU-assisted sampling: writes >= k
    // (value, index) pairs containing the true top-k into values/indices
    // (capacity = min buffer length; implementations may require capacity
    // >= 2k of tie headroom) and the total candidate count into count
    // (uint32). A count above capacity means degenerate ties; the caller
    // must fall back to a full logits readback. Candidates are unordered;
    // the caller sorts. Must run outside a command batch: the count buffer
    // is CPU-cleared immediately before dispatch.
    // x_offset_bytes: byte offset into x for multi-row logits (e.g. clogits_
    // lane * VOCAB * 4). Indices in the output are relative to that row
    // (0..n-1), not absolute buffer indices.
    virtual void topk(const BackendBuffer& x, uint32_t n, uint32_t k,
                      BackendBuffer& values, BackendBuffer& indices, BackendBuffer& count,
                      uint64_t x_offset_bytes = 0) {
        (void)x; (void)n; (void)k; (void)values; (void)indices; (void)count;
        (void)x_offset_bytes;
        throw std::runtime_error("q27: backend has no top-k primitive");
    }
    virtual void kv_store_f16(const BackendBuffer& k, const BackendBuffer& v,
                              BackendBuffer& k_cache, BackendBuffer& v_cache,
                              uint32_t position, uint32_t row_length) = 0;
    virtual void turbo_wht(BackendBuffer& x, uint32_t heads, uint32_t stride,
                           bool inverse) = 0;
    virtual void kv_store_turbo3(const BackendBuffer& k, const BackendBuffer& v,
                                 BackendBuffer& k_cache, BackendBuffer& v_cache,
                                 uint32_t position, uint32_t kv_heads) = 0;
    // partials: caller-owned scratch for the blocked-GQA softmax partials;
    // may be null only when the call cannot route to the blocked kernels
    // (see attention_f16_causal below).
    virtual void attention_turbo3(const BackendBuffer& q, uint32_t q_stride,
                                  const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                                  BackendBuffer& out,
                                  uint32_t seq_len, uint32_t q_heads, uint32_t kv_heads,
                                  uint32_t head_dim, float scale,
                                  BackendBuffer* partials) = 0;
    virtual void attention_f16(const BackendBuffer& q, uint32_t q_stride,
                               const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                               BackendBuffer& out, uint32_t seq_len,
                               uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                               float scale, BackendBuffer* partials) = 0;
    virtual void gdn_gates(const BackendBuffer& alpha, const BackendBuffer& beta_raw,
                           const BackendTensor& ssm_a, const BackendTensor& ssm_dt,
                           BackendBuffer& g, BackendBuffer& beta, uint32_t heads) = 0;
    virtual void conv_step(const BackendBuffer& ring_src, BackendBuffer& ring_dst,
                           const BackendBuffer& qkv, const BackendTensor& conv_weight,
                           BackendBuffer& out, uint32_t channels) = 0;
    virtual void delta_step(const BackendBuffer& state_src, BackendBuffer& state_dst,
                            const BackendBuffer& conv, const BackendBuffer& g,
                            const BackendBuffer& beta, BackendBuffer& out,
                            uint32_t value_heads, uint32_t qk_heads,
                            uint32_t head_dim) = 0;
    virtual void gated_norm_gdn(const BackendBuffer& x, const BackendTensor& weight,
                                const BackendBuffer& gate, BackendBuffer& out,
                                uint32_t heads, uint32_t head_dim, float eps) = 0;

    // Chunked layer-major execution: each call advances `tokens` (2..12)
    // consecutive positions in one dispatch. Recurrent operators commit
    // their carried state exactly once, at the chunk boundary. Backends
    // without chunked support keep the token-serial path.
    virtual void embedding_q8_rows(const BackendTensor& weight, const uint32_t* tokens,
                                   uint32_t count, BackendBuffer& out) {
        (void)weight; (void)tokens; (void)count; (void)out;
        throw std::runtime_error("q27: backend has no chunked execution");
    }
    virtual void rmsnorm_rows_quantized(const BackendBuffer& x, const BackendTensor& weight,
                                        BackendBuffer& out, uint32_t n, uint32_t rows,
                                        float eps, BackendQuantized& quantized) {
        (void)x; (void)weight; (void)out; (void)n; (void)rows; (void)eps; (void)quantized;
        throw std::runtime_error("q27: backend has no chunked execution");
    }
    virtual void matvec_f16_pair_rows(const BackendTensor& a, BackendBuffer& a_out,
                                      const BackendTensor& b, BackendBuffer& b_out,
                                      const BackendBuffer& x, uint32_t rows) {
        (void)a; (void)a_out; (void)b; (void)b_out; (void)x; (void)rows;
        throw std::runtime_error("q27: backend has no chunked execution");
    }
    virtual void gdn_gates_rows(const BackendBuffer& alpha, const BackendBuffer& beta_raw,
                                const BackendTensor& ssm_a, const BackendTensor& ssm_dt,
                                BackendBuffer& g, BackendBuffer& beta,
                                uint32_t heads, uint32_t tokens) {
        (void)alpha; (void)beta_raw; (void)ssm_a; (void)ssm_dt; (void)g; (void)beta;
        (void)heads; (void)tokens;
        throw std::runtime_error("q27: backend has no chunked execution");
    }
    // Chunk state kernels take separate src/dst state bindings (in-place when
    // equal): speculative verification writes dst to a discard slot so the
    // chunk never commits, and acceptance replay commits from parked inputs.
    virtual void conv_chunk(const BackendBuffer& ring_src, BackendBuffer& ring_dst,
                            const BackendBuffer& qkv,
                            const BackendTensor& conv_weight, BackendBuffer& out,
                            uint32_t channels, uint32_t tokens) {
        (void)ring_src; (void)ring_dst; (void)qkv; (void)conv_weight; (void)out;
        (void)channels; (void)tokens;
        throw std::runtime_error("q27: backend has no chunked execution");
    }
    virtual void delta_chunk(const BackendBuffer& state_src, BackendBuffer& state_dst,
                             const BackendBuffer& conv,
                             const BackendBuffer& g, const BackendBuffer& beta,
                             BackendBuffer& out, uint32_t value_heads, uint32_t qk_heads,
                             uint32_t head_dim, uint32_t tokens) {
        (void)state_src; (void)state_dst; (void)conv; (void)g; (void)beta; (void)out;
        (void)value_heads; (void)qk_heads; (void)head_dim; (void)tokens;
        throw std::runtime_error("q27: backend has no chunked execution");
    }
    virtual void l2norm_rows(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                             uint32_t row_stride, uint32_t tokens, float eps) {
        (void)x; (void)heads; (void)head_dim; (void)row_stride; (void)tokens; (void)eps;
        throw std::runtime_error("q27: backend has no chunked execution");
    }
    virtual void rope_neox_rows(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                                uint32_t n_rot, uint32_t stride, uint32_t row_stride,
                                uint32_t position, uint32_t tokens, float freq_base) {
        (void)x; (void)heads; (void)head_dim; (void)n_rot; (void)stride; (void)row_stride;
        (void)position; (void)tokens; (void)freq_base;
        throw std::runtime_error("q27: backend has no chunked execution");
    }
    virtual void kv_store_f16_rows(const BackendBuffer& k, const BackendBuffer& v,
                                   BackendBuffer& k_cache, BackendBuffer& v_cache,
                                   uint32_t position, uint32_t row_length, uint32_t tokens) {
        (void)k; (void)v; (void)k_cache; (void)v_cache; (void)position; (void)row_length;
        (void)tokens;
        throw std::runtime_error("q27: backend has no chunked execution");
    }
    virtual void kv_store_turbo3_rows(const BackendBuffer& k, const BackendBuffer& v,
                                      BackendBuffer& k_cache, BackendBuffer& v_cache,
                                      uint32_t position, uint32_t kv_heads, uint32_t tokens) {
        (void)k; (void)v; (void)k_cache; (void)v_cache; (void)position; (void)kv_heads;
        (void)tokens;
        throw std::runtime_error("q27: backend has no chunked execution");
    }
    // KV-codec attribution store (kl-kv instrument): fp16 cache write with
    // one side (mode 1 = K, mode 2 = V) round-tripped through the turbo3
    // quantizer, so the side's quantization error can be measured alone.
    // head = UINT32_MAX round-trips every head of the selected side; a
    // specific head narrows attribution to one (layer is the caller's cut).
    // flags/scale_off/aux: step-2 scaling arms and the per-feature stats
    // pass (docs/metal/plans/2026-07-16-kv-codec-step2.md). aux may be null when
    // flags need no buffer.
    virtual void kv_store_f16_attrib_rows(const BackendBuffer& k, const BackendBuffer& v,
                                          BackendBuffer& k_cache, BackendBuffer& v_cache,
                                          uint32_t position, uint32_t kv_heads, uint32_t tokens,
                                          uint32_t mode, uint32_t head, uint32_t flags,
                                          uint32_t scale_off, BackendBuffer* aux) {
        (void)k; (void)v; (void)k_cache; (void)v_cache; (void)position; (void)kv_heads;
        (void)tokens; (void)mode; (void)head; (void)flags; (void)scale_off; (void)aux;
        throw std::runtime_error("q27: backend has no KV attribution store");
    }
    // partials: caller-owned scratch for the blocked-GQA softmax partials
    // (engines allocate it once at construction, sized for their own
    // max context — docs/metal/plans/2026-07-17-metal-review-triage.md E2). May
    // be null only when the call cannot route to the blocked kernels.
    virtual void attention_f16_causal(const BackendBuffer& q, uint32_t q_stride,
                                      uint32_t q_row_stride, const BackendBuffer& k_cache,
                                      const BackendBuffer& v_cache,
                                      BackendBuffer& out, uint32_t base_len, uint32_t q_heads,
                                      uint32_t kv_heads, uint32_t head_dim, uint32_t tokens,
                                      float scale, BackendBuffer* partials) {
        (void)q; (void)q_stride; (void)q_row_stride; (void)k_cache; (void)v_cache;
        (void)out; (void)base_len; (void)q_heads; (void)kv_heads;
        (void)head_dim; (void)tokens; (void)scale; (void)partials;
        throw std::runtime_error("q27: backend has no chunked execution");
    }
    virtual void attention_turbo3_causal(const BackendBuffer& q, uint32_t q_stride,
                                         uint32_t q_row_stride, const BackendBuffer& k_cache,
                                         const BackendBuffer& v_cache,
                                         BackendBuffer& out, uint32_t base_len, uint32_t q_heads,
                                         uint32_t kv_heads, uint32_t head_dim, uint32_t tokens,
                                         float scale, BackendBuffer* partials) {
        (void)q; (void)q_stride; (void)q_row_stride; (void)k_cache; (void)v_cache;
        (void)out; (void)base_len; (void)q_heads; (void)kv_heads;
        (void)head_dim; (void)tokens; (void)scale; (void)partials;
        throw std::runtime_error("q27: backend has no chunked execution");
    }
    virtual void sigmoid_gate_mul_rows(BackendBuffer& out, const BackendBuffer& qg,
                                       uint32_t heads, uint32_t head_dim, uint32_t tokens) {
        (void)out; (void)qg; (void)heads; (void)head_dim; (void)tokens;
        throw std::runtime_error("q27: backend has no chunked execution");
    }
    virtual void argmax_rows(const BackendBuffer& x, uint32_t n, uint32_t rows,
                             BackendBuffer& out_indices) {
        (void)x; (void)n; (void)rows; (void)out_indices;
        throw std::runtime_error("q27: backend has no chunked execution");
    }
    // nll[r] = logsumexp(logits[r,:]) - logits[r, targets[r]] over a row-major
    // [rows, n] logit matrix. Targets are uint32 token ids in [0, n).
    virtual void nll_rows(const BackendBuffer& logits, const BackendBuffer& targets,
                          BackendBuffer& nll, uint32_t n, uint32_t rows) {
        (void)logits; (void)targets; (void)nll; (void)n; (void)rows;
        throw std::runtime_error("q27: backend has no NLL primitive");
    }
    virtual void synchronize() = 0;
};

} // namespace q27
