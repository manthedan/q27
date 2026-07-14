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
                          BackendBuffer& scratch, BackendBuffer& out,
                          uint32_t seq_len, uint32_t q_heads, uint32_t kv_heads,
                          uint32_t head_dim, float scale) override;
    void attention_f16(const BackendBuffer& q, uint32_t q_stride,
                       const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                       BackendBuffer& scratch, BackendBuffer& out, uint32_t seq_len,
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
    void synchronize() override;

    uint64_t recommended_working_set_size() const;
    uint64_t max_buffer_length() const;
    uint64_t max_threadgroup_memory_length() const;
    bool supports_quantized_matmul() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace q27
