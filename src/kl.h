#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace q27 {

// Forward KL D(p || q) between softmax(p_logits) and softmax(q_logits) in
// double precision: two logsumexp reductions, then one accumulation pass.
// The baseline distribution p weights the per-token log-ratio, matching the
// whitepaper's KV-tolerance methodology (p = the fp16-KV baseline engine).
inline double forward_kl(const float* p_logits, const float* q_logits, uint32_t n) {
    if (!n) throw std::runtime_error("q27: forward_kl needs a non-empty vocabulary");
    double p_max = -1e300, q_max = -1e300;
    for (uint32_t i = 0; i < n; i++) {
        p_max = std::max(p_max, (double)p_logits[i]);
        q_max = std::max(q_max, (double)q_logits[i]);
    }
    double p_sum = 0.0, q_sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        p_sum += std::exp((double)p_logits[i] - p_max);
        q_sum += std::exp((double)q_logits[i] - q_max);
    }
    const double p_lse = std::log(p_sum) + p_max, q_lse = std::log(q_sum) + q_max;
    double kl = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        const double log_p = (double)p_logits[i] - p_lse;
        kl += std::exp(log_p) * (log_p - ((double)q_logits[i] - q_lse));
    }
    return kl;
}

} // namespace q27
