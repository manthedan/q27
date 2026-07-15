#include "kl.h"
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    // Deterministic synthetic logits.
    const uint32_t n = 2048;
    std::vector<float> p(n), q(n);
    uint32_t lcg = 8642;
    auto uniform = [&]() { lcg = lcg * 1664525u + 1013904223u; return (float)(lcg >> 8) / 8388608.0f - 1.0f; };
    for (uint32_t i = 0; i < n; i++) { p[i] = uniform() * 9.0f; q[i] = p[i] + uniform() * 0.25f; }

    // KL(p, p) is exactly zero (identical doubles cancel term by term).
    if (q27::forward_kl(p.data(), p.data(), n) != 0.0) { fprintf(stderr, "kl: self-KL nonzero\n"); return 1; }
    // Gibbs: forward KL is non-negative.
    const double pq = q27::forward_kl(p.data(), q.data(), n);
    const double qp = q27::forward_kl(q.data(), p.data(), n);
    if (!(pq > 0.0) || !(qp > 0.0)) { fprintf(stderr, "kl: non-positive on perturbed pair\n"); return 1; }
    // Asymmetry: the two directions differ on generic inputs.
    if (pq == qp) { fprintf(stderr, "kl: suspicious symmetry\n"); return 1; }
    // Agreement with a naive softmax reference.
    auto naive = [&](const std::vector<float>& a, const std::vector<float>& b) {
        double as = 0.0, bs = 0.0;
        for (uint32_t i = 0; i < n; i++) { as += std::exp((double)a[i]); bs += std::exp((double)b[i]); }
        double kl = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            const double pa = std::exp((double)a[i]) / as, pb = std::exp((double)b[i]) / bs;
            kl += pa * std::log(pa / pb);
        }
        return kl;
    };
    if (std::fabs(pq - naive(p, q)) > 1e-10 * (1.0 + std::fabs(pq))) {
        fprintf(stderr, "kl: disagrees with naive reference (%.17g vs %.17g)\n", pq, naive(p, q));
        return 1;
    }
    // A concentrated shift on the argmax token moves KL far more than the
    // same shift on a tail token (sanity that p weights the ratio).
    std::vector<float> qt = p; uint32_t top = 0;
    for (uint32_t i = 1; i < n; i++) if (p[i] > p[top]) top = i;
    qt[top] -= 2.0f;
    const double hit_top = q27::forward_kl(p.data(), qt.data(), n);
    qt = p; qt[top == 0 ? 1 : 0] -= 2.0f;
    const double hit_tail = q27::forward_kl(p.data(), qt.data(), n);
    if (!(hit_top > hit_tail)) { fprintf(stderr, "kl: argmax shift not dominant\n"); return 1; }
    puts("forward KL: PASS");
    return 0;
}
