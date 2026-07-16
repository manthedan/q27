// A/B/C MMA roofline — the one-day prefill-maturity decision
// (docs/plans/2026-07-16-mma-roofline.md, round-3 Q1).
//
// Closing the 48.79 -> 52.48 tok/s fork gap needs an 8.1% GEMM-kernel
// speedup with GEMM at 94.3% of prefill GPU time. This tool measures, at
// identical production geometry (dispatch grid, 32x16 tiles, 64-K walk,
// edge clamps, physical MMA count):
//   C   the full production T2 GEMM (q27_matmul_t2_mm_h via matmul_quantized)
//   Beq the same kernel with the unpack/int8->half ALU deleted at EQUAL
//       device traffic (loads C's byte volume, replicates into the same
//       staging stores) — the decision arm: C/Beq isolates plumbing ALU
//   B   the expert-literal arm: half operands at production layout (8x the
//       weight bytes of C) — reported as the traffic-confounded companion;
//       on bandwidth-sensitive shapes it can only understate headroom
//   A   the same MMA sequence from tiles filled once — no staging, no
//       barriers, no flushes; A vs Beq isolates staging/barrier/flush cadence
// Statistics (codex round on the landing commit): per-trial chunk-aggregate
// ratios R_t = sum_i(count_i * tC_i,t) / sum_i(count_i * tBeq_i,t) over 10
// interleaved trials, 95% CI on the mean of R_t. Declaring NO headroom
// requires the UPPER bound <= 1.08; declaring headroom requires the LOWER
// bound above the line; anything straddling is reported inconclusive.
// Anti-vacuity: outputs are zero-poisoned before every arm and must come
// back finite and nonzero. Memory-safe: synthetic buffers only (~250 MiB
// peak), no model mmap.

#include "metal_backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using q27::DType;

namespace {

constexpr uint32_t TRIALS = 16;
constexpr double T_CRIT = 2.131;  // t-distribution, 15 dof, two-sided 95%

struct Shape {
    const char* name;
    uint32_t rows, cols;
    double per_chunk_count;   // dispatches per 64-layer pass at width 96
};

// The production per-chunk projection mix (metal_engine.cpp weight table);
// output head excluded (12-row slices on the serial path). gdn alpha/beta
// [48x5120] excluded: 48 rows round up to two row-groups of mostly-clamped
// lanes, ~0.15% of chunk FLOPs, stated rather than silently dropped.
const Shape SHAPES[] = {
    {"ffn gate/up  [17408x5120]", 17408, 5120, 128},
    {"ffn down     [5120x17408]", 5120, 17408, 64},
    {"gdn qkv      [10240x5120]", 10240, 5120, 48},
    {"gdn gate     [6144x5120]",  6144, 5120, 48},
    {"ssm/attn out [5120x6144]",  5120, 6144, 64},
    {"attn q       [12288x5120]", 12288, 5120, 16},
    {"attn k/v     [1024x5120]",  1024, 5120, 32},
};
constexpr size_t N_SHAPES = sizeof(SHAPES) / sizeof(Shape);
// Cx = the pre-converted-activation candidate (packed weights, LUT unpack,
// half activations from device): measures the ceiling of deleting the
// per-tile char->half converts before any production pre-pass is built.
// D = lever 1 direct-RHS probe (64x32 weight-staged tile, RHS read from
// device via simdgroup_load, int8->half K-major pre-pass charged to the
// arm; docs/plans/2026-07-16-lever1-direct-rhs.md).
enum Arm { C = 0, BEQ = 1, B = 2, A = 3, CX = 4, D = 5, D2 = 6, F = 7, N_ARMS = 8 };
const char* ARM_NAMES[N_ARMS] = {"C", "Beq", "B", "A", "Cx", "D", "D2", "F"};
static_assert(TRIALS % N_ARMS == 0,
              "trial count must be a multiple of the arm count so the rotated "
              "arm order is fully counterbalanced (codex P2)");

struct Stat { double mean, lo, hi; };

Stat stat_of(const std::vector<double>& v) {
    double mean = 0;
    for (double x : v) mean += x;
    mean /= v.size();
    double var = 0;
    for (double x : v) var += (x - mean) * (x - mean);
    const double ci = T_CRIT * std::sqrt(var / (v.size() - 1)) / std::sqrt((double)v.size());
    return {mean, mean - ci, mean + ci};
}

// Synthetic T2 weight for arm C (same generator as metal_gemv_bench).
q27::BackendTensor upload_t2(q27::MetalBackend& backend, const Shape& s,
                             std::vector<uint8_t>& data, std::vector<uint16_t>& scales) {
    data.resize((uint64_t)s.rows * s.cols / 4);
    for (size_t i = 0; i < data.size(); i++) {
        uint8_t v = (uint8_t)(i * 2654435761u >> 24);
        const uint8_t lo = v & 0x55;
        data[i] = (uint8_t)((v & 0xaa & (uint8_t)~(lo << 1)) | lo);  // no 0b11 codes
    }
    scales.assign((uint64_t)s.rows * (s.cols / 128), 0x3c00 /* f16 1.0 */);
    q27::Tensor tensor;
    tensor.name = s.name;
    tensor.dtype = DType::T2_G128;
    tensor.shape = {s.rows, s.cols};
    tensor.data = data.data();
    tensor.data_size = data.size();
    tensor.scales = reinterpret_cast<const uint8_t*>(scales.data());
    tensor.scales_size = scales.size() * sizeof(uint16_t);
    return backend.upload(tensor);
}

std::shared_ptr<q27::BackendBuffer> upload_halves(q27::MetalBackend& backend, uint64_t count,
                                                  bool varied) {
    std::vector<uint16_t> h(count);
    if (varied) {
        // Small exact-in-half integers in [-3, 4]: opaque to the compiler,
        // no overflow across a 17408-long K accumulation.
        static const uint16_t vals[8] = {0xc200, 0xc000, 0xbc00, 0x0000,
                                         0x3c00, 0x4000, 0x4200, 0x4400};
        for (uint64_t i = 0; i < count; i++) h[i] = vals[(i * 2654435761u >> 24) & 7];
    } else {
        for (uint64_t i = 0; i < count; i++) h[i] = 0x3c00;  // f16 1.0
    }
    auto buf = backend.allocate(count * 2);
    backend.write(*buf, 0, h.data(), count * 2);
    return buf;
}

// Zero-poison then run one dispatch: the arm must overwrite the poison with
// finite, nonzero values, so a silently-vacuous arm (bad binding, early
// return) cannot inherit a previous arm's output (codex P2).
bool arm_sane(q27::MetalBackend& backend, q27::BackendBuffer& out, uint64_t floats,
              const std::vector<float>& zeros, const char* what,
              const std::function<void()>& op) {
    backend.write(out, 0, zeros.data(),
                  std::min<uint64_t>(floats, zeros.size()) * 4);
    backend.begin_commands(); op(); backend.end_commands();
    std::vector<float> v(std::min<uint64_t>(floats, 4096));
    backend.read(out, 0, v.data(), v.size() * 4);
    bool nonzero = false;
    for (float f : v) {
        if (!std::isfinite(f)) { fprintf(stderr, "FAIL: %s output not finite\n", what); return false; }
        if (f != 0.0f) nonzero = true;
    }
    if (!nonzero) { fprintf(stderr, "FAIL: %s left the zero poison (vacuous arm)\n", what); return false; }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    uint32_t reps = 8, x_rows = 96;   // 96 = PREFILL_CHUNK_MAX (6 exact 16-token tiles)
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--xrows" && i + 1 < argc) x_rows = (uint32_t)atoi(argv[++i]);
        else if (!a.empty() && a[0] != '-') reps = (uint32_t)atoi(a.c_str());
        else { fprintf(stderr, "usage: %s [reps] [--xrows N]\n", argv[0]); return 1; }
    }
    if (!reps) reps = 8;
    if (!x_rows || x_rows > 96) { fprintf(stderr, "--xrows must be 1..96\n"); return 1; }
    const uint32_t X_ROWS = x_rows;
    // Pin the production GEMM route: an inherited Q27_METAL_GEMM_HALF=0
    // would silently swap arm C to the float-staging kernel.
    setenv("Q27_METAL_GEMM_HALF", "1", 1);
    q27::MetalBackend backend;
    printf("backend: %s — A/B/C MMA roofline, x_rows %u, %u reps x %u interleaved trials\n",
           backend.name().c_str(), X_ROWS, reps, TRIALS);
    printf("excluded: gdn alpha/beta [48x5120] x96 (~0.15%% of chunk FLOPs); output head "
           "(serial-path 12-row slices)\n");

    // Warm the GPU clocks once before any shape.
    {
        auto seed = upload_halves(backend, 32 * 64 + 64 * 16, true);
        auto y = backend.allocate((uint64_t)5120 * X_ROWS * 4);
        backend.begin_commands();
        for (int i = 0; i < 100; i++)
            backend.mma_roofline('a', 5120, 5120, X_ROWS, *seed, nullptr, nullptr, nullptr, *y);
        backend.end_commands();
    }

    // Per-shape, per-arm, per-trial times. Trials interleave arms (C, Beq,
    // B, A within each trial) so thermal/clock drift lands on all arms, and
    // the per-trial chunk aggregate is a paired ratio.
    std::vector<std::vector<std::vector<double>>> t(
        N_SHAPES, std::vector<std::vector<double>>(N_ARMS));

    for (size_t si = 0; si < N_SHAPES; si++) {
        const Shape& s = SHAPES[si];
        std::vector<uint8_t> t2data;
        std::vector<uint16_t> t2scales;
        q27::BackendTensor wt2 = upload_t2(backend, s, t2data, t2scales);
        // C activations: real quantized int8 + per-32 scales via the
        // production quantize kernel.
        std::vector<float> xf((uint64_t)X_ROWS * s.cols);
        for (size_t i = 0; i < xf.size(); i++) xf[i] = (float)((int)(i % 17) - 8) / 8.0f;
        auto xfb = backend.allocate(xf.size() * 4);
        backend.write(*xfb, 0, xf.data(), xf.size() * 4);
        q27::BackendQuantized xq = backend.allocate_quantized(X_ROWS * s.cols);
        backend.quantize(*xfb, xq);
        // B operands: half weights/activations at production layout.
        auto wh = upload_halves(backend, (uint64_t)s.rows * s.cols, true);
        auto xh = upload_halves(backend, (uint64_t)X_ROWS * s.cols, true);
        // Beq operands: C's device byte volume (cols/8 halves per weight
        // row, cols/2 per activation row).
        auto whe = upload_halves(backend, (uint64_t)s.rows * (s.cols / 8), true);
        auto xhe = upload_halves(backend, (uint64_t)X_ROWS * (s.cols / 2), true);
        auto wsh = upload_halves(backend, (uint64_t)s.rows * (s.cols / 128), false);
        std::vector<float> xs1((uint64_t)X_ROWS * (s.cols / 32), 1.0f);
        auto xsb = backend.allocate(xs1.size() * 4);
        backend.write(*xsb, 0, xs1.data(), xs1.size() * 4);
        auto seed = upload_halves(backend, 32 * 64 + 64 * 16, true);
        auto y = backend.allocate((uint64_t)s.rows * X_ROWS * 4);

        // Cx weights: the same packed T2 bytes as C, as a raw buffer.
        auto wt2raw = backend.allocate(t2data.size());
        backend.write(*wt2raw, 0, t2data.data(), t2data.size());

        const std::function<void()> ops[N_ARMS] = {
            [&] { backend.matmul_quantized(wt2, xq, X_ROWS, *y); },
            [&] { backend.mma_roofline('e', s.rows, s.cols, X_ROWS, *whe, wsh.get(), xhe.get(), xsb.get(), *y); },
            [&] { backend.mma_roofline('b', s.rows, s.cols, X_ROWS, *wh, wsh.get(), xh.get(), xsb.get(), *y); },
            [&] { backend.mma_roofline('a', s.rows, s.cols, X_ROWS, *seed, nullptr, nullptr, nullptr, *y); },
            [&] { backend.mma_roofline('x', s.rows, s.cols, X_ROWS, *wt2raw, wsh.get(), xh.get(), xsb.get(), *y); },
            [&] { backend.mma_roofline('d', s.rows, s.cols, X_ROWS, *wt2raw, wsh.get(),
                                       xq.values.get(), xq.scales.get(), *y); },
            [&] { backend.mma_roofline('2', s.rows, s.cols, X_ROWS, *wt2raw, wsh.get(),
                                       xq.values.get(), xq.scales.get(), *y); },
            [&] { backend.mma_roofline('f', s.rows, s.cols, X_ROWS, *wt2raw, wsh.get(),
                                       xq.values.get(), xq.scales.get(), *y); },
        };

        // Anti-vacuity gates (zero-poisoned per arm), then warmup.
        const uint64_t out_floats = (uint64_t)s.rows * X_ROWS;
        std::vector<float> zeros(std::min<uint64_t>(out_floats, 4096), 0.0f);
        for (int arm = 0; arm < N_ARMS; arm++)
            if (!arm_sane(backend, *y, out_floats, zeros, ARM_NAMES[arm], ops[arm])) return 1;
        // Lever-1 correctness: arm D computes the same math as C modulo
        // flush/fold order on exact integer partials — require max relative
        // difference <= 1e-3 on all live outputs before any timing.
        {
            std::vector<float> yc(out_floats), yd(out_floats);
            const std::vector<float> poison(out_floats, 0.0f);
            backend.begin_commands(); ops[C](); backend.end_commands();
            backend.read(*y, 0, yc.data(), out_floats * 4);
            for (int darm : {(int)D, (int)D2, (int)F}) {
                // Arm F is the f16-accumulate probe: slab-bounded half
                // rounding is expected, real error — gated at 5e-2 and
                // REPORTED (a probe read, docs/plans/2026-07-16-f16acc-
                // probe.md), where D/D2 remain near-exact at 1e-3.
                const double tol = darm == (int)F ? 5e-2 : 1e-3;
                // Full-buffer poison: an arm that skips writes must not
                // inherit C's reference values (codex P2); non-finite
                // candidate outputs are rejected explicitly — NaN would
                // otherwise vanish through std::max.
                backend.write(*y, 0, poison.data(), out_floats * 4);
                backend.begin_commands(); ops[darm](); backend.end_commands();
                backend.read(*y, 0, yd.data(), out_floats * 4);
                double worst = 0;
                for (uint64_t i = 0; i < out_floats; i++) {
                    if (!std::isfinite(yd[i])) {
                        fprintf(stderr, "FAIL: %s — arm %s produced a non-finite output\n",
                                s.name, ARM_NAMES[darm]);
                        return 1;
                    }
                    const double denom = std::max(1.0, (double)std::fabs(yc[i]));
                    worst = std::max(worst, (double)std::fabs(yc[i] - yd[i]) / denom);
                }
                if (worst > tol) {
                    fprintf(stderr, "FAIL: %s — arm %s diverges from C (max rel diff %.3e)\n",
                            s.name, ARM_NAMES[darm], worst);
                    return 1;
                }
                if (darm == (int)F)
                    printf("%-28s arm F max rel diff vs C: %.3e (gate 5e-2)\n", s.name, worst);
            }
        }
        for (int arm = 0; arm < N_ARMS; arm++) {
            backend.begin_commands(); ops[arm](); backend.end_commands();
        }

        for (uint32_t trial = 0; trial < TRIALS; trial++) {
            // Rotate the arm order per trial so systematic frequency/thermal
            // drift within a trial doesn't always land on the same side of
            // each paired ratio (codex P2).
            for (int k = 0; k < N_ARMS; k++) {
                const int arm = (k + (int)trial) % N_ARMS;
                const auto start = std::chrono::steady_clock::now();
                backend.begin_commands();
                for (uint32_t r = 0; r < reps; r++) ops[arm]();
                backend.end_commands();
                t[si][arm].push_back(std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count() / reps);
            }
        }
    }

    printf("%-28s %9s %9s %9s %9s %9s %9s | %-18s %8s\n",
           "shape", "C ms", "A ms", "Cx ms", "D ms", "D2 ms", "F ms", "C/F [95% CI]", "C TFLOPs");
    for (size_t si = 0; si < N_SHAPES; si++) {
        const Shape& s = SHAPES[si];
        const Stat c = stat_of(t[si][C]), beq = stat_of(t[si][BEQ]);
        const Stat b = stat_of(t[si][B]), a = stat_of(t[si][A]);
        const Stat cx = stat_of(t[si][CX]), d = stat_of(t[si][D]);
        const Stat d2 = stat_of(t[si][D2]);
        const Stat f = stat_of(t[si][F]);
        (void)beq; (void)b;
        std::vector<double> ratio(TRIALS);
        for (uint32_t k = 0; k < TRIALS; k++) ratio[k] = t[si][C][k] / t[si][F][k];
        const Stat r = stat_of(ratio);
        const double tgs = (double)((s.rows + 31) / 32) * ((X_ROWS + 15) / 16);
        printf("%-28s %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f | %.3f [%.3f,%.3f] %8.2f\n",
               s.name, c.mean * 1e3, a.mean * 1e3, cx.mean * 1e3,
               d.mean * 1e3, d2.mean * 1e3, f.mean * 1e3, r.mean, r.lo, r.hi,
               tgs * s.cols * 1024.0 / c.mean / 1e12);
    }

    // Chunk-aggregate paired ratios per trial (valid CI: one ratio
    // observation per trial, arms measured back-to-back inside the trial).
    auto agg_ratio = [&](Arm num, Arm den) {
        std::vector<double> r(TRIALS);
        for (uint32_t k = 0; k < TRIALS; k++) {
            double tn = 0, td = 0;
            for (size_t si = 0; si < N_SHAPES; si++) {
                tn += SHAPES[si].per_chunk_count * t[si][num][k];
                td += SHAPES[si].per_chunk_count * t[si][den][k];
            }
            r[k] = tn / td;
        }
        return stat_of(r);
    };
    const Stat Req = agg_ratio(C, BEQ), Rb = agg_ratio(C, B), Rba = agg_ratio(BEQ, A);
    const Stat Rcx = agg_ratio(C, CX);
    const Stat Rd = agg_ratio(C, D);
    const Stat Rd2 = agg_ratio(C, D2);
    const Stat Rf = agg_ratio(C, F);
    printf("aggregate C/Beq (decision arm)      : %.3f [%.3f, %.3f]\n", Req.mean, Req.lo, Req.hi);
    printf("aggregate C/B   (expert-literal arm): %.3f [%.3f, %.3f]  (traffic-confounded, "
           "conservative)\n", Rb.mean, Rb.lo, Rb.hi);
    printf("aggregate Beq/A (cadence residual)  : %.3f [%.3f, %.3f]\n", Rba.mean, Rba.lo, Rba.hi);
    printf("aggregate C/Cx  (half-x candidate)  : %.3f [%.3f, %.3f]  (bit-identical if built: "
           "raw int8 is exact in half)\n", Rcx.mean, Rcx.lo, Rcx.hi);
    printf("aggregate C/D   (lever 1 row-stripe)  : %.3f [%.3f, %.3f]  (incl. RHS pre-pass)\n",
           Rd.mean, Rd.lo, Rd.hi);
    printf("aggregate C/D2  (lever 1 quadrant map) : %.3f [%.3f, %.3f]  (incl. RHS pre-pass)\n",
           Rd2.mean, Rd2.lo, Rd2.hi);
    printf("aggregate C/F   (f16-accumulate probe) : %.3f [%.3f, %.3f]\n", Rf.mean, Rf.lo, Rf.hi);
    // f16-acc kill line (docs/plans/2026-07-16-f16acc-probe.md, BaseRT
    // survey import #1): graduate/park on the 1.10 line, CB-sided.
    const char* fv =
        Rf.lo >= 1.10 ? "lower bound >= 1.10: GRADUATE — build the production trial behind Q27_METAL_GEMM_F16ACC + envelope pair"
      : Rf.hi <= 1.10 ? "upper bound <= 1.10: PARK — accumulator width is not the M4 MMA issue limiter"
                      : "INCONCLUSIVE: CI straddles the 1.10 line — extend trials once before parking";
    printf("f16-acc probe verdict (C/F): %s\n", fv);
    if (X_ROWS <= 16) {
        // Lever 1 decision (docs/plans/2026-07-16-lever1-direct-rhs.md):
        // verify-shape verdict on the lower 95% CB, park line 1.3x.
        const Stat best = Rd2.mean > Rd.mean ? Rd2 : Rd;
        const char* lv = best.lo >= 2.0 ? ">=2.0x: target met — integrate behind the verify path"
                       : best.lo >= 1.3 ? "[1.3, 2.0)x: above the park line — integrate, re-check e2e vs the oracle re-sweep bar"
                       : best.hi <  1.3 ? "<1.3x (upper bound, best mapping): PARK lever 1 — the direct-RHS shape doesn't fit the M4"
                                        : "INCONCLUSIVE: CI straddles the 1.3x park line";
        printf("lever 1 verdict at x_rows %u (best-mapping lower CB %.3f): %s\n", X_ROWS, best.lo, lv);
    }
    // No-headroom needs the UPPER bound under the line; headroom needs the
    // LOWER bound above it; anything else is inconclusive (codex P1).
    const char* verdict =
        Req.hi <= 1.08 ? "upper bound <= 1.08: no unpack/staging headroom — M4 prefill MMA is MATURE at this schedule"
      : Req.lo >  1.15 ? "lower bound > 1.15: real implementation headroom — pursue"
      : Req.lo >  1.08 ? "lower bound in (1.08, 1.15]: permit exactly ONE targeted kernel round"
                       : "INCONCLUSIVE: CI straddles the 1.08 line — tighten (more trials) before deciding";
    printf("pre-registered verdict (C/Beq): %s\n", verdict);
    return 0;
}
