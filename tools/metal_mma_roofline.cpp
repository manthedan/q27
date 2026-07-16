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
#include <vector>

using q27::DType;

namespace {

constexpr uint32_t X_ROWS = 96;   // PREFILL_CHUNK_MAX: 6 exact 16-token tiles
constexpr uint32_t TRIALS = 10;
constexpr double T9_975 = 2.262;  // t-distribution, 9 dof, two-sided 95%

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
enum Arm { C = 0, BEQ = 1, B = 2, A = 3, N_ARMS = 4 };
const char* ARM_NAMES[N_ARMS] = {"C", "Beq", "B", "A"};

struct Stat { double mean, lo, hi; };

Stat stat_of(const std::vector<double>& v) {
    double mean = 0;
    for (double x : v) mean += x;
    mean /= v.size();
    double var = 0;
    for (double x : v) var += (x - mean) * (x - mean);
    const double ci = T9_975 * std::sqrt(var / (v.size() - 1)) / std::sqrt((double)v.size());
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
    uint32_t reps = 8;
    if (argc > 1) reps = (uint32_t)atoi(argv[1]);
    if (!reps) reps = 8;
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

        const std::function<void()> ops[N_ARMS] = {
            [&] { backend.matmul_quantized(wt2, xq, X_ROWS, *y); },
            [&] { backend.mma_roofline('e', s.rows, s.cols, X_ROWS, *whe, wsh.get(), xhe.get(), xsb.get(), *y); },
            [&] { backend.mma_roofline('b', s.rows, s.cols, X_ROWS, *wh, wsh.get(), xh.get(), xsb.get(), *y); },
            [&] { backend.mma_roofline('a', s.rows, s.cols, X_ROWS, *seed, nullptr, nullptr, nullptr, *y); },
        };

        // Anti-vacuity gates (zero-poisoned per arm), then warmup.
        const uint64_t out_floats = (uint64_t)s.rows * X_ROWS;
        std::vector<float> zeros(std::min<uint64_t>(out_floats, 4096), 0.0f);
        for (int arm = 0; arm < N_ARMS; arm++)
            if (!arm_sane(backend, *y, out_floats, zeros, ARM_NAMES[arm], ops[arm])) return 1;
        for (int arm = 0; arm < N_ARMS; arm++) {
            backend.begin_commands(); ops[arm](); backend.end_commands();
        }

        for (uint32_t trial = 0; trial < TRIALS; trial++) {
            for (int arm = 0; arm < N_ARMS; arm++) {
                const auto start = std::chrono::steady_clock::now();
                backend.begin_commands();
                for (uint32_t r = 0; r < reps; r++) ops[arm]();
                backend.end_commands();
                t[si][arm].push_back(std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count() / reps);
            }
        }
    }

    printf("%-28s %9s %9s %9s %9s | %-18s %8s\n",
           "shape", "C ms", "Beq ms", "B ms", "A ms", "C/Beq [95% CI]", "C TFLOPs");
    for (size_t si = 0; si < N_SHAPES; si++) {
        const Shape& s = SHAPES[si];
        const Stat c = stat_of(t[si][C]), beq = stat_of(t[si][BEQ]);
        const Stat b = stat_of(t[si][B]), a = stat_of(t[si][A]);
        std::vector<double> ratio(TRIALS);
        for (uint32_t k = 0; k < TRIALS; k++) ratio[k] = t[si][C][k] / t[si][BEQ][k];
        const Stat r = stat_of(ratio);
        const double tgs = (double)((s.rows + 31) / 32) * ((X_ROWS + 15) / 16);
        printf("%-28s %9.3f %9.3f %9.3f %9.3f | %.3f [%.3f,%.3f] %8.2f\n",
               s.name, c.mean * 1e3, beq.mean * 1e3, b.mean * 1e3, a.mean * 1e3,
               r.mean, r.lo, r.hi, tgs * s.cols * 1024.0 / c.mean / 1e12);
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
    printf("aggregate C/Beq (decision arm)      : %.3f [%.3f, %.3f]\n", Req.mean, Req.lo, Req.hi);
    printf("aggregate C/B   (expert-literal arm): %.3f [%.3f, %.3f]  (traffic-confounded, "
           "conservative)\n", Rb.mean, Rb.lo, Rb.hi);
    printf("aggregate Beq/A (cadence residual)  : %.3f [%.3f, %.3f]\n", Rba.mean, Rba.lo, Rba.hi);
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
