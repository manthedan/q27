// A/B/C MMA roofline — the one-day prefill-maturity decision
// (docs/plans/2026-07-16-mma-roofline.md, round-3 Q1).
//
// Closing the 48.79 -> 52.48 tok/s fork gap needs an 8.1% GEMM-kernel
// speedup with GEMM at 94.3% of prefill GPU time. This tool measures, at
// identical production geometry (dispatch grid, 32x16 tiles, 64-K walk,
// edge clamps, physical MMA count):
//   C  the full production T2 GEMM (q27_matmul_t2_mm_h via matmul_quantized)
//   B  the same kernel with half operands from device — no packed unpack,
//      no int8->half; B-C isolates unpack/conversion machinery
//   A  the same MMA sequence from tiles filled once — no staging, no
//      barriers, no flushes; A-B isolates staging/barrier/flush cadence
// Verdict comes from the LOWER 95% confidence bound of R = t_C / t_B
// (10 trials, t-distribution), never a best run. Memory-safe: synthetic
// buffers only (~250 MiB peak), no model mmap.

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

struct Stat { double mean, lo, hi; };

Stat measure(q27::MetalBackend& backend, uint32_t reps,
             const std::function<void()>& op) {
    // Warmup: clock ramp + first-touch, outside the timer.
    backend.begin_commands(); op(); backend.end_commands();
    backend.begin_commands(); op(); backend.end_commands();
    double trials[TRIALS];
    for (uint32_t t = 0; t < TRIALS; t++) {
        const auto start = std::chrono::steady_clock::now();
        backend.begin_commands();
        for (uint32_t r = 0; r < reps; r++) op();
        backend.end_commands();
        trials[t] = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count() / reps;
    }
    double mean = 0;
    for (double v : trials) mean += v;
    mean /= TRIALS;
    double var = 0;
    for (double v : trials) var += (v - mean) * (v - mean);
    const double ci = T9_975 * std::sqrt(var / (TRIALS - 1)) / std::sqrt((double)TRIALS);
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

bool output_sane(q27::MetalBackend& backend, q27::BackendBuffer& out, uint64_t floats,
                 const char* what) {
    std::vector<float> v(std::min<uint64_t>(floats, 4096));
    backend.read(out, 0, v.data(), v.size() * 4);
    bool nonzero = false;
    for (float f : v) {
        if (!std::isfinite(f)) { fprintf(stderr, "FAIL: %s output not finite\n", what); return false; }
        if (f != 0.0f) nonzero = true;
    }
    if (!nonzero) { fprintf(stderr, "FAIL: %s output all zero (vacuous arm)\n", what); return false; }
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
    printf("backend: %s — A/B/C MMA roofline, x_rows %u, %u reps x %u trials, "
           "lower-95%%-CB verdicts\n", backend.name().c_str(), X_ROWS, reps, TRIALS);
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

    printf("%-28s %9s %9s %9s | %-15s %-15s %8s\n",
           "shape", "C ms", "B ms", "A ms", "C/B [lo,hi]", "B/A [lo,hi]", "C TFLOPs");
    double agg_w = 0, agg_cb = 0, agg_cb_lo = 0;
    for (const Shape& s : SHAPES) {
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
        // B operands: half weights/activations/weight-scales, float x-scales.
        auto wh = upload_halves(backend, (uint64_t)s.rows * s.cols, true);
        auto wsh = upload_halves(backend, (uint64_t)s.rows * (s.cols / 128), false);
        auto xh = upload_halves(backend, (uint64_t)X_ROWS * s.cols, true);
        std::vector<float> xs1((uint64_t)X_ROWS * (s.cols / 32), 1.0f);
        auto xsb = backend.allocate(xs1.size() * 4);
        backend.write(*xsb, 0, xs1.data(), xs1.size() * 4);
        auto seed = upload_halves(backend, 32 * 64 + 64 * 16, true);
        auto y = backend.allocate((uint64_t)s.rows * X_ROWS * 4);

        // Anti-vacuity: run each arm once, outputs must be finite + nonzero.
        backend.begin_commands();
        backend.matmul_quantized(wt2, xq, X_ROWS, *y);
        backend.end_commands();
        if (!output_sane(backend, *y, (uint64_t)s.rows * X_ROWS, "arm C")) return 1;
        backend.begin_commands();
        backend.mma_roofline('b', s.rows, s.cols, X_ROWS, *wh, wsh.get(), xh.get(), xsb.get(), *y);
        backend.end_commands();
        if (!output_sane(backend, *y, (uint64_t)s.rows * X_ROWS, "arm B")) return 1;
        backend.begin_commands();
        backend.mma_roofline('a', s.rows, s.cols, X_ROWS, *seed, nullptr, nullptr, nullptr, *y);
        backend.end_commands();
        if (!output_sane(backend, *y, (uint64_t)s.rows * X_ROWS, "arm A")) return 1;

        const Stat c = measure(backend, reps, [&] { backend.matmul_quantized(wt2, xq, X_ROWS, *y); });
        const Stat b = measure(backend, reps, [&] {
            backend.mma_roofline('b', s.rows, s.cols, X_ROWS, *wh, wsh.get(), xh.get(), xsb.get(), *y);
        });
        const Stat a = measure(backend, reps, [&] {
            backend.mma_roofline('a', s.rows, s.cols, X_ROWS, *seed, nullptr, nullptr, nullptr, *y);
        });
        // Physical MMA count: tgs x cols simdgroup-MMAs, 1024 FLOPs each.
        const double tgs = (double)((s.rows + 31) / 32) * ((X_ROWS + 15) / 16);
        const double flops = tgs * s.cols * 1024.0;
        const double cb = c.mean / b.mean, cb_lo = c.lo / b.hi, cb_hi = c.hi / b.lo;
        const double ba = b.mean / a.mean, ba_lo = b.lo / a.hi, ba_hi = b.hi / a.lo;
        printf("%-28s %9.3f %9.3f %9.3f | %.3f [%.3f,%.3f] %.3f [%.3f,%.3f] %8.2f\n",
               s.name, c.mean * 1e3, b.mean * 1e3, a.mean * 1e3,
               cb, cb_lo, cb_hi, ba, ba_lo, ba_hi, flops / c.mean / 1e12);
        const double w = c.mean * s.per_chunk_count;   // C-time share weighting
        agg_w += w;
        agg_cb += w / cb;
        agg_cb_lo += w / cb_lo;
    }
    const double R = agg_w / agg_cb, R_lo = agg_w / agg_cb_lo;
    printf("aggregate B/C headroom R = %.3f (lower 95%% CB %.3f), C-time-weighted\n", R, R_lo);
    const char* verdict = R_lo <= 1.08 ? "<=1.08: no unpack/staging headroom — M4 prefill MMA is MATURE at this schedule"
                        : R_lo <= 1.15 ? "1.08-1.15: permit exactly ONE targeted kernel round"
                                       : ">1.15: real implementation headroom — pursue";
    printf("pre-registered verdict (on the lower CB): %s\n", verdict);
    return 0;
}
