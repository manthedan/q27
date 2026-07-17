// Synthetic decode-GEMV microbenchmark for the Metal backend.
//
// Times matvec_quantized / matvec_quantized_pair at the production decode
// shapes without touching the model artifact, and reports effective weight
// bandwidth. This is the attribution tool for critical-path item 3: decode
// streams every weight byte once per token, so GEMV GB/s bounds tok/s.
//
// Memory-safe: synthetic buffers only (~1.6 GiB peak), no 17 GiB mmap.

#include "metal_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using q27::DType;

namespace {

struct Shape {
    const char* name;
    uint32_t rows, cols;
    DType dtype;
    bool pair; // benches two same-shape weights through matvec_quantized_pair
};

uint64_t data_divisor(DType dtype) {
    return dtype == DType::Q4_G64 ? 2 : dtype == DType::T2_G128 ? 4 :
           dtype == DType::B1_G128 ? 8 : 1;
}

uint64_t data_bytes_for(const Shape& s) {
    if (s.dtype == DType::T3_G128) return (uint64_t)s.rows * (s.cols / 128) * 26;
    return (uint64_t)s.rows * s.cols / data_divisor(s.dtype);
}

uint64_t weight_bytes(const Shape& s) {
    const uint64_t group = s.dtype == DType::Q4_G64 ? 64 : 128;
    const uint64_t data = data_bytes_for(s);
    const uint64_t scales = (uint64_t)s.rows * (s.cols / group) * 2;
    return (data + scales) * (s.pair ? 2 : 1);
}

q27::BackendTensor upload_synthetic(q27::MetalBackend& backend, const Shape& s,
                                    std::vector<uint8_t>& data,
                                    std::vector<uint16_t>& scales) {
    const uint64_t group = s.dtype == DType::Q4_G64 ? 64 : 128;
    data.resize(data_bytes_for(s));
    for (size_t i = 0; i < data.size(); i++) data[i] = (uint8_t)(i * 2654435761u >> 24);
    if (s.dtype == DType::T2_G128)   // keep every 2-bit slot a valid ternary code (no 0b11)
        for (size_t i = 0; i < data.size(); i++) {
            const uint8_t lo = data[i] & 0x55;
            data[i] = (uint8_t)((data[i] & 0xaa & (uint8_t)~(lo << 1)) | lo);
        }
    if (s.dtype == DType::T3_G128)   // keep every byte a valid base-3 code pack
        for (size_t i = 0; i < data.size(); i++) data[i] = (uint8_t)(data[i] % 243);
    scales.assign((uint64_t)s.rows * (s.cols / group), 0x3c00 /* f16 1.0 */);
    q27::Tensor tensor;
    tensor.name = s.name;
    tensor.dtype = s.dtype;
    tensor.shape = {s.rows, s.cols};
    tensor.data = data.data();
    tensor.data_size = data.size();
    tensor.scales = reinterpret_cast<const uint8_t*>(scales.data());
    tensor.scales_size = scales.size() * sizeof(uint16_t);
    return backend.upload(tensor);
}

// Multislot Phase 2, Phase-0 probe (docs/plans/2026-07-16-multislot-phase2-
// probe.md): per decode shape, arm A = 2 sequential single-row quantized
// GEMVs (today's two-slot serial cost), arm B = the 16-token MM tile at
// x_rows=2 (control), arm C = the x2 GEMV candidate. Arm C's two output
// rows must be byte-identical to arm A's before any timing is reported.
// The aggregate s_k weights each shape by its per-token production byte
// share (ffn x3 x64 layers, gdn qkv x48, ssm_out x48, head x1).
int run_slot2_probe(q27::MetalBackend& backend, int reps) {
    // The exact per-token production projection mix (metal_engine.cpp weight
    // shapes; counts: 48 GDN layers, 16 attention layers, 64 FFN, 1 head).
    // Orientation matters — [5120,17408] and [17408,5120] have different
    // row-group occupancy and inner-loop length (codex P2, 2026-07-16).
    struct ProbeShape { Shape shape; double per_token_count; };
    const ProbeShape probes[] = {
        {{"ffn gate/up   [17408x5120]", 17408, 5120, DType::T2_G128, false}, 128},
        {{"ffn down      [5120x17408]", 5120, 17408, DType::T2_G128, false}, 64},
        {{"gdn qkv       [10240x5120]", 10240, 5120, DType::T2_G128, false}, 48},
        {{"gdn gate      [6144x5120]",  6144, 5120, DType::T2_G128, false}, 48},
        {{"ssm/attn out  [5120x6144]",  5120, 6144, DType::T2_G128, false}, 64},
        {{"attn q        [12288x5120]", 12288, 5120, DType::T2_G128, false}, 16},
        {{"attn k/v      [1024x5120]",  1024, 5120, DType::T2_G128, false}, 32},
        {{"output head   [248320x5120]", 248320, 5120, DType::T2_G128, false}, 1},
    };
    printf("excluded from the mix: gdn alpha/beta [48x5120] x96 (~0.15%% of per-token "
           "weight bytes, dispatch-overhead-dominated)\n");
    printf("q* = quantized packed-dot family (chunked-path kernel; NOT serial decode)\n");
    printf("f* = float select-form family (q27_matvec_t2_g128 = production serial decode)\n");
    printf("%-30s %8s %8s %8s %6s | %8s %8s %6s\n",
           "shape", "qA 2x1", "qB mm2", "qC x2", "s_kq", "fA 2x1", "fC x2", "s_kf");
    double agg_num = 0.0, agg_den = 0.0, aggf_num = 0.0, aggf_den = 0.0;
    for (const ProbeShape& p : probes) {
        const Shape& s = p.shape;
        std::vector<uint8_t> data;
        std::vector<uint16_t> scales;
        q27::BackendTensor weight = upload_synthetic(backend, s, data, scales);

        // Two distinct activation rows, contiguous [2, cols]; quantize the
        // pair buffer once (32-blocks never straddle rows, so per-row
        // quantization of the same floats yields identical bytes).
        std::vector<float> x2(2 * (size_t)s.cols);
        for (size_t i = 0; i < x2.size(); i++) x2[i] = (float)((int)(i % 23) - 11) / 11.0f;
        auto xb2 = backend.allocate(x2.size() * sizeof(float));
        backend.write(*xb2, 0, x2.data(), x2.size() * sizeof(float));
        q27::BackendQuantized xq2 = backend.allocate_quantized(2 * s.cols);
        backend.quantize(*xb2, xq2);
        auto xb0 = backend.allocate((uint64_t)s.cols * sizeof(float));
        auto xb1 = backend.allocate((uint64_t)s.cols * sizeof(float));
        backend.write(*xb0, 0, x2.data(), s.cols * sizeof(float));
        backend.write(*xb1, 0, x2.data() + s.cols, s.cols * sizeof(float));
        q27::BackendQuantized xq0 = backend.allocate_quantized(s.cols);
        q27::BackendQuantized xq1 = backend.allocate_quantized(s.cols);
        backend.quantize(*xb0, xq0);
        backend.quantize(*xb1, xq1);

        auto y0 = backend.allocate((uint64_t)s.rows * sizeof(float));
        auto y1 = backend.allocate((uint64_t)s.rows * sizeof(float));
        auto yB = backend.allocate((uint64_t)s.rows * 2 * sizeof(float));
        auto yC = backend.allocate((uint64_t)s.rows * 2 * sizeof(float));
        auto yf0 = backend.allocate((uint64_t)s.rows * sizeof(float));
        auto yf1 = backend.allocate((uint64_t)s.rows * sizeof(float));
        auto yfa = backend.allocate((uint64_t)s.rows * sizeof(float));
        auto yfb = backend.allocate((uint64_t)s.rows * sizeof(float));

        // Correctness gates: each x2 kernel's rows byte-identical to the
        // corresponding single-row kernel outputs.
        backend.begin_commands();
        backend.matvec_quantized(weight, xq0, *y0);
        backend.matvec_quantized(weight, xq1, *y1);
        backend.matvec_quantized_x2(weight, xq2, *yC);
        backend.matvec(weight, *xb0, *yf0);
        backend.matvec(weight, *xb1, *yf1);
        backend.matvec_x2(weight, *xb0, *xb1, *yfa, *yfb);
        backend.end_commands();
        std::vector<float> a0(s.rows), a1(s.rows), c2(2 * (size_t)s.rows);
        std::vector<float> f0(s.rows), f1(s.rows), fa(s.rows), fb(s.rows);
        backend.read(*y0, 0, a0.data(), s.rows * sizeof(float));
        backend.read(*y1, 0, a1.data(), s.rows * sizeof(float));
        backend.read(*yC, 0, c2.data(), 2 * (uint64_t)s.rows * sizeof(float));
        backend.read(*yf0, 0, f0.data(), s.rows * sizeof(float));
        backend.read(*yf1, 0, f1.data(), s.rows * sizeof(float));
        backend.read(*yfa, 0, fa.data(), s.rows * sizeof(float));
        backend.read(*yfb, 0, fb.data(), s.rows * sizeof(float));
        if (memcmp(a0.data(), c2.data(), s.rows * sizeof(float)) != 0 ||
            memcmp(a1.data(), c2.data() + s.rows, s.rows * sizeof(float)) != 0) {
            fprintf(stderr, "FAIL: %s — quantized x2 rows are not byte-identical "
                    "to the single-row kernel\n", s.name);
            return 1;
        }
        if (memcmp(f0.data(), fa.data(), s.rows * sizeof(float)) != 0 ||
            memcmp(f1.data(), fb.data(), s.rows * sizeof(float)) != 0) {
            fprintf(stderr, "FAIL: %s — select-form x2 rows are not byte-identical "
                    "to the single-row kernel\n", s.name);
            return 1;
        }

        auto time_arm = [&](int arm) {
            auto body = [&](int count) {
                backend.begin_commands();
                for (int i = 0; i < count; i++) {
                    if (arm == 0) {
                        backend.matvec_quantized(weight, xq0, *y0);
                        backend.matvec_quantized(weight, xq1, *y1);
                    } else if (arm == 1) {
                        backend.matmul_quantized(weight, xq2, 2, *yB);
                    } else if (arm == 2) {
                        backend.matvec_quantized_x2(weight, xq2, *yC);
                    } else if (arm == 3) {
                        backend.matvec(weight, *xb0, *yf0);
                        backend.matvec(weight, *xb1, *yf1);
                    } else {
                        backend.matvec_x2(weight, *xb0, *xb1, *yfa, *yfb);
                    }
                }
                backend.end_commands();
            };
            body(2); // warmup / first touch
            auto start = std::chrono::steady_clock::now();
            body(reps);
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count() / reps;
        };
        const double a_ms = time_arm(0) * 1e3;
        const double b_ms = time_arm(1) * 1e3;
        const double c_ms = time_arm(2) * 1e3;
        const double fa_ms = time_arm(3) * 1e3;
        const double fc_ms = time_arm(4) * 1e3;
        const double s_kq = a_ms / c_ms;
        const double s_kf = fa_ms / fc_ms;
        printf("%-30s %8.3f %8.3f %8.3f %6.3f | %8.3f %8.3f %6.3f\n",
               s.name, a_ms, b_ms, c_ms, s_kq, fa_ms, fc_ms, s_kf);
        const double w = (double)weight_bytes(s) * p.per_token_count;
        agg_num += w;
        agg_den += w / s_kq;
        aggf_num += w;
        aggf_den += w / s_kf;
    }
    printf("aggregate s_k byte-weighted: quantized %.3f, select-form (production) %.3f\n",
           agg_num / agg_den, aggf_num / aggf_den);
    printf("identity: both x2 kernels byte-identical to their single-row kernels on all shapes\n");
    return 0;
}

// B1 Phase 0B — synthetic kernel economics (docs/plans/2026-07-15-binary-
// tier.md). Three B1 dot structures against the production T2 select-form
// float matvec on the per-token projection mix; the decision metric is the
// production-mix WALL-TIME ratio T_B1/T_T2 (never effective GB/s alone),
// candidate 3's activation preprocessing on the clock by construction
// (matvec_b1_probe encodes it before the dot). Pre-registered bands:
// <=0.60 strong GO, 0.60-0.72 conditional GO, >0.72 kill.
int run_b1_probe(q27::MetalBackend& backend, int reps) {
    struct ProbeShape { Shape shape; double per_token_count; };
    const ProbeShape probes[] = {
        {{"ffn gate/up   [17408x5120]", 17408, 5120, DType::T2_G128, false}, 128},
        {{"ffn down      [5120x17408]", 5120, 17408, DType::T2_G128, false}, 64},
        {{"gdn qkv       [10240x5120]", 10240, 5120, DType::T2_G128, false}, 48},
        {{"gdn gate      [6144x5120]",  6144, 5120, DType::T2_G128, false}, 48},
        {{"ssm/attn out  [5120x6144]",  5120, 6144, DType::T2_G128, false}, 64},
        {{"attn q        [12288x5120]", 12288, 5120, DType::T2_G128, false}, 16},
        {{"attn k/v      [1024x5120]",  1024, 5120, DType::T2_G128, false}, 32},
        {{"output head   [248320x5120]", 248320, 5120, DType::T2_G128, false}, 1},
    };
    printf("B1 probe: reference arm = production T2 select-form float matvec; "
           "c1 = select, c2 = sign-XOR, c3 = int8 bitplane+popcount (preprocess on the clock)\n");
    printf("%-30s %8s %8s %8s %8s | %6s %6s %6s\n",
           "shape", "t2 ms", "c1 ms", "c2 ms", "c3 ms", "r1", "r2", "r3");
    double t2_wall = 0.0, b1_wall[3] = {0.0, 0.0, 0.0};
    double c3_float_err_max = 0.0;
    for (const ProbeShape& p : probes) {
        const Shape& s = p.shape;
        const uint32_t nb = s.cols / 128;
        // Reference arm: synthetic T2 tensor through the production kernel.
        std::vector<uint8_t> t2_data;
        std::vector<uint16_t> t2_scales;
        q27::BackendTensor t2w = upload_synthetic(backend, s, t2_data, t2_scales);

        // B1 arm: raw bits + NONUNIFORM per-(row,group) fp16 scales, all
        // exactly representable (0.5..2.0 step 0.25) so the CPU double refs
        // stay exact — uniform 1.0 scales would leave a wrong/ignored scale
        // index invisible to the gate (codex P2).
        static const uint16_t kScaleBits[7] = {0x3800, 0x3a00, 0x3c00, 0x3d00,
                                               0x3e00, 0x3f00, 0x4000};
        static const float kScaleVal[7] = {0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f};
        auto scale_idx = [](uint32_t r, uint32_t g) { return (r * 31u + g) % 7u; };
        std::vector<uint8_t> bits((uint64_t)s.rows * s.cols / 8);
        for (size_t i = 0; i < bits.size(); i++) bits[i] = (uint8_t)(i * 2654435761u >> 24);
        std::vector<uint16_t> bscales((uint64_t)s.rows * nb);
        for (uint32_t r = 0; r < s.rows; r++)
            for (uint32_t g = 0; g < nb; g++)
                bscales[(uint64_t)r * nb + g] = kScaleBits[scale_idx(r, g)];
        auto bitsb = backend.allocate(bits.size());
        backend.write(*bitsb, 0, bits.data(), bits.size());
        auto scalesb = backend.allocate(bscales.size() * sizeof(uint16_t));
        backend.write(*scalesb, 0, bscales.data(), bscales.size() * sizeof(uint16_t));
        std::vector<float> x(s.cols);
        for (uint32_t i = 0; i < s.cols; i++) x[i] = (float)((int)(i % 23) - 11) / 11.0f;
        auto xb = backend.allocate(x.size() * sizeof(float));
        backend.write(*xb, 0, x.data(), x.size() * sizeof(float));
        auto scratch = backend.allocate((uint64_t)nb * 136);
        auto yt2 = backend.allocate((uint64_t)s.rows * sizeof(float));
        std::shared_ptr<q27::BackendBuffer> yc[3];
        for (int c = 0; c < 3; c++) yc[c] = backend.allocate((uint64_t)s.rows * sizeof(float));

        // CPU references. Float ref (double accumulate): bit ? +x : -x.
        // Int8 model mirrors q27_b1_x_prep / _popcount: float s = amax/127,
        // u = clamp(round(x/s)+128, 0..255), contributions accumulated per
        // group with the same correction terms (integer-exact popcount math).
        std::vector<float> gs(nb), gsumu(nb);
        std::vector<int> u8((uint64_t)s.cols);
        for (uint32_t g = 0; g < nb; g++) {
            float amax = 0.0f;
            for (uint32_t i = 0; i < 128; i++) amax = std::max(amax, std::fabs(x[g * 128 + i]));
            const float sc = amax / 127.0f;
            float sumu = 0.0f;
            for (uint32_t i = 0; i < 128; i++) {
                const float xv = x[g * 128 + i];
                const int u = sc > 0.0f
                    ? (int)std::fmin(std::fmax(std::round(xv / sc) + 128.0f, 0.0f), 255.0f)
                    : 128;
                u8[g * 128 + i] = u;
                sumu += (float)u;
            }
            gs[g] = sc;
            gsumu[g] = sumu;
        }
        std::vector<double> ref(s.rows), ref8(s.rows);
        for (uint32_t r = 0; r < s.rows; r++) {
            double acc = 0.0, acc8 = 0.0;
            const uint64_t rb = (uint64_t)r * s.cols / 8;
            for (uint32_t g = 0; g < nb; g++) {
                double gpos = 0.0, gsum = 0.0;
                long dotu = 0, wpop = 0;
                for (uint32_t i = 0; i < 128; i++) {
                    const uint32_t col = g * 128 + i;
                    const bool bit = bits[rb + col / 8] >> (col % 8) & 1;
                    gsum += x[col];
                    if (bit) { gpos += x[col]; dotu += u8[col]; wpop++; }
                }
                const double d = kScaleVal[scale_idx(r, g)];
                acc += d * (2.0 * gpos - gsum);
                acc8 += d * (double)gs[g] *
                        (2.0 * (double)(dotu - 128 * wpop) - ((double)gsumu[g] - 16384.0));
            }
            ref[r] = acc;
            ref8[r] = acc8;
        }

        // Correctness gates before any timing.
        for (int c = 1; c <= 3; c++) {
            backend.begin_commands();
            backend.matvec_b1_probe(c, s.rows, s.cols, *bitsb, *scalesb, *xb,
                                    scratch.get(), *yc[c - 1]);
            backend.end_commands();
        }
        std::vector<float> y1(s.rows), y2(s.rows), y3(s.rows);
        backend.read(*yc[0], 0, y1.data(), s.rows * sizeof(float));
        backend.read(*yc[1], 0, y2.data(), s.rows * sizeof(float));
        backend.read(*yc[2], 0, y3.data(), s.rows * sizeof(float));
        // A ±1-weight row dot is a random walk, so |ref| can land near zero
        // on some row; a pure relative gate would false-fail on float noise
        // there. Floor the denominator at 1e-3 of the row's input magnitude
        // (sum|x|, row-independent here): float accumulation noise stays
        // orders below it, a single flipped bit changes the dot by O(|x|)
        // and still trips the gate.
        double sum_abs_x = 0.0;
        for (uint32_t i = 0; i < s.cols; i++) sum_abs_x += std::fabs(x[i]);
        const double denom_floor = 1e-3 * sum_abs_x;
        double e1 = 0.0, e12 = 0.0, e3 = 0.0, e3f = 0.0;
        bool nonzero = false;
        for (uint32_t r = 0; r < s.rows; r++) {
            const double d1 = std::fabs(y1[r] - ref[r]) / std::fmax(std::fabs(ref[r]), denom_floor);
            const double d12 =
                std::fabs((double)y1[r] - y2[r]) / std::fmax(std::fabs(ref[r]), denom_floor);
            const double d3 =
                std::fabs(y3[r] - ref8[r]) / std::fmax(std::fabs(ref8[r]), denom_floor);
            const double d3f =
                std::fabs(y3[r] - ref[r]) / std::fmax(std::fabs(ref[r]), denom_floor);
            e1 = std::fmax(e1, d1);
            e12 = std::fmax(e12, d12);
            e3 = std::fmax(e3, d3);
            e3f = std::fmax(e3f, d3f);
            if (y1[r] != 0.0f) nonzero = true;
        }
        c3_float_err_max = std::fmax(c3_float_err_max, e3f);
        if (e1 > 1e-3 || e12 > 1e-3 || e3 > 1e-3) {
            fprintf(stderr, "FAIL: %s — b1 correctness gate (c1 vs ref %.2e, c2 vs c1 %.2e, "
                    "c3 vs int8 model %.2e; bound 1e-3)\n", s.name, e1, e12, e3);
            return 1;
        }
        if (!nonzero || (s.rows > 1 && y1[0] == y1[1])) {
            fprintf(stderr, "FAIL: %s — b1 anti-vacuity (zero or row-identical output)\n", s.name);
            return 1;
        }

        auto time_arm = [&](int arm) {
            auto body = [&](int count) {
                backend.begin_commands();
                for (int i = 0; i < count; i++) {
                    if (arm == 0) backend.matvec(t2w, *xb, *yt2);
                    else backend.matvec_b1_probe(arm, s.rows, s.cols, *bitsb, *scalesb, *xb,
                                                 scratch.get(), *yc[arm - 1]);
                }
                backend.end_commands();
            };
            body(2); // warmup / first touch
            auto start = std::chrono::steady_clock::now();
            body(reps);
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count() / reps;
        };
        const double t2_ms = time_arm(0) * 1e3;
        const double c_ms[3] = {time_arm(1) * 1e3, time_arm(2) * 1e3, time_arm(3) * 1e3};
        printf("%-30s %8.3f %8.3f %8.3f %8.3f | %6.3f %6.3f %6.3f\n", s.name, t2_ms,
               c_ms[0], c_ms[1], c_ms[2], c_ms[0] / t2_ms, c_ms[1] / t2_ms, c_ms[2] / t2_ms);
        t2_wall += p.per_token_count * t2_ms;
        for (int c = 0; c < 3; c++) b1_wall[c] += p.per_token_count * c_ms[c];
    }
    printf("identity: c1/c2 match the float reference and each other, c3 matches its int8 "
           "model, all <= 1e-3 max rel; c3 vs float reference max rel %.3e (quantization "
           "cost, reported not gated)\n", c3_float_err_max);
    printf("production-mix wall ratio T_B1/T_T2 (the pre-registered metric):\n");
    const char* cname[3] = {"c1 select", "c2 sign-XOR", "c3 popcount"};
    for (int c = 0; c < 3; c++) {
        const double r = b1_wall[c] / t2_wall;
        printf("  %-12s %.3f  -> %s\n", cname[c], r,
               r <= 0.60 ? "STRONG GO (<=0.60)"
               : r <= 0.72 ? "conditional GO (0.60-0.72; needs Phase 0A quality margin)"
                           : "KILL (>0.72)");
    }
    printf("per-token T2 wall over the mix: %.1f ms\n", t2_wall);
    return 0;
}

// E6 — Q4/Q8 GEMV efficiency leg, official tier (docs/plans/2026-07-17-e6-
// q4q8-gemv.md). Times the official-tier per-token production projection mix
// on the current quantized kernels exactly as serial decode dispatches them
// (matvec_quantized singles; matvec_quantized_pair on the ffn gate/up
// sibling pair, two distinct synthetic tensors — the cache-honesty rule)
// against the same-run production T2 select-form float matvec on the T2
// projection mix (the machine's demonstrated stream ceiling, never the
// spec-sheet peak). Decision metric R = byte-weighted per-token aggregate
// effective GB/s over the mix / same-run T2 reference aggregate GB/s.
// Pre-registered bands: R >= 0.90 KILL the rewrite; 0.80 <= R < 0.90
// conditional (targeted fix only for a mix-dominant shape below 0.75 of the
// reference); R < 0.80 fund exactly one rewrite round.

struct QSynth {
    std::vector<uint8_t> data;
    std::vector<uint16_t> scales;
    q27::BackendTensor weight;
    std::vector<double> ref_f; // CPU double dequantize-dot against float x
    std::vector<double> ref_i; // CPU double model of the int8-activation kernel
};

// Synthetic Q4_G64/Q8_G128 tensor with NONUNIFORM exactly-representable fp16
// scales (0.5..2.0 step 0.25, the (r*31+g)%7 pattern from the b1 leg —
// uniform 1.0 scales would leave a wrong/ignored scale index invisible to
// the gate). Packing mirrors tools/repack.py quant_q4/quant_q8 exactly: Q4
// stores q+8 nibbles with even columns in low nibbles (group 64), Q8 stores
// int8 clipped to [-127,127] (group 128; repack never emits -128). The CPU
// references decode the packed bytes back, so the gate covers the pack
// layout itself; the int8 model mirrors q27_quantize_x plus the kernels'
// per-32-column exact integer dot in double math (the c3-popcount pattern).
void build_q_synth(q27::MetalBackend& backend, const Shape& s, uint32_t seed,
                   const std::vector<float>& x, const std::vector<int8_t>& xq8,
                   const std::vector<float>& xs, QSynth& t) {
    static const uint16_t kScaleBits[7] = {0x3800, 0x3a00, 0x3c00, 0x3d00,
                                           0x3e00, 0x3f00, 0x4000};
    static const double kScaleVal[7] = {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0};
    const uint32_t group = s.dtype == DType::Q4_G64 ? 64 : 128;
    const uint32_t ng = s.cols / group;
    t.data.resize(data_bytes_for(s));
    for (size_t i = 0; i < t.data.size(); i++) {
        uint8_t b = (uint8_t)((i + (uint64_t)seed * 0x9e3779b9u) * 2654435761u >> 24);
        if (s.dtype == DType::Q8_G128 && b == 0x80) b = 0x7f; // repack clips to [-127,127]
        t.data[i] = b;
    }
    t.scales.resize((uint64_t)s.rows * ng);
    for (uint32_t r = 0; r < s.rows; r++)
        for (uint32_t g = 0; g < ng; g++)
            t.scales[(uint64_t)r * ng + g] = kScaleBits[(r * 31u + g) % 7u];
    q27::Tensor tensor;
    tensor.name = s.name;
    tensor.dtype = s.dtype;
    tensor.shape = {s.rows, s.cols};
    tensor.data = t.data.data();
    tensor.data_size = t.data.size();
    tensor.scales = reinterpret_cast<const uint8_t*>(t.scales.data());
    tensor.scales_size = t.scales.size() * sizeof(uint16_t);
    t.weight = backend.upload(tensor);

    t.ref_f.resize(s.rows);
    t.ref_i.resize(s.rows);
    const uint64_t row_stride = (uint64_t)s.cols / (s.dtype == DType::Q4_G64 ? 2 : 1);
    for (uint32_t r = 0; r < s.rows; r++) {
        const uint64_t rb = (uint64_t)r * row_stride;
        double accf = 0.0, acci = 0.0;
        for (uint32_t b32 = 0; b32 < s.cols / 32; b32++) {
            // A 32-column block never straddles a weight-scale group (64/128)
            // or an activation-scale block — same alignment the kernels use.
            const double ws = kScaleVal[(r * 31u + b32 * 32 / group) % 7u];
            long idot = 0;
            for (uint32_t i = 0; i < 32; i++) {
                const uint32_t c = b32 * 32 + i;
                const int w = s.dtype == DType::Q4_G64
                    ? (int)(t.data[rb + c / 2] >> ((c & 1) * 4) & 15) - 8
                    : (int)(int8_t)t.data[rb + c];
                accf += ws * (double)w * (double)x[c];
                idot += (long)w * xq8[c];
            }
            acci += ws * (double)idot * (double)xs[b32];
        }
        t.ref_f[r] = accf;
        t.ref_i[r] = acci;
    }
}

// With q4_candidate set (--q4-candidate N, the Q4 rewrite-round arms,
// docs/plans/2026-07-17-q4-rewrite-round.md), the Q4_G64 shapes (candidates
// 2-3), the Q8 shapes (candidate 4), or all shapes (candidate 1 = production
// through the probe path, A/B parity) run matvec_q4_probe instead of the
// production dispatch; the ffn sibling pair runs as two probe singles (the
// round has no pair variant). The candidate arm must pass the same CPU
// int8-activation model gate (1e-3, all shapes) plus byte-identity against
// the production single kernel (the round's kernel contract) BEFORE timing;
// the per-shape GB/s + ratio table stays directly comparable to E6 RESULTS.
int run_official_probe(q27::MetalBackend& backend, int reps, int q4_candidate) {
    // The official-tier per-token production projection mix (metal_engine.cpp
    // weight shapes; counts: 48 GDN layers, 16 attention layers, 64 FFN, 1
    // head) under the real v1.3 dtype policy (tools/repack.py policy()):
    // every matrix weight Q4_G64 except attn_k/v (Q8, KV-persistence
    // promotion) and the output head (Q8; official vocab 151936). The T2
    // reference mix differs only at the head (T2 vocab 248320).
    struct ProbeShape { Shape shape; uint32_t t2_rows; double per_token_count; bool pair_arm; };
    const ProbeShape probes[] = {
        {{"ffn gate/up   [17408x5120]", 17408, 5120, DType::Q4_G64, false}, 17408, 128, true},
        {{"ffn down      [5120x17408]", 5120, 17408, DType::Q4_G64, false}, 5120, 64, false},
        {{"gdn qkv       [10240x5120]", 10240, 5120, DType::Q4_G64, false}, 10240, 48, false},
        {{"gdn gate      [6144x5120]",  6144, 5120, DType::Q4_G64, false}, 6144, 48, false},
        {{"ssm/attn out  [5120x6144]",  5120, 6144, DType::Q4_G64, false}, 5120, 64, false},
        {{"attn q        [12288x5120]", 12288, 5120, DType::Q4_G64, false}, 12288, 16, false},
        {{"attn k/v      [1024x5120]",  1024, 5120, DType::Q8_G128, false}, 1024, 32, false},
        {{"output head   [151936x5120]", 151936, 5120, DType::Q8_G128, false}, 248320, 1, false},
    };
    const size_t n_probes = sizeof(probes) / sizeof(probes[0]);
    printf("E6 official-tier Q4/Q8 GEMV leg: production quantized kernels on the official "
           "per-token projection mix;\nreference arm = same-run production T2 select-form "
           "float matvec (backend.matvec) on the T2 mix\n");
    printf("excluded from the mix: MTP draft layer (blk.64.*, Q8) and the output_q4.weight "
           "draft-head copy — both stream\nonly on draft rounds, this leg measures the serial "
           "production token; gdn alpha/beta [48x5120] x96 (~0.15%% of\nper-token weight "
           "bytes, dispatch-overhead-dominated)\n");
    if (q4_candidate)
        printf("q4-round candidate %d arm (%s): affected shapes run matvec_q4_probe; "
               "the ffn sibling pair runs as two probe singles\n", q4_candidate,
               q4_candidate == 1 ? "production kernels through the probe path" :
               q4_candidate == 2 ? "q4 2 rows/simdgroup (retained comparison arm)" :
               q4_candidate == 3 ? "q4 4 rows/simdgroup (promoted production, alias)" :
                                   "q8 r4 twin (KILLED 2026-07-17 — backend throws)");
    double total_q_bytes = 0.0;
    for (const ProbeShape& p : probes)
        total_q_bytes += (double)weight_bytes(p.shape) * p.per_token_count;
    printf("%-30s %-3s %8s %8s %8s | %8s %8s | %6s %6s\n",
           "shape", "dt", "q1 ms", "pair ms", "q GB/s", "t2 ms", "t2 GB/s", "r", "byte%");
    double q_wall_ms = 0.0, t2_wall_ms = 0.0, total_t2_bytes = 0.0;
    double e_float_max = 0.0;
    const char* names[n_probes];
    double shares[n_probes], ratios[n_probes];
    size_t n = 0;
    for (const ProbeShape& p : probes) {
        const Shape& s = p.shape;
        // Candidate routing: 1 = every shape through the probe path,
        // 2-3 = the Q4_G64 shapes, 4 = the Q8_G128 shapes; the rest keep
        // the production dispatch (a promoted candidate replaces only its
        // own kernel, so the mix R matches the promotion re-run semantics).
        const bool use_probe = q4_candidate == 1 ||
            (q4_candidate && (q4_candidate == 4) == (s.dtype == DType::Q8_G128));
        // Activations + the exact CPU int8 model of q27_quantize_x: per
        // 32-block amax/127 float scale, reciprocal-multiply + rint
        // (nearest-even, matching Metal's CUDA-parity form — k3 audit D1),
        // clamp to [-127,127].
        std::vector<float> x(s.cols);
        for (uint32_t i = 0; i < s.cols; i++) x[i] = (float)((int)(i % 23) - 11) / 11.0f;
        std::vector<int8_t> xq8(s.cols);
        std::vector<float> xs(s.cols / 32);
        for (uint32_t b = 0; b < s.cols / 32; b++) {
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) amax = std::fmax(amax, std::fabs(x[b * 32 + i]));
            const float sc = amax / 127.0f;
            xs[b] = sc;
            const float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
            for (uint32_t i = 0; i < 32; i++) {
                const int q = (int)std::rint(x[b * 32 + i] * inv);
                xq8[b * 32 + i] = (int8_t)std::min(127, std::max(-127, q));
            }
        }
        // Two distinct synthetic tensors for the sibling pair (cache honesty:
        // reusing one would let the second dispatch hit warm cache lines).
        QSynth wa, wb;
        build_q_synth(backend, s, 1, x, xq8, xs, wa);
        if (p.pair_arm) build_q_synth(backend, s, 2, x, xq8, xs, wb);
        // Reference arm: synthetic T2 tensor through the production kernel.
        Shape t2s = s;
        t2s.rows = p.t2_rows;
        t2s.dtype = DType::T2_G128;
        std::vector<uint8_t> t2_data;
        std::vector<uint16_t> t2_scales;
        q27::BackendTensor t2w = upload_synthetic(backend, t2s, t2_data, t2_scales);

        auto xb = backend.allocate(x.size() * sizeof(float));
        backend.write(*xb, 0, x.data(), x.size() * sizeof(float));
        q27::BackendQuantized xq = backend.allocate_quantized(s.cols);
        backend.quantize(*xb, xq);
        auto y = backend.allocate((uint64_t)s.rows * sizeof(float));
        auto ypa = p.pair_arm ? backend.allocate((uint64_t)s.rows * sizeof(float)) : nullptr;
        auto ypb = p.pair_arm ? backend.allocate((uint64_t)s.rows * sizeof(float)) : nullptr;
        auto yt2 = backend.allocate((uint64_t)t2s.rows * sizeof(float));

        // Correctness gates before any timing: each arm against the CPU
        // double model of its own tensor, bound 1e-3 max rel with the sum|x|
        // denominator floor (same rationale as the b1 leg: a near-zero row
        // dot must not false-fail on float noise, a real bug still moves the
        // dot by orders more than the floor).
        backend.begin_commands();
        backend.matvec_quantized(wa.weight, xq, *y);
        if (p.pair_arm) backend.matvec_quantized_pair(wa.weight, *ypa, wb.weight, *ypb, xq);
        backend.end_commands();
        std::vector<float> ya(s.rows), yb_a(p.pair_arm ? s.rows : 0), yb_b(p.pair_arm ? s.rows : 0);
        backend.read(*y, 0, ya.data(), s.rows * sizeof(float));
        if (p.pair_arm) {
            backend.read(*ypa, 0, yb_a.data(), s.rows * sizeof(float));
            backend.read(*ypb, 0, yb_b.data(), s.rows * sizeof(float));
        }
        double sum_abs_x = 0.0;
        for (uint32_t i = 0; i < s.cols; i++) sum_abs_x += std::fabs(x[i]);
        const double denom_floor = 1e-3 * sum_abs_x;
        struct GateCheck { const std::vector<float>* y; const QSynth* t; const char* arm; };
        const GateCheck checks[3] = {
            {&ya, &wa, "single"}, {&yb_a, &wa, "pair a"}, {&yb_b, &wb, "pair b"}};
        const size_t n_checks = p.pair_arm ? 3 : 1;
        for (size_t k = 0; k < n_checks; k++) {
            double ei = 0.0, ef = 0.0;
            for (uint32_t r = 0; r < s.rows; r++) {
                const double yr = (*checks[k].y)[r];
                ei = std::fmax(ei, std::fabs(yr - checks[k].t->ref_i[r]) /
                                       std::fmax(std::fabs(checks[k].t->ref_i[r]), denom_floor));
                ef = std::fmax(ef, std::fabs(yr - checks[k].t->ref_f[r]) /
                                       std::fmax(std::fabs(checks[k].t->ref_f[r]), denom_floor));
            }
            if (ei > 1e-3) {
                fprintf(stderr, "FAIL: %s — %s arm vs CPU int8-activation model %.2e "
                        "(bound 1e-3)\n", s.name, checks[k].arm, ei);
                return 1;
            }
            e_float_max = std::fmax(e_float_max, ef);
        }
        bool nonzero = false;
        for (uint32_t r = 0; r < s.rows; r++)
            if (ya[r] != 0.0f) { nonzero = true; break; }
        if (!nonzero || (s.rows > 1 && ya[0] == ya[1])) {
            fprintf(stderr, "FAIL: %s — anti-vacuity (zero or row-identical output)\n", s.name);
            return 1;
        }
        // Candidate gates before any timing: the probe arm against the same
        // CPU int8-activation model, plus byte-identity against the
        // production single kernel (the round's kernel contract: same dot,
        // same scale multiply order). A failing candidate never times.
        if (use_probe) {
            auto ycand = backend.allocate((uint64_t)s.rows * sizeof(float));
            std::vector<float> yc(s.rows), yprod(s.rows);
            const QSynth* tensors[2] = {&wa, &wb};
            const float* prod[2] = {ya.data(), yprod.data()};
            const size_t n_arms = p.pair_arm ? 2 : 1;
            for (size_t k = 0; k < n_arms; k++) {
                backend.begin_commands();
                backend.matvec_q4_probe(q4_candidate, tensors[k]->weight, xq, *ycand);
                // The pair shape's byte gate needs a production SINGLE on the
                // sibling tensor (the pair kernel is not the identity target).
                if (k == 1) backend.matvec_quantized(wb.weight, xq, *y);
                backend.end_commands();
                backend.read(*ycand, 0, yc.data(), s.rows * sizeof(float));
                if (k == 1) backend.read(*y, 0, yprod.data(), s.rows * sizeof(float));
                double ec = 0.0;
                for (uint32_t r = 0; r < s.rows; r++)
                    ec = std::fmax(ec, std::fabs(yc[r] - tensors[k]->ref_i[r]) /
                                           std::fmax(std::fabs(tensors[k]->ref_i[r]), denom_floor));
                if (ec > 1e-3) {
                    fprintf(stderr, "FAIL: %s — candidate %d vs CPU int8-activation model "
                            "%.2e (bound 1e-3)\n", s.name, q4_candidate, ec);
                    return 1;
                }
                if (memcmp(yc.data(), prod[k], s.rows * sizeof(float)) != 0) {
                    fprintf(stderr, "FAIL: %s — candidate %d output is not byte-identical "
                            "to the production kernel\n", s.name, q4_candidate);
                    return 1;
                }
            }
        }

        auto time_arm = [&](int arm) {
            auto body = [&](int count) {
                backend.begin_commands();
                for (int i = 0; i < count; i++) {
                    if (arm == 0) {
                        if (use_probe) backend.matvec_q4_probe(q4_candidate, wa.weight, xq, *y);
                        else backend.matvec_quantized(wa.weight, xq, *y);
                    } else if (arm == 1) {
                        // Candidate pair arm = two probe singles; the 2x
                        // weight-bytes GB/s accounting below still holds.
                        if (use_probe) {
                            backend.matvec_q4_probe(q4_candidate, wa.weight, xq, *ypa);
                            backend.matvec_q4_probe(q4_candidate, wb.weight, xq, *ypb);
                        } else {
                            backend.matvec_quantized_pair(wa.weight, *ypa, wb.weight, *ypb, xq);
                        }
                    } else {
                        backend.matvec(t2w, *xb, *yt2);
                    }
                }
                backend.end_commands();
            };
            body(2); // warmup / first touch
            auto start = std::chrono::steady_clock::now();
            body(reps);
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count() / reps;
        };
        const double q1_ms = time_arm(0) * 1e3;
        const double qp_ms = p.pair_arm ? time_arm(1) * 1e3 : 0.0;
        const double t2_ms = time_arm(2) * 1e3;
        // Production dispatch for the mix: the pair kernel covers two of the
        // per-token count per dispatch on the sibling shape, singles elsewhere.
        const double wb_bytes = (double)weight_bytes(s);
        const double prod_ms = p.pair_arm ? p.per_token_count / 2.0 * qp_ms
                                          : p.per_token_count * q1_ms;
        const double q_gbs = p.pair_arm ? 2.0 * wb_bytes / (qp_ms * 1e-3) / 1e9
                                        : wb_bytes / (q1_ms * 1e-3) / 1e9;
        const double t2_bytes = (double)weight_bytes(t2s);
        const double t2_gbs = t2_bytes / (t2_ms * 1e-3) / 1e9;
        char pair_col[16];
        if (p.pair_arm) snprintf(pair_col, sizeof(pair_col), "%8.3f", qp_ms);
        else snprintf(pair_col, sizeof(pair_col), "%8s", "-");
        const double share = wb_bytes * p.per_token_count / total_q_bytes;
        printf("%-30s %-3s %8.3f %s %8.2f | %8.3f %8.2f | %6.3f %5.1f%%\n",
               s.name, s.dtype == DType::Q4_G64 ? "q4" : "q8", q1_ms, pair_col, q_gbs,
               t2_ms, t2_gbs, q_gbs / t2_gbs, share * 100.0);
        q_wall_ms += prod_ms;
        t2_wall_ms += p.per_token_count * t2_ms;
        total_t2_bytes += t2_bytes * p.per_token_count;
        names[n] = s.name;
        shares[n] = share;
        ratios[n] = q_gbs / t2_gbs;
        n++;
    }
    printf("identity: single and pair kernels match their CPU int8-activation models on all "
           "shapes, <= 1e-3 max rel;\ndistance to the float dequantize-dot reference max rel "
           "%.3e (activation-quantization cost, reported not gated)\n", e_float_max);
    const double q_gbs = total_q_bytes / (q_wall_ms * 1e-3) / 1e9;
    const double ref_gbs = total_t2_bytes / (t2_wall_ms * 1e-3) / 1e9;
    const double ratio = q_gbs / ref_gbs;
    const double parity_ms = total_q_bytes / (ref_gbs * 1e9) * 1e3;
    printf("per-token GEMV wall over the mix: %.2f ms (%.1f tok/s GEMV-bound); "
           "at T2-reference parity: %.2f ms (%.1f tok/s)\n",
           q_wall_ms, 1e3 / q_wall_ms, parity_ms, 1e3 / parity_ms);
    printf("R = %.3f (official mix %.2f GB/s byte-weighted / same-run T2 reference "
           "%.2f GB/s)\n", ratio, q_gbs, ref_gbs);
    if (q4_candidate) {
        // The E6 funding verdict below pre-registered a different decision;
        // the round's own lines are R >= 0.90 on the promotion re-run and
        // >= 10% over production on both worst shapes (attn q, ssm/attn out).
        printf("candidate %d mix: R = %.3f, %.2f GB/s byte-weighted (E6 baseline R = 0.716, "
               "52.44 GB/s; round ship line R >= 0.90)\n", q4_candidate, ratio, q_gbs);
        return 0;
    }
    if (ratio >= 0.90) {
        printf("verdict: R >= 0.90 — no rewrite round funded (the 2026-07-17 q4 round "
               "shipped the r4 restructure at this line; remaining headroom is below the "
               "cost of a round plus regate)\n");
    } else if (ratio >= 0.80) {
        printf("verdict: CONDITIONAL (0.80 <= R < 0.90) — no full round; fund a targeted "
               "fix only if a single mix-dominant shape\n(>= 20%% of per-token bytes) sits "
               "below 0.75 of the reference — name it in RESULTS:\n");
        int hits = 0;
        for (size_t i = 0; i < n; i++)
            if (shares[i] >= 0.20 && ratios[i] < 0.75) {
                printf("  mix-dominant shape below 0.75: %s (byte share %.1f%%, "
                       "ratio %.3f)\n", names[i], shares[i] * 100.0, ratios[i]);
                hits++;
            }
        if (!hits) printf("  no mix-dominant shape below 0.75 — no targeted fix funded\n");
    } else {
        printf("verdict: FUND ONE ROUND (R < 0.80) — exactly one rewrite round, regated by "
               "re-running this leg; the round's own ship line is R >= 0.90 after rewrite\n");
    }
    return 0;
}

// B1 select round-2 candidate leg (--b1-candidate N, docs/plans/2026-07-17-
// b1-select-round2.md): the bonsai-tier per-token projection mix through
// the production int8-activation B1 select GEMV (the same-run A/B control)
// against one candidate arm (1 = production through the probe path, parity
// leg; 2 = 4 rows/simdgroup; 3 = 8 rows/simdgroup). Candidate arms must
// pass the exact CPU int8-activation model (1e-3, denominator floor) AND
// byte-identity vs the production kernel BEFORE any timing — the round's
// kernel contract is same dot, same scale-multiply order. Sub-line (round
// doc): a candidate that fails to beat the production mix wall by >= 10%
// is not promoted regardless of artifact hopes.
int run_b1_round2(q27::MetalBackend& backend, int reps, int b1_candidate) {
    struct ProbeShape { Shape shape; double per_token_count; };
    const ProbeShape probes[] = {
        {{"ffn gate/up   [17408x5120]", 17408, 5120, DType::B1_G128, false}, 128},
        {{"ffn down      [5120x17408]", 5120, 17408, DType::B1_G128, false}, 64},
        {{"gdn qkv       [10240x5120]", 10240, 5120, DType::B1_G128, false}, 48},
        {{"gdn gate      [6144x5120]",  6144, 5120, DType::B1_G128, false}, 48},
        {{"ssm/attn out  [5120x6144]",  5120, 6144, DType::B1_G128, false}, 64},
        {{"attn q        [12288x5120]", 12288, 5120, DType::B1_G128, false}, 16},
        {{"attn k/v      [1024x5120]",  1024, 5120, DType::B1_G128, false}, 32},
        {{"output head   [248320x5120]", 248320, 5120, DType::B1_G128, false}, 1},
    };
    printf("B1 round-2 leg: candidate %d (%s) vs same-run production "
           "q27_matvec_b1_quantized on the bonsai per-token mix\n", b1_candidate,
           b1_candidate == 1 ? "production through the probe path, A/B parity" :
           b1_candidate == 2 ? "4 rows/simdgroup, lane-held x-slice" :
                               "8 rows/simdgroup, issue-depth probe");
    printf("%-30s %8s %8s %8s %8s | %6s\n",
           "shape", "prod ms", "cand ms", "p GB/s", "c GB/s", "r");
    double prod_wall = 0.0, cand_wall = 0.0, total_bytes = 0.0;
    for (const ProbeShape& p : probes) {
        const Shape& s = p.shape;
        const uint32_t nb = s.cols / 128;
        // Synthetic B1 bits + NONUNIFORM exactly-representable fp16 scales
        // (the b1-leg pattern: uniform 1.0 scales would leave a wrong scale
        // index invisible to the gate).
        static const uint16_t kScaleBits[7] = {0x3800, 0x3a00, 0x3c00, 0x3d00,
                                               0x3e00, 0x3f00, 0x4000};
        static const double kScaleVal[7] = {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0};
        auto scale_idx = [](uint32_t r, uint32_t g) { return (r * 31u + g) % 7u; };
        std::vector<uint8_t> bits((uint64_t)s.rows * s.cols / 8);
        for (size_t i = 0; i < bits.size(); i++) bits[i] = (uint8_t)(i * 2654435761u >> 24);
        std::vector<uint16_t> bscales((uint64_t)s.rows * nb);
        for (uint32_t r = 0; r < s.rows; r++)
            for (uint32_t g = 0; g < nb; g++)
                bscales[(uint64_t)r * nb + g] = kScaleBits[scale_idx(r, g)];
        q27::Tensor tensor;
        tensor.name = s.name;
        tensor.dtype = DType::B1_G128;
        tensor.shape = {s.rows, s.cols};
        tensor.data = bits.data();
        tensor.data_size = bits.size();
        tensor.scales = reinterpret_cast<const uint8_t*>(bscales.data());
        tensor.scales_size = bscales.size() * sizeof(uint16_t);
        q27::BackendTensor weight = backend.upload(tensor);

        // Activations + the exact CPU int8 model of q27_quantize_x (per
        // 32-block amax/127, reciprocal-multiply + rint, clamp ±127 — the
        // official-leg pattern verbatim).
        std::vector<float> x(s.cols);
        for (uint32_t i = 0; i < s.cols; i++) x[i] = (float)((int)(i % 23) - 11) / 11.0f;
        std::vector<int8_t> xq8(s.cols);
        std::vector<float> xsc(s.cols / 32);
        for (uint32_t b = 0; b < s.cols / 32; b++) {
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) amax = std::fmax(amax, std::fabs(x[b * 32 + i]));
            const float sc = amax / 127.0f;
            xsc[b] = sc;
            const float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
            for (uint32_t i = 0; i < 32; i++) {
                const int q = (int)std::rint(x[b * 32 + i] * inv);
                xq8[b * 32 + i] = (int8_t)std::min(127, std::max(-127, q));
            }
        }
        // CPU double model of the kernels' per-32-column exact integer dot:
        // idot = 2*sum_{bit=1} xq8 - sum(xq8), scaled by the (row, c/128)
        // weight scale and the block's activation scale.
        std::vector<double> ref_i(s.rows);
        for (uint32_t r = 0; r < s.rows; r++) {
            const uint64_t rb = (uint64_t)r * s.cols / 8;
            double acc = 0.0;
            for (uint32_t b32 = 0; b32 < s.cols / 32; b32++) {
                long pos = 0, tot = 0;
                for (uint32_t i = 0; i < 32; i++) {
                    const uint32_t c = b32 * 32 + i;
                    const int xv = xq8[c];
                    tot += xv;
                    if (bits[rb + c / 8] >> (c % 8) & 1) pos += xv;
                }
                acc += kScaleVal[scale_idx(r, b32 * 32 / 128)] *
                       (double)(2 * pos - tot) * (double)xsc[b32];
            }
            ref_i[r] = acc;
        }

        auto xb = backend.allocate(x.size() * sizeof(float));
        backend.write(*xb, 0, x.data(), x.size() * sizeof(float));
        q27::BackendQuantized xq = backend.allocate_quantized(s.cols);
        backend.quantize(*xb, xq);
        auto yprod = backend.allocate((uint64_t)s.rows * sizeof(float));
        auto ycand = backend.allocate((uint64_t)s.rows * sizeof(float));

        // Correctness gates before any timing.
        backend.begin_commands();
        backend.matvec_quantized(weight, xq, *yprod);
        backend.matvec_b1r2_probe(b1_candidate, weight, xq, *ycand);
        backend.end_commands();
        std::vector<float> yp(s.rows), yc(s.rows);
        backend.read(*yprod, 0, yp.data(), s.rows * sizeof(float));
        backend.read(*ycand, 0, yc.data(), s.rows * sizeof(float));
        double sum_abs_x = 0.0;
        for (uint32_t i = 0; i < s.cols; i++) sum_abs_x += std::fabs(x[i]);
        const double denom_floor = 1e-3 * sum_abs_x;
        double ep = 0.0, ec = 0.0;
        for (uint32_t r = 0; r < s.rows; r++) {
            ep = std::fmax(ep, std::fabs(yp[r] - ref_i[r]) /
                                   std::fmax(std::fabs(ref_i[r]), denom_floor));
            ec = std::fmax(ec, std::fabs(yc[r] - ref_i[r]) /
                                   std::fmax(std::fabs(ref_i[r]), denom_floor));
        }
        if (ep > 1e-3 || ec > 1e-3) {
            fprintf(stderr, "FAIL: %s — vs CPU int8-activation model (prod %.2e, "
                    "cand %.2e, bound 1e-3)\n", s.name, ep, ec);
            return 1;
        }
        bool nonzero = false;
        for (uint32_t r = 0; r < s.rows; r++)
            if (yp[r] != 0.0f) { nonzero = true; break; }
        if (!nonzero || (s.rows > 1 && yp[0] == yp[1])) {
            fprintf(stderr, "FAIL: %s — anti-vacuity (zero or row-identical output)\n",
                    s.name);
            return 1;
        }
        if (memcmp(yc.data(), yp.data(), s.rows * sizeof(float)) != 0) {
            fprintf(stderr, "FAIL: %s — candidate %d output is not byte-identical to "
                    "the production kernel\n", s.name, b1_candidate);
            return 1;
        }

        auto time_arm = [&](bool cand) {
            auto body = [&](int count) {
                backend.begin_commands();
                for (int i = 0; i < count; i++) {
                    if (cand) backend.matvec_b1r2_probe(b1_candidate, weight, xq, *ycand);
                    else backend.matvec_quantized(weight, xq, *yprod);
                }
                backend.end_commands();
            };
            body(2); // warmup / first touch
            auto start = std::chrono::steady_clock::now();
            body(reps);
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count() / reps;
        };
        const double p_ms = time_arm(false) * 1e3;
        const double c_ms = time_arm(true) * 1e3;
        const double wbytes = (double)weight_bytes(s);
        const double p_gbs = wbytes / (p_ms * 1e-3) / 1e9;
        const double c_gbs = wbytes / (c_ms * 1e-3) / 1e9;
        printf("%-30s %8.3f %8.3f %8.2f %8.2f | %6.3f\n",
               s.name, p_ms, c_ms, p_gbs, c_gbs, p_ms / c_ms);
        prod_wall += p.per_token_count * p_ms;
        cand_wall += p.per_token_count * c_ms;
        total_bytes += wbytes * p.per_token_count;
    }
    printf("identity: candidate byte-identical to production on all shapes; both match "
           "the CPU int8-activation model <= 1e-3 max rel\n");
    const double speedup = prod_wall / cand_wall;
    printf("per-token B1 GEMV wall over the mix: production %.2f ms (%.2f GB/s "
           "byte-weighted), candidate %.2f ms (%.2f GB/s)\n",
           prod_wall, total_bytes / (prod_wall * 1e-3) / 1e9,
           cand_wall, total_bytes / (cand_wall * 1e-3) / 1e9);
    printf("candidate %d mix speedup = %.3f (round sub-line >= 1.10 to promote; "
           "ship line is the quiet artifact decode >= 18 tok/s)\n",
           b1_candidate, speedup);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int reps = 20;
    int q4_candidate = 0, b1_candidate = 0;
    bool t2 = false, t3 = false, slot2 = false, b1 = false, official = false;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--dtype" && i + 1 < argc) {
            const std::string d = argv[++i];
            if (d == "t2") t2 = true;
            else if (d == "t3") t3 = true;
            else if (d != "q4q8") { fprintf(stderr, "invalid --dtype (q4q8|t2|t3)\n"); return 1; }
        } else if (arg == "--slot2") {
            slot2 = true;
        } else if (arg == "--b1") {
            b1 = true;
        } else if (arg == "--official") {
            official = true;
        } else if (arg == "--q4-candidate" && i + 1 < argc) {
            q4_candidate = atoi(argv[++i]);
            if (q4_candidate < 1 || q4_candidate > 4) {
                fprintf(stderr, "invalid --q4-candidate (1=production 2=r2 3=production-alias; 4 was killed 2026-07-17)\n");
                return 1;
            }
        } else if (arg == "--b1-candidate" && i + 1 < argc) {
            b1_candidate = atoi(argv[++i]);
            if (b1_candidate < 1 || b1_candidate > 3) {
                fprintf(stderr, "invalid --b1-candidate (1=production-parity 2=4-row 3=8-row)\n");
                return 1;
            }
        } else if (!arg.empty() && arg[0] != '-') {
            reps = atoi(arg.c_str());
        } else {
            fprintf(stderr, "usage: %s [reps] [--dtype q4q8|t2|t3] [--slot2] [--b1] "
                    "[--official] [--q4-candidate N] [--b1-candidate N]\n", argv[0]);
            return 1;
        }
    }
    if (q4_candidate && !official) {
        fprintf(stderr, "--q4-candidate requires --official\n");
        return 1;
    }
    if (reps < 1) reps = 1;
    q27::MetalBackend backend;
    printf("backend: %s, %d reps per shape%s\n", backend.name().c_str(), reps,
           t3 ? ", base-3 ternary weights" : t2 ? ", ternary weights" : "");

    // The t2 table runs the same production shapes with every matrix weight
    // ternary, as the Bonsai artifact packs them (embedding/head included).
    const Shape shapes_q[] = {
        {"ffn_gate single   [17408x5120]", 17408, 5120, DType::Q4_G64, false},
        {"ffn_gate+up pair  [17408x5120]", 17408, 5120, DType::Q4_G64, true},
        {"ffn_gate+up pair  [17408x5120]", 17408, 5120, DType::Q8_G128, true},
        {"ffn_down          [5120x17408]", 5120, 17408, DType::Q4_G64, false},
        {"ffn_down          [5120x17408]", 5120, 17408, DType::Q8_G128, false},
        {"gdn qkv           [10240x5120]", 10240, 5120, DType::Q4_G64, false},
        {"ssm_out           [5120x6144]",  5120, 6144, DType::Q4_G64, false},
        {"output head       [151936x5120]", 151936, 5120, DType::Q8_G128, false},
    };
    const Shape shapes_t2[] = {
        {"ffn_gate single   [17408x5120]", 17408, 5120, DType::T2_G128, false},
        {"ffn_gate+up pair  [17408x5120]", 17408, 5120, DType::T2_G128, true},
        {"ffn_down          [5120x17408]", 5120, 17408, DType::T2_G128, false},
        {"gdn qkv           [10240x5120]", 10240, 5120, DType::T2_G128, false},
        {"ssm_out           [5120x6144]",  5120, 6144, DType::T2_G128, false},
        {"output head       [248320x5120]", 248320, 5120, DType::T2_G128, false},
    };
    const Shape shapes_t3[] = {
        {"ffn_gate single   [17408x5120]", 17408, 5120, DType::T3_G128, false},
        {"ffn_gate+up pair  [17408x5120]", 17408, 5120, DType::T3_G128, true},
        {"ffn_down          [5120x17408]", 5120, 17408, DType::T3_G128, false},
        {"gdn qkv           [10240x5120]", 10240, 5120, DType::T3_G128, false},
        {"ssm_out           [5120x6144]",  5120, 6144, DType::T3_G128, false},
        {"output head       [248320x5120]", 248320, 5120, DType::T3_G128, false},
    };
    const Shape* shapes = t3 ? shapes_t3 : t2 ? shapes_t2 : shapes_q;
    const size_t n_shapes = t3 ? sizeof(shapes_t3) / sizeof(Shape)
                          : t2 ? sizeof(shapes_t2) / sizeof(Shape)
                               : sizeof(shapes_q) / sizeof(Shape);

    // Ramp GPU/memory clocks before timing anything: the first ~second of
    // work otherwise runs at a low power state and understates the first shape.
    {
        const Shape warm{"warmup", 5120, 5120, DType::Q8_G128, false};
        std::vector<uint8_t> data;
        std::vector<uint16_t> scales;
        q27::BackendTensor weight = upload_synthetic(backend, warm, data, scales);
        q27::BackendQuantized xq = backend.allocate_quantized(warm.cols);
        auto y = backend.allocate((uint64_t)warm.rows * sizeof(float));
        backend.begin_commands();
        for (int i = 0; i < 200; i++) backend.matvec_quantized(weight, xq, *y);
        backend.end_commands();
    }

    if (slot2) return run_slot2_probe(backend, reps);
    if (b1) return run_b1_probe(backend, reps);
    if (official) return run_official_probe(backend, reps, q4_candidate);
    if (b1_candidate) return run_b1_round2(backend, reps, b1_candidate);

    double total_seconds = 0.0, total_bytes = 0.0;
    for (size_t si = 0; si < n_shapes; si++) {
        const Shape& shape = shapes[si];
        std::vector<uint8_t> data, data_b;
        std::vector<uint16_t> scales, scales_b;
        q27::BackendTensor weight = upload_synthetic(backend, shape, data, scales);
        // Pair shapes stream two distinct weight tensors, matching production
        // sibling projections; reusing one tensor would let the second
        // dispatch hit cache lines the first already pulled.
        q27::BackendTensor weight_b;
        if (shape.pair) weight_b = upload_synthetic(backend, shape, data_b, scales_b);

        std::vector<float> x(shape.cols);
        for (uint32_t i = 0; i < shape.cols; i++) x[i] = (float)((int)(i % 19) - 9) / 9.0f;
        auto xb = backend.allocate(x.size() * sizeof(float));
        backend.write(*xb, 0, x.data(), x.size() * sizeof(float));
        q27::BackendQuantized xq = backend.allocate_quantized(shape.cols);
        backend.quantize(*xb, xq);
        auto y = backend.allocate((uint64_t)shape.rows * sizeof(float));
        auto y2 = shape.pair ? backend.allocate((uint64_t)shape.rows * sizeof(float)) : nullptr;

        // T2 shapes run the float-activation kernel: exact ternary math needs
        // no activation quantization, so that is the production decode path.
        auto run = [&](int count) {
            backend.begin_commands();
            for (int i = 0; i < count; i++) {
                if (shape.dtype == DType::T2_G128 || shape.dtype == DType::T3_G128) {
                    backend.matvec(weight, *xb, *y);
                    if (shape.pair) backend.matvec(weight_b, *xb, *y2);
                } else if (shape.pair) {
                    backend.matvec_quantized_pair(weight, *y, weight_b, *y2, xq);
                } else {
                    backend.matvec_quantized(weight, xq, *y);
                }
            }
            backend.end_commands();
        };
        run(2); // warmup: first touch of the synthetic weight pages
        auto start = std::chrono::steady_clock::now();
        run(reps);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const double bytes = (double)weight_bytes(shape) * reps;
        printf("%-34s %-3s %8.3f ms/op %8.2f GB/s\n", shape.name,
               shape.dtype == DType::Q4_G64 ? "q4" :
               shape.dtype == DType::T2_G128 ? "t2" :
               shape.dtype == DType::T3_G128 ? "t3" : "q8",
               seconds / reps * 1e3, bytes / seconds / 1e9);
        total_seconds += seconds;
        total_bytes += bytes;
    }
    printf("%-34s %-3s %8s    %11.2f GB/s\n", "aggregate", "", "", total_bytes / total_seconds / 1e9);
    return 0;
}
