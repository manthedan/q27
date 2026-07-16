// Synthetic decode-GEMV microbenchmark for the Metal backend.
//
// Times matvec_quantized / matvec_quantized_pair at the production decode
// shapes without touching the model artifact, and reports effective weight
// bandwidth. This is the attribution tool for critical-path item 3: decode
// streams every weight byte once per token, so GEMV GB/s bounds tok/s.
//
// Memory-safe: synthetic buffers only (~1.6 GiB peak), no 17 GiB mmap.

#include "metal_backend.h"

#include <chrono>
#include <cstdio>
#include <cstring>
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
    return dtype == DType::Q4_G64 ? 2 : dtype == DType::T2_G128 ? 4 : 1;
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

} // namespace

int main(int argc, char** argv) {
    int reps = 20;
    bool t2 = false, t3 = false, slot2 = false;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--dtype" && i + 1 < argc) {
            const std::string d = argv[++i];
            if (d == "t2") t2 = true;
            else if (d == "t3") t3 = true;
            else if (d != "q4q8") { fprintf(stderr, "invalid --dtype (q4q8|t2|t3)\n"); return 1; }
        } else if (arg == "--slot2") {
            slot2 = true;
        } else if (!arg.empty() && arg[0] != '-') {
            reps = atoi(arg.c_str());
        } else {
            fprintf(stderr, "usage: %s [reps] [--dtype q4q8|t2|t3] [--slot2]\n", argv[0]);
            return 1;
        }
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
