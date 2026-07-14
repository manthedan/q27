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

uint64_t weight_bytes(const Shape& s) {
    const uint64_t group = s.dtype == DType::Q8_G128 ? 128 : 64;
    const uint64_t data = (uint64_t)s.rows * s.cols / (s.dtype == DType::Q4_G64 ? 2 : 1);
    const uint64_t scales = (uint64_t)s.rows * (s.cols / group) * 2;
    return (data + scales) * (s.pair ? 2 : 1);
}

q27::BackendTensor upload_synthetic(q27::MetalBackend& backend, const Shape& s,
                                    std::vector<uint8_t>& data,
                                    std::vector<uint16_t>& scales) {
    const uint64_t group = s.dtype == DType::Q8_G128 ? 128 : 64;
    data.resize((uint64_t)s.rows * s.cols / (s.dtype == DType::Q4_G64 ? 2 : 1));
    for (size_t i = 0; i < data.size(); i++) data[i] = (uint8_t)(i * 2654435761u >> 24);
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

} // namespace

int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 20;
    if (reps < 1) reps = 1;
    q27::MetalBackend backend;
    printf("backend: %s, %d reps per shape\n", backend.name().c_str(), reps);

    const Shape shapes[] = {
        {"ffn_gate single   [17408x5120]", 17408, 5120, DType::Q4_G64, false},
        {"ffn_gate+up pair  [17408x5120]", 17408, 5120, DType::Q4_G64, true},
        {"ffn_gate+up pair  [17408x5120]", 17408, 5120, DType::Q8_G128, true},
        {"ffn_down          [5120x17408]", 5120, 17408, DType::Q4_G64, false},
        {"ffn_down          [5120x17408]", 5120, 17408, DType::Q8_G128, false},
        {"gdn qkv           [10240x5120]", 10240, 5120, DType::Q4_G64, false},
        {"ssm_out           [5120x6144]",  5120, 6144, DType::Q4_G64, false},
        {"output head       [151936x5120]", 151936, 5120, DType::Q8_G128, false},
    };

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

    double total_seconds = 0.0, total_bytes = 0.0;
    for (const Shape& shape : shapes) {
        std::vector<uint8_t> data, data_b;
        std::vector<uint16_t> scales, scales_b;
        q27::BackendTensor weight = upload_synthetic(backend, shape, data, scales);
        // Pair shapes stream two distinct weight tensors, matching production
        // sibling projections; reusing one tensor would let the second
        // dispatch hit cache lines the first already pulled.
        q27::BackendTensor weight_b;
        if (shape.pair) weight_b = upload_synthetic(backend, shape, data_b, scales_b);

        std::vector<float> x(shape.cols);
        for (uint32_t i = 0; i < shape.cols; i++) x[i] = (float)((i % 19) - 9) / 9.0f;
        auto xb = backend.allocate(x.size() * sizeof(float));
        backend.write(*xb, 0, x.data(), x.size() * sizeof(float));
        q27::BackendQuantized xq = backend.allocate_quantized(shape.cols);
        backend.quantize(*xb, xq);
        auto y = backend.allocate((uint64_t)shape.rows * sizeof(float));
        auto y2 = shape.pair ? backend.allocate((uint64_t)shape.rows * sizeof(float)) : nullptr;

        auto run = [&](int count) {
            backend.begin_commands();
            for (int i = 0; i < count; i++) {
                if (shape.pair) backend.matvec_quantized_pair(weight, *y, weight_b, *y2, xq);
                else backend.matvec_quantized(weight, xq, *y);
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
               shape.dtype == DType::Q4_G64 ? "q4" : "q8",
               seconds / reps * 1e3, bytes / seconds / 1e9);
        total_seconds += seconds;
        total_bytes += bytes;
    }
    printf("%-34s %-3s %8s    %11.2f GB/s\n", "aggregate", "", "", total_bytes / total_seconds / 1e9);
    return 0;
}
