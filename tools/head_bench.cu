// head_bench: what a single-lane MTP draft step spends in the vocab head vs the
// MTP block's own matmuls, and what a reduced-vocab head would cost (rows
// [0, N) of the same matrix). Same kernels the draft path calls (gemv_q4 /
// gemv_q8 / gemv_t2 by dtype), CUDA-event timed, L2-cold-ish by rotating.
//   build/head_bench model.q27 [rows...]
#include <cstdio>
#include <functional>
#include <cstdlib>
#include <string>
#include <vector>
#include "../src/cuda_common.h"
#include "../src/device_model.h"
#include "../src/kernels.cuh"
#include "../src/loader.h"
using q27::DType;
static void gemv(const q27::DevTensor& w, const q27k::XQuant& xq, float* y, int64_t rows) {
    switch (w.dtype) {
        case DType::Q4_G64: q27k::gemv_q4((const uint8_t*)w.data, (const __half*)w.scales, xq, y, rows, w.cols); break;
        case DType::Q8_G128: q27k::gemv_q8((const int8_t*)w.data, (const __half*)w.scales, xq, y, rows, w.cols); break;
        case DType::T2_G128: q27k::gemv_t2((const uint8_t*)w.data, (const __half*)w.scales, xq, y, rows, w.cols); break;
        case DType::T3_G128: q27k::gemv_t3((const uint8_t*)w.data, (const __half*)w.scales, xq, y, rows, w.cols); break;
        default: fprintf(stderr, "dtype %s unsupported\n", q27::dtype_name(w.dtype)); exit(1);
    }
}
static double time_ms(const std::function<void()>& f, int reps = 200) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 10; i++) f();
    cudaEventRecord(a); for (int i = 0; i < reps; i++) f(); cudaEventRecord(b);
    cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}
int main(int argc, char** argv) {
    q27::Model m = q27::Model::open(argv[1]);
    q27::DeviceModel dm(m);
    const char* hn = m.find("output_q4.weight") ? "output_q4.weight" : "output.weight";
    const q27::DevTensor& h = dm.upload(hn);
    const int64_t cols = (int64_t)h.cols;
    float* x; float* y; CUDA_CHECK(cudaMalloc(&x, 17408 * 4)); CUDA_CHECK(cudaMalloc(&y, h.rows * 4));
    std::vector<float> hx(17408, 0.01f); CUDA_CHECK(cudaMemcpy(x, hx.data(), 17408 * 4, cudaMemcpyHostToDevice));
    q27k::XQuant xq = q27k::xquant_alloc(17408); q27k::quantize_x(x, 17408, xq);
    printf("head %s (%s, %lld x %lld)\n", hn, q27::dtype_name(h.dtype), (long long)h.rows, (long long)cols);
    std::vector<int64_t> rows = {(int64_t)h.rows, 102400, 70656, 47104, 32768};
    for (int i = 2; i < argc; i++) rows.push_back(atoll(argv[i]));
    for (int64_t r : rows) {
        if (r > (int64_t)h.rows) continue;
        double ms = time_ms([&] { gemv(h, xq, y, r); });
        printf("  head rows %7lld: %.4f ms\n", (long long)r, ms);
    }
    // the MTP block's own matmuls (blk.64.*), single lane, in the order a draft step runs them
    double blk = 0; int nb = 0;
    for (const q27::Tensor& t : m.tensors) {
        if (t.name.rfind("blk.64.", 0) != 0 || t.shape.size() != 2) continue;
        if (t.dtype == DType::F32 || t.dtype == DType::F16) continue;
        const q27::DevTensor& w = dm.upload(t.name);
        if (w.cols > 17408) continue;
        q27k::quantize_x(x, w.cols, xq);
        double ms = time_ms([&] { gemv(w, xq, y, (int64_t)w.rows); });
        blk += ms; nb++;
        printf("  %-36s %-8s %6lld x %6lld  %.4f ms\n", t.name.c_str(), q27::dtype_name(w.dtype), (long long)w.rows, (long long)w.cols, ms);
    }
    printf("MTP block matmuls: %d tensors, %.4f ms total\n", nb, blk);
}
