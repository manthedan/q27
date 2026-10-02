// t3_gate: the T3_G128 container gate (Bonsai 2 8 GB packs, 2026-09-20).
//
// Opens the T2 pack and the T3 pack of the same model and, per T3 matrix:
//   1. gemv_t3 vs gemv_t2 on the same quantized activation  -> BITWISE
//   2. gemv_t3_n vs gemv_t2_n at widths 2, 5, 8 (distinct lanes) -> BITWISE per lane
//   3. t3_to_t2_device(T3 device copy) vs the T2 pack's device copy -> BYTE-EQUAL
//      (so the prefill GEMM, which reads that conversion, is the T2 GEMM by
//      construction)
// --bench also times gemv_t2 / gemv_t3 (widths 1, 2, 8) on blk.0's ffn_gate and
// ffn_down and reports effective GB/s: the T3 kernel does ~2 integer ops per
// weight of digit extraction, so this is where a bandwidth-starved card (3060)
// would show the extraction as exposed.
//
//   build/t3_gate <t2pack> <t3pack> [--layers N] [--bench] [--only blk.0.]
//   build/t3_gate --synthetic   (random matrices at off-model shapes, no packs needed)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../src/cuda_common.h"
#include "../src/device_model.h"
#include "../src/kernels.cuh"
#include "../src/loader.h"

using q27::DType;

static std::vector<float> rand_vec(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
static void xq_free(q27k::XQuant& q) {
    if (q.nat) cudaFree(q.nat);
    if (q.eo) cudaFree(q.eo);
    if (q.scale) cudaFree(q.scale);
    if (q.isum) cudaFree(q.isum);
    q = q27k::XQuant{};
}
static int block_index(const std::string& name) {
    if (name.compare(0, 4, "blk.") != 0) return -1;
    return atoi(name.c_str() + 4);
}
// rows differing between two device float vectors (returns count, first index)
static size_t diff_rows(const float* da, const float* db, int64_t n, int64_t* first) {
    std::vector<float> a(n), b(n);
    CUDA_CHECK(cudaMemcpy(a.data(), da, n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), db, n * 4, cudaMemcpyDeviceToHost));
    size_t nd = 0;
    *first = -1;
    for (int64_t i = 0; i < n; i++)
        if (memcmp(&a[i], &b[i], 4) != 0) {
            if (*first < 0) *first = i;
            nd++;
        }
    return nd;
}

// --synthetic: random ternary matrices at shapes the model never uses (tail
// windows of 3 and 4 chunks per lane, single-group rows, rows not a multiple
// of the 8-row block), built host-side in BOTH file layouts from the same
// trits, uploaded through the same device paths the loader uses
// (t2_interleave_device / t3_relayout_device), and held to the same bitwise
// contract as the pack gate: gemv w1/2/5/8 and the T3->T2 conversion.
static int synthetic() {
    const int rows_set[] = {1, 3, 8, 9, 33, 130};
    const int cols_set[] = {128, 256, 384, 640, 1024, 1152, 2176, 3072, 4096, 4224, 4992, 5120, 5248, 6144, 6272, 7296, 9344, 10240, 17408, 20608};
    std::mt19937 rng(7);
    int fails = 0, cases = 0;
    const int NB = 8;
    for (int cols : cols_set) for (int rows : rows_set) {
        cases++;
        const size_t n = (size_t)rows * cols;
        std::vector<uint8_t> code(n);
        for (auto& c : code) c = (uint8_t)(rng() % 3);
        std::vector<uint8_t> t2((size_t)rows * cols / 4, 0), t3((size_t)rows * (cols / 128) * 26, 0);
        for (size_t i = 0; i < n; i++) t2[i / 4] |= (uint8_t)(code[i] << ((i % 4) * 2));
        for (int r = 0; r < rows; r++)
            for (int g = 0; g < cols / 128; g++)
                for (int b = 0; b < 26; b++) {
                    int v = 0, pw = 1;
                    for (int k = 0; k < 5; k++, pw *= 3) {
                        const int j = 5 * b + k;
                        v += (j < 128 ? code[(size_t)r * cols + g * 128 + j] : 1) * pw;
                    }
                    t3[((size_t)r * (cols / 128) + g) * 26 + b] = (uint8_t)v;
                }
        std::vector<__half> sc((size_t)rows * (cols / 128));
        for (auto& x : sc) x = __float2half(0.001f + (rng() % 1000) * 1e-5f);
        uint8_t *d2, *d3raw, *d3, *conv; __half* ds;
        const uint64_t b3 = q27k::t3_device_bytes(rows, cols);
        CUDA_CHECK(cudaMalloc(&d2, t2.size())); CUDA_CHECK(cudaMalloc(&d3raw, t3.size()));
        CUDA_CHECK(cudaMalloc(&d3, b3)); CUDA_CHECK(cudaMalloc(&conv, t2.size()));
        CUDA_CHECK(cudaMalloc(&ds, sc.size() * 2));
        CUDA_CHECK(cudaMemcpy(d2, t2.data(), t2.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d3raw, t3.data(), t3.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ds, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice));
        q27k::t2_interleave_device(d2, t2.size());
        q27k::t3_relayout_device(d3raw, d3, rows, cols);
        CUDA_CHECK(cudaMemset(conv, 0xEE, t2.size()));
        q27k::t3_to_t2_device(d3, conv, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint8_t> ha(t2.size()), hb(t2.size());
        CUDA_CHECK(cudaMemcpy(ha.data(), d2, t2.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hb.data(), conv, t2.size(), cudaMemcpyDeviceToHost));
        bool ok = ha == hb;
        q27k::XQuant xq[NB];
        float* d_x; CUDA_CHECK(cudaMalloc(&d_x, cols * 4));
        for (int k = 0; k < NB; k++) {
            std::vector<float> x = rand_vec(cols, 100u + k + (uint32_t)cols);
            CUDA_CHECK(cudaMemcpy(d_x, x.data(), cols * 4, cudaMemcpyHostToDevice));
            xq[k] = q27k::xquant_alloc(cols); q27k::quantize_x(d_x, cols, xq[k]);
        }
        float *ya, *yb; CUDA_CHECK(cudaMalloc(&ya, (size_t)NB * rows * 4)); CUDA_CHECK(cudaMalloc(&yb, (size_t)NB * rows * 4));
        int64_t first;
        q27k::gemv_t2(d2, ds, xq[0], ya, rows, cols); q27k::gemv_t3(d3, ds, xq[0], yb, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        bool g1 = diff_rows(ya, yb, rows, &first) == 0;
        bool gn = true;
        for (int nb : {2, 5, 8}) {
            float* yas[NB]; float* ybs[NB];
            for (int k = 0; k < NB; k++) { yas[k] = ya + (size_t)k * rows; ybs[k] = yb + (size_t)k * rows; }
            q27k::gemv_t2_n(d2, ds, xq, nb, yas, rows, cols); q27k::gemv_t3_n(d3, ds, xq, nb, ybs, rows, cols);
            CUDA_CHECK(cudaDeviceSynchronize());
            for (int k = 0; k < nb; k++) gn &= diff_rows(yas[k], ybs[k], rows, &first) == 0;
        }
        if (!(ok && g1 && gn)) {
            fails++;
            printf("synthetic rows=%d cols=%d: conv=%s gemv=%s gemv_n=%s  FAIL\n", rows, cols, ok ? "ok" : "DIFF", g1 ? "ok" : "DIFF", gn ? "ok" : "DIFF");
        }
        for (int k = 0; k < NB; k++) xq_free(xq[k]);
        cudaFree(d_x); cudaFree(ya); cudaFree(yb); cudaFree(d2); cudaFree(d3raw); cudaFree(d3); cudaFree(conv); cudaFree(ds);
    }
    printf("t3_gate --synthetic: %d shapes, %d FAIL%s\n", cases, fails, fails ? "" : " -- bitwise at every shape");
    return fails ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc >= 2 && !strcmp(argv[1], "--synthetic")) return synthetic();
    if (argc < 3) {
        fprintf(stderr, "usage: %s <t2pack> <t3pack> [--layers N] [--bench] [--only prefix]\n", argv[0]);
        return 2;
    }
    int layers = 1 << 30;
    bool bench = false;
    std::string only;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench")) bench = true;
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
    }
    q27::Model m2 = q27::Model::open(argv[1]);
    q27::Model m3 = q27::Model::open(argv[2]);
    q27::DeviceModel d2(m2), d3(m3);

    const int NB = 8;
    int n_ok = 0, n_fail = 0, n_t3 = 0;
    uint8_t* scratch = nullptr;
    size_t scratch_bytes = 0;
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    for (const q27::Tensor& t : m3.tensors) {
        if (t.dtype != DType::T3_G128) continue;
        const int bi = block_index(t.name);
        if (bi >= layers) continue;
        if (!only.empty() && t.name.compare(0, only.size(), only) != 0) continue;
        n_t3++;
        const q27::Tensor* s = m2.find(t.name);
        if (!s || s->dtype != DType::T2_G128 || s->rows() != t.rows() || s->cols() != t.cols()) {
            printf("%-36s no matching T2_G128 tensor in the T2 pack: FAIL\n", t.name.c_str());
            n_fail++;
            continue;
        }
        const int64_t rows = (int64_t)t.rows(), cols = (int64_t)t.cols();
        const q27::DevTensor& a = d2.upload(t.name);
        const q27::DevTensor& b = d3.upload(t.name);
        if (b.data_bytes != q27k::t3_device_bytes(rows, cols)) {
            printf("%-36s device bytes %llu != %llu: FAIL\n", t.name.c_str(),
                   (unsigned long long)b.data_bytes, (unsigned long long)q27k::t3_device_bytes(rows, cols));
            n_fail++;
            continue;
        }
        // NB quantized activations, distinct per lane and per tensor
        q27k::XQuant xq[NB];
        float* d_x;
        CUDA_CHECK(cudaMalloc(&d_x, cols * 4));
        for (int n = 0; n < NB; n++) {
            std::vector<float> x = rand_vec(cols, 1000u * (uint32_t)(bi + 1) + 17u * (uint32_t)n + (uint32_t)t.name.size());
            CUDA_CHECK(cudaMemcpy(d_x, x.data(), cols * 4, cudaMemcpyHostToDevice));
            xq[n] = q27k::xquant_alloc(cols);
            q27k::quantize_x(d_x, cols, xq[n]);
        }
        float *ya, *yb;
        CUDA_CHECK(cudaMalloc(&ya, (size_t)NB * rows * 4));
        CUDA_CHECK(cudaMalloc(&yb, (size_t)NB * rows * 4));
        char line[512];
        int pos = snprintf(line, sizeof line, "%-36s [%lldx%lld]", t.name.c_str(), (long long)rows, (long long)cols);
        bool ok = true;
        int64_t first;
        // 1. single-lane
        q27k::gemv_t2((const uint8_t*)a.data, (const __half*)a.scales, xq[0], ya, rows, cols);
        q27k::gemv_t3((const uint8_t*)b.data, (const __half*)b.scales, xq[0], yb, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        size_t nd = diff_rows(ya, yb, rows, &first);
        pos += snprintf(line + pos, sizeof line - pos, " gemv=%s", nd ? "DIFF" : "ok");
        if (nd) { pos += snprintf(line + pos, sizeof line - pos, "(%zu rows, first %lld)", nd, (long long)first); ok = false; }
        // 2. multi-lane
        for (int nb : {2, 5, 8}) {
            float* yas[NB];
            float* ybs[NB];
            for (int n = 0; n < NB; n++) { yas[n] = ya + (size_t)n * rows; ybs[n] = yb + (size_t)n * rows; }
            q27k::gemv_t2_n((const uint8_t*)a.data, (const __half*)a.scales, xq, nb, yas, rows, cols);
            q27k::gemv_t3_n((const uint8_t*)b.data, (const __half*)b.scales, xq, nb, ybs, rows, cols);
            CUDA_CHECK(cudaDeviceSynchronize());
            size_t ndl = 0;
            for (int n = 0; n < nb; n++) ndl += diff_rows(yas[n], ybs[n], rows, &first);
            pos += snprintf(line + pos, sizeof line - pos, " n%d=%s", nb, ndl ? "DIFF" : "ok");
            if (ndl) ok = false;
        }
        // 3. conversion vs the T2 device copy
        {
            const size_t need = (size_t)rows * (size_t)(cols / 4);
            if (need > scratch_bytes) {
                if (scratch) CUDA_CHECK(cudaFree(scratch));
                CUDA_CHECK(cudaMalloc((void**)&scratch, need));
                scratch_bytes = need;
            }
            CUDA_CHECK(cudaMemset(scratch, 0xEE, need));
            q27k::t3_to_t2_device((const uint8_t*)b.data, scratch, rows, cols);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<uint8_t> ha(need), hb(need);
            CUDA_CHECK(cudaMemcpy(ha.data(), a.data, need, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hb.data(), scratch, need, cudaMemcpyDeviceToHost));
            size_t ndb = 0, firstb = (size_t)-1;
            for (size_t i = 0; i < need; i++)
                if (ha[i] != hb[i]) { if (firstb == (size_t)-1) firstb = i; ndb++; }
            pos += snprintf(line + pos, sizeof line - pos, " conv=%s", ndb ? "DIFF" : "ok");
            if (ndb) { pos += snprintf(line + pos, sizeof line - pos, "(%zu B, first %zu)", ndb, firstb); ok = false; }
        }
        // bench on blk.0's two big shapes (and anything under --only)
        if (bench && (bi == 0 || !only.empty()) &&
            (t.name.find("ffn_gate") != std::string::npos || t.name.find("ffn_down") != std::string::npos ||
             !only.empty())) {
            auto timeit = [&](auto&& fn) {
                fn();
                CUDA_CHECK(cudaEventRecord(e0));
                for (int r = 0; r < 20; r++) fn();
                CUDA_CHECK(cudaEventRecord(e1));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                return (double)ms / 20;
            };
            float* yas[NB];
            float* ybs[NB];
            for (int n = 0; n < NB; n++) { yas[n] = ya + (size_t)n * rows; ybs[n] = yb + (size_t)n * rows; }
            const double t2_1 = timeit([&] { q27k::gemv_t2((const uint8_t*)a.data, (const __half*)a.scales, xq[0], ya, rows, cols); });
            const double t3_1 = timeit([&] { q27k::gemv_t3((const uint8_t*)b.data, (const __half*)b.scales, xq[0], yb, rows, cols); });
            const double t2_2 = timeit([&] { q27k::gemv_t2_n((const uint8_t*)a.data, (const __half*)a.scales, xq, 2, yas, rows, cols); });
            const double t3_2 = timeit([&] { q27k::gemv_t3_n((const uint8_t*)b.data, (const __half*)b.scales, xq, 2, ybs, rows, cols); });
            const double t2_8 = timeit([&] { q27k::gemv_t2_n((const uint8_t*)a.data, (const __half*)a.scales, xq, 8, yas, rows, cols); });
            const double t3_8 = timeit([&] { q27k::gemv_t3_n((const uint8_t*)b.data, (const __half*)b.scales, xq, 8, ybs, rows, cols); });
            const double cv = timeit([&] { q27k::t3_to_t2_device((const uint8_t*)b.data, scratch, rows, cols); });
            const double gb2 = (a.data_bytes + a.scales_bytes) / 1e9, gb3 = (b.data_bytes + b.scales_bytes) / 1e9;
            printf("  bench %-28s T2 %.3f/%.3f/%.3f ms (w1/w2/w8, %.0f GB/s at w1) | T3 %.3f/%.3f/%.3f ms (%.0f GB/s) | t3->t2 %.3f ms\n",
                   t.name.c_str(), t2_1, t2_2, t2_8, gb2 / (t2_1 / 1e3), t3_1, t3_2, t3_8, gb3 / (t3_1 / 1e3), cv);
        }
        printf("%s  %s\n", line, ok ? "PASS" : "FAIL");
        if (ok) n_ok++; else n_fail++;
        for (int n = 0; n < NB; n++) xq_free(xq[n]);
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(ya));
        CUDA_CHECK(cudaFree(yb));
    }
    printf("t3_gate: %d T3 tensors, %d PASS, %d FAIL%s\n", n_t3, n_ok, n_fail,
           n_fail ? " -- T3 IS NOT BITWISE T2" : " -- T3 == T2 bitwise (GEMV w1/2/5/8 + prefill conversion)");
    return n_fail ? 1 : 0;
}
