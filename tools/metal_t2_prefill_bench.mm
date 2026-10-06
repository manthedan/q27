// T2 float-activation prefill GEMM microbenchmark (Bonsai 2 chunked prefill).
//
//   build/metal_t2_prefill_bench KERNELS.metal name[:tok[:rows_per_tg[:threads]]] ...
//
// Each kernel has the q27_matmul_t2_mm_f signature (weights, scales, float x,
// out, MatmulArgs); default tile 32 rows x 16 tokens on 128 threads. Runs every
// per-chunk projection shape of a t2-slim layer stack at X_ROWS tokens
// (default 96 = PREFILL_CHUNK_MAX), arms interleaved per trial, and prints GPU
// ms per chunk (token-weighted by calls per chunk) plus max relative output
// difference against the first kernel. Weights are valid ternary codes; inputs
// identical across arms. Synthetic buffers only.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Shape { uint32_t rows, cols, calls; };  // calls per prefill chunk
const Shape kShapes[] = {
    {17408, 5120, 128}, {5120, 17408, 64}, {10240, 5120, 48}, {5120, 6144, 64},
    {6144, 5120, 48},   {12288, 5120, 16}, {1024, 5120, 32},
};
struct MatmulArgs { uint32_t rows, cols, x_rows, simdgroups; };
struct Arm { std::string name; uint32_t tok = 16, rows_per_tg = 32, threads = 128; id<MTLComputePipelineState> pso; };

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 3) { fprintf(stderr, "usage: %s KERNELS.metal name:tok ...\n", argv[0]); return 2; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        NSError* err = nil;
        NSString* src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        std::vector<Arm> arms;
        for (int i = 2; i < argc; i++) {
            Arm a; std::string spec = argv[i];
            // name[:tok[:rows_per_tg[:threads]]]
            std::vector<std::string> f;
            for (size_t p = 0, c; ; p = c + 1) {
                c = spec.find(':', p);
                f.push_back(spec.substr(p, c == std::string::npos ? std::string::npos : c - p));
                if (c == std::string::npos) break;
            }
            a.name = f[0];
            if (f.size() > 1) a.tok = (uint32_t)std::stoul(f[1]);
            if (f.size() > 2) a.rows_per_tg = (uint32_t)std::stoul(f[2]);
            if (f.size() > 3) a.threads = (uint32_t)std::stoul(f[3]);
            id<MTLFunction> fn = [lib newFunctionWithName:@(a.name.c_str())];
            if (!fn) { fprintf(stderr, "no kernel %s\n", a.name.c_str()); return 1; }
            a.pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!a.pso) { fprintf(stderr, "%s: %s\n", a.name.c_str(), err.localizedDescription.UTF8String); return 1; }
            arms.push_back(a);
        }
        const uint32_t x_rows = getenv("X_ROWS") ? (uint32_t)atoi(getenv("X_ROWS")) : 96;
        const int reps = getenv("BENCH_REPS") ? atoi(getenv("BENCH_REPS")) : 8;
        const int trials = getenv("BENCH_TRIALS") ? atoi(getenv("BENCH_TRIALS")) : 5;
        std::mt19937 rng(7);
        printf("device %s, x_rows %u, %d reps x %d trials (min per call), GPU time\n", dev.name.UTF8String, x_rows, reps, trials);
        printf("%-14s %5s", "shape", "calls");
        for (auto& a : arms) printf(" %26s", (a.name + ":" + std::to_string(a.tok)).c_str());
        printf("\n");
        std::vector<double> chunk_ms(arms.size(), 0.0);
        double flops_chunk = 0.0;
        for (const Shape& s : kShapes) {
            const uint64_t wb = (uint64_t)s.rows * s.cols / 4, sb = (uint64_t)s.rows * (s.cols / 128) * 2;
            id<MTLBuffer> w = [dev newBufferWithLength:wb options:MTLResourceStorageModeShared];
            id<MTLBuffer> sc = [dev newBufferWithLength:sb options:MTLResourceStorageModeShared];
            id<MTLBuffer> x = [dev newBufferWithLength:(NSUInteger)s.cols * x_rows * 4 options:MTLResourceStorageModeShared];
            uint8_t* wp = (uint8_t*)w.contents;
            for (uint64_t i = 0; i < wb; i++) {
                uint8_t b = 0;
                for (int f = 0; f < 4; f++) b |= (uint8_t)(rng() % 3) << (2 * f);
                wp[i] = b;
            }
            __fp16* sp = (__fp16*)sc.contents;
            for (uint64_t i = 0; i < sb / 2; i++) sp[i] = (__fp16)(0.01f + 0.02f * (rng() % 1000) / 1000.0f);
            float* xp = (float*)x.contents;
            // X_SCALE stretches activations past half's range (65504) and below
            // its precision, so a kernel that rounded activations to half fails
            // the bitwise comparison below.
            const float x_scale = getenv("X_SCALE") ? (float)atof(getenv("X_SCALE")) : 1.0f;
            for (uint64_t i = 0; i < (uint64_t)s.cols * x_rows; i++)
                xp[i] = x_scale * (std::sin(0.37f * (float)(i % 9973)) + 0.01f * (float)(rng() % 100) + 1e-4f * (float)(rng() % 7));
            MatmulArgs args{s.rows, s.cols, x_rows, 1};
            flops_chunk += 2.0 * s.rows * s.cols * x_rows * s.calls;

            std::vector<id<MTLBuffer>> outs;
            for (size_t k = 0; k < arms.size(); k++)
                outs.push_back([dev newBufferWithLength:(NSUInteger)s.rows * x_rows * 4 options:MTLResourceStorageModeShared]);
            std::vector<double> best(arms.size(), 1e9);
            for (int t = 0; t < trials; t++) {
                for (size_t k0 = 0; k0 < arms.size(); k0++) {
                    const size_t k = (k0 + t) % arms.size();   // rotate arm order per trial
                    const Arm& a = arms[k];
                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:a.pso];
                    [enc setBuffer:w offset:0 atIndex:0];
                    [enc setBuffer:sc offset:0 atIndex:1];
                    [enc setBuffer:x offset:0 atIndex:2];
                    [enc setBuffer:outs[k] offset:0 atIndex:3];
                    [enc setBytes:&args length:sizeof(args) atIndex:4];
                    for (int r = 0; r < reps; r++)
                        [enc dispatchThreadgroups:MTLSizeMake((s.rows + a.rows_per_tg - 1) / a.rows_per_tg,
                                                              (x_rows + a.tok - 1) / a.tok, 1)
                            threadsPerThreadgroup:MTLSizeMake(a.threads, 1, 1)];
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    if (cb.status == MTLCommandBufferStatusError) { fprintf(stderr, "%s failed\n", a.name.c_str()); return 1; }
                    best[k] = std::min(best[k], (cb.GPUEndTime - cb.GPUStartTime) / reps);
                }
            }
            printf("%6ux%-7u %5u", s.rows, s.cols, s.calls);
            const float* ref = (const float*)outs[0].contents;
            for (size_t k = 0; k < arms.size(); k++) {
                const float* o = (const float*)outs[k].contents;
                double max_rel = 0.0;
                uint64_t bit_diffs = 0, nonfinite = 0;
                for (uint64_t i = 0; i < (uint64_t)s.rows * x_rows; i++) {
                    max_rel = std::max(max_rel, (double)std::fabs(o[i] - ref[i]) / (std::fabs(ref[i]) + 1e-2));
                    uint32_t ob, rb;
                    std::memcpy(&ob, &o[i], 4); std::memcpy(&rb, &ref[i], 4);
                    bit_diffs += ob != rb;
                    nonfinite += !std::isfinite(o[i]);
                }
                if (bit_diffs || nonfinite)
                    fprintf(stderr, "  %s %ux%u: %llu values differ bitwise, %llu non-finite\n", arms[k].name.c_str(),
                            s.rows, s.cols, (unsigned long long)bit_diffs, (unsigned long long)nonfinite);
                chunk_ms[k] += best[k] * 1e3 * s.calls;
                char cell[64];
                snprintf(cell, sizeof cell, "%7.3f ms (d %.0e)", best[k] * 1e3, max_rel);
                printf(" %26s", cell);
            }
            printf("\n");
        }
        printf("%-20s", "per 96-token chunk");
        for (size_t k = 0; k < arms.size(); k++) {
            char cell[64];
            snprintf(cell, sizeof cell, "%.0f ms %.2f TFLOP/s", chunk_ms[k], flops_chunk / (chunk_ms[k] * 1e-3) / 1e12);
            printf(" %26s", cell);
        }
        printf("\n");
    }
    return 0;
}
