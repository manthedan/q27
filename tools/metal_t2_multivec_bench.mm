// Small-K T2 multi-vector matvec microbenchmark (speculative verify: K drafted
// tokens through one weight pass).
//
//   build/metal_t2_multivec_bench KERNELS.metal name:K:rows_per_tg[:threads] ...
//
// Each kernel takes (weights, scales, x[K][cols], out[K][rows], {rows, cols, K})
// and is compared, vector by vector and bitwise, with the production
// q27_matvec_t2_g128 run once per vector. Shapes and per-token call counts
// are the Bonsai 2 decode set; weights rotate through >= 256 MiB of copies so
// every call streams from DRAM. Prints ms per full layer stack (one verify
// round) and its ratio to one production decode step.

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

struct Shape { uint32_t rows, cols, calls; };
const Shape kShapes[] = {
    {17408, 5120, 128}, {5120, 17408, 64}, {10240, 5120, 48}, {5120, 6144, 64},
    {6144, 5120, 48},   {248320, 5120, 1}, {12288, 5120, 16}, {1024, 5120, 32},
};
struct MatvecArgs { uint32_t rows, cols; };
struct MvkArgs { uint32_t rows, cols, k; };
struct Arm { std::string name; uint32_t k = 1, rows_per_tg = 32, threads = 256; id<MTLComputePipelineState> pso; };

std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out;
    for (size_t p = 0, q; ; p = q + 1) {
        q = s.find(c, p);
        out.push_back(s.substr(p, q == std::string::npos ? std::string::npos : q - p));
        if (q == std::string::npos) break;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 3) { fprintf(stderr, "usage: %s KERNELS.metal name:K:rows_per_tg[:threads] ...\n", argv[0]); return 2; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        NSError* err = nil;
        NSString* src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&err];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        auto pso = [&](const char* name) {
            id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
            return fn ? [dev newComputePipelineStateWithFunction:fn error:nil] : nil;
        };
        id<MTLComputePipelineState> prod = pso("q27_matvec_t2_g128");
        std::vector<Arm> arms;
        uint32_t kmax = 1;
        for (int i = 2; i < argc; i++) {
            auto f = split(argv[i], ':');
            Arm a; a.name = f[0];
            if (f.size() > 1) a.k = (uint32_t)std::stoul(f[1]);
            if (f.size() > 2) a.rows_per_tg = (uint32_t)std::stoul(f[2]);
            if (f.size() > 3) a.threads = (uint32_t)std::stoul(f[3]);
            a.pso = pso(a.name.c_str());
            if (!a.pso || !prod) { fprintf(stderr, "no kernel %s\n", a.name.c_str()); return 1; }
            kmax = std::max(kmax, a.k);
            arms.push_back(a);
        }
        const int reps = getenv("BENCH_REPS") ? atoi(getenv("BENCH_REPS")) : 20;
        std::mt19937 rng(11);
        printf("device %s; ms per call; 'stack' = one full Bonsai 2 layer stack (decode call mix)\n", dev.name.UTF8String);
        printf("%-14s %5s %12s", "shape", "calls", "prod K=1");
        for (auto& a : arms) printf(" %22s", (a.name + " K=" + std::to_string(a.k)).c_str());
        printf("\n");
        double prod_stack = 0.0;
        std::vector<double> arm_stack(arms.size(), 0.0);
        bool all_same = true;
        for (const Shape& s : kShapes) {
            const uint64_t wb = (uint64_t)s.rows * s.cols / 4, sb = (uint64_t)s.rows * (s.cols / 128) * 2;
            const int copies = (int)std::max<uint64_t>(1, (256ull << 20) / (wb + sb) + 1);
            const int calls = std::max(reps, copies);
            NSMutableArray<id<MTLBuffer>>* ws = [NSMutableArray array];
            NSMutableArray<id<MTLBuffer>>* scs = [NSMutableArray array];
            id<MTLBuffer> w = [dev newBufferWithLength:wb options:MTLResourceStorageModeShared];
            id<MTLBuffer> sc = [dev newBufferWithLength:sb options:MTLResourceStorageModeShared];
            uint8_t* wp = (uint8_t*)w.contents;
            for (uint64_t i = 0; i < wb; i++) {
                uint8_t b = 0;
                for (int f = 0; f < 4; f++) b |= (uint8_t)(rng() % 3) << (2 * f);
                wp[i] = b;
            }
            __fp16* sp = (__fp16*)sc.contents;
            for (uint64_t i = 0; i < sb / 2; i++) sp[i] = (__fp16)(0.01f + 0.02f * (rng() % 1000) / 1000.0f);
            for (int c = 0; c < copies; c++) {
                [ws addObject:c ? [dev newBufferWithBytes:w.contents length:wb options:MTLResourceStorageModeShared] : w];
                [scs addObject:c ? [dev newBufferWithBytes:sc.contents length:sb options:MTLResourceStorageModeShared] : sc];
            }
            id<MTLBuffer> x = [dev newBufferWithLength:(NSUInteger)s.cols * kmax * 4 options:MTLResourceStorageModeShared];
            float* xp = (float*)x.contents;
            for (uint64_t i = 0; i < (uint64_t)s.cols * kmax; i++) xp[i] = std::sin(0.37f * (float)(i % 9973)) + 0.01f * (float)(rng() % 100);

            auto time_kernel = [&](id<MTLComputePipelineState> p, const void* args, size_t args_len, id<MTLBuffer> out,
                                   NSUInteger groups, NSUInteger threads, NSUInteger x_offset) {
                double best = 1e9;
                for (int t = 0; t < 3; t++) {
                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:p];
                    [enc setBuffer:x offset:x_offset atIndex:2];
                    [enc setBuffer:out offset:0 atIndex:3];
                    [enc setBytes:args length:args_len atIndex:4];
                    for (int i = 0; i < calls; i++) {
                        [enc setBuffer:ws[i % copies] offset:0 atIndex:0];
                        [enc setBuffer:scs[i % copies] offset:0 atIndex:1];
                        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) / calls);
                }
                return best;
            };
            // Production reference: one call per vector (outputs kept for the bitwise check).
            std::vector<std::vector<float>> ref(kmax);
            MatvecArgs pargs{s.rows, s.cols};
            double prod_t = 0.0;
            for (uint32_t v = 0; v < kmax; v++) {
                id<MTLBuffer> out = [dev newBufferWithLength:(NSUInteger)s.rows * 4 options:MTLResourceStorageModeShared];
                const double t = time_kernel(prod, &pargs, sizeof pargs, out, (s.rows + 31) / 32, 256,
                                             (NSUInteger)v * s.cols * 4);
                if (v == 0) prod_t = t;
                ref[v].assign((const float*)out.contents, (const float*)out.contents + s.rows);
            }
            prod_stack += prod_t * 1e3 * s.calls;
            printf("%6ux%-7u %5u %9.3f ms", s.rows, s.cols, s.calls, prod_t * 1e3);
            for (size_t k = 0; k < arms.size(); k++) {
                const Arm& a = arms[k];
                MvkArgs args{s.rows, s.cols, a.k};
                id<MTLBuffer> out = [dev newBufferWithLength:(NSUInteger)s.rows * a.k * 4 options:MTLResourceStorageModeShared];
                const double t = time_kernel(a.pso, &args, sizeof args, out, (s.rows + a.rows_per_tg - 1) / a.rows_per_tg,
                                             a.threads, 0);
                arm_stack[k] += t * 1e3 * s.calls;
                uint64_t diffs = 0, nonfinite = 0;
                const float* op = (const float*)out.contents;
                for (uint32_t v = 0; v < a.k; v++)
                    for (uint32_t r = 0; r < s.rows; r++) {
                        const float o = op[(uint64_t)v * s.rows + r];
                        nonfinite += !std::isfinite(o);
                        uint32_t ob, rb; std::memcpy(&ob, &o, 4); std::memcpy(&rb, &ref[v][r], 4);
                        diffs += ob != rb;
                    }
                all_same = all_same && !diffs && !nonfinite;
                char cell[64];
                snprintf(cell, sizeof cell, "%.3f ms %s", t * 1e3, nonfinite ? "NONFINITE" : diffs ? "DIFF" : "same");
                printf(" %22s", cell);
            }
            printf("\n");
        }
        printf("%-20s %9.1f ms", "per layer stack", prod_stack);
        for (size_t k = 0; k < arms.size(); k++) {
            char cell[64];
            snprintf(cell, sizeof cell, "%.1f ms (x%.2f)", arm_stack[k], arm_stack[k] / prod_stack);
            printf(" %22s", cell);
        }
        printf("\n%s\n", all_same ? "outputs: bitwise identical to production per vector" : "outputs: DIFFER (see cells)");
    }
    return 0;
}
