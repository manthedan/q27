// T2_G128 decode-matvec microbenchmark: achieved bandwidth per production
// shape, against a pure-read roofline on the same buffers.
//
//   build/metal_t2_matvec_bench KERNELS.metal [name[:rows_per_tg[:threads]] ...]
//
// Compiles KERNELS.metal (default kernel: q27_matvec_t2_g128:32:256, the
// production geometry), runs each kernel over synthetic T2 weights at the
// Bonsai 2 decode shapes (rows x cols and per-token call counts measured from
// a real t2-slim decode), and prints GB/s per shape plus the token-weighted
// total. Outputs of every kernel are compared with the first one's. The
// roofline row is a float4 sum over the same weight bytes (an upper bound for
// any kernel that must read them once). Kernels share the production
// signature: (weights, scales, x, out, {rows, cols}).
// Synthetic buffers only (<= ~1.6 GiB), no model artifact. Each shape rotates
// through >= 256 MiB of weight copies so calls stream from DRAM, as in decode.

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

struct Shape { uint32_t rows, cols, calls; };  // calls per decode token
// Measured from a t2-slim decode (6.80 GB of T2 weights+scales per token).
const Shape kShapes[] = {
    {17408, 5120, 128}, {5120, 17408, 64}, {10240, 5120, 48}, {5120, 6144, 64},
    {6144, 5120, 48},   {248320, 5120, 1}, {12288, 5120, 16}, {1024, 5120, 32},
};

struct Args { uint32_t rows, cols; };

struct Kernel { std::string name; uint32_t rows_per_tg = 32, threads = 256; id<MTLComputePipelineState> pso; };

uint64_t weight_bytes(const Shape& s) { return (uint64_t)s.rows * s.cols / 4; }
uint64_t scale_bytes(const Shape& s) { return (uint64_t)s.rows * (s.cols / 128) * 2; }

double gpu_seconds(id<MTLCommandBuffer> cb) { return cb.GPUEndTime - cb.GPUStartTime; }

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s KERNELS.metal [name[:rows_per_tg[:threads]] ...]\n", argv[0]);
            return 2;
        }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        NSError* err = nil;
        NSString* src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
        // Append the roofline kernel to whatever source we were given.
        src = [src stringByAppendingString:@R"MSL(
kernel void q27_bench_read_roof(device const float4 *w [[buffer(0)]],
                                device float *out [[buffer(1)]],
                                constant uint &n4 [[buffer(2)]],
                                uint gid [[thread_position_in_grid]],
                                uint grid [[threads_per_grid]]) {
    float4 acc = 0.0f;
    for (uint i = gid; i < n4; i += grid) acc += w[i];
    if (acc.x == 12345.678f) out[gid] = acc.y + acc.z + acc.w;
}
)MSL"];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }

        std::vector<Kernel> kernels;
        for (int i = 2; i < argc; i++) {
            Kernel k; std::string spec = argv[i];
            size_t a = spec.find(':');
            k.name = spec.substr(0, a);
            if (a != std::string::npos) {
                size_t b = spec.find(':', a + 1);
                k.rows_per_tg = (uint32_t)std::stoul(spec.substr(a + 1, b - a - 1));
                if (b != std::string::npos) k.threads = (uint32_t)std::stoul(spec.substr(b + 1));
            }
            kernels.push_back(k);
        }
        if (kernels.empty()) kernels.push_back(Kernel{"q27_matvec_t2_g128", 32, 256, nil});
        for (auto& k : kernels) {
            id<MTLFunction> fn = [lib newFunctionWithName:@(k.name.c_str())];
            if (!fn) { fprintf(stderr, "no kernel %s\n", k.name.c_str()); return 1; }
            k.pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!k.pso) { fprintf(stderr, "pipeline %s: %s\n", k.name.c_str(), err.localizedDescription.UTF8String); return 1; }
        }
        id<MTLFunction> roof_fn = [lib newFunctionWithName:@"q27_bench_read_roof"];
        id<MTLComputePipelineState> roof = [dev newComputePipelineStateWithFunction:roof_fn error:&err];

        const int reps = getenv("BENCH_REPS") ? atoi(getenv("BENCH_REPS")) : 20;
        std::mt19937 rng(1234);
        printf("device %s, %d reps per shape, GPU time per call\n", dev.name.UTF8String, reps);
        printf("%-14s %6s", "shape", "calls");
        printf(" %12s", "roof GB/s");  // codes-only read kernel
        for (auto& k : kernels) printf(" %28s", k.name.c_str());
        printf("\n");

        std::vector<double> token_s(kernels.size(), 0.0);
        double token_roof_s = 0.0, token_roof_bytes = 0.0, token_bytes = 0.0;
        for (const Shape& s : kShapes) {
            const uint64_t wb = weight_bytes(s), sb = scale_bytes(s);
            // Decode streams every weight from DRAM once per token: rotate
            // through enough copies (>= 256 MiB) that no call hits in cache.
            const int copies = (int)std::max<uint64_t>(1, (256ull << 20) / (wb + sb) + 1);
            NSMutableArray<id<MTLBuffer>>* ws = [NSMutableArray array];
            NSMutableArray<id<MTLBuffer>>* scs = [NSMutableArray array];
            id<MTLBuffer> w = [dev newBufferWithLength:wb options:MTLResourceStorageModeShared];
            id<MTLBuffer> sc = [dev newBufferWithLength:sb options:MTLResourceStorageModeShared];
            id<MTLBuffer> x = [dev newBufferWithLength:(NSUInteger)s.cols * 4 options:MTLResourceStorageModeShared];
            uint8_t* wp = (uint8_t*)w.contents;
            for (uint64_t i = 0; i < wb; i++) {
                uint8_t b = 0;
                for (int f = 0; f < 4; f++) b |= (uint8_t)(rng() % 3) << (2 * f);  // codes 0..2 only
                wp[i] = b;
            }
            __fp16* sp = (__fp16*)sc.contents;
            for (uint64_t i = 0; i < sb / 2; i++) sp[i] = (__fp16)(0.01f + 0.02f * (rng() % 1000) / 1000.0f);
            float* xp = (float*)x.contents;
            for (uint32_t i = 0; i < s.cols; i++) xp[i] = std::sin(0.37f * i) + 0.1f * ((int)(rng() % 7) - 3);
            for (int c = 0; c < copies; c++) {
                id<MTLBuffer> wc = c ? [dev newBufferWithBytes:w.contents length:wb options:MTLResourceStorageModeShared] : w;
                id<MTLBuffer> scc = c ? [dev newBufferWithBytes:sc.contents length:sb options:MTLResourceStorageModeShared] : sc;
                [ws addObject:wc]; [scs addObject:scc];
            }
            Args args{s.rows, s.cols};
            // Dispatches per command buffer: at least one per copy, so every
            // trial traverses the whole >= 256 MiB ring (not just `reps` copies).
            const int calls = std::max(reps, copies);

            // Roofline: read the weight bytes once per call.
            id<MTLBuffer> sink = [dev newBufferWithLength:4 * 65536 options:MTLResourceStorageModePrivate];
            const uint32_t n4 = (uint32_t)(wb / 16);
            double roof_best = 1e9;
            for (int r = 0; r < 3; r++) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:roof];
                [enc setBuffer:sink offset:0 atIndex:1];
                [enc setBytes:&n4 length:4 atIndex:2];
                for (int i = 0; i < calls; i++) {
                    [enc setBuffer:ws[i % copies] offset:0 atIndex:0];
                    [enc dispatchThreads:MTLSizeMake(std::min<uint32_t>(n4, 65536), 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                roof_best = std::min(roof_best, gpu_seconds(cb) / calls);
            }
            // The roof kernel reads the codes only; credit just those bytes.
            printf("%6ux%-7u %6u %12.1f", s.rows, s.cols, s.calls, wb / roof_best / 1e9);
            token_roof_s += roof_best * s.calls;
            token_roof_bytes += (double)wb * s.calls;
            token_bytes += (double)(wb + sb) * s.calls;

            std::vector<float> ref;
            for (size_t ki = 0; ki < kernels.size(); ki++) {
                Kernel& k = kernels[ki];
                id<MTLBuffer> out = [dev newBufferWithLength:(NSUInteger)s.rows * 4 options:MTLResourceStorageModeShared];
                double best = 1e9;
                for (int r = 0; r < 3; r++) {
                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:k.pso];
                    [enc setBuffer:x offset:0 atIndex:2];
                    [enc setBuffer:out offset:0 atIndex:3];
                    [enc setBytes:&args length:sizeof(args) atIndex:4];
                    const NSUInteger groups = (s.rows + k.rows_per_tg - 1) / k.rows_per_tg;
                    for (int i = 0; i < calls; i++) {
                        [enc setBuffer:ws[i % copies] offset:0 atIndex:0];
                        [enc setBuffer:scs[i % copies] offset:0 atIndex:1];
                        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(k.threads, 1, 1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    if (cb.status == MTLCommandBufferStatusError) { fprintf(stderr, "\n%s failed\n", k.name.c_str()); return 1; }
                    best = std::min(best, gpu_seconds(cb) / calls);
                }
                token_s[ki] += best * s.calls;
                const float* op = (const float*)out.contents;
                // Bitwise + finiteness (std::max would silently skip NaN).
                double max_rel = 0.0;
                uint64_t bit_diffs = 0, nonfinite = 0;
                for (uint32_t i = 0; i < s.rows; i++) nonfinite += !std::isfinite(op[i]);
                if (ki == 0) ref.assign(op, op + s.rows);
                else
                    for (uint32_t i = 0; i < s.rows; i++) {
                        uint32_t a, b;
                        std::memcpy(&a, &op[i], 4); std::memcpy(&b, &ref[i], 4);
                        bit_diffs += a != b;
                        const double rel = std::fabs(op[i] - ref[i]) / (std::fabs(ref[i]) + 1e-3);
                        if (!(rel <= max_rel)) max_rel = std::isnan(rel) ? INFINITY : rel;
                    }
                char cell[64];
                if (nonfinite) snprintf(cell, sizeof cell, "NONFINITE x%llu", (unsigned long long)nonfinite);
                else if (ki == 0) snprintf(cell, sizeof cell, "%.1f GB/s", (wb + sb) / best / 1e9);
                else snprintf(cell, sizeof cell, "%.1f GB/s (%s %.0e)", (wb + sb) / best / 1e9,
                              bit_diffs ? "diff" : "same", max_rel);
                printf(" %28s", cell);
            }
            printf("\n");
        }
        printf("%-21s %12.1f", "per decode token", token_roof_bytes / token_roof_s / 1e9);
        for (size_t ki = 0; ki < kernels.size(); ki++) {
            char cell[64];
            snprintf(cell, sizeof cell, "%.1f GB/s %.1f ms", token_bytes / token_s[ki] / 1e9, token_s[ki] * 1e3);
            printf(" %28s", cell);
        }
        printf("\n");
    }
    return 0;
}
