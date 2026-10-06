// Metal simdgroup-MMA peak probe: half x float, float x float and half x half
// 8x8x8 MMAs on resident operands, four independent accumulator chains, all
// stored to device memory (dead chains were optimized away in earlier
// versions and read 15-19 TFLOP/s). Mini M4: 3.85 TFLOP/s for all three.
//   c++ -O2 -std=c++17 -fobjc-arc tools/metal_mma_peak.mm -framework Foundation -framework Metal -o build/metal_mma_peak
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <algorithm>
static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
template <typename TA, typename TB>
inline void body(device float *out, uint gid, ushort lane, uint iters, threadgroup float *t) {
    // Operands from memory (non-uniform): no constant folding.
    for (uint i = lane; i < 64; i += 32) t[i] = float((i * 37 + gid) % 17) * 0.01f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup TA *ta = (threadgroup TA *)(t + 64); threadgroup TB *tb = (threadgroup TB *)(t + 128);
    for (uint i = lane; i < 64; i += 32) { ta[i] = TA(t[i]); tb[i] = TB(t[63 - i]); }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<TA, 8, 8> a; simdgroup_load(a, ta, 8);
    simdgroup_matrix<TB, 8, 8> b0; simdgroup_load(b0, tb, 8);
    simdgroup_matrix<TB, 8, 8> b1; simdgroup_load(b1, tb, 8, ulong2(0), true);
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), c1 = c0, c2 = c0, c3 = c0;
    for (uint i = 0; i < iters; i++) {   // 4 independent accumulators for ILP
        simdgroup_multiply_accumulate(c0, a, b0, c0);
        simdgroup_multiply_accumulate(c1, a, b1, c1);
        simdgroup_multiply_accumulate(c2, a, b0, c2);
        simdgroup_multiply_accumulate(c3, a, b1, c3);
    }
    // Every accumulator reaches device memory: no chain is dead.
    device float *o = out + (ulong)gid * 1024 + (lane / 32) * 256;
    simdgroup_store(c0, o, 8); simdgroup_store(c1, o + 64, 8); simdgroup_store(c2, o + 128, 8); simdgroup_store(c3, o + 192, 8);
}
kernel void mma_hf(device float *out [[buffer(0)]], constant uint &iters [[buffer(1)]], uint gid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) { threadgroup float t[256]; body<half, float>(out, gid, lane, iters, t); }
kernel void mma_ff(device float *out [[buffer(0)]], constant uint &iters [[buffer(1)]], uint gid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) { threadgroup float t[256]; body<float, float>(out, gid, lane, iters, t); }
kernel void mma_hh(device float *out [[buffer(0)]], constant uint &iters [[buffer(1)]], uint gid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) { threadgroup float t[256]; body<half, half>(out, gid, lane, iters, t); }
)MSL";
int main() { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError* e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc) options:[MTLCompileOptions new] error:&e];
    if (!lib) { printf("%s\n", e.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLBuffer> out = [dev newBufferWithLength:4096ull * 1024 * 4 options:MTLResourceStorageModeShared];
    for (NSString* name in @[@"mma_hf", @"mma_ff", @"mma_hh"]) {
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&e];
        const uint iters = 4096, groups = 4096, threads = 128;   // 4 simdgroups per TG
        double best = 1e9;
        for (int t = 0; t < 3; t++) {
            id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:p]; [enc setBuffer:out offset:0 atIndex:0]; [enc setBytes:&iters length:4 atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
        }
        const double flops = 2.0 * 512 * 4.0 * iters * (threads / 32) * groups;
        printf("%s: %.2f TFLOP/s\n", name.UTF8String, flops / best / 1e12);
    }
}}
