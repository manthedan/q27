#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace q27 {
namespace {

class MetalBuffer final : public BackendBuffer {
  public:
    explicit MetalBuffer(id<MTLBuffer> buffer) : buffer_(buffer) {}
    uint64_t size() const override { return (uint64_t)buffer_.length; }
    id<MTLBuffer> handle() const { return buffer_; }

  private:
    id<MTLBuffer> buffer_;
};

MetalBuffer& metal_buffer(BackendBuffer& buffer) {
    auto* result = dynamic_cast<MetalBuffer*>(&buffer);
    if (!result) throw std::runtime_error("q27 Metal: buffer belongs to another backend");
    return *result;
}

const MetalBuffer& metal_buffer(const BackendBuffer& buffer) {
    auto* result = dynamic_cast<const MetalBuffer*>(&buffer);
    if (!result) throw std::runtime_error("q27 Metal: buffer belongs to another backend");
    return *result;
}

const MetalBuffer& tensor_data(const BackendTensor& tensor, DType dtype, const char* operation) {
    if (tensor.dtype != dtype || !tensor.data)
        throw std::runtime_error(std::string("q27 Metal: invalid tensor for ") + operation);
    return metal_buffer(*tensor.data);
}

void check_range(uint64_t size, uint64_t offset, uint64_t bytes, const char* operation) {
    if (offset > size || bytes > size - offset)
        throw std::runtime_error(std::string("q27 Metal: buffer range error in ") + operation);
}

// Bounds a tensor access by its logical extent, not just its buffer: shared
// whole-mapping buffers would otherwise let a per-tensor byte-count error
// read silently into the neighboring tensor.
uint64_t tensor_limit(uint64_t buffer_size, uint64_t offset, uint64_t logical_size) {
    if (!logical_size) return buffer_size;
    const uint64_t end = offset > UINT64_MAX - logical_size ? UINT64_MAX : offset + logical_size;
    return end < buffer_size ? end : buffer_size;
}

// Must match the "Q27_SHADER_ABI" tag in q27_kernels.metal. Shaders compile
// from that file at runtime, so a host binary built before a buffer-binding
// change would otherwise misbind silently against a newer shader file.
constexpr const char* kShaderAbiTag = "// Q27_SHADER_ABI 7";

NSString* load_kernel_source() {
    NSFileManager* files = [NSFileManager defaultManager];
    NSMutableArray<NSString*>* candidates = [NSMutableArray array];
    if (const char* override_path = getenv("Q27_METAL_SOURCE"))
        [candidates addObject:[NSString stringWithUTF8String:override_path]];
    [candidates addObject:@"src/metal/q27_kernels.metal"];
    [candidates addObject:@"./src/metal/q27_kernels.metal"];

    for (NSString* path in candidates) {
        if (![files fileExistsAtPath:path]) continue;
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:path
                                                      encoding:NSUTF8StringEncoding
                                                         error:&error];
        if (!source)
            throw std::runtime_error("q27 Metal: cannot read shader source: " +
                                     std::string(error.localizedDescription.UTF8String));
        // Match the tag as a complete logical line (any line ending), so
        // ABI 2 does not accept ABI 20 and CRLF sources still load.
        bool tag_found = false;
        for (NSString* line in [source componentsSeparatedByCharactersInSet:
                                           NSCharacterSet.newlineCharacterSet])
            if ([[line stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet]
                    isEqualToString:@(kShaderAbiTag)]) { tag_found = true; break; }
        if (!tag_found)
            throw std::runtime_error(std::string("q27 Metal: shader ABI mismatch: ") +
                                     path.UTF8String + " does not carry \"" + kShaderAbiTag +
                                     "\"; rebuild this binary against the current shader source");
        return source;
    }
    throw std::runtime_error("q27 Metal: src/metal/q27_kernels.metal not found "
                             "(set Q27_METAL_SOURCE)");
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                          NSString* name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function)
        throw std::runtime_error("q27 Metal: shader function not found: " +
                                 std::string(name.UTF8String));
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                  error:&error];
    if (!pipeline)
        throw std::runtime_error("q27 Metal: pipeline creation failed for " +
                                 std::string(name.UTF8String) + ": " +
                                 std::string(error.localizedDescription.UTF8String));
    if (pipeline.threadExecutionWidth != 32 || pipeline.maxTotalThreadsPerThreadgroup < 256)
        throw std::runtime_error("q27 Metal: matvec requires simd width 32 and 256-thread groups");
    return pipeline;
}

struct MatvecArgs {
    uint32_t rows;
    uint32_t cols;
    uint32_t simdgroups;
};
struct MatvecPairArgs { uint32_t rows_a, rows_b, cols, simdgroups; };
struct MatmulArgs { uint32_t rows,cols,x_rows,simdgroups; };
struct VectorArgs { uint32_t n, groups; float eps; };
struct HeadArgs { uint32_t heads, head_dim, stride, groups; float eps; };
struct GateArgs { uint32_t heads, head_dim; };
struct ConcatArgs { uint32_t a_count, b_count; };
struct RopeArgs { uint32_t heads, head_dim, n_rot, stride, position; float freq_base; };
struct KvStoreArgs { uint32_t position, row_length; };
struct TurboWhtArgs { uint32_t heads, stride, inverse; };
struct TurboStoreArgs { uint32_t position, kv_heads; };
struct AttentionArgs { uint32_t q_stride, seq_len, q_heads, kv_heads, head_dim; float scale; };
struct AttentionGqaArgs { uint32_t q_stride, seq_len, q_heads, kv_heads, head_dim, block, n_blocks; float scale; };
struct AttentionGqaCausalArgs { uint32_t q_stride, q_row_stride, base_len, q_heads, kv_heads, head_dim, block, n_blocks_max, tokens; float scale; };
struct TopkArgs { uint32_t n, k, capacity; };
struct DeltaArgs { uint32_t value_heads, qk_heads, head_dim; };
struct EmbedRowsArgs { uint32_t cols, count, tokens[96]; };
struct RowsNormArgs { uint32_t n, rows, groups; float eps; };
struct MatvecPairRowsArgs { uint32_t rows_a, rows_b, cols, tokens; };
struct GatesRowsArgs { uint32_t heads, tokens; };
struct ConvChunkArgs { uint32_t channels, tokens; };
struct DeltaChunkArgs { uint32_t value_heads, qk_heads, head_dim, tokens; };
struct L2RowsArgs { uint32_t heads, head_dim, row_stride, tokens; float eps; };
struct RopeRowsArgs { uint32_t heads, head_dim, n_rot, stride, row_stride, position, tokens; float freq_base; };
struct KvStoreRowsArgs { uint32_t position, row_length, tokens; };
struct TurboStoreRowsArgs { uint32_t position, kv_heads, tokens; };
struct GateRowsArgs { uint32_t heads, head_dim, tokens; };
struct ArgmaxRowsArgs { uint32_t n, rows; };
struct AttentionCausalArgs { uint32_t q_stride, q_row_stride, base_len, q_heads, kv_heads, head_dim, tokens; float scale; };

} // namespace

struct MetalBackend::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> f32;
    id<MTLComputePipelineState> f16;
    id<MTLComputePipelineState> q8;
    id<MTLComputePipelineState> q4;
    id<MTLComputePipelineState> t2;
    id<MTLComputePipelineState> t3;
    id<MTLComputePipelineState> t2_quantized_matmul_h;
    id<MTLComputePipelineState> mask_logits_p;
    // Half-staging T2 chunk GEMM: default ON (1.28x chunk rate at quality
    // parity on this tier's gates); Q27_METAL_GEMM_HALF=0 opts out.
    bool gemm_half = true;
    id<MTLComputePipelineState> quantize;
    id<MTLComputePipelineState> q8_quantized;
    id<MTLComputePipelineState> q4_quantized;
    id<MTLComputePipelineState> t2_quantized;
    id<MTLComputePipelineState> t2_quantized_x2;
    id<MTLComputePipelineState> t2_x2;
    id<MTLComputePipelineState> f16_pair;
    id<MTLComputePipelineState> q4_quantized_matmul;
    id<MTLComputePipelineState> q8_quantized_matmul;
    id<MTLComputePipelineState> t2_quantized_matmul;
    id<MTLComputePipelineState> embedding;
    id<MTLComputePipelineState> embedding_t2;
    id<MTLComputePipelineState> embedding_dev;
    id<MTLComputePipelineState> embedding_t2_dev;
    id<MTLComputePipelineState> embedding_t2_rows;
    id<MTLComputePipelineState> rms;
    id<MTLComputePipelineState> rms_quantized;
    id<MTLComputePipelineState> rms_heads;
    id<MTLComputePipelineState> l2_heads;
    id<MTLComputePipelineState> silu;
    id<MTLComputePipelineState> add;
    id<MTLComputePipelineState> concat;
    id<MTLComputePipelineState> copy_bytes;
    id<MTLComputePipelineState> sigmoid_gate;
    id<MTLComputePipelineState> rope;
    id<MTLComputePipelineState> argmax;
    id<MTLComputePipelineState> kv_store;
    id<MTLComputePipelineState> turbo_wht;
    id<MTLComputePipelineState> kv_store_turbo3;
    id<MTLComputePipelineState> attention_turbo3;
    id<MTLComputePipelineState> attention;
    id<MTLComputePipelineState> gates;
    id<MTLComputePipelineState> conv;
    id<MTLComputePipelineState> delta;
    id<MTLComputePipelineState> gated_norm;
    id<MTLComputePipelineState> embedding_rows;
    id<MTLComputePipelineState> rms_rows_quantized;
    id<MTLComputePipelineState> f16_pair_rows;
    id<MTLComputePipelineState> gates_rows;
    id<MTLComputePipelineState> conv_chunked;
    id<MTLComputePipelineState> delta_chunked;
    id<MTLComputePipelineState> l2_rows;
    id<MTLComputePipelineState> rope_rows;
    id<MTLComputePipelineState> kv_store_rows;
    id<MTLComputePipelineState> kv_store_turbo3_rows;
    id<MTLComputePipelineState> attention_causal;
    id<MTLComputePipelineState> attention_turbo3_causal_p;
    id<MTLComputePipelineState> sigmoid_gate_rows;
    id<MTLComputePipelineState> argmax_rows_p;
    id<MTLComputePipelineState> nll_rows_p;
    id<MTLComputePipelineState> attention_f16_gqa_p;
    id<MTLComputePipelineState> attention_turbo3_gqa_p;
    id<MTLComputePipelineState> attention_gqa_merge_p;
    id<MTLComputePipelineState> attention_f16_causal_gqa_p;
    id<MTLComputePipelineState> attention_turbo3_causal_gqa_p;
    id<MTLComputePipelineState> attention_gqa_merge_rows_p;
    // R1b token-tiled causal GQA (t2 = production route, t4 + hm = bench).
    id<MTLComputePipelineState> attention_turbo3_gqa_hm_p;
    id<MTLComputePipelineState> attention_turbo3_causal_gqa_t2_p;
    id<MTLComputePipelineState> attention_turbo3_causal_gqa_t4_p;
    id<MTLComputePipelineState> attention_turbo3_causal_gqa_bf2_p;
    id<MTLComputePipelineState> mma_roofline_a_p;
    id<MTLComputePipelineState> mma_roofline_b_p;
    id<MTLComputePipelineState> mma_roofline_b_eq_p;
    id<MTLComputePipelineState> mma_roofline_cx_p;
    id<MTLComputePipelineState> mm_dr_p;
    id<MTLComputePipelineState> mm_dr2_p;
    id<MTLComputePipelineState> x_to_half_t_p;
    id<MTLBuffer> dr_xt_scratch;
    id<MTLComputePipelineState> attention_f16_causal_gqa_t2_p;
    // Q27_METAL_GQA_TILE: causal token-tile factor, 1 (untiled A/B lever)
    // or 2 (default; docs/plans/2026-07-15-cache-block-scheduling.md R1b).
    uint32_t gqa_tile = 2;
    // Q27_METAL_GQA_BLOCK: positions per (kvh, block) threadgroup. Default
    // 1024; smaller blocks trade merge work for threadgroup count — the
    // occupancy lever the latency-bound Phase-0 finding points at. Changing
    // it changes the merge fold order (margin-aware-gates contract class),
    // and chunk↔decode parity holds at any single value.
    uint32_t gqa_block = 1024;
    id<MTLComputePipelineState> topk_logits_p;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    bool batching = false;

    // Model mappings that fit maxBufferLength are wrapped as a single
    // MTLBuffer (tensors bind at offsets), and on macOS 15+ that buffer joins
    // a residency set attached to the queue: pages stay wired between command
    // buffers, so file-backed weight pages are neither faulted in token by
    // token on first touch nor evictable under memory pressure mid-run.
    std::map<const void*, std::weak_ptr<MetalBuffer>> model_wraps;
    API_AVAILABLE(macos(15.0)) id<MTLResidencySet> residency_set;

    // GQA KV-reuse decode attention: sequences at or beyond the threshold
    // route to the blocked kernels that read each KV row once for all
    // q_heads/kv_heads query heads. 0 disables; 1 forces every sequence
    // (parity testing). The block partials scratch grows as context deepens
    // and is GPU-private (never host-read).
    uint32_t gqa_threshold = 2048;
    id<MTLBuffer> gqa_partials;

    void attention_gqa_dispatch(bool turbo3, const MetalBuffer& qb, uint32_t q_stride,
                                const MetalBuffer& kc, const MetalBuffer& vc,
                                MetalBuffer& output, uint32_t seq_len, uint32_t q_heads,
                                uint32_t kv_heads, uint32_t head_dim, float scale) {
        const uint32_t block = gqa_block;
        const uint32_t n_blocks = 1 + (seq_len - 1) / block;   // seq_len >= 1 host-checked
        const uint32_t gqa = q_heads / kv_heads;
        const uint64_t partial_bytes = (uint64_t)q_heads * n_blocks * 258 * 4;
        if (!gqa_partials || gqa_partials.length < partial_bytes)
            gqa_partials = [device newBufferWithLength:(NSUInteger)partial_bytes
                                               options:MTLResourceStorageModePrivate];
        if (!gqa_partials) throw std::runtime_error("q27 Metal: GQA partial allocation failed");
        AttentionGqaArgs args{q_stride, seq_len, q_heads, kv_heads, head_dim,
                              block, n_blocks, scale};
        @autoreleasepool {
            bool own;
            auto enc = encoder_for_operation(own, turbo3 ? "q27_attention_turbo3_gqa"
                                                         : "q27_attention_f16_gqa");
            [enc setComputePipelineState:turbo3 ? attention_turbo3_gqa_p : attention_f16_gqa_p];
            [enc setBuffer:qb.handle() offset:0 atIndex:0];
            [enc setBuffer:kc.handle() offset:0 atIndex:1];
            [enc setBuffer:vc.handle() offset:0 atIndex:2];
            [enc setBuffer:gqa_partials offset:0 atIndex:3];
            [enc setBytes:&args length:sizeof(args) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(kv_heads, n_blocks, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)gqa * 32, 1, 1)];
            // Same serial encoder: the merge reads the partials the first
            // dispatch wrote; serial compute encoders order dispatches.
            [enc setComputePipelineState:attention_gqa_merge_p];
            [enc setBuffer:gqa_partials offset:0 atIndex:0];
            [enc setBuffer:output.handle() offset:0 atIndex:1];
            [enc setBytes:&args length:sizeof(args) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(q_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            if (own) finish_command("GQA attention");
        }
    }

    void attention_gqa_causal_dispatch(bool turbo3, const MetalBuffer& qb, uint32_t q_stride,
                                       uint32_t q_row_stride, const MetalBuffer& kc,
                                       const MetalBuffer& vc, MetalBuffer& output,
                                       uint32_t base_len, uint32_t q_heads, uint32_t kv_heads,
                                       uint32_t head_dim, uint32_t tokens, float scale,
                                       uint64_t q_byte_offset, uint64_t out_byte_offset) {
        const uint32_t block = gqa_block;
        const uint32_t max_seq = base_len + tokens - 1;     // overflow host-checked
        const uint32_t n_blocks_max = 1 + (max_seq - 1) / block;
        const uint32_t gqa = q_heads / kv_heads;
        const uint64_t partial_bytes =
            (uint64_t)tokens * q_heads * n_blocks_max * 258 * 4;
        if (!gqa_partials || gqa_partials.length < partial_bytes)
            gqa_partials = [device newBufferWithLength:(NSUInteger)partial_bytes
                                               options:MTLResourceStorageModePrivate];
        if (!gqa_partials) throw std::runtime_error("q27 Metal: GQA partial allocation failed");
        AttentionGqaCausalArgs args{q_stride, q_row_stride, base_len, q_heads, kv_heads,
                                    head_dim, block, n_blocks_max, tokens, scale};
        // R1b: factor-2 token tiling — bit-identical per token to the untiled
        // kernels, so the chunk↔decode parity contract is unaffected.
        const bool tiled = gqa_tile >= 2 && tokens >= 2;
        @autoreleasepool {
            bool own;
            auto enc = encoder_for_operation(own,
                turbo3 ? (tiled ? "q27_attention_turbo3_causal_gqa_t2" : "q27_attention_turbo3_causal_gqa")
                       : (tiled ? "q27_attention_f16_causal_gqa_t2" : "q27_attention_f16_causal_gqa"));
            [enc setComputePipelineState:turbo3 ? (tiled ? attention_turbo3_causal_gqa_t2_p : attention_turbo3_causal_gqa_p)
                                                : (tiled ? attention_f16_causal_gqa_t2_p : attention_f16_causal_gqa_p)];
            [enc setBuffer:qb.handle() offset:(NSUInteger)q_byte_offset atIndex:0];
            [enc setBuffer:kc.handle() offset:0 atIndex:1];
            [enc setBuffer:vc.handle() offset:0 atIndex:2];
            [enc setBuffer:gqa_partials offset:0 atIndex:3];
            [enc setBytes:&args length:sizeof(args) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(kv_heads, n_blocks_max, tiled ? (tokens + 1) / 2 : tokens)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)gqa * 32, 1, 1)];
            // Same serial encoder: the merge reads the partials the first
            // dispatch wrote; serial compute encoders order dispatches.
            [enc setComputePipelineState:attention_gqa_merge_rows_p];
            [enc setBuffer:gqa_partials offset:0 atIndex:0];
            [enc setBuffer:output.handle() offset:(NSUInteger)out_byte_offset atIndex:1];
            [enc setBytes:&args length:sizeof(args) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(q_heads, tokens, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            if (own) finish_command("GQA chunked attention");
        }
    }

    // Q27_METAL_PROFILE=1: per-dispatch GPU-time attribution. Every operation
    // runs in its own compute encoder bracketed by stage-boundary timestamp
    // samples, so independent dispatches no longer overlap; absolute wall time
    // is inflated, the per-kernel shares are the signal.
    static constexpr uint32_t kMaxProfiledOps = 2048;
    struct ProfileStat { uint64_t calls = 0; uint64_t ticks = 0; };
    bool profile = false;
    id<MTLCounterSampleBuffer> counter_buffer;
    std::vector<const char*> op_labels;
    std::map<std::string, ProfileStat> profile_stats;
    uint64_t profiled_command_buffers = 0;
    double gpu_busy_seconds = 0.0;
    double cpu_wait_seconds = 0.0;
    MTLTimestamp calibration_cpu = 0, calibration_gpu = 0;

    void start_command(bool explicit_batch) {
        if (encoder) throw std::runtime_error("q27 Metal: command batch already active");
        command = [queue commandBuffer];
        if (!command) throw std::runtime_error("q27 Metal: command creation failed");
        if (profile) {
            op_labels.clear();
        } else {
            encoder = [command computeCommandEncoder];
            if (!encoder) throw std::runtime_error("q27 Metal: command creation failed");
        }
        batching = explicit_batch;
    }

    id<MTLComputeCommandEncoder> encoder_for_operation(bool& own_command, const char* label) {
        own_command = !batching;
        if (own_command) start_command(false);
        if (profile) {
            if (op_labels.size() >= kMaxProfiledOps) {
                // Split the batch so the sample buffer never overflows; the
                // queue preserves ordering across the two command buffers.
                const bool was_batching = batching;
                finish_command("profiling split");
                start_command(was_batching);
            }
            if (encoder) { [encoder endEncoding]; encoder = nil; }
            MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
            MTLComputePassSampleBufferAttachmentDescriptor* attachment = pass.sampleBufferAttachments[0];
            attachment.sampleBuffer = counter_buffer;
            attachment.startOfEncoderSampleIndex = op_labels.size() * 2;
            attachment.endOfEncoderSampleIndex = op_labels.size() * 2 + 1;
            encoder = [command computeCommandEncoderWithDescriptor:pass];
            if (!encoder) throw std::runtime_error("q27 Metal: profiled encoder creation failed");
            op_labels.push_back(label);
        }
        return encoder;
    }

    void finish_command(const char* label) {
        if (!command || (!encoder && !profile))
            throw std::runtime_error("q27 Metal: no active command batch");
        if (encoder) { [encoder endEncoding]; encoder = nil; }
        const auto wait_start = std::chrono::steady_clock::now();
        [command commit];
        [command waitUntilCompleted];
        cpu_wait_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start).count();
        if (command.status == MTLCommandBufferStatusError) {
            std::string message = std::string("q27 Metal: ") + label + " failed";
            if (command.error) message += ": " + std::string(command.error.localizedDescription.UTF8String);
            command = nil;
            batching = false;
            throw std::runtime_error(message);
        }
        if (profile) resolve_profile_samples();
        command = nil;
        batching = false;
    }

    void resolve_profile_samples() {
        profiled_command_buffers++;
        if (command.GPUEndTime > command.GPUStartTime)
            gpu_busy_seconds += command.GPUEndTime - command.GPUStartTime;
        if (op_labels.empty()) return;
        NSData* data = [counter_buffer resolveCounterRange:NSMakeRange(0, op_labels.size() * 2)];
        if (data && data.length >= op_labels.size() * 2 * sizeof(MTLCounterResultTimestamp)) {
            const auto* samples = (const MTLCounterResultTimestamp*)data.bytes;
            for (size_t i = 0; i < op_labels.size(); i++) {
                const uint64_t begin = samples[2 * i].timestamp, end = samples[2 * i + 1].timestamp;
                if (begin == MTLCounterErrorValue || end == MTLCounterErrorValue || end < begin) continue;
                ProfileStat& stat = profile_stats[op_labels[i]];
                stat.calls++;
                stat.ticks += end - begin;
            }
        }
        op_labels.clear();
    }

    void report_profile() {
        if (!profile || profile_stats.empty()) return;
        // Convert GPU timestamp ticks to nanoseconds with a session-spanning
        // calibration pair; on Apple Silicon the timebase is usually already
        // nanoseconds, but that is not contractual.
        MTLTimestamp cpu_now = 0, gpu_now = 0;
        [device sampleTimestamps:&cpu_now gpuTimestamp:&gpu_now];
        double ns_per_tick = 1.0;
        if (gpu_now > calibration_gpu && cpu_now > calibration_cpu)
            ns_per_tick = double(cpu_now - calibration_cpu) / double(gpu_now - calibration_gpu);
        std::vector<std::pair<std::string, ProfileStat>> rows(profile_stats.begin(), profile_stats.end());
        std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& b) { return a.second.ticks > b.second.ticks; });
        double total_ns = 0.0;
        uint64_t total_calls = 0;
        for (const auto& row : rows) { total_ns += double(row.second.ticks) * ns_per_tick; total_calls += row.second.calls; }
        fprintf(stderr, "q27 Metal profile (per-op encoders; overlap suppressed, shares are the signal)\n");
        fprintf(stderr, "%-36s %10s %12s %10s %6s\n", "kernel", "calls", "total ms", "avg us", "share");
        for (const auto& row : rows) {
            const double ns = double(row.second.ticks) * ns_per_tick;
            fprintf(stderr, "%-36s %10llu %12.2f %10.1f %5.1f%%\n", row.first.c_str(),
                    (unsigned long long)row.second.calls, ns / 1e6,
                    ns / 1e3 / double(row.second.calls), 100.0 * ns / total_ns);
        }
        fprintf(stderr, "%-36s %10llu %12.2f\n", "total sampled", (unsigned long long)total_calls, total_ns / 1e6);
        fprintf(stderr, "command buffers %llu, GPU busy %.3f s, CPU wait %.3f s\n",
                (unsigned long long)profiled_command_buffers, gpu_busy_seconds, cpu_wait_seconds);
    }
};

MetalBackend::MetalBackend() : impl_(new Impl) {
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) throw std::runtime_error("q27 Metal: no Metal device");
        impl_->queue = [impl_->device newCommandQueue];
        if (!impl_->queue) throw std::runtime_error("q27 Metal: cannot create command queue");

        if (@available(macOS 15.0, *)) {
            const char* env = getenv("Q27_METAL_NO_RESIDENCY");
            if (!env || !*env || *env == '0') {
                MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
                residency_descriptor.label = @"q27 model weights";
                NSError* residency_error = nil;
                impl_->residency_set =
                    [impl_->device newResidencySetWithDescriptor:residency_descriptor
                                                           error:&residency_error];
                if (impl_->residency_set) [impl_->queue addResidencySet:impl_->residency_set];
            }
        }

        MTLCompileOptions* options = [MTLCompileOptions new];
        // Correctness baseline; enable relaxed math only after CUDA comparison.
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError* error = nil;
        impl_->library = [impl_->device newLibraryWithSource:load_kernel_source()
                                                      options:options
                                                        error:&error];
        if (!impl_->library)
            throw std::runtime_error("q27 Metal: shader compilation failed: " +
                                     std::string(error.localizedDescription.UTF8String));
        impl_->f32 = make_pipeline(impl_->device, impl_->library, @"q27_matvec_f32");
        impl_->f16 = make_pipeline(impl_->device, impl_->library, @"q27_matvec_f16");
        impl_->q8 = make_pipeline(impl_->device, impl_->library, @"q27_matvec_q8_g128");
        impl_->q4 = make_pipeline(impl_->device, impl_->library, @"q27_matvec_q4_g64");
        impl_->t2 = make_pipeline(impl_->device, impl_->library, @"q27_matvec_t2_g128");
        impl_->t3 = make_pipeline(impl_->device, impl_->library, @"q27_matvec_t3_g128");
        impl_->mask_logits_p = make_pipeline(impl_->device, impl_->library, @"q27_mask_logits");
        impl_->quantize = make_pipeline(impl_->device, impl_->library, @"q27_quantize_x");
        impl_->q8_quantized = make_pipeline(impl_->device, impl_->library, @"q27_matvec_q8_quantized");
        impl_->q4_quantized = make_pipeline(impl_->device, impl_->library, @"q27_matvec_q4_quantized");
        impl_->t2_quantized = make_pipeline(impl_->device, impl_->library, @"q27_matvec_t2_quantized");
        impl_->t2_quantized_x2 = make_pipeline(impl_->device, impl_->library, @"q27_matvec_t2_quantized_x2");
        impl_->t2_x2 = make_pipeline(impl_->device, impl_->library, @"q27_matvec_t2_g128_x2");
        impl_->f16_pair = make_pipeline(impl_->device, impl_->library, @"q27_matvec_f16_pair");
        // SIMD-scoped matrix multiply is optional on older Intel-family Metal
        // devices. Decode GEMV remains available there; only small-N GEMM is gated.
        if ([impl_->device supportsFamily:MTLGPUFamilyApple7]) {
            impl_->q4_quantized_matmul = make_pipeline(impl_->device, impl_->library, @"q27_matmul_q4_mm");
            impl_->q8_quantized_matmul = make_pipeline(impl_->device, impl_->library, @"q27_matmul_q8_mm");
            impl_->t2_quantized_matmul = make_pipeline(impl_->device, impl_->library, @"q27_matmul_t2_mm");
            impl_->t2_quantized_matmul_h = make_pipeline(impl_->device, impl_->library, @"q27_matmul_t2_mm_h");
            impl_->mma_roofline_a_p = make_pipeline(impl_->device, impl_->library, @"q27_mma_roofline_a");
            impl_->mma_roofline_b_p = make_pipeline(impl_->device, impl_->library, @"q27_mma_roofline_b");
            impl_->mma_roofline_b_eq_p = make_pipeline(impl_->device, impl_->library, @"q27_mma_roofline_b_eq");
            impl_->mma_roofline_cx_p = make_pipeline(impl_->device, impl_->library, @"q27_mma_roofline_cx");
            impl_->mm_dr_p = make_pipeline(impl_->device, impl_->library, @"q27_matmul_t2_mm_dr");
            impl_->mm_dr2_p = make_pipeline(impl_->device, impl_->library, @"q27_matmul_t2_mm_dr2");
            impl_->x_to_half_t_p = make_pipeline(impl_->device, impl_->library, @"q27_x_int8_to_half_t");
            if (const char* env = getenv("Q27_METAL_GEMM_HALF"); env && *env)
                impl_->gemm_half = strtoul(env, nullptr, 10) != 0;
        }
        impl_->embedding = make_pipeline(impl_->device, impl_->library, @"q27_embedding_q8");
        impl_->embedding_t2 = make_pipeline(impl_->device, impl_->library, @"q27_embedding_t2");
        impl_->embedding_dev = make_pipeline(impl_->device, impl_->library, @"q27_embedding_q8_dev");
        impl_->embedding_t2_dev = make_pipeline(impl_->device, impl_->library, @"q27_embedding_t2_dev");
        impl_->embedding_t2_rows = make_pipeline(impl_->device, impl_->library, @"q27_embedding_t2_rows");
        impl_->rms = make_pipeline(impl_->device, impl_->library, @"q27_rmsnorm");
        impl_->rms_quantized = make_pipeline(impl_->device, impl_->library, @"q27_rmsnorm_quantized");
        impl_->rms_heads = make_pipeline(impl_->device, impl_->library, @"q27_rmsnorm_heads");
        impl_->l2_heads = make_pipeline(impl_->device, impl_->library, @"q27_l2norm_heads");
        impl_->silu = make_pipeline(impl_->device, impl_->library, @"q27_silu_mul");
        impl_->add = make_pipeline(impl_->device, impl_->library, @"q27_add_inplace");
        impl_->concat = make_pipeline(impl_->device, impl_->library, @"q27_concat");
        impl_->copy_bytes = make_pipeline(impl_->device, impl_->library, @"q27_copy_bytes");
        impl_->sigmoid_gate = make_pipeline(impl_->device, impl_->library, @"q27_sigmoid_gate_mul");
        impl_->rope = make_pipeline(impl_->device, impl_->library, @"q27_rope_neox");
        impl_->argmax = make_pipeline(impl_->device, impl_->library, @"q27_argmax");
        impl_->kv_store = make_pipeline(impl_->device, impl_->library, @"q27_kv_store_f16");
        impl_->turbo_wht = make_pipeline(impl_->device, impl_->library, @"q27_turbo_wht");
        impl_->kv_store_turbo3 = make_pipeline(impl_->device, impl_->library, @"q27_kv_store_turbo3");
        impl_->attention_turbo3 = make_pipeline(impl_->device, impl_->library, @"q27_attention_turbo3");
        impl_->attention = make_pipeline(impl_->device, impl_->library, @"q27_attention_f16");
        impl_->gates = make_pipeline(impl_->device, impl_->library, @"q27_gdn_gates");
        impl_->conv = make_pipeline(impl_->device, impl_->library, @"q27_conv_step");
        impl_->delta = make_pipeline(impl_->device, impl_->library, @"q27_delta_step");
        impl_->gated_norm = make_pipeline(impl_->device, impl_->library, @"q27_gated_norm_gdn");
        impl_->embedding_rows = make_pipeline(impl_->device, impl_->library, @"q27_embedding_q8_rows");
        impl_->rms_rows_quantized = make_pipeline(impl_->device, impl_->library, @"q27_rmsnorm_rows_quantized");
        impl_->f16_pair_rows = make_pipeline(impl_->device, impl_->library, @"q27_matvec_f16_pair_rows");
        impl_->gates_rows = make_pipeline(impl_->device, impl_->library, @"q27_gdn_gates_rows");
        impl_->conv_chunked = make_pipeline(impl_->device, impl_->library, @"q27_conv_chunk");
        impl_->delta_chunked = make_pipeline(impl_->device, impl_->library, @"q27_delta_chunk");
        impl_->l2_rows = make_pipeline(impl_->device, impl_->library, @"q27_l2norm_rows");
        impl_->rope_rows = make_pipeline(impl_->device, impl_->library, @"q27_rope_neox_rows");
        impl_->kv_store_rows = make_pipeline(impl_->device, impl_->library, @"q27_kv_store_f16_rows");
        impl_->kv_store_turbo3_rows = make_pipeline(impl_->device, impl_->library, @"q27_kv_store_turbo3_rows");
        impl_->attention_causal = make_pipeline(impl_->device, impl_->library, @"q27_attention_f16_causal");
        impl_->attention_turbo3_causal_p = make_pipeline(impl_->device, impl_->library, @"q27_attention_turbo3_causal");
        impl_->sigmoid_gate_rows = make_pipeline(impl_->device, impl_->library, @"q27_sigmoid_gate_mul_rows");
        impl_->argmax_rows_p = make_pipeline(impl_->device, impl_->library, @"q27_argmax_rows");
        impl_->nll_rows_p = make_pipeline(impl_->device, impl_->library, @"q27_nll_rows");
        impl_->attention_f16_gqa_p = make_pipeline(impl_->device, impl_->library, @"q27_attention_f16_gqa");
        impl_->attention_turbo3_gqa_p = make_pipeline(impl_->device, impl_->library, @"q27_attention_turbo3_gqa");
        impl_->attention_gqa_merge_p = make_pipeline(impl_->device, impl_->library, @"q27_attention_gqa_merge");
        impl_->attention_f16_causal_gqa_p = make_pipeline(impl_->device, impl_->library, @"q27_attention_f16_causal_gqa");
        impl_->attention_turbo3_causal_gqa_p = make_pipeline(impl_->device, impl_->library, @"q27_attention_turbo3_causal_gqa");
        impl_->attention_gqa_merge_rows_p = make_pipeline(impl_->device, impl_->library, @"q27_attention_gqa_merge_rows");
        impl_->attention_turbo3_gqa_hm_p = make_pipeline(impl_->device, impl_->library, @"q27_attention_turbo3_gqa_hm");
        impl_->attention_turbo3_causal_gqa_t2_p = make_pipeline(impl_->device, impl_->library, @"q27_attention_turbo3_causal_gqa_t2");
        impl_->attention_turbo3_causal_gqa_t4_p = make_pipeline(impl_->device, impl_->library, @"q27_attention_turbo3_causal_gqa_t4");
        impl_->attention_turbo3_causal_gqa_bf2_p = make_pipeline(impl_->device, impl_->library, @"q27_attention_turbo3_causal_gqa_bf2");
        impl_->attention_f16_causal_gqa_t2_p = make_pipeline(impl_->device, impl_->library, @"q27_attention_f16_causal_gqa_t2");
        if (const char* env = getenv("Q27_METAL_GQA_TILE"); env && *env) {
            const unsigned long tile = strtoul(env, nullptr, 10);
            if (tile != 1 && tile != 2)
                throw std::runtime_error("q27 Metal: Q27_METAL_GQA_TILE must be 1 or 2");
            impl_->gqa_tile = (uint32_t)tile;
        }
        if (const char* env = getenv("Q27_METAL_GQA_BLOCK"); env && *env) {
            const unsigned long block = strtoul(env, nullptr, 10);
            // Power of two in [128, 4096]: kernels stage 8-row tiles, and the
            // straddle split assumes the threshold is block-aligned-agnostic.
            if (block < 128 || block > 4096 || (block & (block - 1)))
                throw std::runtime_error("q27 Metal: Q27_METAL_GQA_BLOCK must be a power of two in [128, 4096]");
            impl_->gqa_block = (uint32_t)block;
        }
        impl_->topk_logits_p = make_pipeline(impl_->device, impl_->library, @"q27_topk_logits");
        if (const char* env = getenv("Q27_METAL_GQA_THRESHOLD"); env && *env)
            impl_->gqa_threshold = (uint32_t)strtoul(env, nullptr, 10);

        if (const char* env = getenv("Q27_METAL_PROFILE"); env && *env && *env != '0') {
            id<MTLCounterSet> timestamps = nil;
            for (id<MTLCounterSet> set in impl_->device.counterSets)
                if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) timestamps = set;
            if (timestamps &&
                [impl_->device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
                MTLCounterSampleBufferDescriptor* descriptor = [MTLCounterSampleBufferDescriptor new];
                descriptor.counterSet = timestamps;
                descriptor.storageMode = MTLStorageModeShared;
                descriptor.sampleCount = Impl::kMaxProfiledOps * 2;
                NSError* error = nil;
                impl_->counter_buffer = [impl_->device newCounterSampleBufferWithDescriptor:descriptor
                                                                                       error:&error];
            }
            if (impl_->counter_buffer) {
                impl_->profile = true;
                [impl_->device sampleTimestamps:&impl_->calibration_cpu
                                    gpuTimestamp:&impl_->calibration_gpu];
                fprintf(stderr, "q27 Metal: per-dispatch profiling enabled\n");
            } else {
                fprintf(stderr, "q27 Metal: Q27_METAL_PROFILE set but stage-boundary "
                                "timestamp sampling is unavailable; profiling disabled\n");
            }
        }
    }
}

MetalBackend::~MetalBackend() {
    @autoreleasepool { impl_->report_profile(); }
}

std::string MetalBackend::name() const {
    return std::string(impl_->device.name.UTF8String);
}

std::shared_ptr<BackendBuffer> MetalBackend::allocate(uint64_t bytes) {
    if (!bytes || bytes > (uint64_t)impl_->device.maxBufferLength ||
        bytes > (uint64_t)std::numeric_limits<NSUInteger>::max())
        throw std::runtime_error("q27 Metal: invalid buffer length");
    id<MTLBuffer> buffer = [impl_->device newBufferWithLength:(NSUInteger)bytes
                                                      options:MTLResourceStorageModeShared];
    if (!buffer) throw std::runtime_error("q27 Metal: buffer allocation failed");
    return std::make_shared<MetalBuffer>(buffer);
}

void MetalBackend::write(BackendBuffer& dst, uint64_t offset, const void* src, uint64_t bytes) {
    MetalBuffer& buffer = metal_buffer(dst);
    check_range(buffer.size(), offset, bytes, "write");
    if (bytes && !src) throw std::runtime_error("q27 Metal: null write source");
    // Same contract zero() already enforces: a host write racing an open
    // command batch changes what already-encoded operations observe.
    if (impl_->batching) throw std::runtime_error("q27 Metal: cannot CPU-write during command batch");
    if (bytes) std::memcpy((uint8_t*)buffer.handle().contents + offset, src, (size_t)bytes);
}

void MetalBackend::read(const BackendBuffer& src, uint64_t offset, void* dst, uint64_t bytes) {
    const MetalBuffer& buffer = metal_buffer(src);
    check_range(buffer.size(), offset, bytes, "read");
    if (bytes && !dst) throw std::runtime_error("q27 Metal: null read destination");
    synchronize();
    if (bytes) std::memcpy(dst, (const uint8_t*)buffer.handle().contents + offset, (size_t)bytes);
}

void MetalBackend::zero(BackendBuffer& dst) {
    MetalBuffer& buffer = metal_buffer(dst);
    if (impl_->batching) throw std::runtime_error("q27 Metal: cannot CPU-clear during command batch");
    std::memset(buffer.handle().contents, 0, (size_t)buffer.size());
}

void MetalBackend::copy(const BackendBuffer& src, uint64_t src_offset,
                        BackendBuffer& dst, uint64_t dst_offset, uint64_t bytes) {
    const MetalBuffer& sb=metal_buffer(src); MetalBuffer& db=metal_buffer(dst);
    check_range(sb.size(),src_offset,bytes,"copy source"); check_range(db.size(),dst_offset,bytes,"copy destination");
    if (!bytes) return;
    if (bytes > UINT32_MAX) throw std::runtime_error("q27 Metal: single copy exceeds kernel limit");
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_copy_bytes"); [enc setComputePipelineState:impl_->copy_bytes];
        [enc setBuffer:sb.handle() offset:(NSUInteger)src_offset atIndex:0];
        [enc setBuffer:db.handle() offset:(NSUInteger)dst_offset atIndex:1];
        [enc setBytes:&bytes length:sizeof(bytes) atIndex:2];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)bytes,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(own) impl_->finish_command("copy");
    }
}

BackendTensor MetalBackend::upload(const Tensor& tensor) {
    BackendTensor result;
    result.dtype = tensor.dtype;
    result.rows = tensor.rows();
    result.cols = tensor.cols();
    result.data = allocate(tensor.data_size);
    result.data_size = tensor.data_size;
    write(*result.data, 0, tensor.data, tensor.data_size);
    if (tensor.scales_size) {
        result.scales = allocate(tensor.scales_size);
        result.scales_size = tensor.scales_size;
        write(*result.scales, 0, tensor.scales, tensor.scales_size);
    }
    return result;
}

BackendTensor MetalBackend::upload(const Model& model, const Tensor& tensor) {
    // Preferred form: one MTLBuffer wraps the whole mapping and every tensor
    // binds at an offset. One buffer instead of two per tensor keeps the
    // per-commit residency/tracking work constant in model size, and gives
    // the residency set a single allocation to wire. Mappings larger than
    // maxBufferLength (the 5.25 bpw tier) keep per-tensor page-aligned views
    // and stay unwired -- they do not fit in memory either way.
    auto wrap_mapping = [&]() -> std::shared_ptr<MetalBuffer> {
        void* base = (void*)model.mapping_base();
        const uint64_t size = model.mapping_size();
        auto& slot = impl_->model_wraps[base];
        if (auto held = slot.lock()) return held;
        madvise(base, (size_t)size, MADV_WILLNEED);
        id<MTLBuffer> buffer = [impl_->device newBufferWithBytesNoCopy:base
                                                                length:(NSUInteger)size
                                                               options:MTLResourceStorageModeShared
                                                           deallocator:nil];
        if (!buffer) throw std::runtime_error("q27 Metal: cannot wrap model mmap");
        std::shared_ptr<MetalBuffer> wrapped;
        if (@available(macOS 15.0, *)) {
            if (id<MTLResidencySet> set = impl_->residency_set) {
                [set addAllocation:buffer];
                [set commit];
                [set requestResidency];
                // The set retains the buffer and keeps its pages wired; drop
                // it when the last tensor goes away so a later munmap cannot
                // leave the set holding a dead address range.
                wrapped = std::shared_ptr<MetalBuffer>(
                    new MetalBuffer(buffer), [set](MetalBuffer* wrapper) {
                        if (@available(macOS 15.0, *)) {
                            [set removeAllocation:wrapper->handle()];
                            [set commit];
                        }
                        delete wrapper;
                    });
            }
        }
        if (!wrapped) wrapped = std::make_shared<MetalBuffer>(buffer);
        slot = wrapped;
        return wrapped;
    };

    auto mapping_offset = [&](const uint8_t* ptr, uint64_t bytes) -> uint64_t {
        if (!ptr || !bytes) throw std::runtime_error("q27 Metal: invalid model view");
        const uintptr_t base = (uintptr_t)model.mapping_base();
        const uintptr_t address = (uintptr_t)ptr;
        if (address < base) throw std::runtime_error("q27 Metal: tensor precedes model mapping");
        const uint64_t offset = (uint64_t)(address - base);
        if (offset > model.mapping_size() || bytes > model.mapping_size() - offset)
            throw std::runtime_error("q27 Metal: tensor outside model mapping");
        return offset;
    };

    if (!model.mapping_base() || !model.mapping_size())
        throw std::runtime_error("q27 Metal: invalid model view");
    if (model.mapping_size() <= (uint64_t)impl_->device.maxBufferLength) {
        BackendTensor result;
        result.dtype = tensor.dtype;
        result.rows = tensor.rows();
        result.cols = tensor.cols();
        result.data_offset = mapping_offset(tensor.data, tensor.data_size);
        result.data = wrap_mapping();
        result.data_size = tensor.data_size;
        if (tensor.scales_size) {
            result.scales_offset = mapping_offset(tensor.scales, tensor.scales_size);
            result.scales = result.data;
            result.scales_size = tensor.scales_size;
        }
        return result;
    }

    auto wrap = [&](const uint8_t* ptr, uint64_t bytes, uint64_t& inner) {
        if (!ptr || !bytes || !model.mapping_base() || !model.mapping_size())
            throw std::runtime_error("q27 Metal: invalid model view");
        const uintptr_t base = (uintptr_t)model.mapping_base();
        const uintptr_t address = (uintptr_t)ptr;
        if (address < base) throw std::runtime_error("q27 Metal: tensor precedes model mapping");
        const uint64_t offset = (uint64_t)(address - base);
        const uint64_t model_size = model.mapping_size();
        if (offset > model_size || bytes > model_size - offset)
            throw std::runtime_error("q27 Metal: tensor outside model mapping");

        const uint64_t page = (uint64_t)getpagesize();
        const uint64_t page_offset = offset - offset % page;
        inner = offset - page_offset;
        if (bytes > UINT64_MAX - inner)
            throw std::runtime_error("q27 Metal: model view size overflow");
        uint64_t view_bytes = inner + bytes;
        if (view_bytes > UINT64_MAX - (page - 1))
            throw std::runtime_error("q27 Metal: model view alignment overflow");
        view_bytes = (view_bytes + page - 1) / page * page;
        if (view_bytes > model_size - page_offset) view_bytes = model_size - page_offset;
        if (inner + bytes > view_bytes || view_bytes > (uint64_t)impl_->device.maxBufferLength)
            throw std::runtime_error("q27 Metal: model view exceeds Metal buffer limits");

        void* view_base = (void*)(base + page_offset);
        id<MTLBuffer> buffer = [impl_->device newBufferWithBytesNoCopy:view_base
                                                               length:(NSUInteger)view_bytes
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        if (!buffer) throw std::runtime_error("q27 Metal: cannot wrap model mmap");
        return std::shared_ptr<BackendBuffer>(std::make_shared<MetalBuffer>(buffer));
    };

    BackendTensor result;
    result.dtype = tensor.dtype;
    result.rows = tensor.rows();
    result.cols = tensor.cols();
    result.data = wrap(tensor.data, tensor.data_size, result.data_offset);
    result.data_size = tensor.data_size;
    if (tensor.scales_size) {
        result.scales = wrap(tensor.scales, tensor.scales_size, result.scales_offset);
        result.scales_size = tensor.scales_size;
    }
    return result;
}

void MetalBackend::begin_commands() {
    @autoreleasepool { impl_->start_command(true); }
}

void MetalBackend::end_commands() {
    @autoreleasepool { impl_->finish_command("command batch"); }
}

void MetalBackend::abort_commands() noexcept {
    @autoreleasepool {
        if (impl_->encoder) [impl_->encoder endEncoding];
        impl_->encoder = nil;
        impl_->command = nil;
        impl_->batching = false;
    }
}

void MetalBackend::matvec(const BackendTensor& weight, const BackendBuffer& x,
                          BackendBuffer& y) {
    if (!weight.data) throw std::runtime_error("q27 Metal: matvec weight has no data");
    if (!weight.rows || !weight.cols || weight.rows > UINT32_MAX || weight.cols > UINT32_MAX)
        throw std::runtime_error("q27 Metal: unsupported matvec dimensions");
    check_range(x.size(), 0, weight.cols * sizeof(float), "matvec input");
    check_range(y.size(), 0, weight.rows * sizeof(float), "matvec output");

    const MetalBuffer& data = metal_buffer(*weight.data);
    const MetalBuffer& input = metal_buffer(x);
    MetalBuffer& output = metal_buffer(y);
    const uint64_t quant_group = weight.dtype == DType::Q8_G128 ? 128 :
                                 weight.dtype == DType::T2_G128 ? 128 :
                                 weight.dtype == DType::T3_G128 ? 128 :
                                 weight.dtype == DType::Q4_G64 ? 64 : 0;
    if (quant_group && weight.cols % quant_group)
        throw std::runtime_error("q27 Metal: matvec columns do not match quantization group");
    if (weight.cols && weight.rows > UINT64_MAX / 4 / weight.cols)
        throw std::runtime_error("q27 Metal: weight shape overflows byte arithmetic");
    uint64_t data_bytes = weight.rows * weight.cols;
    if (weight.dtype == DType::F32) data_bytes *= 4;
    if (weight.dtype == DType::F16) data_bytes *= 2;
    if (weight.dtype == DType::Q4_G64) data_bytes /= 2;
    if (weight.dtype == DType::T2_G128) data_bytes /= 4;
    if (weight.dtype == DType::T3_G128) data_bytes = weight.rows * (weight.cols / 128) * 26;
    check_range(tensor_limit(data.size(), weight.data_offset, weight.data_size), weight.data_offset, data_bytes, "matvec weight");
    id<MTLComputePipelineState> pipeline = nil;
    const char* label = nullptr;
    switch (weight.dtype) {
        case DType::F32: pipeline = impl_->f32; label = "q27_matvec_f32"; break;
        case DType::F16: pipeline = impl_->f16; label = "q27_matvec_f16"; break;
        case DType::Q8_G128: pipeline = impl_->q8; label = "q27_matvec_q8_g128"; break;
        case DType::Q4_G64: pipeline = impl_->q4; label = "q27_matvec_q4_g64"; break;
        case DType::T2_G128: pipeline = impl_->t2; label = "q27_matvec_t2_g128"; break;
        case DType::T3_G128: pipeline = impl_->t3; label = "q27_matvec_t3_g128"; break;
    }
    const MetalBuffer* quant_scales = nullptr;
    if (quant_group) {
        if (!weight.scales) throw std::runtime_error("q27 Metal: quantized weight has no scales");
        quant_scales = &metal_buffer(*weight.scales);
        const uint64_t group = quant_group;
        const uint64_t scale_bytes = weight.rows * (weight.cols / group) * 2;
        check_range(tensor_limit(quant_scales->size(), weight.scales_offset, weight.scales_size), weight.scales_offset, scale_bytes, "matvec scales");
    }

    MatvecArgs args{(uint32_t)weight.rows, (uint32_t)weight.cols, 8};
    @autoreleasepool {
        bool own_command;
        id<MTLComputeCommandEncoder> encoder = impl_->encoder_for_operation(own_command, label);
        [encoder setComputePipelineState:pipeline];
        if (weight.dtype == DType::F32 || weight.dtype == DType::F16) {
            [encoder setBuffer:data.handle() offset:(NSUInteger)weight.data_offset atIndex:0];
            [encoder setBuffer:input.handle() offset:0 atIndex:1];
            [encoder setBuffer:output.handle() offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        } else {
            [encoder setBuffer:data.handle() offset:(NSUInteger)weight.data_offset atIndex:0];
            [encoder setBuffer:quant_scales->handle() offset:(NSUInteger)weight.scales_offset atIndex:1];
            [encoder setBuffer:input.handle() offset:0 atIndex:2];
            [encoder setBuffer:output.handle() offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        }
        // T2/T3 run 4 rows per simdgroup (32 per threadgroup); others 1 (8).
        const bool ternary = weight.dtype == DType::T2_G128 || weight.dtype == DType::T3_G128;
        const NSUInteger row_groups = ternary
            ? (NSUInteger)(weight.rows + 31) / 32 : (NSUInteger)(weight.rows + 7) / 8;
        [encoder dispatchThreadgroups:MTLSizeMake(row_groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (own_command) impl_->finish_command("matvec");
    }
}

void MetalBackend::matvec_pair(const BackendTensor& a, BackendBuffer& a_out,
                               const BackendTensor& b, BackendBuffer& b_out,
                               const BackendBuffer& x) {
    if(a.dtype!=DType::F16 || b.dtype!=DType::F16 || a.cols!=b.cols) {
        matvec(a,x,a_out); matvec(b,x,b_out); return;
    }
    if(!a.data || !b.data || !a.rows || !b.rows || !a.cols || a.rows>UINT32_MAX ||
       b.rows>UINT32_MAX || a.cols>UINT32_MAX)
        throw std::runtime_error("q27 Metal: invalid fused F16 matvec");
    check_range(x.size(),0,a.cols*4,"fused matvec input");
    check_range(a_out.size(),0,a.rows*4,"fused matvec output A");
    check_range(b_out.size(),0,b.rows*4,"fused matvec output B");
    const MetalBuffer& ad=metal_buffer(*a.data); const MetalBuffer& bd=metal_buffer(*b.data);
    const MetalBuffer& input=metal_buffer(x); MetalBuffer& ao=metal_buffer(a_out); MetalBuffer& bo=metal_buffer(b_out);
    check_range(tensor_limit(ad.size(), a.data_offset, a.data_size), a.data_offset,a.rows*a.cols*2,"fused matvec weight A");
    check_range(tensor_limit(bd.size(), b.data_offset, b.data_size), b.data_offset,b.rows*b.cols*2,"fused matvec weight B");
    MatvecPairArgs args{(uint32_t)a.rows,(uint32_t)b.rows,(uint32_t)a.cols,8};
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_matvec_f16_pair"); [enc setComputePipelineState:impl_->f16_pair];
        [enc setBuffer:ad.handle() offset:(NSUInteger)a.data_offset atIndex:0]; [enc setBuffer:ao.handle() offset:0 atIndex:1];
        [enc setBuffer:bd.handle() offset:(NSUInteger)b.data_offset atIndex:2]; [enc setBuffer:bo.handle() offset:0 atIndex:3];
        [enc setBuffer:input.handle() offset:0 atIndex:4]; [enc setBytes:&args length:sizeof(args) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)std::max(a.rows,b.rows),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(own) impl_->finish_command("fused matvec pair");
    }
}

BackendQuantized MetalBackend::allocate_quantized(uint32_t count) {
    if (!count || count % 32) throw std::runtime_error("q27 Metal: quantized activation count must be a multiple of 32");
    BackendQuantized result; result.count=count;
    result.values=allocate(count); result.scales=allocate((uint64_t)(count/32)*sizeof(float));
    return result;
}

void MetalBackend::quantize(const BackendBuffer& x, BackendQuantized& out) {
    if (!out.count || out.count%32 || !out.values || !out.scales)
        throw std::runtime_error("q27 Metal: invalid quantized activation");
    const MetalBuffer& xb=metal_buffer(x); MetalBuffer& values=metal_buffer(*out.values); MetalBuffer& scales=metal_buffer(*out.scales);
    check_range(xb.size(),0,(uint64_t)out.count*4,"quantize input"); check_range(values.size(),0,out.count,"quantize values");
    check_range(scales.size(),0,(uint64_t)(out.count/32)*4,"quantize scales");
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_quantize_x"); [enc setComputePipelineState:impl_->quantize];
        [enc setBuffer:xb.handle() offset:0 atIndex:0]; [enc setBuffer:values.handle() offset:0 atIndex:1];
        [enc setBuffer:scales.handle() offset:0 atIndex:2]; [enc setBytes:&out.count length:4 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(out.count/32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        if(own) impl_->finish_command("activation quantize");
    }
}

void MetalBackend::matvec_quantized(const BackendTensor& weight,
                                     const BackendQuantized& x, BackendBuffer& y) {
    if ((weight.dtype!=DType::Q4_G64 && weight.dtype!=DType::Q8_G128 &&
         weight.dtype!=DType::T2_G128) || !weight.data || !weight.scales)
        throw std::runtime_error("q27 Metal: quantized matvec requires Q4/Q8/T2 weight");
    const uint64_t group=weight.dtype==DType::Q4_G64?64:128;
    if (!weight.rows || !weight.cols || weight.rows>UINT32_MAX || weight.cols>UINT32_MAX ||
        weight.cols%group)
        throw std::runtime_error("q27 Metal: invalid quantized matvec dimensions");
    if (x.count!=weight.cols || !x.values || !x.scales)
        throw std::runtime_error("q27 Metal: quantized matvec activation mismatch");
    check_range(y.size(),0,weight.rows*4,"quantized matvec output");
    const MetalBuffer& data=metal_buffer(*weight.data); const MetalBuffer& ws=metal_buffer(*weight.scales);
    const MetalBuffer& xv=metal_buffer(*x.values); const MetalBuffer& xs=metal_buffer(*x.scales); MetalBuffer& out=metal_buffer(y);
    const uint64_t divisor=weight.dtype==DType::Q4_G64?2:weight.dtype==DType::T2_G128?4:1;
    const uint64_t data_bytes=weight.rows*weight.cols/divisor;
    check_range(tensor_limit(data.size(), weight.data_offset, weight.data_size), weight.data_offset,data_bytes,"quantized matvec weight");
    check_range(tensor_limit(ws.size(), weight.scales_offset, weight.scales_size), weight.scales_offset,weight.rows*(weight.cols/group)*2,"quantized matvec weight scales");
    check_range(xv.size(),0,x.count,"quantized matvec values"); check_range(xs.size(),0,(uint64_t)(x.count/32)*4,"quantized matvec activation scales");
    MatvecArgs args{(uint32_t)weight.rows,(uint32_t)weight.cols,8};
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own,
            weight.dtype==DType::Q8_G128?"q27_matvec_q8_quantized":
            weight.dtype==DType::T2_G128?"q27_matvec_t2_quantized":"q27_matvec_q4_quantized");
        [enc setComputePipelineState:weight.dtype==DType::Q8_G128?impl_->q8_quantized:
                                     weight.dtype==DType::T2_G128?impl_->t2_quantized:impl_->q4_quantized];
        [enc setBuffer:data.handle() offset:(NSUInteger)weight.data_offset atIndex:0];
        [enc setBuffer:ws.handle() offset:(NSUInteger)weight.scales_offset atIndex:1];
        [enc setBuffer:xv.handle() offset:0 atIndex:2]; [enc setBuffer:xs.handle() offset:0 atIndex:3];
        [enc setBuffer:out.handle() offset:0 atIndex:4]; [enc setBytes:&args length:sizeof(args) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(weight.rows+7)/8,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(own) impl_->finish_command("quantized matvec");
    }
}

// N=2 slot-batched select-form T2 GEMV (multislot Phase 2 probe): the
// float-activation production serial-decode path with two independent
// activation/output buffer pairs. Metal-only surface until the probe
// passes its pre-registered decision line.
void MetalBackend::matvec_x2(const BackendTensor& weight,
                             const BackendBuffer& x_a, const BackendBuffer& x_b,
                             BackendBuffer& y_a, BackendBuffer& y_b) {
    if (weight.dtype!=DType::T2_G128 || !weight.data || !weight.scales)
        throw std::runtime_error("q27 Metal: x2 matvec requires T2 weight");
    if (!weight.rows || !weight.cols || weight.rows>UINT32_MAX || weight.cols>UINT32_MAX ||
        weight.cols%128)
        throw std::runtime_error("q27 Metal: invalid x2 matvec dimensions");
    check_range(x_a.size(),0,weight.cols*sizeof(float),"x2 matvec input a");
    check_range(x_b.size(),0,weight.cols*sizeof(float),"x2 matvec input b");
    check_range(y_a.size(),0,weight.rows*sizeof(float),"x2 matvec output a");
    check_range(y_b.size(),0,weight.rows*sizeof(float),"x2 matvec output b");
    const MetalBuffer& data=metal_buffer(*weight.data); const MetalBuffer& ws=metal_buffer(*weight.scales);
    const MetalBuffer& xa=metal_buffer(x_a); const MetalBuffer& xb=metal_buffer(x_b);
    MetalBuffer& ya=metal_buffer(y_a); MetalBuffer& yb=metal_buffer(y_b);
    check_range(tensor_limit(data.size(), weight.data_offset, weight.data_size), weight.data_offset,weight.rows*weight.cols/4,"x2 matvec weight");
    check_range(tensor_limit(ws.size(), weight.scales_offset, weight.scales_size), weight.scales_offset,weight.rows*(weight.cols/128)*2,"x2 matvec weight scales");
    MatvecArgs args{(uint32_t)weight.rows,(uint32_t)weight.cols,8};
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own,"q27_matvec_t2_g128_x2");
        [enc setComputePipelineState:impl_->t2_x2];
        [enc setBuffer:data.handle() offset:(NSUInteger)weight.data_offset atIndex:0];
        [enc setBuffer:ws.handle() offset:(NSUInteger)weight.scales_offset atIndex:1];
        [enc setBuffer:xa.handle() offset:0 atIndex:2]; [enc setBuffer:xb.handle() offset:0 atIndex:3];
        [enc setBuffer:ya.handle() offset:0 atIndex:4]; [enc setBuffer:yb.handle() offset:0 atIndex:5];
        [enc setBytes:&args length:sizeof(args) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(weight.rows+31)/32,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(own) impl_->finish_command("x2 matvec");
    }
}

// N=2 slot-batched T2 GEMV (multislot Phase 2 probe): x carries two
// activation rows ([2, cols] values, [2, cols/32] scales), out is token-
// major [2, rows] — the matmul_quantized layouts at x_rows=2. Metal-only
// surface (not on the Backend interface) until the probe passes its
// pre-registered decision line (docs/plans/2026-07-16-multislot-phase2-
// probe.md).
void MetalBackend::matvec_quantized_x2(const BackendTensor& weight,
                                       const BackendQuantized& x, BackendBuffer& y) {
    if (weight.dtype!=DType::T2_G128 || !weight.data || !weight.scales)
        throw std::runtime_error("q27 Metal: x2 matvec requires T2 weight");
    if (!weight.rows || !weight.cols || weight.rows>UINT32_MAX || weight.cols>UINT32_MAX ||
        weight.cols%128)
        throw std::runtime_error("q27 Metal: invalid x2 matvec dimensions");
    if ((uint64_t)x.count!=(uint64_t)weight.cols*2 || !x.values || !x.scales)
        throw std::runtime_error("q27 Metal: x2 matvec activation mismatch (needs 2 rows)");
    check_range(y.size(),0,weight.rows*2*4,"x2 matvec output");
    const MetalBuffer& data=metal_buffer(*weight.data); const MetalBuffer& ws=metal_buffer(*weight.scales);
    const MetalBuffer& xv=metal_buffer(*x.values); const MetalBuffer& xs=metal_buffer(*x.scales); MetalBuffer& out=metal_buffer(y);
    check_range(tensor_limit(data.size(), weight.data_offset, weight.data_size), weight.data_offset,weight.rows*weight.cols/4,"x2 matvec weight");
    check_range(tensor_limit(ws.size(), weight.scales_offset, weight.scales_size), weight.scales_offset,weight.rows*(weight.cols/128)*2,"x2 matvec weight scales");
    check_range(xv.size(),0,x.count,"x2 matvec values"); check_range(xs.size(),0,(uint64_t)(x.count/32)*4,"x2 matvec activation scales");
    MatvecArgs args{(uint32_t)weight.rows,(uint32_t)weight.cols,8};
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own,"q27_matvec_t2_quantized_x2");
        [enc setComputePipelineState:impl_->t2_quantized_x2];
        [enc setBuffer:data.handle() offset:(NSUInteger)weight.data_offset atIndex:0];
        [enc setBuffer:ws.handle() offset:(NSUInteger)weight.scales_offset atIndex:1];
        [enc setBuffer:xv.handle() offset:0 atIndex:2]; [enc setBuffer:xs.handle() offset:0 atIndex:3];
        [enc setBuffer:out.handle() offset:0 atIndex:4]; [enc setBytes:&args length:sizeof(args) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(weight.rows+7)/8,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(own) impl_->finish_command("x2 quantized matvec");
    }
}

// Two independent packed-dot dispatches now beat a fused pair kernel: the
// rewritten GEMV is weight-stream-bound, so sharing the (cached) activation
// bytes buys nothing, while the fused kernel's doubled register pressure
// measured 47-55 GB/s against 67-69 GB/s for back-to-back singles (Q4,
// 17408x5120-class shapes, M4). The entry point survives as the engine's
// sibling-projection idiom.
void MetalBackend::matvec_quantized_pair(const BackendTensor& a, BackendBuffer& a_out,
                                         const BackendTensor& b, BackendBuffer& b_out,
                                         const BackendQuantized& x) {
    matvec_quantized(a,x,a_out); matvec_quantized(b,x,b_out);
}

void MetalBackend::matmul_quantized(const BackendTensor& weight,const BackendQuantized& x,
                                    uint32_t x_rows,BackendBuffer& y) {
    if(!impl_->q4_quantized_matmul || !impl_->q8_quantized_matmul || !impl_->t2_quantized_matmul) {
        if(x_rows==1) { matvec_quantized(weight,x,y); return; }
        throw std::runtime_error("q27 Metal: quantized matmul requires Apple GPU family 7 or newer");
    }
    if((weight.dtype!=DType::Q4_G64 && weight.dtype!=DType::Q8_G128 &&
        weight.dtype!=DType::T2_G128) || !weight.data || !weight.scales)
        throw std::runtime_error("q27 Metal: quantized matmul requires Q4/Q8/T2 weight");
    uint64_t group=weight.dtype==DType::Q4_G64?64:128;
    // Wide prefill chunks: the kernels tile tokens 16 per threadgroup on
    // grid.y (docs/plans/2026-07-15-wide-chunks.md phase A). 96 = 8x12.
    if(!x_rows || x_rows>96 || !weight.rows || !weight.cols || weight.rows>UINT32_MAX || weight.cols>UINT32_MAX ||
       weight.cols%group || (uint64_t)weight.cols*x_rows>UINT32_MAX || x.count!=weight.cols*x_rows || !x.values || !x.scales)
        throw std::runtime_error("q27 Metal: invalid quantized matmul dimensions");
    check_range(y.size(),0,weight.rows*x_rows*4,"quantized matmul output");
    const MetalBuffer& data=metal_buffer(*weight.data); const MetalBuffer& ws=metal_buffer(*weight.scales);
    const MetalBuffer& xv=metal_buffer(*x.values); const MetalBuffer& xs=metal_buffer(*x.scales); MetalBuffer& out=metal_buffer(y);
    uint64_t divisor=weight.dtype==DType::Q4_G64?2:weight.dtype==DType::T2_G128?4:1;
    check_range(tensor_limit(data.size(), weight.data_offset, weight.data_size), weight.data_offset,weight.rows*weight.cols/divisor,"quantized matmul weight");
    check_range(tensor_limit(ws.size(), weight.scales_offset, weight.scales_size), weight.scales_offset,weight.rows*(weight.cols/group)*2,"quantized matmul weight scales");
    check_range(xv.size(),0,x.count,"quantized matmul values"); check_range(xs.size(),0,(uint64_t)(x.count/32)*4,"quantized matmul scales");
    MatmulArgs args{(uint32_t)weight.rows,(uint32_t)weight.cols,x_rows,1};
    @autoreleasepool {
        // Q27_METAL_GEMM_HALF=1: half-staging T2 GEMM (A/B lever, see
        // docs/plans/2026-07-15-gemm-half-staging.md).
        const bool t2h = impl_->gemm_half && weight.dtype==DType::T2_G128;
        bool own; auto enc=impl_->encoder_for_operation(own,
            weight.dtype==DType::Q8_G128?"q27_matmul_q8_mm":
            weight.dtype==DType::T2_G128?(t2h?"q27_matmul_t2_mm_h":"q27_matmul_t2_mm"):"q27_matmul_q4_mm");
        [enc setComputePipelineState:weight.dtype==DType::Q8_G128?impl_->q8_quantized_matmul:
                                     weight.dtype==DType::T2_G128?(t2h?impl_->t2_quantized_matmul_h:impl_->t2_quantized_matmul):impl_->q4_quantized_matmul];
        [enc setBuffer:data.handle() offset:(NSUInteger)weight.data_offset atIndex:0]; [enc setBuffer:ws.handle() offset:(NSUInteger)weight.scales_offset atIndex:1];
        [enc setBuffer:xv.handle() offset:0 atIndex:2]; [enc setBuffer:xs.handle() offset:0 atIndex:3]; [enc setBuffer:out.handle() offset:0 atIndex:4];
        [enc setBytes:&args length:sizeof(args) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(weight.rows+31)/32,(NSUInteger)(x_rows+15)/16,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        if(own) impl_->finish_command("quantized simdgroup matmul");
    }
}

// A/B/C MMA roofline probe entries (bench-only, docs/plans/2026-07-16-mma-
// roofline.md): same dispatch grid and MatmulArgs as the production T2
// GEMM. Arm 'b' takes half operands + half weight scales + float
// activation scales; arm 'a' takes only the opaque tile seed. Never
// engine-routed.
void MetalBackend::mma_roofline(char arm, uint32_t rows, uint32_t cols, uint32_t x_rows,
                                const BackendBuffer& w_or_seed, const BackendBuffer* w_scales,
                                const BackendBuffer* x, const BackendBuffer* x_scales,
                                BackendBuffer& y) {
    if (!impl_->mma_roofline_a_p || !impl_->mma_roofline_b_p)
        throw std::runtime_error("q27 Metal: MMA roofline requires Apple GPU family 7 or newer");
    if ((arm != 'a' && arm != 'b' && arm != 'e' && arm != 'x' && arm != 'd' && arm != '2') || !rows || !cols ||
        !x_rows || x_rows > 96 || cols % 128)
        throw std::runtime_error("q27 Metal: invalid MMA roofline arguments");
    // Arm 'd' — lever 1 direct-RHS probe (docs/plans/2026-07-16-lever1-
    // direct-rhs.md): w_or_seed = packed T2 weight bytes, x = int8
    // activation values, x_scales = float per-32 scales. Encodes the RHS
    // pre-pass (int8 -> K-major half, charged to this arm) and the 64x32
    // direct-RHS GEMM.
    if (arm == 'd' || arm == '2') {
        if (!w_scales || !x || !x_scales)
            throw std::runtime_error("q27 Metal: roofline arm d needs scales and activations");
        const MetalBuffer& wb = metal_buffer(w_or_seed);
        const MetalBuffer& ws = metal_buffer(*w_scales);
        const MetalBuffer& xb = metal_buffer(*x);
        const MetalBuffer& xs = metal_buffer(*x_scales);
        MetalBuffer& out = metal_buffer(y);
        check_range(wb.size(), 0, (uint64_t)rows * cols / 4, "roofline d weights");
        check_range(ws.size(), 0, (uint64_t)rows * (cols / 128) * 2, "roofline d weight scales");
        check_range(xb.size(), 0, (uint64_t)x_rows * cols, "roofline d activations");
        check_range(xs.size(), 0, (uint64_t)x_rows * (cols / 32) * 4, "roofline d activation scales");
        check_range(out.size(), 0, (uint64_t)rows * x_rows * 4, "roofline d output");
        const uint32_t tokens_pad = (x_rows + 7) / 8 * 8;
        const uint64_t xt_bytes = (uint64_t)cols * tokens_pad * 2;
        if (!impl_->dr_xt_scratch || impl_->dr_xt_scratch.length < xt_bytes)
            impl_->dr_xt_scratch = [impl_->device newBufferWithLength:(NSUInteger)xt_bytes
                                                              options:MTLResourceStorageModePrivate];
        if (!impl_->dr_xt_scratch) throw std::runtime_error("q27 Metal: dr xT allocation failed");
        MatmulArgs args{rows, cols, x_rows, tokens_pad};
        @autoreleasepool {
            bool own; auto enc = impl_->encoder_for_operation(own, "q27_matmul_t2_mm_dr");
            [enc setComputePipelineState:impl_->x_to_half_t_p];
            [enc setBuffer:xb.handle() offset:0 atIndex:0];
            [enc setBuffer:impl_->dr_xt_scratch offset:0 atIndex:1];
            [enc setBytes:&args length:sizeof(args) atIndex:2];
            const uint64_t pre_threads = (uint64_t)cols * tokens_pad;
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((pre_threads + 255) / 256), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            // Same serial encoder: the GEMM reads the xT the pre-pass wrote.
            [enc setComputePipelineState:arm == '2' ? impl_->mm_dr2_p : impl_->mm_dr_p];
            [enc setBuffer:wb.handle() offset:0 atIndex:0];
            [enc setBuffer:ws.handle() offset:0 atIndex:1];
            [enc setBuffer:impl_->dr_xt_scratch offset:0 atIndex:2];
            [enc setBuffer:xs.handle() offset:0 atIndex:3];
            [enc setBuffer:out.handle() offset:0 atIndex:4];
            [enc setBytes:&args length:sizeof(args) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(rows + 63) / 64,
                                                  (NSUInteger)(x_rows + (arm == '2' ? 15 : 31)) /
                                                      (arm == '2' ? 16 : 32), 1)
                threadsPerThreadgroup:MTLSizeMake(arm == '2' ? 128 : 256, 1, 1)];
            if (own) impl_->finish_command("mm direct-rhs probe");
        }
        return;
    }
    const MetalBuffer& wb = metal_buffer(w_or_seed);
    MetalBuffer& out = metal_buffer(y);
    check_range(out.size(), 0, (uint64_t)rows * x_rows * 4, "roofline output");
    MatmulArgs args{rows, cols, x_rows, 1};
    @autoreleasepool {
        if (arm == 'a') {
            check_range(wb.size(), 0, (32 * 64 + 64 * 16) * 2, "roofline seed");
            bool own; auto enc = impl_->encoder_for_operation(own, "q27_mma_roofline_a");
            [enc setComputePipelineState:impl_->mma_roofline_a_p];
            [enc setBuffer:wb.handle() offset:0 atIndex:0];
            [enc setBuffer:out.handle() offset:0 atIndex:1];
            [enc setBytes:&args length:sizeof(args) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(rows + 31) / 32, (NSUInteger)(x_rows + 15) / 16, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            if (own) impl_->finish_command("mma roofline a");
            return;
        }
        if (!w_scales || !x || !x_scales)
            throw std::runtime_error("q27 Metal: roofline arm b needs scales and activations");
        const MetalBuffer& ws = metal_buffer(*w_scales);
        const MetalBuffer& xb = metal_buffer(*x);
        const MetalBuffer& xs = metal_buffer(*x_scales);
        // 'e' (equal-traffic) reads C's device byte volume: cols/8 halves per
        // weight row, cols/2 halves per activation row. 'x' (pre-converted
        // activations) reads C's packed T2 weight bytes and full half
        // activations.
        const uint64_t wbytes = arm == 'e' ? (uint64_t)rows * (cols / 8) * 2
                              : arm == 'x' ? (uint64_t)rows * cols / 4
                                           : (uint64_t)rows * cols * 2;
        const uint64_t xdiv = arm == 'e' ? 2 : 1;
        check_range(wb.size(), 0, wbytes, "roofline b weights");
        check_range(ws.size(), 0, (uint64_t)rows * (cols / 128) * 2, "roofline b weight scales");
        check_range(xb.size(), 0, (uint64_t)x_rows * (cols / xdiv) * 2, "roofline b activations");
        check_range(xs.size(), 0, (uint64_t)x_rows * (cols / 32) * 4, "roofline b activation scales");
        bool own; auto enc = impl_->encoder_for_operation(own,
            arm == 'e' ? "q27_mma_roofline_b_eq" :
            arm == 'x' ? "q27_mma_roofline_cx" : "q27_mma_roofline_b");
        [enc setComputePipelineState:arm == 'e' ? impl_->mma_roofline_b_eq_p :
                                     arm == 'x' ? impl_->mma_roofline_cx_p
                                                : impl_->mma_roofline_b_p];
        [enc setBuffer:wb.handle() offset:0 atIndex:0];
        [enc setBuffer:ws.handle() offset:0 atIndex:1];
        [enc setBuffer:xb.handle() offset:0 atIndex:2];
        [enc setBuffer:xs.handle() offset:0 atIndex:3];
        [enc setBuffer:out.handle() offset:0 atIndex:4];
        [enc setBytes:&args length:sizeof(args) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(rows + 31) / 32, (NSUInteger)(x_rows + 15) / 16, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        if (own) impl_->finish_command("mma roofline b");
    }
}

void MetalBackend::embedding_q8(const BackendTensor& weight, uint32_t token,
                                 BackendBuffer& out) {
    const bool t2 = weight.dtype == DType::T2_G128;
    if ((weight.dtype != DType::Q8_G128 && !t2) || !weight.data || !weight.scales ||
        token >= weight.rows ||
        !weight.rows || !weight.cols || weight.rows > UINT32_MAX || weight.cols > UINT32_MAX ||
        weight.cols % 128)
        throw std::runtime_error("q27 Metal: invalid embedding tensor/token");
    check_range(out.size(), 0, weight.cols * 4, "embedding output");
    const MetalBuffer& data = metal_buffer(*weight.data);
    const MetalBuffer& scales = metal_buffer(*weight.scales);
    check_range(tensor_limit(data.size(), weight.data_offset, weight.data_size), weight.data_offset,
                weight.rows * weight.cols / (t2 ? 4 : 1), "embedding weight");
    check_range(tensor_limit(scales.size(), weight.scales_offset, weight.scales_size), weight.scales_offset,
                weight.rows * (weight.cols / 128) * 2, "embedding scales");
    MetalBuffer& output = metal_buffer(out);
    @autoreleasepool {
        bool own; id<MTLComputeCommandEncoder> enc = impl_->encoder_for_operation(own,
            t2 ? "q27_embedding_t2" : "q27_embedding_q8");
        [enc setComputePipelineState:t2 ? impl_->embedding_t2 : impl_->embedding];
        [enc setBuffer:data.handle() offset:(NSUInteger)weight.data_offset atIndex:0];
        [enc setBuffer:scales.handle() offset:(NSUInteger)weight.scales_offset atIndex:1];
        [enc setBuffer:output.handle() offset:0 atIndex:2];
        [enc setBytes:&token length:sizeof(token) atIndex:3];
        uint32_t cols = (uint32_t)weight.cols;
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(cols, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (own) impl_->finish_command("embedding");
    }
}

// GPU-resident greedy decode: same row lookup, but the token id lives in a
// device buffer written by the previous step's argmax — chained steps need
// no CPU sync. The id cannot be range-checked host-side; argmax only writes
// ids < vocab, and the kernels never index past the id row.
void MetalBackend::embedding_from_device(const BackendTensor& weight, const BackendBuffer& token,
                                         BackendBuffer& out) {
    const bool t2 = weight.dtype == DType::T2_G128;
    if ((weight.dtype != DType::Q8_G128 && !t2) || !weight.data || !weight.scales ||
        !weight.rows || !weight.cols || weight.rows > UINT32_MAX || weight.cols > UINT32_MAX ||
        weight.cols % 128)
        throw std::runtime_error("q27 Metal: invalid embedding tensor");
    check_range(out.size(), 0, weight.cols * 4, "embedding output");
    const MetalBuffer& tokb = metal_buffer(token);
    check_range(tokb.size(), 0, 4, "embedding token id");
    const MetalBuffer& data = metal_buffer(*weight.data);
    const MetalBuffer& scales = metal_buffer(*weight.scales);
    check_range(tensor_limit(data.size(), weight.data_offset, weight.data_size), weight.data_offset,
                weight.rows * weight.cols / (t2 ? 4 : 1), "embedding weight");
    check_range(tensor_limit(scales.size(), weight.scales_offset, weight.scales_size), weight.scales_offset,
                weight.rows * (weight.cols / 128) * 2, "embedding scales");
    MetalBuffer& output = metal_buffer(out);
    @autoreleasepool {
        bool own; id<MTLComputeCommandEncoder> enc = impl_->encoder_for_operation(own,
            t2 ? "q27_embedding_t2_dev" : "q27_embedding_q8_dev");
        [enc setComputePipelineState:t2 ? impl_->embedding_t2_dev : impl_->embedding_dev];
        [enc setBuffer:data.handle() offset:(NSUInteger)weight.data_offset atIndex:0];
        [enc setBuffer:scales.handle() offset:(NSUInteger)weight.scales_offset atIndex:1];
        [enc setBuffer:output.handle() offset:0 atIndex:2];
        [enc setBuffer:tokb.handle() offset:0 atIndex:3];
        uint32_t cols = (uint32_t)weight.cols;
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(cols, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (own) impl_->finish_command("embedding (device token)");
    }
}

void MetalBackend::rmsnorm(const BackendBuffer& x, const BackendTensor& weight,
                            BackendBuffer& out, uint32_t n, float eps) {
    const MetalBuffer& w = tensor_data(weight, DType::F32, "rmsnorm");
    const MetalBuffer& input = metal_buffer(x); MetalBuffer& output = metal_buffer(out);
    check_range(input.size(), 0, (uint64_t)n * 4, "rmsnorm input");
    check_range(output.size(), 0, (uint64_t)n * 4, "rmsnorm output");
    check_range(tensor_limit(w.size(), weight.data_offset, weight.data_size), weight.data_offset, (uint64_t)n * 4, "rmsnorm weight");
    VectorArgs args{n, 8, eps};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_rmsnorm");
        [enc setComputePipelineState:impl_->rms];
        [enc setBuffer:input.handle() offset:0 atIndex:0];
        [enc setBuffer:w.handle() offset:(NSUInteger)weight.data_offset atIndex:1];
        [enc setBuffer:output.handle() offset:0 atIndex:2];
        [enc setBytes:&args length:sizeof(args) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if (own) impl_->finish_command("rmsnorm");
    }
}

void MetalBackend::rmsnorm_quantized(const BackendBuffer& x,const BackendTensor& weight,
                                     BackendBuffer& out,uint32_t n,float eps,
                                     BackendQuantized& quantized) {
    if(!n || n%32 || quantized.count!=n || !quantized.values || !quantized.scales)
        throw std::runtime_error("q27 Metal: invalid fused rmsnorm quantization");
    const MetalBuffer& w=tensor_data(weight,DType::F32,"fused rmsnorm");
    const MetalBuffer& input=metal_buffer(x); MetalBuffer& output=metal_buffer(out);
    MetalBuffer& values=metal_buffer(*quantized.values); MetalBuffer& scales=metal_buffer(*quantized.scales);
    check_range(input.size(),0,(uint64_t)n*4,"fused rmsnorm input"); check_range(output.size(),0,(uint64_t)n*4,"fused rmsnorm output");
    check_range(tensor_limit(w.size(), weight.data_offset, weight.data_size), weight.data_offset,(uint64_t)n*4,"fused rmsnorm weight"); check_range(values.size(),0,n,"fused rmsnorm values");
    check_range(scales.size(),0,(uint64_t)(n/32)*4,"fused rmsnorm scales"); VectorArgs args{n,8,eps};
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_rmsnorm_quantized"); [enc setComputePipelineState:impl_->rms_quantized];
        [enc setBuffer:input.handle() offset:0 atIndex:0]; [enc setBuffer:w.handle() offset:(NSUInteger)weight.data_offset atIndex:1];
        [enc setBuffer:output.handle() offset:0 atIndex:2]; [enc setBuffer:values.handle() offset:0 atIndex:3];
        [enc setBuffer:scales.handle() offset:0 atIndex:4]; [enc setBytes:&args length:sizeof(args) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(own) impl_->finish_command("fused rmsnorm quantize");
    }
}

void MetalBackend::rmsnorm_heads(BackendBuffer& x, const BackendTensor& weight,
                                  uint32_t heads, uint32_t head_dim, uint32_t stride,
                                  float eps) {
    const MetalBuffer& w = tensor_data(weight, DType::F32, "head rmsnorm");
    MetalBuffer& input = metal_buffer(x);
    check_range(input.size(), 0, ((uint64_t)(heads - 1) * stride + head_dim) * 4, "head rmsnorm input");
    check_range(tensor_limit(w.size(), weight.data_offset, weight.data_size), weight.data_offset, (uint64_t)head_dim * 4, "head rmsnorm weight");
    HeadArgs args{heads, head_dim, stride, 8, eps};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_rmsnorm_heads");
        [enc setComputePipelineState:impl_->rms_heads];
        [enc setBuffer:input.handle() offset:0 atIndex:0];
        [enc setBuffer:w.handle() offset:(NSUInteger)weight.data_offset atIndex:1];
        [enc setBytes:&args length:sizeof(args) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if (own) impl_->finish_command("head rmsnorm");
    }
}

void MetalBackend::l2norm_heads(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                                 float eps) {
    MetalBuffer& input = metal_buffer(x);
    check_range(input.size(), 0, (uint64_t)heads * head_dim * 4, "l2norm input");
    HeadArgs args{heads, head_dim, head_dim, 8, eps};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_l2norm_heads");
        [enc setComputePipelineState:impl_->l2_heads];
        [enc setBuffer:input.handle() offset:0 atIndex:0];
        [enc setBytes:&args length:sizeof(args) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if (own) impl_->finish_command("l2norm");
    }
}

void MetalBackend::silu_mul(const BackendBuffer& gate, const BackendBuffer& up,
                             BackendBuffer& out, uint32_t n) {
    const MetalBuffer& g = metal_buffer(gate); const MetalBuffer& u = metal_buffer(up);
    MetalBuffer& o = metal_buffer(out);
    check_range(g.size(),0,(uint64_t)n*4,"silu gate"); check_range(u.size(),0,(uint64_t)n*4,"silu up");
    check_range(o.size(),0,(uint64_t)n*4,"silu output");
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_silu_mul"); [enc setComputePipelineState:impl_->silu];
        [enc setBuffer:g.handle() offset:0 atIndex:0]; [enc setBuffer:u.handle() offset:0 atIndex:1];
        [enc setBuffer:o.handle() offset:0 atIndex:2]; [enc setBytes:&n length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(own) impl_->finish_command("silu multiply");
    }
}

void MetalBackend::add_inplace(BackendBuffer& x, const BackendBuffer& y, uint32_t n) {
    MetalBuffer& a=metal_buffer(x); const MetalBuffer& b=metal_buffer(y);
    check_range(a.size(),0,(uint64_t)n*4,"add x"); check_range(b.size(),0,(uint64_t)n*4,"add y");
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_add_inplace"); [enc setComputePipelineState:impl_->add];
        [enc setBuffer:a.handle() offset:0 atIndex:0]; [enc setBuffer:b.handle() offset:0 atIndex:1];
        [enc setBytes:&n length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(own) impl_->finish_command("residual add");
    }
}

void MetalBackend::concat(const BackendBuffer& a, uint32_t a_count,
                           const BackendBuffer& b, uint32_t b_count,
                           BackendBuffer& out) {
    const MetalBuffer& ab=metal_buffer(a); const MetalBuffer& bb=metal_buffer(b); MetalBuffer& ob=metal_buffer(out);
    check_range(ab.size(),0,(uint64_t)a_count*4,"concat a"); check_range(bb.size(),0,(uint64_t)b_count*4,"concat b");
    check_range(ob.size(),0,(uint64_t)(a_count+b_count)*4,"concat output"); ConcatArgs args{a_count,b_count};
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_concat"); [enc setComputePipelineState:impl_->concat];
        [enc setBuffer:ab.handle() offset:0 atIndex:0]; [enc setBuffer:bb.handle() offset:0 atIndex:1]; [enc setBuffer:ob.handle() offset:0 atIndex:2];
        [enc setBytes:&args length:sizeof(args) atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)a_count+b_count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(own) impl_->finish_command("concat");
    }
}

void MetalBackend::sigmoid_gate_mul(BackendBuffer& out, const BackendBuffer& qg,
                                     uint32_t heads, uint32_t head_dim) {
    MetalBuffer& o=metal_buffer(out); const MetalBuffer& gates=metal_buffer(qg);
    const uint32_t n=heads*head_dim; GateArgs args{heads,head_dim};
    check_range(o.size(),0,(uint64_t)n*4,"sigmoid output"); check_range(gates.size(),0,(uint64_t)n*2*4,"sigmoid gates");
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_sigmoid_gate_mul"); [enc setComputePipelineState:impl_->sigmoid_gate];
        [enc setBuffer:o.handle() offset:0 atIndex:0]; [enc setBuffer:gates.handle() offset:0 atIndex:1];
        [enc setBytes:&args length:sizeof(args) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(own) impl_->finish_command("sigmoid gate");
    }
}

void MetalBackend::rope_neox(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                              uint32_t n_rot, uint32_t stride, uint32_t position,
                              float freq_base) {
    if (!n_rot || n_rot > head_dim || (n_rot & 1)) throw std::runtime_error("q27 Metal: invalid RoPE dimensions");
    MetalBuffer& input=metal_buffer(x); RopeArgs args{heads,head_dim,n_rot,stride,position,freq_base};
    check_range(input.size(),0,((uint64_t)(heads-1)*stride+head_dim)*4,"rope input");
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_rope_neox"); [enc setComputePipelineState:impl_->rope];
        [enc setBuffer:input.handle() offset:0 atIndex:0]; [enc setBytes:&args length:sizeof(args) atIndex:1];
        [enc dispatchThreads:MTLSizeMake(n_rot/2,heads,1) threadsPerThreadgroup:MTLSizeMake(n_rot/2,1,1)];
        if(own) impl_->finish_command("rope");
    }
}

void MetalBackend::argmax(const BackendBuffer& x, uint32_t n, BackendBuffer& out_index) {
    const MetalBuffer& input=metal_buffer(x); MetalBuffer& output=metal_buffer(out_index);
    check_range(input.size(),0,(uint64_t)n*4,"argmax input"); check_range(output.size(),0,4,"argmax output");
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_argmax"); [enc setComputePipelineState:impl_->argmax];
        [enc setBuffer:input.handle() offset:0 atIndex:0]; [enc setBuffer:output.handle() offset:0 atIndex:1];
        [enc setBytes:&n length:4 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(own) impl_->finish_command("argmax");
    }
}

void MetalBackend::topk(const BackendBuffer& x, uint32_t n, uint32_t k,
                        BackendBuffer& values, BackendBuffer& indices, BackendBuffer& count) {
    const MetalBuffer& input=metal_buffer(x);
    MetalBuffer& vb=metal_buffer(values); MetalBuffer& ib=metal_buffer(indices); MetalBuffer& cb=metal_buffer(count);
    if (!n || !k || k > 256) throw std::runtime_error("q27 Metal: top-k requires 1..256 candidates");
    check_range(input.size(),0,(uint64_t)n*4,"top-k input"); check_range(cb.size(),0,4,"top-k count");
    const uint64_t capacity=std::min(vb.size(),ib.size())/4;
    if (capacity < 2*(uint64_t)k) throw std::runtime_error("q27 Metal: top-k output capacity below 2k");
    if (impl_->batching) throw std::runtime_error("q27 Metal: top-k requires its own command");
    std::memset(cb.handle().contents,0,4);
    TopkArgs args{n,k,(uint32_t)std::min<uint64_t>(capacity,UINT32_MAX)};
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_topk_logits"); [enc setComputePipelineState:impl_->topk_logits_p];
        [enc setBuffer:input.handle() offset:0 atIndex:0]; [enc setBuffer:vb.handle() offset:0 atIndex:1];
        [enc setBuffer:ib.handle() offset:0 atIndex:2]; [enc setBuffer:cb.handle() offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
        if(own) impl_->finish_command("top-k");
    }
}

void MetalBackend::mask_logits(BackendBuffer& logits, const BackendBuffer& masks,
                                uint64_t mask_offset, uint32_t n) {
    MetalBuffer& lb = metal_buffer(logits);
    const MetalBuffer& mb = metal_buffer(masks);
    if (!n) throw std::runtime_error("q27 Metal: empty mask_logits");
    check_range(lb.size(), 0, (uint64_t)n * 4, "mask_logits logits");
    check_range(mb.size(), mask_offset, ((uint64_t)n + 31) / 32 * 4, "mask_logits mask");
    if (mask_offset % 4) throw std::runtime_error("q27 Metal: mask offset must be word-aligned");
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_mask_logits");
        [enc setComputePipelineState:impl_->mask_logits_p];
        [enc setBuffer:lb.handle() offset:0 atIndex:0];
        [enc setBuffer:mb.handle() offset:(NSUInteger)mask_offset atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (own) impl_->finish_command("mask logits");
    }
}

void MetalBackend::kv_store_f16(const BackendBuffer& k, const BackendBuffer& v,
                                 BackendBuffer& k_cache, BackendBuffer& v_cache,
                                 uint32_t position, uint32_t row_length) {
    const MetalBuffer& kb=metal_buffer(k); const MetalBuffer& vb=metal_buffer(v);
    MetalBuffer& kc=metal_buffer(k_cache); MetalBuffer& vc=metal_buffer(v_cache);
    check_range(kb.size(),0,(uint64_t)row_length*4,"K row"); check_range(vb.size(),0,(uint64_t)row_length*4,"V row");
    check_range(kc.size(),(uint64_t)position*row_length*2,(uint64_t)row_length*2,"K cache");
    check_range(vc.size(),(uint64_t)position*row_length*2,(uint64_t)row_length*2,"V cache");
    KvStoreArgs args{position,row_length};
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_kv_store_f16"); [enc setComputePipelineState:impl_->kv_store];
        [enc setBuffer:kb.handle() offset:0 atIndex:0]; [enc setBuffer:vb.handle() offset:0 atIndex:1];
        [enc setBuffer:kc.handle() offset:0 atIndex:2]; [enc setBuffer:vc.handle() offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(row_length,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(own) impl_->finish_command("KV store");
    }
}

void MetalBackend::turbo_wht(BackendBuffer& x, uint32_t heads, uint32_t stride,
                              bool inverse) {
    if (!heads || stride < 256) throw std::runtime_error("q27 Metal: invalid turbo3 WHT dimensions");
    MetalBuffer& xb = metal_buffer(x);
    check_range(xb.size(), 0, ((uint64_t)(heads - 1) * stride + 256) * 4, "turbo3 WHT");
    TurboWhtArgs args{heads, stride, inverse ? 1u : 0u};
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_turbo_wht"); [enc setComputePipelineState:impl_->turbo_wht];
        [enc setBuffer:xb.handle() offset:0 atIndex:0]; [enc setBytes:&args length:sizeof(args) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)heads*2,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        if(own) impl_->finish_command("turbo3 WHT");
    }
}

void MetalBackend::kv_store_turbo3(const BackendBuffer& k, const BackendBuffer& v,
                                    BackendBuffer& k_cache, BackendBuffer& v_cache,
                                    uint32_t position, uint32_t kv_heads) {
    if (!kv_heads) throw std::runtime_error("q27 Metal: invalid turbo3 KV dimensions");
    const MetalBuffer& kb=metal_buffer(k); const MetalBuffer& vb=metal_buffer(v);
    MetalBuffer& kc=metal_buffer(k_cache); MetalBuffer& vc=metal_buffer(v_cache);
    const uint64_t row_bytes=(uint64_t)kv_heads*2*50;
    check_range(kb.size(),0,(uint64_t)kv_heads*256*4,"turbo3 K row");
    check_range(vb.size(),0,(uint64_t)kv_heads*256*4,"turbo3 V row");
    check_range(kc.size(),(uint64_t)position*row_bytes,row_bytes,"turbo3 K cache");
    check_range(vc.size(),(uint64_t)position*row_bytes,row_bytes,"turbo3 V cache");
    TurboStoreArgs args{position,kv_heads};
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_kv_store_turbo3"); [enc setComputePipelineState:impl_->kv_store_turbo3];
        [enc setBuffer:kb.handle() offset:0 atIndex:0]; [enc setBuffer:vb.handle() offset:0 atIndex:1];
        [enc setBuffer:kc.handle() offset:0 atIndex:2]; [enc setBuffer:vc.handle() offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)kv_heads*2,2,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        if(own) impl_->finish_command("turbo3 KV store");
    }
}

void MetalBackend::attention_turbo3(const BackendBuffer& q, uint32_t q_stride,
                                     const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                                     BackendBuffer& out, uint32_t seq_len,
                                     uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                                     float scale) {
    if (!seq_len || !kv_heads || q_heads%kv_heads || head_dim != 256)
        throw std::runtime_error("q27 Metal: invalid turbo3 attention dimensions");
    const MetalBuffer& qb=metal_buffer(q); const MetalBuffer& kc=metal_buffer(k_cache); const MetalBuffer& vc=metal_buffer(v_cache);
    MetalBuffer& output=metal_buffer(out);
    check_range(qb.size(),0,((uint64_t)(q_heads-1)*q_stride+head_dim)*4,"turbo3 attention Q");
    const uint64_t cache_bytes=(uint64_t)seq_len*kv_heads*2*50;
    check_range(kc.size(),0,cache_bytes,"turbo3 K cache"); check_range(vc.size(),0,cache_bytes,"turbo3 V cache");
    check_range(output.size(),0,(uint64_t)q_heads*head_dim*4,"turbo3 attention output");
    const uint32_t gqa=q_heads/kv_heads;
    if (impl_->gqa_threshold && seq_len >= impl_->gqa_threshold && gqa >= 2 && gqa <= 8) {
        impl_->attention_gqa_dispatch(true,qb,q_stride,kc,vc,output,seq_len,q_heads,kv_heads,head_dim,scale);
        return;
    }
    AttentionArgs args{q_stride,seq_len,q_heads,kv_heads,head_dim,scale};
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_attention_turbo3"); [enc setComputePipelineState:impl_->attention_turbo3];
        [enc setBuffer:qb.handle() offset:0 atIndex:0]; [enc setBuffer:kc.handle() offset:0 atIndex:1]; [enc setBuffer:vc.handle() offset:0 atIndex:2];
        [enc setBuffer:output.handle() offset:0 atIndex:3]; [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(q_heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(own) impl_->finish_command("turbo3 attention");
    }
}

// Phase-0 R1 probe: same dispatch shape as the GQA decode path, head-major
// cache addressing. The merge kernel is reused unchanged (partials layout is
// identical); its AttentionGqaArgs is built alongside the probe args.
void MetalBackend::attention_turbo3_gqa_headmajor(const BackendBuffer& q, uint32_t q_stride,
                                                  const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                                                  BackendBuffer& out, uint32_t seq_len, uint32_t seq_cap,
                                                  uint32_t q_heads, uint32_t kv_heads,
                                                  uint32_t head_dim, float scale) {
    const uint32_t gqa = kv_heads ? q_heads / kv_heads : 0;
    if (!seq_len || seq_len > seq_cap || !kv_heads || q_heads % kv_heads ||
        head_dim != 256 || gqa < 2 || gqa > 8)
        throw std::runtime_error("q27 Metal: invalid head-major probe dimensions");
    const MetalBuffer& qb = metal_buffer(q);
    const MetalBuffer& kc = metal_buffer(k_cache);
    const MetalBuffer& vc = metal_buffer(v_cache);
    MetalBuffer& output = metal_buffer(out);
    check_range(qb.size(), 0, ((uint64_t)(q_heads - 1) * q_stride + head_dim) * 4, "hm probe Q");
    const uint64_t cache_bytes = (uint64_t)seq_cap * kv_heads * 100;
    check_range(kc.size(), 0, cache_bytes, "hm probe K cache");
    check_range(vc.size(), 0, cache_bytes, "hm probe V cache");
    check_range(output.size(), 0, (uint64_t)q_heads * head_dim * 4, "hm probe output");
    const uint32_t block = impl_->gqa_block;
    const uint32_t n_blocks = 1 + (seq_len - 1) / block;
    const uint64_t partial_bytes = (uint64_t)q_heads * n_blocks * 258 * 4;
    if (!impl_->gqa_partials || impl_->gqa_partials.length < partial_bytes)
        impl_->gqa_partials = [impl_->device newBufferWithLength:(NSUInteger)partial_bytes
                                                         options:MTLResourceStorageModePrivate];
    if (!impl_->gqa_partials) throw std::runtime_error("q27 Metal: GQA partial allocation failed");
    struct HmArgs { uint32_t q_stride, seq_len, seq_cap, q_heads, kv_heads, head_dim, block, n_blocks; float scale; };
    HmArgs args{q_stride, seq_len, seq_cap, q_heads, kv_heads, head_dim, block, n_blocks, scale};
    AttentionGqaArgs margs{q_stride, seq_len, q_heads, kv_heads, head_dim, block, n_blocks, scale};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_attention_turbo3_gqa_hm");
        [enc setComputePipelineState:impl_->attention_turbo3_gqa_hm_p];
        [enc setBuffer:qb.handle() offset:0 atIndex:0];
        [enc setBuffer:kc.handle() offset:0 atIndex:1];
        [enc setBuffer:vc.handle() offset:0 atIndex:2];
        [enc setBuffer:impl_->gqa_partials offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(kv_heads, n_blocks, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)gqa * 32, 1, 1)];
        [enc setComputePipelineState:impl_->attention_gqa_merge_p];
        [enc setBuffer:impl_->gqa_partials offset:0 atIndex:0];
        [enc setBuffer:output.handle() offset:0 atIndex:1];
        [enc setBytes:&margs length:sizeof(margs) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(q_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        if (own) impl_->finish_command("hm probe attention");
    }
}

// Phase-0 R1b probe: token-tiled causal GQA at tile 2 or 4. Interleaved
// production cache layout; merge kernel reused unchanged.
// R3 probe entry (bench-only): barrier-free direct-read block-partial causal
// GQA, token factor 2, with an explicit block-size override — the sweep is
// the experiment (docs/plans/2026-07-16-r3-barrier-free-attention.md). At
// block == gqa_block the output is bit-identical to the t2 route.
void MetalBackend::attention_turbo3_causal_gqa_bf(const BackendBuffer& q, uint32_t q_stride,
                                                  uint32_t q_row_stride,
                                                  const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                                                  BackendBuffer& out, uint32_t base_len,
                                                  uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                                                  uint32_t tokens, uint32_t block, float scale) {
    const uint32_t gqa = kv_heads ? q_heads / kv_heads : 0;
    if (!base_len || !tokens || !kv_heads || q_heads % kv_heads || head_dim != 256 ||
        gqa < 2 || gqa > 8 || !block || block % 8 ||
        base_len > UINT32_MAX - tokens)
        throw std::runtime_error("q27 Metal: invalid bf probe dimensions");
    const MetalBuffer& qb = metal_buffer(q);
    const MetalBuffer& kc = metal_buffer(k_cache);
    const MetalBuffer& vc = metal_buffer(v_cache);
    MetalBuffer& output = metal_buffer(out);
    const uint32_t max_seq = base_len + tokens - 1;
    check_range(qb.size(), 0,
                ((uint64_t)(tokens - 1) * q_row_stride + (uint64_t)(q_heads - 1) * q_stride + head_dim) * 4,
                "bf probe Q");
    const uint64_t cache_bytes = (uint64_t)max_seq * kv_heads * 2 * 50;
    check_range(kc.size(), 0, cache_bytes, "bf probe K cache");
    check_range(vc.size(), 0, cache_bytes, "bf probe V cache");
    check_range(output.size(), 0, (uint64_t)tokens * q_heads * head_dim * 4, "bf probe output");
    const uint32_t n_blocks_max = 1 + (max_seq - 1) / block;
    const uint64_t partial_bytes = (uint64_t)tokens * q_heads * n_blocks_max * 258 * 4;
    if (!impl_->gqa_partials || impl_->gqa_partials.length < partial_bytes)
        impl_->gqa_partials = [impl_->device newBufferWithLength:(NSUInteger)partial_bytes
                                                         options:MTLResourceStorageModePrivate];
    if (!impl_->gqa_partials) throw std::runtime_error("q27 Metal: GQA partial allocation failed");
    AttentionGqaCausalArgs args{q_stride, q_row_stride, base_len, q_heads, kv_heads,
                                head_dim, block, n_blocks_max, tokens, scale};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_attention_turbo3_causal_gqa_bf2");
        [enc setComputePipelineState:impl_->attention_turbo3_causal_gqa_bf2_p];
        [enc setBuffer:qb.handle() offset:0 atIndex:0];
        [enc setBuffer:kc.handle() offset:0 atIndex:1];
        [enc setBuffer:vc.handle() offset:0 atIndex:2];
        [enc setBuffer:impl_->gqa_partials offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(kv_heads, n_blocks_max, (tokens + 1) / 2)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)gqa * 32, 1, 1)];
        [enc setComputePipelineState:impl_->attention_gqa_merge_rows_p];
        [enc setBuffer:impl_->gqa_partials offset:0 atIndex:0];
        [enc setBuffer:output.handle() offset:0 atIndex:1];
        [enc setBytes:&args length:sizeof(args) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(q_heads, tokens, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        if (own) impl_->finish_command("bf probe attention");
    }
}

void MetalBackend::attention_turbo3_causal_gqa_tiled(const BackendBuffer& q, uint32_t q_stride,
                                                     uint32_t q_row_stride,
                                                     const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                                                     BackendBuffer& out, uint32_t base_len,
                                                     uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                                                     uint32_t tokens, uint32_t tile, float scale) {
    const uint32_t gqa = kv_heads ? q_heads / kv_heads : 0;
    if (!base_len || !tokens || !kv_heads || q_heads % kv_heads || head_dim != 256 ||
        gqa < 2 || gqa > 8 || (tile != 2 && tile != 4) ||
        base_len > UINT32_MAX - tokens)
        throw std::runtime_error("q27 Metal: invalid tiled probe dimensions");
    const MetalBuffer& qb = metal_buffer(q);
    const MetalBuffer& kc = metal_buffer(k_cache);
    const MetalBuffer& vc = metal_buffer(v_cache);
    MetalBuffer& output = metal_buffer(out);
    const uint32_t max_seq = base_len + tokens - 1;
    check_range(qb.size(), 0,
                ((uint64_t)(tokens - 1) * q_row_stride + (uint64_t)(q_heads - 1) * q_stride + head_dim) * 4,
                "tiled probe Q");
    const uint64_t cache_bytes = (uint64_t)max_seq * kv_heads * 2 * 50;
    check_range(kc.size(), 0, cache_bytes, "tiled probe K cache");
    check_range(vc.size(), 0, cache_bytes, "tiled probe V cache");
    check_range(output.size(), 0, (uint64_t)tokens * q_heads * head_dim * 4, "tiled probe output");
    const uint32_t block = impl_->gqa_block;
    const uint32_t n_blocks_max = 1 + (max_seq - 1) / block;
    const uint64_t partial_bytes = (uint64_t)tokens * q_heads * n_blocks_max * 258 * 4;
    if (!impl_->gqa_partials || impl_->gqa_partials.length < partial_bytes)
        impl_->gqa_partials = [impl_->device newBufferWithLength:(NSUInteger)partial_bytes
                                                         options:MTLResourceStorageModePrivate];
    if (!impl_->gqa_partials) throw std::runtime_error("q27 Metal: GQA partial allocation failed");
    AttentionGqaCausalArgs args{q_stride, q_row_stride, base_len, q_heads, kv_heads,
                                head_dim, block, n_blocks_max, tokens, scale};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own,
            tile == 2 ? "q27_attention_turbo3_causal_gqa_t2" : "q27_attention_turbo3_causal_gqa_t4");
        [enc setComputePipelineState:tile == 2 ? impl_->attention_turbo3_causal_gqa_t2_p
                                               : impl_->attention_turbo3_causal_gqa_t4_p];
        [enc setBuffer:qb.handle() offset:0 atIndex:0];
        [enc setBuffer:kc.handle() offset:0 atIndex:1];
        [enc setBuffer:vc.handle() offset:0 atIndex:2];
        [enc setBuffer:impl_->gqa_partials offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(kv_heads, n_blocks_max, (tokens + tile - 1) / tile)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)gqa * 32, 1, 1)];
        [enc setComputePipelineState:impl_->attention_gqa_merge_rows_p];
        [enc setBuffer:impl_->gqa_partials offset:0 atIndex:0];
        [enc setBuffer:output.handle() offset:0 atIndex:1];
        [enc setBytes:&args length:sizeof(args) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(q_heads, tokens, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        if (own) impl_->finish_command("tiled probe attention");
    }
}

void MetalBackend::attention_f16(const BackendBuffer& q, uint32_t q_stride,
                                  const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                                  BackendBuffer& out, uint32_t seq_len,
                                  uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                                  float scale) {
    if (!seq_len || !kv_heads || q_heads%kv_heads || !head_dim || head_dim > 256)
        throw std::runtime_error("q27 Metal: invalid attention dimensions");
    const MetalBuffer& qb=metal_buffer(q); const MetalBuffer& kc=metal_buffer(k_cache); const MetalBuffer& vc=metal_buffer(v_cache);
    MetalBuffer& output=metal_buffer(out);
    check_range(qb.size(),0,((uint64_t)(q_heads-1)*q_stride+head_dim)*4,"attention Q");
    const uint64_t cache_bytes=(uint64_t)seq_len*kv_heads*head_dim*2;
    check_range(kc.size(),0,cache_bytes,"attention K cache"); check_range(vc.size(),0,cache_bytes,"attention V cache");
    check_range(output.size(),0,(uint64_t)q_heads*head_dim*4,"attention output");
    const uint32_t gqa=q_heads/kv_heads;
    if (impl_->gqa_threshold && seq_len >= impl_->gqa_threshold && gqa >= 2 && gqa <= 8) {
        impl_->attention_gqa_dispatch(false,qb,q_stride,kc,vc,output,seq_len,q_heads,kv_heads,head_dim,scale);
        return;
    }
    AttentionArgs args{q_stride,seq_len,q_heads,kv_heads,head_dim,scale};
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_attention_f16"); [enc setComputePipelineState:impl_->attention];
        [enc setBuffer:qb.handle() offset:0 atIndex:0]; [enc setBuffer:kc.handle() offset:0 atIndex:1]; [enc setBuffer:vc.handle() offset:0 atIndex:2];
        [enc setBuffer:output.handle() offset:0 atIndex:3]; [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(q_heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(own) impl_->finish_command("FP16 attention");
    }
}

void MetalBackend::gdn_gates(const BackendBuffer& alpha, const BackendBuffer& beta_raw,
                              const BackendTensor& ssm_a, const BackendTensor& ssm_dt,
                              BackendBuffer& g, BackendBuffer& beta, uint32_t heads) {
    const MetalBuffer& ar=metal_buffer(alpha); const MetalBuffer& br=metal_buffer(beta_raw);
    const MetalBuffer& a=tensor_data(ssm_a,DType::F32,"GDN a"); const MetalBuffer& dt=tensor_data(ssm_dt,DType::F32,"GDN dt");
    MetalBuffer& go=metal_buffer(g); MetalBuffer& bo=metal_buffer(beta);
    check_range(ar.size(),0,(uint64_t)heads*4,"GDN alpha"); check_range(br.size(),0,(uint64_t)heads*4,"GDN beta raw");
    check_range(go.size(),0,(uint64_t)heads*4,"GDN g"); check_range(bo.size(),0,(uint64_t)heads*4,"GDN beta");
    check_range(tensor_limit(a.size(), ssm_a.data_offset, ssm_a.data_size), ssm_a.data_offset,(uint64_t)heads*4,"GDN a"); check_range(tensor_limit(dt.size(), ssm_dt.data_offset, ssm_dt.data_size), ssm_dt.data_offset,(uint64_t)heads*4,"GDN dt");
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_gdn_gates"); [enc setComputePipelineState:impl_->gates];
        [enc setBuffer:ar.handle() offset:0 atIndex:0]; [enc setBuffer:br.handle() offset:0 atIndex:1];
        [enc setBuffer:a.handle() offset:(NSUInteger)ssm_a.data_offset atIndex:2]; [enc setBuffer:dt.handle() offset:(NSUInteger)ssm_dt.data_offset atIndex:3];
        [enc setBuffer:go.handle() offset:0 atIndex:4]; [enc setBuffer:bo.handle() offset:0 atIndex:5]; [enc setBytes:&heads length:4 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(heads,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; if(own) impl_->finish_command("GDN gates");
    }
}

void MetalBackend::conv_step(const BackendBuffer& ring_src, BackendBuffer& ring_dst,
                              const BackendBuffer& qkv, const BackendTensor& conv_weight,
                              BackendBuffer& out, uint32_t channels) {
    const MetalBuffer& src=metal_buffer(ring_src); MetalBuffer& dst=metal_buffer(ring_dst); const MetalBuffer& q=metal_buffer(qkv);
    const MetalBuffer& w=tensor_data(conv_weight,DType::F32,"GDN convolution"); MetalBuffer& o=metal_buffer(out);
    check_range(src.size(),0,(uint64_t)channels*3*4,"conv ring source"); check_range(dst.size(),0,(uint64_t)channels*3*4,"conv ring destination");
    check_range(q.size(),0,(uint64_t)channels*4,"conv input"); check_range(tensor_limit(w.size(), conv_weight.data_offset, conv_weight.data_size), conv_weight.data_offset,(uint64_t)channels*4*4,"conv weight"); check_range(o.size(),0,(uint64_t)channels*4,"conv output");
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_conv_step"); [enc setComputePipelineState:impl_->conv];
        [enc setBuffer:src.handle() offset:0 atIndex:0]; [enc setBuffer:dst.handle() offset:0 atIndex:1]; [enc setBuffer:q.handle() offset:0 atIndex:2];
        [enc setBuffer:w.handle() offset:(NSUInteger)conv_weight.data_offset atIndex:3]; [enc setBuffer:o.handle() offset:0 atIndex:4]; [enc setBytes:&channels length:4 atIndex:5];
        [enc dispatchThreads:MTLSizeMake(channels,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; if(own) impl_->finish_command("GDN convolution");
    }
}

void MetalBackend::delta_step(const BackendBuffer& state_src, BackendBuffer& state_dst,
                               const BackendBuffer& conv, const BackendBuffer& g,
                               const BackendBuffer& beta, BackendBuffer& out,
                               uint32_t value_heads, uint32_t qk_heads,
                               uint32_t head_dim) {
    if(head_dim!=128 || qk_heads!=16 || impl_->delta.maxTotalThreadsPerThreadgroup<512) throw std::runtime_error("q27 Metal: unsupported DeltaNet shape");
    const MetalBuffer& src=metal_buffer(state_src); MetalBuffer& dst=metal_buffer(state_dst); const MetalBuffer& cv=metal_buffer(conv);
    const MetalBuffer& gb=metal_buffer(g); const MetalBuffer& bb=metal_buffer(beta); MetalBuffer& o=metal_buffer(out);
    const uint64_t state_bytes=(uint64_t)value_heads*head_dim*head_dim*4;
    check_range(src.size(),0,state_bytes,"delta state source"); check_range(dst.size(),0,state_bytes,"delta state destination");
    check_range(cv.size(),0,(uint64_t)(qk_heads*2+value_heads)*head_dim*4,"delta conv"); check_range(gb.size(),0,(uint64_t)value_heads*4,"delta g"); check_range(bb.size(),0,(uint64_t)value_heads*4,"delta beta"); check_range(o.size(),0,(uint64_t)value_heads*head_dim*4,"delta output");
    DeltaArgs args{value_heads,qk_heads,head_dim};
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_delta_step"); [enc setComputePipelineState:impl_->delta];
        [enc setBuffer:src.handle() offset:0 atIndex:0]; [enc setBuffer:dst.handle() offset:0 atIndex:1]; [enc setBuffer:cv.handle() offset:0 atIndex:2];
        [enc setBuffer:gb.handle() offset:0 atIndex:3]; [enc setBuffer:bb.handle() offset:0 atIndex:4]; [enc setBuffer:o.handle() offset:0 atIndex:5]; [enc setBytes:&args length:sizeof(args) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(value_heads,1,1) threadsPerThreadgroup:MTLSizeMake(512,1,1)]; if(own) impl_->finish_command("DeltaNet recurrence");
    }
}

void MetalBackend::gated_norm_gdn(const BackendBuffer& x, const BackendTensor& weight,
                                   const BackendBuffer& gate, BackendBuffer& out,
                                   uint32_t heads, uint32_t head_dim, float eps) {
    const MetalBuffer& xb=metal_buffer(x); const MetalBuffer& w=tensor_data(weight,DType::F32,"GDN norm"); const MetalBuffer& gb=metal_buffer(gate); MetalBuffer& o=metal_buffer(out);
    const uint64_t bytes=(uint64_t)heads*head_dim*4;
    check_range(xb.size(),0,bytes,"GDN norm input"); check_range(gb.size(),0,bytes,"GDN norm gate"); check_range(o.size(),0,bytes,"GDN norm output"); check_range(tensor_limit(w.size(), weight.data_offset, weight.data_size), weight.data_offset,(uint64_t)head_dim*4,"GDN norm weight");
    HeadArgs args{heads,head_dim,head_dim,8,eps};
    @autoreleasepool {
        bool own; auto enc=impl_->encoder_for_operation(own, "q27_gated_norm_gdn"); [enc setComputePipelineState:impl_->gated_norm];
        [enc setBuffer:xb.handle() offset:0 atIndex:0]; [enc setBuffer:w.handle() offset:(NSUInteger)weight.data_offset atIndex:1]; [enc setBuffer:gb.handle() offset:0 atIndex:2]; [enc setBuffer:o.handle() offset:0 atIndex:3]; [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; if(own) impl_->finish_command("GDN gated norm");
    }
}

void MetalBackend::embedding_q8_rows(const BackendTensor& weight, const uint32_t* tokens,
                                      uint32_t count, BackendBuffer& out) {
    if (!tokens || !count || count > 96)
        throw std::runtime_error("q27 Metal: chunked embedding requires 1..12 tokens");
    const bool t2 = weight.dtype == DType::T2_G128;
    if ((weight.dtype != DType::Q8_G128 && !t2) || !weight.data || !weight.scales ||
        !weight.rows || !weight.cols || weight.rows > UINT32_MAX || weight.cols > UINT32_MAX ||
        weight.cols % 128)
        throw std::runtime_error("q27 Metal: invalid chunked embedding tensor");
    EmbedRowsArgs args{(uint32_t)weight.cols, count, {}};
    for (uint32_t i = 0; i < count; i++) {
        if (tokens[i] >= weight.rows) throw std::runtime_error("q27 Metal: chunked embedding token out of range");
        args.tokens[i] = tokens[i];
    }
    check_range(out.size(), 0, (uint64_t)count * weight.cols * 4, "chunked embedding output");
    const MetalBuffer& data = metal_buffer(*weight.data);
    const MetalBuffer& scales = metal_buffer(*weight.scales);
    check_range(tensor_limit(data.size(), weight.data_offset, weight.data_size), weight.data_offset,
                weight.rows * weight.cols / (t2 ? 4 : 1), "chunked embedding weight");
    check_range(tensor_limit(scales.size(), weight.scales_offset, weight.scales_size), weight.scales_offset, weight.rows * (weight.cols / 128) * 2, "chunked embedding scales");
    MetalBuffer& output = metal_buffer(out);
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own,
            t2 ? "q27_embedding_t2_rows" : "q27_embedding_q8_rows");
        [enc setComputePipelineState:t2 ? impl_->embedding_t2_rows : impl_->embedding_rows];
        [enc setBuffer:data.handle() offset:(NSUInteger)weight.data_offset atIndex:0];
        [enc setBuffer:scales.handle() offset:(NSUInteger)weight.scales_offset atIndex:1];
        [enc setBuffer:output.handle() offset:0 atIndex:2];
        [enc setBytes:&args length:sizeof(args) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(weight.cols, count, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (own) impl_->finish_command("chunked embedding");
    }
}

void MetalBackend::rmsnorm_rows_quantized(const BackendBuffer& x, const BackendTensor& weight,
                                          BackendBuffer& out, uint32_t n, uint32_t rows,
                                          float eps, BackendQuantized& quantized) {
    if (!n || n % 32 || !rows || quantized.count != n * rows || !quantized.values || !quantized.scales)
        throw std::runtime_error("q27 Metal: invalid chunked rmsnorm quantization");
    const MetalBuffer& w = tensor_data(weight, DType::F32, "chunked rmsnorm");
    const MetalBuffer& input = metal_buffer(x); MetalBuffer& output = metal_buffer(out);
    MetalBuffer& values = metal_buffer(*quantized.values); MetalBuffer& scales = metal_buffer(*quantized.scales);
    const uint64_t total = (uint64_t)n * rows;
    check_range(input.size(), 0, total * 4, "chunked rmsnorm input");
    check_range(output.size(), 0, total * 4, "chunked rmsnorm output");
    check_range(tensor_limit(w.size(), weight.data_offset, weight.data_size), weight.data_offset, (uint64_t)n * 4, "chunked rmsnorm weight");
    check_range(values.size(), 0, total, "chunked rmsnorm values");
    check_range(scales.size(), 0, (total / 32) * 4, "chunked rmsnorm scales");
    RowsNormArgs args{n, rows, 8, eps};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_rmsnorm_rows_quantized");
        [enc setComputePipelineState:impl_->rms_rows_quantized];
        [enc setBuffer:input.handle() offset:0 atIndex:0];
        [enc setBuffer:w.handle() offset:(NSUInteger)weight.data_offset atIndex:1];
        [enc setBuffer:output.handle() offset:0 atIndex:2];
        [enc setBuffer:values.handle() offset:0 atIndex:3];
        [enc setBuffer:scales.handle() offset:0 atIndex:4];
        [enc setBytes:&args length:sizeof(args) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if (own) impl_->finish_command("chunked rmsnorm quantize");
    }
}

void MetalBackend::matvec_f16_pair_rows(const BackendTensor& a, BackendBuffer& a_out,
                                        const BackendTensor& b, BackendBuffer& b_out,
                                        const BackendBuffer& x, uint32_t rows) {
    if (a.dtype != DType::F16 || b.dtype != DType::F16 || a.cols != b.cols || !a.data || !b.data ||
        !a.rows || !b.rows || !a.cols || a.rows > UINT32_MAX || b.rows > UINT32_MAX || a.cols > UINT32_MAX)
        throw std::runtime_error("q27 Metal: chunked F16 matvec pair requires compatible F16 weights");
    if (!rows || rows > 96) throw std::runtime_error("q27 Metal: chunked F16 matvec pair requires 1..96 rows");
    check_range(x.size(), 0, (uint64_t)rows * a.cols * 4, "chunked matvec input");
    check_range(a_out.size(), 0, (uint64_t)rows * a.rows * 4, "chunked matvec output A");
    check_range(b_out.size(), 0, (uint64_t)rows * b.rows * 4, "chunked matvec output B");
    const MetalBuffer& ad = metal_buffer(*a.data); const MetalBuffer& bd = metal_buffer(*b.data);
    const MetalBuffer& input = metal_buffer(x);
    MetalBuffer& ao = metal_buffer(a_out); MetalBuffer& bo = metal_buffer(b_out);
    check_range(tensor_limit(ad.size(), a.data_offset, a.data_size), a.data_offset, a.rows * a.cols * 2, "chunked matvec weight A");
    check_range(tensor_limit(bd.size(), b.data_offset, b.data_size), b.data_offset, b.rows * b.cols * 2, "chunked matvec weight B");
    MatvecPairRowsArgs args{(uint32_t)a.rows, (uint32_t)b.rows, (uint32_t)a.cols, rows};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_matvec_f16_pair_rows");
        [enc setComputePipelineState:impl_->f16_pair_rows];
        [enc setBuffer:ad.handle() offset:(NSUInteger)a.data_offset atIndex:0]; [enc setBuffer:ao.handle() offset:0 atIndex:1];
        [enc setBuffer:bd.handle() offset:(NSUInteger)b.data_offset atIndex:2]; [enc setBuffer:bo.handle() offset:0 atIndex:3];
        [enc setBuffer:input.handle() offset:0 atIndex:4]; [enc setBytes:&args length:sizeof(args) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)std::max(a.rows,b.rows), rows, 1)
                threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if (own) impl_->finish_command("chunked F16 matvec pair");
    }
}

void MetalBackend::gdn_gates_rows(const BackendBuffer& alpha, const BackendBuffer& beta_raw,
                                  const BackendTensor& ssm_a, const BackendTensor& ssm_dt,
                                  BackendBuffer& g, BackendBuffer& beta,
                                  uint32_t heads, uint32_t tokens) {
    if (!heads || !tokens || tokens > 96) throw std::runtime_error("q27 Metal: invalid chunked GDN gates");
    const MetalBuffer& ar = metal_buffer(alpha); const MetalBuffer& br = metal_buffer(beta_raw);
    const MetalBuffer& a = tensor_data(ssm_a, DType::F32, "chunked GDN a");
    const MetalBuffer& dt = tensor_data(ssm_dt, DType::F32, "chunked GDN dt");
    MetalBuffer& go = metal_buffer(g); MetalBuffer& bo = metal_buffer(beta);
    const uint64_t total = (uint64_t)heads * tokens * 4;
    check_range(ar.size(), 0, total, "chunked GDN alpha"); check_range(br.size(), 0, total, "chunked GDN beta raw");
    check_range(go.size(), 0, total, "chunked GDN g"); check_range(bo.size(), 0, total, "chunked GDN beta");
    check_range(tensor_limit(a.size(), ssm_a.data_offset, ssm_a.data_size), ssm_a.data_offset, (uint64_t)heads * 4, "chunked GDN a");
    check_range(tensor_limit(dt.size(), ssm_dt.data_offset, ssm_dt.data_size), ssm_dt.data_offset, (uint64_t)heads * 4, "chunked GDN dt");
    GatesRowsArgs args{heads, tokens};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_gdn_gates_rows");
        [enc setComputePipelineState:impl_->gates_rows];
        [enc setBuffer:ar.handle() offset:0 atIndex:0]; [enc setBuffer:br.handle() offset:0 atIndex:1];
        [enc setBuffer:a.handle() offset:(NSUInteger)ssm_a.data_offset atIndex:2];
        [enc setBuffer:dt.handle() offset:(NSUInteger)ssm_dt.data_offset atIndex:3];
        [enc setBuffer:go.handle() offset:0 atIndex:4]; [enc setBuffer:bo.handle() offset:0 atIndex:5];
        [enc setBytes:&args length:sizeof(args) atIndex:6];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)heads * tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        if (own) impl_->finish_command("chunked GDN gates");
    }
}

void MetalBackend::conv_chunk(const BackendBuffer& ring_src, BackendBuffer& ring_dst,
                              const BackendBuffer& qkv,
                              const BackendTensor& conv_weight, BackendBuffer& out,
                              uint32_t channels, uint32_t tokens) {
    if (!channels || !tokens || tokens > 96) throw std::runtime_error("q27 Metal: invalid chunked convolution");
    const MetalBuffer& rs = metal_buffer(ring_src); MetalBuffer& rd = metal_buffer(ring_dst);
    const MetalBuffer& q = metal_buffer(qkv);
    const MetalBuffer& w = tensor_data(conv_weight, DType::F32, "chunked GDN convolution");
    MetalBuffer& o = metal_buffer(out);
    check_range(rs.size(), 0, (uint64_t)channels * 3 * 4, "chunked conv ring src");
    check_range(rd.size(), 0, (uint64_t)channels * 3 * 4, "chunked conv ring dst");
    check_range(q.size(), 0, (uint64_t)channels * tokens * 4, "chunked conv input");
    check_range(tensor_limit(w.size(), conv_weight.data_offset, conv_weight.data_size), conv_weight.data_offset, (uint64_t)channels * 4 * 4, "chunked conv weight");
    check_range(o.size(), 0, (uint64_t)channels * tokens * 4, "chunked conv output");
    ConvChunkArgs args{channels, tokens};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_conv_chunk");
        [enc setComputePipelineState:impl_->conv_chunked];
        [enc setBuffer:rs.handle() offset:0 atIndex:0]; [enc setBuffer:rd.handle() offset:0 atIndex:1];
        [enc setBuffer:q.handle() offset:0 atIndex:2];
        [enc setBuffer:w.handle() offset:(NSUInteger)conv_weight.data_offset atIndex:3];
        [enc setBuffer:o.handle() offset:0 atIndex:4]; [enc setBytes:&args length:sizeof(args) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(channels,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if (own) impl_->finish_command("chunked GDN convolution");
    }
}

void MetalBackend::delta_chunk(const BackendBuffer& state_src, BackendBuffer& state_dst,
                               const BackendBuffer& conv,
                               const BackendBuffer& g, const BackendBuffer& beta,
                               BackendBuffer& out, uint32_t value_heads, uint32_t qk_heads,
                               uint32_t head_dim, uint32_t tokens) {
    if (head_dim != 128 || qk_heads != 16 || !tokens || tokens > 96 ||
        impl_->delta_chunked.maxTotalThreadsPerThreadgroup < 512)
        throw std::runtime_error("q27 Metal: unsupported chunked DeltaNet shape");
    const MetalBuffer& ss = metal_buffer(state_src); MetalBuffer& sd = metal_buffer(state_dst);
    const MetalBuffer& cv = metal_buffer(conv);
    const MetalBuffer& gb = metal_buffer(g); const MetalBuffer& bb = metal_buffer(beta);
    MetalBuffer& o = metal_buffer(out);
    const uint64_t state_bytes = (uint64_t)value_heads * head_dim * head_dim * 4;
    check_range(ss.size(), 0, state_bytes, "chunked delta state src");
    check_range(sd.size(), 0, state_bytes, "chunked delta state dst");
    check_range(cv.size(), 0, (uint64_t)(qk_heads * 2 + value_heads) * head_dim * tokens * 4, "chunked delta conv");
    check_range(gb.size(), 0, (uint64_t)value_heads * tokens * 4, "chunked delta g");
    check_range(bb.size(), 0, (uint64_t)value_heads * tokens * 4, "chunked delta beta");
    check_range(o.size(), 0, (uint64_t)value_heads * head_dim * tokens * 4, "chunked delta output");
    DeltaChunkArgs args{value_heads, qk_heads, head_dim, tokens};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_delta_chunk");
        [enc setComputePipelineState:impl_->delta_chunked];
        [enc setBuffer:ss.handle() offset:0 atIndex:0]; [enc setBuffer:sd.handle() offset:0 atIndex:1];
        [enc setBuffer:cv.handle() offset:0 atIndex:2];
        [enc setBuffer:gb.handle() offset:0 atIndex:3]; [enc setBuffer:bb.handle() offset:0 atIndex:4];
        [enc setBuffer:o.handle() offset:0 atIndex:5]; [enc setBytes:&args length:sizeof(args) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(value_heads,1,1) threadsPerThreadgroup:MTLSizeMake(512,1,1)];
        if (own) impl_->finish_command("chunked DeltaNet recurrence");
    }
}

void MetalBackend::l2norm_rows(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                               uint32_t row_stride, uint32_t tokens, float eps) {
    if (!heads || !tokens || tokens > 96 || row_stride < heads * head_dim)
        throw std::runtime_error("q27 Metal: invalid chunked l2norm");
    MetalBuffer& input = metal_buffer(x);
    check_range(input.size(), 0, ((uint64_t)(tokens - 1) * row_stride + (uint64_t)heads * head_dim) * 4,
                "chunked l2norm input");
    L2RowsArgs args{heads, head_dim, row_stride, tokens, eps};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_l2norm_rows");
        [enc setComputePipelineState:impl_->l2_rows];
        [enc setBuffer:input.handle() offset:0 atIndex:0];
        [enc setBytes:&args length:sizeof(args) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(heads,tokens,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if (own) impl_->finish_command("chunked l2norm");
    }
}

void MetalBackend::rope_neox_rows(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                                  uint32_t n_rot, uint32_t stride, uint32_t row_stride,
                                  uint32_t position, uint32_t tokens, float freq_base) {
    if (!n_rot || n_rot > head_dim || (n_rot & 1) || !tokens || tokens > 96)
        throw std::runtime_error("q27 Metal: invalid chunked RoPE dimensions");
    MetalBuffer& input = metal_buffer(x);
    check_range(input.size(), 0,
                ((uint64_t)(tokens - 1) * row_stride + (uint64_t)(heads - 1) * stride + head_dim) * 4,
                "chunked rope input");
    RopeRowsArgs args{heads, head_dim, n_rot, stride, row_stride, position, tokens, freq_base};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_rope_neox_rows");
        [enc setComputePipelineState:impl_->rope_rows];
        [enc setBuffer:input.handle() offset:0 atIndex:0]; [enc setBytes:&args length:sizeof(args) atIndex:1];
        [enc dispatchThreads:MTLSizeMake(n_rot/2,heads,tokens) threadsPerThreadgroup:MTLSizeMake(n_rot/2,1,1)];
        if (own) impl_->finish_command("chunked rope");
    }
}

void MetalBackend::kv_store_f16_rows(const BackendBuffer& k, const BackendBuffer& v,
                                     BackendBuffer& k_cache, BackendBuffer& v_cache,
                                     uint32_t position, uint32_t row_length, uint32_t tokens) {
    if (!tokens || tokens > 96) throw std::runtime_error("q27 Metal: invalid chunked KV store");
    const MetalBuffer& kb = metal_buffer(k); const MetalBuffer& vb = metal_buffer(v);
    MetalBuffer& kc = metal_buffer(k_cache); MetalBuffer& vc = metal_buffer(v_cache);
    check_range(kb.size(), 0, (uint64_t)row_length * tokens * 4, "chunked K rows");
    check_range(vb.size(), 0, (uint64_t)row_length * tokens * 4, "chunked V rows");
    check_range(kc.size(), (uint64_t)position * row_length * 2, (uint64_t)row_length * tokens * 2, "chunked K cache");
    check_range(vc.size(), (uint64_t)position * row_length * 2, (uint64_t)row_length * tokens * 2, "chunked V cache");
    KvStoreRowsArgs args{position, row_length, tokens};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_kv_store_f16_rows");
        [enc setComputePipelineState:impl_->kv_store_rows];
        [enc setBuffer:kb.handle() offset:0 atIndex:0]; [enc setBuffer:vb.handle() offset:0 atIndex:1];
        [enc setBuffer:kc.handle() offset:0 atIndex:2]; [enc setBuffer:vc.handle() offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(row_length,tokens,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if (own) impl_->finish_command("chunked KV store");
    }
}

void MetalBackend::kv_store_turbo3_rows(const BackendBuffer& k, const BackendBuffer& v,
                                        BackendBuffer& k_cache, BackendBuffer& v_cache,
                                        uint32_t position, uint32_t kv_heads, uint32_t tokens) {
    if (!kv_heads || !tokens || tokens > 96)
        throw std::runtime_error("q27 Metal: invalid chunked turbo3 KV store");
    const MetalBuffer& kb = metal_buffer(k); const MetalBuffer& vb = metal_buffer(v);
    MetalBuffer& kc = metal_buffer(k_cache); MetalBuffer& vc = metal_buffer(v_cache);
    const uint64_t row_bytes = (uint64_t)kv_heads * 2 * 50;
    check_range(kb.size(), 0, (uint64_t)kv_heads * 256 * tokens * 4, "chunked turbo3 K rows");
    check_range(vb.size(), 0, (uint64_t)kv_heads * 256 * tokens * 4, "chunked turbo3 V rows");
    check_range(kc.size(), (uint64_t)position * row_bytes, row_bytes * tokens, "chunked turbo3 K cache");
    check_range(vc.size(), (uint64_t)position * row_bytes, row_bytes * tokens, "chunked turbo3 V cache");
    TurboStoreRowsArgs args{position, kv_heads, tokens};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_kv_store_turbo3_rows");
        [enc setComputePipelineState:impl_->kv_store_turbo3_rows];
        [enc setBuffer:kb.handle() offset:0 atIndex:0]; [enc setBuffer:vb.handle() offset:0 atIndex:1];
        [enc setBuffer:kc.handle() offset:0 atIndex:2]; [enc setBuffer:vc.handle() offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)kv_heads*2,2,tokens)
                threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        if (own) impl_->finish_command("chunked turbo3 KV store");
    }
}

void MetalBackend::attention_f16_causal(const BackendBuffer& q, uint32_t q_stride,
                                        uint32_t q_row_stride, const BackendBuffer& k_cache,
                                        const BackendBuffer& v_cache,
                                        BackendBuffer& out, uint32_t base_len, uint32_t q_heads,
                                        uint32_t kv_heads, uint32_t head_dim, uint32_t tokens,
                                        float scale) {
    if (!base_len || !kv_heads || q_heads % kv_heads || !head_dim || head_dim > 256 ||
        !tokens || tokens > 96)
        throw std::runtime_error("q27 Metal: invalid chunked attention dimensions");
    const MetalBuffer& qb = metal_buffer(q); const MetalBuffer& kc = metal_buffer(k_cache);
    const MetalBuffer& vc = metal_buffer(v_cache);
    MetalBuffer& output = metal_buffer(out);
    const uint32_t max_seq = base_len + tokens - 1;
    check_range(qb.size(), 0,
                ((uint64_t)(tokens-1)*q_row_stride + (uint64_t)(q_heads-1)*q_stride + head_dim)*4,
                "chunked attention Q");
    const uint64_t cache_bytes = (uint64_t)max_seq * kv_heads * head_dim * 2;
    check_range(kc.size(), 0, cache_bytes, "chunked attention K cache");
    check_range(vc.size(), 0, cache_bytes, "chunked attention V cache");
    check_range(output.size(), 0, (uint64_t)tokens * q_heads * head_dim * 4, "chunked attention output");
    if (base_len > UINT32_MAX - tokens - impl_->gqa_block)
        throw std::runtime_error("q27 Metal: chunked attention sequence too large");
    // Each chunk row must take the same path serial decode takes at that
    // row's sequence length, or chunked prefill and serial ingestion diverge
    // at the threshold. Rows past the threshold go to the GQA kernels; a
    // chunk straddling it splits into two dispatches.
    const uint32_t gqa = q_heads / kv_heads;
    uint32_t gqa_from = tokens;
    if (impl_->gqa_threshold && gqa >= 2 && gqa <= 8)
        gqa_from = base_len >= impl_->gqa_threshold ? 0
                 : std::min(tokens, impl_->gqa_threshold - base_len);
    if (gqa_from < tokens) {
        impl_->attention_gqa_causal_dispatch(false, qb, q_stride, q_row_stride, kc, vc, output,
                                             base_len + gqa_from, q_heads, kv_heads, head_dim,
                                             tokens - gqa_from, scale,
                                             (uint64_t)gqa_from * q_row_stride * 4,
                                             (uint64_t)gqa_from * q_heads * head_dim * 4);
        if (gqa_from == 0) return;
    }
    AttentionCausalArgs args{q_stride, q_row_stride, base_len, q_heads, kv_heads, head_dim, gqa_from, scale};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_attention_f16_causal");
        [enc setComputePipelineState:impl_->attention_causal];
        [enc setBuffer:qb.handle() offset:0 atIndex:0]; [enc setBuffer:kc.handle() offset:0 atIndex:1];
        [enc setBuffer:vc.handle() offset:0 atIndex:2];
        [enc setBuffer:output.handle() offset:0 atIndex:3]; [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(q_heads,gqa_from,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if (own) impl_->finish_command("chunked FP16 attention");
    }
}

void MetalBackend::attention_turbo3_causal(const BackendBuffer& q, uint32_t q_stride,
                                           uint32_t q_row_stride, const BackendBuffer& k_cache,
                                           const BackendBuffer& v_cache,
                                           BackendBuffer& out, uint32_t base_len, uint32_t q_heads,
                                           uint32_t kv_heads, uint32_t head_dim, uint32_t tokens,
                                           float scale) {
    if (!base_len || !kv_heads || q_heads % kv_heads || head_dim != 256 || !tokens || tokens > 96)
        throw std::runtime_error("q27 Metal: invalid chunked turbo3 attention dimensions");
    const MetalBuffer& qb = metal_buffer(q); const MetalBuffer& kc = metal_buffer(k_cache);
    const MetalBuffer& vc = metal_buffer(v_cache);
    MetalBuffer& output = metal_buffer(out);
    const uint32_t max_seq = base_len + tokens - 1;
    check_range(qb.size(), 0,
                ((uint64_t)(tokens-1)*q_row_stride + (uint64_t)(q_heads-1)*q_stride + head_dim)*4,
                "chunked turbo3 attention Q");
    const uint64_t cache_bytes = (uint64_t)max_seq * kv_heads * 2 * 50;
    check_range(kc.size(), 0, cache_bytes, "chunked turbo3 K cache");
    check_range(vc.size(), 0, cache_bytes, "chunked turbo3 V cache");
    check_range(output.size(), 0, (uint64_t)tokens * q_heads * head_dim * 4, "chunked turbo3 attention output");
    if (base_len > UINT32_MAX - tokens - impl_->gqa_block)
        throw std::runtime_error("q27 Metal: chunked turbo3 attention sequence too large");
    // Each chunk row must take the same path serial decode takes at that
    // row's sequence length, or chunked prefill and serial ingestion diverge
    // at the threshold. Rows past the threshold go to the GQA kernels; a
    // chunk straddling it splits into two dispatches.
    const uint32_t gqa = q_heads / kv_heads;
    uint32_t gqa_from = tokens;
    if (impl_->gqa_threshold && gqa >= 2 && gqa <= 8)
        gqa_from = base_len >= impl_->gqa_threshold ? 0
                 : std::min(tokens, impl_->gqa_threshold - base_len);
    if (gqa_from < tokens) {
        impl_->attention_gqa_causal_dispatch(true, qb, q_stride, q_row_stride, kc, vc, output,
                                             base_len + gqa_from, q_heads, kv_heads, head_dim,
                                             tokens - gqa_from, scale,
                                             (uint64_t)gqa_from * q_row_stride * 4,
                                             (uint64_t)gqa_from * q_heads * head_dim * 4);
        if (gqa_from == 0) return;
    }
    AttentionCausalArgs args{q_stride, q_row_stride, base_len, q_heads, kv_heads, head_dim, gqa_from, scale};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_attention_turbo3_causal");
        [enc setComputePipelineState:impl_->attention_turbo3_causal_p];
        [enc setBuffer:qb.handle() offset:0 atIndex:0]; [enc setBuffer:kc.handle() offset:0 atIndex:1];
        [enc setBuffer:vc.handle() offset:0 atIndex:2];
        [enc setBuffer:output.handle() offset:0 atIndex:3]; [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(q_heads,gqa_from,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if (own) impl_->finish_command("chunked turbo3 attention");
    }
}

void MetalBackend::sigmoid_gate_mul_rows(BackendBuffer& out, const BackendBuffer& qg,
                                         uint32_t heads, uint32_t head_dim, uint32_t tokens) {
    if (!heads || !head_dim || !tokens || tokens > 96)
        throw std::runtime_error("q27 Metal: invalid chunked sigmoid gate");
    MetalBuffer& o = metal_buffer(out); const MetalBuffer& gates = metal_buffer(qg);
    const uint64_t n = (uint64_t)heads * head_dim * tokens;
    check_range(o.size(), 0, n * 4, "chunked sigmoid output");
    check_range(gates.size(), 0, n * 2 * 4, "chunked sigmoid gates");
    GateRowsArgs args{heads, head_dim, tokens};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_sigmoid_gate_mul_rows");
        [enc setComputePipelineState:impl_->sigmoid_gate_rows];
        [enc setBuffer:o.handle() offset:0 atIndex:0]; [enc setBuffer:gates.handle() offset:0 atIndex:1];
        [enc setBytes:&args length:sizeof(args) atIndex:2];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if (own) impl_->finish_command("chunked sigmoid gate");
    }
}

void MetalBackend::argmax_rows(const BackendBuffer& x, uint32_t n, uint32_t rows,
                               BackendBuffer& out_indices) {
    // Cap matches VERIFY_CHUNK_MAX (lever 2): the verify/oracle path argmaxes
    // up to 48 rows; the kernel is one threadgroup per row, width-agnostic.
    if (!n || !rows || rows > 48) throw std::runtime_error("q27 Metal: invalid chunked argmax");
    const MetalBuffer& input = metal_buffer(x); MetalBuffer& output = metal_buffer(out_indices);
    check_range(input.size(), 0, (uint64_t)n * rows * 4, "chunked argmax input");
    check_range(output.size(), 0, (uint64_t)rows * 4, "chunked argmax output");
    ArgmaxRowsArgs args{n, rows};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_argmax_rows");
        [enc setComputePipelineState:impl_->argmax_rows_p];
        [enc setBuffer:input.handle() offset:0 atIndex:0];
        [enc setBuffer:output.handle() offset:0 atIndex:1];
        [enc setBytes:&args length:sizeof(args) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if (own) impl_->finish_command("chunked argmax");
    }
}

void MetalBackend::nll_rows(const BackendBuffer& logits, const BackendBuffer& targets,
                            BackendBuffer& nll, uint32_t n, uint32_t rows) {
    if (!n || !rows || rows > 12) throw std::runtime_error("q27 Metal: invalid NLL shape");
    const MetalBuffer& input = metal_buffer(logits);
    const MetalBuffer& tgt = metal_buffer(targets);
    MetalBuffer& out = metal_buffer(nll);
    check_range(input.size(), 0, (uint64_t)n * rows * 4, "NLL logits");
    check_range(tgt.size(), 0, (uint64_t)rows * 4, "NLL targets");
    check_range(out.size(), 0, (uint64_t)rows * 4, "NLL output");
    struct NllRowsArgs { uint32_t n; uint32_t rows; } args{n, rows};
    @autoreleasepool {
        bool own; auto enc = impl_->encoder_for_operation(own, "q27_nll_rows");
        [enc setComputePipelineState:impl_->nll_rows_p];
        [enc setBuffer:input.handle() offset:0 atIndex:0];
        [enc setBuffer:tgt.handle() offset:0 atIndex:1];
        [enc setBuffer:out.handle() offset:0 atIndex:2];
        [enc setBytes:&args length:sizeof(args) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (own) impl_->finish_command("nll rows");
    }
}

void MetalBackend::synchronize() {
    @autoreleasepool {
        if (impl_->batching)
            throw std::runtime_error("q27 Metal: end command batch before synchronizing");
        id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
        if (!command) throw std::runtime_error("q27 Metal: command creation failed");
        [command commit];
        [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError)
            throw std::runtime_error("q27 Metal: synchronization failed");
    }
}

void MetalBackend::profile_reset() {
    if (!impl_->profile) return;
    impl_->profile_stats.clear();
    impl_->profiled_command_buffers = 0;
    impl_->gpu_busy_seconds = 0.0;
    impl_->cpu_wait_seconds = 0.0;
}

uint64_t MetalBackend::recommended_working_set_size() const {
    return (uint64_t)impl_->device.recommendedMaxWorkingSetSize;
}

uint32_t MetalBackend::gqa_block_size() const {
    return impl_->gqa_block;
}

bool MetalBackend::gqa_blocked_reachable(uint32_t context) const {
    return impl_->gqa_threshold != 0 && context >= impl_->gqa_threshold;
}

void MetalBackend::set_gemm_half(bool enabled) {
    impl_->gemm_half = enabled;
}

void MetalBackend::set_gqa_threshold(uint32_t threshold) {
    impl_->gqa_threshold = threshold;
}

uint64_t MetalBackend::max_buffer_length() const {
    return (uint64_t)impl_->device.maxBufferLength;
}

bool MetalBackend::supports_quantized_matmul() const {
    return impl_->q4_quantized_matmul && impl_->q8_quantized_matmul && impl_->t2_quantized_matmul;
}

uint64_t MetalBackend::max_threadgroup_memory_length() const {
    return (uint64_t)impl_->device.maxThreadgroupMemoryLength;
}

} // namespace q27
