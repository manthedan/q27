// Phase-0 synthetic deep-KV attention benchmark (cache-block scheduling,
// docs/metal/plans/2026-07-15-cache-block-scheduling.md).
//
// Measures the GQA attention dispatch cost at depth against synthetic KV
// caches — no model, memory-safe — separating the decode grid from the
// causal chunk grid, and A/Bs the two Phase-0 probe kernels:
//   R1  probe: head-major turbo3 KV layout (contiguous per-head runs)
//   R1b probe: token-tiled causal GQA (tile 2/4; KV stream / tile factor)
// Both probes are verified bit-identical against the production GQA kernels
// on the same logical cache before any timing is reported.
//
// Effective GB/s is LOGICAL: the bytes a factor-1 kernel must stream
// (decode: seq * row; chunk: sum over tokens of the visible prefix), so a
// tiling win shows up as logical bandwidth above the physical stream bound.
//
// The GQA threshold is forced to 1 (unless the caller set it) so every
// depth routes the blocked GQA kernels — the objects under test.

#include "metal_backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr uint32_t N_HEAD = 24;
constexpr uint32_t N_KV = 4;
constexpr uint32_t HEAD_DIM = 256;
constexpr uint32_t Q_STRIDE = 2 * HEAD_DIM;               // decode qg layout
constexpr uint32_t Q_ROW_STRIDE = 2 * N_HEAD * HEAD_DIM;  // chunk q rows
constexpr float SCALE = 1.0f / 16.0f;

// Varied, always-finite synthetic turbo3 rows: per 50-byte block a small
// positive half scale and pseudo-random code bytes (every code decodes to a
// finite centroid, so any byte pattern is a valid block).
void fill_turbo3_interleaved(std::vector<uint8_t>& cache, uint32_t seq, uint32_t salt) {
    cache.resize((size_t)seq * N_KV * 2 * 50);
    for (size_t b = 0; b < cache.size() / 50; b++) {
        uint8_t* block = cache.data() + b * 50;
        const uint16_t h = (uint16_t)(0x2c00 | ((b * 2654435761u + salt) & 0x03ff));
        std::memcpy(block, &h, 2);
        for (uint32_t i = 2; i < 50; i++)
            block[i] = (uint8_t)(((b * 50 + i) * 2246822519u + salt * 374761393u) >> 13);
    }
}

// Same logical rows re-addressed head-major: (kvh * seq + pos) * 100.
void relayout_headmajor(const std::vector<uint8_t>& il, std::vector<uint8_t>& hm, uint32_t seq) {
    hm.resize(il.size());
    for (uint32_t pos = 0; pos < seq; pos++)
        for (uint32_t kvh = 0; kvh < N_KV; kvh++)
            std::memcpy(hm.data() + ((size_t)kvh * seq + pos) * 100,
                        il.data() + ((size_t)pos * N_KV + kvh) * 100, 100);
}

void fill_f16(std::vector<uint8_t>& cache, uint32_t seq, uint32_t salt) {
    cache.resize((size_t)seq * N_KV * HEAD_DIM * 2);
    for (size_t i = 0; i < cache.size(); i += 2) {
        const uint16_t h = (uint16_t)(0x3000 | ((i * 2654435761u + salt) >> 20 & 0x03ff));
        std::memcpy(&cache[i], &h, 2);
    }
}

std::shared_ptr<q27::BackendBuffer> upload_bytes(q27::MetalBackend& backend,
                                                 const std::vector<uint8_t>& bytes) {
    auto buf = backend.allocate(bytes.size());
    backend.write(*buf, 0, bytes.data(), bytes.size());
    return buf;
}

double wall_per_op(q27::MetalBackend& backend, uint32_t reps,
                   const std::function<void(uint32_t)>& op) {
    // Warmup outside the timer (clock ramp, first-touch, pipeline caches).
    backend.begin_commands(); op(0); backend.end_commands();
    backend.begin_commands(); op(1 % reps); backend.end_commands();
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t r = 0; r < reps; r++) {
        backend.begin_commands();
        op(r);
        backend.end_commands();
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / reps;
}

bool read_equal(q27::MetalBackend& backend, q27::BackendBuffer& a, q27::BackendBuffer& b,
                uint64_t bytes, const char* what) {
    std::vector<uint8_t> ba(bytes), bb(bytes);
    backend.read(a, 0, ba.data(), bytes);
    backend.read(b, 0, bb.data(), bytes);
    if (std::memcmp(ba.data(), bb.data(), bytes) == 0) return true;
    size_t i = 0;
    while (i < bytes && ba[i] == bb[i]) i++;
    fprintf(stderr, "FAIL: %s diverges at byte %zu of %llu\n", what, i, (unsigned long long)bytes);
    return false;
}

} // namespace

int main(int argc, char** argv) {
    uint32_t seq = 32768, tokens = 12, reps = 50;
    std::string kv = "turbo3";
    bool verify = true;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--seq" && i + 1 < argc) seq = (uint32_t)atoi(argv[++i]);
        else if (arg == "--tokens" && i + 1 < argc) tokens = (uint32_t)atoi(argv[++i]);
        else if (arg == "--reps" && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (arg == "--kv" && i + 1 < argc) kv = argv[++i];
        else if (arg == "--no-verify") verify = false;
        else {
            fprintf(stderr, "usage: %s [--seq N] [--tokens N] [--reps N] [--kv turbo3|fp16] [--no-verify]\n",
                    argv[0]);
            return 1;
        }
    }
    if (!seq || !tokens || tokens > 96 || !reps || seq <= tokens ||
        (kv != "turbo3" && kv != "fp16")) {
        fprintf(stderr, "invalid arguments\n");
        return 1;
    }
    // Route every depth through the blocked GQA kernels, and pin the
    // production causal route to the UNTILED kernel so the "gqa" rows are a
    // stable factor-1 baseline for the probe ratios (the engine default is
    // tiled; its in-situ cost is the t2 row). Overwrite any inherited
    // values — an exported TILE=2 would make baseline and t2 the same
    // kernel, an exported THRESHOLD=0 would measure the legacy kernels
    // (codex P2, 2026-07-15).
    setenv("Q27_METAL_GQA_THRESHOLD", "1", 1);
    setenv("Q27_METAL_GQA_TILE", "1", 1);
    // Pin the block too: the R3 bf2 verify and ratios assume the staged
    // baselines run at block 1024 — an inherited Q27_METAL_GQA_BLOCK would
    // change the merge fold and fail the memcmp, or (with --no-verify)
    // silently compare mismatched blocks (codex P2, 2026-07-16).
    setenv("Q27_METAL_GQA_BLOCK", "1024", 1);

    q27::MetalBackend backend;
    const bool turbo3 = kv == "turbo3";
    const uint64_t row_bytes = turbo3 ? (uint64_t)N_KV * 2 * 50 : (uint64_t)N_KV * HEAD_DIM * 2;
    const uint64_t cache_bytes = (uint64_t)seq * row_bytes;
    // Enough cache copies that cycling them defeats the SLC — the engine
    // cycles 16 live attention layers, so steady state streams from DRAM.
    const uint32_t copies = (uint32_t)std::min<uint64_t>(16,
        std::max<uint64_t>(1, (256ull << 20) / (2 * cache_bytes) + 1));
    printf("backend: %s, seq %u, %s KV, %u cache copies (%.1f MiB cycled), %u reps\n",
           backend.name().c_str(), seq, kv.c_str(), copies,
           copies * 2 * cache_bytes / (1024.0 * 1024.0), reps);

    // Q and output buffers.
    std::vector<float> qhost((size_t)tokens * Q_ROW_STRIDE);
    for (size_t i = 0; i < qhost.size(); i++)
        qhost[i] = (float)((int)((i * 1103515245u + 12345u) >> 16 & 1023) - 512) / 512.0f;
    auto qbuf = backend.allocate(qhost.size() * 4);
    backend.write(*qbuf, 0, qhost.data(), qhost.size() * 4);
    auto out_a = backend.allocate((uint64_t)tokens * N_HEAD * HEAD_DIM * 4);
    auto out_b = backend.allocate((uint64_t)tokens * N_HEAD * HEAD_DIM * 4);
    // Blocked-GQA partials scratch (caller-owned since audit E2): sized for
    // the deepest fold this bench dispatches — the R3 sweep's smallest block
    // (128) produces the most partial rows.
    auto partials = backend.allocate_private(
        (uint64_t)tokens * N_HEAD * (1 + ((uint64_t)seq - 1) / 128) * 258 * 4);

    // Caches.
    std::vector<std::shared_ptr<q27::BackendBuffer>> kc(copies), vc(copies);
    std::vector<std::shared_ptr<q27::BackendBuffer>> kh, vh;
    {
        std::vector<uint8_t> il, hm;
        for (uint32_t c = 0; c < copies; c++) {
            if (turbo3) {
                fill_turbo3_interleaved(il, seq, c * 2 + 1);
                kc[c] = upload_bytes(backend, il);
                relayout_headmajor(il, hm, seq);
                kh.push_back(upload_bytes(backend, hm));
                fill_turbo3_interleaved(il, seq, c * 2 + 2);
                vc[c] = upload_bytes(backend, il);
                relayout_headmajor(il, hm, seq);
                vh.push_back(upload_bytes(backend, hm));
            } else {
                fill_f16(il, seq, c * 2 + 1);
                kc[c] = upload_bytes(backend, il);
                fill_f16(il, seq, c * 2 + 2);
                vc[c] = upload_bytes(backend, il);
            }
        }
    }

    const uint32_t base_len = seq - tokens + 1;   // deepest chunk token sees seq-1 positions
    const double decode_gb = (double)cache_bytes * 2;
    double chunk_gb = 0;
    for (uint32_t t = 0; t < tokens; t++) chunk_gb += (double)(base_len + t) * row_bytes * 2;

    auto decode_op = [&](q27::BackendBuffer& k, q27::BackendBuffer& v, q27::BackendBuffer& out) {
        if (turbo3) backend.attention_turbo3(*qbuf, Q_STRIDE, k, v, out, seq, N_HEAD, N_KV, HEAD_DIM, SCALE, partials.get());
        else backend.attention_f16(*qbuf, Q_STRIDE, k, v, out, seq, N_HEAD, N_KV, HEAD_DIM, SCALE, partials.get());
    };
    auto chunk_op = [&](q27::BackendBuffer& k, q27::BackendBuffer& v, q27::BackendBuffer& out) {
        if (turbo3) backend.attention_turbo3_causal(*qbuf, Q_STRIDE, Q_ROW_STRIDE, k, v, out,
                                                    base_len, N_HEAD, N_KV, HEAD_DIM, tokens, SCALE,
                                                    partials.get());
        else backend.attention_f16_causal(*qbuf, Q_STRIDE, Q_ROW_STRIDE, k, v, out,
                                          base_len, N_HEAD, N_KV, HEAD_DIM, tokens, SCALE,
                                          partials.get());
    };

    // ---- Verify probes bit-identical before timing anything ----
    if (verify && turbo3) {
        backend.begin_commands(); decode_op(*kc[0], *vc[0], *out_a); backend.end_commands();
        backend.begin_commands();
        backend.attention_turbo3_gqa_headmajor(*qbuf, Q_STRIDE, *kh[0], *vh[0], *out_b,
                                               seq, seq, N_HEAD, N_KV, HEAD_DIM, SCALE, *partials);
        backend.end_commands();
        if (!read_equal(backend, *out_a, *out_b, (uint64_t)N_HEAD * HEAD_DIM * 4,
                        "head-major decode probe")) return 1;
        backend.begin_commands(); chunk_op(*kc[0], *vc[0], *out_a); backend.end_commands();
        for (uint32_t tile : {2u, 4u}) {
            backend.begin_commands();
            backend.attention_turbo3_causal_gqa_tiled(*qbuf, Q_STRIDE, Q_ROW_STRIDE, *kc[0], *vc[0],
                                                      *out_b, base_len, N_HEAD, N_KV, HEAD_DIM,
                                                      tokens, tile, SCALE, *partials);
            backend.end_commands();
            char what[64];
            snprintf(what, sizeof what, "token-tiled causal probe t%u", tile);
            if (!read_equal(backend, *out_a, *out_b, (uint64_t)tokens * N_HEAD * HEAD_DIM * 4, what))
                return 1;
        }
        // R3: at the production block size the barrier-free kernel must be
        // bit-identical to the staged kernels (same dequant, dot order, and
        // fold); other block sizes change the fold and are timing-only here.
        backend.begin_commands();
        backend.attention_turbo3_causal_gqa_bf(*qbuf, Q_STRIDE, Q_ROW_STRIDE, *kc[0], *vc[0],
                                               *out_b, base_len, N_HEAD, N_KV, HEAD_DIM,
                                               tokens, 1024, SCALE, *partials);
        backend.end_commands();
        if (!read_equal(backend, *out_a, *out_b, (uint64_t)tokens * N_HEAD * HEAD_DIM * 4,
                        "barrier-free causal probe bf2 @ block 1024"))
            return 1;
        printf("verify: head-major + token-tiled (t2/t4) + barrier-free (bf2 @1024) "
               "bit-identical to the untiled GQA kernels\n");
    }

    // ---- Timing ----
    const double t_decode = wall_per_op(backend, reps, [&](uint32_t r) {
        decode_op(*kc[r % copies], *vc[r % copies], *out_a);
    });
    printf("decode  gqa       : %8.3f ms/dispatch, %6.1f GB/s logical\n",
           t_decode * 1e3, decode_gb / t_decode / 1e9);
    if (turbo3) {
        const double t_hm = wall_per_op(backend, reps, [&](uint32_t r) {
            backend.attention_turbo3_gqa_headmajor(*qbuf, Q_STRIDE, *kh[r % copies], *vh[r % copies],
                                                   *out_a, seq, seq, N_HEAD, N_KV, HEAD_DIM, SCALE,
                                                   *partials);
        });
        printf("decode  gqa-hm    : %8.3f ms/dispatch, %6.1f GB/s logical (%.2fx)\n",
               t_hm * 1e3, decode_gb / t_hm / 1e9, t_decode / t_hm);
    }
    const double t_chunk = wall_per_op(backend, reps, [&](uint32_t r) {
        chunk_op(*kc[r % copies], *vc[r % copies], *out_a);
    });
    printf("chunk%-3u gqa       : %8.3f ms/dispatch, %6.1f GB/s logical\n",
           tokens, t_chunk * 1e3, chunk_gb / t_chunk / 1e9);
    if (turbo3) {
        double t_t2 = 0;
        for (uint32_t tile : {2u, 4u}) {
            const double t_tiled = wall_per_op(backend, reps, [&](uint32_t r) {
                backend.attention_turbo3_causal_gqa_tiled(*qbuf, Q_STRIDE, Q_ROW_STRIDE,
                                                          *kc[r % copies], *vc[r % copies], *out_a,
                                                          base_len, N_HEAD, N_KV, HEAD_DIM,
                                                          tokens, tile, SCALE, *partials);
            });
            if (tile == 2) t_t2 = t_tiled;
            printf("chunk%-3u gqa-t%u    : %8.3f ms/dispatch, %6.1f GB/s logical (%.2fx)\n",
                   tokens, tile, t_tiled * 1e3, chunk_gb / t_tiled / 1e9, t_chunk / t_tiled);
        }
        // R3 block sweep: ratios are against the production t2 route (the
        // deploy decision baseline), not the untiled kernel.
        for (uint32_t block : {128u, 256u, 512u, 1024u, 2048u}) {
            const double t_bf = wall_per_op(backend, reps, [&](uint32_t r) {
                backend.attention_turbo3_causal_gqa_bf(*qbuf, Q_STRIDE, Q_ROW_STRIDE,
                                                       *kc[r % copies], *vc[r % copies], *out_a,
                                                       base_len, N_HEAD, N_KV, HEAD_DIM,
                                                       tokens, block, SCALE, *partials);
            });
            printf("chunk%-3u gqa-bf2 B%-5u: %8.3f ms/dispatch, %6.1f GB/s logical (%.2fx vs t2)\n",
                   tokens, block, t_bf * 1e3, chunk_gb / t_bf / 1e9, t_t2 / t_bf);
        }
    }
    return 0;
}
