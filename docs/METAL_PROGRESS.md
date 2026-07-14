# Metal implementation progress

**Updated:** 2026-07-13  
**Test device:** Apple M4, 24 GiB unified memory, 17.8 GiB recommended Metal working set

This is the execution ledger for the Metal port. CUDA remains the behavioral reference. “Baseline” means the end-to-end operation exists and has local CPU/synthetic coverage; it does not mean the optimized CUDA-equivalent gate has passed.

| # | Checkpoint | Status | Evidence / remaining gate |
|---|------------|--------|---------------------------|
| 1 | Decode primitives | **Baseline complete** | `make test-metal`, including activation quantization and quantized Q4/Q8 matvec |
| 2 | One-token Gated DeltaNet | **Baseline complete** | CPU recurrence test; persistent 64-layer state; device snapshot/restore implemented |
| 3 | FP16 attention | **Baseline complete** | CPU-reference GQA/KV test; production tiling remains |
| 4 | Full serial decode | **Baseline complete** | Official 27B artifact loads zero-copy; 16-token canonical continuation matches the live CUDA server exactly; 128-token trajectory divergence is under investigation |
| 5 | Batched prefill | **In progress** | Prompt tokens are teacher-forced in bounded eight-token command chunks; projection GEMM and layer-major recurrence remain |
| 6 | MTP widths 2/4/8/12 | **Baseline complete** | Widths 2/4/8/12 produce the same 12-token canonical output; native MTP, serial target verification, acceptance accounting, snapshots and suffix drafting are wired; batched verification remains performance work |
| 7 | Prefix cache and server | **Baseline complete** | Device snapshots (including resident logits), longest-prefix LRU, deterministic CPU top-k/top-p sampling, Metal CLI/server, and OpenAI/Anthropic endpoint smoke tests exist; streaming/tool constraints remain parity work |
| 8 | turbo3 and long context | **In progress** | Metal WHT, 50-byte codec, KV writer/reader, engine mode, synthetic quality and 32K/262K allocation gates pass; long-context retrieval/perplexity remain blocked on practical batched prefill |
| 9 | Performance | **In progress** | mmap views, command batching, activation-quantized GEMV, paired projections, fused RMSNorm+quantization, and 8-row simdgroup dispatch landed; tiled kernels remain |

## Completed foundation

- [x] Startup `.q27` and `.tok` validation before backend upload
- [x] Opaque backend buffers/tensors (`src/backend.h`)
- [x] Metal device, queue, runtime shader compilation, shared buffers
- [x] Page-aligned, no-copy `MTLBuffer` views over the model mmap
- [x] Explicit command-buffer begin/end batching
- [x] F32/F16/Q8_G128/Q4_G64 matvec correctness
- [x] Lightweight Apple Silicon regression target (`make test-metal`)

## Checkpoint details

### 1 — Decode primitives

Implemented embedding lookup, RMSNorm, per-head RMSNorm/L2 normalization, SiLU/SwiGLU, residual add, sigmoid gating, partial NeoX RoPE, FP16 KV append, concatenation, device copy, argmax, and grouped-quantized matvec.

### 2 — Gated DeltaNet

Implemented gate computation, convolution-ring update, 128×128 recurrent DeltaNet update, gated normalization, persistent state for all 48 GDN layers, and GPU-side capture/restore of recurrent/ring state.

### 3 — Attention

Implemented 24:4 GQA mapping, causal decode attention, per-head Q/K normalization, FP16 KV append, and CPU-reference tests. The baseline kernel is deliberately simple and is not FlashAttention.

### 4 — Full serial decode

`src/metal/metal_engine.cpp` validates the exact Qwen3.6-27B architecture, maps all 867 tensors without a weight copy, executes 64 base layers, applies the output head, and performs greedy decode.

Local official-artifact smoke gates:

- canonical five-token prompt `760,6511,314,9338,369` → ` Paris`;
- one-token prompt `760` → ` following`;
- CPU RSS stayed small because weights alias the mmap (earlier measured peak footprint about 274 MiB), although the GPU still streams/paginates the 17 GiB file.

The `yukon` RTX 3090 server is now available as a CUDA oracle. `tools/metal_cuda_gate.py --cuda-ssh yukon` compares committed decoded text without copying the model. The 16-token canonical continuation is byte-exact across CUDA and Metal (SHA-256 `6c1d4328...`). Post-prompt logits have matching argmax and top-10 membership, cosine `0.9999038`, RMSE `0.07029`, and max absolute error `0.2125`. At 128 tokens the paths first differ at `on` versus `along`; teacher-forcing the shared prefix makes both backends choose `along`, indicating a low-margin autoregressive/prefill numeric-path distinction rather than an architecture mismatch. Deeper layer probes remain useful.

### 5 — Prefill

Teacher-forced prompt ingestion records tokens in bounded eight-token command chunks and computes logits only for the final prompt token. This removes per-token CPU synchronization while preventing unbounded encoder growth and preserving recurrent dependency order. Multi-token projection GEMM and tiled causal attention are still required for practical long-prompt performance.

### 6 — MTP

Layer 64 is wired natively: embedding/hidden normalization, concatenation and projection, prompt-history MTP KV warming, MTP attention/FFN, shared-head norm, Q4 draft head, autoregressive draft lanes, target verification, consecutive acceptance, bonus prediction, and per-run draft/accept counters. Widths 2, 4, 8 and 12 all produced the exact same 12-token canonical continuation. They are functional gates, not speed claims: measured wall rates were respectively `0.55`, `0.01`, `0.06`, and `0.23 tok/s`, showing why serial verification is not production-ready. The backend-neutral `SuffixDraft` is also available through `--suffix 2..12`, with its CPU gate included in `make test-cpu`. Small-N multi-lane verification GEMM remains checkpoint-9 work.

### 7 — Prefix/server

`MetalEngine::capture_state` and `restore_state` copy GDN state, active KV rows, MTP KV, position, last normalized hidden state, and resident logits on-device. `build/q27-metal` supports token IDs or text prompts, deterministic temperature/top-k/top-p sampling, logit dumps, MTP, and suffix drafting. `build/q27-metal-server` adds a mutex-serialized Metal service with longest-prefix snapshot LRU and baseline `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/v1/messages`, and `/v1/responses` endpoints. The health endpoint is browser-smoke-tested. Streaming event formats, constrained tools, multi-slot scheduling, GPU-side sampling, and the full CUDA server compatibility suite remain.

### 8 — turbo3/long context

`--kv turbo3` stores each 256-dimensional KV head as two 50-byte blocks per token. This is **400 bytes per K or V token row**, versus 2,048 bytes for FP16: 5.12× smaller. The 17 attention caches (16 base + MTP), K+V, require approximately:

| Context | FP16 KV | turbo3 KV |
|---------|---------:|----------:|
| 32K | 2.1 GiB | 0.40 GiB |
| 128K | 8.5 GiB | 1.6 GiB |
| 262K | 17.0 GiB | 3.4 GiB |

The engine refuses a requested cache above half the device’s recommended Metal working set and recommends turbo3/reduced context rather than risking memory pressure. Synthetic WHT round-trip and end-to-end turbo3 attention tests run without the model artifact. Current synthetic complete-path quality is within the registered gate (observed NRMSE 0.1095, cosine 0.9940; gate NRMSE ≤0.30, cosine ≥0.95).

Turbo3 allocation/startup passes at 32K and the full 262144-token limit. KV memory is now logically cleared by resetting `position_`; rows are always written before becoming visible, avoiding an unnecessary O(context) memset. The 262K allocation-only gate starts in 0.11 s with about 207 MiB resident because reserved Metal buffers remain physically lazy. Retrieval, perplexity, and GQA=6 turbo3-K quality are not yet passed: serial token-major prefill makes those runs impractical. The upstream risk note still applies: if turbo3-K quality fails, use FP16/Q8 K plus turbo3 V rather than weakening the quality gate.

### 9 — Performance

Landed: zero-copy mmap weights, one command buffer per prompt/step, group-32 int8 activation quantization reused across sibling projections, integer-accumulating Q4/Q8 GEMV, eight independent simdgroups per threadgroup, shared-input paired projection dispatches, and fused RMSNorm+activation quantization. The 16-token CUDA gate improved from 33.35 s (`0.48 tok/s`) to 25.39 s (`0.63 tok/s`) while preserving exact output; the 128-token run averaged `1.22 tok/s`. Peak process footprint stayed about 276 MiB. Kernel throughput remains far below parity. A small-N Q4 projection prototype (12 activation rows over a `[17408,5120]` matrix) measured `36.9 ms` versus `27.6 ms` for 12 serial GEMVs (`0.75x`) and was reverted; sharing weights through threadgroup barriers was slower still. A viable matrix path needs simdgroup-matrix/packed-dot tiling rather than serial per-token reductions inside a SIMD group. Remaining high-impact work: packed SIMD dot instructions, small-N MTP verification GEMM, fused GDN, tiled decode/prefill attention, GPU-side acceptance/sampling, and Instruments attribution.

## Memory-safe test policy

The downloaded `qwen36-27b-mtp.q27` is already the smallest official tier (default 5.25 bpw, about 17 GiB). The q6/q6k files are larger, not lighter. Therefore:

1. `make test-cpu test-metal` is the default gate and uses only small synthetic buffers; it also passes on the 16 GiB `mac-mini` M4 (10.7 GiB recommended Metal working set).
2. Do not run width sweeps as separate full-model processes on the 24 GiB machine; they repeatedly page the 17 GiB mmap.
3. Run a single official-artifact smoke test only at explicit milestones.
4. Use `--kv turbo3` for long-context work; do not allocate full 262K FP16 KV.

## Commands

```sh
make test-cpu
make test-metal
make build/q27-metal build/q27-metal-server

# CUDA/Metal committed-token gate through Yukon's loopback CUDA server
./tools/metal_cuda_gate.py MODEL TOKENIZER --cuda-ssh yukon -n 16

# Full artifact smoke test (heavy weight streaming; run sparingly)
./build/q27-metal models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
  models/qwen36-27b-mtp/qwen36-27b-mtp.tok \
  --tokens 760,6511,314,9338,369 -n 1 --ctx 8

# Validate both artifacts and architecture without streaming model weights
./build/q27-metal MODEL TOKENIZER --validate-only --ctx 8

# Memory-constrained long-context mode
./build/q27-metal MODEL TOKENIZER --tokens IDS --ctx 131072 --kv turbo3
```
