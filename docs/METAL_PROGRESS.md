# Metal implementation progress

**Updated:** 2026-07-14  
**Test device:** Apple M4, 24 GiB unified memory, 17.8 GiB recommended Metal working set

This is the execution ledger for the Metal port. CUDA remains the behavioral reference. “Baseline” means the end-to-end operation exists and has local CPU/synthetic coverage; it does not mean the optimized CUDA-equivalent gate has passed.

| # | Checkpoint | Status | Evidence / remaining gate |
|---|------------|--------|---------------------------|
| 1 | Decode primitives | **Baseline complete** | `make test-metal`, including activation quantization and quantized Q4/Q8 matvec |
| 2 | One-token Gated DeltaNet | **Baseline complete** | CPU recurrence test; persistent 64-layer state; device snapshot/restore implemented |
| 3 | FP16 attention | **Baseline complete** | CPU-reference GQA/KV test; production tiling remains |
| 4 | Full serial decode | **Baseline complete** | Official 27B artifact loads zero-copy; 16-token canonical continuation matches the live CUDA server exactly; 128-token trajectory divergence is under investigation |
| 5 | Batched prefill | **Baseline complete** | Layer-major 2–12-token chunked prefill is integrated end-to-end through the simdgroup projection GEMM; chunk-aware GDN recurrence and causal attention match the serial path (op gates bit-exact, official-artifact A/B committed tokens identical, 4.3× prefill wall-clock); production tiled attention remains |
| 6 | MTP widths 2/4/8/12 | **Baseline complete** | Batched layer-major target verification with one CPU sync per round replaced serial verification; canonical committed tokens identical to greedy (byte-exact over a 32-token trajectory); wall rate at greedy parity (`0.71` vs `0.73 tok/s`) versus the `0.01–0.55 tok/s` serial-verify sweep; GPU-resident drafting/acceptance and removing the partial-round commit re-encode remain performance work |
| 7 | Prefix cache and server | **Baseline complete** | Device snapshots (including resident logits), longest-prefix LRU, deterministic CPU top-k/top-p sampling, Metal CLI/server, and OpenAI/Anthropic endpoint smoke tests exist; streaming/tool constraints remain parity work |
| 8 | turbo3 and long context | **In progress** | Metal WHT, 50-byte codec, KV writer/reader, engine mode, synthetic quality and 32K/262K allocation gates pass; chunked prefill now covers turbo3 KV (A/B identical), but 32K+ retrieval/perplexity still need more prefill throughput than the current ~3.6 tok/s |
| 9 | Performance | **In progress** | mmap views, command batching, paired projections, fused RMSNorm+quantization, GEMV, and a 3.29x N=12 simdgroup projection kernel landed; engine-level tiled scheduling remains |

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

Prompt ingestion is now layer-major: `MetalEngine::encode_chunk` advances 2–12 tokens per command buffer through batched embedding, per-row fused RMSNorm+quantization, the 1–12-row simdgroup projection GEMM, a chunk-sequential convolution-ring kernel, a chunk-sequential DeltaNet kernel that keeps the 128×128 state in registers and commits it once per chunk, per-token-position RoPE, batched FP16/turbo3 KV append, and causal chunk attention (token *t* attends to the warm cache plus in-chunk keys `0..t`). The final prompt token always runs the serial path so it produces logits and leaves the last normalized hidden state for MTP drafting and prefix snapshots; MTP-warmed prompts (`generate_mtp`) stay fully serial because warming needs every token's final hidden state. `--prefill serial` restores the old path for A/B comparison.

Gates: every chunked operation has a serial-reference parity test in `make test-metal` (most are bit-exact, including the conv ring and turbo3 store bytes); official-artifact A/B runs produced identical committed tokens for the canonical 16-token FP16 continuation, a 126-token prompt, and a turbo3 run. The 126-token prompt prefilled in ~35 s chunked versus ~153 s serial (4.3×, about 3.6 tok/s prefill). Causal chunk attention still materializes probabilities in a reserved `12 × heads × context` scratch buffer (physically lazy until long prompts touch it); online-softmax tiled attention in critical-path item 4 removes it.

### 6 — MTP

Layer 64 is wired natively: embedding/hidden normalization, concatenation and projection, prompt-history MTP KV warming, MTP attention/FFN, shared-head norm, Q4 draft head, autoregressive draft lanes, target verification, consecutive acceptance, bonus prediction, and per-run draft/accept counters. The backend-neutral `SuffixDraft` is also available through `--suffix 2..12`, with its CPU gate included in `make test-cpu`.

Target verification is now batched (2026-07-14). Each round drafts serially through layer 64, then verifies every lane — pending token plus drafts — in a single optimistic committing layer-major chunk with a batched output head and per-lane argmax: one CPU synchronization per round instead of one per committed token. All 48 GDN recurrent/ring states (~157 MiB) are checkpointed on-device before the chunk; full acceptance keeps the state, partial acceptance restores the checkpoint and re-encodes only the accepted prefix. Committed tokens follow the exact serial-walk semantics (the final output token is never encoded; `x1_`/logits describe the last encoded token), so snapshots and continuations stay compatible. The draft width adapts to measured acceptance starting from 4 — committed output is width-invariant, and a wide first round otherwise pays for many serial drafts through a cold draft head (the recorded serial sweep's `0.01 tok/s` outlier was this pathology).

Gates: widths 4, 8 and 12 produce the canonical greedy output, and a 32-token `--mtp 8` run matched greedy byte-exactly at 48.6% draft acceptance, `0.71` versus `0.73 tok/s` — parity with greedy where serial verification ran `0.01–0.55 tok/s`. `Q27_MTP_TRACE=1` prints per-round draft/verify/commit timing: currently ~1.0 s verify plus ~0.9 s commit re-encode per partial round. The re-encode is the main structural overhead; removing it (per-token GDN state capture or GPU-side acceptance fused into the chunk) and GPU-resident drafting are the remaining levers.

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

Landed: zero-copy mmap weights, one command buffer per prompt/step, group-32 int8 activation quantization reused across sibling projections, integer-accumulating Q4/Q8 GEMV, eight independent simdgroups per threadgroup, shared-input paired projection dispatches, and fused RMSNorm+activation quantization. The 16-token CUDA gate improved from 33.35 s (`0.48 tok/s`) to 25.39 s (`0.63 tok/s`) while preserving exact output; the 128-token run averaged `1.22 tok/s`. Peak process footprint stayed about 276 MiB. Kernel throughput remains far below parity. Initial small-N Q4 prototypes that performed serial per-token SIMD reductions measured `0.62–0.75x` versus serial GEMV and were reverted. Replacing them with 8×8 float `simdgroup_matrix` tiles and on-tile Q4/Q8 dequantization changed the result: 12 activation rows over `[17408,5120]` measure `8.36 ms` versus `27.49 ms` for 12 serial GEMVs (`3.29x`). The 1–12-row primitive has Q4/Q8 CPU-reference tests and passes on both M4 hosts. Layer-major chunked prefill now schedules it end-to-end: a 126-token prompt ingests in ~35 s versus ~153 s token-serial (4.3×, ~3.6 tok/s prefill) with identical committed tokens. Remaining high-impact work: batched MTP verification on the same substrate, packed SIMD dot instructions for GEMV, fused GDN, tiled attention, GPU-side acceptance/sampling, and Instruments attribution.

## Mature-decode critical path

The remaining work should proceed in this order; isolated kernel wins do not close a checkpoint until the engine schedules them end-to-end.

1. **Layer-major 8–12 token execution:** ✅ done (2026-07-14). Batched embeddings/RMSNorm/quantization, projections through the simdgroup GEMM, chunk-aware GDN recurrence and causal attention, state committed at chunk boundaries. MTP-warmed prompts remain serial until item 2 provides batched layer-64 execution.
2. **Batched MTP verification:** ✅ done (2026-07-14), except the GPU acceptance walk. Candidate lanes verify in one layer-major chunk, GDN state is checkpointed per round and only accepted rows survive, width adapts to measured acceptance, and committed tokens are byte-identical to greedy. The acceptance walk stays on the CPU (one small read per round); moving it to the GPU only pays once drafting is GPU-resident, which is item-3 work alongside removing the partial-round commit re-encode.
3. **Greedy hot-path optimization:** profile GPU time and effective bandwidth, add packed-dot Q4/Q8 GEMV, fuse GDN stages and residual/normalization boundaries, and reduce per-token dispatch count. A provisional base-M4 maturity target is `3–5 tok/s` or at least 50% of the measured sustainable bandwidth roofline.
4. **Production attention:** add online-softmax tiled FP16/turbo3 decode, GQA KV reuse, chunk-causal prefill attention, and long-context cache-block scheduling.
5. **Quality closure:** collect deeper CUDA layer/state/top-k probes, then run 32K/128K/262K retrieval, perplexity, and turbo3 quality gates. Numerical tolerance is allowed across architectures; semantic, ranking, and committed-token regressions are not.
6. **Serving closure:** add true token streaming, GPU sampling, tool constraints, stop handling, cancellation/backpressure, multi-slot scheduling, and the CUDA server compatibility suite.
7. **Hardware validation:** benchmark cold/warm short decode, long decode, prefill, and MTP on base M4 and available Max/Ultra-class machines; report memory mode and effective bandwidth with every result.

The immediate implementation target is item 3. Chunk passes currently sustain roughly 15 GB/s of effective weight bandwidth against a much higher hardware roofline, the partial-acceptance commit re-encode doubles MTP round cost, and drafting still runs one synchronized command buffer per lane — all three are hot-path scheduling/bandwidth work.

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

# A/B the layer-major chunked prompt ingestion against the token-serial path
./build/q27-metal MODEL TOKENIZER --prompt TEXT -n 1 --prefill serial
```
