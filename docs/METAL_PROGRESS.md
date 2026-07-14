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
| 8 | turbo3 and long context | **In progress** | Metal WHT, 50-byte codec, KV writer/reader, engine mode, synthetic quality and 32K/262K allocation gates pass; chunked prefill covers turbo3 KV (A/B identical) and both turbo3 attention kernels are now online-softmax and flat in context (synthetic: 22.9 tok/s chunk prefill at ctx 480, 4.4 tok/s decode at ctx 1024) — the 32K retrieval/perplexity runs are unblocked and are the next gate |
| 9 | Performance | **In progress** | mmap views, command batching, fused RMSNorm+quantization, packed-dot GEMV (aggregate 35.6→84.8 GB/s synthetic; 128-token decode 1.22→3.01 tok/s, inside the 3–5 tok/s target band), per-dispatch GPU profiling (`Q27_METAL_PROFILE`), synthetic full-decode-step and full-prefill-chunk benches, online-softmax decode **and chunk-causal** attention, widened F16 pair projections, and a tiled simdgroup-matrix chunk GEMM landed; resident-weight decode ceiling 218 ms/token, prefill-chunk ceiling 2121→525 ms/chunk (5.7→22.9 tok/s synthetic); turbo3 decode/chunk attention now online-softmax too (`--kv turbo3` bench modes: decode 732→227 ms/token at ctx 1024, chunk 884→525 ms/chunk at ctx 480, both flat in context); GQA KV reuse and remaining GEMM/GEMV headroom remain |

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

Implemented 24:4 GQA mapping, causal decode attention, per-head Q/K normalization, FP16 KV append, and CPU-reference tests. The FP16 decode kernel is now online-softmax (2026-07-14): one threadgroup per query head, eight simdgroups stripe the sequence with running max/denominator/weighted-value kept in registers, partials merge in log2 rounds through threadgroup memory, and no probability scratch is materialized. Parity gates cover the production shape (24:4, head_dim 256) at sequence lengths that exercise empty stripes and running-max updates. The chunked FP16 causal prefill kernel now mirrors the same online-softmax structure exactly, one threadgroup per (query head, chunk token), so a chunk token's attention output is bit-identical to the serial decode kernel and the probability scratch is gone from the FP16 path (2026-07-14 evening; production-shape chunked-vs-serial gates at warm 2 and 130 are bit-exact). The turbo3 decode and chunk kernels followed on 2026-07-14 (late): both mirror the same online-softmax structure with the 50-byte turbo3 blocks dequantized on the fly, chunk output is bit-identical to turbo3 decode at equal sequence lengths, and no attention kernel materializes probabilities anymore — the scratch buffers, their engine allocations, and the `scratch` parameters are deleted from the backend API (shader ABI 4→5). Gates: a production-shape turbo3 CPU-reference test (24:4, head_dim 256, seq 5/133, dequantizing the stored blocks so the kernel math is checked exactly), the chunked-vs-serial turbo3 gate upgraded to bit-exact, the synthetic quality gate unchanged (NRMSE 0.10952, cosine 0.99399), the canonical 16-token CUDA gate byte-exact (SHA `6c1d4328...`), and a turbo3 chunked-vs-serial artifact A/B with identical committed tokens.

### 4 — Full serial decode

`src/metal/metal_engine.cpp` validates the exact Qwen3.6-27B architecture, maps all 867 tensors without a weight copy, executes 64 base layers, applies the output head, and performs greedy decode.

Local official-artifact smoke gates:

- canonical five-token prompt `760,6511,314,9338,369` → ` Paris`;
- one-token prompt `760` → ` following`;
- CPU RSS stayed small because weights alias the mmap (earlier measured peak footprint about 274 MiB), although the GPU still streams/paginates the 17 GiB file.

The `yukon` RTX 3090 server is now available as a CUDA oracle. `tools/metal_cuda_gate.py --cuda-ssh yukon` compares committed decoded text without copying the model. The 16-token canonical continuation is byte-exact across CUDA and Metal (SHA-256 `6c1d4328...`). Post-prompt logits have matching argmax and top-10 membership, cosine `0.9999038`, RMSE `0.07029`, and max absolute error `0.2125`. At 128 tokens the paths first differ at `on` versus `along`; teacher-forcing the shared prefix makes both backends choose `along`, indicating a low-margin autoregressive/prefill numeric-path distinction rather than an architecture mismatch. Deeper layer probes remain useful. (The 2026-07-14 packed-dot GEMV changed float accumulation order, so the exact 128-token divergence point may have moved; the 16-token canonical gate was re-verified byte-exact after that change.)

### 5 — Prefill

Prompt ingestion is now layer-major: `MetalEngine::encode_chunk` advances 2–12 tokens per command buffer through batched embedding, per-row fused RMSNorm+quantization, the 1–12-row simdgroup projection GEMM, a chunk-sequential convolution-ring kernel, a chunk-sequential DeltaNet kernel that keeps the 128×128 state in registers and commits it once per chunk, per-token-position RoPE, batched FP16/turbo3 KV append, and causal chunk attention (token *t* attends to the warm cache plus in-chunk keys `0..t`). The final prompt token always runs the serial path so it produces logits and leaves the last normalized hidden state for MTP drafting and prefix snapshots; MTP-warmed prompts (`generate_mtp`) stay fully serial because warming needs every token's final hidden state. `--prefill serial` restores the old path for A/B comparison.

Gates: every chunked operation has a serial-reference parity test in `make test-metal` (most are bit-exact, including the conv ring, turbo3 store bytes, and now chunk-causal FP16 attention; the chunk GEMM is tolerance-gated, see item 9); official-artifact A/B runs produced identical committed tokens for the canonical 16-token FP16 continuation, a 126-token prompt, and a turbo3 run, and the canonical 16-token CUDA gate re-passed byte-exact after the 2026-07-14 GEMM/attention rewrites. The 126-token prompt prefilled in ~35 s chunked versus ~153 s serial (4.3×, about 3.6 tok/s prefill) before those rewrites; the synthetic resident-weight chunk ceiling is now 22.9 tok/s, so the next artifact prefill measurement should be paging-dominated. No attention path materializes probabilities anymore; the reserved `12 × heads × context` scratch buffer and its allocation are deleted (2026-07-14 late, with the turbo3 online-softmax rewrite).

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

Turbo3 allocation/startup passes at 32K and the full 262144-token limit. KV memory is now logically cleared by resetting `position_`; rows are always written before becoming visible, avoiding an unnecessary O(context) memset. The 262K allocation-only gate starts in 0.11 s with about 207 MiB resident because reserved Metal buffers remain physically lazy. Retrieval, perplexity, and GQA=6 turbo3-K quality are not yet passed, but the throughput blocker is gone: turbo3 prefill and decode attention are online-softmax and flat in context (2026-07-14 late; see item 9), so 32K runs at ~20 tok/s synthetic prefill are now practical. The upstream risk note still applies: if turbo3-K quality fails, use FP16/Q8 K plus turbo3 V rather than weakening the quality gate.

### 9 — Performance

Landed: zero-copy mmap weights, one command buffer per prompt/step, group-32 int8 activation quantization reused across sibling projections, integer-accumulating Q4/Q8 GEMV, eight independent simdgroups per threadgroup, and fused RMSNorm+activation quantization. Initial small-N Q4 prototypes that performed serial per-token SIMD reductions measured `0.62–0.75x` versus serial GEMV and were reverted. Replacing them with 8×8 float `simdgroup_matrix` tiles and on-tile Q4/Q8 dequantization changed the result: 12 activation rows over `[17408,5120]` measure `8.36 ms` versus `27.49 ms` for 12 serial GEMVs (`3.29x`). The 1–12-row primitive has Q4/Q8 CPU-reference tests and passes on both M4 hosts. Layer-major chunked prefill now schedules it end-to-end: a 126-token prompt ingests in ~35 s versus ~153 s token-serial (4.3×, ~3.6 tok/s prefill) with identical committed tokens. Batched MTP verification runs on the same substrate at greedy parity.

**Packed-dot GEMV rewrite (2026-07-14).** The original decode GEMV kernels loaded one weight byte per lane and ran a full 32-lane `simd_sum` per 32 columns — instruction-issue-bound, not bandwidth-bound; `build/metal_gemv_bench` (synthetic, memory-safe) measured Q4 at `20.5 GB/s` and Q8 at `40 GB/s` with identical wall time per op for both dtypes. The rewrite loads 16 weight values per lane (`int4`/packed `uint2`), computes exact integer dots against 16 int8 activations, applies one combined scale per lane, and reduces once per row. Steady-state: Q4 `67–70 GB/s` (3.3x), Q8 `86–90 GB/s` (2.2x), aggregate `35.6 → 84.8 GB/s` — roughly 70–85% of the M4's practical stream bandwidth. A vectorized mask/shuffle Q4 nibble decode measured *slower* (39–44 GB/s) than scalar shift-extract and was rejected. The fused quantized pair kernels were retired: with the reduction bottleneck gone, two packed single dispatches (`67–69 GB/s`) beat the fused pair (`47–55 GB/s`, doubled register pressure); `matvec_quantized_pair` survives as an entry point issuing two dispatches. Decode: the 16-token cold CUDA gate improved `25.39 s → 15.77 s` (`0.63 → 1.01 tok/s` including first-touch weight paging) with byte-exact committed text (SHA `6c1d4328...` re-verified), and a warm 128-token official-artifact run improved `1.22 → 3.01 tok/s`, inside the provisional `3–5 tok/s` maturity band. Peak process footprint stayed small (weights alias the mmap).

**Incident and new gate.** The first packed rewrite shipped a per-lane activation-scale indexing bug (each lane's second 16-column half read the *next lane's* scale). Every synthetic test passed — the narrow 64/128-column shapes never reach the vectorized main loop, which requires ≥1024 columns — and only the official-artifact CUDA gate caught the garbage output. `test_metal` now includes `test_quantized_wide`: Q4/Q8 GEMV parity at 1088/1152/5120 columns with deliberately varied per-group weight scales and per-32-column activation scales, verified to fail against the buggy kernel. Lesson recorded: any kernel with a width-gated fast path needs a gate shape that actually enters it. Note the packed accumulation order changes float rounding, so low-margin greedy trajectories beyond the canonical 16 tokens can legitimately shift versus earlier Metal runs.

**Decode attribution and hot-path fixes (2026-07-14, second pass).** Two attribution tools landed. `Q27_METAL_PROFILE=1` makes the backend run every dispatch in its own compute encoder bracketed by stage-boundary timestamp samples and prints a per-kernel GPU-time table at teardown (overlap is suppressed in this mode, so shares — not absolute wall — are the signal; overhead measured ~10%). `build/metal_decode_bench` replays the exact per-token dispatch sequence of serial decode against resident synthetic weights at production shapes, isolating kernel cost from mmap paging. The measured token at ctx 128 (13.28 GiB synthetic weight stream): Q4 GEMV 194 ms (77.6%), naive decode attention 29.2 ms (11.7%, `1.82 ms` per dispatch, linear in context), Q8 output head 15.3 ms (~84 GB/s, healthy), F16 alpha/beta pair 4.4 ms (10.8 GB/s), everything else ~5 ms; GPU-busy ≈ CPU-wait, so dispatch/sync overhead is a non-issue. The resident-weight ceiling was 245 ms/token (4.08 tok/s) against 330 ms observed on the artifact — ~85 ms/token of the real run is weight paging, not kernels.

Fixes from that data: decode FP16 attention is now online-softmax (one threadgroup per query head, eight simdgroups striping the sequence with running max/denominator/value in registers, log2 merge through threadgroup memory, no probability scratch) — `1824 → 65 µs` per dispatch at ctx 128, and a ctx-4096 decode step now costs about the same as ctx-128 (247 vs 217 ms/token) where the serial kernel would have added ~900 ms/token. The F16 pair projections moved from simdgroup-per-row (1,536 threads total for the 48-row alpha/beta shape) to a 256-thread threadgroup per row with `packed_half4` dots — `87 → 17 µs` per dispatch — and the chunked rows variant mirrors the same accumulation structure exactly so chunked and serial results stay bit-identical. Resident-weight ceiling after both: `218 ms/token (4.59 tok/s)`. Gates: production-shape attention parity tests (24:4 GQA, head_dim 256, seq 5 and 133, running-max swings), a wide F16 pair gate (48×5120 plus a scalar-tail width), the canonical 16-token CUDA gate byte-exact (SHA `6c1d4328...` re-verified), and chunked-vs-serial prefill A/B identical over the same 16 tokens.

**Second incident, same shape as the first.** The initial post-change CUDA gate FAILED with coherent-but-wrong text, and two days of kernels were nearly blamed: the gate had run a **stale `build/q27-metal`**. Shaders compile from `q27_kernels.metal` at *runtime*, so the old host binary silently bound the new attention kernel's buffers at the old indices (attention output landed in the dead scratch buffer). Unit binaries had been rebuilt; the CLI had not. `q27_kernels.metal` now carries a `Q27_SHADER_ABI` tag that `metal_backend.mm` verifies at library load — a host/shader layout mismatch now fails at startup with a rebuild instruction instead of decoding garbage. Lesson recorded: rebuild every binary before an artifact gate, and bump the ABI tag on any buffer-index or argument-struct change.

Remaining high-impact work, reordered by measured leverage: (1) prefill: Instruments/profiler attribution of the ~15 GB/s effective chunk bandwidth, then tiled chunk-causal prefill attention (the profiler now works for prefill runs too); (2) rest of item 4: online-softmax turbo3 decode attention, GQA KV reuse (six query heads re-read each KV row today), and long-context cache-block scheduling; (3) remaining GEMV headroom toward ~100 GB/s (the Q4 aggregate inside a full token measures ~62 GB/s against 67–70 GB/s in isolation); (4) removing the partial-round MTP commit re-encode and GPU-resident drafting/acceptance/sampling.

**Prefill attribution and the chunk GEMM rewrite (2026-07-14, third pass).** `build/metal_prefill_bench` replays the exact `chunk_forward` dispatch sequence against resident synthetic weights (memory-safe, ~1.6 GiB), the prefill counterpart of `metal_decode_bench`. Attribution was unambiguous: the original 8×8 simdgroup-matrix chunk GEMM was **96% of prefill GPU time** (6.6 ms per dispatch, ~7 GB/s effective on FFN shapes; chunk attention was under 2% at short contexts). Two rewrites followed. A packed-dot scalar GEMM mirroring the GEMV structure (bit-exact against the serial path) recovered only 11%: with 12 token accumulators it is scalar-issue-bound at ~4 ops per multiply-add — two char→int converts per MAC dominate — and a 4-row register-blocked variant spilled and ran 1.6× *slower*. The landed kernel (`q27_matmul_q4_mm`/`q8_mm`) is a tiled simdgroup-matrix GEMM: 128-thread threadgroups stage 64-column K-tiles in threadgroup memory (weights as raw dequantized integers, activations prescaled by their group scale) and accumulate a 32-row × 16-token tile with 8×8×8 `simdgroup_multiply_accumulate`, flushing through per-simdgroup scratch once per weight-scale group — the matrix unit converts operands once per staged tile instead of once per MAC. Chunk ceiling: `2121 → 525 ms/chunk` (`5.7 → 22.9 tok/s` synthetic prefill at chunk 12); GEMM dispatch `6.6 → 1.3 ms`. A 64-row/256-thread variant measured within noise and was rejected for the occupancy-safer 32-row tile. Numerics: chunked projections are no longer bit-exact against the serial GEMV (K-tile float accumulation order differs; the weight side stays integer-exact, the activation side rounds once per value at staging), so the matmul op gates are tolerance-based (3e-4) with wide shapes that enter the tiled main path, and the end-to-end gates carry exactness: the canonical 16-token CUDA gate re-passed byte-exact (SHA `6c1d4328...`) after both rewrites. The same pass replaced the chunk-causal FP16 attention baseline — which serialized scores on thread 0 through the probability scratch — with the decode kernel's online-softmax structure (bit-exact against serial decode attention at production shapes, gates at warm 2/130); per-dispatch cost stays ~0.6 ms and under 2% share out to ctx 480 where the old kernel scaled linearly per token. Shader ABI 2→4 across the two commits (matmul dispatch geometry, then attention buffer bindings).

**Turbo3 attention online-softmax (2026-07-14, late).** Both benches gained `--kv turbo3` modes that replay the engine's turbo3 dispatch sequence (WHT on Q, 50-byte KV store, turbo3 attention, inverse WHT), committed first so the probability-scratch kernels were measured before replacement: decode attention was 32.9 ms/dispatch at ctx 1024 (71.7% of token time, 272→732 ms/token from ctx 128→1024) and chunk-causal attention 25.0 ms/dispatch at ctx ~480 (45.2% of chunk time) — both linear in context, which is what made 32K turbo3 runs impractical. The rewrite mirrors the FP16 online-softmax structure in both kernels with turbo3 blocks dequantized on the fly. After: decode 0.57 ms/dispatch (3.1% share) and 226–227 ms/token flat from ctx 128 to 1024 (4.4 tok/s synthetic); chunk 1.54 ms/dispatch (3.2% share) and 512–525 ms/chunk flat from ctx 120 to 480 — the turbo3 chunk ceiling now equals the FP16 ceiling (22.9 tok/s). With no probability-scratch user left, the scratch buffers, engine allocations (`attn_scratch_`, `cattn_scratch_`), and the `scratch` parameters on all three attention entry points are deleted; shader ABI 4→5 (turbo3 attention buffer bindings). Gates: production-shape turbo3 CPU-reference test that dequantizes the stored blocks (seq 5/133, exercising empty stripes and running-max swings), chunked-vs-serial turbo3 parity upgraded to bit-exact, synthetic quality gate unchanged (NRMSE 0.10952, cosine 0.99399), canonical 16-token CUDA gate byte-exact (SHA `6c1d4328...`), and a turbo3 chunked-vs-serial artifact A/B with identical committed tokens. Note the turbo3 accumulation order changed, so turbo3 trajectories may legitimately shift versus earlier turbo3 runs (the FP16 path is untouched).

## Mature-decode critical path

The remaining work should proceed in this order; isolated kernel wins do not close a checkpoint until the engine schedules them end-to-end.

1. **Layer-major 8–12 token execution:** ✅ done (2026-07-14). Batched embeddings/RMSNorm/quantization, projections through the simdgroup GEMM, chunk-aware GDN recurrence and causal attention, state committed at chunk boundaries. MTP-warmed prompts remain serial until item 2 provides batched layer-64 execution.
2. **Batched MTP verification:** ✅ done (2026-07-14), except the GPU acceptance walk. Candidate lanes verify in one layer-major chunk, GDN state is checkpointed per round and only accepted rows survive, width adapts to measured acceptance, and committed tokens are byte-identical to greedy. The acceptance walk stays on the CPU (one small read per round); moving it to the GPU only pays once drafting is GPU-resident, which is item-3 work alongside removing the partial-round commit re-encode.
3. **Greedy hot-path optimization:** ✅ done (2026-07-14). Packed-dot Q4/Q8 GEMV landed (synthetic aggregate `35.6 → 84.8 GB/s`; warm 128-token decode `1.22 → 3.01 tok/s`). The non-GEMV residual is now attributed (`Q27_METAL_PROFILE` + `build/metal_decode_bench`): attention and the F16 pair were the offenders and both are fixed; ~85 ms/token of the observed artifact rate is weight paging, and the resident-weight kernel ceiling is `218 ms/token (4.59 tok/s)`.
4. **Production attention:** decode FP16 online-softmax landed (2026-07-14, `1824 → 65 µs` per dispatch at ctx 128, flat to ctx 4096, no probability scratch), the chunk-causal FP16 prefill kernel mirrors it (2026-07-14 evening; bit-exact vs serial decode attention, under 2% of chunk time out to ctx 480), and the turbo3 decode and chunk kernels followed (2026-07-14 late: `32.9 → 0.57 ms` and `25.0 → 1.54 ms` per dispatch, both flat in context, probability scratch fully deleted from the codebase, ABI 4→5). The prefill-side GEMM attribution and rewrite also landed (see item 9: `5.7 → 22.9 tok/s` synthetic chunk ceiling). Remaining: GQA KV reuse and long-context cache-block scheduling.
5. **Quality closure:** collect deeper CUDA layer/state/top-k probes, then run 32K/128K/262K retrieval, perplexity, and turbo3 quality gates. Numerical tolerance is allowed across architectures; semantic, ranking, and committed-token regressions are not.
6. **Serving closure:** add true token streaming, GPU sampling, tool constraints, stop handling, cancellation/backpressure, multi-slot scheduling, and the CUDA server compatibility suite.
7. **Hardware validation:** benchmark cold/warm short decode, long decode, prefill, and MTP on base M4 and available Max/Ultra-class machines; report memory mode and effective bandwidth with every result.

Item 4's kernel work is done on every path (FP16 and turbo3, decode and chunk; only GQA KV reuse and cache-block scheduling remain). The next targets, in leverage order: (1) the item-5 quality closure runs that the turbo3 rewrite unblocked — 32K retrieval and perplexity under `--kv turbo3` at ~20 tok/s synthetic prefill, plus deeper CUDA probes for the 128-token trajectory question; (2) the MTP partial-acceptance commit re-encode and GPU-resident drafting (item 6 details); (3) GQA KV reuse (six query heads re-read each KV row today) and residual chunk-GEMM headroom (still ~25 GB/s effective weight stream against 67–90 GB/s for decode GEMV — half-precision staging and double-buffered K-tiles are the untried levers).

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

# Synthetic decode-GEMV bandwidth attribution (memory-safe, no model artifact)
make build/metal_gemv_bench && ./build/metal_gemv_bench

# Full synthetic decode step at production shapes (memory-safe, ~1.6 GiB);
# resident-weight kernel ceiling, isolates kernel cost from mmap paging.
# Add --kv turbo3 to replay the turbo3 KV dispatch sequence instead of FP16.
make build/metal_decode_bench && ./build/metal_decode_bench --tokens 16 --seq 128

# Full synthetic prefill chunk sequence at production shapes (memory-safe,
# ~1.6 GiB); resident-weight chunk ceiling, prefill counterpart of the above
# (also takes --kv turbo3)
make build/metal_prefill_bench && ./build/metal_prefill_bench --prompt 120 --chunk 12

# Per-kernel GPU-time attribution for any Metal run (per-op encoders,
# ~10% overhead; shares are the signal, table prints at process exit)
Q27_METAL_PROFILE=1 ./build/metal_decode_bench --tokens 8 --seq 128

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
