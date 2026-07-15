# Ternary weight tier (Bonsai g128) for the Metal engine

**Status: proposed** (2026-07-14). Follows the Bonsai 27B whitepaper review (PrismML,
`bonsai-27b-whitepaper.pdf`, July 2026): ternary/binary end-to-end quantizations of the
same Qwen3.6-27B base we target, Apache-licensed, shipped as GGUF and MLX packs.

## Motivation

The 24 GiB M4's decode rate is footprint-bound, not kernel-bound: our GEMV runs at
~85 GB/s but streams a 17 GiB artifact that cannot stay resident, and ~85 ms/token of
observed decode is weight paging. No remaining kernel work moves that wall. A ternary
artifact moves the wall directly:

| tier            | true bpw | streamed/token | resident on 24 GiB? | ceiling @ ~85 GB/s |
|-----------------|----------|----------------|---------------------|--------------------|
| official 5.25   | 5.25     | ~17 GB         | no (pages)          | ~4.6 tok/s (measured ceiling; ~3.0 observed) |
| Bonsai ternary  | ~2.25 deployed (2-bit slots + scales) | ~7.0 GB | yes, with room for 32K turbo3 KV + OS | **~11–12 tok/s** |

Their own Apple deployment is a llama.cpp fork: 4-bit KV, no working speculation on
Apple Silicon (batch-1 verification "does not yet amortize"), 18 tok/s on an M4 **Pro**
(~2.3× our memory bandwidth). Our engine brings sub-2-bit turbo3 KV (1.56 bits/value,
32K depth-flat as of tonight's NLL run), layer-major chunked prefill, and the gate
discipline. Their bytes in our engine plausibly beats their stack on the same hardware.

## Dual-tier support is native to the design

No fork, no mode switch. `dtype` is **per-tensor** in the q27 container (FORMAT.md) and
the engine already mixes F16/Q8_G128/Q4_G64 per tensor; the artifact chosen at load time
determines everything downstream. A ternary artifact is just a `.q27` file whose matrix
weights carry a new dtype. One binary serves either artifact; every gate, the server,
the CLI, the NLL harness, and the benches work unchanged. Shared wholesale: attention
(FP16 + turbo3), GDN chunk machinery, layer-major prefill, engine, server, tokenizer
(same base model — verify vocab identity in Phase 1). New: one dtype, its repack path,
and the weight-side inner loops of GEMV/GEMM.

Known asymmetries of the ternary tier, accepted up front:

1. **No MTP head.** Bonsai's packs almost certainly drop the MTP head (their speculation
   is a separate DSpark drafter; llama.cpp has no MTP). Day one: greedy + **suffix
   drafting** (`generate_suffix`, model-free, works unchanged). Optional experiment
   later: graft the official artifact's MTP head onto the ternary trunk — verification
   is lossless so correctness is free; acceptance rate against ternary hidden states is
   the empirical question.
2. **No byte-exact oracle.** The CUDA 16-token gate pins the official tier to the CUDA
   reference byte-for-byte. For ternary there is no same-weights reference in our stack;
   gates become (a) bit-exact unit parity against CPU references for the new kernels,
   (b) committed-token comparison against their llama.cpp fork running the same pack
   (text-level, tolerance mindset — accumulation orders differ), and (c) on-device NLL
   A/B against the official tier.

## Phase 0 — validation spike (one afternoon, go/no-go)

Download the ternary GGUF (~7.2–7.6 GB; try mainline llama.cpp first, their
`PrismML-Eng/llama.cpp` fork if the Q2_0 pack needs it) and run it as a user on this
M4. Answers before any engineering: does decode actually land near the bandwidth
prediction; do *our* prompts (agentic, instruction-heavy — their weakest categories:
IFBench 68→58.5, τ²-Bench 82.9→73.6) hold up subjectively; does perplexity on our
wikitext tokens look sane via llama.cpp's own tooling. **No-go** if quality is
unusable for our purposes — file the findings, stay on course. Machine rule: 7.2 + 17
GiB > 24 GiB — the two artifacts are NEVER resident simultaneously; all A/Bs serial.

### Phase 0 verdict (2026-07-14, night): **GO**

Run on the base M4 (24 GiB), Ternary-Bonsai-27B Q2_0 7.17 GB pack, PrismML fork
binaries `prism-b9591-62061f9` (mainline b9960 cannot run the packs — their "Q2_0" is
a fork-only ggml type, id 42, aborts in the Metal backend; offsets also reveal the
7.17 GB pack is g128-scaled under a g64-sized type name. The Q2_g64 pack loads in
mainline but still aborts on compute. Their fork binaries are the only stock path,
and the authoritative dequant reference for Phase 1).

- **Throughput (their stack):** quiet machine (2026-07-14, post-reboot, fork
  `prism-b9591-62061f9`): tg128 `8.41 ± 1.36` tok/s, pp512 `52.48 ± 1.64` tok/s —
  right at the whitepaper's M4 Pro figure scaled by ~2.3× bandwidth (predicts ~8).
  Effective weight stream ≈ 6.66 GiB × 8.41 ≈ 60 GB/s before KV/activation traffic,
  vs our decode GEMV's ~85 GB/s. (The original loaded-machine run — tg128
  `5.17 ± 0.15`, pp512 `24.2 ± 1.8`, desktop load + swap debt from the day's 17 GiB
  runs — is superseded; quote the quiet numbers.) The q27 upside is real but
  narrower than the loaded run suggested: ~85 GB/s GEMV efficiency projects to
  ~11–12 tok/s, roughly **1.3–1.4×** their stack on this hardware, plus what our
  prefill/turbo3-KV/suffix-drafting advantages add on top.
- **Behavioral probes (thinking mode, their llama-server): all pass.** JSON-only
  constrained output: exact. 7-constraint list-formatting probe (their weakest
  category): every constraint honored. Native OpenAI tool call: correct function,
  correct arguments, no leakage. `merge_intervals` with a planted touching-intervals
  edge case: correct.
- **Wikitext-2 PPL (ctx-512 chunks, 24 chunks): `10.04 ± 0.36`.** Same corpus and
  tokenizer as our 32K gate (official artifact: 5.318 at 32K single-pass, 4.90 in the
  0–2k bucket). Protocols differ, but the direction is clear: **roughly 1.7–1.9× the
  perplexity of the full-quality tier** — a real LM-quality gap, larger than the
  whitepaper's benchmark-retention framing suggests (they never report weight-quant
  PPL), yet clearly not the collapse regime given the behavioral results.
- **Server-build quirks recorded:** their `llama-cli` enters interactive mode despite
  `-no-cnv` (use llama-server or llama-bench); per-request `thinking_budget_tokens`
  is ignored on this build; thinking cannot be disabled per-request.

Verdict rationale: the two-tier architecture prices this correctly — ternary buys
resident decode and ~2-4× throughput on 24 GiB machines at a real but non-collapsing
quality cost; the official artifact remains the quality tier. Proceed to Phase 1.
Phase 3's gate 3 (32K NLL A/B on our own harness, same protocol both tiers) upgrades
the PPL comparison from indicative to definitive.

### Phase 1 kickoff notes (post-reboot re-setup)

- Models already on disk (keep both): `models/ternary-bonsai-27b/Ternary-Bonsai-27B-Q2_0.gguf`
  (7.17 GB, native g128 — the Phase 1 repack source) and `...-Q2_g64.gguf` (7.59 GB,
  mainline-shaped but still fork-only at compute time).
- Fork binaries (wiped from /tmp on reboot; re-fetch as needed):
  `https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b9591-62061f9/llama-prism-b9591-62061f9-bin-macos-arm64.tar.gz`
- First actions: (1) re-run `llama-bench -m models/ternary-bonsai-27b/Ternary-Bonsai-27B-Q2_0.gguf`
  on the quiet machine and update the Phase 0 verdict's throughput note; (2) read the
  fork's ggml type-42 dequant source (`PrismML-Eng/llama.cpp`) and document the slot
  encoding here before writing any repack code; (3) wikitext raw re-downloads via
  `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`
  (the tokenized `data/wikitext2-test.tokens.bin` is already in-repo and reproducible
  via `build/tokenize_to_bin`).

### Fork "Q2_0" (ggml type 42) slot encoding — authoritative (read 2026-07-14, tag `prism-b9591-62061f9`)

Source of truth: `ggml/src/ggml-common.h` (`block_q2_0`), `ggml/src/ggml-quants.c`
(`quantize_row_q2_0_ref`, `dequantize_row_q2_0`), `ggml/src/ggml-metal/ggml-metal.metal`
(`dequantize_q2_0`). CPU and Metal decoders are bit-identical.

- **Block:** `QK2_0 = 128` values → `{ ggml_half d; uint8_t qs[32]; }`, 34 bytes,
  **2.125 bpw**. The fork is natively g128 — the "g64" in the pack filename is
  cosmetic; no scale-pair collapse is needed for the 7.17 GB Q2_0 pack. A row of k
  elements is k/128 consecutive 34-byte blocks (standard ggml row layout: scale first,
  then the 32 quant bytes).
- **Slot packing:** sequential, 4 values per byte, **LSB-first**. Element `j` of a
  block lives in `qs[j/4]` at bit offset `(j%4)*2`. No Q4_0-style lo/hi nibble split.
- **Decode:** `value = ((int)q - 1) * d` with `q = (qs[j/4] >> ((j%4)*2)) & 3`,
  i.e. codes `{0,1,2,3} → {−1, 0, +1, +2}`. **The codespace is NOT strictly ternary:
  code 3 decodes to +2.** The fork's own quantizer can never emit it (scale is the
  block amax, so `round(w/d) ∈ [−1,1]`), but the Bonsai pack comes from their QAT
  pipeline, not this quantizer — the repack must scan every slot and hard-fail on
  code 3 (or, if it ever appears, T2_G128 must grow a +2 story; don't decide that
  silently).
- **Scale:** one FP16 `d` per 128 values; dequant multiplies in FP32 after the
  integer subtract. Reference dequant order: FP16→FP32 the scale once per block, then
  `(q−1)*d` per element.
- Sibling type `GGML_TYPE_Q1_0` (`QK1_0 = 128`, 1 bit/element, 18-byte block) exists
  in the fork — the binary-tier follow-up has the same authoritative-source answer
  waiting.

## Phase 1 — format + repack

- `DType::T2_G128 = 4` (or `T2_G64` if the source scales don't collapse — see below).
  Container stays version 1; dtype byte extends per FORMAT.md. 2-bit slots on the
  contiguous/reduction axis, FP16 scale blob, mirroring the Q4_G64 layout conventions
  (packing order chosen for 128-byte coalesced reads, documented in FORMAT.md).
- `tools/repack.py` grows a ternary path (`quant_policy: bonsai-t2-v1`): read their
  GGUF pack, verify values ∈ {−1, 0, +1} per slot exactly, re-pack into our layout.
  ~~Whitepaper: the g64 Q2_0 pack is the native g128 representation with each scale
  repeated per 64-value block — verify pairs are equal and collapse to g128; keep g64
  as fallback.~~ **Resolved by the source reading:** the fork's type 42 is natively
  g128 (`QK2_0 = 128`); the 7.17 GB Q2_0 pack needs no scale collapse, and its code
  bytes are already in exactly the layout T2_G128 wants — the repack is a lossless
  byte-copy plus scale-blob separation.
- Verify tokenizer identity (same base model) against our `.tok`; MTP-head tensors:
  confirm absent, and record what the engine needs to tolerate their absence cleanly.
- Gate: repacked artifact round-trips — dequantized q27 tensors bit-match dequantized
  GGUF tensors for every weight.

### Phase 1 status (2026-07-14 night): repack DONE, all gates passed

Artifact: `models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27` (7.15 GB, md5
`8babb56d…` in the dir's CHECKSUMS.md5), repacked from the Q2_0 pack in 143 s.
`gguf-py` doesn't know the fork's types; repack.py now forges enum members for
type 42 (and 41, the binary sibling) so the mainline package reads their packs.

- **Round-trip gate: PASSED.** All 498 Q2_0 tensors byte-copy losslessly into
  T2_G128 (same code bytes, same decode formula); the chunked bit-exact dequant
  comparison (fork reference formula vs FORMAT.md formula) passed on every tensor.
  All F32 tensors pass through untouched. rel-RMSE 0.0000 across the board.
- **Strictly ternary: CONFIRMED.** Zero code-3 slots across all 26.89 B ternary
  weights — the +2 code is unused, as hoped; T2_G128's forbidden-code rule is safe.
  Sparsity: 29.7% zeros overall (range 23.4% `blk.0.ssm_alpha` → 34.1%
  `blk.33.ssm_beta`) — a ternary-GEMV skip path has real headroom if ever needed.
- **Tokenizer identity: CONFIRMED byte-exact.** `tools/export_tokenizer.py` output
  from the ternary GGUF is byte-identical to `models/qwen36-27b-mtp/qwen36-27b-mtp.tok`
  (248,320 tokens, 247,587 merges, bos 248044, eos 248046). The ternary tier reuses
  the official `.tok`; no second tokenizer file needed.
- **Pack census** (851 tensors, arch `qwen35`, blk 0–63, ctx 262144 — same dims as
  the official artifact): Q2_0 = every matmul weight *including* `token_embd`,
  `output`, `ssm_alpha`, `ssm_beta`; F32 = all norms, `ssm_a`, `ssm_dt.bias`,
  `ssm_conv1d`. MTP (`blk.64.*`): **absent**, as predicted.
- **What the engine must tolerate (Phase 3 loader/dispatch checklist):**
  1. dtype byte 4 → T2_G128 kernels (GEMV + chunk GEMM, Phase 2).
  2. `token_embd.weight` in T2 → ternary row-lookup dequant (trivial: 40 groups/row).
  3. Missing `blk.64.*` and missing `output_q4.weight` → disable MTP drafting
     cleanly (greedy + suffix drafting only); today's loader treats blk.64 as
     required — needs an explicit "no MTP layer" mode keyed off the tensor table.
  4. `ssm_alpha`/`ssm_beta` arrive as T2 (official tier has them F16) — dispatch is
     per-tensor already, but any code assuming their dtype must be checked.

## Phase 2 — Metal kernels

- **Ternary GEMV** as a packed-dot variant of the existing Q4/Q8 structure: each lane
  loads 16 2-bit values, dot against 16 int8 activations is add/sub/skip — exact
  integer math, one combined scale per lane, one reduction per row. Same skeleton, new
  weight decode.
- **Ternary tiled chunk GEMM** (`q27_matmul_t2_mm`): stage dequantized integer weights
  in threadgroup memory exactly like the Q4/Q8 tiles; the simdgroup-matrix accumulate
  path is unchanged.
- Unit gates (the incident lesson, both of them): CPU-reference parity including wide
  shapes that enter the vectorized main loop (≥1024 cols) with deliberately varied
  per-group scales; `metal_gemv_bench`/`metal_prefill_bench` gain `--dtype t2` synthetic
  modes for bandwidth attribution. New kernels add entry points; if any buffer-index or
  argument-struct change touches existing kernels, bump `Q27_SHADER_ABI`.

### Phase 2 status (2026-07-14 late night): kernels DONE, all gates passed

- **Ternary GEMV (`q27_matvec_t2_g128`), 93.2 GB/s aggregate** on the production
  decode shapes (M4, synthetic resident weights) — above the Q8 kernel's 87 GB/s
  and fully bandwidth-bound. Projects to **~12.5 tok/s resident-decode kernel
  ceiling** (7.15 GB / 90 GB/s ≈ 79 ms/token) before attention/overhead — on the
  plan's ~11–12 prediction. Two structures failed first (recorded because the
  wrong-turn is instructive): a Q4-style shift/mask/imad packed-dot and a
  byte→float4 threadgroup-LUT+fma variant both saturated ~125 Gelem/s — the M4's
  issue-bound ceiling for per-element decode — which at 2.27 bpw is only ~33 GB/s,
  *slower in wall time than Q4 on the same shape*. The winning structure (read
  from the PrismML fork's kernel, then re-measured ourselves): **select-form dot**
  (`Σ(c−1)y = Σ_lo y + 2Σ_hi y − Σy`, two conditional adds per element, no
  multiplies, no decode) plus **4 rows per simdgroup** with the y-slice in
  registers so activation reads amortize across rows.
- **Design decision for Phase 3 dispatch: the production T2 GEMV takes FLOAT
  activations** (`matvec`, not `matvec_quantized`). Ternary math with float x is
  exact (+x/−x/0), needs no activation quantization, and the select-form kernel
  wants floats anyway. The engine's decode path already materializes the float
  rmsnorm output alongside the quantized copy, so routing is a per-dtype branch,
  not a new buffer. `q27_matvec_t2_quantized` (int8 x, scalar integer-exact)
  exists as the non-family-7 matmul fallback and parity reference.
- **Ternary chunk GEMM (`q27_matmul_t2_mm`)**: q8_mm structure with 2-bit staging
  (the 64-col K-tile subdivides the 128-col scale group identically). Prefill
  chunk rate is unchanged vs Q4/Q8 (470 vs 484 ms/chunk synthetic) — prefill is
  compute-bound, ternary neither helps nor hurts it.
- **Embedding row-lookup kernels** (`q27_embedding_t2`, `_t2_rows`) landed;
  `embedding_q8`/`embedding_q8_rows` entry points route per-dtype.
- **Gates** (all in `make test-metal`, all passing on a clean rebuild): narrow
  exact T2 matvec + embedding row (128 cols), wide float-path parity at 1152/5120
  cols with 37 rows (threadgroup-spanning + clamped-row tail) and varied
  per-group scales, wide int-path parity at 1152/5120, tiled GEMM parity at
  {9×256, 9×1024, 17×1152} × n∈{1,4,5,8,9,12}. `test-cpu` green.
- **ABI unchanged (6)** — new entry points only; the T2 grid shape (32 rows/tg)
  is host-side. Loader/inspect know dtype 4; `repack.py`-produced artifacts load.
- Benches: `metal_gemv_bench [reps] [--dtype q4q8|t2]`,
  `metal_prefill_bench ... [--dtype t2]` (t2 = every projection + embedding
  ternary; ssm_alpha/beta stay F16 pending the Phase 3 dispatch decision).
- Deferred to merge time: the byte-exact CUDA 16-token gate on the official
  artifact (shared loader.cpp touched — additive enum only, but the gate is
  mandatory for shared-path merges) and a `metal_decode_bench --dtype t2` mode
  (Phase 4 lists it).

## Phase 3 — engine integration + quality gates

- Loader/dispatch: route T2 tensors to the new kernels (dispatch is already
  per-dtype); everything else untouched.
- Gates, in order:
  1. `make test-metal` green with the ternary artifact validating (`--validate-only`).
  2. 16-token greedy continuation vs their fork running the same pack (on yukon or
     local, serial slot): committed text compared at tolerance/text level; investigate
     any early divergence rather than accepting it.
  3. **32K turbo3 NLL A/B** with tonight's harness: ternary vs official 5.25 bpw on
     `data/wikitext2-test.tokens.bin`, same buckets. This quantifies the tier's true
     quality cost on-device, independent of their benchmark claims, and simultaneously
     answers whether sub-2-bit KV composes with sub-2-bit weights (their §4.4 suggests
     low-bit weights *increase* KV tolerance; we get the measurement for free).
  4. Suffix-drafting A/B: committed tokens identical to ternary greedy.

### Phase 3 status (2026-07-15 early): integration DONE; gates 1, 2, 4 passed, gate 3 running

The ternary artifact now runs end-to-end on the q27 Metal engine — the first
fully-resident 27B decode on this machine.

- **Loader/validation:** `validate_architecture()` branches on
  `quant_policy == "bonsai-t2-v1"`: 64 blocks (no `nextn_predict_layers`, no
  blk.64 in the attention map), T2 `token_embd`/`output`/`ssm_alpha`/`ssm_beta`,
  `group_t2 == 128`, and an explicit assertion that no MTP tensors exist in a
  ternary artifact. `has_mtp_` gates every MTP entry point with a clear error
  ("use --suffix drafting"); MTP KV caches are not allocated for ternary.
- **Dispatch:** new `project`/`project_pair` helpers route T2 projections to the
  float-activation select-form GEMV (and skip the now-dead activation-quantize
  dispatches); Q4/Q8 keep the packed-dot path through the same helpers.
  Chunked prefill stays on `matmul_quantized` (the T2 GEMM); ternary
  `ssm_alpha`/`ssm_beta` chunk through the T2 GEMM instead of the F16 pair-rows
  kernel (same output layout). Unrouted sites still work — the quantized entry
  points accept T2 — so nothing can dispatch-fail on dtype.
- **Gate 1 (validate + suites): PASSED.** `--validate-only` OK on both tiers;
  `make test-metal test-cpu` green.
- **Gate 2 (vs the fork, same pack): PASSED, byte-identical** — stronger than
  the tolerance-level expectation. Greedy continuations match the fork's
  llama-server (`temperature 0, top_k 1`; note: `"samplers": []` makes their
  server nondeterministic — use top_k 1) token-for-token on "The capital of
  France is" (16 and 48 tokens) and "def fibonacci(n):" (32 tokens).
- **Gate 4 (suffix drafting): PASSED** — committed tokens identical to greedy
  (0 drafts accepted on the test prompt; drafting effectiveness is a Phase 4
  question, identity is the gate).
- **Official-tier regression:** canonical smoke (`--tokens 760,6511,314,9338,369`)
  still produces "Paris" after the dispatch refactor. The byte-exact CUDA
  16-token gate remains mandatory at merge time.
- **Observed throughput (unprofiled, base M4, greedy serial):** ~8.2–9.4 tok/s
  over 256-token runs (run-to-run variance, likely thermal), vs the fork's
  8.41 ± 1.36 on the same machine. `Q27_METAL_PROFILE` attribution puts the
  T2 GEMV at ~83 ms/token — exactly the bandwidth prediction — so the gap to
  ~11 tok/s is per-token overhead outside the GEMV (argmax readback sync per
  step, command-buffer turnaround, attention/GDN residual): Phase 4 material.
- **Gate 3 (32K turbo3 NLL A/B): deferred to an overnight run** (killed at
  ~2k/32768 positions — a 2 h machine-exclusive job mid-session was the wrong
  trade; it gates no current decision). Two harness notes for the rerun and all
  future NLL work:
  1. `--nll` now prints a **running mean NLL/PPL every 2048 positions**, so a
     long pass yields its verdict in the first minutes and the tail only
     refines the deep buckets.
  2. The cheap protocol when only the weight-quality question matters: a
     short pass over the same stream compares bucket-exactly against the
     official 32K run's shallow buckets (0–2k PPL 4.904, 2k–8k PPL 6.884) —
     `--nll-long 8192 --ctx 8192 --kv turbo3` reproduces both, `4096` the
     first. Attention is linear-in-position, so an 8K pass is ~1/16 the
     attention work of 32K. Only the weights×KV depth-interaction question
     needs the full pass, once per artifact/KV config, never per code change.
  Overnight command:
  `./build/q27-metal models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27 models/qwen36-27b-mtp/qwen36-27b-mtp.tok --nll data/wikitext2-test.tokens.bin --nll-long 32768 --ctx 32768 --kv turbo3`

## Phase 4 — performance validation

`metal_decode_bench --dtype t2` resident ceiling; artifact decode tg128-style (expect
no paging term at all — first fully-resident 27B on this machine); prefill chunk rate;
`Q27_METAL_PROFILE` attribution to confirm the GEMV share lands near the bandwidth
bound. Report vs the ~11–12 tok/s prediction and vs their published M4 Pro numbers
scaled by bandwidth.

### Phase 4 first measurements (2026-07-15 early)

- **Resident decode ceiling (`metal_decode_bench --dtype t2`, full step incl.
  attention/GDN/argmax sync): 12.40 tok/s fp16 KV, 12.66 tok/s turbo3**
  (80.7 / 79.0 ms/token, ~85 GB/s effective weight stream) — above the ~11–12
  projection.
- **Observed artifact decode: ~8.2–9.4 tok/s** warm (vs the fork's 8.41 ± 1.36
  on this machine). Gap to ceiling ≈ 37 ms/token; the weights are mmap-wrapped
  rather than resident synthetic buffers, so the suspects are page
  residency/wiring of the mmap-backed MTLBuffers (the artifact *fits* now —
  unlike the 17 GiB tier this tax should be removable: residency sets or
  pre-touch/pin of the weight range) and per-step command-buffer + argmax
  readback turnaround. Attribution next.
- Prefill chunk rate: T2 ≈ Q4/Q8 (470 vs 484 ms/chunk synthetic) — compute-
  bound, as expected; ternary doesn't change the prefill wall.

### Phase 4 paging chase — attribution and verdict (2026-07-15)

- **vm_stat attribution:** steady-state decode refaults nothing (pageins ≈ 0
  between tokens); the paging cost is **launch-time only** — each process
  start refaulted 3–6 GiB at ~1.8 GB/s during the first tokens, inside the
  tok/s timer, because the page cache is dropped between runs under memory
  pressure.
- **Landed (metal_backend.mm):** whole-mapping single-MTLBuffer wrap with
  per-tensor offsets when the mapping fits `maxBufferLength` (13.3 GiB here;
  the 16.5 GiB official tier keeps per-tensor views), `madvise(WILLNEED)`,
  and an MTLResidencySet on macOS 15+ attached to the queue
  (`Q27_METAL_NO_RESIDENCY=1` opt-out). Weights now wire at load ("ready"
  0.1 s → 2.5–5 s) and cannot be evicted mid-serve. Tests green, official
  tier fallback verified, greedy output byte-identical.
- **A/B verdict:** interleaved 3×3 base/res/no-residency showed no
  steady-state difference beyond ±0.5 tok/s noise. The 8.2–9.4-vs-12.66 gap
  was (a) launch fault-in inside the timer and (b) thermal mismatch — the
  ceiling itself re-measured mid-session was 10.57 tok/s, and profiling the
  artifact shows GPU busy 96% of wall with the T2 GEMV at 84.7% / ~77
  ms/token: decode is kernel-bound at the bandwidth limit, within a few
  percent of the same-moment ceiling.
- **Protocol rule:** ceiling and artifact numbers are only comparable when
  measured back-to-back at matched thermal state.

### Gate 3 — 32K turbo3 NLL verdict (2026-07-15, PASS)

Full single pass, no resets, `Q27_METAL_GQA_THRESHOLD=0` (proven kernels
only). Overall mean NLL 2.4590, **PPL 11.693** (official tier: 5.318).
Buckets vs official: 0–2k 9.483/4.904 (1.93×), 2k–8k 15.409/6.884 (2.24×),
8k–16k 7.456/3.525 (2.11×), 16k–32k 13.554/5.988 (2.26×). The bucket shape
matches the official run exactly — 8–16k best in both, deepest bucket
better than 2k–8k in both — so variation tracks content, not position:
**no ternary×turbo3 depth compounding; the tier's quality cost is a
depth-uniform ~2.1–2.3×**, consistent with Phase 0. The 0–2k and 2k–8k
buckets reproduce the 8K short-pass bucket-exactly, validating the cheap
paired protocol for this artifact.

**Flagged follow-up (performance, not quality):** wall 33,201.62 s
(0.99 tok/s) vs the official artifact's 7,458.87 s (4.39 tok/s) for the
same pass on the same machine — 4.4× slower despite streaming fewer weight
bytes. Quadratic attention work explains the within-run decay, not the
cross-artifact gap; suspects are the T2 float-activation chunk GEMM route
and overnight power state. Any timed rerun should also measure the causal
GQA path (default threshold), which was −7.8% already at 8K.
**(Resolved — see the wall-anomaly verdict below: overnight power state,
not the T2 path.)**

### Gate 3 wall anomaly — second-rig attribution pair verdict (2026-07-15)

Caffeinated sequential 16K turbo3 NLL pair on the certified 16 GiB M4
mac-mini (suite green, T2 artifact md5-verified, whole-mapping +
residency-set path — first 16 GiB machine to take it), same command as
gate 3, idle machine:

- **Legacy kernels (`Q27_METAL_GQA_THRESHOLD=0`): wall 1240.34 s
  (13.21 tok/s)**, overall PPL 10.088, buckets 9.483 / 15.409 / 7.456 —
  bit-identical to gate 3's first three buckets — and no slowdown anywhere
  in the pass (gate 3's collapse concentrated after ~position 11k).
- **Verdict: hypothesis B.** Identical deterministic computation at 13×
  gate 3's 0.99 tok/s exonerates the T2 float-activation chunk GEMM route;
  the 4.4× anomaly was the 24 GB machine's overnight power/pressure state.
  Gate 3's quality verdict stands; nothing reruns. Protocol rule: long
  unattended runs must be `caffeinate`d, and timed records note power
  state. (Optional: a short caffeinated confirmation slice on the 24 GB
  machine if same-machine confirmation is ever wanted.)
- **GQA 16K point: causal GQA (default threshold 2048) wall 991.59 s
  (16.52 tok/s) — −20.1% vs legacy at 16K**, extending the −7.8%-at-8K
  curve; 0–2k bucket bit-identical (9.483 — sub-threshold chunks share the
  legacy path), deeper buckets +0.11–0.17% PPL (15.435 / 7.464 vs
  15.409 / 7.456; summation-reorder noise, same magnitude as the 8K A/B).

## Whitepaper-derived follow-ups (2026-07-15, full read of bonsai-27b-whitepaper.pdf)

Ranked; items 1–2 are concrete next levers, the rest are references and
protocol notes.

1. **Adopt their KV-tolerance methodology as a q27 gate (paper §4.4,
   Table 6).** They measure KV-quantization damage as **output forward-KL
   against the same model's FP16-KV baseline** — on-policy (own MATH-500
   generations) and off-policy (BABILong-16K) — not corpus NLL. This
   isolates the KV effect from weight quality, which gate 3 could only
   disentangle by bucket-shape argument. q27 version: one process, one
   model mapping, two engines (fp16 KV + turbo3 KV share the weight wrap),
   teacher-force the same wikitext stream through both, per-position
   forward-KL, depth-bucketed. ~2× the cost of one 8K pass. This is also
   the right instrument for any sub-2-bit KV experiment (their early
   results say the key cache tolerates sub-2-bit, and that low-bit weights
   tolerate KV noise *better* — 0.0011–0.0029 nats vs FP16's 0.0137–0.222
   — a claim worth reproducing on turbo3 before trusting).
   **LANDED (2026-07-15, `--kl-kv` / `--kl-kv-self`, design in
   docs/plans/2026-07-15-kl-kv-gate.md). First 8K measurement: mean
   0.0115 nats, depth-flat (0.0123 / 0.0113), max 2.83 — 4–10× their
   low-bit band, just under their FP16 floor; their tolerance claim does
   not reproduce at magnitude for T2+turbo3. See the METAL_PROGRESS
   entry.**
2. **Native ternary packing (paper §4.3 + §9 roadmap).** Their own kernels
   store each trit in a 2-bit slot (deployed 7.2 GB vs 5.9 GB
   information-theoretic); ours inherit that layout (T2_G128). Base-3
   packing (5 trits/byte ≈ 1.6 bpw + scales ≈ 1.71 effective) cuts the
   decode weight stream ~15–18% — at the bandwidth bound that is a direct
   ~12.4 → ~14+ tok/s ceiling move. Kernel shape: 243-entry byte→5-trit
   LUT feeding the existing select-form dot. They list this as *their*
   future work; landing it first extends the q27 edge.
3. **Prefill reference point (paper §5.1, Table 8).** Their fork does
   pp512 52.48 tok/s on this exact M4 (Phase 0 measurement; 125–133 on M4
   Pro). Our chunk rate (~470 ms per 12-token chunk ≈ ~25 tok/s) is roughly
   half their stack on the same machine — independent confirmation that
   chunk-GEMM staging (half-precision staging, double-buffered K-tiles) is
   the top prefill lever, with a concrete beatable number.
4. **Behavioral-gate sampling config (paper §B.1).** Their evals run
   thinking mode at temperature 0.7, top-p 0.95, top-k 20 (greedy is not
   their recommended operating point). Our behavioral probes should match
   that config once GPU-assisted sampling (stage 2) lands — which also
   makes top_k=20 the exact GPU-candidate case.
5. **Drafting reality check (paper §6.2 + §9).** Their DSpark drafter is
   net-positive only on CUDA; they state batch-1 verification does not
   amortize on Apple Silicon — the same wall our MTP/suffix experiments
   hit. Treat GPU-resident drafting on Metal as research, not a scheduled
   lever; their ~1.34–1.37× CUDA speedup bounds the prize if the
   verification cost problem is solved.
6. **Verify llama.cpp's batch-1 Metal MTP claim; if real, lift the
   scheduling (2026-07-15, from the Gemma 4 / Unsloth release trail).**
   llama.cpp mainlined MTP speculative decoding (Unsloth changelog
   2026-05-18 / 06-03: auto-enabled for MTP GGUFs, "hardware-specific
   customized settings", "~2x faster", Apple Silicon prebuilts; Gemma 4
   shipped family-wide MTP drafters 2026-04-16). This directly contradicts
   item 5's amortization wall — if their gain holds at batch 1 on Metal.
   Cheap verification first: read their speculative-MTP scheduling code
   (draft width per hardware tier, when-to-verify policy) and bench their
   prebuilt on this M4. If it holds, port the *scheduling policy* onto our
   existing batched MTP verify (official tier has the MTP layer; verify
   path is byte-exact) — not their code. If it does not hold at batch 1 on
   Metal, record the measurement next to item 5 and keep drafting parked.

## Non-goals (this plan)

Binary 1.125-bpw tier (follow-up — same kernel skeleton, do after ternary proves out);
CUDA ternary kernels (Metal first); vision tower (q27 is text-only); DSpark-style
external drafter (suffix drafting covers day one; MTP-head graft is the experiment
worth trying first).

## Risks / open questions

- **Slot encoding unknown** until their fork is read (Phase 1 hard dependency).
- **Quality claims unverified** — Phase 0 exists to kill the plan cheaply if they
  don't hold. Their numbers are self-published, single-harness.
- **MLX-only tensors**: if the GGUF pack turns out to be scale+bias (their MLX packing
  stores two FP16 values/group), the repack must reconstruct scale-only form; the
  whitepaper says GGUF stores one scale — verify.
- **The official tier remains the quality anchor**: byte-exact CUDA gate and full
  `make test-metal` on the 5.25 bpw artifact stay mandatory for every merge touching
  shared code paths (kernels are additive, but loader/dispatch changes are shared).
