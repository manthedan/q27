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

- **Throughput (their stack, loaded machine):** tg128 `5.17 ± 0.15` tok/s, pp512
  `24.2 ± 1.8` tok/s. Far below the whitepaper's M4 Pro figures even after ~2.3×
  bandwidth scaling (predicts ~8); effective weight stream ~37 GB/s vs our kernels'
  ~85 GB/s. **The engineering upside grew:** a q27 ternary tier at our measured GEMV
  efficiency projects to ~8–12 tok/s — roughly 2× their own product on this hardware.
  (Number taken under desktop load + swap debt from the day's 17 GiB runs; re-bench on
  a quiet machine before quoting externally.)
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

## Phase 1 — format + repack

- `DType::T2_G128 = 4` (or `T2_G64` if the source scales don't collapse — see below).
  Container stays version 1; dtype byte extends per FORMAT.md. 2-bit slots on the
  contiguous/reduction axis, FP16 scale blob, mirroring the Q4_G64 layout conventions
  (packing order chosen for 128-byte coalesced reads, documented in FORMAT.md).
- `tools/repack.py` grows a ternary path (`quant_policy: bonsai-t2-v1`): read their
  GGUF pack, verify values ∈ {−1, 0, +1} per slot exactly, re-pack into our layout.
  Whitepaper: the g64 Q2_0 pack is the native g128 representation with each scale
  repeated per 64-value block — verify pairs are equal and collapse to g128; keep g64
  as fallback. Read their fork's dequant source for the authoritative slot encoding
  first; do not guess it.
- Verify tokenizer identity (same base model) against our `.tok`; MTP-head tensors:
  confirm absent, and record what the engine needs to tolerate their absence cleanly.
- Gate: repacked artifact round-trips — dequantized q27 tensors bit-match dequantized
  GGUF tensors for every weight.

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

## Phase 4 — performance validation

`metal_decode_bench --dtype t2` resident ceiling; artifact decode tg128-style (expect
no paging term at all — first fully-resident 27B on this machine); prefill chunk rate;
`Q27_METAL_PROFILE` attribution to confirm the GEMV share lands near the bandwidth
bound. Report vs the ~11–12 tok/s prediction and vs their published M4 Pro numbers
scaled by bandwidth.

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
