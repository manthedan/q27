# Binary weight tier (Bonsai 1.125 bpw) for the Metal engine

**Status: proposed** (2026-07-15). The ternary plan's non-goals section parked this
tier behind "after ternary proves out." That precondition is met: gate 3 passed
(depth-uniform ~2.1–2.3× PPL, 32K), fork parity was byte-identical, T2 serves.
This plan mirrors `2026-07-14-ternary-tier.md` phase-for-phase; where a step is
identical to the ternary version it says so instead of restating it.

Source pack: PrismML `Bonsai-27B` (binary {−1,+1} sibling of Ternary-Bonsai,
same Qwen3.6-27B base, Apache-2.0). GGUF `Q1_0_g128` (ggml type 41 in their
fork, the type-42 ternary's sibling) at **1.125 bpw deployed**, ~3.9 GB; also
shipped as `-unpacked` masters and an MLX 1-bit pack. Whitepaper retention:
**89.5% of FP16** aggregate (ternary: 94.6%) — the quality risk is the whole
reason Phase 0 exists.

## Motivation

Same wall as the ternary tier, one step further down the ladder. Decode is
weight-stream-bound; the artifact size *is* the ceiling:

| tier | bpw deployed | artifact | resident ceiling @ ~85 GB/s |
|------|-------------:|---------:|------------------------------|
| official | 5.25 | ~17 GB | ~4.6 tok/s (pages on 24 GiB) |
| ternary T2 | 2.125 | 7.15 GB | ~12.4–12.7 tok/s (measured) |
| ternary T3 (planned) | 1.75 | ~5.9 GB | ~14+ tok/s (projected) |
| **binary B1** | **1.125** | **~3.9 GB** | **~20–23 tok/s (projected)** |

Projection basis: T2 resident decode measured 79–81 ms/token with the GEMV at
~77 ms (84.7% share); halving streamed weight bytes puts the GEMV near ~39 ms
and the token near ~43 ms if the non-GEMV residual (~3–4 ms) holds. Treat
20–23 as a band, not a number, until `metal_decode_bench --dtype b1` exists.
Fixed per-step costs (readback sync, command-buffer turnaround) double in
relative share at this speed — the GPU-resident greedy feedback loop
(ds4-survey item 4) pays twice as much on this tier.

**The mini becomes a first-class serving box.** 3.9 GB fits a **single**
buffer view under the mini's 8.0 GiB `maxBufferLength` (no overlapping-view
machinery needed), fully wired via the existing residency-set path, with room
for 262K turbo3 KV (3.4 GiB) beside it in 16 GiB. A full 27B at ~20 tok/s
with maximum context on the smaller machine is the headline if quality holds.

Three-tier framing: official / ternary / binary is a quality–speed ladder on
one binary — per-tensor dtype (FORMAT.md) means the artifact chosen at load
time determines everything; engine, server, gates, benches shared wholesale.

## Known asymmetries (inherited from the ternary tier, plus one)

1. **No MTP head** — same as ternary; greedy + suffix drafting day one.
   Drafting remains parked per the ternary doc's follow-up 6 resolution
   (batch-1 Metal spec-decode corroborated as a net loss on low-bandwidth
   Apple Silicon).
2. **No byte-exact oracle** — same gate substitute as ternary: (a) bit-exact
   CPU-reference unit parity for the new kernels, (b) committed-token
   comparison vs their fork on the same pack (`top_k: 1`, two prompts,
   16/32/48 tokens — the ternary protocol verbatim), (c) on-device NLL A/B
   vs the official AND ternary tiers.
3. **Quality risk is materially higher than ternary's.** 89.5% aggregate
   retention, and ternary's weakest categories (IFBench, τ²-Bench — exactly
   our agentic workload) will be weaker still. Hence:

## Phase 0 — validation spike with a written kill criterion (go/no-go)

Run on the **mini** (the 24 GB machine is the daily driver; ~3.9 GB download
fits easily). Their fork binaries (`prism-b9591-62061f9`) are already
certified there from the ternary spike.

1. Download the Q1_0 pack; md5 into `CHECKSUMS.md5`; verify tokenizer
   byte-identity against `qwen36-27b-mtp.tok` (held for ternary; re-verify).
2. `llama-bench` under their fork, caffeinated: tg128/pp512 — sanity vs the
   bandwidth projection (expect roughly 2× their ternary 8.41 tok/s on the
   24 GB M4-class machine).
3. Quality, cheap paired protocol (validated on ternary): 8K NLL on our
   wikitext tokens via their tooling, ratio vs the official tier's 8K
   buckets; plus subjective agentic smoke (tool-call formatting, small edit
   tasks, instruction adherence — the ternary spike's prompt set).

**Provisional kill criterion (adjust before running, then honor it):**
- 8K NLL ratio vs official ≤ ~3.5× and agentic smoke coherent → **serving
  tier**; proceed to Phases 1–4.
- ~3.5–5× or noticeably degraded-but-usable agentic output → **demote to
  experimental/throughput tier** (benches, long-context stress, drafter
  research substrate); build Phases 1–3 anyway only if the throughput use
  case justifies it — record the decision.
- > ~5× or agentic collapse (malformed tool calls, incoherent edits) →
  **no-go**; file findings in this doc, tier stays dead.

Machine rule: Phase 0 lives entirely on the mini under its memory-safe
policy. The binary pack (3.9 GB) + ternary artifact (7.15 GB) can physically
coexist on either machine, but **the one-model-load rule stays in force** —
it exists because of a real crash; amend it only deliberately, with a
measured pressure test, never implicitly.

### Phase 0A results (2026-07-15 night, 24 GB M4 — Daniel-authorized overnight batch, `logs/overnight-20260715/`)

Two sessions, ratios within-session (machine under desktop load in session
B: T2 tg128 6.21 vs 8.59 in session A vs 8.41 recorded quiet):

- **Throughput (their stack): B1 is NOT bandwidth-bound — their Q1_0
  kernel is issue-bound on M4.** Session A: B1 tg128 `10.05 ± 0.27` vs T2
  `8.59 ± 1.02` (**1.17×**); session B: `7.05 ± 1.01` vs `6.21 ± 0.18`
  (1.14×). At 0.53× the bytes (3.53 vs 6.66 GiB) a bandwidth-bound kernel
  would give ~1.9×; effective stream is ~35 GB/s vs T2's ~57 in the same
  stack. B1 pp512 `32.64 ± 3.99` (also below T2's 52.5 — their prefill
  pays the unpack too). **Consequence: their stack is not the B1 ceiling,
  and the q27 Phase 0B kernel prize is larger than the in-stack numbers
  suggest** — the whitepaper's "native low-bit kernels are future work"
  framing is corroborated at 1.125 bpw, same lesson as our T3 kill but on
  their side of the fence.
- **Wikitext-2 PPL (ctx-512 × 24 chunks, identical protocol to the ternary
  spike): `11.69 ± 0.43`** vs T2's `10.04 ± 0.36` → **B1/T2 ≈ 1.16×**, a
  far gentler increment than the whitepaper's aggregate-retention gap
  (89.5 vs 94.6) implied. Vs the official tier (~4.90–5.32, protocol
  indicative): ~2.2–2.4× — comfortably inside the ≤ ~3.5× serving-tier
  band of the provisional kill criterion.
- **Behavioral probes: 4/4 PASS** (greedy, their llama-server; thinking
  mode cannot be disabled on this build — 1024-token budgets truncate
  inside the thinking phase, 6144 clears it; a footgun for any scripted
  gate on this stack). Native tool call: correct function, exact args, no
  leakage. JSON-only: machine-checked exact (keys/types/arity, no fences,
  no prose). 7-constraint list: every constraint honored. Code edit
  (planted touching-intervals case): correct fix (`s <= out[-1][1]`),
  code only. Prompts are reconstructions of the ternary-spike categories,
  not the identical strings (originals weren't recorded — they are now,
  in `logs/overnight-20260715/run.sh`).

**Phase 0A verdict: GO on quality.** PPL sits in the serving-tier band
(~2.2–2.4× official, 1.16× over T2), the agentic-shaped probes are clean,
and the increment over the already-resident T2 tier is small. The tier's
fate now rests on Phase 0B kernel economics (below) — their issue-bound
kernel means the in-stack 1.17× is a floor argument, not a ceiling one.

### Phase 0B — synthetic kernel economics (split adopted per expert reviews 2–3)

The vendor-stack quality gates above are **Phase 0A** (make them
machine-checkable per the review-2 triage, finding 7: exact-JSON, native
tool call, instruction task, code edit with tests, planted edge case,
long-context retrieval, 8K NLL vs BOTH official and T2 — report B1/T2, not
only B1/official). **Phase 0B** runs in parallel, before any repack or
engine work: B1 GEMV + embedding only, in `metal_gemv_bench`, benchmarking
all three dot structures from the round-3 answers doc §Q4 —

1. positive-mask select (baseline): `scale·(2·Σ_pos − Σy)`, one conditional
   accumulate per element;
2. float sign-bit XOR: flip the IEEE sign bit from the weight bit — exact
   for finite activations, no Σy correction;
3. int8 bitplane + popcount: 8 and+popcounts per 32 columns, with the
   activation preprocessing (int8 quant + bitplane transpose + Σx, fused)
   **inside the measured time**.

**Pre-registered kill lines (expert's, endorsed): the metric is production
projection-mix WALL-TIME ratio T_B1/T_T2 — never effective GB/s alone.**
≤0.60 with predicted full token ≤50 ms → strong GO (the 20 tok/s thesis
holds). 0.60–0.72 → conditional GO, only if Phase 0A quality lands
materially closer to T2 than the 89.5% retention suggests. >0.72 → kill the
20–23 serving headline (ratio 0.60 requires ~85% of T2's ~93 GB/s stream ≈
79 GB/s; practical early kill line ~70 GB/s across the full projection
mix). A candidate that wins only when preprocessing is excluded from
measurement is a kill, not a pass. After integration, rerun resident K=8 —
the fixed orchestration share doubles as tokens get cheaper.

## Type-41 encoding — read at source before any repack code

Same discipline as ternary (that plan, "authoritative source" section): read
`PrismML-Eng/llama.cpp` `block_q1_0` / `quantize_row_q1_0_ref` /
`dequantize_q1_0` (CPU and Metal) at the pinned tag and document the layout
HERE before writing repack code. Expected shape (verify, do not assume):
g128, one fp16 scale + 128 code bits per group (16 code bytes), code
`c ∈ {0,1}` → `(2c−1)·d`, sequential LSB-first like type 42. Open questions
the source read must answer: bit order within bytes, scale derivation (amax
vs mean-abs — QAT pipelines differ), and whether any tensor in the pack uses
a different dtype mix than the ternary pack's census.

**READ 2026-07-16 night (tag prism-b9591-62061f9, `~/prism-fork/src`),
decode side confirmed exactly as expected:** `block_q1_0` = 18-byte
`{fp16 d; u8 qs[16]}` per 128 (ggml-common.h:180–184); bit `j` lives at
`qs[j/8]` offset `j%8` — **sequential LSB-first**, same convention as type
42 — and decodes `bit ? d : -d` (dequantize_row_q1_0, ggml-quants.c:419).
Bit-order question answered; the scale-derivation question (amax vs
mean-abs) remains open but is only needed for PRODUCING packs — repack is
decode-only. **Phase 1's B1_G128 layout (dtype 6) is now implemented and
first produced** by the dspark drafter repack (`repack_b1` in
tools/repack.py: verbatim code bytes + split fp16 scale blob, chunked
bit-exact round-trip gate; the dspark pack's `token_embd.weight`,
248320×5120, passes). The full-tier repack of `Bonsai-27B-Q1_0.gguf` and
the census/cross-check against the `-unpacked` masters remain this plan's
own Phase 1 work.

## Phase 1 — format + repack

- `B1_G128` = **dtype 6** in FORMAT.md (5 is reserved for T3_G128). Row data
  `(cols/128) × 16` bytes; scales fp16 `[rows, cols/128]` — same scale-blob
  separation as T2. Meta `quant_policy: bonsai-b1-v1`.
- **Prefer the `-unpacked` masters as the repack source** with the GGUF as a
  cross-check (both must dequantize identically); this also future-proofs a
  CUDA binary tier. Unlike ternary there is no illegal code to hard-fail on
  (both code values are valid), so integrity rests on the unpacked↔GGUF
  cross-check plus the chunked bit-exact round-trip gate (ternary gate,
  reused).
- Pack census (mirror ternary Phase 1): tensor count, which tensors are
  binary vs F16/Q8, dims vs the ternary pack, MTP layer confirmed absent.

## Phase 2 — Metal kernels

Same skeleton as T2; the select-form dot simplifies — {−1,+1} needs
`2·Σ_{c=1} y − Σy`: one conditional add per element plus one shared Σy per
128-group (T2 needs two conditional adds). Deliverables mirror the T2 list:

- `q27_matvec_b1_g128` (float-activation production path; structure chosen
  by Phase 0B's three-candidate bench — positive-mask select, sign-XOR, or
  bitplane-popcount. B1 moves half T2's bytes per column, so the issue-rate
  ceiling returns as the central risk at 1.125 bpw; Phase 0B's wall-time
  kill lines catch it before any engine wiring).
- `q27_matvec_b1_quantized` (int8-x integer-exact parity variant),
- `q27_matmul_b1_mm` on the (now half-staged) GEMM staging pattern,
- `q27_embedding_b1`/`_rows` with per-dtype routing,
- `--dtype b1` modes in `metal_gemv_bench` / `metal_prefill_bench` /
  `metal_decode_bench`.

Gates: narrow-exact, wide float-path and int-path (threadgroup-spanning row
clamp, varied scales — the shapes that caught the packed-GEMV scale bug),
GEMM-tile parity; all in `make test-metal`, green on a clean rebuild
(stale-binary rule; bump `Q27_SHADER_ABI` only if bindings change — additive
entry points should keep it).

## Phase 3 — loader/dispatch

`bonsai-b1-v1` branch beside `bonsai-t2-v1` in `validate_architecture()`:
64 blocks, no MTP (`has_mtp_` gating reused), B1 projections routed to the
float-x GEMV, chunked prefill on the B1 GEMM, T2's `project`/`project_pair`
helpers generalized to a dtype switch rather than duplicated. `--validate-only`
both tiers; single-view whole-mapping + residency on both machines.

## Phase 4 — on-artifact gates + perf

Mirror ternary Phase 3/4: 16/32/48-token greedy byte-identical to their fork
on the same pack; suffix-drafting committed tokens identical to greedy;
official-tier canonical smoke unchanged; byte-exact CUDA 16-token gate at
merge (loader touched). Then `metal_decode_bench --dtype b1` resident ceiling
vs artifact decode back-to-back at matched thermal state (protocol rule), and
the 8K→32K NLL ladder (expect the ternary result's shape: depth-uniform
ratio, buckets tracking content; any depth *interaction* is new information
and gates long-context serving on this tier).

## Relationship to T3 (scope boundary)

**Binary needs no packing follow-up.** 1.125 bpw is already
information-theoretically dense for {−1,+1}+scale (128 code bits + 16 scale
bits per group); there is no base-3-style trick to apply. T3's ~18% stream
cut is ternary-only. The two streams share only the GEMM staging pattern and
the per-dtype routing — they do not compete for kernel surface.

## Follow-up (recorded, not scheduled): drafter options for the no-MTP tiers

Two candidate drafters exist if drafting is ever revived on these tiers;
whitepaper §6 (read 2026-07-15) settles their ranking:

1. **DSpark (their released drafter) — the stronger option at the same
   cost.** Six-layer block-parallel transformer (DFlash family) conditioned
   on normalized hidden-state taps from five evenly spaced target layers,
   K=4 block drafting in one pass, sequential head for intra-block
   dependencies, confidence head + hardware-aware scheduler verifying only
   positive-expected-return tokens. Drafter-unique weights ~0.5 GB
   (embeddings/head shared with target). Measured on H100 greedy: accepted
   length 3.6–3.7, **1.34×/1.37×** over ternary/binary low-bit-kernels-only.
   Our `dflash-block-verify-design.md` (CUDA side) already worked out the
   block-verify economics for exactly this drafter contract.
2. **Small-sibling 1.7B drafter (our idea, NOT in their roadmap)** — same
   ~0.5 GB stream per draft token but generic (not trained against the 27B),
   autoregressive (sequential draft latency where DSpark is block-parallel),
   and a multi-arch engine project on our side. **Dominated by DSpark at
   equal byte budget; keep only as a fallback** if DSpark's Metal
   port/contract proves impractical.

Both stay parked behind the same wall, documented independently by our
ledger (ternary doc follow-up 6 resolution) and their §9: batch-1
verification does not amortize on low-bandwidth Apple Silicon; they name
making DSpark net-positive on-device an open roadmap item. Revive only after
the GPU-resident sync work (ds4-survey item 4) removes per-round overhead,
and gate with the mtp-draft-head discipline: offline acceptance/economics
probe first, zero engine work. For completeness:
binary-27B-as-drafter-for-ternary-27B was checked and does not pay (0.55×
stream cost per draft token loses at any realistic acceptance).

## Whitepaper §9 roadmap notes relevant to this tier (read 2026-07-15)

- **"Agentic coding, next":** a Bonsai 27B variant tuned for agentic coding
  "will follow shortly." Directly targets this tier's biggest quality risk
  (our workload is their weakest category). If Phase 0 lands in the demote/
  kill band, re-run the same gates against that variant when it ships before
  writing the tier off — the artifact is a drop-in for the same dtype.
- **Sub-2-bit KV is future work for them; we ship it** (turbo3, 1.56
  bits/value, 32K depth-flat). Native low-bit packing is also their future
  work (our T3 race). Worth stating: on KV and packing, q27 is ahead of the
  model vendor's own roadmap; drafters are the one place they're ahead of us.

## Non-goals (this plan)

T3 packing (own plan doc); CUDA binary kernels (Metal first, same as
ternary); MLX packs (different runtime; the mlx-1bit pack is a benchmark
comparison target only); AWQ-4bit packs (mainstream-stack repacks, strictly
dominated by native tiers on our hardware); any amendment of the
one-model-load rule.
