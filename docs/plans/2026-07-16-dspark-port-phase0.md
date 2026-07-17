# DSpark port — Phase 0: contract, repack, and gate design (no engine code)

**RESOLVED (2026-07-16 evening, operator-authorized Phase-2 legs): the
pre-registered Phase-3 gate fires PARK — decisively.** Leg B
(`logs/dspark-phase2-20260716/`, fork harness `test-dspark-real-eval`, T2
target + drafter, 24 prompts / 712 rounds / 2,285 tokens): **τ = 3.209
committed/round** (single block + bonus); conditional acceptance is FLAT in
depth (d(1..4) = 0.772/0.780/0.779/0.778 — mask positions do not decay
within a block); **P(full block) = 0.365**; category spread is large (chat
τ 2.40 / reasoning 3.63 / code 4.02). Against the lever-2 curve
(BE(16) = 4.13 tok/round, G = 89.4 ms): single-block S = 0.78; the chained
upper bound — generous to DSpark twice over (chained blocks assumed to
accept at block-1 rates despite strictly degraded features, drafter forward
cost excluded) — is **E ≤ 4.42 at w=16 → S ≤ 1.07 < the 1.3 bar**. True
chained acceptance can only be lower, so the park is decisive, not
marginal. Corroboration: the fork's own stack measures speedup 0.528 on the
same run — DSpark loses on M4 even at home. Even code-only traffic
(τ 4.02 < BE 4.13) fails single-block. **Per this plan's own kill line: the
port parks HERE with the contract recorded; suffix-burst widening (no
drafter, no acceptance loss) inherits lever 2's win instead.** What is
banked for any future revisit (M5/M6-class hardware or a stronger drafter):
the full contract below, the lossless repacked artifact, 7 rounds of
real-run reference fixtures in the fork's tier-2 ref.bin layout
(`fixtures/`, checksummed; fork instrumentation diff alongside), the
harness build recipe, and `tools/dspark_gate_analysis.py` — re-running the
gate on new hardware is one command per leg.

**Status update (2026-07-16 night): lever 2 LANDED (S(48) = 3.94×,
2026-07-16-lever2-verify-width.md) and the Phase-3 gate now STRADDLES.**
The measured curve gives break-even ~4.1 tok/round at w=16 (the chained-
block width class); the expected chained-block committed rate 5–6.5 puts
S ≈ 1.2–1.57× — straddling the pre-registered 1.3× line. Arithmetic can no
longer decide go/park: the deciding number is the fork's measured per-block
acceptance decay, which is exactly what **Phase 2's reference fixtures**
produce. Phase 0 residue + Phases 1–2 therefore proceed as planned (CPU-
only, this machine); Phase 3 stays blocked until the fixture-measured
acceptance-weighted rate is plugged into the lever-2 curve. Full triage:
2026-07-16-paper-scan-triage.md §DSpark.

**Status: Phase 0 contract DONE; economics path narrowed (2026-07-16 pm).**
Lever 1 (phase B direct-RHS) PARKED the same afternoon by its own kill line
(C/D2 = 0.782 at width 12 — 2026-07-16-lever1-direct-rhs.md), so the ~430 ms
round is NOT coming down via chunk-GEMM efficiency on the M4. The port
decision now rides verify-round-cost **lever 2 only** (verify width past 12,
mini's queue): at a flat ~430–465 ms round, single-block DSpark (~3.8
committed/round) stays a loss; the viable shape is **chained blocks at
w = 8–16**, whose expected committed tokens/round (~5–6.5 at the fork's
measured per-block acceptance, decaying with chain depth) sits right at the
break-even — thin, and decided by measurement, not argument. **Updated gate
for Phase 3: the post-lever-2 oracle re-sweep must show S at the chained
width ≥ 1.3× at the acceptance-weighted committed rate before any engine
work.** If lever 2's curve flattens below that, the port parks here with the
contract recorded — and suffix-burst widening (which needs no drafter at
all) inherits lever 2's win instead. Machine: 24 GB M4 for design/repack
(CPU).

## Why now, and the pre-registered kill line

Gate 0 (sibling-drafter probe §Gate 0): break-even is ~5 committed
tokens/round at today's flat ~430 ms verify round. DSpark's measured M4
acceptance (~2.8 of each 4-block → ~3.8 committed/round with the bonus) is a
net LOSS today. Phase B targets ≥2× effective weight stream at w ≤ 16 →
round ~250 ms → break-even ~2.9 → DSpark clears at ~1.3× net. **Kill line:
if phase B misses its own park line (<1.3× stream at width 12) AND the
repriced oracle sweep keeps break-even above ~4 tokens/round, this port
parks at Phase 0 with the contract recorded — no engine work.** Phase 0 is
deliberately the conflict-free long-lead slice (new files only; zero overlap
with phase B's kernel work).

## Drafter contract (extracted from the shipped pack, 2026-07-16)

`prism-ml/Bonsai-27B-gguf :: Bonsai-27B-dspark-Q4_1.gguf` — 1.79 GB, arch
`dspark`, "3.6B":

- **Trunk:** 6 standard attention blocks — n_embd 5120, ffn 5120 (gate/up/
  down), 40 Q heads / 4 KV heads @ head_dim 128 with q/k norms, RMS eps 1e-6,
  rope base 1e7, ctx 4096. All Q4_1 (type 3). No GDN — the drafter is pure
  attention, so the q27 engine's attention path covers it; no conv/DeltaNet
  state to manage.
- **Taps:** `dspark.fc.weight [25600, 5120]` — concat of FIVE target hidden
  states (5 × 5120) from target layers **[1, 16, 31, 46, 61]**, projected to
  5120. `dspark.hidden_norm.weight [5120]` normalizes (order vs fc per the
  fork reference — verify in Phase 1 gate).
- **Block proposal:** `dspark.dspark.block_size = 4`, mask token 248319 —
  block-parallel: one drafter forward proposes 4 tokens (mask-fill), not a
  serial chain.
- **Shared vocab head:** own `token_embd`/`output` [5120, 248320] (Q4_1) +
  `output_norm` — vocab-identical to the target (the property that killed
  the 1.7B sibling and makes DSpark the only viable drafter).
- **Markov head:** `markov_head_a [256, 248320]` (type 30 = **BF16**, see
  resolved unknowns below) + `markov_head_b [256, 248320]` (Q4_1), rank
  256 — the confidence-scheduling prior (whitepaper §6).
- **Confidence head:** `[5376, 1]` + bias — 5376 = 5120 + 256: hidden concat
  markov features (`confidence_head_with_markov = true`). Gates block
  acceptance scheduling.
- **log-SNR conditioning:** `log_snr_fc1 [128, 5120]` (type 30) + fc2
  [5120, 5120] + biases, clamp [-9, 9] — diffusion-style noise conditioning
  on the mask-fill.
- ~~Unknown to resolve from the fork source (Phase 0 residue)~~ **ALL THREE
  RESOLVED at source (2026-07-16 night, tag prism-b9591-62061f9, shallow
  clone at `~/prism-fork/src`):**
  1. **Type 30 = plain `GGML_TYPE_BF16`** (mainline numbering, ggml.h:420) —
     not a fork packed format. Three BF16 tensors, not two: `log_snr_fc1`,
     **`log_snr_fc2` [5120,5120]** (contract correction — was assumed Q4_1),
     `markov_head_a`. All widen exactly to F32 in the repack.
  2. **Tap timing: the layer's full residual-stream OUTPUT** — `l_out` after
     attention residual + FFN residual + cvec, BEFORE the next layer's
     attn_norm (models/qwen35.cpp per-layer tap; capture slots keep caller
     order, so fc's concat order is the requested [1,16,31,46,61]). The tap
     is un-normalized; the drafter applies **fc first, then hidden_norm**
     (models/dspark.cpp:119) — the order question from the contract above is
     settled the fc-first way. Capture rows use the masked output-row layout
     and REQUIRE logits/output requested on every tapped row (a prefill-
     driver obligation; llama-context get_embeddings_capture_ith).
  3. **Feature staging: incremental per-token, no window, no shift.** One
     capture row (5×5120 f32) per newly-accepted target token accumulates in
     `ctx_feat`; each draft round feeds ONLY the new rows (dummy-token batch
     rows whose embeddings are replaced by the staged features before the
     residual stream forms) + the block, then crops the drafter KV back to
     the anchor and clears the staging (common/speculative.cpp draft-dspark
     impl). The drafter keeps persistent KV over context rows; the only
     bound is its own ctx 4096.
- **Port-critical details from the same read:** block position 0 is seeded
  with the REAL last-accepted token (anchor), NOT mask_token_id — only
  positions 1..3 are masked; the markov resample is **strictly sequential
  within the block** (k conditions on the actually-sampled k−1 — the fork
  comments record that batching it over mask tokens is a bug class that
  already hit their MLX port); the fork's own CLI/server do NOT engage
  capture — `tests/test-dspark-real-eval.cpp` / `test-dspark-forward.cpp`
  are the true reference drivers, so Phase 2 fixtures should ride the test
  harness, not `--spec-type` on the server.
- **Contract correction:** `token_embd.weight` is **Q1_0 (type 41, binary)**,
  not Q4_1 — the shared-vocab embedding rides the fork's binary format. The
  `output` head IS Q4_1 as recorded.

## Memory budget (24 GB M4, T2 target)

T2 target 7.15 GB + drafter 1.79 GB + KV (both) + taps ≈ **~9.5 GB total** —
comfortably resident; no policy amendment needed (the one-model-load rule
amendment question from the sibling plan dissolves: this is one target + a
1.79 GB sidecar, not 17+17).

## Phase plan (each gated)

0. **This doc + contract** (done above) + fork-source reads for the three
   unknowns. No GPU. **DONE 2026-07-16 night — all three resolved, see
   contract section.**
1. **Repack:** extend `tools/repack.py` for arch `dspark` (Q4_1 + type-30 at
   source, same lossless discipline as bonsai-t2-v1; hard-fail unknown
   slots). Gate: bit-exact round-trip, all 79 tensors. **DONE 2026-07-16
   night — GATE PASSES: all 79 tensors, every verbatim tensor (46 Q4_1 →
   dtype 7 `Q4_1_G32`, 1 Q1_0 token_embd → dtype 6 `B1_G128` per the
   binary-tier plan's pre-registered Phase-1 layout) chunked bit-exact
   round-trip vs the fork reference dequant; 3 BF16 → F32 exact, RMSE 0.0000
   on all 79. Artifact `models/binary-bonsai-27b/bonsai-27b-dspark.q27`
   (1.97 GB; +0.18 GB over source = the exact BF16→F32 widening), md5
   `ec338c42…` in the dir's CHECKSUMS.md5; source pack md5 re-verified
   against its recorded checksum before repacking. FORMAT.md documents both
   new dtypes. Note: dtype 6 is hereby first PRODUCED (drafter token_embd
   only); the full B1 tier remains the binary plan's own Phase 1. Codex
   round: P2 fixed (group-divisibility hard-fails were `assert`s, gone
   under `python -O` — now explicit raises, T2 path included); P1
   ACCEPTED AS THE PHASE BOUNDARY: `src/loader.{h,cpp}` rejects dtypes
   > 5, so the artifact is deliberately unloadable until Phase 3's
   engine work — loader dtype support lands WITH Phase 3, behind this
   plan's pre-registered gate, not before.**
2. **Reference gates BEFORE engine work:** the fork runs DSpark on this
   machine (`--spec-type draft-dspark`, 2026-07-15 measurement). Instrument
   or replay it to dump, for a fixed prompt: the 5 tap vectors, fc output,
   and the 4-token block proposal per round. These become CPU-reference
   fixtures — the port's correctness gates are "same taps in → same block
   out," not end-to-end vibes.
3. **Engine integration (BLOCKED on phase B + repriced oracle):** tap
   capture at layers [1,16,31,46,61] (Metal: copy 5×5120 f32 rows during the
   target's verify chunk — the layer-major loop already visits each layer),
   drafter forward as a second small engine instance (attention-only,
   existing kernels), block proposals feeding the EXISTING batched verify
   (oracle_round's teacher-forced path generalizes: lanes from the drafter
   instead of the reference). Width: one block = 4 lanes; chained blocks
   (8–12 lanes, decaying acceptance) only if the repriced break-even needs
   them — measure, don't assume.
4. **Economics gate:** oracle-style A/B on live prompts vs the post-phase-B
   serial rate. Ship only ≥1.15× net on the agentic mix (below that, the
   complexity isn't paid for; the number is pre-registered here).

## Non-goals

Confidence-head scheduling and log-SNR conditioning in v1 (greedy block
accept/reject through the batched verify is the correctness-equivalent
baseline; scheduling is a Phase-4+ acceptance optimization measured
separately); any drafter training; CUDA-side DSpark (upstream's fork already
has the reference; our CUDA tier has native MTP).
