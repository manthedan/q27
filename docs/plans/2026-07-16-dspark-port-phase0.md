# DSpark port — Phase 0: contract, repack, and gate design (no engine code)

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
- **Markov head:** `markov_head_a [256, 248320]` (type 30 = fork packed) +
  `markov_head_b [256, 248320]` (Q4_1), rank 256 — the confidence-scheduling
  prior (whitepaper §6).
- **Confidence head:** `[5376, 1]` + bias — 5376 = 5120 + 256: hidden concat
  markov features (`confidence_head_with_markov = true`). Gates block
  acceptance scheduling.
- **log-SNR conditioning:** `log_snr_fc1 [128, 5120]` (type 30) + fc2
  [5120, 5120] + biases, clamp [-9, 9] — diffusion-style noise conditioning
  on the mask-fill.
- **Unknown to resolve from the fork source (Phase 0 residue):** type 30's
  exact encoding (fork packed format — repack.py precedent: read at source
  like type 42 was); tap timing (hidden states pre- or post-layernorm at the
  5 layers); feature-window mode (per-token 128–256 slots vs per-cycle
  shift — dflash design doc says build per-token).

## Memory budget (24 GB M4, T2 target)

T2 target 7.15 GB + drafter 1.79 GB + KV (both) + taps ≈ **~9.5 GB total** —
comfortably resident; no policy amendment needed (the one-model-load rule
amendment question from the sibling plan dissolves: this is one target + a
1.79 GB sidecar, not 17+17).

## Phase plan (each gated)

0. **This doc + contract** (done above) + fork-source reads for the three
   unknowns. No GPU.
1. **Repack:** extend `tools/repack.py` for arch `dspark` (Q4_1 + type-30 at
   source, same lossless discipline as bonsai-t2-v1; hard-fail unknown
   slots). Gate: bit-exact round-trip, all 79 tensors.
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
