# KV-codec step 1 — kl-kv tail instrumentation + K/V attribution arms

Status: PRE-REGISTERED before measurement (results appended below).
Parent: 2026-07-16-paper-scan-triage.md §"New workstream: P1 KV codec",
step 1 (instrument prep). Calibration bar this feeds: turbo3 mean 0.0115
nats depth-flat, max per-position 2.83 (2026-07-15, 8K wikitext, mini).

## Instrument changes

- `--kl-kv` report gains a tail block: p50/p90/p99/p99.5/max per-position
  KL with the max's position, plus consecutive-high-KL run structure at
  two fixed thresholds (0.1 and 0.5 nats): positions above, number of
  maximal runs, longest run and its start. Same quantile convention as
  the envelope instrument (sorted index ⌊p·n⌋, clamped).
- New attribution arms `--kl-kv-k` / `--kl-kv-v`: the subject engine
  keeps the fp16 KV cache and fp16 attention kernels, but the store
  round-trips ONE side (K or V) through the exact turbo3 quantizer —
  normalize, ±1 sign flips, butterfly, 8-centroid nearest, half-rounded
  norm-correction scale — and back through the inverse transform
  (`q27_kv_store_f16_attrib_rows`; scale passes through half exactly as
  the packed block header does). The KL against the fp16 baseline is
  then attributable to that one side's quantization alone, with no mixed-
  format attention kernel. Engine knob `set_kv_attrib` (fp16-KV engines
  only, before any token); routed at all four fp16 store sites (decode,
  chunk, both MTP paths) so the semantics hold on any stream.

## Run plan

T2 artifact, wikitext2, 8,192 positions (`--nll-long 8192 --ctx 8192`) —
the exact config of the 2026-07-15 calibration — three arms: `--kl-kv`
(both sides), `--kl-kv-k`, `--kl-kv-v`. ~14 min each on the mini.

## Pre-registered reads

1. **Reproduction check (both-sides arm):** the GPU is deterministic
   (envelope repeat class = exactly 0), so the re-run must reproduce
   mean 0.0115 / max 2.83 exactly. A deviation is a regression in this
   change, not a measurement.
2. **Additivity:** K-arm + V-arm mean KL ≈ both-arm mean (smoke at 128
   positions: 0.00255 + 0.00541 = 0.00796 vs 0.00765, ~4% sub-additive).
   Strong super-additivity would mean joint K×V interaction dominates and
   step 4's allocation cannot treat sides independently.
3. **Tail ownership:** which side's arm carries the 2.83-class max and
   the p99. This orders step 2's scaling arms and step 3's census.
   (Smoke suggests V ≈ 2× K on the mean; the tail may differ.)
4. **Run structure:** longest run at 0.1/0.5 nats. Runs ≫ 1 ⇒ damage
   propagates through state (GDN carry), isolated spikes ⇒ per-position
   near-ties; the two point at different codec remedies.

No pass/fail gate — this is instrumentation; the numbers become the
attribution baseline for steps 2–4.

---

## Results (appended post-measurement, same afternoon, M4 16 GB)

8,192 wikitext2 positions per arm, ctx 8192, current default route:

| arm | mean KL | p99 / p99.5 | max (nats) | runs >0.1 | runs >0.5 |
|---|---|---|---|---|---|
| both sides (turbo3) | 0.01165 | 0.089 / 0.163 | 2.937 @pos 1000 | 69 pos, 64 runs, longest 3 | 6 pos, all length 1 |
| K-only | 0.00677 | 0.064 / 0.118 | 2.522 @pos 1000 | 48 pos, 44 runs, longest 3 | 5 pos, all length 1 |
| V-only | 0.00526 | 0.036 / 0.054 | 0.441 @pos 3877 | 12 pos, 11 runs, longest 2 | **0** |

Reads, in pre-registered order:

1. **Reproduction — resolved as a route-default change, then reproduced
   exactly.** The both-sides re-run deviated from the 2026-07-15
   calibration (mean 0.01165 vs 0.01153, max 2.937 vs 2.828). Timeline:
   the calibration was recorded 13:11 Jul 15; the half-staged MMA chunk
   GEMM (float-class numerics — the envelope GEMM class) became the
   default at 14:28 the same day. A/B at 2,048 positions:
   `Q27_METAL_GQA_TILE` 1 vs 2 — **bit-identical** (tiling is not the
   differencer at ≤2K); `Q27_METAL_GEMM_HALF=0` — **reproduces the
   calibration's 0–2k bucket to every printed digit** (mean 0.0122604,
   max 2.82816). Attribution is 100% GEMM half-staging; the A1 16-token
   tile grid and wide-chunk reshapes are bit-preserving on this path,
   exactly as their landing gates claimed. Margin-aware note: this is
   the predicted behavior — an instrument's absolute calibration shifts
   within its class envelope when a same-class route lands; the shift
   (mean +1.0%, per-position max +0.11 at the same position) sits well
   inside the GEMM class contract, and the old bar stays recoverable
   via the class knob.
2. **Additivity holds:** 0.00677 + 0.00526 = 0.01203 vs 0.01165
   (~3% sub-additive). Steps 2–4 may treat sides independently.
3. **The heavy tail is K-owned.** V-only never reaches 0.5 nats
   (max 0.441); K-only reaches 2.52 at exactly the both-arm max
   position (1000). K also carries 58% of the mean at 8K — inverting
   the 128-position smoke where V led ~2:1, so V dominates shallow
   contexts and K dominates at depth. This is direct instrument support
   for the Block-GTQ premise (structured allocation pays on K/attention
   scores) and orders step 2's scaling arms K-first.
4. **The tail is spikes, not bursts:** longest run 3 at 0.1 nats, every
   0.5-run length 1. Per-position near-ties, not state-propagated
   damage — a codec remedy needs better worst-row K fidelity, not
   anti-drift machinery.

### Updated calibration bar (current default route, this instrument)

Both-sides: **mean 0.01165 / p99 0.089 / max 2.94**. Step 2's
pre-registered graduation bar restated against it: an arm must cut
max ≥ 5× (2.94 → ≤ 0.59) or p99 ≥ 3× (0.089 → ≤ 0.030) at unchanged
mean (±10%). Given read 3, arms that scale K are the live candidates;
V-side scaling cannot move the max by more than its own 0.44 ceiling.
The 2026-07-15 numbers remain valid for the float-staged route
(`Q27_METAL_GEMM_HALF=0`) and are reproducible to the digit.
