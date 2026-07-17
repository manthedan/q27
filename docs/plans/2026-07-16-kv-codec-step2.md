# KV-codec step 2 — KVarN-style scaling arms on the turbo3 round-trip

Status: PRE-REGISTERED, STAGED while the step-3 census runs (arms execute
after it frees the GPU; build/q27-metal is not rebuilt until the census
completes — its run fingerprint pins the binary).
Parent: 2026-07-16-paper-scan-triage.md §KV codec step 2, informed by
step 1 (2026-07-16-kv-codec-step1.md): the tail is K-owned (K max 2.52 /
V max 0.44 at 8K) and spiky, so arms run K-FIRST; V-side scaling cannot
move the both-sides max past V's own 0.44 ceiling.

## Mechanism — measurement rides the round-trip harness

The attribution round-trip (step 1) evaluates any candidate scaling rule
with ZERO attention-kernel changes: apply the rule inside the store-time
quantize→dequant round-trip and read the KL against the fp16 baseline.
The turbo3 quantizer already normalizes per (token, 128-feature group)
— the token axis is covered. KVarN's claim is that the FEATURE axis
(per-dimension variance, stable across tokens) also matters. Arms:

- **current** — step 1's numbers (K 0.00677 / 2.52; both 0.01165 / 2.94).
- **scale32** — the per-group scale kept in f32 instead of half through
  the round-trip: isolates scale-precision from scale-structure. (This
  would change the 50-byte block if productized; as an arm it splits the
  attribution cheaply.)
- **feature** — per-feature descale before the quantizer, rescale after
  the inverse transform: x' = x/s_d, round-trip, out·s_d. Scales
  s[side][attn_idx][head][dim] = RMS over the corpus, collected by a
  stats pass (`--kl-kv-stats`: subject stores clean fp16 + accumulates
  per-feature Σx² for both sides; its KL is exactly 0 by construction —
  a rode-along determinism canary). Clamp: s ≥ 1e-3 × the side's mean
  RMS (recorded constant; guards dead features).
  Two-pass oracle scales are deliberate: if FIXED whole-corpus scales
  cannot cut the tail, no calibration-free online estimate will — this
  is the strongest version of the KVarN hypothesis, so a park is final.
- **both** — feature + scale32.

## Run plan (after census; ~11 min/arm at 8K, matching step 1)

Stats pass (8K), then K:feature, K:scale32, K:both. V arms only if a K
arm graduates or comes close (V's ceiling makes V-first irrelevant).
Driver: `tools/kv_step2_arms.sh` (rebuild-at-top, pinned route knobs,
fingerprint, self-check + stats-zero canaries, resumable).

## Pre-registered graduation bar (triage, restated vs current route)

An arm graduates only if, at unchanged mean (±10% of 0.00677 for K arms
— the arm must not buy the tail by fattening the body):
- max-KL ≥ 5× cut: K-only 2.52 → ≤ 0.50, or
- p99 ≥ 3× cut: K-only 0.064 → ≤ 0.021.

Below the bar: record and STOP — the tail is then not a scale artifact,
and the codec effort moves to step 4's structured allocation (Block-GTQ
shape) with the census's cell targets. Secondary read: scale32 alone
approaching the bar would say the half-rounded scale is the defect —
a much cheaper production fix than any per-feature machinery.
