# FP8-KV control arm — pre-registration (2026-07-17, QUEUED not started)

k3 roadmap item #1, adopted 2026-07-17 (Daniel: "sounds good, do it" —
queued behind the two funded kernel rounds). The missing baseline for the
KV-codec P1 workstream: turbo3 (WHT + 3-bit + per-block scale, 100 B per
head-token-half... 50 B per 128-half) is measured at KL mean 0.0115 / max
2.83 nats with a K-owned tail, but has never been compared against the
obvious simpler codec. If plain e4m3 matches turbo3's KL, the WHT
machinery buys nothing but bytes (SPEC already priced fp8 at 1.1 GB vs
turbo3's 0.4 GB at 32K); if fp8 is worse, turbo3 is validated by control
rather than by default. Either result redirects KV-codec P1 steps 2–4.

## Scope

- e4m3 store kernel pair (serial + rows variants) — manual bit-math (MSL
  has no native fp8): sign, 4-bit exponent bias 7, 3-bit mantissa, RNE,
  saturate to ±448; NaN-free by construction on finite inputs.
- Dequant in every attention consumer the fp8 dtype reaches: per-head
  decode kernel, blocked-GQA pair, causal rows pair (grep the turbo3
  dtype's consumer list and mirror it exactly — the k3 estimate of "two
  kernels" undercounts; honest count is ~5 sites plus engine/CLI wiring
  for a third `--kv` value).
- Rides the standing --kl-kv instrument unchanged: fp16 baseline vs fp8
  subject, same corpus, same cells as the turbo3 run.

## Pre-registered readout (bands before the run)

Same-protocol KL vs the recorded turbo3 numbers (mean 0.0115, max 2.83):
- fp8 mean KL ≤ 1.1× turbo3's → turbo3's structure NOT earning its keep
  on quality; the 2.56× byte cost becomes the whole argument, and P1
  steps 2–4 pivot to "when do bytes matter" (context-depth banded).
- fp8 mean KL ≥ 2× turbo3's → turbo3 validated by control; P1 proceeds
  as planned; record and close.
- Between → report both numbers, no unilateral pivot; the K-owned tail
  comparison (per-cell census) decides.

~3–4 days honest scope. No model quality claims until --kl-kv lands.

## RESULTS

(pending)
