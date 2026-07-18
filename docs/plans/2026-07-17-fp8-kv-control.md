# FP8-KV control arm — pre-registration (2026-07-17, QUEUED not started)

k3 roadmap item #1, adopted 2026-07-17 (operator: "sounds good, do it" —
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

## Method amendment (2026-07-17, recorded BEFORE the readout ran)

The full-dtype build order is inverted: the QUALITY readout ships first
as `--kl-kv-fp8` — attrib-store mode 4, an e4m3 round-trip of BOTH
sides at store time on the fp16-cache subject — and the production fp8
dtype (store/dequant kernels across the ~8-consumer turbo3 mirror, third
`--kv` value) is built only if the readout says fp8 wins and bytes then
matter.

Why the round-trip arm is valid here when it was demoted for turbo3
tails: the demotion was about SEMANTICS divergence — turbo3's
production attention dequantizes and sums in the WHT domain, so
attrib-domain values ≠ production values at knife-edge positions. e4m3
is transform-free: the round-tripped fp16 values are bit-identical to
what a real fp8 cache would dequantize into attention. The arm is
production-exact for this codec, means AND tails, so the comparison
against the PRODUCTION turbo3 numbers (kl8k.log: mean 0.0115289, max
2.82816 @8K) is apples-to-apples. Bytes need no measurement — e4m3 is
1 B/elem by construction (1024 B/token/side vs turbo3's 400).

Guard rails: SHADER_ABI 10→11 (a stale shader silently stores clean
fp16 for an unknown mode — KL would read 0, vacuously "fp8 perfect";
the mode-3 landing hit exactly this class); unit test pins the e4m3
grid bit-exactly on both sides (RNE ties both directions, subnormal
ties at 2^-9, min-normal boundary, ±448 saturation) against hand
goldens + a CPU mirror across every binade, and doubles as the
anti-vacuity guard.

## RESULTS (2026-07-17, mini; logs/fp8_kv/)

Protocol: production-exact e4m3 arm (method amendment above) vs the
recorded PRODUCTION turbo3 run (kl8k.log), same corpus, same 8,191
positions, same route envs.

- **8K formal arm: fp8 mean 0.000626 nats, max 0.0941 @7719, p99
  0.00565 — vs turbo3's 0.0115289 / 2.82816. Mean 18.4× LOWER (the
  band asked ≤1.1×), max 30× lower, depth-flat (0.00057 → 0.00064
  running mean), ZERO positions above 0.1.** The pos-1000 event does
  not exist under e4m3. 1,536-pos smoke agrees (0.00058 / 0.0355; sits
  2.6× above the fp16-route envelope floor, so the read is not
  vacuous — and the ABI bump + bit-exact store test guard the
  silent-fp16 failure mode independently.
- **Band 1 fires: turbo3's WHT+3-bit structure is NOT earning its keep
  on quality. It is a RATE point, now validated as only that: its whole
  case is 2.56× fewer bytes (800 vs 2048 B/token both-sides).** P1
  steps 2–4 pivot, per pre-registration, to "when do bytes matter"
  (context-depth banded).
- Trade table at 8K (bytes relative to turbo3):
  turbo3 1.0× → mean 0.0115, max 2.83;
  funded L7-full 1.258× → 0.0104, 0.755;
  fp8 2.56× → 0.00063, 0.094;
  fp16 5.12× → 0.
  fp8 beats the funded config 8× on max and 16× on mean for ~2× its
  byte premium — the mid-rate L7 code (option 2 of the funding
  decision) now competes against "fp8 on the hot cells", which needs
  no format design at all.
- Production-dtype work (the ~8-kernel turbo3 mirror, third --kv
  value) stays UNBUILT per the amendment until a serving decision
  actually wants fp8 bytes in the cache; the quality question this
  plan was opened for is answered.
