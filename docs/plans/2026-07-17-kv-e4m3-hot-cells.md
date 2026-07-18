# e4m3 on the hot cells — pre-registration (2026-07-17, before any arm ran)

the operator: "hot cells" (follow-up to the fp8-KV control arm's band-1 result,
2026-07-17-fp8-kv-control.md). Question: does e4m3 fidelity on ONLY layer
7's 8 cells — turbo3 everywhere else — kill the production pos-1000 event
the way the funded fp16 L7-full config does? Prices at stake per token
(cell = layer×head×side; turbo3 100 B, e4m3 256 B, fp16 512 B):

- funded fp16 L7-full: +8×412 B = **+25.8%** of turbo3 KV (max 2.94→0.755)
- mid-rate 2× code (option 2, undesigned): ~**+12.9%**, amplification risk
- **e4m3 L7-full: +8×156 B = +9.75%** — cheaper than the design option,
  and needs no format work to MEASURE.

The genuine risk is amplification: 4b (l7h1v) and the layer sweep
(h0+h1 union) both showed PARTIAL fidelity amplifying the knife-edge
event. e4m3 is partial fidelity relative to the fp16 sides that
graduated. This is exactly the question the production instrument must
answer; the attrib instrument is demoted for this class.

## Instrument (production-exact, same discipline as the funded gate)

The KV exception machinery runs unchanged (turbo3 store + dispatch,
side caches, f16 window re-attention overwriting the masked heads'
rows); the SIDE-STORE kernel gains a codec argument — codec 1 rounds
each value onto the e4m3 grid (the proven `q27_e4m3_roundtrip`, RNE,
±448, subnormals) before storing as half. e4m3 values are exactly
representable in half, so every downstream kernel is untouched and the
window attention computes on exactly the operands a real 1-byte e4m3
side cache would dequantize. Values-exact; only the BYTES are simulated
(the buffers stay half-sized — +9.75% is the price of the real side
format, a small follow-up if this graduates: side caches are our own
code, not the 8-kernel turbo3 mirror).

- Env: `Q27_METAL_KV_CELLS_CODEC=e4m3` (default fp16; rejected loudly
  without `Q27_METAL_KV_FP16_CELLS`, unknown values rejected).
- SHADER_ABI 11→12 (argument-struct change on an existing kernel —
  the Q4-round lesson; a stale shader would silently store fp16 sides
  and reproduce the funded config's numbers, faking a PASS).
- Snapshot config identity: header reserved bit 2 carries the side
  codec; mismatch rejects loudly both ways. (Binaries older than this
  commit ignore bit 2 — recorded cross-version caveat, instrument-
  grade.) Server disk-store tag: config token 'x'→'y' under e4m3.
- Unit: side-store test extended — codec 1 lands e4m3 grid values
  bit-exactly (vs the CPU mirror), codec 0 byte-identical to before.
- Domain nuance, recorded: the side store runs in the WHT domain by
  design (outputs compose with the turbo3 path pre-inverse-WHT), so
  this arm measures **e4m3 on WHT-domain rows** — which is exactly what
  a real 1-byte side cache in this machinery would store. It is a
  different codec point from the control arm's original-domain e4m3
  (full cache); both are production-exact for their respective
  configs.

## Pre-registered arms and bands (in order; each later arm needs the earlier)

Protocol identical to the production re-selection sweep: T2 artifact,
wikitext-2, `--kl-kv` with subject = turbo3 + env, route envs
GEMM_HALF=1 TILE=2 THRESHOLD=2048 BLOCK=1024. Comparators (recorded):
1,536-pos control max 2.772 @1000; fp16 L7-full 0.672 @741. 8K control
max 2.94 / mean 0.0116; funded fp16 L7-full 0.7546 / 0.0104.

1. **1,536 screen** (e4m3 cells 8–15): PASS to the 8K arm if max ≤
   1.30. KILL if max > 2.0 — a partial cut in control's class is not
   competitive with the funded config; record amplification if max >
   control 2.772.
2. **8K formal**: max ≤ 1.30 AND mean ≤ 0.0116 → e4m3-L7 GRADUATES as
   a priced alternative: +9.75% for the event kill, vs +25.8% funded.
   The selection between them (and whether to fund the 1-byte side
   format) is the operator's, per standing practice.
3. Suite green + default-off byte identity ride along (codec unset =
   fp16 sides; the funded config's behavior must be bit-unchanged).

## RESULTS (2026-07-17, mini; logs/kv_e4m3/)

**1,536 screen: KILLED at the pre-registered line.** e4m3 L7-full:
max 2.278 @pos 1000 (kill line > 2.0; control 2.772, fp16 sides
0.672), mean 0.0130 — WORSE than control's 0.0122 (+6%; fp16 sides:
0.0100). p99 0.116 / p99.5 0.227. The 8K arm does not run.

Reading: partial fidelity on the hot cells is not a partial kill — it
is the anti-additive knife-edge again, now at 8-bit-float fidelity.
e4m3-in-WHT on all four L7 pairs lands almost exactly where fp16 on
the l7h1 pair alone did (2.265): the event reconstitutes through the
residual quantization, and the perturbed near-tie sum drags the BODY
down too (mean above control). Not a vacuity: the number moved off
both the fp16-sides value (0.672) and control (2.772), the unit test
pins the codec path bit-exactly, and ABI 12 forced the shader.

Dispositions:
- The funded fp16 L7-full config (+25.8%) stands as the ONLY measured
  kill of the production event. Its price is what the event costs.
- Option 2 (mid-rate ~2× turbo3-family code on L7, ~+12.9%) is now
  strongly disfavored by evidence, not just unmeasured: an 8-bit
  FLOAT code failed to kill; a ~6-bit code has no plausible path.
  Recommend closing it unless the operator wants the arm anyway.
- The e4m3 machinery (codec knob, ABI 12, snapshot/tag identity)
  stays in-tree as the reproducible killed experiment, default-off,
  funded-config behavior bit-unchanged (codec-0 store byte-identity
  unit-pinned).
- Program note: three independent instruments now agree the pos-1000
  event is a knife-edge multi-cell near-tie that only FULL fp16 on
  all of L7 resolves — quantization structure (WHT vs float grid) and
  rate (3 vs 8 bit) both fail below that threshold.
