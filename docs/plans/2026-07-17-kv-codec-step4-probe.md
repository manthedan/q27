# KV-codec step 4 probe: the census exception list, measured jointly

2026-07-17, mini. Pre-registered BEFORE implementation/measurement.

## Question

The census (step 3) says KV damage is not flat: L7 h1 owns the ≥0.3-nat
single-cell tail class with both its cells (V 1.204 / K 1.100) and L63
h2/h3 V own the mean (0.000777 / 0.00113, the two highest cell means).
Step 2 parked all scaling levers (oracle scales miss the bars). The one
live codec shape left is targeted precision: **hold exactly those 4
cells in fp16, quantize the other 124 as usual — what does it buy on
the both-sides instrument?** Census read 4 (sub-additivity 2.32×) says
per-cell numbers overstate joint effects, so this must be measured
jointly, not summed.

Cost of the hold-out at ctx 2048+: 4/128 cells ≈ 3.1% of KV bytes
(fp16 vs ~3.2 bpv turbo3 for those rows ≈ +2.7% total KV bytes).

## Instrument extension (mode 3: exception mask)

`q27_kv_store_f16_attrib_rows` gains mode 3 = quantize BOTH sides
through the turbo3 round-trip UNLESS the (head, side) bit is set in a
per-layer 8-bit exception mask (bit = head*2 + side, matching census
cell numbering `attn_idx*8 + head*2 + side`). The mask rides the
existing `head` argument; engine holds 16 per-attn-layer masks; MTP KV
(not a census cell) is never excepted, matching how modes 1/2 treat it
and keeping the empty-mask control comparable to the step-1 both-sides
class. CLI: `--kl-kv-except N[,N...]` (census cell numbers; empty set
via `--kl-kv-except -` is the control arm).

## Gates before measurement

1. Full-mask vacuity: `--kl-kv-except` all 128 cells → subject stores
   clean fp16 everywhere → KL exactly 0 (same class as --kl-kv-self).
2. Unit suites stay green (mode 3 touches only the attrib kernel's
   selection; production paths don't pass mode 3).
3. Control-arm class check (arm A below doubles as it).

## Arms (8,191 positions, ctx 8192, pinned route, ~11 min each)

- **A (control):** mode 3, empty mask — everything quantized through
  the attrib round-trip. Expectation: lands near the step-1 additivity
  sum for both sides (K-attrib 0.00677 + V-attrib 0.00526 ≈ 0.0120)
  rather than exactly the turbo3-engine 0.01165 (same-class, ~3%
  sub-additive). If A is wildly off (>±15% of 0.0120), the mode-3
  plumbing is suspect — STOP.
- **B (probe):** mode 3, mask = census cells {10, 11, 125, 127}
  (L7 h1 K, L7 h1 V, L63 h2 V, L63 h3 V).

## Pre-registered reads

1. **Tail:** does B cut A's max ≥ 2× and/or move A's argmax off the
   worst position? (The census says L7 h1 owns the single-cell tail;
   if the JOINT tail has a different owner — read-3's multi-cell sum —
   the hold-out may barely move max. Either answer shapes step 4.)
2. **Mean:** B vs A mean delta, against the naive per-cell prediction
   (Σ of the 4 held cells' census means = 0.00311 at 2K scale —
   sub-additivity says expect LESS). Report the realized fraction.
3. **Decision:** the probe GRADUATES into a codec design step only if
   B cuts A's max ≥ 2× or p99 ≥ 1.5× at ≤ 3.5% KV-byte cost. Below
   that, targeted-precision joins scaling in the parked list and the
   KV-codec program STOPS at "turbo3 as shipped, damage priced in"
   (0.0115 mean / 2.83 max class, recorded 2026-07-15) — the honest
   outcome of steps 1–4 would then be: the tail is multi-cell,
   position-local, and not addressable by any per-cell lever measured.

Layer-only marginals, V-only holds, or bigger masks are follow-ups
only if B graduates.
