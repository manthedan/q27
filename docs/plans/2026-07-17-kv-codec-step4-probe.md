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

Cost of the hold-out (CORRECTED pre-result, 2026-07-17 00:25, codex P2
on the step-2 commit — the original "~3% of KV bytes" counted cells,
not bytes): a turbo3 cell stores 100 B/token (two 50-byte blocks), an
fp16 cell 512 B/token, so 4 held cells cost 4×(512−100)/(128×100) =
**+12.9% of turbo3 KV storage**. This amendment was written while the
8K arms were still running (control mid-pass, probe not started) —
no result was visible when the bar below was re-priced.

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
3. **Decision (re-priced with the corrected 12.9% cost, pre-result):**
   the cut bars stand — B graduates only on max ≥ 2× or p99 ≥ 1.5× vs
   A. But at +12.9% KV bytes a graduating result funds a DESIGN step,
   not a shippable config: find a cheaper encoding of the same
   protection (fp16 for K of L7 h1 alone = +3.2%; or a mid-rate code
   for the listed cells), and any design must beat the null alternative
   of simply spending +12.9% uniformly (e.g. a richer global block
   code). Below the cut bars, targeted-precision joins scaling in the
   parked list and the KV-codec program STOPS at "turbo3 as shipped,
   damage priced in" (0.0115 mean / 2.83 max class, recorded
   2026-07-15) — the honest outcome of steps 1–4 would then be: the
   tail is multi-cell, position-local, and not addressable by any
   per-cell lever measured.

Layer-only marginals, V-only holds, or bigger masks are follow-ups
only if B graduates.

## Results (2026-07-17 00:42, 16 GB mini — GRADUATES on the max bar)

8,191 positions/arm, ctx 8192, pinned route, fingerprinted
(logs/kv_step4/); vacuity gate exactly 0 at launch.

| arm | mean KL | p90 | p99 | p99.5 | max |
|---|---|---|---|---|---|
| A control (all quantized) | 0.011871 | 0.0230 | 0.1006 | 0.169 | 2.478 @1000 |
| B probe (4 cells fp16) | 0.009329 | 0.0185 | 0.0924 | 0.151 | 1.117 @7719 |

Reads, in pre-registered order:

1. **Tail: B cuts A's max 2.22× (2.478 → 1.117) — the ≥2× bar CLEARS —
   and the argmax moves off position 1000 entirely (→ 7719).** Nuance
   against census read 3: no single cell owns pos-1000 (every cell
   < 0.3 there), yet jointly holding these 4 cells kills the event —
   the multi-cell sum is substantially composed of/coupled through the
   held cells. The p99 body barely moves (1.09×, fails its 1.5× bar):
   step 2's "broad near-tie population" verdict stands for the body.
2. **Mean: −21.4% (0.011871 → 0.009329), realized Δ 0.00254 = 71% of
   the naive per-cell sum (0.00359)** — sub-additive again, milder at
   4 cells than the census's 128-cell 2.32×.
3. **Decision (re-priced bar): GRADUATES on max — funds the DESIGN
   step, not a shippable config.** At +12.9% KV bytes the protection
   is real but expensive; the design step must find a cheaper encoding
   (candidates from the data: L7h1-only hold — both its cells — since
   the max cut is the graduating read and L63 V mostly buys mean;
   fp16-K-of-L7h1 alone at +3.2%; or a mid-rate 2× code for the listed
   cells at ~+6%) and must beat the null of spending +12.9% uniformly
   on a richer global block code. Control-arm validity: mean 0.011871
   sits dead-center in the pre-registered class band (~0.0120 ±15%)
   and reproduces the both-sides tail signature (max @pos 1000, 2.478
   vs the turbo3 engine's 2.94 — same event, attrib-instrument class).

Instrument note (codex P1, fixed same night): SHADER_ABI 9 → 10 for
mode 3 — a stale shader would store everything clean and read KL 0,
which the vacuity gate cannot catch (it expects 0); the driver now
also asserts the control arm's class band, the canary that CAN fire
under that skew.
