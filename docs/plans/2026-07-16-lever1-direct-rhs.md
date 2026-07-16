# Lever 1 — direct-RHS chunk GEMM (verify-round-cost plan, kernel phase)

Status: PRE-REGISTERED before measurement (results appended below the
line). Parent: 2026-07-16-verify-round-cost.md lever 1 (P0 confirmed at
97.5% verify-batch; kill lines inherited). Design source: ds4-survey item
2/3 phase B — "stage only the weight tile, read the RHS directly from
device, spend all threadgroup memory on weights", classic tile 64 rows ×
32 tokens.

## Why the current kernel is slow at verify widths (from today's roofline)

At w ≤ 16 the weight stream is already read exactly once, so "effective
GB/s" is just weight-bytes / kernel-time: the wall is kernel time, not
traffic. Today's mm_h at x_rows 96 runs ~2.65 TFLOP/s physical MMA vs the
~3.65 TFLOP/s A-arm ceiling (cadence residual Beq/A = 1.27), and a
12-token verify pays 16-token-tile MMA work (25% padding). Ceiling
arithmetic at w=12 on ffn [17408×5120]: useful 2.14 GFLOP at 3.65 TFLOP/s
with zero padding ≈ 0.59 ms ≈ 40 GB/s effective — the plan's ≥2× target
(21 → 40+) is exactly "run at the MMA ceiling with no wasted lanes."
Direct-RHS attacks the two gaps: RHS staging + its barrier share
(cadence), and the fixed 16-token tile (padding at w=12 stays 25% via an
8-column-bounded loop, but the 64-row tile halves flush/store overhead per
row and the freed threadgroup memory keeps the weight tile resident).

## Kernel design (q27_matmul_t2_mm_dr, bench-first)

- Tile: 64 rows × up-to-32 tokens per threadgroup, 256 threads =
  8 simdgroups; each simdgroup owns an 8-row stripe × all live token
  columns. Grid: (rows/64) × ceil(x_rows/32). All production row counts
  are multiples of 64 (17408, 5120, 10240, 6144, 12288, 1024, 248320) —
  row clamps kept for safety, never exercised in production.
- Weight staging: 64 rows × 32 K half tile (4 KB) via the byte-LUT unpack
  landed this morning (`q27_t2_half4_lut`), one threadgroup barrier per
  K-tile. Weight side stays integer-exact in half.
- RHS: **not staged.** The verify/prefill activations are pre-converted
  once per dispatch by a tiny pre-pass kernel (`q27_x_int8_to_half_t`):
  int8 [x_rows, cols] → half **K-major** [cols, 32-token-padded], raw
  int8 values (exact in half), zero-padded token slots. simdgroup_load
  reads B fragments directly from device with natural (non-transposed)
  addressing. Pre-pass writes ≤ 1.1 MB — noise next to the 22 MB weight
  stream.
- Numerics: per-K-tile flush, and K-tile = 32 = exactly one activation-
  scale group, so the racc fold applies ws(row, c/128) × xs(tok, c/32) to
  exact integer partial sums — same exactness argument as mm_h (int8 ×
  trit sums bounded well inside float). Fold order differs from mm_h
  (32-K flush vs 64-K sub-slab pairs), so outputs are NOT bit-identical
  to mm_h: op-level tolerance gate (existing 3e-4 chunk-GEMM class) +
  end-to-end artifact committed-token byte-identity carry the quality
  contract, exactly as the mm → mm_h transition did.
- Token-column loop bounded by ceil(live/8) fragments (w=12 → 2 columns,
  25% pad — same as today; w=32 → 4). No specialization constants in the
  bench phase; runtime args only (FC variants are a later production
  refinement per ds4 item 3 if the kernel graduates).

## Measurement (extend metal_mma_roofline: --xrows flag + arm 'd')

Arms C (production mm_h) vs D (direct-RHS incl. its pre-pass dispatch) at
x_rows ∈ {12, 96}, same paired-trial statistics as the roofline. The
pre-pass cost is charged to arm D (it is part of the lever's dispatch
sequence). Correctness before timing: D vs C outputs, max relative
difference ≤ 1e-3 on live outputs (fold-order-only difference on exact
integer partials; expected ~1e-6).

## Pre-registered decision lines (inherited from the parent plan)

- Primary metric: t_C / t_D at x_rows = 12 (the verify shape), aggregate
  over the production mix, lower 95% CB.
- **< 1.3× → PARK lever 1** ("the direct-RHS shape doesn't fit the M4"),
  re-scope around lever 2 only.
- **≥ 1.3× → integrate** behind the verify path (and evaluate for
  prefill), then the parent plan's gates: chunk shape suite, artifact
  committed tokens byte-identical, oracle re-sweep with **S(12) ≥ 3× as
  the acceptance measure** (round ~430 → ~250 ms class).
- x_rows = 96 is reported for the prefill double-pay claim but does not
  gate the verify decision.

---

## Results (appended post-measurement, same day, M4 16 GB)

**VERDICT: PARK lever 1** (the parent plan's kill line: < 1.3× at width 12
→ "the direct-RHS shape doesn't fit the M4; re-scope around lever 2
only"). Best mapping C/D2 = **0.782 [0.780, 0.785]** at x_rows 12 — the
direct-RHS kernel is 22% SLOWER than production mm_h, with the park line
at 1.3× FASTER. Not close; no CI ambiguity.

### Two mappings, one diagnosis (both correctness-gated ≤ 1e-3 vs C)

| arm | mapping | C/arm @ w=12 |
|---|---|---|
| D | 8 simdgroups × (8-row stripe × all tokens): every simdgroup issues IDENTICAL device B loads (8× redundant) | 0.361 [0.360, 0.362] |
| D2 | 4 simdgroups × (32-row × 8-token quadrants): B redundancy 2×, all simdgroups live at w=12 | 0.782 [0.780, 0.785] |

The D → D2 doubling confirms the attribution: redundant device
`simdgroup_load`s of the RHS are the dominant cost, and even at 2×
redundancy the device fragment loads plus the per-32-K flush cadence
(double mm_h's 64-K sub-slab structure) lose to production's staged-Xt
design. The third nail is this morning's Cx arm: removing mm_h's RHS
staging cost entirely is worth only ~2% — there was never much to win by
not staging the RHS, and direct device reads cost far more than the
staging they replace.

### The transfer hypothesis is refuted, and the why is recorded

ds4's "direct-RHS was the clear win" runs on Metal-4 cooperative tensors
(M5/M6). The survey's hypothesis was that the *structure* — don't stage
the RHS, spend threadgroup memory on weights — transfers to plain
`simdgroup_matrix` code. Measured: it does not, on M4. Plain
`simdgroup_load` from device pays per-fragment issue cost per simdgroup
with no cooperative amortization, which is exactly what threadgroup
staging amortizes. The idea can be re-priced if a future rig has
cooperative tensors; the probe surface stays in tree
(`metal_mma_roofline --xrows 12`, arms D/D2).

### Consequences (parent plan re-ranked per its own kill line)

- **Lever 2 (verify width past 12) is now the whole prize**: decouple
  `VERIFY_CHUNK_MAX` from the NLL/KL width-12 contract; gates as
  pre-registered (tiled parity at widths actually dispatched, oracle
  sweep w ∈ {16, 24, 48}, committed byte-identity vs width-12 verify;
  kill line: S(48) < 3× → stop at the width the curve flattens).
- Note for the oracle re-sweep baseline: today's byte-LUT + vector-store
  round already ships +2.5% on the chunk GEMM the verify batch rides, so
  the ~424 ms round should re-measure slightly better before lever 2
  adds anything.
- Unregistered residue (recorded, not scheduled): a 64-row tile WITH
  staged RHS (not direct-RHS) was not measured; the Cx result caps its
  RHS-side upside at ~2%, so it would need to win on flush/row-tile
  economics alone.
