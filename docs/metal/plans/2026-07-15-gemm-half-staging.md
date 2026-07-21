# Chunk-GEMM half-precision staging (prefill lever)

Prefill is compute-bound in the tiled simdgroup GEMM (this M4: 409.9
ms/chunk q4q8 synthetic, 29.28 tok/s, 31.6 GB/s effective weight stream vs
88–98 for decode GEMV). The fork does pp512 52.48 tok/s on this same
machine — the concrete beatable number (whitepaper item 3). Apple GPUs run
`simdgroup_half8x8` multiply-accumulate at twice the float8x8 rate and half
tiles halve threadgroup traffic/footprint; the untried levers named in the
ledger are half staging and double-buffered K-tiles.

## Numerics ladder (probe speed first, refine exactness second)

- **Variant A (speed probe, this doc's phase 1):** Wt and Xt staged as
  half, `simdgroup_half8x8` accumulators, per-64-K-slab flush through
  scratch into float racc with weight-scale folding — structurally
  identical to today's kernel, types swapped. Weight side stays exact
  (ints ≤ 8 and trits are exact in half); the activation side picks up
  one extra rounding (int8 × x_scale is exact in float24, rounded in
  half11); slab partials accumulate in half (bounded: ≤ 64 products).
  Chunk output is already tolerance-gated vs decode (the chunk↔decode
  bit-parity contract covers attention, not projections), but variant A
  must still pass the matmul shape suite bounds and the artifact
  committed-token A/B before it can be default.
- **Variant B (exactness refinement, only if A's speed prize ≥ ~1.3×):**
  stage BOTH sides as raw integers in half (exact), flush every 32 K so
  both the per-row weight scale and the per-token-per-32-group activation
  scale fold at the flush (racc component (row, token) scales by
  ws(row, slab) × xs(token, slab)). Restores exact integer staging at the
  cost of 2× flush overhead; half accumulation of integer products remains
  rounded above 2048, so this is still not bit-exact — measure both error
  and speed before choosing.
- **Double-buffered K-tiles:** orthogonal; try after A/B verdict.

## Prototype scope

`q27_matmul_t2_mm_h` first (this rig's tier, end-to-end benchable on the
artifact), behind `Q27_METAL_GEMM_HALF=1` in the backend's matmul dispatch.
Q4/Q8 variants follow only if the T2 prize is real.

## Gates

1. `test_matmul_shape` suite run against the half kernel (same tolerance
   bounds; if half staging cannot meet them, that is a verdict, not a
   reason to loosen bounds silently).
2. `metal_prefill_bench --dtype t2` A/B, env off/on, back-to-back.
3. Artifact: 126-token prompt ingest wall + committed tokens vs env-off;
   2K NLL spot (≤0.1% PPL delta) if committed tokens diverge.
4. Codex autoreview.

## Kill criteria

Variant A < 1.15× chunk rate, or shape-suite tolerance failures that
variant B cannot repair: record and park, T3-style.

## Outcome (2026-07-15): variant G′ LANDED — 1.22× chunk rate, float-class numerics

Full ladder, `metal_prefill_bench --dtype t2`, float baseline 396.8–398.7
ms/chunk (30.1–30.2 tok/s), suite = `test_matmul_shape` bounds:

| variant | staging | accumulate | flush | ms/chunk | rate | verdict |
|---|---|---|---|---|---|---|
| A | x·scale in half | half | 64-K | 310.8 | 1.28× | suite FAIL (1.4e-3 > bound) |
| B | raw ints in half | half | 32-K | 323.7 | 1.23× | suite pass, **2K NLL +0.43% FAIL** (sums round past 2048) |
| F | raw ints in half | half | 16-K (sums ≤ 2032, exact) | 376.7 | 1.05× | clean, **below 1.15× kill line** |
| G | raw ints in half | **float (mixed MMA)** | 32-K | 335.5 | 1.18× | clean |
| **G′** | raw ints in half | float (mixed MMA) | 64-K staged, two acc pairs, one barrier region | **326.4** | **1.22×** | **clean — landed** |

Gates: shape suite pass (exit-code-checked — a `tail` pipe had masked
variant A's failure at first; lesson recorded), artifact committed tokens
byte-identical, 2K NLL 9.491 vs 9.483 (+0.08%, within the 0.1% spot bound —
racc fold-order noise, same class as the GQA reorder deltas). Default ON
for T2 (`Q27_METAL_GEMM_HALF=0` opts out). Mixed-precision
`simdgroup_multiply_accumulate` (half operands, float accumulators) is
the load-bearing discovery: half tiles halve threadgroup traffic while
float accumulation keeps int8×trit sums exact to 2^24.

Follow-ups: Q4/Q8 half-staging variants (same mechanical change) need the
24 GB machine — the official-tier artifact gates (incl. MTP-verify
acceptance) cannot run on this rig. Double-buffered K-tiles remain untried
on top of G′.
