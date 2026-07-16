# A/B/C MMA roofline — the one-day prefill-maturity decision (2026-07-16)

Status: PRE-REGISTERED before measurement (results appended below the
line). Source: round-3 answers Q1 (ADOPT). The question: closing the
48.79 → 52.48 tok/s fork gap needs an 8.1% GEMM-kernel speedup with GEMM
at 94.3% of prefill GPU time — is that headroom real, or is M4 prefill MMA
mature at this schedule? "94.3%" proves Amdahl dominance and "3.3 GB/s"
proves DRAM irrelevance, but neither measures the fraction of peak
`simdgroup_multiply_accumulate` issue rate reached.

## The three kernels (identical production geometry)

Identical dispatch grid (`ceil(rows/32) × ceil(x_rows/16)` threadgroups of
128), tile size (32-row × 16-token, 64-K tiles), edge behavior (row/token
clamps), and per-simdgroup MMA sequence (2 sub-slabs × 4 steps × 2 MMAs =
16 half 8×8×8 MMAs per 64-K tile; physical MMA count per dispatch =
tgs × cols simdgroup-MMAs, 1024 FLOPs each).

- **C — full production T2 G′**: `q27_matmul_t2_mm_h` invoked through the
  production `matmul_quantized` entry (gemm-half default) on synthetic T2
  weights. The baseline being judged.
- **B — half-plumbing roofline** (`q27_mma_roofline_b`, bench-only): the
  mm_h body with the packed-trit unpack and int8→half conversion deleted —
  weight and activation operands are already half in device memory and are
  staged into the same threadgroup tiles with the same store pattern, same
  barriers, same MMA loops, same per-64-K flush with the same
  ws×xs float scale-fold epilogue. B−C isolates the cost of unpack +
  conversion machinery ONLY. (B streams 8× the weight bytes of C — half vs
  2-bit; at these dispatch rates that is still far below the stream roof,
  and the direction of any bandwidth penalty is conservative: it can only
  understate headroom, never fabricate it.)
- **A — MMA-core roofline** (`q27_mma_roofline_a`, bench-only): tiles
  filled once from an opaque device seed, then the same per-64-K MMA loop
  count with accumulators carried across the whole K walk (dependency
  preserved, nothing resettable/DCE-able), one final fold + store. No
  per-tile staging, no barriers, no flushes. A−B isolates staging/barrier/
  flush cadence at the same MMA count.

Anti-vacuity guards (lesson applied): outputs of all three arms are read
back and must be finite and nonzero; C is additionally the production
route, so its output is real by construction.

## Shapes and weighting

The production per-chunk projection mix at x_rows = 96 (PREFILL_CHUNK_MAX;
96 = 6 exact 16-token tiles, and every production row count is a multiple
of 32 — no edge tiles): ffn gate/up [17408×5120]×128, ffn down
[5120×17408]×64, gdn qkv [10240×5120]×48, gdn gate [6144×5120]×48,
ssm/attn out [5120×6144]×64, attn q [12288×5120]×16, attn k/v
[1024×5120]×32 per 64-layer chunk pass; head excluded (12-row slices on
the serial path). Aggregate R weights each shape by its C-time share —
that is how the fork gap composes.

## Pre-registered reads (expert table, adopted verbatim)

Ratios use the **lower 95% confidence bound** of R = t_C / t_B (10 trials,
t-distribution on the mean; conservative bound = C_low / B_high), never a
best run.

| signal | verdict |
|---|---|
| B/C ≤ ~1.08 | no unpack/staging headroom to close the gap — **stop; classify M4 prefill MMA as mature** |
| B/C 1.08–1.15 | permit **exactly one** targeted kernel round |
| B/C > 1.15 | real implementation headroom; pursue |
| A/B small | at the shape-specific MMA ceiling |
| A/B large | barriers / tile movement / accumulator folding are the residual problem |

Lever ordering if (and only if) the roofline reopens work, from the
round-3 doc: K=128 flush cadence only if plumbing accounts for >8–10%;
prescaled-half operands reopen only if B/C implicates raw-int
staging/flush; barrier reduction only on a measured production
arrival-stall; half accumulators stay dead (the +0.43% NLL was a real
regression); no scalar/MMA interleave bet.

Bottom line adopted with the design: one attribution experiment is worth
chasing; the 7% itself is not an open-ended optimization stream. If the
read says mature, prefill closes as mature and the roadmap moves on.

---

## Results (appended post-measurement, same day, M4 16 GB)

**VERDICT: B/C ≤ 1.08 → STOP. M4 prefill MMA is classified MATURE at this
schedule.** Aggregate R = t_C/t_B = **1.010**, lower 95% CB **0.992**
(C-time-weighted across the production mix). The unpack + int8→half
machinery the fork doesn't carry costs ~1% — the 8.1% GEMM speedup needed
to close the 48.79 → 52.48 fork gap is not there. Per the adopted bottom
line, the prefill-maturity decision is closed; the remaining gap is
schedule-level, not implementation debt in the T2 plumbing.

### Numbers (8 reps × 10 trials, ms/dispatch, x_rows 96)

| shape | C | B | A | C/B [95% CI] | B/A [95% CI] | C TFLOP/s |
|---|---|---|---|---|---|---|
| ffn gate/up [17408×5120] | 6.955 | 6.662 | 4.780 | 1.044 [1.036,1.052] | 1.394 [1.380,1.408] | 2.46 |
| ffn down [5120×17408] | 6.916 | 7.575 | 4.826 | **0.913** [0.893,0.934] | 1.570 [1.532,1.608] | 2.47 |
| gdn qkv [10240×5120] | 4.214 | 4.005 | 2.892 | 1.052 [1.032,1.073] | 1.385 [1.350,1.421] | 2.39 |
| gdn gate [6144×5120] | 2.531 | 2.447 | 1.754 | 1.034 [1.007,1.063] | 1.395 [1.347,1.445] | 2.39 |
| ssm/attn out [5120×6144] | 2.519 | 2.415 | 1.737 | 1.043 [1.005,1.083] | 1.391 [1.345,1.438] | 2.40 |
| attn q [12288×5120] | 5.025 | 4.817 | 3.479 | 1.043 [1.025,1.062] | 1.385 [1.356,1.414] | 2.40 |
| attn k/v [1024×5120] | 0.484 | 0.482 | 0.337 | 1.006 [0.892,1.132] | 1.429 [1.315,1.550] | 2.08 |
| **aggregate** | | | | **1.010 (lo 0.992)** | | |

Anti-vacuity gates passed (all arms finite + nonzero); CIs are tight
except the small attn k/v shape, whose weight in the aggregate is minor.
ffn down's B < C is the pre-registered conservative direction realized:
half weights stream 8× the device bytes of packed T2, and at [5120,17408]
geometry that traffic actually bites — meaning production's packed
streaming is a net WIN there, not debt. Even the most optimistic per-shape
upper bound (1.13, attn k/v) stays under the 1.15 pursue line.

### Attribution reading (A/B), recorded for the file

B/A ≈ 1.39–1.57: staging + barriers + per-64-K flush cadence holds
~28–36% of arm B's wall — production runs ~2.4 TFLOP/s of physical MMA
against a ~3.5 TFLOP/s shape-specific ceiling (arm A). Per the expert
table that names "barriers / tile movement / accumulator folding" as the
residual problem — but the pre-registered lever ordering permits barrier
work only on a measured *production arrival-stall*, and the adopted
bottom line stands: one attribution experiment, not an open-ended
optimization stream. The K=128 flush-cadence probe condition ("plumbing
accounts for >8–10%") is NOT met (plumbing = 1%), and prescaled-half
variant A stays closed (B/C does not implicate raw-int staging). Recorded
as the known shape of the remaining headroom should a future round-4
expert review choose to re-open it with a stall measurement.

### Consequences

- **Prefill maturity: DECIDED (mature).** The margin-aware program item
  "re-price prescaled-half variant A after the roofline" resolves to
  CLOSED — the roofline does not implicate raw-int staging.
- Roofline surface stays in tree (bench-only): `q27_mma_roofline_a/b`,
  `MetalBackend::mma_roofline`, `tools/metal_mma_roofline.cpp` — one
  command re-establishes the verdict on other hardware.
