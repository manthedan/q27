# Official-tier Q4 matvec rewrite round — pre-registration (2026-07-17)

The one round E6 funded (docs/plans/2026-07-17-e6-q4q8-gemv.md RESULTS:
R = 0.716). Daniel's go 2026-07-17 ("sounds good, do it" on the composite
queue). Ship line carried verbatim: **R ≥ 0.90 on the E6 leg re-run, byte
gates green; < 0.90 ships nothing** and the tier keeps the current kernels.

## Diagnosis driving the candidates

Every rewrite target is Q4_G64 (attn q 0.517, ssm/attn out 0.572, ffn pair
0.731 at 43.9% of bytes; the Q8 shapes are at 0.889/1.251). The production
q27_matvec_q4_quantized runs ONE row per simdgroup: each row's simdgroup
re-issues the full activation load (x16 int4 pairs + x_scales) — on
attn q [12288×5120] that is ~63 MB of (cached but issue-bearing) x-loads
against ~31 MB of weight stream, plus one simd_sum and scale-read chain per
row. T2 round-1 measured this exact overhead class and killed it with
4-rows-per-simdgroup + lane-held x-slices (93 GB/s). The Q4 nibble-decode
ALU itself (~4 ops/elem, no integer-dot instruction on Apple GPUs) already
survived one probe (vectorized decode REJECTED at 39–44 vs 54–64 scalar) —
this round does NOT retry decode tricks.

## Candidates (bench arms first, promotion after)

- **A — multi-row restructure (T2 round-1 pattern, primary):** 4 rows per
  simdgroup, each lane holding its 32-column int8 x-slice (2×int4) and
  x-scale in registers; inner loop reads only weights (4 independent
  16 B streams — better ILP), one fused scale-multiply per row-chunk,
  simd_sum ×4 at the end. Decode ALU unchanged by design.
- **B — half-MMA staging (T2 round-2 pattern, fallback):** dequantized
  half tiles through threadgroup memory + simdgroup MMA. GEMV drives at
  most 1/8 of the MMA tile — expected to lose to A on utilization; run
  only if A lands < 0.90 on the leg.
- **A′ — Q8 twin:** if A ships for Q4, the same restructure applies to
  q27_matvec_q8_quantized (head 0.889 has less headroom; A′ ships only
  if it does not regress any Q8 shape).

Arms ride metal_gemv_bench as bench-only candidate PSOs (b1-probe
pattern), gated against the existing exact CPU int8 models (1e-3) on all
8 shapes BEFORE any timing. Promotion = replacing the production kernel
body, then: test-metal, the 16-token official-tier byte gate
(--tokens identity arms vs current binary), and the E6 leg re-run.

## Pre-registered verdict

- E6 leg re-run R ≥ 0.90 → ship; record per-shape table.
- R < 0.90 after both candidates → kill, ship nothing, record the honest
  per-shape table and close the official-tier GEMV question as
  "issue-bound, structure-resistant" alongside the T3 autopsy.
- Per-candidate sub-line: an arm that fails to beat the production kernel
  on BOTH named worst shapes (attn q, ssm/attn out) by ≥ 10% in its bench
  form is not promoted regardless of mix R (avoids shipping noise).

Timing legs on the quiet 24 GB M4 (nothing else loaded); correctness legs
contention-tolerant.

## RESULTS

(pending)
