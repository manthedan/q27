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

**FINAL VERDICT (post-codex corrected methodology): C/Beq = 1.135
[1.134, 1.136] → the (1.08, 1.15] read fires — NOT mature; exactly ONE
targeted kernel round is permitted, aimed at the unpack/conversion ALU.**

### The verdict flipped in the codex round — recorded honestly

The first measurement (landing commit 048fd31) read "mature": aggregate
t_C/t_B = 1.010, lower CB 0.992. Codex raised two P1s that overturned it:

1. **Arm B was traffic-confounded.** Half operands at production layout
   stream 8× the weight bytes of C's packed T2; on ffn_down that traffic
   made B *slower* than C (0.913), dragging the aggregate toward 1 and
   masking the unpack cost. Fix: arm **Beq** loads exactly C's device byte
   volume (one half2 = 4 B per weight tile-slot, one half4 = 8 B per
   activation tile-slot), replicates into the same staging stores, keeps
   barriers/MMA/flush identical — C/Beq isolates plumbing ALU at equal
   traffic. The expert-literal B stays reported as the confounded
   companion (C/B = 1.016 [1.014, 1.018]).
2. **Wrong confidence-bound direction.** Declaring "no headroom" requires
   the UPPER bound ≤ 1.08 (the lower bound only ever establishes
   headroom); the tool now has an explicit inconclusive zone.

Plus two P2s, both fixed: outputs are zero-poisoned before every arm's
anti-vacuity check (an arm that silently writes nothing can no longer
inherit the previous arm's output), and the aggregate CI is now valid —
per-trial paired chunk-aggregate ratios (arms interleaved within each
trial), CI over the 10 ratio observations, instead of a harmonic
combination of non-simultaneous per-shape bounds.

A second codex round found two more bias sources, both fixed and
re-measured: the arm order inside each trial is now rotated (a fixed
C-first order folds thermal/frequency drift into the same side of every
paired ratio), and Beq's activation staging loads were split into two
4-byte half2 loads to match C's two char4 loads (one 8-byte load
understates load-issue cost). Re-measured verdict: **1.135
[1.134, 1.136] — unchanged**; the number was already honest, now the
methodology can prove it.

### Numbers (8 reps × 10 interleaved trials, ms/dispatch, x_rows 96)

| shape | C | Beq | B | A | C/Beq [95% CI] | C TFLOP/s |
|---|---|---|---|---|---|---|
| ffn gate/up [17408×5120] | 6.753 | 5.944 | 6.473 | 4.671 | 1.136 [1.134,1.139] | 2.53 |
| ffn down [5120×17408] | 6.771 | 5.973 | 7.282 | 4.630 | 1.134 [1.131,1.136] | 2.53 |
| gdn qkv [10240×5120] | 4.000 | 3.526 | 3.829 | 2.778 | 1.134 [1.131,1.138] | 2.52 |
| gdn gate [6144×5120] | 2.442 | 2.152 | 2.337 | 1.693 | 1.135 [1.128,1.142] | 2.47 |
| ssm/attn out [5120×6144] | 2.422 | 2.136 | 2.323 | 1.680 | 1.134 [1.130,1.139] | 2.49 |
| attn q [12288×5120] | 4.783 | 4.217 | 4.587 | 3.319 | 1.134 [1.132,1.136] | 2.53 |
| attn k/v [1024×5120] | 0.466 | 0.413 | 0.453 | 0.331 | 1.129 [1.109,1.148] | 2.16 |
| **aggregate C/Beq** | | | | | **1.135 [1.134, 1.136]** | |
| aggregate C/B (literal, confounded) | | | | | 1.014 [1.013, 1.016] | |
| aggregate Beq/A (cadence residual) | | | | | 1.276 [1.274, 1.278] | |

The per-shape C/Beq is remarkably uniform (1.13–1.14 everywhere): the
plumbing tax is a fixed fraction of tile work, independent of shape — a
kernel-structural cost, exactly what a targeted round can attack.

### What the one permitted round should target

- The 13.5% is the per-tile unpack + convert ALU: 16 shift/mask/sub →
  int→half conversions per weight uint, plus 8 char→half conversions per
  activation slot. Primary candidate: **bit-pattern trit→half unpack** —
  a trit's half encoding is one of three constants (0xBC00/0x0000/0x3C00),
  so a select/LUT construction can replace the int→half convert chain.
  Activation-side char→half is the secondary target.
- The K=128 flush-cadence probe condition ("plumbing accounts for
  >8–10%") is now MET (13.5%) — it may ride the same round if the unpack
  fix alone doesn't clear the fork-gap arithmetic (needs 8.1%).
- Prescaled-half variant A stays closed unless the round's own
  measurement implicates the activation-staging side specifically.
- Beq/A = 1.276: staging/barrier/flush cadence still holds ~28% at equal
  MMA count (C runs ~2.5 TFLOP/s physical MMA vs A's ~3.65 ceiling), but
  the barrier lever stays gated on a measured production arrival-stall,
  per the round-3 ordering.
- The expert contract is explicit: this is ONE round with the fork-gap
  arithmetic as its bar (8.1% GEMM-kernel speedup ≈ the measured 13.5%
  budget minus whatever is irreducible), not an open-ended stream. If the
  round lands short, prefill closes as mature with the attempt recorded.

### Consequences

- **Prefill maturity: NOT yet decided mature** — the 048fd31 "mature"
  read is retracted (methodology, not measurement noise). One targeted
  unpack round is authorized by the pre-registered table.
- Roofline surface stays in tree (bench-only): `q27_mma_roofline_a/b/b_eq`,
  `MetalBackend::mma_roofline`, `tools/metal_mma_roofline.cpp` — one
  command re-establishes the verdict, now with valid statistics.

---

## The authorized round (same day)

Four candidates were built and measured against the roofline harness; two
landed, two were rejected by their own numbers. A post-landing codex round
(1 P1 + 1 P2) found the CONTROL arms had fallen behind production — Beq
still used 16 scalar stores after C moved to vector stores, and Cx still
used the scalar LUT — so the residual attributions below are from the
re-run with controls mirroring production staging exactly. (The landed
gain itself is control-independent: Beq is identical across the round, so
it cancels as a bridge.)

Landed production-GEMM speedup: **4.8%** aggregate (C/Beq 1.135 → 1.083
with clean controls; ffn gate/up 6.75 → 6.42 ms = +5.2%, attn q +2.9%;
final C ≈ 2.65 TFLOP/s physical MMA). Whole-prefill replay
(`metal_prefill_bench --dtype t2`, two reproducible runs per arm, pre
shader pinned via `Q27_METAL_SOURCE` with the old mm_h body spliced into
the current file): 1303.7 → 1271.9 ms/chunk = **+2.5% end-to-end prefill**
(45.99/46.06 → 47.19/47.16 tok/s). The isolated-harness number exceeds
the in-sequence number — dispatch interleaving and cache behavior differ —
so +2.5% is the deployable claim; the roofline number is attribution.

| step | change | outcome |
|---|---|---|
| 1 | scalar 4-entry LUT trit→half, deleting int-sub + convert per element | landed |
| 2 | half4 vector stores (4 per tile-slot instead of 16 scalar) | landed |
| 3 | byte-LUT: 256-entry `half4` table, one constant gather per 4 trits, deleting the shift/mask chains | landed |
| — | half-activation pre-pass ceiling (arm Cx = production weight staging + device-half activations) | C/Cx = **1.021** with clean controls — PARKED as borderline (a ~2% GEMM gain needs a kernel + ABI bump + engine buffer plumbing); re-openable if a round-4 happens |
| — | K=128 staging (two 64-K sub-tiles per barrier round, flush/fold order unchanged = still bit-identical) | REJECTED — 1.191, a 10% regression: 16 KB threadgroup footprint costs more occupancy than the halved barrier cadence saves; K=64 stands, bracketed from both sides (K=32 −4%, K=128 −10%) |

Identity gates for the landed changes: the staged halves are identical
values, so outputs are bit-identical by construction — verified by
`test_metal`/`test_metal_ops` and a pre/post artifact A/B (saved pre-round
binary + pre-round shader via `Q27_METAL_SOURCE` vs the new tree; 113-token
prompt through two 96-wide chunks + 48 greedy tokens; 215 bytes,
byte-identical, both arms verified non-vacuous — real generation at
position 161).

## FINAL disposition

**The landed gain (4.8% GEMM / 2.5% whole-prefill) is short of the 8.1%
fork-gap bar → prefill closes as MATURE with the attempt recorded**, per
the adopted round-3 bottom line ("one attribution experiment... not an
open-ended optimization stream"). The gain ships (it is free —
bit-identical). Residual attribution for any future round-4 re-open, all
with clean controls: remaining plumbing C/Beq = 1.083 [1.082, 1.083]
(byte-extract + constant gathers + staging stores + load-issue — the
staging structure itself; the tool prints its threshold line against this
residual, but the one authorized round is SPENT); flush/barrier cadence
Beq/A = 1.274 (now bracketed: neither K=32 nor K=128 beats K=64); the
parked Cx pre-pass at 2.1%; C ≈ 2.65 vs A ≈ 3.65 TFLOP/s physical MMA.
