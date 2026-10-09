# Optimization round 2 (2026-10-09, mini M4 16 GiB, b2 t2-slim)

Four candidates after v0.7.1. Measured before building; two are closed here
so they are not re-tried without new evidence.

The machine was not quiet: about 10% GPU utilization from other processes
(WindowServer, screen sharing, a video wallpaper). Decisions therefore rest
on cost ratios, exact replays and interleaved A/B runs, not on single
wall-clock numbers.

## 1. Suffix bursts below the 12-token match floor: no-go

GPU cost of one suffix verify round (the verify chunk plus the replay),
forced to a fixed width on a fully repetitive prompt:

| live lanes | 2 | 4 | 6 | 8 | 9 | 12 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPU ms | 199 | 209 | 217 | 222 | 327 | 378 | 393 |

A serial step costs about 88 ms (83-96 ms between 32 and 1.5K tokens of
context). Up to 8 lanes the 8-token tile sets the cost; from 9 lanes it is
the 16-token tile.

Today's rule (`SUFFIX_MIN_MATCH` 12, live = match + 1, capped at the width)
always verifies 13-16 lanes, so the 8-token tile never runs in a burst. The
question was whether shorter matches (4-11) at 8 lanes, or capping long
matches at 8, would pay off.

**Method.** Real b2 greedy streams on four prompts: a Python file edit, a C++
header edit, an explanation of a file, and prose; 500 tokens each. These
were replayed exactly through the `SuffixDraft` proposer under each policy,
using the cost table above.

**Result.** Every policy landed between -1.5% and +2.2% of today's total, which
is inside run-to-run noise. Two reasons:

- Edit streams already burst on almost every round at width 16.
- Explanation and prose streams almost never match.

Capping every burst at 8 lanes was worse (+1.2% to +2.2%). The suffix-mode
serial fallback costs the same as plain decode (90.5 vs 92-96 ms per step at
1.1K context).

## 2. Prefill: GEMM is 88% of it; a 64-row tile buys ~3%

Per-kernel profile (`Q27_METAL_PROFILE=1`) of a 960-token prefill, which
took 19.9 s of GPU time:

| Kernel | Share |
| --- | --- |
| T2 float GEMM | 88% |
| Causal attention | 5.9% |
| DeltaNet chunk | 2.1% |
| Everything else | under 1% each |

The GEMM runs at about 2.7 TFLOP/s, 70-74% of the measured 3.85 TFLOP/s
matrix peak. The peak is the same for half x float, float x float and half x
half on M4.

Sweep in `tools/metal_t2_prefill_bench.mm` (96-token chunk, the full layer
stack's shapes; every variant bit-identical, d = 0):

| variant (rows per simdgroup / tile rows x tokens / K per stage) | ms per chunk | vs today |
| --- | --- | --- |
| today: 8 / 32x16 / 64 | 1714 | — |
| 16 / 64x16 / 64 | 1657 | -3.3% |
| 32 / 128x16 / 64 | 2550 | +49% |
| 8 / 32x16 / 128 | 2124 | +24% |
| 16 / 64x16 / 128 | 2758 | +61% |
| 16 / 64x32 / 64 | 1832 | +7% |
| 8 / 32x32 / 64 | 1805 | +5% |

The 64-row tile wins from 48 tokens up, ties at 17-32, and is 2% slower at
9-16, where it halves the threadgroups. It now runs for chunks over 32 tokens
(e278384).

- Measured in the engine: the profiled GEMM fell 2.9% (mean of two
  interleaved pairs).
- Token ids are identical to master for greedy (960 + 33), suffix-burst
  (200) and sampled (64) runs.
- Expected whole-prefill gain: about 2.5%.

What remains is structural. Larger register blocking or longer K stages lose
occupancy, and there is no faster matrix datatype on M4. Further prefill gains
on this GPU need a different algorithm (fewer FLOPs), not tuning.

## 3. Not done here

- **Qwen3.8 decode fusions.** Needs the 24 GB laptop. The mini cannot hold or
  run the 14.6 GB pack.
- **T3 chunked prefill and prefix snapshots.** b2 is T2 on every Mac, so this
  only matters if T3 is chosen on purpose.
- **Wall-clock journeys baseline.** Still needs a quiet machine.
