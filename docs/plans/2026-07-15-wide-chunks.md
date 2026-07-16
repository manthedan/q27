# Wide prefill chunks + direct-RHS GEMM (ds4-survey items 2+3)

Prefill is compute-bound in the chunk GEMM at ~20 GB/s effective weight
stream after G′ half staging (36.8 tok/s vs the fork's 52.5 on this M4).
The structural cause: every staged weight tile amortizes over at most 12
activations (CHUNK_MAX=12, historically tied to the GEMM's 16-token tile
and MTP verify width). ds4 prefills 4096-token chunks with 64-row × 32-token
× 32-K tiles and a direct-RHS path that never stages activations.

## The arithmetic-intensity trap (why phasing matters)

Widening the engine chunk alone and grid-tiling the existing 16-token GEMM
does NOT cut the weight stream — each token-tile threadgroup re-reads the
weights, so intensity stays at 16 tokens per weight byte. The weight-stream
win requires the kernel to hold MORE TOKENS PER THREADGROUP: wider Xt (or
no Xt at all — direct device reads) and bigger weight tiles. Both phases
are needed; only their combination moves the ceiling.

## Phases

- **A. Chunk-width decoupling (engine plumbing).** `PREFILL_CHUNK_MAX = 96`
  (8 × 12: token-id array stays a 384 B dispatch constant; MTP verify keeps
  its own 12). c-buffers sized by the prefill width (~40 MB total at 96;
  `clogits_`/output-head stays 12-wide — the head runs only on the final
  12-slice in prefill, and NLL slices the head per 12 inside a wide chunk).
  Backend `x_rows` guards widen; the GEMM dispatches a token-tile grid
  (`ceil(x_rows/16)` in grid.y) as a correctness bridge. GDN chunk is
  sequential-in-T in-kernel (any T); causal GQA already handles straddle
  splits. Gates: chunked-vs-serial committed tokens identical at widths
  {12, 24, 96}; NLL 2K spot unchanged; `metal_prefill_bench --chunk` sweep
  (expect modest win from fewer chunk boundaries — GDN/attention batching —
  not from the GEMM).
- **B. Direct-RHS wide-token GEMM.** New T2 kernel: 64-row × 32-token tiles,
  weight tile staged in half (G′ discovery), activations read directly from
  device memory as int8 (+ per-32-group scale), float accumulators, scale
  folding at the 32-K flush as G′ does. Threadgroup budget: Wt 64×64 half
  = 8 KB, no Xt, scratch ~4 KB. Weights stream once per 32 tokens — 2×
  intensity — and the freed TG memory doubles the row tile, halving
  dispatch count. Gates: shape suite at the new tile shapes (row/token
  remainders), committed-token A/B, bench sweep. ds4 evidence: 128-token
  retest was neutral-or-slower vs their shape; 32 is the sweet spot they
  shipped.
- **C. Function-constant specialization** (edge-tile variants, simdgroup
  counts) — opportunistic after B, only if the profile shows bounds
  branches.

## Kill criteria

Phase A: any committed-token divergence (no tolerance here — same kernels,
same math, just batching). Phase B: < 1.2× chunk rate over post-A baseline,
or shape-suite failures — record and keep A.

## Non-goals

MTP verify width changes; sampled-path chunking; CUDA parity; ds4's
Metal-4 cooperative tensors (M5/M6-gated).

## Outcomes (2026-07-15)

**Phase A: landed, 1.39×** (35.09 → 48.79 tok/s at chunk 96 on the 384-token
synthetic T2 mix; 48 saturates). Committed tokens byte-identical to serial
and to the pre-widening chunked path at widths 17 and 96. Commit `9bf49c6`
plus the ABI 6→7 bump (codex finding on the review of 9bf49c6).

**Phase B: killed by its own criterion — 0.96×, kernel removed.** The
64-row × 32-token kernel (`q27_matmul_t2_mm_w`: K-stage 32 = one
activation-scale group, fold order bit-matched to `_h`, Wt 4 KB + Xt 2 KB +
Sc 8 KB, 256 threads) was built, passed the full shape suite with the wide
widths genuinely dispatched (17/33/64/96, row remainders 100/64), and
measured 46.94–47.00 vs 48.81–48.86 tok/s for `_h` across repeated A/Bs —
stable, not noise. Attribution kills the premise, not the execution: at
chunk 96 the GEMM is 94.3% of GPU time but the effective weight stream is
3.3 GB/s against ~100 GB/s machine bandwidth — **prefill is MMA/ALU-bound,
not DRAM-bound, so doubling tokens-per-weight-byte buys nothing here**, and
the finer flush/barrier cadence (per 32-K instead of per 64-K) costs 4%.
ds4's direct-RHS win presupposes DRAM-bound prefill (M5-class MMA
throughput); it does not transfer to the M4. Both kernels sit at the same
~2.5 TFLOP effective MMA plateau — the remaining 7% to the fork's pp512
52.48 is not GEMM-tile-shape addressable; half *accumulators* would be the
2× ALU lever but already failed the numerics ladder (see
2026-07-15-gemm-half-staging.md). Kept from the attempt: the widened shape
suite (widths 17/33 in the dispatch loop; row/token remainder shapes
100×1152×33 and 64×128×64 — cols kept moderate because the pre-scaled fp32
staging in Q4/Q8 amplifies cancellation rounding past the 3e-4 gate at
cols=5120, found when the new shape tripped Q8).

**Phase C: not entered** — its trigger (bounds-branch overhead in the
profile) is absent; the profile shows raw MMA dominance.
