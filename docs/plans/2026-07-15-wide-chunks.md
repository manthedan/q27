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
