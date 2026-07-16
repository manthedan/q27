# Margin-aware quality gates — adopting the numeric-equivalence contract

Daniel approved (2026-07-15 night) moving off byte-identity as the DEFAULT
gate for kernel changes, adopting the external expert's five-level contract
(2026-07-15-expert-review-integration.md, "Numeric-equivalence contract").
This doc operationalizes it for q27.

## What changes and what does not

**Stays, unchanged (L1 — exact structural):** repack round-trips, tokenizer
identity, KV addressing, the canonical 16-token CUDA gate, RNG-stream
parity contracts. These are structure, not numerics.

**Stays where it holds for free:** invariants that are bit-exact by
construction keep their memcmp gates — e.g. the R1b t2 kernels are
bit-identical per token by design, and the chunk↔decode straddle contract
remains cheap to prove. Adopting the contract does not discard proven
invariants; it changes the default for NEW numeric work.

**Changes:** a kernel variant that reorders reductions or stages at
different precision is no longer auto-rejected for failing byte-identity or
an ad-hoc 3e-4. It gates on:
- **L2** — teacher-forced logit metrics per position (max/RMS logit error,
  KL vs reference, top-k overlap, reference-token rank/margin) over a fixed
  corpus slice.
- **L3** — margin-aware top-1: with per-logit error bound E, a top-1
  mismatch at margin m > 2E is a bug; m ≤ 2E is legitimate divergence.
- **L5** — free-running comparisons stop at the first legitimate divergence;
  after it, only task metrics compare.

**Not a license for lossy kernels:** quality regressions beyond the
empirical envelope still fail. Half accumulators' +0.43% NLL sits well
outside the measured reorder noise (±0.17%) and stays dead unless a
variant lands inside the envelope.

## Phase 1 — build the envelope before using it (instrument)

Tolerances are built empirically, not chosen. Sources for the envelope, all
legitimate same-model reduction-order pairs on the T2 artifact:
- chunked vs serial encode (the known fold-order class)
- `Q27_METAL_GEMM_HALF` 0 vs 1
- `Q27_METAL_GQA_THRESHOLD` 0 vs default (legacy vs blocked attention)
- repeat runs (should be 0 — determinism check rides along)

Instrument: extend the `--kl-kv` machinery (`teacher_force_logits` +
`src/kl.h`) into a `--envelope` mode that runs two engine configs in
lockstep over N positions and records per-position max-logit delta, KL,
top-1 agreement, and margin at every disagreement. Output: p99.9 of each
metric + margin → recorded as the contract constants in this doc; hard
alarm thresholds set one order above. Runs on the mac-mini, ~256–2048
positions, minutes not hours.

## Reopened levers, re-priced under the new contract

1. **Attention block-size {128..2048} and query-head-grouping {1,2,3,6}
   sweeps** (expert P3 sweep, now the top attention lever post-Phase-0):
   different block sizes change merge fold order — previously unshippable,
   now L2/L3-gated. Bench first on `metal_attn_bench` (wall only), then
   envelope-gate the winner.
2. **Prefill prescaled-half staging (variant A)** — died on the 3e-4 shape
   suite; the rounding is per-value staging rounding, exactly the class the
   contract bounds empirically. Re-price AFTER the three-roofline
   attribution says whether the exactness-forced flush cadence is the wall
   (if pure-half GEMM at identical geometry is not much faster, nothing to
   reopen).
3. NOT reopened: half accumulators (quality class, outside envelope), T3
   packing (killed on wall time, not numerics), R1 layout (killed on wall
   time), phase-B direct-RHS (killed on wall time).

## Order of work (GPU-serialized behind the overnight batch)

1. Three-roofline prefill attribution + block-size/head-grouping bench
   sweep (wall-time answers, no contract needed).
2. Envelope instrument + first envelope measurement (contract constants).
3. Ship or kill the sweep winner and variant A under L2/L3.
