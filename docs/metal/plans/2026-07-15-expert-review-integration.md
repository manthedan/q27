# External expert review — integration notes (2026-07-15)

An outside expert reviewed the project from `EXPERT-BRIEF-2026-07-15.md`.
Access caveat: they could only see the public **`master`** branch (the
GitHub default branch was still `master` — now fixed to `metal`), so their
review is grounded in the CUDA ledger, ds4's public code, and published
Apple Silicon characterization — NOT our Metal ledgers/plans. Where they
independently reconstructed our own conclusions from first principles,
that's meaningful confirmation; where they guessed at gaps we've already
closed, corrections are listed at the bottom. This doc distills the review
into deltas against existing plan docs. Full response text is with the operator.

## Independent confirmations (no action; confidence++)

- Decode GEMV at the stream ceiling: their arithmetic (7.17 GB × 12.4 tok/s
  ≈ 88.9 GB/s) matches our measured 84–93 GB/s exactly; the 9→12.5 gap is
  orchestration, not kernels — this is `resident-greedy`'s thesis.
- Prefill compute-bound with the 12-token chunk as prime suspect —
  `wide-chunks.md`'s thesis, reconstructed without seeing it.
- Their P0/P1 priorities = our two in-flight docs. Roadmap ordering holds.
- Packing rooflines match ours (1.75 bpw → 15.1 tok/s ceiling; binary →
  23.4). Their warning that 1.75 bpw's ~22% ceiling gain can be eaten by
  unpack cost = t3-packing's existing Phase 0 kill gate, reinforced.

## Deltas to adopt, by doc

### wide-chunks.md (P1)

1. **Sweep protocol upgrade**: isolated projection bench over
   T={12..512} × M={16,32,64} × N={8,16,32} × RHS={direct,half-staged} ×
   **N-reuse={1,2,4,8}** before committing to Phase B's fixed 64×32 shape.
   N-reuse (decode a packed weight tile once, consume several N subtiles
   before advancing K) is a structural alternative to bigger token tiles —
   same intensity win, less threadgroup pressure. Not currently in the doc.
2. **Fit t(T)=a+bT** across the sweep: large `a` → dispatch/setup/underfill;
   large `b` with poor MMA utilization → unpack/fragment construction.
3. **Three rooflines at identical shape/geometry**: pure-half GEMM (no
   unpack), unpack-only (dummy MMA), full ternary GEMM. Partitions the gap
   into matrix-path ceiling / unpack cost / staging cost / multiply cost.
4. **Expand-to-half diagnostic** (not production): for T≥128, expand one
   matrix to half in rolling scratch, run best half GEMM, discard. If
   dramatically faster → unpack is the wall; modestly → shape/matrix path;
   loses → packed kernel already right.
5. External datapoint (their cite: arXiv 2606.12765, M4 Max
   characterization): Metal-4 MPP ≈ only 1.05–1.21× over simdgroup_matrix,
   fp8 slower than fp16 (emulated). Confirms our M4 skip of cooperative
   tensors and caps expectations: the play is efficient unpack-to-half +
   well-utilized simdgroup_matrix, not a hidden low-bit datapath.

### resident-greedy.md (P0)

1. **Split into A/B experiments**: (A) prequeue K one-token command buffers
   (device-side token feedback, no CPU readback between buffers, one
   queue's ordering) vs (B) K full steps in one command buffer. A isolates
   queue starvation/readback; B additionally measures per-CB scheduling
   cost. Flags proposed: `Q27_METAL_RESIDENT_CB_AHEAD=K` /
   `Q27_METAL_RESIDENT_STEPS_PER_CB=K`; sweep K={1,2,4,8,16}.
2. **DecodeState device struct** with `done` flag: EOS/length sets done,
   remaining scheduled steps become deterministic no-ops. Positions are
   host-encoded constants (no CPU dependency).
3. **Keep MTLSharedEvent OUT of the model dependency chain** — same-queue
   ordering already expresses it; use events/completion handlers only to
   notify the CPU that tokens are ready (streaming).
4. **lm_head fused selection**: don't materialize full logits on the greedy
   path — lm_head blocks write local (max_logit, token_id) pairs, small
   second-stage reduction picks the winner (lower-ID tie-break) directly
   into DecodeState. Output traffic drops and the feedback loop becomes
   explicit. Full logits stay available for serving/sampled paths.
5. **Correctness gates to add**: EOS at each step 0..K−1, max-length at
   each step, ring wraparound, cancellation after every step, prefix-cache
   save immediately after a batched step, solo-vs-multislot interleave.
   Promotion gate: ≥95% of the resident ceiling, zero CPU waits on the
   critical path, no idle gaps in Metal System Trace.
6. **Log GPU P-state**: the same paper reports short-kernel P-state
   bimodality on M4-class GPUs (short work finishes before high clocks).
   K-step buffers may gain from clock ramp as much as scheduling — bench
   logs must record first-iteration-separately, warm/cold queue, median
   not min. (Applies to ALL short-kernel benches; see cross-cutting.)

### cache-block-scheduling.md (P3) — design challenge to resolve

Expert argues **global head-major conflicts with eviction, prefix sharing,
and multislot**; prefers block-major/head-inner:
`[cache_block][kv_head][K|V][token_in_block][packed_dim]` — blocks
individually movable/evictable/refcountable, token-block split-K natural.
Also: sweep query-head grouping {1,2,3,6} (6 accumulators may cost
occupancy; 2–3 plausible optimum), block sizes {64,128,256,512}, and report
efficiency η = min-required-KV-bytes / (kernel time × practical BW) with
thresholds (>70% at floor; <40% layout/occupancy problem; falling-with-
depth → TLB/block-table). The R1 head-major proposal should answer this
challenge explicitly before kernels are written — at minimum the layout
must not preclude per-block eviction/refcounting given multislot (P2) and
disk prefix persistence (ds4-survey item 5) both want block granularity.
Their second cite (arXiv 2604.16957) supports turbo3's
direct-compressed-consumption approach at long context (vs
dequantize-then-attend) — magnitude not transferable, direction confirmed.

### metal-multislot.md (P2) — design challenge to resolve

Current design is whole-generation leases then interleaved yields =
**time slicing**. Expert: treat multislot as **weight-read amortization** —
for the large linears, batch active slots into a true N dimension so one
decoded weight tile feeds all slots. Two always-committing columns beat
speculative lanes economically (no acceptance loss, no drafter bytes).
Their priority: finish multislot BEFORE any drafter investment. Note the
existing lease design remains the right v1 for correctness/isolation; the
challenge applies to the steady-state execution model. Slot-batched N also
composes with wide-chunks' GEMM work (same kernel surface).

### sibling-drafter-probe.md (P4) — gate structure upgraded

New gate 0, ahead of everything: **oracle verifier test** in OUR engine —
feed the target's own known future tokens as free proposals (D=0, max
acceptance), measure S(w) = A·G / (V(w)+O(w)) at w={1,2,4,8,12,16}.
Oracle <1.05× → verifier economics structurally dead on this hardware,
close the question; 1.1–1.25× → insufficient headroom for a sibling;
>1.25–1.4× → real drafter has a budget. Also record the **V(w)/G curve**
directly — if V(8) is several ×G, the verify GEMM isn't reusing weight
work, which ties speculation's fate to wide-chunks (P1): do not final-call
speculation until macro-panel GEMM and resident feedback are done. The
their-stack probe (1.7B + DSpark force-enable) remains cheap and parallel;
suffix drafting (D≈0) deserves its own traffic-split gate (novel code /
prose / repetition-heavy) independent of any learned drafter.

### Numeric-equivalence contract (new; formalizes existing practice)

Five levels, replacing ad-hoc tolerance choices:
L1 exact structural (repack round-trip, tokenizer, KV addresses, canonical
16-token — all existing); L2 teacher-forced logit metrics per position
(max/RMS error, KL, top-k overlap, ref-token rank, margin); L3
**margin-aware top-1**: with per-logit error bound E, mismatch at margin
m>2E → bug; m≤2E → legitimate divergence (tighter: m>|e1|+|e2|); L4
layer-localized checkpoints as hashes+error summaries in CI, full dumps
behind a debug flag; L5 free-running → compare task metrics only after
first legitimate divergence. **Tolerances built empirically**: envelope
from {two Metal reduction orders, serial-vs-batched, CUDA kernel pair,
repeat runs}, use p99.9 + margin with a hard alarm — not an arbitrary
3e-4. Speculative commit invariant: committed tokens checked against the
verifier row (argmax-or-near within L3 margins), never whole-stream
identity — ds4 adopted this and it caught a real verifier indexing bug.

## New workstreams worth opening (small, high information)

1. **Mixed-precision quality-rescue tier**: swap one ternary matrix class
   at a time back to official-tier weights, teacher-force a fixed corpus,
   rank tensors by KL/PPL recovered per MiB. A few promoted projections may
   beat both pure tiers while keeping 24 GB residency. (Composes with the
   binary tier: same experiment prices a binary+promoted hybrid.)
2. **Ternary zero-structure analysis**: ledger already records 29.7% zeros;
   measure all-zero rates in 16/32/64 blocks, run distribution, row/K-tile
   stability. Gate any sparse path on **bytes not fetched**, not skipped
   multiplies. Random zeros → drop the idea.
3. **Separate decode vs prefill weight layouts** (park until wide-chunks
   lands): optimal GEMV packing ≠ optimal GEMM packing; an alternate
   prefill-oriented layout for the dominant projections, generated at load
   time, may fit the ternary tier's budget. Gate on the three-roofline
   attribution first — pointless if unpack isn't the prefill wall.

## Cross-cutting bench discipline (adopt everywhere)

Per-bench logging: GPU clock/P-state where obtainable, first iteration
reported separately, warm vs cold queue, median + distribution (never min
alone), idle intervals, thermal state. Extends the existing
caffeinate/matched-thermal protocol rules from run-level to bench-level.

## Corrections to relay back (things the expert couldn't see)

1. Prefix checkpoints ALREADY include GDN recurrent state, MTP KV,
   position, and resident logits (`capture_state`/`restore_state`,
   checkpoint 7) — flagged as "conspicuously absent," actually shipped.
2. Teacher-forced NLL harness with CUDA-matched position buckets exists
   (`--nll-long`); L2 of their contract is partially built.
3. Thermal/caffeinate protocol rules exist at run level (ledger, 2026-07-15
   entries); their delta is bench-level clock logging, adopted above.
4. The GQA kernel already assigns one simdgroup per query head (register
   pressure differs from their 6-accumulator concern); the grouping sweep
   is still worth running on the decode-into-registers reuse path.
5. Default branch now `metal` — the Metal ledgers/plans they couldn't find
   are at `docs/metal/METAL_PROGRESS.md` and `docs/plans/2026-07-15-*`.

## Priority deltas accepted

- Multislot rises above drafter work (P2 > P4) — consistent with the
  drafting-parked verdict; multislot needs no acceptance luck.
- Formats (T3/binary) gate on **wall time, not bytes**: separate
  `decode_ns_per_weight_byte` / `prefill_ns_per_output_element` /
  `verify_ns_per_candidate` gates per format. Binary keeps its large
  headroom (23.4 ceiling); T3's 22% margin is fragile.
- The two first experiments they propose are exactly the two in-flight
  docs (resident-greedy A/B; macro-panel GEMM spike) — proceed, with the
  upgraded protocols above.
