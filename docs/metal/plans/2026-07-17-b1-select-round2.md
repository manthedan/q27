# B1 select kernel round 2 — pre-registration (2026-07-17)

The one kernel look funded by Phase 0A-q27's decode-economics leg
(docs/metal/plans/2026-07-15-binary-tier.md RESULTS: artifact decode 17.28/17.09
tok/s = CONDITIONAL band; resident ceiling 19.39 tok/s at 69.9 GB/s vs T2's
same-bench 11.54 at 78.5 — the select kernel streams at 0.89 of T2's rate).
The operator's go 2026-07-17 ("let's work on 3" on the post-census queue).
Ship line carried verbatim: **artifact decode ≥ 18 tok/s
(metal_decode_bench --dtype b1, quiet-machine regated); < 18 ships nothing**
and the tier keeps the current kernel.

## Diagnosis driving the candidates

Production `q27_matvec_b1_quantized` runs ONE row per simdgroup — the same
overhead class E6 measured on Q4 and round 1 killed there: every row's
simdgroup re-issues the full activation load (x16 int4 pairs + x_scales)
and pays its own simd_sum + scale-read chain. B1 makes this class *worse*
than Q4's: the weight stream is 1 bit/weight (a lone 4 B uint per lane per
1024-col chunk — Q4's was 16 B), so per-row load chains are shorter, the
issue mix is x-load-heavier relative to weight bytes, and the kernel has
less latency to hide behind. That is consistent with streaming 0.89× of T2
at the decode dispatch mix despite Phase 0B's GEMV-only bandwidth parity.
The select ALU itself (q27_dot16_b1 positive-mask form) won Phase 0B
against sign-xor (2.6×) and matched popcount without its 0.115 int8
quantization cost — this round does NOT retry dot-form tricks.

## Candidates (bench arms first, promotion after)

- **A — multi-row restructure (the twice-proven pattern, primary):** 4 rows
  per simdgroup, each lane holding its 32-column int8 x-slice (2×int4) and
  x-scale in registers; inner loop reads only weights (4 independent 4 B
  streams — ILP across rows), one fused scale-multiply per row-chunk,
  simd_sum ×4 at the end. Select ALU unchanged by design.
- **B — 8-row variant (issue-depth probe, run only if A < sub-line):** same
  lane-held x-slice, 8 independent weight chains per lane; tests whether
  B1's 4 B chains need twice the row depth Q4 needed to fill the issue
  window, at the cost of ×8 simd_sum/scale chains.

Arms ride metal_gemv_bench as bench-only candidate PSOs (the b1-probe
pattern from Phase 0B), gated against the exact CPU reference on all
production B1 shapes BEFORE any timing. Promotion = replacing the
production kernel body, then: SHADER_ABI bump (dispatch-topology change —
the Q4 round's own lesson), test-metal, the 16-token B1 byte gate
(--tokens identity vs the current binary), and the quiet-machine artifact
decode regate.

## Pre-registered verdict

- Quiet artifact decode ≥ 18 tok/s → ship; record the honest table
  (per-shape GB/s + artifact + resident ceiling).
- < 18 after both candidates → kill, ship nothing, record the table and
  close the B1 decode question as "issue-bound at 4 B chains,
  structure-resistant" beside the Q4 autopsy.
- Per-candidate sub-line: an arm that fails to beat the production kernel's
  byte-weighted B1 GEMV mix by ≥ 10% in bench form is not promoted
  regardless of artifact hopes (the 17.3 → 18 gap is +4% of a wall that is
  not all GEMV; a sub-10% bench win cannot carry it and would ship noise).

Timing legs on the quiet 24 GB M4 (T2 server stopped, nothing else
loaded); correctness legs contention-tolerant. Candidate-vs-production
ratios are read same-process (contention-robust per Phase 0B practice);
the ship call itself only from the quiet regate.

## RESULTS (2026-07-17 evening, 24 GB M4, T2 server stopped for all legs;
logs/b1-round2-20260717/)

**SHIPPED — with the honest attribution that the artifact prize was not
where the funding thought it was.**

Candidate r2 (4 rows/simdgroup): correctness gates PASS on all 8 shapes
(CPU int8-activation model ≤ 1e-3 AND byte-identity vs production, before
any timing), then **mix speedup 2.359** on the clean decision leg
(production 158.66 ms → 67.26 ms per-token GEMV wall; per-shape 1.33–3.34,
worst attn k/v, best output head) — the ≥ 1.10 sub-line cleared by 2×.
A contended indicative run (resident T2 server) read 2.43 and is recorded
as contaminated. r3 (8-row) never ran — r2 cleared the sub-line first;
retained in-tree as the unrun issue-depth arm. Promotion: r2 body into
`q27_matvec_b1_quantized`, dispatch 8 → 32 rows/group, **SHADER_ABI
12 → 13** (dispatch topology, the Q4 round's lesson), probe candidate 2
aliased to production. 16-token B1 byte gate IDENTICAL pre/post;
test-metal green; post-promotion probe-path parity leg re-passes all gates.

**The ship line is met but the kernel did not move it: artifact decode
pre 18.90/18.67 → post 18.64/18.75 tok/s (same-session A/B, warm pack) —
noise.** Resident ceiling 53.7 → 52.0 ms/token (18.64 → 19.23 tok/s,
69.3 GB/s effective): +3%. The funded 17.28/17.09 baseline was a COLD-pack
number — its named "paging residue" was the whole 17.3 → 18 gap, and a warm
pack crosses the line with the old kernel. Diagnosis correction for the
record: the select GEMV *is* issue-bound in isolated dispatch (the 2.36×
is real and the per-dispatch wall matched Q4's at a quarter of the bytes),
but the engine's decode stream already overlaps kernels enough to hide it —
decode sits at the memory wall (3.36 GiB/token at ~69 GB/s ≈ 19 tok/s
ceiling, any kernel). The promoted kernel ships because it strictly
dominates (byte-identical, faster or equal everywhere, +3% ceiling, 2.36×
isolated — relevant to batched/multi-stream contexts), not because it
bought decode wall.

Residue: B1 streams 69.3 GB/s vs T2's same-bench 78.5 — the remaining 13%
is stream-rate at the full-decode dispatch mix, now known NOT to be GEMV
issue-rate; unfunded. Legs ran with the server stopped but the machine not
input-idle; the decision metrics are same-run A/B ratios (contention-
robust) and the absolute ≥ 18 legs can only improve on a quiet machine.

## CORRECTION (2026-07-19, expert review 4): route attribution

The causal claim above — "the engine's decode stream already overlaps
kernels enough to hide the 2.36×" — is **not established by this
experiment.** The promoted kernel is `q27_matvec_b1_quantized` (the
int8-activation entry point). The pure-B1 serial-decode path does not call
it:

    B1 tensor
      → MetalEngine::project()        (metal_engine.cpp:1146-1149)
      → is_bonsai_dtype → backend_.matvec(float activation)
      → q27_matvec_b1_g128            (metal_backend.mm:1032)

`project()` routes both B1 and T2 to `backend_.matvec`, not
`matvec_quantized`; the backend then selects `q27_matvec_b1_g128` for B1.
The float-activation kernel was already four rows per simdgroup. The
round-2 change modified the **int8** entry point, which the B1 artifact
serial path never invokes (the int8 copy is written 129×/token and read
0×/token — see `2026-07-19-b1-serial-decode-fusion.md`).

Therefore:

1. The artifact wall failing to move is **expected**; the changed kernel
   was not on that critical path.
2. "Engine overlap hides the 2.36× kernel gain" is **not** a supported
   conclusion of this round.
3. The promoted kernel may still be correct and useful to direct callers
   (chunked prefill / official-tier), but it was not a production B1
   serial-decode optimization. The measured B1 speed and the memory-wall
   observation stand; only the causal attribution does not.
4. **New gate — route liveness:** before timing any candidate artifact,
   the profiler must show a nonzero call count for the candidate kernel in
   that exact artifact run (P0 route audit). This round would have failed
   that gate.
