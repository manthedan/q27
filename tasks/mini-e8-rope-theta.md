# Task for mac-mini: E8 — RoPE theta-precision probe at depth (P2)

From the k3 audit triage (docs/plans/2026-07-17-k3-audit-triage.md, D3/E8);
operator-authorized 2026-07-17 ("do it" on the parallelization plan). **Queued
behind mini-e2-gqa-partials — finish that first**, this one is model-free and
patient.

**Question (verified in source):** both backends compute RoPE phases with the
same expression (q27_kernels.metal:880 and :3333 match src/blocks.cu:62-64
operand-for-operand), so there is no cross-backend shift by construction — but
both evaluate cos/sin at up to 262,143 rad in float, and no existing gate
isolates rope-phase error from KV-codec error at depth (32K NLL/needles are
end-to-end). The KV-codec P1 workstream needs the rope term sized on its own.

**Scope:**
1. New standalone probe, `tools/rope_theta_probe.cpp` (own Makefile target,
   not linked into anything production). Read the exact phase expression OUT
   OF THE KERNELS — do not re-derive it. Constants: rope sections [11,11,10,0],
   freq_base 1e7, HEAD_DIM 256 (verify against loader enforcement).
2. Three arms per (position, dim-pair): (a) double-precision reference
   (double phase, sincos in double), (b) CPU float mimicking the kernel
   expression order in float, (c) the Metal kernel itself via a tiny dispatch
   over a synthetic Q/K row (this arm exercises the GPU's cos/sin
   implementations, which are not libm's).
3. Positions: {1023, 8191, 32767, 131071, 262143}; all rope-active dim pairs.
   Report per position: max |Δphase| (b vs a), max |Δcos|+|Δsin| (b vs a and
   c vs a), and the worst-case rotated-component relative error on a
   unit-norm pair.
4. Banded readout (informational probe — no kill line, it feeds KV-codec P1):
   worst rotated-component error < 1e-4 at 262,143 → rope term is negligible
   vs the KV-codec quantization floor; 1e-4..1e-3 → report as a real term for
   the P1 error budget; > 1e-3 → flag for a paired-double or position-folding
   fix proposal (proposal only, no kernel edits under this task).
5. CUDA arm is NOT yours — the orchestrator runs the same probe body on yukon
   after the tool lands (keep the arm-(b) code path compilable without Metal).

**Rules of the house (unchanged):** edit/compile plus THIS probe's runs (it is
model-free, no model load, fine beside your other work); suites before any
push; append findings to this file's RESULTS section rather than rewriting.

## RESULTS

(pending)
