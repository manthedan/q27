# CUDA/Metal parity audit (2026-07-17) — triage + fixes

External static audit, two LOW findings, both source-verified REAL and
both fixed the same night. Its rejected/excluded table was checked
against the tree: every disposition matches ours (the two "stale — landed
at audited HEAD" rows are the suffix server integration and B1 tier,
which landed mid-audit).

## A (LOW, adopted): argmax exact-tie divergence

Verified: every CUDA argmax-family reduction (`k_argmax`, `argmax_masked`,
`k_gumbel_d`, `k_sample_stop`, fused `k_draft_top2`) shared `am_pack`'s
`(monotonic_value << 32) | idx` under max/atomicMax — exact-value ties
resolved to the HIGHEST surviving index, while Metal's argmax kernels and
the CPU `max_element` reference both take the LOWEST. Consequence: greedy
decode could commit different tokens on an exact fp32 logit tie (never
observed in a byte gate — ties are rare — but semantically real, and the
byte gates' cleanliness silently depended on tie absence).

Fix: `am_pack` now packs the BITWISE-INVERTED index (`~idx`), so the same
max reductions prefer the lowest index; `am_unpack_idx` recovers it at the
two extraction sites. One rule, all five kernels, no behavior change off
ties. Tests: `test_kernels.cu` argmax battery now asserts
lowest-index-on-tie on EVERY case plus a new triple-tie case — full suite
ALL PASS on yukon (3090, oracle swapped out and restored exactly:
health + live completion + 23,224 MiB footprint). The prebuilt oracle
binary predates the fix; it inherits it at the next prebuilt refresh —
recorded, not urgent (ties unobserved in gates to date).

## D (LOW, adopted): top-p boundary-tie truncation

Verified: both CPU sampling paths (`sample_logits_cpu`,
`sample_candidates_cpu`) kept only the minimal prefix crossing the top-p
cutoff — a token whose logit exactly ties the boundary token could be
silently excluded by sort order, where the CUDA nucleus keeps every token
at or above the threshold value. Fix: the retained prefix now extends
through all exact boundary ties (value-keyed) in both paths, keeping them
draw-for-draw identical. Test: three-way exact tie at top_p=0.5 — all
three tokens must be reachable (fails pre-fix, both paths); the existing
full-vs-candidates consistency sweep stays green.

## Found while probing (not in the audit): Metal server never rendered tools

The B1 behavioral probe battery's toolcall leg FAILed and the
investigation showed the Metal chat path applied the chat template with
messages only — `tools` fed the constrainer's trigger list but the model
never saw the schema (the CUDA server renders `tools_preamble` into the
system text). Fix: `tools_preamble`/`strip_ctrl` extracted to
`src/tool_preamble.h` (shared header — api_common.h's Utf8Gate collides
with the Metal stream gate, so whole-header reuse was not possible),
Metal `messages_from` now merges the preamble exactly as `chatml_prompt`
does. Probe re-run: PASS (correct function, exact args). CUDA server
rebuild verified on yukon after the header split.

## Codex round on the fixes (zero P1)

- **P2 adopted — bench guards**: `metal_decode_bench --dtype b1` still ran
  activation quantization the engine skips for bonsai dtypes (`!t2`
  guards); fixed to `!t2 && !b1`, ceiling re-measured 19.16 → **19.39
  tok/s @ 69.9 GB/s** (the extra dispatches cost ~0.6 ms/token).
- **P2 adopted — tie order across sampling paths**: with boundary ties now
  in the nucleus, the full-logits path's unspecified sort order could map
  the same draw to a different token than the candidates path's
  index-ascending order. Full path now tie-breaks index-ascending; new
  test asserts draw-for-draw identity on a tied boundary (300 draws).
- **P3 rejected with reasoning**: the `am_pack(-FLT_MAX, 0)` sentinel
  "winning" ties against real −FLT_MAX candidates cannot materialize —
  the per-thread scans use strict `>` from bv = −FLT_MAX, so a candidate
  at exactly −FLT_MAX is never selected pre- or post-fix; the tie rule
  governs reachable values only. Repacking sentinels as INT_MAX would
  additionally risk an out-of-range extraction on degenerate
  all-(−FLT_MAX) inputs where today's behavior returns a valid index 0.
