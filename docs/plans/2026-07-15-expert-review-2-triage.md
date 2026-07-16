# Expert review round 2 (with repo access) — verified triage

2026-07-15. The external expert re-reviewed with full `metal`-branch access
(ledger, plans, engine/server/kernels/tests). This triage verifies each
finding against source at `38c744a` and records what was **overtaken by the
mini's same-day landings** (the review snapshot predates c514714..38c744a:
Phase B kill, R1 park, R1b graduation, R2 probe). Full review text is with
Daniel. Statuses: CONFIRMED (verified in source), ADOPT (design/process
change accepted), OVERTAKEN (resolved by measurement before the review
arrived), PARKED-AGREED.

## P0 — confirmed correctness findings (fix before further server/multislot work)

1. **Tool-constraint mask can leak across requests. FIXED (same night)** —
   defensive `set_tool_constraint(-1)` at request entry + a non-throwing
   scope-exit guard (`tc.end()` + mask clear) covering every exit path.
   The prescribed sink-throw byte-compare gate and backend failpoint remain
   follow-ups. Original finding: **CONFIRMED**
   (`metal_server.cpp:249-250`). Cleanup (`tc.end()`,
   `set_tool_constraint(-1)`) runs only on the normal return path; engine
   exceptions mid-generation (backend `runtime_error`s) skip it, the mutex
   releases, and the next request decodes under the previous request's
   mask. Fix shape (expert's, endorsed): defensive reset at request entry +
   scope-exit cleanup that never masks the original exception. Gate: throw
   from the token sink after grammar engagement, then run a clean
   unconstrained request and byte-compare against a fresh engine; add a
   backend failpoint (`throw_after_dispatch`) for the realistic path.
2. **Shared KV reservation not constructor-exception-safe. FIXED (same
   night)** — constructor-local RAII reservation guard (rolls back unless
   the constructor completes) + underflow assert in the destructor.
   Failure-injection gate remains a follow-up (needs a backend failpoint).
   Original finding: **CONFIRMED**
   (`metal_engine.cpp:233-236`): `shared_->cache_bytes` is incremented
   before ~15 subsequent allocations; a constructor throw never runs the
   destructor (~line 192), leaving a phantom reservation that rejects later
   valid engines — precisely the multislot startup-pressure scenario. Fix
   shape: constructor-local reservation guard (RAII, commit-at-end), plus
   an underflow assert on subtraction. Single-threaded `Shared` — no
   atomics needed.
3. **Wide prefill path not covered by NLL/KL gates. CONFIRMED**
   (`metal_engine.h:130`, `metal_engine.cpp:956`): NLL/teacher-forcing
   still slices at `CHUNK_MAX=12`, so widths 48/96 carry only
   committed-token A/Bs — decision-level, not distribution-level, coverage.
   The width-17 corruption found during Phase A (while a vacuous shape gate
   passed) makes this non-theoretical. ADOPT the expert's gate: run
   `chunk_forward` at widths 17/48/96, head applied in 12-row slices,
   compare per-row NLL + logits vs 12-row chunking (max/mean NLL delta,
   top-1 agreement, top-k overlap, margin at each flip), keep the
   committed-token A/B as a separate gate.

## Design corrections adopted

4. **Screening error budget is per-block, must be cumulative. ADOPT**
   (plan-doc fix, blocks any future Phase 0b screening; R1b graduation is
   unaffected). Per-block `ε` bounds sum across skipped blocks — skip a set
   S only when Σ_{b∈S} U_b ≤ ε·L_K (U_b = n_b·e^{s·u_b − M} from exact
   anchor scores), or allocate ε/N_eligible per block (sorted cumulative
   scan is less pessimistic). Also: a softmax-mass bound does not bound the
   output vector unless ‖v‖ ≤ R is maintained (then error ≤ 2Rε/(1+ε)) —
   either track a value-norm bound or relabel the method
   probability-mass-bounded and keep KL/NLL/needle as the output evidence.
   Disambiguate scaled-vs-unscaled scores in the doc's notation.
5. **Suffix drafting has never had its fair test. ADOPT — this is the
   highest-value non-P0 item.** `generate_suffix()` verifies serially
   (one weight read per verified token); the batched layer-major verifier
   exists only inside the MTP path. Factor it into a drafter-independent
   `verify_lanes(lanes, remaining)` used by (a) oracle proposals, (b)
   suffix proposals, (c) MTP, (d) any future drafter. Oracle gates
   (supersede the probe doc's thresholds): <1.05× → stop all batched
   drafting; 1.05–1.20× → suffix on repetition-heavy traffic only; >1.20×
   → suffix first, learned-drafter economics second. Measure emitted (not
   encoded) tokens — q27 never encodes the final output token — and
   include full round cost: verify CB, argmax readback, GDN replay,
   commit. Run only after prefill/GEMM work settles (same kernel controls
   verifier economics). Keep MTP-warm prompts out of the oracle
   measurement (`prefill()` is token-serial under `warm_mtp` — ledger-known).
6. **Multislot scheduling quantum vs latency claim. ADOPT.** A 96-token
   chunk at ~48.8 tok/s holds the GPU ~2 s; yield-after-`encode_chunk`
   cannot deliver "~one decode token" waits during prefill. Make prefill
   width a runtime policy, not an engine constant: 96 for offline/bulk, 48
   (or adaptive) for serving, shrink to 12–24 when waiters exist; report
   gate-wait separately for arrival-during-decode vs arrival-during-
   prefill (and during MTP verify). The honest v1 guarantee is "at most
   one active scheduling quantum." Also from the review: admission check
   must budget fixed engine state + configured snapshots + GQA partial
   peak (not KV alone); state the route-mutex → GPU-lease lock order and
   add a two-thread stress gate; make post-cancel MTP state explicitly
   non-cacheable with a cancel-at-every-lane-position gate.
7. **Binary Phase 0 gates were too subjective. ADOPT — split.**
   Phase 0A (vendor stack, machine-checkable, pre-registered): exact-JSON
   response, valid native tool call, multi-condition instruction task,
   small code edit with tests, planted edge case, long-context retrieval,
   8K NLL vs BOTH official and T2 (report B1/T2, not only B1/official —
   the real question is the increment over the already-resident tier).
   Phase 0B (synthetic q27 kernel economics BEFORE repack/integration):
   B1 GEMV + embedding only; bytes read, GB/s, projection-mix time,
   predicted full-step from the measured non-GEMV residual. Pre-register
   the speed-vs-quality proposition; 14–15 tok/s at materially worse
   quality does not support the 20–23 headline. After integration, rerun
   resident K=8 (fixed orchestration share grows as tokens get cheaper).
8. **Numeric-equivalence contract upgraded to pairwise margins. ADOPT.**
   Replace single global error bounds with per-competitor certificates:
   top-1 flip impossible when z_a − z_j > |e_a| + |e_j|. Disagreement
   despite the inequality → bug; when it fails → tolerance-legitimate;
   same top-1 with large KL/rank movement → numerical regression worth
   investigation. Envelopes per operation and shape from accepted
   implementation variants — the repo-wide 3e-4 stays a unit-test
   convenience, not the definition of equivalence. Speculative commits
   keep the stricter row-match invariant.

## Overtaken by same-day measurements (no action; review snapshot was stale)

- **Phase B RHS-ingress advice** — Phase B was run and killed at 0.96×
  against a pre-registered 1.2× bar (`c514714`): prefill is **MMA-bound**
  (GEMM 94.3% of GPU time at 3.3 GB/s effective weight stream), not
  DRAM-bound, so the int8→half bridge question and tile sweeps are moot on
  M4. The expert's "global half-int mirror" idea is recorded as a reopen
  condition only if a future measurement shows staging (not MMA issue) as
  the wall. Their stale-kill-criterion point was fair but is also moot.
- **R1 block layout debate** — R1 head-major parked at 1.00× (`fdb8c67`):
  attention is latency-bound, not bandwidth-bound (fp16 pays no wall for
  5× bytes), so BOTH layout proposals lose their premise at current
  depths. The expert's block-major/head-inner sketch stays in the
  cache-block doc as the shape to prefer IF eviction/refcounting/multislot
  later require movable blocks — a functionality argument, not a
  bandwidth one.
- **R1b token tiling** — graduated at 2.0× with factor 2 (`90e16fb`);
  the expert predicted "factor 2 may win" over 4. Confirmed by
  measurement. Straddle contract re-proven through the tiled route.
- **Resident greedy demotion** — expert concurs with the mini's
  measurement (11.52→11.66; the 8-9 vs 12.5 gap was thermal). Kept as
  clean code; re-measure after B1 where the fixed share doubles; no
  streaming/EOS/constraint extension; lm-head argmax fusion deferred
  until a tier makes output-head overhead measurable.

## Process items adopted

- **Failure-injection testing** as the next test frontier: extract
  weight-free host units (budget reservation, constraint guard, multislot
  routing/lock-order, adaptive quantum policy, acceptance bookkeeping,
  prefix selection policy) into `make test-metal`-runnable tests with
  backend failpoints (`throw_after_allocate/dispatch/read`). The synthetic
  suite's gap is host orchestration, not kernels — findings 1–2 prove it.
- **Stable-boundary prefix snapshots**: the needle runs re-prefilled the
  same 31K haystack because each prompt differed only in the trailing
  question. A capacity-1/2 policy snapshotting an application-supplied or
  message-boundary-stable prefix eliminates enormous repeat prefill for
  document-QA/coding traffic. Composes with the ds4-survey disk-KV item.
- **Doc drift cleanup before Phase B–class work**: comments still say
  "2–12 tokens" post-96-wide ingestion; the backend half-GEMM comment
  reports the pre-fix 1.28× rather than landed 1.22×. Stale structural
  comments fed the vacuous-width gate class Phase A exposed.

## Revised execution order (merging expert order with overtaken items)

> **Superseded 2026-07-15** by round 3
> (`2026-07-15-expert-review-3-answers.md`), which inserts the one-day
> A/B/C MMA roofline and the R3 barrier-free attention probe ahead of
> serving work and promotes N=2 slot-batched linears above the generic
> verifier. The P0-first rule is unchanged.

1. **P0 correctness**: findings 1–3 (constraint RAII, reservation guard,
   wide NLL/logit gate) + screening-math fix in the cache-block doc.
2. **Two-slot serving** with finding 6's contract (runtime prefill quantum,
   honest wait metrics, admission budget, lock order, cancel invariant).
3. **Binary Phase 0A/0B in parallel** (vendor-stack quality + synthetic
   kernel economics; integration only if both gates pass).
4. **Generic `verify_lanes` + oracle/suffix measurement** (after GEMM work
   settles; suffix on repetition traffic is the live speculation question).
5. **Parked, unchanged**: T3 (killed on measurement), learned
   sibling/DSpark drafting, further T2 resident-sync work, screening
   Phase 0b (until the proof is rewritten).
