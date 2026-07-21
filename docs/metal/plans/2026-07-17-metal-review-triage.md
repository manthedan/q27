# Metal audit (docs/metal/metal-review-2026-07-17.md) — source-verified triage

Reviewed 2026-07-16 night against the live tree (the audit is fresh — it
covers `q27_b1_x_prep`, landed the same evening in e94e1cf). House rule
applied: every load-bearing claim verified in source before judgment.

## Headline findings — both fail verification

- **A1 (b1_x_prep "race by accident", HIGH) → NOT A BUG, comment
  accurate.** The kernel comment explicitly names the two-array layout as
  the reason no inter-phase barrier is needed (q27_kernels.metal:529-531);
  the overlap the reviewer describes is between DISTINCT threadgroup
  arrays — no race exists and the stated reason is the correct one. The
  reviewer's consolidation hazard is real as a future-edit risk only.
  ADOPT the one-barrier hardening as P3 (bench-only kernel, one barrier
  of cost). Severity HIGH is wrong.
- **A2 (topk `count < k` silent corruption, HIGH) → IMPOSSIBLE BY
  CONSTRUCTION.** Pass 1 stops with `above < k` and proves the boundary
  bin holds ≥ k−above items; pass 2 exits with `cumulative ≥ remaining`
  on BOTH branches (at `bin==0` cumulative equals the whole boundary bin,
  which pass 1 bounded below by `remaining`); pass 3 emits
  `above + cumulative ≥ k`. Ties only inflate the count, and that side
  has the capacity fallback (metal_engine.cpp:1628). ADOPT as P3
  defensive check (`count >= k || full fallback` — one line, guards
  future kernel edits), plus E8's stress tests. "Most consequential open
  bug" is retracted by proof.

## Verified and adopted

- **A6/E1 (whole-mapping tensor-extent validation) — the review's real
  top item.** Already-recorded codex debt, unimplemented; now flagged by
  two independent reviews. PRIORITY BUMPED: this is the one remaining
  silent-corruption path of the class that burned 2026-07-14. P1.
- **E6 (Q4/Q8 GEMV select-form / half-MMA leg) — genuinely new roadmap
  item, the review's best contribution.** The official tier's dominant
  decode cost sits at 67–90 GB/s while T2's two rewrites reached ~93;
  neither T2 trick has been tried on Q4/Q8. Queue a `metal_gemv_bench`
  leg with pre-registered bands before any kernel work. P2 (perf,
  official tier).
- **E8 (argmax/topk stress tests: all-−inf, tie storms, n % 256 ≠ 0,
  count == k−1)** — cheap, real test-coverage gap. P2.
- **B1/E2 (gqa_partials per-engine + shrink)** — real, belongs with the
  multislot/snapshots Phase-2 slot work (mini's lane). P2.
- **C6/E3 (gemm_half / gqa_threshold backend-scoped)** — factually
  correct; the envelope instrument flips them SEQUENTIALLY in one thread
  by design, so no live bug. ADOPT as documented single-engine-mutation
  contract (comment) now, per-engine scoping if a second concurrent
  engine ever needs different knobs. P3.
- **A3/E5 (T3 odd-nb tail)** — real but T3 is a PARKED tier; adopt the
  cheap `cols % 512` load-time refusal. P3.
- **A4, A5/E11, B4/E12, B5, C4** — comment-level/defensive nits, all
  accurate on the code, none load-bearing today. Batch as one hygiene
  pass. P3.
- **E10 (bench-only PSO surface in production init)** — real; fold into
  the Homebrew packaging lane (the installed binary is where startup
  compile time matters). P3.

## Confirmations (no action, review agrees with recorded state)

C1 (Q4/Q8 issue-bound: matches parked lever), C5 (4-blocks-in-flight:
matches recorded register-pressure kill), B3/E9 (readback latency floor:
matches GPU-sampling stage-3 roadmap), C3 (realloc is growth-only, not
per-dispatch — reviewer overstates, sizing hoist folded into E2), C2
(x2 kernels: parked-by-measurement, keep with PARKED comment per E10
pass), D1–D4 (numerics: all confirm SPEC/chronicle).

## Overall read

High-quality audit: accurate source reads, honest confirmations of
recorded decisions, and one genuinely new workstream (E6). Its two HIGH
severities don't survive verification — the pattern to keep: severity
claims about "silent corruption" get proven or retracted before they
drive work. Nothing in the audit contradicts the shipped gates.

---

## Fix-batch results (2026-07-16 night, same session; local machine)

Four MORE audit claims fell during implementation, each verified in
source: **A6/E1 was stale** — bind-time logical-extent enforcement
already exists at every enumerable tensor bind site (`tensor_limit` on
`data_size`/`scales_size` from all three upload paths, plus loader-side
declared-size equality, in-mapping bounds, and blob-overlap rejection);
the actual gap was test coverage, closed with a failing-capable negative
test (short data_size and short scales_size must throw; well-formed
controls pass). **E11 was factually wrong** — both `delta_step` and
`delta_chunk` already throw on non-(128,16) shapes. **E5's load-time
refusal REJECTED** — odd-nb T3 is a *tested* slow path (`test_t3_wide`,
cols=1152), so refusing it would reject provably-correct shapes; kernel
comment instead. **B5's `[command cancel]` is not a real MTLCommandBuffer
API** — contract comment is the whole fix. Landed: A2 min-k defensive
bound, E8 argmax/topk stress tests (all-−inf, tie storm, odd-n,
boundary-exact-k), A1/E7 barrier + corrected comment, E3 mutation
contract, E12/B5 comments, C2 PARKED markers. Codex: zero findings.
Suites green. E2 assigned to the mini (tasks/mini-e2-gqa-partials.md).
Scorecard for the audit after full verification: both HIGHs refuted, one
"unimplemented P1" already implemented, one host-guard claim wrong, one
suggested API nonexistent, one refusal would have broken a tested shape —
against real contributions E6 (Q4/Q8 GEMV leg, queued), E8 (tests,
landed), E10/E2 (queued in the right lanes).
