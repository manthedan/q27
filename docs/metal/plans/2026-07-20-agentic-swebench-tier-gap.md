# Agentic SWE-bench tier-gap probe — metric pre-registration (2026-07-20)

Motivation: the 120-prompt short-form capability suite SATURATED at
max_tokens 4096 (2026-07-19 plan, verdict c8a7bdb): all four tiers score
~1.0, so it has no discriminative power between B1 and T2. The operator
wants real agent tasks that challenge the packs enough to see WHERE they
fail, bounded in time, without Claude Code (heavy startup prefill we cannot
amortise) and without shipping sysprompt snapshots. This is that probe.

This is a CAPABILITY/TIER-GAP probe (does t2 beat b1 on agentic coding?),
not the throughput replay bench (2026-07-17 plan) and not the saturated
short-form suite (2026-07-19 plan). Pre-registered BEFORE any measurement.

## Design (locked with operator, 2026-07-20)

- **Harness**: q27-native Python agent loop (`tools/eval/agent_runner.py`)
  against our own Anthropic API. NO Claude Code, NO Docker. Tools exposed
  to the model: `read_file(path)`, `write_file(path, content)`,
  `edit_file(path, old, new)`, `grep(pattern)`, `glob(pattern)`,
  `run_bash(cmd)` (bounded). Loop: assistant message -> tool calls -> tool
  results -> next assistant message, up to the turn cap. Prefix caching ON
  so the repo context + system prompt is paid once per instance (the
  amortisation the snapshots pitch would formalise).
- **Budget**: 20 turns/instance, 8192 max_tokens/turn, thinking mode ON
  (matches the 4096-capability config that eliminated the empty-response
  artifact; 8192 gives headroom for the larger diffs agent work emits).
  Per-instance wall cap 20 min; the agent can stop early (no tool calls =
  done).
- **Arms**: `b1-base` and `t2-base` ONLY (cleanest tier gap). One resident
  model at a time, serial, quiet machine, overnight.
- **Set**: `bench/swebench/manifest.hard.json` = the existing 12 pinned
  instances (already selected, prior results) + ~8 HARDER instances from
  the same fast repos, difficulty >= "15 min - 1 hour", deterministic
  sorted-by-id, per-repo quota weighted to where hard instances exist
  (xarray/pylint/pytest; flask exhausted at 1, requests has 2). ~20 total.
- **Server config**: frozen protocol block per arm, recorded in provenance
  exactly as the capability suite does. ctx 8192, prefix cache on, greedy.

## Metrics (fixed here, pre-registered)

- **gold_file_hit** (primary, cheap): the run's diff touches >=1 of the
  instance's `gold_files` (same signal as the 2026-07-14 run.sh). Reported
  per arm as hit/total.
- **stop_reason discipline**: every generation row records stop_reason;
  only `end_turn`/`stop` rows are scoreable, `max_tokens`/`length` rows are
  dropped from numerator AND denominator and counted in `excl` (identical
  rule to the fixed accuracy.py). Rows without stop_reason fail closed.
- **turns / wall / out_tok per instance**, and per-arm aggregate decode
  tok/s from the server trace (secondary; not a gate).
- **failure-mode read** (the point): for each arm, classify misses as
  (a) wrong-file (edited files disjoint from gold), (b) no-edit (empty
  diff), (c) truncated (max_tokens stop), (d) turn-cap exhausted, (e) tool
  error / loop stall. The tier gap is read off WHERE b1 fails that t2 does
  not.

## Ship line / verdict rule

- The probe DISCRIMINATES if the two arms differ by >= 2 gold_file_hit on
  the ~20-instance set AND the misses cluster in different failure modes.
- If both arms hit ~all (like the 2026-07-14 12/12, 10/12 gold_hit run),
  the SET is still too easy -> record "agentic probe saturates, need
  >1-hour difficulty" and stop; do NOT escalate scope.
- If both arms miss ~all, the harness/budget is the bottleneck (likely
  prefix/tooling), record that and stop; do NOT blame capability.
- NO pass/fail test execution (operator chose gold_file_hit only for v1).
  A future v2 may run each instance's FAIL_TO_PASS tests in-container for a
  true pass-rate; that is explicitly OUT of scope here.

## Bounds

- ~20 instances x 2 arms, ~6-12 min/instance deep-budget serial ->
  ~4-8 h one overnight run. If the first smoke instance exceeds the
  20-min wall cap, stop and re-budget BEFORE the full run.
- One-resident-model rule; no other GPU work concurrent.

## RESULTS

(pending)
