# Perf ratchets

Speed claims in release notes are gates, not one-off measurements. Two kinds:

| Gate | What | Noise | Where it runs |
| --- | --- | --- | --- |
| `tools/perf_ceilings.sh` | Metal command buffers and encoded operations for fixed greedy workloads (`ceilings.tsv`) | none: exact counts | tier B; valid on a busy machine |
| `tools/perf_journeys.sh` | load time, 512-token prefill tok/s, decode tok/s, suffix-copy time (best of 3) against `baseline-<hw.model>-<GiB>.tsv` | 10% band | tier B, quiet machine only |
| TUI `steady_frame_cost_is_independent_of_session_length` | lines word-wrapped per steady-state frame, 10 vs 400 turns | none | tier C (`cargo test`) |

## Rules

- A count above its ceiling fails. When a change lowers a count, run
  `tools/perf_ceilings.sh --ratchet` and commit the lowered ceiling with the
  change. Raising a ceiling needs a reviewed reason in the commit message.
- Record a journey baseline (`tools/perf_journeys.sh --record`) only on an
  idle machine, for a new machine or a reviewed speedup.
- A new ratchet counts only after it was shown to fail on a deliberately
  slowed build.

## Evidence that the gates can fail (2026-10-05, mini M4 16 GiB, t2-slim)

Two mutants built from a scratch copy of `src/` (working tree untouched):

- **tiny:** one redundant `rmsnorm_quantized` on layer 0 per token.
  Ceilings FAIL: prefill512.operations 9802 > 9801, decode64.operations
  97856 > 97792 (+0.07%), suffix_copy48.operations 62644 > 62605. Far below
  any wall-clock band; only the exact count sees it.
- **big:** a second `ffn_down` projection in every layer. Ceilings FAIL:
  decode64.operations 105984 > 97792 (+8.4%), suffix_copy48 +8.0%,
  prefill512 +1.3%.

Proxy vs wall clock (P4): the **big** mutant against the real binary,
interleaved to cancel background load, t2-slim greedy decode of 128 tokens:

| round | real tok/s | big tok/s |
| --- | --- | --- |
| 1 | 11.82 | 9.03 |
| 2 | 10.88 | 8.91 |
| 3 | 11.79 | 9.88 |

Operations +8.4% came with decode 16-19% slower: the count moves with real
cost and catches it exactly, but it weighs every operation equally, so it
detects a regression without sizing it (one `ffn_down` matmul is far heavier
than an average op). Size comes from the journeys. Real-binary spread across
rounds was ~8%, which is why journeys take the best of 3 against a 10% band.

## Baselines

`baseline-Mac16,10-16g.tsv` (the mini) is **provisional**: recorded
2026-10-05 with `mediaanalysisd` and Spotlight using ~130% CPU in the
background (load average 3.3). Its 512-token prefill reads 36.5 tok/s against
41 tok/s in the metal-v0.7.0 notes. Re-record on an idle machine before
treating a journey FAIL as a regression.

## Seen in the counts

On t2-slim the suffix-burst copy of 48 tokens uses 82 command buffers against
12 for 48 plain decode tokens: bursts save ~15% of operations but add a CPU
round trip per burst round. A lead for agent copy/edit speed.
