# Agentic replay bench — metric pre-registration (2026-07-17)

Adopted from the operator's roadmap review ("sure", 2026-07-17 evening):
the README's 5090 headline is an agentic replay number (202.7 t/s on
SWE-bench replay) while every M4 number is synthetic or single-prompt.
The trace instrument now records this operator's real agent traffic —
replaying it against the server produces "M4 agentic effective tok/s":
one number that prices snapshots + suffix bursts + queueing + the whole
serving loop together. It becomes the PRODUCT gate metric (mixed-pack
serving claims, homebrew Phase-4, suffix gate-6 refreshes) — beside the
kernel-level ship lines, never instead of them.

Metric definitions are fixed HERE, before the first measurement, so
every future number is comparable:

- **agentic effective output tok/s** (the headline) = total generated
  tokens / total turn wall, summed over the corpus. Wall = request
  sent -> stream closed; turns replay serially, back-to-back,
  think-time excluded (this measures server capacity, not session
  latency — stated wherever the number is quoted).
- **TTFT** = request sent -> first content chunk; report median and
  p95 beside the headline (a single scalar hides the snapshot story,
  our main differentiator).
- **context throughput** = total prompt tokens / total TTFT.
- Token accounting and terminal cause are exact via the replay server's
  own trace (boot-unique id join). A missing join, output-count mismatch,
  or terminal-cause mismatch invalidates the run; nothing is approximated.
  The runner accepts only a proxy-disabled literal loopback origin, captures
  artifact/build/shader/tokenizer/config/platform/boot plus the install-local
  evaluation host identity, rechecks server identity after all turns, and
  embeds that provenance in the summary.

## Mechanics (tools/, all stdlib, selftested)

- `agentic_replay_extract.py` — trace -> corpus. The RENDERED prompt
  replays VERBATIM through /v1/completions (raw-prompt endpoint, full
  serving machinery), so replay is byte-identical to recorded traffic
  and needs no message reconstruction. Teacher-forced by construction:
  turn N's prompt embeds the RECORDED prior turns — deterministic and
  comparable across packs/commits, stated in the writeup. Drops are
  counted, never silent: truncated rendered (64 KB trace cap),
  cancelled/errored turns, sub-512-token prompts (gate fixtures),
  non-generation apis. The corpus retains max_tokens, sampling controls,
  stop sequences, constrained-tool names, output count, terminal cause, source
  boot ID, and full source model/runtime identity; incompatible source boots
  fail extraction. Replay sends controls verbatim and must reproduce
  count/cause exactly.
- `agentic_replay_bench.py` — the serial replay runner + summary.
  AUTHORED AND MOCK-TESTED ONLY; a model consumer under
  COORDINATION.md (coordinated GPU slot, owner expects the traffic).
- `agentic_replay_selftest.py` — 18 checks incl. must-fail directions
  (all-filtered corpus exit 2, dead server nonzero, empty corpus
  nonzero); ALL PASS.

## Corpus discipline

A published number names its corpus (turn count, prompt/output token
totals, date range) and the serving line that produced it. Corpora are
frozen files: re-extract under a new name, never mutate one a number
was published against. First extraction smoke (2026-07-17 evening
trace): 14 turns, 39,109 prompt tokens, 405 output tokens — too thin
to publish; the corpus accumulates from real pi/codex traffic and the
first real run waits for a representative week plus a coordinated
slot. No performance bands are pre-registered — the first quiet run
IS the baseline; bands attach when a specific decision adopts this
metric as its gate.

## RESULTS

(pending — first real run after the corpus fattens; quiet machine,
serving line recorded)
