# Task-level suite on the mixed packs — pre-registration (2026-07-19)

Operator-directed, 2026-07-19: "I would also love to see how our mixed packs
score on a task level suite, can we try that on our packs?" — scope: 4 arms,
gold-accuracy driver built.

## Why, and what this is NOT

A5 (KL-to-official) permanently closed the mixed-pack **serving** thesis
(`2026-07-18-kl-pair-a5.md` RESULTS): both cheap_pair and gdn_pair rotate
away from official on both corpora. This run does NOT reopen that. It is
**diagnostic color on the one axis we have never measured for a mixed
pack: task-level accuracy.** The whitepaper's central claim is task-level
(the ternary holds long-chain-reasoning benchmarks that conventional
sub-4-bit quants collapse). The open diagnostic question: does the KL
rotation we found cost *actual task accuracy*, or is it a silent
distributional drift that tasks tolerate? NLL says nothing here; KL says
nothing here; only a gold-graded task run answers it.

There is no ship line and no gate that re-opens serving. The result is a
measurement with a pre-registered interpretation, recorded either way.

## Harness (all pre-existing, self-tested)

`tools/eval/` (authored for the census ship gate,
`2026-07-17-census-capability-spotcheck.md`): 120 gold-labelled prompts
across three files — `choice.jsonl` (60 MC: logic/science/knowledge/math/
code/reasoning), `numeric.jsonl` (40 math word problems), `freeform.jsonl`
(20 short-answer knowledge). Extractors strip think/tool wrappers and use
anchored forms only; graders score against `gold` (exact / numeric-with-
tolerance / MC letter). Every extractor and grader proves it can fail in
`selftest.py`.

**New code (this run):** `tools/eval/accuracy.py` — a thin gold-accuracy
driver that consumes SAVED per-arm generation JSONL (`{id, prompt_id,
text}`), maps each prompt file to its extractor+grader, and reports
per-file and overall accuracy per arm. It loads no model; generation is
the existing `gen_runner.py` / `run_arm.sh` path. Self-test added: a
deliberately-wrong generation file MUST score below a correct one and the
driver MUST exit 1 when an arm is worse-than-floor (fail-closed, exit-code
checked, never grep-for-PASS).

## Arms (4), serial, one resident model, greedy

- **t2-base** (control / quality tier) — the strong reference.
- **b1-base** (floor / small tier) — the weak reference.
- **gdn_pair** (mixed, NLL-strongest, A5-KILLED) — rebuilt locally via
  `tools/q27_mix.py --take '(attn_qkv|ssm_(alpha|beta))\.weight'`, md5
  verified `107647e9cba0f01a003934011644c2fe` before generation.
- **m1-candidate** (mixed cheap_pair, A5-KILLED) — already local at
  `models/bonsai-27b-m1/` (md5 `91db7fdd…`).

Generation: greedy, thinking mode (server default), one prompt at a time
via the existing runner, against a single-arm server on :8213, torn down
between arms. Provenance: the runner's existing sidecar (artifact MD5 +
server/shader/tokenizer hashes + boot) pins each arm's generations.

## Pre-registered interpretation (not a gate)

Report per-arm accuracy: overall and per file (choice/numeric/freeform),
plus the two reference deltas that make the mixed-pack question legible:

- **gap-to-t2** = acc(arm) − acc(t2-base). Negative = the mixed pack lost
  task accuracy vs the quality tier.
- **gap-recovered-vs-b1** = (acc(arm) − acc(b1)) / (acc(t2) − acc(b1)) for
  acc(t2) > acc(b1). The task-level analogue of the census's NLL gap
  metric: how much of the b1→t2 task-accuracy gap the graft recovers.

Read (no ship consequence either way):
- If gdn_pair / m1-candidate land near t2-base on accuracy despite the
  A5 KL-rotation → the KL drift is **task-silent**: distributional
  rotation the tasks tolerate. Consistent with the whitepaper's
  "short-form masks the difference" caution in reverse.
- If they drop toward b1-base → the KL rotation **costs real capability**;
  the task suite corroborates A5 from a second, independent axis.
- Either way it is recorded; the serving thesis stays closed.

## Cost / house rules

~120 prompts × 4 arms × greedy, serial, one resident model (teardown
between arms permitted — no live traffic). Estimated well under an hour of
GPU on this M4. No new model downloads, no CUDA, no EvalScope/lm-eval —
the scope is deliberately our own 120-prompt probe, NOT the whitepaper's
15-benchmark H100 suite. gdn_pair pack is transient (/tmp), deleted after
its generation pass.

## RESULTS (2026-07-19, 24 GB M4, one GPU queue) — CONFOUNDED, do not read as capability

All four arms generated and published with provenance (`logs/eval-census/`).
`t2-base` first attempt FAILED closed: its server received SIGTERM mid-run
(prompts c43–c49 connection-refused; run_arm.sh's boot_id check correctly
refused the partial). Re-run clean (0 failures). Scored with
`tools/eval/accuracy.py` (selftest green, provenance + run-binding enforced):

| arm | choice | numeric | freeform | overall |
|---|---|---|---|---|
| t2-base      | 44/60 (0.73) | 34/40 (0.85) | 14/20 (0.70) | 92/120 (0.767) |
| b1-base      | 49/60 (0.82) | 37/40 (0.93) | 18/20 (0.90) | 104/120 (0.867) |
| gdn-pair     | 54/60 (0.90) | 37/40 (0.93) | 19/20 (0.95) | 110/120 (0.917) |
| m1-candidate | 56/60 (0.93) | 40/40 (1.00) | 20/20 (1.00) | 116/120 (0.967) |

**The ranking is a generation-completeness artifact, not capability.** The
empty-response count (<5 chars) anti-correlates almost perfectly with the
accuracy order: t2-base 22 empty (choice 13, freeform 7, numeric 2), b1-base
13, gdn-pair 10, m1-candidate 2. Whichever server returned fewer empty
responses "scored" higher. T2 — the quality tier — produced the MOST empty
outputs (e.g. c11/c15/c26/c27/c31 returned `""` with no error field; c20
truncated mid-sentence). An empty generation scores 0 regardless of the
model's actual capability, so the accuracy ordering tracks **serving
robustness under this 120-prompt serial greedy workload**, not task
capability. `gap-recovered` is undefined (acc(t2) < acc(b1)) for the same
reason.

**What this run actually measured:** a per-arm server-empty-response rate
under 1024-max-token thinking-mode greedy generation. m1-candidate's server
was the most robust (2 empty); t2-base's the least (22 empty). This is a
REAL signal worth its own investigation (why does the T2 server return
empty/truncated at ~18% on this prompt set — thinking-mode budget
exhaustion? an early-EOS / finish-reason bug? suffix/constraint
interaction?) but it is NOT the task-accuracy axis the pre-registration
sought, and it does NOT speak to the A5-closed serving thesis.

**Completed-only re-score (empty rows excluded from numerator AND denominator):**
t2 91/98 (0.929), b1 103/107 (0.963), gdn 108/110 (0.982), m1 116/118
(0.983). The gap narrows but the order holds — and the residual is a SECOND
generation artifact, not capability: T2's remaining failures are
**truncations** of verbose reasoning at the 1024 max_tokens cap (n09/n26 cut
mid-sentence; n10 cut at "The answer is 1" dropping the final digit of "12";
c20/c55 cut mid-sentence), while m1/gdn emit terser reasoning that completes
in budget. T2 ALSO has the most empty responses (the dominant term) — two
distinct generation/serving artifacts (empty-response rate + verbose-reasoning
truncation), both orthogonal to task capability.

**Verdict: this run measured generation/serving robustness, not task
capability.** Two confounds, both on the serving axis: (1) per-arm
empty-response rate under 1024-token thinking greedy (t2 22 > b1 13 > gdn 10
> m1 2); (2) T2's verbose chain-of-thought truncating at the token cap.
Neither says anything about what the models know. The capability question
this suite was built for remains OPEN.

**ROOT CAUSE FOUND (2026-07-19, diagnostic replay, T2 server on :8213):**
the empty responses are **thinking-mode budget exhaustion**, NOT a server
bug and NOT capability. `gen_runner.py` sends `--max-tokens 1024` with
thinking mode ON (server default) and extracts only `type=="text"` blocks
from the Anthropic response. Replaying c11 at max_tokens=1024 against a
live T2 server: `stop_reason=max_tokens`, `usage.output_tokens=1024`,
`content` = **one `thinking` block and NO `text` block** — the model's
2722-char `<think>` consumed the entire 1024-token budget before any
visible answer was emitted, so `extract_text` returns `""`. Replaying the
same prompt at max_tokens=2048: `stop_reason=end_turn`, `content` =
`[thinking, text]`, correct answer ("The answer is C"). The per-arm empty
ordering (t2 22 > b1 13 > gdn 10 > m1 2) is exactly the per-arm
thinking-verbosity ordering: the more verbose the model's reasoning, the
more often its 1024 budget is exhausted inside the think block. This is a
**harness configuration bug**, not a model or server defect — the eval set
max_tokens too low for thinking-mode generation.

**Fix (harness, one line):** the capability suite must budget thinking +
answer, not answer alone. Either (a) raise gen_runner's `--max-tokens` so
NO completion truncates (verify `stop_reason=="end_turn"` for every row —
a `max_tokens` stop is a truncated think, unscoreable), or (b) disable
thinking for the eval if the intent is short-form answers. Scoring must
then use only `end_turn` rows. The 1024 cap was sized for the ANSWER, not
the reasoning the server prepends.

**Note the second artifact is the same root cause in miniature:** T2's
"completed-but-truncated" rows (n10 cut at "The answer is 1", c20
mid-sentence) are think+answer overrunning 1024 so the answer itself
truncates. Both confounds collapse to one: budget too small for
thinking-mode. Re-score after a 2048+ re-run; the capability axis stays
OPEN until then. Generation corpus + provenance retained in
`logs/eval-census/`.
