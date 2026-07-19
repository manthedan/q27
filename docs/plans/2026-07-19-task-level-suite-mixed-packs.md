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
