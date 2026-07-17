# tools/eval — capability-eval harness v1

Triage item U2 (docs/plans/2026-07-17-ds4-product-triage.md): gate
infrastructure for "does the agent still finish tasks" questions that
NLL/KL/needle cannot answer. Pure CPU, Python 3 stdlib only, no model
loads anywhere in this directory.

## What exists

- `extract.py` — answer extractors: numeric (boxed / answer-tag /
  final-line), multiple-choice letter (anchored forms only; a letter in
  prose never extracts), freeform short answer with
  case/whitespace/article normalization. All extractors strip
  `<think>`, `<tool_call>`, `<tool_response>` blocks and dangling
  `{"tool_call":` fragments first (shapes per src/api_common.h);
  truncated-mid-think output yields None, not a leaked thought.
- `grade.py` — graders returning `(verdict, reason)`: exact-match,
  numeric-with-tolerance, multiple-choice. `None` prediction always
  fails.
- `agreement.py` — output-agreement gate between two SAVED generation
  files (weak tier vs strong tier, ds4's sufficient first gate).
- `selftest.py` — golden self-tests. Every extractor and grader proves
  it can fail: deliberately-wrong answers grade False, label-shadow
  and distractor traps must not extract, and the agreement gate is
  driven to exit 1 in a subprocess with the real exit code checked.
- `gen_runner.py` — the generation runner that produces agreement.py
  JSONL from a running q27 server (`--api anthropic|openai`, serial
  requests, one retry, failed prompts recorded with empty text + error
  field and a nonzero exit). AUTHORED AND MOCK-TESTED ONLY: it must
  only ever RUN inside a coordinated GPU slot against a server whose
  owner expects the traffic (COORDINATION.md rules apply — it is a
  model consumer even though this directory loads no model).
- `selftest_gen.py` — mock-server self-test for gen_runner.py (stdlib
  HTTPServer; both API shapes, dead-server and duplicate-id must-fail
  directions, real exit codes).

## Run the self-tests

    python3 tools/eval/selftest.py
    python3 tools/eval/selftest_gen.py

Exit 0 = green. Any failure exits 1 and lists each failing check.
Check the exit code, never grep for "PASS" (masked-failure lesson,
tasks/lessons.md).

## agreement.py contract

Input: two JSONL files, one object per line:

    {"id": <any>, "prompt_id": <str|int>, "text": <str>}

Pairing is by `prompt_id` (fallback `id`); duplicates within a file
are an error (exit 2). `text` is the raw model output including any
think/tool wrappers — stripping is the harness's job.

    python3 tools/eval/agreement.py weak.jsonl strong.jsonl \
        --mode numeric --min-rate 0.9

Modes: `numeric` (isclose, `--rel-tol`, default 1e-4), `choice`,
`freeform`. Rate = agreeing pairs / all paired items; a pair where
either side is unextractable counts as disagree (unextractable output
is a capability signal, not missing data), with `none_a/none_b/
none_both` reported so a degenerate corpus is visible. Exits 1 when
rate < `--min-rate`, 2 on input errors, 0 otherwise.

## Not yet built (deliberate, keep scope creep visible)

- Capability question sets (GPQA-class) — only if a decision ever
  needs them, per the triage verdict.
- The first real agreement RUN (gen_runner.py exists but has never
  touched a live server) — a later GPU-slot task with its own
  pre-registered prompt set and threshold.
- Continuation-fixture regression corpus (ds4 glm5.2 model) — after
  the first real agreement run exists.
- Task-completion / agentic probes — separate from answer-grading.
