# tools/eval — capability-eval harness v1

Triage item U2 (docs/metal/plans/2026-07-17-ds4-product-triage.md): gate
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
- `prompts/{choice,numeric,freeform}.jsonl` — the census capability
  spot-check set (120 items: 60 anchored MC, 40 numeric word problems,
  20 freeform short answers; fields prompt_id/prompt/gold/category).
  Authored for the mixed-pack ship gate
  (docs/metal/plans/2026-07-17-census-capability-spotcheck.md). Agreement is a
  supporting read only; the amended A6 gate is ground-truth accuracy
  non-inferiority and its verdict driver is not yet authored.
- `promptlint.py` — structural lint for the prompt set: unique ids
  across all files, anchored-answer instruction present, options
  well-formed, and every gold ROUND-TRIPS through its real extractor
  (an item whose gold cannot extract would score as disagreement for
  every arm — a corpus bug, not a capability signal). Proves failure:
  duplicate id / missing anchor / bad gold each fail loudly.
- `arms.tsv` + `run_arm.sh` — frozen artifact filename/MD5/size for each
  decision arm and one serial generation pass over the three prompt files.
  The runner is localhost-only, verifies the local full-file MD5 plus the
  server's opt-in `/health?identity=1` SHA1 over the resident mmap plus its
  artifact filename (without exposing the absolute deployment path), then
  writes a mandatory artifact + runtime provenance sidecar (server/shader/
  tokenizer hashes, normalized protocol/numeric-path settings, hardware/OS,
  an install-local private-file host identity, and server boot; verdict requires
  exact runtime/machine equality across arms,
  while the runner rechecks the same boot before publication). Generation
  publishes from a temporary workspace with
  provenance last; per-row IDs are bound to the sidecar run ID, so interrupted
  reruns fail closed instead of mixing stale modes
  (COORDINATION.md applies: coordinated GPU slot only; greedy).
- `spotcheck_verdict.py` — supporting per-arm agreement report vs the
  reference arm (subprocesses agreement.py — one source of truth), with
  the capability-gap-recovered metric. It rejects incomplete/wrong prompt
  ID sets and missing/mismatched artifact provenance. It cannot clear
  amended A6; exit 0 means the report completed, while worse-than-floor
  still exits 1.

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

- The first real agreement RUN (gen_runner.py exists but has never
  touched a live server) — now concretely queued as the census
  capability spot-check on the mini (plan doc above has the arms,
  protocol, and pre-registered bands); still a GPU-slot task.
- Continuation-fixture regression corpus (ds4 glm5.2 model) — after
  the first real agreement run exists.
- Task-completion / agentic probes — separate from answer-grading.
