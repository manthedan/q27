#!/usr/bin/env python3
"""Golden self-tests for the eval harness. Exits nonzero on ANY failure.

Load-bearing property (masked-failure lesson): every extractor and
grader must PROVE IT CAN FAIL. Deliberately-wrong answers MUST grade
False, unanchored letters MUST NOT extract, and the agreement gate MUST
exit 1 on a below-threshold corpus. Exit codes are checked directly --
never grep output for PASS.

Run: python3 tools/eval/selftest.py
"""

import json
import os
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True  # keep tools/eval free of __pycache__
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # ...including subprocesses
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract import (strip_wrappers, extract_numeric, extract_choice,
                     extract_freeform, normalize_freeform)
from grade import grade_exact, grade_numeric, grade_choice

HERE = os.path.dirname(os.path.abspath(__file__))
FAILURES = []


def check(name, got, want):
    if got != want:
        FAILURES.append("%s: got %r, want %r" % (name, got, want))


def check_verdict(name, result, want):
    verdict, reason = result
    if verdict is not want:
        FAILURES.append("%s: verdict %r (want %r), reason: %s"
                        % (name, verdict, want, reason))


# --- wrapper stripping ------------------------------------------------------

check("strip closed think",
      strip_wrappers("<think>secret 99</think>The answer is 4."),
      "The answer is 4.")
check("strip tool_call",
      strip_wrappers('x<tool_call>\n{"name":"bash","arguments":{}}\n</tool_call>y'),
      "xy")
check("strip tool_response",
      strip_wrappers("<tool_response>\n12 files\n</tool_response>done"),
      "done")
check("strip dangling bare-JSON opener",
      strip_wrappers('Answer: 7\n{"tool_call": {"name":'),
      "Answer: 7\n")
check("truncated think drops to end",
      strip_wrappers("<think>the answer must be 42 because"),
      "")

# --- numeric extraction -----------------------------------------------------

check("boxed numeric", extract_numeric(r"So \boxed{42}."), 42.0)
check("boxed beats final line", extract_numeric(r"\boxed{7}" + "\nAlso 9"), 7.0)
check("answer tag", extract_numeric("Adding up, the answer is 108."), 108.0)
check("final-line last number, distractors ignored",
      extract_numeric("There are 12 apples and 5 oranges.\nTotal: 17"), 17.0)
check("comma and dollar", extract_numeric("Answer: $1,234.50"), 1234.5)
check("negative", extract_numeric("The answer is -3.5"), -3.5)
check("think-block number not leaked",
      extract_numeric("<think>maybe 99?</think>\nThe answer is 4"), 4.0)
check("no number -> None", extract_numeric("I cannot determine this."), None)
check("truncated mid-think -> None",
      extract_numeric("<think>let me compute 6*7"), None)
check("empty -> None", extract_numeric(""), None)

# MUST-FAIL proof: a wrong extraction would be caught here.
check("distractor NOT extracted as answer",
      extract_numeric("Start with 500 widgets.\nThe answer is 3.") == 500.0,
      False)

# --- choice extraction ------------------------------------------------------

check("answer-is letter", extract_choice("The answer is B."), "B")
check("boxed letter", extract_choice(r"\boxed{C}"), "C")
check("bare final line", extract_choice("Reasoning here.\n(D)"), "D")
check("option tag", extract_choice("I pick option C because it fits."), "C")
check("lowercase folded", extract_choice("answer: d"), "D")
check("label shadow: prose letter NOT extracted",
      extract_choice("A quick check shows the plan works."), None)
check("letter inside word NOT extracted",
      extract_choice("The answer is Brussels."), None)
check("think letter not leaked",
      extract_choice("<think>surely B</think>No idea."), None)
check("empty -> None", extract_choice(""), None)

# --- freeform extraction ----------------------------------------------------

check("answer tag freeform",
      extract_freeform("Thinking...\nThe answer is The Eiffel Tower."),
      "eiffel tower")
check("final line fallback", extract_freeform("blah\nParis"), "paris")
check("article+case+ws folding",
      normalize_freeform("  An   Elephant. "), "elephant")
check("think not leaked",
      extract_freeform("<think>maybe Rome</think>Paris"), "paris")
check("empty -> None", extract_freeform("<think>hmm</think>"), None)

# --- graders: pass AND fail directions --------------------------------------

check_verdict("exact pass", grade_exact("The Eiffel Tower", "eiffel tower"), True)
check_verdict("exact FAIL on wrong answer", grade_exact("louvre", "eiffel tower"), False)
check_verdict("exact FAIL on None", grade_exact(None, "x"), False)

check_verdict("numeric pass", grade_numeric(42.0, 42), True)
check_verdict("numeric tolerance pass", grade_numeric(1.00001, 1.0, rel_tol=1e-4), True)
check_verdict("numeric FAIL outside tol", grade_numeric(1.1, 1.0, rel_tol=1e-4), False)
check_verdict("numeric FAIL on wrong answer", grade_numeric(17, 18), False)
check_verdict("numeric FAIL on None", grade_numeric(None, 5), False)
check_verdict("numeric FAIL on junk", grade_numeric("banana", 5), False)

check_verdict("choice pass", grade_choice("b", "B"), True)
check_verdict("choice FAIL on wrong letter", grade_choice("A", "B"), False)
check_verdict("choice FAIL on None", grade_choice(None, "B"), False)
check_verdict("choice FAIL on multi-char", grade_choice("AB", "A"), False)

# --- end-to-end: extract then grade a deliberately-wrong transcript ---------

wrong = "<think>hmm</think>\nAfter checking, the answer is 41."
check_verdict("meta: known-bad transcript grades FAIL",
              grade_numeric(extract_numeric(wrong), 42), False)
truncated = "<think>I should compute 6*7 which is"
check_verdict("meta: truncated transcript grades FAIL",
              grade_numeric(extract_numeric(truncated), 42), False)

# --- agreement.py subprocess: real exit codes -------------------------------

def run_agreement(args):
    p = subprocess.run(
        [sys.executable, os.path.join(HERE, "agreement.py")] + args,
        capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


with tempfile.TemporaryDirectory() as td:
    a = os.path.join(td, "a.jsonl")
    b = os.path.join(td, "b.jsonl")
    # 4 pairs: agree, agree, disagree, one-side-unextractable
    write_jsonl(a, [
        {"id": 1, "prompt_id": "p1", "text": "The answer is 10"},
        {"id": 2, "prompt_id": "p2", "text": "<think>x</think>Answer: 3"},
        {"id": 3, "prompt_id": "p3", "text": "The answer is 7"},
        {"id": 4, "prompt_id": "p4", "text": "no clue at all"},
    ])
    write_jsonl(b, [
        {"id": 1, "prompt_id": "p1", "text": "Result: 10"},
        {"id": 2, "prompt_id": "p2", "text": "So the answer is 3."},
        {"id": 3, "prompt_id": "p3", "text": "The answer is 8"},
        {"id": 4, "prompt_id": "p4", "text": "Answer: 5"},
    ])

    rc, out = run_agreement([a, b, "--mode", "numeric"])
    check("agreement: no gate exits 0", rc, 0)
    check("agreement: rate line correct",
          "pairs=4 agree=2 rate=0.5000" in out, True)
    check("agreement: unextractable counted",
          "none_a=1" in out and "none_both=0" in out, True)

    rc, _ = run_agreement([a, b, "--mode", "numeric", "--min-rate", "0.4"])
    check("agreement: gate above threshold exits 0", rc, 0)

    # MUST-FAIL proof for the gate itself
    rc, out = run_agreement([a, b, "--mode", "numeric", "--min-rate", "0.9"])
    check("agreement: gate below threshold exits 1", rc, 1)
    check("agreement: gate failure stated", "GATE FAIL" in out, True)

    bad = os.path.join(td, "bad.jsonl")
    with open(bad, "w") as f:
        f.write("{not json\n")
    rc, _ = run_agreement([bad, b, "--mode", "numeric"])
    check("agreement: bad input exits nonzero", rc != 0, True)

    dup = os.path.join(td, "dup.jsonl")
    write_jsonl(dup, [{"prompt_id": "p1", "text": "1"},
                      {"prompt_id": "p1", "text": "2"}])
    rc, _ = run_agreement([dup, b, "--mode", "numeric"])
    check("agreement: duplicate prompt_id exits nonzero", rc != 0, True)

    # choice mode end-to-end with a label-shadow trap in one file
    ca = os.path.join(td, "ca.jsonl")
    cb = os.path.join(td, "cb.jsonl")
    write_jsonl(ca, [{"prompt_id": "q1", "text": "The answer is B."},
                     {"prompt_id": "q2", "text": "A close look reveals nothing."}])
    write_jsonl(cb, [{"prompt_id": "q1", "text": "answer: b"},
                     {"prompt_id": "q2", "text": "The answer is A."}])
    rc, out = run_agreement([ca, cb, "--mode", "choice"])
    check("agreement choice: shadow trap disagrees, not false-agrees",
          "pairs=2 agree=1" in out, True)
    check("agreement choice exits 0 without gate", rc, 0)

# --- report -----------------------------------------------------------------

if FAILURES:
    print("SELFTEST FAILED: %d failure(s)" % len(FAILURES))
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("selftest: all checks passed (extractors, graders, agreement gate; "
      "must-fail directions exercised)")
sys.exit(0)
