#!/usr/bin/env python3
"""Structural lint for tools/eval/prompts/*.jsonl. Stdlib only, no model.

Every item must survive its own harness: a synthetic reply built from the
item's gold ("The answer is <gold>.") is pushed through the REAL extractor
for its mode and must come back equal to the gold (choice: the letter;
numeric: isclose; freeform: normalized string). An item whose gold cannot
round-trip would score as disagreement for every arm — a corpus bug, not
a capability signal. Also checks: unique prompt_ids across ALL files (one
namespace — arms' output files are keyed by prompt_id alone), non-empty
prompts, the anchored-answer instruction present, choice options A)-D)
(or A)-C)) present with gold among them.

Exit 0 green; exit 1 lists every failing check (masked-failure lesson:
check the exit code, never grep for PASS).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import (extract_choice, extract_freeform, extract_numeric,
                     normalize_freeform)

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
MODES = ("choice", "numeric", "freeform")
ANCHOR = "The answer is"

errors = []


def err(msg):
    errors.append(msg)


def load(mode):
    path = os.path.join(PROMPT_DIR, mode + ".jsonl")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append((ln, json.loads(line)))
            except json.JSONDecodeError as e:
                err("%s:%d: bad JSON: %s" % (path, ln, e))
    return path, items


def roundtrip(mode, gold):
    reply = ("Some brief reasoning first.\n"
             "The answer is %s." % gold)
    if mode == "choice":
        return extract_choice(reply) == gold
    if mode == "numeric":
        got = extract_numeric(reply)
        return got is not None and math.isclose(got, float(gold),
                                                rel_tol=1e-4, abs_tol=1e-9)
    return extract_freeform(reply) == normalize_freeform(gold)


def main():
    seen_ids = {}
    counts = {}
    for mode in MODES:
        path, items = load(mode)
        counts[mode] = len(items)
        for ln, obj in items:
            where = "%s:%d" % (path, ln)
            pid = obj.get("prompt_id")
            if not pid:
                err(where + ": missing prompt_id")
                continue
            if pid in seen_ids:
                err("%s: prompt_id %r already used at %s"
                    % (where, pid, seen_ids[pid]))
            seen_ids[pid] = where
            prompt = obj.get("prompt", "")
            gold = obj.get("gold", "")
            if not prompt.strip():
                err(where + ": empty prompt")
            if not str(gold).strip():
                err(where + ": empty gold")
                continue
            if ANCHOR not in prompt:
                err(where + ": prompt lacks the anchored-answer instruction")
            if mode == "choice":
                if gold not in "ABCDE" or len(gold) != 1:
                    err(where + ": choice gold %r is not a single letter" % gold)
                    continue
                opts = [c for c in "ABCD" if ("%s) " % c) in prompt]
                if len(opts) < 3:
                    err(where + ": fewer than 3 lettered options found")
                if gold not in opts:
                    err(where + ": gold %r not among options %r" % (gold, opts))
            if mode == "numeric":
                try:
                    float(gold)
                except ValueError:
                    err(where + ": numeric gold %r is not a number" % gold)
                    continue
            if not roundtrip(mode, gold):
                err(where + ": gold %r fails extractor round-trip" % gold)
    for mode, n in counts.items():
        print("promptlint: %-8s %d items" % (mode, n))
    if errors:
        for e in errors:
            print("FAIL: " + e)
        return 1
    print("promptlint: ALL PASS (%d items, ids unique across files)"
          % len(seen_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
