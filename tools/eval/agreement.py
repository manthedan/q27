#!/usr/bin/env python3
"""Output-agreement gate between two saved generation files (weak tier
vs strong tier, triage item U2 first gate). Consumes SAVED generations
only; it never runs a model.

JSONL contract, one object per line:
    {"id": <any>, "prompt_id": <str/int>, "text": <str>}
Items are paired by prompt_id (fallback: id). Duplicate prompt_ids
within a file are an error.

Usage:
    agreement.py A.jsonl B.jsonl --mode {numeric,choice,freeform}
                 [--min-rate R] [--rel-tol T] [--quiet]

Agreement rate = agreeing pairs / all paired items. A pair where either
side yields no extractable answer counts as DISAGREE (an unextractable
weak-tier answer is a capability signal, not missing data); pairs where
BOTH sides are unextractable also disagree, and are reported separately
so a degenerate corpus is visible. Exits 1 if rate < --min-rate,
2 on input errors.
"""

import argparse
import json
import math
import sys

from extract import extract_numeric, extract_choice, extract_freeform

EXTRACTORS = {
    "numeric": extract_numeric,
    "choice": extract_choice,
    "freeform": extract_freeform,
}


def load_jsonl(path):
    items = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit("%s:%d: bad JSON: %s" % (path, ln, e))
            key = obj.get("prompt_id", obj.get("id"))
            if key is None:
                sys.exit("%s:%d: no prompt_id or id" % (path, ln))
            if "text" not in obj:
                sys.exit("%s:%d: no text field" % (path, ln))
            if key in items:
                sys.exit("%s:%d: duplicate prompt_id %r" % (path, ln, key))
            items[key] = obj["text"]
    if not items:
        sys.exit("%s: no items" % path)
    return items


def agree(a, b, mode, rel_tol):
    if a is None or b is None:
        return False
    if mode == "numeric":
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)
    return a == b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file_a", help="generations A (e.g. weak tier)")
    ap.add_argument("file_b", help="generations B (e.g. strong tier)")
    ap.add_argument("--mode", required=True, choices=sorted(EXTRACTORS))
    ap.add_argument("--min-rate", type=float, default=None,
                    help="exit 1 if agreement rate < this")
    ap.add_argument("--rel-tol", type=float, default=1e-4,
                    help="numeric-mode relative tolerance")
    ap.add_argument("--quiet", action="store_true",
                    help="summary only, no per-item lines")
    args = ap.parse_args(argv)

    ex = EXTRACTORS[args.mode]
    a_items, b_items = load_jsonl(args.file_a), load_jsonl(args.file_b)
    keys = sorted(set(a_items) & set(b_items), key=str)
    only_a = len(a_items) - len(keys)
    only_b = len(b_items) - len(keys)
    if not keys:
        sys.exit("no shared prompt_ids between %s and %s"
                 % (args.file_a, args.file_b))

    n_agree = n_none_a = n_none_b = n_none_both = 0
    for k in keys:
        ea, eb = ex(a_items[k]), ex(b_items[k])
        ok = agree(ea, eb, args.mode, args.rel_tol)
        n_agree += ok
        n_none_a += ea is None
        n_none_b += eb is None
        n_none_both += ea is None and eb is None
        if not args.quiet:
            print("%-6s %-20s a=%r b=%r"
                  % ("AGREE" if ok else "differ", str(k), ea, eb))

    rate = n_agree / len(keys)
    print("pairs=%d agree=%d rate=%.4f none_a=%d none_b=%d none_both=%d"
          % (len(keys), n_agree, rate, n_none_a, n_none_b, n_none_both))
    if only_a or only_b:
        print("unpaired: %d only in A, %d only in B" % (only_a, only_b))
    if args.min_rate is not None and rate < args.min_rate:
        print("GATE FAIL: rate %.4f < min-rate %.4f" % (rate, args.min_rate))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
