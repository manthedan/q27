#!/usr/bin/env python3
"""Census capability spot-check verdict
(docs/plans/2026-07-17-census-capability-spotcheck.md). Stdlib only,
consumes SAVED generations — never runs a model.

Reads per-arm generation files written by run_arm.sh
(<dir>/<arm>.<mode>.jsonl) and computes each arm's output-agreement rate
against the reference arm by SUBPROCESSING agreement.py per (arm, mode) —
one source of truth for agreement semantics (unextractable counts as
disagree). Overall rate is pair-weighted across modes.

The pre-registered decision metric mirrors the census's own currency:

    capability gap recovered = (A(candidate) - A(floor)) / (1 - A(floor))

where A(x) is x's overall agreement rate with the reference arm, floor is
the pure weak tier (b1-base), and candidate is a census arm. Bands live in
the plan doc and are echoed here.

    spotcheck_verdict.py --dir logs/eval-census --ref t2-base \
        --floor b1-base --arms gdn-qkv combo
"""
import argparse
import os
import re
import subprocess
import sys

MODES = ("choice", "numeric", "freeform")
SUMMARY = re.compile(
    r"pairs=(\d+) agree=(\d+) rate=([\d.]+) "
    r"none_a=(\d+) none_b=(\d+) none_both=(\d+)")


def agreement(dir_, arm, ref, mode):
    a = os.path.join(dir_, "%s.%s.jsonl" % (arm, mode))
    b = os.path.join(dir_, "%s.%s.jsonl" % (ref, mode))
    for p in (a, b):
        if not os.path.exists(p):
            sys.exit("missing generation file: %s" % p)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "agreement.py")
    r = subprocess.run([sys.executable, script, a, b,
                        "--mode", mode, "--quiet"],
                       capture_output=True, text=True)
    if r.returncode not in (0, 1):
        sys.exit("agreement.py failed on %s vs %s (%s): %s"
                 % (arm, ref, mode, (r.stdout + r.stderr).strip()))
    m = SUMMARY.search(r.stdout)
    if not m:
        sys.exit("cannot parse agreement.py summary for %s/%s: %r"
                 % (arm, mode, r.stdout))
    pairs, agree_n = int(m.group(1)), int(m.group(2))
    return pairs, agree_n, int(m.group(4)), int(m.group(5))


def overall(dir_, arm, ref):
    tp = ta = 0
    per_mode = {}
    nones = 0
    for mode in MODES:
        pairs, agree_n, none_a, _ = agreement(dir_, arm, ref, mode)
        per_mode[mode] = (agree_n, pairs)
        tp += pairs
        ta += agree_n
        nones += none_a
    return ta / tp, per_mode, nones, tp


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="logs/eval-census")
    ap.add_argument("--ref", required=True,
                    help="reference arm name (the tier the claim is against)")
    ap.add_argument("--floor", required=True,
                    help="floor arm name (pure weak tier)")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="candidate arm names")
    args = ap.parse_args()

    print("capability spot-check: agreement with reference '%s' "
          "(greedy, %s)" % (args.ref, "/".join(MODES)))
    a_floor, pm, nones, pairs = overall(args.dir, args.floor, args.ref)
    fmt = "  %-12s overall %.4f  (choice %d/%d, numeric %d/%d, " \
          "freeform %d/%d, unextractable %d)"

    def row(name, rate, pm, nones):
        print(fmt % (name, rate,
                     pm["choice"][0], pm["choice"][1],
                     pm["numeric"][0], pm["numeric"][1],
                     pm["freeform"][0], pm["freeform"][1], nones))

    row(args.floor, a_floor, pm, nones)
    gap = 1.0 - a_floor
    fail = False
    for arm in args.arms:
        a, pm, nones, _ = overall(args.dir, arm, args.ref)
        row(arm, a, pm, nones)
        if gap <= 0:
            print("  %-12s   floor already at ceiling (gap 0) — "
                  "report only, no recovery defined" % arm)
            continue
        rec = (a - a_floor) / gap
        verdict = ("SHIP-SUPPORTING (>= 0.50)" if rec >= 0.50 else
                   "CONDITIONAL (0 <= x < 0.50)" if rec >= 0.0 else
                   "FAIL — worse than the pure weak tier (mirage caught)")
        print("  %-12s   capability gap recovered %.1f%% -> %s"
              % (arm, 100.0 * rec, verdict))
        fail |= rec < 0.0
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
