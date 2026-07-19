#!/usr/bin/env python3
"""Gold-accuracy driver for the capability spot-check set
(docs/plans/2026-07-19-task-level-suite-mixed-packs.md). Stdlib only,
consumes SAVED per-arm generation JSONL written by run_arm.sh
(<dir>/<arm>.<mode>.jsonl, rows {"id","prompt_id","text"}) and scores each
row against the gold answer in tools/eval/prompts/<mode>.jsonl using the
SAME extractors + graders as selftest.py. Loads no model.

Reports per-mode and overall accuracy per arm, plus the two reference
deltas that make the mixed-pack question legible (diagnostic only — this
is NOT a ship gate and reopens nothing):

    gap-to-t2          = acc(arm) - acc(ref arm)         (overall, vs t2)
    gap-recovered      = (acc(arm) - acc(floor)) / (acc(ref) - acc(floor))
                         for acc(ref) > acc(floor), else undefined (nan)

Fail-closed: a missing/truncated generation row, an unknown prompt_id, or
an arm whose overall accuracy is strictly below the floor arm exits 1
(exit codes are the contract — never grep for PASS).

    accuracy.py --dir logs/eval-census --ref t2-base --floor b1-base \
        --arms gdn-pair m1-candidate
"""
import argparse
import json
import math
import os
import sys

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import extract  # noqa: E402
import grade    # noqa: E402

MODES = ("choice", "numeric", "freeform")
EXPECTED_COUNTS = {"choice": 60, "numeric": 40, "freeform": 20}

# mode -> (extractor, grader). Freeform grades by exact-match after
# normalize_freeform on both sides (grade_exact); numeric by tolerance.
_PIPE = {
    "choice": (extract.extract_choice, grade.grade_choice),
    "numeric": (extract.extract_numeric, grade.grade_numeric),
    "freeform": (extract.extract_freeform, grade.grade_exact),
}


def load_prompts(mode):
    path = os.path.join(_HERE, "prompts", mode + ".jsonl")
    gold = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid, g = r.get("prompt_id"), r.get("gold")
            if pid is None or g is None:
                sys.exit(f"prompts/{mode}.jsonl line {ln}: missing prompt_id/gold")
            if pid in gold:
                sys.exit(f"prompts/{mode}.jsonl line {ln}: duplicate prompt_id {pid!r}")
            gold[pid] = str(g)
    if len(gold) != EXPECTED_COUNTS[mode]:
        sys.exit(f"prompts/{mode}.jsonl: expected {EXPECTED_COUNTS[mode]} items, "
                 f"found {len(gold)}")
    return gold


def load_generations(path, arm, mode):
    rows = {}
    if not os.path.exists(path):
        sys.exit(f"missing generation file for arm {arm!r}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("prompt_id")
            if pid is None:
                sys.exit(f"{path} line {ln}: missing prompt_id")
            if pid in rows:
                sys.exit(f"{path} line {ln}: duplicate prompt_id {pid!r}")
            rows[pid] = r.get("text", "")
    return rows


def score_arm(arm, gen_dir, golds):
    """Return (per_mode {mode:(correct,total)}, overall (correct,total)).
    Fail-closed on any missing/unknown prompt_id."""
    per_mode = {}
    tot_c = tot_n = 0
    for mode in MODES:
        gold = golds[mode]
        rows = load_generations(os.path.join(gen_dir, f"{arm}.{mode}.jsonl"),
                                arm, mode)
        if set(rows) != set(gold):
            missing = sorted(set(gold) - set(rows))[:5]
            extra = sorted(set(rows) - set(gold))[:5]
            sys.exit(f"{arm}.{mode}: prompt_id set mismatch "
                     f"(missing {missing}, extra {extra}) — incomplete run, "
                     f"refusing to score")
        extractor, grader = _PIPE[mode]
        correct = 0
        for pid, g in gold.items():
            pred = extractor(rows[pid])
            ok, _reason = grader(pred, g)
            if ok:
                correct += 1
        per_mode[mode] = (correct, len(gold))
        tot_c += correct
        tot_n += len(gold)
    return per_mode, (tot_c, tot_n)


def acc(ct):
    c, n = ct
    return c / n if n else math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="generation dir (run_arm.sh OUT)")
    ap.add_argument("--ref", required=True, help="reference arm (quality tier)")
    ap.add_argument("--floor", required=True, help="floor arm (weak tier)")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="candidate arms to score (ref/floor are scored too)")
    args = ap.parse_args()

    golds = {m: load_prompts(m) for m in MODES}
    all_arms = [args.ref, args.floor] + [a for a in args.arms
                                         if a not in (args.ref, args.floor)]
    results = {}
    for arm in all_arms:
        per_mode, overall = score_arm(arm, args.dir, golds)
        results[arm] = (per_mode, overall)

    ref_acc = acc(results[args.ref][1])
    floor_acc = acc(results[args.floor][1])

    hdr = f"{'arm':16s} " + " ".join(f"{m:>16s}" for m in MODES) + \
          f" {'overall':>10s} {'gap-to-ref':>11s} {'gap-rec':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for arm in all_arms:
        per_mode, overall = results[arm]
        cells = " ".join(f"{c}/{n} ({c/n:.2f})".rjust(16) for m in MODES
                         for c, n in [per_mode[m]])
        a = acc(overall)
        gap = a - ref_acc
        rec = (a - floor_acc) / (ref_acc - floor_acc) \
            if ref_acc > floor_acc else math.nan
        print(f"{arm:16s} {cells} {overall[0]}/{overall[1]} ({a:.3f}) "
              f"{gap:+.3f}".ljust(len(hdr) - 9) + (f"{rec:7.2f}" if not math.isnan(rec) else "    nan"))

    # fail-closed: any candidate strictly below the floor arm exits 1.
    rc = 0
    for arm in all_arms:
        if arm in (args.ref, args.floor):
            continue
        if acc(results[arm][1]) < floor_acc:
            print(f"BELOW-FLOOR: {arm} overall {acc(results[arm][1]):.3f} "
                  f"< floor {args.floor} {floor_acc:.3f}", file=sys.stderr)
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
