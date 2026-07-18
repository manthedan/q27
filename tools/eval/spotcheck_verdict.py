#!/usr/bin/env python3
"""Census capability spot-check verdict
(docs/plans/2026-07-17-census-capability-spotcheck.md). Stdlib only,
consumes SAVED generations — never runs a model.

Reads per-arm generation files written by run_arm.sh
(<dir>/<arm>.<mode>.jsonl) and computes each arm's output-agreement rate
against the reference arm by SUBPROCESSING agreement.py per (arm, mode) —
one source of truth for agreement semantics (unextractable counts as
disagree). Overall rate is pair-weighted across modes.

The supporting agreement metric mirrors the census's own currency:

    capability gap recovered = (A(candidate) - A(floor)) / (1 - A(floor))

where A(x) is x's overall agreement rate with the reference arm, floor is
the pure weak tier (b1-base), and candidate is a census arm. This is NOT the
amended A6 ground-truth non-inferiority gate: exit 0 means only that the
agreement report completed. Bands live in the plan doc and are echoed here.

    spotcheck_verdict.py --dir logs/eval-census --ref t2-base \
        --floor b1-base --arms m1-candidate
"""
import argparse
import json
import os
import re
import subprocess
import sys

MODES = ("choice", "numeric", "freeform")
EXPECTED_COUNTS = {"choice": 60, "numeric": 40, "freeform": 20}
PROTOCOL_KEYS = {"context", "kv", "mtp", "suffix", "slots", "prefix_entries",
    "constrain_tools", "snapshots", "snapshot_auto_min", "snapshot_max_bytes",
    "max_tokens_default", "kv_fp16_except",
    "kv_fp16_cell_masks", "kv_side_codec", "gemm_half", "gemm_half_q4",
    "gqa_tile", "gqa_block", "gqa_threshold", "gpu_sample", "resident",
    "bare_system", "tool_strict", "test_failpoints", "tokenizer", "tokenizer_sha1"}
SUMMARY = re.compile(
    r"pairs=(\d+) agree=(\d+) rate=([\d.]+) "
    r"none_a=(\d+) none_b=(\d+) none_both=(\d+)")


def prompt_ids(path, required_id_prefix=None):
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit("%s:%d: bad JSON: %s" % (path, ln, e))
            if required_id_prefix is not None:
                row_id = obj.get("id")
                if not isinstance(row_id, str) or not row_id.startswith(required_id_prefix):
                    sys.exit("%s:%d: row id %r is not bound to run prefix %r"
                             % (path, ln, row_id, required_id_prefix))
            key = obj.get("prompt_id", obj.get("id"))
            if key is None:
                sys.exit("%s:%d: no prompt_id or id" % (path, ln))
            if key in ids:
                sys.exit("%s:%d: duplicate prompt_id %r" % (path, ln, key))
            ids.add(key)
    return ids


def arm_manifest():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arms.tsv")
    arms = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 5 or parts[0] in arms:
                sys.exit("%s:%d: malformed/duplicate arm manifest row" % (path, ln))
            arms[parts[0]] = {"arm": parts[0], "model": parts[1],
                              "md5": parts[2], "resident_sha1": parts[3],
                              "bytes": int(parts[4])}
    return arms


def validate_provenance(dir_, arm):
    manifests = arm_manifest()
    if arm not in manifests:
        sys.exit("arm %r is not frozen in tools/eval/arms.tsv" % arm)
    path = os.path.join(dir_, "%s.provenance.json" % arm)
    if not os.path.exists(path):
        sys.exit("missing provenance file: %s" % path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            got = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        sys.exit("bad provenance file %s: %s" % (path, e))
    artifact = {k: got.get(k) for k in manifests[arm]}
    if artifact != manifests[arm]:
        sys.exit("artifact provenance mismatch for %s: got %r, expected %r"
                 % (arm, artifact, manifests[arm]))
    runtime = got.get("runtime")
    required = {"identity_schema", "server_sha1", "shader_abi", "shader_sha1",
                "protocol", "platform", "eval_host_id"}
    if not isinstance(runtime, dict) or set(runtime) != required or runtime.get("identity_schema") != 2:
        sys.exit("missing/incomplete runtime provenance for %s: %r" % (arm, runtime))
    protocol = runtime.get("protocol")
    if not isinstance(protocol, dict) or set(protocol) != PROTOCOL_KEYS:
        sys.exit("missing/incomplete protocol provenance for %s: %r" % (arm, protocol))
    platform = runtime.get("platform")
    if not isinstance(platform, dict) or set(platform) != {"sysname", "release", "machine", "metal_device"}:
        sys.exit("missing/incomplete platform provenance for %s: %r" % (arm, platform))
    server_boot = got.get("server_boot")
    if not isinstance(server_boot, str) or not server_boot:
        sys.exit("missing server_boot in provenance for %s" % arm)
    run_id = got.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        sys.exit("missing run_id in provenance for %s" % arm)
    return runtime, run_id


def frozen_prompt_ids(mode):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "prompts", "%s.jsonl" % mode)
    ids = prompt_ids(path)
    expected = EXPECTED_COUNTS[mode]
    if len(ids) != expected:
        sys.exit("frozen %s corpus has %d ids; expected %d"
                 % (mode, len(ids), expected))
    return ids


def agreement(dir_, arm, ref, mode):
    a = os.path.join(dir_, "%s.%s.jsonl" % (arm, mode))
    b = os.path.join(dir_, "%s.%s.jsonl" % (ref, mode))
    for p in (a, b):
        if not os.path.exists(p):
            sys.exit("missing generation file: %s" % p)
    _, a_run = validate_provenance(dir_, arm)
    _, b_run = validate_provenance(dir_, ref)
    a_ids = prompt_ids(a, "%s-%s-%s-" % (arm, a_run, mode))
    b_ids = prompt_ids(b, "%s-%s-%s-" % (ref, b_run, mode))
    expected = EXPECTED_COUNTS[mode]
    if len(a_ids) != expected or len(b_ids) != expected:
        sys.exit("incomplete %s arm: %s has %d ids, %s has %d; expected %d"
                 % (mode, arm, len(a_ids), ref, len(b_ids), expected))
    frozen_ids = frozen_prompt_ids(mode)
    if a_ids != frozen_ids or b_ids != frozen_ids:
        sys.exit("prompt-id mismatch from frozen %s corpus for %s vs %s"
                 % (mode, arm, ref))
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
    validate_provenance(dir_, arm)
    validate_provenance(dir_, ref)
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

    arm_names = list(dict.fromkeys([args.ref, args.floor] + args.arms))
    runtimes = {name: validate_provenance(args.dir, name)[0] for name in arm_names}
    reference_runtime = runtimes[args.ref]
    mismatched = [name for name in arm_names if runtimes[name] != reference_runtime]
    if mismatched:
        sys.exit("runtime provenance mismatch vs %s: %s" %
                 (args.ref, ", ".join(mismatched)))

    print("capability spot-check: SUPPORTING AGREEMENT ONLY — "
          "cannot clear amended A6 ground-truth gate")
    print("agreement with reference '%s' (greedy, %s)"
          % (args.ref, "/".join(MODES)))
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
        verdict = ("SUPPORTING AGREEMENT READ (>= 0.50; NOT A6)"
                   if rec >= 0.50 else
                   "PARTIAL AGREEMENT TRANSFER (0 <= x < 0.50; NOT A6)"
                   if rec >= 0.0 else
                   "FAIL — worse than the pure weak tier (mirage caught)")
        print("  %-12s   capability gap recovered %.1f%% -> %s"
              % (arm, 100.0 * rec, verdict))
        fail |= rec < 0.0
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
