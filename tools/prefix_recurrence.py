#!/usr/bin/env python3
# Prefix-recurrence economics for the --trace stream (I1 design check,
# docs/plans/2026-07-17-i1-strip-design-check.md): the proactive strip /
# re-materialization lever is worth building only if real serving traffic
# recurs to the same prompt families with short gaps. This prices it.
#
# Family key: first 4 KB of the rendered prompt (system prompts lead every
# agentic body; the 2026-07-17 T2 finding showed pi's is session-independent).
# A "warming save" is credited when a request's family was seen within
# --gap seconds AND the previous occurrence paid a cold prefill — an idle-slot
# rebuild after that first cold prefill would have made the later requests
# warm. Reported as tokens/seconds saved vs the auto-snapshot baseline (which
# only helps AFTER a ≥4096-token bank exists).
#
#   usage: tools/prefix_recurrence.py [trace.jsonl] [--gap 900] [--since days]
import json, sys, argparse, hashlib, time
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("trace", nargs="?", default="/Users/macthedan/.q27/trace.jsonl")
ap.add_argument("--gap", type=int, default=900, help="max seconds between family recurrences to credit warming")
ap.add_argument("--since", type=float, default=7.0, help="ignore events older than this many days")
a = ap.parse_args()

reqs = {}
prefix = {}   # id -> (tier, hit)
saves = 0
cutoff = time.time() - a.since * 86400
for line in open(a.trace):
    try: d = json.loads(line)
    except Exception: continue
    if d.get("ts", 0) < cutoff: continue
    k = d.get("kind")
    if k == "request":
        r = d.get("rendered", "")
        head = r if isinstance(r, str) else r.get("head", "")
        fam = hashlib.sha1(head[:4096].encode()).hexdigest()[:16]
        reqs[d.get("id", f"#{len(reqs)}")] = (fam, d.get("prompt_tokens", 0), d.get("ts", 0), d.get("api"))
    elif k == "prefix":
        prefix[d.get("id")] = (d.get("tier"), d.get("hit", 0))
    elif k in ("snapshot_save", "snapshot_bank"):
        saves += 1

fam_hist = defaultdict(list)   # fam -> [(ts, prompt_tokens, tier)]
for rid, (fam, ptok, ts, api) in reqs.items():
    tier, hit = prefix.get(rid, ("?", 0))
    fam_hist[fam].append((ts, ptok, tier))

n_req = len(reqs)
n_fam = len(fam_hist)
recur_reqs = sum(len(v) - 1 for v in fam_hist.values())
warmable_tokens = 0
cold_tokens = 0
rows = []
for fam, hist in sorted(fam_hist.items(), key=lambda kv: -len(kv[1])):
    hist.sort()
    fam_warm = 0
    prev_ts = None
    for ts, ptok, tier in hist:
        cold = ptok if tier == "cold" else ptok - 0  # hit covers the rest
        if tier == "cold": cold_tokens += ptok
        if prev_ts is not None and ts - prev_ts <= a.gap and tier == "cold":
            fam_warm += ptok
        prev_ts = ts
    warmable_tokens += fam_warm
    rows.append((len(hist), fam, hist[0][1], fam_warm, max(0, len(hist) - 1)))

print(f"trace window: {n_req} requests across {n_fam} prompt families; {saves} snapshot save/bank events")
print(f"recurrence: {recur_reqs}/{n_req} requests ({100.0*recur_reqs/max(n_req,1):.0f}%) hit an already-seen family")
print(f"cold prefill paid: {cold_tokens:,} tokens; of that, {warmable_tokens:,} ({100.0*warmable_tokens/max(cold_tokens,1):.0f}%)")
print(f"  was a within-{a.gap}s family recurrence an idle-slot warming pass would have hidden")
print(f"  (≈ {warmable_tokens/28/60:.1f} min of T2-class 28 tok/s prefill)")
print("\ntop families (count, first-prompt-tokens, warmable-tokens, repeats):")
for cnt, fam, ptok, warm, reps in rows[:10]:
    print(f"  {cnt:4d}  {fam}  first={ptok:<7,} warmable={warm:<8,} repeats={reps}")
