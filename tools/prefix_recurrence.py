#!/usr/bin/env python3
# Prefix-recurrence economics for the --trace stream (I1 design check,
# docs/metal/plans/2026-07-17-i1-strip-design-check.md): the proactive strip /
# re-materialization lever is worth building only if real serving traffic
# recurs to the same prompt families with short gaps. This prices it.
#
# Family key: first 4 KB of the rendered prompt (system prompts lead every
# agentic body; the 2026-07-17 T2 finding showed pi's is session-independent).
# Savings are then capped by the exact common prefix in each request's bounded
# token_head trace field; family membership alone never credits a suffix.
# A "warming save" is credited when a request's family was seen within
# --gap seconds AND the previous occurrence paid prefill work — an idle-slot
# rebuild after that paid suffix could have made the later request warm. Partial
# memory/disk hits count only their uncovered suffix. Reported as tokens/seconds
# saved vs the auto-snapshot baseline (which
# only helps AFTER a ≥4096-token bank exists).
#
#   usage: tools/prefix_recurrence.py [trace.jsonl] [--gap 900] [--since days]
import json, sys, argparse, hashlib, time, os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("trace", nargs="?", default=os.path.expanduser("~/.q27/trace.jsonl"))
ap.add_argument("--gap", type=int, default=900, help="max seconds between family recurrences to credit warming")
ap.add_argument("--since", type=float, default=7.0, help="ignore events older than this many days")
a = ap.parse_args()

reqs = {}
prefix = {}   # (boot, id) -> (tier, hit)
completed, cancelled, errored = set(), set(), set()
saves = 0
boot = 0
GENERATIVE_APIS = {"completions", "chat", "messages", "responses"}
cutoff = time.time() - a.since * 86400
for line in open(a.trace):
    try: d = json.loads(line)
    except Exception: continue
    k = d.get("kind")
    # IDs restart at every boot. Advance this scope even for old events before
    # applying the time cutoff, or a window boundary can collapse two boots.
    if k == "boot":
        boot += 1
        continue
    if d.get("ts", 0) < cutoff: continue
    rid = d.get("id")
    key = (boot, rid)
    if k == "request" and d.get("api") in GENERATIVE_APIS and rid:
        r = d.get("rendered", "")
        head = r if isinstance(r, str) else r.get("head", "")
        fam = hashlib.sha1(head[:4096].encode()).hexdigest()[:16]
        token_head = d.get("token_head")
        if not isinstance(token_head, list) or not all(isinstance(x, int) for x in token_head):
            token_head = []
        reqs[key] = (fam, d.get("prompt_tokens", 0), d.get("ts", 0),
                     d.get("api"), tuple(token_head), head)
    elif k == "prefix" and rid:
        prefix[key] = (d.get("tier"), d.get("hit", 0))
    elif k == "outcome" and rid:
        completed.add(key)
    elif k == "cancel" and rid:
        cancelled.add(key)
    elif k == "error" and rid:
        errored.add(key)
    elif k in ("snapshot_save", "snapshot_bank"):
        saves += 1

fam_hist = defaultdict(list)   # fam -> [(ts, prompt_tokens, tier, hit, token_head, rendered)]
for key, (fam, ptok, ts, api, token_head, rendered) in reqs.items():
    # Queue/generation cancellations and errored/dangling requests are not a
    # completed prefill sample. Exclude them rather than charging a full miss;
    # a future partial-prefill metric can admit only measured completed work.
    if key not in completed or key in cancelled or key in errored:
        continue
    tier, hit = prefix.get(key, ("?", 0))
    fam_hist[fam].append((ts, ptok, tier, hit, token_head, rendered))

n_req = sum(map(len, fam_hist.values()))
n_fam = len(fam_hist)
recur_reqs = sum(len(v) - 1 for v in fam_hist.values())
warmable_tokens = 0
paid_prefill_tokens = 0
rows = []
for fam, hist in sorted(fam_hist.items(), key=lambda kv: -len(kv[1])):
    hist.sort()
    fam_warm = 0
    prev_ts = None
    prev_paid = 0
    prev_ptok = 0
    prev_tokens = ()
    prev_rendered = None
    for ts, ptok, tier, hit, token_head, rendered in hist:
        # Every request pays only its uncovered suffix. This includes partial
        # memory/disk hits; treating every non-cold tier as free materially
        # understates both baseline work and the possible warming save.
        hit = min(max(0, hit), ptok)
        paid = ptok - hit
        paid_prefill_tokens += paid
        # Same-family means only a shared 4 KB header. Credit no unique suffix:
        # bound warming by the exact common token head captured by the server.
        # For legacy traces, identical complete rendered prompts are the only
        # safe fallback; all other missing-token-head pairs receive zero.
        shared = 0
        if prev_tokens and token_head:
            shared = next((i for i, (x, y) in enumerate(zip(prev_tokens, token_head))
                           if x != y), min(len(prev_tokens), len(token_head)))
        elif prev_rendered == rendered and prev_rendered is not None and prev_ptok == ptok:
            shared = ptok
        shared = min(shared, prev_ptok, ptok)
        savable = max(0, min(paid, shared - hit))
        if prev_ts is not None and ts - prev_ts <= a.gap and prev_paid > 0:
            fam_warm += savable
        prev_ts, prev_paid, prev_ptok = ts, paid, ptok
        prev_tokens, prev_rendered = token_head, rendered
    warmable_tokens += fam_warm
    rows.append((len(hist), fam, hist[0][1], fam_warm, max(0, len(hist) - 1)))

print(f"trace window: {n_req} requests across {n_fam} prompt families; {saves} snapshot save/bank events")
print(f"recurrence: {recur_reqs}/{n_req} requests ({100.0*recur_reqs/max(n_req,1):.0f}%) hit an already-seen family")
print(f"prefill paid (uncovered suffixes): {paid_prefill_tokens:,} tokens; of that, {warmable_tokens:,} ({100.0*warmable_tokens/max(paid_prefill_tokens,1):.0f}%)")
print(f"  was a within-{a.gap}s family recurrence an idle-slot warming pass would have hidden")
print(f"  (≈ {warmable_tokens/28/60:.1f} min of T2-class 28 tok/s prefill)")
print("\ntop families (count, first-prompt-tokens, warmable-tokens, repeats):")
for cnt, fam, ptok, warm, reps in rows[:10]:
    print(f"  {cnt:4d}  {fam}  first={ptok:<7,} warmable={warm:<8,} repeats={reps}")
