#!/bin/bash
# Gate for --trace whole-session logging (triage I2, ds4-product-triage).
# Run AFTER tools/agentic_parity_gate.sh + tools/responses_parity_gate.sh
# against a server started with --trace <path>: this script adds one
# /v1/completions smoke (the suites don't cover it) and then asserts the
# JSONL stream recorded the session.
#   usage: tools/trace_gate.sh <trace-path> [base-url]
set -u
TRACE="${1:?usage: tools/trace_gate.sh <trace-path> [base-url]}"
BASE="${2:-http://127.0.0.1:8213}"
FAIL=0
note() { printf '%s\n' "$*"; }
pass() { note "PASS: $*"; }
fail() { note "FAIL: $*"; FAIL=1; }

HEALTH=$(curl -s --max-time 10 "$BASE/health")
HEALTH_ROW=$(printf '%s' "$HEALTH" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(str(d.get("runtime",{}).get("protocol",{}).get("test_failpoints",False))+"\t"+d.get("boot_id",""))')
IFS=$'\t' read -r FP HEALTH_BOOT <<<"$HEALTH_ROW"
[ "$FP" = "True" ] && [ -n "$HEALTH_BOOT" ] || { fail "server lacks test failpoints or boot identity"; exit 1; }
TRACE_OFFSET=0
[ -e "$TRACE" ] && TRACE_OFFSET=$(wc -c < "$TRACE" | tr -d ' ')

# completions smoke so every endpoint family has request+outcome events
curl -s -H "Content-Type: application/json" --max-time 120 "$BASE/v1/completions" \
    -d '{"model":"q27-metal","max_tokens":16,"prompt":"The capital of France is"}' >/dev/null

# Deterministic diagnostic legs require the release-trace server to have been
# started with Q27_METAL_TEST_FAILPOINTS=1 (recorded in /health runtime config).
RM=$(curl -s -H "Content-Type: application/json" --max-time 30 "$BASE/v1/responses" \
    -d '{"model":"q27-metal","max_output_tokens":16,"input":"trace malformed recovery", "stream":true,"q27_test_malformed_wrapper":true}')
RM_ID=$(printf '%s' "$RM" | python3 -c 'import json,sys
for l in sys.stdin:
    if l.startswith("data: "):
        try:
            d=json.loads(l[6:])
            if d.get("type")=="response.created": print(d.get("response",{}).get("id","")); break
        except: pass')
printf '%s' "$RM" | python3 -c '
import json,sys
e=[]
for l in sys.stdin:
    if l.startswith("data: "):
        try:e.append(json.loads(l[6:]))
        except:pass
adds={(x.get("output_index"),x.get("item",{}).get("id")) for x in e if x.get("type")=="response.output_item.added"}
dones=[x for x in e if x.get("type")=="response.output_item.done"]
ok=bool(dones) and all((x.get("output_index"),x.get("item",{}).get("id")) in adds for x in dones)
raise SystemExit(0 if ok and any(x.get("type")=="response.completed" for x in e) else 1)' \
    || fail "forced malformed-wrapper lifecycle did not complete with paired items"

RE=$(curl -s -H "Content-Type: application/json" --max-time 30 "$BASE/v1/responses" \
    -d '{"model":"q27-metal","max_output_tokens":16,"input":"trace engine failure", "stream":true,"q27_test_engine_error":true}')
RE_ID=$(printf '%s' "$RE" | python3 -c 'import json,sys
for l in sys.stdin:
    if l.startswith("data: "):
        try:
            d=json.loads(l[6:])
            if d.get("type")=="response.created": print(d.get("response",{}).get("id","")); break
        except: pass')
printf '%s' "$RE" | python3 -c '
import json,sys
e=[]
for l in sys.stdin:
    if l.startswith("data: "):
        try:e.append(json.loads(l[6:]))
        except:pass
failed=next((x.get("response",{}) for x in e if x.get("type")=="response.failed"),{})
items=failed.get("output",[])
text="".join(c.get("text","") for i in items if i.get("type")=="message" for c in i.get("content",[]))
ok=failed.get("status")=="failed" and "partial-before-failure <thi" in text and any(i.get("status")=="incomplete" for i in items)
raise SystemExit(0 if ok else 1)' || fail "forced engine error lost/inaccurately completed partial output"

CANCEL_MARK="q27-trace-cancel-$$-$RANDOM"
CANCEL_BODY=$(python3 -c 'import json,sys; print(json.dumps({"model":"q27-metal","max_tokens":32,"prompt":sys.argv[1]+" "+"cancel prefill "*20000,"stream":True}))' "$CANCEL_MARK")
# Load-dependent race (2026-07-18): a single 50 ms abort can fire before a
# busy server begins prefill, so no cancel registers for THIS mark. Retry
# the probe a few times with a slightly wider window, and only stop early
# once a cancel event for a request carrying THIS mark has actually landed
# in the trace. Worst case ~6 s; the binding assertion below then has a
# real cancel to find instead of racing the probe.
cancel_registered() {
    python3 - "$TRACE" "$TRACE_OFFSET" "$CANCEL_MARK" <<'PY'
import json,sys
path,offset,mark=sys.argv[1],int(sys.argv[2]),sys.argv[3]
reqs=set(); cancels=set()
try:
    with open(path,"rb") as f:
        while True:
            pos=f.tell(); raw=f.readline()
            if not raw: break
            try: d=json.loads(raw)
            except Exception: continue
            if pos<offset: continue
            rid=d.get("id")
            if d.get("kind")=="request" and rid and mark in str(d.get("rendered","")): reqs.add(rid)
            elif d.get("kind")=="cancel" and rid: cancels.add(rid)
except FileNotFoundError:
    pass
raise SystemExit(0 if (reqs & cancels) else 1)
PY
}
attempt=0
while [ $attempt -lt 3 ]; do
    curl -s --max-time 0.2 -H "Content-Type: application/json" "$BASE/v1/completions" -d "$CANCEL_BODY" >/dev/null 2>&1 || true
    sleep 2   # allow the per-chunk liveness probe and trace flush to observe close
    cancel_registered && break
    attempt=$((attempt+1))
done

[ -s "$TRACE" ] || { fail "trace file missing/empty: $TRACE"; exit 1; }
[ -n "$RM_ID" ] && [ -n "$RE_ID" ] || { fail "forced Responses legs returned no response IDs"; exit 1; }

# Bind deterministic assertions to THIS invocation and THIS server boot, not
# merely to compatible historical records in the append-only file.
python3 - "$TRACE" "$HEALTH_BOOT" "$TRACE_OFFSET" "$RM_ID" "$RE_ID" "$CANCEL_MARK" <<'EOF'
import json,sys
path,expected_boot,offset,rm_id,re_id,cancel_mark=sys.argv[1:]
offset=int(offset)
latest_boot=None; post_requests={}; post_recoveries=set(); post_errors=[]; post_cancels=set()
with open(path,"rb") as f:
    while True:
        pos=f.tell(); raw=f.readline()
        if not raw: break
        try: d=json.loads(raw)
        except Exception: continue
        if d.get("kind")=="boot": latest_boot=d.get("boot_id")
        if pos < offset: continue
        rid=d.get("id")
        if d.get("kind")=="request" and rid: post_requests[rid]=d
        elif d.get("kind")=="tool_recovery" and rid: post_recoveries.add(rid)
        elif d.get("kind")=="error" and rid: post_errors.append((d.get("status"),rid))
        elif d.get("kind")=="cancel" and rid: post_cancels.add(rid)
fails=[]
if latest_boot!=expected_boot: fails.append("trace latest boot does not match /health")
if rm_id not in post_requests or rm_id not in post_recoveries: fails.append("forced recovery ID missing after gate offset")
if re_id not in post_requests or (500,re_id) not in post_errors: fails.append("forced error ID missing after gate offset")
cancel_ids={rid for rid,d in post_requests.items() if cancel_mark in str(d.get("rendered",""))}
if not cancel_ids or not (cancel_ids & post_cancels): fails.append("forced cancellation ID missing after gate offset")
if fails:
    for x in fails: print("FAIL: "+x)
    raise SystemExit(1)
print("PASS: current invocation IDs bound to health boot and post-run trace offset")
EOF
BOUND_RC=$?
[ $BOUND_RC -eq 0 ] || fail "trace current-invocation binding failed"

python3 - "$TRACE" <<'EOF'
import json,sys
path=sys.argv[1]
kinds=[]
apis=set(); outcomes=set(); errors=[]; prefixes=0; boots=0
request_ids=set(); recoveries=[]; cancels=[]
bad=0; prev_tms=-1; monotonic=True
with open(path) as f:
    for ln,line in enumerate(f,1):
        line=line.strip()
        if not line: continue
        try: d=json.loads(line)
        except Exception: bad+=1; continue
        k=d.get("kind"); kinds.append(k)
        if k=="boot":
            boots+=1
            # Append-only traces may contain old successful boots. Every
            # release assertion below is scoped to the latest boot only.
            kinds=[k]; apis=set(); outcomes=set(); errors=[]; prefixes=0; bad=0
            request_ids=set(); recoveries=[]; cancels=[]
            prev_tms=-1; monotonic=True
        if k=="request":
            apis.add(d.get("api"))
            if d.get("id"): request_ids.add(d.get("id"))
        if k=="outcome": outcomes.add(d.get("api"))
        if k=="error": errors.append((d.get("status"),d.get("type"),d.get("id")))
        if k=="prefix": prefixes+=1
        if k=="tool_recovery": recoveries.append(d.get("id"))
        if k=="cancel": cancels.append(d.get("id"))
        t=d.get("tms",-1)
        if t<prev_tms: monotonic=False
        prev_tms=t
def report(ok,msg): print(("PASS" if ok else "FAIL")+": "+msg); sys.exit(0 if False else 0) or (not ok and sys.stdout.write("")) or ok
fails=[]
if bad: fails.append(f"{bad} unparseable JSONL line(s)")
if boots<1: fails.append("no boot event")
need={"completions","chat","messages","count_tokens","responses"}
if not need<=apis: fails.append(f"request events missing apis: {sorted(need-apis)}")
if not {"completions","chat","messages","responses"}<=outcomes: fails.append(f"outcome events missing apis: {sorted({'completions','chat','messages','responses'}-outcomes)}")
if not any(s==400 for s,_,_ in errors): fails.append("no 400-class error event (G9 controls should have produced them)")
if prefixes<1: fails.append("no prefix decision events")
if not recoveries: fails.append("no forced tool_recovery event")
elif any(not rid or rid not in request_ids for rid in recoveries):
    fails.append("tool_recovery event missing/corrupt request id")
if not cancels: fails.append("no cancellation event")
elif any(not rid or rid not in request_ids for rid in cancels):
    fails.append("cancel event missing/corrupt request id")
if not any(s==500 and rid in request_ids for s,_,rid in errors):
    fails.append("no correlated forced 500 engine-error event")
if not monotonic: fails.append("tms not monotonic")
print(f"trace-kinds: {sorted(set(k for k in kinds if k))}")
print(f"trace-apis: requests={sorted(apis)} outcomes={sorted(outcomes)} errors={len(errors)} recoveries={len(recoveries)} cancels={len(cancels)}")
if fails:
    for m in fails: print("FAIL: "+m)
    sys.exit(1)
print("PASS: trace stream complete (boot, 5 api families, outcomes, 400s, prefix decisions, monotonic tms)")
EOF
RC=$?
[ $RC -eq 0 ] && pass "trace gate: ALL PASS" || fail "trace gate: failures above"
exit $FAIL
