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

# completions smoke so every endpoint family has request+outcome events
curl -s -H "Content-Type: application/json" --max-time 120 "$BASE/v1/completions" \
    -d '{"model":"q27-metal","max_tokens":16,"prompt":"The capital of France is"}' >/dev/null
sleep 1   # trace flushes per event, but stay gracious on a loaded box

[ -s "$TRACE" ] || { fail "trace file missing/empty: $TRACE"; exit 1; }

python3 - "$TRACE" <<'EOF'
import json,sys
path=sys.argv[1]
kinds=[]
apis=set(); outcomes=set(); errors=[]; prefixes=0; boots=0
bad=0; prev_tms=-1; monotonic=True
with open(path) as f:
    for ln,line in enumerate(f,1):
        line=line.strip()
        if not line: continue
        try: d=json.loads(line)
        except Exception: bad+=1; continue
        k=d.get("kind"); kinds.append(k)
        if k=="boot": boots+=1
        if k=="request": apis.add(d.get("api"))
        if k=="outcome": outcomes.add(d.get("api"))
        if k=="error": errors.append((d.get("status"),d.get("type")))
        if k=="prefix": prefixes+=1
        t=d.get("tms",-1)
        if k=="boot": prev_tms=-1   # append-across-restarts: tms resets per process
        if t<prev_tms: monotonic=False
        prev_tms=t
def report(ok,msg): print(("PASS" if ok else "FAIL")+": "+msg); sys.exit(0 if False else 0) or (not ok and sys.stdout.write("")) or ok
fails=[]
if bad: fails.append(f"{bad} unparseable JSONL line(s)")
if boots<1: fails.append("no boot event")
need={"completions","chat","messages","count_tokens","responses"}
if not need<=apis: fails.append(f"request events missing apis: {sorted(need-apis)}")
if not {"completions","chat","messages","responses"}<=outcomes: fails.append(f"outcome events missing apis: {sorted({'completions','chat','messages','responses'}-outcomes)}")
if not any(s==400 for s,_ in errors): fails.append("no 400-class error event (G9 controls should have produced them)")
if prefixes<1: fails.append("no prefix decision events")
if not monotonic: fails.append("tms not monotonic")
print(f"trace-kinds: {sorted(set(k for k in kinds if k))}")
print(f"trace-apis: requests={sorted(apis)} outcomes={sorted(outcomes)} errors={len(errors)}")
if fails:
    for m in fails: print("FAIL: "+m)
    sys.exit(1)
print("PASS: trace stream complete (boot, 5 api families, outcomes, 400s, prefix decisions, monotonic tms)")
EOF
RC=$?
[ $RC -eq 0 ] && pass "trace gate: ALL PASS" || fail "trace gate: failures above"
exit $FAIL
