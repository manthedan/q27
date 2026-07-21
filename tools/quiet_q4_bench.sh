#!/bin/bash
# Quiet-machine ship-line bench for the Q4/Q8 chunk-GEMM schedule port
# (pre-registered in docs/metal/plans/2026-07-17-t2-prefill-throughput.md):
#   "a watcher waits for >=10 min input idle, stops the T2 server, runs
#    float/half at 8352 (x2) and 960, restarts the server. >=1.7x ships
#    default-ON; <1.7x ships nothing but the parked kernels + the honest
#    table."
# Ship line: official-tier prefill bench (8352/96, quiet) >= 1.7x current
# (>= 39 tok/s vs the 23.2 tok/s quiet baseline of 2026-07-17 night).
# Hold file: while /tmp/q27-quiet-bench.hold exists the watcher waits
# (other work owns the server). Logs + verdict: logs/q4port-20260717/.
set -u -o pipefail
cd "$(dirname "$0")/.."
LOGDIR=logs/q4port-20260717
VERDICT="$LOGDIR/quiet_bench.verdict"
HOLD=/tmp/q27-quiet-bench.hold
IDLE_NEED=600          # seconds of input idle before firing
PORT=8213
SERVER_LOG="$LOGDIR/t2-server-quiet-restart.log"
SERVER_MODEL=models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27
SERVER_TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
TRACE_PATH=/Users/macthedan/.q27/trace.jsonl
ORIGINAL_HEALTH=
MODEL_SHA=
BENCH_SHA=
SHADER_SHA=
NEEDS_RESTART=0
SERVICE_DRAINED=0
VERDICT_TMP=

idle_s() { ioreg -c IOHIDSystem 2>/dev/null | awk '/HIDIdleTime/{printf "%d",$NF/1000000000; exit}'; }

# The server pid must match the SERVER process, not the caffeinate wrapper
# (whose command line also contains the pattern): match argv[0] exactly.
# Killing the wrapper orphans the 17 GB server — the 2026-07-17 OOM crashes.
server_pid() { ps -Ao pid,args | awk -v port="$PORT" '$2 ~ /(^|\/)q27-metal-server$/ && $0 ~ ("--port " port) {print $1}' | head -1; }
# Admin credential for drain/resume (autoreview P2: no longer the public
# boot_id). Read the token the server printed to its stderr log at startup;
# the running server's stderr is redirect-appended to a log we can read.
# We locate it via the process's own log: the canonical restart log for a
# server this script started, else the newest server log under logs/.
admin_token() {
    local tok log
    for log in "$SERVER_LOG" $(ls -t logs/server-restarts/*.log logs/*/t2-server*.log 2>/dev/null); do
        [ -f "$log" ] || continue
        tok=$(grep 'admin token (X-Q27-Admin-Token):' "$log" 2>/dev/null | tail -1 | awk '{print $NF}')
        [ -n "$tok" ] && { printf '%s' "$tok"; return 0; }
    done
    return 1
}
# Detection is broader than termination: never kill an unrelated model job,
# but refuse to load while ANY known q27 model consumer remains resident.
model_pids() { ps -Ao pid,args | awk -v me="$$" '$1!=me && $2 ~ /(^|\/)[^\/]*(q27|metal|failpoint)[^\/]*$/ {print $1" "$0}'; }

is_descendant() { # pid ancestor
    local p="$1" root="$2" pp
    while [ "$p" -gt 1 ] 2>/dev/null; do
        [ "$p" = "$root" ] && return 0
        pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ') || return 1
        [ -n "$pp" ] || return 1
        p="$pp"
    done
    return 1
}

start_server() {
    local residents code launcher p rows conflict health boot
    # The 26-minute benchmark creates a race window: repeat the one-model
    # preflight immediately before launch, not only before the first leg.
    residents=$(model_pids)
    [ -z "$residents" ] || { printf 'restart blocked by resident model process:\n%s\n' "$residents" >&2; return 1; }
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$PORT/health 2>/dev/null)
    [ "$code" != 200 ] || { echo "restart blocked: port $PORT already serves /health" >&2; return 1; }
    # Close the preflight race once more before allocating the 17 GB model.
    [ -z "$(model_pids)" ] || { echo "restart race: model process appeared during preflight" >&2; return 1; }

    # Mirror the canonical serving line (flag form since the knobs->flags
    # promotion; --trace was silently dropped by the 2026-07-17 restart —
    # the registered script debt this block closes).
    nohup caffeinate -i nice env \
        -u Q27_METAL_KV_FP16_CELLS -u Q27_METAL_KV_CELLS_CODEC \
        -u Q27_METAL_GEMM_HALF -u Q27_METAL_GEMM_HALF_Q4 \
        -u Q27_METAL_GQA_TILE -u Q27_METAL_GQA_BLOCK -u Q27_METAL_GQA_THRESHOLD \
        -u Q27_METAL_GPU_SAMPLE -u Q27_METAL_RESIDENT -u Q27_METAL_NO_RESIDENCY \
        -u Q27_METAL_PROFILE -u Q27_METAL_SOURCE -u Q27_BARE -u Q27_TOOL_STRICT \
        -u Q27_METAL_TEST_FAILPOINTS -u Q27_METAL_FAIL_FINISH -u Q27_METAL_SNAP_CRASH \
        -u Q27_MTP_TRACE -u Q27_ORACLE_TRACE -u Q27_SUFFIX_TRACE -u Q27_METAL_BUDGET_MB \
        -u Q27_METAL_SNAPSHOT_DIR -u Q27_METAL_SNAPSHOT_MAX_MB -u Q27_METAL_SNAPSHOT_AUTO \
        -u Q27_METAL_MAX_TOKENS_DEFAULT ./build/q27-metal-server \
        "$SERVER_MODEL" "$SERVER_TOK" \
        --ctx 131072 --port $PORT --suffix 32 --slots 2 --kv fp16 --prefix-entries 1 \
        --snapshot-dir /Users/macthedan/.q27/snapshots --snapshot-max-mb 8192 \
        --snapshot-auto 4096 --max-tokens-default 16384 \
        --trace "$TRACE_PATH" > "$SERVER_LOG" 2>&1 &
    launcher=$!
    for _ in $(seq 1 60); do
        rows=$(model_pids); conflict=0
        while read -r p _; do
            [ -z "$p" ] || is_descendant "$p" "$launcher" || conflict=1
        done <<< "$rows"
        if [ "$conflict" = 1 ]; then
            echo "restart race: unrelated model process appeared" >&2
            break
        fi
        p=$(server_pid)
        if [ -n "$p" ] && is_descendant "$p" "$launcher"; then
            health=$(curl -fsS --max-time 4 "http://127.0.0.1:$PORT/health?identity=1" 2>/dev/null) || health=
            boot=$(/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 - "$ORIGINAL_HEALTH" "$health" <<'PY' 2>/dev/null
import json,sys
try:
    old,new=map(json.loads,sys.argv[1:3])
    need=lambda ok: None if ok else (_ for _ in ()).throw(ValueError("identity mismatch"))
    # Restore the exact serving identity, not a permissive subset. boot_id is
    # intentionally new; every model/build/shader/tokenizer/platform/protocol
    # field and the resident artifact must otherwise match byte-for-byte.
    need(new["status"]=="ok" and new["model"]==old["model"])
    need(new["artifact_sha1"]==old["artifact_sha1"])
    need(new["runtime"]==old["runtime"])
    need(new["trace"]==old["trace"]=={"enabled":True,"healthy":True})
    need(new["serving"]==old["serving"]=={"draining":False,"active_requests":0})
    need(isinstance(new["boot_id"],str) and new["boot_id"] and new["boot_id"]!=old["boot_id"])
    print(new["boot_id"])
except Exception: raise SystemExit(1)
PY
) || boot=
            if [ -n "$boot" ]; then
                echo "restart bound pid=$p boot_id=$boot artifact_sha1=$MODEL_SHA"
                return 0
            fi
        fi
        sleep 2
    done
    p=$(server_pid)
    [ -n "$p" ] && is_descendant "$p" "$launcher" && kill "$p" 2>/dev/null
    kill "$launcher" 2>/dev/null
    return 1
}

cleanup_exit() {
    local rc=$?
    trap - EXIT HUP INT TERM
    [ -z "$VERDICT_TMP" ] || rm -f "$VERDICT_TMP"
    if [ "$SERVICE_DRAINED" = 1 ] && [ -n "$(server_pid)" ]; then
        local tok; tok=$(admin_token)
        if [ -n "$tok" ] && curl -fsS --max-time 3 -X POST -H "X-Q27-Admin-Token: $tok" \
            "http://127.0.0.1:$PORT/admin/resume" >/dev/null 2>&1; then
            NEEDS_RESTART=0
        else rc=1
        fi
    fi
    if [ "$NEEDS_RESTART" = 1 ]; then
        echo "quiet_q4_bench: restoring canonical service on exit" >&2
        start_server || { echo "FATAL: emergency serving restore failed" >&2; rc=1; }
    fi
    exit "$rc"
}

fingerprint_ok() {
    [ "$(shasum -a 1 ./build/metal_prefill_bench 2>/dev/null | awk '{print $1}')" = "$BENCH_SHA" ] &&
    [ "$(shasum -a 1 src/metal/q27_kernels.metal 2>/dev/null | awk '{print $1}')" = "$SHADER_SHA" ]
}

trace_quiet() { # boot id: no active traced request and terminal gap >=45s
    /usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 - "$TRACE_PATH" "$1" <<'PY' 2>/dev/null
import json,os,sys,time
try:
    path,boot=sys.argv[1:3]
    with open(path,"rb") as f:
        size0=os.fstat(f.fileno()).st_size
        rows=[]
        for raw in f:
            if raw.strip(): rows.append(json.loads(raw))
        size1=os.fstat(f.fileno()).st_size
    if size0 != size1: raise ValueError("trace changed while read")
    starts=[i for i,r in enumerate(rows) if r.get("kind")=="boot" and r.get("boot_id")==boot]
    if not starts: raise ValueError("current boot absent from trace")
    events=rows[starts[-1]+1:]
    active=set()
    for r in events:
        rid=r.get("id")
        if r.get("kind")=="request" and rid and r.get("api")!="count_tokens": active.add(rid)
        if r.get("kind") in ("outcome","cancel","error") and rid: active.discard(rid)
    if active or not events: raise ValueError("active/empty trace")
    last=events[-1]
    if last.get("kind") not in ("outcome","cancel","error"): raise ValueError("last event not terminal")
    if time.time()-float(last.get("ts",0)) < 45: raise ValueError("terminal gap <45s")
except Exception: raise SystemExit(1)
PY
}

rate() { grep -oE '[0-9]+\.[0-9]+ tok/s' "$1" | tail -1 | grep -oE '^[0-9.]+'; }

run_leg() { # name env prompt log
    local name="$1" half="$2" prompt="$3" log="$LOGDIR/quiet_$1.log"
    local idle0 idle1 wall0 wall1 elapsed r
    fingerprint_ok || return 1
    idle0=$(idle_s); wall0=$(date +%s)
    [ -n "$idle0" ] && [ "$idle0" -ge "$IDLE_NEED" ] || return 1
    # Q27_METAL_GEMM_HALF_Q4 since the park commit (79a2cf3) — the old
    # Q27_METAL_GEMM_HALF no longer routes the Q4 GEMM, and a re-armed rerun
    # under it would bench float-vs-float and print a fabricated ratio
    # (review 2026-07-17).
    env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin Q27_METAL_GEMM_HALF_Q4=$half \
        ./build/metal_prefill_bench --dtype q4q8 --prompt "$prompt" > "$log" 2>&1 || return 1
    fingerprint_ok || return 1
    idle1=$(idle_s); wall1=$(date +%s); elapsed=$((wall1-wall0))
    # HIDIdleTime must advance with wall time. This catches any input reset
    # during a leg, even if a very long leg later climbs above ten minutes.
    [ -n "$idle1" ] && [ $((idle1+3)) -ge $((idle0+elapsed)) ] || return 1
    r=$(rate "$log") || return 1
    [[ "$r" =~ ^[0-9]+\.[0-9]+$ ]] || return 1
    printf '%s\n' "$r"
}

echo "quiet_q4_bench: watcher up $(date '+%H:%M:%S'), waiting for idle>=${IDLE_NEED}s and no hold file"
while :; do
    [ -e "$VERDICT" ] && { echo "verdict already exists, exiting"; exit 0; }
    if [ ! -e "$HOLD" ]; then
        i=$(idle_s)
        [ -n "$i" ] && [ "$i" -ge "$IDLE_NEED" ] && break
    fi
    sleep 60
done
# Build after the unbounded wait and force every host-code dependency to be
# recompiled. Fingerprints are then held invariant around every timed leg.
make -B build/metal_prefill_bench >/dev/null || { echo "FAIL: cannot rebuild measurement binary" >&2; exit 1; }
BENCH_SHA=$(shasum -a 1 ./build/metal_prefill_bench 2>/dev/null | awk '{print $1}')
SHADER_SHA=$(shasum -a 1 src/metal/q27_kernels.metal 2>/dev/null | awk '{print $1}')
[[ "$BENCH_SHA" =~ ^[0-9a-f]{40}$ && "$SHADER_SHA" =~ ^[0-9a-f]{40}$ ]] || {
    echo "FAIL: cannot fingerprint measurement binary/shader" >&2; exit 1;
}
echo "quiet window detected $(date '+%H:%M:%S') — validating canonical T2 server"
PID=$(server_pid)
[ -n "$PID" ] || { echo "FAIL: no canonical server process to restore" >&2; exit 1; }
MODEL_SHA=$(shasum -a 1 "$SERVER_MODEL" 2>/dev/null | awk '{print $1}')
SERVER_SHA=$(shasum -a 1 ./build/q27-metal-server 2>/dev/null | awk '{print $1}')
TOK_SHA=$(shasum -a 1 "$SERVER_TOK" 2>/dev/null | awk '{print $1}')
ORIGINAL_HEALTH=$(curl -fsS --max-time 8 "http://127.0.0.1:$PORT/health?identity=1" 2>/dev/null) || ORIGINAL_HEALTH=
ORIGINAL_BOOT=$(/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 - "$MODEL_SHA" "$SERVER_SHA" "$TOK_SHA" "$ORIGINAL_HEALTH" <<'PY' 2>/dev/null
import json,re,sys
try:
    model_sha,server_sha,tok_sha=sys.argv[1:4]; h=json.loads(sys.argv[4])
    r=h["runtime"]; p=r["protocol"]
    need=lambda ok: None if ok else (_ for _ in ()).throw(ValueError("identity mismatch"))
    need(all(re.fullmatch(r"[0-9a-f]{40}",x or "") for x in (model_sha,server_sha,tok_sha)))
    need(h["status"]=="ok" and h["model"]=="ternary-bonsai-27b-t2.q27")
    need(h["artifact_sha1"]==model_sha and r["server_sha1"]==server_sha)
    need(p["tokenizer_sha1"]==tok_sha and p["tokenizer"]=="qwen36-27b-mtp.tok")
    need(r["identity_schema"]==3 and isinstance(h["boot_id"],str) and h["boot_id"])
    need(h["trace"]=={"enabled":True,"healthy":True})
    need(h["serving"]=={"draining":False,"active_requests":0})
    need(p["context"]==131072 and p["kv"]=="fp16" and p["mtp"]==0)
    need(p["suffix"]==32 and p["slots"]==2 and p["prefix_entries"]==1)
    need(p["constrain_tools"] is False and p["snapshots"] is True)
    need(p["snapshot_auto_min"]==4096 and p["snapshot_max_bytes"]==8192*1024*1024)
    need(p["max_tokens_default"]==16384 and p["kv_fp16_except"] is False)
    need(p["kv_fp16_cell_masks"]=="0"*32 and p["kv_side_codec"]=="none")
    need(p["gemm_half"] is True and p["gemm_half_q4"] is False)
    need(p["gqa_tile"]==2 and p["gqa_block"]==1024 and p["gqa_threshold"]==2048)
    need(p["gpu_sample"] is True and p["resident"] is True and p["bare_system"] is False)
    need(p["tool_strict"] is False and p["test_failpoints"] is False)
    print(h["boot_id"])
except Exception: raise SystemExit(1)
PY
) || ORIGINAL_BOOT=
[ -n "$ORIGINAL_BOOT" ] || { echo "FAIL: resident service is not the expected canonical T2 identity; refusing stop" >&2; exit 1; }
[ ! -e "$HOLD" ] || { echo "FAIL: hold file appeared; refusing server stop" >&2; exit 1; }
trace_quiet "$ORIGINAL_BOOT" || { echo "FAIL: current trace has an active/recent request; refusing server stop" >&2; exit 1; }
sleep 1
trace_quiet "$ORIGINAL_BOOT" || { echo "FAIL: trace changed or quiet gap vanished; refusing server stop" >&2; exit 1; }
# Atomically close admission before the final stop decision. Entry scopes are
# counted before parse/render/tokenization; a racing request is either counted
# to completion or rejected while draining.
trap cleanup_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
ADMIN_TOKEN=$(admin_token) || { echo "FAIL: cannot read admin token from server log" >&2; exit 1; }
curl -fsS --max-time 4 -X POST -H "X-Q27-Admin-Token: $ADMIN_TOKEN" \
    "http://127.0.0.1:$PORT/admin/drain" >/dev/null || {
    echo "FAIL: cannot drain canonical service" >&2; exit 1;
}
SERVICE_DRAINED=1
NEEDS_RESTART=1
DRAIN_READY=0
for _ in $(seq 1 90); do
    DRAIN_HEALTH=$(curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null) || DRAIN_HEALTH=
    if /usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 - "$ORIGINAL_BOOT" "$DRAIN_HEALTH" <<'PY' 2>/dev/null
import json,sys
try:
    h=json.loads(sys.argv[2])
    if h.get("boot_id")!=sys.argv[1] or h.get("serving")!={"draining":True,"active_requests":0}:
        raise ValueError("not drained")
    if h.get("trace")!={"enabled":True,"healthy":True}: raise ValueError("trace unhealthy")
except Exception: raise SystemExit(1)
PY
    then
        trace_quiet "$ORIGINAL_BOOT" && { DRAIN_READY=1; break; }
    fi
    sleep 1
done
[ "$DRAIN_READY" = 1 ] || { echo "FAIL: drain/trace quiet condition not reached" >&2; exit 1; }
FINAL_IDLE=$(idle_s)
[ -n "$FINAL_IDLE" ] && [ "$FINAL_IDLE" -ge "$IDLE_NEED" ] || {
    echo "FAIL: input activity resumed before stop" >&2; exit 1;
}
[ ! -e "$HOLD" ] || { echo "FAIL: hold file appeared before stop" >&2; exit 1; }
CURRENT_PID=$(server_pid)
MODEL_PID_SET=$(model_pids | awk '{print $1}')
[ "$CURRENT_PID" = "$PID" ] && [ "$(printf '%s\n' "$MODEL_PID_SET" | awk 'NF{n++} END{print n+0}')" = 1 ] && \
    [ "$MODEL_PID_SET" = "$PID" ] && kill -0 "$PID" 2>/dev/null || {
    echo "FAIL: model process set/PID changed before stop; refusing signal" >&2
    exit 1
}
echo "validated pid=$PID boot_id=$ORIGINAL_BOOT artifact_sha1=$MODEL_SHA trace_gap>=45s active=0 drained — stopping T2 server"
if [ -n "$PID" ]; then
    kill "$PID" || { echo "FAIL: could not signal resident server pid=$PID" >&2; exit 1; }
    for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
    kill -0 "$PID" 2>/dev/null && {
        echo "FAIL: resident server pid=$PID still alive after 30s; refusing model load" >&2
        exit 1
    }
fi
SERVICE_DRAINED=0
# Re-resolve instead of trusting the original PID (wrapper/restart races), and
# reject any healthy server our exact argv matcher failed to identify.
[ -z "$(server_pid)" ] || { echo "FAIL: a q27-metal-server still matches port $PORT" >&2; exit 1; }
RESIDENT=$(model_pids)
[ -z "$RESIDENT" ] || { printf 'FAIL: model-loading q27 process remains; refusing benchmark:\n%s\n' "$RESIDENT" >&2; exit 1; }
if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ]; then
    echo "FAIL: a server is still healthy on port $PORT; refusing model load" >&2
    exit 1
fi
sleep 15   # GPU + memory drain: one model resident at a time

bench_failed=0
F1=$(run_leg float8352_a 0 8352) || bench_failed=1
H1=$(run_leg half8352_a 1 8352) || bench_failed=1
F2=$(run_leg float8352_b 0 8352) || bench_failed=1
H2=$(run_leg half8352_b 1 8352) || bench_failed=1
F9=$(run_leg float960 0 960) || bench_failed=1
H9=$(run_leg half960 1 960) || bench_failed=1

echo "quiet_q4_bench: benches done $(date '+%H:%M:%S') — restarting T2 server"
restart_failed=0
if start_server; then
    NEEDS_RESTART=0
    echo "server healthy"
else
    echo "SERVER RESTART FAILED — see $SERVER_LOG" >&2
    restart_failed=1
fi
[ "$bench_failed" = 0 ] || {
    echo "FAIL: one or more benchmark replicas failed/unparsed; no verdict written" >&2
    exit 1
}
[ "$restart_failed" = 0 ] || {
    echo "FAIL: serving restart failed; no verdict written" >&2
    exit 1
}

VERDICT_TMP="$VERDICT.tmp.$$"
rm -f "$VERDICT_TMP"
/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 - "$F1" "$F2" "$H1" "$H2" "$F9" "$H9" "$BENCH_SHA" "$SHADER_SHA" > "$VERDICT_TMP" <<'EOF'
import math,sys
vals=[float(x) for x in sys.argv[1:7]]
if not all(math.isfinite(x) for x in vals):
    print("VERDICT: INVALID — missing/non-finite benchmark replica")
    raise SystemExit(2)
f1,f2,h1,h2,f9,h9 = vals
bench_sha,shader_sha=sys.argv[7:9]
fmax = max(f1,f2); hmin = min(h1,h2)
ratio = hmin/fmax if fmax==fmax and fmax>0 else float('nan')
print(f"quiet ship-line bench (8352/96, dtype q4q8, machine idle>=10min)")
print(f"  measurement binary sha1: {bench_sha}")
print(f"  shader source sha1: {shader_sha}")
print(f"  float 8352 tok/s: {f1} {f2}   (max {fmax:.2f}; quiet baseline 23.2)")
print(f"  half  8352 tok/s: {h1} {h2}   (min {hmin:.2f})")
print(f"  float 960  tok/s: {f9}        half 960 tok/s: {h9}")
print(f"  ratio min(half)/max(float) = {ratio:.3f}  (ship line >= 1.7, i.e. >= 39 tok/s)")
ship = ratio==ratio and ratio >= 1.7 and hmin >= 39.0
suspect = not (18.0 <= fmax <= 30.0)   # float baseline failed to reproduce
if suspect:
    print("VERDICT: SUSPECT — float baseline failed to reproduce the quiet 23.2 tok/s;"
          " run rejected and verdict file removed")
    raise SystemExit(3)
elif ship:
    print("VERDICT: SHIP — q27_matmul_q4_mm_h stays default-ON (Q27_METAL_GEMM_HALF_Q4=1)")
else:
    print("VERDICT: PARK — <1.7x; route Q4 back to the float-staged kernel, keep the"
          " parked half kernels + this honest table per the pre-registration")
EOF
verdict_rc=$?
cat "$VERDICT_TMP"
if [ "$verdict_rc" != 0 ] || [ "$(grep -Ec '^VERDICT: (SHIP|PARK)' "$VERDICT_TMP")" != 1 ]; then
    echo "FAIL: verdict computation rejected measurements" >&2
    [ "$verdict_rc" != 0 ] || verdict_rc=2
    exit "$verdict_rc"
fi
mv -f "$VERDICT_TMP" "$VERDICT"
trap - EXIT HUP INT TERM
echo "quiet_q4_bench: done $(date '+%H:%M:%S')"
