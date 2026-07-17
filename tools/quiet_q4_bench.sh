#!/bin/bash
# Quiet-machine ship-line bench for the Q4/Q8 chunk-GEMM schedule port
# (pre-registered in docs/plans/2026-07-17-t2-prefill-throughput.md):
#   "a watcher waits for >=10 min input idle, stops the T2 server, runs
#    float/half at 8352 (x2) and 960, restarts the server. >=1.7x ships
#    default-ON; <1.7x ships nothing but the parked kernels + the honest
#    table."
# Ship line: official-tier prefill bench (8352/96, quiet) >= 1.7x current
# (>= 39 tok/s vs the 23.2 tok/s quiet baseline of 2026-07-17 night).
# Hold file: while /tmp/q27-quiet-bench.hold exists the watcher waits
# (other work owns the server). Logs + verdict: logs/q4port-20260717/.
set -u
cd "$(dirname "$0")/.."
LOGDIR=logs/q4port-20260717
VERDICT="$LOGDIR/quiet_bench.verdict"
HOLD=/tmp/q27-quiet-bench.hold
IDLE_NEED=600          # seconds of input idle before firing
PORT=8213
SERVER_LOG="$LOGDIR/t2-server-quiet-restart.log"

idle_s() { ioreg -c IOHIDSystem 2>/dev/null | awk '/HIDIdleTime/{printf "%d",$NF/1000000000; exit}'; }

# The server pid must match the SERVER process, not the caffeinate wrapper
# (whose command line also contains the pattern): match argv[0] exactly.
# Killing the wrapper orphans the 17 GB server — the 2026-07-17 OOM crashes.
server_pid() { ps -Ao pid,args | awk -v p="./build/q27-metal-server" -v port="$PORT" '$2==p && $0 ~ ("--port " port) {print $1}' | head -1; }

start_server() {
    env Q27_METAL_SNAPSHOT_DIR=/Users/macthedan/.q27/snapshots \
        Q27_METAL_MAX_TOKENS_DEFAULT=16384 \
        nohup caffeinate -i nice ./build/q27-metal-server \
        models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27 \
        models/qwen36-27b-mtp/qwen36-27b-mtp.tok \
        --ctx 131072 --port $PORT --suffix 32 > "$SERVER_LOG" 2>&1 &
    for _ in $(seq 1 60); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && return 0
        sleep 2
    done
    return 1
}

rate() { grep -oE '[0-9]+\.[0-9]+ tok/s' "$1" | tail -1 | grep -oE '^[0-9.]+'; }

run_leg() { # name env prompt log
    local name="$1" half="$2" prompt="$3" log="$LOGDIR/quiet_$1.log"
    Q27_METAL_GEMM_HALF=$half ./build/metal_prefill_bench --dtype q4q8 --prompt "$prompt" > "$log" 2>&1
    rate "$log"
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
echo "quiet window detected $(date '+%H:%M:%S') — stopping T2 server"
PID=$(server_pid)
[ -n "$PID" ] && kill "$PID" && for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
sleep 15   # GPU + memory drain: one model resident at a time

F1=$(run_leg float8352_a 0 8352); H1=$(run_leg half8352_a 1 8352)
F2=$(run_leg float8352_b 0 8352); H2=$(run_leg half8352_b 1 8352)
F9=$(run_leg float960 0 960);     H9=$(run_leg half960 1 960)

echo "quiet_q4_bench: benches done $(date '+%H:%M:%S') — restarting T2 server"
start_server && echo "server healthy" || echo "SERVER RESTART FAILED — see $SERVER_LOG"

python3 - "$F1" "$F2" "$H1" "$H2" "$F9" "$H9" <<'EOF' | tee "$VERDICT"
import sys
f1,f2,h1,h2,f9,h9 = (float(x) if x not in ('','None') else float('nan') for x in sys.argv[1:7])
fmax = max(f1,f2); hmin = min(h1,h2)
ratio = hmin/fmax if fmax==fmax and fmax>0 else float('nan')
print(f"quiet ship-line bench (8352/96, dtype q4q8, machine idle>=10min)")
print(f"  float 8352 tok/s: {f1} {f2}   (max {fmax:.2f}; quiet baseline 23.2)")
print(f"  half  8352 tok/s: {h1} {h2}   (min {hmin:.2f})")
print(f"  float 960  tok/s: {f9}        half 960 tok/s: {h9}")
print(f"  ratio min(half)/max(float) = {ratio:.3f}  (ship line >= 1.7, i.e. >= 39 tok/s)")
ship = ratio==ratio and ratio >= 1.7 and hmin >= 39.0
suspect = not (18.0 <= fmax <= 30.0)   # float baseline failed to reproduce
if suspect:
    print("VERDICT: SUSPECT — float baseline failed to reproduce the quiet 23.2 tok/s;"
          " treat as contaminated, delete this verdict to re-arm the watcher")
elif ship:
    print("VERDICT: SHIP — q27_matmul_q4_mm_h stays default-ON (Q27_METAL_GEMM_HALF=1)")
else:
    print("VERDICT: PARK — <1.7x; route Q4 back to the float-staged kernel, keep the"
          " parked half kernels + this honest table per the pre-registration")
EOF
echo "quiet_q4_bench: done $(date '+%H:%M:%S')"
