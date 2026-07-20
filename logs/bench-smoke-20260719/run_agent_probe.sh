#!/bin/bash
# Agentic SWE-bench tier-gap probe: serial b1 + t2, one resident model at a
# time. Usage: run_agent_probe.sh [arm] [instance_id]   (both optional)
#   no args           -> full probe: b1 all, then t2 all
#   arm only          -> that arm, all instances
#   arm + instance_id -> single smoke instance
set -u -o pipefail
cd "$(dirname "$0")/../.."
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
MAN=bench/swebench/manifest.hard.json
WORK=${AGENT_WORK:-/tmp/swebench-work}
LOG=logs/bench-smoke-20260719
mkdir -p "$WORK" "$LOG"
export Q27_EVAL_MAX_TOKENS=8192

ARM_MODEL_b1="models/binary-bonsai-27b/bonsai-27b-b1.q27"
ARM_MODEL_t2="models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27"

idle_s() { ioreg -c IOHIDSystem 2>/dev/null | awk '/HIDIdleTime/{printf "%d",$NF/1000000000; exit}'; }
wait_idle() { while :; do local i; i=$(idle_s); [ -n "$i" ] && [ "$i" -ge 90 ] && return 0; sleep 15; done; }
srv_pid() { ps -Ao pid,args | awk '$2 ~ /(^|\/)q27-metal-server$/ && $0 ~ /--port 8213/ {print $1}' | head -1; }
wait_free() { while pgrep -f "build/q27-metal " >/dev/null 2>&1 || [ -n "$(srv_pid)" ]; do sleep 10; done; sleep 10; }

boot() { wait_free
    nohup caffeinate -i nice ./build/q27-metal-server "$1" "$TOK" \
        --ctx 8192 --port 8213 --max-tokens-default 8192 \
        > "$LOG/agent-server.log" 2>&1 &
    for _ in $(seq 1 90); do curl -s -o /dev/null --max-time 2 http://127.0.0.1:8213/health && return 0; sleep 2; done
    echo "server never healthy" >&2; return 1; }
stop() { local p; p=$(srv_pid); [ -n "$p" ] && kill "$p" 2>/dev/null; for _ in $(seq 1 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done; wait_free; }

arm() { # name model [only]
    wait_idle
    boot "$2" || return 1
    echo "=== arm $1 $(date '+%H:%M:%S') ==="
    tools/eval/agent_prep.sh "$MAN" "$WORK" ${3:-} || { echo "prep failed" >&2; stop; return 1; }
    python3 tools/eval/agent_runner.py --manifest "$MAN" --work "$WORK" \
        --out "logs/agent-probe/$1.jsonl" --base-url http://127.0.0.1:8213 \
        --max-turns 20 --max-tokens 8192 ${3:+--only "$3"} || true
    stop
    echo "arm $1 done $(date '+%H:%M:%S')"
}

mkdir -p logs/agent-probe
ARM=${1:-}; ONLY=${2:-}
wait_idle
if [ -z "$ARM" ]; then
    arm b1 "$ARM_MODEL_b1"
    arm t2 "$ARM_MODEL_t2"
else
    case "$ARM" in b1) arm b1 "$ARM_MODEL_b1" "$ONLY";; t2) arm t2 "$ARM_MODEL_t2" "$ONLY";; *) echo "unknown arm $ARM" >&2; exit 2;; esac
fi
echo "agent probe done $(date '+%H:%M:%S')"
