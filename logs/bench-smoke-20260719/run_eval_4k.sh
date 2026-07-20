#!/bin/bash
# Capability suite re-run at max_tokens 4096 (confound-controlled, 2026-07-20).
# One resident model at a time; run_arm.sh (patched) records stop_reason +
# thinking and the scorer drops truncated rows. Output -> logs/eval-census-4k.
set -u -o pipefail
cd "$(dirname "$0")/../.."
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
EVAL=logs/eval-census-4k
LOG=logs/bench-smoke-20260719
mkdir -p "$EVAL"
export Q27_EVAL_MAX_TOKENS=4096

idle_s() { ioreg -c IOHIDSystem 2>/dev/null | awk '/HIDIdleTime/{printf "%d",$NF/1000000000; exit}'; }
wait_idle() { while :; do local i; i=$(idle_s); [ -n "$i" ] && [ "$i" -ge 120 ] && return 0; sleep 15; done; }
srv_pid() { ps -Ao pid,args | awk '$2 ~ /(^|\/)q27-metal-server$/ && $0 ~ /--port 8213/ {print $1}' | head -1; }
wait_free() { while pgrep -f "build/q27-metal " >/dev/null 2>&1 || [ -n "$(srv_pid)" ]; do sleep 10; done; sleep 15; }

boot() { # model
    wait_free
    nohup caffeinate -i nice ./build/q27-metal-server "$1" "$TOK" \
        --ctx 8192 --port 8213 --max-tokens-default 8192 \
        > "$LOG/eval4k-server.log" 2>&1 &
    for _ in $(seq 1 90); do curl -s -o /dev/null --max-time 2 http://127.0.0.1:8213/health && return 0; sleep 2; done
    echo "server never healthy" >&2; return 1
}
stop() { local p; p=$(srv_pid); [ -n "$p" ] && kill "$p" 2>/dev/null; for _ in $(seq 1 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done; wait_free; }

arm() { # name model
    wait_idle
    boot "$2" || return 1
    echo "=== arm $1 $(date '+%H:%M:%S') ==="
    tools/eval/run_arm.sh "$1" http://127.0.0.1:8213 "$2" "$EVAL" || { echo "arm $1 FAILED" >&2; stop; return 1; }
    stop
    echo "arm $1 done"
}

echo "eval-4k queue start $(date '+%H:%M:%S')"
wait_idle
arm t2-base      models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27
arm b1-base      models/binary-bonsai-27b/bonsai-27b-b1.q27
arm gdn-pair     /tmp/gdn-pair-arm.q27
arm m1-candidate models/bonsai-27b-m1/bonsai-27b-m1.q27
echo "eval-4k queue done $(date '+%H:%M:%S')"
