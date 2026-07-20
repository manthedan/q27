#!/bin/bash
# Master GPU queue (one resident model at a time, quiet machine, 2026-07-19).
# Two phases in one slot, at the other agent's request:
#   A. Cross-tier oracle sweep + resident-greedy (b1, official)  [standing residue]
#   B. Task-level capability suite, 4 arms (t2,b1,gdn,m1)        [fb10879 pre-reg]
# The one-model rule is firm: every phase tears down before the next boots.
set -u -o pipefail
cd "$(dirname "$0")/../.."
LOG=logs/bench-smoke-20260719
EVAL=logs/eval-census
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
PROMPT="Write a detailed, factual overview of how tidal forces shape planetary ring systems."
IDLE_NEED=120
mkdir -p "$EVAL"

idle_s() { ioreg -c IOHIDSystem 2>/dev/null | awk '/HIDIdleTime/{printf "%d",$NF/1000000000; exit}'; }
wait_idle() { while :; do local i; i=$(idle_s); [ -n "$i" ] && [ "$i" -ge "$IDLE_NEED" ] && return 0; sleep 15; done; }
resident() { pgrep -f "build/q27-metal " | grep -v grep || true; }
srv_pid() { ps -Ao pid,args | awk '$2 ~ /(^|\/)q27-metal-server$/ && $0 ~ /--port 8213/ {print $1}' | head -1; }
wait_free() { while [ -n "$(resident)" ] || [ -n "$(srv_pid)" ]; do sleep 10; done; sleep 15; }

# ---------- Phase A: cross-tier sweep ----------
sweep_tier() { # name model
    local name="$1" model="$2"
    wait_idle; wait_free
    echo "=== sweep $name $(date '+%H:%M:%S') ==="
    for w in 12 16 32 48; do
        Q27_ORACLE_TRACE=1 caffeinate -dims ./build/q27-metal "$model" "$TOK" \
            --ctx 512 --prompt "$PROMPT" -n 128 --oracle "$w" \
            > "$LOG/sweep-${name}-oracle-w${w}.log" 2>&1
    done
    caffeinate -dims ./build/q27-metal "$model" "$TOK" \
        --ctx 512 --prompt "$PROMPT" -n 128 \
        > "$LOG/sweep-${name}-greedy.log" 2>&1
    echo "sweep $name done"
}

# ---------- Phase B: 4-arm capability suite ----------
boot_eval_server() { # model
    local model="$1"
    wait_free
    nohup caffeinate -i nice ./build/q27-metal-server "$model" "$TOK" \
        --ctx 8192 --port 8213 --max-tokens-default 8192 \
        > "$LOG/eval-server.log" 2>&1 &
    local i
    for i in $(seq 1 90); do
        curl -s -o /dev/null --max-time 2 http://127.0.0.1:8213/health && return 0
        sleep 2
    done
    echo "eval server never healthy" >&2; return 1
}
stop_eval_server() {
    local p; p=$(srv_pid)
    [ -n "$p" ] && kill "$p" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    wait_free
}
run_arm() { # arm model
    local arm="$1" model="$2"
    wait_idle
    boot_eval_server "$model" || return 1
    echo "=== arm $arm $(date '+%H:%M:%S') ==="
    tools/eval/run_arm.sh "$arm" http://127.0.0.1:8213 "$model" "$EVAL" || { echo "arm $arm FAILED" >&2; stop_eval_server; return 1; }
    stop_eval_server
    echo "arm $arm done"
}

echo "GPU queue start $(date '+%H:%M:%S')"
wait_idle

# Phase A
sweep_tier b1       models/binary-bonsai-27b/bonsai-27b-b1.q27
sweep_tier official models/qwen36-27b-mtp/qwen36-27b-mtp.q27

# Phase B
run_arm t2-base      models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27
run_arm b1-base      models/binary-bonsai-27b/bonsai-27b-b1.q27
run_arm gdn-pair     /tmp/gdn-pair-arm.q27
run_arm m1-candidate models/bonsai-27b-m1/bonsai-27b-m1.q27

echo "GPU queue done $(date '+%H:%M:%S')"
ls "$EVAL"
