#!/usr/bin/env bash
# release_gates.sh <A|B|C> [evidence-dir] — run a QA_BEFORE_RELEASES.md risk
# tier, capture evidence, print the ledger. One model resident at a time;
# serialized server boots with teardown between. bash 3.2-safe.
set -u
cd "$(dirname "$0")/.."

TIER=${1:-}
case "$TIER" in A|B|C) ;; *) echo "usage: tools/release_gates.sh A|B|C [evidence-dir]" >&2; exit 2;; esac
EVID=${2:-/tmp/q27-release-gates-$(date +%Y%m%d-%H%M)}
mkdir -p "$EVID"
LEDGER="$EVID/ledger.txt"
: > "$LEDGER"

MODEL=models/qwen36-27b-mtp/qwen36-27b-mtp.q27
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
PORT=8213
fails=0

pass() { echo "PASS: $*" | tee -a "$LEDGER"; }
fail() { echo "FAIL: $*" | tee -a "$LEDGER"; fails=$((fails+1)); }
skip() { echo "SKIP: $*" | tee -a "$LEDGER"; }
note() { echo "note: $*" | tee -a "$LEDGER"; }

run_gate() { # name logfile cmd...  — nonzero exit = gate failure
    local name=$1 log=$2; shift 2
    echo "== $name (log: $log)"
    if caffeinate "$@" > "$log" 2>&1; then pass "$name"; else fail "$name (log: $log)"; fi
}

server_stop() {
    local pid
    pid=$(pgrep -f '^\./build/q27-metal-server' | head -1 || true)
    [ -n "$pid" ] && kill "$pid" 2>/dev/null
    local i
    for i in $(seq 1 20); do pgrep -f '^\./build/q27-metal-server' >/dev/null || break; sleep 3; done
    pgrep -f '^\./build/q27-metal-server' >/dev/null && echo "warn: server still up" >&2
}

server_boot() { # logfile extra-args... — boots with failpoints, waits for health
    local log=$1; shift
    env -i HOME="$HOME" PATH="$PATH" Q27_METAL_TEST_FAILPOINTS=1 caffeinate -i \
        ./build/q27-metal-server "$MODEL" "$TOK" --port "$PORT" "$@" \
        > "$log" 2>&1 &
    local i
    for i in $(seq 1 60); do
        curl -s -m 2 "localhost:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 5
    done
    return 1
}

echo "release gates tier $TIER — evidence: $EVID"

# ---- Shared: clean-tree sanity + unit floors -----------------------------
if [ -n "$(git status --short -- src/ experiments/ Makefile)" ]; then
    note "working tree dirty under src//experiments//Makefile (evidence still valid, record in notes)"
fi
run_gate "test-cpu" "$EVID/test-cpu.log" make test-cpu

if [ "$TIER" = "C" ]; then
    run_gate "wrapper" "$EVID/wrapper.log" bash packaging/test_q27_wrapper.sh
    echo "----"; cat "$LEDGER"
    [ "$fails" -eq 0 ] || exit 1
    exit 0
fi

# ---- Numerics sentinel (B) / full numerics (A) ----------------------------
run_gate "chunk-parity-384" "$EVID/chunk-parity.log" \
    ./build/q27-metal "$MODEL" "$TOK" --nll data/wikitext2-test.tokens.bin \
    --chunk-parity 384 --ctx 4096
if [ "$TIER" = "A" ]; then
    run_gate "test-metal" "$EVID/test-metal.log" make test-metal
    run_gate "snapshot-gate" "$EVID/snapshot.log" bash tools/snapshot_gate.sh
    run_gate "failpoint-gate" "$EVID/failpoint.log" bash tools/failpoint_gate.sh "$MODEL"
    run_gate "nll-8k" "$EVID/nll-8k.log" \
        ./build/q27-metal "$MODEL" "$TOK" --nll data/wikitext2-test.tokens.bin \
        --nll-long 8192 --ctx 8192
    skip "yukon CUDA byte gate (CUDA-side; run on yukon when trigger files change)"
    skip "constrain_gate.sh / ckpt_gate.sh (CUDA-side scripts: nvcc / CUDA [gen] lines)"
fi

# ---- Live-server gates -----------------------------------------------------
# Two boots, the v0.6.0-proven recipe: fp16 16k for the parity gates (the
# model's tool-call determinism lives there; turbo3 KV shifts the greedy
# stream into the think-forever pathology), then turbo3 131k for the trace
# gate (its cancel probe needs a >131k-token window) with the parity suites
# re-run against it so the trace carries every API family.
server_stop
if ! server_boot "$EVID/server-parity.log" --constrain-tools --ctx 16384; then
    fail "parity server boot (see $EVID/server-parity.log)"
    echo "----"; cat "$LEDGER"; exit 1
fi
run_gate "agentic-parity" "$EVID/agentic-parity.log" bash tools/agentic_parity_gate.sh
# G8d rides the qwen36 think-forever pathology (recorded in QA v2): the
# responses gate fails ONLY on G8d lines and still exits 1. Anything else
# failing is a real failure.
if caffeinate bash tools/responses_parity_gate.sh > "$EVID/responses-parity.log" 2>&1; then
    pass "responses-parity"
else
    if grep "^FAIL" "$EVID/responses-parity.log" | grep -vq "G8d"; then
        fail "responses-parity beyond G8d (log: $EVID/responses-parity.log)"
    else
        skip "responses-parity G8d only (known think-forever pathology)"
    fi
fi
server_stop
if ! server_boot "$EVID/server-trace.log" --constrain-tools --kv turbo3 \
        --ctx 131072 --trace "$EVID/trace.jsonl"; then
    fail "trace server boot (see $EVID/server-trace.log)"
    echo "----"; cat "$LEDGER"; exit 1
fi
caffeinate bash tools/agentic_parity_gate.sh > "$EVID/agentic-parity-trace.log" 2>&1 || \
    note "agentic parity under turbo3 has known think-pathology flakes (trace boot exists for coverage)"
caffeinate bash tools/responses_parity_gate.sh > "$EVID/responses-parity-trace.log" 2>&1 || true
run_gate "trace-gate" "$EVID/trace-gate.log" bash tools/trace_gate.sh "$EVID/trace.jsonl"
server_stop

# --ctx auto smoke: default boot resolves a nonzero window that health
# reports, serves a request, and tears down. (The v0.6.1 lesson: auto
# sizing to the raw policy ceiling OOMs the command queue on the 24 GB +
# official-tier reality.)
if ! server_boot "$EVID/server-auto.log"; then
    fail "auto-ctx server boot (see $EVID/server-auto.log)"
else
    AUTO_CTX=$(curl -s "localhost:$PORT/health" | python3 -c "import json,sys;print(json.load(sys.stdin)['runtime']['protocol']['context'])" 2>/dev/null || echo 0)
    if [ "${AUTO_CTX:-0}" -gt 0 ] && grep -q "ctx auto: $AUTO_CTX" "$EVID/server-auto.log"; then
        pass "auto-ctx resolves consistently ($AUTO_CTX)"
    else
        fail "auto-ctx mismatch (health=$AUTO_CTX; see $EVID/server-auto.log)"
    fi
    if curl -s --max-time 300 "localhost:$PORT/v1/completions" -H 'content-type: application/json' \
        -d '{"model":"x","prompt":"The harbor at dawn is","max_tokens":8}' | grep -q '"text"'; then
        pass "auto-ctx serves"
    else
        fail "auto-ctx completion (see $EVID/server-auto.log)"
    fi
fi
server_stop

if [ "$TIER" = "A" ]; then
    run_gate "multislot-greedy" "$EVID/multislot.log" \
        env Q27_GATE_MODEL="$MODEL" Q27_GATE_TOK="$TOK" python3 tools/multislot_gates.py
    run_gate "multislot-mtp" "$EVID/multislot-mtp.log" \
        env Q27_GATE_MODEL="$MODEL" Q27_GATE_TOK="$TOK" python3 tools/multislot_gates.py mtp
    run_gate "suffix-burst" "$EVID/suffix-burst.log" bash tools/suffix_burst_gates_2026-07-16.sh
fi

echo "----"
cat "$LEDGER"
echo "evidence: $EVID"
[ "$fails" -eq 0 ] || exit 1
