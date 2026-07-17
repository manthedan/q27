#!/bin/sh
# Official-tier gate backlog — 24 GB M4 ONLY (17 GiB artifact pages; nothing
# else may touch the GPU). STAGED 2026-07-16; run only on the operator's explicit go:
#   caffeinate -dims sh tools/official_tier_gates_2026-07-16.sh
#
# Backlog items covered (chronicle refs in parentheses):
#   1. Measured batched-MTP gate — the multislot Phase 1 honesty-note
#      retraction: --mtp 8 vs greedy byte-compare + spec stats (the first
#      "identical" readout was two empty error files; this one must be real).
#   2. Resident-greedy re-measure — decides whether the old Phase-4
#      observed-vs-ceiling gap was real overhead or clocks (resident entry).
#   3. Gate 0 oracle sweep on the OFFICIAL tier — cross-tier round anatomy
#      next to T2's flat ~430 ms (Q27_ORACLE_TRACE for the P0 split).
#   4. Early-EOS/cancel position-integrity gate (open codex finding: batched
#      MTP commits the accepted prefix before sinks fire) — server-based
#      post-EOS rerun byte-identity. NOTE: this gate targets the recorded bug
#      but has not yet been proven able to fail; treat a pass as weak evidence
#      until a forced-EOS negative control exists.
# NOT covered: Q4/Q8 GEMM-half variants (blocked on kernel work, not on gates).
set -x
cd "$(dirname "$0")/.." || exit 1
M=models/qwen36-27b-mtp/qwen36-27b-mtp.q27
T=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
LOG=logs/official-gates-$(date +%Y%m%d)
mkdir -p "$LOG"
PROMPT="Write a detailed, factual overview of how tidal forces shape planetary ring systems."

# Stale-binary rule (operationalized 2026-07-15): rebuild first, fatal on fail.
make build/q27-metal build/q27-metal-server > "$LOG/rebuild.log" 2>&1 || {
    echo "rebuild failed" > "$LOG/DONE"; exit 1; }

# --- 1. measured batched-MTP gate: greedy vs --mtp 8, byte-compare ---
./build/q27-metal "$M" "$T" --ctx 512 --prompt "$PROMPT" -n 48 \
    > "$LOG/greedy48.log" 2>&1
Q27_MTP_TRACE=1 ./build/q27-metal "$M" "$T" --ctx 512 --prompt "$PROMPT" -n 48 --mtp 8 \
    > "$LOG/mtp48.log" 2>&1
G=$(grep '^generated:' "$LOG/greedy48.log"); S=$(grep '^generated:' "$LOG/mtp48.log")
[ -n "$G" ] || { echo "VACUOUS: greedy leg produced no output" >> "$LOG/mtp_gate.verdict"; }
[ -n "$S" ] || { echo "VACUOUS: mtp leg produced no output" >> "$LOG/mtp_gate.verdict"; }
if [ -n "$G" ] && [ "$G" = "$S" ]; then echo "MTP GATE PASS (non-empty, byte-identical)" >> "$LOG/mtp_gate.verdict"
else echo "MTP GATE: DIVERGED or empty — inspect (chunk-vs-serial low-margin class is the known benign cause)" >> "$LOG/mtp_gate.verdict"; fi
grep speculation "$LOG/mtp48.log" >> "$LOG/mtp_gate.verdict"

# --- 2. resident-greedy re-measure (128 tokens, resident vs opt-out) ---
Q27_METAL_RESIDENT=1 ./build/q27-metal "$M" "$T" --ctx 512 --prompt "$PROMPT" -n 128 \
    > "$LOG/resident1.log" 2>&1
Q27_METAL_RESIDENT=0 ./build/q27-metal "$M" "$T" --ctx 512 --prompt "$PROMPT" -n 128 \
    > "$LOG/resident0.log" 2>&1

# --- 3. oracle sweep, official tier (round anatomy via trace at w=12) ---
for W in 4 8 12; do
  Q27_ORACLE_TRACE=1 ./build/q27-metal "$M" "$T" --ctx 512 --prompt "$PROMPT" -n 128 --oracle $W \
      > "$LOG/oracle-w$W.log" 2>&1
done

# --- 4. early-EOS rerun integrity (server; weak gate, see header) ---
./build/q27-metal-server "$M" "$T" --ctx 4096 --port 8121 > "$LOG/server.log" 2>&1 &
SRV=$!
i=0; until curl -s -o /dev/null http://127.0.0.1:8121/health; do
  i=$((i+1))
  if [ $i -gt 120 ]; then
    echo "EOS RERUN: SERVER NEVER HEALTHY (gate NOT run)" > "$LOG/eos_gate.verdict"
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; exit 1
  fi
  sleep 2; done
req() { curl -sf http://127.0.0.1:8121/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with exactly the word done and nothing else."}],"temperature":0,"top_k":1,"max_tokens":512}'; }
# Empty/failed responses would cmp as identical — the vacuous-empty class.
# Require curl success AND nonempty bodies before any comparison (codex P2).
if ! req > "$LOG/eos1.json" || ! req > "$LOG/eos2.json" \
   || ! [ -s "$LOG/eos1.json" ] || ! [ -s "$LOG/eos2.json" ]; then
  echo "EOS RERUN: REQUEST FAILED OR EMPTY (gate NOT run)" > "$LOG/eos_gate.verdict"
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; exit 1
fi
curl -s http://127.0.0.1:8121/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name the capital of France, one word."}],"temperature":0,"top_k":1,"max_tokens":64}' > "$LOG/post_eos.json"
if cmp -s "$LOG/eos1.json" "$LOG/eos2.json"; then echo "EOS RERUN: identical" > "$LOG/eos_gate.verdict"
else echo "EOS RERUN: DIVERGED — recorded bug class likely engaged" > "$LOG/eos_gate.verdict"; fi
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null

date > "$LOG/DONE"
