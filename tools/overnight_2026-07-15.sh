#!/bin/sh
# Overnight batch 2026-07-15 (mac-mini, GPU-serialized, run under caffeinate):
#   caffeinate -dims sh tools/overnight_2026-07-15.sh
# 1. 32K needle full pass (task #9) — also the first real-workload pass over
#    the R1b tiled causal GQA route at depth.
# 2. 16K turbo3 NLL wall A/B: tile 2 (default) vs tile 1 — the R1b wall gate.
#    Phase-0 projection: ~-19% total wall; NLL must be bit-identical (the
#    tiled kernels are bit-identical per token).
set -x
cd "$(dirname "$0")/.." || exit 1
M=models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27
T=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
LOG=logs/overnight-2026-07-15
mkdir -p "$LOG"

# --- 1. needle full pass ---
./build/q27-metal-server "$M" "$T" --ctx 32768 --kv turbo3 --port 8117 \
    > "$LOG/needle_server.log" 2>&1 &
SERVER=$!
i=0
until curl -s -o /dev/null "http://127.0.0.1:8117/v1/models" 2>/dev/null; do
    i=$((i+1)); [ $i -gt 120 ] && { echo "server never came up" >> "$LOG/needle.log"; break; }
    sleep 5
done
python3 tools/needle_32k.py http://127.0.0.1:8117 > "$LOG/needle.log" 2>&1
echo "needle exit: $?" >> "$LOG/needle.log"
kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null

# --- 2. 16K NLL wall A/B (tiled arm first) ---
./build/q27-metal "$M" "$T" --nll data/wikitext2-test.tokens.bin \
    --nll-long 16384 --ctx 16384 --kv turbo3 > "$LOG/nll16k_tile2.log" 2>&1
echo "tile2 exit: $?" >> "$LOG/nll16k_tile2.log"

Q27_METAL_GQA_TILE=1 ./build/q27-metal "$M" "$T" --nll data/wikitext2-test.tokens.bin \
    --nll-long 16384 --ctx 16384 --kv turbo3 > "$LOG/nll16k_tile1.log" 2>&1
echo "tile1 exit: $?" >> "$LOG/nll16k_tile1.log"

date > "$LOG/DONE"
