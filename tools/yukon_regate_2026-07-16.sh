#!/bin/sh
# Yukon CUDA oracle re-gate after the 2026-07-16 upstream merge (signalnine
# continuous-batching stream: engine.cuh/server.cu/conductor.h refactor).
# STAGED; run only on Daniel's explicit go — loads the model on yukon AND
# the T2/official artifact locally (GPU-serialized machine rules apply).
#
# 1. Rebuild yukon's checkout at the merged metal branch (upstream claims
#    byte-identity through the LaneView/conductor refactor; we measure).
# 2. Re-run the 16-token byte-exact canonical gate (SHA-256 6c1d4328... was
#    the pre-merge record; the gate compares live, not against the hash).
set -x
cd "$(dirname "$0")/.." || exit 1
LOG=logs/yukon-regate-$(date +%Y%m%d)
mkdir -p "$LOG"

ssh yukon 'cd ~/q27 && git fetch origin && git checkout metal && git pull --ff-only origin metal && make -j build/q27 build/q27-server' \
    > "$LOG/yukon-rebuild.log" 2>&1 || { echo "yukon rebuild failed" > "$LOG/DONE"; exit 1; }

./tools/metal_cuda_gate.py models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
    models/qwen36-27b-mtp/qwen36-27b-mtp.tok --cuda-ssh yukon -n 16 \
    > "$LOG/gate16.log" 2>&1
GATE=$?
echo "gate exit: $GATE" >> "$LOG/gate16.log"
# A failed canonical gate must fail the script — DONE alone must never be
# read as success (codex P2 on the merge review).
if [ "$GATE" -ne 0 ]; then echo "GATE FAILED ($GATE)" > "$LOG/DONE"; exit "$GATE"; fi
echo "GATE PASS $(date)" > "$LOG/DONE"
