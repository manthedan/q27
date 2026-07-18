#!/bin/bash
# Census capability spot-check — one arm's generation pass
# (docs/plans/2026-07-17-census-capability-spotcheck.md).
#
#   usage: run_arm.sh <arm-name> <base-url> [outdir]
#   e.g.:  run_arm.sh t2-base http://127.0.0.1:8213
#
# COORDINATION.md applies: gen_runner is a model consumer — run ONLY inside
# a coordinated GPU slot against a server whose owner expects the traffic.
# The caller boots one server per pack (one model resident at a time) and
# runs this once per arm. Requests are serial and greedy (gen_runner sends
# no temperature; the server's missing-temperature default is 0.0), so an
# arm's outputs are deterministic for its pack + binary.
set -euo pipefail
cd "$(dirname "$0")/../.."
ARM=$1; URL=$2; OUT=${3:-logs/eval-census}
mkdir -p "$OUT"
for mode in choice numeric freeform; do
    python3 tools/eval/gen_runner.py "tools/eval/prompts/$mode.jsonl" \
        --out "$OUT/$ARM.$mode.jsonl" --base-url "$URL" \
        --api anthropic --max-tokens 1024 --tag "$ARM-$mode"
done
echo "run_arm: $ARM complete -> $OUT/$ARM.{choice,numeric,freeform}.jsonl"
