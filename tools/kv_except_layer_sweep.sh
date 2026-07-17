#!/bin/zsh
# Production-domain exception sweep (docs/plans/2026-07-17-kv-except-production.md,
# gate-3 FAIL follow-up): which attention layer's head-pairs, protected in
# PRODUCTION (Q27_METAL_KV_FP16_CELLS), kill the pos-1000 tail event? The
# attrib-derived list did not transfer at the knife-edge (l7h1: attrib
# 2.48->1.12 vs production 2.94->2.27); the mean transferred. 1,536-position
# arms (~2 min) — the event is at pos 1000.
set -u
cd "$(dirname "$0")/.."
[ -z "${SWEEP_CAFF:-}" ] && exec env SWEEP_CAFF=1 caffeinate -i "$0" "$@"

BIN=build/q27-metal
M=models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27
T=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
C=data/wikitext2-test.tokens.bin
OUT=logs/kv_except
mkdir -p "$OUT"

export Q27_METAL_GEMM_HALF=1 Q27_METAL_GQA_TILE=2
export Q27_METAL_GQA_THRESHOLD=2048 Q27_METAL_GQA_BLOCK=1024

run_arm() {
    local name=$1 cells=$2
    local log="$OUT/sweep_$name.log"
    [ -s "$log" ] && grep -q "overall mean KL" "$log" && return 0
    if [ "$cells" = "NONE" ]; then
        "$BIN" "$M" "$T" --nll "$C" --nll-long 1536 --ctx 2048 --kl-kv > "$log" 2>&1
    else
        Q27_METAL_KV_FP16_CELLS="$cells" \
            "$BIN" "$M" "$T" --nll "$C" --nll-long 1536 --ctx 2048 --kl-kv > "$log" 2>&1
    fi
    grep -q "overall mean KL" "$log" ||
        { echo "sweep: arm $name FAILED (see $log)" | tee "$OUT/SWEEP_ABORTED"; exit 1; }
}

run_arm control NONE
for i in $(seq 0 15); do
    cells=$(seq -s, $((i*8)) $((i*8+7)))
    run_arm "L$((i*4+3))" "$cells"
done

{
    echo "production layer sweep, 1536 positions, ctx 2048 (event @pos 1000)"
    for f in "$OUT"/sweep_*.log; do
        printf "%-14s %s\n" "$(basename $f .log)" "$(grep -o 'overall mean KL.*' "$f")"
    done
} > "$OUT/sweep_summary.txt"
cat "$OUT/sweep_summary.txt"
echo "sweep: complete"
