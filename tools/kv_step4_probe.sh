#!/bin/zsh
# KV-codec step 4 probe: joint 4-cell hold-out arms
# (docs/plans/2026-07-17-kv-codec-step4-probe.md). Arm A control (mode 3,
# empty mask) then arm B probe (census cells 10,11,125,127 held fp16),
# 8,191 positions each at the step-1/2 corpus config.
set -u
cd "$(dirname "$0")/.."

# Self-caffeinate end-to-end (2026-07-16 lesson: sleep reaps background
# jobs between caffeinated children).
[ -z "${STEP4_CAFF:-}" ] && exec env STEP4_CAFF=1 caffeinate -i "$0" "$@"

BIN=build/q27-metal
MODEL=models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
CORPUS=data/wikitext2-test.tokens.bin
OUT=logs/kv_step4
mkdir -p "$OUT"

if pgrep -f "q27-metal " >/dev/null 2>&1; then
    echo "step4: another q27-metal process is running; refusing to start"; exit 1
fi

make "$BIN" || { echo "step4: BUILD FAILED" | tee "$OUT/ABORTED"; exit 1; }

export Q27_METAL_GEMM_HALF=1
export Q27_METAL_GQA_TILE=2
export Q27_METAL_GQA_THRESHOLD=2048
export Q27_METAL_GQA_BLOCK=1024

FP="$(git rev-parse HEAD 2>/dev/null || echo nogit) $(md5 -q "$BIN") $(md5 -q "$MODEL") $(md5 -q "$CORPUS") gemm_half=1 tile=2 thr=2048 blk=1024 nll_long=8192"
if [ -f "$OUT/fingerprint" ]; then
    [ "$(cat "$OUT/fingerprint")" = "$FP" ] ||
        { echo "step4: fingerprint mismatch — move $OUT aside" | tee "$OUT/ABORTED"; exit 1; }
else
    if ls "$OUT"/*.log >/dev/null 2>&1 || [ -e "$OUT/step4_summary.txt" ]; then
        echo "step4: $OUT has unfingerprinted logs; move it aside" | tee "$OUT/ABORTED"; exit 1
    fi
    echo "$FP" > "$OUT/fingerprint"
fi

# Vacuity gate rides every launch: full exception mask must be exactly 0.
"$BIN" "$MODEL" "$TOK" --nll "$CORPUS" --nll-long 128 --ctx 256 \
    --kl-kv-except "$(seq -s, 0 127)" 2>&1 |
    grep -q "overall mean KL 0 nats, max 0" ||
    { echo "step4: VACUITY GATE NONZERO" | tee "$OUT/ABORTED"; exit 1; }

run_arm() {
    local name=$1 list=$2
    local log="$OUT/$name.log"
    [ -s "$log" ] && grep -q "overall mean KL" "$log" && return 0  # resumable
    caffeinate -i "$BIN" "$MODEL" "$TOK" --nll "$CORPUS" --nll-long 8192 --ctx 8192 \
        --kl-kv-except "$list" > "$log" 2>&1
    grep -q "overall mean KL" "$log" ||
        { echo "step4: arm $name FAILED (see $log)" | tee "$OUT/ABORTED"; exit 1; }
}

run_arm control -
run_arm probe4 10,11,125,127

{
    echo "step-1 both-sides baseline (turbo3 engine): mean 0.01165  p99 0.089  max 2.94 @pos 1000"
    echo "graduation bar: probe4 cuts control max >=2x OR p99 >=1.5x (mean read reported, no bar)"
    for a in control probe4; do
        printf "%-8s %s\n" "$a" "$(grep 'kl-kv tail:' "$OUT/$a.log")"
        printf "%-8s %s\n" "$a" "$(grep 'overall mean KL' "$OUT/$a.log")"
    done
} > "$OUT/step4_summary.txt"
cat "$OUT/step4_summary.txt"
echo "step4: complete"
