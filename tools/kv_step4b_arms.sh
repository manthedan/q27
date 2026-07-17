#!/bin/zsh
# KV-codec step 4b: design-step decomposition arms (control anchor + l7h1 / l7h1k / l63v holds)
# (docs/plans/2026-07-17-kv-codec-step4-probe.md). Arm A control (mode 3,
# empty mask) then arm B probe (census cells 10,11,125,127 held fp16),
# 8,191 positions each at the step-1/2 corpus config.
set -u
cd "$(dirname "$0")/.."

# Self-caffeinate end-to-end (2026-07-16 lesson: sleep reaps background
# jobs between caffeinated children).
[ -z "${STEP4B_CAFF:-}" ] && exec env STEP4B_CAFF=1 caffeinate -i "$0" "$@"

BIN=build/q27-metal
MODEL=models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
CORPUS=data/wikitext2-test.tokens.bin
OUT=logs/kv_step4b
mkdir -p "$OUT"

if pgrep -f "q27-metal " >/dev/null 2>&1; then
    echo "step4b: another q27-metal process is running; refusing to start"; exit 1
fi

make "$BIN" || { echo "step4b: BUILD FAILED" | tee "$OUT/ABORTED"; exit 1; }

export Q27_METAL_GEMM_HALF=1
export Q27_METAL_GQA_TILE=2
export Q27_METAL_GQA_THRESHOLD=2048
export Q27_METAL_GQA_BLOCK=1024

FP="$(git rev-parse HEAD 2>/dev/null || echo nogit) $(md5 -q "$BIN") $(md5 -q "$MODEL") $(md5 -q "$CORPUS") gemm_half=1 tile=2 thr=2048 blk=1024 nll_long=8192"
if [ -f "$OUT/fingerprint" ]; then
    [ "$(cat "$OUT/fingerprint")" = "$FP" ] ||
        { echo "step4b: fingerprint mismatch — move $OUT aside" | tee "$OUT/ABORTED"; exit 1; }
else
    if ls "$OUT"/*.log >/dev/null 2>&1 || [ -e "$OUT/step4b_summary.txt" ]; then
        echo "step4b: $OUT has unfingerprinted logs; move it aside" | tee "$OUT/ABORTED"; exit 1
    fi
    echo "$FP" > "$OUT/fingerprint"
fi

# Vacuity gate rides every launch: full exception mask must be exactly 0.
"$BIN" "$MODEL" "$TOK" --nll "$CORPUS" --nll-long 128 --ctx 256 \
    --kl-kv-except "$(seq -s, 0 127)" 2>&1 |
    grep -q "overall mean KL 0 nats, max 0" ||
    { echo "step4b: VACUITY GATE NONZERO" | tee "$OUT/ABORTED"; exit 1; }

run_arm() {
    local name=$1 list=$2
    local log="$OUT/$name.log"
    [ -s "$log" ] && grep -q "overall mean KL" "$log" && return 0  # resumable
    caffeinate -i "$BIN" "$MODEL" "$TOK" --nll "$CORPUS" --nll-long 8192 --ctx 8192 \
        --kl-kv-except "$list" > "$log" 2>&1
    grep -q "overall mean KL" "$log" ||
        { echo "step4b: arm $name FAILED (see $log)" | tee "$OUT/ABORTED"; exit 1; }
}

run_arm control -
# Control-arm class assert (codex P1 on the mode-3 commit): a stale
# pre-mode-3 shader stores everything clean and reads KL 0 — and the
# vacuity gate above EXPECTS 0, so it cannot catch that skew. The
# control arm can: its mean must land in the pre-registered class band
# (~0.0120 +-15%, the step-1 additivity sum).
cmean=$(grep -o 'overall mean KL [0-9.e-]*' "$OUT/control.log" | awk '{print $4}')
awk -v m="$cmean" 'BEGIN{exit !(m >= 0.0102 && m <= 0.0138)}' ||
    { echo "step4b: CONTROL OUT OF CLASS (mean $cmean vs [0.0102,0.0138]) — mode-3 plumbing suspect" |
      tee "$OUT/ABORTED"; exit 1; }
run_arm l7h1   10,11
run_arm l7h1k  10
run_arm l63v   125,127
# Amendment arm (registered 01:26 pre-measurement): V of L7 h1 alone.
run_arm l7h1v  11

{
    echo "4a anchors: control 0.011871 / p99 0.1006 / max 2.478 @1000; probe4 0.009329 / 0.0924 / 1.117 @7719"
    echo "reads: cheapest arm retaining max >=2x vs control becomes the design candidate"
    for a in control l7h1 l7h1k l63v l7h1v; do
        printf "%-8s %s\n" "$a" "$(grep 'kl-kv tail:' "$OUT/$a.log")"
        printf "%-8s %s\n" "$a" "$(grep 'overall mean KL' "$OUT/$a.log")"
    done
} > "$OUT/step4b_summary.txt"
cat "$OUT/step4b_summary.txt"
echo "step4b: complete"
