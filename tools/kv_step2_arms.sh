#!/bin/zsh
# KV-codec step 2: KVarN-style scaling arms on the turbo3 round-trip
# (docs/plans/2026-07-16-kv-codec-step2.md). Stats pass (per-feature RMS,
# KL-zero canary rides along), then K:scale32 / K:feature / K:both at the
# step-1 corpus config (--nll-long 8192 => 8,191 positions, comparable to
# step 1's K arm: mean 0.00677, p99 0.064, max 2.52).
# Graduation (pre-registered): max <= 0.50 or p99 <= 0.021 at mean +-10%.
set -u
cd "$(dirname "$0")/.."

BIN=build/q27-metal
MODEL=models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
CORPUS=data/wikitext2-test.tokens.bin
OUT=logs/kv_step2
mkdir -p "$OUT"

# Never contend with a running census/batch for the GPU or the binary.
if pgrep -f "q27-metal" >/dev/null 2>&1; then
    echo "step2: another q27-metal process is running; refusing to start"; exit 1
fi
if [ -d logs/kv_census ] && [ ! -f logs/kv_census/census_summary.tsv ] && [ ! -f logs/kv_census/ABORTED ]; then
    echo "step2: census incomplete (no summary, no ABORTED); refusing to steal its binary/GPU"; exit 1
fi

make "$BIN" || { echo "step2: BUILD FAILED" | tee "$OUT/ABORTED"; exit 1; }

export Q27_METAL_GEMM_HALF=1
export Q27_METAL_GQA_TILE=2
export Q27_METAL_GQA_THRESHOLD=2048
export Q27_METAL_GQA_BLOCK=1024

FP="$(git rev-parse HEAD 2>/dev/null || echo nogit) $(md5 -q "$BIN") $(md5 -q "$MODEL") $(md5 -q "$CORPUS") gemm_half=1 tile=2 thr=2048 blk=1024 nll_long=8192"
if [ -f "$OUT/fingerprint" ]; then
    [ "$(cat "$OUT/fingerprint")" = "$FP" ] ||
        { echo "step2: fingerprint mismatch — move $OUT aside" | tee "$OUT/ABORTED"; exit 1; }
else
    if ls "$OUT"/*.log >/dev/null 2>&1 || [ -e "$OUT/step2_summary.txt" ]; then
        echo "step2: $OUT has unfingerprinted logs; move it aside" | tee "$OUT/ABORTED"; exit 1
    fi
    echo "$FP" > "$OUT/fingerprint"
fi

# Smoke gates (128 positions, ~15 s each): self-check exactly zero through
# the new plumbing; a feature-arm smoke must run and be nonzero-small.
"$BIN" "$MODEL" "$TOK" --nll "$CORPUS" --nll-long 128 --ctx 256 --kl-kv-self 2>&1 |
    grep -q "overall mean KL 0 nats, max 0" ||
    { echo "step2: SELF-CHECK NONZERO" | tee "$OUT/ABORTED"; exit 1; }

run_arm() {
    local name=$1; shift
    local log="$OUT/$name.log"
    [ -s "$log" ] && grep -q "overall mean KL" "$log" && return 0  # resumable
    caffeinate -i "$BIN" "$MODEL" "$TOK" --nll "$CORPUS" --nll-long 8192 --ctx 8192 "$@" > "$log" 2>&1
    grep -q "overall mean KL" "$log" ||
        { echo "step2: arm $name FAILED (see $log)" | tee "$OUT/ABORTED"; exit 1; }
}

# Stats pass first (its KL-zero canary is enforced by the binary itself).
run_arm stats --kl-kv-stats "$OUT/feature_stats.bin"
[ -s "$OUT/feature_stats.bin" ] || { echo "step2: stats file missing" | tee "$OUT/ABORTED"; exit 1; }

# Feature-arm smoke at 128 positions before committing 11 minutes to it.
"$BIN" "$MODEL" "$TOK" --nll "$CORPUS" --nll-long 128 --ctx 256 \
    --kl-kv-k --kv-rt-feature "$OUT/feature_stats.bin" 2>&1 |
    grep -q "overall mean KL" ||
    { echo "step2: feature-arm smoke FAILED" | tee "$OUT/ABORTED"; exit 1; }

run_arm k_scale32 --kl-kv-k --kv-rt-scale32
run_arm k_feature --kl-kv-k --kv-rt-feature "$OUT/feature_stats.bin"
run_arm k_both    --kl-kv-k --kv-rt-scale32 --kv-rt-feature "$OUT/feature_stats.bin"

{
    echo "step-1 K baseline: mean 0.00677  p99 0.064  p99.5 0.118  max 2.52 @pos 1000"
    echo "graduation bar: max <= 0.50 OR p99 <= 0.021, at mean within [0.00609, 0.00745]"
    for a in k_scale32 k_feature k_both; do
        printf "%-10s %s\n" "$a" "$(grep 'kl-kv tail:' "$OUT/$a.log")"
        printf "%-10s %s\n" "$a" "$(grep 'overall mean KL' "$OUT/$a.log")"
    done
} > "$OUT/step2_summary.txt"
cat "$OUT/step2_summary.txt"
echo "step2: complete"
