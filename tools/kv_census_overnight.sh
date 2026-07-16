#!/bin/zsh
# KV-codec step 3: 128-cell sensitivity census (overnight batch, mac-mini).
# One process per cell: quantize exactly one (attention layer, KV head, K|V)
# cell through the turbo3 round-trip, 2,048 wikitext2 positions vs the fp16
# baseline. cell id = attn_idx*8 + head*2 + side (side 0=K, 1=V).
# ~2.5 min/cell => ~5.3 h. Plan: docs/plans/2026-07-16-kv-codec-census.md
set -u
cd "$(dirname "$0")/.."

BIN=build/q27-metal
MODEL=models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
CORPUS=data/wikitext2-test.tokens.bin
OUT=logs/kv_census
mkdir -p "$OUT"

# Rebuild at the top of the batch — a stale binary silently mismeasures
# (2026-07-16 overnight lesson, commit 38c744a).
make "$BIN" || { echo "census: BUILD FAILED, aborting" | tee "$OUT/ABORTED"; exit 1; }

# Pin the numeric route explicitly with overwrite — inherited env knobs can
# collapse arms onto one kernel (codex P2 on eaee44b) or silently change the
# attention fold order (codex P2 on d6ebfa5). These are today's defaults;
# the census is defined against this route.
export Q27_METAL_GEMM_HALF=1
export Q27_METAL_GQA_TILE=2
export Q27_METAL_GQA_THRESHOLD=2048
export Q27_METAL_GQA_BLOCK=1024

# Run fingerprint: resuming after ANY identity change (source, artifact,
# corpus, route pins, position count) must not mix logs from two different
# experiments into one summary (codex P1 on d6ebfa5).
FP="$(git rev-parse HEAD 2>/dev/null || echo nogit) $(md5 -q "$BIN") $(md5 -q "$MODEL") $(md5 -q "$CORPUS") gemm_half=$Q27_METAL_GEMM_HALF tile=$Q27_METAL_GQA_TILE thr=$Q27_METAL_GQA_THRESHOLD blk=$Q27_METAL_GQA_BLOCK nll_long=2049 ctx=2048"
if [ -f "$OUT/fingerprint" ]; then
    [ "$(cat "$OUT/fingerprint")" = "$FP" ] ||
        { echo "census: fingerprint mismatch — $OUT holds a different experiment; move it aside" | tee "$OUT/ABORTED"; exit 1; }
else
    echo "$FP" > "$OUT/fingerprint"
fi

# Canary: the instrument must be exactly zero against itself.
"$BIN" "$MODEL" "$TOK" --nll "$CORPUS" --nll-long 128 --ctx 256 --kl-kv-self 2>&1 |
    grep -q "overall mean KL 0 nats, max 0" ||
    { echo "census: SELF-CHECK NONZERO, aborting" | tee "$OUT/ABORTED"; exit 1; }

for cell in $(seq 0 127); do
    log=$(printf "%s/cell_%03d.log" "$OUT" "$cell")
    [ -s "$log" ] && grep -q "overall mean KL" "$log" && continue  # resumable
    # 2049 tokens => 2,048 evaluated positions (the KL path encodes n-1;
    # codex P2 on d6ebfa5).
    caffeinate -i "$BIN" "$MODEL" "$TOK" --nll "$CORPUS" \
        --nll-long 2049 --ctx 2048 --kl-kv-cell "$cell" > "$log" 2>&1
    grep -q "overall mean KL" "$log" ||
        { echo "census: cell $cell FAILED (see $log), aborting" | tee "$OUT/ABORTED"; exit 1; }
done

# Summary TSV: cell, layer, head, side, mean, p99, p99.5, max, max_pos
{
    printf "cell\tlayer\thead\tside\tmean\tp99\tp99_5\tmax\tmax_pos\n"
    for cell in $(seq 0 127); do
        log=$(printf "%s/cell_%03d.log" "$OUT" "$cell")
        layer=$(( (cell / 8) * 4 + 3 )); head=$(( (cell / 2) % 4 ))
        side=$([ $((cell % 2)) -eq 0 ] && echo K || echo V)
        mean=$(sed -n 's/.*overall mean KL \([0-9.e-]*\) nats.*/\1/p' "$log")
        tail_line=$(grep "kl-kv tail:" "$log")
        p99=$(echo "$tail_line"    | sed -n 's/.*p99 \([0-9.e-]*\) .*/\1/p')
        p995=$(echo "$tail_line"   | sed -n 's/.*p99\.5 \([0-9.e-]*\) .*/\1/p')
        mx=$(echo "$tail_line"     | sed -n 's/.*max \([0-9.e-]*\) @pos.*/\1/p')
        mxpos=$(echo "$tail_line"  | sed -n 's/.*@pos \([0-9]*\).*/\1/p')
        printf "%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$cell" "$layer" "$head" "$side" "$mean" "$p99" "$p995" "$mx" "$mxpos"
    done
} > "$OUT/census_summary.tsv"
echo "census: complete, summary at $OUT/census_summary.tsv"
