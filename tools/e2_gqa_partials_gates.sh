#!/bin/zsh
# E2 gates: per-engine gqa_partials move (tasks/mini-e2-gqa-partials.md).
#   1. unit suite (test_metal_ops: GQA-path parity, straddle, tiled parity,
#      chunk-vs-serial — all attention classes now run on caller-owned
#      partials scratch)
#   2. multislot suite G1..G7 green with the moved buffer
#   3. A/B byte-identity decode pre/post move against the stashed pre-E2
#      binary (build/q27-metal-pre-e2, dc43c52 tree): chunked AND serial
#      prefill classes, deep enough to cross the GQA threshold (2048) so
#      the blocked kernels are exercised, plus greedy generation tokens.
# NOTE: byte-identity here spans BOTH the E2 move and yukon's E1 batch
# (48484dd..); a mismatch means bisect before blaming either.
set -u
cd "$(dirname "$0")/.."

# Self-caffeinate the whole run (protocol rule: every long run) and
# skip already-complete arms (logs are moved into place only on
# success), so a killed run resumes instead of repeating ~40 min.
[ -z "${E2_CAFF:-}" ] && exec env E2_CAFF=1 caffeinate -i "$0" "$@"

MODEL=models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
CORPUS=data/wikitext2-test.tokens.bin
PRE=build/q27-metal-pre-e2
OUT=logs/e2_gates
mkdir -p "$OUT"

# Refuse to run while the census (or any q27-metal batch) is live: the
# rebuild below would invalidate its binary fingerprint mid-run.
if [ -d logs/kv_census ] && [ ! -f logs/kv_census/census_summary.tsv ] && [ ! -f logs/kv_census/ABORTED ]; then
    echo "e2 gates: census incomplete — refusing to rebuild under it"; exit 1
fi
pgrep -f 'q27-metal' >/dev/null && { echo "e2 gates: q27-metal process alive — not rebuilding under it"; exit 1; }
[ -x "$PRE" ] || { echo "e2 gates: pre-E2 binary $PRE missing"; exit 1; }

make build/q27-metal build/q27-metal-server build/test_metal_ops || exit 1

# Pin the numeric route both binaries measure under (today's defaults).
export Q27_METAL_GEMM_HALF=1
export Q27_METAL_GQA_TILE=2
export Q27_METAL_GQA_THRESHOLD=2048
export Q27_METAL_GQA_BLOCK=1024

fail=0

echo "== gate 1: unit suite =="
if [ -f "$OUT/unit.PASS" ]; then echo "unit suite PASS (cached)"; else
    ./build/test_metal_ops > "$OUT/unit.log" 2>&1 \
        && { echo "unit suite PASS"; touch "$OUT/unit.PASS"; } \
        || { echo "unit suite FAIL (see $OUT/unit.log)"; fail=1; }
fi

echo "== gate 2: multislot G1..G7 =="
if [ -f "$OUT/multislot.PASS" ]; then echo "multislot PASS (cached)"; else
    python3 tools/multislot_gates.py > "$OUT/multislot.log" 2>&1 \
        && { echo "multislot PASS"; touch "$OUT/multislot.PASS"; } \
        || { echo "multislot FAIL (see $OUT/multislot.log)"; fail=1; }
fi

echo "== gate 3: A/B byte-identity vs pre-E2 binary =="
# 4096 evaluated positions at ctx 4096: rows below AND above the 2048
# threshold, chunk-straddle included; then the same through serial prefill
# (one-token decode class); then greedy generation tokens.
for arm in "chunk:--prefill chunk" "serial:--prefill serial"; do
    name="${arm%%:*}"; flags="${arm#*:}"
    for side in pre post; do
        bin=$([ $side = pre ] && echo "$PRE" || echo build/q27-metal)
        log="$OUT/nll_${name}_${side}.log"
        [ -f "$log" ] && { echo "A/B $name $side: cached"; continue; }
        if "$bin" "$MODEL" "$TOK" --nll "$CORPUS" --nll-long 4097 --ctx 4096 ${=flags} \
            > "$log.tmp" 2>&1; then mv "$log.tmp" "$log"
        else echo "A/B $name $side run FAILED (see $log.tmp)"; fail=1; fi
    done
    if diff <(grep -v 'Metal model ready\|wall' "$OUT/nll_${name}_pre.log") \
            <(grep -v 'Metal model ready\|wall' "$OUT/nll_${name}_post.log") >/dev/null; then
        echo "A/B $name: byte-identical"
    else
        echo "A/B $name: MISMATCH (diff $OUT/nll_${name}_pre.log $OUT/nll_${name}_post.log)"; fail=1
    fi
done
for side in pre post; do
    bin=$([ $side = pre ] && echo "$PRE" || echo build/q27-metal)
    log="$OUT/gen_${side}.log"
    [ -f "$log" ] && { echo "A/B gen $side: cached"; continue; }
    if "$bin" "$MODEL" "$TOK" --prompt "The capital of France is" -n 64 \
        > "$log.tmp" 2>&1; then mv "$log.tmp" "$log"
    else echo "A/B gen $side run FAILED (see $log.tmp)"; fail=1; fi
done
if diff <(grep -v 'Metal model ready\|tok/s\|wall' "$OUT/gen_pre.log") \
        <(grep -v 'Metal model ready\|tok/s\|wall' "$OUT/gen_post.log") >/dev/null; then
    echo "A/B generation: identical"
else
    echo "A/B generation: MISMATCH"; fail=1
fi

if [ $fail -eq 0 ]; then echo "E2 GATES: ALL PASS" | tee "$OUT/RESULT"
else echo "E2 GATES: FAILURES (see $OUT)" | tee "$OUT/RESULT"; fi
exit $fail
