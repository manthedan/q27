#!/bin/zsh
# Prefix-snapshot Phase 1 gates (docs/plans/2026-07-16-prefix-snapshots.md):
# fresh-process byte identity for both KV dtypes, then the reject matrix
# (truncated file, corrupted identity, wrong dtype, position > context).
# Every reject case must FAIL LOUD and leave generation able to error out
# before any state mutation.
set -u
cd "$(dirname "$0")/.."

BIN=build/q27-metal
MODEL=models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
TMP=$(mktemp -d /tmp/q27snap.XXXXXX)
trap 'rm -rf "$TMP"' EXIT
PROMPT="The disk snapshot gate replays a saved prefix and must continue byte-identically:"
N=64
CTX=512
fails=0

gen() { grep '^generated:' "$1"; }

for kv in fp16 turbo3; do
    ref="$TMP/ref.$kv"; sav="$TMP/sav.$kv"; lod="$TMP/lod.$kv"; snap="$TMP/snap.$kv"
    "$BIN" "$MODEL" "$TOK" --prompt "$PROMPT" -n $N --ctx $CTX --kv $kv > "$ref" 2>"$ref.err" ||
        { echo "FAIL[$kv]: reference run errored"; fails=$((fails+1)); continue; }
    "$BIN" "$MODEL" "$TOK" --prompt "$PROMPT" -n $N --ctx $CTX --kv $kv --save-state "$snap" > "$sav" 2>"$sav.err" ||
        { echo "FAIL[$kv]: save run errored"; fails=$((fails+1)); continue; }
    "$BIN" "$MODEL" "$TOK" --load-state "$snap" -n $N --ctx $CTX --kv $kv > "$lod" 2>"$lod.err" ||
        { echo "FAIL[$kv]: load run errored"; fails=$((fails+1)); continue; }
    if [ "$(gen "$ref")" = "$(gen "$sav")" ] && [ "$(gen "$ref")" = "$(gen "$lod")" ]; then
        echo "PASS[$kv]: save-run and fresh-process load both byte-identical to reference ($N tokens)"
        grep '^state:' "$sav.err" "$lod.err" | sed 's/^/  /'
    else
        echo "FAIL[$kv]: continuation diverged"; fails=$((fails+1))
    fi
done

expect_reject() {
    local label=$1; shift
    local out="$TMP/rej.$RANDOM"
    if "$@" > "$out" 2>&1; then
        echo "FAIL[reject]: $label was ACCEPTED"; fails=$((fails+1))
    elif grep -q '^generated:' "$out"; then
        echo "FAIL[reject]: $label errored but still generated"; fails=$((fails+1))
    else
        echo "PASS[reject]: $label -> $(tail -1 "$out")"
    fi
}

SNAP="$TMP/snap.fp16"
head -c 1000 "$SNAP" > "$TMP/truncated.snap"
expect_reject "truncated file" "$BIN" "$MODEL" "$TOK" --load-state "$TMP/truncated.snap" -n 4 --ctx $CTX
cp "$SNAP" "$TMP/badid.snap"
printf '\xff\xff\xff\xff' | dd of="$TMP/badid.snap" bs=1 seek=20 count=4 conv=notrunc 2>/dev/null
expect_reject "corrupted artifact identity" "$BIN" "$MODEL" "$TOK" --load-state "$TMP/badid.snap" -n 4 --ctx $CTX
expect_reject "wrong kv dtype" "$BIN" "$MODEL" "$TOK" --load-state "$SNAP" -n 4 --ctx $CTX --kv turbo3
expect_reject "position exceeds context" "$BIN" "$MODEL" "$TOK" --load-state "$SNAP" -n 4 --ctx 4

if [ $fails -eq 0 ]; then echo "snapshot gate: ALL PASS"; else echo "snapshot gate: $fails FAILURE(S)"; exit 1; fi
