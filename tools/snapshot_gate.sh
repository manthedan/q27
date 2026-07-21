#!/bin/zsh
# Prefix-snapshot Phase 1 gates (docs/metal/plans/2026-07-16-prefix-snapshots.md):
# fresh-process byte identity for both KV dtypes, then the reject matrix
# (truncated file, corrupted identity, wrong dtype, position > context),
# then the crash-consistency legs (k3 audit B1/E2) driven by the engine's
# Q27_METAL_SNAP_CRASH failpoints. Every reject case must FAIL LOUD and
# leave generation able to error out before any state mutation.
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

# Whole generated text, not just the marker line: decoded continuations
# usually start with a newline, which left the old grep comparing only
# the empty "generated:" line (v2 hardening; strictly stronger check).
gen() { sed -n '/^generated:/,$p' "$1"; }

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

# KV fp16 exception arm (snapshot v2, 2026-07-17-kv-except-snapshot-v2.md):
# the funded L7-full config saves/loads with its side rows — fresh-process
# continuation must stay byte-identical under the env.
L7=8,9,10,11,12,13,14,15
kv=l7full
ref="$TMP/ref.$kv"; sav="$TMP/sav.$kv"; lod="$TMP/lod.$kv"; snap="$TMP/snap.$kv"
Q27_METAL_KV_FP16_CELLS=$L7 "$BIN" "$MODEL" "$TOK" --prompt "$PROMPT" -n $N --ctx $CTX --kv turbo3 > "$ref" 2>"$ref.err" ||
    { echo "FAIL[$kv]: reference run errored"; fails=$((fails+1)); }
Q27_METAL_KV_FP16_CELLS=$L7 "$BIN" "$MODEL" "$TOK" --prompt "$PROMPT" -n $N --ctx $CTX --kv turbo3 --save-state "$snap" > "$sav" 2>"$sav.err" ||
    { echo "FAIL[$kv]: save run errored"; fails=$((fails+1)); }
Q27_METAL_KV_FP16_CELLS=$L7 "$BIN" "$MODEL" "$TOK" --load-state "$snap" -n $N --ctx $CTX --kv turbo3 > "$lod" 2>"$lod.err" ||
    { echo "FAIL[$kv]: load run errored"; fails=$((fails+1)); }
if [ "$(gen "$ref")" = "$(gen "$sav")" ] && [ "$(gen "$ref")" = "$(gen "$lod")" ] && [ -n "$(gen "$ref")" ]; then
    echo "PASS[$kv]: exception-engine save-run and fresh-process load byte-identical to reference ($N tokens)"
    grep '^state:' "$sav.err" "$lod.err" | sed 's/^/  /'
else
    echo "FAIL[$kv]: exception-engine continuation diverged"; fails=$((fails+1))
fi

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
# Truncation INSIDE the final blob: every earlier blob is intact, so a
# validator that only walks lengths would bless it and pass 2 would
# partially restore (codex P2 on 39d74a0).
SNAPBYTES=$(wc -c < "$SNAP")
head -c $((SNAPBYTES - 100)) "$SNAP" > "$TMP/tail_truncated.snap"
expect_reject "truncation inside the final blob" "$BIN" "$MODEL" "$TOK" --load-state "$TMP/tail_truncated.snap" -n 4 --ctx $CTX
cp "$SNAP" "$TMP/badid.snap"
printf '\xff\xff\xff\xff' | dd of="$TMP/badid.snap" bs=1 seek=20 count=4 conv=notrunc 2>/dev/null
expect_reject "corrupted artifact identity" "$BIN" "$MODEL" "$TOK" --load-state "$TMP/badid.snap" -n 4 --ctx $CTX
expect_reject "wrong kv dtype" "$BIN" "$MODEL" "$TOK" --load-state "$SNAP" -n 4 --ctx $CTX --kv turbo3
expect_reject "position exceeds context" "$BIN" "$MODEL" "$TOK" --load-state "$SNAP" -n 4 --ctx 4

# v2 exception reject matrix: presence and cell-list identity both matter.
XSNAP="$TMP/snap.l7full"
expect_reject "env-unset snapshot into an exception engine" \
    env Q27_METAL_KV_FP16_CELLS=$L7 "$BIN" "$MODEL" "$TOK" --load-state "$TMP/snap.turbo3" -n 4 --ctx $CTX --kv turbo3
expect_reject "exception snapshot into an env-unset engine" \
    "$BIN" "$MODEL" "$TOK" --load-state "$XSNAP" -n 4 --ctx $CTX --kv turbo3
expect_reject "mismatched exception cell lists (equal size)" \
    env Q27_METAL_KV_FP16_CELLS=0,1,2,3,4,5,6,7 "$BIN" "$MODEL" "$TOK" --load-state "$XSNAP" -n 4 --ctx $CTX --kv turbo3
XBYTES=$(wc -c < "$XSNAP")
head -c $((XBYTES - 100)) "$XSNAP" > "$TMP/xtail_truncated.snap"
expect_reject "truncation inside the final side blob" \
    env Q27_METAL_KV_FP16_CELLS=$L7 "$BIN" "$MODEL" "$TOK" --load-state "$TMP/xtail_truncated.snap" -n 4 --ctx $CTX --kv turbo3

# Crash-consistency legs (k3 audit B1/E2). Q27_METAL_SNAP_CRASH makes
# save_state _exit(42) (a) after the last blob write, BEFORE the content
# fsync — the target path must be absent (fresh path) or still the previous
# intact snapshot, and a leftover .tmp is ignorable — or (b) after the
# rename + directory fsync — the snapshot must load and continue
# byte-identically to an uncrashed save.
CRASH="$TMP/crash.snap"
crash_save() {
    Q27_METAL_SNAP_CRASH=$1 "$BIN" "$MODEL" "$TOK" --prompt "$PROMPT" -n $N --ctx $CTX --kv fp16 \
        --save-state "$CRASH" > "$TMP/crash.$1.out" 2>&1
}

rm -f "$CRASH" "$CRASH.tmp"
crash_save before-fsync; rc=$?
if [ $rc -ne 42 ]; then
    echo "FAIL[crash]: before-fsync failpoint exited $rc, expected 42"; fails=$((fails+1))
elif [ -e "$CRASH" ]; then
    echo "FAIL[crash]: before-fsync crash left a target snapshot"; fails=$((fails+1))
else
    echo "PASS[crash]: before-fsync crash left no target snapshot (.tmp ignorable)"
fi

cp "$SNAP" "$CRASH"
crash_save before-fsync; rc=$?
"$BIN" "$MODEL" "$TOK" --load-state "$CRASH" -n $N --ctx $CTX --kv fp16 > "$TMP/crash.prev" 2>"$TMP/crash.prev.err"
if [ $rc -ne 42 ]; then
    echo "FAIL[crash]: before-fsync failpoint (re-save) exited $rc, expected 42"; fails=$((fails+1))
elif [ "$(gen "$TMP/ref.fp16")" = "$(gen "$TMP/crash.prev")" ]; then
    echo "PASS[crash]: before-fsync crash left the previous snapshot intact and loadable"
else
    echo "FAIL[crash]: crashed re-save corrupted the previous snapshot"; fails=$((fails+1))
fi

rm -f "$CRASH" "$CRASH.tmp"
crash_save after-rename; rc=$?
"$BIN" "$MODEL" "$TOK" --load-state "$CRASH" -n $N --ctx $CTX --kv fp16 > "$TMP/crash.b" 2>"$TMP/crash.b.err"
if [ $rc -ne 42 ]; then
    echo "FAIL[crash]: after-rename failpoint exited $rc, expected 42"; fails=$((fails+1))
elif [ "$(gen "$TMP/ref.fp16")" = "$(gen "$TMP/crash.b")" ]; then
    echo "PASS[crash]: after-rename crash loads and continues byte-identically to an uncrashed save"
else
    echo "FAIL[crash]: after-rename snapshot missing, unloadable, or continuation diverged"; fails=$((fails+1))
fi

if [ $fails -eq 0 ]; then echo "snapshot gate: ALL PASS"; else echo "snapshot gate: $fails FAILURE(S)"; exit 1; fi
