#!/bin/sh
# B2/E4 post-throw engine-state gate wrapper (k3 audit; pre-registered leg 4).
# Usage: tools/failpoint_gate.sh <model.q27>
# Sweeps Q27_METAL_FAIL_FINISH until the injection lands inside the step loop
# (the countdown also ticks during load/ingest finishes), then diffs the
# post-reset regeneration against an unarmed reference run.
set -u
MODEL="${1:?usage: failpoint_gate.sh model.q27}"
BIN=build/failpoint_gate
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

"$BIN" "$MODEL" > "$OUT/ref.txt" || { echo "FAIL: unarmed reference run"; exit 1; }

for n in 4 8 16 24 32 48 64; do
    Q27_METAL_FAIL_FINISH=$n "$BIN" "$MODEL" > "$OUT/armed.txt" 2> "$OUT/armed.err"
    rc=$?
    case $rc in
        0)  if cmp -s "$OUT/ref.txt" "$OUT/armed.txt"; then
                echo "failpoint gate: PASS (N=$n; $(head -1 "$OUT/armed.err"))"
                exit 0
            fi
            echo "FAIL: post-reset regeneration diverges from clean engine (N=$n)"
            diff "$OUT/ref.txt" "$OUT/armed.txt" | head
            exit 1;;
        3|4) continue;;   # fired during load/ingest — need a larger N
        5)   continue;;   # never fired — window overshot, keep going
        *)   echo "FAIL: gate assertions (N=$n, rc=$rc)"; cat "$OUT/armed.err"; exit 1;;
    esac
done
echo "FAIL: no N in the sweep landed the injection inside the step loop"
exit 1
