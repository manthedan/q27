#!/usr/bin/env bash
# perf_ceilings.sh [--ratchet] [PACK...] — deterministic work-count gate.
#
# Runs fixed greedy workloads through build/q27-metal --counters and compares
# the Metal command buffers and GPU dispatches against perf/ceilings.tsv.
# Same binary + pack + workload give the same counts on every run, so the gate
# is exact (no noise band): a count above its ceiling fails. A count below its
# ceiling is reported; --ratchet then lowers that ceiling (ceilings only go down).
# A metric with no ceiling yet fails until --ratchet records it.
#
# Packs: arguments, else Q27_GATE_PACKS, else the t2-slim and t3-slim packs
# under models/bonsai2/; tokenizer Q27_GATE_TOK. One model resident at a time.
# bash 3.2-safe.
set -euo pipefail
cd "$(dirname "$0")/.."

ratchet=0
if [ "${1:-}" = "--ratchet" ]; then ratchet=1; shift; fi
if [ $# -gt 0 ]; then packs="$*"; else
    packs=${Q27_GATE_PACKS:-"models/bonsai2/bonsai2-27b-t2-slim.q27 models/bonsai2/bonsai2-27b-t3-slim.q27"}
fi
tok=${Q27_GATE_TOK:-models/bonsai2/bonsai2.tok}
ceilings=${Q27_PERF_CEILINGS:-perf/ceilings.tsv}
cli=${Q27_METAL_CLI:-build/q27-metal}
[ -x "$cli" ] || { echo "missing $cli (make build/q27-metal)" >&2; exit 2; }
[ -f "$tok" ] || { echo "missing tokenizer $tok" >&2; exit 2; }

work=$(mktemp -d "${TMPDIR:-/tmp}/q27-perf-ceilings.XXXXXX")
trap 'rm -rf "$work"' EXIT
measured="$work/measured.tsv"
: > "$measured"

# Fixed token ids (tokenizer-independent): a 512-token prompt for prefill and
# an 11-token prompt for decode.
long_prompt=$(awk 'BEGIN { for (i = 0; i < 512; i++) printf "%s%d", (i ? "," : ""), 1000 + i % 5000 }')
short_prompt="760,1210,264,2805,32462,911,279,9396,323,279,12884"
copy_prompt="Repeat the following paragraph exactly, word for word:

The lighthouse keeper climbed the spiral stairs each evening, counted one hundred and twelve steps, trimmed the wick, wound the clockwork that turned the lens, and wrote the weather in a leather logbook before the first ship passed the point.

Repeated paragraph:
"

# run TAG PACK ARGS... -> "generated command_buffers dispatches". Callers
# assign the result (r=$(run ...) || exit 1) so a failure stops the gate.
run() {
    local tag=$1 pack=$2 out; shift 2
    out=$("$cli" "$pack" "$tok" "$@" --counters 2>&1 >/dev/null) || {
        echo "$out" >&2; echo "q27-metal failed ($tag)" >&2; return 1; }
    out=$(echo "$out" | sed -n 's/^counters: prompt=[0-9]* generated=\([0-9]*\) command_buffers=\([0-9]*\) dispatches=\([0-9]*\)$/\1 \2 \3/p')
    [ -n "$out" ] || { echo "q27-metal printed no counters line ($tag)" >&2; return 1; }
    echo "$out"
}

record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$measured"; }

for pack in $packs; do
    [ -f "$pack" ] || { echo "missing pack $pack" >&2; exit 2; }
    name=$(basename "$pack" .q27)
    echo "== $name"

    r=$(run prefill512 "$pack" --tokens "$long_prompt" -n 1 --ctx 1024) || exit 1
    set -- $r
    [ "$1" = 1 ] || { echo "$name: prefill workload generated $1 tokens, expected 1" >&2; exit 1; }
    record "$name" prefill512.command_buffers "$2"
    record "$name" prefill512.dispatches "$3"

    # Decode cost per token = (n=65 run) - (n=1 run) over the same prompt: the
    # difference is exactly 64 decode steps, prefill cancels out.
    r=$(run decode1 "$pack" --tokens "$short_prompt" -n 1 --ctx 512) || exit 1
    set -- $r
    [ "$1" = 1 ] || { echo "$name: decode workload generated $1 tokens, expected 1" >&2; exit 1; }
    base_cb=$2 base_ops=$3
    r=$(run decode65 "$pack" --tokens "$short_prompt" -n 65 --ctx 512) || exit 1
    set -- $r
    [ "$1" = 65 ] || { echo "$name: decode workload stopped at $1 tokens (EOS); change short_prompt" >&2; exit 1; }
    record "$name" decode64.command_buffers $(( $2 - base_cb ))
    record "$name" decode64.dispatches $(( $3 - base_ops ))

    # Suffix bursts on a copy task (the agent's file-edit shape). Counts
    # depend on draft acceptance, which is deterministic under greedy decode.
    # T3 packs have no batched prefill, so no batched suffix (wrapper skips it).
    case "$name" in *-t3*) continue ;; esac
    r=$(run suffix-copy "$pack" --prompt "$copy_prompt" -n 48 --ctx 1024 --suffix 16) || exit 1
    set -- $r
    [ "$1" = 48 ] || { echo "$name: suffix workload stopped at $1 tokens (EOS); change copy_prompt" >&2; exit 1; }
    record "$name" suffix_copy48.command_buffers "$2"
    record "$name" suffix_copy48.dispatches "$3"
done

# No packs (e.g. an empty argument) means nothing was measured: not a pass.
[ -s "$measured" ] || { echo "no workloads measured (empty pack list?)" >&2; exit 2; }

# Compare against ceilings; with --ratchet, write the lowered table back.
header=$(printf 'pack\tmetric\tceiling')
[ -f "$ceilings" ] || printf '%s\n' "$header" > "$ceilings"
# An empty or headerless table would make awk read the measurements as the
# table and compare nothing.
[ "$(head -n 1 "$ceilings")" = "$header" ] || {
    echo "malformed $ceilings: first line must be the header (pack, metric, ceiling)" >&2; exit 2; }
awk -F'\t' -v ratchet="$ratchet" -v out="$work/new.tsv" '
    NR == FNR { if (FNR > 1) { key = $1 "\t" $2; ceil[key] = $3; order[++n] = key } else header = $0; next }
    {
        key = $1 "\t" $2; val = $3 + 0; seen[key] = 1; measured_pack[$1] = 1
        if (!(key in ceil)) {
            if (ratchet) { printf "NEW   %-28s %-34s %12d\n", $1, $2, val; ceil[key] = val; order[++n] = key }
            else { printf "FAIL  %-28s %-34s %12d (no ceiling; run --ratchet)\n", $1, $2, val; fails++ }
            next
        }
        c = ceil[key] + 0
        if (val > c) { printf "FAIL  %-28s %-34s %12d > ceiling %d (+%.2f%%)\n", $1, $2, val, c, 100 * (val - c) / c; fails++ }
        else if (val < c) {
            printf "%s %-28s %-34s %12d < ceiling %d (-%.2f%%)\n", (ratchet ? "LOWER" : "BELOW"), $1, $2, val, c, 100 * (c - val) / c
            if (ratchet) ceil[key] = val; else below++
        }
        else printf "ok    %-28s %-34s %12d\n", $1, $2, val
    }
    END {
        # A ceiling for a measured pack that no workload produced: the gate
        # would silently stop covering it.
        for (i = 1; i <= n; i++) {
            split(order[i], kp, "\t")
            if ((kp[1] in measured_pack) && !(order[i] in seen)) {
                printf "FAIL  %-28s %-34s not measured (workload removed? delete the row in a reviewed commit)\n", kp[1], kp[2]; fails++
            }
        }
        if (ratchet && !fails) {
            print header > out
            for (i = 1; i <= n; i++) print order[i] "\t" ceil[order[i]] > out
        }
        if (below && !ratchet) print "note: counts below ceiling; run tools/perf_ceilings.sh --ratchet to lock in the win"
        exit fails ? 1 : 0
    }' "$ceilings" "$measured" && status=0 || status=$?

if [ "$status" -eq 0 ] && [ "$ratchet" = 1 ] && [ -f "$work/new.tsv" ]; then
    cp "$work/new.tsv" "$ceilings"
    echo "ceilings written: $ceilings"
fi
[ "$status" -eq 0 ] && echo "perf ceilings: PASS" || echo "perf ceilings: FAIL"
exit "$status"
