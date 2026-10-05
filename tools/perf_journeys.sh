#!/usr/bin/env bash
# perf_journeys.sh [--record] [PACK...] — wall-clock journeys against a
# per-machine baseline.
#
# Journeys (each the best of Q27_PERF_REPS runs, default 3, to shed noise):
#   load_s          model ready (mapping + upload), from q27-metal's own log
#   prefill512_tps  512-token prompt, chunked prefill
#   decode_tps      128 greedy decode tokens (n=129 run minus n=1 run)
#   suffix_copy_s   copy task with suffix bursts (T2 packs only)
# Baselines live in perf/baseline-<hw.model>-<GiB>.tsv, since speeds differ by
# machine. A journey more than Q27_PERF_BAND percent (default 10) slower than
# its baseline fails; --record writes the measured values as the new baseline.
# Unlike perf_ceilings.sh this gate is noisy by nature: run it on an idle
# machine, one GPU job at a time. bash 3.2-safe.
set -euo pipefail
cd "$(dirname "$0")/.."

record=0
if [ "${1:-}" = "--record" ]; then record=1; shift; fi
if [ $# -gt 0 ]; then packs="$*"; else
    packs=${Q27_GATE_PACKS:-"models/bonsai2/bonsai2-27b-t2-slim.q27 models/bonsai2/bonsai2-27b-t3-slim.q27"}
fi
tok=${Q27_GATE_TOK:-models/bonsai2/bonsai2.tok}
cli=${Q27_METAL_CLI:-build/q27-metal}
reps=${Q27_PERF_REPS:-3}
band=${Q27_PERF_BAND:-10}
machine="$(sysctl -n hw.model)-$(( $(sysctl -n hw.memsize) / 1073741824 ))g"
baseline=${Q27_PERF_BASELINE:-perf/baseline-$machine.tsv}
[ -x "$cli" ] || { echo "missing $cli (make build/q27-metal)" >&2; exit 2; }
[ -f "$tok" ] || { echo "missing tokenizer $tok" >&2; exit 2; }

work=$(mktemp -d "${TMPDIR:-/tmp}/q27-perf-journeys.XXXXXX")
trap 'rm -rf "$work"' EXIT
measured="$work/measured.tsv"
: > "$measured"

long_prompt=$(awk 'BEGIN { for (i = 0; i < 512; i++) printf "%s%d", (i ? "," : ""), 1000 + i % 5000 }')
short_prompt="760,1210,264,2805,32462,911,279,9396,323,279,12884"
copy_prompt="Repeat the following paragraph exactly, word for word:

The lighthouse keeper climbed the spiral stairs each evening, counted one hundred and twelve steps, trimmed the wick, wound the clockwork that turned the lens, and wrote the weather in a leather logbook before the first ship passed the point.

Repeated paragraph:
"

# timed EXPECT TAG PACK ARGS... -> "load_seconds generate_seconds generated";
# fails unless exactly EXPECT tokens were generated (an early EOS would
# otherwise read as a speedup).
timed() {
    local expect=$1 tag=$2 pack=$3 out load gen n; shift 3
    out=$("$cli" "$pack" "$tok" "$@" 2>&1 >/dev/null) || {
        echo "$out" >&2; echo "q27-metal failed ($tag)" >&2; return 1; }
    load=$(echo "$out" | sed -n 's/^Metal model ready on .* in \([0-9.]*\) s .*/\1/p')
    gen=$(echo "$out" | sed -n 's/^\([0-9]*\) tokens in \([0-9.]*\) s .*/\2/p')
    n=$(echo "$out" | sed -n 's/^\([0-9]*\) tokens in .*/\1/p')
    [ -n "$load" ] && [ -n "$gen" ] || { echo "$out" >&2; echo "unparsed timing ($tag)" >&2; return 1; }
    [ "$n" = "$expect" ] || { echo "$tag: generated $n tokens, expected $expect (EOS?)" >&2; return 1; }
    echo "$load $gen $n"
}

# best_of EXPECT TAG PACK ARGS... -> best (minimum) "load gen n" over $reps runs
best_of() {
    local i r l g n="" best_load="" best_gen=""
    for i in $(seq "$reps"); do
        r=$(timed "$@") || return 1
        read -r l g n <<EOF
$r
EOF
        if [ -z "$best_load" ] || awk -v a="$l" -v b="$best_load" 'BEGIN { exit !(a < b) }'; then best_load=$l; fi
        if [ -z "$best_gen" ] || awk -v a="$g" -v b="$best_gen" 'BEGIN { exit !(a < b) }'; then best_gen=$g; fi
    done
    echo "$best_load $best_gen $n"
}

record_row() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$measured"; }  # pack metric value better

for pack in $packs; do
    [ -f "$pack" ] || { echo "missing pack $pack" >&2; exit 2; }
    name=$(basename "$pack" .q27)
    echo "== $name (best of $reps)"

    r=$(best_of 1 prefill512 "$pack" --tokens "$long_prompt" -n 1 --ctx 1024) || exit 1
    set -- $r
    record_row "$name" load_s "$1" lower
    record_row "$name" prefill512_tps "$(awk -v s="$2" 'BEGIN { printf "%.2f", 512 / s }')" higher

    r=$(best_of 1 decode1 "$pack" --tokens "$short_prompt" -n 1 --ctx 512) || exit 1
    set -- $r; base=$2
    r=$(best_of 129 decode129 "$pack" --tokens "$short_prompt" -n 129 --ctx 512) || exit 1
    set -- $r
    record_row "$name" decode_tps "$(awk -v a="$2" -v b="$base" 'BEGIN { printf "%.2f", 128 / (a - b) }')" higher

    case "$name" in *-t3*) continue ;; esac
    r=$(best_of 48 suffix-copy "$pack" --prompt "$copy_prompt" -n 48 --ctx 1024 --suffix 16) || exit 1
    set -- $r
    record_row "$name" suffix_copy_s "$2" lower
done

# No packs (e.g. an empty argument) means nothing was measured: neither a
# pass nor something to record.
[ -s "$measured" ] || { echo "no journeys measured (empty pack list?)" >&2; exit 2; }

# Validate an existing baseline before either path: the --record merge treats
# line 1 as the header and would drop a headerless file's first row.
if [ -f "$baseline" ]; then
    [ "$(head -n 1 "$baseline")" = "$(printf 'pack\tmetric\tvalue\tbetter')" ] || {
        echo "malformed $baseline: first line must be the header (pack, metric, value, better)" >&2; exit 2; }
fi

if [ "$record" = 1 ]; then
    # Replace the measured packs' rows; keep every other pack's baseline.
    mkdir -p "$(dirname "$baseline")"
    {
        printf 'pack\tmetric\tvalue\tbetter\n'
        if [ -f "$baseline" ]; then
            awk -F'\t' 'NR == FNR { m[$1] = 1; next } FNR > 1 && !($1 in m)' "$measured" "$baseline"
        fi
        cat "$measured"
    } > "$work/baseline.new"
    mv "$work/baseline.new" "$baseline"
    column -t -s"$(printf '\t')" "$baseline"
    echo "baseline written: $baseline"
    exit 0
fi
[ -f "$baseline" ] || { echo "no baseline for $machine ($baseline); run with --record on an idle machine" >&2; exit 1; }

awk -F'\t' -v band="$band" '
    NR == FNR { if (FNR > 1) { base[$1 "\t" $2] = $3; order[++n] = $1 "\t" $2 }; next }
    {
        key = $1 "\t" $2; v = $3 + 0; seen[key] = 1; measured_pack[$1] = 1
        if (!(key in base)) { printf "FAIL  %-24s %-16s %10.2f (no baseline; --record on an idle machine)\n", $1, $2, v; fails++; next }
        b = base[key] + 0
        # Slowdown in percent, whichever direction is better.
        slow = ($4 == "higher") ? 100 * (b - v) / b : 100 * (v - b) / b
        tag = slow > band ? "FAIL" : "ok  "
        if (slow > band) fails++
        printf "%s  %-24s %-16s %10.2f  baseline %10.2f  %+6.1f%% %s\n", tag, $1, $2, v, b, -slow, (slow > 0 ? "slower" : "faster")
    }
    END {
        # A baseline metric for a measured pack that no journey produced.
        for (i = 1; i <= n; i++) {
            split(order[i], kp, "\t")
            if ((kp[1] in measured_pack) && !(order[i] in seen)) {
                printf "FAIL  %-24s %-16s not measured (journey removed? re-record in a reviewed commit)\n", kp[1], kp[2]; fails++
            }
        }
        exit fails ? 1 : 0
    }' "$baseline" "$measured" && status=0 || status=$?
[ "$status" -eq 0 ] && echo "perf journeys: PASS (band ${band}%)" || echo "perf journeys: FAIL (band ${band}%)"
exit "$status"
