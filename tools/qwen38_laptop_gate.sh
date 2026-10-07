#!/usr/bin/env bash
# qwen38_laptop_gate.sh [evidence-dir] — the one-command Qwen3.8 release gate
# for a 24 GB Apple Silicon Mac (v0.7.1). Run from a checkout of the release
# commit; everything runs strictly one after another (one model resident).
#
#   1. preflight: macOS arm64, >= 24 GiB RAM, an M4 / M4 Pro (published
#      canonical), no spaces in paths; builds the binaries
#   2. pack: QWEN38_MODEL/QWEN38_TOKENIZER if set, else ~/.q27/models/q38
#      (Q27_HOME), pulling it with `packaging/bin/q27 pull q38` (SHA-256
#      verified, ~14.6 GB) when missing
#   3. gates: canonical registry contracts, Metal backend suite, Qwen3.8 policy
#      + canonical digest, serving (OpenAI/Responses/Anthropic tool round
#      trips), native agent through the packaged wrapper
#   4. perf: first work-count ceilings for the pack, written to the evidence
#      dir (bring perf-ceilings-q38.tsv back to commit them)
#
# Prints a ledger and exits nonzero if any gate failed. Evidence (logs, ledger,
# machine facts) lands in evidence-dir (default /tmp/q27-qwen38-gate-<time>).
# bash 3.2-safe.
set -u
cd "$(dirname "$0")/.."
root=$(pwd)
EVID=${1:-/tmp/q27-qwen38-gate-$(date +%Y%m%d-%H%M)}
mkdir -p "$EVID"
LEDGER="$EVID/ledger.txt"
: > "$LEDGER"
fails=0
pass() { echo "PASS: $*" | tee -a "$LEDGER"; }
fail() { echo "FAIL: $*" | tee -a "$LEDGER"; fails=$((fails+1)); }
note() { echo "note: $*" | tee -a "$LEDGER"; }
finish() {
    echo "----"; cat "$LEDGER"; echo "evidence: $EVID"
    [ "$fails" -eq 0 ] && { echo "qwen38 laptop gate: PASS"; exit 0; }
    echo "qwen38 laptop gate: FAIL ($fails)"; exit 1
}
run_gate() { # name logfile cmd...
    local name=$1 log=$2; shift 2
    echo "== $name (log: $log)"
    if caffeinate -i "$@" > "$log" 2>&1; then pass "$name"; else fail "$name (log: $log)"; fi
}

# ---- 1. preflight ------------------------------------------------------------
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || { fail "needs macOS on Apple Silicon"; finish; }
mem_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))
chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)
note "commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown) on $chip, ${mem_gib} GiB, macOS $(sw_vers -productVersion)"
[ "$mem_gib" -ge 24 ] || { fail "Qwen3.8 needs >= 24 GiB RAM (this Mac: ${mem_gib} GiB)"; finish; }
# The canonical digest is published for M4 / M4 Pro only (metal_canonical_gate.sh);
# fail before a 15 GB download on anything else unless CANON_ARCH is set.
# Same chip probe and list as metal_canonical_gate.sh.
if [ -z "${CANON_ARCH:-}" ]; then
    case "$chip" in
        "Apple M4"|"Apple M4 Pro") ;;
        *) fail "no published Qwen3.8 canonical for '$chip' (M4 / M4 Pro only); set CANON_ARCH after deriving one"; finish ;;
    esac
fi
# Paths with spaces would split in make and the perf scripts.
case "$root$EVID${QWEN38_MODEL:-}${QWEN38_TOKENIZER:-}${Q27_HOME:-}" in
    *" "*) fail "paths must not contain spaces (checkout, evidence dir, pack); use a symlink"; finish ;;
esac
# Build first: the pull itself needs build/q27-lock-exec.
make build/q27-metal build/q27-metal-server build/q27-agent build/q27-lock-exec > "$EVID/build.log" 2>&1 \
    || { fail "build (log: $EVID/build.log)"; finish; }
if [ -n "$(git status --short -- src/ tools/ packaging/ Makefile 2>/dev/null)" ]; then
    note "working tree has edits under src/tools/packaging/Makefile: evidence is not for a clean commit"
fi

# ---- 2. pack -----------------------------------------------------------------
if [ -n "${QWEN38_MODEL:-}" ] || [ -n "${QWEN38_TOKENIZER:-}" ]; then
    model=${QWEN38_MODEL:-}; tok=${QWEN38_TOKENIZER:-}
else
    home=${Q27_HOME:-$HOME/.q27/models}
    . packaging/lib/q27_bench_lib.sh
    model="$home/q38/$(q27_registry_field q38 4)"
    tok="$home/q38/$(q27_registry_field q38 16)"
    if [ ! -f "$model" ] || [ ! -f "$tok" ]; then
        # Free space on the filesystem the pull writes to (nearest existing dir).
        probe=$home; while [ ! -d "$probe" ]; do probe=$(dirname "$probe"); done
        free_gib=$(df -g "$probe" | awk 'NR==2 { print $4 }')
        [ "${free_gib:-0}" -ge 16 ] || { fail "pulling q38 needs ~15 GB free (have ${free_gib} GB)"; finish; }
        run_gate "pull q38 (SHA-256 verified)" "$EVID/pull.log" env Q27_HOME="$home" packaging/bin/q27 pull q38
    else
        note "pack present: $model"
    fi
fi
for f in "$model" "$tok"; do
    case "$f" in /*) ;; *) f="$root/$f" ;; esac
    [ -f "$f" ] || { fail "missing $f"; finish; }
done
case "$model" in /*) ;; *) model="$root/$model" ;; esac
case "$tok" in /*) ;; *) tok="$root/$tok" ;; esac
note "model $model ($(du -h "$model" | cut -f1))"

# ---- 3. gates ----------------------------------------------------------------
run_gate "test-canonical-registry" "$EVID/canonical-registry.log" make test-canonical-registry
run_gate "test-metal-backend" "$EVID/metal-backend.log" make test-metal-backend
run_gate "qwen38 policy + canonical" "$EVID/qwen38.log" make test-metal-qwen38 "QWEN38_MODEL=$model" "QWEN38_TOKENIZER=$tok"
run_gate "qwen38 serving" "$EVID/qwen38-serving.log" make test-metal-qwen38-serving "QWEN38_MODEL=$model" "QWEN38_TOKENIZER=$tok"
run_gate "qwen38 native agent" "$EVID/qwen38-agent.log" make test-metal-qwen38-agent "QWEN38_MODEL=$model" "QWEN38_TOKENIZER=$tok"

# ---- 4. perf: first ceilings for this pack -----------------------------------
printf 'pack\tmetric\tceiling\n' > "$EVID/perf-ceilings-q38.tsv"
run_gate "perf ceilings recorded (q38)" "$EVID/perf-ceilings.log" \
    env Q27_PERF_CEILINGS="$EVID/perf-ceilings-q38.tsv" Q27_GATE_TOK="$tok" \
    tools/perf_ceilings.sh --ratchet "$model"
finish
