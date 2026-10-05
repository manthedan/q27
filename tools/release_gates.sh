#!/usr/bin/env bash
# release_gates.sh <A|B|C> [evidence-dir] — run a QA_BEFORE_RELEASES.md risk
# tier for the Bonsai 2 release line, capture evidence, print the ledger.
# Gates run strictly one after another: never two resident models (or two
# GPU jobs) at once. bash 3.2-safe.
#
#   C  hermetic: CPU suites, repack, packaging, shader discovery, TUI tests
#   B  C + Metal backend suite + live Bonsai 2 gates on every pack
#   A  B + independent Prism reference oracle (needs PRISM_DIR) + session soak
#
# Packs: Q27_GATE_PACKS (space-separated .q27 paths), default the t2-slim and
# t3-slim packs under models/bonsai2/; tokenizer Q27_GATE_TOK.
set -u
cd "$(dirname "$0")/.."

TIER=${1:-}
case "$TIER" in A|B|C) ;; *) echo "usage: tools/release_gates.sh A|B|C [evidence-dir]" >&2; exit 2;; esac
EVID=${2:-/tmp/q27-release-gates-$(date +%Y%m%d-%H%M)}
mkdir -p "$EVID"
LEDGER="$EVID/ledger.txt"
: > "$LEDGER"

PACKS=${Q27_GATE_PACKS:-"models/bonsai2/bonsai2-27b-t2-slim.q27 models/bonsai2/bonsai2-27b-t3-slim.q27"}
TOK=${Q27_GATE_TOK:-models/bonsai2/bonsai2.tok}
PY=${PYTHON:-.venv/bin/python}
fails=0

pass() { echo "PASS: $*" | tee -a "$LEDGER"; }
fail() { echo "FAIL: $*" | tee -a "$LEDGER"; fails=$((fails+1)); }
skip() { echo "SKIP: $*" | tee -a "$LEDGER"; }
note() { echo "note: $*" | tee -a "$LEDGER"; }

run_gate() { # name logfile cmd...  — nonzero exit = gate failure
    local name=$1 log=$2; shift 2
    echo "== $name (log: $log)"
    if caffeinate -i "$@" > "$log" 2>&1; then pass "$name"; else fail "$name (log: $log)"; fi
}

finish() {
    echo "----"
    cat "$LEDGER"
    echo "evidence: $EVID"
    [ "$fails" -eq 0 ] || exit 1
    exit 0
}

echo "release gates tier $TIER — evidence: $EVID"
note "commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git status --short -- src/ experiments/ packaging/ tools/ Makefile 2>/dev/null)" ]; then
    note "working tree dirty under src/experiments/packaging/tools/Makefile"
fi

# ---- Tier C: hermetic --------------------------------------------------------
run_gate "test-tools" "$EVID/test-tools.log" make test-tools
run_gate "test-inspect" "$EVID/test-inspect.log" make test-inspect
run_gate "test-agent" "$EVID/test-agent.log" make test-agent
run_gate "test-repack" "$EVID/test-repack.log" make test-repack test-bonsai2-repack PYTHON="$PY"
run_gate "test-packaging" "$EVID/test-packaging.log" make test-packaging
run_gate "test-shader-discovery" "$EVID/test-shader-discovery.log" make test-shader-discovery
run_gate "tui-cargo-test" "$EVID/tui.log" \
    cargo test --manifest-path experiments/q27-tui/Cargo.toml --locked
[ "$TIER" = "C" ] && finish

# ---- Tier B: Metal + live Bonsai 2 gates -------------------------------------
[ -f "$TOK" ] || { fail "tokenizer missing: $TOK"; finish; }
run_gate "test-metal-backend" "$EVID/test-metal-backend.log" make test-metal-backend
make build/q27-metal-server build/q27-metal-server-test agent >"$EVID/build.log" 2>&1 ||
    { fail "build (log: $EVID/build.log)"; finish; }
for pack in $PACKS; do
    tag=$(basename "$pack" .q27)
    [ -f "$pack" ] || { fail "$tag: pack missing ($pack)"; continue; }
    run_gate "$tag serving" "$EVID/$tag-serving.log" \
        python3 tools/test_bonsai2_serving.py build/q27-metal-server "$pack" "$TOK"
    run_gate "$tag native" "$EVID/$tag-native.log" \
        python3 tools/test_bonsai2_native.py build/q27-agent "$pack" "$TOK"
    run_gate "$tag recovery" "$EVID/$tag-recovery.log" \
        make test-metal-recovery MODEL="$pack" TOKENIZER="$TOK"
    run_gate "$tag snapshot-reuse" "$EVID/$tag-snapshot-reuse.log" \
        python3 tools/test_bonsai2_snapshot_reuse.py build/q27-metal-server "$pack" "$TOK"
done
[ "$TIER" = "B" ] && finish

# ---- Tier A: reference oracle + soak -----------------------------------------
if [ -n "${PRISM_DIR:-}" ]; then
    run_gate "prism-oracle" "$EVID/prism-oracle.log" env PRISM_DIR="$PRISM_DIR" tools/bonsai2_gate.sh
else
    skip "prism-oracle (set PRISM_DIR to a pinned Prism build; see docs/metal/BONSAI2.md)"
fi
first_pack=${PACKS%% *}
run_gate "session-soak" "$EVID/session-soak.log" \
    python3 tools/test_bonsai2_session_soak.py build/q27-agent "$first_pack" "$TOK" 16384 8
finish
