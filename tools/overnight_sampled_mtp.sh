#!/usr/bin/env bash
# Overnight / quiet-machine suite for Metal sampled MTP.
# Plan: docs/metal/plans/2026-07-21-metal-sampled-mtp.md (Phase 2 + perf A/B).
#
# Runs offline unit legs first (no model), then live legs on an official MTP pack.
# Designed to be left unattended: writes a timestamped OUTDIR with logs + SUMMARY.
#
# Usage (quiet machine, nothing else holding the model):
#   MODEL=models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
#   TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok \
#   tools/overnight_sampled_mtp.sh
#
# Optional env:
#   OUTDIR=logs/overnight-...   BIN=build/q27-metal
#   MTP=4  N_GATE=48  N_AB=96  N_TEMP=32  SEED=42  CTX=2048
#   SKIP_UNITS=1  SKIP_LIVE=1   # debug slices
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${BIN:-$ROOT/build/q27-metal}"
MODEL="${MODEL:-}"
TOK="${TOK:-}"
MTP="${MTP:-4}"
N_GATE="${N_GATE:-48}"
N_AB="${N_AB:-96}"
N_TEMP="${N_TEMP:-32}"
SEED="${SEED:-42}"
CTX="${CTX:-2048}"
PROMPT="${PROMPT:-Write a short Python function that returns the nth Fibonacci number with a docstring.}"
SKIP_UNITS="${SKIP_UNITS:-0}"
SKIP_LIVE="${SKIP_LIVE:-0}"

STAMP=$(date +%Y%m%d-%H%M%S)
OUTDIR="${OUTDIR:-$ROOT/logs/overnight-sampled-mtp-$STAMP}"
mkdir -p "$OUTDIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/run.log" >&2; }
die() { log "FAIL: $*"; echo "FAIL: $*" >>"$OUTDIR/SUMMARY.md"; exit 1; }

# ---- preamble ----
{
  echo "# Overnight sampled-MTP suite — $STAMP"
  echo
  echo "- host: $(uname -n) $(uname -m)"
  echo "- outdir: \`$OUTDIR\`"
  echo "- model: \`${MODEL:-unset}\`"
  echo "- bin: \`$BIN\`"
  echo "- mtp=$MTP ctx=$CTX seed=$SEED N_gate=$N_GATE N_ab=$N_AB N_temp=$N_TEMP"
  echo
  echo "## Legs"
  echo
} >"$OUTDIR/SUMMARY.md"

log "outdir=$OUTDIR"

# Refuse to start if another q27 engine process is holding weights (best-effort).
# Match process names, not path substrings (q27-agent-tui was a false positive).
if pgrep -x 'q27-metal-server' >/dev/null 2>&1 ||
   pgrep -x 'q27-agent' >/dev/null 2>&1 ||
   pgrep -f '(^|/)(build/)?q27-metal( |$)' >/dev/null 2>&1; then
  log "WARN: another q27 process may be running — live legs can thrash memory"
  echo "- **warn:** other q27 process detected at start" >>"$OUTDIR/SUMMARY.md"
fi

# ---- Leg 0: offline units (no model) ----
if [[ "$SKIP_UNITS" != "1" ]]; then
  log "Leg 0: unit tests"
  {
    make build/test_sampling build/test_metal_ops
    ./build/test_sampling
    ./build/test_metal_ops
  } >"$OUTDIR/00-units.out" 2>"$OUTDIR/00-units.err" || {
    tail -30 "$OUTDIR/00-units.err" | tee -a "$OUTDIR/run.log" >&2
    die "unit tests failed (see 00-units.err)"
  }
  echo "- [x] Leg 0 units PASS" >>"$OUTDIR/SUMMARY.md"
  log "Leg 0 PASS"
else
  echo "- [ ] Leg 0 units SKIPPED" >>"$OUTDIR/SUMMARY.md"
fi

if [[ "$SKIP_LIVE" == "1" ]]; then
  echo "- [ ] Live legs SKIPPED (SKIP_LIVE=1)" >>"$OUTDIR/SUMMARY.md"
  log "live skipped; done"
  exit 0
fi

[[ -x "$BIN" ]] || die "missing $BIN (make build/q27-metal)"
[[ -n "$MODEL" && -f "$MODEL" ]] || die "set MODEL= to official MTP .q27"
[[ -n "$TOK" && -f "$TOK" ]] || die "set TOK= to matching .tok"

# ---- Leg 1: correctness gate ----
log "Leg 1: metal_sampled_mtp_gate.sh"
export BIN MODEL TOK MTP N="$N_GATE" SEED CTX PROMPT
if ! tools/metal_sampled_mtp_gate.sh \
    >"$OUTDIR/01-gate.out" 2>"$OUTDIR/01-gate.err"; then
  tail -40 "$OUTDIR/01-gate.err" | tee -a "$OUTDIR/run.log" >&2
  die "gate failed (see 01-gate.err)"
fi
echo "- [x] Leg 1 gate PASS" >>"$OUTDIR/SUMMARY.md"
# Capture key lines
rg -n 'PASS:|FAIL:|speculation:|WARN:' "$OUTDIR/01-gate.err" \
  >>"$OUTDIR/SUMMARY.md" || true
log "Leg 1 PASS"

# Helper: run one CLI generation; append t/s line to SUMMARY.
# Args: label extra-args...
run_ab() {
  local label=$1; shift
  local out="$OUTDIR/ab-$label.out"
  local err="$OUTDIR/ab-$label.err"
  log "A/B: $label"
  set +e
  /usr/bin/time -p "$BIN" "$MODEL" "$TOK" --ctx "$CTX" --mtp "$MTP" \
    "$@" -n "$N_AB" --prompt "$PROMPT" >"$out" 2>"$err"
  local rc=$?
  set -e
  [[ $rc -eq 0 ]] || die "A/B $label failed exit=$rc"
  local tps
  tps=$(rg -o '[0-9.]+ tok/s' "$err" | head -1 || true)
  local spec
  spec=$(rg 'speculation:' "$err" | tail -1 || true)
  echo "- **$label**: ${tps:-n/a} — ${spec:-no spec line}" >>"$OUTDIR/SUMMARY.md"
  log "A/B $label: ${tps:-n/a}"
}

# ---- Leg 2: wall t/s A/B (quiet-machine claim surface) ----
log "Leg 2: wall t/s A/B (N=$N_AB)"
echo >>"$OUTDIR/SUMMARY.md"
echo "### Leg 2 wall t/s (N=$N_AB)" >>"$OUTDIR/SUMMARY.md"
run_ab greedy
run_ab sample-mtp --temperature 0.7 --top-p 0.95 --top-k 20 --seed "$SEED"
log "A/B: plain-sample"
set +e
env Q27_SAMPLE_PLAIN=1 /usr/bin/time -p "$BIN" "$MODEL" "$TOK" --ctx "$CTX" --mtp "$MTP" \
  --temperature 0.7 --top-p 0.95 --top-k 20 --seed "$SEED" \
  -n "$N_AB" --prompt "$PROMPT" >"$OUTDIR/ab-plain-sample.out" 2>"$OUTDIR/ab-plain-sample.err"
rc=$?
set -e
[[ $rc -eq 0 ]] || die "A/B plain-sample failed"
tps=$(rg -o '[0-9.]+ tok/s' "$OUTDIR/ab-plain-sample.err" | head -1 || true)
echo "- **plain-sample**: ${tps:-n/a} (Q27_SAMPLE_PLAIN=1)" >>"$OUTDIR/SUMMARY.md"
log "A/B plain-sample: ${tps:-n/a}"
echo "- [x] Leg 2 A/B completed" >>"$OUTDIR/SUMMARY.md"
log "Leg 2 done"

# ---- Leg 3: acceptance-vs-temp (longer) ----
log "Leg 3: acceptance vs temp (N=$N_TEMP)"
echo >>"$OUTDIR/SUMMARY.md"
echo "### Leg 3 acceptance vs temp (N=$N_TEMP)" >>"$OUTDIR/SUMMARY.md"
for T in 0.0 0.3 0.7 1.0 1.5; do
  err="$OUTDIR/temp-$T.err"
  out="$OUTDIR/temp-$T.out"
  set +e
  if [[ "$T" == "0.0" ]]; then
    env Q27_MTP_TRACE=1 "$BIN" "$MODEL" "$TOK" --ctx "$CTX" --mtp "$MTP" \
      -n "$N_TEMP" --prompt "$PROMPT" >"$out" 2>"$err"
  else
    env Q27_MTP_TRACE=1 "$BIN" "$MODEL" "$TOK" --ctx "$CTX" --mtp "$MTP" \
      --temperature "$T" --top-p 0.95 --top-k 20 --seed "$SEED" \
      -n "$N_TEMP" --prompt "$PROMPT" >"$out" 2>"$err"
  fi
  rc=$?
  set -e
  [[ $rc -eq 0 ]] || die "temp leg T=$T failed"
  spec=$(rg 'speculation:' "$err" | tail -1 || true)
  tps=$(rg -o '[0-9.]+ tok/s' "$err" | head -1 || true)
  rounds=$(rg -c 'mtp (sample )?round:' "$err" || true)
  echo "- T=$T: ${tps:-n/a}; rounds≈${rounds:-0}; $spec" >>"$OUTDIR/SUMMARY.md"
  log "T=$T ${tps:-n/a} $spec"
done
echo "- [x] Leg 3 acceptance-vs-temp done" >>"$OUTDIR/SUMMARY.md"

{
  echo
  echo "## Done"
  echo
  echo "Finished at $(date -Iseconds). Review \`$OUTDIR/SUMMARY.md\` and paste"
  echo "key rows into docs/metal/plans/2026-07-21-metal-sampled-mtp.md progress log."
} >>"$OUTDIR/SUMMARY.md"

log "ALL LEGS COMPLETE — see $OUTDIR/SUMMARY.md"
cat "$OUTDIR/SUMMARY.md" >&2
