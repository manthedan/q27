#!/usr/bin/env bash
# Metal sampled-MTP smoke/gate (Phase 1d / Phase 2 of
# docs/metal/plans/2026-07-21-metal-sampled-mtp.md).
#
# Requires an official MTP pack (has blk.64). Bonsai will not exercise MTP.
# Usage:
#   MODEL=path/to.q27 TOK=path/to.tok tools/metal_sampled_mtp_gate.sh
# Optional:
#   BIN=build/q27-metal  MTP=4  N=48  SEED=42  CTX=2048
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$ROOT/build/q27-metal}"
MODEL="${MODEL:-}"
TOK="${TOK:-}"
MTP="${MTP:-4}"
N="${N:-48}"
SEED="${SEED:-42}"
CTX="${CTX:-2048}"
PROMPT="${PROMPT:-Write a short Python function that returns the nth Fibonacci number.}"

die() { echo "metal_sampled_mtp_gate: $*" >&2; exit 1; }

[[ -x "$BIN" ]] || die "missing binary $BIN (make build/q27-metal)"
[[ -n "$MODEL" && -f "$MODEL" ]] || die "set MODEL= to an official MTP .q27 pack"
[[ -n "$TOK" && -f "$TOK" ]] || die "set TOK= to the matching .tok"

run() {
  local label=$1; shift
  echo "=== $label ===" >&2
  "$@"
}

# 1) Seeded identity: same seed → identical generated line
out1=$(mktemp); out2=$(mktemp)
run "seeded identity A" "$BIN" "$MODEL" "$TOK" --ctx "$CTX" --mtp "$MTP" \
  --temperature 0.7 --top-p 0.95 --top-k 20 --seed "$SEED" -n "$N" \
  --prompt "$PROMPT" >"$out1"
run "seeded identity B" "$BIN" "$MODEL" "$TOK" --ctx "$CTX" --mtp "$MTP" \
  --temperature 0.7 --top-p 0.95 --top-k 20 --seed "$SEED" -n "$N" \
  --prompt "$PROMPT" >"$out2"
if ! diff -u "$out1" "$out2" >/dev/null; then
  echo "FAIL: seeded identity diverged" >&2
  diff -u "$out1" "$out2" >&2 || true
  rm -f "$out1" "$out2"
  exit 1
fi
echo "PASS: seeded identity" >&2

# 2) Seed varies (different seed should usually differ; warn-only if equal)
out3=$(mktemp)
run "seed varies" "$BIN" "$MODEL" "$TOK" --ctx "$CTX" --mtp "$MTP" \
  --temperature 0.7 --top-p 0.95 --top-k 20 --seed $((SEED+1)) -n "$N" \
  --prompt "$PROMPT" >"$out3"
if cmp -s "$out1" "$out3"; then
  echo "WARN: seed+1 matched seed (possible but rare at N=$N)" >&2
else
  echo "PASS: seed varies" >&2
fi

# 3) Sampled != greedy (temp 0) — expect different streams on most prompts
outg=$(mktemp)
run "greedy mtp control" "$BIN" "$MODEL" "$TOK" --ctx "$CTX" --mtp "$MTP" \
  -n "$N" --prompt "$PROMPT" >"$outg"
if cmp -s "$out1" "$outg"; then
  echo "WARN: sampled matched greedy (unusual at T=0.7)" >&2
else
  echo "PASS: sampled != greedy" >&2
fi

# 4) Q27_SAMPLE_PLAIN A/B both produce trajectories (no crash)
outp=$(mktemp)
run "plain sample force" env Q27_SAMPLE_PLAIN=1 "$BIN" "$MODEL" "$TOK" --ctx "$CTX" --mtp "$MTP" \
  --temperature 0.7 --top-p 0.95 --top-k 20 --seed "$SEED" -n "$N" \
  --prompt "$PROMPT" >"$outp"
grep -q '^generated:' "$outp" || die "plain sample produced no generated: line"
echo "PASS: Q27_SAMPLE_PLAIN trajectory" >&2

# 5) Acceptance telemetry (tokens/round) via Q27_MTP_TRACE on a short run
echo "=== acceptance-vs-temp (trace, short) ===" >&2
for T in 0.0 0.3 0.7 1.0; do
  if [[ "$T" == "0.0" ]]; then
    tr=$(env Q27_MTP_TRACE=1 "$BIN" "$MODEL" "$TOK" --ctx "$CTX" --mtp "$MTP" \
      -n 32 --prompt "$PROMPT" 2>&1 >/dev/null | tee /dev/stderr | rg -c 'mtp round:' || true)
    echo "T=$T greedy mtp_round lines: ${tr:-0}" >&2
  else
    tr=$(env Q27_MTP_TRACE=1 "$BIN" "$MODEL" "$TOK" --ctx "$CTX" --mtp "$MTP" \
      --temperature "$T" --top-p 0.95 --top-k 20 --seed "$SEED" -n 32 \
      --prompt "$PROMPT" 2>&1 >/dev/null | tee /dev/stderr | rg -c 'mtp sample round:' || true)
    echo "T=$T mtp_sample_round lines: ${tr:-0}" >&2
  fi
done

rm -f "$out1" "$out2" "$out3" "$outg" "$outp"
echo "metal_sampled_mtp_gate: PASS (smoke)" >&2
