#!/usr/bin/env bash
# Sampled-MTP vs plain-sampling distribution gate (codex branch-review P2;
# docs/metal/plans/2026-07-21-metal-sampled-mtp.md pending row).
#
# The correctness claim of sampled MTP is distributional: its output must
# match plain sample_from_logits. Unit tests cover the rejection math on
# synthetic distributions; this script checks the END-TO-END claim on a
# real pack: lane mapping, EOS/commit handling, pending-logit state, and
# top-k row offsets can only break it here.
#
# Method: three legs, same prompt + recipe.
#   plainA (Q27_SAMPLE_PLAIN, seed A)   — reference
#   plainB (Q27_SAMPLE_PLAIN, seed B)   — sampling-noise floor
#   mtp    (sampled MTP,       seed C)  — path under test
# Unigram token histograms over generated ids; total-variation distance.
# Verdict: TV(mtp, plainA) <= max(1.5 * TV(plainA, plainB), TV + 0.05).
# This is smoke-strength (N ~ 1024/leg): it catches gross distribution
# breaks (wrong accept mass, row-offset bugs), not small biases; the full
# seed x top-p x top-k matrix stays an overnight item.
#
# Usage:
#   MODEL=path.q27 TOK=path.tok tools/metal_sampled_mtp_dist.sh
#   N=1024 MTP=4 CTX=2048 TEMP=0.7 TOPP=0.95 TOPK=20 PROMPT="..." (defaults)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$ROOT/build/q27-metal}"
MODEL="${MODEL:-}"
TOK="${TOK:-}"
N="${N:-1024}"
MTP="${MTP:-4}"
CTX="${CTX:-2048}"
TEMP="${TEMP:-0.7}"
TOPP="${TOPP:-0.95}"
TOPK="${TOPK:-20}"
PROMPT="${PROMPT:-Write a short Python module with five small utility functions and a main block that exercises them.}"

die() { echo "metal_sampled_mtp_dist: $*" >&2; exit 1; }
[[ -x "$BIN" ]] || die "missing binary $BIN"
[[ -n "$MODEL" && -f "$MODEL" ]] || die "set MODEL= to an official MTP pack"
[[ -n "$TOK" && -f "$TOK" ]] || die "set TOK= to the matching .tok"
command -v python3 >/dev/null || die "python3 required"

OUT="$(mktemp -d "${TMPDIR:-/tmp}/q27-dist.XXXXXX")"
trap 'rm -rf "$OUT"' EXIT

run_leg() { # label seed extra-env...
  local label=$1 seed=$2; shift 2
  echo "== leg $label (N=$N seed=$seed) ==" >&2
  env "$@" "$BIN" "$MODEL" "$TOK" --ctx "$CTX" --mtp "$MTP" \
    --temperature "$TEMP" --top-p "$TOPP" --top-k "$TOPK" --seed "$seed" \
    -n "$N" --prompt "$PROMPT" --dump-token-ids "$OUT/$label.ids" \
    >"$OUT/$label.out" 2>"$OUT/$label.err" \
    || die "leg $label failed: $(tail -2 "$OUT/$label.err")"
  [[ -s "$OUT/$label.ids" ]] || die "leg $label produced no token ids"
}

# --dump-token-ids writes the generated id stream (whitespace-separated).
command -v rg >/dev/null && rg -q "dump-token-ids" "$ROOT/src/metal/metal_cli.cpp" \
  || die "this q27-metal lacks --dump-token-ids (add the flag or update this script)"

run_leg plainA 101 Q27_SAMPLE_PLAIN=1
run_leg plainB 202 Q27_SAMPLE_PLAIN=1
run_leg mtp    303 Q27_MTP_TRACE=1
grep -q "mtp sample round:" "$OUT/mtp.err" \
  || die "mtp leg produced no sample rounds (route fell back to plain?)"

python3 - "$OUT" "$N" <<'PY'
import sys, collections, re

out, n = sys.argv[1], int(sys.argv[2])
def ids(leg):
    text = open(f"{out}/{leg}.ids").read()
    return [int(t) for t in re.findall(r"\d+", text)][:n]

def hist(a):
    c = collections.Counter(a)
    total = float(len(a))
    return {k: v / total for k, v in c.items()}

def tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)

legs = {l: ids(l) for l in ("plainA", "plainB", "mtp")}
for l, a in legs.items():
    print(f"{l}: {len(a)} tokens, {len(set(a))} distinct", file=sys.stderr)
h = {l: hist(a) for l, a in legs.items()}
floor = tv(h["plainA"], h["plainB"])
test  = tv(h["plainA"], h["mtp"])
bar   = max(1.5 * floor, floor + 0.05)
print(f"TV(plainA,plainB)={floor:.4f}  TV(plainA,mtp)={test:.4f}  bar={bar:.4f}")
if test <= bar:
    print("DIST-GATE: PASS (mtp within plain-vs-plain noise)")
    sys.exit(0)
print("DIST-GATE: FAIL (mtp distribution shifted beyond noise floor)", file=sys.stderr)
sys.exit(1)
PY
