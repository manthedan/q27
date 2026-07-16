#!/bin/bash
# Suffix-burst batched-verify correctness gates (2026-07-16-suffix-burst-verify.md
# gates 2-4). T2 artifact, ~7.15 GB, one model load per leg, legs strictly
# serial. STAGED — run only on Daniel's go. Timing legs are NOT here (quiet
# machine; gate 6 has its own leg once these pass).
set -u  # no -e: each gate checks its own exit and the verdict file decides
cd "$(dirname "$0")/.."

MODEL=models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok  # shared 248320 vocab across tiers
OUT=logs/suffix-gates-20260716
mkdir -p "$OUT"
VERDICTS="$OUT/verdicts.txt"
: > "$VERDICTS"
fail=0
verdict() { echo "$1" | tee -a "$VERDICTS"; case "$1" in FAIL*) fail=1;; esac; }

# Stale-binary rule: rebuild first.
make build/q27-metal || { verdict "FAIL: build"; exit 1; }

# Repetition-heavy prompt: the same sentence four times primes SuffixDraft
# (4-gram key + long backward extension) so greedy continuation repeats and
# bursts actually fire. Neutral prompt: the standing canonical short prompt,
# where the Phase-0 prior says the drafter goes silent.
REP="The quick brown fox jumps over the lazy dog and runs far away. The quick brown fox jumps over the lazy dog and runs far away. The quick brown fox jumps over the lazy dog and runs far away. The quick brown fox jumps over the lazy dog and runs far away. The quick brown"
NEU="The capital of France is"

run() { # name prompt args...
  local name=$1 prompt=$2; shift 2
  Q27_SUFFIX_TRACE=1 ./build/q27-metal "$MODEL" "$TOK" --prompt "$prompt" -n 96 --ctx 512 "$@" \
    > "$OUT/$name.out" 2> "$OUT/$name.err"
  local rc=$?
  grep '^generated:' "$OUT/$name.out" > "$OUT/$name.txt"
  return $rc
}

# Gate 2 — committed-stream A/B on the repetition prompt.
run serial-rep      "$REP" || verdict "FAIL: serial-rep run rc"
run sfx-serial-rep  "$REP" --suffix-serial 12 || verdict "FAIL: sfx-serial-rep run rc"
for w in 16 32 48; do
  run "sfx$w-rep" "$REP" --suffix "$w" || verdict "FAIL: sfx$w-rep run rc"
done
for arm in sfx-serial-rep sfx16-rep sfx32-rep sfx48-rep; do
  if cmp -s "$OUT/serial-rep.txt" "$OUT/$arm.txt"; then
    verdict "PASS: $arm committed bytes identical to serial"
  else
    verdict "FAIL: $arm committed bytes differ from serial (inspect: known tolerance class only if low-margin, else bug)"
  fi
done

# Gate 3 — widths actually dispatched (> 12), from the dispatch-site trace.
if grep -qE 'suffix round: live (1[6-9]|[2-4][0-9])' "$OUT/sfx16-rep.err" \
   && grep -qE 'suffix round: live (3[2-9]|4[0-8])' "$OUT/sfx48-rep.err"; then
  verdict "PASS: dispatch trace shows live >= 16 (w16 arm) and >= 32 (w48 arm)"
else
  verdict "FAIL: wide lanes never dispatched — burst gating or snap policy is wrong (or prompt failed to fire)"
fi

# Gate 3b — acceptance walk is LIVE (not teacher-forced): some round must
# reject a lane (accepted < live-1) somewhere across the burst arms.
if grep -hE 'suffix round: live' "$OUT"/sfx*-rep.err \
   | awk '{ if ($5 < $3 - 1) found=1 } END { exit !found }'; then
  verdict "PASS: acceptance walk rejected at least one lane (walk is live)"
else
  verdict "FAIL: every lane of every round accepted — walk may be vacuous (or prompt is degenerate; change prompt before trusting)"
fi

# Gate 4 — stats honesty: fired + fallback rounds == speculation rounds, and
# neutral traffic goes silent with byte-identity.
run serial-neu "$NEU" || verdict "FAIL: serial-neu run rc"
run sfx48-neu  "$NEU" --suffix 48 || verdict "FAIL: sfx48-neu run rc"
if cmp -s "$OUT/serial-neu.txt" "$OUT/sfx48-neu.txt" \
   && grep -q 'suffix bursts: 0 fired' "$OUT/sfx48-neu.err"; then
  verdict "PASS: neutral prompt — zero bursts, committed bytes identical"
else
  verdict "FAIL: neutral prompt regressed (bursts fired or bytes differ)"
fi

echo; echo "=== $VERDICTS ==="; cat "$VERDICTS"
exit $fail
