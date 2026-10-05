#!/usr/bin/env bash
# Hermetic tests for tools/perf_ceilings.sh and tools/perf_journeys.sh: a fake
# q27-metal prints controlled counters/timings, so every gate decision (pass,
# fail, ratchet, record, missing rows, early EOS) is checked without a GPU.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
work=$(mktemp -d "${TMPDIR:-/tmp}/q27-perf-gates-test.XXXXXX")
trap 'rm -rf "$work"' EXIT

# Fake CLI. Per generated token: FAKE_CB command buffers, FAKE_DISP dispatches,
# FAKE_SPT seconds; FAKE_EOS caps the generated count; FAKE_FAIL=1 exits 1;
# FAKE_SUFFIX_EOS caps suffix runs only; FAKE_NOCOUNT=1 omits the counters line. FAKE_SLOW_PACK slows one pack 10x.
cat > "$work/q27-metal" <<'SH'
#!/usr/bin/env bash
pack=$1; shift 2
n=1; counters=0; suffix=0
while [ $# -gt 0 ]; do
  case "$1" in -n) n=$2; shift ;; --counters) counters=1 ;; --suffix) suffix=1; shift ;; esac; shift
done
if [ "$suffix" = 1 ] && [ -n "${FAKE_SUFFIX_EOS:-}" ]; then
  # FAKE_SUFFIX_EOS_CALL=k: only the k-th suffix run stops early.
  calls=$(( $(cat "$FAKE_STATE/suffix_calls" 2>/dev/null || echo 0) + 1 ))
  echo "$calls" > "$FAKE_STATE/suffix_calls"
  [ "$calls" = "${FAKE_SUFFIX_EOS_CALL:-$calls}" ] && n=$FAKE_SUFFIX_EOS
fi
[ "${FAKE_FAIL:-0}" = 1 ] && { echo "fake failure" >&2; exit 1; }
[ -n "${FAKE_EOS:-}" ] && [ "$n" -gt "$FAKE_EOS" ] && n=$FAKE_EOS
spt=${FAKE_SPT:-0.1}
case "$pack" in *"${FAKE_SLOW_PACK:-none}"*) spt=$(awk -v s="$spt" 'BEGIN { print s * 10 }') ;; esac
echo "Metal model ready on Fake in 1.00 s (shader sha1 0)" >&2
echo "$n tokens in $(awk -v n="$n" -v s="$spt" 'BEGIN { printf "%.4f", 1 + n * s }') s (1.0 tok/s), position 1" >&2
if [ "$counters" = 1 ] && [ "${FAKE_NOCOUNT:-0}" != 1 ]; then
  echo "counters: prompt=1 generated=$n command_buffers=$(( 3 + n * ${FAKE_CB:-1} )) dispatches=$(( 100 + n * ${FAKE_DISP:-10} ))" >&2
fi
echo "generated:x"
SH
chmod +x "$work/q27-metal"
mkdir -p "$work/models"
: > "$work/models/fake-t2.q27"; : > "$work/models/fake-t3.q27"; : > "$work/tok"
export Q27_METAL_CLI="$work/q27-metal" Q27_GATE_TOK="$work/tok" FAKE_STATE="$work"
export Q27_PERF_CEILINGS="$work/ceilings.tsv" Q27_PERF_BASELINE="$work/baseline.tsv" Q27_PERF_REPS=2
export Q27_PERF_BAND=10
unset Q27_GATE_PACKS
T2="$work/models/fake-t2.q27" T3="$work/models/fake-t3.q27"
ceil="$root/tools/perf_ceilings.sh" jour="$root/tools/perf_journeys.sh"

pass=0
check() { # name expected-exit expected-output-regex cmd...
  local name=$1 want=$2 re=$3 out rc; shift 3
  if out=$("$@" 2>&1); then rc=0; else rc=$?; fi
  if [ "$rc" != "$want" ] || ! printf '%s\n' "$out" | grep -Eq -- "$re"; then
    echo "FAIL: $name (exit $rc, want $want; want /$re/)"; printf '%s\n' "$out" | sed 's/^/    /'; exit 1
  fi
  pass=$((pass + 1)); echo "ok: $name"
}

# ---- perf_ceilings.sh ----
check "no ceilings: fails" 1 "no ceiling; run --ratchet" "$ceil" "$T2" "$T3"
check "ratchet records" 0 "NEW +fake-t2 +suffix_copy48.dispatches" "$ceil" --ratchet "$T2" "$T3"
grep -q "fake-t3	suffix_copy48" "$work/ceilings.tsv" && { echo "FAIL: T3 has a suffix row"; exit 1; }
check "equal: pass" 0 "perf ceilings: PASS" "$ceil" "$T2" "$T3"
check "one more dispatch per token: fails" 1 "FAIL +fake-t2 +decode64.dispatches" env FAKE_DISP=11 "$ceil" "$T2"
check "lower: passes with a note" 0 "BELOW.*decode64.dispatches" env FAKE_DISP=9 "$ceil" "$T2"
check "ratchet lowers" 0 "LOWER.*decode64.dispatches" env FAKE_DISP=9 "$ceil" --ratchet "$T2"
check "after lowering, old count fails" 1 "FAIL +fake-t2 +decode64.dispatches" "$ceil" "$T2"
cp "$work/ceilings.tsv" "$work/ceilings.before"
check "ratchet never raises" 1 "FAIL" "$ceil" --ratchet "$T2"
cmp -s "$work/ceilings.tsv" "$work/ceilings.before" || { echo "FAIL: ratchet rewrote ceilings after a failure"; exit 1; }
printf 'fake-t2\tgone.metric\t5\n' >> "$work/ceilings.tsv"
check "ceiling row no workload measures: fails" 1 "gone.metric +not measured" env FAKE_DISP=9 "$ceil" "$T2"
check "unmeasured pack rows are left alone" 0 "PASS" "$ceil" "$T3"
check "cli failure fails" 1 "q27-metal failed" env FAKE_FAIL=1 "$ceil" "$T3"
check "missing counters line fails" 1 "no counters line" env FAKE_NOCOUNT=1 "$ceil" "$T3"
check "early EOS fails" 1 "stopped at" env FAKE_EOS=20 "$ceil" "$T3"
check "prefill generating nothing fails" 1 "prefill workload generated 0" env FAKE_EOS=0 "$ceil" "$T3"
check "empty pack list fails" 2 "no workloads measured" "$ceil" ""
check "early EOS on suffix copy fails" 1 "suffix workload stopped at 40" env FAKE_SUFFIX_EOS=40 "$ceil" "$T2"

# ---- perf_journeys.sh ----
check "no baseline: fails" 1 "no baseline" "$jour" "$T2"
check "record" 0 "baseline written" "$jour" --record "$T2" "$T3"
check "same speed: pass" 0 "perf journeys: PASS" "$jour" "$T2" "$T3"
check "2x slower: fails" 1 "FAIL +fake-t2 +decode_tps" env FAKE_SPT=0.2 "$jour" "$T2"
check "faster: passes" 0 "PASS" env FAKE_SPT=0.05 "$jour" "$T2"
check "record one pack keeps the other" 0 "baseline written" "$jour" --record "$T2"
grep -q "^fake-t3	decode_tps" "$work/baseline.tsv" || { echo "FAIL: --record dropped fake-t3"; exit 1; }
check "other pack still gated after partial record" 1 "FAIL +fake-t3" env FAKE_SLOW_PACK=t3 "$jour" "$T2" "$T3"
grep -v "^fake-t3	decode_tps" "$work/baseline.tsv" > "$work/b" && mv "$work/b" "$work/baseline.tsv"
check "metric missing from baseline: fails" 1 "FAIL +fake-t3 +decode_tps .*no baseline" "$jour" "$T3"
check "early EOS on suffix copy fails" 1 "expected 48" env FAKE_SUFFIX_EOS=40 "$jour" "$T2"
rm -f "$work/suffix_calls"
check "early EOS in a later repetition only fails" 1 "expected 48" env FAKE_SUFFIX_EOS=40 FAKE_SUFFIX_EOS_CALL=2 "$jour" "$T2"
check "cli failure fails" 1 "q27-metal failed" env FAKE_FAIL=1 "$jour" "$T2"
cp "$work/baseline.tsv" "$work/baseline.before"
check "empty pack list: record refuses" 2 "no journeys measured" "$jour" --record ""
cmp -s "$work/baseline.tsv" "$work/baseline.before" || { echo "FAIL: empty --record changed the baseline"; exit 1; }
check "empty pack list: check fails" 2 "no journeys measured" "$jour" ""
cp "$work/baseline.tsv" "$work/baseline.keep"
printf 'fake-t2\textra_s\t1.0\tlower\n' >> "$work/baseline.tsv"
check "baseline row no journey measures: fails" 1 "extra_s +not measured" "$jour" "$T2"
: > "$work/baseline.tsv"
check "empty baseline: fails" 2 "malformed" "$jour" "$T2"
mv "$work/baseline.keep" "$work/baseline.tsv"
cp "$work/ceilings.tsv" "$work/ceilings.keep"
: > "$work/ceilings.tsv"
check "empty ceilings: fails" 2 "malformed" "$ceil" "$T2"
check "empty ceilings: ratchet refuses too" 2 "malformed" "$ceil" --ratchet "$T2"
mv "$work/ceilings.keep" "$work/ceilings.tsv"

echo "perf gate tests: $pass passed"
