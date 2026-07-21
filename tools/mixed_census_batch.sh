#!/bin/zsh
# Mixed weight-tier census batch (docs/metal/plans/2026-07-17-mixed-tier-census.md):
# baselines (all-B1, all-T2) then one class-flip arm at a time — build the
# mixed pack with q27_mix.py, validate, 8K wikitext NLL, DELETE the pack
# (13 GiB free on the mini; one arm pack exists at any moment). Resumable:
# arms with a completed log are skipped, so a crashed batch re-runs only
# what is missing. One model load at a time by construction (serial loop).
set -u
set -o pipefail
DRIVER=${0:A}  # save before entering fingerprint(); zsh sets $0 to the function name there
cd "$(dirname "$DRIVER")/.."
[ -z "${SWEEP_CAFF:-}" ] && exec env SWEEP_CAFF=1 caffeinate -i "$DRIVER" "$@"

BIN=build/q27-metal
B1=models/bonsai-27b-b1/bonsai-27b-b1.q27
T2=models/ternary-bonsai-27b/ternary-bonsai-27b-t2.q27
TOK=models/qwen36-27b-mtp/qwen36-27b-mtp.tok
C=data/wikitext2-test.tokens.bin
OUT=logs/mixed_census
ARM_PACK="models/census_arm.$$.q27"  # run-unique transient
# Shared across ALL NLL batch drivers (census, combo, future): the lock
# guards exclusive Metal/measurement access, not just this results dir —
# two different drivers passing their own pgrep checks pre-launch could
# otherwise run concurrently and contaminate both experiments.
LOCK="logs/q27_metal_batch.lock"
mkdir -p "$OUT"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "census: $LOCK exists; another q27 batch driver is running (remove only after verifying it is stale)"
    exit 1
fi
echo "$$" > "$LOCK/pid"
cleanup() { rm -f "$ARM_PACK"; rm -rf "$LOCK"; }
trap cleanup EXIT

if pgrep -x q27-metal >/dev/null 2>&1 || pgrep -x q27-metal-server >/dev/null 2>&1; then
    echo "census: another q27 Metal process is running; refusing to start"; exit 1
fi

export Q27_METAL_GEMM_HALF=1 Q27_METAL_GQA_TILE=2
export Q27_METAL_GQA_THRESHOLD=2048 Q27_METAL_GQA_BLOCK=1024

# Same-machine/driver identity for resume. Hash the platform UUID rather
# than writing it to a committed fingerprint; OS build + GPU identify the
# Metal runtime available to this binary. Every probe fails closed: an
# empty hash must never turn "same unknown Mac" into a valid resume.
platform_fail() {
    echo "census: cannot establish machine/Metal runtime identity" | tee "$OUT/ABORTED"
    exit 1
}
HW_MODEL=$(sysctl -n hw.model 2>/dev/null) || platform_fail
[ -n "$HW_MODEL" ] || platform_fail
HW_UUID=$(ioreg -rd1 -c IOPlatformExpertDevice 2>/dev/null |
    awk -F'"' '/IOPlatformUUID/{print $(NF-1); exit}') || platform_fail
[ -n "$HW_UUID" ] || platform_fail
HW_HASH=$(printf '%s' "$HW_UUID" | shasum -a 256 | awk '{print $1}') || platform_fail
case "$HW_HASH" in (''|*[!0-9a-f]*) platform_fail;; esac
[ "${#HW_HASH}" = 64 ] || platform_fail
unset HW_UUID
OS_VERSION=$(sw_vers -productVersion 2>/dev/null) || platform_fail
OS_VERSION_BUILD=$(sw_vers -buildVersion 2>/dev/null) || platform_fail
[ -n "$OS_VERSION" ] && [ -n "$OS_VERSION_BUILD" ] || platform_fail
OS_BUILD="$OS_VERSION-$OS_VERSION_BUILD"
METAL_GPU=$(system_profiler SPDisplaysDataType 2>/dev/null |
    awk -F': ' '/Chipset Model/{print $2; exit}') || platform_fail
[ -n "$METAL_GPU" ] || platform_fail

# Run the engine under a clean environment so inherited Q27/Metal debug,
# codec, shader-override, or residency knobs cannot redefine an arm. The
# shader path is consequently the checkout file hashed below.
q27() {
    env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
        Q27_METAL_GEMM_HALF="$Q27_METAL_GEMM_HALF" \
        Q27_METAL_GQA_TILE="$Q27_METAL_GQA_TILE" \
        Q27_METAL_GQA_THRESHOLD="$Q27_METAL_GQA_THRESHOLD" \
        Q27_METAL_GQA_BLOCK="$Q27_METAL_GQA_BLOCK" \
        "$BIN" "$@"
}

# Stale-binary rule (2026-07-15 lesson): rebuild before any measurement.
make build/q27-metal > "$OUT/rebuild.log" 2>&1 || {
    echo "census: rebuild failed (see $OUT/rebuild.log)" | tee "$OUT/ABORTED"; exit 1; }

# Resumability must never combine arms from different binaries, artifacts,
# corpora, route pins, or driver revisions into one readout. Include both
# HEAD and this script's bytes: the latter catches an uncommitted edit too.
fingerprint() {
    local head driver mixer host shader b1 t2 tok corpus
    head=$(git rev-parse HEAD 2>/dev/null) || return 1
    driver=$(md5 -q "$DRIVER") || return 1
    mixer=$(md5 -q tools/q27_mix.py) || return 1
    host=$(md5 -q "$BIN") || return 1
    shader=$(md5 -q src/metal/q27_kernels.metal) || return 1
    b1=$(md5 -q "$B1") || return 1
    t2=$(md5 -q "$T2") || return 1
    tok=$(md5 -q "$TOK") || return 1
    corpus=$(md5 -q "$C") || return 1
    echo "$head driver=$driver mixer=$mixer host=$host shader=$shader b1=$b1 t2=$t2 tok=$tok corpus=$corpus hw_model=$HW_MODEL hw_hash=$HW_HASH os_build=$OS_BUILD metal_gpu=$METAL_GPU clean_env=1 gemm_half=$Q27_METAL_GEMM_HALF tile=$Q27_METAL_GQA_TILE thr=$Q27_METAL_GQA_THRESHOLD blk=$Q27_METAL_GQA_BLOCK nll_long=8192 ctx=8192"
}
FP=$(fingerprint) ||
    { echo "census: cannot fingerprint a required input" | tee "$OUT/ABORTED"; exit 1; }
if [ -f "$OUT/fingerprint" ]; then
    [ "$(cat "$OUT/fingerprint")" = "$FP" ] ||
        { echo "census: fingerprint mismatch — move $OUT aside" | tee "$OUT/ABORTED"; exit 1; }
else
    # rebuild.log is produced above; any other file means this is not a
    # fresh experiment and must not be blessed retroactively.
    stale=$(find "$OUT" -type f ! -name rebuild.log -print -quit)
    [ -z "$stale" ] ||
        { echo "census: $OUT has unfingerprinted results; move it aside" | tee "$OUT/ABORTED"; exit 1; }
    echo "$FP" > "$OUT/fingerprint"
fi
rm -f "$OUT/ABORTED"

# The shader and mixer are read at runtime and each child reopens every
# artifact. Recheck the complete identity around every measurement so an
# edit during this multi-day process cannot silently split the experiment.
check_fingerprint() {
    local now
    now=$(fingerprint) ||
        { echo "census: cannot fingerprint a required input" | tee "$OUT/ABORTED"; exit 1; }
    [ "$now" = "$FP" ] ||
        { echo "census: input changed while batch was running — aborting" | tee "$OUT/ABORTED"; exit 1; }
}

file_md5() {
    md5 -q "$1" || { echo "census: cannot hash $1" >&2; return 1; }
}

valid_nll_log() {
    python3 - "$1" <<'PY'
import math, re, sys
text = open(sys.argv[1], errors="replace").read()
values = re.findall(r"overall mean NLL\s+(\S+)", text)
if len(values) != 1:
    raise SystemExit(1)
try:
    value = float(values[0])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) else 1)
PY
}

nll_run() {
    local model=$1 log=$2 expected_md5=$3
    local tmp="$log.tmp" actual_md5 rc
    # Only the atomic final pathname means complete: it is published after
    # a zero exit, one finite NLL, and global + exact-model post-run hashes.
    [ -s "$log" ] && valid_nll_log "$log" && return 0
    rm -f "$tmp"
    for attempt in 1 2; do
        check_fingerprint
        q27 "$model" "$TOK" --nll "$C" --nll-long 8192 --ctx 8192 > "$tmp" 2>&1
        rc=$?
        if [ "$rc" = 0 ] && valid_nll_log "$tmp"; then
            check_fingerprint
            actual_md5=$(file_md5 "$model") ||
                { echo "census: post-run model hash failed" | tee "$OUT/ABORTED"; exit 1; }
            [ "$actual_md5" = "$expected_md5" ] ||
                { echo "census: model changed during $(basename "$log")" | tee "$OUT/ABORTED"; exit 1; }
            mv "$tmp" "$log" ||
                { echo "census: cannot publish $(basename "$log")" | tee "$OUT/ABORTED"; exit 1; }
            return 0
        fi
        if [ "$attempt" = 1 ]; then
            mv "$tmp" "$log.attempt1" ||
                { echo "census: cannot preserve failed $(basename "$log")" | tee "$OUT/ABORTED"; exit 1; }
            echo "census: $(basename "$log") attempt 1 failed (exit $rc or invalid NLL); retrying once"
        fi
    done
    mv "$tmp" "$log.failed" 2>/dev/null || true
    echo "census: NLL run FAILED twice (see $log.failed)" | tee "$OUT/ABORTED"; exit 1
}

# Baselines anchor the gap on THIS box and binary (the recorded 1.055
# B1/T2 ratio is a 24 GB M4 number; ratios must be same-machine).
B1_MD5=$(file_md5 "$B1") || { echo "census: B1 hash failed" | tee "$OUT/ABORTED"; exit 1; }
T2_MD5=$(file_md5 "$T2") || { echo "census: T2 hash failed" | tee "$OUT/ABORTED"; exit 1; }
nll_run "$B1" "$OUT/base_b1.log" "$B1_MD5"
nll_run "$T2" "$OUT/base_t2.log" "$T2_MD5"

arm() {
    local name=$1 take=$2
    local log="$OUT/arm_$name.log"
    [ -s "$log" ] && valid_nll_log "$log" && return 0
    check_fingerprint
    python3 tools/q27_mix.py "$B1" "$T2" "$ARM_PACK" --take "$take" \
        > "$OUT/mix_$name.log" 2>&1 ||
        { echo "census: mix FAILED for $name" | tee "$OUT/ABORTED"; exit 1; }
    local arm_md5
    arm_md5=$(md5 -q "$ARM_PACK") ||
        { echo "census: cannot fingerprint transient pack for $name" | tee "$OUT/ABORTED"; exit 1; }
    q27 "$ARM_PACK" "$TOK" --validate-only --ctx 8 >> "$OUT/mix_$name.log" 2>&1 ||
        { echo "census: validate FAILED for $name" | tee "$OUT/ABORTED"; exit 1; }
    check_fingerprint
    [ "$(md5 -q "$ARM_PACK")" = "$arm_md5" ] ||
        { echo "census: transient pack changed during validation for $name" | tee "$OUT/ABORTED"; exit 1; }
    nll_run "$ARM_PACK" "$log" "$arm_md5"
    rm -f "$ARM_PACK"
}

# Depth bands over the 64 blocks (thirds).
E='([0-9]|1[0-9]|20)'      # blk 0-20
M='(2[1-9]|3[0-9]|4[0-2])' # blk 21-42
L='(4[3-9]|5[0-9]|6[0-3])' # blk 43-63

# Attention classes x bands (attention layers are blk 3,7,...,63).
arm attnq_early  "blk\\.$E\\.attn_q\\.weight"
arm attnq_mid    "blk\\.$M\\.attn_q\\.weight"
arm attnq_late   "blk\\.$L\\.attn_q\\.weight"
arm attnkv_early "blk\\.$E\\.attn_(k|v)\\.weight"
arm attnkv_mid   "blk\\.$M\\.attn_(k|v)\\.weight"
arm attnkv_late  "blk\\.$L\\.attn_(k|v)\\.weight"
arm attnout_early "blk\\.$E\\.attn_output\\.weight"
arm attnout_mid   "blk\\.$M\\.attn_output\\.weight"
arm attnout_late  "blk\\.$L\\.attn_output\\.weight"
# FFN classes x bands (every block). Gate and up are separate: their
# combined 13.9-14.6% byte cost exceeded the census's <=10% eligibility bar.
arm ffngate_early "blk\\.$E\\.ffn_gate\\.weight"
arm ffngate_mid   "blk\\.$M\\.ffn_gate\\.weight"
arm ffngate_late  "blk\\.$L\\.ffn_gate\\.weight"
arm ffnup_early   "blk\\.$E\\.ffn_up\\.weight"
arm ffnup_mid     "blk\\.$M\\.ffn_up\\.weight"
arm ffnup_late    "blk\\.$L\\.ffn_up\\.weight"
arm ffndown_early "blk\\.$E\\.ffn_down\\.weight"
arm ffndown_mid   "blk\\.$M\\.ffn_down\\.weight"
arm ffndown_late  "blk\\.$L\\.ffn_down\\.weight"
# GDN classes, full depth (48 GDN blocks; small tensors or single class).
arm gdn_qkv      "attn_qkv\\.weight"
arm gdn_gate     "attn_gate\\.weight"
arm gdn_out      "ssm_out\\.weight"
arm gdn_alphabeta "ssm_(alpha|beta)\\.weight"
# These special matrices were called "tier-pinned" in the sketch, but the
# source artifacts actually differ B1->T2 and each costs 4.7% of the full
# delta. Test separately; both are eligible under the <=10% census gate.
arm embedding   '^token_embd\.weight$'
arm output_head '^output\.weight$'
# The siblings' same-dtype tensors are not byte-identical. Test them as one
# zero-byte-delta cohort so their contribution cannot be misclassified as a
# diffuse quantized-matrix gap; zoom by class only if this cohort indicts.
arm shared_f32 '(^output_norm\.weight$|\.(attn_norm|post_attention_norm|attn_q_norm|attn_k_norm|ssm_norm)\.weight$|\.ssm_(a|dt\.bias|conv1d\.weight)$)'

# Summary table: arm, overall NLL, delta vs B1 base, gap recovered.
check_fingerprint
SUMMARY_TMP="$OUT/census_summary.txt.tmp"
python3 - "$OUT" <<'PY' | tee "$SUMMARY_TMP"
import glob, math, os, re, sys
out = sys.argv[1]
def nll(path):
    values = re.findall(r"overall mean NLL\s+(\S+)", open(path, errors="replace").read())
    if len(values) != 1:
        raise RuntimeError(f"{path}: expected exactly one overall NLL")
    value = float(values[0])
    if not math.isfinite(value):
        raise RuntimeError(f"{path}: non-finite overall NLL")
    return value
b1, t2 = nll(f"{out}/base_b1.log"), nll(f"{out}/base_t2.log")
gap = b1 - t2
if gap <= 0:
    raise RuntimeError(f"expected positive B1-T2 gap, got {gap}")
print(f"base B1 {b1:.4f}  base T2 {t2:.4f}  gap {gap:.4f} nats")
files = sorted(glob.glob(f"{out}/arm_*.log"))
if len(files) != 25:
    raise RuntimeError(f"expected 25 arm logs, found {len(files)}")
rows = []
for f in files:
    name = os.path.basename(f)[4:-4]
    v = nll(f)
    mix = f"{out}/mix_{name}.log"
    matches = re.findall(r"byte delta ([+-]?[0-9.]+) MB", open(mix).read())
    if len(matches) != 1:
        raise RuntimeError(f"{mix}: expected exactly one byte delta")
    take_mb = float(matches[0])
    rows.append((name, v, (b1 - v) / gap, take_mb))
rows.sort(key=lambda r: -r[2])
for name, v, rec, mb in rows:
    print(f"{name:16s} NLL {v:.4f}  gap recovered {100*rec:5.1f}%  bytes {mb:+8.1f} MB")
PY
summary_rc=$?
[ "$summary_rc" = 0 ] ||
    { echo "census: summary generation FAILED" | tee "$OUT/ABORTED"; exit 1; }
mv "$SUMMARY_TMP" "$OUT/census_summary.txt" ||
    { echo "census: cannot publish summary" | tee "$OUT/ABORTED"; exit 1; }
echo "census batch: COMPLETE"
