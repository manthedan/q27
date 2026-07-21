#!/bin/zsh
# Mixed-tier COMBO batch (docs/metal/plans/2026-07-17-mixed-tier-census.md, combo
# phase): the first-pass census proved single-class deltas are non-additive
# (gdn_qkv alone beats all-T2), so the three registered compositions of the
# gate-passing classes are measured directly. Derived from
# tools/mixed_census_batch.sh with the same fail-closed harness: own results
# dir + lifetime lock, fresh same-box baselines, complete fingerprint
# rechecked around every measurement, transient PID-unique packs, atomic
# publication. Resumable: arms with a completed log are skipped.
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
OUT=logs/mixed_combo
ARM_PACK="models/combo_arm.$$.q27"  # run-unique transient
# Shared across ALL NLL batch drivers (census, combo, future): the lock
# guards exclusive Metal/measurement access, not just this results dir —
# two different drivers passing their own pgrep checks pre-launch could
# otherwise run concurrently and contaminate both experiments.
LOCK="logs/q27_metal_batch.lock"
mkdir -p "$OUT"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "combo: $LOCK exists; another q27 batch driver is running (remove only after verifying it is stale)"
    exit 1
fi
echo "$$" > "$LOCK/pid"
cleanup() { rm -f "$ARM_PACK"; rm -rf "$LOCK"; }
trap cleanup EXIT

if pgrep -x q27-metal >/dev/null 2>&1 || pgrep -x q27-metal-server >/dev/null 2>&1; then
    echo "combo: another q27 Metal process is running; refusing to start"; exit 1
fi

export Q27_METAL_GEMM_HALF=1 Q27_METAL_GQA_TILE=2
export Q27_METAL_GQA_THRESHOLD=2048 Q27_METAL_GQA_BLOCK=1024

# Same-machine/driver identity for resume; every probe fails closed (see
# the census driver for rationale — this block is intentionally identical).
platform_fail() {
    echo "combo: cannot establish machine/Metal runtime identity" | tee "$OUT/ABORTED"
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

# Clean environment: inherited Q27/Metal knobs cannot redefine an arm.
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
    echo "combo: rebuild failed (see $OUT/rebuild.log)" | tee "$OUT/ABORTED"; exit 1; }

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
    { echo "combo: cannot fingerprint a required input" | tee "$OUT/ABORTED"; exit 1; }
if [ -f "$OUT/fingerprint" ]; then
    [ "$(cat "$OUT/fingerprint")" = "$FP" ] ||
        { echo "combo: fingerprint mismatch — move $OUT aside" | tee "$OUT/ABORTED"; exit 1; }
else
    stale=$(find "$OUT" -type f ! -name rebuild.log -print -quit)
    [ -z "$stale" ] ||
        { echo "combo: $OUT has unfingerprinted results; move it aside" | tee "$OUT/ABORTED"; exit 1; }
    echo "$FP" > "$OUT/fingerprint"
fi
rm -f "$OUT/ABORTED"

check_fingerprint() {
    local now
    now=$(fingerprint) ||
        { echo "combo: cannot fingerprint a required input" | tee "$OUT/ABORTED"; exit 1; }
    [ "$now" = "$FP" ] ||
        { echo "combo: input changed while batch was running — aborting" | tee "$OUT/ABORTED"; exit 1; }
}

file_md5() {
    md5 -q "$1" || { echo "combo: cannot hash $1" >&2; return 1; }
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
                { echo "combo: post-run model hash failed" | tee "$OUT/ABORTED"; exit 1; }
            [ "$actual_md5" = "$expected_md5" ] ||
                { echo "combo: model changed during $(basename "$log")" | tee "$OUT/ABORTED"; exit 1; }
            mv "$tmp" "$log" ||
                { echo "combo: cannot publish $(basename "$log")" | tee "$OUT/ABORTED"; exit 1; }
            return 0
        fi
        if [ "$attempt" = 1 ]; then
            mv "$tmp" "$log.attempt1" ||
                { echo "combo: cannot preserve failed $(basename "$log")" | tee "$OUT/ABORTED"; exit 1; }
            echo "combo: $(basename "$log") attempt 1 failed (exit $rc or invalid NLL); retrying once"
        fi
    done
    mv "$tmp" "$log.failed" 2>/dev/null || true
    echo "combo: NLL run FAILED twice (see $log.failed)" | tee "$OUT/ABORTED"; exit 1
}

# Fresh same-box baselines: combo results must never borrow anchors from a
# different driver fingerprint, and two 8K runs are cheap insurance.
B1_MD5=$(file_md5 "$B1") || { echo "combo: B1 hash failed" | tee "$OUT/ABORTED"; exit 1; }
T2_MD5=$(file_md5 "$T2") || { echo "combo: T2 hash failed" | tee "$OUT/ABORTED"; exit 1; }
nll_run "$B1" "$OUT/base_b1.log" "$B1_MD5"
nll_run "$T2" "$OUT/base_t2.log" "$T2_MD5"

arm() {
    local name=$1 take=$2
    local log="$OUT/arm_$name.log"
    [ -s "$log" ] && valid_nll_log "$log" && return 0
    check_fingerprint
    python3 tools/q27_mix.py "$B1" "$T2" "$ARM_PACK" --take "$take" \
        > "$OUT/mix_$name.log" 2>&1 ||
        { echo "combo: mix FAILED for $name" | tee "$OUT/ABORTED"; exit 1; }
    local arm_md5
    arm_md5=$(md5 -q "$ARM_PACK") ||
        { echo "combo: cannot fingerprint transient pack for $name" | tee "$OUT/ABORTED"; exit 1; }
    q27 "$ARM_PACK" "$TOK" --validate-only --ctx 8 >> "$OUT/mix_$name.log" 2>&1 ||
        { echo "combo: validate FAILED for $name" | tee "$OUT/ABORTED"; exit 1; }
    check_fingerprint
    [ "$(md5 -q "$ARM_PACK")" = "$arm_md5" ] ||
        { echo "combo: transient pack changed during validation for $name" | tee "$OUT/ABORTED"; exit 1; }
    nll_run "$ARM_PACK" "$log" "$arm_md5"
    rm -f "$ARM_PACK"
}

# Registered combos — compositions of the census's gate-passing classes
# only (gdn_qkv 114.4%, gdn_alphabeta 46.9%, attnq_mid 17.6%). Mid band
# matches the census exactly: blk 21-42.
AQM='blk\.(2[1-9]|3[0-9]|4[0-2])\.attn_q\.weight'
arm gdn_pair   '(attn_qkv|ssm_(alpha|beta))\.weight'
arm gate_trio  "((attn_qkv|ssm_(alpha|beta))\\.weight|$AQM)"
arm cheap_pair "(ssm_(alpha|beta)\\.weight|$AQM)"

# Summary: NLL, gap recovered, mixed/T2 NLL ratio, pack size, ship band
# (pre-registered: ratio <= 1.06 at <= 5.0 GB ship; <= 1.12 conditional).
check_fingerprint
SUMMARY_TMP="$OUT/combo_summary.txt.tmp"
python3 - "$OUT" "$B1" <<'PY' | tee "$SUMMARY_TMP"
import glob, math, os, re, sys
out, b1_path = sys.argv[1], sys.argv[2]
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
# Decimal units throughout: the mix log's "byte delta MB" is decimal
# (bytes/1e6), so the base must be decimal GB too or the 5.0 GB ship
# gate compares mixed units and fails open.
b1_gb = os.path.getsize(b1_path) / 1e9
print(f"base B1 {b1:.4f}  base T2 {t2:.4f}  gap {gap:.4f} nats  B1 pack {b1_gb:.2f} GB")
files = sorted(glob.glob(f"{out}/arm_*.log"))
if len(files) != 3:
    raise RuntimeError(f"expected 3 combo logs, found {len(files)}")
rows = []
for f in files:
    name = os.path.basename(f)[4:-4]
    v = nll(f)
    mix = f"{out}/mix_{name}.log"
    matches = re.findall(r"byte delta ([+-]?[0-9.]+) MB", open(mix).read())
    if len(matches) != 1:
        raise RuntimeError(f"{mix}: expected exactly one byte delta")
    take_mb = float(matches[0])
    ratio = v / t2
    pack_gb = b1_gb + take_mb / 1000
    if ratio <= 1.06 and pack_gb <= 5.0:
        band = "SHIP-BAND"
    elif ratio <= 1.12 and pack_gb <= 5.0:
        band = "conditional"
    else:
        band = "outside"
    rows.append((name, v, (b1 - v) / gap, ratio, take_mb, pack_gb, band))
rows.sort(key=lambda r: -r[2])
for name, v, rec, ratio, mb, gb, band in rows:
    print(f"{name:10s} NLL {v:.4f}  gap recovered {100*rec:6.1f}%  vs T2 {ratio:.4f}  bytes {mb:+7.1f} MB  pack {gb:.2f} GB  {band}")
PY
summary_rc=$?
[ "$summary_rc" = 0 ] ||
    { echo "combo: summary generation FAILED" | tee "$OUT/ABORTED"; exit 1; }
mv "$SUMMARY_TMP" "$OUT/combo_summary.txt" ||
    { echo "combo: cannot publish summary" | tee "$OUT/ABORTED"; exit 1; }
echo "combo batch: COMPLETE"
