#!/usr/bin/env bash
# 8 GB-card simulation on the 3090 for the T3 packs (2026-09-20): the hog
# leaves FREE GB (headless 8 GB card: ~8.3 GB before the process context;
# with a display: ~7.5), the 12g server build boots the T3 slim pack with
# Q27_FIXED_STACK_GB=0.6 (the measured 12g stack, 0.54, + margin) and the
# Ampere-default turbo5k KV, and must reproduce the full-memory turbo5k texts.
set -u
S=${OUT:-/tmp/bonsai2-gates}
M=/mnt/ai/projects/q27-master
T3=/mnt/ai/models/bonsai2-27b/q27/bonsai2-27b-t3-slim.q27
T3M=/mnt/ai/models/bonsai2-27b/q27/bonsai2-27b-t3-mtp-slim.q27
BASE_USED=${BASE_USED:-0.7}   # GB the 3090 holds at rest with the transcribers stopped (hog log 2026-09-20: 25.30 total, 15.09 held, 8.81 free); check "hog: holding" in hog.log
# SPECS: ';'-separated "leg free_gb pack log_prefix" items
IFS=';' read -ra SPECLIST <<< "${SPECS:-plain 8.3 $T3 g8a;plain 8.0 $T3 g8b;plain 7.6 $T3 g8c;mtp 8.0 $T3M g8m}"
for spec in "${SPECLIST[@]}"; do
  set -- $spec; leg=$1; free=$2; pack=$3; pfx=$4
  hog=$(python3 -c "print(round((25.30 - $BASE_USED - $free) / 1.073741824, 2))")
  echo "[g8 $(date '+%H:%M:%S')] ===== $leg pack=$(basename $pack) free~$free GB (hog $hog GiB) ====="
  PFX=$pfx SLIM=$pack BIN=$M/build/q27-server-12g HOG=$hog FIXED=${FIXED:-0.6} KV=${KV:-turbo5k} CTX=${CTX:-49152} \
    LEGS=$leg REF=${REF:-$S/t3s_t3k5.json} bash $M/bench/bonsai2/gate12g_3090.sh 2>&1 | grep -v "^hog:"
done
echo "[g8 $(date '+%H:%M:%S')] G8-DONE"
