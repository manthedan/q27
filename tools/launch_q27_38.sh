#!/usr/bin/env bash
# Production launch recipe for the q27-38 serving unit (Qwen3.8-27B-MTP on the
# 5090). q27-38 is a TRANSIENT systemd-run --user unit: `systemctl --user stop
# q27-38` deletes it, this script recreates it. Modes:
#   d2-pfx   PRODUCTION since 2026-09-08 evening: d2 + the prefix-cache tiers ON
#            (tmpfs-backed disk tier, RAM tier off) with the P16b shared cut;
#            docs/plans/2026-09-08-prefill-attack.md phase 0 (prefill wall
#            255 -> 110 s on the 12-instance Claude Code run). Pass
#            -E Q27_SYSBLK=1 to log system-block geometry per request.
#   d2       the 2026-09-08 daytime config: DFlash2 Q8 pack, sampled walk, MMA
#            verify, NO prefix cache (every first turn and every returning turn
#            after a side request re-prefills cold)
#   ladder   the pre-09-08 production config (MTP ladder + suffix drafter)
#   bonsai2  the d2-pfx config serving PrismML's Ternary Bonsai 2 27B
#            (docs/plans/2026-09-18-bonsai2-ternary.md; BONSAI2_MODEL overrides
#            the artifact). The pack has no MTP block, so DFlash2 is the only
#            drafter; its cache root is its own (different numerics), and the
#            t2 artifact needs the 5090 (22.4 GB resident until the T2 prefill
#            GEMM drops the .q4x shadows).
# Extra `-E K=V` after the mode are passed to systemd-run (e.g. -E Q27_SYSBLK=1).
set -euo pipefail
Q=/mnt/ai/projects/q27
BIN=$Q/build/q27-server
MODEL=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27
TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
PACK8=/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q8-serve.d2w
ARGS="--host 172.17.0.1 --port 8081 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0"
D2ENV="-E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_BATCH=0 -E Q27_DFLASH2=$PACK8 -E Q27_DFLASH2_RESERVE_GB=3 -E Q27_D2_TIMING=1"
# P16 disk tier on tmpfs (zero SSD wear; /dev/shm had 62 GB free on 09-08),
# max_tokens raised so 48K+ conversations persist (default 32768 would
# silently never persist them), step left at the 8192 default. The P16c RAM
# tier is OFF by default: tmpfs alone restored in 0.47 s on 07-24 vs 0.53 s
# with the tier, and each RAM slot pins pfx_bytes(max_tokens) ~2.44 GB on top
# of the engine's two pinned staging buffers (~4.9 GB). PFX_RAM_GB=16 gives
# six slots. Use a DIFFERENT PFX_DIR per numerical variant (kernel changes,
# ladder vs d2): the blob format does not encode kernel numerics. Since
# 2026-09-08 (k) a DFlash2 engine skips the MTP KV warm during prefill, so
# every blob under this root has UNWARMED MTP rows: it is a DFlash2-only root.
# A ladder config restoring from it would draft from garbage (correct output,
# acceptance loss the bitwise gates cannot see). The ladder mode below has no
# cache flags on purpose; give it its own root if that ever changes.
# Request-body recording (2026-09-08, item 2): REQ_LOG=<file> appends one
# JSONL line per request (seq, t_ms, api, path, raw body) so bench/replay/
# replay.py can feed two binaries the identical sequence. Real session
# content -- keep it local, delete when done.
REQLOG_ENV=""; [ -n "${REQ_LOG:-}" ] && REQLOG_ENV="-E Q27_REQ_LOG=$REQ_LOG"
PFX_DIR=${PFX_DIR:-/dev/shm/q27-pfx}
# 131072-token entries since 2026-09-13 (was 65536): returning to an evicted
# 69K-token conversation restores in 0.5 s instead of re-prefilling 45K
# tokens (16 s) -- docs/perf-next-2026-09-12.md. Costs ~4.6 GB more pinned
# host memory (2 x 4.72 GB staging on the one production slot).
PFXARGS="--prefix-cache $PFX_DIR --prefix-cache-max-gb ${PFX_MAX_GB:-40} --prefix-cache-ram-gb ${PFX_RAM_GB:-0} --prefix-cache-max-tokens ${PFX_MAX_TOK:-131072}"
mode=${1:-}; shift || true
systemctl --user stop q27-38 2>/dev/null || true
systemctl --user reset-failed q27-38 2>/dev/null || true
case "$mode" in
  d2)      systemd-run --user --unit q27-38 $D2ENV $REQLOG_ENV "$@" $BIN $MODEL $TOK $ARGS ;;
  d2-pfx)  mkdir -p "$PFX_DIR"
           systemd-run --user --unit q27-38 $D2ENV $REQLOG_ENV "$@" $BIN $MODEL $TOK $ARGS $PFXARGS ;;
  ladder)  systemd-run --user --unit q27-38 -E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 $REQLOG_ENV "$@" $BIN $MODEL $TOK $ARGS ;;
  bonsai2) MODEL=${BONSAI2_MODEL:-/mnt/ai/models/bonsai2-27b/q27/bonsai2-27b-t2.q27}
           # the Bonsai-trained DFlash2 drafter (ProCreations, repacked by
           # tools/dflash2_pack.py --q8): 3.80 vs 3.47 tok/round over the Qwen3.8
           # pack, 227.7 vs 178.8 t/s aggregate over the MTP ladder on the
           # single-slot campaign (BUILDLOG 2026-09-18 (as), (au)). Multi-slot
           # wants the T2+MTP pack instead (BONSAI2_MODEL=...t2-mtp.q27 with
           # Q27_BATCH=1 and no Q27_DFLASH2).
           BZPACK=${BONSAI2_PACK:-/mnt/ai/models/bonsai2-27b-dflash2-bf16/bonsai2-dflash2-q8-serve.d2w}
           BZD2ENV="-E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_BATCH=0 -E Q27_DFLASH2=$BZPACK -E Q27_DFLASH2_RESERVE_GB=3 -E Q27_D2_TIMING=1"
           PFX_DIR=${PFX_DIR_BONSAI2:-/dev/shm/q27-pfx-bonsai2}; mkdir -p "$PFX_DIR"
           PFXARGS="--prefix-cache $PFX_DIR --prefix-cache-max-gb ${PFX_MAX_GB:-40} --prefix-cache-ram-gb ${PFX_RAM_GB:-0} --prefix-cache-max-tokens ${PFX_MAX_TOK:-65536}"
           systemd-run --user --unit q27-38 $BZD2ENV $REQLOG_ENV "$@" $BIN $MODEL $TOK $ARGS $PFXARGS ;;
  *) echo "usage: $0 d2|d2-pfx|ladder|bonsai2 [-E K=V ...]" >&2; exit 2 ;;
esac
# readiness: key on THIS invocation (a --since window can match the previous
# unit's line) and on the listener ("listening on"); the DFlash2 "serving ON"
# line prints during engine setup, before the socket is bound
inv=$(systemctl --user show q27-38 -p InvocationID --value)
for i in $(seq 1 240); do
  if journalctl --user _SYSTEMD_INVOCATION_ID="$inv" -o cat --no-pager 2>/dev/null | grep -q "listening on"; then
    echo "q27-38 ($mode) serving; invocation $inv"; exit 0
  fi
  systemctl --user is-active --quiet q27-38 || { echo "q27-38 died during startup" >&2; journalctl --user _SYSTEMD_INVOCATION_ID="$inv" -o cat --no-pager | tail -5; exit 1; }
  sleep 2
done
echo "q27-38 never reported 'listening on'" >&2; exit 1
