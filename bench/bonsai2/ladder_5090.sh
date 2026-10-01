#!/usr/bin/env bash
# Concurrency ladder (the 2026-08-14/08-19 protocol: --slots 8 --ctx 16384,
# Q27_KV=fp8 Q27_BATCH=1, C concurrent salted chat requests at temp 0.6,
# max_tokens 8192, aggregate = sum(dec)/union of decode intervals from the
# server's [req] lines) for the pure-T2 Bonsai 2 pack on the 5090:
#   fused   default (draftless width-2 lane pairs in fused rounds)
#   bzsolo  Q27_BONSAI_FUSED=0 (members pinned to solo rounds = the control)
# Production is stopped for the run and relaunched at the end.
set -u
M=/mnt/ai/projects/q27-master; Q=/mnt/ai/projects/q27
S=${OUT:-/tmp/bonsai2-gates}; mkdir -p $S
T2=/mnt/ai/models/bonsai2-27b/q27/bonsai2-27b-t2.q27; TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
ARGS="--host 127.0.0.1 --port 8093 --slots 8 --ctx 16384"
log() { echo "[ladder $(date '+%H:%M:%S')] $*"; }
busy=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | grep GPU-e592b842 | grep -v "$Q/build/q27-server" || true)
[ -n "$busy" ] && { echo "5090 busy: $busy"; exit 3; }
systemctl --user stop q27-38; sleep 3
run_leg() { # $1 label, rest = -E envs
  local label=$1; shift
  systemctl --user reset-failed bz-server 2>/dev/null
  systemd-run --user --unit bz-server -E CUDA_VISIBLE_DEVICES=0 -E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_BATCH=1 "$@" \
    -p StandardOutput=file:$S/ladder_$label.log -p StandardError=file:$S/ladder_$label.log \
    $M/build/q27-server $T2 $TOK $ARGS >/dev/null
  for i in $(seq 1 200); do c=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8093/health); [ "$c" = 200 ] && break; systemctl --user is-active --quiet bz-server || { log "$label died"; tail -4 $S/ladder_$label.log; return 1; }; sleep 2; done
  log "== $label: $(grep -m1 -E 'clamping|slot [0-9]+ ready' $S/ladder_$label.log) | $(grep -m1 'vram: free' $S/ladder_$label.log)"
  # sampled sanity (the ladder samples at temp 0.6; the fused sampled tail runs max_draft 0): one short reply, eyeballed
  curl -s -m 120 http://127.0.0.1:8093/v1/messages -H 'content-type: application/json' -H 'x-api-key: x' \
    -d '{"model":"q27","max_tokens":120,"temperature":0.6,"messages":[{"role":"user","content":"In two sentences, what does a hash table do?"}]}' \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('  sampled:', repr(''.join(b.get('text','') for b in d['content'])[:200]))"
  for C in ${CS:-1 2 4 8}; do
    log "-- $label C=$C"
    python3 $M/bench/ladder/ladder.py $S/ladder_$label.log http://127.0.0.1:8093 $C ${MAXTOK:-8192} 2>&1 | grep -E "aggregate|FAILED|WARN"
  done
  systemctl --user stop bz-server; sleep 3
}
run_leg fused
run_leg bzsolo -E Q27_BONSAI_FUSED=0
bash $Q/tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1 | tail -1
log "LADDER-DONE"
