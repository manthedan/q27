#!/usr/bin/env bash
# 12 GB-card simulation on the 3090: a hog holds VRAM so the server sees ~12 GB
# free; the 12g server build + the slim pack must boot, project a usable ctx,
# and reproduce the plain reference texts (T2 embed/head are exact).
set -u
M=/mnt/ai/projects/q27-master; S=${OUT:-/tmp/bonsai2-gates}; mkdir -p $S
SLIM=${SLIM:-/mnt/ai/models/bonsai2-27b/q27/bonsai2-27b-t2-slim.q27}; TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
BIN=${BIN:-$M/build/q27-server-12g}; HOG=${HOG:-11.4}
ARGS="--host 127.0.0.1 --port 8090 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0 ${CTX:+--ctx $CTX}"
log() { echo "[12g $(date '+%H:%M:%S')] $*"; }
sudo -n systemctl stop vox-transcriber vox-transcriber-gmrs; sleep 3
b3=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | grep GPU-5a723c5e || true)
[ -n "$b3" ] && { log "3090 busy: $b3"; sudo -n systemctl start vox-transcriber vox-transcriber-gmrs; exit 3; }
systemctl --user reset-failed bz-hog 2>/dev/null
systemd-run --user --unit bz-hog -E CUDA_VISIBLE_DEVICES=1 -p StandardOutput=file:$S/hog.log -p StandardError=file:$S/hog.log python3 $M/bench/bonsai2/vram_hog.py $HOG cuda:0
for i in $(seq 1 30); do grep -q "hog: holding" $S/hog.log 2>/dev/null && break; sleep 1; done; cat $S/hog.log
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed -n 1p
run_leg() { # $1 label, rest = -E envs
  local label=$1; shift
  systemctl --user reset-failed bz-server 2>/dev/null
  systemd-run --user --unit bz-server -E CUDA_VISIBLE_DEVICES=1 -E Q27_KV=${KV:-fp8} -E Q27_PRINT_WSUM=1 "$@" \
    -p StandardOutput=file:$S/${PFX:-g12}_$label.log -p StandardError=file:$S/${PFX:-g12}_$label.log $BIN $SLIM $TOK $ARGS --slots 1 >/dev/null
  for i in $(seq 1 200); do c=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/health); [ "$c" = 200 ] && break; systemctl --user is-active --quiet bz-server || { log "$label died"; tail -6 $S/${PFX:-g12}_$label.log | cut -c1-200; return 1; }; sleep 2; done
  log "== $label"; grep -E "wsum|vram: free|\[pool\]|slot 0 ready|clamp|listening|dflash2 serving|bonsai2:" $S/${PFX:-g12}_$label.log | cut -c1-160
  python3 - "$label" "$S" "${PFX:-g12}" "${REF:-$S/mg_t2plain.json}" <<'PY'
import json, urllib.request, sys, time
label=sys.argv[1]; S=sys.argv[2]; base="http://127.0.0.1:8090"
def msg(prompt, n, think):
    b={"model":"q27","max_tokens":n,"stream":False,"temperature":0,"messages":[{"role":"user","content":prompt}]}
    if think: b["thinking"]={"type":"enabled","budget_tokens":n-16}
    req=urllib.request.Request(base+"/v1/messages", data=json.dumps(b).encode(), headers={"content-type":"application/json","x-api-key":"local","anthropic-version":"2023-06-01"})
    with urllib.request.urlopen(req, timeout=900) as r: d=json.load(r)
    return "".join((x.get("thinking","") if x.get("type")=="thinking" else x.get("text","")) for x in d["content"]), d["usage"]["output_tokens"]
P={"short":("Reply with the single word ok.",32,False),
   "cities":("What is the capital of France, and name two other French cities? Answer in one sentence.",200,True),
   "long":("Write a 300-word explanation of how a hash table handles collisions, with a short Python example.",700,True),
   "code":("Write a Python module with a class LRUCache(capacity) supporting get(key) and put(key, value) in O(1), with docstrings and a small pytest test file. Explain the data structure choice briefly first.",1500,True)}
out={}
for name,(p,n,th) in P.items():
    t0=time.time(); t,ntok=msg(p,n,th); out[name]=t; print(f"  {name}: {ntok} tokens {time.time()-t0:.1f}s", flush=True)
json.dump(out, open(f"{S}/{sys.argv[3]}_{label}.json","w"))
ref=json.load(open(sys.argv[4]))
for k in ref:
    a=ref[k]; b=out[k]
    if a==b: print(f"  {k:7s}: IDENTICAL to the reference ({len(a)} chars)")
    else:
        i=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
        print(f"  {k:7s}: DIFFER at char {i} of {len(a)}/{len(b)}")
PY
  grep "\[req\]" $S/${PFX:-g12}_$label.log | sed 's/.*dec=\([0-9]*\) dec_ms=\([0-9.]*\).*rounds=\([0-9]*\) tps=\([0-9.]*\).*/    dec=\1 dec_ms=\2 rounds=\3 tps=\4/' | head -4
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n 1p
  systemctl --user stop bz-server; sleep 2
}
for leg in ${LEGS:-plain d2}; do
  case $leg in
    plain) run_leg plain -E Q27_BATCH=0 -E Q27_FIXED_STACK_GB=${FIXED:-0.9} ;;
    fusedplain) run_leg fusedplain -E Q27_FIXED_STACK_GB=${FIXED:-0.9} ;;   # the server default: conductor on, k=1 fused rounds
    mtp)   run_leg mtp -E Q27_BATCH=0 -E Q27_FIXED_STACK_GB=${FIXED:-0.9} ;;
    d2)    run_leg d2 -E Q27_BATCH=0 -E Q27_FIXED_STACK_GB=${FIXED:-0.9} -E Q27_DFLASH2=/mnt/ai/models/bonsai2-27b-dflash2-bf16/bonsai2-dflash2-q8-serve.d2w -E Q27_DFLASH2_RESERVE_GB=1 ;;
  esac
done
systemctl --user stop bz-hog
sudo -n systemctl start vox-transcriber vox-transcriber-gmrs
log "G12-DONE"
