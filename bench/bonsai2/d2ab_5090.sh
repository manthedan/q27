#!/usr/bin/env bash
# 5090 A/B of DFlash2 drafter packs on the pure-T2 Bonsai 2 target:
#   plain            identity reference (no drafter)
#   qwen-exact       Qwen3.8 Q8 pack,  Q27_D2_VGEMM=0 -> greedy text must == plain
#   bonsai-exact     Bonsai 2 Q8 pack, Q27_D2_VGEMM=0 -> greedy text must == plain
#   qwen / bonsai    default (vgemm) config for the timing numbers
# Same binary, same prompts, same day: the Qwen3.8 pack is the control.
set -u
M=/mnt/ai/projects/q27-master; Q=/mnt/ai/projects/q27
S=${OUT:-/tmp/bonsai2-gates}; mkdir -p $S
T2=${MODEL:-/mnt/ai/models/bonsai2-27b/q27/bonsai2-27b-t2.q27}; TAG=${TAG:-}; TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
PACKQ=/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q8-serve.d2w
PACKB=${PACKB:-/mnt/ai/models/bonsai2-27b-dflash2-bf16/bonsai2-dflash2-q8-serve.d2w}
ARGS="--host 127.0.0.1 --port 8093 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0"
log() { echo "[d2ab $(date '+%H:%M:%S')] $*"; }
busy=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | grep GPU-e592b842 | grep -v "$Q/build/q27-server" || true)
[ -n "$busy" ] && { echo "5090 busy: $busy"; exit 3; }
systemctl --user stop q27-38; sleep 3
run_server() { # $1 label, rest = extra -E envs
  local label=$1; shift
  systemctl --user reset-failed bz-server 2>/dev/null
  systemd-run --user --unit bz-server -E CUDA_VISIBLE_DEVICES=0 -E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_D2_TIMING=1 "$@" \
    -p StandardOutput=file:$S/d2ab_$TAG$label.log -p StandardError=file:$S/d2ab_$TAG$label.log $M/build/q27-server $T2 $TOK $ARGS >/dev/null
  for i in $(seq 1 150); do c=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8093/health); [ "$c" = 200 ] && break; systemctl --user is-active --quiet bz-server || { log "$label died"; tail -3 $S/d2ab_$TAG$label.log; return 1; }; sleep 2; done
  log "== $label: $(grep -m1 'dflash2 serving' $S/d2ab_$TAG$label.log || echo 'no drafter')"
  python3 - "$TAG$label" <<'PY'
import json, urllib.request, sys, time
label=sys.argv[1]; base="http://127.0.0.1:8093"; out={}
def msg(prompt, n, think):
    b={"model":"q27","max_tokens":n,"stream":False,"temperature":0,"messages":[{"role":"user","content":prompt}]}
    if think: b["thinking"]={"type":"enabled","budget_tokens":n-16}
    req=urllib.request.Request(base+"/v1/messages", data=json.dumps(b).encode(), headers={"content-type":"application/json","x-api-key":"local","anthropic-version":"2023-06-01"})
    t0=time.time()
    with urllib.request.urlopen(req, timeout=900) as r: d=json.load(r)
    return "".join((x.get("thinking","") if x.get("type")=="thinking" else x.get("text","")) for x in d["content"]), d["usage"]["output_tokens"], time.time()-t0
P={"short":("Reply with the single word ok.",32,False),
   "cities":("What is the capital of France, and name two other French cities? Answer in one sentence.",200,True),
   "long":("Write a 300-word explanation of how a hash table handles collisions, with a short Python example.",700,True),
   "code":("Write a Python module with a class LRUCache(capacity) supporting get(key) and put(key, value) in O(1), with docstrings and a small pytest test file. Explain the data structure choice briefly first.",1500,True)}
for name,(p,n,th) in P.items():
    t,ntok,dt=msg(p,n,th); out[name]=t; print(f"  {name}: {ntok} tokens in {dt:.1f}s ({ntok/dt:.1f} t/s incl. prefill)")
json.dump(out, open(f"" + __import__('os').environ.get('OUT','/tmp/bonsai2-gates') + "/d2ab_{label}.json","w"))
PY
  grep "\[req\]" $S/d2ab_$TAG$label.log | sed 's/.*dec=\([0-9]*\) dec_ms=\([0-9.]*\).*rounds=\([0-9]*\) tps=\([0-9.]*\).*/    dec=\1 dec_ms=\2 rounds=\3 tps=\4/'
  grep "\[d2timing\]" $S/d2ab_$TAG$label.log | tail -1
  systemctl --user stop bz-server; sleep 2
}
for r in ${RUNS:-plain qwen-exact bonsai-exact qwen bonsai}; do
  case $r in
    plain)        run_server plain ;;
    qwen-exact)   run_server qwen-exact   -E Q27_DFLASH2=$PACKQ -E Q27_DFLASH2_RESERVE_GB=3 -E Q27_BATCH=0 -E Q27_D2_VGEMM=0 ;;
    bonsai-exact) run_server bonsai-exact -E Q27_DFLASH2=$PACKB -E Q27_DFLASH2_RESERVE_GB=3 -E Q27_BATCH=0 -E Q27_D2_VGEMM=0 ;;
    qwen)         run_server qwen         -E Q27_DFLASH2=$PACKQ -E Q27_DFLASH2_RESERVE_GB=3 -E Q27_BATCH=0 ;;
    bonsai)       run_server bonsai       -E Q27_DFLASH2=$PACKB -E Q27_DFLASH2_RESERVE_GB=3 -E Q27_BATCH=0 ;;
  esac
done
TAG=$TAG python3 - <<'PY'
import json, os, glob
S="" + __import__('os').environ.get('OUT','/tmp/bonsai2-gates') + ""; T=os.environ.get("TAG","")
a=json.load(open(f"{S}/d2ab_{T}plain.json"))
for lab in ("qwen-exact","bonsai-exact"):
    if not os.path.exists(f"{S}/d2ab_{T}{lab}.json"): continue
    b=json.load(open(f"{S}/d2ab_{T}{lab}.json"))
    for k in a:
        if a[k]==b[k]: print(f"  identity {lab:13s} {k:7s}: IDENTICAL ({len(a[k])} chars)")
        else:
            i=next((i for i,(x,y) in enumerate(zip(a[k],b[k])) if x!=y), min(len(a[k]),len(b[k])))
            print(f"  identity {lab:13s} {k:7s}: DIFFER at char {i} of {len(a[k])}/{len(b[k])}")
PY
[ "${NORELAUNCH:-0}" = 1 ] || bash $Q/tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1 | tail -1
log "D2AB-DONE"
