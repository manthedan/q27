#!/usr/bin/env bash
# 3090 identity gate for Bonsai 2 fused (multi-slot) rounds on the pure-T2 pack.
# Four server configs, the same four greedy prompts sent as two concurrent
# pairs (so the 2-slot legs form k=2 unions; a lone request on a 2-slot
# server exercises the k=1 fused path), plus each prompt alone:
#   solo1   --slots 1, Q27_BATCH=0            (no conductor: plain token graph, the reference)
#   fused1  --slots 1, Q27_BATCH=1            (conductor, k=1 fused rounds via always_fused)
#   fused   --slots 2, Q27_BATCH=1            (conductor: fused width-2 lane pairs)
#   bzsolo  --slots 2, Q27_BONSAI_FUSED=0     (conductor, members pinned solo)
#   fifo    --slots 2, Q27_BATCH=0            (no conductor, time-sliced)
# Every text must be identical across the four (sm_86: the GEMV family and the
# width-1 vs width-2 lane kernels are bitwise; the 5090 is not the instrument).
set -u
M=/mnt/ai/projects/q27-master
S=${OUT:-/tmp/bonsai2-gates}; mkdir -p $S
T2=/mnt/ai/models/bonsai2-27b/q27/bonsai2-27b-t2.q27; TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
ARGS="--host 127.0.0.1 --port 8090 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0 --ctx 16384"
log() { echo "[fg $(date '+%H:%M:%S')] $*"; }
sudo -n systemctl stop vox-transcriber vox-transcriber-gmrs; sleep 3
b3=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | grep GPU-5a723c5e || true)
[ -n "$b3" ] && { log "3090 busy: $b3"; sudo -n systemctl start vox-transcriber vox-transcriber-gmrs; exit 3; }
run_leg() { # $1 label, $2 slots, rest = -E envs
  local label=$1 slots=$2; shift 2
  systemctl --user reset-failed bz-server 2>/dev/null
  systemd-run --user --unit bz-server -E CUDA_VISIBLE_DEVICES=1 -E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_BATCH_DBG=1 "$@" \
    -p StandardOutput=file:$S/fg_$label.log -p StandardError=file:$S/fg_$label.log \
    $M/build/q27-server $T2 $TOK $ARGS --slots $slots >/dev/null
  for i in $(seq 1 200); do c=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/health); [ "$c" = 200 ] && break; systemctl --user is-active --quiet bz-server || { log "$label died"; tail -4 $S/fg_$label.log; return 1; }; sleep 2; done
  log "== $label: $(grep -m1 -E 'slot [0-9]+ ready|slots' $S/fg_$label.log | head -1) | $(grep -m1 'spec graphs' $S/fg_$label.log)"
  python3 - "$label" <<'PY'
import json, urllib.request, sys, time, threading
label=sys.argv[1]; base="http://127.0.0.1:8090"
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
def run(name):
    p,n,th=P[name]; t0=time.time(); t,ntok=msg(p,n,th); out[name]=t; print(f"  {name}: {ntok} tokens {time.time()-t0:.1f}s", flush=True)
# concurrent pairs (long+code, short+cities), then each alone
for pair in (("long","code"),("short","cities")):
    ths=[threading.Thread(target=run,args=(nm,)) for nm in pair]
    [t.start() for t in ths]; [t.join() for t in ths]
solo={}
for name in P:
    p,n,th=P[name]; t,_=msg(p,n,th); solo[name]=t
    print(f"  {name} alone: {'== paired' if t==out[name] else 'DIFFERS from paired'}", flush=True)
json.dump({"paired":out,"alone":solo}, open(f"" + __import__('os').environ.get('OUT','/tmp/bonsai2-gates') + "/fg_{label}.json","w"))
PY
  grep -c "\[batch\]\|\[fused\]\|union" $S/fg_$label.log | sed 's/^/    batch-log lines: /'
  grep "\[req\]" $S/fg_$label.log | sed 's/.*dec=\([0-9]*\) dec_ms=\([0-9.]*\).*rounds=\([0-9]*\) tps=\([0-9.]*\).*/    dec=\1 dec_ms=\2 rounds=\3 tps=\4/' | head -8
  systemctl --user stop bz-server; sleep 2
}
run_leg solo1 1 -E Q27_BATCH=0
run_leg fused1 1
run_leg fused 2
run_leg bzsolo 2 -E Q27_BONSAI_FUSED=0
run_leg fifo 2 -E Q27_BATCH=0
sudo -n systemctl start vox-transcriber vox-transcriber-gmrs
python3 - <<'PY'
import json
S="" + __import__('os').environ.get('OUT','/tmp/bonsai2-gates') + ""
ref=json.load(open(f"{S}/fg_solo1.json"))
for lab in ("fused1","fused","bzsolo","fifo"):
    d=json.load(open(f"{S}/fg_{lab}.json"))
    for k in ref["paired"]:
        for mode in ("paired","alone"):
            a=ref["paired"][k]; b=d[mode][k]
            if a==b: print(f"  {lab:7s} {mode:6s} {k:7s}: IDENTICAL to solo1 ({len(a)} chars)")
            else:
                i=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
                print(f"  {lab:7s} {mode:6s} {k:7s}: DIFFER at char {i} of {len(a)}/{len(b)}")
PY
log "FG-DONE"
