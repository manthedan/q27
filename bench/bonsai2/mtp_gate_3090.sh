#!/usr/bin/env bash
# 3090 gates for the MTP block on the Bonsai 2 target (pure-T2 body + the
# third-party head as blk.64):
#  1. CLI canonical: plain (pure-T2 pack) vs --spec ladder (T2+MTP pack), the
#     house prompt, 128 greedy tokens -> the `generated:` lines must match, and
#     the ladder run reports its accepted tokens per round.
#  2. Server identity: pure-T2 plain (Q27_BATCH=0, 1 slot) vs T2+MTP ladder
#     (1 slot) vs T2+MTP fused 2-slot (conductor, concurrent pairs) on the four
#     greedy prompts -> every text identical; [req] rounds give tok/round.
set -u
M=/mnt/ai/projects/q27-master
S=${OUT:-/tmp/bonsai2-gates}; mkdir -p $S
T2=/mnt/ai/models/bonsai2-27b/q27/bonsai2-27b-t2.q27
T2M=/mnt/ai/models/bonsai2-27b/q27/bonsai2-27b-t2-mtp.q27
TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
ARGS="--host 127.0.0.1 --port 8090 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0 --ctx 16384"
log() { echo "[mg $(date '+%H:%M:%S')] $*"; }
sudo -n systemctl stop vox-transcriber vox-transcriber-gmrs; sleep 3
b3=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | grep GPU-5a723c5e || true)
[ -n "$b3" ] && { log "3090 busy: $b3"; sudo -n systemctl start vox-transcriber vox-transcriber-gmrs; exit 3; }
export CUDA_VISIBLE_DEVICES=1
log "== 1. CLI canonical: plain (t2) vs --spec ladder (t2-mtp)"
Q27_KV=fp8 timeout 600 $M/build/q27 $T2 --tokens "760,6511,314,9338,369" -n 128 --ctx 2048 > $S/mg_cli_plain.log 2>&1
Q27_KV=fp8 timeout 600 $M/build/q27 $T2M --tokens "760,6511,314,9338,369" -n 128 --ctx 2048 --spec > $S/mg_cli_spec.log 2>&1
a=$(grep -m1 "^generated:" $S/mg_cli_plain.log | md5sum | cut -c1-8); b=$(grep -m1 "^generated:" $S/mg_cli_spec.log | md5sum | cut -c1-8)
log "generated md5: plain $a  spec $b  -> $([ "$a" = "$b" ] && echo IDENTICAL || echo DIFFER)"
grep -E "round outcomes|tokens/round|tok/round|t/s|wsum|bonsai2:|spec graphs" $S/mg_cli_spec.log | head -6 | cut -c1-200
grep -E "t/s|wsum" $S/mg_cli_plain.log | head -3 | cut -c1-200
run_leg() { # $1 label, $2 model, $3 slots, rest = -E envs
  local label=$1 model=$2 slots=$3; shift 3
  systemctl --user reset-failed bz-server 2>/dev/null
  systemd-run --user --unit bz-server -E CUDA_VISIBLE_DEVICES=1 -E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 "$@" \
    -p StandardOutput=file:$S/mg_$label.log -p StandardError=file:$S/mg_$label.log \
    $M/build/q27-server $model $TOK $ARGS --slots $slots >/dev/null
  for i in $(seq 1 200); do c=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/health); [ "$c" = 200 ] && break; systemctl --user is-active --quiet bz-server || { log "$label died"; tail -4 $S/mg_$label.log; return 1; }; sleep 2; done
  log "== $label: $(grep -m1 -E 'bonsai2:' $S/mg_$label.log | cut -c1-120)"
  python3 - "$label" "$S" <<'PY'
import json, urllib.request, sys, time, threading
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
def run(name):
    p,n,th=P[name]; t0=time.time(); t,ntok=msg(p,n,th); out[name]=t; print(f"  {name}: {ntok} tokens {time.time()-t0:.1f}s", flush=True)
for pair in (("long","code"),("short","cities")):
    ths=[threading.Thread(target=run,args=(nm,)) for nm in pair]
    [t.start() for t in ths]; [t.join() for t in ths]
json.dump(out, open(f"{S}/mg_{label}.json","w"))
PY
  grep "\[req\]" $S/mg_$label.log | sed 's/.*dec=\([0-9]*\) dec_ms=\([0-9.]*\).*rounds=\([0-9]*\) tps=\([0-9.]*\).*/    dec=\1 dec_ms=\2 rounds=\3 tps=\4/' | head -4
  systemctl --user stop bz-server; sleep 2
}
run_leg t2plain $T2 1 -E Q27_BATCH=0
run_leg mtp1 $T2M 1
run_leg mtp2 $T2M 2
sudo -n systemctl start vox-transcriber vox-transcriber-gmrs
python3 - "$S" <<'PY'
import json, sys
S=sys.argv[1]
ref=json.load(open(f"{S}/mg_t2plain.json"))
for lab in ("mtp1","mtp2"):
    d=json.load(open(f"{S}/mg_{lab}.json"))
    for k in ref:
        a=ref[k]; b=d[k]
        if a==b: print(f"  {lab:6s} {k:7s}: IDENTICAL to t2 plain ({len(a)} chars)")
        else:
            i=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
            print(f"  {lab:6s} {k:7s}: DIFFER at char {i} of {len(a)}/{len(b)}")
PY
log "MG-DONE"
