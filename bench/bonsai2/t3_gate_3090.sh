#!/usr/bin/env bash
# T3_G128 container gates on the 3090 (Bonsai 2 8 GB packs, 2026-09-20).
#   gate     build/t3_gate: every T3 matrix vs the T2 pack -- gemv w1/2/5/8 and
#            the prefill T2 conversion, all BITWISE; --bench times the two GEMVs
#   kernels  build/test_kernels on the T3 pack (reference tolerances + gemv_n)
#   ninv     build/ninv_test on the T3 pack (N-invariance of gemv_t3_n)
#   cli      CLI canonical: T2 vs T3 plain, fp8 KV, 128 greedy -> identical text
#   server   1-slot Q27_BATCH=0 servers, Ampere-default KV (turbo5k), T2 vs T3
#            on the four greedy prompts -> identical texts; [req] tps
#   nll      --nll chunk 512 / ctx 512 on both packs (the README tier protocol)
set -u
M=/mnt/ai/projects/q27-master
S=${OUT:-/tmp/bonsai2-gates}; mkdir -p $S
T2=${T2:-/mnt/ai/models/bonsai2-27b/q27/bonsai2-27b-t2-slim.q27}
T3=${T3:-/mnt/ai/models/bonsai2-27b/q27/bonsai2-27b-t3-slim.q27}
TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
NLLCORPUS=${NLLCORPUS:-/mnt/ai/data/wikitext-2-raw/wiki.test.qwen38.i32}
ARGS="--host 127.0.0.1 --port 8090 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0 --ctx 16384"
log() { echo "[t3 $(date '+%H:%M:%S')] $*"; }
sudo -n systemctl stop vox-transcriber vox-transcriber-gmrs; sleep 3
b3=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | grep GPU-5a723c5e || true)
[ -n "$b3" ] && { log "3090 busy: $b3"; sudo -n systemctl start vox-transcriber vox-transcriber-gmrs; exit 3; }
export CUDA_VISIBLE_DEVICES=1
run_server() { # $1 label, $2 model, rest = -E envs
  local label=$1 model=$2; shift 2
  systemctl --user reset-failed bz-server 2>/dev/null
  systemd-run --user --unit bz-server -E CUDA_VISIBLE_DEVICES=1 -E Q27_PRINT_WSUM=1 -E Q27_BATCH=0 "$@" \
    -p StandardOutput=truncate:$S/t3s_$label.log -p StandardError=truncate:$S/t3s_$label.log \
    $M/build/q27-server $model $TOK $ARGS --slots 1 >/dev/null
  for i in $(seq 1 200); do c=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/health); [ "$c" = 200 ] && break; systemctl --user is-active --quiet bz-server || { log "$label died"; tail -4 $S/t3s_$label.log; return 1; }; sleep 2; done
  log "== $label: $(grep -m1 -E 'KV cache:' $S/t3s_$label.log | cut -c1-80) | $(grep -m1 -E 'wsum' $S/t3s_$label.log | cut -c1-80)"
  python3 - "$label" "$S" <<'PY'
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
json.dump(out, open(f"{S}/t3s_{label}.json","w"))
PY
  grep "\[req\]" $S/t3s_$label.log | sed 's/.*dec=\([0-9]*\) dec_ms=\([0-9.]*\).*rounds=\([0-9]*\) tps=\([0-9.]*\).*/    dec=\1 dec_ms=\2 rounds=\3 tps=\4/' | head -4
  systemctl --user stop bz-server; sleep 2
}
for leg in ${LEGS:-gate kernels ninv cli server nll}; do
  case $leg in
    gate)
      log "== t3_gate: T2 vs T3 bitwise (gemv w1/2/5/8 + prefill conversion), --bench"
      $M/build/t3_gate $T2 $T3 --bench > $S/t3_gate.log 2>&1
      tail -1 $S/t3_gate.log; grep -c " PASS" $S/t3_gate.log | sed 's/^/    PASS lines: /'; grep " FAIL" $S/t3_gate.log | head -5
      grep "bench" $S/t3_gate.log ;;
    kernels)
      log "== test_kernels on the T3 pack"
      $M/build/test_kernels $T3 > $S/t3_test_kernels.log 2>&1
      echo "    PASS $(grep -c PASS $S/t3_test_kernels.log) FAIL $(grep -c FAIL $S/t3_test_kernels.log)"; grep FAIL $S/t3_test_kernels.log | head -5
      grep -E "gemv10|T3_G128|gemv_n" $S/t3_test_kernels.log | head -8 ;;
    ninv)
      log "== ninv_test on the T3 pack"
      $M/build/ninv_test $T3 > $S/t3_ninv.log 2>&1; tail -3 $S/t3_ninv.log ;;
    cli)
      log "== CLI canonical: T2 vs T3 plain (fp8 KV, 128 greedy)"
      for p in t2 t3; do pk=$T2; [ $p = t3 ] && pk=$T3
        Q27_KV=fp8 timeout 900 $M/build/q27 $pk --tokens "760,6511,314,9338,369" -n 128 --ctx 2048 > $S/t3_cli_$p.log 2>&1
      done
      a=$(grep -m1 "^generated:" $S/t3_cli_t2.log | md5sum | cut -c1-8); b=$(grep -m1 "^generated:" $S/t3_cli_t3.log | md5sum | cut -c1-8)
      log "generated md5: t2 $a  t3 $b  -> $([ "$a" = "$b" ] && echo IDENTICAL || echo DIFFER)"
      for p in t2 t3; do echo "    $p: $(grep -E 't/s' $S/t3_cli_$p.log | head -2 | tr '\n' ' ' | cut -c1-200)"; done ;;
    server)
      run_server t2k5 $T2
      run_server t3k5 $T3
      python3 - "$S" <<'PY'
import json, sys
S=sys.argv[1]
a=json.load(open(f"{S}/t3s_t2k5.json")); b=json.load(open(f"{S}/t3s_t3k5.json"))
for k in a:
    if a[k]==b[k]: print(f"  {k:7s}: IDENTICAL T2 vs T3 ({len(a[k])} chars)")
    else:
        i=next((i for i,(x,y) in enumerate(zip(a[k],b[k])) if x!=y), min(len(a[k]),len(b[k])))
        print(f"  {k:7s}: DIFFER at char {i} of {len(a[k])}/{len(b[k])}")
PY
      ;;
    nll)
      log "== --nll chunk 512 ctx 512 (tier protocol): T3 then T2"
      for p in t3 t2; do pk=$T3; [ $p = t2 ] && pk=$T2
        Q27_KV=fp8 timeout 3600 $M/build/q27 $pk --nll $NLLCORPUS --nll-chunk 512 --ctx 512 > $S/t3_nll_$p.log 2>&1
        echo "    $p: $(grep -E 'NLL|PPL|ppl|nll' $S/t3_nll_$p.log | tail -2 | tr '\n' ' ' | cut -c1-220)"
      done ;;
  esac
done
sudo -n systemctl start vox-transcriber vox-transcriber-gmrs
log "T3-DONE"
