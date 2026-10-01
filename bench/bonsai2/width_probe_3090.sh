#!/usr/bin/env bash
set -u
S=${OUT:-/tmp/bonsai2-gates}; mkdir -p $S
sudo -n systemctl stop vox-transcriber vox-transcriber-gmrs; sleep 3
export CUDA_VISIBLE_DEVICES=1 Q27_KV=fp8
M=/mnt/ai/projects/q27-master; T2=/mnt/ai/models/bonsai2-27b/q27/bonsai2-27b-t2.q27
# 550 greedy tokens of plain decode after the code prompt -> a 617-token prompt
timeout 600 $M/build/q27 $T2 --tokens-file $S/req_code.ids -n 550 --ctx 4096 > $S/deep_gen.log 2>&1
python3 - <<'PY'
S="" + __import__('os').environ.get('OUT','/tmp/bonsai2-gates') + ""
ids=[int(x) for x in open(f"{S}/req_code.ids").read().replace("\n","").split(",") if x.strip()]
gen=[]
for ln in open(f"{S}/deep_gen.log", errors="replace"):
    if ln.startswith("generated:"): gen=[int(x) for x in ln.split()[1:]]
open(f"{S}/req_code_deep.ids","w").write(",".join(map(str, ids+gen)))
print("deep prompt ids:", len(ids)+len(gen))
PY
for mode in "" "--batched"; do
  echo "== Bonsai pure T2, DEEP prompt, prefill ${mode:-serial}"
  timeout 900 $M/build/width_probe $T2 --tokens-file $S/req_code_deep.ids $mode 2>&1 | grep -E "^  (fold|reject|trunc|graph|width [248] vs)|width_probe:|refused" | head -14
done
sudo -n systemctl start vox-transcriber vox-transcriber-gmrs
echo DEEP-DONE
