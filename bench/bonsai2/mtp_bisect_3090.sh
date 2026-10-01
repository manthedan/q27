#!/usr/bin/env bash
# CLI bisect on the 3090: does the ladder (T2+MTP pack) diverge from plain
# (pure T2) within 1500 greedy tokens of the canonical prompt, and which
# ladder ingredient makes it? Each run prints `generated:` ids.
set -u
M=/mnt/ai/projects/q27-master; S=${OUT:-/tmp/bonsai2-gates}; mkdir -p $S
T2=/mnt/ai/models/bonsai2-27b/q27/bonsai2-27b-t2.q27; T2M=/mnt/ai/models/bonsai2-27b/q27/bonsai2-27b-t2-mtp.q27
export CUDA_VISIBLE_DEVICES=1 Q27_KV=fp8
sudo -n systemctl stop vox-transcriber vox-transcriber-gmrs; sleep 3
run() { local label=$1; shift; env "$@" timeout 900 $M/build/q27 ${MODEL} --tokens "760,6511,314,9338,369" -n 1500 --ctx 4096 ${EXTRA:-} > $S/bis_$label.log 2>&1; echo "$label: $(grep -m1 '^generated:' $S/bis_$label.log | md5sum | cut -c1-8) $(grep -oE '[0-9.]+ tokens/round' $S/bis_$label.log | head -1)"; }
MODEL=$T2 EXTRA="" run plain_t2
MODEL=$T2M EXTRA="" run plain_t2mtp          # plain decode on the MTP pack (no --spec): the pack's base tensors only
MODEL=$T2M EXTRA="--spec" run spec_default
MODEL=$T2M EXTRA="--spec" run spec_pmin0 Q27_PMIN=0
MODEL=$T2M EXTRA="--spec" run spec_dexit0 Q27_DEXIT=0
MODEL=$T2M EXTRA="--spec" run spec_maxd4 Q27_MAXD=4 Q27_PMIN=0 Q27_DEXIT=0
MODEL=$T2M EXTRA="--spec" run spec_gemv99 Q27_GEMM_MIN=99
sudo -n systemctl start vox-transcriber vox-transcriber-gmrs
python3 - <<'PY'
import glob, os
S="" + __import__('os').environ.get('OUT','/tmp/bonsai2-gates') + ""
def ids(p):
    for ln in open(p, errors="replace"):
        if ln.startswith("generated:"): return ln.split()[1:]
    return []
ref=ids(f"{S}/bis_plain_t2.log")
for lab in ("plain_t2mtp","spec_default","spec_pmin0","spec_dexit0","spec_maxd4","spec_gemv99"):
    g=ids(f"{S}/bis_{lab}.log")
    i=next((i for i,(a,b) in enumerate(zip(ref,g)) if a!=b), None)
    print(f"  {lab:14s}: {'IDENTICAL' if i is None and len(g)==len(ref) else f'first diff at token {i} (len {len(ref)}/{len(g)})'}")
PY
echo BISECT-DONE
