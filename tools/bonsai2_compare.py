#!/usr/bin/env python3
"""Compare serial logits under the preregistered Bonsai 2 parity limits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare(reference, candidate):
    ref=np.fromfile(reference,dtype="<f4").astype(np.float64)
    got=np.fromfile(candidate,dtype="<f4").astype(np.float64)
    if not ref.size or ref.size%248320 or got.shape!=ref.shape:
        raise ValueError("logit file lengths must agree and contain complete vocabulary rows")
    if not np.isfinite(ref).all() or not np.isfinite(got).all():
        raise ValueError("nonfinite logits")
    ref=ref.reshape(-1,248320);got=got.reshape(ref.shape)
    def logprobs(x):
        x=x-x.max(axis=1,keepdims=True)
        return x-np.log(np.exp(x).sum(axis=1,keepdims=True))
    lp,lq=logprobs(ref),logprobs(got)
    kl=(np.exp(lp)*(lp-lq)).sum(axis=1)
    rmse=np.sqrt(np.mean((ref-got)**2,axis=1))/np.maximum(np.sqrt(np.mean(ref**2,axis=1)),1e-12)
    top=np.partition(ref,-2,axis=1)[:,-2:]
    margin=top[:,1]-top[:,0]
    ids=ref.argmax(axis=1);actual=got.argmax(axis=1)
    wrong=np.flatnonzero((ids!=actual)&(margin>0.25)).tolist()
    passed=bool(rmse.max()<=0.02 and kl.mean()<=0.01 and kl.max()<=0.05 and not wrong)
    return dict(passed=passed,positions=len(ref),max_relative_rmse=float(rmse.max()),
                mean_kl=float(kl.mean()),max_kl=float(kl.max()),
                argmax_matches=int((ids==actual).sum()),high_margin_mismatches=wrong,
                reference_argmax=ids.tolist(),candidate_argmax=actual.tolist(),
                reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256(Path(candidate).read_bytes()).hexdigest())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference");p.add_argument("candidate");p.add_argument("--output")
    a=p.parse_args();result=compare(a.reference,a.candidate)
    text=json.dumps(result,indent=2)+"\n"
    print(text,end="")
    if a.output:Path(a.output).write_text(text)
    raise SystemExit(0 if result["passed"] else 1)
