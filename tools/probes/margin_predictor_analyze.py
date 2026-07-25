import numpy as np, itertools, sys
rows=[]
for l in open("margins.csv"):
    p=l.strip().split(",")
    if len(p)<4: continue
    cap,n,md=int(p[0]),int(p[1]),int(p[2])
    m=[float(x) for x in p[3:3+md]]
    if len(m)!=md: continue
    rows.append((cap,n,md,m))
md_mode=max(set(r[2] for r in rows), key=[r[2] for r in rows].count)
rows=[r for r in rows if r[2]==md_mode]
X=np.array([r[3] for r in rows]); y=np.array([r[1] for r in rows]); cap=np.array([r[0] for r in rows])
N=len(y)
print(f"rounds={N}  md={md_mode}  n range {y.min()}..{y.max()}  mean n={y.mean():.2f}")
print("n distribution:", {int(v):int(c) for v,c in zip(*np.unique(y,return_counts=True))})

def topk_acc(pred_table, keys, y, k=1):
    """pred_table: key -> list of n sorted by train frequency"""
    hit=0
    glob=list(np.argsort(-np.bincount(y))[:k])
    for kk,yy in zip(keys,y):
        cand=pred_table.get(kk, glob)[:k]
        if yy in cand: hit+=1
    return hit/len(y)

def build(keys,y,k=3):
    from collections import defaultdict, Counter
    d=defaultdict(Counter)
    for kk,yy in zip(keys,y): d[kk][yy]+=1
    return {kk:[v for v,_ in c.most_common(k)] for kk,c in d.items()}

def evaluate(name, keyfn, tr, te):
    ktr=[keyfn(X[i]) for i in tr]; kte=[keyfn(X[i]) for i in te]
    t=build(ktr,y[tr])
    return name, topk_acc(t,kte,y[te],1), topk_acc(t,kte,y[te],2)

splits={}
idx=np.arange(N)
rng=np.random.default_rng(7); perm=rng.permutation(N)
splits["random (i.i.d., optimistic)"]=(perm[:N//2], perm[N//2:])
splits["block (held-out traffic)"]=(idx[:int(N*0.6)], idx[int(N*0.6):])

# quantile bin edges from the whole set (feature definition, not label info)
qs=[np.quantile(X[:,k],[0.25,0.5,0.75]) for k in range(md_mode)]
def binv(v,k): return int(np.searchsorted(qs[k],v))

feats=[
 ("cap only (leading run @ theta=0.5)", lambda x: int(np.argmax(np.concatenate([x<0.5,[True]])))),
 ("margin[0] quartile",                 lambda x: binv(x[0],0)),
 ("all margins, quartile-binned",       lambda x: tuple(binv(x[k],k) for k in range(md_mode))),
 ("all margins, median-binned",         lambda x: tuple(int(x[k]>qs[k][1]) for k in range(md_mode))),
 ("run-length sweep + margin[0] bin",   lambda x: (int(np.argmax(np.concatenate([x<0.5,[True]]))), binv(x[0],0))),
]
for sname,(tr,te) in splits.items():
    print(f"\n--- split: {sname}   (train {len(tr)}, test {len(te)}) ---")
    print(f"{'predictor':<40} {'top-1':>7} {'top-2':>7}")
    base=np.bincount(y[te]).max()/len(te)
    print(f"{'always-predict-mode (no signal)':<40} {base*100:6.1f}% {'-':>7}")
    for nm,fn in feats:
        _,a1,a2=evaluate(nm,fn,tr,te)
        print(f"{nm:<40} {a1*100:6.1f}% {a2*100:6.1f}%")

# best single threshold for the run-length rule
print("\n--- leading-run threshold sweep (predict n = run+1), whole set ---")
best=None
for t in [0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0]:
    pred=np.array([int(np.argmax(np.concatenate([x<t,[True]])))+1 for x in X])
    acc=(pred==y).mean()
    print(f"  theta={t:<5} exact-n accuracy {acc*100:5.1f}%")
    if best is None or acc>best[1]: best=(t,acc)
print(f"  best theta={best[0]} at {best[1]*100:.1f}%")

# how much does the margin actually tell us? mutual-information-ish check
print("\n--- is the signal there at all? mean margin by outcome ---")
for k in range(md_mode):
    acc_k = y > k          # step k was accepted
    a=X[acc_k,k].mean(); b=X[~acc_k,k].mean()
    print(f"  step {k}: mean margin when accepted {a:6.2f} | when rejected {b:6.2f} | gap {a-b:+.2f}")
