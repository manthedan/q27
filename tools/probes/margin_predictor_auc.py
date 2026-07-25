import numpy as np
rows=[]
for l in open("margins.csv"):
    p=l.strip().split(",")
    if len(p)<4: continue
    cap,n,md=int(p[0]),int(p[1]),int(p[2])
    m=[float(x) for x in p[3:3+md]]
    if len(m)==md: rows.append((cap,n,md,m))
rows=[r for r in rows if r[2]==4]
X=np.array([r[3] for r in rows]); y=np.array([r[1] for r in rows]); N=len(y)

def auc(scores, labels):
    o=np.argsort(scores); r=np.empty(N_:=len(scores)); r[o]=np.arange(1,N_+1)
    p=labels.sum(); q=len(labels)-p
    if p==0 or q==0: return float('nan')
    return (r[labels==1].sum()-p*(p+1)/2)/(p*q)

print("Separation of the drafter's own confidence, per step")
print(f"{'step':<6} {'n_acc':>7} {'n_rej':>7} {'mean(acc)':>10} {'mean(rej)':>10} {'AUC':>7}")
for k in range(1,4):
    lab=(y>k).astype(int)
    print(f"{k:<6} {lab.sum():7d} {len(lab)-lab.sum():7d} {X[lab==1,k].mean():10.2f} "
          f"{X[lab==0,k].mean():10.2f} {auc(X[:,k],lab):7.3f}")

# a few more feature ideas, honest held-out block split
from collections import defaultdict, Counter
def build(keys,yy,k=3):
    d=defaultdict(Counter)
    for a,b in zip(keys,yy): d[a][b]+=1
    return {a:[v for v,_ in c.most_common(k)] for a,c in d.items()}
def acc_at(t,keys,yy,k):
    g=list(np.argsort(-np.bincount(yy))[:k]); h=0
    for a,b in zip(keys,yy):
        if b in t.get(a,g)[:k]: h+=1
    return h/len(yy)
idx=np.arange(N); tr=idx[:int(N*.6)]; te=idx[int(N*.6):]
qs=[np.quantile(X[:,k],[0.5]) for k in range(4)]
def b(v,k): return int(v>qs[k][0])
feats={
 "min margin over steps (binned)": lambda x: int(np.digitize(x.min(),[0.5,1.0,2.0,4.0])),
 "sum of margins (binned)":        lambda x: int(np.digitize(x.sum(),[4,8,12,16,24])),
 "run@1.5 + min-margin bin":       lambda x: (int(np.argmax(np.concatenate([x<1.5,[True]]))), int(np.digitize(x.min(),[0.5,1.5,3.0]))),
 "all margins median-binned":      lambda x: tuple(b(x[k],k) for k in range(4)),
 "run@1.5 only":                   lambda x: int(np.argmax(np.concatenate([x<1.5,[True]]))),
}
print(f"\nheld-out block split (train {len(tr)} / test {len(te)})")
print(f"{'predictor':<34} {'top-1':>7} {'top-2':>7}")
for nm,fn in feats.items():
    t=build([fn(X[i]) for i in tr], y[tr])
    kte=[fn(X[i]) for i in te]
    print(f"{nm:<34} {acc_at(t,kte,y[te],1)*100:6.1f}% {acc_at(t,kte,y[te],2)*100:6.1f}%")
