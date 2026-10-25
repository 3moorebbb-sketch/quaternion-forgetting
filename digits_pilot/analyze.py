import json, numpy as np
R=[json.loads(l) for l in open('sweep.jsonl')]
assert all(r['finite'] for r in R), "non-finite run"
A=['qs3','qfree','real28','real64']; RO=['shared','split']; O=['sgd','adam','mep']
def rows(a,ro,o,lr,seeds): return sorted([r for r in R if r['arch']==a and r['readout']==ro and r['opt']==o and r['lr']==lr and r['seed'] in seeds],key=lambda r:r['seed'])
TUNE,TEST=range(1,4),range(4,16)
best={}
for a in A:
  for ro in RO:
    for o in O:
      lrs=sorted(set(r['lr'] for r in R if r['opt']==o))
      lr=max(lrs,key=lambda lr:np.mean([r['acc'] for r in rows(a,ro,o,lr,TUNE)]))
      best[(a,ro,o)]=(lr,rows(a,ro,o,lr,TEST))
rng=np.random.default_rng(0)
def boot(x,y=None,paired=False):
  x=np.array(x); y=None if y is None else np.array(y); d=[]
  for _ in range(10000):
    i=rng.integers(0,len(x),len(x))
    if y is None: d.append(x[i].mean())
    elif paired: d.append((x[i]-y[i]).mean())
    else: j=rng.integers(0,len(y),len(y)); d.append(x[i].mean()-y[j].mean())
  d=np.sort(d); return d[249],d[9749]
g=lambda a,ro,o,k:[r[k] for r in best[(a,ro,o)][1]]
f=lambda d,c:f"{d:+5.1f} [{c[0]:+.1f}, {c[1]:+.1f}]"
print("Held-out seeds 4–15 (n=12), lr chosen on seeds 1–3 by final ACC.  ACC = mean accuracy over 5 tasks after task 5; FGT = average forgetting of tasks 1–4")
for ro in RO:
  print(f"\n[{ro} head]            "+"".join(f"{o:>24s}" for o in O))
  for a in A: print(f"  {a:7s} ACC±sd / FGT  "+"".join(f"{np.mean(g(a,ro,o,'acc')):6.1f}±{np.std(g(a,ro,o,'acc'),ddof=1):3.1f} / {np.mean(g(a,ro,o,'fgt')):4.1f} (lr {best[(a,ro,o)][0]})".rjust(24) for o in O))
print("\n1) Readout: split − shared, ACC")
for a in A: print(f"  {a:7s}"+"".join(f"  {o}: "+f(np.mean(g(a,'split',o,'acc'))-np.mean(g(a,'shared',o,'acc')),boot(g(a,'split',o,'acc'),g(a,'shared',o,'acc'),True)) for o in O))
print("\n2) S³ vs free (same initial function, same batches; paired): qfree − qs3, ACC")
for ro in RO: print(f"  {ro:7s}"+"".join(f"  {o}: "+f(np.mean(g('qfree',ro,o,'acc'))-np.mean(g('qs3',ro,o,'acc')),boot(g('qfree',ro,o,'acc'),g('qs3',ro,o,'acc'),True)) for o in O))
print("\n3) Quaternion (free) vs real, ACC")
for ro in RO:
  for b in ['real28','real64']: print(f"  {ro:7s} qfree − {b}:"+"".join(f"  {o}: "+f(np.mean(g('qfree',ro,o,'acc'))-np.mean(g(b,ro,o,'acc')),boot(g('qfree',ro,o,'acc'),g(b,ro,o,'acc'))) for o in O))
print("\n4) Optimizer (paired by seed), ACC:  adam − sgd   |   mep − adam")
for ro in RO:
  for a in A:
    x=g(a,ro,'adam','acc'); y=g(a,ro,'sgd','acc'); z=g(a,ro,'mep','acc')
    print(f"  {ro:7s} {a:7s} adam−sgd {f(np.mean(x)-np.mean(y),boot(x,y,True))}   mep−adam {f(np.mean(z)-np.mean(x),boot(z,x,True))}")
print("\n5) Probe vs head on old tasks (1–4) after task 5, best config per optimizer")
for ro in RO:
  for a in A: print(f"  {ro:7s} {a:7s}"+"".join(f"  {o}: head {np.mean(g(a,ro,o,'head_old')):5.1f} probe {np.mean(g(a,ro,o,'probe_old')):5.1f}" for o in O))
print("\nlearning accuracy (acc on each task right after learning it), shared head:", {a:round(float(np.mean(g(a,'shared','sgd','la'))),1) for a in A})
json.dump({f"{a}|{ro}|{o}":{"lr":best[(a,ro,o)][0],"acc":float(np.mean(g(a,ro,o,'acc'))),"acc_sd":float(np.std(g(a,ro,o,'acc'),ddof=1)),"fgt":float(np.mean(g(a,ro,o,'fgt'))),"head_old":float(np.mean(g(a,ro,o,'head_old'))),"probe_old":float(np.mean(g(a,ro,o,'probe_old')))} for a in A for ro in RO for o in O},open('summary.json','w'),indent=1)
