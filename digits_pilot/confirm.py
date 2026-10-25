import json, numpy as np
from multiprocessing import Pool
import dengine as E
T = E.make_tasks()
CFG = [('qfree','split','sgd',0.1),('real28','split','sgd',0.1),('real64','split','sgd',0.03),
       ('qfree','split','adam',0.001),('real28','split','adam',0.001),('real64','split','adam',0.001),
       ('qs3','shared','sgd',0.03),('qfree','shared','sgd',0.03),('qs3','shared','adam',0.001),('qfree','shared','adam',0.001)]
jobs=[c+(s,) for c in CFG for s in range(16,46)]
def work(j): return json.dumps(E.run(*j, tasks=T, probe=False))
if __name__=='__main__':
    with Pool(2) as p, open('confirm.jsonl','w') as f:
        for line in p.imap_unordered(work, jobs, chunksize=4): f.write(line+'\n')
    R=[json.loads(l) for l in open('confirm.jsonl')]
    g=lambda a,ro,o:np.array([r['acc'] for r in sorted(R,key=lambda r:r['seed']) if r['arch']==a and r['readout']==ro and r['opt']==o])
    rng=np.random.default_rng(1)
    def boot(x,y,paired):
        d=[]
        for _ in range(10000):
            i=rng.integers(0,len(x),len(x))
            d.append((x[i]-y[i]).mean() if paired else x[i].mean()-y[rng.integers(0,len(y),len(y))].mean())
        d=np.sort(d); return d[249],d[9749]
    print("Fresh seeds 16–45 (n=30), settings fixed from the sweep")
    for o in ['sgd','adam']:
        for b in ['real28','real64']:
            x,y=g('qfree','split',o),g(b,'split',o); c=boot(x,y,False)
            print(f"  split  qfree − {b} ({o}): {x.mean()-y.mean():+.1f} [{c[0]:+.1f}, {c[1]:+.1f}]   ({x.mean():.1f} vs {y.mean():.1f})")
    for o in ['sgd','adam']:
        x,y=g('qfree','shared',o),g('qs3','shared',o); c=boot(x,y,True)
        print(f"  shared qfree − qs3 ({o}, paired): {x.mean()-y.mean():+.1f} [{c[0]:+.1f}, {c[1]:+.1f}]   ({x.mean():.1f} vs {y.mean():.1f})")
