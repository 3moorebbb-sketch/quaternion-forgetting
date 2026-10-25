import json, sys, itertools
from multiprocessing import Pool
import dengine as E
T = E.make_tasks()
LR = {'sgd': [0.03, 0.1, 0.3], 'adam': [0.001, 0.003, 0.01], 'mep': [0.001, 0.003, 0.01]}
jobs = [(a, r, o, lr, s) for a in ['qs3', 'qfree', 'real28', 'real64'] for r in ['shared', 'split']
        for o in ['sgd', 'adam', 'mep'] for lr in LR[o] for s in range(1, 16)]
def work(j): return json.dumps(E.run(*j, tasks=T))
if __name__ == '__main__':
    with Pool(2) as p, open('sweep.jsonl', 'w') as f:
        for i, line in enumerate(p.imap_unordered(work, jobs, chunksize=4)):
            f.write(line + '\n')
            if i % 200 == 0: print(i, flush=True)
    print('done', len(jobs))
