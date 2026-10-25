# Quaternion continual-learning study

Node.js 18+, no dependencies. All results are seeded and reproduce exactly.

- `engine.js` — networks (qs3, qfree, qs3sep, qfreesep, real14, real24), six Task-B methods, evaluation.
- `archsweep.js <arch> <seed>` — lr × knob grid for one network and seed (seeds 1–12 in the study).
- `analyze_arch.js` — tune on seeds 1–6, report held-out 7–12, bootstrap CIs.
- `confirm.js <arch> <seed>` — plain-SGD confirmation runs (seeds 13–36).
- `ablate.js <seed> <riem|euclid|nofloor|precond|wspace> <lr>` — S³-cost ablations from one shared checkpoint.
- `adam.js <seed> <arch> <native|wspace> <lr>` — Task B with Adam instead of SGD (seeds 13–36).
- `probe.js <arch> <seed>` — linear probes on the last hidden layer, frozen-head and memory-refit readouts (seeds 13–36).
- `qcl_workbench.html` — the interactive page; runs the same engine in the browser.
- `results/` — raw outputs behind every number in the write-up.
