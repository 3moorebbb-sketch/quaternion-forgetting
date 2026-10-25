# Shared readouts, not S³: what drives forgetting in a toy quaternion network

[![DOI](https://zenodo.org/badge/1432961308.svg)](https://doi.org/10.5281/zenodo.23638141)

**[Live workbench](https://3moorebbb-sketch.github.io/quaternion-forgetting/)** · **[Write-up (PDF)](paper.pdf)**

Does putting a network's weights on the unit 3-sphere help it remember an old task while learning a new one? Not in these experiments.

In a toy problem, a small network learns Task A (a circle), then Task B (a diagonal half-plane), and we measure how much of Task A survives. Forgetting in a quaternion network comes mostly from how the tasks read the output, not from the sphere. A pilot on real images (split digits) confirms that. It also finds a modest advantage for quaternion layers once each task has its own output. One toy result, the S³ step-size cost, did not transfer.

## Findings

| | Result | Status |
| --- | --- | --- |
| 1 | **The readout matters most.** Giving each task its own output recovers 15–23 points of Task A in the toy and adds 13–18 points of final accuracy on split digits. Reading tasks from the i and j parts of one output quaternion behaves like a single shared head. | Known pattern (shared vs per-task heads); holds on digits |
| 2 | **Quaternion layers forget less with split outputs (digits).** The free quaternion network beats a real MLP of about the same size by +2.2 points under SGD [1.7, 2.7] and +1.4 under Adam [0.8, 1.9], on 30 fresh seeds, and matches one with 3.4× the parameters. It learns each task as well and forgets less. The toy showed no such advantage. | Pilot result; needs a weight-sharing (PHM) control on a standard benchmark |
| 3 | **The S³ step-size cost is toy-specific.** In the toy, writing w = A·q shrinks direction updates by A²; undoing that recovers +9.4 points [6.0, 13.0], under SGD and Adam. On split digits, S³ and free weights tie under SGD and S³ is slightly better under Adam (+0.8). | Did not transfer |
| 4 | **Old tasks are misread, not erased.** A linear probe still reads Task A at 98.7–99.4% in the toy, and old digit tasks at 92–99% while a shared head scores 76–80%. | Replicates [Davari et al., 2022](https://mlanthology.org/cvpr/2022/davari2022cvpr-probing/); holds on digits |
| 5 | **Good protection erases the differences (toy).** With replay or refreshed orthogonal-gradient projection, every network keeps 95–97% of Task A. | — |

Brackets are 95% bootstrap intervals. In both studies, settings were tuned on some seeds and reported on others, and headline comparisons were rerun on fresh seeds. One exploratory toy result did not replicate and is reported as null in the write-up.

## Repository layout

```
index.html        Interactive workbench (toy study: live runs, multi-seed benchmark, architecture study)
paper.pdf         Write-up with figures, related work and limitations
engine.js         Toy-study networks, update rules and evaluation; seeded, no dependencies
archsweep.js      Learning-rate × knob sweep for one network and seed (seeds 1–12)
analyze_arch.js   Tune on seeds 1–6, report held-out seeds 7–12, bootstrap intervals
confirm.js        Plain-SGD confirmation runs (fresh seeds 13–36)
ablate.js         S³-cost ablations from one shared checkpoint
adam.js           The same comparisons with Adam for Task B
probe.js          Linear probes, frozen-head and memory-refit readouts
results/          Raw outputs behind every toy-study number
digits_pilot/     Split-digits pilot (Python): engine, sweep, analysis, confirmation, raw results
```

## Reproducing

**Toy study.** Requires Node.js 18 or later; there is nothing to install.

```bash
mkdir -p as && node archsweep.js qs3 1   # one network, one seed, full sweep (writes to as/)
cp -r results/archsweep_seeds1-12 as && node analyze_arch.js   # held-out table and intervals
node confirm.js qfreesep 13           # one confirmation run
node ablate.js 13 precond 0.035       # one ablation: riem | euclid | nofloor | precond | wspace
node adam.js 13 qs3sep native 0.005   # Adam: native | wspace
node probe.js qs3 13                  # probes for one network and seed
```

**Digits pilot.** Requires Python 3 with NumPy and scikit-learn; the data ships with scikit-learn, so nothing is downloaded.

```bash
cd digits_pilot
python3 sweep.py      # 1,080 runs, about 4 minutes on 2 cores -> sweep.jsonl
python3 analyze.py    # learning rate chosen on seeds 1–3, reported on seeds 4–15
python3 confirm.py    # key comparisons on fresh seeds 16–45
```

Every run is deterministic. In the toy study, the checkpoint RNG is `seed·7919+1`, memory is `seed·7919+2` and the Task B stream is `seed·104729+3` (mulberry32). Gradients in both engines were checked against finite differences (relative error at most 3×10⁻⁶). The workbench runs the toy engine in the browser and reproduces the study data exactly for matching seeds and settings.

## Networks compared

**Toy study**

| Network | Weights | Task readout | Parameters |
| --- | --- | --- | --- |
| Quaternion, S³ | A·q, unit q on S³, Riemannian updates | one output quaternion: A reads i, B reads j | 292 |
| Quaternion, free | w ∈ ℝ⁴, Euclidean updates, same initial function | shared, as above | 244 |
| Quaternion, S³, split | as S³ | one output quaternion per task | 326 |
| Quaternion, free, split | as free | split | 272 |
| Real MLP 14 | 2→14→14→2, tanh | an output row per task | 282 |
| Real MLP 24 | 2→24→24→2, width-matched | per task | 722 |

The six Task B update rules are plain SGD, weight anchoring, output anchoring, replay, orthogonal-gradient projection with a fixed basis, and the same projection with its basis recomputed every 10 steps.

**Digits pilot.** scikit-learn's 8×8 digits as five binary tasks learned in sequence (0/1, 2/3, 4/5, 6/7, 8/9), with a shared head or one head per task. Networks: S³ and free quaternion networks (16→16→16 quaternions, 2,200–3,100 parameters), and real MLPs 28 wide (about the same parameters) and 64 wide (same width). Optimizers: SGD, Adam, and an Adam variant without first-moment bias correction.

## Limitations

Both studies are small: two-dimensional inputs in the toy, 8×8 images in the pilot, networks of a few hundred to a few thousand parameters, and one task order. A quaternion layer is also a real matrix with structured weight sharing, so the digits advantage may come from that sharing rather than from quaternion algebra. The next test is split MNIST with parameterized hypercomplex (PHM) layers as a weight-sharing control.

## How this was made

The experiments, analysis code and write-up were produced with AI assistance (Claude, by Anthropic), directed and reviewed by the author. All numbers come from the code and raw results in this repository.

## License

Code is released under the MIT License. The write-up and figures are released under CC BY 4.0.

## Citation

If you use this work, please cite it using the metadata in [`CITATION.cff`](CITATION.cff), or GitHub's "Cite this repository" button.
