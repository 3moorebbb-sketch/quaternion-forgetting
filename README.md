# Shared readouts, not S³: what drives forgetting in a toy quaternion network

[![DOI](https://zenodo.org/badge/1432961308.svg)](https://doi.org/10.5281/zenodo.23638141)

**[Live workbench](https://3moorebbb-sketch.github.io/quaternion-forgetting/)** · **[Write-up (PDF)](paper.pdf)**

Does putting a network's weights on the unit 3-sphere help it remember an old task while learning a new one? In this toy setting, no.

A small network learns Task A (a circle), then Task B (a diagonal half-plane), and we measure how much of Task A survives. The forgetting in a quaternion network comes mostly from how the two tasks read the output. The rest comes from writing each weight as amplitude × unit quaternion, not from the sphere itself.

## Findings

| | Result | Status |
| --- | --- | --- |
| 1 | **The readout matters most.** Giving each task its own output quaternion recovers 15–23 points of Task A. Reading both tasks from the i and j parts of one output quaternion behaves like a single shared head. | Known pattern (shared vs per-task heads) |
| 2 | **The S³ cost is a step-size effect.** Writing w = A·q shrinks every direction update by A². Dropping the unit-norm constraint changes nothing (+0.0); dividing the q step by A² recovers +9.4 points [6.0, 13.0]. The cost survives Adam (+9.4 to +10.6). | Mechanism known (weight normalization); the forgetting cost appears unreported |
| 3 | **No quaternion advantage.** With split outputs and free weights, the quaternion network matches real MLPs of equal size or width to within 0.3 points. | Null result |
| 4 | **Task A is misread, not erased.** After Task B, a linear probe still reads Task A at 98.7–99.4% from every network's hidden layer. Refitting only the readout on 100 stored points recovers 90–96%. | Replicates [Davari et al., 2022](https://mlanthology.org/cvpr/2022/davari2022cvpr-probing/) |
| 5 | **Good protection erases the differences.** With replay or refreshed orthogonal-gradient projection, every network keeps 95–97% of Task A. | — |

Figures are Task A accuracy when Task B first reaches 95% (results 1–3), or at step 1,000 (results 4–5). Brackets are 95% bootstrap intervals. Headline comparisons were rerun on 24 fresh seeds that played no part in tuning. One exploratory result did not replicate and is reported as null in the write-up.

## Repository layout

```
index.html        Interactive workbench (live runs, multi-seed benchmark, architecture study)
paper.pdf         Write-up with figures, related work and limitations
engine.js         Networks, update rules and evaluation; seeded, no dependencies
archsweep.js      Learning-rate × knob sweep for one network and seed (seeds 1–12)
analyze_arch.js   Tune on seeds 1–6, report held-out seeds 7–12, bootstrap intervals
confirm.js        Plain-SGD confirmation runs (fresh seeds 13–36)
ablate.js         S³-cost ablations from one shared checkpoint
adam.js           The same comparisons with Adam for Task B
probe.js          Linear probes, frozen-head and memory-refit readouts
results/          Raw outputs behind every number in the write-up
```

## Reproducing

Requires Node.js 18 or later. There is nothing to install.

```bash
mkdir -p as && node archsweep.js qs3 1   # one network, one seed, full sweep (writes to as/)
cp -r results/archsweep_seeds1-12 as && node analyze_arch.js   # held-out table and intervals
node confirm.js qfreesep 13           # one confirmation run
node ablate.js 13 precond 0.035       # one ablation: riem | euclid | nofloor | precond | wspace
node adam.js 13 qs3sep native 0.005   # Adam: native | wspace
node probe.js qs3 13                  # probes for one network and seed
```

Every run is deterministic. The checkpoint RNG is `seed·7919+1`, memory is `seed·7919+2` and the Task B stream is `seed·104729+3` (mulberry32). Gradients were checked against finite differences (relative error at most 3×10⁻⁶). The workbench runs the same engine in the browser and reproduces the study data exactly for matching seeds and settings.

## Networks compared

| Network | Weights | Task readout | Parameters |
| --- | --- | --- | --- |
| Quaternion, S³ | A·q, unit q on S³, Riemannian updates | one output quaternion: A reads i, B reads j | 292 |
| Quaternion, free | w ∈ ℝ⁴, Euclidean updates, same initial function | shared, as above | 244 |
| Quaternion, S³, split | as S³ | one output quaternion per task | 326 |
| Quaternion, free, split | as free | split | 272 |
| Real MLP 14 | 2→14→14→2, tanh | an output row per task | 282 |
| Real MLP 24 | 2→24→24→2, width-matched | per task | 722 |

The six Task B update rules are plain SGD, weight anchoring, output anchoring, replay, orthogonal-gradient projection with a fixed basis, and the same projection with its basis recomputed every 10 steps.

## Limitations

This is a toy problem: one task pair, two-dimensional inputs, networks of a few hundred parameters, and SGD or Adam for Task B. The natural next steps are the same 2×2 comparison on a standard benchmark such as split MNIST with quaternion layers, and the A² cost under optimizers that decouple magnitude and direction.

## How this was made

The experiments, analysis code and write-up were produced with AI assistance (Claude, by Anthropic), directed and reviewed by the author. All numbers come from the code and raw results in this repository.

## License

Code is released under the MIT License. The write-up and figures are released under CC BY 4.0.

## Citation

If you use this work, please cite it using the metadata in [`CITATION.cff`](CITATION.cff), or GitHub's "Cite this repository" button.
