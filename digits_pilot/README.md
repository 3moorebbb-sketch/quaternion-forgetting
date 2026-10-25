# Split-digits pilot

Does the toy quaternion study transfer to real images? scikit-learn 8×8 digits, five binary tasks
(0/1, 2/3, 4/5, 6/7, 8/9) learned in sequence; shared head (domain-incremental) vs per-task heads
(task-incremental); S³ and free quaternion networks (identical initial function), and real MLPs
matched on parameters (28 wide) and width (64 wide); SGD, Adam and the MEP optimizer.

Python 3 with NumPy and scikit-learn; no downloads.

    python3 sweep.py      # 1,080 runs (≈4 min on 2 cores) -> sweep.jsonl
    python3 analyze.py    # lr chosen on seeds 1–3, reported on seeds 4–15, bootstrap CIs
    python3 confirm.py    # key comparisons on fresh seeds 16–45
    python3 mep_equiv_adam.py   # MEP optimizer == Adam without first-moment bias correction

Gradients were checked against finite differences (relative error ≤ 4×10⁻⁷).
