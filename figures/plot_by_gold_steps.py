import json
import argparse
from collections import defaultdict
from statistics import NormalDist

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

_NORM = NormalDist()


def count_gold_steps(answer: str) -> int:
    lines = [l.strip() for l in answer.strip().split("\n") if l.strip()]
    return sum(1 for l in lines if not l.startswith("####"))


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center, center - half, center + half


def bootstrap_mean_ci(values, alpha=0.05, n_boot=10000, seed=0):
    """BCa bootstrap CI for the mean of a bounded [0, 1] quantity.

    Returns (mean, lo, hi). The interval is non-parametric, respects the
    [0, 1] support by construction, and does not assume symmetry -- which
    matters because stable_match_frac tends to pile up against 1.0.

    Special cases:
      * all values in {0, 1}  -> Wilson interval (the mean is a binomial
        proportion; the bootstrap degenerates at the boundary)
      * zero sample variance  -> degenerate point interval
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    m = float(v.mean())

    if np.all((v == 0.0) | (v == 1.0)):
        _, lo, hi = wilson_ci(float(v.sum()), n)
        return m, max(0.0, lo), min(1.0, hi)

    if n == 1 or np.ptp(v) == 0.0:
        return m, m, m

    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)

    def percentile_interval():
        return (m,
                float(np.quantile(boot, alpha / 2)),
                float(np.quantile(boot, 1 - alpha / 2)))

    # Bias correction. The mid-p form (half credit for ties) keeps z0 finite
    # when many resamples land exactly on the observed mean.
    below = np.count_nonzero(boot < m) + 0.5 * np.count_nonzero(boot == m)
    p0 = below / n_boot
    if not 0.0 < p0 < 1.0:
        return percentile_interval()
    z0 = _NORM.inv_cdf(p0)

    # Acceleration, estimated by jackknife (leave-one-out means).
    jack = (v.sum() - v) / (n - 1)
    u = jack.mean() - jack
    denom = 6.0 * float(np.sum(u**2)) ** 1.5
    a = float(np.sum(u**3)) / denom if denom > 0 else 0.0

    quantiles = []
    for q in (alpha / 2, 1 - alpha / 2):
        zq = _NORM.inv_cdf(q)
        scale = 1 - a * (z0 + zq)
        if scale <= 0:
            return percentile_interval()
        adj = z0 + (z0 + zq) / scale
        if not np.isfinite(adj):
            return percentile_interval()
        quantiles.append(min(max(_NORM.cdf(adj), 1 / n_boot), 1 - 1 / n_boot))

    lo, hi = (float(np.quantile(boot, q)) for q in quantiles)
    return m, max(0.0, lo), min(1.0, hi)


DISPLAY_NAME = {"sft": "ERM", "coconut": "CoCoNuT", "codi": "CoDi"}

BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({"font.size": 14})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["sft", "coconut", "codi"],
                        help="Which model's results to plot")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="Minimum samples per bin to include (default: 5)")
    parser.add_argument("--n-boot", type=int, default=10000,
                        help="Bootstrap resamples for the stable-match CI")
    parser.add_argument("--seed", type=int, default=0,
                        help="Bootstrap seed, so reruns give identical bands")
    args = parser.parse_args()

    print("Loading GSM8k test split...", flush=True)
    dataset = load_dataset("gsm8k", "main", split="test")
    gold_steps = [count_gold_steps(ex["answer"]) for ex in dataset]

    results_path = f"results/{args.model}.json"
    print(f"Loading {results_path}...", flush=True)
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    assert len(results) == len(dataset), (
        f"Result count {len(results)} != dataset size {len(dataset)}"
    )

    buckets = defaultdict(lambda: {"correct": [], "stable_match_frac": []})

    for entry in results:
        idx = entry["sample_idx"]
        num_gold = gold_steps[idx]
        is_correct = int(entry["original_result"]["is_correct"][-1])
        smf = entry["original_result"]["stable_match_frac"]
        buckets[num_gold]["correct"].append(is_correct)
        buckets[num_gold]["stable_match_frac"].append(smf)

    step_counts = sorted(k for k, v in buckets.items() if len(v["correct"]) >= args.min_samples)

    acc_means, acc_lo, acc_hi = [], [], []
    smf_means, smf_lo, smf_hi = [], [], []

    for s in step_counts:
        correct = buckets[s]["correct"]
        smf_vals = buckets[s]["stable_match_frac"]

        n = len(correct)
        _, lo, hi = wilson_ci(sum(correct), n)
        acc_means.append(sum(correct) / n)
        acc_lo.append(lo)
        acc_hi.append(hi)

        m, lo2, hi2 = bootstrap_mean_ci(smf_vals, n_boot=args.n_boot, seed=args.seed)
        smf_means.append(m)
        smf_lo.append(lo2)
        smf_hi.append(hi2)

    x = np.array(step_counts)
    acc_means = np.array(acc_means)
    smf_means = np.array(smf_means)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(x, acc_means, marker="o", markersize=8, linewidth=2,
            color=BLUE, label="Accuracy")
    ax.fill_between(x, acc_lo, acc_hi, alpha=0.15, color=BLUE, linewidth=0)

    ax.plot(x, smf_means, marker="s", markersize=8, linewidth=2,
            color=ORANGE, label="Stable match fraction")
    ax.fill_between(x, smf_lo, smf_hi, alpha=0.15, color=ORANGE, linewidth=0)

    ax.set_xlabel("Number of steps in gold reasoning trace")
    ax.set_ylabel("Fraction")
    ax.set_title(f"{DISPLAY_NAME[args.model]}: accuracy & stable match fraction\nby gold reasoning steps")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="center right")
    ax.grid(axis="y", linewidth=0.5, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate sample counts
    for i, s in enumerate(step_counts):
        n = len(buckets[s]["correct"])
        ax.annotate(f"n={n}", (x[i], 0.02), ha="center", color=MUTED)

    out = f"{args.model}_by_gold_steps.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()