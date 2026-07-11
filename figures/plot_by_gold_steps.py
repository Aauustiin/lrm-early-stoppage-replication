import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_common import (
    FIGURES_DIR, DISPLAY_NAME, BLUE, ORANGE,
    load_gold_steps, load_results, wilson_ci, bootstrap_mean_ci,
)

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

    gold_steps = load_gold_steps()
    data = load_results(args.model, expected_len=len(gold_steps))
    results = data["results"]

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

    out = FIGURES_DIR / f"{args.model}_by_gold_steps.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()