import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

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


def mean_ci(values, z=1.96):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return m, m - z * se, m + z * se


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["sft", "coconut", "codi"],
                        help="Which model's results to plot")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="Minimum samples per bin to include (default: 5)")
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

        m, lo2, hi2 = mean_ci(smf_vals)
        smf_means.append(m)
        smf_lo.append(lo2)
        smf_hi.append(hi2)

    x = np.array(step_counts)
    acc_means = np.array(acc_means)
    smf_means = np.array(smf_means)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x, acc_means, marker="o", color="steelblue", label="Accuracy")
    ax.fill_between(x, acc_lo, acc_hi, alpha=0.2, color="steelblue")

    ax.plot(x, smf_means, marker="s", color="darkorange", label="Stable match fraction")
    ax.fill_between(x, smf_lo, smf_hi, alpha=0.2, color="darkorange")

    ax.set_xlabel("Number of steps in gold reasoning trace")
    ax.set_ylabel("Fraction")
    ax.set_title(f"{args.model.upper()} — accuracy & stable match fraction by gold reasoning steps")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)

    # Annotate sample counts
    for i, s in enumerate(step_counts):
        n = len(buckets[s]["correct"])
        ax.annotate(f"n={n}", (x[i], 0.02), ha="center", fontsize=7, color="gray")

    out = f"{args.model}_by_gold_steps.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
