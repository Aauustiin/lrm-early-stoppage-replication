"""
Bins GSM8k test samples by number of steps in the gold reasoning trace and
plots, for each model:
  - CODI / COCONUT: mean(stable_match_frac * 6)   [effective latent tokens used]
  - SFT:            mean(num_steps * stable_match_frac)  [effective text steps used]
All three lines on one axes, with 95% CI bands.
"""

import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset


def count_gold_steps(answer: str) -> int:
    lines = [l.strip() for l in answer.strip().split("\n") if l.strip()]
    return sum(1 for l in lines if not l.startswith("####"))


def mean_ci(values, z=1.96):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return m, m - z * se, m + z * se


MODELS = {
    "sft":     {"label": "SFT",     "color": "steelblue"},
    "coconut": {"label": "CoCoNuT", "color": "darkorange"},
    "codi":    {"label": "CODI",    "color": "forestgreen"},
}


def effective_metric(entry, model: str) -> float:
    r = entry["original_result"]
    if model == "sft":
        return r["num_steps"] * r["stable_match_frac"]
    else:  # coconut / codi — 6 latent steps per implicit reasoning step
        return r["stable_match_frac"] * 6


def main():
    print("Loading GSM8k test split...", flush=True)
    dataset = load_dataset("gsm8k", "main", split="test")
    gold_steps = [count_gold_steps(ex["answer"]) for ex in dataset]

    MIN_SAMPLES = 5

    fig, ax = plt.subplots(figsize=(9, 5))

    all_x = set()

    model_data = {}
    for model in MODELS:
        path = f"results/{model}.json"
        print(f"Loading {path}...", flush=True)
        with open(path) as f:
            data = json.load(f)

        results = data["results"]
        assert len(results) == len(dataset)

        buckets = defaultdict(list)
        for entry in results:
            idx = entry["sample_idx"]
            num_gold = gold_steps[idx]
            buckets[num_gold].append(effective_metric(entry, model))

        model_data[model] = {k: v for k, v in buckets.items() if len(v) >= MIN_SAMPLES}
        all_x.update(model_data[model].keys())

    step_counts = sorted(all_x)

    for model, cfg in MODELS.items():
        buckets = model_data[model]
        xs, ys, los, his = [], [], [], []
        for s in step_counts:
            if s not in buckets:
                continue
            m, lo, hi = mean_ci(buckets[s])
            xs.append(s)
            ys.append(m)
            los.append(lo)
            his.append(hi)

        xs = np.array(xs)
        ax.plot(xs, ys, marker="o", color=cfg["color"], label=cfg["label"])
        # ax.fill_between(xs, los, his, alpha=0.2, color=cfg["color"])

    ax.set_xlabel("Number of steps in gold reasoning trace")
    ax.set_ylabel("Effective steps used\n(stable_match_frac × total steps)")
    ax.set_title("Effective steps used by gold reasoning trace length")
    ax.set_xticks(step_counts)
    ax.legend()
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)

    out = "effective_steps_by_gold_steps.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
