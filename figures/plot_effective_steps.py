"""
Bins GSM8k test samples by number of steps in the gold reasoning trace and
plots, for each model:
  - CODI / COCONUT: mean(stable_match_frac * 6)   [effective latent tokens used]
  - SFT:            mean(num_steps * stable_match_frac)  [effective text steps used]
All three lines on one axes, with 95% CI bands.
"""

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_common import (
    MODELS, DISPLAY_NAME, MODEL_COLOR, FIGURES_DIR,
    load_gold_steps, load_results, bootstrap_mean_ci,
)

MARKERS = {"coconut": "o", "codi": "s", "sft": "^"}


def effective_metric(entry, model: str) -> float:
    r = entry["original_result"]
    if model == "sft":
        return r["num_steps"] * r["stable_match_frac"]
    else:  # coconut / codi — 6 latent steps per implicit reasoning step
        return r["stable_match_frac"] * 6


def main():
    gold_steps = load_gold_steps()

    MIN_SAMPLES = 5

    fig, ax = plt.subplots(figsize=(9, 5))

    all_x = set()

    model_data = {}
    for model in MODELS:
        data = load_results(model, expected_len=len(gold_steps))
        results = data["results"]

        buckets = defaultdict(list)
        for entry in results:
            idx = entry["sample_idx"]
            num_gold = gold_steps[idx]
            buckets[num_gold].append(effective_metric(entry, model))

        model_data[model] = {k: v for k, v in buckets.items() if len(v) >= MIN_SAMPLES}
        all_x.update(model_data[model].keys())

    step_counts = sorted(all_x)

    for model in MODELS:
        buckets = model_data[model]
        xs, ys, los, his = [], [], [], []
        for s in step_counts:
            if s not in buckets:
                continue
            # Effective steps used isn't a [0, 1] fraction (it ranges up to
            # ~6-8 depending on the model), so don't clamp the BCa interval.
            m, lo, hi = bootstrap_mean_ci(buckets[s], bounds=None)
            xs.append(s)
            ys.append(m)
            los.append(lo)
            his.append(hi)

        xs = np.array(xs)
        ax.plot(xs, ys, marker=MARKERS[model], color=MODEL_COLOR[model], label=DISPLAY_NAME[model])
        ax.fill_between(xs, los, his, alpha=0.2, color=MODEL_COLOR[model], linewidth=0)

    ax.set_xlabel("Number of steps in gold reasoning trace")
    ax.set_ylabel("Effective steps used\n(stable_match_frac × total steps)")
    ax.set_title("Effective steps used by gold reasoning trace length\n(shaded band: 95% CI)")
    ax.set_xticks(step_counts)
    ax.legend()
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)

    out = FIGURES_DIR / "effective_steps_by_gold_steps.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
