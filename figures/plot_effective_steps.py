"""
Bins GSM8k test samples by number of steps in the gold reasoning trace and
plots, for each model, the average raw stable_match step count in that
bucket (the smallest k such that the model's answer at step k onward always
matches its final answer -- not divided by the model's step budget).
All three lines on one axes, with 95% CI bands.
"""

import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_common import (
    MODELS, DISPLAY_NAME, MODEL_COLOR, MODEL_LINESTYLE, MODEL_MARKER,
    FIGURES_DIR, FIGURE_WIDTH_IN,
    load_gold_steps, load_results, bootstrap_mean_ci,
    set_style, style_axes, legend_below, save_figure,
)

set_style()


def effective_metric(entry, model: str) -> float:
    return entry["original_result"]["stable_match"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "prosqa", "prontoqa"],
                        default="gsm8k", help="Which dataset's results to plot (default: gsm8k)")
    args = parser.parse_args()

    gold_steps = load_gold_steps(args.dataset)

    MIN_SAMPLES = 5

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 0.96))

    all_x = set()

    model_data = {}
    for model in MODELS:
        data = load_results(model, dataset=args.dataset, expected_len=len(gold_steps))
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
            # Raw stable_match isn't a [0, 1] fraction (it ranges up to
            # ~6-8 depending on the model), so don't clamp the BCa interval.
            m, lo, hi = bootstrap_mean_ci(buckets[s], bounds=None)
            xs.append(s)
            ys.append(m)
            los.append(lo)
            his.append(hi)

        xs = np.array(xs)
        ax.plot(xs, ys, marker=MODEL_MARKER[model], linestyle=MODEL_LINESTYLE[model],
                markersize=5, linewidth=1.8, color=MODEL_COLOR[model], label=DISPLAY_NAME[model])
        ax.fill_between(xs, los, his, alpha=0.2, color=MODEL_COLOR[model], linewidth=0)

    ax.set_xlabel("Gold reasoning steps")
    ax.set_ylabel("Stable Match")
    ax.set_xticks(step_counts)
    legend_below(ax, ncol=3, y=-0.2,
                 handlelength=1.3, handletextpad=0.4, columnspacing=1.0)
    style_axes(ax)

    suffix = "" if args.dataset == "gsm8k" else f"_{args.dataset}"
    save_figure(fig, FIGURES_DIR / f"effective_steps_by_gold_steps{suffix}.png")


if __name__ == "__main__":
    main()
