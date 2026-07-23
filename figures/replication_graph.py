import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    RESULTS_DIR, FIGURES_DIR, DISPLAY_NAME, FIGURE_WIDTH_IN,
    bootstrap_mean_ci, set_style, style_axes, legend_below, save_figure,
)

set_style()

MODEL_KEYS = ['sft', 'coconut', 'codi']
CATEGORIES = [DISPLAY_NAME[m] for m in MODEL_KEYS]

# Numbers reported by the original paper being replicated. These aren't ours
# to compute -- there's no per-sample data behind a single reported number,
# so no CI is possible for them -- they stay hardcoded point estimates.
ORIGINAL = {
    'first-match': [98, 54, 44],
    'stable-match': [99, 69, 54],
}

# Per-sample field backing each metric's average, used to bootstrap a 95% CI
# for our own replication numbers.
PER_SAMPLE_FIELD = {
    'first-match': 'first_match_frac',
    'stable-match': 'stable_match_frac',
}

METRIC_TITLE = {
    'first-match': 'First Match',
    'stable-match': 'Stable Match',
}


def load_replication(metric: str):
    """Return (values, los, his), each a list over MODEL_KEYS, in percent.
    `values` are the point estimates; (los, his) is a 95% bootstrap CI over
    the per-sample fractions backing each average.
    """
    field = PER_SAMPLE_FIELD[metric]
    values, los, his = [], [], []
    for model in MODEL_KEYS:
        with open(RESULTS_DIR / f'{model}.json') as f:
            data = json.load(f)
        fracs = [r['original_result'][field] for r in data['results']]
        m, lo, hi = bootstrap_mean_ci(fracs)
        values.append(round(m * 100, 1))
        los.append(lo * 100)
        his.append(hi * 100)
    return values, los, his


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metric', nargs='?', default='first-match',
                         choices=['first-match', 'stable-match'],
                         help="Which metric to plot (default: first-match)")
    args = parser.parse_args()

    original = ORIGINAL[args.metric]
    replication, replication_lo, replication_hi = load_replication(args.metric)

    x = np.arange(len(CATEGORIES))
    width = 0.32
    offset = 0.19  # > width/2: leaves a gap so wide value labels don't touch

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 0.95))

    bars1 = ax.bar(x - offset, original, width, label='original',
                   facecolor='white', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + offset, replication, width, label='replication',
                   facecolor='white', edgecolor='black', linewidth=1.2,
                   hatch='///')

    # 95% CI on the replication bars only -- the "original" values are
    # single numbers reported by the paper, with no underlying data to
    # bootstrap a CI from.
    err_lo = [v - lo for v, lo in zip(replication, replication_lo)]
    err_hi = [hi - v for v, hi in zip(replication, replication_hi)]
    ax.errorbar(x + offset, replication, yerr=[err_lo, err_hi],
                fmt='none', ecolor='black', elinewidth=1.2, capsize=4, zorder=3)

    # value labels on top of each bar, rotated so they don't need to be
    # wider than the (narrow, body-text-sized-label) bars they sit above
    for bar, val in zip(list(bars1), original):
        ax.text(bar.get_x() + bar.get_width()/2, val + 4,
                f'{val}%', ha='center', va='bottom', rotation=90)
    for bar, val, hi in zip(list(bars2), replication, replication_hi):
        ax.text(bar.get_x() + bar.get_width()/2, hi + 4,
                f'{val}%', ha='center', va='bottom', rotation=90)

    ax.set_ylim(0, 138)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_title(METRIC_TITLE[args.metric], pad=4)
    legend_below(ax, ncol=2, y=-0.1)
    style_axes(ax)

    save_figure(fig, FIGURES_DIR / f"{args.metric.replace('-', '_')}.png")


if __name__ == '__main__':
    main()
