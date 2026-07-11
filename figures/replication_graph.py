import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from plot_common import RESULTS_DIR, FIGURES_DIR, bootstrap_mean_ci

CATEGORIES = ['ERM', 'Coconut', 'CODI']
MODEL_KEYS = ['sft', 'coconut', 'codi']

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
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 6))

    bars1 = ax.bar(x - width/2, original, width, label='original',
                   facecolor='white', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, replication, width, label='replication',
                   facecolor='white', edgecolor='black', linewidth=1.5,
                   hatch='///')

    # 95% CI on the replication bars only -- the "original" values are
    # single numbers reported by the paper, with no underlying data to
    # bootstrap a CI from.
    err_lo = [v - lo for v, lo in zip(replication, replication_lo)]
    err_hi = [hi - v for v, hi in zip(replication, replication_hi)]
    ax.errorbar(x + width/2, replication, yerr=[err_lo, err_hi],
                fmt='none', ecolor='black', elinewidth=1.5, capsize=5, zorder=3)

    # value labels on top of each bar
    for bar, val in zip(list(bars1), original):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f'{val}%', ha='center', va='bottom', fontsize=14)
    for bar, val, hi in zip(list(bars2), replication, replication_hi):
        ax.text(bar.get_x() + bar.get_width()/2, hi + 1.5,
                f'{val}%', ha='center', va='bottom', fontsize=14)

    ax.set_ylim(0, 110)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title(f"{METRIC_TITLE[args.metric]}\n(replication error bars: 95% CI)",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / f"{args.metric.replace('-', '_')}.png"
    plt.savefig(out, dpi=150)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
