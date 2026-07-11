import json

import matplotlib.pyplot as plt
import numpy as np

from plot_common import MODELS, DISPLAY_NAME, PALETTE, RESULTS_DIR, FIGURES_DIR, cluster_bootstrap_proportion_ci

CATEGORIES = ["Original", "Augmented", "Other", "Tie"]
COLORS = PALETTE[:4]
STATUS_KEY = {"Original": "original", "Augmented": "augmented", "Other": "other", "Tie": "tie"}

# Each question contributes several slicing steps, and steps from the same
# question are correlated (not independent trials) -- so the CI is a cluster
# bootstrap over questions rather than a plain per-step Wilson interval.
proportions = {m: {} for m in MODELS}
ci_lo = {m: {} for m in MODELS}
ci_hi = {m: {} for m in MODELS}
for name in MODELS:
    with open(RESULTS_DIR / f"{name}.json") as f:
        data = json.load(f)
    groups = [r["slicing_answer_status"] for r in data["results"] if "slicing_answer_status" in r]
    ci = cluster_bootstrap_proportion_ci(groups, list(STATUS_KEY.values()))
    for cat, key in STATUS_KEY.items():
        point, lo, hi = ci[key]
        proportions[name][cat] = point
        ci_lo[name][cat] = lo
        ci_hi[name][cat] = hi

x = np.arange(len(MODELS))
bar_width = 0.5

fig, ax = plt.subplots(figsize=(7, 5))

bottoms = np.zeros(len(MODELS))
for cat, color in zip(CATEGORIES, COLORS):
    vals = np.array([proportions[m][cat] for m in MODELS])
    los = np.array([ci_lo[m][cat] for m in MODELS])
    his = np.array([ci_hi[m][cat] for m in MODELS])
    ax.bar(x, vals, bar_width, bottom=bottoms, color=color, label=cat)

    # 95% CI (cluster bootstrap), drawn at each segment's own boundary with
    # that category's own margin -- not the cumulative stack position.
    tops = bottoms + vals
    ax.errorbar(x, tops, yerr=[vals - los, his - vals],
                fmt='none', ecolor='black', elinewidth=1.2, capsize=4, zorder=3)

    # Annotate segments that are large enough to label
    for i, (v, bot) in enumerate(zip(vals, bottoms)):
        if v >= 0.04:
            ax.text(
                x[i], bot + v / 2,
                f'{v:.1%}',
                ha='center', va='center',
                fontsize=14, color='white'
            )
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels([DISPLAY_NAME[m] for m in MODELS], fontsize=14)
ax.set_ylabel('Proportion', fontsize=14)
ax.set_title('Token Patching Answer Status by Model\n(error bars: 95% CI)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.tick_params(axis='both', labelsize=14)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=len(CATEGORIES),
          framealpha=0.9, fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = FIGURES_DIR / 'slicing_status.png'
plt.savefig(out, dpi=150)
print(f'Saved to {out}')
