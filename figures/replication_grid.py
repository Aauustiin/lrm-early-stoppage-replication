"""Draft replacement for Figure 2 (first_match.png / stable_match.png):
a 2x3 grid of the same original-vs-replication bar chart, one panel per
(metric, dataset) pair -- rows are First Match / Stable Match, columns are
GSM8K / ProsQA / PrOntoQA. Each panel is styled identically to
replication_graph.py's single-dataset figure; this script just tiles six of
them and shares one legend. Not yet wired into acl_latex.tex -- run
standalone (`python replication_grid.py`) to inspect replication_grid.png.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    RESULTS_DIR, FIGURES_DIR, DISPLAY_NAME,
    bootstrap_mean_ci, style_axes, save_figure, _ACL_BODY_FONT_PT,
)

# Unlike plot_common's other figures (placed at \columnwidth or
# 0.48\linewidth, both ~7.7cm -- see plot_common.FIGURE_WIDTH_IN's
# docstring), this one is placed at width=\textwidth in acl_latex.tex, so
# its font has to be calibrated against the full text width instead of
# plot_common.font_size_for_width's \columnwidth assumption -- otherwise it
# renders smaller than every other figure's ~8pt physical text once placed.
# a4paper (21cm) with acl.sty's `margin=2.5cm` geometry call gives
# \textwidth = 21 - 2*2.5 = 16cm.
_ACL_TEXTWIDTH_IN = 16.0 / 2.54

# Authoring canvas width: only sets how much room the 6 subplots get, not
# the final physical text size (that's fixed by the ratio below) -- chosen
# generously so per-panel bar/label spacing looks like the single-panel
# figures' own canvas.
GRID_WIDTH_IN = 14.0
GRID_HEIGHT_IN = GRID_WIDTH_IN * 0.62

plt.rcParams.update({
    "font.size": round(_ACL_BODY_FONT_PT * GRID_WIDTH_IN / _ACL_TEXTWIDTH_IN),
    "font.family": "serif",
})

MODEL_KEYS = ['sft', 'coconut', 'codi']
CATEGORIES = [DISPLAY_NAME[m] for m in MODEL_KEYS]

DATASET_KEYS = ['gsm8k', 'prosqa', 'prontoqa']
DATASET_DISPLAY = {'gsm8k': 'GSM8K', 'prosqa': 'ProsQA', 'prontoqa': 'PrOntoQA'}

METRICS = ['first-match', 'stable-match']
METRIC_TITLE = {'first-match': 'First match fraction', 'stable-match': 'Stable match fraction'}
PER_SAMPLE_FIELD = {'first-match': 'first_match_frac', 'stable-match': 'stable_match_frac'}

# Numbers reported by the original paper being replicated (Figure 3 of
# Dilgren & Wiegreffe 2026), read off that figure's bar labels -- there's no
# per-sample data behind a single reported number, so no CI is possible for
# them, same as replication_graph.py's ORIGINAL. sft/coconut/codi order, in
# percent. Note CODI (not Coconut) carries the small nonzero ProsQA bars --
# Coconut is exactly 0% on both ProsQA and PrOntoQA, both metrics.
ORIGINAL = {
    'gsm8k': {
        'first-match': [98, 54, 44],
        'stable-match': [99, 69, 54],
    },
    'prosqa': {
        'first-match': [92, 0, 2],
        'stable-match': [98, 0, 4],
    },
    'prontoqa': {
        'first-match': [46, 0, 0],
        'stable-match': [47, 0, 0],
    },
}


# coconut.json forces COCONUT to answer immediately after its latent
# reasoning tokens; this figure uses the no-forced-answer variant instead
# (see evaluate_coconut.py's --no-force-answer), consistent with
# gold_steps_grid.py, so its curve reflects the model's own answer timing
# like CODI's and the ERM's. No no-force-answer variant exists for
# ProsQA/PrOntoQA, so only the gsm8k entry is overridden.
RESULTS_FILENAME_OVERRIDE = {('coconut', 'gsm8k'): 'coconut_no_force_answer.json'}


def results_path(model: str, dataset: str):
    override = RESULTS_FILENAME_OVERRIDE.get((model, dataset))
    if override:
        return RESULTS_DIR / override
    suffix = '' if dataset == 'gsm8k' else f'_{dataset}'
    return RESULTS_DIR / f'{model}{suffix}.json'


def load_replication(dataset: str, metric: str):
    """Return (values, los, his), each a list over MODEL_KEYS, in percent.
    `values` are the point estimates; (los, his) is a 95% bootstrap CI over
    the per-sample fractions backing each average.
    """
    field = PER_SAMPLE_FIELD[metric]
    values, los, his = [], [], []
    for model in MODEL_KEYS:
        with open(results_path(model, dataset)) as f:
            data = json.load(f)
        fracs = [r['original_result'][field] for r in data['results']]
        m, lo, hi = bootstrap_mean_ci(fracs)
        values.append(round(m * 100, 1))
        los.append(lo * 100)
        his.append(hi * 100)
    return values, los, his


def plot_panel(ax, dataset: str, metric: str):
    original = ORIGINAL[dataset][metric]
    replication, replication_lo, replication_hi = load_replication(dataset, metric)

    x = np.arange(len(CATEGORIES))
    width = 0.32
    offset = 0.19  # > width/2: leaves a gap so wide value labels don't touch

    bars1 = ax.bar(x - offset, original, width, label='original',
                   facecolor='white', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + offset, replication, width, label='replication',
                   facecolor='white', edgecolor='black', linewidth=1.2,
                   hatch='///')

    # max(0, ...): rounding `replication` to 1 decimal (for the on-bar label)
    # while leaving the CI bound unrounded can put e.g. hi a hair below the
    # rounded point estimate at the 0%/100% boundary, which would otherwise
    # go negative.
    err_lo = [max(0.0, v - lo) for v, lo in zip(replication, replication_lo)]
    err_hi = [max(0.0, hi - v) for v, hi in zip(replication, replication_hi)]
    ax.errorbar(x + offset, replication, yerr=[err_lo, err_hi],
                fmt='none', ecolor='black', elinewidth=1.2, capsize=4, zorder=3)

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
    style_axes(ax)

    return bars1, bars2


def main():
    fig, axes = plt.subplots(
        2, 3, figsize=(GRID_WIDTH_IN, GRID_HEIGHT_IN),
        sharey=True,
    )

    legend_handles = None
    for row, metric in enumerate(METRICS):
        for col, dataset in enumerate(DATASET_KEYS):
            ax = axes[row, col]
            bars1, bars2 = plot_panel(ax, dataset, metric)
            legend_handles = (bars1, bars2)
            if row == 0:
                ax.set_title(DATASET_DISPLAY[dataset], pad=4)
            if col == 0:
                ax.set_ylabel(METRIC_TITLE[metric])

    fig.legend(handles=list(legend_handles), labels=['original', 'replication'],
               loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.subplots_adjust(wspace=0.08, hspace=0.28)

    # width_in matches the authoring canvas (GRID_WIDTH_IN), not a LaTeX
    # placement width -- see save_figure's docstring; the font size above is
    # what actually calibrates physical text size for \textwidth placement.
    save_figure(fig, FIGURES_DIR / "replication_grid.png", width_in=GRID_WIDTH_IN)


if __name__ == '__main__':
    main()
