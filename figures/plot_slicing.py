import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    MODELS, DISPLAY_NAME, RESULTS_DIR, FIGURES_DIR,
    FIGURE_WIDTH_IN, cluster_bootstrap_proportion_ci,
    set_style, style_axes, legend_below, save_figure,
)

set_style()


# Bottom-to-top stacking order (also the legend order, since ax.bar(...,
# label=cat) is called in this order): Original on the bottom, Other in
# the middle, Augmented on top.
CATEGORIES = ["Original", "Other", "Augmented"]
# Blue/terracotta (Original/Augmented, the two informative outcomes) sit at
# OKLCH L ~= 0.70, chroma 0.12 -- clears the dataviz skill's 0.10
# identity-hue floor (CVD Delta-E ~79, target is >=12). Now that the value
# labels sit beside the bars rather than on the fill, fill lightness no
# longer has to satisfy a black-text-contrast floor, so Other (the
# remaining uninformative outcome) is a lighter, near-neutral grey (L ~=
# 0.85) than the informative segments rather than matched to their
# lightness -- it reads as a quieter middle band between the two hues.
COLORS = ["#6aa1e8", "#d4d4d4", "#dd8363"]
STATUS_KEY = {"Original": "original", "Other": "other", "Augmented": "augmented"}


def slicing_statuses(r, effective_only):
    """The non-tie slicing_answer_status entries for one question's result
    `r`, optionally restricted to trials that spliced into an *effective*
    step.

    A trial at index i spliced the augmented step into position i
    (0-indexed), using i+1 total steps up to that point (see
    evaluate_coconut.py / evaluate_codi.py / evaluate_sft.py: the loop runs
    `for i in range(num_steps)` and forces an answer from i original steps
    + 1 augmented step at position i). "Effective" means that step
    position was one the *unperturbed* original trajectory actually
    needed -- i.e. i < stable_match, where stable_match (the same quantity
    as Figure 5's "effective steps used") is the smallest step count after
    which the original run's answer had already locked onto its final
    value. Steps at or after stable_match are redundant: the model had
    already decided, so swapping them tests something that couldn't have
    changed the outcome.
    """
    statuses = r["slicing_answer_status"]
    if effective_only:
        stable_match = r["original_result"]["stable_match"]
        statuses = statuses[:stable_match]
    return [s for s in statuses if s != "tie"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--effective-only", action="store_true",
        help=(
            "Restrict to slicing trials that spliced into an effective "
            "step (before the original run's answer had stabilized), "
            "dropping trials on steps that were already redundant. See "
            "slicing_statuses() docstring."
        ),
    )
    args = parser.parse_args()

    # Each question contributes several slicing steps, and steps from the
    # same question are correlated (not independent trials) -- so the CI is
    # a cluster bootstrap over questions rather than a plain per-step
    # Wilson interval.
    proportions = {m: {} for m in MODELS}
    ci_lo = {m: {} for m in MODELS}
    ci_hi = {m: {} for m in MODELS}
    for name in MODELS:
        with open(RESULTS_DIR / f"{name}.json") as f:
            data = json.load(f)
        groups = [
            slicing_statuses(r, args.effective_only)
            for r in data["results"] if "slicing_answer_status" in r
        ]
        # Drop clusters left empty (by excluding "tie", or -- with
        # --effective-only -- by a stable_match of 0) so they don't
        # contribute a zero-weight entry to the cluster bootstrap.
        groups = [g for g in groups if g]
        ci = cluster_bootstrap_proportion_ci(groups, list(STATUS_KEY.values()))
        for cat, key in STATUS_KEY.items():
            point, lo, hi = ci[key]
            proportions[name][cat] = point
            ci_lo[name][cat] = lo
            ci_hi[name][cat] = hi

    x = np.arange(len(MODELS))
    bar_width = 0.42

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 1.05))

    bottoms = np.zeros(len(MODELS))
    for cat, color in zip(CATEGORIES, COLORS):
        vals = np.array([proportions[m][cat] for m in MODELS])
        los = np.array([ci_lo[m][cat] for m in MODELS])
        his = np.array([ci_hi[m][cat] for m in MODELS])
        ax.bar(x, vals, bar_width, bottom=bottoms, color=color, label=cat)

        # 95% CI (cluster bootstrap), drawn at each segment's own boundary
        # with that category's own margin -- not the cumulative stack
        # position.
        tops = bottoms + vals
        ax.errorbar(x, tops, yerr=[vals - los, his - vals],
                    fmt='none', ecolor='black', elinewidth=1.2, capsize=4, zorder=3)

        # Label beside each segment rather than inside it -- inside labels
        # for small segments (e.g. Augmented at 5-6% for COCONUT/CODI) had
        # no room to sit cleanly within the fill. Placed in the gap to the
        # bar's right (narrowed bar_width above makes room), at the
        # segment's vertical midpoint; ha='left' keeps them clear of the
        # bar itself.
        for i, (v, bot) in enumerate(zip(vals, bottoms)):
            ax.text(
                x[i] + bar_width / 2 + 0.03, bot + v / 2,
                f'{v:.0%}',
                ha='left', va='center',
                color='black',
            )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAME[m] for m in MODELS], rotation=0, ha='center')
    ax.set_ylabel('Proportion')
    ax.set_ylim(0, 1.05)
    # Extra right margin for the last bar's (ERM's) side labels, which have
    # no following bar to make room for them the way COCONUT/CODI's do.
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    # y=-0.12 renders the same *visual* legend-to-axis gap as Figures 3 and
    # 5 (plot_by_gold_steps.py / plot_effective_steps.py, both y=-0.2) --
    # the axes-relative y fraction those use doesn't transfer directly to
    # this figure's taller aspect ratio (FIGURE_WIDTH_IN * 1.05 vs 0.96),
    # so this was tuned by measuring the actual rendered gap rather than
    # copying the fraction. A 3-column legend row that's wider than the
    # axes would force the tight-bbox save to widen the canvas beyond
    # FIGURE_WIDTH_IN, which silently shrinks the effective font size once
    # LaTeX scales the PNG back down to column width (see
    # plot_common.FIGURE_WIDTH_IN) -- tightened spacing keeps this row
    # within the axes width instead.
    legend_below(ax, ncol=3, y=-0.12, handlelength=1.2, handletextpad=0.5, columnspacing=1.0)
    style_axes(ax)

    out_name = "slicing_status_effective.png" if args.effective_only else "slicing_status.png"
    save_figure(fig, FIGURES_DIR / out_name)


if __name__ == "__main__":
    main()
