"""1x3 grid of accuracy / stable-match vs. gold-reasoning-steps panels
(COCONUT, CODI, ERM) sharing one legend and one set of axis labels --
replaces the three separately-placed *_by_gold_steps.png images previously
tiled via LaTeX \\hfill in acl_latex.tex's fig:steps figure.
"""

import argparse

import matplotlib.pyplot as plt

from plot_common import (
    FIGURES_DIR, DISPLAY_NAME, BLUE, ORANGE, _ACL_BODY_FONT_PT,
    load_gold_steps, load_results, gold_steps_accuracy_stable_match,
    style_axes, save_figure,
)

# Placed at width=\textwidth inside a figure* in acl_latex.tex (replacing the
# three 0.32\linewidth-each panels) -- same \textwidth calibration as
# replication_grid.py. a4paper (21cm) with acl.sty's margin=2.5cm geometry
# gives \textwidth = 16cm.
_ACL_TEXTWIDTH_IN = 16.0 / 2.54
GRID_WIDTH_IN = 14.0
GRID_HEIGHT_IN = 6.2

plt.rcParams.update({
    "font.size": round(_ACL_BODY_FONT_PT * GRID_WIDTH_IN / _ACL_TEXTWIDTH_IN),
    "font.family": "serif",
})

MODELS = ["coconut", "codi", "sft"]

# coconut.json forces COCONUT to answer immediately after its latent
# reasoning tokens; this figure uses the no-forced-answer variant instead
# (see evaluate_coconut.py's --no-force-answer) so its curve, like CODI's
# and the ERM's, reflects the model's own answer timing. No such variant
# was run for ProsQA/PrOntoQA, so the override only applies to gsm8k.
RESULTS_FILENAME = {"gsm8k": {"coconut": "coconut.json"}}


def plot_panel(ax, model: str, dataset: str, min_samples=5, n_boot=10000, seed=0):
    gold_steps = load_gold_steps(dataset)
    data = load_results(model, dataset=dataset, expected_len=len(gold_steps),
                         filename=RESULTS_FILENAME.get(dataset, {}).get(model))
    x, acc_means, acc_lo, acc_hi, smf_means, smf_lo, smf_hi = \
        gold_steps_accuracy_stable_match(data["results"], gold_steps,
                                          min_samples=min_samples,
                                          n_boot=n_boot, seed=seed)

    line_acc, = ax.plot(x, acc_means, marker="o", markersize=5, linewidth=1.8,
                         linestyle="-", color=BLUE, label="Accuracy")
    ax.fill_between(x, acc_lo, acc_hi, alpha=0.15, color=BLUE, linewidth=0)

    line_smf, = ax.plot(x, smf_means, marker="s", markersize=5, linewidth=1.8,
                         linestyle="--", color=ORANGE, label="Stable match")
    ax.fill_between(x, smf_lo, smf_hi, alpha=0.15, color=ORANGE, linewidth=0)

    ax.set_title(DISPLAY_NAME[model])
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    style_axes(ax)

    return line_acc, line_smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "prosqa", "prontoqa"],
                        default="gsm8k", help="Which dataset's results to plot (default: gsm8k)")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(GRID_WIDTH_IN, GRID_HEIGHT_IN), sharey=True)

    handles = None
    for ax, model in zip(axes, MODELS):
        handles = plot_panel(ax, model, args.dataset)

    axes[0].set_ylabel("Fraction")
    fig.supxlabel("Gold reasoning steps", y=0.17)

    fig.legend(handles=list(handles), labels=["Accuracy", "Stable match"],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.subplots_adjust(wspace=0.08, bottom=0.34)

    suffix = "" if args.dataset == "gsm8k" else f"_{args.dataset}"
    save_figure(fig, FIGURES_DIR / f"gold_steps_grid{suffix}.png", width_in=GRID_WIDTH_IN)


if __name__ == "__main__":
    main()
