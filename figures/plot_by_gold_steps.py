import argparse

import matplotlib.pyplot as plt

from plot_common import (
    FIGURES_DIR, DISPLAY_NAME, BLUE, ORANGE, FIGURE_WIDTH_IN,
    load_gold_steps, load_results, gold_steps_accuracy_stable_match,
    style_axes, legend_below, save_figure, _ACL_BODY_FONT_PT,
)

# This figure is placed 3-up at width=0.32\linewidth inside a figure* (a
# two-column \textwidth spread) in acl_latex.tex, not at plot_common's
# set_style()-default \columnwidth/0.48-pair target (~7.7cm) -- same
# \textwidth calibration as replication_grid.py. a4paper (21cm) with
# acl.sty's margin=2.5cm geometry gives \textwidth = 16cm.
_ACL_TEXTWIDTH_IN = 16.0 / 2.54
_PANEL_LINEWIDTH_FRAC = 0.32

plt.rcParams.update({
    "font.size": round(_ACL_BODY_FONT_PT * FIGURE_WIDTH_IN
                        / (_PANEL_LINEWIDTH_FRAC * _ACL_TEXTWIDTH_IN)),
    "font.family": "serif",
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["sft", "coconut", "codi"],
                        help="Which model's results to plot")
    parser.add_argument("--dataset", choices=["gsm8k", "prosqa", "prontoqa"],
                        default="gsm8k", help="Which dataset's results to plot (default: gsm8k)")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="Minimum samples per bin to include (default: 5)")
    parser.add_argument("--n-boot", type=int, default=10000,
                        help="Bootstrap resamples for the stable-match CI")
    parser.add_argument("--seed", type=int, default=0,
                        help="Bootstrap seed, so reruns give identical bands")
    args = parser.parse_args()

    gold_steps = load_gold_steps(args.dataset)
    data = load_results(args.model, dataset=args.dataset, expected_len=len(gold_steps))
    x, acc_means, acc_lo, acc_hi, smf_means, smf_lo, smf_hi = gold_steps_accuracy_stable_match(
        data["results"], gold_steps, min_samples=args.min_samples,
        n_boot=args.n_boot, seed=args.seed)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 0.95))

    ax.plot(x, acc_means, marker="o", markersize=5, linewidth=1.8, linestyle="-",
            color=BLUE, label="Accuracy")
    ax.fill_between(x, acc_lo, acc_hi, alpha=0.15, color=BLUE, linewidth=0)

    ax.plot(x, smf_means, marker="s", markersize=5, linewidth=1.8, linestyle="--",
            color=ORANGE, label="Stable match")
    ax.fill_between(x, smf_lo, smf_hi, alpha=0.15, color=ORANGE, linewidth=0)

    ax.set_xlabel("Gold reasoning steps")
    ax.set_ylabel("Fraction")
    title = DISPLAY_NAME[args.model]
    if args.dataset != "gsm8k":
        title += f" ({args.dataset})"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    legend_below(ax, ncol=2, y=-0.2)
    style_axes(ax)

    suffix = "" if args.dataset == "gsm8k" else f"_{args.dataset}"
    save_figure(fig, FIGURES_DIR / f"{args.model}_by_gold_steps{suffix}.png")


if __name__ == "__main__":
    main()
