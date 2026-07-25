"""Shared helpers for the plotting scripts (plot_slicing.py,
plot_effective_steps.py, plot_by_gold_steps.py).
"""

import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image as PILImage

FIGURES_DIR = Path(__file__).resolve().parent
RESULTS_DIR = FIGURES_DIR.parent / "results"

MODELS = ["coconut", "codi", "sft"]
DISPLAY_NAME = {"sft": "ERM", "coconut": "COCONUT", "codi": "CODI"}

# Colorblind-safe categorical palette (fixed hue order), validated with the
# dataviz skill's CVD/contrast checker: blue, aqua, yellow, green, violet,
# red, magenta, orange. Slices of this list are reused across every figure
# instead of each script picking its own colors.
PALETTE = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]

# One color per model, in MODELS order (blue/aqua/yellow).
MODEL_COLOR = dict(zip(MODELS, PALETTE))

# The blue/orange pair used for 2-series metric comparisons (e.g. accuracy
# vs. stable-match fraction), validated as a high-contrast pair.
BLUE, ORANGE = PALETTE[0], PALETTE[7]

# Distinct line style / marker / bar hatch per fixed-order slot, paired with
# PALETTE so identity survives grayscale printing too, not just color --
# ACL formatting guidance asks figures not to rely on color alone. Applied
# in the same fixed order as PALETTE/MODEL_COLOR.
LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D"]
HATCHES = ["", "///", "xx", ".."]

MODEL_LINESTYLE = dict(zip(MODELS, LINESTYLES))
MODEL_MARKER = dict(zip(MODELS, MARKERS))

# --- Shared figure sizing --------------------------------------------------
# Every figure in the paper is placed at (or very close to) one ACL column
# width -- 7.7cm, per paper/formatting.md -- whether it's a single-column
# figure (`width=\columnwidth`) or one half of a two-up `figure*` pair
# (`width=0.48\linewidth` of a ~16.1cm two-column spread, ~7.7cm too). Every
# figure is authored at FIGURE_WIDTH_IN with a matching font size so that
# once LaTeX scales the PNG down to that width, in-figure text renders at
# a fixed physical size, _ACL_BODY_FONT_PT (deliberately smaller than the
# paper's 11pt body text -- see formatting.md -- so in-figure text reads as
# a caption/label register, not body copy). A script can author at a wider
# native canvas (e.g. to give crowded x-tick labels more room) as long as
# it scales the font size to match -- that's what
# font_size_for_width()/set_style(width_in=...) are for.
#
# This invariant only holds if the saved PNG's actual (tight-bbox) width
# matches the width_in used to compute its font size -- e.g. a legend or
# rotated tick labels that overflow the nominal canvas will widen the
# saved image beyond width_in, which silently shrinks the effective text
# size once LaTeX scales it back down to column width. Check a script's
# actual output width (px / FIGURE_DPI) against its width_in if its text
# looks off relative to the other figures.
FIGURE_WIDTH_IN = 6.0
_ACL_COLUMN_WIDTH_IN = 7.7 / 2.54
_ACL_BODY_FONT_PT = 8
FIGURE_DPI = 200


def font_size_for_width(width_in=FIGURE_WIDTH_IN):
    return round(_ACL_BODY_FONT_PT * width_in / _ACL_COLUMN_WIDTH_IN)


FONT_SIZE = font_size_for_width()

MUTED = "#898781"
GRID = "#e1e0d9"


def set_style(width_in=FIGURE_WIDTH_IN):
    """Apply the shared figure design system (call once, before plotting)."""
    plt.rcParams.update({
        "font.size": font_size_for_width(width_in),
        "font.family": "serif",  # match the paper's Times body text
    })


def style_axes(ax):
    """Shared axis chrome: no top/right spine, muted gridlines behind data."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.6, color=GRID, zorder=0)
    ax.set_axisbelow(True)


def legend_below(ax, ncol, y=-0.32, **kwargs):
    """Frameless legend in a single row below the axes -- at body-text-sized
    fonts an in-plot legend collides with data far too easily, so every
    figure in this project puts its legend in the same place instead.

    Extra **kwargs (e.g. handlelength, handletextpad, columnspacing) pass
    through to ax.legend() -- useful for a many-column legend whose default
    spacing would render wider than the axes, forcing bbox_inches="tight"
    to pad the saved canvas and making the actual plot look shrunken.
    """
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
                      frameon=False, **kwargs)


def _tight_width_in(fig):
    """Actual PNG width (inches) after the bbox_inches="tight" trim -- may
    differ from the canvas width_in a script authored at, since trimming
    isn't proportional to font size (see save_figure)."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.12)
    buf.seek(0)
    with PILImage.open(buf) as im:
        return im.size[0] / FIGURE_DPI


def save_figure(fig, out, width_in=FIGURE_WIDTH_IN):
    """Save with a tight-bbox trim (removes excess whitespace, and rescues
    content -- e.g. a rotated axis label -- that would otherwise clip) and
    report the path. Relies on bbox_inches alone for layout, not
    tight_layout(), since the two can fight each other's margins.

    `width_in` is the canvas width the script authored at (and passed to
    set_style()/font_size_for_width() to pick its font size) -- normally
    FIGURE_WIDTH_IN. The bbox_inches="tight" trim doesn't remove whitespace
    proportionally to font size, so the actual saved width can drift from
    width_in (a wide legend or dense tick labels trim less; a sparse plot
    trims more). Left uncorrected, that drift silently mis-scales the
    in-figure text once LaTeX scales the PNG back down to column width: a
    saved image *wider* than width_in gets shrunk *more* than the font size
    assumed, so its text renders smaller than intended (and vice versa for
    a narrower saved image) -- so two figures at the same nominal font size
    can render at different physical sizes in the paper. This measures the
    actual width and, if it's off from width_in by more than 1%, scales
    every text artist's font size by (measured / width_in) -- bigger for an
    over-wide save, smaller for an under-wide one -- before the final save.
    """
    measured = _tight_width_in(fig)
    scale = measured / width_in
    if abs(scale - 1) > 0.01:
        for t in fig.findobj(matplotlib.text.Text):
            t.set_fontsize(t.get_fontsize() * scale)
        measured = _tight_width_in(fig)
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.12)
    print(f"Saved {out} ({measured:.3f}in actual vs {width_in:.3f}in nominal)")


_NORM = NormalDist()


def count_gold_steps(answer: str) -> int:
    """Count reasoning steps in a GSM8k gold answer (lines before ####)."""
    lines = [l.strip() for l in answer.strip().split("\n") if l.strip()]
    return sum(1 for l in lines if not l.startswith("####"))


# ProsQA/PrOntoQA aren't mirrored on the Hugging Face Hub, so their test
# splits are fetched directly from the replicated paper's own repo, pinned to
# a fixed commit. Duplicated from eval_common.py's _LRM_PAPER_REPO_* rather
# than imported -- this project's figures/ scripts are meant to run
# standalone from any working directory (see README), and every other
# gold-step source here (including GSM8K's, below) is likewise fetched
# directly rather than reusing eval_common.py's copy.
_LRM_PAPER_REPO_COMMIT = "32f413d8d55239d9bc54bb6b6ec37b0630891ed4"
_LRM_PAPER_REPO_RAW = (
    "https://raw.githubusercontent.com/connordilgren/are-lrms-easily-interpretable"
    f"/{_LRM_PAPER_REPO_COMMIT}"
)


def load_gold_steps(dataset: str = "gsm8k"):
    """Return the per-sample gold reasoning step counts for `dataset`'s test
    split, in the same sample order as evaluate_*.py's `sample_idx`.

    GSM8K: derived from the gold answer's calculator-notation lines (see
    count_gold_steps). ProsQA/PrOntoQA: each test sample's gold reasoning
    chain is already segmented into a "steps" list by the source data, so
    the gold step count is just its length -- no parsing needed.
    """
    if dataset == "gsm8k":
        print("Loading GSM8k test split...", flush=True)
        ds = load_dataset("gsm8k", "main", split="test")
        return [count_gold_steps(ex["answer"]) for ex in ds]
    elif dataset in ("prosqa", "prontoqa"):
        print(f"Loading {dataset} test split...", flush=True)
        url = f"{_LRM_PAPER_REPO_RAW}/data/{dataset}_test.json"
        with urllib.request.urlopen(url) as response:
            samples = json.load(response)
        return [len(s["steps"]) for s in samples]
    else:
        raise ValueError(f"Unknown dataset {dataset!r}")


def load_results(model: str, dataset: str = "gsm8k", expected_len: int | None = None,
                  filename: str | None = None):
    """Load a results/*.json file for `model`/`dataset`. If `expected_len` is
    given (typically len(gold_steps) from load_gold_steps()), assert the
    results cover every test sample.

    Filename matches evaluate_sft.py/evaluate_coconut.py/evaluate_codi.py's
    output convention: results/{model}.json for gsm8k (the default dataset),
    results/{model}_{dataset}.json otherwise. Pass `filename` to override
    this, e.g. to pick a same-model variant like coconut_no_force_answer.json.
    """
    if filename is None:
        filename = f"{model}.json" if dataset == "gsm8k" else f"{model}_{dataset}.json"
    path = RESULTS_DIR / filename
    print(f"Loading {path}...", flush=True)
    with open(path) as f:
        data = json.load(f)

    if expected_len is not None:
        assert len(data["results"]) == expected_len, (
            f"Result count {len(data['results'])} != dataset size {expected_len}"
        )
    return data


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center, center - half, center + half


def bootstrap_mean_ci(values, alpha=0.05, n_boot=10000, seed=0, bounds=(0.0, 1.0)):
    """BCa bootstrap CI for the mean of `values`.

    Returns (mean, lo, hi). The interval is non-parametric and does not
    assume symmetry -- which matters e.g. for stable_match_frac, which tends
    to pile up against 1.0.

    `bounds`: optional (lo, hi) to clamp the interval to the quantity's known
    support (default (0, 1), the common case for this project's *_frac
    fields). Pass None to skip clamping, for quantities without a fixed
    range (e.g. "effective steps used").

    Special cases:
      * all values in {0, 1} and bounds == (0, 1) -> Wilson interval (the
        mean is a binomial proportion; the bootstrap degenerates at the
        boundary)
      * zero sample variance  -> degenerate point interval
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    m = float(v.mean())

    def clamp(lo, hi):
        if bounds is None:
            return lo, hi
        return max(bounds[0], lo), min(bounds[1], hi)

    if bounds == (0.0, 1.0) and np.all((v == 0.0) | (v == 1.0)):
        _, lo, hi = wilson_ci(float(v.sum()), n)
        return m, *clamp(lo, hi)

    if n == 1 or np.ptp(v) == 0.0:
        return m, m, m

    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)

    def percentile_interval():
        return (m,
                float(np.quantile(boot, alpha / 2)),
                float(np.quantile(boot, 1 - alpha / 2)))

    # Bias correction. The mid-p form (half credit for ties) keeps z0 finite
    # when many resamples land exactly on the observed mean.
    below = np.count_nonzero(boot < m) + 0.5 * np.count_nonzero(boot == m)
    p0 = below / n_boot
    if not 0.0 < p0 < 1.0:
        return percentile_interval()
    z0 = _NORM.inv_cdf(p0)

    # Acceleration, estimated by jackknife (leave-one-out means).
    jack = (v.sum() - v) / (n - 1)
    u = jack.mean() - jack
    denom = 6.0 * float(np.sum(u**2)) ** 1.5
    a = float(np.sum(u**3)) / denom if denom > 0 else 0.0

    quantiles = []
    for q in (alpha / 2, 1 - alpha / 2):
        zq = _NORM.inv_cdf(q)
        scale = 1 - a * (z0 + zq)
        if scale <= 0:
            return percentile_interval()
        adj = z0 + (z0 + zq) / scale
        if not np.isfinite(adj):
            return percentile_interval()
        quantiles.append(min(max(_NORM.cdf(adj), 1 / n_boot), 1 - 1 / n_boot))

    lo, hi = (float(np.quantile(boot, q)) for q in quantiles)
    return m, *clamp(lo, hi)


def gold_steps_accuracy_stable_match(results, gold_steps, min_samples=5, n_boot=10000, seed=0):
    """Bucket `results` (a load_results(...)["results"] list) by each
    sample's gold reasoning step count and compute per-bucket accuracy and
    stable-match-fraction means with CIs. Shared by plot_by_gold_steps.py's
    single-model figures and gold_steps_grid.py's combined 3-panel figure so
    the two can't drift out of sync.

    Returns (step_counts, acc_means, acc_lo, acc_hi, smf_means, smf_lo, smf_hi),
    all aligned to step_counts (ascending; buckets with fewer than
    min_samples entries are dropped).
    """
    buckets = defaultdict(lambda: {"correct": [], "stable_match_frac": []})
    for entry in results:
        idx = entry["sample_idx"]
        num_gold = gold_steps[idx]
        is_correct = int(entry["original_result"]["is_correct"][-1])
        smf = entry["original_result"]["stable_match_frac"]
        buckets[num_gold]["correct"].append(is_correct)
        buckets[num_gold]["stable_match_frac"].append(smf)

    step_counts = sorted(k for k, v in buckets.items() if len(v["correct"]) >= min_samples)

    acc_means, acc_lo, acc_hi = [], [], []
    smf_means, smf_lo, smf_hi = [], [], []
    for s in step_counts:
        correct = buckets[s]["correct"]
        smf_vals = buckets[s]["stable_match_frac"]

        n = len(correct)
        _, lo, hi = wilson_ci(sum(correct), n)
        acc_means.append(sum(correct) / n)
        acc_lo.append(lo)
        acc_hi.append(hi)

        m, lo2, hi2 = bootstrap_mean_ci(smf_vals, n_boot=n_boot, seed=seed)
        smf_means.append(m)
        smf_lo.append(lo2)
        smf_hi.append(hi2)

    return (np.array(step_counts), np.array(acc_means), acc_lo, acc_hi,
            np.array(smf_means), smf_lo, smf_hi)


def cluster_bootstrap_proportion_ci(groups, categories, alpha=0.05, n_boot=10000, seed=0):
    """Percentile bootstrap CI for the proportion of each category in a set
    of categorical labels that are *nested* within clusters (e.g. several
    per-step labels belonging to the same question) -- resampling individual
    labels would understate uncertainty because labels from the same cluster
    are correlated, so this resamples whole clusters instead.

    `groups`: list of clusters, each a list of category labels (e.g. one
    list of "original"/"augmented"/"other"/"tie" per question).
    `categories`: the category values to compute intervals for.

    Returns {category: (point_estimate, lo, hi)}.
    """
    rng = np.random.default_rng(seed)
    n = len(groups)
    sizes = np.array([len(g) for g in groups])
    counts = {cat: np.array([g.count(cat) for g in groups]) for cat in categories}
    total = sizes.sum()
    point = {cat: counts[cat].sum() / total for cat in categories}

    idx = rng.integers(0, n, size=(n_boot, n))
    boot_sizes = sizes[idx].sum(axis=1)  # (n_boot,)

    result = {}
    for cat in categories:
        boot_counts = counts[cat][idx].sum(axis=1)  # (n_boot,)
        boot_props = boot_counts / boot_sizes
        lo = float(np.quantile(boot_props, alpha / 2))
        hi = float(np.quantile(boot_props, 1 - alpha / 2))
        result[cat] = (point[cat], lo, hi)
    return result
