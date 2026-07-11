"""Shared helpers for the plotting scripts (plot_slicing.py,
plot_effective_steps.py, plot_by_gold_steps.py).
"""

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
from datasets import load_dataset

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

_NORM = NormalDist()


def count_gold_steps(answer: str) -> int:
    """Count reasoning steps in a GSM8k gold answer (lines before ####)."""
    lines = [l.strip() for l in answer.strip().split("\n") if l.strip()]
    return sum(1 for l in lines if not l.startswith("####"))


def load_gold_steps():
    """Load the GSM8K test split and return its per-sample gold step counts."""
    print("Loading GSM8k test split...", flush=True)
    dataset = load_dataset("gsm8k", "main", split="test")
    return [count_gold_steps(ex["answer"]) for ex in dataset]


def load_results(model: str, expected_len: int | None = None):
    """Load results/{model}.json. If `expected_len` is given (typically
    len(gold_steps) from load_gold_steps()), assert the results cover every
    GSM8K test sample.
    """
    path = RESULTS_DIR / f"{model}.json"
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
