# lrm-early-stoppage-replication

Replication and analysis of early-stopping / stable-answer behavior on
GSM8K, ProsQA, and PrOntoQA for three reasoning approaches, written up in
`paper/acl_latex.tex`:

- **ERM / SFT** — a GPT-2 model fine-tuned with explicit textual
  chain-of-thought, using the public `connordilgren/gpt2-{dataset}-cot`
  checkpoints (`evaluate_sft.py`).
- **CoCoNuT** — a GPT-2 model that reasons in continuous "latent thought"
  space instead of text, using the public `connordilgren/gpt2-{dataset}-coconut`
  checkpoints (`evaluate_coconut.py`, model code vendored in `coconut.py`).
- **CODI** — another continuous-thought model, using the public
  `zen-E/CODI-gpt2` (GSM8K) / `connordilgren/gpt2-{dataset}-codi` (ProsQA,
  PrOntoQA) checkpoints (`evaluate_codi.py`, model code vendored in
  `codi.py`).

Every one of the `evaluate_*.py` scripts below takes a `--dataset
{gsm8k,prosqa,prontoqa}` flag (default `gsm8k`). For each model/dataset
combination, every test question is run through:

1. **Early stopping** — record the model's answer if you stop it after each
   reasoning step (text step for ERM, latent "thought" for CoCoNuT/CODI),
   and see how early the answer first matches / permanently matches the
   final answer (`first_match_frac` / `stable_match_frac`). This is the
   analysis behind the paper's Section 3 replication (Table 1, Figures 2-3)
   and its Section 4 task-difficulty analysis (Figure 4), and is what all
   three datasets run.
2. **Question augmentation** — build a minimal-pair "augmented" question
   whose correct answer is guaranteed to differ from the original, and
   re-run early stopping on it. GSM8K: swap the first number in the
   question for a random same-magnitude number. ProsQA: swap the query's
   two named class options everywhere they appear in the question (a pure
   symbol relabelling, so the correct answer is guaranteed to flip). See
   `eval_common.augment_question`.
3. **Slicing** — splice a reasoning step from the augmented run into the
   original run's trace, and check whether the resulting answer still
   tracks the original question, has switched to the augmented question, or
   neither (`original` / `augmented` / `other` / `tie`). This is the
   analysis behind the paper's Section 5 copying-bias experiment.

Steps 2-3 (Section 5 of the paper) run for GSM8K and ProsQA. PrOntoQA has
no augmentable structure recognised yet — `augment_question()` naturally
returns `None` for it, so every PrOntoQA sample takes the `"no_augmentation"`
skip path and only the early-stopping analysis (step 1) actually runs; the
paper explicitly leaves a PrOntoQA version of Section 5 for future work.

There are also a few standalone scripts for analyses **not currently used
in the paper** — see [Other analyses](#other-analyses-not-in-the-paper)
below.

## Setup

```
pip install torch transformers datasets matplotlib numpy huggingface_hub
# evaluate_codi.py additionally needs:
pip install peft safetensors accelerate
```

GSM8K downloads via the `datasets` library on first run (cached
afterward). ProsQA/PrOntoQA aren't mirrored on the Hugging Face Hub, so
their test splits are downloaded directly (via `eval_common.load_test_samples`)
from the replicated paper's own repository
(`connordilgren/are-lrms-easily-interpretable`), pinned to a fixed commit
for reproducibility. If you hit Hugging Face Hub rate limits, set an
`HF_TOKEN` environment variable.

## Directory layout

- `eval_common.py` — helpers shared by the `evaluate_*.py` scripts (dataset
  loading for all three datasets, reasoning-step splitting, question
  augmentation, first/stable-match math, slicing categorisation, random
  same-magnitude numbers, result aggregation + saving).
- `coconut.py` / `codi.py` — vendored third-party model implementations
  (Meta's CoCoNuT, CODI); not meant to be edited.
- `results/` — JSON outputs from the evaluations.
- `figures/` — plotting scripts, `plot_common.py` (shared helpers, palette,
  CI math), and the generated PNGs.
- `slurm/` — job scripts for running the evaluations on the University of
  York's Viking HPC cluster.
- `paper/` — the ACL-format writeup (`acl_latex.tex`), built in place (see
  [Building the paper](#building-the-paper)).

## 1. Run an evaluation

Each script processes the full test split of the chosen dataset and writes
an aggregate + per-question JSON file to `results/`. These need a GPU and
the relevant model checkpoint, and take a while to run (multiple
`generate()` calls per question).

### `evaluate_sft.py` (ERM baseline)

```
python evaluate_sft.py [--dataset {gsm8k,prosqa,prontoqa}] [--checkpoint-path PATH]
```

| Option | Meaning |
|---|---|
| `--dataset` | Which dataset's test split to evaluate on (default: `gsm8k`). |
| `--checkpoint-path` | Path to a local GPT-2 SFT checkpoint (`state_dict`) to load, overriding the default. Default: automatically download `connordilgren/gpt2-{dataset}-cot`'s `checkpoint_25` from the Hugging Face Hub. |

Output: `results/sft.json` (`gsm8k`), or `results/sft_{dataset}.json`
otherwise.

### `evaluate_coconut.py` (CoCoNuT)

```
python evaluate_coconut.py [--dataset {gsm8k,prosqa,prontoqa}] [--no-force-answer]
```

| Option | Meaning |
|---|---|
| `--dataset` | Which dataset's test split to evaluate on (default: `gsm8k`). |
| `--no-force-answer` | By default, the string `"### "` is inserted immediately after the `<\|end-latent\|>` token so the model is forced to answer right away instead of continuing on with its own explicit-CoT text. Pass this flag to disable that and let the model decide on its own — used by some of the paper's GSM8K figures (see below) so COCONUT's curve reflects its own answer timing, like CODI's and the ERM's. No such variant was run for ProsQA/PrOntoQA. |

Automatically downloads the public `connordilgren/gpt2-{dataset}-coconut`
checkpoint from the Hugging Face Hub (`checkpoint_33`/`_40`/`_36` for
gsm8k/prosqa/prontoqa respectively -- each dataset's CoCoNuT checkpoint
converged at a different training epoch). To use a different checkpoint,
edit `CHECKPOINT_REPO` / `CHECKPOINT_FILE` near the top of the file.

Output: `results/coconut.json` (default dataset/force-answer), with
`_{dataset}` and/or `_no_force_answer` suffixes added as applicable (e.g.
`results/coconut_prosqa_no_force_answer.json`).

### `evaluate_codi.py` (CODI)

```
python evaluate_codi.py [--dataset {gsm8k,prosqa,prontoqa}]
```

| Option | Meaning |
|---|---|
| `--dataset` | Which dataset's test split to evaluate on (default: `gsm8k`). |

Automatically downloads the public checkpoint from the Hugging Face Hub:
`zen-E/CODI-gpt2` for `gsm8k` (the original authors' checkpoint), or
`connordilgren/gpt2-{dataset}-codi` for `prosqa`/`prontoqa` (connordilgren's
own reruns of the same upstream CODI training codebase on those datasets).

Output: `results/codi.json` (`gsm8k`), or `results/codi_{dataset}.json`
otherwise.

## 2. Running the evaluations on Viking (SLURM)

`slurm/` has job scripts for running the evaluations above on the
University of York's Viking HPC cluster (single-GPU `gpu` partition jobs,
one per evaluation, plus a second CoCoNuT job for the `--no-force-answer`
variant). Each script's python invocation matches its section-1 form above,
and `evaluate_sft.py` / `evaluate_coconut.py` / `evaluate_copying_bias.py`
pick up the checkpoint paths already hardcoded in those scripts — edit the
scripts (not the job files) if you need to point at a different
checkpoint.

Setup (once, on a login node — no GPU needed):

```
git clone <this repo> && cd early-stoppage-replication
bash slurm/setup_env.sh   # creates slurm/../.venv with all dependencies
```

Then edit `--account=YOUR_ACCOUNT` in whichever job script(s) you plan to
submit (Viking requires a project account code; see
`using_viking/submitting_jobs.html` in the Viking docs if you don't know
yours).

Submit all five GSM8K evaluations at once, as a job array (each array task
gets its own GPU allocation, run concurrently):

```
sbatch slurm/evaluate_all_array.slurm
```

Or submit an individual evaluation on its own:

```
sbatch slurm/evaluate_sft.slurm
sbatch slurm/evaluate_coconut_force_answer.slurm     # default: "### " forced after <|end-latent|>
sbatch slurm/evaluate_coconut_no_force_answer.slurm  # model decides when to stop on its own
sbatch slurm/evaluate_codi.slurm
sbatch slurm/evaluate_copying_bias.slurm
```

For the ProsQA/PrOntoQA replication (early-stopping, and — for ProsQA —
the augmentation/slicing analysis too), submit the six-task array covering
all three models on both datasets:

```
sbatch slurm/evaluate_prosqa_prontoqa_array.slurm
```

Submit from the repo root (as above) — every job script locates
`slurm/common.sh` via `$SLURM_SUBMIT_DIR`, the directory `sbatch` was run
from, so it needs to be `slurm/common.sh` relative to wherever you invoke
`sbatch`.

All job scripts request the `gpu` partition (nVidia A40, 1 GPU, 16G host
memory, 1-day time limit — raise `--time` towards the partition's 3-day cap
if a job is killed for running out of time) and log to
`<job-name>-<job-id>.log` / `.err` in the directory you submit from.
`slurm/common.sh`, sourced by every job script, loads the Python module,
`cd`s to the repo root, and activates the venv from `setup_env.sh` — it's
not meant to be run directly.

> **Why `$SLURM_SUBMIT_DIR` and not a path relative to the script itself?**
> `sbatch` copies the job script into a per-job spool directory
> (`/var/spool/slurmd.spool/...`) and runs it from there, so a
> `$(dirname "${BASH_SOURCE[0]}")`-style trick resolves to that spool
> directory, not `slurm/` in your checkout, and fails to find `common.sh`.
> `$SLURM_SUBMIT_DIR` isn't affected by the copy.

Note: the ERM/SFT scripts are small enough (GPT-2 small, short forced
generations) that a full 500-1,319-question evaluation also completes in a
few minutes on CPU — you don't strictly need a GPU allocation for
`evaluate_sft.py`. CoCoNuT/CODI's `generate()` calls are more expensive and
are the ones that actually benefit from a GPU.

## 3. Figures

Figure scripts only read `results/*.json` (no GPU needed) plus the gold
reasoning-step counts for the ones binning by that (GSM8K's test split via
`datasets`, ProsQA/PrOntoQA's via the same repo the results themselves come
from). Run them from anywhere — output paths are resolved relative to the
script's own location, not the working directory.

### `figures/gold_steps_grid.py` — used in the paper (Figure 3, Figure 7)

```
python figures/gold_steps_grid.py [--dataset {gsm8k,prosqa,prontoqa}]
```

1x3 grid (COCONUT, CODI, ERM) of accuracy (Wilson 95% CI) and stable-match
fraction (BCa bootstrap 95% CI) against the number of gold reasoning steps,
sharing one legend and axis labels — this is what's actually embedded in
the paper, not `plot_by_gold_steps.py` below.

Output: `figures/gold_steps_grid.png` (`gsm8k`), or
`figures/gold_steps_grid_{dataset}.png` otherwise.

### `figures/plot_slicing.py` — used in the paper (Figure 5, Figure 6)

```
python figures/plot_slicing.py [--dataset {gsm8k,prosqa,prontoqa}] [--effective-only]
```

| Option | Meaning |
|---|---|
| `--dataset` | Which dataset's results to plot (default: `gsm8k`). PrOntoQA has no slicing data (see the dataset note above), so only `gsm8k`/`prosqa` are meaningful. |
| `--effective-only` | Restrict to slicing trials that spliced into an *effective* step -- one before the original run's answer had already stabilized (index `i` < that question's `stable_match`, the same quantity behind `plot_effective_steps.py`). Drops trials on already-redundant steps, where a splice couldn't have changed the answer regardless of any copying bias. **The paper only uses the `--effective-only` figures** (the unrestricted ones are commented out in `acl_latex.tex`, kept as a reference point in the source). |

Plots all three models at once. Stacked bar chart of the slicing-analysis
answer status (`original` / `augmented` / `other` / `tie`) per model, with
95% CI error bars from a cluster bootstrap over *questions* (steps within a
question aren't independent, so resampling individual steps would
understate the uncertainty). "0%" segment labels are suppressed (a segment
can round to 0% while still containing a few real trials — see the
category's raw count in `results/*.json` if you need the exact number).

Output: `figures/slicing_status{_effective}{_dataset}.png`, e.g.
`figures/slicing_status_effective_prosqa.png`.

### `figures/plot_effective_steps.py` — used in the paper (Figure 4)

```
python figures/plot_effective_steps.py [--dataset {gsm8k,prosqa,prontoqa}]
```

For all three models, plots the mean "effective steps used"
(`stable_match_frac × total steps`) against the number of gold reasoning
steps, with BCa bootstrap 95% CI bands. The paper only uses the `gsm8k`
(default) output.

Output: `figures/effective_steps_by_gold_steps.png` (`gsm8k`), or
`figures/effective_steps_by_gold_steps_{dataset}.png` otherwise.

### `figures/replication_grid.py` — used in the paper (Figure 2)

```
python figures/replication_grid.py
```

No CLI options; always plots all three datasets. 2x3 grid (rows: first
match / stable match; columns: GSM8K / ProsQA / PrOntoQA) comparing the
original paper's reported numbers (no CI — there's no per-sample data
behind a single published number) against our own replication numbers (95%
bootstrap CI error bars), for all three models. Uses
`results/coconut_no_force_answer.json` for the GSM8K/COCONUT panel (see
`evaluate_coconut.py --no-force-answer` above); every other cell uses the
default (forced-answer) result file.

Output: `figures/replication_grid.png`

### Superseded standalone scripts (not used in the paper)

`figures/plot_by_gold_steps.py` (single-model version of
`gold_steps_grid.py`) and `figures/replication_graph.py` (single-dataset,
single-metric version of `replication_grid.py`) still work and are useful
for a quick one-off look at a single model/dataset/metric, but neither's
output is embedded in `acl_latex.tex` — the grid versions above replaced
them. See each script's `--help` for usage.

## Reproducing the paper's figures and tables

Assuming you already have all the model checkpoints available (see
Section 1), this is the full sequence from a clean `results/` directory to
everything `acl_latex.tex` includes:

```
# 1. Evaluations -- GSM8K (10 result files; COCONUT needs both variants)
python evaluate_sft.py
python evaluate_coconut.py
python evaluate_coconut.py --no-force-answer
python evaluate_codi.py

# 1b. ProsQA (adds the augmentation/slicing analysis automatically)
python evaluate_sft.py --dataset prosqa
python evaluate_coconut.py --dataset prosqa
python evaluate_codi.py --dataset prosqa

# 1c. PrOntoQA (early-stopping only -- see the dataset note above)
python evaluate_sft.py --dataset prontoqa
python evaluate_coconut.py --dataset prontoqa
python evaluate_codi.py --dataset prontoqa

# 2. Figures
python figures/replication_grid.py                        # Figure 2 (fig:replication)
python figures/gold_steps_grid.py                          # Figure 3 (fig:steps)
python figures/plot_effective_steps.py                     # Figure 4 (fig:effective)
python figures/plot_slicing.py --effective-only            # Figure 5 (fig:copy-effective)
python figures/plot_slicing.py --dataset prosqa --effective-only   # Figure 6 (fig:copy-effective-prosqa)
python figures/gold_steps_grid.py --dataset prosqa         # Figure 7, appendix (fig:steps-prosqa)
```

Table 1 (`tab:performance`)'s accuracy numbers and the appendix's
sample-exclusion tables (`tab:exclusion*`) are transcribed by hand from
each run's `accuracy` / `slicing_*_count` fields in `results/*.json` — they
aren't auto-generated into the `.tex` file.

## Building the paper

`paper/acl_latex.tex` is a standard two-pass `pdflatex` document; build it
in place (its figure `\includegraphics` paths are relative to `paper/`,
which is why the PNGs above are read straight from `figures/` — copy or
symlink them into `paper/` first, or point `\graphicspath` there, if
they're not already alongside `acl_latex.tex`):

```
cd paper
pdflatex -interaction=nonstopmode -halt-on-error acl_latex.tex
pdflatex -interaction=nonstopmode -halt-on-error acl_latex.tex   # second pass resolves cross-references
```

Output: `paper/acl_latex.pdf`.

## Other analyses (not in the paper)

A few scripts explore questions the paper doesn't currently report on —
kept for reference / future work, not wired into any figure above:

- **`evaluate_copying_bias.py`** / **`evaluate_copying_bias_prosqa.py`** —
  an earlier, simpler version of the Section 5 copying-bias question: at
  each truncation length, swap the last explicit reasoning step's numeric
  result (GSM8K) or last word (ProsQA) for an implausible replacement and
  see how often the ERM's forced answer copies it. Superseded in the paper
  by the augmentation/slicing analysis above, which tests a *valid*
  minimal-pair alternative rather than an implausible one, but its results
  are still interesting as a simpler baseline. Output:
  `results/copying_bias.json` / `results/copying_bias_prosqa.json`.
