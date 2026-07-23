# lrm-early-stoppage-replication

Replication and analysis of early-stopping / stable-answer behavior on
GSM8K, ProsQA, and PrOntoQA for three reasoning approaches:

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
   analysis behind the paper's Section 3 replication (Table 1, Figure 2) and
   is what all three datasets run.
2. **Question augmentation** — swap the first number in the question for a
   random same-magnitude number, and re-run early stopping on it.
3. **Slicing** — splice a reasoning step from the augmented run into the
   original run's trace, and check whether the resulting answer still
   tracks the original question, has switched to the augmented question, or
   neither (`original` / `augmented` / `other` / `tie`).

Steps 2-3 (Sections 4-5 of the paper) are GSM8K-only in practice: they swap
a number in the question, and ProsQA/PrOntoQA questions don't contain any
— `augment_question()` naturally returns `None` for them, so every
ProsQA/PrOntoQA sample takes the "no_number" skip path and only the
early-stopping analysis (step 1) actually runs.

There's also a fourth, ERM/GSM8K-only analysis (`evaluate_copying_bias.py`):
swap the numeric result of the ERM's final explicit reasoning step for a
random same-magnitude number and force an answer, to see how often the
ERM's answer just copies that (otherwise meaningless) number.

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
| `--no-force-answer` | By default, the string `"### "` is inserted immediately after the `<\|end-latent\|>` token so the model is forced to answer right away instead of continuing on with its own explicit-CoT text. Pass this flag to disable that and let the model decide on its own — useful for an ablation comparing forced vs. free-form answering. |

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

### `evaluate_copying_bias.py` (ERM copying-bias experiment, GSM8K-only)

```
python evaluate_copying_bias.py [--checkpoint-path PATH]
```

| Option | Meaning |
|---|---|
| `--checkpoint-path` | Path to the GPT-2 SFT checkpoint (`state_dict`) to load. Same default/meaning as `evaluate_sft.py`'s `gsm8k` default. |

Same model, GSM8K test split, and per-truncation-length forced-answer
protocol as `evaluate_sft.py`'s early-stopping analysis (Section 3), except
at each truncation length the numeric result of the truncated trace's last
explicit reasoning step (e.g. the `24` in `<<48/2=24>>`) is swapped for a
random number of the same magnitude (`rand_same_magnitude()` from
`eval_common.py`) before forcing an answer, instead of leaving it as
generated. This gives one trial per truncation length per question (skipped
for a given length if that step's result isn't a plain positive integer).
GSM8K-only, like the augmentation/slicing analysis: it relies on the same
"swap a number in the reasoning trace" mechanic, which doesn't apply to
ProsQA/PrOntoQA.

Output: `results/copying_bias.json` — includes `match_rate` (fraction of
trials where the model's forced answer equals the random number),
`num_trials`, `num_samples`, and per-question `results` (each with a list
of per-truncation-length `trials`).

## 2. Running the evaluations on Viking (SLURM)

`slurm/` has job scripts for running the evaluations above on the
University of York's Viking HPC cluster (single-GPU `gpu` partition jobs,
one per evaluation, plus a second CoCoNuT job for the `--no-force-answer`
variant). Each script's python invocation matches its section-1 form above,
and `evaluate_sft.py` / `evaluate_coconut.py` / `evaluate_copying_bias.py`
pick up the checkpoint paths already hardcoded in those scripts — edit the
scripts (not the job files) if you need to point at a different checkpoint.

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

For the ProsQA/PrOntoQA Section 3 replication (early-stopping /
matching-metrics only -- see the dataset note above), submit the
six-task array covering all three models on both datasets:

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

## 3. Figures

Figure scripts only read `results/*.json` (no GPU needed) plus the GSM8K
test split for the ones binning by gold-step count. Run them from anywhere
— output paths are resolved relative to the script's own location, not the
working directory.

### `figures/plot_by_gold_steps.py`

```
python figures/plot_by_gold_steps.py MODEL [--min-samples N] [--n-boot N] [--seed N]
```

| Option | Meaning |
|---|---|
| `MODEL` | One of `sft`, `coconut`, `codi` (required). |
| `--min-samples` | Minimum samples a gold-step bucket needs to be plotted (default: 5). |
| `--n-boot` | Bootstrap resamples for the stable-match-fraction CI (default: 10000). |
| `--seed` | Bootstrap RNG seed, so bands are identical across reruns (default: 0). |

Plots accuracy (Wilson 95% CI) and stable-match fraction (BCa bootstrap 95%
CI) against the number of gold reasoning steps, for one model.

Output: `figures/{model}_by_gold_steps.png`

### `figures/plot_slicing.py`

```
python figures/plot_slicing.py [--effective-only]
```

| Option | Meaning |
|---|---|
| `--effective-only` | Restrict to slicing trials that spliced into an *effective* step -- one before the original run's answer had already stabilized (index `i` < that question's `stable_match`, the same quantity behind `plot_effective_steps.py`). Drops trials on already-redundant steps, where a splice couldn't have changed the answer regardless of any copying bias. |

Plots all three models at once. Stacked bar chart of the slicing-analysis
answer status (`original` / `augmented` / `other` / `tie`) per model, with
95% CI error bars from a cluster bootstrap over *questions* (steps within a
question aren't independent, so resampling individual steps would
understate the uncertainty).

Output: `figures/slicing_status.png` (`figures/slicing_status_effective.png` with `--effective-only`)

### `figures/plot_effective_steps.py`

```
python figures/plot_effective_steps.py
```

No CLI options. For all three models, plots the mean "effective steps
used" (`stable_match_frac × total steps`) against the number of gold
reasoning steps, with BCa bootstrap 95% CI bands.

Output: `figures/effective_steps_by_gold_steps.png`

### `figures/replication_graph.py`

```
python figures/replication_graph.py [METRIC]
```

| Option | Meaning |
|---|---|
| `METRIC` | `first-match` or `stable-match` (default: `first-match`). |

Bar chart comparing the original paper's reported numbers (no CI — there's
no per-sample data behind a single published number) against our own
replication numbers (95% bootstrap CI error bars), for all three models.

Output: `figures/first_match.png` or `figures/stable_match.png`

## Typical workflow

```
# 1. Run the evaluations you need (GPU + checkpoints required)
python evaluate_sft.py
python evaluate_coconut.py
python evaluate_codi.py
python evaluate_copying_bias.py

# 1b. Section 3 (early-stopping / matching-metrics) on the other two
#     datasets -- see the dataset note above for why only this analysis
#     (not augmentation/slicing) runs for these two
python evaluate_sft.py --dataset prosqa
python evaluate_coconut.py --dataset prosqa
python evaluate_codi.py --dataset prosqa
python evaluate_sft.py --dataset prontoqa
python evaluate_coconut.py --dataset prontoqa
python evaluate_codi.py --dataset prontoqa

# 2. Figures (only need results/*.json from step 1; GSM8K-only for now --
#    none of the figures/*.py scripts plot the prosqa/prontoqa results yet)
python figures/plot_by_gold_steps.py sft
python figures/plot_slicing.py
python figures/plot_slicing.py --effective-only
python figures/plot_effective_steps.py
python figures/replication_graph.py
```
