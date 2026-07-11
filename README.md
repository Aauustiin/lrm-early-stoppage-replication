# lrm-early-stoppage-replication

Replication and analysis of early-stopping / stable-answer behavior on GSM8K
for three reasoning approaches:

- **ERM / SFT** — a GPT-2 model fine-tuned on GSM8K with explicit textual
  chain-of-thought (`evaluate_sft.py`).
- **CoCoNuT** — a GPT-2 model that reasons in continuous "latent thought"
  space instead of text (`evaluate_coconut.py`, model code vendored in
  `coconut.py`).
- **CODI** — another continuous-thought model, using the public
  `zen-E/CODI-gpt2` checkpoint (`evaluate_codi.py`, model code vendored in
  `codi.py`).

For each model, every GSM8K test question is run through three analyses:

1. **Early stopping** — record the model's answer if you stop it after each
   reasoning step (text step for ERM, latent "thought" for CoCoNuT/CODI),
   and see how early the answer first matches / permanently matches the
   final answer (`first_match_frac` / `stable_match_frac`).
2. **Question augmentation** — swap the first number in the question for a
   random same-magnitude number, and re-run early stopping on it.
3. **Slicing** — splice a reasoning step from the augmented run into the
   original run's trace, and check whether the resulting answer still
   tracks the original question, has switched to the augmented question, or
   neither (`original` / `augmented` / `other` / `tie`).

## Setup

```
pip install torch transformers datasets matplotlib numpy
# evaluate_codi.py additionally needs:
pip install peft huggingface_hub safetensors accelerate
```

All scripts download the GSM8K test split via the `datasets` library on
first run (cached afterward). If you hit Hugging Face Hub rate limits, set
an `HF_TOKEN` environment variable.

## Directory layout

- `eval_common.py` — helpers shared by the three `evaluate_*.py` scripts
  (question augmentation, first/stable-match math, slicing categorisation,
  result aggregation + saving).
- `coconut.py` / `codi.py` — vendored third-party model implementations
  (Meta's CoCoNuT, CODI); not meant to be edited.
- `results/` — JSON outputs from the evaluations.
- `figures/` — plotting scripts, `plot_common.py` (shared helpers, palette,
  CI math), and the generated PNGs.

## 1. Run an evaluation

Each script processes the full GSM8K test split and writes an aggregate +
per-question JSON file to `results/`. These need a GPU and the relevant
model checkpoint, and take a while to run (multiple `generate()` calls per
question).

### `evaluate_sft.py` (ERM baseline)

```
python evaluate_sft.py [--checkpoint-path PATH]
```

| Option | Meaning |
|---|---|
| `--checkpoint-path` | Path to the GPT-2 SFT checkpoint (`state_dict`) to load. Default: `/users/cns542/scratch/coconut/gsm-cot/checkpoint_7` (a cluster-specific path — override it for your own environment). |

Output: `results/sft.json`

### `evaluate_coconut.py` (CoCoNuT)

```
python evaluate_coconut.py [--no-force-answer]
```

| Option | Meaning |
|---|---|
| `--no-force-answer` | By default, the string `"### "` is inserted immediately after the `<\|end-latent\|>` token so the model is forced to answer right away instead of continuing on with its own explicit-CoT text. Pass this flag to disable that and let the model decide on its own — useful for an ablation comparing forced vs. free-form answering. |

The checkpoint path is hardcoded (`CHECKPOINT_PATH` near the top of the
file) rather than a CLI flag — edit it to point at your own CoCoNuT
checkpoint.

Output: `results/coconut.json` (default), or
`results/coconut_no_force_answer.json` (with `--no-force-answer`).

### `evaluate_codi.py` (CODI)

```
python evaluate_codi.py
```

No CLI options. Automatically downloads the public `zen-E/CODI-gpt2`
checkpoint from the Hugging Face Hub.

Output: `results/codi.json`

## 2. Figures

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
python figures/plot_slicing.py
```

No CLI options — plots all three models at once. Stacked bar chart of the
slicing-analysis answer status (`original` / `augmented` / `other` / `tie`)
per model, with 95% CI error bars from a cluster bootstrap over *questions*
(steps within a question aren't independent, so resampling individual steps
would understate the uncertainty).

Output: `figures/slicing_status.png`

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

# 2. Figures (only need results/*.json from step 1)
python figures/plot_by_gold_steps.py sft
python figures/plot_slicing.py
python figures/plot_effective_steps.py
python figures/replication_graph.py
```
