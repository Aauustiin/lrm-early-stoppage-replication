"""Shared helpers for evaluate_sft.py / evaluate_coconut.py / evaluate_codi.py.

Each of those scripts runs a different model, but they all follow the same
early-stopping / question-augmentation / slicing protocol and dump the same
result schema, across three datasets: GSM8K, ProsQA, and PrOntoQA. This
module holds the model-agnostic, dataset-aware pieces.
"""

import math
import operator
import random
import re
import json
import urllib.request

DATASETS = ("gsm8k", "prosqa", "prontoqa")

# ProsQA/PrOntoQA (unlike GSM8K) aren't mirrored on the Hugging Face Hub, so
# their test splits are downloaded from the replicated paper's own repo.
# Pinned to a commit (not a floating branch ref) since the repo has no
# versioned release -- a `main`-branch fetch could silently start returning
# a different test split.
_LRM_PAPER_REPO_COMMIT = "32f413d8d55239d9bc54bb6b6ec37b0630891ed4"
_LRM_PAPER_REPO_RAW = (
    "https://raw.githubusercontent.com/connordilgren/are-lrms-easily-interpretable"
    f"/{_LRM_PAPER_REPO_COMMIT}"
)


def load_test_samples(dataset):
    """Return the test split of `dataset` (one of DATASETS) as a list of
    (question, ground_truth_answer) pairs.

    GSM8K comes from its standard Hugging Face Hub mirror. ProsQA/PrOntoQA
    come from the replicated paper's repo (see _LRM_PAPER_REPO_COMMIT) --
    their "answer" field is already the bare final answer (e.g. "True" or
    "Sally is a sterpus."), unlike GSM8K's raw answer field, which is a
    full reasoning trace ending in "#### 42" and needs splitting.
    """
    if dataset == "gsm8k":
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("openai/gsm8k", "main")["test"]
        return [
            (s["question"], s["answer"].split("####")[1].strip())
            for s in ds
        ]
    elif dataset in ("prosqa", "prontoqa"):
        url = f"{_LRM_PAPER_REPO_RAW}/data/{dataset}_test.json"
        with urllib.request.urlopen(url) as response:
            samples = json.load(response)
        return [(s["question"], s["answer"]) for s in samples]
    else:
        raise ValueError(f"Unknown dataset {dataset!r}, expected one of {DATASETS}")


def split_reasoning_steps(dataset, full_output_text):
    """Split a CoT/SFT model's own generated reasoning (the decoded
    question + generated continuation) into individual steps, for use in
    evaluate_sft.py's early-stopping loop.

    GSM8K's gold traces (and the CoT model's learned output format) are
    preprocessed down to just calculator-notation annotations, so each step
    is extracted as one "<<...>>" span.

    ProsQA and PrOntoQA have no calculator notation -- their gold traces
    (and, by extension, what the CoT model was trained to generate) are
    newline-separated natural-language sentences: `question + "\\n" +
    "\\n".join(steps) + "\\n### " + answer` (see the replicated paper's
    dataset.py). Since neither dataset's questions contain embedded
    newlines, every non-empty line after the first (the question) and
    before the answer delimiter is one step.
    """
    if dataset == "gsm8k":
        return re.findall(r"<<.*?>>", full_output_text)
    elif dataset in ("prosqa", "prontoqa"):
        lines = full_output_text.split("\n")[1:]  # drop the question line
        steps = []
        for line in lines:
            if "#" in line:
                break
            if line.strip():
                steps.append(line.strip())
        return steps
    else:
        raise ValueError(f"Unknown dataset {dataset!r}, expected one of {DATASETS}")


# The exact token sequence that immediately precedes the answer in all three
# datasets' CoT training format (see split_reasoning_steps's docstring), used
# by evaluate_sft.py to force the ERM to answer right away when testing a
# truncated reasoning trace. Confirmed directly against a real GSM8K CoT
# completion (`...\n<<9*2=18>>\n### 18`) and against the replicated paper's
# own early-stopping reference script (experiments/early_stopping/run.py),
# which hardcodes "###" for every dataset -- there's no "#### " (four-hash)
# variant anywhere in the actual training pipeline, despite that being
# standard GSM8K/Cobbes-style notation. No trailing space: the model
# generates that itself as part of its learned continuation, same as the
# reference implementation's token-level `get_delimiter_tokens`.
FORCE_ANSWER_DELIM = "###"


def step_token_counts(tokenizer, gen_ids, num_steps):
    """Token-index prefixes of `gen_ids` (the CoT model's own raw generated
    token IDs for one sample, i.e. output_ids[0][len(input_ids):]) covering
    0, 1, ..., num_steps - 1 *complete* reasoning steps, for use by
    evaluate_sft.py's early-stopping loop.

    Step boundaries are found by decoding each generated token individually
    and checking for "\\n", directly on the token stream -- not by decoding
    the full text, splitting into step strings, rejoining a prefix, and
    re-encoding. That text-roundtrip can retokenize differently than the
    model's original generation (BPE merges depend on surrounding context),
    putting the forced prompt slightly out of distribution. This was
    confirmed to be the cause of near-total garbage forced-early answers on
    PrOntoQA (e.g. repeated byte-fallback tokens instead of "True"/"False")
    at truncation levels that should have had ample context to answer
    cleanly -- switching to token-level slicing eliminated it entirely in
    spot checks. The reference implementation
    (experiments/early_stopping/run.py's find_step_boundaries /
    extract_reasoning_tokens_cot in the replicated paper's repo) does the
    same thing.
    """
    counts = [0]
    for i, token_id in enumerate(gen_ids.tolist()):
        if len(counts) >= num_steps:
            break
        if "\n" in tokenizer.decode([token_id]):
            counts.append(i + 1)
    return counts


def find_first_number(text):
    """Return (start, end, int_value) of the first positive integer in text."""
    for m in re.finditer(r'(?<!\d)(\d[\d,]*)(?!\d)', text):
        try:
            value = int(m.group(1).replace(',', ''))
        except ValueError:
            continue
        if value > 0:
            return m.start(1), m.end(1), value
    return None, None, None


def rand_same_magnitude(n):
    """Random integer with same order of magnitude as n, guaranteed != n."""
    mag = 10 ** math.floor(math.log10(n))
    while True:
        r = random.randint(mag, mag * 10 - 1)
        if r != n:
            return r


def augment_question(question):
    """Swap the first positive integer in `question` for a same-magnitude
    random one. Returns (augmented_question, orig_number), or (None, None)
    if the question has no number to swap.
    """
    start, end, orig = find_first_number(question)
    if orig is None:
        return None, None
    new_num = rand_same_magnitude(orig)
    return question[:start] + str(new_num) + question[end:], orig


def match_fractions(answers, num_steps, eq=operator.eq):
    """Given per-step answers (answers[-1] is the final answer), return
    (first_match, stable_match, first_match_frac, stable_match_frac):
      * first_match:  index of the first answer equal to the final answer.
      * stable_match: smallest k such that answers[k:] are all equal to the
        final answer.
    """
    first_match = next(idx for idx, a in enumerate(answers) if eq(a, answers[-1]))

    stable_match = num_steps
    for k in range(num_steps + 1):
        if all(eq(a, answers[-1]) for a in answers[k:]):
            stable_match = k
            break

    if num_steps == 0:
        first_match_frac = 0
        stable_match_frac = 0
    else:
        first_match_frac = first_match / num_steps
        stable_match_frac = stable_match / num_steps

    return first_match, stable_match, first_match_frac, stable_match_frac


def slicing_status(sliced_answer, orig_answer, aug_answer, eq=operator.eq):
    """Categorise a spliced-step answer relative to the original/augmented
    answers at that step: 'tie' (they already agreed), 'original',
    'augmented', or 'other'.
    """
    if eq(orig_answer, aug_answer):
        return "tie"
    if eq(sliced_answer, orig_answer):
        return "original"
    if eq(sliced_answer, aug_answer):
        return "augmented"
    return "other"


def aggregate_and_save(results, model, dataset, out_path):
    """Compute the shared summary stats (average match fractions, slicing
    proportions, accuracy) and write `{summary..., results}` to out_path.
    """
    average_first_match_frac = sum(r["original_result"]["first_match_frac"] for r in results) / len(results)
    average_stable_match_frac = sum(r["original_result"]["stable_match_frac"] for r in results) / len(results)

    slicing_original_count = sum(r.get("slicing_original_count") or 0 for r in results)
    slicing_augmented_count = sum(r.get("slicing_augmented_count") or 0 for r in results)
    slicing_other_count = sum(r.get("slicing_other_count") or 0 for r in results)
    slicing_tie_count = sum(r.get("slicing_tie_count") or 0 for r in results)
    slicing_total = (
        slicing_original_count + slicing_augmented_count
        + slicing_other_count + slicing_tie_count
    )

    def proportion(count):
        return count / slicing_total if slicing_total else 0

    accuracy = sum(r["original_result"]["is_correct"][-1] for r in results) / len(results)

    output = {
        "model": model,
        "dataset": dataset,
        "average_first_match_frac": average_first_match_frac,
        "average_stable_match_frac": average_stable_match_frac,
        "slicing_original_count": slicing_original_count,
        "slicing_augmented_count": slicing_augmented_count,
        "slicing_other_count": slicing_other_count,
        "slicing_tie_count": slicing_tie_count,
        "slicing_original_proportion": proportion(slicing_original_count),
        "slicing_augmented_proportion": proportion(slicing_augmented_count),
        "slicing_other_proportion": proportion(slicing_other_count),
        "slicing_tie_proportion": proportion(slicing_tie_count),
        "accuracy": accuracy,
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
