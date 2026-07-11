"""Shared helpers for evaluate_sft.py / evaluate_coconut.py / evaluate_codi.py.

Each of those scripts runs a different model, but they all follow the same
early-stopping / question-augmentation / slicing protocol on GSM8K and dump
the same result schema. This module holds the model-agnostic pieces.
"""

import math
import operator
import random
import re
import json


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
