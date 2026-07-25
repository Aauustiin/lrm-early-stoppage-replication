"""Tests a candidate shallow heuristic for the ERM (SFT model) on PrOntoQA:
does its forced-early-stopping answer match "true iff the CoT-so-far and the
query end in the same word, reversed if the query's final clause contains
'not'"?

PrOntoQA's query is never literal copying (unlike GSM8k/ProsQA): the answer
is True/False and never appears verbatim in the CoT. This checks a
plausible shallow substitute -- last-word matching between the (possibly
truncated) CoT and the query, with the query's own negation flipping the
predicted polarity -- against every forced answer in the early-stopping
experiment (results/sft_prontoqa.json), not just the final one.

Run after evaluate_sft.py --dataset prontoqa has produced that file; this
script does no model inference of its own.
"""
import re
import json
from collections import defaultdict

# Same convention as evaluate_copying_bias_prosqa.py: last run of word
# characters in a string, ignoring trailing punctuation.
LAST_WORD_RE = re.compile(r"(\w+)[^\w]*$")

# PrOntoQA's trailing query sentence, e.g. "True or false: Stella is not
# floral." -- captures the clause after the "True or false:" prefix.
QUERY_CLAUSE_RE = re.compile(r"True or false: (.+)\.\s*$")


def extract_last_word(text):
    text = text.strip()
    if not text:
        return None
    match = LAST_WORD_RE.search(text)
    return match.group(1).lower() if match else None


def heuristic_prediction(query_word, query_has_not, cot_so_far):
    """True/False per the literal heuristic: same last word as the query,
    reversed if the query clause contains "not". `cot_so_far` is the list
    of explicit reasoning steps visible at this forced-answer point (may be
    empty, at the very first forced answer before any step is generated).
    """
    cot_word = extract_last_word(" ".join(cot_so_far))
    same_word = cot_word is not None and cot_word == query_word
    predicted_true = same_word != query_has_not
    return "True" if predicted_true else "False"


def main():
    with open("results/sft_prontoqa.json") as f:
        data = json.load(f)

    total = 0
    matches = 0
    by_step = defaultdict(lambda: [0, 0])  # step index -> [matches, total]

    for r in data["results"]:
        orr = r["original_result"]
        question = orr["question"]
        steps = orr["steps"]
        num_steps = orr["num_steps"]
        model_answers = orr["model_answers"]  # len == num_steps + 1

        clause = QUERY_CLAUSE_RE.search(question).group(1)
        query_has_not = re.search(r"\bnot\b", clause) is not None
        query_word = extract_last_word(clause)

        for i in range(num_steps + 1):
            pred = heuristic_prediction(query_word, query_has_not, steps[:i])
            actual = model_answers[i]
            total += 1
            matches += (pred == actual)
            key = "final" if i == num_steps else i
            by_step[key][0] += (pred == actual)
            by_step[key][1] += 1

    print(f"Overall agreement (all forced-answer points): {matches}/{total} = {matches/total:.1%}")
    print()
    print("By step index (0 = forced answer before any CoT step is visible):")
    for key in sorted(by_step, key=lambda k: (k == "final", k)):
        m, t = by_step[key]
        print(f"  {key}: {m}/{t} = {m/t:.1%}")


if __name__ == "__main__":
    main()
