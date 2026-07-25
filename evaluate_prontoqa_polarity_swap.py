"""Causal test of the "fixed" shallow heuristic proposed for the ERM (SFT
model) on PrOntoQA: does the model's forced-early-stopping answer flip when
the *polarity* of the last explicit reasoning step is toggled ("X is Y."
<-> "X is not Y."), holding everything else -- including the predicate word
-- fixed?

analyze_prontoqa_heuristic.py already showed the ERM's forced final answer
is well predicted (99.6%, 797/800) by "True iff the CoT's last step and the
query match in both predicate word and polarity, else False" -- but that
was purely observational. This script performs the causal intervention the
observational test can't: take each sample's already-computed CoT (from
results/sft_prontoqa.json, produced by evaluate_sft.py --dataset prontoqa),
flip only the "not" in its last step, and force a fresh answer from the
resulting trace. If the heuristic actually drives the model's answer (not
just correlates with it), the forced answer should flip in lockstep with
the polarity edit.

Requires results/sft_prontoqa.json to already exist.
"""
import re
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_common import FORCE_ANSWER_DELIM

# Same public-checkpoint convention as evaluate_sft.py.
CHECKPOINT_REPO = "connordilgren/gpt2-prontoqa-cot"
CHECKPOINT_FILE = "checkpoint_25"

# PrOntoQA's final explicit reasoning step (and its trailing query clause)
# is always exactly "<Name> is [not] <predicate>." -- confirmed against all
# 800 test samples' actual final steps, with no exceptions.
POLARITY_RE = re.compile(r"^(\w+ is)( not)? (\w+\.)$")


def toggle_polarity(step_text):
    """Flip a "<Name> is [not] <predicate>." step's polarity, keeping the
    subject and predicate word exactly as they were. Returns None if
    step_text doesn't match the expected shape.
    """
    m = POLARITY_RE.match(step_text)
    if m is None:
        return None
    subject_is, has_not, predicate = m.groups()
    if has_not:
        return f"{subject_is} {predicate}"
    return f"{subject_is} not {predicate}"


def force_answer(question, steps, model, tokenizer):
    prompt = question + "\n" + "\n".join(steps) + "\n" + FORCE_ANSWER_DELIM
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(prompt_ids)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text.split("#")[-1].replace(",", "").strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    checkpoint_path = hf_hub_download(repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_FILE)
    saved_weights = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved_weights, strict=False)
    model = model.to(device)
    model.eval()

    with open("results/sft_prontoqa.json") as f:
        data = json.load(f)

    results = []
    for r in data["results"]:
        orr = r["original_result"]
        question = orr["question"]
        steps = orr["steps"]
        original_answer = orr["model_answers"][-1]

        if not steps:
            results.append({"sample_idx": r["sample_idx"], "skipped": "no_steps"})
            continue

        flipped_last_step = toggle_polarity(steps[-1])
        if flipped_last_step is None:
            results.append({"sample_idx": r["sample_idx"], "skipped": "unparseable_last_step"})
            continue

        swapped_steps = steps[:-1] + [flipped_last_step]
        swapped_answer = force_answer(question, swapped_steps, model, tokenizer)

        predicted_answer = {"True": "False", "False": "True"}.get(original_answer)

        results.append({
            "sample_idx": r["sample_idx"],
            "question": question,
            "original_last_step": steps[-1],
            "flipped_last_step": flipped_last_step,
            "original_answer": original_answer,
            "predicted_answer": predicted_answer,
            "swapped_answer": swapped_answer,
            "matches_prediction": swapped_answer == predicted_answer,
        })

    n_skipped = sum(1 for r in results if "skipped" in r)
    scored = [r for r in results if r.get("predicted_answer") is not None]
    n_matches = sum(1 for r in scored if r["matches_prediction"])

    summary = {
        "n_total": len(results),
        "n_skipped": n_skipped,
        "n_scored": len(scored),
        "n_matches_prediction": n_matches,
        "match_rate": (n_matches / len(scored)) if scored else 0,
        "results": results,
    }

    out_path = "results/prontoqa_polarity_swap.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")
    if scored:
        print(f"Polarity flip matched heuristic prediction: {n_matches}/{len(scored)} = {n_matches/len(scored):.1%}")


if __name__ == "__main__":
    main()
