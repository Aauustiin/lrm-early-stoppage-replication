import re
import random
import argparse
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_common import (
    load_test_samples, split_reasoning_steps, find_replacement_word,
    FORCE_ANSWER_DELIM,
)

# Last run of word characters in a string, e.g. the "sterpus" in "a sterpus."
# or in "Sally is a sterpus" (no trailing period).
LAST_WORD_RE = re.compile(r"(\w+)[^\w]*$")

# Same ERM checkpoint family as evaluate_sft.py -- public checkpoint from
# https://huggingface.co/connordilgren/gpt2-prosqa-cot (a raw state_dict, not
# a from_pretrained-style model).
CHECKPOINT_REPO = "connordilgren/gpt2-prosqa-cot"
CHECKPOINT_FILE = "checkpoint_25"


def generate_text(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def extract_last_word(text):
    match = LAST_WORD_RE.search(text.strip())
    return match.group(1) if match else None


def replace_last_word(step_text, new_word):
    """Replace the last run of word characters in `step_text` with
    `new_word`, keeping any trailing punctuation (e.g. the "." in "Every
    yimpus is a sterpus.") intact.
    """
    match = LAST_WORD_RE.search(step_text)
    return step_text[:match.start(1)] + new_word + step_text[match.end(1):]


def run_copying_bias(question, model, tokenizer):
    """Generate the ERM's chain-of-thought for `question`, then mirror
    Section 3's early-stopping protocol: for every truncation length k (the
    CoT cut down to its first k explicit reasoning steps), replace the last
    word of the last step in that truncated trace -- steps[k-1] -- with the
    first "Every Y is a Z." word from the problem statement that isn't
    already that word, and force an answer from the perturbed, truncated
    trace.

    Returns one trial dict per truncation length whose last step has an
    extractable last word and a valid replacement word (a truncation is
    skipped for that k only if either is missing).
    """
    full_output_text = generate_text(question + "\n", model, tokenizer)
    steps = split_reasoning_steps("prosqa", full_output_text)

    trials = []
    for k in range(1, len(steps) + 1):
        last_step = steps[k - 1]
        orig_word = extract_last_word(last_step)
        if orig_word is None:
            continue

        replacement_word = find_replacement_word(question, orig_word)
        if replacement_word is None:
            continue

        modified_step = replace_last_word(last_step, replacement_word)

        prefix = "".join(step + "\n" for step in steps[:k - 1])
        prompt = question + "\n" + prefix + modified_step + "\n" + FORCE_ANSWER_DELIM
        perturbed_output_text = generate_text(prompt, model, tokenizer)
        perturbed_answer = perturbed_output_text.split("#")[-1].strip()
        perturbed_answer_last_word = extract_last_word(perturbed_answer)

        trials.append({
            "truncated_at_step": k,
            "num_steps": len(steps),
            "original_last_word": orig_word,
            "replacement_word": replacement_word,
            "perturbed_answer": perturbed_answer,
            "matches_replacement": perturbed_answer_last_word == replacement_word,
        })

    return trials


# ---------------------------------------------------------------------------
# Main. Same experimental setup as evaluate_sft.py's early_stopping (same
# model, checkpoint, and ProsQA test split, one forced answer per truncation
# length), except at each truncation length we perturb the last word of the
# truncated trace's last step rather than leaving it as generated, to see how
# often the ERM's forced answer just copies that (otherwise unsupported)
# word.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help=(
            "Path to a local GPT-2 SFT checkpoint (state_dict) to load. "
            f"Default: download {CHECKPOINT_REPO}'s {CHECKPOINT_FILE} from "
            "the Hugging Face Hub."
        ),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(42)

    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    checkpoint_path = args.checkpoint_path or hf_hub_download(
        repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_FILE
    )
    saved_weights = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved_weights, strict=False)

    model = model.to(device)
    model.eval()

    samples = load_test_samples("prosqa")

    results = []
    all_trials = []
    for sample_idx, (question, _ground_truth_answer) in enumerate(samples):
        trials = run_copying_bias(question, model, tokenizer)
        results.append({"sample_idx": sample_idx, "trials": trials})
        all_trials.extend(trials)

    match_rate = (
        sum(t["matches_replacement"] for t in all_trials) / len(all_trials)
        if all_trials else 0.0
    )

    output = {
        "model": args.checkpoint_path or f"{CHECKPOINT_REPO}/{CHECKPOINT_FILE}",
        "dataset": "prosqa",
        "match_rate": match_rate,
        "num_trials": len(all_trials),
        "num_samples": len(results),
        "results": results,
    }

    out_path = "results/copying_bias_prosqa.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nMatch rate: {match_rate:.1%} ({len(all_trials)} trials over {len(results)} samples)")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
