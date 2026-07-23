import re
import random
import argparse
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from eval_common import rand_same_magnitude

# Matches the numeric result at the end of a "<<expr=result>>" calculator step.
STEP_RESULT_RE = re.compile(r"=(-?\d[\d,]*)>>$")

# Same ERM checkpoint as evaluate_sft.py -- public checkpoint from
# https://huggingface.co/connordilgren/gpt2-gsm8k-cot (a raw state_dict, not
# a from_pretrained-style model).
CHECKPOINT_REPO = "connordilgren/gpt2-gsm8k-cot"
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


def run_copying_bias(question, model, tokenizer):
    """Generate the ERM's chain-of-thought for `question`, then mirror
    Section 3's early-stopping protocol: for every truncation length k (the
    CoT cut down to its first k explicit reasoning steps), replace the
    result of the last step in that truncated trace -- steps[k-1] -- with a
    random same-magnitude number, and force an answer from the perturbed,
    truncated trace.

    Returns one trial dict per truncation length whose last step has a
    plain positive-integer result to swap (a truncation whose last step's
    result isn't a plain positive integer is skipped for that k only).
    """
    full_output_text = generate_text(question + "\n", model, tokenizer)
    steps = re.findall(r"<<.*?>>", full_output_text)

    trials = []
    for k in range(1, len(steps) + 1):
        last_step = steps[k - 1]
        match = STEP_RESULT_RE.search(last_step)
        if match is None:
            continue
        try:
            orig_result = int(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if orig_result <= 0:
            continue

        random_number = rand_same_magnitude(orig_result)
        modified_step = (
            last_step[:match.start(1)] + str(random_number) + last_step[match.end(1):]
        )

        prefix = "".join(step + "\n" for step in steps[:k - 1])
        prompt = question + "\n" + prefix + modified_step + "\n#### "
        perturbed_output_text = generate_text(prompt, model, tokenizer)
        perturbed_answer = perturbed_output_text.split("#")[-1].replace(",", "").strip()

        trials.append({
            "truncated_at_step": k,
            "num_steps": len(steps),
            "original_last_step_result": orig_result,
            "random_number": random_number,
            "perturbed_answer": perturbed_answer,
            "matches_random": perturbed_answer == str(random_number),
        })

    return trials


# ---------------------------------------------------------------------------
# Main. Same experimental setup as evaluate_sft.py's early_stopping (same
# model, checkpoint, and GSM8k test split, one forced answer per truncation
# length), except at each truncation length we perturb the numeric result of
# the truncated trace's last step rather than leaving it as generated, to see
# how often the ERM's forced answer just copies that (unrelated) number.
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

    # Download (unless a local override was given) and load model weights
    checkpoint_path = args.checkpoint_path or hf_hub_download(
        repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_FILE
    )
    saved_weights = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved_weights, strict=False)

    model = model.to(device)
    model.eval()

    ds = load_dataset("openai/gsm8k", "main")

    results = []
    all_trials = []
    for sample_idx, sample in enumerate(ds["test"]):
        question = sample["question"]
        trials = run_copying_bias(question, model, tokenizer)
        results.append({"sample_idx": sample_idx, "trials": trials})
        all_trials.extend(trials)

    match_rate = (
        sum(t["matches_random"] for t in all_trials) / len(all_trials)
        if all_trials else 0.0
    )

    output = {
        "model": args.checkpoint_path or f"{CHECKPOINT_REPO}/{CHECKPOINT_FILE}",
        "dataset": "openai/gsm8k",
        "match_rate": match_rate,
        "num_trials": len(all_trials),
        "num_samples": len(results),
        "results": results,
    }

    out_path = "results/copying_bias.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nMatch rate: {match_rate:.1%} ({len(all_trials)} trials over {len(results)} samples)")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
