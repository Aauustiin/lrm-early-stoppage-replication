import random
import argparse
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_common import (
    DATASETS, augment_question, match_fractions, slicing_status,
    aggregate_and_save, load_test_samples, split_reasoning_steps,
    FORCE_ANSWER_DELIM, step_token_counts,
)

# Public checkpoints from https://huggingface.co/connordilgren -- raw
# state_dicts, not from_pretrained-style models (same convention as
# evaluate_codi.py's zen-E/CODI-gpt2 download). All three datasets' CoT
# checkpoints share the same checkpoint filename.
CHECKPOINT_FILE = "checkpoint_25"


def checkpoint_repo(dataset):
    return f"connordilgren/gpt2-{dataset}-cot"


def early_stopping(question, ground_truth_answer, model, tokenizer, dataset):
    answers = []
    is_correct = []

    input_ids = tokenizer.encode(question + "\n", return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)

    # Generate a response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_ids = output_ids[0][input_ids.shape[1]:]
    full_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    final_answer = full_output_text.split("#")[-1].replace(",", "").strip()

    steps = split_reasoning_steps(dataset, full_output_text)
    num_steps = len(steps)

    # Force each truncated answer by slicing the model's own generated token
    # IDs directly and appending the delimiter's token IDs -- not by
    # decoding to text, rejoining a step prefix, and re-encoding a new
    # prompt string, which can retokenize differently at the truncation
    # boundary than the original generation actually used (see
    # step_token_counts's docstring).
    delim_ids = torch.tensor(
        tokenizer.encode(FORCE_ANSWER_DELIM, add_special_tokens=False),
        device=model.device,
    )
    for token_count in step_token_counts(tokenizer, gen_ids, num_steps):
        prompt_ids = torch.cat([input_ids[0], gen_ids[:token_count], delim_ids]).unsqueeze(0)
        prompt_attention_mask = torch.ones_like(prompt_ids)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = output_text.split("#")[-1].replace(",", "").strip()
        answers.append(answer)
        if ground_truth_answer is not None:
            is_correct.append(answer == ground_truth_answer)

    answers.append(final_answer)
    if ground_truth_answer is not None:
        is_correct.append(final_answer == ground_truth_answer)

    first_match, stable_match, first_match_frac, stable_match_frac = match_fractions(answers, num_steps)

    if ground_truth_answer is not None:
        results = {
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "full_output_text": full_output_text,
            "steps": steps,
            "num_steps": num_steps,
            "model_answers": answers,
            "is_correct": is_correct,
            "first_match": first_match,
            "first_match_frac": first_match_frac,
            "stable_match": stable_match,
            "stable_match_frac": stable_match_frac,
        }
    else:
        results = {
            "question": question,
            "full_output_text": full_output_text,
            "steps": steps,
            "num_steps": num_steps,
            "model_answers": answers,
            "first_match": first_match,
            "first_match_frac": first_match_frac,
            "stable_match": stable_match,
            "stable_match_frac": stable_match_frac,
        }
    return results


# ---------------------------------------------------------------------------
# Main. Mirrors evaluate_coconut.py / evaluate_codi.py:
#   * early-stopping analysis on the original question,
#   * number-augmentation (swap first positive integer for a same-magnitude
#     random one) + early-stopping analysis on the augmented question,
#   * slicing analysis: splice the i-th augmented reasoning step into the first
#     i original steps and categorise the answer as original/augmented/other/tie.
# For the latent models a "step" is a continuous thought; here it is a textual
# reasoning step (GSM8K: a <<...>> calculator annotation; ProsQA/PrOntoQA: a
# newline-delimited sentence -- see eval_common.split_reasoning_steps), so the
# number of steps is data-dependent and the slicing analysis is only run when
# the original and augmented questions yield the same number of steps.
#
# The augmentation/slicing analysis (Section 5 of the paper) is GSM8K-only in
# practice: it swaps a number in the question, and ProsQA/PrOntoQA questions
# don't contain any -- augment_question() naturally returns None for them, so
# every ProsQA/PrOntoQA sample takes the "no_number" skip path below and only
# the early-stopping / matching-metrics analysis (Section 3) actually runs.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=DATASETS, default="gsm8k",
        help="Which dataset's test split to evaluate on (default: gsm8k).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help=(
            "Path to a local GPT-2 SFT checkpoint (state_dict) to load. "
            f"Default: download connordilgren/gpt2-{{dataset}}-cot's "
            f"{CHECKPOINT_FILE} from the Hugging Face Hub."
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
        repo_id=checkpoint_repo(args.dataset), filename=CHECKPOINT_FILE
    )
    saved_weights = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved_weights, strict=False)

    # Move to GPU and set eval mode
    model = model.to(device)
    model.eval()

    samples = load_test_samples(args.dataset)

    results = []

    # Process each question in the dataset
    for sample_idx, (question, ground_truth_answer) in enumerate(samples):
        original_result = early_stopping(
            question, ground_truth_answer, model, tokenizer, args.dataset
        )

        aug_question, _ = augment_question(question)
        if aug_question is None:
            results.append({
                "sample_idx": sample_idx,
                "original_result": original_result,
                "skipped": "no_number",
            })
            continue

        augmented_result = early_stopping(
            aug_question, None, model, tokenizer, args.dataset
        )

        # Slicing analysis: for each step i, run with the first i original
        # reasoning steps + the i-th augmented step, then force the answer.
        # Only meaningful when both runs produced the same (non-zero) number of
        # steps, since the step alignment is positional.
        if (original_result["num_steps"] == augmented_result["num_steps"]
                and original_result["num_steps"] > 0):
            answer_status = []
            answers = []
            for i in range(original_result["num_steps"]):
                if i == 0:
                    prompt = question + "\n" + augmented_result["steps"][i] + "\n" + FORCE_ANSWER_DELIM
                else:
                    prompt = (
                        question + "\n"
                        + "\n".join(original_result["steps"][:i]) + "\n"
                        + augmented_result["steps"][i] + "\n" + FORCE_ANSWER_DELIM
                    )

                prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                prompt_attention_mask = torch.ones_like(prompt_ids)
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_attention_mask,
                        max_new_tokens=128,
                        pad_token_id=tokenizer.eos_token_id
                    )
                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                answer = output_text.split("#")[-1].replace(",", "").strip()
                answers.append(answer)

                answer_status.append(slicing_status(
                    answer,
                    original_result["model_answers"][i + 1],
                    augmented_result["model_answers"][i + 1],
                ))

            results.append({
                "sample_idx": sample_idx,
                "original_result": original_result,
                "augmented_result": augmented_result,
                "slicing_answers": answers,
                "slicing_answer_status": answer_status,
                "slicing_original_count": answer_status.count("original"),
                "slicing_augmented_count": answer_status.count("augmented"),
                "slicing_other_count": answer_status.count("other"),
                "slicing_tie_count": answer_status.count("tie"),
            })
        else:
            results.append({
                "sample_idx": sample_idx,
                "original_result": original_result,
                "augmented_result": augmented_result,
                "skipped": "step_count_mismatch",
            })

    model_name = args.checkpoint_path or f"{checkpoint_repo(args.dataset)}/{CHECKPOINT_FILE}"
    out_path = "results/sft.json" if args.dataset == "gsm8k" else f"results/sft_{args.dataset}.json"
    aggregate_and_save(results, model_name, args.dataset, out_path)


if __name__ == "__main__":
    main()