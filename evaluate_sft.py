import math
import re
import random
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

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

def early_stopping(question, ground_truth_answer, model, tokenizer):
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

    full_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    final_answer = full_output_text.split("#")[-1].replace(",", "").strip()

    steps = re.findall(r"<<.*?>>", full_output_text)
    num_steps = len(steps)

    partial_reasoning_traces = [""]
    for step in steps[:-1]:
        partial_reasoning_traces.append(partial_reasoning_traces[-1] + step + "\n")

    for partial_reasoning_trace in partial_reasoning_traces:
        prompt = question + "\n" + partial_reasoning_trace + "#### "
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
        if ground_truth_answer is not None:
            is_correct.append(answer == ground_truth_answer)

    answers.append(final_answer)
    if ground_truth_answer is not None:
        is_correct.append(final_answer == ground_truth_answer)

    for idx, answer in enumerate(answers):
        if answer == answers[-1]:
            first_match = idx
            break

    stable_match = num_steps
    for k in range(num_steps + 1):
        if all(a == answers[-1] for a in answers[k:]):
            stable_match = k
            break

    if num_steps == 0:
        first_match_frac = 0
        stable_match_frac = 0
    else:
        first_match_frac = first_match / num_steps
        stable_match_frac = stable_match / num_steps

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
# <<...>> reasoning step, so the number of steps is data-dependent and the
# slicing analysis is only run when the original and augmented questions yield
# the same number of steps.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/users/cns542/scratch/coconut/gsm-cot/checkpoint_7",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(42)

    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load model weights
    saved_weights = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(saved_weights, strict=False)

    # Move to GPU and set eval mode
    model = model.to(device)
    model.eval()

    ds = load_dataset("openai/gsm8k", "main")

    results = []

    # Process each question in the dataset
    for sample_idx, sample in enumerate(ds["test"]):
        question = sample["question"]
        ground_truth_answer = sample["answer"].split("####")[1].strip()
        original_result = early_stopping(question, ground_truth_answer, model, tokenizer)

        start, end, orig = find_first_number(question)
        if orig is None:
            results.append({
                "sample_idx": sample_idx,
                "original_result": original_result,
                "skipped": "no_number",
            })
            continue

        new_num = rand_same_magnitude(orig)
        aug_question = question[:start] + str(new_num) + question[end:]
        augmented_result = early_stopping(aug_question, None, model, tokenizer)

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
                    prompt = question + "\n" + augmented_result["steps"][i] + "\n#### "
                else:
                    prompt = (
                        question + "\n"
                        + "\n".join(original_result["steps"][:i]) + "\n"
                        + augmented_result["steps"][i] + "\n#### "
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

                if original_result["model_answers"][i + 1] == augmented_result["model_answers"][i + 1]:
                    answer_status.append("tie")
                elif answer == original_result["model_answers"][i + 1]:
                    answer_status.append("original")
                elif answer == augmented_result["model_answers"][i + 1]:
                    answer_status.append("augmented")
                else:
                    answer_status.append("other")

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

    # Aggregate.
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
    slicing_original_proportion = slicing_original_count / slicing_total if slicing_total else 0
    slicing_augmented_proportion = slicing_augmented_count / slicing_total if slicing_total else 0
    slicing_other_proportion = slicing_other_count / slicing_total if slicing_total else 0
    slicing_tie_proportion = slicing_tie_count / slicing_total if slicing_total else 0
    accuracy = sum(r["original_result"]["is_correct"][-1] for r in results) / len(results)

    output = {
        "model": args.checkpoint_path,
        "dataset": "openai/gsm8k",
        "average_first_match_frac": average_first_match_frac,
        "average_stable_match_frac": average_stable_match_frac,
        "slicing_original_count": slicing_original_count,
        "slicing_augmented_count": slicing_augmented_count,
        "slicing_other_count": slicing_other_count,
        "slicing_tie_count": slicing_tie_count,
        "slicing_original_proportion": slicing_original_proportion,
        "slicing_augmented_proportion": slicing_augmented_proportion,
        "slicing_other_proportion": slicing_other_proportion,
        "slicing_tie_proportion": slicing_tie_proportion,
        "accuracy": accuracy,
        "results": results,
    }

    with open("gpt2_sft_gsm_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to gpt2_sft_gsm_results.json")


if __name__ == "__main__":
    main()