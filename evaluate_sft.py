import math
import re
import random
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def count_gold_steps(answer: str) -> int:
    """Count reasoning steps in a GSM8k gold answer (lines before ####)."""
    lines = [l.strip() for l in answer.strip().split("\n") if l.strip()]
    return sum(1 for l in lines if not l.startswith("####"))

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default="/users/cns542/scratch/coconut/gsm-cot/checkpoint_7")
    parser.add_argument("--experiment", type=str, default="replication")
    # Random seed
    # Output file path
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(42)

    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load model weights
    saved_weights = torch.load(
        args.checkpoint_path,
        map_location=device
    )
    model.load_state_dict(saved_weights, strict=False)

    # Move to GPU and set eval mode
    model = model.to(device)
    model.eval()

    ds = load_dataset("openai/gsm8k", "main")

    if args.experiment == "replication":
        results = []

        for sample_idx, sample in enumerate(ds["test"]):
            question = sample["question"]
            ground_truth_answer = sample["answer"].split("####")[1].strip()
            input_ids = tokenizer.encode(question + "\n", return_tensors="pt").to(model.device)
            attention_mask = torch.ones_like(input_ids)
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
    elif args.experiment == "patching":
        results = []

        for sample_idx, sample in enumerate(ds["test"]):
            orig_question = sample["question"]
            ground_truth_answer = sample["answer"].split("####")[1].strip()
            num_gold_steps = count_gold_steps(sample["answer"])

            orig_input_ids = tokenizer.encode(orig_question + "\n", return_tensors="pt").to(model.device)
            orig_attention_mask = torch.ones_like(orig_input_ids)
            with torch.no_grad():
                orig_output_ids = model.generate(
                    input_ids=orig_input_ids,
                    attention_mask=orig_attention_mask,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id
                )
            orig_output_text = tokenizer.decode(orig_output_ids[0], skip_special_tokens=True)
            orig_answer = orig_output_text.split("#")[-1].replace(",", "").strip()
            orig_steps = re.findall(r"<<.*?>>", orig_output_text)
            orig_num_steps = len(orig_steps)

            start, end, orig_num = find_first_number(orig_question)
            if orig_num is None:
                results.append({
                    "sample_idx": sample_idx,
                    "orig_question": orig_question,
                    "ground_truth_answer": ground_truth_answer,
                    "num_gold_steps": num_gold_steps,
                    "orig_answer": orig_answer,
                    "is_correct": float(orig_answer) == float(ground_truth_answer),
                    "orig_steps": orig_steps,
                    "orig_num_steps": orig_num_steps,
                    "aug_question": None,
                    "aug_answer": None,
                    "aug_steps": None,
                    "aug_num_steps": None,
                    "patching_answers": None,
                    "patching_answer_categorisations": None,
                    "error": "The question does not contain a positive integer."
                })
                continue

            aug_num = rand_same_magnitude(orig_num)
            aug_question = orig_question[:start] + str(aug_num) + orig_question[end:]

            aug_input_ids = tokenizer.encode(aug_question + "\n", return_tensors="pt").to(model.device)
            aug_attention_mask = torch.ones_like(aug_input_ids)
            with torch.no_grad():
                aug_output_ids = model.generate(
                    input_ids=aug_input_ids,
                    attention_mask=aug_attention_mask,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id
                )
            aug_output_text = tokenizer.decode(aug_output_ids[0], skip_special_tokens=True)
            aug_answer = aug_output_text.split("#")[-1].replace(",", "").strip()
            aug_steps = re.findall(r"<<.*?>>", aug_output_text)
            aug_num_steps = len(aug_steps)

            if (orig_num_steps != aug_num_steps) or (orig_answer == aug_answer):
                error = []
                if orig_num_steps != aug_num_steps:
                    error.append("The model produced a different number of reasoning steps for the original and augmented questions.")
                if orig_answer == aug_answer:
                    error.append("The model's answers to the original and augmented questions are the same.")
                
                results.append({
                    "sample_idx": sample_idx,
                    "orig_question": orig_question,
                    "ground_truth_answer": ground_truth_answer,
                    "num_gold_steps": num_gold_steps,
                    "orig_answer": orig_answer,
                    "is_correct": float(orig_answer) == float(ground_truth_answer),
                    "orig_steps": orig_steps,
                    "orig_num_steps": orig_num_steps,
                    "aug_question": aug_question,
                    "aug_answer": aug_answer,
                    "aug_steps": aug_steps,
                    "aug_num_steps": aug_num_steps,
                    "patching_answers": None,
                    "patching_answer_categorisations": None,
                    "error": ". ".join(error)
                })
                continue
            else:
                patching_answers = []
                patching_answer_categorisations = []

                for i in range(orig_num_steps):
                    prompt = orig_question + "\n" + "\n".join(orig_steps[:i]) + "\n" + aug_steps[i] + "\n" + "\n".join(aug_steps[i+1:]) + "\n#### "
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
                    patching_answers.append(answer)

                    if answer == orig_answer:
                        patching_answer_categorisations.append("orig")
                    elif answer == aug_answer:
                        patching_answer_categorisations.append("aug")
                    else:
                        patching_answer_categorisations.append("other")

                results.append({
                    "sample_idx": sample_idx,
                    "orig_question": orig_question,
                    "ground_truth_answer": ground_truth_answer,
                    "num_gold_steps": num_gold_steps,
                    "orig_answer": orig_answer,
                    "is_correct": float(orig_answer) == float(ground_truth_answer),
                    "orig_steps": orig_steps,
                    "orig_num_steps": orig_num_steps,
                    "aug_question": aug_question,
                    "aug_answer": aug_answer,
                    "aug_steps": aug_steps,
                    "aug_num_steps": aug_num_steps,
                    "patching_answers": patching_answers,
                    "patching_answer_categorisations": patching_answer_categorisations,
                    "error": None
                })

        output = {
            "accuracy": sum(r["is_correct"] for r in results) / len(results),
            "results": results,
        }


    # Process each question in the dataset
    for sample_idx, sample in enumerate(ds["test"]):
        question = sample["question"]
        ground_truth_answer = sample["answer"].split("####")[1].strip()
        original_result = early_stopping(question, ground_truth_answer, model, tokenizer)

        
        augmented_result = early_stopping(aug_question, None, model, tokenizer)

        if original_result["num_steps"] == augmented_result["num_steps"]:
            answer_status = []
            answers = []
            for i in range(0, original_result["num_steps"]):
                if i == 0:
                    prompt = question + "\n" + augmented_result["steps"][i] + "\n#### "
                else:
                    prompt = question + "\n" + "\n".join(original_result["steps"][:i]) + "\n" + augmented_result["steps"][i] + "\n#### "

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

                if original_result["model_answers"][i+1] == augmented_result["model_answers"][i+1]:
                    answer_status.append("tie")
                elif answer == original_result["model_answers"][i+1]:
                    answer_status.append("original")
                elif answer == augmented_result["model_answers"][i+1]:
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
                "slicing_tie_count": answer_status.count("tie")
            })
        else:
            results.append({
                "sample_idx": sample_idx,
                "original_result": original_result,
                "augmented_result": augmented_result,
            })


    average_first_match_frac = sum(r["original_result"]["first_match_frac"] for r in results) / len(results)
    average_stable_match_frac = sum(r["original_result"]["stable_match_frac"] for r in results) / len(results)
    slicing_original_count = sum(r.get("slicing_original_count") or 0 for r in results)
    slicing_augmented_count = sum(r.get("slicing_augmented_count") or 0 for r in results)
    slicing_other_count = sum(r.get("slicing_other_count") or 0 for r in results)
    slicing_tie_count = sum(r.get("slicing_tie_count") or 0 for r in results)
    slicing_original_proportion = slicing_original_count / (slicing_original_count + slicing_augmented_count + slicing_other_count + slicing_tie_count)
    slicing_augmented_proportion = slicing_augmented_count / (slicing_original_count + slicing_augmented_count + slicing_other_count + slicing_tie_count)
    slicing_other_proportion = slicing_other_count / (slicing_original_count + slicing_augmented_count + slicing_other_count + slicing_tie_count)
    slicing_tie_proportion = slicing_tie_count / (slicing_original_count + slicing_augmented_count + slicing_other_count + slicing_tie_count)
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