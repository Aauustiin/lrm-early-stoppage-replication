import math
import re
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datasets import load_dataset
from coconut import Coconut

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


def early_stopping(question, ground_truth_answer, model, tokenizer, device):
    answers = []
    is_correct = []

    input_ids = tokenizer.encode(question + "\n", return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    # Generate a response
    with torch.no_grad():
        output_ids, latent_hidden_states = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            num_latent_reasoning_tokens = 6,
            output_latent_hidden_states=True
        )

    full_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    final_answer = full_output_text.split("#")[-1].replace(",", "").strip()
    latent_hidden_states = torch.stack(latent_hidden_states)
    num_steps = 6

    for i in range(6):
        if i == 0:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id,
                    num_latent_reasoning_tokens = 0
                )
        else:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id,
                    num_latent_reasoning_tokens = i,
                    latent_reasoning_tokens = latent_hidden_states[:i]
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
            "latent_reasoning_tokens": latent_hidden_states,
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
            "latent_reasoning_tokens": latent_hidden_states,
            "num_steps": num_steps,
            "model_answers": answers,
            "first_match": first_match,
            "first_match_frac": first_match_frac,
            "stable_match": stable_match,
            "stable_match_frac": stable_match_frac,
        }
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(42)

    # Initialise tokeniser with special tokens
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))

    answer_prefix_ids = tokenizer.encode("### ", add_special_tokens=False)

    # Initialise model
    model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id, answer_prefix_ids=answer_prefix_ids)

    # Load model weights
    saved_weights = torch.load(
        "/users/cns542/scratch/coconut/gsm-coconut-true/checkpoint_11",
        map_location=device
    )
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
        original_result = early_stopping(question, ground_truth_answer, model, tokenizer, device)

        start, end, orig = find_first_number(question)
        if orig is None:
            # Remove latent reasoning tokens from results to save space
            del original_result["latent_reasoning_tokens"]
            results.append({"sample_idx": sample_idx, "original_result": original_result, "skipped": "no_number"})
            continue
        new_num = rand_same_magnitude(orig)
        aug_question = question[:start] + str(new_num) + question[end:]
        augmented_result = early_stopping(aug_question, None, model, tokenizer, device)

        answer_status = []
        answers = []
        for i in range(0, original_result["num_steps"]):
            input_ids = tokenizer.encode(question + "\n", return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id,
                    num_latent_reasoning_tokens = i+1,
                    latent_reasoning_tokens = torch.cat((original_result["latent_reasoning_tokens"][:i], augmented_result["latent_reasoning_tokens"][i:i+1]), dim=0)
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

        # Remove latent reasoning tokens from results to save space
        del original_result["latent_reasoning_tokens"]
        del augmented_result["latent_reasoning_tokens"]

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
        "model": "/users/cns542/scratch/coconut/gsm-coconut-true/checkpoint_11",
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

    with open("gpt2_coconut_gsm_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to gpt2_coconut_gsm_results.json")


if __name__ == "__main__":
    main()