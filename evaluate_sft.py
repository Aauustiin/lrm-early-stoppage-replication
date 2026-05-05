import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datasets import load_dataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load model weights
    saved_weights = torch.load(
        "/users/cns542/scratch/coconut/gsm-cot/checkpoint_7",
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

        answers = []
        answer_correctness = []

        input_ids = tokenizer.encode(question, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids)

        # Generate a response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128
            )

        full_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        full_answer_output = full_output_text.split("#")[-1].replace(",", "").strip()

        steps = re.findall(r"<<.*?>>", full_output_text)
        num_steps = len(steps)
        question_suffixes = ([""] + re.findall(r"<<.*?>>", full_output_text))[:-1]

        for idx in range(len(question_suffixes)):
            prompt = question + "".join(question_suffixes[:idx+1]) + "###"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            prompt_attention_mask = torch.ones_like(prompt_ids)
            with torch.no_grad():
                step_output_ids = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attention_mask,
                    max_new_tokens=128
                )
            step_output_text = tokenizer.decode(step_output_ids[0], skip_special_tokens=True)
            answer_output = step_output_text.split("#")[-1].replace(",", "").strip()
            answers.append(answer_output)
            answer_correctness.append(answer_output == ground_truth_answer)

        answers.append(full_answer_output)
        answer_correctness.append(full_answer_output == ground_truth_answer)

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

        results.append({
            "question_idx": sample_idx,
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "model_reasoning": steps,   
            "num_steps": num_steps,
            "model_answers": answers,
            "is_correct": answer_correctness,
            "first_match": first_match,
            "first_match_frac": first_match_frac,
            "stable_match": stable_match,
            "stable_match_frac": stable_match_frac,
        })

    with open("gpt2_sft_gsm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to gpt2_sft_gsm_results.json")

    avg_first_match_frac = sum(r["first_match_frac"] for r in results) / len(results)
    avg_stable_match_frac = sum(r["stable_match_frac"] for r in results) / len(results)
    print(f"Average first_match_frac:  {avg_first_match_frac:.4f}")
    print(f"Average stable_match_frac: {avg_stable_match_frac:.4f}")

if __name__ == "__main__":
    main()