import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from coconut import Coconut
from datasets import load_dataset

num_steps = 6

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Initialise model
    model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    # Load model weights
    saved_weights = torch.load(
        "/users/cns542/scratch/coconut/gsm-coconut-true/checkpoint_11",
        map_location=device
    )
    model.load_state_dict(saved_weights, strict=False)

    # Move to GPU and set eval mode
    model = model.to(device)
    model.eval()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    ds = load_dataset("openai/gsm8k", "main")

    # Store results
    results = []

    # Process each question in the dataset
    for sample_idx, sample in enumerate(ds["test"]):
        answers = []
        answer_correctness = []
        reasoning = []
        question = sample["question"]
        ground_truth_answer = sample["answer"].split("####")[1].strip()
        for num_latent_tokens in range(7):
            # Tokenize prompt with latent tokens
            prompt = f"{question}\n<|start-latent|>" + "<|latent|>" * num_latent_tokens + "<|end-latent|>"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)

            # Generate a response
            with torch.no_grad():
                output_ids, latent_hidden_states = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    output_latent_hidden_states=True
                )   

            # Extract answer and reasoning
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            answer_output = output_text.split("#")[-1].replace(",", "").strip()
            sample_reasoning = output_text.split("#")[0].strip() if "#" in output_text else ""
            if question in sample_reasoning:
                sample_reasoning = sample_reasoning.replace(question, "").strip()
            is_correct = answer_output == ground_truth_answer

            answers.append(answer_output)
            answer_correctness.append(is_correct)
            reasoning.append(sample_reasoning)

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
            "model_reasoning": reasoning,   
            "num_steps": num_steps,
            "model_answers": answers,
            "is_correct": answer_correctness,
            "first_match": first_match,
            "first_match_frac": first_match_frac,
            "stable_match": stable_match,
            "stable_match_frac": stable_match_frac,
        })

    with open("coconut_gsm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {"coconut_gsm_results.json"}")

    avg_first_match_frac = sum(r["first_match_frac"] for r in results) / len(results)
    avg_stable_match_frac = sum(r["stable_match_frac"] for r in results) / len(results)
    print(f"Average first_match_frac:  {avg_first_match_frac:.4f}")
    print(f"Average stable_match_frac: {avg_stable_match_frac:.4f}")


if __name__ == "__main__":
    main()