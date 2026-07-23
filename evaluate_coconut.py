import argparse
import random
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut

from eval_common import (
    DATASETS, augment_question, match_fractions, slicing_status,
    aggregate_and_save, load_test_samples,
)


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

    first_match, stable_match, first_match_frac, stable_match_frac = match_fractions(answers, num_steps)

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


# Public checkpoints from https://huggingface.co/connordilgren -- raw
# state_dicts, not from_pretrained-style models (same convention as
# evaluate_codi.py's zen-E/CODI-gpt2 download). Unlike the CoT checkpoints,
# each dataset's CoCoNuT checkpoint has its own filename (a different
# training-epoch count each converged at).
CHECKPOINT_REPO = {
    "gsm8k": "connordilgren/gpt2-gsm8k-coconut",
    "prosqa": "connordilgren/gpt2-prosqa-coconut",
    "prontoqa": "connordilgren/gpt2-prontoqa-coconut",
}
CHECKPOINT_FILE = {
    "gsm8k": "checkpoint_33",
    "prosqa": "checkpoint_40",
    "prontoqa": "checkpoint_36",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=DATASETS, default="gsm8k",
        help="Which dataset's test split to evaluate on (default: gsm8k).",
    )
    parser.add_argument(
        "--no-force-answer",
        action="store_true",
        help=(
            "Don't insert '### ' right after <|end-latent|>. By default it is "
            "forced in so the model must answer immediately instead of "
            "continuing on with explicit CoT text; pass this flag to let the "
            "model decide on its own instead (for comparison)."
        ),
    )
    args = parser.parse_args()

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

    answer_prefix_ids = (
        None if args.no_force_answer
        else tokenizer.encode("### ", add_special_tokens=False)
    )

    # Initialise model
    model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id, answer_prefix_ids=answer_prefix_ids)

    # Download and load model weights
    checkpoint_path = hf_hub_download(
        repo_id=CHECKPOINT_REPO[args.dataset], filename=CHECKPOINT_FILE[args.dataset]
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
        original_result = early_stopping(question, ground_truth_answer, model, tokenizer, device)

        aug_question, _ = augment_question(question)
        if aug_question is None:
            # Remove latent reasoning tokens from results to save space
            del original_result["latent_reasoning_tokens"]
            results.append({"sample_idx": sample_idx, "original_result": original_result, "skipped": "no_number"})
            continue
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

            answer_status.append(slicing_status(
                answer,
                original_result["model_answers"][i + 1],
                augmented_result["model_answers"][i + 1],
            ))

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

    out_name = "coconut"
    if args.dataset != "gsm8k":
        out_name += f"_{args.dataset}"
    if args.no_force_answer:
        out_name += "_no_force_answer"
    model_name = f"{CHECKPOINT_REPO[args.dataset]}/{CHECKPOINT_FILE[args.dataset]}"
    aggregate_and_save(results, model_name, args.dataset, f"results/{out_name}.json")


if __name__ == "__main__":
    main()