"""
CODI-gpt2 evaluation on GSM8K.

Mirrors the evaluation done in evaluate_coconut.py / evaluate_sft.py:
  * For each test question, run the model with N=0..6 latent thoughts and
    record the answer at each prefix (early-stopping analysis).
  * Augment the question by replacing its first positive integer with a
    same-magnitude random integer, and run the same early-stopping
    analysis on the augmented question.
  * Slicing analysis: for each step i in 0..5, run with `i` latent
    thoughts taken from the original run and 1 latent thought (at
    position i) taken from the augmented run, and tag the resulting
    answer as 'original', 'augmented', 'other' or 'tie'.
  * Aggregate accuracy, average first/stable match fractions, and
    slicing counts/proportions, and dump to JSON.

Place this file next to `codi.py` (the one originally found in src/ of the
CODI repo). It downloads the public checkpoint from
https://huggingface.co/zen-E/CODI-gpt2 and runs it on GSM8K's test split.

Requirements:
    pip install torch transformers peft huggingface_hub safetensors accelerate datasets
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import re
import sys
import random
import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import hf_hub_download
from peft import LoraConfig, TaskType
from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. Import CODI from codi.py (assumed to be in the same directory).
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codi import CODI, ModelArguments, TrainingArguments  # noqa: E402
from eval_common import augment_question, match_fractions, slicing_status, aggregate_and_save  # noqa: E402

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_bf16 = device.type == "cuda"  # bf16 on GPU, float32 on CPU
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# 2. Answer extraction. CODI emits the answer directly after <EOT> (no
#    "#### " prefix as in the coconut / SFT training data), so we pull the
#    last number out of the decoded text. We compare answers via float
#    equality with a string fallback to absorb "42" vs "42.0" type mismatches
#    that the regex can produce.
# ---------------------------------------------------------------------------
def extract_answer(text):
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else "(no number found)"


def answers_equal(a, b):
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return a == b


# ---------------------------------------------------------------------------
# 3. Generation: encode question -> N latent thoughts -> feed <EOT> -> decode.
#    The N latent thoughts that are actually fed into the model as
#    `inputs_embeds` are returned as a (N, hidden_size) tensor; these are the
#    analogue of coconut's `latent_hidden_states` and are what gets sliced and
#    replayed in the augmentation analysis.
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model, tokenizer, training_args, prompt,
             max_new_tokens=256,
             num_latent_thoughts=6,
             latent_reasoning_tokens=None):
    """
    Args:
        prompt: question text.
        num_latent_thoughts: total number of latent-thought iterations.
        latent_reasoning_tokens: optional tensor of shape (k, hidden_size)
            holding `k` pre-supplied continuous-thought vectors. When given,
            the first `k` iterations feed in these vectors instead of the
            model-computed latent; the remaining ``num_latent_thoughts - k``
            iterations feed in the model-computed latent as usual.
            Requires ``k <= num_latent_thoughts``.

    Returns:
        output_text: decoded answer string (special tokens removed).
        latents_fed: tensor of shape (num_latent_thoughts, hidden_size) with
            the latent vectors that were fed in at each iteration.
    """
    enc = tokenizer(prompt.strip(), return_tensors="pt").to(device)
    input_ids, attn = enc.input_ids, enc.attention_mask

    # Append the <BOT> sentinel token (no <EOS> first, since remove_eos=True).
    bot = torch.full((1, 1), model.bot_id, dtype=torch.long, device=device)
    input_ids = torch.cat([input_ids, bot], dim=1)
    attn = torch.cat([attn, torch.ones_like(bot)], dim=1)

    # --- (a) Encode the question, take the last hidden state as the 1st latent.
    out = model.codi(
        input_ids=input_ids,
        attention_mask=attn,
        use_cache=True,
        output_hidden_states=True,
    )
    pkv = out.past_key_values
    latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
    if training_args.use_prj:
        latent = model.prj(latent)

    # Validate any pre-supplied latents.
    n_prefix = 0
    if latent_reasoning_tokens is not None:
        assert latent_reasoning_tokens.dim() == 2, (
            "latent_reasoning_tokens must be 2D (k, hidden_size); "
            f"got shape {tuple(latent_reasoning_tokens.shape)}"
        )
        n_prefix = latent_reasoning_tokens.shape[0]
        assert n_prefix <= num_latent_thoughts, (
            f"latent_reasoning_tokens has {n_prefix} entries but "
            f"num_latent_thoughts is {num_latent_thoughts}"
        )

    # --- (b) Iterate the latent thoughts in continuous space, optionally
    #         overriding the first n_prefix of them with pre-supplied vectors.
    latents_fed = []
    for j in range(num_latent_thoughts):
        if j < n_prefix:
            latent_to_feed = latent_reasoning_tokens[j].to(
                device=device, dtype=latent.dtype
            ).view(1, 1, -1)
        else:
            latent_to_feed = latent

        latents_fed.append(latent_to_feed.squeeze(0).squeeze(0).detach().clone())

        out = model.codi(
            inputs_embeds=latent_to_feed,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=pkv,
        )
        pkv = out.past_key_values
        latent = out.hidden_states[-1][:, -1, :].unsqueeze(1)
        if training_args.use_prj:
            latent = model.prj(latent)

    # --- (c) Feed the <EOT> sentinel and autoregressively decode the answer.
    embed = model.get_embd(model.codi, model.model_name)
    eot_id = torch.tensor([model.eot_id], dtype=torch.long, device=device)
    cur = embed(eot_id).unsqueeze(0)  # shape [1, 1, hidden]

    generated = []
    for _ in range(max_new_tokens):
        out = model.codi(inputs_embeds=cur, use_cache=True, past_key_values=pkv)
        pkv = out.past_key_values
        # Trim the trailing <EOT>/sentinel logits so we never sample them.
        logits = out.logits[:, -1, : model.codi.config.vocab_size - 1]

        if training_args.greedy:
            next_id = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits / 0.1, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

        token = next_id.item()
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        cur = embed(next_id).unsqueeze(1)

    output_text = tokenizer.decode(generated, skip_special_tokens=True)

    if latents_fed:
        latents_tensor = torch.stack(latents_fed, dim=0)
    else:
        latents_tensor = torch.empty(
            0, model.dim, device=device, dtype=latent.dtype
        )

    return output_text, latents_tensor


# ---------------------------------------------------------------------------
# 4. Early-stopping analysis. Run the model with the full latent budget once
#    (capturing the latents), then re-run with progressively more pre-supplied
#    latents from that full run. Mirrors evaluate_coconut.py's early_stopping.
# ---------------------------------------------------------------------------
def early_stopping(question, ground_truth_answer, model, tokenizer,
                   training_args, num_steps=6):
    answers = []
    is_correct = []

    # Full run: capture latents.
    full_output_text, latent_hidden_states = generate(
        model, tokenizer, training_args, question,
        num_latent_thoughts=num_steps,
    )
    final_answer = extract_answer(full_output_text)

    # Replays with 0, 1, ..., num_steps-1 latents pre-supplied from the full run.
    for i in range(num_steps):
        if i == 0:
            output_text, _ = generate(
                model, tokenizer, training_args, question,
                num_latent_thoughts=0,
            )
        else:
            output_text, _ = generate(
                model, tokenizer, training_args, question,
                num_latent_thoughts=i,
                latent_reasoning_tokens=latent_hidden_states[:i],
            )
        answer = extract_answer(output_text)
        answers.append(answer)
        if ground_truth_answer is not None:
            is_correct.append(answers_equal(answer, ground_truth_answer))

    answers.append(final_answer)
    if ground_truth_answer is not None:
        is_correct.append(answers_equal(final_answer, ground_truth_answer))

    first_match, stable_match, first_match_frac, stable_match_frac = match_fractions(
        answers, num_steps, eq=answers_equal
    )

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
    if ground_truth_answer is not None:
        # Insert ground-truth fields right after `question` (cosmetic; matches
        # the field order used in evaluate_coconut.py / evaluate_sft.py).
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
    return results


# ---------------------------------------------------------------------------
# 5. Main.
# ---------------------------------------------------------------------------
def main():
    random.seed(42)

    # Args + LoRA config (match the released checkpoint, scripts/test_gpt2.sh).
    model_args = ModelArguments(
        model_name_or_path="openai-community/gpt2",
        lora_r=128,
        lora_alpha=32,
        lora_init=True,
        full_precision=True,
        train=True,  # only triggers init(), a no-op when restore_from=""
    )

    training_args = TrainingArguments(
        output_dir="./_codi_tmp",
        bf16=use_bf16,
        use_lora=True,
        num_latent=6,
        inf_latent_iterations=6,
        use_prj=True,
        prj_dim=768,             # GPT-2 hidden size
        prj_no_ln=False,
        prj_dropout=0.0,
        remove_eos=True,
        greedy=True,
        print_loss=False,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"],
        init_lora_weights=True,
    )

    # Build the model, download the checkpoint, load weights.
    print("Building CODI model (downloads base GPT-2 if not cached)...")
    model = CODI(model_args, training_args, lora_config)

    print("Downloading CODI-gpt2 checkpoint from HuggingFace Hub...")
    ckpt_path = hf_hub_download(
        repo_id="zen-E/CODI-gpt2", filename="pytorch_model.bin"
    )
    print(f"Checkpoint at: {ckpt_path}")

    print("Loading weights...")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"  ({len(unexpected)} unexpected keys, ignored)")
    model.codi.tie_weights()

    model.to(device)
    if use_bf16:
        model.to(torch.bfloat16)
    model.eval()

    # Tokenizer.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=False, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id

    # Dataset.
    ds = load_dataset("openai/gsm8k", "main")

    results = []
    num_steps_default = 6

    for sample_idx, sample in enumerate(ds["test"]):
        question = sample["question"]
        ground_truth_answer = sample["answer"].split("####")[1].strip().replace(",", "")

        original_result = early_stopping(
            question, ground_truth_answer, model, tokenizer, training_args,
            num_steps=num_steps_default,
        )

        aug_question, _ = augment_question(question)
        if aug_question is None:
            # No number to swap: record the original result and move on.
            del original_result["latent_reasoning_tokens"]
            results.append({
                "sample_idx": sample_idx,
                "original_result": original_result,
                "skipped": "no_number",
            })
            continue

        augmented_result = early_stopping(
            aug_question, None, model, tokenizer, training_args,
            num_steps=num_steps_default,
        )

        # Slicing analysis: for each i in 0..num_steps-1, run with the first i
        # latents from the original + the i-th latent from the augmented.
        answer_status = []
        slicing_answers = []
        for i in range(original_result["num_steps"]):
            mixed_latents = torch.cat(
                (
                    original_result["latent_reasoning_tokens"][:i],
                    augmented_result["latent_reasoning_tokens"][i:i + 1],
                ),
                dim=0,
            )
            output_text, _ = generate(
                model, tokenizer, training_args, question,
                num_latent_thoughts=i + 1,
                latent_reasoning_tokens=mixed_latents,
            )
            answer = extract_answer(output_text)
            slicing_answers.append(answer)

            answer_status.append(slicing_status(
                answer,
                original_result["model_answers"][i + 1],
                augmented_result["model_answers"][i + 1],
                eq=answers_equal,
            ))

        # Drop the latent tensors before serialising.
        del original_result["latent_reasoning_tokens"]
        del augmented_result["latent_reasoning_tokens"]

        results.append({
            "sample_idx": sample_idx,
            "original_result": original_result,
            "augmented_result": augmented_result,
            "slicing_answers": slicing_answers,
            "slicing_answer_status": answer_status,
            "slicing_original_count": answer_status.count("original"),
            "slicing_augmented_count": answer_status.count("augmented"),
            "slicing_other_count": answer_status.count("other"),
            "slicing_tie_count": answer_status.count("tie"),
        })

    aggregate_and_save(results, "zen-E/CODI-gpt2", "openai/gsm8k", "results/codi.json")


if __name__ == "__main__":
    main()