"""
Minimal CODI-gpt2 inference script.

Place this file next to `model.py` (the one originally found in src/ of the
CODI repo). It downloads the public checkpoint from
https://huggingface.co/zen-E/CODI-gpt2 and runs it on a single test prompt.

Requirements:
    pip install torch transformers peft huggingface_hub safetensors accelerate
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import re
import sys
import torch
import json
import torch.nn.functional as F
import transformers
from huggingface_hub import hf_hub_download
from peft import LoraConfig, TaskType
from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. Import CODI from model.py (assumed to be in the same directory).
#    The original repo had this file inside `src/`; we just import it directly.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import CODI, ModelArguments, TrainingArguments  # noqa: E402

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------------------------
# 2. Build the args + LoRA config that match the released checkpoint.
#    These values come from the official scripts/test_gpt2.sh of the CODI repo
#    (also reproduced in the SIM-CoT model card, which uses the same setup).
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_bf16 = device.type == "cuda"  # bf16 on GPU, float32 on CPU
print(f"Using device: {device}")

model_args = ModelArguments(
    model_name_or_path="openai-community/gpt2",
    lora_r=128,
    lora_alpha=32,
    lora_init=True,
    full_precision=True,
    train=True,  # only triggers init() which is a no-op when restore_from=""
)

training_args = TrainingArguments(
    output_dir="./_codi_tmp",  # required by HF TrainingArguments, never used
    bf16=use_bf16,
    use_lora=True,
    num_latent=6,
    inf_latent_iterations=6,
    use_prj=True,
    prj_dim=768,             # GPT-2 hidden size
    prj_no_ln=False,
    prj_dropout=0.0,
    remove_eos=True,         # matches the released checkpoint's training
    greedy=True,
    print_loss=False,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=model_args.lora_r,
    lora_alpha=model_args.lora_alpha,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj", "c_fc"],  # GPT-2 module names
    init_lora_weights=True,
)

# ---------------------------------------------------------------------------
# 3. Build the model, download the checkpoint, load weights.
# ---------------------------------------------------------------------------
print("Building CODI model (downloads base GPT-2 if not cached)...")
model = CODI(model_args, training_args, lora_config)

print("Downloading CODI-gpt2 checkpoint from HuggingFace Hub...")
ckpt_path = hf_hub_download(repo_id="zen-E/CODI-gpt2", filename="pytorch_model.bin")
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

# ---------------------------------------------------------------------------
# 4. Tokenizer (mirrors test.py's setup).
# ---------------------------------------------------------------------------
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path, use_fast=False, padding_side="left"
)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = model.pad_token_id

# ---------------------------------------------------------------------------
# 5. Inference: encode question -> N latent thoughts -> feed <EOT> -> decode.
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(prompt: str, max_new_tokens: int = 256, num_latent_thoughts: int = 6) -> str:
    enc = tokenizer(prompt.strip(), return_tensors="pt").to(device)
    input_ids, attn = enc.input_ids, enc.attention_mask

    # Append the <BOT> sentinel token (no <EOS> first, since remove_eos=True)
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

    # --- (b) Iterate the latent thoughts in continuous space.
    for _ in range(num_latent_thoughts):
        out = model.codi(
            inputs_embeds=latent,
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
        # Trim the trailing <EOT> logit so we never sample the sentinel itself.
        logits = out.logits[:, -1, : model.codi.config.vocab_size - 1]

        if training_args.greedy:
            next_id = torch.argmax(logits, dim=-1)  # shape [1]
        else:
            probs = F.softmax(logits / 0.1, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

        token = next_id.item()
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        cur = embed(next_id).unsqueeze(1)  # shape [1, 1, hidden]

    return tokenizer.decode(generated, skip_special_tokens=True)


ds = load_dataset("openai/gsm8k", "main")

results = []

for sample_idx, sample in enumerate(ds["test"]):
    answers = []
    answer_correctness = []
    reasoning = []
    question = sample["question"]
    ground_truth_answer = sample["answer"].split("####")[1].strip().replace(",", "")

    for num_latent_thoughts in range(7):
        text = generate(question, num_latent_thoughts=num_latent_thoughts)
        nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
        final = nums[-1] if nums else "(no number found)"
        answers.append(final)

        try:
            is_correct = float(final) == float(ground_truth_answer)
        except ValueError:
            is_correct = False
        answer_correctness.append(is_correct)

        reasoning.append(("<|Latent|>"*num_latent_thoughts) + text)

    for idx, answer in enumerate(answers):
        if answer == answers[-1]:
            first_match = idx
            break

    num_steps = 6

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
print(f"\nResults saved to {"codi_gsm_results.json"}")

avg_first_match_frac = sum(r["first_match_frac"] for r in results) / len(results)
avg_stable_match_frac = sum(r["stable_match_frac"] for r in results) / len(results)
print(f"Average first_match_frac:  {avg_first_match_frac:.4f}")
print(f"Average stable_match_frac: {avg_stable_match_frac:.4f}")