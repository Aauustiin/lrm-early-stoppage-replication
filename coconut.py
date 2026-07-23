# Adapted from https://github.com/facebookresearch/coconut, the official
# implementation of Hao et al., "Training Large Language Models to Reason in
# a Continuous Latent Space".
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.cache_utils import DynamicCache

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "latent_hidden_states"])
MAX_N_LATENT = 8

def slice_cache(cache, end_idx):
    """Return a new DynamicCache containing only the first end_idx positions of `cache`."""
    new_cache = DynamicCache()
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        # DynamicCache API
        for layer_idx in range(len(cache.key_cache)):
            k = cache.key_cache[layer_idx][:, :, :end_idx, :]
            v = cache.value_cache[layer_idx][:, :, :end_idx, :]
            new_cache.update(k, v, layer_idx)
    else:
        # Legacy tuple format — be defensive about extra fields
        for layer_idx, layer in enumerate(cache):
            k, v = layer[0], layer[1]
            new_cache.update(k[:, :, :end_idx, :], v[:, :, :end_idx, :], layer_idx)
    return new_cache

def list_to_cache(past_key_values):
    """Convert a list of (key, value) tuples to a DynamicCache object."""
    if past_key_values is None:
        return None
    cache = DynamicCache()
    for layer_idx, (key, value) in enumerate(past_key_values):
        cache.update(key, value, layer_idx)
    return cache


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        answer_prefix_ids=None,
    ):
        """
        Args:
            answer_prefix_ids: Optional sequence of token ids (e.g. those for
                "#### ") that are automatically inserted right after
                ``<end_latent>`` in ``generate()``, before any output tokens
                are produced. Set to ``None`` (default) to disable.
        """

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.answer_prefix_ids = self._normalize_prefix_ids(answer_prefix_ids)

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    @staticmethod
    def _normalize_prefix_ids(answer_prefix_ids):
        """Accept a list, 1D tensor, or 2D tensor of shape (1, n) and return a
        flat list of Python ints (or None)."""
        if answer_prefix_ids is None:
            return None
        if isinstance(answer_prefix_ids, torch.Tensor):
            if answer_prefix_ids.dim() == 2:
                assert answer_prefix_ids.shape[0] == 1, (
                    "answer_prefix_ids 2D tensor must have batch dim 1; "
                    f"got shape {tuple(answer_prefix_ids.shape)}"
                )
                answer_prefix_ids = answer_prefix_ids[0]
            assert answer_prefix_ids.dim() == 1, (
                "answer_prefix_ids tensor must be 1D or (1, n); "
                f"got shape {tuple(answer_prefix_ids.shape)}"
            )
            return answer_prefix_ids.tolist()
        return [int(t) for t in answer_prefix_ids]

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        position_ids,
        latent_reasoning_tokens=None,
        **kwargs,
    ):
        """
        Args:
            latent_reasoning_tokens: Optional tensor of shape (i, hidden_size)
                holding `i` pre-supplied continuous-thought vectors. When given,
                the first `i` latent placeholder positions in ``input_ids`` are
                filled with these vectors instead of the model-computed hidden
                states; the remaining latent positions are computed normally.
        """

        logits = []
        latent_hidden_states = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        # Number of pre-supplied latent reasoning tokens. The first `n_prefix`
        # latent positions take their continuous thought from
        # `latent_reasoning_tokens` instead of from the model-computed hidden
        # state; the remaining `max_n_latents - n_prefix` are computed normally.
        n_prefix = 0
        if latent_reasoning_tokens is not None:
            assert latent_reasoning_tokens.dim() == 2, (
                "latent_reasoning_tokens must be 2D (i, hidden_size); "
                f"got shape {tuple(latent_reasoning_tokens.shape)}"
            )
            n_prefix = latent_reasoning_tokens.shape[0]
            assert n_prefix <= max_n_latents, (
                f"latent_reasoning_tokens has {n_prefix} entries but only "
                f"{max_n_latents} latent placeholder positions are in input_ids"
            )

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=slice_cache(kv_cache, next_compute_range[0]),
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            
            kv_cache = outputs.past_key_values

            # For the first `n_prefix` passes the latent thought is the
            # user-supplied vector rather than the model-computed hidden state.
            use_prefix = pass_idx < n_prefix
            if use_prefix:
                prefix_thought = latent_reasoning_tokens[pass_idx].to(
                    dtype=inputs_embeds.dtype, device=inputs_embeds.device
                )

            for instance_idx, mask_list in enumerate(latent_lists):
                if len(mask_list) > pass_idx:
                    token_idx = mask_list[pass_idx]
                    if use_prefix:
                        thought_hidden = prefix_thought.clone()
                    else:
                        thought_hidden = hidden_states[
                            instance_idx, token_idx - 1 - hidden_states_offset, :
                        ].clone()
                    latent_hidden_states.append(thought_hidden)

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                if use_prefix:
                    # use the user-supplied latent reasoning token
                    tensor_list[batch_idx][token_idx] = prefix_thought
                else:
                    # replace it with the preceding last hidden states
                    tensor_list[batch_idx][token_idx] = hidden_states[
                        batch_idx, token_idx - 1 - hidden_states_offset, :
                    ]

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        final_past_key_values = slice_cache(kv_cache, next_compute_range[0]) if kv_cache is not None else None

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
            past_key_values=final_past_key_values,
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(
            loss=loss,
            inputs_embeds=inputs_embeds,
            logits=logits,
            latent_hidden_states=latent_hidden_states
        )

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        num_latent_reasoning_tokens=None,
        latent_reasoning_tokens=None,
        output_latent_hidden_states=False,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):
        """
        Args:
            input_ids: (1, prompt_len) prompt tokens.
                - If ``num_latent_reasoning_tokens`` is None (default),
                  ``input_ids`` is expected to already contain the
                  ``<start_latent>``, ``<latent>`` and ``<end_latent>`` tokens
                  (legacy behaviour).
                - If ``num_latent_reasoning_tokens`` is specified, ``input_ids``
                  should be the plain prompt; this method appends
                  ``<start_latent>`` + N copies of ``<latent>`` +
                  ``<end_latent>`` to it.
            attention_mask: (1, prompt_len) attention mask for ``input_ids``.
            max_new_tokens: max number of output tokens to generate after the
                latent reasoning phase.
            num_latent_reasoning_tokens: Optional int. Total number of latent
                reasoning steps the model should perform before generating the
                output.
            latent_reasoning_tokens: Optional tensor of shape ``(i, hidden_size)``
                with ``i`` pre-computed continuous-thought vectors. These are
                used as the first ``i`` latent reasoning tokens; the remaining
                ``num_latent_reasoning_tokens - i`` are generated by the model.
                Requires ``num_latent_reasoning_tokens`` to be set and
                ``i <= num_latent_reasoning_tokens``.
        """

        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        device = input_ids.device

        # Validate `latent_reasoning_tokens` if provided.
        if latent_reasoning_tokens is not None:
            assert num_latent_reasoning_tokens is not None, (
                "`latent_reasoning_tokens` requires "
                "`num_latent_reasoning_tokens` to also be specified."
            )
            assert latent_reasoning_tokens.dim() == 2, (
                "`latent_reasoning_tokens` must be 2D (i, hidden_size); "
                f"got shape {tuple(latent_reasoning_tokens.shape)}"
            )
            i_prefix = latent_reasoning_tokens.shape[0]
            assert i_prefix <= num_latent_reasoning_tokens, (
                f"`latent_reasoning_tokens` has {i_prefix} entries but "
                f"`num_latent_reasoning_tokens` is "
                f"{num_latent_reasoning_tokens}"
            )
            latent_reasoning_tokens = latent_reasoning_tokens.to(device)

        # If a desired number of latent reasoning tokens is given, build a new
        # input by appending <start_latent> + N * <latent> + <end_latent>.
        if num_latent_reasoning_tokens is not None:
            assert num_latent_reasoning_tokens >= 0, (
                "`num_latent_reasoning_tokens` must be non-negative"
            )
            latent_ids = (
                [self.start_latent_id]
                + [self.latent_token_id] * num_latent_reasoning_tokens
                + [self.end_latent_id]
            )
            latent_block = torch.tensor(
                [latent_ids], device=device, dtype=input_ids.dtype
            )
            input_ids = torch.cat([input_ids, latent_block], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones_like(latent_block, dtype=attention_mask.dtype),
                ],
                dim=1,
            )

        tokens = input_ids[0].detach().tolist()

        # ---- 1. Run latent reasoning ----
        labels = input_ids.clone()
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
            latent_reasoning_tokens=latent_reasoning_tokens,
        )

        inputs_embeds = outputs.inputs_embeds
        latent_hidden_states = outputs.latent_hidden_states

        # ---- 1b. Insert answer prefix (e.g. "#### ") after <end_latent> ----
        # These tokens are conditioned on by the model before it generates the
        # first output token, and they also appear in the returned sequence.
        if self.answer_prefix_ids:
            prefix_ids_tensor = torch.tensor(
                [self.answer_prefix_ids], device=device, dtype=input_ids.dtype
            )
            prefix_embeds = self.embedding(prefix_ids_tensor)
            inputs_embeds = torch.cat([inputs_embeds, prefix_embeds], dim=1)
            tokens.extend(self.answer_prefix_ids)

        # ---- 2. Build initial KV cache from full sequence ----
        base_outputs = self.base_causallm(
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )

        past = base_outputs.past_key_values

        # First generated token
        next_token = torch.argmax(base_outputs.logits[0, -1]).item()
        tokens.append(next_token)

        new_token_embed = self.embedding(
            torch.tensor([[next_token]], device=input_ids.device)
        )

        # ---- 3. Cached autoregressive decoding ----
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(
                inputs_embeds=new_token_embed,
                past_key_values=past,
                use_cache=True,
            )

            self.gen_forward_cnt += 1

            past = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[0, -1]).item()

            if next_token == self.eos_token_id:
                break

            tokens.append(next_token)

            new_token_embed = self.embedding(
                torch.tensor([[next_token]], device=input_ids.device)
            )

        # ---- 4. Synced GPUs (unchanged logic) ----
        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(
                    inputs_embeds=new_token_embed,
                    past_key_values=past,
                    use_cache=True,
                )

        # ---- 5. Return format (unchanged) ----
        tokens_tensor = torch.tensor(tokens, device=input_ids.device).view(1, -1)

        if output_latent_hidden_states:
            return tokens_tensor, latent_hidden_states
        elif output_embedding:
            return tokens_tensor, inputs_embeds
        else:
            return tokens_tensor