# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Auto Config class."""

import importlib
import os
import re
import warnings
from collections import OrderedDict
from typing import List, Union

import dataclasses
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

from fish_speech.tokenizer import SEMANTIC_TOKENS, FishTokenizer
from fish_speech.utils import RankedLogger

from .lora import LoraConfig, setup_lora

log = RankedLogger(__name__, rank_zero_only=True)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class BaseModelArgs:
    model_type: str = "base"

    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_qkv_bias: bool = False

    # Codebook configs
    codebook_size: int = 160
    num_codebooks: int = 4

    # Gradient checkpointing
    use_gradient_checkpointing: bool = True

    # Initialize the model
    initializer_range: float = 0.02

    # Dummy vars
    is_reward_model: bool = False
    share_codebook_embeddings: bool = True
    scale_codebook_embeddings: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)

        if path.is_dir():
            path = path / "config.json"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        match data["model_type"]:
            case "naive":
                cls = NaiveModelArgs
            case "dual_ar":
                cls = DualARModelArgs
            case _:
                raise ValueError(f"Unknown model type: {data['model_type']}")

        return cls(**data)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, ensure_ascii=False)


@dataclass
class NaiveModelArgs(BaseModelArgs):
    model_type: str = "naive"


@dataclass
class DualARModelArgs(BaseModelArgs):
    model_type: str = "dual_ar"
    n_fast_layer: int = 4
    fast_dim: int | None = None
    fast_n_head: int | None = None
    fast_n_local_heads: int | None = None
    fast_head_dim: int | None = None
    fast_intermediate_size: int | None = None
    fast_attention_qkv_bias: bool | None = None

    def __post_init__(self):
        super().__post_init__()

        self.fast_dim = self.fast_dim or self.dim
        self.fast_n_head = self.fast_n_head or self.n_head
        self.fast_n_local_heads = self.fast_n_local_heads or self.n_local_heads
        self.fast_head_dim = self.fast_head_dim or self.head_dim
        self.fast_intermediate_size = (
            self.fast_intermediate_size or self.intermediate_size
        )
        self.fast_attention_qkv_bias = (
            self.fast_attention_qkv_bias
            if self.fast_attention_qkv_bias is not None
            else self.attention_qkv_bias
        )


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_len, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


@dataclass
class TransformerForwardResult:
    token_logits: Tensor
    codebook_logits: Tensor


@dataclass
class BaseTransformerForwardResult:
    logits: Tensor
    hidden_states: Tensor


class BaseTransformer(nn.Module):
    def __init__(
        self,
        config: BaseModelArgs,
        tokenizer: FishTokenizer | AutoTokenizer,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.semantic_token_ids = [
            tokenizer.get_token_id(SEMANTIC_TOKEN) for SEMANTIC_TOKEN in SEMANTIC_TOKENS
        ]

        # Slow transformer
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.dim,
        )
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks,
            config.dim,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config, use_sdpa=True) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if self.config.tie_word_embeddings is False:
            self.output = nn.Linear(
                config.dim,
                config.vocab_size,
                bias=False,
            )

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.max_seq_len,
                config.dim // config.n_head,
                config.rope_base,
            ),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.max_seq_len,
                    config.max_seq_len,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )

        # For kv cache
        self.max_batch_size = -1
        self.max_seq_len = -1

        if init_weights:
            self.apply(self._init_weights)

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        if self.max_seq_len >= max_seq_len and self.max_batch_size >= max_batch_size:
            return

        head_dim = self.config.dim // self.config.n_head
        max_seq_len = find_multiple(max_seq_len, 8)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_len,
                self.config.n_local_heads,
                head_dim,
                dtype=dtype,
            )

    def embed(self, x: Tensor) -> Tensor:
        vocab_embeds = [self.embeddings(x[:, 0])]
        for i in range(self.config.num_codebooks):
            emb = self.codebook_embeddings(x[:, i + 1] + i * self.config.codebook_size)
            semantic_token_ids_tensor = torch.tensor(
                self.semantic_token_ids, device=x.device
            )
            emb[~torch.isin(x[:, 0], semantic_token_ids_tensor)] = 0

        x = torch.stack(vocab_embeds, dim=3)
        x = x.sum(dim=3)

        return x

    def forward(
        self,
        inp: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> BaseTransformerForwardResult:
        seq_len = inp.size(2)

        # Here we want to merge the embeddings of the codebooks
        x = self.embed(inp)

        freqs_cis = self.freqs_cis[:seq_len]

        # Not that the causal mask here follows the definition of scaled_dot_product_attention
        # That is, FALSE means masked out
        # To maintain consistency, key_padding_mask use TRUE to mask out
        mask = None
        if key_padding_mask is not None:
            mask = self.causal_mask[None, None, :seq_len, :seq_len]  # (B, N, Q, K)
            mask = mask & key_padding_mask[:, None, None, :].logical_not()

        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, freqs_cis, mask, use_reentrant=True)
            else:
                x = layer(x, freqs_cis, mask)

        # We got slow_out here
        slow_out = self.norm(x)

        if self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings.weight)
        else:
            token_logits = self.output(slow_out)

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=x,
        )

    def forward_generate(
        self,
        inp: Tensor,
        input_pos: Optional[Tensor] = None,
        vq_masks: Optional[Tensor] = None,  # this is not used in fact
        return_all: bool = False,
    ) -> BaseTransformerForwardResult:
        # This is used for generation, optimized for torch compile
        # assert (
        #     self.max_seq_len != -1 and self.max_batch_size != -1
        # ), "Please call setup_caches before forward_generate"

        embeds = []
        for i in range(self.config.num_codebooks):
            if self.config.share_codebook_embeddings:
                _tokens = inp[:, i + 1] + i * self.config.codebook_size
            else:
                _tokens = inp[:, i + 1]

            emb = self.codebook_embeddings(_tokens)
            embeds.append(emb)

        vq_embeds_sum = torch.stack(embeds, dim=1).sum(dim=1)
        # if self.config.use_codebook_mlp:
        #     vq_embeds_sum = vq_embeds_sum / self.config.num_codebooks
        #     vq_embeds_sum = self.codebook_mlp(vq_embeds_sum)

        vq_masks = (inp[:, 0] >= self.tokenizer.semantic_begin_id) & (
            inp[:, 0] <= self.tokenizer.semantic_end_id
        )

        vq_embeds_sum[~vq_masks] = 0
        x = self.embeddings(inp[:, 0]) + vq_embeds_sum

        if input_pos is None:
            input_pos = torch.arange(inp.shape[-1], device=x.device)
            max_seq_len = inp.shape[-1]
        else:
            max_seq_len = self.max_seq_len

        mask = self.causal_mask[None, None, input_pos, :max_seq_len]  # (B, N, Q, K)
        freqs_cis = self.freqs_cis[input_pos]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask, input_pos=input_pos)

        # If prefill, we only calculate the logits of last token
        if x.size(1) > 1 and not return_all:
            x = x[:, -1:]

        # We got slow_out here
        slow_out = self.norm(x)

        if self.config.is_reward_model:
            token_logits = self.score_output(slow_out)
        elif self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings.weight)
        else:
            token_logits = self.output(slow_out)

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=x,
        )

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @staticmethod
    def from_pretrained(
        path: str,
        load_weights: bool = False,
        max_length: int | None = None,
        lora_config: LoraConfig | None = None,
        rope_base: int | None = None,
        is_agent: bool = False,
    ) -> "BaseTransformer":
        config = BaseModelArgs.from_pretrained(str(path))
        if max_length is not None:
            config.max_seq_len = max_length
            log.info(f"Override max_seq_len to {max_length}")

        if rope_base is not None:
            config.rope_base = rope_base
            log.info(f"Override rope_base to {rope_base}")

        match config.model_type:
            case "naive":
                model_cls = NaiveTransformer
            case "dual_ar":
                model_cls = DualARTransformer
            case _:
                raise ValueError(f"Unknown model type: {config.model_type}")

        if is_agent:
            tokenizer = AutoTokenizer.from_pretrained(str(path))
        else:
            tokenizer_path = str(path) + "/tokenizer.tiktoken"
            tokenizer = FishTokenizer(tokenizer_path)

        log.info(f"Loading model from {path}, config: {config}")
        model = model_cls(config, tokenizer=tokenizer)

        if lora_config is not None:
            setup_lora(model, lora_config)
            log.info(f"LoRA setup: {lora_config}")

        if load_weights is False:
            log.info("Randomly initialized model")
        else:

            if "int8" in str(Path(path)):
                logger.info("Using int8 weight-only quantization!")
                from tools.llama.quantize import WeightOnlyInt8QuantHandler

                simple_quantizer = WeightOnlyInt8QuantHandler(model)
                model = simple_quantizer.convert_for_runtime()

            if "int4" in str(Path(path)):
                logger.info("Using int4 quantization!")
                path_comps = path.name.split("-")
                assert path_comps[-2].startswith("g")
                groupsize = int(path_comps[-2][1:])
                from tools.llama.quantize import WeightOnlyInt4QuantHandler

                simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
                model = simple_quantizer.convert_for_runtime()

            weights = torch.load(
                Path(path) / "model.pth",
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )

            if "state_dict" in weights:
                logger.warning(
                    "Using a TextToSemantic LightningModule checkpoint, "
                    "please make sure it is a full model, not a LoRA model."
                )
                weights = weights["state_dict"]

            if next(iter(weights.keys())).startswith("model."):
                logger.info(
                    f"Remove prefix 'model.' created by TextToSemantic LightningModule from keys"
                )
                new_weights = OrderedDict()
                for k, v in weights.items():
                    new_weights[k.replace("model.", "")] = v
                weights = new_weights

            # Verify the name and shape of parameters since strict=False in load_state_dict.
            for k, v in model.named_parameters():
                if k not in weights:
                    logger.warning(f"No weight for {k}")
                elif v.shape != weights[k].shape:
                    logger.warning(
                        f"Shape mismatch for {k}: {v.shape} vs {weights[k].shape}"
                    )

            err = model.load_state_dict(weights, strict=False, assign=True)
            log.info(f"Loaded weights with error: {err}")

        return model

    def save_pretrained(self, path: str, drop_lora: bool = False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.config.save(path / "config.json")
        state_dict = self.state_dict()

        if drop_lora:
            for key in list(state_dict.keys()):
                if "lora" not in key:
                    continue

                state_dict.pop(key)
                log.info(f"Drop LoRA parameter: {key}")

        torch.save(state_dict, path / "model.pth")
        self.tokenizer.save_pretrained(path)


class NaiveTransformer(BaseTransformer):
    def __init__(self, config: NaiveModelArgs, tokenizer: FishTokenizer) -> None:
        super().__init__(config, init_weights=False, tokenizer=tokenizer)

        self.codebook_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.codebook_output = nn.Linear(
            config.dim,
            config.codebook_size * config.num_codebooks,
            bias=False,
        )

        self.apply(self._init_weights)

    def decode(self, result: BaseTransformerForwardResult) -> TransformerForwardResult:
        token_logits = result.logits
        x = result.hidden_states

        # Codebook
        codebook_logits = self.codebook_output(self.codebook_norm(x))
        codebook_logits = rearrange(
            codebook_logits, "b n (c d) -> b n c d", c=self.config.num_codebooks
        )

        return TransformerForwardResult(
            token_logits=token_logits,
            codebook_logits=codebook_logits,
        )

    def forward(
        self,
        inp: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        result = super().forward(
            inp=inp,
            key_padding_mask=key_padding_mask,
        )
        return self.decode(result)

    def forward_generate(
        self, x: Tensor, input_pos: Optional[Tensor] = None
    ) -> TransformerForwardResult:
        result = super().forward_generate(x, input_pos)
        return self.decode(result)


class DualARTransformer(BaseTransformer):
    def __init__(self, config: NaiveModelArgs, tokenizer: FishTokenizer) -> None:
        super().__init__(config, init_weights=False, tokenizer=tokenizer)

        # Project to fast dim if needed
        if config.fast_dim is not None and config.fast_dim != config.dim:
            self.fast_project_in = nn.Linear(config.dim, config.fast_dim)
        else:
            self.fast_project_in = nn.Identity()

        # Fast transformer
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)

        # The equivalent bs is so large that sdpa doesn't work
        override_config = dataclasses.replace(
            config,
            dim=config.fast_dim,
            n_head=config.fast_n_head,
            n_local_heads=config.fast_n_local_heads,
            head_dim=config.fast_head_dim,
            intermediate_size=config.fast_intermediate_size,
            attention_qkv_bias=config.fast_attention_qkv_bias,
        )

        self.fast_layers = nn.ModuleList(
            TransformerBlock(override_config, use_sdpa=False)
            for _ in range(config.n_fast_layer)
        )
        self.fast_norm = RMSNorm(config.fast_dim, eps=config.norm_eps)
        self.fast_output = nn.Linear(
            config.fast_dim,
            config.codebook_size,
            bias=False,
        )

        self.register_buffer(
            "fast_freqs_cis",
            precompute_freqs_cis(
                config.num_codebooks,
                config.fast_dim // config.fast_n_head,
                config.rope_base,
            ),
            persistent=False,
        )
        self.apply(self._init_weights)

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        super().setup_caches(max_batch_size, max_seq_len, dtype)

        head_dim = self.config.fast_dim // self.config.fast_n_head

        # Fast transformer
        # The max seq len here is the number of codebooks
        for b in self.fast_layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                self.config.num_codebooks,
                self.config.fast_n_local_heads,
                head_dim,
                dtype=dtype,
            )

    def forward(
        self,
        inp: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        parent_result = super().forward(inp, key_padding_mask)
        token_logits = parent_result.logits
        x = parent_result.hidden_states
        x = self.fast_project_in(x)

        # Fast transformer
        fast_seq_len = self.config.num_codebooks
        fast_mask = self.causal_mask[
            None, None, :fast_seq_len, :fast_seq_len
        ]  # (B, N, Q, K)

        # Drop the last token and rotate left
        codebooks = inp[:, 1:-1, 1:]
        codebooks = F.pad(codebooks, (0, 1), value=0)
        codebook_embeddings = self.fast_embeddings(codebooks)
        x = torch.cat([x[:, None], codebook_embeddings], dim=1)
        b, s = x.size(0), x.size(2)
        x = rearrange(x, "b n s d -> (b s) n d")  # flatten the batch and seq_len

        # Remove padded part
        codebooks = rearrange(codebooks, "b n s -> (b s) n")
        codebook_mask = (codebooks == 0).all(dim=-1)

        if torch.all(codebook_mask):
            # If all codebooks are padded, we keep first 8 to make sure the model runs
            codebook_mask[:8] = False

        x_bs, x_len = x.size(0), x.size(1)
        x = x[~codebook_mask]

        for layer in self.fast_layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(
                    layer, x, self.fast_freqs_cis, fast_mask, use_reentrant=True
                )
            else:
                x = layer(x, self.fast_freqs_cis, fast_mask)

        # unflatten the batch and num_codebooks
        fast_out = self.fast_norm(x)
        codebook_logits = self.fast_output(fast_out)

        # Re-pad the codebook_logits
        buffer = torch.zeros(
            x_bs,
            x_len,
            codebook_logits.size(-1),
            device=codebook_logits.device,
            dtype=codebook_logits.dtype,
        )
        buffer[~codebook_mask] = codebook_logits
        codebook_logits = buffer

        assert codebook_logits.shape[1] == self.config.num_codebooks
        codebook_logits = rearrange(
            codebook_logits,
            "(b s) n d -> b s n d",
            b=b,
            s=s,
            n=self.config.num_codebooks,
        )

        return TransformerForwardResult(
            token_logits=token_logits,
            codebook_logits=codebook_logits,
        )

    def forward_generate_fast(
        self, x: Tensor, input_pos: Optional[Tensor] = None
    ) -> Tensor:
        # Fast transformer
        x = x.view(1, 1, -1)

        fast_mask = self.causal_mask[
            None, None, input_pos, : self.config.num_codebooks
        ]  # (B, N, Q, K)
        fast_freqs_cis = self.fast_freqs_cis[input_pos]

        for layer in self.fast_layers:
            x = layer(x, fast_freqs_cis, fast_mask, input_pos=input_pos)

        # unflatten the batch and num_codebooks
        fast_out = self.fast_norm(x)  # only take the last token
        codebook_logits = self.fast_output(fast_out)

        return codebook_logits

    def forward_generate(
        self,
        x: Tensor,
        input_pos: Optional[Tensor] = None,
        vq_masks: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        x = super().forward_generate(x, input_pos, vq_masks)
        x.hidden_states = self.fast_project_in(x.hidden_states)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True) -> None:
        super().__init__()
        self.attention = Attention(config, use_sdpa=use_sdpa)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Tensor = None
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(
            config.dim, total_head_dim, bias=config.attention_qkv_bias
        )
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.dropout = config.dropout
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.use_sdpa = use_sdpa
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if self.use_sdpa:
            if mask is None:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    y = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=True,
                        # No third party attn_mask here to use flash_attention
                    )
            else:
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        else:
            y = self.eq_scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        return self.wo(y)

    def eq_scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
    ) -> torch.Tensor:
        # This is a standard scaled dot product attention
        # It's low efficient, but it doesn't raise cuda error

        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(1, 1, L, S, dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        return attn_weight @ value


class FeedForward(nn.Module):
    def __init__(self, config: BaseModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import CONFIG_NAME, logging


logger = logging.get_logger(__name__)


CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("dual_ar", "DualARModelArgs"),
        # Add configs here
        ("albert", "AlbertConfig"),
        ("align", "AlignConfig"),
        ("altclip", "AltCLIPConfig"),
        ("audio-spectrogram-transformer", "ASTConfig"),
        ("autoformer", "AutoformerConfig"),
        ("bark", "BarkConfig"),
        ("bart", "BartConfig"),
        ("beit", "BeitConfig"),
        ("bert", "BertConfig"),
        ("bert-generation", "BertGenerationConfig"),
        ("big_bird", "BigBirdConfig"),
        ("bigbird_pegasus", "BigBirdPegasusConfig"),
        ("biogpt", "BioGptConfig"),
        ("bit", "BitConfig"),
        ("blenderbot", "BlenderbotConfig"),
        ("blenderbot-small", "BlenderbotSmallConfig"),
        ("blip", "BlipConfig"),
        ("blip-2", "Blip2Config"),
        ("bloom", "BloomConfig"),
        ("bridgetower", "BridgeTowerConfig"),
        ("bros", "BrosConfig"),
        ("camembert", "CamembertConfig"),
        ("canine", "CanineConfig"),
        ("chameleon", "ChameleonConfig"),
        ("chinese_clip", "ChineseCLIPConfig"),
        ("chinese_clip_vision_model", "ChineseCLIPVisionConfig"),
        ("clap", "ClapConfig"),
        ("clip", "CLIPConfig"),
        ("clip_text_model", "CLIPTextConfig"),
        ("clip_vision_model", "CLIPVisionConfig"),
        ("clipseg", "CLIPSegConfig"),
        ("clvp", "ClvpConfig"),
        ("code_llama", "LlamaConfig"),
        ("codegen", "CodeGenConfig"),
        ("cohere", "CohereConfig"),
        ("conditional_detr", "ConditionalDetrConfig"),
        ("convbert", "ConvBertConfig"),
        ("convnext", "ConvNextConfig"),
        ("convnextv2", "ConvNextV2Config"),
        ("cpmant", "CpmAntConfig"),
        ("ctrl", "CTRLConfig"),
        ("cvt", "CvtConfig"),
        ("dac", "DacConfig"),
        ("data2vec-audio", "Data2VecAudioConfig"),
        ("data2vec-text", "Data2VecTextConfig"),
        ("data2vec-vision", "Data2VecVisionConfig"),
        ("dbrx", "DbrxConfig"),
        ("deberta", "DebertaConfig"),
        ("deberta-v2", "DebertaV2Config"),
        ("decision_transformer", "DecisionTransformerConfig"),
        ("deformable_detr", "DeformableDetrConfig"),
        ("deit", "DeiTConfig"),
        ("depth_anything", "DepthAnythingConfig"),
        ("deta", "DetaConfig"),
        ("detr", "DetrConfig"),
        ("dinat", "DinatConfig"),
        ("dinov2", "Dinov2Config"),
        ("distilbert", "DistilBertConfig"),
        ("donut-swin", "DonutSwinConfig"),
        ("dpr", "DPRConfig"),
        ("dpt", "DPTConfig"),
        ("efficientformer", "EfficientFormerConfig"),
        ("efficientnet", "EfficientNetConfig"),
        ("electra", "ElectraConfig"),
        ("encodec", "EncodecConfig"),
        ("encoder-decoder", "EncoderDecoderConfig"),
        ("ernie", "ErnieConfig"),
        ("ernie_m", "ErnieMConfig"),
        ("esm", "EsmConfig"),
        ("falcon", "FalconConfig"),
        ("falcon_mamba", "FalconMambaConfig"),
        ("fastspeech2_conformer", "FastSpeech2ConformerConfig"),
        ("flaubert", "FlaubertConfig"),
        ("flava", "FlavaConfig"),
        ("fnet", "FNetConfig"),
        ("focalnet", "FocalNetConfig"),
        ("fsmt", "FSMTConfig"),
        ("funnel", "FunnelConfig"),
        ("fuyu", "FuyuConfig"),
        ("gemma", "GemmaConfig"),
        ("gemma2", "Gemma2Config"),
        ("git", "GitConfig"),
        ("glm", "GlmConfig"),
        ("glpn", "GLPNConfig"),
        ("gpt-sw3", "GPT2Config"),
        ("gpt2", "GPT2Config"),
        ("gpt_bigcode", "GPTBigCodeConfig"),
        ("gpt_neo", "GPTNeoConfig"),
        ("gpt_neox", "GPTNeoXConfig"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseConfig"),
        ("gptj", "GPTJConfig"),
        ("gptsan-japanese", "GPTSanJapaneseConfig"),
        ("granite", "GraniteConfig"),
        ("granitemoe", "GraniteMoeConfig"),
        ("graphormer", "GraphormerConfig"),
        ("grounding-dino", "GroundingDinoConfig"),
        ("groupvit", "GroupViTConfig"),
        ("hiera", "HieraConfig"),
        ("hubert", "HubertConfig"),
        ("ibert", "IBertConfig"),
        ("idefics", "IdeficsConfig"),
        ("idefics2", "Idefics2Config"),
        ("idefics3", "Idefics3Config"),
        ("imagegpt", "ImageGPTConfig"),
        ("informer", "InformerConfig"),
        ("instructblip", "InstructBlipConfig"),
        ("instructblipvideo", "InstructBlipVideoConfig"),
        ("jamba", "JambaConfig"),
        ("jetmoe", "JetMoeConfig"),
        ("jukebox", "JukeboxConfig"),
        ("kosmos-2", "Kosmos2Config"),
        ("layoutlm", "LayoutLMConfig"),
        ("layoutlmv2", "LayoutLMv2Config"),
        ("layoutlmv3", "LayoutLMv3Config"),
        ("led", "LEDConfig"),
        ("levit", "LevitConfig"),
        ("lilt", "LiltConfig"),
        ("llama", "LlamaConfig"),
        ("llava", "LlavaConfig"),
        ("llava_next", "LlavaNextConfig"),
        ("llava_next_video", "LlavaNextVideoConfig"),
        ("llava_onevision", "LlavaOnevisionConfig"),
        ("longformer", "LongformerConfig"),
        ("longt5", "LongT5Config"),
        ("luke", "LukeConfig"),
        ("lxmert", "LxmertConfig"),
        ("m2m_100", "M2M100Config"),
        ("mamba", "MambaConfig"),
        ("mamba2", "Mamba2Config"),
        ("marian", "MarianConfig"),
        ("markuplm", "MarkupLMConfig"),
        ("mask2former", "Mask2FormerConfig"),
        ("maskformer", "MaskFormerConfig"),
        ("maskformer-swin", "MaskFormerSwinConfig"),
        ("mbart", "MBartConfig"),
        ("mctct", "MCTCTConfig"),
        ("mega", "MegaConfig"),
        ("megatron-bert", "MegatronBertConfig"),
        ("mgp-str", "MgpstrConfig"),
        ("mimi", "MimiConfig"),
        ("mistral", "MistralConfig"),
        ("mixtral", "MixtralConfig"),
        ("mllama", "MllamaConfig"),
        ("mobilebert", "MobileBertConfig"),
        ("mobilenet_v1", "MobileNetV1Config"),
        ("mobilenet_v2", "MobileNetV2Config"),
        ("mobilevit", "MobileViTConfig"),
        ("mobilevitv2", "MobileViTV2Config"),
        ("moshi", "MoshiConfig"),
        ("mpnet", "MPNetConfig"),
        ("mpt", "MptConfig"),
        ("mra", "MraConfig"),
        ("mt5", "MT5Config"),
        ("musicgen", "MusicgenConfig"),
        ("musicgen_melody", "MusicgenMelodyConfig"),
        ("mvp", "MvpConfig"),
        ("nat", "NatConfig"),
        ("nemotron", "NemotronConfig"),
        ("nezha", "NezhaConfig"),
        ("nllb-moe", "NllbMoeConfig"),
        ("nougat", "VisionEncoderDecoderConfig"),
        ("nystromformer", "NystromformerConfig"),
        ("olmo", "OlmoConfig"),
        ("olmo2", "Olmo2Config"),
        ("olmoe", "OlmoeConfig"),
        ("omdet-turbo", "OmDetTurboConfig"),
        ("oneformer", "OneFormerConfig"),
        ("open-llama", "OpenLlamaConfig"),
        ("openai-gpt", "OpenAIGPTConfig"),
        ("opt", "OPTConfig"),
        ("owlv2", "Owlv2Config"),
        ("owlvit", "OwlViTConfig"),
        ("paligemma", "PaliGemmaConfig"),
        ("patchtsmixer", "PatchTSMixerConfig"),
        ("patchtst", "PatchTSTConfig"),
        ("pegasus", "PegasusConfig"),
        ("pegasus_x", "PegasusXConfig"),
        ("perceiver", "PerceiverConfig"),
        ("persimmon", "PersimmonConfig"),
        ("phi", "PhiConfig"),
        ("phi3", "Phi3Config"),
        ("phimoe", "PhimoeConfig"),
        ("pix2struct", "Pix2StructConfig"),
        ("pixtral", "PixtralVisionConfig"),
        ("plbart", "PLBartConfig"),
        ("poolformer", "PoolFormerConfig"),
        ("pop2piano", "Pop2PianoConfig"),
        ("prophetnet", "ProphetNetConfig"),
        ("pvt", "PvtConfig"),
        ("pvt_v2", "PvtV2Config"),
        ("qdqbert", "QDQBertConfig"),
        ("qwen2", "Qwen2Config"),
        ("qwen2_audio", "Qwen2AudioConfig"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoderConfig"),
        ("qwen2_moe", "Qwen2MoeConfig"),
        ("qwen2_vl", "Qwen2VLConfig"),
        ("rag", "RagConfig"),
        ("realm", "RealmConfig"),
        ("recurrent_gemma", "RecurrentGemmaConfig"),
        ("reformer", "ReformerConfig"),
        ("regnet", "RegNetConfig"),
        ("rembert", "RemBertConfig"),
        ("resnet", "ResNetConfig"),
        ("retribert", "RetriBertConfig"),
        ("roberta", "RobertaConfig"),
        ("roberta-prelayernorm", "RobertaPreLayerNormConfig"),
        ("roc_bert", "RoCBertConfig"),
        ("roformer", "RoFormerConfig"),
        ("rt_detr", "RTDetrConfig"),
        ("rt_detr_resnet", "RTDetrResNetConfig"),
        ("rwkv", "RwkvConfig"),
        ("sam", "SamConfig"),
        ("seamless_m4t", "SeamlessM4TConfig"),
        ("seamless_m4t_v2", "SeamlessM4Tv2Config"),
        ("segformer", "SegformerConfig"),
        ("seggpt", "SegGptConfig"),
        ("sew", "SEWConfig"),
        ("sew-d", "SEWDConfig"),
        ("siglip", "SiglipConfig"),
        ("siglip_vision_model", "SiglipVisionConfig"),
        ("speech-encoder-decoder", "SpeechEncoderDecoderConfig"),
        ("speech_to_text", "Speech2TextConfig"),
        ("speech_to_text_2", "Speech2Text2Config"),
        ("speecht5", "SpeechT5Config"),
        ("splinter", "SplinterConfig"),
        ("squeezebert", "SqueezeBertConfig"),
        ("stablelm", "StableLmConfig"),
        ("starcoder2", "Starcoder2Config"),
        ("superpoint", "SuperPointConfig"),
        ("swiftformer", "SwiftFormerConfig"),
        ("swin", "SwinConfig"),
        ("swin2sr", "Swin2SRConfig"),
        ("swinv2", "Swinv2Config"),
        ("switch_transformers", "SwitchTransformersConfig"),
        ("t5", "T5Config"),
        ("table-transformer", "TableTransformerConfig"),
        ("tapas", "TapasConfig"),
        ("time_series_transformer", "TimeSeriesTransformerConfig"),
        ("timesformer", "TimesformerConfig"),
        ("timm_backbone", "TimmBackboneConfig"),
        ("trajectory_transformer", "TrajectoryTransformerConfig"),
        ("transfo-xl", "TransfoXLConfig"),
        ("trocr", "TrOCRConfig"),
        ("tvlt", "TvltConfig"),
        ("tvp", "TvpConfig"),
        ("udop", "UdopConfig"),
        ("umt5", "UMT5Config"),
        ("unispeech", "UniSpeechConfig"),
        ("unispeech-sat", "UniSpeechSatConfig"),
        ("univnet", "UnivNetConfig"),
        ("upernet", "UperNetConfig"),
        ("van", "VanConfig"),
        ("video_llava", "VideoLlavaConfig"),
        ("videomae", "VideoMAEConfig"),
        ("vilt", "ViltConfig"),
        ("vipllava", "VipLlavaConfig"),
        ("vision-encoder-decoder", "VisionEncoderDecoderConfig"),
        ("vision-text-dual-encoder", "VisionTextDualEncoderConfig"),
        ("visual_bert", "VisualBertConfig"),
        ("vit", "ViTConfig"),
        ("vit_hybrid", "ViTHybridConfig"),
        ("vit_mae", "ViTMAEConfig"),
        ("vit_msn", "ViTMSNConfig"),
        ("vitdet", "VitDetConfig"),
        ("vitmatte", "VitMatteConfig"),
        ("vits", "VitsConfig"),
        ("vivit", "VivitConfig"),
        ("wav2vec2", "Wav2Vec2Config"),
        ("wav2vec2-bert", "Wav2Vec2BertConfig"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerConfig"),
        ("wavlm", "WavLMConfig"),
        ("whisper", "WhisperConfig"),
        ("xclip", "XCLIPConfig"),
        ("xglm", "XGLMConfig"),
        ("xlm", "XLMConfig"),
        ("xlm-prophetnet", "XLMProphetNetConfig"),
        ("xlm-roberta", "XLMRobertaConfig"),
        ("xlm-roberta-xl", "XLMRobertaXLConfig"),
        ("xlnet", "XLNetConfig"),
        ("xmod", "XmodConfig"),
        ("yolos", "YolosConfig"),
        ("yoso", "YosoConfig"),
        ("zamba", "ZambaConfig"),
        ("zoedepth", "ZoeDepthConfig"),
    ]
)


MODEL_NAMES_MAPPING = OrderedDict(
    [
        ("dual_ar", "DualARTransformer"),
        # Add full (and cased) model names here
        ("albert", "ALBERT"),
        ("align", "ALIGN"),
        ("altclip", "AltCLIP"),
        ("audio-spectrogram-transformer", "Audio Spectrogram Transformer"),
        ("autoformer", "Autoformer"),
        ("bark", "Bark"),
        ("bart", "BART"),
        ("barthez", "BARThez"),
        ("bartpho", "BARTpho"),
        ("beit", "BEiT"),
        ("bert", "BERT"),
        ("bert-generation", "Bert Generation"),
        ("bert-japanese", "BertJapanese"),
        ("bertweet", "BERTweet"),
        ("big_bird", "BigBird"),
        ("bigbird_pegasus", "BigBird-Pegasus"),
        ("biogpt", "BioGpt"),
        ("bit", "BiT"),
        ("blenderbot", "Blenderbot"),
        ("blenderbot-small", "BlenderbotSmall"),
        ("blip", "BLIP"),
        ("blip-2", "BLIP-2"),
        ("bloom", "BLOOM"),
        ("bort", "BORT"),
        ("bridgetower", "BridgeTower"),
        ("bros", "BROS"),
        ("byt5", "ByT5"),
        ("camembert", "CamemBERT"),
        ("canine", "CANINE"),
        ("chameleon", "Chameleon"),
        ("chinese_clip", "Chinese-CLIP"),
        ("chinese_clip_vision_model", "ChineseCLIPVisionModel"),
        ("clap", "CLAP"),
        ("clip", "CLIP"),
        ("clip_text_model", "CLIPTextModel"),
        ("clip_vision_model", "CLIPVisionModel"),
        ("clipseg", "CLIPSeg"),
        ("clvp", "CLVP"),
        ("code_llama", "CodeLlama"),
        ("codegen", "CodeGen"),
        ("cohere", "Cohere"),
        ("conditional_detr", "Conditional DETR"),
        ("convbert", "ConvBERT"),
        ("convnext", "ConvNeXT"),
        ("convnextv2", "ConvNeXTV2"),
        ("cpm", "CPM"),
        ("cpmant", "CPM-Ant"),
        ("ctrl", "CTRL"),
        ("cvt", "CvT"),
        ("dac", "DAC"),
        ("data2vec-audio", "Data2VecAudio"),
        ("data2vec-text", "Data2VecText"),
        ("data2vec-vision", "Data2VecVision"),
        ("dbrx", "DBRX"),
        ("deberta", "DeBERTa"),
        ("deberta-v2", "DeBERTa-v2"),
        ("decision_transformer", "Decision Transformer"),
        ("deformable_detr", "Deformable DETR"),
        ("deit", "DeiT"),
        ("deplot", "DePlot"),
        ("depth_anything", "Depth Anything"),
        ("depth_anything_v2", "Depth Anything V2"),
        ("deta", "DETA"),
        ("detr", "DETR"),
        ("dialogpt", "DialoGPT"),
        ("dinat", "DiNAT"),
        ("dinov2", "DINOv2"),
        ("distilbert", "DistilBERT"),
        ("dit", "DiT"),
        ("donut-swin", "DonutSwin"),
        ("dpr", "DPR"),
        ("dpt", "DPT"),
        ("efficientformer", "EfficientFormer"),
        ("efficientnet", "EfficientNet"),
        ("electra", "ELECTRA"),
        ("encodec", "EnCodec"),
        ("encoder-decoder", "Encoder decoder"),
        ("ernie", "ERNIE"),
        ("ernie_m", "ErnieM"),
        ("esm", "ESM"),
        ("falcon", "Falcon"),
        ("falcon_mamba", "FalconMamba"),
        ("fastspeech2_conformer", "FastSpeech2Conformer"),
        ("flan-t5", "FLAN-T5"),
        ("flan-ul2", "FLAN-UL2"),
        ("flaubert", "FlauBERT"),
        ("flava", "FLAVA"),
        ("fnet", "FNet"),
        ("focalnet", "FocalNet"),
        ("fsmt", "FairSeq Machine-Translation"),
        ("funnel", "Funnel Transformer"),
        ("fuyu", "Fuyu"),
        ("gemma", "Gemma"),
        ("gemma2", "Gemma2"),
        ("git", "GIT"),
        ("glm", "GLM"),
        ("glpn", "GLPN"),
        ("gpt-sw3", "GPT-Sw3"),
        ("gpt2", "OpenAI GPT-2"),
        ("gpt_bigcode", "GPTBigCode"),
        ("gpt_neo", "GPT Neo"),
        ("gpt_neox", "GPT NeoX"),
        ("gpt_neox_japanese", "GPT NeoX Japanese"),
        ("gptj", "GPT-J"),
        ("gptsan-japanese", "GPTSAN-japanese"),
        ("granite", "Granite"),
        ("granitemoe", "GraniteMoeMoe"),
        ("graphormer", "Graphormer"),
        ("grounding-dino", "Grounding DINO"),
        ("groupvit", "GroupViT"),
        ("herbert", "HerBERT"),
        ("hiera", "Hiera"),
        ("hubert", "Hubert"),
        ("ibert", "I-BERT"),
        ("idefics", "IDEFICS"),
        ("idefics2", "Idefics2"),
        ("idefics3", "Idefics3"),
        ("imagegpt", "ImageGPT"),
        ("informer", "Informer"),
        ("instructblip", "InstructBLIP"),
        ("instructblipvideo", "InstructBlipVideo"),
        ("jamba", "Jamba"),
        ("jetmoe", "JetMoe"),
        ("jukebox", "Jukebox"),
        ("kosmos-2", "KOSMOS-2"),
        ("layoutlm", "LayoutLM"),
        ("layoutlmv2", "LayoutLMv2"),
        ("layoutlmv3", "LayoutLMv3"),
        ("layoutxlm", "LayoutXLM"),
        ("led", "LED"),
        ("levit", "LeViT"),
        ("lilt", "LiLT"),
        ("llama", "LLaMA"),
        ("llama2", "Llama2"),
        ("llama3", "Llama3"),
        ("llava", "LLaVa"),
        ("llava_next", "LLaVA-NeXT"),
        ("llava_next_video", "LLaVa-NeXT-Video"),
        ("llava_onevision", "LLaVA-Onevision"),
        ("longformer", "Longformer"),
        ("longt5", "LongT5"),
        ("luke", "LUKE"),
        ("lxmert", "LXMERT"),
        ("m2m_100", "M2M100"),
        ("madlad-400", "MADLAD-400"),
        ("mamba", "Mamba"),
        ("mamba2", "mamba2"),
        ("marian", "Marian"),
        ("markuplm", "MarkupLM"),
        ("mask2former", "Mask2Former"),
        ("maskformer", "MaskFormer"),
        ("maskformer-swin", "MaskFormerSwin"),
        ("matcha", "MatCha"),
        ("mbart", "mBART"),
        ("mbart50", "mBART-50"),
        ("mctct", "M-CTC-T"),
        ("mega", "MEGA"),
        ("megatron-bert", "Megatron-BERT"),
        ("megatron_gpt2", "Megatron-GPT2"),
        ("mgp-str", "MGP-STR"),
        ("mimi", "Mimi"),
        ("mistral", "Mistral"),
        ("mixtral", "Mixtral"),
        ("mllama", "Mllama"),
        ("mluke", "mLUKE"),
        ("mms", "MMS"),
        ("mobilebert", "MobileBERT"),
        ("mobilenet_v1", "MobileNetV1"),
        ("mobilenet_v2", "MobileNetV2"),
        ("mobilevit", "MobileViT"),
        ("mobilevitv2", "MobileViTV2"),
        ("moshi", "Moshi"),
        ("mpnet", "MPNet"),
        ("mpt", "MPT"),
        ("mra", "MRA"),
        ("mt5", "MT5"),
        ("musicgen", "MusicGen"),
        ("musicgen_melody", "MusicGen Melody"),
        ("mvp", "MVP"),
        ("myt5", "myt5"),
        ("nat", "NAT"),
        ("nemotron", "Nemotron"),
        ("nezha", "Nezha"),
        ("nllb", "NLLB"),
        ("nllb-moe", "NLLB-MOE"),
        ("nougat", "Nougat"),
        ("nystromformer", "Nyströmformer"),
        ("olmo", "OLMo"),
        ("olmo2", "OLMo2"),
        ("olmoe", "OLMoE"),
        ("omdet-turbo", "OmDet-Turbo"),
        ("oneformer", "OneFormer"),
        ("open-llama", "OpenLlama"),
        ("openai-gpt", "OpenAI GPT"),
        ("opt", "OPT"),
        ("owlv2", "OWLv2"),
        ("owlvit", "OWL-ViT"),
        ("paligemma", "PaliGemma"),
        ("patchtsmixer", "PatchTSMixer"),
        ("patchtst", "PatchTST"),
        ("pegasus", "Pegasus"),
        ("pegasus_x", "PEGASUS-X"),
        ("perceiver", "Perceiver"),
        ("persimmon", "Persimmon"),
        ("phi", "Phi"),
        ("phi3", "Phi3"),
        ("phimoe", "Phimoe"),
        ("phobert", "PhoBERT"),
        ("pix2struct", "Pix2Struct"),
        ("pixtral", "Pixtral"),
        ("plbart", "PLBart"),
        ("poolformer", "PoolFormer"),
        ("pop2piano", "Pop2Piano"),
        ("prophetnet", "ProphetNet"),
        ("pvt", "PVT"),
        ("pvt_v2", "PVTv2"),
        ("qdqbert", "QDQBert"),
        ("qwen2", "Qwen2"),
        ("qwen2_audio", "Qwen2Audio"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoder"),
        ("qwen2_moe", "Qwen2MoE"),
        ("qwen2_vl", "Qwen2VL"),
        ("rag", "RAG"),
        ("realm", "REALM"),
        ("recurrent_gemma", "RecurrentGemma"),
        ("reformer", "Reformer"),
        ("regnet", "RegNet"),
        ("rembert", "RemBERT"),
        ("resnet", "ResNet"),
        ("retribert", "RetriBERT"),
        ("roberta", "RoBERTa"),
        ("roberta-prelayernorm", "RoBERTa-PreLayerNorm"),
        ("roc_bert", "RoCBert"),
        ("roformer", "RoFormer"),
        ("rt_detr", "RT-DETR"),
        ("rt_detr_resnet", "RT-DETR-ResNet"),
        ("rwkv", "RWKV"),
        ("sam", "SAM"),
        ("seamless_m4t", "SeamlessM4T"),
        ("seamless_m4t_v2", "SeamlessM4Tv2"),
        ("segformer", "SegFormer"),
        ("seggpt", "SegGPT"),
        ("sew", "SEW"),
        ("sew-d", "SEW-D"),
        ("siglip", "SigLIP"),
        ("siglip_vision_model", "SiglipVisionModel"),
        ("speech-encoder-decoder", "Speech Encoder decoder"),
        ("speech_to_text", "Speech2Text"),
        ("speech_to_text_2", "Speech2Text2"),
        ("speecht5", "SpeechT5"),
        ("splinter", "Splinter"),
        ("squeezebert", "SqueezeBERT"),
        ("stablelm", "StableLm"),
        ("starcoder2", "Starcoder2"),
        ("superpoint", "SuperPoint"),
        ("swiftformer", "SwiftFormer"),
        ("swin", "Swin Transformer"),
        ("swin2sr", "Swin2SR"),
        ("swinv2", "Swin Transformer V2"),
        ("switch_transformers", "SwitchTransformers"),
        ("t5", "T5"),
        ("t5v1.1", "T5v1.1"),
        ("table-transformer", "Table Transformer"),
        ("tapas", "TAPAS"),
        ("tapex", "TAPEX"),
        ("time_series_transformer", "Time Series Transformer"),
        ("timesformer", "TimeSformer"),
        ("timm_backbone", "TimmBackbone"),
        ("trajectory_transformer", "Trajectory Transformer"),
        ("transfo-xl", "Transformer-XL"),
        ("trocr", "TrOCR"),
        ("tvlt", "TVLT"),
        ("tvp", "TVP"),
        ("udop", "UDOP"),
        ("ul2", "UL2"),
        ("umt5", "UMT5"),
        ("unispeech", "UniSpeech"),
        ("unispeech-sat", "UniSpeechSat"),
        ("univnet", "UnivNet"),
        ("upernet", "UPerNet"),
        ("van", "VAN"),
        ("video_llava", "VideoLlava"),
        ("videomae", "VideoMAE"),
        ("vilt", "ViLT"),
        ("vipllava", "VipLlava"),
        ("vision-encoder-decoder", "Vision Encoder decoder"),
        ("vision-text-dual-encoder", "VisionTextDualEncoder"),
        ("visual_bert", "VisualBERT"),
        ("vit", "ViT"),
        ("vit_hybrid", "ViT Hybrid"),
        ("vit_mae", "ViTMAE"),
        ("vit_msn", "ViTMSN"),
        ("vitdet", "VitDet"),
        ("vitmatte", "ViTMatte"),
        ("vits", "VITS"),
        ("vivit", "ViViT"),
        ("wav2vec2", "Wav2Vec2"),
        ("wav2vec2-bert", "Wav2Vec2-BERT"),
        ("wav2vec2-conformer", "Wav2Vec2-Conformer"),
        ("wav2vec2_phoneme", "Wav2Vec2Phoneme"),
        ("wavlm", "WavLM"),
        ("whisper", "Whisper"),
        ("xclip", "X-CLIP"),
        ("xglm", "XGLM"),
        ("xlm", "XLM"),
        ("xlm-prophetnet", "XLM-ProphetNet"),
        ("xlm-roberta", "XLM-RoBERTa"),
        ("xlm-roberta-xl", "XLM-RoBERTa-XL"),
        ("xlm-v", "XLM-V"),
        ("xlnet", "XLNet"),
        ("xls_r", "XLS-R"),
        ("xlsr_wav2vec2", "XLSR-Wav2Vec2"),
        ("xmod", "X-MOD"),
        ("yolos", "YOLOS"),
        ("yoso", "YOSO"),
        ("zamba", "Zamba"),
        ("zoedepth", "ZoeDepth"),
    ]
)

# This is tied to the processing `-` -> `_` in `model_type_to_module_name`. For example, instead of putting
# `transfo-xl` (as in `CONFIG_MAPPING_NAMES`), we should use `transfo_xl`.
DEPRECATED_MODELS = [
    "bort",
    "deta",
    "efficientformer",
    "ernie_m",
    "gptsan_japanese",
    "graphormer",
    "jukebox",
    "mctct",
    "mega",
    "mmbt",
    "nat",
    "nezha",
    "open_llama",
    "qdqbert",
    "realm",
    "retribert",
    "speech_to_text_2",
    "tapex",
    "trajectory_transformer",
    "transfo_xl",
    "tvlt",
    "van",
    "vit_hybrid",
    "xlm_prophetnet",
]

SPECIAL_MODEL_TYPE_TO_MODULE_NAME = OrderedDict(
    [
        ("openai-gpt", "openai"),
        ("data2vec-audio", "data2vec"),
        ("data2vec-text", "data2vec"),
        ("data2vec-vision", "data2vec"),
        ("donut-swin", "donut"),
        ("kosmos-2", "kosmos2"),
        ("maskformer-swin", "maskformer"),
        ("xclip", "x_clip"),
        ("clip_vision_model", "clip"),
        ("qwen2_audio_encoder", "qwen2_audio"),
        ("clip_text_model", "clip"),
        ("siglip_vision_model", "siglip"),
        ("chinese_clip_vision_model", "chinese_clip"),
        ("rt_detr_resnet", "rt_detr"),
    ]
)


def model_type_to_module_name(key):
    """Converts a config key to the corresponding module."""
    # Special treatment
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        key = SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]

        if key in DEPRECATED_MODELS:
            key = f"deprecated.{key}"
        return key

    key = key.replace("-", "_")
    if key in DEPRECATED_MODELS:
        key = f"deprecated.{key}"

    return key


def config_class_to_model_type(config):
    """Converts a config class name to the corresponding model type"""
    for key, cls in CONFIG_MAPPING_NAMES.items():
        if cls == config:
            return key
    # if key not found check in extra content
    for key, cls in CONFIG_MAPPING._extra_content.items():
        if cls.__name__ == config:
            return key
    return None


class _LazyConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        module_name = model_type_to_module_name(key)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # Some of the mappings have entries model_type -> config of another model type. In that case we try to grab the
        # object at the top level.
        transformers_module = importlib.import_module("transformers")
        return getattr(transformers_module, value)

    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(self._extra_content.values())

    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items())

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping.keys() and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
        self._extra_content[key] = value


CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)


class _LazyLoadAllMappings(OrderedDict):
    """
    A mapping that will load all pairs of key values at the first access (either by indexing, requestions keys, values,
    etc.)

    Args:
        mapping: The mapping to load.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._initialized = False
        self._data = {}

    def _initialize(self):
        if self._initialized:
            return

        for model_type, map_name in self._mapping.items():
            module_name = model_type_to_module_name(model_type)
            module = importlib.import_module(f".{module_name}", "transformers.models")
            mapping = getattr(module, map_name)
            self._data.update(mapping)

        self._initialized = True

    def __getitem__(self, key):
        self._initialize()
        return self._data[key]

    def keys(self):
        self._initialize()
        return self._data.keys()

    def values(self):
        self._initialize()
        return self._data.values()

    def items(self):
        self._initialize()
        return self._data.keys()

    def __iter__(self):
        self._initialize()
        return iter(self._data)

    def __contains__(self, item):
        self._initialize()
        return item in self._data


def _get_class_name(model_class: Union[str, List[str]]):
    if isinstance(model_class, (list, tuple)):
        return " or ".join([f"[`{c}`]" for c in model_class if c is not None])
    return f"[`{model_class}`]"


def _list_model_options(indent, config_to_class=None, use_model_types=True):
    if config_to_class is None and not use_model_types:
        raise ValueError("Using `use_model_types=False` requires a `config_to_class` dictionary.")
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {model_type: f"[`{config}`]" for model_type, config in CONFIG_MAPPING_NAMES.items()}
        else:
            model_type_to_name = {
                model_type: _get_class_name(model_class)
                for model_type, model_class in config_to_class.items()
                if model_type in MODEL_NAMES_MAPPING
            }
        lines = [
            f"{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({MODEL_NAMES_MAPPING[model_type]} model)"
            for model_type in sorted(model_type_to_name.keys())
        ]
    else:
        config_to_name = {
            CONFIG_MAPPING_NAMES[config]: _get_class_name(clas)
            for config, clas in config_to_class.items()
            if config in CONFIG_MAPPING_NAMES
        }
        config_to_model_name = {
            config: MODEL_NAMES_MAPPING[model_type] for model_type, config in CONFIG_MAPPING_NAMES.items()
        }
        lines = [
            f"{indent}- [`{config_name}`] configuration class:"
            f" {config_to_name[config_name]} ({config_to_model_name[config_name]} model)"
            for config_name in sorted(config_to_name.keys())
        ]
    return "\n".join(lines)


def replace_list_option_in_docstrings(config_to_class=None, use_model_types=True):
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        if docstrings is None:
            # Example: -OO
            return fn
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^(\s*)List options\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = re.search(r"^(\s*)List options\s*$", lines[i]).groups()[0]
            if use_model_types:
                indent = f"{indent}    "
            lines[i] = _list_model_options(indent, config_to_class=config_to_class, use_model_types=use_model_types)
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current"
                f" docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoConfig.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )

    @classmethod
    @replace_list_option_in_docstrings()
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate one of the configuration classes of the library from a pretrained model configuration.

        The configuration class to instantiate is selected based on the `model_type` property of the config object that
        is loaded, or when it's missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model configuration hosted inside a model repo on
                      huggingface.co.
                    - A path to a *directory* containing a configuration file saved using the
                      [`~PretrainedConfig.save_pretrained`] method, or the [`~PreTrainedModel.save_pretrained`] method,
                      e.g., `./my_model_directory/`.
                    - A path or url to a saved configuration JSON *file*, e.g.,
                      `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs(additional keyword arguments, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Examples:

        ```python
        >>> from transformers import AutoConfig

        >>> # Download configuration from huggingface.co and cache.
        >>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")

        >>> # Download configuration from huggingface.co (user-uploaded) and cache.
        >>> config = AutoConfig.from_pretrained("dbmdz/bert-base-german-cased")

        >>> # If configuration file is in a directory (e.g., was saved using *save_pretrained('./test/saved_model/')*).
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/")

        >>> # Load a specific configuration file.
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/my_configuration.json")

        >>> # Change some config attributes when loading a pretrained config.
        >>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
        >>> config.output_attentions
        True

        >>> config, unused_kwargs = AutoConfig.from_pretrained(
        ...     "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        ... )
        >>> config.output_attentions
        True

        >>> unused_kwargs
        {'foo': False}
        ```"""
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        kwargs["_from_auto"] = True
        kwargs["name_or_path"] = pretrained_model_name_or_path
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        code_revision = kwargs.pop("code_revision", None)

        config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
        has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        if has_remote_code and trust_remote_code:
            class_ref = config_dict["auto_map"]["AutoConfig"]
            config_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, code_revision=code_revision, **kwargs
            )
            if os.path.isdir(pretrained_model_name_or_path):
                config_class.register_for_auto_class()
            return config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif "model_type" in config_dict:
            try:
                config_class = CONFIG_MAPPING[config_dict["model_type"]]
            except KeyError:
                raise ValueError(
                    f"The checkpoint you are trying to load has model type `{config_dict['model_type']}` "
                    "but Transformers does not recognize this architecture. This could be because of an "
                    "issue with the checkpoint, or because your version of Transformers is out of date."
                )
            return config_class.from_dict(config_dict, **unused_kwargs)
        else:
            # Fallback: use pattern matching on the string.
            # We go from longer names to shorter names to catch roberta before bert (for instance)
            for pattern in sorted(CONFIG_MAPPING.keys(), key=len, reverse=True):
                if pattern in str(pretrained_model_name_or_path):
                    return CONFIG_MAPPING[pattern].from_dict(config_dict, **unused_kwargs)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            f"Should have a `model_type` key in its {CONFIG_NAME}, or contain one of the following strings "
            f"in its name: {', '.join(CONFIG_MAPPING.keys())}"
        )

    @staticmethod
    def register(model_type, config, exist_ok=False):
        """
        Register a new configuration for this class.

        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`PretrainedConfig`]): The config to register.
        """
        if issubclass(config, PretrainedConfig) and config.model_type != model_type:
            raise ValueError(
                "The config you are passing has a `model_type` attribute that is not consistent with the model type "
                f"you passed (config has {config.model_type} and you passed {model_type}. Fix one of those so they "
                "match!"
            )
        CONFIG_MAPPING.register(model_type, config, exist_ok=exist_ok)
