# coding=utf-8
# Adapted from
# https://huggingface.co/models?filter=glm

# Copyright 2023 The vLLM team.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Inference-only GLM model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from sglang.srt.configs.glm import GLMConfig
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

LoraConfig = None


class GLMMLP_V2(nn.Module):
    def __init__(
            self,
            intermediate_size: int,
            config: GLMConfig,
            quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [intermediate_size] * 2,
            bias=config.use_bias,
            quant_config=quant_config,
                                )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
        )

        if not config.use_swiglu:
            raise ValueError("Unsupported activation. Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class GLMAttention(nn.Module):
    def __init__(
            self,
            config: GLMConfig,
            layer_id: int = 0,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        # hidden_states大小
        self.hidden_size = config.hidden_size
        self.use_rotary = config.use_rotary
        self.rotary_type = config.rotary_type
        # q头数
        self.total_num_heads = config.num_attention_heads
        # kv头数（mha下跟total_num_heads相等，mqa或gqa小于total_num_heads）
        self.total_kv_heads = config.num_key_value_heads
        # 并发数
        tp_size = get_tensor_model_parallel_world_size()

        # q 头或 kv头数要被tp整除
        assert self.total_num_heads % tp_size == 0
        assert self.total_kv_heads % tp_size == 0
        assert self.total_num_heads >= self.total_kv_heads

        # 当前gpu rank下的q头
        self.num_heads = self.total_num_heads // tp_size
        # 每个q头的维度
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
        # 当前gpu的q_size
        self.q_size = self.head_dim * self.num_heads

        # 当前gpu rank下的kv头
        self.num_kv_heads = self.total_kv_heads // tp_size
        # 当前gpu的k/v维度
        self.kv_size = max(1, self.num_kv_heads * self.head_dim)

        self.scale = self.head_dim ** -0.5

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_kv_heads,
            bias=(config.use_bias or config.use_qkv_bias),
            quant_config=quant_config,
        )

        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            )

        if not self.use_rotary or self.rotary_type != 'full-1d':
            raise ValueError("Unsupported model arch. Only full-1d rope is supported for now.")

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        context_layer = self.attn(q, k, v, forward_batch)
        attn_output, _ = self.dense(context_layer)
        return attn_output


class GLMBlock(nn.Module):
    def __init__(
            self,
            config: GLMConfig,
            layer_id: int = 0,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size

        intermediate_size = config.intermediate_size if config.intermediate_size \
            else (4 * hidden_size)

        self.input_layernorm = RMSNorm(hidden_size, eps=1.0e-5) if config.use_rmsnorm \
            else nn.LayerNorm(hidden_size, eps=1.0e-5)

        self.attention = GLMAttention(config, layer_id, quant_config, )

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=1.0e-5) if config.use_rmsnorm \
            else nn.LayerNorm(hidden_size, eps=1.0e-5)

        if config.moe_config and config.num_experts > 0:
            raise ValueError("Unsupported MoE Model.")
        else:
            if not (config.mlp_version == "v2" or config.gate_up):
                raise ValueError("Unsupported GLM MLP_V1 Model.")
            mlp_class = GLMMLP_V2
        self.mlp = mlp_class(intermediate_size, config, quant_config)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            forward_batch: ForwardBatch,
            residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.attention(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class GLMModel(nn.Module):

    def __init__(
            self,
            config: GLMConfig,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.word_embeddings = VocabParallelEmbedding(self.vocab_size, self.embed_dim)

        # tie_word_embeddings为true，复用tie_word_embeddings，反之是独立的
        self.lm_head = self.word_embeddings if config.tie_word_embeddings \
            else ParallelLMHead(self.vocab_size, self.embed_dim)

        self.embedding_dropout = torch.nn.Dropout(config.embedding_dropout_prob)

        self.layers = nn.ModuleList(
            [
                GLMBlock(config, i, quant_config=quant_config)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.final_layernorm = RMSNorm(self.embed_dim, eps=1.0e-5) if config.use_rmsnorm \
            else nn.LayerNorm(self.embed_dim, eps=1.0e-5)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.word_embeddings(input_ids)
        # hidden_states = self.embedding_dropout(hidden_states)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class GLMForCausalLM(nn.Module):
    def __init__(
            self,
            config: GLMConfig,
            quant_config: Optional[QuantizationConfig] = None,
            cache_config=None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.transformer = GLMModel(config, quant_config)
        self.lm_head = self.transformer.lm_head
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            forward_batch: ForwardBatch,
            input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head.weight, forward_batch
        )


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # lm_head 在tie_word_embeddings=true时跳过，与embeddings共享；反之则独立加载
            if (("v_head" in name) or ("inv_freq" in name) or
                    (self.config.tie_word_embeddings and "lm_head" in name)):
                continue

            # 如果是norm head的方式，需要在初始化的时候对lm_head进行处理
            if self.config.norm_head and "lm_head.weight" in name:
                import torch.nn.functional as F
                loaded_weight = F.normalize(loaded_weight, dim=0, p=2, eps=1e-7)

            # key对齐，默认以 transformer. 开头
            name = name[len('glm.'):] if name.startswith('glm.') else name
            name = name[len('transformer.'):] if name.startswith('transformer.transformer.') else name
            if not name.startswith('transformer.'):
                name = 'transformer.' + name
            # moe router/gate改名
            name = name.replace('.router.classifier.', '.gate.')
            # moe v1 - ffn去掉冗余的experts关键词
            moe_mlp_v1 = bool('mlp.experts.' in name and self.config.mlp_version != "v2")
            if moe_mlp_v1:
                name = name.replace('mlp.experts.', 'mlp.')
            # moe v2改名
            name = name.replace('.expert_', '.')

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # 跳过不存在的module key
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # 跳过不存在的module key
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

# class GLMForConditionalGeneration(GLMForCausalLM):
#     pass

# class GLMModel(GLMForCausalLM):
#     pass

# EntryClass = [GLMForCausalLM, GLMForConditionalGeneration, GLMModel]

EntryClass = [GLMForCausalLM]

