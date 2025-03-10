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
from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)

from sglang.srt.configs.glm import GLMConfig
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import EPMoE
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers

LoraConfig = None


class GLMMLP_V2(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: GLMConfig,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: Optional[bool] = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [intermediate_size] * 2,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

        if not config.use_swiglu:
            raise ValueError("Unsupported activation. Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class GLMMoE_V2(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: GLMConfig,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: Optional[bool] = True,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.norm_expert_prob = config.norm_expert_prob
        self.hidden_size = config.hidden_size
        self.num_shared_experts = config.num_shared_experts
        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
        self.experts = MoEImpl(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=config.expert_intermediate_size,
            renormalize=self.norm_expert_prob,
            quant_config=quant_config,
            tp_size=self.tp_size,
            prefix=add_prefix("experts", prefix),
        )

        if self.num_shared_experts > 0:
            intermediate_size = (config.expert_intermediate_size *
                                 self.num_shared_experts)
            self.shared_experts = GLMMLP_V2(
                intermediate_size=intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        if self.num_shared_experts > 0:
            shared_output = self.shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits)

        if self.num_shared_experts > 0:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_size)


class GLMAttention(nn.Module):
    def __init__(
        self,
        config: GLMConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
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
            prefix=add_prefix("query_key_value", prefix),
        )

        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=add_prefix("dense", prefix),
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
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.hidden_size

        intermediate_size = config.intermediate_size if config.intermediate_size \
            else (4 * hidden_size)

        assert config.use_rmsnorm, "DO NOT support nn.LayerNorm for bailing in SGLang"
        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        self.attention = GLMAttention(
            config,
            layer_id,
            quant_config,
            prefix=add_prefix("attention", prefix),
        )

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        if config.moe_config and config.num_experts > 0:
            mlp_class = GLMMoE_V2
        else:
            if not (config.mlp_version == "v2" or config.gate_up):
                raise ValueError("Unsupported GLM MLP_V1 Model for bailing.")
            mlp_class = GLMMLP_V2
        self.mlp = mlp_class(intermediate_size, config, quant_config, prefix=add_prefix("mlp", prefix),)

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
        prefix: str = ""
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.word_embeddings = VocabParallelEmbedding(
            self.vocab_size,
            self.embed_dim,
            quant_config=quant_config,
        )

        # tie_word_embeddings为true，复用tie_word_embeddings，反之是独立的
        self.lm_head = self.word_embeddings if config.tie_word_embeddings \
            else ParallelLMHead(self.vocab_size, self.embed_dim, quant_config=quant_config)

        self.embedding_dropout = torch.nn.Dropout(config.embedding_dropout_prob)

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: GLMBlock(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix
            ),
            prefix=add_prefix("layers", prefix),
        )

        assert config.use_rmsnorm
        self.final_layernorm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.word_embeddings(input_ids)
        else:
            hidden_states = input_embeds

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
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.transformer = GLMModel(config, quant_config, prefix=add_prefix("transformer", ""),)
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
        hidden_states = self.transformer(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
        expert_params_mapping = MoEImpl.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

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
            # moe v2改名
            name = name.replace('.expert_', '.')

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
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
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
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


class GLMForConditionalGeneration(GLMForCausalLM):
    pass


EntryClass = [GLMForCausalLM, GLMForConditionalGeneration]
