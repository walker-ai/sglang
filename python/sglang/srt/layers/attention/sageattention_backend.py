from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
from sglang.srt.utils import get_compiler_backend



if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from sgl_kernel.sage_attn import sageattn, sageattn_varlen

@dataclass
class SageAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 1
    # Maximum sequence length for key
    max_seq_len_k: int = 0
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    # Window size (typically used by Gemma)
    window_size: tuple = (-1, -1)
    # Page table, the index of KV Cache Tables/Blocks
    page_table: torch.Tensor = None

    # Encoder metadata
    # Cumulative sequence lengths for encoder key
    encoder_cu_seqlens_k: torch.Tensor = None
    # Maximum sequence length for encoder key
    encoder_max_seq_len_k: int = 0
    # Sequence lengths for the forward batch
    encoder_lens_int32: torch.Tensor = None
    # Page table for the encoder
    encoder_page_table: torch.Tensor = None

    @dataclass
    class LocalAttentionMetadata:
        local_query_start_loc: torch.Tensor = None  # cu_seqlens_q for local attention
        local_seqused_k: torch.Tensor = None  # sequence lengths for local attention
        local_block_table: torch.Tensor = None  # block table for local attention
        local_max_query_len: int = 0  # max query length for local attention
        local_max_seq_len: int = 0  # max sequence length for local attention

    local_attn_metadata: Optional[LocalAttentionMetadata] = None


class SageAttentionBackend(AttentionBackend):
    """SageAttention backend implementation."""

    def __init__(
        self,
        model_runner,
        skip_prefill: bool = False,
    ):
        super().__init__()

        self.forward_metadata: SageAttentionMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        self.skip_prefill = skip_prefill

        self.use_mla = False

        # Local attention settings
        self.attention_chunk_size = (
            model_runner.attention_chunk_size
            if hasattr(model_runner, "attention_chunk_size")
            else None
        )

        max_bs = model_runner.req_to_token_pool.size
        self.kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )


    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr

        metadata = SageAttentionMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode():
            # TODO(walker-ai): currently not support speculative decoding, Normal Decode
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()

            kv_indices = torch.empty(
                forward_batch.seq_lens_sum, dtype=torch.int32, device=self.device
            )

            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )

            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            self.kv_indices = kv_indices

            # self._init_local_attn_metadata(metadata, device)
        elif forward_batch.forward_mode.is_extend():
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)

            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(forward_batch.extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.cu_seqlens_q = metadata.cu_seqlens_k
            

            # Setup local attention if enabled
            # if forward_batch.forward_mode == ForwardMode.EXTEND:
            #     self._init_local_attn_metadata(metadata, device)

        self.forward_metadata = metadata
        

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # TODO(walker-ai): support multi-head latent attention
    ):
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    raise NotImplementedError("SageAttention does not support MLA currently.")

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        max_seq_len_q = metadata.max_seq_len_q
        max_seq_len_k = metadata.max_seq_len_k
        cu_seqlens_q = metadata.cu_seqlens_q
        cu_seqlens_k = metadata.cu_seqlens_k


        # Use Sage Attention for prefill
        if not self.use_mla:
            # Do multi-head attention
            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            # q = q.view(
            #     forward_batch.batch_size, -1, layer.tp_q_head_num, layer.head_dim
            # )

            # k = k.view(
            #     forward_batch.batch_size, -1, layer.tp_k_head_num, layer.head_dim
            # )

            # v= v.view(
            #     forward_batch.batch_size, -1, layer.tp_v_head_num, layer.head_dim
            # )

            # key_cache = key_cache.view(
            #     -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            # )
            # value_cache = value_cache.view(
            #     -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            # )

            # result = sageattn(
            #     q=q,
            #     k=k,
            #     v=v,
            #     is_causal=True,
            # )

            q = q.view(
                cu_seqlens_q[-1], layer.tp_q_head_num, layer.head_dim
            )

            k = k.view(
                cu_seqlens_k[-1], layer.tp_k_head_num, layer.head_dim
            )

            v= v.view(
                cu_seqlens_k[-1], layer.tp_v_head_num, layer.head_dim
            )
            
            result = sageattn_varlen(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seq_len_q,
                max_seqlen_k=max_seq_len_k,
                is_causal=True,
            )

         
            o = result

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            

            
    
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # TODO(walker-ai): support multi-head latent attention
    ):
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        max_seq_len_q = metadata.max_seq_len_q
        max_seq_len_k = metadata.max_seq_len_k
        cu_seqlens_q = metadata.cu_seqlens_q
        cu_seqlens_k = metadata.cu_seqlens_k

        if not self.use_mla:
            # Do multi-head attention

            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            # key_cache = key_cache.view(
            #     forward_batch.batch_size, -1, layer.tp_k_head_num, layer.head_dim
            # )
            # value_cache = value_cache.view(
            #     forward_batch.batch_size, -1, layer.tp_v_head_num, layer.head_dim
            # )

            kv_indices_ref = torch.cat(
                [
                    self.req_to_token[forward_batch.req_pool_indices[i], :forward_batch.seq_lens[i]] 
                    for i in range(forward_batch.batch_size)
                ],
                dim=0,
            ).contiguous()


            k = key_cache[kv_indices_ref, :, :]
            v = value_cache[kv_indices_ref, :, :]

            # k = key_cache[self.kv_indices, :, :]
            # v = value_cache[self.kv_indices, :, :]

            k = k.view(
                cu_seqlens_k[-1], layer.tp_k_head_num, layer.head_dim
            )

            v = v.view(
                cu_seqlens_k[-1], layer.tp_k_head_num, layer.head_dim
            )

            q_reshaped = q.contiguous().view(
                cu_seqlens_q[-1], layer.tp_q_head_num, layer.head_dim
            )

            # Default: single-token self-attention
            result = sageattn_varlen(
                q=q_reshaped,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seq_len_q,
                max_seqlen_k=max_seq_len_k,
            )

            o = result

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)