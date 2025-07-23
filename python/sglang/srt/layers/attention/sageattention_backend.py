from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
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

        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )

        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )

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

        BLKQ = 128 # 从你的函数中获取
        BLKK = 64  # 从你的函数中获取

        # 计算 q_scale 的最大长度总和
        self.max_q_blocks_per_seq = (1 + BLKQ - 1) // BLKQ
        self.max_q_scale_len_sum = self.max_q_blocks_per_seq * max_bs

        # 计算 k_scale 的最大长度总和
        self.max_k_blocks_per_seq = (1 + BLKK - 1) // BLKK
        self.max_k_scale_len_sum = self.max_k_blocks_per_seq * max_bs

        self.q_scale_out = torch.zeros(
            (16 * 1, self.num_q_heads),
            device=self.device,
            dtype=torch.float32,
        )

        self.k_scale_out = torch.zeros(
            (16 * 1, self.num_kv_heads),
            device=self.device,
            dtype=torch.float32,
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
            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )

            # kv_indice part
            kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                forward_batch.seq_lens_sum, dtype=torch.int32, device=self.device
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

            # update self.kv_indices
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
            
            # kv_indice part
            kv_indptr[1 : bs + 1] = torch.cumsum(
                forward_batch.extend_prefix_lens, dim=0
            )
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                forward_batch.extend_prefix_lens.sum().item(),
                dtype=torch.int32,
                device=self.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.extend_prefix_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            # update self.kv_indices
            self.kv_indices = kv_indices

        self.forward_metadata = metadata
        
    def init_cuda_graph_state(self, max_bs: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        self.cuda_cache_seqlens_int32 =  torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self.cuda_graph_kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_kv_indices = torch.zeros(
            (max_bs * self.max_context_len,),
            dtype=torch.int32,
            device="cuda",
        )
        self.cuda_graph_cu_seqlens_q = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_cu_seqlens_k = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=self.device
        )

        """ For qk_int8 uantization """
        # self.q_int8_out = torch.zeros(
        #     (max_bs * self.max_context_len, )
        # )
        BLKQ = 128 # 从你的函数中获取
        BLKK = 64  # 从你的函数中获取

        # 计算 q_scale 的最大长度总和
        self.max_q_blocks_per_seq = (1 + BLKQ - 1) // BLKQ
        self.max_q_scale_len_sum = self.max_q_blocks_per_seq * max_bs

        # 计算 k_scale 的最大长度总和
        self.max_k_blocks_per_seq = (1 + BLKK - 1) // BLKK
        self.max_k_scale_len_sum = self.max_k_blocks_per_seq * max_bs


        # for per_block
        # self.cuda_graph_q_scale_buffer = torch.zeros(
        #     (self.max_q_scale_len_sum, self.num_q_heads),
        #     device=self.device,
        #     dtype=torch.float32
        # )

        # self.cuda_graph_k_scale_buffer = torch.zeros(
        #     (self.max_k_scale_len_sum, self.num_kv_heads),
        #     device=self.device,
        #     dtype=torch.float32,
        # )


        # Create a persistent metadata object for CUDA graph
        self.forward_metadata = SageAttentionMetadata(
            cu_seqlens_q=self.cuda_graph_cu_seqlens_q,
            cu_seqlens_k=self.cuda_graph_cu_seqlens_k,
        )
    
    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        """Initialize forward metadata for capturing CUDA graph."""
        metadata = self.forward_metadata
        device = self.device

        if forward_mode.is_decode():
            metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)

            metadata.max_seq_len_k = seq_lens.max().item()

            # Use copy_ to fill the pre-allocated tensors.
            # This operation IS captured by CUDA graph.
            metadata.cu_seqlens_q[:bs + 1].copy_(
                torch.arange(0, bs + 1, dtype=torch.int32, device=device)
            )
            metadata.cu_seqlens_k[:bs + 1].copy_(
                torch.nn.functional.pad(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            )

            # Re-use pre-allocated tensors for CUDA graph capture
            kv_indptr = self.cuda_graph_kv_indptr[: bs + 1]
            kv_indices = self.cuda_graph_kv_indices

            kv_indptr[1 : bs + 1].copy_(
                torch.cumsum(seq_lens, dim=0)
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token, 
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
        else:
            raise NotImplementedError(f"Unsupported cuda-graph capture {forward_mode=}")
            
        self.kv_indices = self.cuda_graph_kv_indices

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Init the metadata for a forward pass for replaying a cuda graph."""
        seq_lens = seq_lens[:bs]
        
        metadata = self.forward_metadata
        device = self.device

    
        # print(f"DEBUG: bs={bs}, seq_lens.shape={seq_lens.shape}")
        # print(f"DEBUG: metadata.cu_seqlens_k.shape={metadata.cu_seqlens_k.shape}")
        # print(f"DEBUG: metadata.cache_seqlens_int32.shape={metadata.cache_seqlens_int32.shape}")

        
        if forward_mode.is_decode():
            
            metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)

            metadata.max_seq_len_k = seq_lens_cpu.max().item()

            # Update the pre-allocated tensors
            metadata.cu_seqlens_q[:bs + 1].copy_(
                torch.arange(0, bs + 1, dtype=torch.int32, device=device)
            )

            # tmp = torch.nn.functional.pad(torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0))
            # print(f"DEBUG: tmp = {tmp}, tmp.shape={tmp.shape}")

            # print(f"DEBUG: metadata.cu_seqlens_k = {metadata.cu_seqlens_k}, metadata.cu_seqlens_k.shape={metadata.cu_seqlens_k.shape}")

            metadata.cu_seqlens_k[:bs + 1].copy_(
                torch.nn.functional.pad(
                    torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
                )
            )
            
            # Re-use pre-allocated tensors for CUDA graph capture
            kv_indptr = self.cuda_graph_kv_indptr[: bs + 1]
            kv_indices = self.cuda_graph_kv_indices

            kv_indptr[1 : bs + 1].copy_(
                torch.cumsum(seq_lens, dim=0)
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token, 
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

        else:
            raise NotImplementedError(f"Unsupported cuda-graph replay {forward_mode=}")

        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

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
                    self.kv_indices = cache_loc.to(torch.int32)
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

            # print(f"DEBUG: cache_loc = {cache_loc}")
            # print(f"DEBUG: k = {k}, k.shape = {k.shape}")
            # print(f"DEBUG: v = {v}, v.shape = {v.shape}")

            # print(f"DEBUG: key_cache = {key_cache[cache_loc, :, :]}")
            # print(f"DEBUG: value_cache = {value_cache[cache_loc], :, :]}")

            # print(f"DEBUG: self.kv_indices = {self.kv_indices}, self.kv_indices.dtype = {self.kv_indices.dtype}")
            
            # print(f"DEBUG: execute prefill")
            result = sageattn_varlen(
                q=q,
                key_cache=key_cache,
                value_cache=value_cache,
                kv_indices=self.kv_indices,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seq_len_q,
                max_seqlen_k=max_seq_len_k,
                is_causal=True,
                q_scale_out=self.q_scale_out,
                k_scale_out=self.k_scale_out,
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

            # kv_indices_ref = torch.cat(
            #     [
            #         self.req_to_token[forward_batch.req_pool_indices[i], :forward_batch.seq_lens[i]] 
            #         for i in range(forward_batch.batch_size)
            #     ],
            #     dim=0,
            # ).contiguous()


            # k = key_cache[kv_indices_ref, :, :]
            # v = value_cache[kv_indices_ref, :, :]

            # k = key_cache[self.kv_indices, :, :]
            # v = value_cache[self.kv_indices, :, :]

            # k = k.view(
            #     cu_seqlens_k[-1], layer.tp_k_head_num, layer.head_dim
            # )

            # v = v.view(
            #     cu_seqlens_k[-1], layer.tp_k_head_num, layer.head_dim
            # )

            q_reshaped = q.contiguous().view(
                -1, layer.tp_q_head_num, layer.head_dim
            )

            # Default: single-token self-attention
            result = sageattn_varlen(
                q=q_reshaped,
                key_cache=key_cache,
                value_cache=value_cache,
                kv_indices=self.kv_indices,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seq_len_q,
                max_seqlen_k=max_seq_len_k,
                q_scale_out=self.q_scale_out,
                k_scale_out=self.k_scale_out,
            )

            o = result

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)