from __future__ import annotations

"""
Minimal real-path demo: HiRadixCache + delta-cache (cuSZp) with host pool eviction.
Run directly (not pytest). Requires CUDA torch, sgl_kernel kvcacheio, and libcuSZp.
"""

from pathlib import Path
import time

import torch
import torch.distributed as dist

import os

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.distributed import parallel_state


def main() -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = "/home/wangyitao/hicache_demo"
    cuszp_so = Path("/home/wangyitao/tools/cuSZp/install/lib/libcuSZp.so")
    if not cuszp_so.exists():
        print(f"[skip] cuSZp shared library not found at {cuszp_so}")
        return

    if not torch.cuda.is_available():
        print("[skip] CUDA is required for this demo.")
        return

    initialized_here = False
    if not dist.is_initialized():
        parallel_state.init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
            distributed_init_method="env://",
        )
        initialized_here = True
    else:
        # Ensure WORLD group exists even if torch.distributed is already inited
        parallel_state.init_distributed_environment(
            world_size=dist.get_world_size(),
            rank=dist.get_rank(),
            local_rank=0,
            backend="gloo",
            distributed_init_method="env://",
        )

    # Initialize tensor/pipeline model-parallel groups (size=1) for HiCacheController
    parallel_state.ensure_model_parallel_initialized(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        backend="gloo",
    )
    tp_group = dist.group.WORLD

    device = "cuda"
    page_size = 1
    gpu_tokens = 128  # GPU KV capacity (tokens)
    head_num = 2
    head_dim = 8
    layer_num = 2

    kv_cache = MHATokenToKVPool(
        size=gpu_tokens,
        page_size=page_size,
        dtype=torch.bfloat16,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        device=device,
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=gpu_tokens,
        dtype=torch.bfloat16,
        device=device,
        kvcache=kv_cache,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=32,
        max_context_len=256,
        device=device,
        enable_memory_saver=False,
    )

    cache = HiRadixCache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        tp_cache_group=tp_group,
        page_size=page_size,
        hicache_ratio=2.0,  # host capacity ~= 2x GPU
        hicache_size=0,  # use ratio; avoids allocating huge GB directly
        hicache_write_policy="write_through",
        hicache_io_backend="kernel",
        hicache_mem_layout="page_first",
        enable_metrics=False,
        eviction_policy="lru",
        hicache_storage_backend="file",  # enable file backend
        hicache_storage_prefetch_policy="best_effort",
        model_name=None,
        storage_backend_extra_config=None,
        is_eagle=False,
        enable_delta_cache=True,
        compression_backend="cuszp",
    )
    # For demo: allow loading back even for small nodes (default threshold=10 tokens).
    cache.load_back_threshold = 0

    base_len = 6
    base_indices = allocator.alloc(base_len)
    delta_indices = allocator.alloc(base_len)
    assert base_indices is not None and delta_indices is not None

    # Fill base/delta KV on GPU
    for layer in range(layer_num):
        base_k = torch.randn((base_len, head_num, head_dim), device=device, dtype=torch.float32)
        base_v = torch.randn((base_len, head_num, head_dim), device=device, dtype=torch.float32)
        delta_k = base_k + 1e-4
        delta_v = base_v - 1e-4
        kv_cache.k_buffer[layer].index_copy_(0, base_indices, base_k.to(torch.bfloat16))
        kv_cache.v_buffer[layer].index_copy_(0, base_indices, base_v.to(torch.bfloat16))
        kv_cache.k_buffer[layer].index_copy_(0, delta_indices, delta_k.to(torch.bfloat16))
        kv_cache.v_buffer[layer].index_copy_(0, delta_indices, delta_v.to(torch.bfloat16))

    # Insert a base node (length matches tokens)
    tokens = list(range(1, base_len + 1))
    base_key = RadixKey(tokens, extra_key="lora0")
    node = TreeNode()
    node.key = base_key
    node.value = base_indices
    node.is_base = True
    node.parent = cache.root_node
    # 为 storage backend 准备哈希（与插入逻辑一致，page_size=1）
    node.hash_value = []
    last_hash = None
    for idx in range(0, len(tokens), page_size):
        h = cache.cache_controller.get_hash_str(
            tokens[idx : idx + page_size], prior_hash=last_hash
        )
        node.hash_value.append(h)
        last_hash = h
    cache.root_node.children[tokens[0]] = node
    cache.evictable_size_ += len(base_indices)

    # Create delta node (delta stays on GPU initially)
    delta_key = RadixKey(tokens, extra_key="lora1")
    delta_node = cache.recompute_diff(base_indices, delta_indices, "lora1", delta_key)
    delta_node.base_node = node
    node.delta_node["lora1"] = delta_node

    # Backup base to host -> storage, then evict from GPU.
    cache.write_backup(node)
    # 等待 write stream 完成，确保 ack 队列可消费，从而触发写入 storage backend。
    cache.cache_controller.write_stream.synchronize()
    cache.writing_check()
    # 等待存储线程写完后处理 ack_backup 队列
    cc = cache.cache_controller
    for _ in range(100):
        if cc.backup_queue.empty():
            break
        time.sleep(0.01)
    cache.drain_storage_control_queues()
    cache._evict_backuped(node)

    # Match with delta key -> should load base back from host, then reconstruct delta
    res = cache.match_prefix(delta_key)
    print(f"[demo] reconstructed indices shape: {tuple(res.device_indices.shape)}")

    # Validate reconstruction accuracy (lossy; use relaxed threshold)
    max_k_err = 0.0
    max_v_err = 0.0
    for layer in range(layer_num):
        recon_k = kv_cache.k_buffer[layer].index_select(0, res.device_indices).to(torch.float32)
        recon_v = kv_cache.v_buffer[layer].index_select(0, res.device_indices).to(torch.float32)
        ref_k = kv_cache.k_buffer[layer].index_select(0, delta_indices).to(torch.float32)
        ref_v = kv_cache.v_buffer[layer].index_select(0, delta_indices).to(torch.float32)
        max_k_err = max(max_k_err, (recon_k - ref_k).abs().max().item())
        max_v_err = max(max_v_err, (recon_v - ref_v).abs().max().item())
    # cuSZp 为有损压缩，这里采用宽松阈值，仅用于示例演示
    print(f"[demo] max K/V reconstruction error: {max_k_err:.4f} / {max_v_err:.4f}")
    assert max_k_err < 6.0 and max_v_err < 6.0

    print("[demo] HiCache + delta-cache roundtrip succeeded (host backup + GPU evict + reconstruct).")
    # 提示存储目录
    store_dir = os.environ.get("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", "/tmp/hicache")
    print(f"[demo] HiCache file backend dir: {store_dir}")

    if initialized_here:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
