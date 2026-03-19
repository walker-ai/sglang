from __future__ import annotations

"""
Sanity test: HiCache + delta-cache (cuSZp) keeps allocator/tree accounting in sync.
Requires CUDA + libcuSZp.so.
"""

import os
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from sglang.srt.distributed import parallel_state
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode


def _setup_dist():
    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "12345")
        parallel_state.init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
            distributed_init_method="env://",
        )
        created = True
    else:
        parallel_state.init_distributed_environment(
            world_size=dist.get_world_size(),
            rank=dist.get_rank(),
            local_rank=0,
            backend="gloo",
            distributed_init_method="env://",
        )

    parallel_state.ensure_model_parallel_initialized(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        backend="gloo",
    )
    return created


def _teardown_dist(created: bool):
    if created:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()


def test_hicache_delta_cache_accounting(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")

    cuszp_so = Path("/home/wangyitao/tools/cuSZp/install/lib/libcuSZp.so")
    if not cuszp_so.exists():
        pytest.skip(f"cuSZp shared library not found at {cuszp_so}")

    storage_dir = tmp_path / "hicache_file"
    os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = str(storage_dir)

    created_pg = _setup_dist()
    try:
        device = "cuda"
        page_size = 1
        gpu_tokens = 64
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
            size=8,
            max_context_len=128,
            device=device,
            enable_memory_saver=False,
        )

        cache = HiRadixCache(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            tp_cache_group=dist.group.WORLD,
            page_size=page_size,
            hicache_ratio=2.0,
            hicache_size=0,
            hicache_write_policy="write_through",
            hicache_io_backend="kernel",
            hicache_mem_layout="page_first",
            enable_metrics=False,
            eviction_policy="lru",
            hicache_storage_backend="file",
            hicache_storage_prefetch_policy="best_effort",
            model_name="demo",
            storage_backend_extra_config=None,
            is_eagle=False,
            enable_delta_cache=True,
            compression_backend="cuszp",
        )
        cache.load_back_threshold = 0

        base_len = 7
        base_indices = allocator.alloc(base_len)
        delta_indices = allocator.alloc(base_len)
        assert base_indices is not None and delta_indices is not None

        for layer in range(layer_num):
            base_k = torch.randn((base_len, head_num, head_dim), device=device, dtype=torch.float32)
            base_v = torch.randn((base_len, head_num, head_dim), device=device, dtype=torch.float32)
            delta_k = base_k + 1e-4
            delta_v = base_v - 1e-4
            kv_cache.k_buffer[layer].index_copy_(0, base_indices, base_k.to(torch.bfloat16))
            kv_cache.v_buffer[layer].index_copy_(0, base_indices, base_v.to(torch.bfloat16))
            kv_cache.k_buffer[layer].index_copy_(0, delta_indices, delta_k.to(torch.bfloat16))
            kv_cache.v_buffer[layer].index_copy_(0, delta_indices, delta_v.to(torch.bfloat16))

        tokens = list(range(1, base_len + 1))
        base_key = RadixKey(tokens, extra_key="lora0")
        node = TreeNode()
        node.key = base_key
        node.value = base_indices
        node.is_base = True
        node.parent = cache.root_node
        # hash for storage
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

        delta_key = RadixKey(tokens, extra_key="lora1")
        delta_node = cache.recompute_diff(base_indices, delta_indices, "lora1", delta_key)
        delta_node.base_node = node
        node.delta_node["lora1"] = delta_node

        cache.write_backup(node)
        cache.cache_controller.write_stream.synchronize()
        cache.writing_check()
        # wait storage thread
        for _ in range(200):
            if cache.cache_controller.backup_queue.empty():
                break
            time.sleep(0.01)
        cache.drain_storage_control_queues()
        cache._evict_backuped(node)

        # Reconstruct once to exercise load_back + delta reconstruct
        res = cache.match_prefix(delta_key)
        assert res.device_indices.numel() == base_len
        # Free reconstructed indices to simulate request completion.
        allocator.free(res.device_indices)

        # Accounting check
        cache.recompute_evictable_size()
        cache.prune_stale_values(set(allocator.free_pages.tolist()))
        available = allocator.available_size()
        evictable = cache.evictable_size()
        protected = cache.protected_size()
        assert available + evictable + protected == allocator.size

    finally:
        _teardown_dist(created_pg)
