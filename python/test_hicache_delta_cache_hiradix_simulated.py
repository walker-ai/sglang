from __future__ import annotations

from pathlib import Path

import pytest


def test_hiradix_delta_reconstruct_with_host_base_simulated():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA for cuSZp backend.")

    cuszp_so = Path("/home/wangyitao/tools/cuSZp/install/lib/libcuSZp.so")
    if not cuszp_so.exists():
        pytest.skip(f"cuSZp shared library not found at {cuszp_so}")

    import sglang.srt.mem_cache.radix_cache as radix_cache
    import sglang.srt.mem_cache.hiradix_cache as hiradix_cache

    RadixCache = radix_cache.RadixCache
    RadixKey = radix_cache.RadixKey
    TreeNode = radix_cache.TreeNode
    HiRadixCache = hiradix_cache.HiRadixCache

    class TinyKVCache:
        def __init__(self, *, size: int, layers: int, head_num: int, head_dim: int, dtype, device):
            self.k_buffer = [
                torch.zeros((size, head_num, head_dim), dtype=dtype, device=device)
                for _ in range(layers)
            ]
            self.v_buffer = [
                torch.zeros((size, head_num, head_dim), dtype=dtype, device=device)
                for _ in range(layers)
            ]

    class TinyAllocator:
        def __init__(self, *, size: int, device):
            self.device = torch.device(device)
            self._free = list(range(size))
            self._kv_cache = TinyKVCache(
                size=size,
                layers=2,
                head_num=2,
                head_dim=8,
                dtype=torch.bfloat16,
                device=self.device,
            )

        def get_kvcache(self):
            return self._kv_cache

        def alloc(self, n: int):
            if len(self._free) < n:
                return None
            out = [self._free.pop(0) for _ in range(n)]
            return torch.tensor(out, dtype=torch.int64, device=self.device)

        def free(self, indices):
            if indices is None:
                return
            if hasattr(indices, "tolist"):
                vals = indices.tolist()
            else:
                vals = list(indices)
            self._free.extend(int(x) for x in vals)

    allocator = TinyAllocator(size=256, device="cuda")
    kv = allocator.get_kvcache()

    base_len = 6
    base_indices = allocator.alloc(base_len)
    delta_indices = allocator.alloc(base_len)
    assert base_indices is not None and delta_indices is not None

    for layer in range(len(kv.k_buffer)):
        base_k = torch.randn((base_len, 2, 8), device=allocator.device, dtype=torch.float32)
        base_v = torch.randn((base_len, 2, 8), device=allocator.device, dtype=torch.float32)
        delta_k = base_k + 1e-4
        delta_v = base_v - 1e-4
        kv.k_buffer[layer].index_copy_(0, base_indices, base_k.to(torch.bfloat16))
        kv.v_buffer[layer].index_copy_(0, base_indices, base_v.to(torch.bfloat16))
        kv.k_buffer[layer].index_copy_(0, delta_indices, delta_k.to(torch.bfloat16))
        kv.v_buffer[layer].index_copy_(0, delta_indices, delta_v.to(torch.bfloat16))

    # 构造 HiRadixCache：绕过完整 __init__（里面会初始化分布式和 host pool），只调用 RadixCache.__init__。
    cache = HiRadixCache.__new__(HiRadixCache)
    # 先放置若干 HiRadix 依赖的属性，避免 __getattr__ 报错。
    class _DummyHostPool:
        def clear(self):
            pass

    class _DummyCacheController:
        def __init__(self):
            self.write_policy = "write_through"
            self.ack_write_queue = []
            self.ack_load_queue = []
            self.prefetch_rate_limited = lambda: False
            self.prefetch_tokens_occupied = 0
            self.mem_pool_device_allocator = allocator
            self.mem_pool_host = None
            self.storage_backend = None
        def reset(self):
            self.ack_write_queue.clear()
            self.ack_load_queue.clear()
    cache.cache_controller = _DummyCacheController()
    cache.enable_storage = False
    cache.enable_storage_metrics = False
    cache.tp_world_size = 1
    cache.tp_group = None
    cache.token_to_kv_pool_host = _DummyHostPool()

    RadixCache.__init__(
        cache,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        disable=False,
        enable_delta_cache=True,
        compression_backend="cuszp",
        delta_blob_data_tier="device",
    )

    # Base 节点视为已下放到 host：value=None, host_value=base_indices(cpu)
    root = cache.root_node
    node = TreeNode()
    node.parent = root
    node.key = RadixKey([1, 2, 3], extra_key="lora0")
    node.is_base = True
    node.value = None
    node.host_value = base_indices.cpu()  # backuped 属性通过 host_value 判定
    root.children[1] = node  # token-only child key

    # 写入 delta 数据
    delta_node = cache.recompute_diff(
        base_indices, delta_indices, "lora1", RadixKey([1, 2, 3], extra_key="lora1")
    )
    delta_node.base_node = node
    node.delta_node["lora1"] = delta_node

    load_back_called = {"v": 0}

    def fake_load_back(n, mem_quota=None):
        load_back_called["v"] += 1
        n.value = base_indices
        return base_indices

    cache.load_back = fake_load_back  # type: ignore[assignment]

    res = cache.match_prefix(RadixKey([1, 2, 3], extra_key="lora1"))
    assert load_back_called["v"] >= 1
    assert res.device_indices.numel() == base_len

    # 精度校验放宽（cuSZp 有损）
    for layer in range(len(kv.k_buffer)):
        recon_k = kv.k_buffer[layer].index_select(0, res.device_indices).to(torch.float32)
        recon_v = kv.v_buffer[layer].index_select(0, res.device_indices).to(torch.float32)
        ref_k = kv.k_buffer[layer].index_select(0, delta_indices).to(torch.float32)
        ref_v = kv.v_buffer[layer].index_select(0, delta_indices).to(torch.float32)
        assert (recon_k - ref_k).abs().max().item() < 3.0
        assert (recon_v - ref_v).abs().max().item() < 3.0
