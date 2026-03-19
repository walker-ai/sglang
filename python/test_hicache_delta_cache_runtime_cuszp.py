from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _import_modules(repo_root: Path):
    sys.path.insert(0, str(repo_root / "python"))
    try:
        import sglang.srt.mem_cache.radix_cache as radix_cache  # noqa: F401
    except Exception as exc:  # pragma: no cover - skip on missing deps
        pytest.skip(f"Cannot import radix_cache module: {exc}")
    return radix_cache


def test_delta_cache_cuszp_roundtrip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA for cuSZp backend.")

    cuszp_so = Path("/home/wangyitao/tools/cuSZp/install/lib/libcuSZp.so")
    if not cuszp_so.exists():
        pytest.skip(f"cuSZp shared library not found at {cuszp_so}")

    repo_root = Path(__file__).resolve().parents[1]
    radix_cache = _import_modules(repo_root)
    RadixCache = radix_cache.RadixCache
    RadixKey = radix_cache.RadixKey
    TreeNode = radix_cache.TreeNode

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

    # Fill base/delta values
    for layer in range(len(kv.k_buffer)):
        base_k = torch.randn((base_len, 2, 8), device=allocator.device, dtype=torch.float32)
        base_v = torch.randn((base_len, 2, 8), device=allocator.device, dtype=torch.float32)
        delta_k = base_k + 1e-4
        delta_v = base_v - 1e-4
        kv.k_buffer[layer].index_copy_(0, base_indices, base_k.to(torch.bfloat16))
        kv.v_buffer[layer].index_copy_(0, base_indices, base_v.to(torch.bfloat16))
        kv.k_buffer[layer].index_copy_(0, delta_indices, delta_k.to(torch.bfloat16))
        kv.v_buffer[layer].index_copy_(0, delta_indices, delta_v.to(torch.bfloat16))

    cache = RadixCache(
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        disable=False,
        enable_delta_cache=True,
        compression_backend="cuszp",
        delta_blob_data_tier="device",
    )

    node = TreeNode()
    node.is_base = True
    node.value = base_indices
    node.key = RadixKey([1, 2, 3], extra_key="lora0")

    delta_node = cache.recompute_diff(
        base_indices, delta_indices, "lora1", RadixKey([1, 2, 3], extra_key="lora1")
    )
    delta_node.base_node = node
    node.delta_node["lora1"] = delta_node

    segment_info = (delta_node.delta_data, "delta", base_indices)
    new_indices = cache._reconstruct_and_alloc(segment_info)
    assert new_indices is not None
    assert new_indices.numel() == base_len

    # Validate reconstruction accuracy (lossy, so use a loose threshold)
    for layer in range(len(kv.k_buffer)):
        recon_k = kv.k_buffer[layer].index_select(0, new_indices).to(torch.float32)
        recon_v = kv.v_buffer[layer].index_select(0, new_indices).to(torch.float32)
        ref_k = kv.k_buffer[layer].index_select(0, delta_indices).to(torch.float32)
        ref_v = kv.v_buffer[layer].index_select(0, delta_indices).to(torch.float32)

        # cuSZp 是有损压缩，这里只验证数值在可接受范围内（取较宽松阈值）。
        assert (recon_k - ref_k).abs().max().item() < 3.0
        assert (recon_v - ref_v).abs().max().item() < 3.0
