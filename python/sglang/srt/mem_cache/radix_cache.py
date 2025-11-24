from __future__ import annotations

from typing import Dict

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import time
from collections import defaultdict
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union

import torch
import numpy as np

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:

    def __init__(self, token_ids: List[int], extra_key: Optional[str] = None):
        # token ids sequence
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''})"


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

        # 新增字段用于差分存储
        self.base_node: Optional[TreeNode] = None  # 存储差分节点所依赖的基础节点映射
        self.delta_node: Dict[str, TreeNode] = {}  # 存储所有差分节点 {extra_key: node}  
        self.is_base: bool = False  # 标记是否为基础节点 
        
        self.delta_data: Optional[tuple] = None  # <-- 新的 (将存储 e.g., (handle_K_list, handle_V_list, "diff_list"))

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _check_extra_key(key0: RadixKey, key1: RadixKey):
    if key0.extra_key != key1.extra_key:
        raise ValueError(
            f"_key_match should be run on the same extra key, but got key0.extra_key={key0.extra_key} != key1.extra_key={key1.extra_key}"
        )


def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size

    return i

def _key_match_token_only(key0: RadixKey, key1: RadixKey):
    # TODO: 实现仅用 token 来进行匹配的函数

    # 其实就是不进行 extra_key 的检查，pass _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i

def _key_match_extra_key(key0: RadixKey, key1: RadixKey):
    if key0 != key1:
        return 0
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i

def get_child_key_token_only(key: RadixKey, page_size: int = 1):
    if page_size == 1:
        return key.token_ids[0]
    else:
        return tuple(key.token_ids[:page_size])

def get_child_key(key: RadixKey, page_size: int = 1):
    if page_size == 1:
        plain_key = key.token_ids[0]
    else:
        plain_key = tuple(key.token_ids[:page_size])
    if key.extra_key is None:
        return plain_key
    else:
        return (key.extra_key, plain_key)


def _convert_to_bigram_key(tokens: List[int]) -> List[Tuple[int, int]]:
    # EAGLE uses bigram keys in the radix tree since draft sequence is the one-token-shifted version of target
    # [1, 2, 3, 4] -> [(1,2), (2,3), (3,4)]
    if len(tokens) < 2:
        return []
    if isinstance(tokens[0], tuple):
        return tokens
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


class RadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        disable: bool = False,
        enable_kv_cache_events: bool = False,
        eviction_policy: str = "lru",
        is_eagle: bool = False,
        enable_delta_cache: Optional[bool] = False,
        compression_backend: Optional[str] = None,  # sz or cuSZp
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.disable = disable
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue = []
        self.is_eagle = is_eagle
        self.enable_delta_cache = enable_delta_cache
        self.compression_backend = compression_backend

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if enable_delta_cache:
            # 延迟引入
            from sglang.srt.mem_cache.delta_cache import CuSZpBackend, SZBackend

            self.key_match_fn_token_only = _key_match_token_only
            self.key_match_fn_extra_key = _key_match_extra_key
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key_token_only

            # 通用存储池：可以存 Tensor(cuSZp)，也可以存 numpy （SZ）
            self.delta_data_pool: Dict[int, Union[torch.Tensor, np.ndarray]] = {}

            self.total_delta_bytes = 0  # 压缩后的物理字节数
            self.total_delta_len = 0    # 逻辑上的 Token 总数量
            self.delta_data_counter: int = 0

            if self.compression_backend == "cuszp":
                self.compressor = CuSZpBackend(self.device)
            elif self.compression_backend == "sz":
                self.compressor = SZBackend()
            else:
                raise ValueError(f"Unknown compression backend: {compression_backend}")
        else:
            if self.page_size == 1:
                self.key_match_fn = _key_match_page_size1
                self.get_child_key_fn = get_child_key
            else:
                self.key_match_fn = partial(_key_match_paged, page_size=page_size)
                self.get_child_key_fn = partial(get_child_key, page_size=page_size)

        if is_eagle:
            self.key_convert_fn = _convert_to_bigram_key
        else:
            self.key_convert_fn = lambda key: key

        if eviction_policy.lower() == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif eviction_policy.lower() == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif eviction_policy.lower() == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif eviction_policy.lower() == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif eviction_policy.lower() == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        else:
            raise ValueError(
                f"Unknown eviction policy: {eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo'."
            )
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self._record_all_cleared_event()

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            key (RadixKey): The lookup key containing a list of token ids and an
                optional ``extra_key`` namespace tag. If ``page_size > 1`` the
                length is internally truncated to a multiple of ``page_size``
                before matching. Passing an empty key returns an empty result
                with the root as the last node.
            **kwargs: Reserved for future extensions (ignored currently).

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0). ``last_device_node`` and
            ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        key.token_ids = self.key_convert_fn(key.token_ids)

        def empty_match_result():
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        if self.disable or len(key) == 0:
            return empty_match_result()

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        if len(key) == 0:
            return empty_match_result()

        # --- (!! 核心修改 !!) ---
        
        # 1. 收集重建信息
        #    collected_segments_info: List[tuple(data, type, metadata)]
        collected_segments_info, last_node = self._match_prefix_helper(self.root_node, key)

        if not collected_segments_info:
            # 没有匹配到任何片段
            return MatchResult(
                device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
                last_device_node=last_node,
                last_host_node=last_node,
            )

        # 2. 处理片段 (重建或直接使用)
        final_indices_list = []
        for segment_info in collected_segments_info:
            # segment_info[1] 是 "type" ("base" 或 "delta")
            if segment_info[1] == "base":
                # 'data' (segment_info[0]) 已经是索引
                final_indices_list.append(segment_info[0])
            elif segment_info[1] == "delta":
                # 'data' 是句柄元组, 'metadata' 是 base 索引
                # 调用重建函数
                newly_allocated_indices = self._reconstruct_and_alloc(segment_info)
                
                if newly_allocated_indices is None:
                    # 重建失败 (例如 OOM)
                    # 停止匹配, 只返回到此为止的索引
                    break 
                
                final_indices_list.append(newly_allocated_indices)
        
        # 3. 组装最终结果
        if final_indices_list:
            value = torch.cat(final_indices_list)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        # --- (结束核心修改) ---

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def insert(self, key: RadixKey, value=None, chunked=False):
        if self.disable:
            return 0

        key.token_ids = self.key_convert_fn(key.token_ids)

        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)

        if self.is_eagle:
            # Make sure the value len equal to the EAGLE bigram key len
            value = value[: len(key)]

        return self._insert_helper(self.root_node, key, value)

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        all_token_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :all_token_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:all_token_len]
        # For EAGLE radix cache, we will convert the key to bigram key, e.g. [1,2,3,4] -> [(1,2), (2,3), (3,4)], the length will -1. ((len([(1,2), (2,3), (3,4)]) = len([1,2,3,4]) - 1))
        # So for the corresponding kv length should also -1. Then we get the actual_kv_len, and use it to do later calculation and slicing.
        actual_kv_len = all_token_len - 1 if self.is_eagle else all_token_len
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :all_token_len
        ]

        if self.page_size != 1:
            page_aligned_len = actual_kv_len // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = actual_kv_len
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        page_aligned_token_len = (
            page_aligned_len + 1 if self.is_eagle else page_aligned_len
        )

        old_prefix_len = len(req.prefix_indices)
        if self.is_eagle and old_prefix_len > req.last_matched_prefix_len:
            # In EAGLE chunked prefill case, the prefix_indices included one unmatched token (kv_indices[actual_kv_len:])
            # Here we -1 to make sure the kv of the unmatched token can be freed correctly to avoid memory leak
            old_prefix_len -= 1

        # Radix Cache takes one ref in memory pool
        if is_insert:
            new_prefix_len = self.insert(
                RadixKey(token_ids[:page_aligned_token_len], req.extra_key),
                page_aligned_kv_indices,
            )
            # Free the duplicates that were already in the tree
            self.token_to_kv_pool_allocator.free(
                kv_indices[old_prefix_len:new_prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[old_prefix_len:page_aligned_len]
            )

        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        all_token_len = len(token_ids)
        # For EAGLE radix cache, we will convert the key to bigram key, e.g. [1,2,3,4] -> [(1,2), (2,3), (3,4)], the length will -1. ((len([(1,2), (2,3), (3,4)]) = len([1,2,3,4]) - 1))
        # So for the corresponding kv length should also -1. Then we get the actual_kv_len, and use it to do later calculation and slicing.
        actual_kv_len = all_token_len - 1 if self.is_eagle else all_token_len
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :all_token_len
        ]

        if self.page_size != 1:
            page_aligned_len = actual_kv_len // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = actual_kv_len
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        # For EAGLE, the page_aligned_len is for the bigram key, the normal key len should +1
        page_aligned_token_len = (
            page_aligned_len + 1 if self.is_eagle else page_aligned_len
        )
        page_aligned_token_ids = token_ids[:page_aligned_token_len]

        old_prefix_len = len(req.prefix_indices)
        if self.is_eagle and old_prefix_len > req.last_matched_prefix_len:
            # In EAGLE chunked prefill case, the prefix_indices included one unmatched token (kv_indices[actual_kv_len:])
            # Here we -1 to make sure the kv of the unmatched token can be freed correctly to avoid memory leak
            old_prefix_len -= 1

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(
            RadixKey(page_aligned_token_ids, req.extra_key),
            page_aligned_kv_indices,
            chunked=chunked,
        )
        self.token_to_kv_pool_allocator.free(kv_indices[old_prefix_len:new_prefix_len])

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node, _, _ = self.match_prefix(
            RadixKey(token_ids=page_aligned_token_ids, extra_key=req.extra_key)
        )
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(old_prefix_len, len(new_indices))),
            new_indices[old_prefix_len:],
        )

        # The last_matched_prefix_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `last_matched_prefix_len` field to make sure the partial part can be freed correctly.
        req.last_matched_prefix_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        if self.page_size != 1:
            # Handle partial page, the partial part should be freed in the next cache_unfinished_req and final cache_finished_req.
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            if self.is_eagle:
                # Attach the kv index of the last token for EAGLE, it can be used in chunked prefill
                req.prefix_indices = torch.cat(
                    [new_indices, kv_indices[actual_kv_len:]]
                )
            else:
                req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, num_tokens: int):
        if self.disable:
            return

        leaves = self._collect_leaves()
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)

            # 级联的 delta_node 全部删掉
            self._free_node_delta_resources(x)

            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)
    
    def store_in_delta_pool(self, compressed_data, delta_len=0) -> int:
        """
        (修改) 存储 GPU Tensor 并使用 numel * element_size 统计字节。
        """
        handle = self.delta_data_counter
        self.delta_data_pool[handle] = compressed_data

        if isinstance(compressed_data, torch.Tensor):
            # cuSZp case
            size_bytes = compressed_data.numel() * compressed_data.element_size()
        else:
            # SZ Case (numpy)
            size_bytes = compressed_data.nbytes
        
        self.total_delta_bytes += size_bytes

        self.delta_data_counter += 1
        return handle
    
    def release_from_delta_pool(self, delta_data_tuple: tuple):
        """(修改) 释放 GPU Tensor 并更新计数。"""

        if self.compression_backend == "sz":
            if (delta_data_tuple is None or len(delta_data_tuple) != 7
                or delta_data_tuple[6] != "diff_list_v2"):
                return 
            handle_K_list, handle_V_list, k_shape, _, _, _, _ = delta_data_tuple

        elif self.compression_backend == "cuszp":
            if (delta_data_tuple is None or len(delta_data_tuple) != 9
                or delta_data_tuple[6] != "diff_list_v2"):
                return 
            handle_K_list, handle_V_list, k_shape, _, _, _, _, _, _ = delta_data_tuple
        
        # 释放逻辑长度
        delta_token_len = k_shape[0]
        self.total_delta_len -= delta_token_len

        for handle in handle_K_list + handle_V_list:
            if handle in self.delta_data_pool:
                data = self.delta_data_pool[handle]
                
                # [Dynamic] 释放时同样判断类型
                if isinstance(data, torch.Tensor):
                    size_bytes = data.numel() * data.element_size()
                else:
                    size_bytes = data.nbytes
                self.total_delta_bytes -= size_bytes
                del self.delta_data_pool[handle]

    def _get_dynamic_error_bound(self, base_tensor: torch.Tensor, delta_tensor: torch.Tensor) -> float:
        """
        (新增) 根据 Base 和 Delta 的余弦相似度动态决定 Error Bound (eb_abs)。
        逻辑：相似度越高 -> Diff 越小 -> 需要越高的精度 (越小的 eb_abs)。
        """
        # 为了计算速度，可以只采样一部分，或者直接 flatten 计算
        # 这里为了准确性使用 flatten (注意性能开销，显存拷贝是主要瓶颈，这个计算相对较快)
        sim = torch.nn.functional.cosine_similarity(
            base_tensor.flatten().float(), 
            delta_tensor.flatten().float(), 
            dim=0
        ).item()
        
        # 简单的分级策略 (阈值可根据实验调整)
        if sim > 0.975:
            return 1e-4
        elif sim > 0.85:
            return 1e-3 
        else:
            return 1e-2
        
    def compress(self, target_tensor, eb_abs):
        # TODO: 实现 compress 函数
        
        diff_tensor = target_tensor
        diff_numpy = diff_tensor.detach().cpu().float().numpy()  # 转为 NumPy 数组

        diff_compressed, _ = sz.compress(diff_numpy, eb_mode=0, eb_abs=eb_abs, eb_rel=0, eb_pwr=0)

        diff_tensor_size = diff_tensor.numel() * diff_tensor.element_size()
        compressed_size = diff_compressed.nbytes

        return diff_compressed
    
    def decompress(self, compressed_diff, shape, dtype) -> torch.Tensor:
        decompressed_diff = sz.decompress(compressed_diff, shape, original_dtype=dtype)

        # 转为 torch 张量
        decompressed_diff = torch.from_numpy(decompressed_diff).to('cuda').to(torch.bfloat16)
        return decompressed_diff
    
    def recompute_diff(self, base_indices, delta_indices, extra_key, key_segment):
        """
        (修改) 调用 compress_cuSZp。
        """
        start_time = time.perf_counter()

        device = self.token_to_kv_pool_allocator.device
        base_indices = base_indices.to(device)
        delta_indices = delta_indices.to(device)

        handle_K_list = []
        handle_V_list = []

        # 用于存储每一层的 error bound
        k_eb_list = []
        v_eb_list = []

        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        num_layers = len(kv_cache.k_buffer) 

        k_shape, v_shape = None, None
        k_dtype, v_dtype = None, None

        total_orig_bytes = 0
        total_comp_bytes = 0
        
        delta_len = len(base_indices)
        self.total_delta_len += delta_len

        for i in range(num_layers): 
            base_k_buffer_layer, base_v_buffer_layer = kv_cache.k_buffer[i], kv_cache.v_buffer[i]

            base_K_tensor = base_k_buffer_layer.index_select(0, base_indices)
            base_V_tensor = base_v_buffer_layer.index_select(0, base_indices)
            delta_K_tensor = base_k_buffer_layer.index_select(0, delta_indices)
            delta_V_tensor = base_v_buffer_layer.index_select(0, delta_indices)

            k_eb = self._get_dynamic_error_bound(base_K_tensor, delta_K_tensor)
            v_eb = self._get_dynamic_error_bound(base_V_tensor, delta_V_tensor)

            k_eb_list.append(k_eb)
            v_eb_list.append(v_eb)
            
            diff_K = delta_K_tensor - base_K_tensor
            diff_V = delta_V_tensor - base_V_tensor

            if i == 0:
                k_shape, k_dtype = diff_K.shape, diff_K.dtype
                v_shape, v_dtype = diff_V.shape, diff_V.dtype

            compressed_K = self.compressor.compress(diff_K, error_bound=k_eb)
            compressed_V = self.compressor.compress(diff_V, error_bound=v_eb)
            
            # 统计原始字节数 (BF16 = 2 bytes), 注意 SZ 返回 numpy, cuSZp 返回 Tensor
            orig_size = (diff_K.numel() * diff_K.element_size()) + (diff_V.numel() * diff_V.element_size())
            total_orig_bytes += orig_size
            
            # 统计压缩字节数
            comp_size_k = compressed_K.numel() * compressed_K.element_size() if isinstance(compressed_K, torch.Tensor) else compressed_K.nbytes
            comp_size_v = compressed_V.numel() * compressed_V.element_size() if isinstance(compressed_V, torch.Tensor) else compressed_V.nbytes
            total_comp_bytes += (comp_size_k + comp_size_v)
            
            handle_K_list.append(self.store_in_delta_pool(compressed_K, delta_len))
            handle_V_list.append(self.store_in_delta_pool(compressed_V, delta_len))

        used_slots, log_toks, saved_mem = self.get_memory_stats()
        print(f"[DeltaCache Stats] Total Bytes: {self.total_delta_bytes} | Saved: {saved_mem:.2f} MB")

        new_delta_node = TreeNode()
        new_delta_node.key = key_segment

        new_delta_node.delta_data = (
            handle_K_list, handle_V_list, 
            k_shape, v_shape, k_dtype, v_dtype, 
            "diff_list_v2",
            k_eb_list, v_eb_list
        )
        new_delta_node.value = delta_indices.cpu() 
        
        duration = (time.perf_counter() - start_time) * 1000
        ratio = total_orig_bytes / total_comp_bytes if total_comp_bytes > 0 else 0
        print(f"[DeltaCache] 📉 Write(Diff-GPU): LoRA={extra_key} | Ratio={ratio:.2f}x | Time={duration:.2f}ms")
        
        return new_delta_node
    
    def get_memory_stats(self):
        """
        计算并返回 Delta Cache 的内存节省统计信息。
        """
        # self.total_delta_bytes, sz返回一个 numpy 数组，uint8 格式，因此一个就是一字节

        # 计算一个 delta_bytes占多少 token槽位：2bytes(bf16) * 32 (num_layers) * 8*128 (hidden_states) = total_bytes
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        total_kv_slots = self.total_delta_bytes / (32 * 8 * 128 * 2)  # 1 个 slot 占用空间：32 * 8 * 128 * 2 字节

        # 计算节省了多少空间
        # 原本占用空间：
        origin_occupy_space = self.total_delta_len * 2 * 32 * 1024 * 2  # prompt 长度 * 2(kv) * 32(num_layers) * 4096(hidden_states) * 2(bf16) bytes
        # 现在占用空间：
        cur_occupy_space = self.total_delta_bytes

        saved_space = (origin_occupy_space - cur_occupy_space) / (1024 * 1024)  # MB
        return total_kv_slots, self.total_delta_len, saved_space

    ##### Internal Helper Functions #####

    def _match_prefix_helper_legacy(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node
    
    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        """
        (新增) 遍历树, 收集重建所需的信息片段。
        """
        if not self.enable_delta_cache:
            value_list, last_node = self._match_prefix_helper_legacy(node, key)
            # 转换格式以匹配新输出
            collected_segments_info = [(val, "base", None) for val in value_list]
            return collected_segments_info, last_node

        # --- 新的差分逻辑 ---
        node.last_access_time = time.monotonic()
        request_extra_key = key.extra_key
        search_key = key
        child_key = self.get_child_key_fn(search_key) # token-only

        collected_segments_info = [] # 存储 (data, type, metadata) 元组
        last_node = node # 跟踪 token 路径的末端
        extra_key_matched = True

        while len(search_key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            last_node = child # 总是更新 last_node 到 token 路径的末端

            prefix_len = self.key_match_fn_token_only(child.key, search_key)

            if prefix_len < len(child.key):
                # --- 分裂情况 ---
                new_split_node = self._split_node(child.key, child, prefix_len)
                last_node = new_split_node # Token 路径在此结束

                if extra_key_matched:
                    segment_info = self._get_value_for_key(new_split_node, request_extra_key)
                    data, type, metadata = segment_info
                    if type == "miss":
                        extra_key_matched = False
                    else:
                        collected_segments_info.append(segment_info)
                break
            else:
                # --- 完整节点匹配情况 ---
                if extra_key_matched:
                    segment_info = self._get_value_for_key(child, request_extra_key)
                    data, type, metadata = segment_info
                    if type == "miss":
                        extra_key_matched = False
                    else:
                        collected_segments_info.append(segment_info)

                node = child
                search_key = search_key[prefix_len:]
                if len(search_key):
                    child_key = self.get_child_key_fn(search_key)

        return collected_segments_info, last_node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        """
        (完整) 将一个节点分裂为前缀 (new_node) 和后缀 (child)。
        """
        # --- 1. 基础设置 (与旧版相同) ---
        self._record_remove_event(child)
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len] 
        child_key_suffix = child.key[split_len:]
        
        # --- 2. 差分缓存分裂逻辑 ---
        if self.enable_delta_cache:
            
            # --- 2a. Base 版本分裂 (保持不变) ---
            if child.is_base:
                new_node.is_base = True
                new_node.value = child.value[:split_len] # Base 前缀索引 (配方)
                child.value = child.value[split_len:]    # Base 后缀索引 (配方)
                new_node.key.extra_key = child.key.extra_key
            
            # --- 2b. Delta 版本分裂 (!! 核心重构 !!) ---
            new_node.delta_node = {}
            suffix_delta_nodes = {} 
            
            for extra_key, delta_node in child.delta_node.items():
                
                # i. 释放旧的、无效的 Diff 数据 (句柄)
                self.release_from_delta_pool(delta_node.delta_data)

                # ii. 获取“配方”(原始 Delta 索引)
                original_delta_indices = delta_node.value
                
                # iii. 切片“配方” (Tensor 切片)
                prefix_delta_indices = original_delta_indices[:split_len]
                suffix_delta_indices = original_delta_indices[split_len:]

                # iv. 重新计算并存储 "前缀" Diff
                if len(prefix_delta_indices) > 0 and new_node.value is not None:
                    new_prefix_delta = self.recompute_diff(
                        new_node.value, prefix_delta_indices, 
                        extra_key, delta_node.key[:split_len]
                    )
                    new_prefix_delta.base_node = new_node
                    new_node.delta_node[extra_key] = new_prefix_delta

                # v. 重新计算并存储 "后缀" Diff
                if len(suffix_delta_indices) > 0 and child.value is not None:
                    new_suffix_delta = self.recompute_diff(
                        child.value, suffix_delta_indices,
                        extra_key, delta_node.key[split_len:]
                    )
                    new_suffix_delta.base_node = child
                    suffix_delta_nodes[extra_key] = new_suffix_delta
            
            child.delta_node = suffix_delta_nodes
            
        else:
            # --- 旧的非差分逻辑 (保持不变) ---
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        # --- 结束 ---

        # --- 3. 收尾 (与旧版相同) ---
        child.parent = new_node
        child.key = child_key_suffix 
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        self._record_store_event(new_node)
        self._record_store_event(child)

        return new_node
    
    def _insert_helper_legacy(self, node: TreeNode, key: RadixKey, value):
        """ 这是原版的 _insert_helper，用于非差分模式 """
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._record_store_event(new_node)
        return total_prefix_length

    def _insert_helper(self, node: TreeNode, key: RadixKey, value):
        # 如果禁用了差分缓存，则使用旧的（非差分）逻辑
        if not self.enable_delta_cache:
            return self._insert_helper_legacy(node, key, value)
        
        # --- 差分缓存开启时的逻辑 ---
        
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key) 
        original_key = key
        original_value = value

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            
            prefix_len = self.key_match_fn_token_only(node.key, key)
            radix_prefix_len = self.key_match_fn_extra_key(node.key, key)
            total_prefix_length += radix_prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            
            if prefix_len < len(node.key):
                # --- Case A: 节点分裂 ---
                new_node = self._split_node(node.key, node, prefix_len)
                
                # 在分裂出的新父节点上存储
                self._store_value_in_node(new_node, original_key, original_value) 
                node = new_node
                # return total_prefix_length + prefix_len

            # --- Case B: 完美匹配节点，继续 ---
            
            # ******** 关键修复 ********
            # 我们刚刚完美匹配了 'node'。我们 *必须* 在此节点上存储
            # 我们的 (extra_key, value) 版本。
            self._store_value_in_node(node, original_key, original_value)
            # **************************

            # total_prefix_length += prefix_len
            # key = key[prefix_len:]
            # (我们不再需要切片 'value' 循环变量)

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            # --- Case D: 匹配在节点边界停止 (需要添加新子节点) ---
            # (例如，我们匹配了 [1,2,3], 现在需要添加 [4,5])
            new_node = TreeNode()
            new_node.parent = node 
            new_node.key = key # 剩余的 key [4, 5]
            new_node.value = value
            
            # 存储
            self._store_value_in_node(new_node, original_key, original_value)
            
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._record_store_event(new_node)
        return total_prefix_length
    
    def _store_value_in_node(self, node: TreeNode, key_with_extra: RadixKey, original_value: torch.Tensor):
        """
        (完整) 在给定的 token 节点上存储特定 extra_key 的 value。
        """
        if not self.enable_delta_cache:
             node.value = original_value
             return

        extra_key = key_with_extra.extra_key

        # --- 1. 计算切片 (与您之前的代码相同) ---
        end_pos = 0
        temp_node = node
        nodes_path = []
        while temp_node.parent is not None:
            nodes_path.append(temp_node)
            temp_node = temp_node.parent
        end_pos = sum(len(n.key) for n in reversed(nodes_path))
        start_pos = end_pos - len(node.key)
        if start_pos >= len(original_value):
            return 
        end_pos = min(end_pos, len(original_value))
        value_segment = original_value[start_pos : end_pos]
        key_segment = key_with_extra[start_pos : end_pos]
        if len(value_segment) == 0:
            return

        # --- 2. 存储逻辑 ---
        
        # Case 1: 节点是空的 (没有 base)，将其设为 base
        if not node.is_base:
            node.is_base = True
            node.key.extra_key = extra_key 
            node.value = value_segment # Base 存储 *索引* (Tensor)
            # [Log] 记录 Base 创建
            print(f"[DeltaCache] 🟦 Set Base: LoRA={extra_key} | NodeID={node.id}")
        
        # Case 2: 插入的 extra_key 与 base 相同 (相同的不进行覆盖 base)
        elif node.key.extra_key == extra_key:
            if node.value is not None and not torch.equal(node.value, value_segment):
                self.token_to_kv_pool_allocator.free(node.value)
                node.value = value_segment # 存储新的 Base 索引
            elif node.value is None:
                node.value = value_segment
        
        # Case 3: 插入的 extra_key 是一个新的 "delta"
        else:
            # [Log] 触发 Delta 存储逻辑
            print(f"[DeltaCache] 🔀 Detect Variant: Base={node.key.extra_key} vs New={extra_key}")

            base_indices = node.value    # Base 配方
            delta_indices = value_segment # Delta 配方

            # --- 3a. 调用 recompute_diff ---
            new_delta_node = self.recompute_diff(
                base_indices, delta_indices, extra_key, key_segment
            )
            
            # --- 3b. 将新节点插入 delta_node 字典 ---
            existing_delta_node = node.delta_node.get(extra_key)
            if existing_delta_node:
                # [Log] 如果覆盖了旧的 Delta
                print(f"[DeltaCache] 🔄 Overwriting existing Delta for {extra_key}")
                self.release_from_delta_pool(existing_delta_node.delta_data) # 释放旧 handle
            
            new_delta_node.base_node = node
            node.delta_node[extra_key] = new_delta_node
            
            # --- 3c. (!! 关键 !!) 释放 Delta 的 *原始* 索引 ---
            self.token_to_kv_pool_allocator.free(delta_indices)

    def _get_value_for_key(self, node: TreeNode, extra_key: Optional[str]) -> tuple:
        """
        (新增) 获取重建所需的信息。
        返回: tuple (data, type, metadata)
          - type="base": data=indices (Tensor), metadata=None
          - type="delta": data=delta_data_tuple, metadata=base_indices (Tensor)
          - type="miss": data=None, metadata=None
        """
        if not self.enable_delta_cache:
            # 旧逻辑: 只返回索引或 None
            return (node.value, "base" if node.value is not None else "miss", None)

        # --- 新的差分逻辑 ---
        
        # Case 1: 请求的 extra_key 匹配 Base
        if node.is_base and node.key.extra_key == extra_key:
            if node.value is not None:
                return (node.value, "base", None)
            else:
                return (None, "miss", None) 

        # Case 2: 请求的 extra_key 匹配一个 Delta
        delta_node = node.delta_node.get(extra_key)
        if delta_node:
            # 检查 delta_data 是否是我们期望的格式
            if (delta_node.delta_data and isinstance(delta_node.delta_data, tuple)
                and delta_node.delta_data[6] == "diff_list_v2"):
                
                # 我们需要 Base 节点的索引来进行重建
                base_indices = node.value # Base 节点的索引存储在 .value
                if base_indices is None:
                     print(f"WARNING: Delta found for {extra_key} but base node {node.id} has no value!")
                     return (None, "miss", None)

                # 返回: (句柄元组, "delta", Base的索引)
                return (delta_node.delta_data, "delta", base_indices)
            else:
                print(f"WARNING: Delta node {delta_node.id} for {extra_key} has invalid delta_data!")
                return (None, "miss", None)

        # Case 3: Miss. Token 路径匹配了, 但此 LoRA 在此节点无数据
        return (None, "miss", None)
    
    def _reconstruct_and_alloc(self, segment_info: tuple) -> Optional[torch.Tensor]:
        """
        (修改) 调用 decompress_cuSZp。
        """
        data, type, metadata = segment_info

        if type == "base":
            return data 
        
        if type == "miss":
            return None
            
        if type == "delta":
            start_time = time.perf_counter()

            delta_data_tuple = data
            base_indices = metadata 
            
            if self.compression_backend == "sz":
                # sz
                (handle_K_list, handle_V_list, 
                 k_shape, v_shape, k_dtype, v_dtype, 
                 _, k_eb_list, v_eb_list) = delta_data_tuple
            else:
                # cuszp
                (handle_K_list, handle_V_list, 
                k_shape, v_shape, k_dtype, v_dtype, 
                _, k_eb_list, v_eb_list) = delta_data_tuple

            num_tokens_to_alloc = k_shape[0]
            if num_tokens_to_alloc == 0:
                return torch.tensor([], dtype=torch.int64, device=self.device)
                
            new_indices = self.token_to_kv_pool_allocator.alloc(num_tokens_to_alloc)
            if new_indices is None:
                print(f"[DeltaCache] ❌ OOM Warning")
                return None

            kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
            num_layers = len(kv_cache.k_buffer) 
            
            base_indices = base_indices.to(self.device)

            for i in range(num_layers):
                base_K_tensor = kv_cache.k_buffer[i].index_select(0, base_indices)
                base_V_tensor = kv_cache.v_buffer[i].index_select(0, base_indices)
                
                handle_K = handle_K_list[i]
                compressed_K = self.delta_data_pool[handle_K]
    
                # 通用解压器
                diff_K = self.compressor.decompress(compressed_K, base_K_tensor.shape, k_dtype, k_eb_list[i])
                
                handle_V = handle_V_list[i]
                compressed_V = self.delta_data_pool[handle_V]
                
                diff_V = self.compressor.decompress(compressed_V, base_V_tensor.shape, v_dtype, v_eb_list[i])
                
                reconstructed_K = base_K_tensor + diff_K
                reconstructed_V = base_V_tensor + diff_V
                
                kv_cache.k_buffer[i].index_copy_(0, new_indices, reconstructed_K)
                kv_cache.v_buffer[i].index_copy_(0, new_indices, reconstructed_V)

            duration = (time.perf_counter() - start_time) * 1000
            print(f"[DeltaCache] 📈 Read(Reconstruct-GPU): Tokens={num_tokens_to_alloc} | Time={duration:.2f}ms")

            return new_indices
            
        return None 

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)
    
    def _free_node_delta_resources(self, node: TreeNode):
        """
        (新增) 辅助函数：释放一个节点下挂载的所有 Delta 资源。
        这是为了防止 Memory Leak。当一个 Base 节点被 Evict 时，
        依附于它的所有 Delta 节点也必须被销毁，因为没有 Base 它们无法还原。
        """
        if not self.enable_delta_cache:
            return

        # 遍历该节点挂载的所有 Delta 变体
        for extra_key, delta_node in node.delta_node.items():
            if delta_node.delta_data is not None:
                # [Critical] 必须从全局池中释放内存，否则会泄露
                self.release_from_delta_pool(delta_node.delta_data)
                
        
        # 清空字典，断开引用
        node.delta_node.clear()

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _collect_leaves(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _record_store_event(self, node: TreeNode):
        # One BlockStored per ``page_size`` chunk.
        if self.enable_kv_cache_events:
            # First chunk links to the last page of the parent node (if any).
            if node.parent is None or node != self.root_node:
                parent_block_hash = None
            else:
                last_page_start = (
                    (len(node.parent.key) - 1) // self.page_size
                ) * self.page_size
                parent_parent_tokens = node.parent.key.token_ids[last_page_start:]
                parent_block_hash = hash(tuple(parent_parent_tokens))

            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash(tuple(page_tokens))

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                    )
                )

                # Chain next chunk to this one.
                parent_block_hash = block_hash

    def _record_remove_event(self, node: TreeNode):
        # One BlockRemoved per chunk.
        if self.enable_kv_cache_events:
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue
                block_hash = hash(tuple(page_tokens))
                self.kv_event_queue.append(BlockRemoved(block_hashes=[block_hash]))

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    tree = RadixCache(None, None, page_size=1, disable=False)

    # Example token id sequences (as lists of ints)
    # tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
    # tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key=None))
    # tree.insert(RadixKey(token_ids=[1, 2, 4, 5], extra_key=None))
    # tree.insert(RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    # tree.insert(RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key=None))
    # tree.pretty_print()

    # print(tree.match_prefix(RadixKey(token_ids=[1, 2, 3, 13, 14], extra_key=None)))

    tree.insert(RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key=None))
    print(tree.match_prefix(RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.pretty_print()
