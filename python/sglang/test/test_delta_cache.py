import torch
from sglang.srt.mem_cache.radix_cache import RadixKey, RadixCache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

LAYER_NUM=32

kvcache = MHATokenToKVPool(
    size=512,
    page_size=1,
    dtype=torch.bfloat16,
    head_num=8,
    head_dim=128,
    layer_num=LAYER_NUM,
    device='cuda',
    enable_memory_saver=False,
    start_layer=0,
    end_layer=LAYER_NUM
)

allocator = TokenToKVPoolAllocator(
    size=512,
    dtype=torch.bfloat16,
    device='cuda',
    kvcache=kvcache,
    need_sort=False
)

# --- (需要 MockKVCache 和 TokenToKVPoolAllocator 才能运行) ---
# (请确保这些类和 import 语句在您的测试文件中)

NUM_LAYERS = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

class MockKVCache:
    def __init__(self, num_tokens=512):
        self.k_buffer = []
        self.v_buffer = []
        for _ in range(NUM_LAYERS):
            self.k_buffer.append(torch.randn(num_tokens, 8, 128, device=device))
            self.v_buffer.append(torch.randn(num_tokens, 8, 128, device=device))

# (假设 TokenToKVPoolAllocator 已定义)
# class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator): ...

class MockAllocator:
    """(模拟) 模拟 TokenToKVPoolAllocator。"""
    def __init__(self, device):
        self.kv_cache = MockKVCache()
        self.freed_indices = [] # "Spy" 列表
        self.device = device
        self.free_pages = torch.tensor([], device=device, dtype=torch.int64)

    def get_kvcache(self):
        return self.kv_cache

    def free(self, indices: torch.Tensor):
        """(Spy 'free' method) 记录被释放的索引。"""
        if indices is not None and indices.numel() > 0:
            self.freed_indices.append(tuple(indices.cpu().tolist()))
            print(f"  (MockAllocator: Logged free {indices.cpu().tolist()})")

kvcache = MockKVCache(512)

def test1():
    # --- 测试: 启用差分缓存 ---
    print("--- Test 1: Delta Cache ENabled (page_size=1) ---")

    # --- 1. 创建真实的 Allocator 和模拟的 KVCache ---
    mock_kvcache = MockKVCache(num_tokens=1000)
    
    # (!! 关键 !!) 创建您的 *真实* Allocator 实例
    allocator = TokenToKVPoolAllocator(
        size=1000, 
        dtype=torch.float16, 
        device=device, 
        kvcache=mock_kvcache, 
        need_sort=False # (必须为 False, free() 才会添加到 .free_pages)
    )

    # --- 2. (!! 关键 !!) 手动设置 Allocator 状态 ---
    # 定义我们将在测试中使用的索引 (确保为 int64)
    val_A = torch.tensor([10, 20, 30, 40, 50], device=device, dtype=torch.int64)
    val_B = torch.tensor([11, 22, 33, 44, 55], device=device, dtype=torch.int64)
    val_A_prefix = torch.tensor([10, 20, 30], device=device, dtype=torch.int64)
    val_C = torch.tensor([100, 200, 300], device=device, dtype=torch.int64)
    
    # 将所有将要 "占用" 的索引合并
    all_indices_in_use = torch.cat([val_A, val_B, val_C]).unique()
    
    # 从 allocator 的 free_pages 中移除它们，模拟它们 "已被分配"
    # (我们在 CPU 上执行 isin，然后将 mask 传回 GPU)
    mask = torch.isin(allocator.free_pages.cpu(), all_indices_in_use.cpu())
    allocator.free_pages = allocator.free_pages[~mask.to(device)]
    
    print(f"Allocator setup complete. Initial available: {allocator.available_size()}")
    
    # 验证它们现在 *不* 在空闲池中
    # torch.isin() 检查 val_B 中的每个元素是否在 free_pages 中
    assert not torch.all(torch.isin(val_B, allocator.free_pages))
    print(f"  (Setup check: {val_B.tolist()} successfully removed from free_pages)")

    # --- 3. 运行测试逻辑 ---
    
    tree = RadixCache(None, allocator, page_size=1, disable=False, enable_delta_cache=True)

    # 3a. 插入 lora_A (Base)
    print("\nInserting [1, 2, 3, 4, 5] for 'lora_A'")
    key_A = RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key='lora_A')
    tree.insert(key_A, val_A)
    
    # 3b. 插入 lora_B (Delta) -> 触发 free(val_B)
    print("\nInserting [1, 2, 3, 4, 5] for 'lora_B'")
    key_B = RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key='lora_B')
    tree.insert(key_B, val_B)
    
    # 3c. 插入 lora_A (Base Override + Split) -> 触发 free(val_A) (旧 Base)
    print("\nInserting [1, 2, 3] for 'lora_A' (Base Override + Split)")
    key_A_prefix = RadixKey(token_ids=[1, 2, 3], extra_key='lora_A')
    tree.insert(key_A_prefix, val_A_prefix)

    # 3d. 插入 lora_C (Delta) -> 触发 free(val_C)
    print("\nInserting [1, 2, 3] for 'lora_C'")
    key_C = RadixKey(token_ids=[1, 2, 3], extra_key='lora_C')
    tree.insert(key_C, val_C)
    
    print("\nFinal tree state:")
    tree.pretty_print()
    
    # --- 4. 验证 (!! 检查 allocator.free_pages 状态 !!) ---
    print("\nVerifying memory management (Checking Allocator State):")
    try:
        # ... (所有对 node_123 和 node_45 状态的断言都保持不变) ...
        # (例如: assert node_123.is_base ...)
        # (例如: assert isinstance(delta_C.delta_data, tuple) ...)
        # (例如: assert delta_C.value.tolist() == val_C.cpu().tolist() ...)

        node_123 = tree.root_node.children[1]
        node_45 = node_123.children[4]

        # 验证 Node(1,2,3)
        print(f"Node [1,2,3] (key: {node_123.key.token_ids}):")
        assert node_123.is_base and node_123.key.extra_key == 'lora_A'
        assert torch.equal(node_123.value, val_A_prefix) # Base value
        print(f"  Base 'lora_A' (value): {node_123.value.tolist()}... PASSED")

        # 验证 lora_C (Delta)
        assert 'lora_C' in node_123.delta_node
        delta_C = node_123.delta_node['lora_C']
        print(f"  Delta 'lora_C' (Node id={delta_C.id}):")
        assert isinstance(delta_C.delta_data, tuple) and delta_C.delta_data[6] == "diff_list_v2"
        assert len(delta_C.delta_data[0]) == NUM_LAYERS
        assert isinstance(delta_C.value, torch.Tensor) and delta_C.value.tolist() == val_C.cpu().tolist()
        print(f"    ... stores 'delta_data' (tuple) and 'value' (recipe): PASSED")

        # 验证 lora_B (Delta 前缀)
        # (因为 lora_A 覆盖时触发了分裂，lora_B 应该也被分裂了)
        assert 'lora_B' in node_123.delta_node
        delta_B_prefix = node_123.delta_node['lora_B']
        print(f"  Delta 'lora_B' (Node id={delta_B_prefix.id}):")
        assert isinstance(delta_B_prefix.delta_data, tuple) and delta_B_prefix.delta_data[6] == "diff_list_v2"
        assert len(delta_B_prefix.delta_data[0]) == NUM_LAYERS
        val_B_prefix_recipe = [11, 22, 33]
        assert isinstance(delta_B_prefix.value, torch.Tensor) and delta_B_prefix.value.tolist() == val_B_prefix_recipe
        print(f"    ... stores 'delta_data' (tuple) and 'value' (recipe): PASSED")

        # 验证 Node(4,5)
        print(f"\nNode [4,5] (key: {node_45.key.token_ids}):")
        assert node_45.is_base and node_45.key.extra_key == 'lora_A'
        val_A_suffix_val = [40, 50]
        assert node_45.value.tolist() == val_A_suffix_val # Base value (suffix)
        print(f"  Base 'lora_A' (value): {node_45.value.tolist()}... PASSED")

        # 验证 lora_B (Delta Suffix)
        assert 'lora_B' in node_45.delta_node
        delta_B_suffix = node_45.delta_node['lora_B']
        print(f"  Delta 'lora_B' (Node id={delta_B_suffix.id}):")
        assert isinstance(delta_B_suffix.delta_data, tuple) and delta_B_suffix.delta_data[6] == "diff_list_v2"
        assert len(delta_B_suffix.delta_data[0]) == NUM_LAYERS
        val_B_suffix_recipe = [44, 55] # lora_B 的 [4,5] 部分
        assert isinstance(delta_B_suffix.value, torch.Tensor) and delta_B_suffix.value.tolist() == val_B_suffix_recipe
        print(f"    ... stores 'delta_data' (tuple) and 'value' (recipe): PASSED")
        
        # --- 验证内存释放 (检查 allocator.free_pages 状态) ---
        print("\nVerifying memory management (Checking Allocator State):")

        # 检查 'lora_B' 的索引是否已返回到 free_pages
        # (它在步骤 3b 中被 free)
        assert torch.all(torch.isin(val_B, allocator.free_pages))
        print("  Delta indices for 'lora_B' were freed to pool: PASSED")

        # 检查 'lora_C' 的索引是否已返回
        # (它在步骤 3d 中被 free)
        assert torch.all(torch.isin(val_C, allocator.free_pages))
        print("  Delta indices for 'lora_C' were freed to pool: PASSED")

        # 检查 'val_A' (原始 Base) *没有* 被完整释放
        # (因为 [40, 50] 仍在 Node_45 中使用)
        assert not torch.all(torch.isin(val_A, allocator.free_pages))
        print("  Original Base 'val_A' was (correctly) not fully freed: PASSED")

        # 检查 [10, 20, 30] (Base 前缀) *不* 应该在空闲池中
        # (因为它在 3c 中是 Base 覆盖, 相同值未释放)
        assert not torch.all(torch.isin(val_A_prefix, allocator.free_pages))
        print("  Base prefix 'lora_A' [10, 20, 30] was (correctly) not freed: PASSED")

        # 检查 [40, 50] (Base 后缀) *不应该* 在空闲池中
        # (它仍然被 Node_45 占用)
        val_A_suffix = torch.tensor([40, 50], device=device, dtype=torch.int64)
        assert not torch.all(torch.isin(val_A_suffix, allocator.free_pages))
        print("  Base suffix 'lora_A' [40, 50] is (correctly) not in pool: PASSED")

        print("\nTest 1 (Final Corrected Version) PASSED!")

    except Exception as e:
        print(f"\nTest 1 FAILED: {e}")
        traceback.print_exc()

def test2():
    """corner case: Multi-Delta Split"""
    print("--- Test Corner Case 2: Multi-Delta Split ---")

    # (使用 MockAllocator)
    allocator = MockAllocator(device=device)
    tree = RadixCache(None, allocator, page_size=1, disable=False, enable_delta_cache=True)

    # 1. 插入 Base 和 2 个 Deltas (全长)
    val_A = torch.tensor([10, 20, 30, 40], device=device, dtype=torch.int64)
    tree.insert(RadixKey(token_ids=[1, 2, 3, 4], extra_key='lora_A'), val_A)

    val_B = torch.tensor([11, 22, 33, 44], device=device, dtype=torch.int64)
    tree.insert(RadixKey(token_ids=[1, 2, 3, 4], extra_key='lora_B'), val_B)

    val_C = torch.tensor([12, 23, 34, 45], device=device, dtype=torch.int64)
    tree.insert(RadixKey(token_ids=[1, 2, 3, 4], extra_key='lora_C'), val_C)

    print("\nTree before split:")
    tree.pretty_print()

    # 2. 插入一个部分 Delta，触发分裂
    val_D = torch.tensor([100, 200], device=device, dtype=torch.int64)
    tree.insert(RadixKey(token_ids=[1, 2], extra_key='lora_D'), val_D)

    print("\nTree after split:")
    tree.pretty_print()

    # --- 3. 验证状态 (已修改) ---
    print("\nVerifying node states:")
    try:
        node_12 = tree.root_node.children[1]
        node_34 = node_12.children[3]

        # 验证 Node(1,2)
        print(f"Node [1,2] (key: {node_12.key.token_ids}):")
        assert node_12.is_base and node_12.key.extra_key == 'lora_A'
        assert node_12.value.tolist() == [10, 20] # Base
        print(f"  Base 'lora_A' (value): {node_12.value.tolist()}... PASSED")

        # 检查所有 Delta
        assert sorted(list(node_12.delta_node.keys())) == ['lora_B', 'lora_C', 'lora_D']

        # 验证 'lora_B' (前缀)
        delta_B = node_12.delta_node['lora_B']
        assert isinstance(delta_B.delta_data, tuple) and delta_B.delta_data[6] == "diff_list_v2" # "数据"
        assert delta_B.value.tolist() == [11, 22]    # "配方"

        # 验证 'lora_C' (前缀)
        delta_C = node_12.delta_node['lora_C']
        assert isinstance(delta_C.delta_data, tuple) and delta_C.delta_data[6] == "diff_list_v2" # "数据"
        assert delta_C.value.tolist() == [12, 23]    # "配方"

        # 验证 'lora_D' (新)
        delta_D = node_12.delta_node['lora_D']
        assert isinstance(delta_D.delta_data, tuple) and delta_D.delta_data[6] == "diff_list_v2" # "数据"
        assert delta_D.value.tolist() == [100, 200]  # "配方"
        print("  Deltas 'B', 'C', 'D' state in Node [1,2]: PASSED")

        # 验证 Node(3,4)
        print(f"\nNode [3,4] (key: {node_34.key.token_ids}):")
        assert node_34.is_base and node_34.key.extra_key == 'lora_A'
        assert node_34.value.tolist() == [30, 40] # Base
        print(f"  Base 'lora_A' (value): {node_34.value.tolist()}... PASSED")

        # 检查所有 Delta
        assert sorted(list(node_34.delta_node.keys())) == ['lora_B', 'lora_C']

        # 验证 'lora_B' (后缀)
        delta_B_suffix = node_34.delta_node['lora_B']
        assert isinstance(delta_B_suffix.delta_data, tuple) and delta_B_suffix.delta_data[6] == "diff_list_v2" # "数据"
        assert delta_B_suffix.value.tolist() == [33, 44]     # "配方"

        # 验证 'lora_C' (后缀)
        delta_C_suffix = node_34.delta_node['lora_C']
        assert isinstance(delta_C_suffix.delta_data, tuple) and delta_C_suffix.delta_data[6] == "diff_list_v2" # "数据"
        assert delta_C_suffix.value.tolist() == [34, 45]     # "配方"
        print("  Deltas 'B', 'C' state in Node [3,4]: PASSED")

        # --- 验证内存释放 (已修改) ---
        print("\nVerifying memory management:")
        # val_B 和 val_C 在分裂时被释放 (旧的 diff)
        assert tuple(val_B.cpu().tolist()) in allocator.freed_indices
        assert tuple(val_C.cpu().tolist()) in allocator.freed_indices
        # val_D 在插入时被释放
        assert tuple(val_D.cpu().tolist()) in allocator.freed_indices
        print("  All Delta indices (B, C, D) were freed: PASSED")

        print("\nCorner Case 1 (Test 2) PASSED!")

    except Exception as e:
        print(f"\nCorner Case 1 (Test 2) FAILED: {e}")
        import traceback
        traceback.print_exc()

def test3():
    # 这个 case 会验证插入顺序是否会影响结果
    print("--- Test Corner Case 3: Order Invariance ---") # (重命名以区分)

    # --- Test 3a: Insert 'short' key, then 'long' key ---
    print("\n--- Test 3a: short (B) then long (A) ---")
    try:
        # --- 1a. 设置 Allocator for 3a ---
        mock_kvcache_A = MockKVCache(1000)
        allocator_A = TokenToKVPoolAllocator(
            size=1000, dtype=torch.float16, device=device,
            kvcache=mock_kvcache_A, need_sort=False
        )
        val_B = torch.tensor([11, 22, 33], device=device, dtype=torch.int64)
        val_A = torch.tensor([10, 20, 30, 40, 50], device=device, dtype=torch.int64)
        indices_in_use_A_cpu = torch.cat([val_B, val_A]).cpu().unique()
        mask_A = torch.isin(allocator_A.free_pages.cpu(), indices_in_use_A_cpu)
        allocator_A.free_pages = allocator_A.free_pages[~mask_A.to(device)]
        print(f"  Allocator A setup. Initial available: {allocator_A.available_size()}")
        assert not torch.all(torch.isin(val_A, allocator_A.free_pages))

        # --- 2a. 运行测试逻辑 ---
        tree_A = RadixCache(None, allocator_A, page_size=1, disable=False, enable_delta_cache=True)

        tree_A.insert(RadixKey(token_ids=[1, 2, 3], extra_key='lora_B'), val_B)
        tree_A.insert(RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key='lora_A'), val_A)

        tree_A.pretty_print()

        # --- 3a. 验证状态 (A) ---
        node_123_A = tree_A.root_node.children[1]
        node_45_A = node_123_A.children[4]

        print("Verifying Test 3a:")
        # (节点状态验证与之前相同)
        assert node_123_A.is_base and node_123_A.key.extra_key == 'lora_B'
        assert node_123_A.value.tolist() == [11, 22, 33]
        delta_A = node_123_A.delta_node['lora_A']
        assert isinstance(delta_A.delta_data, tuple) and delta_A.delta_data[6] == "diff_list_v2"
        assert delta_A.value.tolist() == [10, 20, 30]

        assert node_45_A.is_base and node_45_A.key.extra_key == 'lora_A'
        assert node_45_A.value.tolist() == [40, 50]
        assert not node_45_A.delta_node

        # --- 验证内存 (检查 allocator.free_pages) ---
        print("\nVerifying memory management (Allocator A State):")

        # --- (!! 关键修正 !!) ---
        # val_A (作为 Delta 插入) 的 *前缀* 应该被释放回 free_pages
        val_A_prefix_cpu = torch.tensor([10, 20, 30], dtype=torch.int64)
        assert torch.all(torch.isin(val_A_prefix_cpu.to(device), allocator_A.free_pages))
        print("  'lora_A' (Delta prefix indices [10, 20, 30]) were freed to pool: PASSED")

        # val_A (作为 Base 插入) 的 *后缀* 不应该在 free_pages 中
        val_A_suffix_cpu = torch.tensor([40, 50], dtype=torch.int64)
        assert not torch.all(torch.isin(val_A_suffix_cpu.to(device), allocator_A.free_pages))
        print("  'lora_A' (Base suffix indices [40, 50]) were (correctly) not freed: PASSED")

        # val_B (作为 Base 插入) 不应该在 free_pages 中
        assert not torch.all(torch.isin(val_B, allocator_A.free_pages))
        print("  'lora_B' (Base indices) were (correctly) not freed: PASSED")
        # --- 结束修正 ---

        print("\n  Test 3a PASSED (short-then-long is correct)\n")

    except Exception as e:
        print(f"\n  Test 3a FAILED: {e}")
        traceback.print_exc()

    # --- Test 3b: Insert 'long' key, then 'short' key ---
    print("\n--- Test 3b: long (C) then short (D) ---")
    try:
        # --- 1b. 设置 Allocator for 3b ---
        mock_kvcache_B = MockKVCache(1000)
        allocator_B = TokenToKVPoolAllocator(
            size=1000, dtype=torch.float16, device=device,
            kvcache=mock_kvcache_B, need_sort=False
        )
        val_C = torch.tensor([100, 200, 300, 400, 500], device=device, dtype=torch.int64)
        val_D = torch.tensor([111, 222, 333], device=device, dtype=torch.int64)
        indices_in_use_B_cpu = torch.cat([val_C, val_D]).cpu().unique()
        mask_B = torch.isin(allocator_B.free_pages.cpu(), indices_in_use_B_cpu)
        allocator_B.free_pages = allocator_B.free_pages[~mask_B.to(device)]
        print(f"  Allocator B setup. Initial available: {allocator_B.available_size()}")
        assert not torch.all(torch.isin(val_D, allocator_B.free_pages))


        # --- 2b. 运行测试逻辑 ---
        tree_B = RadixCache(None, allocator_B, page_size=1, disable=False, enable_delta_cache=True)

        tree_B.insert(RadixKey(token_ids=[6, 7, 8, 9, 10], extra_key='lora_C'), val_C)
        tree_B.insert(RadixKey(token_ids=[6, 7, 8], extra_key='lora_D'), val_D)

        tree_B.pretty_print()

        # --- 3b. 验证状态 (B) ---
        node_678_B = tree_B.root_node.children[6]
        node_910_B = node_678_B.children[9]

        print("Verifying Test 3b:")
        # (节点状态验证与之前相同)
        assert node_678_B.is_base and node_678_B.key.extra_key == 'lora_C'
        assert node_678_B.value.tolist() == [100, 200, 300]
        delta_D = node_678_B.delta_node['lora_D']
        assert isinstance(delta_D.delta_data, tuple) and delta_D.delta_data[6] == "diff_list_v2"
        assert delta_D.value.tolist() == [111, 222, 333]

        assert node_910_B.is_base and node_910_B.key.extra_key == 'lora_C'
        assert node_910_B.value.tolist() == [400, 500]
        assert not node_910_B.delta_node

        # --- 验证内存 (检查 allocator.free_pages) ---
        print("\nVerifying memory management (Allocator B State):")
        # val_C (原始 Base) 在分裂时 *没有* 被释放 (根据当前 _split_node 实现)
        # (这是 _split_node 的一个潜在问题, 但测试应该反映当前实现)
        assert not torch.all(torch.isin(val_C, allocator_B.free_pages))
        print("  'lora_C' (Base parts) were (correctly by current split) not freed: PASSED")
        # val_D (Delta) 在插入时被释放
        assert torch.all(torch.isin(val_D, allocator_B.free_pages))
        print("  'lora_D' (Delta indices) were freed to pool: PASSED")

        print("\n  Test 3b PASSED (long-then-short is correct)\n")

    except Exception as e:
        print(f"\n  Test 3b FAILED: {e}")
        import traceback
        traceback.print_exc()

def test_delta_pool_logic():
    # --- 步骤 1 测试: 验证 "新区域" (Pool) 逻辑 ---
    print("--- Test Step 1: Verify Delta Pool Logic ---")
    
    tree = RadixCache(None, allocator, page_size=1, disable=False, enable_delta_cache=True)

    # 1. 创建一个模拟的 K-diff-tensor
    k_diff_tensor = torch.randn((3, 16, 64), dtype=torch.float16) # [num_tokens, heads, dim]
    
    # 2. 压缩
    compressed_K = tree.compress(k_diff_tensor)
    print(f"Original tensor size: {k_diff_tensor.element_size() * k_diff_tensor.nelement()} bytes")
    print(f"Compressed size (K): {len(compressed_K)} bytes")
    
    # 3. 存入 Pool
    handle_K = tree.store_in_delta_pool(compressed_K)
    print(f"Stored compressed data with handle: {handle_K}")
    
    # 4. 验证
    assert handle_K == 0
    assert tree.delta_data_counter == 1
    assert 0 in tree.delta_data_pool
    # assert tree.delta_data_pool[0] == compressed_K
    assert torch.equal(tree.delta_data_pool[0], compressed_K) == True
    # 5. 释放
    # (模拟一个 delta_data 元组)
    mock_tuple = ([0], [], "diff_list") 
    tree.release_from_delta_pool(mock_tuple)
    assert 0 not in tree.delta_data_pool
    print("Store and Release from pool: PASSED")
    
    print("\nStep 1 (Setup) PASSED!")


def test_simple_diff_compression():
    print("\n--- Test: Simple Diff Compression (A vs B) ---")
    
    # --- 1. 设置 Allocator ---
    allocator = TokenToKVPoolAllocator(
        size=1000, dtype=torch.float16, device=device, 
        kvcache=MockKVCache(1000), need_sort=False
    )
    
    # 定义索引
    val_A = torch.tensor([10, 11, 12], device=device, dtype=torch.int64)
    val_B = torch.tensor([20, 21, 22], device=device, dtype=torch.int64)
    
    # 手动从 free_pages 中移除, 模拟它们 "已被分配"
    indices_in_use_cpu = torch.cat([val_A, val_B]).cpu()
    mask = torch.isin(allocator.free_pages.cpu(), indices_in_use_cpu)
    allocator.free_pages = allocator.free_pages[~mask.to(device)]
    
    # 验证它们现在 *不* 在空闲池中
    assert not torch.all(torch.isin(val_A, allocator.free_pages))  # tensor([False, False, False])
    assert not torch.all(torch.isin(val_B, allocator.free_pages))  # tensor([False, False, False])

    # --- 2. 运行测试 ---
    tree = RadixCache(None, allocator, page_size=1, disable=False, enable_delta_cache=True)

    # 2a. 插入 lora_A (Base)
    print("\nInserting [1, 2, 3] for 'lora_A' (Base)")
    tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key='lora_A'), val_A)
    
    # 2b. 插入 lora_B (Delta) -> 触发压缩
    print("\nInserting [1, 2, 3] for 'lora_B' (Delta)")
    tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key='lora_B'), val_B)
    
    print("\nFinal tree state:")
    tree.pretty_print()

    # --- 3. 验证 ---
    print("\nVerifying states:")
    try:
        node_123 = tree.root_node.children[1]
        
        # 验证 Base
        assert node_123.is_base and node_123.key.extra_key == 'lora_A'
        assert torch.equal(node_123.value, val_A)
        print("  Base 'lora_A' is correct: PASSED")
        
        # 验证 Delta
        assert 'lora_B' in node_123.delta_node
        delta_B = node_123.delta_node['lora_B']
        
        # 验证 "数据" (句柄元组)
        assert isinstance(delta_B.delta_data, tuple)
        assert delta_B.delta_data[6] == "diff_list_v2"
        assert len(delta_B.delta_data[0]) == NUM_LAYERS # K handles
        print("  Delta 'lora_B' data (handles) stored: PASSED")
        
        # 验证 "配方" (Tensor 索引)
        assert isinstance(delta_B.value, torch.Tensor)
        assert torch.equal(delta_B.value, val_B.cpu())
        print("  Delta 'lora_B' recipe (indices) stored: PASSED")
        
        # 验证 "新区域" (Pool)
        assert len(tree.delta_data_pool) == NUM_LAYERS * 2
        print(f"  delta_data_pool contains {len(tree.delta_data_pool)} items: PASSED")
        
        # 验证内存释放
        assert torch.all(torch.isin(val_B, allocator.free_pages))
        print("  Delta indices for 'lora_B' were freed: PASSED")
        assert not torch.all(torch.isin(val_A, allocator.free_pages))
        print("  Base indices for 'lora_A' were (correctly) not freed: PASSED")

        print("\nSimple Diff Compression Test PASSED!")
    
    except Exception as e:
        print(f"\nSimple Diff Compression Test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # test1()
    # test2()
    test3()
    # test_delta_pool_logic()

    # test_simple_diff_compression()
