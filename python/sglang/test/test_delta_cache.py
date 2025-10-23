import torch
from sglang.srt.mem_cache.radix_cache import RadixKey, RadixCache

def test1():
    # --- 测试: 启用差分缓存 ---
    print("--- Test 1: Delta Cache ENabled (page_size=1) ---")

    tree = RadixCache(None, None, page_size=1, disable=False, enable_delta_cache=True)

    # 1. 插入 lora_A (将成为 base)
    print("\nInserting [1, 2, 3, 4, 5] for 'lora_A'")
    key_A = RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key='lora_A')
    val_A = torch.tensor([10, 20, 30, 40, 50])
    tree.insert(key_A, val_A)
    tree.pretty_print()
    
    # 2. 插入 lora_B (将成为 delta)
    print("\nInserting [1, 2, 3, 4, 5] for 'lora_B'")
    key_B = RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key='lora_B')
    val_B = torch.tensor([11, 22, 33, 44, 55])
    tree.insert(key_B, val_B)
    
    print("\nTree after lora_B (structure should be unchanged, but deltas added):")
    tree.pretty_print()

    # 3a. 插入 lora_A 的 [1,2,3] 版本 (覆盖)
    tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key='lora_A'), torch.tensor([10, 20, 30]))
    print("\nTree after [1, 2, 3] (should split [1,2,3,4,5] node):")
    tree.pretty_print()

    # 3b. 插入 lora_C (导致分裂)
    print("\nInserting [1, 2, 3] for 'lora_C'")
    key_C = RadixKey(token_ids=[1, 2, 3], extra_key='lora_C')
    val_C = torch.tensor([100, 200, 300])
    
    tree.insert(key_C, val_C)
    print("\nTree after [1, 2, 3] for 'lora_C")
    tree.pretty_print()
    
    
    # 验证分裂后的节点状态
    print("\nVerifying node states:")
    try:
        node_123 = tree.root_node.children[1]
        node_45 = node_123.children[4]
        
        print(f"Node [1,2,3] (key: {node_123.key.token_ids}):")
        print(f"  is_base: {node_123.is_base}, base_key: {node_123.key.extra_key}")
        print(f"  base_value: {node_123.value.tolist()}")
        print(f"  delta_keys: {list(node_123.delta_node.keys())}")
        
        # 验证 lora_C 的 delta
        delta_C = node_123.delta_node['lora_C']
        print(f"  delta 'lora_C' value: {delta_C.delta_data.tolist()}")
        assert delta_C.delta_data.tolist() == val_C.tolist()
        
        print(f"\nNode [4,5] (key: {node_45.key.token_ids}):")
        print(f"  is_base: {node_45.is_base}, base_key: {node_45.key.extra_key}")
        print(f"  base_value: {node_45.value.tolist()}")
        print(f"  delta_keys: {list(node_45.delta_node.keys())}")
        
        # 验证 lora_B 的 delta 是否被正确传递到
        delta_B_suffix = node_45.delta_node['lora_B']
        print(f"  delta 'lora_B' suffix value: {delta_B_suffix.delta_data.tolist()}")
        
        # lora_B 的 [4,5] 部分
        assert delta_B_suffix.delta_data.tolist() == [44, 55] 

        print("\nStep 2 Test PASSED!")
    
    except Exception as e:
        print(f"\nStep 2 Test FAILED: {e}")
        import traceback
        traceback.print_exc()

def test2():
    """corner case"""
    print("--- Test Corner Case 2: Multi-Delta Split ---")
    
    tree = RadixCache(None, None, page_size=1, disable=False, enable_delta_cache=True)

    # 1. 插入 Base 和 2 个 Deltas (全长)
    val_A = torch.tensor([10, 20, 30, 40])
    tree.insert(RadixKey(token_ids=[1, 2, 3, 4], extra_key='lora_A'), val_A)
    
    val_B = torch.tensor([11, 22, 33, 44])
    tree.insert(RadixKey(token_ids=[1, 2, 3, 4], extra_key='lora_B'), val_B)
    
    val_C = torch.tensor([12, 23, 34, 45])
    tree.insert(RadixKey(token_ids=[1, 2, 3, 4], extra_key='lora_C'), val_C)
    
    print("\nTree before split:")
    tree.pretty_print()

    # 2. 插入一个部分 Delta，触发分裂
    val_D = torch.tensor([100, 200])
    tree.insert(RadixKey(token_ids=[1, 2], extra_key='lora_D'), val_D)
    
    print("\nTree after split:")
    tree.pretty_print()

    # 3. 验证状态
    print("\nVerifying node states:")
    try:
        node_12 = tree.root_node.children[1]
        node_34 = node_12.children[3]
        
        # 验证 Node(1,2)
        print(f"Node [1,2] (key: {node_12.key.token_ids}):")
        print(f"  Base: {node_12.key.extra_key}")
        print(f"  Base Value: {node_12.value.tolist()}")
        print(f"  Delta Keys: {sorted(list(node_12.delta_node.keys()))}")
        
        assert node_12.value.tolist() == [10, 20]
        assert sorted(list(node_12.delta_node.keys())) == ['lora_B', 'lora_C', 'lora_D']
        assert node_12.delta_node['lora_B'].delta_data.tolist() == [11, 22]
        assert node_12.delta_node['lora_D'].delta_data.tolist() == [100, 200]

        # 验证 Node(3,4)
        print(f"\nNode [3,4] (key: {node_34.key.token_ids}):")
        print(f"  Base: {node_34.key.extra_key}")
        print(f"  Base Value: {node_34.value.tolist()}")
        print(f"  Delta Keys: {sorted(list(node_34.delta_node.keys()))}")
        
        assert node_34.value.tolist() == [30, 40]
        assert sorted(list(node_34.delta_node.keys())) == ['lora_B', 'lora_C']
        assert node_34.delta_node['lora_B'].delta_data.tolist() == [33, 44]
        assert node_34.delta_node['lora_C'].delta_data.tolist() == [34, 45]

        print("\nCorner Case 1 Test PASSED!")
    
    except Exception as e:
        print(f"\nCorner Case 1 Test FAILED: {e}")
        import traceback
        traceback.print_exc()

def test3():
    tree = RadixCache(None, None, page_size=1, disable=False, enable_delta_cache=True)
    tree.insert(RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key='lora_B'), torch.tensor([11, 22, 33, 44, 55]))
    tree.insert(RadixKey(token_ids=[1, 2, 3], extra_key='lora_A'), torch.tensor([10, 20, 30]))
    

    tree.pretty_print()


    print(123)


if __name__ == "__main__":
    # test1()
    # test2()
    test3()