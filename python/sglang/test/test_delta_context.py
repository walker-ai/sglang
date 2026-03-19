import unittest
import torch
import time

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

from sglang.srt.mem_cache.app_state_monitor import AppState, global_app_monitor

from sglang.srt.mem_cache.radix_cache import (
    RadixCache, 
    RadixKey, 
    TreeNode
)

from sglang.srt.mem_cache.evict_policy import MobiLoRAEvictionStrategy

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


class TestMobiLoRAContext(unittest.TestCase):

    def setUp(self):
        """每个测试用例开始前的初始化"""
        # 1. 重置全局 Monitor
        global_app_monitor.app_states.clear()
        global_app_monitor.update_interval_s = 0.0
        
        # 2. 初始化 Cache，启用 MobiLoRA 策略
        self.kv_cache = kvcache
        self.allocator = allocator

        self.tree_cache = RadixCache(None, allocator, page_size=1, disable=False, enable_delta_cache=True, compression_backend="cuszp", eviction_policy="mobilora")
        
        # 3. 设置一些测试用的 App ID
        self.app_fg = "com.app.foreground"
        self.app_bg = "com.app.background"
        self.app_killed = "com.app.killed"
        
        # 4. 初始化它们的状态
        self.tree_cache.global_app_monitor.update_state(self.app_fg, AppState.FOREGROUND)
        self.tree_cache.global_app_monitor.update_state(self.app_bg, AppState.BACKGROUND)
        self.tree_cache.global_app_monitor.update_state(self.app_killed, AppState.KILLED)

    def test_01_insert_context_association(self):
        """测试：插入节点时，是否正确关联了 App ID"""
        key = RadixKey([1, 2, 3], app_id=self.app_fg)
        value = torch.tensor([1, 2, 3])
        
        self.tree_cache.insert(key, value)
        
        # 获取叶子节点
        # 路径: Root -> [1, 2, 3] (假设没分裂)
        # 注意：RadixTree 的结构取决于实现，这里假设一次性插入且无分裂
        # 我们通过 match_prefix 找到最后节点来验证
        match_res = self.tree_cache.match_prefix(key)
        last_node = match_res.last_host_node
        
        print(f"\n[Test 01] Node associated apps: {last_node.associated_apps}")
        
        # 验证：节点必须包含 self.app_fg
        self.assertIn(self.app_fg, last_node.associated_apps)
        # 验证：节点不应该包含其他 App
        self.assertNotIn(self.app_bg, last_node.associated_apps)

    def test_02_path_inheritance_on_split(self):
        """测试：节点分裂时，父节点是否继承了子节点的 App 关联"""
        # 1. App A 插入 [1, 2, 3, 4]
        key_a = RadixKey([1, 2, 3, 4], app_id="AppA")
        self.tree_cache.insert(key_a, torch.tensor([1, 2, 3, 4]))
        
        # 2. App B 插入 [1, 2, 5, 6] -> 这会导致在 [1, 2] 处分裂
        key_b = RadixKey([1, 2, 5, 6], app_id="AppB")
        self.tree_cache.insert(key_b, torch.tensor([1, 2, 5, 6]))
        
        # 3. 验证公共父节点 [1, 2]
        # 我们构造一个 key [1, 2] 去查找
        match_res = self.tree_cache.match_prefix(RadixKey([1, 2], app_id="AppC"))
        parent_node = match_res.last_host_node
        
        print(f"\n[Test 02] Parent Node Key: {parent_node.key.token_ids}")
        print(f"[Test 02] Parent associated apps: {parent_node.associated_apps}")
        
        # 验证：公共父节点必须同时包含 AppA 和 AppB
        # 因为 AppA 用过 [1,2,3,4]，意味着它也用过 [1,2]
        self.assertIn("AppA", parent_node.associated_apps)
        self.assertIn("AppB", parent_node.associated_apps)
        # 顺便验证当前查找的 AppC 也被加上了
        self.assertIn("AppC", parent_node.associated_apps)

    def test_03_shared_node_update(self):
        """测试：多个 App 访问同一节点，associated_apps 是否正确累加"""
        key = RadixKey([10, 11], app_id="App_1")
        self.tree_cache.insert(key, torch.tensor([10, 11]))
        
        # App_2 再次访问/插入相同的 Key
        key2 = RadixKey([10, 11], app_id="App_2")
        self.tree_cache.insert(key2, torch.tensor([10, 11]))
        
        match_res = self.tree_cache.match_prefix(key)
        node = match_res.last_host_node
        
        self.assertEqual(len(node.associated_apps), 2)
        self.assertTrue({"App_1", "App_2"}.issubset(node.associated_apps))

    def test_04_eviction_strategy_calculation(self):
        """测试：Eviction 策略是否能正确区分前台、后台和 Killed 应用"""
        strategy = self.tree_cache.eviction_strategy
        
        # 模拟三个节点
        node_fg = TreeNode()
        node_fg.value = [1] * 10 # 长度 10
        node_fg.associated_apps.add(self.app_fg) # 前台
        
        node_bg = TreeNode()
        node_bg.value = [1] * 10
        node_bg.associated_apps.add(self.app_bg) # 后台
        
        node_killed = TreeNode()
        node_killed.value = [1] * 10
        node_killed.associated_apps.add(self.app_killed) # Killed
        
        # 计算分数
        # 注意：EvictionStrategy 依赖 last_access_time，我们先强制设为相同以排除干扰
        now = time.monotonic()
        node_fg.last_access_time = now
        node_bg.last_access_time = now
        node_killed.last_access_time = now
        
        score_fg = strategy.get_priority(node_fg)
        score_bg = strategy.get_priority(node_bg)
        score_killed = strategy.get_priority(node_killed)
        
        print(f"\n[Test 04] Scores -> FG: {score_fg:.4f}, BG: {score_bg:.4f}, Killed: {score_killed:.4f}")
        
        # 验证：前台分数 > 后台分数 > Killed 分数
        self.assertGreater(score_fg, score_bg)
        self.assertGreater(score_bg, score_killed)

    def test_05_dynamic_state_change(self):
        """测试：当 Monitor 中状态改变时，节点的计算分数是否实时变化"""
        strategy = self.tree_cache.eviction_strategy
        
        node = TreeNode()
        node.value = [1] * 5
        app_id = "dynamic.app"
        node.associated_apps.add(app_id)
        node.last_access_time = time.monotonic()
        
        # 1. 初始状态：FOREGROUND
        global_app_monitor.update_state(app_id, AppState.FOREGROUND)
        score_1 = strategy.get_priority(node)
        
        # 2. 状态变更为：KILLED
        global_app_monitor.update_state(app_id, AppState.KILLED)
        score_2 = strategy.get_priority(node)
        
        print(f"\n[Test 05] Score change: {score_1:.4f} -> {score_2:.4f}")
        
        # 验证：分数应该大幅下降
        self.assertGreater(score_1, score_2)
        
        # 验证具体的数值逻辑 (假设 lambda_s=10, killed=0)
        # log(0 + 1) = 0，第一项应该变为 0
        # 只有时间分和长度分保留

    def test_06_submodularity_diminishing_returns(self):
        """测试：次模性（边际效益递减）。多个前台 App 共享一个节点，分数的增长应该是递减的"""
        strategy = self.tree_cache.eviction_strategy
        
        node = TreeNode()
        node.value = [1] * 10
        node.last_access_time = time.monotonic()
        
        # 0 个 App
        score_0 = strategy.get_priority(node)
        
        # 1 个前台 App
        app1 = "app1"
        global_app_monitor.update_state(app1, AppState.FOREGROUND)
        node.associated_apps.add(app1)
        score_1 = strategy.get_priority(node)
        
        # 2 个前台 App
        app2 = "app2"
        global_app_monitor.update_state(app2, AppState.FOREGROUND)
        node.associated_apps.add(app2)
        score_2 = strategy.get_priority(node)
        
        gain_1 = score_1 - score_0
        gain_2 = score_2 - score_1
        
        print(f"\n[Test 06] Gain 1st App: {gain_1:.4f}")
        print(f"[Test 06] Gain 2nd App: {gain_2:.4f}")
        
        # 验证：第一个 App 带来的增益 应该大于 第二个 App 带来的增益 (由于 log 函数)
        self.assertGreater(gain_1, gain_2)

if __name__ == '__main__':
    unittest.main()
