from __future__ import annotations

import logging
import signal
import sys
import time
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.utils.common import disable_request_logging, pyspy_dump_schedulers

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerRuntimeCheckerMixin:

    def _check_hybrid_memory(self: Scheduler):
        (
            full_num_used,
            swa_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        ) = self._get_swa_token_info()
        memory_leak = full_num_used != 0 or swa_num_used != 0
        token_msg = (
            f"{self.full_tokens_per_layer=}, {full_available_size=}, {full_evictable_size=}, {self.tree_cache.full_protected_size()=}\n"
            f"{self.swa_tokens_per_layer=}, {swa_available_size=}, {swa_evictable_size=}, {self.tree_cache.swa_protected_size()=}\n"
        )
        return memory_leak, token_msg

    def _check_mamba_memory(self: Scheduler):
        (
            full_num_used,
            mamba_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        ) = self._get_mamba_token_info()
        memory_leak = (
            full_num_used != self.tree_cache.full_protected_size()
            or mamba_num_used != self.tree_cache.mamba_protected_size()
        )
        token_msg = (
            f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}\n"
            f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}\n"
        )
        return memory_leak, token_msg

    def _estimate_inflight_tokens(self: Scheduler) -> int:
        """Estimate tokens currently allocated but not yet reflected in tree/protected sizes."""
        current_batch: ScheduleBatch = self.last_batch
        running_batch: ScheduleBatch = getattr(self, "running_batch", None)

        extend_size = 0
        if current_batch is not None:
            for req in current_batch.reqs:
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
                fill_len = len(req.fill_ids) if req.fill_ids is not None else 0
                prefix_len = len(req.prefix_indices) if req.prefix_indices is not None else 0

                if current_batch.forward_mode.is_decode():
                    if req.finished():
                        unreleased_len = 1
                    else:
                        unreleased_len = seq_len - prefix_len
                else:
                    unreleased_len = fill_len - prefix_len

                extend_size += unreleased_len

        if (
            current_batch is not None
            and current_batch.forward_mode.is_extend()
            and running_batch is not None
            and not running_batch.is_empty()
            and running_batch.forward_mode.is_decode()
        ):
            for req in running_batch.reqs:
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
                prefix_len = len(req.prefix_indices) if req.prefix_indices is not None else 0

                if req.finished():
                    unreleased_len = 0
                else:
                    unreleased_len = seq_len - prefix_len - 1

                extend_size += unreleased_len

        return extend_size

    def _check_radix_cache_memory(self: Scheduler):
        # Always recompute evictable from the tree to avoid stale counters.
        if hasattr(self.tree_cache, "recompute_evictable_size"):
            try:
                evictable_size = self.tree_cache.recompute_evictable_size()
            except Exception:
                evictable_size = self.tree_cache.evictable_size()
        else:
            evictable_size = self.tree_cache.evictable_size()

        protected_size = self.tree_cache.protected_size()
        extend_size = self._estimate_inflight_tokens()
        # Derive available by subtracting known used/protected from capacity.
        available_size = self.max_total_num_tokens - (evictable_size + protected_size)
        # Use derived accounting for leak check.
        memory_leak = (available_size + evictable_size + protected_size + extend_size) != (
            self.max_total_num_tokens
        )
        if memory_leak:
            locked_nodes = []
            value_nodes = []

            def _collect(node):
                for child in node.children.values():
                    if getattr(child, "lock_ref", 0) > 0:
                        locked_nodes.append(
                            {
                                "len": len(child.key),
                                "lock_ref": child.lock_ref,
                                "evicted": bool(getattr(child, "value", None) is None),
                                "is_base": getattr(child, "is_base", False),
                                "extra": getattr(child.key, "extra_key", None),
                            }
                        )
                    val = getattr(child, "value", None)
                    if val is not None:
                        value_nodes.append(
                            {
                                "len": len(val),
                                "key_len": len(child.key),
                                "extra": getattr(child.key, "extra_key", None),
                                "node_id": getattr(child, "id", None),
                            }
                        )
                    _collect(child)

            _collect(self.tree_cache.root_node)
            recomputed_evictable = sum(v["len"] for v in value_nodes)

            # Compare allocator vs tree index sets to locate missing/extra indices.
            missing = []
            extra = []
            dup_count = -1
            free_len = release_len = used_est = -1
            try:
                free_pages = getattr(self.token_to_kv_pool_allocator, "free_pages", None)
                release_pages = getattr(self.token_to_kv_pool_allocator, "release_pages", None)
                free_list = []
                if free_pages is not None:
                    free_list.extend(free_pages.tolist())
                if release_pages is not None:
                    free_list.extend(release_pages.tolist())
                free_set = set(int(x) for x in free_list)
                free_len = len(free_set)
                used_set_est = set(range(1, self.token_to_kv_pool_allocator.size + 1)) - free_set
                used_est = len(used_set_est)

                tree_indices = []
                # collect actual tensor refs to avoid double conversion
                for node in self.tree_cache.root_node.children.values():
                    pass
                def _collect_indices(node):
                    val = getattr(node, "value", None)
                    if val is not None:
                        try:
                            tree_indices.extend(int(i) for i in val.tolist())
                        except Exception:
                            pass
                    for ch in node.children.values():
                        _collect_indices(ch)
                _collect_indices(self.tree_cache.root_node)
                tree_set = set(tree_indices)
                missing = list(used_set_est - tree_set)
                extra = list(tree_set - used_set_est)
                dup_count = len(tree_indices) - len(tree_set)
            except Exception:
                pass
            # 限制日志长度，最多展示前 50 个节点
            value_preview = value_nodes[:50]
            locked_preview = locked_nodes[:50]
            logger.error(
                "KV pool accounting mismatch: "
                f"{self.max_total_num_tokens=}, {available_size=}, {evictable_size=}, {protected_size=}, {extend_size=}, "
                f"tree_evictable={getattr(self.tree_cache, 'evictable_size_', 'n/a')}, "
                f"tree_protected={getattr(self.tree_cache, 'protected_size_', 'n/a')}, "
                f"locked_nodes(total={len(locked_nodes)}): {locked_preview}, "
                f"value_nodes(total={len(value_nodes)}): {value_preview}, "
                f"recomputed_evictable={recomputed_evictable}, dup_in_tree={dup_count}, "
                f"missing_in_tree_sample={missing[:20]}, extra_in_tree_sample={extra[:20]}, "
                f"free_total={free_len}, used_est={used_est}"
            )
        token_msg = (
            f"{self.max_total_num_tokens=}, {available_size=}, "
            f"{evictable_size=}, {protected_size=}, {extend_size=}\n"
        )
        return memory_leak, token_msg

    def _check_runtime_mem_leak(self: Scheduler):
        current_batch: ScheduleBatch = self.last_batch

        if current_batch is None:
            return

        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()

        extend_size = 0
        for i, req in enumerate(current_batch.reqs):
            seq_len = len(req.origin_input_ids) + len(req.output_ids)
            fill_len = len(req.fill_ids) if req.fill_ids is not None else 0
            prefix_len = (
                len(req.prefix_indices) if req.prefix_indices is not None else 0
            )

            if current_batch.forward_mode.is_decode():
                if req.finished():
                    unreleased_len = 1
                else:
                    unreleased_len = seq_len - prefix_len
            else:
                unreleased_len = fill_len - prefix_len

            extend_size += unreleased_len

        if (
            current_batch.forward_mode.is_extend()
            and self.running_batch is not None
            and not self.running_batch.is_empty()
            and self.running_batch.forward_mode.is_decode()
        ):
            for i, req in enumerate(self.running_batch.reqs):
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
                prefix_len = (
                    len(req.prefix_indices) if req.prefix_indices is not None else 0
                )

                if req.finished():
                    unreleased_len = 0
                else:
                    unreleased_len = seq_len - prefix_len - 1

                extend_size += unreleased_len

        total_tokens = available_size + evictable_size + protected_size + extend_size

        assert (
            total_tokens == self.max_total_num_tokens
        ), f"Mem Leak Detected! {total_tokens=} vs {self.max_total_num_tokens=}"

    def _check_req_pool(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        if len(self.req_to_token_pool.free_slots) != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise ValueError(msg)

    def check_memory(self: Scheduler):
        if self.is_hybrid:
            memory_leak, token_msg = self._check_hybrid_memory()
        elif self.is_hybrid_gdn and isinstance(self.tree_cache, MambaRadixCache):
            memory_leak, token_msg = self._check_mamba_memory()
        else:
            memory_leak, token_msg = self._check_radix_cache_memory()

        if memory_leak:
            # Best-effort pruning of stale tree nodes pointing to already-freed indices.
            try:
                import torch

                free_pages = getattr(self.token_to_kv_pool_allocator, "free_pages", None)
                release_pages = getattr(self.token_to_kv_pool_allocator, "release_pages", None)
                free_list = []
                if free_pages is not None:
                    free_list.extend(free_pages.tolist())
                if release_pages is not None:
                    free_list.extend(release_pages.tolist())
                free_set = set(int(x) for x in free_list)
                if hasattr(self.tree_cache, "prune_stale_values"):
                    self.tree_cache.prune_stale_values(free_set)
                    # Re-evaluate after pruning
                    _, _, available_size, evictable_size = self._get_token_info()
                    protected_size = self.tree_cache.protected_size()
                    if (available_size + evictable_size) == (
                        self.max_total_num_tokens - protected_size
                    ):
                        return
            except Exception:
                logger.exception("Failed to prune stale values during memory check")

            msg = "token_to_kv_pool_allocator memory leak detected! " f"{token_msg}"
            raise ValueError(msg)

        self._check_req_pool()

        if (
            self.enable_metrics
            and self.current_scheduler_metrics_enabled()
            and time.perf_counter() > self.metrics_collector.last_log_time + 30
        ):
            # During idle time, also collect metrics every 30 seconds.
            if self.is_hybrid:
                (
                    full_num_used,
                    swa_num_used,
                    full_token_usage,
                    swa_token_usage,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_swa_token_info()
                num_used = max(full_num_used, swa_num_used)
                token_usage = max(full_token_usage, swa_token_usage)
            elif self.is_hybrid_gdn:
                (
                    num_used,
                    _,
                    token_usage,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_mamba_token_info()
            else:
                num_used, token_usage, _, _ = self._get_token_info()
            num_running_reqs = len(self.running_batch.reqs)
            self.stats.num_running_reqs = num_running_reqs
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = round(token_usage, 2)
            self.stats.gen_throughput = 0
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.num_grammar_queue_reqs = len(self.grammar_queue)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )
            self.metrics_collector.log_stats(self.stats)
        self._publish_kv_events()

    def check_tree_cache(self: Scheduler):
        if (self.is_hybrid and isinstance(self.tree_cache, SWARadixCache)) or (
            self.is_hybrid_gdn and isinstance(self.tree_cache, MambaRadixCache)
        ):
            self.tree_cache.sanity_check()

    def self_check_during_idle(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)
            if queue_size:
                return

        self.check_memory()
        self.check_tree_cache()
        self.new_token_ratio = self.init_new_token_ratio
        self.maybe_sleep_on_idle()

    def watchdog_thread(self: Scheduler):
        """A watch dog thread that will try to kill the server itself if one forward batch takes too long."""
        self.watchdog_last_forward_ct = 0
        self.watchdog_last_time = time.perf_counter()

        while True:
            current = time.perf_counter()
            if self.cur_batch is not None:
                if self.watchdog_last_forward_ct == self.forward_ct:
                    if current > self.watchdog_last_time + self.watchdog_timeout:
                        break
                else:
                    self.watchdog_last_forward_ct = self.forward_ct
                    self.watchdog_last_time = current
            time.sleep(self.watchdog_timeout // 2)

        if not disable_request_logging():
            # Print batch size and memory pool info to check whether there are de-sync issues.
            if self.is_hybrid:
                _, info_msg = self._check_hybrid_memory()
            elif self.is_hybrid_gdn and isinstance(self.tree_cache, MambaRadixCache):
                _, info_msg = self._check_mamba_memory()
            else:
                _, info_msg = self._check_radix_cache_memory()
            logger.error(
                f"{self.cur_batch.batch_size()=}\n"
                f"{self.cur_batch.reqs=}\n"
                f"{info_msg}"
            )

        pyspy_dump_schedulers()
        logger.error(f"Watchdog timeout ({self.watchdog_timeout=})")
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        # Wait for some time so that the parent process can print the error.
        time.sleep(5)
        self.parent_process.send_signal(signal.SIGQUIT)
