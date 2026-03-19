from __future__ import annotations

from abc import ABC, abstractmethod
import math
import time
from typing import Callable, TYPE_CHECKING, Tuple, Union

from sglang.srt.mem_cache.app_state_monitor import AppState, global_app_monitor

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.creation_time


PhiType = Union[str, Callable[[float], float]]


class MobiLoRAEvictionStrategy(EvictionStrategy):
    def __init__(
        self,
        lambda_s: float = 1.0,
        lambda_t: float = 1.0,
        lambda_l: float = 1.0,
        phi_s: PhiType | None = None,
        phi_t: PhiType | None = None,
        phi_l: PhiType | None = None,
        time_decay_tau: float = 1.0,
        length_norm: float = 1.0,
        app_ttl_s: float = 30.0,
        app_agg: str = "sum",
    ):
        self.lambda_s = float(lambda_s)
        self.lambda_t = float(lambda_t)
        self.lambda_l = float(lambda_l)
        self.phi_s = self._resolve_phi(phi_s)
        self.phi_t = self._resolve_phi(phi_t)
        self.phi_l = self._resolve_phi(phi_l)
        self.time_decay_tau = max(float(time_decay_tau), 1e-6)
        self.length_norm = max(float(length_norm), 1e-6)
        self.app_ttl_s = max(float(app_ttl_s), 0.0)
        self.app_agg = "max" if str(app_agg).lower() == "max" else "sum"

    @staticmethod
    def _resolve_phi(phi: PhiType | None) -> Callable[[float], float]:
        if phi is None:
            return math.log1p
        if callable(phi):
            return phi
        choice = str(phi).lower()
        if choice == "identity":
            return lambda x: float(x)
        if choice == "sqrt":
            return lambda x: math.sqrt(max(x, 0.0))
        if choice == "log1p":
            return math.log1p
        return math.log1p

    def _app_state_sum(self, node: "TreeNode") -> float:
        scores = []
        last_seen = getattr(node, "associated_app_last_seen", None)
        now = time.monotonic()
        if last_seen:
            for app_id, ts in last_seen.items():
                age = max(now - ts, 0.0)
                if self.app_ttl_s > 0 and age > self.app_ttl_s:
                    continue
                decay = 1.0
                if self.app_ttl_s <= 0:
                    denom = 1.0 + (age / max(self.time_decay_tau, 1e-6))
                    decay = 1.0 / denom
                state = global_app_monitor.get_state(app_id)
                score = float(state.value) - float(AppState.BACKGROUND.value)
                scores.append(score * decay)
        if not scores:
            associated_apps = getattr(node, "associated_apps", None)
            if associated_apps:
                for app_id in associated_apps:
                    state = global_app_monitor.get_state(app_id)
                    score = float(state.value) - float(AppState.BACKGROUND.value)
                    scores.append(score)
        if not scores:
            return 0.0
        if self.app_agg == "max":
            return max(scores)
        return sum(scores)

    @staticmethod
    def _apply_phi_signed(phi: Callable[[float], float], value: float) -> float:
        if value < 0.0:
            return -phi(-value)
        return phi(value)

    def _lru_score(self, node: "TreeNode") -> float:
        now = time.monotonic()
        age = max(now - node.last_access_time, 0.0)
        return 1.0 / (1.0 + (age / self.time_decay_tau))

    def _length_score(self, node: "TreeNode") -> float:
        if getattr(node, "value", None) is None:
            return 0.0
        value = node.value
        if hasattr(value, "numel"):
            length = float(value.numel())
        else:
            length = float(len(value))
        return length / self.length_norm

    def get_priority(self, node: "TreeNode") -> float:
        state_score = self._app_state_sum(node)
        lru_score = self._lru_score(node)
        length_score = self._length_score(node)
        state_term = self._apply_phi_signed(self.phi_s, state_score)
        return (
            self.lambda_s * state_term
            + self.lambda_t * self.phi_t(lru_score)
            + self.lambda_l * self.phi_l(length_score)
        )

    def debug_components(
        self, node: "TreeNode"
    ) -> Tuple[float, float, float, float, float, float, float]:
        state_score = self._app_state_sum(node)
        lru_score = self._lru_score(node)
        length_score = self._length_score(node)
        phi_s_val = self._apply_phi_signed(self.phi_s, state_score)
        phi_t_val = self.phi_t(lru_score)
        phi_l_val = self.phi_l(length_score)
        utility = (
            self.lambda_s * phi_s_val
            + self.lambda_t * phi_t_val
            + self.lambda_l * phi_l_val
        )
        return (
            state_score,
            lru_score,
            length_score,
            phi_s_val,
            phi_t_val,
            phi_l_val,
            utility,
        )


class HiCacheAwareEvictionStrategy(MobiLoRAEvictionStrategy):
    """
    HiCache-aware eviction policy.

    It builds on MobiLoRA scoring and adds:
    - extra app-state bias to keep foreground data on GPU longer
    - backup bonus to evict nodes already backed up to host/storage first
    """

    def __init__(
        self,
        *args,
        hicache_state_weight: float = 2.0,
        backuped_bonus: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hicache_state_weight = float(hicache_state_weight)
        self.backuped_bonus = float(backuped_bonus)

    def get_priority(self, node: "TreeNode") -> float:
        base_priority = super().get_priority(node)
        state_score = self._app_state_sum(node)
        priority = base_priority + self.hicache_state_weight * state_score
        if getattr(node, "backuped", False):
            priority -= self.backuped_bonus * (1.0 + self._length_score(node))
        return priority
