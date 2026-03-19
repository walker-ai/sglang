from enum import Enum
import hashlib
import os
import random
import time
from typing import Dict, Iterable, Optional

# [MobiLoRA Design 2] 定义应用的三种生命周期状态 
class AppState(Enum):
    FOREGROUND = 10.0  # 活跃状态，给予高分
    BACKGROUND = 1.0   # 后台状态，给予低分
    KILLED = 0.0       # 进程被杀，无价值，应尽快驱逐 [cite: 213]

# [MobiLoRA Design 2] 全局应用状态监控器 [cite: 211]
# 在实际集成中，你需要将 Android/iOS 的生命周期回调与此 Monitor 绑定
class AppStateMonitor:
    def __init__(
        self,
        update_interval_s: float = 1.0,
        state_weights: Optional[Dict[AppState, float]] = None,
        seed: Optional[int] = None,
        mode: Optional[str] = None,
        foreground_ttl_s: float = 2.0,
        background_ttl_s: float = 30.0,
    ):
        # 记录每个 app_id 的当前状态（用于手动覆盖或随机模式）
        self.app_states: Dict[str, AppState] = {}
        # 记录每个 app_id 的最后活跃时间（用于 trace 模式）
        self.app_last_seen: Dict[str, float] = {}
        self.update_interval_s = max(update_interval_s, 0.0)
        self.timeline_interval_s = max(update_interval_s, 1e-6)
        self._last_update_time = 0.0
        self._rng = random.Random(seed)
        self._timeline_seed = int(seed) if seed is not None else 0
        self._state_weights = state_weights or {
            AppState.FOREGROUND: 0.6,
            AppState.BACKGROUND: 0.3,
            AppState.KILLED: 0.1,
        }
        self.mode = self._normalize_mode(mode or os.getenv("SGLANG_APP_STATE_MODE"))
        self.foreground_ttl_s = max(foreground_ttl_s, 0.0)
        self.background_ttl_s = max(background_ttl_s, 0.0)

        env_fg = os.getenv("SGLANG_APP_FOREGROUND_TTL_S")
        env_bg = os.getenv("SGLANG_APP_BACKGROUND_TTL_S")
        if env_fg is not None:
            try:
                self.foreground_ttl_s = max(float(env_fg), 0.0)
            except ValueError:
                pass
        if env_bg is not None:
            try:
                self.background_ttl_s = max(float(env_bg), 0.0)
            except ValueError:
                pass

    @staticmethod
    def _normalize_mode(mode: Optional[str]) -> str:
        if not mode:
            return "touch"
        mode = mode.lower()
        if mode in ("trace", "touch"):
            return "touch"
        if mode in ("timeline", "time"):
            return "timeline"
        if mode not in ("touch", "random", "timeline"):
            return "touch"
        return mode

    def update_state(self, app_id: str, state: AppState):
        self.app_states[app_id] = state

    def configure(
        self,
        mode: Optional[str] = None,
        update_interval_s: Optional[float] = None,
        state_weights: Optional[Dict[AppState, float]] = None,
        seed: Optional[int] = None,
        foreground_ttl_s: Optional[float] = None,
        background_ttl_s: Optional[float] = None,
    ) -> None:
        if mode is not None:
            self.mode = self._normalize_mode(mode)
        if update_interval_s is not None:
            self.update_interval_s = max(float(update_interval_s), 0.0)
            self.timeline_interval_s = max(float(update_interval_s), 1e-6)
        if state_weights is not None:
            self._state_weights = state_weights
        if seed is not None:
            self._rng = random.Random(seed)
            self._timeline_seed = int(seed)
        if foreground_ttl_s is not None:
            self.foreground_ttl_s = max(float(foreground_ttl_s), 0.0)
        if background_ttl_s is not None:
            self.background_ttl_s = max(float(background_ttl_s), 0.0)

    def maybe_update_states(self, now: Optional[float] = None):
        if self.mode != "random":
            return
        if self.update_interval_s <= 0:
            return
        if now is None:
            now = time.monotonic()
        if now - self._last_update_time < self.update_interval_s:
            return
        self._last_update_time = now
        self._random_update_states()

    def _random_update_states(self, app_ids: Optional[Iterable[str]] = None):
        if app_ids is None:
            app_ids = list(self.app_states.keys())
        if not app_ids:
            return
        states = list(AppState)
        weights = [self._state_weights.get(state, 1.0) for state in states]
        for app_id in app_ids:
            self.app_states[app_id] = self._rng.choices(states, weights=weights, k=1)[0]

    def touch(self, app_id: str, now: Optional[float] = None) -> None:
        if not app_id:
            return
        if self.mode != "touch":
            return
        if now is None:
            now = time.monotonic()
        self.app_last_seen[app_id] = now

    def _timeline_state(self, app_id: str, now: float) -> AppState:
        interval = max(self.timeline_interval_s, 1e-6)
        bucket = int(now / interval)
        digest = hashlib.sha256(
            f"{self._timeline_seed}:{app_id}:{bucket}".encode("utf-8")
        ).digest()
        seed_val = int.from_bytes(digest[:8], "little")
        rng = random.Random(seed_val)
        states = list(AppState)
        weights = [self._state_weights.get(state, 1.0) for state in states]
        return rng.choices(states, weights=weights, k=1)[0]

    def get_state(self, app_id: str) -> AppState:
        if app_id in self.app_states:
            return self.app_states[app_id]
        if self.mode == "random":
            self.maybe_update_states()
            if app_id not in self.app_states:
                self._random_update_states([app_id])
            return self.app_states.get(app_id, AppState.KILLED)
        if self.mode == "timeline":
            now = time.monotonic()
            return self._timeline_state(app_id, now)

        now = time.monotonic()
        last_seen = self.app_last_seen.get(app_id)
        if last_seen is None:
            return AppState.KILLED
        age = max(now - last_seen, 0.0)
        if age <= self.foreground_ttl_s:
            return AppState.FOREGROUND
        if age <= self.foreground_ttl_s + self.background_ttl_s:
            return AppState.BACKGROUND
        return AppState.KILLED

    def get_score(self, app_id: str) -> float:
        return self.get_state(app_id).value

# 全局单例，供 Cache 访问
global_app_monitor = AppStateMonitor()
