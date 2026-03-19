import argparse
import asyncio
import json
import math
import queue
import random
import threading
import time
from datetime import datetime
from typing import Optional, List

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm

from sglang.bench_serving import (
    RequestFuncOutput,
    get_tokenizer,
    remove_prefix,
    sample_random_requests,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)

# ==========================================
# 1. 新增：LoRA 选择器，支持不同的分布规律
# ==========================================
class LoraSelector:
    def __init__(self, lora_paths: List[str], distribution: str = "uniform", zipf_a: float = 1.5, seed: int = 42, weights: Optional[List[float]] = None):
        self.lora_paths = lora_paths
        self.distribution = distribution
        self.zipf_a = zipf_a  # Zipf 分布的参数，越大越集中
        self.py_rng = random.Random(seed)
        # 如果选择 weighted 但未传权重，则默认平均分配
        if self.distribution == "weighted" and not weights:
            if lora_paths:
                self.weights = [1.0 / len(lora_paths)] * len(lora_paths)
                print(f"[LoraSelector] weighted 未提供权重，使用均匀权重 {self.weights}")
            else:
                self.weights = None
        else:
            self.weights = weights
        if self.weights and self.distribution != "weighted":
            self.distribution = "weighted"

        
        if not lora_paths:
            # 如果没提供 lora，就全返空字符串，退化为 Base Model
            self.lora_paths = [""]
        
        print(f"Load LoRA Selector: {len(self.lora_paths)} LoRAs, Distribution: {distribution}")

    def select(self) -> str:
        if not self.lora_paths or self.lora_paths == [""]:
            return ""

        if self.distribution == "uniform":
            return self.py_rng.choice(self.lora_paths)
        
        elif self.distribution == "zipf":
            idx = (np.random.zipf(self.zipf_a) - 1) % len(self.lora_paths)
            return self.lora_paths[idx]
        elif self.distribution == "weighted" and self.weights:
            choice = self.py_rng.choices(self.lora_paths, weights=self.weights, k=1)[0]
            return choice
        elif self.distribution == "weighted" and not self.weights:
            print("[Warning] weighted 分布缺少权重，退化为首个 LoRA")
            return self.lora_paths[0]
        
        else:
            return self.lora_paths[0]

# ==========================================
# 2. 新增：App Trace，模拟热点 App 活跃/闲置
# ==========================================
class AppTrace:
    def __init__(
        self,
        app_ids: List[str],
        mode: str = "static",
        hotset_size: int = 3,
        phase_requests: int = 50,
        phase_sleep_s: float = 0.0,
        inactive_killed_after_phases: int = 2,
        foreground_app_count: int = 1,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.app_ids = list(app_ids)
        self.mode = mode
        self.hotset_size = min(max(hotset_size, 1), len(self.app_ids))
        self.phase_requests = max(int(phase_requests), 1)
        self.phase_sleep_s = max(float(phase_sleep_s), 0.0)
        self._rng = random.Random(seed)
        self._requests_since_phase = 0
        self._phase_idx = 0
        self._active_apps: set = set()
        self._foreground_apps: set = set()
        self._last_active_phase = {app_id: 0 for app_id in self.app_ids}
        self.inactive_killed_after_phases = max(int(inactive_killed_after_phases), 0)
        self.foreground_app_count = max(int(foreground_app_count), 1)
        self.verbose = verbose

        if self.mode == "hotset":
            self._rotate(force=True)
        else:
            self._active_apps = set(self.app_ids)
            self._choose_foreground_apps()

    def _choose_foreground_apps(self) -> None:
        if not self._active_apps:
            self._foreground_apps = set()
            return
        ordered_active = sorted(self._active_apps)
        fg_count = min(self.foreground_app_count, len(ordered_active))
        self._foreground_apps = set(
            self._rng.sample(ordered_active, fg_count)
        )

    def _rotate(self, force: bool = False) -> bool:
        if self.mode != "hotset":
            return False
        if not force and self._requests_since_phase < self.phase_requests:
            return False
        self._requests_since_phase = 0
        self._phase_idx += 1
        if self.hotset_size >= len(self.app_ids):
            self._active_apps = set(self.app_ids)
        else:
            self._active_apps = set(self._rng.sample(self.app_ids, self.hotset_size))
        for app_id in self._active_apps:
            self._last_active_phase[app_id] = self._phase_idx
        self._choose_foreground_apps()
        if self.verbose:
            print(
                f"[AppTrace] phase={self._phase_idx} "
                f"active_apps={sorted(self._active_apps)} "
                f"foreground_apps={sorted(self._foreground_apps)}"
            )
        return True

    def is_app_active(self, app_id: str) -> bool:
        if self.mode != "hotset":
            return True
        return app_id in self._active_apps

    def _active_app_list(self) -> List[str]:
        if self.mode != "hotset":
            return list(self.app_ids)
        if not self._active_apps:
            return list(self.app_ids)
        return sorted(self._active_apps)

    def inactive_app_list(self) -> List[str]:
        if self.mode != "hotset":
            return []
        inactive = [app_id for app_id in self.app_ids if app_id not in self._active_apps]
        return inactive

    def active_non_foreground_list(self) -> List[str]:
        if self.mode != "hotset":
            return []
        return sorted(
            app_id
            for app_id in self._active_apps
            if app_id not in self._foreground_apps
        )

    def sample_active_app(self, rng: random.Random) -> str:
        candidates = self._active_app_list()
        return rng.choice(candidates)

    def sample_foreground_app(self, rng: random.Random) -> Optional[str]:
        if not self._foreground_apps:
            return None
        return rng.choice(sorted(self._foreground_apps))

    def get_app_states(self) -> dict:
        if self.mode != "hotset":
            if not self._foreground_apps:
                self._choose_foreground_apps()
            return {
                app_id: ("foreground" if app_id in self._foreground_apps else "background")
                for app_id in self.app_ids
            }
        states = {}
        for app_id in self.app_ids:
            if app_id in self._foreground_apps:
                state = "foreground"
            elif app_id in self._active_apps:
                state = "background"
            else:
                last_active = self._last_active_phase.get(app_id, 0)
                if (
                    self.inactive_killed_after_phases > 0
                    and self._phase_idx - last_active >= self.inactive_killed_after_phases
                ):
                    state = "killed"
                else:
                    state = "background"
            states[app_id] = state
        return states

    def on_request_sent(self) -> bool:
        if self.mode != "hotset":
            return False
        self._requests_since_phase += 1
        return self._rotate()

    def force_rotate(self) -> bool:
        return self._rotate(force=True)

# ==========================================
# 3. 参数解析修改
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark for Delta Cache with Dynamic LoRA Switching."
    )
    # --- 针对端侧/Delta Cache 的默认参数调整 ---
    parser.add_argument("--num-clients", type=int, default=1, help="模拟的用户数量 (端侧通常为 1)")
    parser.add_argument("--max-parallel", type=int, default=1, help="最大并发请求数 (端侧串行处理设为 1)")
    parser.add_argument("--num-rounds", type=int, default=10, help="每个用户进行的对话轮数")
    # ---------------------------------------
    
    parser.add_argument("--request-length", type=int, default=512)
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--output-log-file", type=str, default="")
    parser.add_argument("--stats-log-file", type=str, default="", help="路径：保存 print_stats 输出，便于区分模型输出和运行日志")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-apps",
        type=int,
        default=10,
        help="模拟的 app 数量 (app0..appN-1)",
    )
    parser.add_argument(
        "--app-trace-mode",
        type=str,
        choices=["static", "hotset"],
        default="static",
        help="app trace 模式: static(所有 app 活跃), hotset(仅热点 app 活跃)",
    )
    parser.add_argument(
        "--app-hotset-size",
        type=int,
        default=3,
        help="热点 app 数量 (hotset 模式)",
    )
    parser.add_argument(
        "--app-phase-requests",
        type=int,
        default=50,
        help="每发送多少请求切换一次热点 app",
    )
    parser.add_argument(
        "--app-phase-sleep",
        type=float,
        default=0.0,
        help="每次切换热点 app 之后的休眠秒数",
    )
    parser.add_argument(
        "--inactive-app-prob",
        type=float,
        default=0.0,
        help="hotset 模式下，从非热点 app 抽样的概率",
    )
    parser.add_argument(
        "--foreground-app-count",
        type=int,
        default=1,
        help="前台 app 数量（端侧通常为 1）",
    )
    parser.add_argument(
        "--foreground-app-prob",
        type=float,
        default=0.7,
        help="hotset 模式下，选择前台 app 的概率",
    )
    parser.add_argument(
        "--foreground-history-mode",
        type=str,
        choices=["append", "reset"],
        default="append",
        help="前台 app 的历史策略：append 追加多轮，reset 每次重置",
    )
    parser.add_argument(
        "--background-history-mode",
        type=str,
        choices=["append", "reset"],
        default="reset",
        help="后台 app 的历史策略：append 追加多轮，reset 每次重置",
    )
    parser.add_argument(
        "--killed-history-mode",
        type=str,
        choices=["append", "reset"],
        default="reset",
        help="killed app 的历史策略：append 追加多轮，reset 每次重置",
    )
    parser.add_argument(
        "--app-killed-after-phases",
        type=int,
        default=2,
        help="hotset 模式下，非活跃 app 连续多少个 phase 后视为 killed",
    )
    parser.add_argument(
        "--app-trace-verbose",
        action="store_true",
        help="打印热点 app 切换日志",
    )
    
    # --- 新增：LoRA 列表和分布控制 ---
    parser.add_argument(
        "--lora-list", 
        type=str, 
        default="lora0,lora1,lora2", 
        help="逗号分隔的 LoRA 名称列表，例如: 'lora_sql,lora_chat,lora_code'"
    )
    parser.add_argument(
        "--lora-dist", 
        type=str, 
        choices=["uniform", "zipf", "weighted"], 
        default="uniform",
        help="LoRA 选择的分布规律"
    )
    parser.add_argument("--lora-weights", type=str, default="", help="weighted 分布时的权重，如 0.7,0.2,0.1")
    parser.add_argument("--warmup-loras", type=str, default="", help="在正式压测前先用这些 LoRA 做一次预热，逗号分隔")
    parser.add_argument("--disable-random-sample", action="store_true")
    parser.add_argument("--sub-question-input-length", type=int, default=0)
    parser.add_argument("--request-rate", type=float, default=float("inf"), help="请求发送速率，inf 表示处理完立刻发下一个")

    return parser.parse_args()

async def async_request_sglang_generate(payload, url, pbar: Optional[tqdm] = None):
    # (保持原有的异步请求逻辑不变)
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {}
        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        output = RequestFuncOutput()
        try:
            async with session.post(url=url, json=payload, headers=headers) as response:
                if response.status == 200:
                    prompt_tokens = 0
                    cached_tokens = 0
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes: continue
                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]": pass
                        else:
                            data = json.loads(chunk)
                            if data["text"]:
                                timestamp = time.perf_counter()
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft
                                    prompt_tokens = (data.get("meta_info") or {}).get("prompt_tokens", 0)
                                    cached_tokens = (data.get("meta_info") or {}).get("cached_tokens", 0)
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)
                                most_recent_timestamp = timestamp
                                generated_text = data["text"]
                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.prompt_len = prompt_tokens
                    output.cached_tokens = cached_tokens
                    output.generated_len = len(output.itl) + 1
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception as e:
            output.success = False
            output.error = str(e)
    if pbar: pbar.update(1)
    return output

def gen_payload(prompt, output_len, lora_path="", app_id="", app_state=None, app_states=None):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
        "stream_options": {"include_usage": True},
        "lora_path": lora_path, # 这里将动态传入不同的 LoRA
    }
    if app_id:
        payload["app_id"] = app_id
    if app_state:
        payload["app_state"] = app_state
    if app_states:
        payload["app_states"] = app_states
    return payload

def run_warmup(host, port, model_path, warmup_loras: List[str], prompt_len: int, output_len: int):
    """在正式压测前，对指定 LoRA 做一次轻量预热，避免计时阶段写入 delta。"""
    if not warmup_loras:
        return
    url = f"http://{host}:{port}/generate"
    tokenizer = get_tokenizer(model_path)
    req = sample_random_requests(
        input_len=prompt_len,
        output_len=output_len,
        num_prompts=1,
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path="",
        random_sample=True,
        seed=1234,
    )[0].prompt
    for l in warmup_loras:
        payload = gen_payload(req, output_len, lora_path=l)
        try:
            requests.post(url, json=payload, timeout=120)
            print(f"[Warmup] sent warmup request for LoRA={l}")
        except Exception as e:
            print(f"[Warmup] failed for LoRA={l}: {e}")

class ReadyQueue:
    def __init__(self, init_requests=None):
        self.lock = threading.Lock()
        self.requests = init_requests or []
    def append(self, item):
        with self.lock: self.requests.append(item)
    def pop(self):
        with self.lock:
            if not self.requests: return None
            # FIFO 或者是 Random 并不重要，因为并发是 1
            return self.requests.pop(0)
    def pop_active(self, is_active=None):
        with self.lock:
            if not self.requests:
                return None
            if is_active is None:
                return self.requests.pop(0)
            for i, item in enumerate(self.requests):
                if is_active(item):
                    return self.requests.pop(i)
        return None

class WorkloadGenerator:
    def __init__(self, args):
        self.url = f"http://{args.host}:{args.port}/generate"
        self.tokenizer = get_tokenizer(args.model_path)
        self.num_clients = args.num_clients
        self.num_rounds = args.num_rounds
        self.max_parallel = args.max_parallel
        self.output_length = args.output_length
        self.total_requests = max(self.num_clients * self.num_rounds, 1)
        app_count = max(args.num_apps, 1)
        self.app_ids = [f"app{i}" for i in range(app_count)]
        app_rng = random.Random(args.seed + 1)
        self.app_rng = random.Random(args.seed + 2)
        if args.num_clients >= app_count:
            shuffled = self.app_ids[:]
            app_rng.shuffle(shuffled)
            self.client_app_map = {
                i: shuffled[i % app_count] for i in range(args.num_clients)
            }
        else:
            self.client_app_map = {
                i: app_rng.choice(self.app_ids) for i in range(args.num_clients)
            }
        used_app_ids = sorted(set(self.client_app_map.values()))
        if args.app_trace_mode == "static" and len(used_app_ids) < app_count:
            print(
                f"[AppTrace] Warning: num_clients({args.num_clients}) < num_apps({app_count}), "
                f"only {len(used_app_ids)} apps will be used."
            )

        trace_app_ids = self.app_ids
        self.app_trace = AppTrace(
            app_ids=trace_app_ids,
            mode=args.app_trace_mode,
            hotset_size=args.app_hotset_size,
            phase_requests=args.app_phase_requests,
            phase_sleep_s=args.app_phase_sleep,
            inactive_killed_after_phases=args.app_killed_after_phases,
            foreground_app_count=args.foreground_app_count,
            seed=args.seed + 7,
            verbose=args.app_trace_verbose,
        )
        self.fixed_foreground_app = (
            args.app_trace_mode == "hotset" and args.foreground_app_count == 1
        )
        self.foreground_anchor_app = (
            self.app_ids[0] if (self.fixed_foreground_app and self.app_ids) else None
        )
        if self.foreground_anchor_app:
            req_len = int(args.request_length)
            if req_len >= 1536:
                self.foreground_prompt_segments = 1
                self.background_prompt_segments = 1
            elif req_len >= 1024:
                self.foreground_prompt_segments = 2
                self.background_prompt_segments = 1
            elif req_len >= 512:
                self.foreground_prompt_segments = 3
                self.background_prompt_segments = 1
            else:
                self.foreground_prompt_segments = 4
                self.background_prompt_segments = 2
        else:
            self.foreground_prompt_segments = 1
            self.background_prompt_segments = 1
        self.replay_enabled = self.foreground_anchor_app is not None
        self.replay_interval = None
        self._requests_since_replay = 0
        if self.replay_enabled:
            # 通过固定间隔重复前台 anchor prompt，制造可复用前缀；
            # 间隔内穿插不同 app 形成 churn，放大淘汰策略差异。
            if args.request_length >= 1536:
                self.replay_interval = 3
            elif args.request_length >= 1024:
                self.replay_interval = 4
            elif args.request_length >= 512:
                self.replay_interval = 6
            else:
                self.replay_interval = 8
            self._requests_since_replay = self.replay_interval - 1
            print(
                "[Replay] enabled "
                f"anchor={self.foreground_anchor_app} "
                f"interval={self.replay_interval} "
                f"fg_segments={self.foreground_prompt_segments} "
                f"bg_segments={self.background_prompt_segments}"
            )
        self.inactive_app_prob = min(max(float(args.inactive_app_prob), 0.0), 1.0)
        self.foreground_app_prob = min(max(float(args.foreground_app_prob), 0.0), 1.0)
        self._requests_since_foreground = 0
        self._foreground_interval = None
        if 0.0 < self.foreground_app_prob < 0.5:
            self._foreground_interval = max(
                int(math.ceil(1.0 / self.foreground_app_prob)), 1
            )
        self._non_fg_cycle = []
        self._non_fg_cycle_index = 0
        self._non_fg_cycle_key = ()
        self.history_mode_by_state = {
            "foreground": args.foreground_history_mode,
            "background": args.background_history_mode,
            "killed": args.killed_history_mode,
        }
        if self.app_trace.mode == "hotset":
            print(
                "[AppTrace] "
                f"mode=hotset hotset_size={self.app_trace.hotset_size} "
                f"phase_requests={self.app_trace.phase_requests} "
                f"phase_sleep_s={self.app_trace.phase_sleep_s} "
                f"foreground_app_count={self.app_trace.foreground_app_count} "
                f"inactive_app_prob={self.inactive_app_prob} "
                f"foreground_app_prob={self.foreground_app_prob}"
            )
        
        # --- 初始化 LoRA 选择器 ---
        lora_list = [l.strip() for l in args.lora_list.split(",") if l.strip()]
        weights = None
        if args.lora_dist == "weighted" and args.lora_weights:
            try:
                weights = [float(x) for x in args.lora_weights.split(",") if x.strip() != ""]
                # 调整权重长度与 lora_list 一致
                if len(weights) < len(lora_list):
                    missing = len(lora_list) - len(weights)
                    # 用平均剩余填充
                    fill = 1.0 if sum(weights) <= 0 else max(1e-6, sum(weights) / len(weights))
                    weights.extend([fill] * missing)
                elif len(weights) > len(lora_list):
                    weights = weights[:len(lora_list)]
                total_w = sum(weights)
                if total_w <= 0:
                    print("[Warning] lora-weights 总和非正，忽略 weights")
                    weights = None
                else:
                    weights = [w / total_w for w in weights]
            except Exception:
                print("[Warning] 解析 lora-weights 失败，忽略 weights")
                weights = None
        if weights:
            print(f"[LoraSelector] Using weighted dist, paths={lora_list}, weights={weights}")
        self.lora_selector = LoraSelector(
            lora_paths=lora_list,
            distribution=args.lora_dist,
            seed=args.seed,
            weights=weights,
        )
        self.app_lora_map = {
            app_id: self.lora_selector.select() for app_id in self.app_ids
        }

        # --- 准备 Prompt ---
        app_init_inputs = sample_random_requests(
            input_len=args.request_length,
            output_len=args.output_length,
            num_prompts=app_count,
            range_ratio=1.0,
            tokenizer=self.tokenizer,
            dataset_path=args.dataset_path,
            random_sample=not args.disable_random_sample,
            seed=args.seed,
        )
        app_init_prompts = [i.prompt for i in app_init_inputs]

        # 准备多轮对话需要的“新问题”池子
        sub_len = (
            args.sub_question_input_length
            if args.sub_question_input_length != 0
            else args.request_length
        )
        prompt_budget = self.total_requests
        max_segments = max(self.foreground_prompt_segments, self.background_prompt_segments)
        prompt_budget = self.total_requests * max_segments + len(self.app_ids) * max_segments
        self.sub_question_inputs = sample_random_requests(
            input_len=sub_len,
            output_len=args.output_length,
            num_prompts=prompt_budget,
            range_ratio=1.0,
            tokenizer=self.tokenizer,
            dataset_path=args.dataset_path,
            random_sample=not args.disable_random_sample,
            seed=args.seed,
        )

        self.app_records = {}
        for i, app_id in enumerate(self.app_ids):
            prompt = app_init_prompts[i]
            if self.foreground_anchor_app and app_id == self.foreground_anchor_app:
                for _ in range(max(self.foreground_prompt_segments - 1, 0)):
                    extra_prompt = self._next_prompt()
                    if extra_prompt:
                        prompt += extra_prompt
            self.app_records[app_id] = {
                "count": 0,
                "history": prompt,
                "last_phase_idx": -1,
            }

        # --- 构造初始请求 (占位，真正的 payload 在发送时生成) ---
        init_requests = list(range(min(args.num_clients, self.total_requests)))
        self.ready_queue = ReadyQueue(init_requests=init_requests)

        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=self.total_requests)
        
        self.sent_requests = 0
        self.completed_requests = 0

        self.detailed_logs = []  # <--- 必须添加这一行初始化！
        
        # 统计数据容器
        self.performance_metrics = {
            "ttft": [], "latency": [], "cached_tokens": [], "prompt_len": []
        }
        self.app_metrics = {}
        self.state_metrics = {}

    def _next_prompt(self) -> str:
        if not self.sub_question_inputs:
            return ""
        return self.sub_question_inputs.pop().prompt

    def _select_lora_for_app(self, app_id: str, app_state: str) -> str:
        if app_state == "foreground":
            return self.app_lora_map.get(app_id, self.lora_selector.select())
        return self.lora_selector.select()

    def _next_non_foreground_app(self) -> Optional[str]:
        non_fg = self.app_trace.active_non_foreground_list()
        if not non_fg:
            return None
        key = tuple(non_fg)
        if key != self._non_fg_cycle_key:
            self._non_fg_cycle_key = key
            self._non_fg_cycle = list(non_fg)
            self.app_rng.shuffle(self._non_fg_cycle)
            self._non_fg_cycle_index = 0
        if self._non_fg_cycle_index >= len(self._non_fg_cycle):
            self.app_rng.shuffle(self._non_fg_cycle)
            self._non_fg_cycle_index = 0
        app_id = self._non_fg_cycle[self._non_fg_cycle_index]
        self._non_fg_cycle_index += 1
        return app_id

    def _select_app_id(self, client_id: int) -> str:
        if self.replay_enabled and self.foreground_anchor_app:
            if self.replay_interval and self._requests_since_replay >= self.replay_interval - 1:
                self._requests_since_replay = 0
                self._requests_since_foreground = 0
                return self.foreground_anchor_app
            app_id = None
            if self.app_trace.mode == "hotset":
                inactive_apps = [
                    a
                    for a in self.app_trace.inactive_app_list()
                    if a != self.foreground_anchor_app
                ]
                if inactive_apps:
                    app_id = self.app_rng.choice(inactive_apps)
            if app_id is None:
                app_id = self._next_non_foreground_app()
            if app_id is None and self.app_trace.mode == "hotset":
                app_id = self.app_trace.sample_active_app(self.app_rng)
            if app_id is None:
                app_id = self.client_app_map.get(client_id, self.app_ids[0])
            if app_id == self.foreground_anchor_app and len(self.app_ids) > 1:
                alternatives = [a for a in self.app_ids if a != self.foreground_anchor_app]
                app_id = self.app_rng.choice(alternatives)
            self._requests_since_replay += 1
            self._requests_since_foreground += 1
            return app_id
        if self.app_trace.mode == "hotset":
            if self._foreground_interval is not None:
                if self._requests_since_foreground >= self._foreground_interval - 1:
                    fg_app = (
                        self.foreground_anchor_app
                        if self.foreground_anchor_app
                        else self.app_trace.sample_foreground_app(self.app_rng)
                    )
                    if fg_app:
                        self._requests_since_foreground = 0
                        return fg_app
            else:
                if (
                    self.foreground_app_prob > 0.0
                    and self.app_rng.random() < self.foreground_app_prob
                ):
                    fg_app = (
                        self.foreground_anchor_app
                        if self.foreground_anchor_app
                        else self.app_trace.sample_foreground_app(self.app_rng)
                    )
                    if fg_app:
                        self._requests_since_foreground = 0
                        return fg_app
            if self.inactive_app_prob > 0.0 and self.app_rng.random() < self.inactive_app_prob:
                inactive_apps = self.app_trace.inactive_app_list()
                if inactive_apps:
                    self._requests_since_foreground += 1
                    return self.app_rng.choice(inactive_apps)
            non_fg = self._next_non_foreground_app()
            if non_fg:
                self._requests_since_foreground += 1
                return non_fg
            app_id = self.app_trace.sample_active_app(self.app_rng)
            self._requests_since_foreground += 1
            return app_id
        return self.client_app_map.get(client_id, self.app_ids[0])

    def _build_payload_for_app(self, app_id: str):
        record = self.app_records[app_id]
        app_states = self.app_trace.get_app_states()
        if self.foreground_anchor_app:
            for key, state in app_states.items():
                if key == self.foreground_anchor_app:
                    app_states[key] = "foreground"
                elif state == "foreground":
                    app_states[key] = "background"
        if self.replay_enabled and self.app_trace.mode == "hotset":
            for inactive_app in self.app_trace.inactive_app_list():
                if inactive_app != self.foreground_anchor_app:
                    app_states[inactive_app] = "killed"
        app_state = app_states.get(app_id) or "background"
        phase_idx = getattr(self.app_trace, "_phase_idx", 0)
        history_mode = self.history_mode_by_state.get(app_state, "append")
        if self.foreground_anchor_app and app_state == "foreground":
            prompt = record["history"]
        elif history_mode == "reset":
            segment_count = 1
            if self.foreground_anchor_app and app_state != "foreground":
                segment_count = self.background_prompt_segments
            segments = []
            for _ in range(segment_count):
                seg = self._next_prompt()
                if seg:
                    segments.append(seg)
            prompt = "".join(segments)
            record["history"] = prompt
            record["last_phase_idx"] = phase_idx
        else:
            if app_state == "foreground" and self.app_trace.mode == "hotset":
                prompt = record["history"]
            else:
                if record["count"] > 0:
                    new_question = self._next_prompt()
                    record["history"] += new_question
                prompt = record["history"]
        selected_lora = self._select_lora_for_app(app_id, app_state)
        return gen_payload(
            prompt,
            self.output_length,
            lora_path=selected_lora,
            app_id=app_id,
            app_state=app_state,
            app_states=app_states,
        )

    def _update_app_metrics(self, app_id: str, response: RequestFuncOutput):
        key = app_id or "unknown"
        stats = self.app_metrics.get(key)
        if stats is None:
            stats = {
                "count": 0,
                "ttft_sum": 0.0,
                "latency_sum": 0.0,
                "cached_tokens_sum": 0,
                "prompt_len_sum": 0,
            }
            self.app_metrics[key] = stats
        stats["count"] += 1
        stats["ttft_sum"] += float(response.ttft or 0.0)
        stats["latency_sum"] += float(response.latency or 0.0)
        stats["cached_tokens_sum"] += int(response.cached_tokens or 0)
        stats["prompt_len_sum"] += int(response.prompt_len or 0)

    def _update_state_metrics(self, app_state: str, response: RequestFuncOutput):
        key = app_state or "unknown"
        stats = self.state_metrics.get(key)
        if stats is None:
            stats = {
                "count": 0,
                "ttft_sum": 0.0,
                "latency_sum": 0.0,
                "cached_tokens_sum": 0,
                "prompt_len_sum": 0,
            }
            self.state_metrics[key] = stats
        stats["count"] += 1
        stats["ttft_sum"] += float(response.ttft or 0.0)
        stats["latency_sum"] += float(response.latency or 0.0)
        stats["cached_tokens_sum"] += int(response.cached_tokens or 0)
        stats["prompt_len_sum"] += int(response.prompt_len or 0)

    async def handle_request(self, item):
        client_id, payload = item

        # =====================================================
        # [新增] 打印请求详情
        # =====================================================
        lora_used = payload.get("lora_path", "None")
        prompt_text = payload.get("text", "")
        # 为了防止日志刷屏，Prompt 只截取前 100 个字符预览，并显示总长度
        prompt_preview = prompt_text[:100].replace("\n", "\\n") 
        if len(prompt_text) > 100:
            prompt_preview += "..."
            
        print(f"\n>>> [Request Sending] Client {client_id}")
        print(f"    LoRA Path : {lora_used}")
        print(f"    Prompt Len: {len(prompt_text)} chars")
        print(f"    Prompt    : {prompt_preview}")
        print("-" * 60)
        # =====================================================

        # print(f"[DEBUG] Sending req for Client {client_id}, LoRA={payload['lora_path']}") # 可选：调试打印
        response = await async_request_sglang_generate(payload, self.url, self.pbar)
        self.response_queue.put((client_id, response, payload))

    def request_sender(self):
        # 简单的发送循环，控制并发数
        async def request_loop():
            while True:
                if self.sent_requests - self.completed_requests < self.max_parallel:
                    if self.sent_requests >= self.total_requests:
                        if self.pbar.n == self.pbar.total:
                            break
                        await asyncio.sleep(0.01)
                        continue
                    client_id = self.ready_queue.pop()
                    if client_id is not None:
                        app_id = self._select_app_id(client_id)
                        payload = self._build_payload_for_app(app_id)
                        asyncio.create_task(self.handle_request((client_id, payload)))
                        self.sent_requests += 1
                        rotated = self.app_trace.on_request_sent()
                        if rotated:
                            self._requests_since_foreground = 0
                            self._non_fg_cycle = []
                            self._non_fg_cycle_index = 0
                            self._non_fg_cycle_key = ()
                        if rotated and self.app_trace.phase_sleep_s > 0:
                            await asyncio.sleep(self.app_trace.phase_sleep_s)
                    else:
                        if self.pbar.n == self.pbar.total: break
                        rotated = self.app_trace.force_rotate()
                        if rotated:
                            self._requests_since_foreground = 0
                            self._non_fg_cycle = []
                            self._non_fg_cycle_index = 0
                            self._non_fg_cycle_key = ()
                        if rotated and self.app_trace.phase_sleep_s > 0:
                            await asyncio.sleep(self.app_trace.phase_sleep_s)
                        else:
                            await asyncio.sleep(0.01)
                else:
                    await asyncio.sleep(0.01)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(request_loop())
        loop.close()

    def response_handler(self):
        while True:
            try:
                client_id, response, payload = self.response_queue.get(timeout=5)
                if not response.success:
                    print(f"Error client {client_id}: {response.error}")
                    self.completed_requests += 1
                    continue

                app_id = payload.get("app_id", "")
                app_state = payload.get("app_state", "") or "unknown"
                record = self.app_records.get(app_id)
                if record is None:
                    record = {"count": 0, "history": payload.get("text", "")}
                    self.app_records[app_id] = record
                current_round = record["count"]
                lora_used = payload.get("lora_path", "")
                input_text = payload.get("text", "")

                # === [新增] 记录详细日志 ===
                log_entry = {
                    "client_id": client_id,
                    "round_idx": current_round,
                    "lora": lora_used,
                    "app_id": app_id,
                    "app_state": app_state,
                    "input_text": input_text,
                    "output_text": response.generated_text,
                    "ttft": response.ttft,
                    "latency": response.latency,
                    "cached_tokens": response.cached_tokens,
                    "prompt_len": response.prompt_len,
                    "timestamp": datetime.now().isoformat()
                }
                self.detailed_logs.append(log_entry)
                # ===========================
                
                # 记录指标
                self.performance_metrics["ttft"].append(response.ttft)
                self.performance_metrics["latency"].append(response.latency)
                self.performance_metrics["cached_tokens"].append(response.cached_tokens)
                self.performance_metrics["prompt_len"].append(response.prompt_len)
                self._update_app_metrics(app_id, response)
                self._update_state_metrics(app_state, response)

                # 更新历史：前台默认不追加生成文本，避免降低前台重复命中率
                history_mode = self.history_mode_by_state.get(app_state, "append")
                if history_mode == "append":
                    if not (app_state == "foreground" and self.app_trace.mode == "hotset"):
                        record["history"] += response.generated_text
                record["count"] += 1
                self.completed_requests += 1

                # --- 构造下一轮请求 ---
                if self.completed_requests < self.total_requests:
                    if self.sent_requests < self.total_requests:
                        self.ready_queue.append(client_id)
                
                if self.pbar.n == self.pbar.total: break

            except queue.Empty:
                if self.pbar.n >= self.pbar.total: break

    def run(self):
        t1 = threading.Thread(target=self.request_sender, daemon=True)
        t2 = threading.Thread(target=self.response_handler, daemon=True)
        
        st = time.perf_counter()
        t1.start(); t2.start()
        t1.join(); t2.join()
        duration = time.perf_counter() - st
        
        self.pbar.close()

        # === [新增] 写入文件逻辑 ===
        print(f"\nWriting logs to {args.output_log_file}...")
        # 按 client 和 round 排序，方便阅读
        self.detailed_logs.sort(key=lambda x: (x['client_id'], x['round_idx']))

        try:
            with open(args.output_log_file, "w", encoding="utf-8") as f:
                for entry in self.detailed_logs:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print("✅ Logs saved successfully.")
        except Exception as e:
            print(f"❌ Failed to save logs: {e}")
        # ===========================

        self.print_stats(duration)

    def print_stats(self, duration):
        total_reqs = len(self.performance_metrics["ttft"])
        avg_ttft = sum(self.performance_metrics["ttft"]) / total_reqs if total_reqs else 0
        avg_latency = sum(self.performance_metrics["latency"]) / total_reqs if total_reqs else 0
        
        total_prompt_tokens = sum(self.performance_metrics["prompt_len"])
        total_cached_tokens = sum(self.performance_metrics["cached_tokens"])
        hit_rate = total_cached_tokens / total_prompt_tokens if total_prompt_tokens else 0

        lines = [
            "",
            "="*40,
            f"Delta Cache Benchmark (Concurrency={self.max_parallel})",
            "="*40,
            f"Total Requests : {total_reqs}",
            f"Duration       : {duration:.2f} s",
            f"Avg TTFT       : {avg_ttft*1000:.2f} ms",
            f"Avg Latency    : {avg_latency:.2f} s",
            f"Cache Hit Rate : {hit_rate*100:.2f} %",
        ]

        if self.app_metrics:
            lines.append("-"*40)
            counts_line = " ".join(
                [
                    f"{app_id}={stats['count']}"
                    for app_id, stats in sorted(
                        self.app_metrics.items(),
                        key=lambda x: (-x[1]["count"], x[0]),
                    )
                ]
            )
            lines.append(f"App Request Counts: {counts_line}")
            lines.append("Per-App Metrics:")
            for app_id, stats in sorted(
                self.app_metrics.items(),
                key=lambda x: (-x[1]["count"], x[0]),
            ):
                count = stats["count"]
                avg_ttft_ms = (stats["ttft_sum"] / count * 1000.0) if count else 0.0
                avg_latency = (stats["latency_sum"] / count) if count else 0.0
                hit_rate_app = (
                    stats["cached_tokens_sum"] / stats["prompt_len_sum"]
                    if stats["prompt_len_sum"]
                    else 0.0
                )
                lines.append(
                    f"  {app_id}: count={count} "
                    f"avg_ttft={avg_ttft_ms:.2f}ms "
                    f"avg_latency={avg_latency:.2f}s "
                    f"hit_rate={hit_rate_app*100:.2f}%"
                )

        if self.state_metrics:
            lines.append("-"*40)
            counts_line = " ".join(
                [
                    f"{state}={stats['count']}"
                    for state, stats in sorted(
                        self.state_metrics.items(),
                        key=lambda x: (-x[1]["count"], x[0]),
                    )
                ]
            )
            lines.append(f"App State Counts: {counts_line}")
            lines.append("App State Metrics:")
            for state, stats in sorted(
                self.state_metrics.items(),
                key=lambda x: (-x[1]["count"], x[0]),
            ):
                count = stats["count"]
                avg_ttft_ms = (stats["ttft_sum"] / count * 1000.0) if count else 0.0
                avg_latency = (stats["latency_sum"] / count) if count else 0.0
                hit_rate_state = (
                    stats["cached_tokens_sum"] / stats["prompt_len_sum"]
                    if stats["prompt_len_sum"]
                    else 0.0
                )
                lines.append(
                    f"  {state}: count={count} "
                    f"avg_ttft={avg_ttft_ms:.2f}ms "
                    f"avg_latency={avg_latency:.2f}s "
                    f"hit_rate={hit_rate_state*100:.2f}%"
                )

        lines.append("="*40)

        for line in lines:
            print(line)

        if args.stats_log_file:
            try:
                with open(args.stats_log_file, "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
            except Exception as e:
                print(f"Failed to write stats log to {args.stats_log_file}: {e}")
            try:
                policy = "unknown"
                if "policylru" in args.stats_log_file:
                    policy = "lru"
                elif "policymobilora" in args.stats_log_file:
                    policy = "mobilora"
                summary = {
                    "timestamp": datetime.now().isoformat(),
                    "policy": policy,
                    "total_requests": total_reqs,
                    "duration_s": duration,
                    "avg_ttft_ms": avg_ttft * 1000.0,
                    "avg_latency_s": avg_latency,
                    "hit_rate": hit_rate,
                    "stats_log_file": args.stats_log_file,
                    "output_log_file": args.output_log_file,
                }
                summary_path = args.stats_log_file + ".jsonl"
                with open(summary_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Failed to write stats summary to {args.stats_log_file}.jsonl: {e}")

if __name__ == "__main__":
    args = parse_args()
    
    # 可以在这里先请求一下 /flush_cache 接口清空服务器状态
    try:
        requests.post(f"http://{args.host}:{args.port}/flush_cache", timeout=5)
    except:
        print("Warning: Failed to flush cache.")

    warmup_list = [l.strip() for l in args.warmup_loras.split(",") if l.strip()]
    if warmup_list:
        run_warmup(args.host, args.port, args.model_path, warmup_list, args.request_length, args.output_length)

    gen = WorkloadGenerator(args)
    gen.run()
