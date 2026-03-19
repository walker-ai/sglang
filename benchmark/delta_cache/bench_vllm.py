import argparse
import asyncio
import json
import queue
import random
import threading
import time
from datetime import datetime
import re
from typing import Optional, List

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm

from sglang.bench_serving import (
    RequestFuncOutput,
    get_tokenizer,
    sample_random_requests,
    remove_prefix,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)


class LoraSelector:
    """LoRA 选择器，支持 uniform / zipf 分布。"""

    def __init__(
        self,
        lora_paths: List[str],
        distribution: str = "uniform",
        seed: int = 42,
        weights: Optional[List[float]] = None,
    ):
        self.lora_paths = lora_paths or [""]
        self.distribution = distribution
        self.py_rng = random.Random(seed)
        # weighted 模式未提供权重时默认均匀
        if self.distribution == "weighted" and not weights:
            if self.lora_paths:
                self.weights = [1.0 / len(self.lora_paths)] * len(self.lora_paths)
                print(f"[LoraSelector] weighted 未提供权重，使用均匀权重 {self.weights}")
            else:
                self.weights = None
        else:
            self.weights = weights
        if self.weights and self.distribution != "weighted":
            self.distribution = "weighted"

        print(
            f"Load LoRA Selector: {len(self.lora_paths)} LoRAs, "
            f"Distribution: {distribution}"
        )

    def select(self) -> str:
        if not self.lora_paths or self.lora_paths == [""]:
            return ""

        if self.distribution == "uniform":
            return self.py_rng.choice(self.lora_paths)
        if self.distribution == "zipf":
            idx = (np.random.zipf(1.5) - 1) % len(self.lora_paths)
            return self.lora_paths[idx]
        if self.distribution == "weighted" and self.weights:
            choice = self.py_rng.choices(self.lora_paths, weights=self.weights, k=1)[0]
            return choice
        if self.distribution == "weighted" and not self.weights:
            print("[Warning] weighted 分布缺少权重，退化为首个 LoRA")
            return self.lora_paths[0]
        return self.lora_paths[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM (OpenAI API) with multi-round LoRA switching."
    )
    parser.add_argument("--num-clients", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--request-length", type=int, default=512)
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="用于 tokenizer 统计 prompt 长度，同时作为默认的 model 名传给 vLLM。",
    )
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--log-file", type=str, default="vllm_delta_cache.jsonl")
    parser.add_argument(
        "--output-log-file",
        type=str,
        default="",
        help="保存请求/响应明细 JSONL，留空则使用 --log-file",
    )
    parser.add_argument(
        "--stats-log-file",
        type=str,
        default="",
        help="保存汇总指标",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lora-list",
        type=str,
        default="lora0,lora1,lora2",
        help="逗号分隔的 LoRA 名称列表。",
    )
    parser.add_argument(
        "--lora-dist",
        type=str,
        choices=["uniform", "zipf", "weighted"],
        default="uniform",
        help="LoRA 选择分布。",
    )
    parser.add_argument("--lora-weights", type=str, default="", help="weighted 分布时的权重，如 0.8,0.2")
    parser.add_argument("--disable-random-sample", action="store_true")
    parser.add_argument("--sub-question-input-length", type=int, default=0)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="请求速率，暂未严格限流（与原 bench_sglang 行为一致）。",
    )
    parser.add_argument(
        "--vllm-log-path",
        type=str,
        default="",
        help="可选，指定 vLLM 服务端日志文件路径，用于从日志中解析 Prefix cache hit rate。",
    )
    return parser.parse_args()


async def async_request_vllm_completions(payload, url, pbar: Optional[tqdm] = None):
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
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")
                        chunk = remove_prefix(chunk.strip(), " ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            continue
                        data = json.loads(chunk)
                        if data["choices"][0]["text"]:
                            timestamp = time.perf_counter()
                            if ttft == 0.0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)
                            most_recent_timestamp = timestamp
                            generated_text += data["choices"][0]["text"]
                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.prompt_len = 0
                    output.output_len = payload.get("max_tokens", 0)
                    output.cached_tokens = 0
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception as e:
            output.success = False
            output.error = str(e)
    if pbar:
        pbar.update(1)
    return output


def gen_payload(prompt, output_len, model_name: str, lora_path=""):
    payload = {
        "model": lora_path,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": output_len,
        "stream": True,
    }
    # if lora_path:
    #     payload["lora_path"] = lora_path
    return payload


class ReadyQueue:
    def __init__(self, init_requests=None):
        self.lock = threading.Lock()
        self.requests = init_requests or []

    def append(self, item):
        with self.lock:
            self.requests.append(item)

    def pop(self):
        with self.lock:
            if not self.requests:
                return None
            return self.requests.pop(0)


class WorkloadGenerator:
    def __init__(self, args):
        self.url = f"http://{args.host}:{args.port}/v1/completions"
        self.tokenizer = get_tokenizer(args.model_path)
        self.num_clients = args.num_clients
        self.num_rounds = args.num_rounds
        self.max_parallel = args.max_parallel
        self.output_length = args.output_length
        self.model_name = args.model_path

        lora_list = [l.strip() for l in args.lora_list.split(",") if l.strip()]
        weights = None
        if args.lora_dist == "weighted" and args.lora_weights:
            try:
                weights = [float(x) for x in args.lora_weights.split(",") if x.strip() != ""]
                if len(weights) < len(lora_list):
                    missing = len(lora_list) - len(weights)
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

        self.candidate_inputs = sample_random_requests(
            input_len=args.request_length,
            output_len=args.output_length,
            num_prompts=args.num_clients,
            range_ratio=1.0,
            tokenizer=self.tokenizer,
            dataset_path=args.dataset_path,
            random_sample=not args.disable_random_sample,
            seed=args.seed,
        )
        self.candidate_inputs = [i.prompt for i in self.candidate_inputs]

        sub_len = (
            args.sub_question_input_length
            if args.sub_question_input_length != 0
            else args.request_length
        )
        self.sub_question_inputs = sample_random_requests(
            input_len=sub_len,
            output_len=args.output_length,
            num_prompts=args.num_clients * max(args.num_rounds - 1, 1),
            range_ratio=1.0,
            tokenizer=self.tokenizer,
            dataset_path=args.dataset_path,
            random_sample=not args.disable_random_sample,
            seed=args.seed,
        )

        init_requests = []
        for i in range(args.num_clients):
            prompt = self.candidate_inputs[i]
            selected_lora = self.lora_selector.select()
            payload = gen_payload(prompt, args.output_length, self.model_name, selected_lora)
            prompt_len = len(self.tokenizer.encode(prompt))
            init_requests.append((i, payload, prompt_len))

        self.client_records = {
            i: {"round": 0, "history": init_requests[i][1]["prompt"]}
            for i in range(args.num_clients)
        }
        self.ready_queue = ReadyQueue(init_requests=init_requests)
        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=args.num_clients * args.num_rounds)
        self.sent_requests = 0
        self.completed_requests = 0
        self.detailed_logs = []
        self.performance_metrics = {
            "ttft": [],
            "latency": [],
            "cached_tokens": [],
            "prompt_len": [],
        }

    async def handle_request(self, item):
        client_id, payload, prompt_len = item
        lora_used = payload.get("lora_path", "")
        prompt_text = payload.get("prompt", "")
        prompt_preview = prompt_text[:100].replace("\n", "\\n")
        if len(prompt_text) > 100:
            prompt_preview += "..."

        print(f"\n>>> [Request Sending] Client {client_id}")
        print(f"    LoRA Path : {lora_used or '(base)'}")
        print(f"    Prompt Len: {len(prompt_text)} chars")
        print(f"    Prompt    : {prompt_preview}")
        print("-" * 60)

        response = await async_request_vllm_completions(payload, self.url, self.pbar)
        self.response_queue.put((client_id, response, payload, prompt_len))

    def request_sender(self):
        async def request_loop():
            while True:
                if self.sent_requests - self.completed_requests < self.max_parallel:
                    new_request = self.ready_queue.pop()
                    if new_request:
                        asyncio.create_task(self.handle_request(new_request))
                        self.sent_requests += 1
                    else:
                        if self.pbar.n == self.pbar.total:
                            break
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
                client_id, response, payload, prompt_len = self.response_queue.get(timeout=5)
                if not response.success:
                    print(f"Error client {client_id}: {response.error}")
                    self.completed_requests += 1
                    continue

                current_round = self.client_records[client_id]["round"]
                lora_used = payload.get("lora_path", "")
                input_text = payload.get("prompt", "")

                log_entry = {
                    "client_id": client_id,
                    "round_idx": current_round,
                    "lora": lora_used,
                    "input_text": input_text,
                    "output_text": response.generated_text,
                    "ttft": response.ttft,
                    "latency": response.latency,
                    "cached_tokens": getattr(response, "cached_tokens", 0),
                    "prompt_len": prompt_len,
                    "timestamp": datetime.now().isoformat(),
                }
                self.detailed_logs.append(log_entry)

                self.performance_metrics["ttft"].append(response.ttft)
                self.performance_metrics["latency"].append(response.latency)
                self.performance_metrics["cached_tokens"].append(
                    getattr(response, "cached_tokens", 0)
                )
                self.performance_metrics["prompt_len"].append(prompt_len)

                self.client_records[client_id]["history"] += response.generated_text
                self.client_records[client_id]["round"] += 1
                self.completed_requests += 1

                if self.client_records[client_id]["round"] < self.num_rounds:
                    new_question = self.sub_question_inputs.pop().prompt
                    self.client_records[client_id]["history"] += new_question
                    next_lora = self.lora_selector.select()
                    next_payload = gen_payload(
                        self.client_records[client_id]["history"],
                        self.output_length,
                        self.model_name,
                        next_lora,
                    )
                    next_prompt_len = len(
                        self.tokenizer.encode(self.client_records[client_id]["history"])
                    )
                    self.ready_queue.append((client_id, next_payload, next_prompt_len))

                if self.pbar.n == self.pbar.total:
                    break

            except queue.Empty:
                if self.pbar.n >= self.pbar.total:
                    break

    def run(self):
        t1 = threading.Thread(target=self.request_sender, daemon=True)
        t2 = threading.Thread(target=self.response_handler, daemon=True)

        st = time.perf_counter()
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        duration = time.perf_counter() - st
        self.pbar.close()

        log_file = args.output_log_file or args.log_file
        print(f"\nWriting logs to {log_file}...")
        self.detailed_logs.sort(key=lambda x: (x["client_id"], x["round_idx"]))
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                for entry in self.detailed_logs:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print("✅ Logs saved successfully.")
        except Exception as e:
            print(f"❌ Failed to save logs: {e}")

        self.print_stats(duration)

    def print_stats(self, duration):
        total_reqs = len(self.performance_metrics["ttft"])
        avg_ttft = (
            sum(self.performance_metrics["ttft"]) / total_reqs if total_reqs else 0
        )
        avg_latency = (
            sum(self.performance_metrics["latency"]) / total_reqs if total_reqs else 0
        )
        total_prompt_tokens = sum(self.performance_metrics["prompt_len"])
        total_cached_tokens = sum(self.performance_metrics["cached_tokens"])
        hit_rate = total_cached_tokens / total_prompt_tokens if total_prompt_tokens else 0

        log_hit_rate = parse_prefix_hit_rate_from_log(getattr(args, "vllm_log_path", ""))

        lines = [
            "",
            "=" * 40,
            f"vLLM Benchmark (Concurrency={self.max_parallel})",
            "=" * 40,
            f"Total Requests : {total_reqs}",
            f"Duration       : {duration:.2f} s",
            f"Avg TTFT       : {avg_ttft*1000:.2f} ms",
            f"Avg Latency    : {avg_latency:.2f} s",
            f"Cache Hit Rate : {hit_rate*100:.2f} % (vLLM OpenAI 接口未返回 cached_tokens，仅供参考)",
        ]
        if log_hit_rate is not None:
            lines.append(f"Prefix Cache Hit Rate (server log): {log_hit_rate:.2f} %")
        lines.append("=" * 40)

        for line in lines:
            print(line)

        if args.stats_log_file:
            try:
                with open(args.stats_log_file, "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
            except Exception as e:
                print(f"Failed to write stats log to {args.stats_log_file}: {e}")


def parse_prefix_hit_rate_from_log(log_path: str) -> Optional[float]:
    """从 vLLM 日志文件中解析最后一次出现的 Prefix cache hit rate 值。"""
    if not log_path:
        return None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        pattern = re.compile(r"Prefix cache hit rate:\s*([0-9.]+)%")
        for line in reversed(lines):
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    except Exception:
        return None
    return None


if __name__ == "__main__":
    args = parse_args()
    wg = WorkloadGenerator(args)
    wg.run()
