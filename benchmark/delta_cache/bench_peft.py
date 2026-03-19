import argparse
import json
import queue
import random
import threading
import time
from datetime import datetime
from typing import Optional, List

import numpy as np
import requests
from tqdm import tqdm

from sglang.bench_serving import (
    RequestFuncOutput,
    get_tokenizer,
    remove_prefix,
    sample_random_requests,
)


class LoraSelector:
    """LoRA 选择器，支持 uniform / zipf 分布。"""

    def __init__(
        self,
        lora_names: List[str],
        distribution: str = "uniform",
        seed: int = 42,
        weights: Optional[List[float]] = None,
    ):
        self.lora_names = lora_names or [""]
        self.distribution = distribution
        self.py_rng = random.Random(seed)
        # weighted 模式未提供权重时默认均匀
        if self.distribution == "weighted" and not weights:
            if self.lora_names:
                self.weights = [1.0 / len(self.lora_names)] * len(self.lora_names)
                print(f"[LoraSelector] weighted 未提供权重，使用均匀权重 {self.weights}")
            else:
                self.weights = None
        else:
            self.weights = weights
        if self.weights and self.distribution != "weighted":
            self.distribution = "weighted"

        print(
            f"Load LoRA Selector: {len(self.lora_names)} LoRAs, "
            f"Distribution: {distribution}"
        )

    def select(self) -> str:
        if not self.lora_names or self.lora_names == [""]:
            return ""

        if self.distribution == "uniform":
            return self.py_rng.choice(self.lora_names)
        if self.distribution == "zipf":
            idx = (np.random.zipf(1.5) - 1) % len(self.lora_names)
            return self.lora_names[idx]
        if self.distribution == "weighted" and self.weights:
            return self.py_rng.choices(self.lora_names, weights=self.weights, k=1)[0]
        if self.distribution == "weighted" and not self.weights:
            print("[Warning] weighted 分布缺少权重，退化为首个 LoRA")
            return self.lora_names[0]
        return self.lora_names[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark HF+PEFT server with multi-round LoRA switching."
    )
    parser.add_argument("--num-clients", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--request-length", type=int, default=512)
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8190)
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="用于 tokenizer 统计 prompt 长度。",
    )
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--log-file", type=str, default="peft_output.jsonl")
    parser.add_argument("--stats-log-file", type=str, default="peft_stats.log")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lora-list",
        type=str,
        default="lora0,lora1",
        help="逗号分隔的 LoRA 名称列表，需要与服务端预加载的 adapter 名一致。",
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
        help="请求速率（当前未严格限流，与 bench_sglang 行为一致）。",
    )
    return parser.parse_args()


def gen_payload(prompt: str, output_len: int, lora_name: str):
    return {
        "text": prompt,
        "max_new_tokens": output_len,
        "lora": lora_name or None,
    }


def sync_request_peft(payload, url) -> RequestFuncOutput:
    output = RequestFuncOutput()
    st = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=120)
        latency = time.perf_counter() - st
        if resp.status_code == 200:
            data = resp.json()
            output.generated_text = remove_prefix(data.get("text", ""), "data: ").strip()
            output.success = True
            output.latency = latency
            output.ttft = latency  # 无流式返回，用总耗时近似
            output.prompt_len = 0
            output.cached_tokens = 0
        else:
            output.success = False
            output.error = f"HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        output.success = False
        output.error = str(e)
    return output


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
        self.url = f"http://{args.host}:{args.port}/generate"
        self.tokenizer = get_tokenizer(args.model_path)
        self.num_clients = args.num_clients
        self.num_rounds = args.num_rounds
        self.max_parallel = args.max_parallel
        self.output_length = args.output_length

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
            lora_names=lora_list,
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
            payload = gen_payload(prompt, args.output_length, selected_lora)
            prompt_len = len(self.tokenizer.encode(prompt))
            init_requests.append((i, payload, prompt_len))

        self.client_records = {
            i: {"round": 0, "history": init_requests[i][1]["text"]}
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

    def handle_request(self, item):
        client_id, payload, prompt_len = item
        lora_used = payload.get("lora", "")
        response = sync_request_peft(payload, self.url)
        self.response_queue.put((client_id, response, payload, prompt_len))

    def request_sender(self):
        while True:
            if self.sent_requests - self.completed_requests < self.max_parallel:
                new_request = self.ready_queue.pop()
                if new_request:
                    threading.Thread(
                        target=self.handle_request, args=(new_request,), daemon=True
                    ).start()
                    self.sent_requests += 1
                else:
                    if self.pbar.n == self.pbar.total:
                        break
                    time.sleep(0.01)
            else:
                time.sleep(0.01)

    def response_handler(self):
        while True:
            try:
                client_id, response, payload, prompt_len = self.response_queue.get(timeout=5)
                if not response.success:
                    print(f"Error client {client_id}: {response.error}")
                    self.completed_requests += 1
                    continue

                current_round = self.client_records[client_id]["round"]
                lora_used = payload.get("lora", "")
                input_text = payload.get("text", "")

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
                self.pbar.update(1)

                if self.client_records[client_id]["round"] < self.num_rounds:
                    new_question = self.sub_question_inputs.pop().prompt
                    self.client_records[client_id]["history"] += new_question
                    next_lora = self.lora_selector.select()
                    next_payload = gen_payload(
                        self.client_records[client_id]["history"],
                        self.output_length,
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

        print(f"\nWriting logs to {args.log_file}...")
        self.detailed_logs.sort(key=lambda x: (x["client_id"], x["round_idx"]))
        try:
            with open(args.log_file, "w", encoding="utf-8") as f:
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

        lines = [
            "",
            "=" * 40,
            f"PEFT Benchmark (Concurrency={self.max_parallel})",
            "=" * 40,
            f"Total Requests : {total_reqs}",
            f"Duration       : {duration:.2f} s",
            f"Avg TTFT       : {avg_ttft*1000:.2f} ms",
            f"Avg Latency    : {avg_latency:.2f} s",
            f"Cache Hit Rate : {hit_rate*100:.2f} % (PEFT 不返回 cached_tokens，仅占位)",
            "=" * 40,
        ]
        for line in lines:
            print(line)

        if args.stats_log_file:
            try:
                with open(args.stats_log_file, "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
            except Exception as e:
                print(f"Failed to write stats log to {args.stats_log_file}: {e}")


if __name__ == "__main__":
    args = parse_args()
    wg = WorkloadGenerator(args)
    wg.run()
