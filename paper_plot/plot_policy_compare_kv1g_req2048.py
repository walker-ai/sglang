#!/usr/bin/env python
"""
Plot policy comparison for a single config (kv=1g, req=2048) using stats log jsonl.

Output:
    paper_plot/policy_compare_kv1g_req2048.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
STATS_ROOT = ROOT / "paper"
OUTPUT_PATH = Path(__file__).with_name("policy_compare_kv1g_req2048.png")

FNAME_RE = re.compile(
    r".*_req2048_.*_kv1g_policy(?P<policy>[^_]+)_stats\.log\.jsonl$"
)

POLICY_ORDER = ["lru", "lfu", "fifo", "mru", "filo", "mobilora"]


def read_last_json(path: Path) -> dict | None:
    last = None
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = line
        if not last:
            return None
        return json.loads(last)
    except Exception:
        return None


def collect_metrics() -> dict[str, dict]:
    results: dict[str, dict] = {}
    for path in STATS_ROOT.glob("*_stats.log.jsonl"):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        policy = m.group("policy")
        data = read_last_json(path)
        if not data:
            continue
        results[policy] = data
    return results


def main() -> None:
    results = collect_metrics()
    missing = [p for p in POLICY_ORDER if p not in results]
    if missing:
        print(f"Warning: missing policies: {missing}")

    policies = [p for p in POLICY_ORDER if p in results]
    ttft_ms = [results[p]["avg_ttft_ms"] for p in policies]
    hit_rate = [results[p]["hit_rate"] * 100.0 for p in policies]

    colors = ["#9ecae1"] * len(policies)
    if "mobilora" in policies:
        colors[policies.index("mobilora")] = "#fdae6b"

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)

    axes[0].bar(policies, hit_rate, color=colors)
    axes[0].set_ylabel("Cache Hit Rate (%)")
    axes[0].set_title("KV=1GB, Request Len=2048")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(policies, ttft_ms, color=colors)
    axes[1].set_ylabel("Avg TTFT (ms)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].set_xlabel("Eviction Policy")
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
