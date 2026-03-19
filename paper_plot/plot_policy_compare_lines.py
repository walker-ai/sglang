#!/usr/bin/env python
"""
Line plot for eviction policy comparison across configs.

Outputs:
    paper_plot/policy_compare_lines.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
STATS_ROOT = ROOT / "paper"
OUTPUT_PATH = Path(__file__).with_name("policy_compare_lines.png")

POLICIES = ["lru", "lfu", "fifo", "mru", "filo", "mobilora"]

KV_VALUES = [1, 2, 4]
REQ_VALUES = [1024, 2048]
CONFIGS = [
    {"label": f"kv{kv}g req{req}", "kv": kv, "req": req}
    for kv in KV_VALUES
    for req in REQ_VALUES
]


def read_last_json(path: Path) -> Optional[dict]:
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


def find_latest_stats(req: int, kv: int, policy: str) -> Optional[dict]:
    pattern = f"*req{req}_*kv{kv}g_policy{policy}_stats.log.jsonl"
    candidates = sorted(STATS_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return read_last_json(candidates[-1])


def collect_metrics() -> Dict[str, Dict[str, List[float]]]:
    metrics: Dict[str, Dict[str, List[float]]] = {}
    for policy in POLICIES:
        metrics[policy] = {"hit_rate": [], "ttft_ms": []}
        for cfg in CONFIGS:
            data = find_latest_stats(cfg["req"], cfg["kv"], policy)
            if not data:
                metrics[policy]["hit_rate"].append(np.nan)
                metrics[policy]["ttft_ms"].append(np.nan)
                continue
            metrics[policy]["hit_rate"].append(data["hit_rate"] * 100.0)
            metrics[policy]["ttft_ms"].append(data["avg_ttft_ms"])
    return metrics


def main() -> None:
    metrics = collect_metrics()
    labels = [cfg["label"] for cfg in CONFIGS]
    x = np.arange(len(CONFIGS))

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    cmap = plt.get_cmap("tab10")

    for idx, policy in enumerate(POLICIES):
        color = "#fdae6b" if policy == "mobilora" else cmap(idx % 10)
        axes[0].plot(
            x,
            metrics[policy]["hit_rate"],
            marker="o",
            label=policy,
            color=color,
        )
        axes[1].plot(
            x,
            metrics[policy]["ttft_ms"],
            marker="o",
            label=policy,
            color=color,
        )

    axes[0].set_ylabel("Cache Hit Rate (%)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].set_title("Policy Comparison (placeholders for missing configs)")

    axes[1].set_ylabel("Avg TTFT (ms)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].set_xlabel("Config")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)

    axes[0].legend(ncol=3, frameon=False, loc="upper right")
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
