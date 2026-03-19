"""
Parse stats logs under paper/ and plot summary bars for TTFT / Latency / Cache Hit Rate.

Usage:
    python paper_plot/plot_stats_from_logs.py

It will:
  - scan paper/*_stats.log
  - parse filename tags: bench, dc, hc, lora_count, lora_dist, req, out, clients, rounds, kv_mem
  - parse metrics from file content: Avg TTFT (ms), Avg Latency (s), Cache Hit Rate (%)
  - save a bar chart to paper_plot/stats_summary.png
  - print a table and list missing combinations (cross product of observed dc/hc/lora_count/lora_dist/kv_mem)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STATS_ROOT = Path(__file__).resolve().parent.parent / "paper"
OUTPUT_PATH = Path(__file__).with_name("stats_summary.png")

# Filename pattern: <bench>_dcX_hcY_l<count>-<dist>_reqR_outO_cC_rN_kvZg_stats.log
FNAME_RE = re.compile(
    r"(?P<bench>\w+)_dc(?P<dc>[01])_hc(?P<hc>[01])_l(?P<lora_count>\d+)-(?P<lora_dist>\w+)"
    r"_req(?P<req>\d+)_out(?P<out>\d+)_c(?P<clients>\d+)_r(?P<rounds>\d+)_kv(?P<kv_mem>\d+)g_stats\.log$"
)

TTFT_RE = re.compile(r"Avg TTFT\s*:\s*([\d.]+)\s*ms", re.I)
LAT_RE = re.compile(r"Avg Latency\s*:\s*([\d.]+)\s*s", re.I)
HIT_RE = re.compile(r"Cache Hit Rate\s*:\s*([\d.]+)\s*%", re.I)


@dataclass(frozen=True)
class ExpKey:
    bench: str
    dc: int
    hc: int
    lora_count: int
    lora_dist: str
    kv_mem: int
    req: int
    out: int
    clients: int
    rounds: int

    def label(self) -> str:
        # concise label for plotting
        return (
            f"{self.bench}|dc{self.dc}|hc{self.hc}|"
            f"l{self.lora_count}-{self.lora_dist}|kv{self.kv_mem}g|"
            f"req{self.req}|out{self.out}"
        )


def parse_stats_files(root: Path) -> Dict[ExpKey, Dict[str, float]]:
    results: Dict[ExpKey, Dict[str, float]] = {}
    for path in root.glob("*_stats.log"):
        m = FNAME_RE.search(path.name)
        if not m:
            continue
        key = ExpKey(
            bench=m.group("bench"),
            dc=int(m.group("dc")),
            hc=int(m.group("hc")),
            lora_count=int(m.group("lora_count")),
            lora_dist=m.group("lora_dist"),
            kv_mem=int(m.group("kv_mem")),
            req=int(m.group("req")),
            out=int(m.group("out")),
            clients=int(m.group("clients")),
            rounds=int(m.group("rounds")),
        )
        text = path.read_text(encoding="utf-8", errors="ignore")
        ttft_m = TTFT_RE.search(text)
        lat_m = LAT_RE.search(text)
        hit_m = HIT_RE.search(text)
        if not (ttft_m and lat_m and hit_m):
            continue
        results[key] = {
            "ttft_ms": float(ttft_m.group(1)),
            "latency_s": float(lat_m.group(1)),
            "hit_rate": float(hit_m.group(1)),
            "path": str(path),
        }
    return results


def compute_missing(keys: List[ExpKey]) -> List[Tuple[int, int, int, str, int]]:
    """Return missing combinations based on observed value sets (dc, hc, lora_count, lora_dist, kv_mem)."""
    dcs = sorted({k.dc for k in keys})
    hcs = sorted({k.hc for k in keys})
    lcounts = sorted({k.lora_count for k in keys})
    ldist = sorted({k.lora_dist for k in keys})
    kvs = sorted({k.kv_mem for k in keys})

    observed = {(k.dc, k.hc, k.lora_count, k.lora_dist, k.kv_mem) for k in keys}
    missing = []
    for dc in dcs:
        for hc in hcs:
            for lc in lcounts:
                for ld in ldist:
                    for kv in kvs:
                        tup = (dc, hc, lc, ld, kv)
                        if tup not in observed:
                            missing.append(tup)
    return missing


def plot_metrics(results: Dict[ExpKey, Dict[str, float]], output_path: Path) -> None:
    if not results:
        print("No stats logs found.")
        return

    keys = list(results.keys())
    # 分不同 LoRA 配置出子图：按 (lora_count, lora_dist, kv_mem) 分组
    groups: Dict[Tuple[int, str, int], List[ExpKey]] = {}
    for k in keys:
        g = (k.lora_count, k.lora_dist, k.kv_mem)
        groups.setdefault(g, []).append(k)

    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 3, figsize=(12, max(3, 3 * n_groups)))
    if n_groups == 1:
        axes = np.expand_dims(axes, axis=0)  # normalize shape

    for row_idx, (gkey, gkeys) in enumerate(sorted(groups.items())):
        lc, ld, kv = gkey
        labels = [f"{k.bench}-dc{k.dc}-hc{k.hc}" for k in gkeys]
        ttft_vals = [results[k]["ttft_ms"] for k in gkeys]
        lat_vals = [results[k]["latency_s"] * 1000 for k in gkeys]  # ms
        hit_vals = [results[k]["hit_rate"] for k in gkeys]

        x = np.arange(len(gkeys))
        width = 0.4

        bars1 = axes[row_idx, 0].bar(x, ttft_vals, width=width, color="#9ecae1")
        axes[row_idx, 0].set_title(f"TTFT (ms) | l{lc}-{ld} kv{kv}g")
        axes[row_idx, 0].bar_label(bars1, padding=1, fontsize=7, fmt="%.0f")
        if ttft_vals:
            ymax = max(ttft_vals)
            axes[row_idx, 0].set_ylim(0, ymax * 1.15)

        bars2 = axes[row_idx, 1].bar(x, lat_vals, width=width, color="#fdae6b")
        axes[row_idx, 1].set_title("Latency (ms)")
        axes[row_idx, 1].bar_label(bars2, padding=1, fontsize=7, fmt="%.0f")
        if lat_vals:
            ymax = max(lat_vals)
            axes[row_idx, 1].set_ylim(0, ymax * 1.15)

        bars3 = axes[row_idx, 2].bar(x, hit_vals, width=width, color="#a1d99b")
        axes[row_idx, 2].set_title("Cache Hit Rate (%)")
        axes[row_idx, 2].bar_label(bars3, padding=1, fontsize=7, fmt="%.1f")
        if hit_vals:
            ymax = max(hit_vals)
            axes[row_idx, 2].set_ylim(0, ymax * 1.1)

        for ax in axes[row_idx]:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, fontsize=8, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    data = parse_stats_files(STATS_ROOT)
    print(f"Parsed {len(data)} stats logs from {STATS_ROOT}")
    for k, v in data.items():
        print(f"{k.label()}: TTFT={v['ttft_ms']:.2f} ms, Lat={v['latency_s']:.2f} s, Hit={v['hit_rate']:.2f}%")

    missing = compute_missing(list(data.keys()))
    if missing:
        print("Missing combinations (dc,hc,lora_count,lora_dist,kv_mem):")
        for tup in missing:
            print(f"  {tup}")

    plot_metrics(data, OUTPUT_PATH)
