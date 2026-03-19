"""
Parse stats logs under paper/ and plot summary bars with matplotlib.

Usage:
    python paper_plot/plot_stats_from_logs_matplotlib.py

Outputs:
    paper_plot/stats_summary_by_kv_nomobilora_star.png
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from matplotlib import font_manager

_CJK_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if _CJK_FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_CJK_FONT_PATH))
    cjk_font = font_manager.FontProperties(fname=str(_CJK_FONT_PATH))
    plt.rcParams["font.family"] = cjk_font.get_name()
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


STATS_ROOT = Path(__file__).resolve().parent.parent / "paper"
OUTPUT_PATH = Path(__file__).with_name("stats_summary_by_kv_nomobilora_star.png")
PDF_OUTPUT_PATH = OUTPUT_PATH.with_suffix(".pdf")
STATS_TABLE_PATH = Path(__file__).with_name("stats_table.csv")

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


def parse_stats_table(path: Path) -> Dict[ExpKey, Dict[str, float]]:
    if not path.exists():
        return {}
    results: Dict[ExpKey, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                key = ExpKey(
                    bench=row["bench"],
                    dc=int(row["dc"]),
                    hc=int(row["hc"]),
                    lora_count=int(row["lora_count"]),
                    lora_dist=row["lora_dist"],
                    kv_mem=int(row["kv_mem"]),
                    req=int(row["req"]),
                    out=int(row["out"]),
                    clients=int(row["clients"]),
                    rounds=int(row["rounds"]),
                )
                results[key] = {
                    "ttft_ms": float(row["ttft_ms"]),
                    "latency_s": float(row["latency_s"]),
                    "hit_rate": float(row["hit_rate"]),
                    "path": row.get("path", ""),
                }
            except (KeyError, ValueError):
                continue
    return results


def _scenario_label(lora_count: int, lora_dist: str) -> str:
    return f"lora{lora_count}-{lora_dist}"


def _scenario_label_with_kv(lora_count: int, lora_dist: str, kv_mem: int) -> str:
    return f"lora{lora_count}-{lora_dist} kv{kv_mem}g"


def plot_by_kv(
    results: Dict[ExpKey, Dict[str, float]],
    output_path: Path,
) -> None:
    if not results:
        print("No stats data found.")
        return

    system_order = [
        ("vllm", 0, 0),
        ("sgl", 0, 0),
        ("sgl", 1, 0),
    ]
    system_labels = {
        ("sgl", 0, 0): "SGLang",
        ("vllm", 0, 0): "vLLM",
        ("sgl", 1, 0): "本方法",
    }
    system_colors = {
        ("sgl", 0, 0): "#9ecae1",
        ("vllm", 0, 0): "#bcbddc",
        ("sgl", 1, 0): "#fdae6b",
    }

    scenarios = sorted({(k.lora_count, k.lora_dist, k.kv_mem) for k in results})
    kv_values = sorted({kv for (_, _, kv) in scenarios})

    # Scenario list per kv (sorted by lora_count then dist)
    scenarios_by_kv: Dict[int, List[Tuple[int, str, int]]] = {}
    for kv in kv_values:
        scenarios_by_kv[kv] = sorted(
            [s for s in scenarios if s[2] == kv],
            key=lambda s: (s[0], s[1]),
        )

    def get_metric(
        scenario: Tuple[int, str, int],
        system: Tuple[str, int, int],
        metric: str,
    ) -> float:
        bench, dc, hc = system
        for k, v in results.items():
            if (
                k.lora_count == scenario[0]
                and k.lora_dist == scenario[1]
                and k.kv_mem == scenario[2]
                and k.dc == dc
                and k.hc == hc
                and k.bench == bench
            ):
                if metric == "latency_ms":
                    return v["latency_s"] * 1000.0
                return v[metric]
        return float("nan")

    metrics = [
        ("ttft_ms", "TTFT (ms)"),
        ("latency_ms", "Latency (ms)"),
        ("hit_rate", "Cache Hit Rate (%)"),
    ]

    # Global y max per metric to keep consistent scale across kv.
    metric_max: Dict[str, float] = {}
    for metric, _ in metrics:
        vals: List[float] = []
        for scenario in scenarios:
            for system in system_order:
                val = get_metric(scenario, system, metric)
                if not np.isnan(val):
                    vals.append(val)
        metric_max[metric] = max(vals) * 1.15 if vals else 1.0

    n_rows = len(kv_values)
    n_cols = len(metrics)
    fig_height = max(3, 3.4 * len(kv_values))
    fig = plt.figure(figsize=(13.5, fig_height))
    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        height_ratios=[1.15] * len(kv_values),
    )

    axes = [
        [fig.add_subplot(gs[r, c]) for c in range(n_cols)]
        for r in range(len(kv_values))
    ]

    for row_idx, kv in enumerate(kv_values):
        scenario_list = scenarios_by_kv[kv]
        x = np.arange(len(scenario_list))
        width = min(0.28, 0.8 / len(system_order))

        for col_idx, (metric_key, metric_title) in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            for m_idx, system in enumerate(system_order):
                vals = [
                    get_metric(s, system, metric_key) for s in scenario_list
                ]
                offset = (m_idx - (len(system_order) - 1) / 2) * width
                bars = ax.bar(
                    x + offset,
                    vals,
                    width=width,
                    label=system_labels[system],
                    color=system_colors[system],
                )
                ax.bar_label(bars, padding=1, fontsize=5, fmt="%.0f")

            ax.set_title(f"{metric_title} | KV-Mem for {kv} GB")
            ax.set_ylim(0, metric_max[metric_key])
            ax.set_xticks(x)
            ax.set_xticklabels(
                [_scenario_label(s[0], s[1]) for s in scenario_list],
                rotation=30,
                fontsize=8,
                ha="right",
            )
            ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Only add legend once.
    axes[0][0].legend(
        loc="upper right",
        fontsize=7,
        frameon=False,
        handlelength=1.0,
        labelspacing=0.3,
        borderpad=0.2,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    print(f"Saved plot to {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    data = parse_stats_files(STATS_ROOT)
    if not data:
        data = parse_stats_table(STATS_TABLE_PATH)
    plot_by_kv(data, OUTPUT_PATH)
