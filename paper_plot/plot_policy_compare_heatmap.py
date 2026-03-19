#!/usr/bin/env python
"""
Heatmap comparison for eviction policies across kv-mem and request length.

Outputs:
    paper_plot/policy_compare_heatmap.png
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import colors as mcolors
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
STATS_ROOT = ROOT / "paper" / "2026-0127-mobilora_vs_baseline"
OUTPUT_HIT_PATH = Path(__file__).with_name("policy_compare_hit_rate.png")
OUTPUT_TTFT_PATH = Path(__file__).with_name("policy_compare_ttft.png")

POLICIES = ["lru", "lfu", "fifo", "mru", "filo", "mobilora"]
POLICY_LABELS = ["LRU", "LFU", "FIFO", "MRU", "FILO", "本方法"]
KV_VALUES = [1, 2, 4]
REQ_VALUES = [1024, 2048]

def format_config_label(kv: int, req: int) -> str:
    kv_label = f"{kv}G"
    if req % 1024 == 0:
        req_label = f"{req // 1024}K"
    else:
        req_label = f"{req}"
    return f"{kv_label}@{req_label}"


CONFIGS = [(kv, req, format_config_label(kv, req)) for kv in KV_VALUES for req in REQ_VALUES]


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


def collect_matrix(metric: str) -> np.ndarray:
    mat = np.full((len(POLICIES), len(CONFIGS)), np.nan, dtype=float)
    for i, policy in enumerate(POLICIES):
        for j, (kv, req, _) in enumerate(CONFIGS):
            data = find_latest_stats(req, kv, policy)
            if not data:
                continue
            if metric == "hit_rate":
                mat[i, j] = data["hit_rate"] * 100.0
            elif metric == "ttft_ms":
                mat[i, j] = data["avg_ttft_ms"]
    return mat


def plot_heatmap(
    ax, mat: np.ndarray, title: str, cmap: str, vmin=None, vmax=None, norm=None
):
    masked = np.ma.masked_invalid(mat)
    imshow_kwargs = {"cmap": cmap, "aspect": "auto"}
    if norm is not None:
        imshow_kwargs["norm"] = norm
    else:
        imshow_kwargs["vmin"] = vmin
        imshow_kwargs["vmax"] = vmax
    im = ax.imshow(masked, **imshow_kwargs)
    ax.set_title(title)
    ax.set_yticks(range(len(POLICIES)))
    ax.set_yticklabels(POLICY_LABELS)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([c[2] for c in CONFIGS])
    ax.tick_params(axis="x", labelrotation=0, labelsize=8, pad=2)
    ax.set_ylabel("")
    # show values
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                continue
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8, color="black")
    return im


BAD_COLOR = "#e6e6e6"


def truncate_colormap(name: str, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    base = plt.get_cmap(name)
    colors = base(np.linspace(minval, maxval, n))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        f"{name}_trunc_{minval:.2f}_{maxval:.2f}", colors
    )
    cmap.set_bad(BAD_COLOR)
    return cmap


def robust_range(mat: np.ndarray, lower: float = 5.0, upper: float = 95.0):
    vals = mat[np.isfinite(mat)]
    if vals.size == 0:
        return None, None
    if vals.size <= 3:
        return float(np.nanmin(vals)), float(np.nanmax(vals))
    vmin = float(np.nanpercentile(vals, lower))
    vmax = float(np.nanpercentile(vals, upper))
    if np.isclose(vmin, vmax):
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    return vmin, vmax


def configure_fonts() -> None:
    font_path = os.environ.get("SGLANG_PLOT_FONT_PATH")
    if font_path:
        try:
            font_manager.fontManager.addfont(font_path)
            prop = font_manager.FontProperties(fname=font_path)
            matplotlib.rcParams["font.sans-serif"] = [prop.get_name()]
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            pass

    path_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    ]
    for path in path_candidates:
        if not os.path.exists(path):
            continue
        try:
            font_manager.fontManager.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            matplotlib.rcParams["font.sans-serif"] = [prop.get_name()]
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            pass

    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
    ]
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue


def main() -> None:
    configure_fonts()
    hit = collect_matrix("hit_rate")
    ttft = collect_matrix("ttft_ms")

    hit_vmin, hit_vmax = robust_range(hit, 5.0, 95.0)
    hit_norm = (
        mcolors.PowerNorm(gamma=0.6, vmin=hit_vmin, vmax=hit_vmax)
        if hit_vmin is not None and hit_vmax is not None
        else None
    )

    fig1, ax1 = plt.subplots(1, 1, figsize=(7.0, 3.5))
    im1 = plot_heatmap(
        ax1,
        hit,
        "Cache Hit Rate (%)",
        cmap=truncate_colormap("YlGnBu", 0.35, 0.95),
        vmin=hit_vmin,
        vmax=hit_vmax,
        norm=hit_norm,
    )
    fig1.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
    plt.tight_layout()

    ttft_vmin, ttft_vmax = robust_range(ttft, 5.0, 95.0)
    ttft_norm = (
        mcolors.PowerNorm(gamma=0.75, vmin=ttft_vmin, vmax=ttft_vmax)
        if ttft_vmin is not None and ttft_vmax is not None
        else None
    )

    fig2, ax2 = plt.subplots(1, 1, figsize=(7.0, 3.5))
    im2 = plot_heatmap(
        ax2,
        ttft,
        "Avg TTFT (ms)",
        cmap=truncate_colormap("OrRd", 0.35, 0.95),
        vmin=ttft_vmin,
        vmax=ttft_vmax,
        norm=ttft_norm,
    )
    fig2.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    plt.tight_layout()

    OUTPUT_HIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(OUTPUT_HIT_PATH, dpi=220)
    fig2.savefig(OUTPUT_TTFT_PATH, dpi=220)
    print(f"Saved plot to {OUTPUT_HIT_PATH}")
    print(f"Saved plot to {OUTPUT_TTFT_PATH}")


if __name__ == "__main__":
    main()
