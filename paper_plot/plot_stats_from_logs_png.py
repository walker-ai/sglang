"""
Parse stats logs under paper/ and generate PNG bar charts (no matplotlib).

Usage:
    python paper_plot/plot_stats_from_logs_png.py

Outputs:
    paper_plot/ttft_bars.png
    paper_plot/latency_bars.png
    paper_plot/hit_rate_bars.png
    paper_plot/stats_table.csv
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import csv
import re
from typing import Dict, List, Tuple, Optional


STATS_ROOT = Path(__file__).resolve().parent.parent / "paper"
OUTPUT_DIR = Path(__file__).resolve().parent

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


def build_groups(
    results: Dict[ExpKey, Dict[str, float]]
) -> Dict[Tuple[int, str, int], Dict[Tuple[str, int, int], Dict[str, float]]]:
    groups: Dict[
        Tuple[int, str, int], Dict[Tuple[str, int, int], Dict[str, float]]
    ] = {}
    for key, metrics in results.items():
        scenario = (key.lora_count, key.lora_dist, key.kv_mem)
        mode = (key.bench, key.dc, key.hc)
        groups.setdefault(scenario, {})[mode] = metrics
    return groups


def _format_group_label(lora_count: int, lora_dist: str) -> str:
    return f"lora{lora_count}-{lora_dist}"


def _safe_max(values: List[Optional[float]]) -> float:
    filtered = [v for v in values if v is not None]
    return max(filtered) if filtered else 1.0


def _load_font(size: int):
    try:
        from PIL import ImageFont
        # Try common system font first.
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ):
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        return ImageFont.load_default()
    except Exception:
        return None


def render_png_bars_by_kv(
    groups: Dict[Tuple[int, str, int], Dict[Tuple[str, int, int], Dict[str, float]]],
    metric_key: str,
    title: str,
    unit: str,
    output_path: Path,
    value_fmt: str,
) -> None:
    from PIL import Image, ImageDraw

    scale = float(os.getenv("PLOT_SCALE", "2.0"))
    if scale <= 0:
        scale = 1.0

    mode_order = [
        ("sgl", 0, 0),
        ("vllm", 0, 0),
        ("sgl", 1, 0),
        ("sgl", 1, 1),
    ]
    mode_labels = {
        ("sgl", 0, 0): "baseline",
        ("vllm", 0, 0): "vllm",
        ("sgl", 1, 0): "dc",
        ("sgl", 1, 1): "dc+hc",
    }
    mode_colors = {
        ("sgl", 0, 0): (158, 202, 225),
        ("vllm", 0, 0): (188, 189, 220),
        ("sgl", 1, 0): (253, 174, 107),
        ("sgl", 1, 1): (161, 217, 155),
    }

    scenarios = sorted(groups.keys())
    kv_values = sorted({kv for (_, _, kv) in scenarios})
    kv_groups: Dict[int, List[Tuple[int, str, int]]] = {}
    for kv in kv_values:
        kv_groups[kv] = sorted(
            [s for s in scenarios if s[2] == kv],
            key=lambda s: (s[0], s[1]),
        )
    max_group_count = max((len(v) for v in kv_groups.values()), default=0)

    all_vals: List[Optional[float]] = []
    for scenario in scenarios:
        for mode in mode_order:
            metrics = groups[scenario].get(mode)
            if metrics is None:
                all_vals.append(None)
                continue
            val = metrics[metric_key]
            if metric_key == "latency_s":
                val = val * 1000.0
            all_vals.append(val)
    max_val = _safe_max(all_vals)
    max_val = max_val * 1.15 if max_val > 0 else 1.0

    # Layout constants
    bar_width = int(32 * scale)
    bar_gap = int(12 * scale)
    group_gap = int(26 * scale)
    left_margin = int(80 * scale)
    right_margin = int(30 * scale)
    top_margin = int(40 * scale)
    bottom_margin = int(40 * scale)
    panel_gap = int(40 * scale)
    panel_height = int(220 * scale)

    group_width = len(mode_order) * bar_width + (len(mode_order) - 1) * bar_gap
    plot_width = (
        left_margin
        + right_margin
        + max_group_count * group_width
        + max(0, max_group_count - 1) * group_gap
    )
    total_height = (
        top_margin
        + len(kv_values) * panel_height
        + max(0, len(kv_values) - 1) * panel_gap
        + bottom_margin
    )

    img = Image.new("RGB", (plot_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_title = _load_font(int(16 * scale))
    font_axis = _load_font(int(12 * scale))
    font_small = _load_font(int(11 * scale))
    font_tiny = _load_font(int(10 * scale))

    # Title
    title_w = draw.textlength(title, font=font_title) if font_title else 0
    draw.text(((plot_width - title_w) / 2, 12), title, fill=(0, 0, 0), font=font_title)

    tick_count = 5
    for panel_idx, kv in enumerate(kv_values):
        panel_top = top_margin + panel_idx * (panel_height + panel_gap)
        panel_bottom = panel_top + panel_height
        x0 = left_margin
        x1 = plot_width - right_margin

        # Axes
        draw.line((x0, panel_bottom, x1, panel_bottom), fill=(50, 50, 50), width=1)
        draw.line((x0, panel_top, x0, panel_bottom), fill=(50, 50, 50), width=1)

        # Panel label
        panel_label = f"kv{kv}g"
        draw.text((x0, panel_top - 18), panel_label, fill=(0, 0, 0), font=font_axis)
        if panel_idx == 0:
            draw.text((x0, panel_top - 32), unit, fill=(0, 0, 0), font=font_axis)

        # Y ticks + grid
        for i in range(tick_count + 1):
            val = max_val * i / tick_count
            y = panel_top + panel_height * (1.0 - val / max_val)
            draw.line((x0, y, x1, y), fill=(220, 220, 220), width=1)
            label = value_fmt % val
            label_w = draw.textlength(label, font=font_small) if font_small else 0
            draw.text((x0 - 8 - label_w, y - 6), label, fill=(0, 0, 0), font=font_small)

        # Bars
        scenarios_for_kv = kv_groups.get(kv, [])
        for g_idx, scenario in enumerate(scenarios_for_kv):
            group_x = x0 + g_idx * (group_width + group_gap)
            for m_idx, mode in enumerate(mode_order):
                metrics = groups[scenario].get(mode)
                if metrics is None:
                    val = None
                else:
                    val = metrics[metric_key]
                    if metric_key == "latency_s":
                        val = val * 1000.0
                if val is None:
                    bar_h = 0
                else:
                    bar_h = (val / max_val) * panel_height
                bar_x0 = group_x + m_idx * (bar_width + bar_gap)
                bar_x1 = bar_x0 + bar_width
                bar_y0 = panel_bottom - bar_h
                draw.rectangle(
                    (bar_x0, bar_y0, bar_x1, panel_bottom),
                    fill=mode_colors[mode],
                    outline=None,
                )
                if val is not None:
                    val_text = value_fmt % val
                    text_w = draw.textlength(val_text, font=font_tiny) if font_tiny else 0
                    draw.text(
                        (bar_x0 + (bar_width - text_w) / 2, bar_y0 - 12),
                        val_text,
                        fill=(0, 0, 0),
                        font=font_tiny,
                    )

            # Group label
            label = _format_group_label(scenario[0], scenario[1])
            label_w = draw.textlength(label, font=font_small) if font_small else 0
            label_x = group_x + (group_width - label_w) / 2
            draw.text((label_x, panel_bottom + 10), label, fill=(0, 0, 0), font=font_small)

    # Legend
    legend_x = plot_width - right_margin - 120
    legend_y = top_margin + 6
    for i, mode in enumerate(mode_order):
        y = legend_y + i * 18
        draw.rectangle((legend_x, y, legend_x + 12, y + 12), fill=mode_colors[mode])
        draw.text((legend_x + 16, y - 2), mode_labels[mode], fill=(0, 0, 0), font=font_small)

    img.save(output_path)


def write_csv(
    results: Dict[ExpKey, Dict[str, float]],
    output_path: Path,
) -> None:
    headers = [
        "bench",
        "dc",
        "hc",
        "lora_count",
        "lora_dist",
        "kv_mem",
        "req",
        "out",
        "clients",
        "rounds",
        "ttft_ms",
        "latency_s",
        "hit_rate",
        "path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for key, metrics in sorted(
            results.items(),
            key=lambda kv: (
                kv[0].lora_count,
                kv[0].lora_dist,
                kv[0].kv_mem,
                kv[0].dc,
                kv[0].hc,
            ),
        ):
            writer.writerow(
                [
                    key.bench,
                    key.dc,
                    key.hc,
                    key.lora_count,
                    key.lora_dist,
                    key.kv_mem,
                    key.req,
                    key.out,
                    key.clients,
                    key.rounds,
                    metrics["ttft_ms"],
                    metrics["latency_s"],
                    metrics["hit_rate"],
                    metrics["path"],
                ]
            )


def main() -> None:
    results = parse_stats_files(STATS_ROOT)
    if not results:
        print(f"No stats logs found under {STATS_ROOT}")
        return
    groups = build_groups(results)

    render_png_bars_by_kv(
        groups,
        metric_key="ttft_ms",
        title="TTFT (ms) by scenario",
        unit="ms",
        output_path=OUTPUT_DIR / "ttft_bars.png",
        value_fmt="%.0f",
    )
    render_png_bars_by_kv(
        groups,
        metric_key="latency_s",
        title="Latency (ms) by scenario",
        unit="ms",
        output_path=OUTPUT_DIR / "latency_bars.png",
        value_fmt="%.0f",
    )
    render_png_bars_by_kv(
        groups,
        metric_key="hit_rate",
        title="Cache Hit Rate (%) by scenario",
        unit="%",
        output_path=OUTPUT_DIR / "hit_rate_bars.png",
        value_fmt="%.1f",
    )
    write_csv(results, OUTPUT_DIR / "stats_table.csv")
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
