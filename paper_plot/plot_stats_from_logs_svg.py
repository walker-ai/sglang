"""
Parse stats logs under paper/ and generate simple SVG bar charts (no matplotlib).

Usage:
    python paper_plot/plot_stats_from_logs_svg.py

Outputs:
    paper_plot/ttft_bars.svg
    paper_plot/latency_bars.svg
    paper_plot/hit_rate_bars.svg
    paper_plot/stats_table.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
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
) -> Dict[Tuple[int, str, int], Dict[Tuple[int, int], Dict[str, float]]]:
    groups: Dict[Tuple[int, str, int], Dict[Tuple[int, int], Dict[str, float]]] = {}
    for key, metrics in results.items():
        scenario = (key.lora_count, key.lora_dist, key.kv_mem)
        mode = (key.dc, key.hc)
        groups.setdefault(scenario, {})[mode] = metrics
    return groups


def _format_group_label(lora_count: int, lora_dist: str, kv_mem: int) -> List[str]:
    dist_map = {"uniform": "u", "weighted": "w"}
    dist_short = dist_map.get(lora_dist, lora_dist[:1])
    return [f"l{lora_count}-{dist_short}", f"kv{kv_mem}g"]


def _safe_max(values: List[Optional[float]]) -> float:
    filtered = [v for v in values if v is not None]
    return max(filtered) if filtered else 1.0


def render_svg_bars_by_kv(
    groups: Dict[Tuple[int, str, int], Dict[Tuple[int, int], Dict[str, float]]],
    metric_key: str,
    title: str,
    unit: str,
    output_path: Path,
    value_fmt: str,
) -> None:
    mode_order = [(0, 0), (1, 0), (1, 1)]
    mode_labels = {
        (0, 0): "baseline",
        (1, 0): "dc",
        (1, 1): "dc+hc",
    }
    mode_colors = {
        (0, 0): "#9ecae1",
        (1, 0): "#fdae6b",
        (1, 1): "#a1d99b",
    }

    scenarios = sorted(groups.keys())
    kv_values = sorted({kv for (_, _, kv) in scenarios})
    kv_groups: Dict[int, List[Tuple[int, str, int]]] = {
        kv: [s for s in scenarios if s[2] == kv] for kv in kv_values
    }
    max_group_count = max(len(v) for v in kv_groups.values()) if kv_groups else 0

    # Global max for shared y scale
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
    n_groups = max_group_count
    n_modes = len(mode_order)
    bar_width = 18
    bar_gap = 6
    group_gap = 20
    left_margin = 60
    right_margin = 20
    top_margin = 40
    bottom_margin = 40
    panel_gap = 40
    panel_height = 200
    plot_width = (
        left_margin
        + right_margin
        + n_groups * (n_modes * bar_width + (n_modes - 1) * bar_gap)
        + (n_groups - 1) * group_gap
    )
    total_height = (
        top_margin
        + len(kv_values) * panel_height
        + (len(kv_values) - 1) * panel_gap
        + bottom_margin
    )

    def y_for_val(v: float) -> float:
        return panel_top + panel_height * (1.0 - v / max_val)

    lines: List[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{plot_width}" height="{total_height}">'
    )
    # Title
    lines.append(
        f'<text x="{plot_width/2:.1f}" y="20" text-anchor="middle" '
        f'font-size="14" font-family="Arial">{title}</text>'
    )

    tick_count = 5
    for panel_idx, kv in enumerate(kv_values):
        panel_top = top_margin + panel_idx * (panel_height + panel_gap)
        x0 = left_margin
        y0 = panel_top + panel_height
        panel_label = f"kv{kv}g"

        # Axes
        lines.append(
            f'<line x1="{x0}" y1="{y0}" x2="{plot_width-right_margin}" y2="{y0}" stroke="#333"/>'
        )
        lines.append(
            f'<line x1="{x0}" y1="{panel_top}" x2="{x0}" y2="{y0}" stroke="#333"/>'
        )

        # Panel label
        lines.append(
            f'<text x="{x0}" y="{panel_top-8}" text-anchor="start" font-size="11" font-family="Arial">{panel_label}</text>'
        )

        # Y ticks
        for i in range(tick_count + 1):
            val = max_val * i / tick_count
            y = y_for_val(val)
            lines.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{plot_width-right_margin}" y2="{y:.1f}" stroke="#ddd"/>'
            )
            lines.append(
                f'<text x="{x0-6}" y="{y+4:.1f}" text-anchor="end" font-size="10" font-family="Arial">{value_fmt % val}</text>'
            )
        if panel_idx == 0:
            lines.append(
                f'<text x="{x0}" y="{panel_top-22}" text-anchor="start" font-size="11" font-family="Arial">{unit}</text>'
            )

        # Bars
        scenarios_for_kv = kv_groups.get(kv, [])
        for g_idx, scenario in enumerate(scenarios_for_kv):
            group_x = (
                left_margin
                + g_idx * (n_modes * bar_width + (n_modes - 1) * bar_gap + group_gap)
            )
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
                x = group_x + m_idx * (bar_width + bar_gap)
                y = y0 - bar_h
                color = mode_colors[mode]
                lines.append(
                    f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_h:.1f}" fill="{color}"/>'
                )
                # value label
                if val is not None:
                    lines.append(
                        f'<text x="{x + bar_width/2:.1f}" y="{y-4:.1f}" text-anchor="middle" '
                        f'font-size="9" font-family="Arial">{value_fmt % val}</text>'
                    )
                else:
                    lines.append(
                        f'<text x="{x + bar_width/2:.1f}" y="{y-4:.1f}" text-anchor="middle" '
                        f'font-size="9" font-family="Arial">NA</text>'
                    )

            # Group label (two lines)
            label_lines = _format_group_label(*scenario)
            label_x = group_x + (n_modes * bar_width + (n_modes - 1) * bar_gap) / 2
            label_y = y0 + 16
            lines.append(
                f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle" font-size="10" font-family="Arial">'
                f'<tspan x="{label_x:.1f}" dy="0">{label_lines[0]}</tspan>'
                f'<tspan x="{label_x:.1f}" dy="12">{label_lines[1]}</tspan>'
                f"</text>"
            )

    # Legend
    legend_x = plot_width - right_margin - 120
    legend_y = top_margin + 4
    for i, mode in enumerate(mode_order):
        y = legend_y + i * 16
        lines.append(f'<rect x="{legend_x}" y="{y-10}" width="12" height="12" fill="{mode_colors[mode]}"/>')
        lines.append(
            f'<text x="{legend_x+16}" y="{y}" font-size="10" font-family="Arial">{mode_labels[mode]}</text>'
        )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


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

    render_svg_bars_by_kv(
        groups,
        metric_key="ttft_ms",
        title="TTFT (ms) by scenario",
        unit="ms",
        output_path=OUTPUT_DIR / "ttft_bars.svg",
        value_fmt="%.0f",
    )
    render_svg_bars_by_kv(
        groups,
        metric_key="latency_s",
        title="Latency (ms) by scenario",
        unit="ms",
        output_path=OUTPUT_DIR / "latency_bars.svg",
        value_fmt="%.0f",
    )
    render_svg_bars_by_kv(
        groups,
        metric_key="hit_rate",
        title="Cache Hit Rate (%) by scenario",
        unit="%",
        output_path=OUTPUT_DIR / "hit_rate_bars.svg",
        value_fmt="%.1f",
    )
    write_csv(results, OUTPUT_DIR / "stats_table.csv")
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
