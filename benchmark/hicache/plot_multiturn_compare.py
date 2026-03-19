#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


LOWER_IS_BETTER = {
    "average_ttft",
    "p90_ttft",
    "median_ttft",
    "average_latency",
    "p90_latency",
    "median_latency",
}

HIGHER_IS_BETTER = {
    "throughput",
    "input_token_throughput",
    "output_token_throughput",
    "cache_hit_rate",
}

# Requested default focus metrics (avoid mixed-scale "everything in one axis").
DEFAULT_METRICS = [
    "median_ttft",
    "cache_hit_rate",
    "p90_ttft",
    "median_latency",
    "p90_latency",
]

# Some environments (notably conda+MKL) may abort when MKL uses Intel threading + SHM.
# For plotting, forcing GNU threading is typically safe and avoids those crashes.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")


@dataclass(frozen=True)
class RunRecord:
    label: str
    source: Path
    timestamp: Optional[datetime]
    tag: str
    request_rate: float
    summary: dict[str, Any]
    round_metrics: Optional[dict[str, Any]]


def _parse_timestamp(v: Any) -> Optional[datetime]:
    if not v:
        return None
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return None
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON in {path}:{i}: {e}") from e
    return out


def _coerce_float(v: Any, *, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _matches_tag(row: dict[str, Any], tag: Optional[str]) -> bool:
    if tag is None:
        return True
    return str(row.get("tag", "")) == tag


def _extract_runs(*, label: str, path: Path, tag: Optional[str]) -> list[RunRecord]:
    rows = _load_jsonl(path)
    runs: list[RunRecord] = []
    for row in rows:
        if not _matches_tag(row, tag):
            continue
        summary = row.get("summary")
        if not isinstance(summary, dict):
            continue
        request_rate = _coerce_float(summary.get("request_rate"), default=float("nan"))
        if math.isnan(request_rate):
            continue
        runs.append(
            RunRecord(
                label=label,
                source=path,
                timestamp=_parse_timestamp(row.get("timestamp")),
                tag=str(row.get("tag", "")),
                request_rate=request_rate,
                summary=summary,
                round_metrics=row.get("round") if isinstance(row.get("round"), dict) else None,
            )
        )
    return runs


def _select_per_rate(
    runs: list[RunRecord], *, mode: str
) -> dict[float, RunRecord]:
    if mode == "latest":
        chosen: dict[float, RunRecord] = {}
        for r in runs:
            cur = chosen.get(r.request_rate)
            if cur is None:
                chosen[r.request_rate] = r
                continue
            if (cur.timestamp or datetime.min) <= (r.timestamp or datetime.min):
                chosen[r.request_rate] = r
        return chosen
    raise ValueError(f"Unsupported mode: {mode}")


def _round_series(round_metrics: dict[str, Any], key: str) -> list[float]:
    points: list[tuple[int, float]] = []
    for round_key, payload in round_metrics.items():
        if not round_key.startswith("round_"):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            idx = int(round_key.split("_", 1)[1])
        except Exception:
            continue
        v = _coerce_float(payload.get(key), default=float("nan"))
        if math.isnan(v):
            continue
        points.append((idx, v))
    points.sort(key=lambda x: x[0])
    if not points:
        return []
    max_idx = max(i for i, _ in points)
    series = [float("nan")] * (max_idx + 1)
    for i, v in points:
        series[i] = v
    return series


def _metric_list(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    keys = set(a.keys()) & set(b.keys())
    return [m for m in DEFAULT_METRICS if m in keys]


def _improvement_pct(metric: str, off: float, on: float) -> Optional[float]:
    if math.isnan(off) or math.isnan(on):
        return None
    if off == 0:
        return None
    if metric in LOWER_IS_BETTER:
        return (off - on) / off * 100.0
    if metric in HIGHER_IS_BETTER:
        return (on - off) / off * 100.0
    return None


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install it (e.g. `pip install matplotlib`) "
            "or run this script in an environment that already has it."
        ) from e
    return plt


def _plot_summary_grid(
    *,
    out_path: Path,
    title: str,
    off: RunRecord,
    on: RunRecord,
    metrics: list[str],
) -> None:
    plt = _require_matplotlib()

    n = len(metrics)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12.6, 3.4 * rows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        off_v = _coerce_float(off.summary.get(metric))
        on_v = _coerce_float(on.summary.get(metric))
        ax.barh([0, 1], [off_v, on_v], color=["#4C78A8", "#F58518"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([off.label, on.label])
        ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.4)
        ax.set_title(metric)
        if metric == "cache_hit_rate":
            ax.set_xlim(0.0, 1.0)
        for y, v in enumerate([off_v, on_v]):
            if not math.isnan(v):
                ax.text(v, y, f" {v:.4g}", va="center", ha="left")

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_metrics_vs_rate(
    *,
    out_path: Path,
    title: str,
    off_by_rate: dict[float, RunRecord],
    on_by_rate: dict[float, RunRecord],
    metrics: list[str],
) -> None:
    plt = _require_matplotlib()

    rates = sorted(set(off_by_rate.keys()) & set(on_by_rate.keys()))
    if not rates:
        return

    shown_metrics = metrics[:6] if len(metrics) > 6 else metrics
    n = len(shown_metrics)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12.6, 3.6 * rows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for idx, metric in enumerate(shown_metrics):
        ax = axes_flat[idx]
        off_vals = [_coerce_float(off_by_rate[r].summary.get(metric)) for r in rates]
        on_vals = [_coerce_float(on_by_rate[r].summary.get(metric)) for r in rates]
        ax.plot(rates, off_vals, marker="o", label=off_by_rate[rates[0]].label)
        ax.plot(rates, on_vals, marker="o", label=on_by_rate[rates[0]].label)
        ax.set_title(metric)
        ax.set_xlabel("request_rate (req/s)")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
        if metric in LOWER_IS_BETTER:
            ax.set_ylabel("seconds")
        ax.legend(loc="best")

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_per_round(
    *,
    out_path: Path,
    title: str,
    metric: str,
    off: RunRecord,
    on: RunRecord,
) -> bool:
    if not off.round_metrics or not on.round_metrics:
        return False

    plt = _require_matplotlib()
    off_series = _round_series(off.round_metrics, metric)
    on_series = _round_series(on.round_metrics, metric)
    if not off_series or not on_series:
        return False

    rounds = list(range(max(len(off_series), len(on_series))))
    off_y = off_series + [float("nan")] * (len(rounds) - len(off_series))
    on_y = on_series + [float("nan")] * (len(rounds) - len(on_series))

    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.plot(rounds, off_y, marker="o", label=off.label)
    ax.plot(rounds, on_y, marker="o", label=on.label)
    ax.set_xlabel("round")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    if metric.endswith("ttft"):
        ax.set_ylabel("seconds")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _format_value(metric: str, v: float) -> str:
    if math.isnan(v):
        return ""
    if metric == "cache_hit_rate":
        return f"{v:.6f}"
    if abs(v) >= 1000:
        return f"{v:.4g}"
    return f"{v:.6g}"


def _plot_overall_table(
    *,
    out_path: Path,
    title: str,
    off: RunRecord,
    on: RunRecord,
    metrics: list[str],
) -> None:
    plt = _require_matplotlib()

    rows: list[list[str]] = []
    for m in metrics:
        off_v = _coerce_float(off.summary.get(m))
        on_v = _coerce_float(on.summary.get(m))
        imp = _improvement_pct(m, off_v, on_v)
        imp_s = "" if imp is None else f"{imp:+.2f}%"
        rows.append(
            [
                m,
                _format_value(m, off_v),
                _format_value(m, on_v),
                imp_s,
            ]
        )

    fig_h = max(2.8, 0.38 * (len(rows) + 1) + 1.6)
    fig, ax = plt.subplots(figsize=(10.8, fig_h))
    ax.axis("off")
    ax.set_title(title)

    table = ax.table(
        cellText=rows,
        colLabels=["metric", off.label, on.label, "improvement"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)

    # Light styling: bold header row.
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_overall_improvement(
    *,
    out_path: Path,
    title: str,
    off: RunRecord,
    on: RunRecord,
    metrics: list[str],
) -> None:
    plt = _require_matplotlib()

    labels: list[str] = []
    values: list[float] = []
    for m in metrics:
        off_v = _coerce_float(off.summary.get(m))
        on_v = _coerce_float(on.summary.get(m))
        imp = _improvement_pct(m, off_v, on_v)
        if imp is None:
            continue
        labels.append(m)
        values.append(imp)

    if not labels:
        return

    fig_h = max(3.0, 0.42 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(10.8, fig_h))
    ys = list(range(len(labels)))
    ax.barh(ys, values, color="#54A24B")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.set_xlabel("improvement (%)  (positive = better)")
    ax.set_title(title)
    ax.invert_yaxis()
    for y, v in enumerate(values):
        ax.text(v, y, f" {v:+.2f}%", va="center", ha="left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare bench_multiturn.py JSONL outputs (hicache on vs off) and plot key metrics."
    )
    p.add_argument("--on", type=Path, required=True, help="JSONL log from hicache-enabled server run")
    p.add_argument("--off", type=Path, required=True, help="JSONL log from hicache-disabled server run")
    p.add_argument("--tag-on", type=str, default=None, help="Only use rows whose tag equals this value")
    p.add_argument("--tag-off", type=str, default=None, help="Only use rows whose tag equals this value")
    p.add_argument("--label-on", type=str, default="hicache_on")
    p.add_argument("--label-off", type=str, default="hicache_off")
    p.add_argument("--out-dir", type=Path, default=Path("benchmark/hicache/plots"))
    p.add_argument("--mode", type=str, default="latest", choices=["latest"])
    p.add_argument(
        "--select-rate",
        type=float,
        default=None,
        help="If set, generate per-round plots for this request_rate (exact match).",
    )
    p.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help=(
            "Comma-separated summary metrics to compare/plot. "
            f"Default: {','.join(DEFAULT_METRICS)}"
        ),
    )
    args = p.parse_args()

    on_runs = _extract_runs(label=args.label_on, path=args.on, tag=args.tag_on)
    off_runs = _extract_runs(label=args.label_off, path=args.off, tag=args.tag_off)
    if not on_runs:
        raise SystemExit(f"No usable runs found in {args.on} (tag={args.tag_on!r}).")
    if not off_runs:
        raise SystemExit(f"No usable runs found in {args.off} (tag={args.tag_off!r}).")

    on_by_rate = _select_per_rate(on_runs, mode=args.mode)
    off_by_rate = _select_per_rate(off_runs, mode=args.mode)
    rates = sorted(set(on_by_rate.keys()) & set(off_by_rate.keys()))
    if not rates:
        raise SystemExit(
            f"No overlapping request_rate between {args.on} and {args.off}. "
            f"on has {sorted(on_by_rate.keys())}, off has {sorted(off_by_rate.keys())}."
        )

    selected_rate = args.select_rate if args.select_rate is not None else rates[-1]
    if selected_rate not in on_by_rate or selected_rate not in off_by_rate:
        raise SystemExit(
            f"--select-rate {selected_rate} is not available in both logs. "
            f"available overlap: {rates}"
        )

    on_sel = on_by_rate[selected_rate]
    off_sel = off_by_rate[selected_rate]
    requested_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not requested_metrics:
        raise SystemExit("--metrics is empty.")
    # Respect requested order, but only keep metrics present in both summaries.
    present = set(off_sel.summary.keys()) & set(on_sel.summary.keys())
    metrics = [m for m in requested_metrics if m in present]
    if not metrics:
        raise SystemExit("No comparable metrics found between on/off summaries.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Dump normalized tables for easy inspection / spreadsheets.
    summary_rows: list[dict[str, Any]] = []
    for rate in rates:
        for rec in (off_by_rate[rate], on_by_rate[rate]):
            row = {
                "label": rec.label,
                "source": str(rec.source),
                "tag": rec.tag,
                "timestamp": rec.timestamp.isoformat() if rec.timestamp else "",
                "request_rate": rec.request_rate,
            }
            for m in metrics:
                row[m] = rec.summary.get(m)
            summary_rows.append(row)
    _write_csv(args.out_dir / "summary_table.csv", summary_rows)

    improvement_rows: list[dict[str, Any]] = []
    for rate in rates:
        off = off_by_rate[rate]
        on = on_by_rate[rate]
        for m in metrics:
            off_v = _coerce_float(off.summary.get(m))
            on_v = _coerce_float(on.summary.get(m))
            imp = _improvement_pct(m, off_v, on_v)
            improvement_rows.append(
                {
                    "request_rate": rate,
                    "metric": m,
                    "off": off_v,
                    "on": on_v,
                    "improvement_pct": imp if imp is not None else "",
                }
            )
    _write_csv(args.out_dir / "improvement_table.csv", improvement_rows)

    # A compact "overall" view for the selected request_rate.
    overall_rows: list[dict[str, Any]] = []
    for rec in (off_sel, on_sel):
        row = {"label": rec.label, "request_rate": selected_rate}
        for m in metrics:
            row[m] = rec.summary.get(m)
        overall_rows.append(row)
    _write_csv(args.out_dir / f"overall_rate_{selected_rate:g}.csv", overall_rows)

    # Plots.
    title_suffix = f"(request_rate={selected_rate:g} req/s)"
    _plot_summary_grid(
        out_path=args.out_dir / "summary_compare.png",
        title=f"bench_multiturn summary (selected metrics): {args.label_on} vs {args.label_off} {title_suffix}",
        off=off_sel,
        on=on_sel,
        metrics=metrics,
    )
    _plot_overall_table(
        out_path=args.out_dir / f"overall_table_rate_{selected_rate:g}.png",
        title=f"Overall summary table {title_suffix}",
        off=off_sel,
        on=on_sel,
        metrics=metrics,
    )
    _plot_overall_improvement(
        out_path=args.out_dir / f"overall_improvement_rate_{selected_rate:g}.png",
        title=f"Overall improvement {title_suffix}",
        off=off_sel,
        on=on_sel,
        metrics=metrics,
    )
    if len(rates) > 1:
        _plot_metrics_vs_rate(
            out_path=args.out_dir / "summary_vs_rate.png",
            title=f"bench_multiturn metrics vs request_rate: {args.label_on} vs {args.label_off}",
            off_by_rate=off_by_rate,
            on_by_rate=on_by_rate,
            metrics=metrics,
        )

    _plot_per_round(
        out_path=args.out_dir / f"per_round_average_ttft_rate_{selected_rate:g}.png",
        title=f"Per-round average_ttft {title_suffix}",
        metric="average_ttft",
        off=off_sel,
        on=on_sel,
    )
    _plot_per_round(
        out_path=args.out_dir / f"per_round_cache_hit_rate_rate_{selected_rate:g}.png",
        title=f"Per-round cache_hit_rate {title_suffix}",
        metric="cache_hit_rate",
        off=off_sel,
        on=on_sel,
    )

    print(f"Wrote plots and tables to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
