"""
Grouped bar chart for TTFT (ms) comparisons.

Edit `ttft_data` with your groups and numbers, then run:
    python ttft_bars.py
This saves a PNG next to the script.
"""

from pathlib import Path

import matplotlib

# Use a non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Example TTFT values in milliseconds; replace with your own.
ttft_data = {
    "Group A": {"baseline1": 180, "baseline2": 155, "baseline3": 140, "ours": 95},
    "Group B": {"baseline1": 210, "baseline2": 190, "baseline3": 170, "ours": 120},
    "Group C": {"baseline1": 160, "baseline2": 150, "baseline3": 140, "ours": 100},
}

SYSTEMS = ["baseline1", "baseline2", "baseline3", "ours"]
SYSTEM_LABELS = {
    "baseline1": "Baseline 1",
    "baseline2": "Baseline 2",
    "baseline3": "Baseline 3",
    "ours": "Ours",
}
SYSTEM_COLORS = {
    "baseline1": "#9ecae1",
    "baseline2": "#fdae6b",
    "baseline3": "#a1d99b",
    "ours": "#ef3b2c",
}


def plot_ttft_bars(data: dict, output_path: Path) -> None:
    """Plot grouped bars comparing baselines against ours for each group."""
    groups = list(data.keys())
    x = np.arange(len(groups))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, system in enumerate(SYSTEMS):
        values = [data[group][system] for group in groups]
        offset = (idx - (len(SYSTEMS) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=SYSTEM_LABELS.get(system, system),
            color=SYSTEM_COLORS.get(system),
        )
        ax.bar_label(bars, padding=2, fontsize=9, fmt="%.0f")

    ax.set_xlabel("Group")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT comparison by group")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    output_file = Path(__file__).with_suffix(".png")
    plot_ttft_bars(ttft_data, output_file)
