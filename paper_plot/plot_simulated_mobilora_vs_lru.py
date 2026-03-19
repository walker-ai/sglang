import os

os.environ.setdefault("KMP_SHM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_line(ax, x_labels, y_lru, y_mobilora, ylabel, title):
    x = list(range(len(x_labels)))
    ax.plot(x, y_lru, marker="o", linewidth=2.0, label="LRU")
    ax.plot(x, y_mobilora, marker="o", linewidth=2.0, label="MobiLoRA")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="best")


def main():
    # Simulated data for presentation. Edit these arrays to match your setup.
    group_params = [
        {"kv_gb": 1, "fg_prob": 0.05, "inactive_prob": 0.50, "hotset": 6, "phase_req": 30},
        {"kv_gb": 1, "fg_prob": 0.10, "inactive_prob": 0.50, "hotset": 6, "phase_req": 30},
        {"kv_gb": 2, "fg_prob": 0.05, "inactive_prob": 0.50, "hotset": 6, "phase_req": 30},
        {"kv_gb": 2, "fg_prob": 0.10, "inactive_prob": 0.60, "hotset": 6, "phase_req": 30},
        {"kv_gb": 2, "fg_prob": 0.10, "inactive_prob": 0.70, "hotset": 6, "phase_req": 30},
    ]
    groups = ["Config A", "Config B", "Config C", "Config D", "Config E"]

    # Cache hit rate in percent.
    lru_hit_rate = [2.5, 3.1, 4.2, 5.0, 5.8]
    mobilora_hit_rate = [8.4, 9.8, 11.6, 13.1, 14.3]

    # TTFT in milliseconds.
    lru_ttft_ms = [650, 620, 600, 585, 570]
    mobilora_ttft_ms = [580, 555, 535, 515, 500]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.0), dpi=160)
    plot_line(
        ax1,
        groups,
        lru_hit_rate,
        mobilora_hit_rate,
        ylabel="Cache Hit Rate (%)",
        title="Cache Hit Rate by Config",
    )
    plot_line(
        ax2,
        groups,
        lru_ttft_ms,
        mobilora_ttft_ms,
        ylabel="Avg TTFT (ms)",
        title="TTFT by Config",
    )
    fig.tight_layout()
    fig.savefig("sglang/paper_plot/mobilora_vs_lru_combined.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
