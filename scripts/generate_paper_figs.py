"""Generate paper figures from downloaded CSV results."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figs")

# IEEE style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "Baseline": "#1f77b4",
    "+NWD": "#ff7f0e",
    "+NWD+P2": "#2ca02c",
    "+NWD+P2+SimAM+PConv": "#9467bd",
}

CSVS = {
    "Baseline": os.path.join(FIGS_DIR, "baseline.csv"),
    "+NWD": os.path.join(FIGS_DIR, "nwd_only.csv"),
    "+NWD+P2": os.path.join(FIGS_DIR, "nwd_p2.csv"),
    "+NWD+P2+SimAM+PConv": os.path.join(FIGS_DIR, "nwd_p2_simam_pconv.csv"),
}


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def fig1_training_curves():
    """Fig 1: mAP50 training curves for ablation experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    for name, path in CSVS.items():
        df = load_csv(path)
        epochs = df["epoch"].values
        map50 = df["metrics/mAP50(B)"].values
        # Smooth with EMA for readability
        alpha = 0.9
        smooth = np.zeros_like(map50)
        smooth[0] = map50[0]
        for i in range(1, len(map50)):
            smooth[i] = alpha * smooth[i-1] + (1 - alpha) * map50[i]

        axes[0].plot(epochs, smooth, label=name, color=COLORS[name], linewidth=1.2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("mAP50")
    axes[0].set_title("(a) mAP50 Convergence")
    axes[0].legend(loc="lower right", framealpha=0.8)
    axes[0].set_xlim(0, 300)
    axes[0].set_ylim(0.4, 1.0)
    axes[0].grid(True, alpha=0.3)

    # Loss curves
    for name, path in CSVS.items():
        df = load_csv(path)
        epochs = df["epoch"].values
        box_loss = df["train/box_loss"].values
        alpha = 0.9
        smooth = np.zeros_like(box_loss)
        smooth[0] = box_loss[0]
        for i in range(1, len(box_loss)):
            smooth[i] = alpha * smooth[i-1] + (1 - alpha) * box_loss[i]
        axes[1].plot(epochs, smooth, label=name, color=COLORS[name], linewidth=1.2)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Box Loss")
    axes[1].set_title("(b) Training Box Loss")
    axes[1].legend(loc="upper right", framealpha=0.8)
    axes[1].set_xlim(0, 300)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "training_curves.png")
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")


def fig2_ablation_bar():
    """Fig 2: Ablation bar chart showing incremental improvements."""
    # Full ablation data (including SimAM and BiFPN negatives)
    names = ["Baseline", "+NWD", "+NWD+P2", "+SimAM", "+SimAM\n+PConv", "+BiFPN\n(Full)"]
    map50 = [0.9600, 0.9705, 0.9781, 0.9519, 0.9749, 0.9365]
    deltas = [0, 0.0105, 0.0076, -0.0262, 0.0230, -0.0384]
    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#d62728"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(range(len(names)), map50, color=bar_colors, width=0.6, edgecolor="black", linewidth=0.5)

    # Annotate bars
    for i, (bar, val, d) in enumerate(zip(bars, map50, deltas)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        if i > 0:
            sign = "+" if d >= 0 else ""
            color = "green" if d >= 0 else "red"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.015,
                    f"{sign}{d:.4f}", ha="center", va="top", fontsize=7, color=color)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("mAP50")
    ax.set_title("Ablation Study: Incremental Component Analysis")
    ax.set_ylim(0.90, 1.00)
    ax.axhline(y=map50[0], color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "ablation_bar.png")
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")


def fig3_pr_comparison():
    """Fig 3: Precision-Recall comparison between Baseline and +NWD+P2."""
    fig, ax = plt.subplots(figsize=(4, 3.5))

    for name in ["Baseline", "+NWD+P2"]:
        df = load_csv(CSVS[name])
        p = df["metrics/precision(B)"].values
        r = df["metrics/recall(B)"].values
        # Sort by recall for smooth curve
        idx = np.argsort(r)
        ax.scatter(r[idx], p[idx], s=8, alpha=0.3, color=COLORS[name])
        # Plot best point
        map50 = df["metrics/mAP50(B)"].values
        best_ep = np.argmax(map50)
        ax.scatter(r[best_ep], p[best_ep], s=80, marker="*", color=COLORS[name],
                   edgecolors="black", linewidth=0.5, zorder=5,
                   label=f"{name} (best: P={p[best_ep]:.3f}, R={r[best_ep]:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall (per epoch)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "pr_comparison.png")
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    fig1_training_curves()
    fig2_ablation_bar()
    fig3_pr_comparison()
    print("\nAll figures generated in paper/figs/")
