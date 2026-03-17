"""
plot_results.py — 生成论文所需的所有可视化图表

运行时机：collect_results.py 完成后
用法：python scripts/plot_results.py

生成图表：
  paper/figs/training_curves.pdf     — 训练曲线（mAP50 vs epoch）
  paper/figs/ablation_bar.pdf        — 消融实验柱状图
  paper/figs/pareto_curve.pdf        — 精度-效率权衡散点图
  paper/figs/overfitting_analysis.pdf — 过拟合分析（纯NWD vs 两阶段）
"""

import os
import sys
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# ---- 论文级别 matplotlib 配置 ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

# 色盲友好配色（Paul Tol Bright）
COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44",
           "#66CCEE", "#AA3377", "#BBBBBB", "#000000"]


def load_results_csv(exp_dir):
    """加载 ultralytics results.csv。"""
    # 兼容路径嵌套
    candidates = [
        Path(exp_dir) / "results.csv",
        Path("runs/detect/runs/detect") / Path(exp_dir).name / "results.csv",
        Path("runs/detect") / Path(exp_dir).name / "results.csv",
        Path("runs/ablation") / Path(exp_dir).name / "results.csv",
    ]
    csv_path = None
    for c in candidates:
        if c.exists():
            csv_path = c
            break
    if csv_path is None:
        print(f"  [WARN] No results.csv for {exp_dir}")
        return None

    data = {"epoch": [], "map50": [], "map50_95": [], "box_loss": []}
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        idx_map = {
            "epoch": next((i for i, h in enumerate(header) if "epoch" in h.lower()), None),
            "map50": next((i for i, h in enumerate(header) if "map50(b)" in h.lower() and "95" not in h.lower()), None),
            "map50_95": next((i for i, h in enumerate(header) if "map50-95" in h.lower()), None),
            "box_loss": next((i for i, h in enumerate(header) if "box_loss" in h.lower()), None),
        }
        for row in reader:
            row = [r.strip() for r in row]
            for key, idx in idx_map.items():
                if idx is not None:
                    try:
                        data[key].append(float(row[idx]))
                    except (ValueError, IndexError):
                        data[key].append(0.0)

    return {k: np.array(v) for k, v in data.items()}


def plot_training_curves(experiments, save_path):
    """绘制多实验训练曲线对比图。

    Args:
        experiments: list of (label, exp_dir)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=False)

    for i, (label, exp_dir) in enumerate(experiments):
        data = load_results_csv(exp_dir)
        if data is None or len(data["epoch"]) == 0:
            continue
        color = COLORS[i % len(COLORS)]
        ax1.plot(data["epoch"], data["map50"] * 100, color=color,
                 label=label, alpha=0.85)
        ax2.plot(data["epoch"], data["map50_95"] * 100, color=color,
                 label=label, alpha=0.85)

    ax1.set_ylabel("mAP@0.5 (%)")
    ax1.legend(loc="lower right", fontsize=7)
    ax1.set_ylim(bottom=0)

    ax2.set_ylabel("mAP@0.5:0.95 (%)")
    ax2.set_xlabel("Epoch")
    ax2.legend(loc="lower right", fontsize=7)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ablation_bar(results_json_path, save_path):
    """消融实验柱状图。"""
    if not Path(results_json_path).exists():
        print(f"  [SKIP] {results_json_path} not found")
        return

    with open(results_json_path) as f:
        results = json.load(f)

    ablation = [r for r in results if any(
        kw in r["name"] for kw in ["Baseline", "NWD", "P2", "SimAM", "PConv", "Full"]
    )]
    if not ablation:
        return

    names = [r["name"] for r in ablation]
    map50 = [r["map50"] * 100 for r in ablation]
    map50_95 = [r["map50_95"] * 100 for r in ablation]
    baseline = map50[0]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.2), 4))
    bars1 = ax.bar(x - width/2, map50, width, label="mAP@0.5",
                   color=COLORS[0], alpha=0.85)
    bars2 = ax.bar(x + width/2, map50_95, width, label="mAP@0.5:0.95",
                   color=COLORS[1], alpha=0.85)

    # Baseline 参考线
    ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1,
               alpha=0.7, label=f"Baseline ({baseline:.1f}%)")

    # 数值标注
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mAP (%)")
    ax.set_title("Ablation Study")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=max(0, min(map50) - 5))

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pareto(results_json_path, save_path):
    """精度-效率权衡散点图。"""
    if not Path(results_json_path).exists():
        print(f"  [SKIP] {results_json_path} not found")
        return

    with open(results_json_path) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, r in enumerate(results):
        is_ours = "Two-Stage" in r["name"] or "SOTA" in r["name"]
        color = COLORS[0] if is_ours else COLORS[6]
        marker = "*" if is_ours else "o"
        size = 150 if is_ours else 60
        ax.scatter(r["params_M"], r["map50"] * 100,
                   c=color, marker=marker, s=size, zorder=5)
        ax.annotate(r["name"].replace("Two-Stage", "TS"),
                    (r["params_M"], r["map50"] * 100),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7)

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_title("Accuracy-Efficiency Trade-off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_overfitting_analysis(nwd_exp_dir, two_stage_exp_dir, save_path):
    """过拟合分析：纯 NWD vs 两阶段训练。"""
    data_nwd = load_results_csv(nwd_exp_dir)
    data_ts = load_results_csv(two_stage_exp_dir)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    if data_nwd is not None and len(data_nwd["epoch"]) > 0:
        ax.plot(data_nwd["epoch"], data_nwd["map50"] * 100,
                color=COLORS[1], label="Pure NWD (direct training)",
                linewidth=2, alpha=0.85)
        # 标注峰值
        peak_idx = int(np.argmax(data_nwd["map50"]))
        peak_val = data_nwd["map50"][peak_idx] * 100
        peak_ep = data_nwd["epoch"][peak_idx]
        ax.axvline(x=peak_ep, color=COLORS[1], linestyle="--",
                   alpha=0.5, linewidth=1)
        ax.annotate(f"Peak\n{peak_val:.1f}%\nep.{int(peak_ep)}",
                    xy=(peak_ep, peak_val),
                    xytext=(peak_ep + max(5, len(data_nwd["epoch"]) * 0.05),
                            peak_val - 5),
                    fontsize=7,
                    arrowprops=dict(arrowstyle="->", color=COLORS[1]))

    if data_ts is not None and len(data_ts["epoch"]) > 0:
        ax.plot(data_ts["epoch"], data_ts["map50"] * 100,
                color=COLORS[0], label="Two-Stage (CIoU→hybrid SA-NWD)",
                linewidth=2, alpha=0.85)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val mAP@0.5 (%)")
    ax.set_title("Training Stability: Pure NWD vs Two-Stage")
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 60)
    print("  Generating Paper Figures")
    print("=" * 60)

    figs_dir = Path("paper_figures")
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Training curves
    # ------------------------------------------------------------------ #
    print("\n[1] Training curves ...")
    experiments = [
        ("Baseline (YOLOv8n)",        "runs/baseline"),
        ("+P2 Head",                   "runs/p2"),
        ("+SimAM",                     "runs/simam"),
        ("+PConv",                     "runs/pconv"),
        ("+NWD (pure)",                "runs/nwd_pure"),
        ("Full Two-Stage",             "runs/two_stage"),
    ]
    plot_training_curves(
        experiments,
        save_path=str(figs_dir / "training_curves.pdf"),
    )

    # ------------------------------------------------------------------ #
    # 2. Ablation bar chart
    # ------------------------------------------------------------------ #
    print("\n[2] Ablation bar chart ...")
    plot_ablation_bar(
        results_json_path="results/ablation_results.json",
        save_path=str(figs_dir / "ablation_bar.pdf"),
    )

    # ------------------------------------------------------------------ #
    # 3. Pareto / accuracy-efficiency scatter
    # ------------------------------------------------------------------ #
    print("\n[3] Pareto scatter ...")
    plot_pareto(
        results_json_path="results/comparison_results.json",
        save_path=str(figs_dir / "pareto.pdf"),
    )

    # ------------------------------------------------------------------ #
    # 4. Overfitting / training-stability analysis
    # ------------------------------------------------------------------ #
    print("\n[4] Overfitting analysis ...")
    plot_overfitting_analysis(
        nwd_exp_dir="runs/nwd_pure",
        two_stage_exp_dir="runs/two_stage",
        save_path=str(figs_dir / "overfitting_analysis.pdf"),
    )

    print("\n" + "=" * 60)
    print("  All figures saved to:", figs_dir.resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()
