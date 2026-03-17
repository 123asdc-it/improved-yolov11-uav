"""
训练曲线可视化：从 results.csv 提取各实验的训练轨迹

生成论文图：
  1. mAP50 vs epoch 对比图（多实验叠加）
  2. 过拟合分析图（纯 NWD vs hybrid SA-NWD+CIoU）

用法：python scripts/plot_training_curves.py --results_dirs dir1 dir2 ... --names name1 name2 ...
"""

import os
import sys
import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("需要安装 matplotlib: pip install matplotlib")
    sys.exit(1)


# ============================================================
# 论文级 matplotlib 配置
# ============================================================
def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
    })


# Paul Tol Bright (colorblind-safe)
COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#000000"]


def load_results_csv(results_dir):
    """加载 ultralytics 的 results.csv 文件。

    Returns:
        dict with keys: epoch, map50, map50_95, precision, recall, box_loss, cls_loss, dfl_loss
    """
    csv_path = Path(results_dir) / "results.csv"
    if not csv_path.exists():
        print(f"  警告: {csv_path} 不存在")
        return None

    data = {
        "epoch": [], "map50": [], "map50_95": [],
        "precision": [], "recall": [],
        "box_loss": [], "cls_loss": [], "dfl_loss": [],
    }

    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        # 清理 header 空格
        header = [h.strip() for h in header]

        # 找列索引
        col_map = {}
        for key, patterns in {
            "epoch": ["epoch"],
            "map50": ["metrics/mAP50(B)"],
            "map50_95": ["metrics/mAP50-95(B)"],
            "precision": ["metrics/precision(B)"],
            "recall": ["metrics/recall(B)"],
            "box_loss": ["train/box_loss"],
            "cls_loss": ["train/cls_loss"],
            "dfl_loss": ["train/dfl_loss"],
        }.items():
            for pat in patterns:
                if pat in header:
                    col_map[key] = header.index(pat)
                    break

        for row in reader:
            row = [r.strip() for r in row]
            for key, col_idx in col_map.items():
                try:
                    data[key].append(float(row[col_idx]))
                except (IndexError, ValueError):
                    data[key].append(0.0)

    return {k: np.array(v) for k, v in data.items()}


def plot_map50_comparison(all_data, names, save_path):
    """绘制 mAP50 训练曲线对比图。"""
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    for i, (data, name) in enumerate(zip(all_data, names)):
        if data is None:
            continue
        color = COLORS[i % len(COLORS)]
        epochs = data["epoch"]
        ax1.plot(epochs, data["map50"], color=color, label=name, alpha=0.85)
        ax2.plot(epochs, data["map50_95"], color=color, label=name, alpha=0.85)

    ax1.set_ylabel("mAP@0.5")
    ax1.legend(loc="lower right")
    ax1.set_ylim(bottom=0.0)

    ax2.set_ylabel("mAP@0.5:0.95")
    ax2.set_xlabel("Epoch")
    ax2.legend(loc="lower right")
    ax2.set_ylim(bottom=0.0)

    plt.tight_layout()
    fig.savefig(save_path, format="pdf")
    print(f"  保存: {save_path}")
    plt.close()


def plot_loss_comparison(all_data, names, save_path):
    """绘制 loss 训练曲线对比图。"""
    set_paper_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    for i, (data, name) in enumerate(zip(all_data, names)):
        if data is None:
            continue
        color = COLORS[i % len(COLORS)]
        epochs = data["epoch"]
        ax1.plot(epochs, data["box_loss"], color=color, label=name, alpha=0.85)
        ax2.plot(epochs, data["cls_loss"], color=color, label=name, alpha=0.85)
        ax3.plot(epochs, data["dfl_loss"], color=color, label=name, alpha=0.85)

    ax1.set_ylabel("Box Loss")
    ax1.legend(loc="upper right", fontsize=7)
    ax2.set_ylabel("Cls Loss")
    ax3.set_ylabel("DFL Loss")
    ax3.set_xlabel("Epoch")

    plt.tight_layout()
    fig.savefig(save_path, format="pdf")
    print(f"  保存: {save_path}")
    plt.close()


def plot_overfitting_analysis(data_nwd, data_hybrid, save_path):
    """过拟合分析图：纯 NWD vs hybrid SA-NWD+CIoU。"""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))

    if data_nwd is not None:
        epochs = data_nwd["epoch"]
        map50 = data_nwd["map50"]
        ax.plot(epochs, map50, color=COLORS[1], label="Pure NWD Loss", linewidth=2)
        # 标注过拟合区域
        peak_idx = np.argmax(map50)
        if peak_idx < len(map50) - 10:
            ax.axvline(x=epochs[peak_idx], color=COLORS[1], linestyle="--",
                       alpha=0.5, linewidth=1)
            ax.annotate(f"Peak: {map50[peak_idx]:.3f}\n(epoch {int(epochs[peak_idx])})",
                        xy=(epochs[peak_idx], map50[peak_idx]),
                        xytext=(epochs[peak_idx] + 20, map50[peak_idx] - 0.03),
                        fontsize=8, arrowprops=dict(arrowstyle="->", color=COLORS[1]))

    if data_hybrid is not None:
        epochs = data_hybrid["epoch"]
        map50 = data_hybrid["map50"]
        ax.plot(epochs, map50, color=COLORS[0], label="Hybrid SA-NWD+CIoU", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5")
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=0.5)

    plt.tight_layout()
    fig.savefig(save_path, format="pdf")
    print(f"  保存: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dirs", nargs="+", required=True,
                        help="各实验的结果目录（含 results.csv）")
    parser.add_argument("--names", nargs="+", required=True,
                        help="各实验的显示名称")
    parser.add_argument("--output_dir", default="paper/figs",
                        help="输出目录")
    parser.add_argument("--overfitting_nwd", default=None,
                        help="纯 NWD 实验目录（用于过拟合分析图）")
    parser.add_argument("--overfitting_hybrid", default=None,
                        help="Hybrid 实验目录（用于过拟合分析图）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("加载训练数据...")
    all_data = []
    for d, n in zip(args.results_dirs, args.names):
        print(f"  {n}: {d}")
        data = load_results_csv(d)
        all_data.append(data)

    print("\n绘制图表...")
    plot_map50_comparison(all_data, args.names,
                          os.path.join(args.output_dir, "training_curves_map.pdf"))
    plot_loss_comparison(all_data, args.names,
                         os.path.join(args.output_dir, "training_curves_loss.pdf"))

    # 过拟合分析图
    if args.overfitting_nwd:
        data_nwd = load_results_csv(args.overfitting_nwd)
        data_hybrid = load_results_csv(args.overfitting_hybrid) if args.overfitting_hybrid else None
        plot_overfitting_analysis(data_nwd, data_hybrid,
                                  os.path.join(args.output_dir, "overfitting_analysis.pdf"))

    print("\n完成！")


if __name__ == "__main__":
    main()
