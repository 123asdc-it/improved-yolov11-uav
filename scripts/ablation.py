"""
消融实验：逐步添加各改进模块，验证每个模块的贡献

修复：每组实验使用独立子进程，避免 CUDA 上下文污染导致的 DataLoader 卡死。

实验配置：
  0. Baseline (YOLOv11n)
  1. +P2 检测头
  2. +P2 +EMA
  3. +P2 +EMA +PConv
  4. +P2 +EMA +PConv +BiFPN (Full)

用法：cd 项目根目录，然后 python scripts/ablation.py
"""

import os
import sys
import json
import subprocess
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))


# ============================================================
# 公共训练参数
# ============================================================
TRAIN_ARGS = dict(
    data="configs/data.yaml",
    imgsz=1280,
    epochs=300,
    patience=50,
    batch=8,
    lr0=0.01,
    cos_lr=True,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    workers=4,
    cache=False,
)


# ============================================================
# YAML 配置生成
# ============================================================
def create_ablation_yamls():
    os.makedirs("configs/ablation", exist_ok=True)

    ablation_p2 = {
        "nc": 1,
        "scales": {"n": [0.50, 0.25, 1024]},
        "backbone": [
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
            [-1, 2, "C3k2", [256, False, 0.25]],
            [-1, 1, "Conv", [256, 3, 2]],
            [-1, 2, "C3k2", [512, False, 0.25]],
            [-1, 1, "Conv", [512, 3, 2]],
            [-1, 2, "C3k2", [512, True]],
            [-1, 1, "Conv", [1024, 3, 2]],
            [-1, 2, "C3k2", [1024, True]],
            [-1, 1, "SPPF", [1024, 5]],
            [-1, 2, "C2PSA", [1024]],
        ],
        "head": [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 6], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 4], 1, "Concat", [1]],
            [-1, 2, "C3k2", [256, False]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 2], 1, "Concat", [1]],
            [-1, 2, "C3k2", [128, False]],
            [-1, 1, "Conv", [128, 3, 2]],
            [[-1, 16], 1, "Concat", [1]],
            [-1, 2, "C3k2", [256, False]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 13], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],
            [-1, 2, "C3k2", [1024, True]],
            [[19, 22, 25, 28], 1, "Detect", ["nc"]],
        ],
    }

    ablation_p2_ema = {
        "nc": 1,
        "scales": {"n": [0.50, 0.25, 1024]},
        "backbone": [
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
            [-1, 2, "C3k2", [256, False, 0.25]],
            [-1, 1, "Conv", [256, 3, 2]],
            [-1, 2, "C3k2", [512, False, 0.25]],
            [-1, 1, "EMA", []],
            [-1, 1, "Conv", [512, 3, 2]],
            [-1, 2, "C3k2", [512, True]],
            [-1, 1, "EMA", []],
            [-1, 1, "Conv", [1024, 3, 2]],
            [-1, 2, "C3k2", [1024, True]],
            [-1, 1, "SPPF", [1024, 5]],
            [-1, 2, "C2PSA", [1024]],
        ],
        "head": [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 8], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 5], 1, "Concat", [1]],
            [-1, 2, "C3k2", [256, False]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 2], 1, "Concat", [1]],
            [-1, 2, "C3k2", [128, False]],
            [-1, 1, "Conv", [128, 3, 2]],
            [[-1, 18], 1, "Concat", [1]],
            [-1, 2, "C3k2", [256, False]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 15], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 12], 1, "Concat", [1]],
            [-1, 2, "C3k2", [1024, True]],
            [[21, 24, 27, 30], 1, "Detect", ["nc"]],
        ],
    }

    ablation_p2_ema_pconv = {
        "nc": 1,
        "scales": {"n": [0.50, 0.25, 1024]},
        "backbone": [
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
            [-1, 2, "C3k2", [256, False, 0.25]],
            [-1, 1, "Conv", [256, 3, 2]],
            [-1, 2, "C3k2", [512, False, 0.25]],
            [-1, 1, "EMA", []],
            [-1, 1, "Conv", [512, 3, 2]],
            [-1, 2, "C3k2", [512, True]],
            [-1, 1, "EMA", []],
            [-1, 1, "Conv", [1024, 3, 2]],
            [-1, 2, "C3k2", [1024, True]],
            [-1, 1, "SPPF", [1024, 5]],
            [-1, 2, "C2PSA", [1024]],
        ],
        "head": [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 8], 1, "Concat", [1]],
            [-1, 2, "PConv_C3k2", [512]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 5], 1, "Concat", [1]],
            [-1, 2, "PConv_C3k2", [256]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 2], 1, "Concat", [1]],
            [-1, 2, "PConv_C3k2", [128]],
            [-1, 1, "Conv", [128, 3, 2]],
            [[-1, 18], 1, "Concat", [1]],
            [-1, 2, "PConv_C3k2", [256]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 15], 1, "Concat", [1]],
            [-1, 2, "PConv_C3k2", [512]],
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 12], 1, "Concat", [1]],
            [-1, 2, "C3k2", [1024, True]],
            [[21, 24, 27, 30], 1, "Detect", ["nc"]],
        ],
    }

    for name, cfg in [("ablation_p2", ablation_p2),
                      ("ablation_p2_ema", ablation_p2_ema),
                      ("ablation_p2_ema_pconv", ablation_p2_ema_pconv)]:
        path = f"configs/ablation/{name}.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=None, allow_unicode=True, sort_keys=False)
        print(f"  Created: {path}")


# ============================================================
# 单实验运行脚本（每组在独立子进程中执行）
# ============================================================
SINGLE_EXP_SCRIPT = '''
import os, sys, json
PROJECT_ROOT = {project_root!r}
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
import register_modules
from ultralytics import YOLO

weight_path = {weight_path!r}
if os.path.exists(weight_path):
    print(f"[SKIP] {{weight_path}} already exists")
    model = YOLO(weight_path)
    metrics = model.val(data="configs/data.yaml", imgsz=1280, split="val")
else:
    model = YOLO({model_cfg!r})
    model.train(
        data="configs/data.yaml",
        imgsz=1280, epochs=300, patience=50, batch=8,
        lr0=0.01, cos_lr=True, mosaic=1.0, mixup=0.1,
        copy_paste=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        workers=4, cache=False,
        pretrained={pretrained!r},
        project="runs/ablation", name={exp_name!r}, exist_ok=True,
    )
    metrics = model.val(data="configs/data.yaml", imgsz=1280)

result = {{
    "name": {display_name!r},
    "map50": round(float(metrics.box.map50), 4),
    "map": round(float(metrics.box.map), 4),
}}
print("RESULT:", json.dumps(result))
'''


def run_experiment_subprocess(display_name, model_cfg, exp_name, pretrained="yolo11n.pt"):
    """Run a single experiment in a fresh subprocess (clean CUDA context)."""
    weight_path = f"runs/ablation/{exp_name}/weights/best.pt"

    script = SINGLE_EXP_SCRIPT.format(
        project_root=PROJECT_ROOT,
        weight_path=weight_path,
        model_cfg=model_cfg,
        pretrained=pretrained,
        exp_name=exp_name,
        display_name=display_name,
    )

    print(f"\n{'='*60}")
    print(f" Subprocess: {exp_name} ({display_name})")
    print(f"{'='*60}")

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=PROJECT_ROOT,
    )

    result_line = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("RESULT:"):
            result_line = line[len("RESULT:"):].strip()

    proc.wait()
    if proc.returncode != 0:
        print(f"ERROR: {exp_name} exited with code {proc.returncode}")
        return {"name": display_name, "map50": 0.0, "map": 0.0}

    if result_line:
        return json.loads(result_line)
    return {"name": display_name, "map50": 0.0, "map": 0.0}


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print(" Ablation Study: YOLOv11n Improvements")
    print("=" * 60)

    print("\nGenerating ablation configs...")
    create_ablation_yamls()

    experiments = [
        ("Baseline",      "yolo11n.pt",                          "baseline",       "yolo11n.pt"),
        ("+P2",           "configs/ablation/ablation_p2.yaml",   "p2_only",        "yolo11n.pt"),
        ("+P2+EMA",       "configs/ablation/ablation_p2_ema.yaml","p2_ema",         "yolo11n.pt"),
        ("+P2+EMA+PConv", "configs/ablation/ablation_p2_ema_pconv.yaml","p2_ema_pconv","yolo11n.pt"),
        ("Full (Ours)",   "configs/yolo11n-improved.yaml",       "full_improved",  "yolo11n.pt"),
    ]

    results = []
    for display_name, model_cfg, exp_name, pretrained in experiments:
        r = run_experiment_subprocess(display_name, model_cfg, exp_name, pretrained)
        results.append(r)
        # Save intermediate results after each experiment
        os.makedirs("runs/ablation", exist_ok=True)
        with open("runs/ablation/ablation_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print(f" Ablation Results Summary")
    print(f"{'='*60}")
    print(f"{'Experiment':<25} {'mAP50':>8} {'mAP50-95':>10}")
    print("-" * 45)
    for r in results:
        print(f"{r['name']:<25} {r['map50']:>8.4f} {r['map']:>10.4f}")
    print("=" * 60)
    print("\nResults saved to runs/ablation/ablation_results.json")


if __name__ == "__main__":
    main()
