"""
阶段1：YOLOv11n 基线训练
在恒源云 RTX3060 上运行

用法：cd 项目根目录，然后 python scripts/train_baseline.py
"""

import os
import sys
import json

# 确保工作目录为项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    model.train(
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
        project="runs/detect",
        name="yolo11n_baseline",
        exist_ok=True,
    )

    # Val set evaluation
    val_metrics = model.val(data="configs/data.yaml", imgsz=1280, split="val")

    # Test set evaluation (final number for paper)
    test_metrics = model.val(data="configs/data.yaml", imgsz=1280, split="test")

    results = {
        "val_map50": round(float(val_metrics.box.map50), 4),
        "val_map": round(float(val_metrics.box.map), 4),
        "test_map50": round(float(test_metrics.box.map50), 4),
        "test_map": round(float(test_metrics.box.map), 4),
    }

    print(f"\n{'='*55}")
    print(f"Baseline Results")
    print(f"{'='*55}")
    print(f"  Val  mAP50:    {results['val_map50']:.4f}")
    print(f"  Val  mAP50-95: {results['val_map']:.4f}")
    print(f"  Test mAP50:    {results['test_map50']:.4f}")
    print(f"  Test mAP50-95: {results['test_map']:.4f}")
    print(f"{'='*55}")

    os.makedirs("runs/detect", exist_ok=True)
    with open("runs/detect/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to runs/detect/baseline_results.json")


if __name__ == "__main__":
    main()
