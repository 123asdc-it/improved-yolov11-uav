"""
阶段2：YOLOv11n-Improved 改进模型训练
改进点：NWD损失/标签分配 + P2检测头 + SimAM注意力 + PConv轻量化 + BiFPN加权融合

用法：cd 项目根目录，然后 python scripts/train_improved.py
"""

import os
import sys
import json

# 确保工作目录为项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import register_modules  # ★ 必须在 YOLO 之前导入
from ultralytics_modules.nwd import patch_all_nwd
from ultralytics import YOLO


def main():
    # Apply NWD loss + NWD-TAL before model creation
    patch_all_nwd(loss_constant=12.0, tal_constant=12.0)

    model = YOLO("configs/yolo11n-improved.yaml")

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
        pretrained="yolo11n.pt",
        project="runs/detect",
        name="yolo11n_improved",
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
    print(f"Improved Model Results")
    print(f"{'='*55}")
    print(f"  Val  mAP50:    {results['val_map50']:.4f}")
    print(f"  Val  mAP50-95: {results['val_map']:.4f}")
    print(f"  Test mAP50:    {results['test_map50']:.4f}")
    print(f"  Test mAP50-95: {results['test_map']:.4f}")
    print(f"{'='*55}")

    os.makedirs("runs/detect", exist_ok=True)
    with open("runs/detect/improved_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to runs/detect/improved_results.json")


if __name__ == "__main__":
    main()
