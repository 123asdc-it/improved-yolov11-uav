"""
SOTA 模型训练：YOLOv11n-SOTA
改进：RepVGG backbone + SimAM attention + CARAFE P2 + PConv + BiFPN + NWD loss/TAL

用法：cd 项目根目录，然后 python scripts/train_sota.py
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import register_modules
from ultralytics_modules.nwd import patch_all_nwd
from ultralytics import YOLO


def main():
    # Apply NWD loss + NWD-TAL before model creation
    patch_all_nwd(loss_constant=12.0, tal_constant=12.0)

    model = YOLO("configs/yolo11n-sota.yaml")

    model.train(
        data="configs/data.yaml",
        imgsz=1280,
        epochs=300,
        patience=50,
        batch=8,
        lr0=0.01,
        cos_lr=True,
        # Enhanced augmentation for small dataset
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,   # increased from 0.1
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        fliplr=0.5,
        # Pretrained weights (partial load from yolo11n)
        pretrained="yolo11n.pt",
        project="runs/detect",
        name="yolo11n_sota",
        exist_ok=True,
    )

    # Val evaluation
    val_metrics = model.val(data="configs/data.yaml", imgsz=1280, split="val")

    # Val + TTA
    val_tta_metrics = model.val(data="configs/data.yaml", imgsz=1280,
                                split="val", augment=True)

    # Test evaluation
    test_metrics = model.val(data="configs/data.yaml", imgsz=1280, split="test")

    # Test + TTA (final paper number)
    test_tta_metrics = model.val(data="configs/data.yaml", imgsz=1280,
                                 split="test", augment=True)

    results = {
        "val_map50":      round(float(val_metrics.box.map50), 4),
        "val_map":        round(float(val_metrics.box.map), 4),
        "val_tta_map50":  round(float(val_tta_metrics.box.map50), 4),
        "test_map50":     round(float(test_metrics.box.map50), 4),
        "test_map":       round(float(test_metrics.box.map), 4),
        "test_tta_map50": round(float(test_tta_metrics.box.map50), 4),
    }

    print(f"\n{'='*60}")
    print(f" SOTA Model Results")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k:<22}: {v:.4f}")
    print(f"{'='*60}")

    os.makedirs("runs/detect", exist_ok=True)
    with open("runs/detect/sota_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to runs/detect/sota_results.json")


if __name__ == "__main__":
    main()
