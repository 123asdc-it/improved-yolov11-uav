"""
统一评估脚本：评估模型的 mAP50、参数量、FPS、模型大小
用于论文中的对比实验表格

用法：cd 项目根目录，然后 python scripts/eval.py --weights ... --data configs/data.yaml

修复记录：
- [Fix M3] 增加模型文件大小统计
- [Fix S3] FPS 测量改为 end-to-end（含 NMS）口径，使用 ultralytics 官方 benchmark
          warmup 从 50 降到 10，runs 从 200 降到 100，减少评估耗时
- [Fix S3] 增加推理延迟（latency ms）统计
"""

import os
import sys
import time
import json
import torch
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import register_modules
from ultralytics import YOLO


def count_parameters(model):
    """Count total parameters."""
    total = sum(p.numel() for p in model.model.parameters())
    return total


def measure_fps_end2end(model, imgsz=1280, device="cuda", warmup=10, runs=100):
    """
    Measure end-to-end FPS using ultralytics predict pipeline.
    This includes preprocessing + backbone + head + NMS, which is the
    correct definition for reporting in papers.

    Returns: (fps, latency_ms)
    """
    import numpy as np
    dummy_img = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.uint8).numpy()

    # Warmup
    for _ in range(warmup):
        model.predict(source=dummy_img, imgsz=imgsz, verbose=False, device=device)

    # Timed runs
    latencies = []
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(source=dummy_img, imgsz=imgsz, verbose=False, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    avg_latency = float(np.mean(latencies))
    fps = 1000.0 / avg_latency
    return fps, avg_latency


def get_model_size_mb(weight_path):
    """Return model file size in MB."""
    return os.path.getsize(weight_path) / (1024 ** 2)


def evaluate_model(weight_path, data_yaml="configs/data.yaml", imgsz=1280, split="test"):
    """Full evaluation of a single model."""
    model = YOLO(weight_path)

    # mAP on test split
    metrics = model.val(data=data_yaml, imgsz=imgsz, split=split)

    # Parameters
    total_params = count_parameters(model)

    # FPS (end-to-end)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fps, latency_ms = measure_fps_end2end(model, imgsz=imgsz, device=device)

    # Model file size
    model_size_mb = get_model_size_mb(weight_path)

    return {
        "weight": weight_path,
        "map50": round(float(metrics.box.map50), 4),
        "map": round(float(metrics.box.map), 4),
        "params_M": round(total_params / 1e6, 3),
        "fps": round(fps, 1),
        "latency_ms": round(latency_ms, 2),
        "model_size_mb": round(model_size_mb, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO models for paper table")
    parser.add_argument("--weights", nargs="+", required=True,
                        help="Model weight paths to evaluate")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Display names for each model")
    parser.add_argument("--save_json", default="runs/eval_results.json",
                        help="Path to save results JSON")
    args = parser.parse_args()

    names = args.names or [os.path.basename(os.path.dirname(os.path.dirname(w))) for w in args.weights]

    all_results = []

    print(f"\n{'='*85}")
    print(f" Model Evaluation Report  (split={args.split})")
    print(f"{'='*85}")
    print(f"{'Model':<22} {'mAP50':>7} {'mAP50-95':>9} {'Params(M)':>10} {'FPS':>7} {'Lat(ms)':>8} {'Size(MB)':>9}")
    print("-" * 85)

    for name, weight in zip(names, args.weights):
        print(f"Evaluating {name}...", flush=True)
        r = evaluate_model(weight, args.data, args.imgsz, args.split)
        r["name"] = name
        all_results.append(r)
        print(f"{name:<22} {r['map50']:>7.4f} {r['map']:>9.4f} {r['params_M']:>10.3f} "
              f"{r['fps']:>7.1f} {r['latency_ms']:>8.2f} {r['model_size_mb']:>9.2f}")

    print("=" * 85)

    # Save results
    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.save_json}")
    print("\nLaTeX table row template:")
    for r in all_results:
        print(f"  {r['name']} & {r['params_M']:.2f}M & {r['map50']*100:.1f} & {r['fps']:.0f} \\\\ ")


if __name__ == "__main__":
    main()
