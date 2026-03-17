"""
统一评估脚本：收集论文所需的完整指标体系

精度指标：mAP50, mAP75, mAP50-95, Precision, Recall, F1
效率指标：Params(M), FLOPs(G), FPS, Latency(ms), ModelSize(MB)

用法：
  cd 项目根目录
  python scripts/eval.py --weights path/to/best.pt --data configs/data.yaml
  python scripts/eval.py --weights model1.pt model2.pt --names "Baseline" "Ours"
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


def get_flops(model, imgsz=1280):
    """Estimate FLOPs using ultralytics built-in profiler."""
    try:
        from ultralytics.utils.torch_utils import get_flops as _get_flops
        return round(_get_flops(model.model, imgsz) / 1e9, 2)
    except Exception:
        try:
            # Fallback: profile with a dummy input
            from thop import profile
            dummy = torch.zeros(1, 3, imgsz, imgsz).to(
                next(model.model.parameters()).device
            )
            flops, _ = profile(model.model, inputs=(dummy,), verbose=False)
            return round(flops / 1e9, 2)
        except Exception:
            return 0.0


def evaluate_model(weight_path, data_yaml="configs/data.yaml", imgsz=1280, split="test"):
    """Full evaluation of a single model — collects all metrics for paper."""
    model = YOLO(weight_path)

    # === Accuracy metrics ===
    metrics = model.val(data=data_yaml, imgsz=imgsz, split=split)

    # Standard detection metrics
    map50 = round(float(metrics.box.map50), 4)
    map75 = round(float(metrics.box.map75), 4)
    map50_95 = round(float(metrics.box.map), 4)
    precision = round(float(metrics.box.mp), 4)
    recall = round(float(metrics.box.mr), 4)
    f1 = round(2 * precision * recall / (precision + recall + 1e-8), 4)

    # Per-IoU-threshold AP (for detailed analysis)
    # metrics.box.maps gives per-class AP at each IoU threshold

    # === Efficiency metrics ===
    total_params = count_parameters(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fps, latency_ms = measure_fps_end2end(model, imgsz=imgsz, device=device)
    model_size_mb = get_model_size_mb(weight_path)
    flops_g = get_flops(model, imgsz)

    return {
        "weight": weight_path,
        # Accuracy
        "map50": map50,
        "map75": map75,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # Efficiency
        "params_M": round(total_params / 1e6, 3),
        "flops_G": flops_g,
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

    # === Table 1: Accuracy ===
    print(f"\n{'='*100}")
    print(f" Evaluation Report  (split={args.split}, imgsz={args.imgsz})")
    print(f"{'='*100}")

    hdr = f"{'Model':<20} {'mAP50':>7} {'mAP75':>7} {'mAP50-95':>9} {'P':>7} {'R':>7} {'F1':>7}"
    print(f"\n  [Accuracy]")
    print(f"  {hdr}")
    print(f"  {'-'*65}")

    for name, weight in zip(names, args.weights):
        print(f"  Evaluating {name}...", flush=True)
        r = evaluate_model(weight, args.data, args.imgsz, args.split)
        r["name"] = name
        all_results.append(r)
        print(f"  {name:<20} {r['map50']:>7.4f} {r['map75']:>7.4f} {r['map50_95']:>9.4f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}")

    # === Table 2: Efficiency ===
    print(f"\n  [Efficiency]")
    print(f"  {'Model':<20} {'Params(M)':>10} {'FLOPs(G)':>9} {'FPS':>7} {'Lat(ms)':>8} {'Size(MB)':>9}")
    print(f"  {'-'*65}")
    for r in all_results:
        print(f"  {r['name']:<20} {r['params_M']:>10.3f} {r['flops_G']:>9.2f} "
              f"{r['fps']:>7.1f} {r['latency_ms']:>8.2f} {r['model_size_mb']:>9.2f}")

    print(f"\n{'='*100}")

    # Save results
    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.save_json}")

    # LaTeX template
    print("\n  LaTeX accuracy row template:")
    for r in all_results:
        print(f"  {r['name']} & {r['map50']*100:.1f} & {r['map75']*100:.1f} "
              f"& {r['map50_95']*100:.1f} & {r['precision']*100:.1f} "
              f"& {r['recall']*100:.1f} & {r['f1']*100:.1f} \\\\")
    print("\n  LaTeX efficiency row template:")
    for r in all_results:
        print(f"  {r['name']} & {r['params_M']:.2f} & {r['flops_G']:.1f} "
              f"& {r['fps']:.0f} & {r['latency_ms']:.1f} & {r['model_size_mb']:.1f} \\\\")


if __name__ == "__main__":
    main()
