"""
SA-NWD 对比实验：验证核心创新的效果

在消融实验 Full 组（纯 NWD）的基础上，对比：
  Exp 6: Full + hybrid SA-NWD+CIoU (alpha=0.5)    — 混合损失效果
  Exp 7: Full + hybrid SA-NWD+CIoU + NWD-NMS      — 完整框架

每组使用独立子进程运行，避免 CUDA 上下文污染。

用法：cd 项目根目录，然后 python scripts/run_sa_nwd_comparison.py
"""

import os
import sys
import json
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

SA_NWD_PROJECT = "/root/drone_detection/runs/sa_nwd"

TRAIN_ARGS = dict(
    data="configs/data.yaml",
    imgsz=1280,
    epochs=300,
    patience=100,
    batch=8,
    lr0=0.01,
    cos_lr=True,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    warmup_epochs=5,
    workers=4,
    cache=False,
)

SINGLE_EXP_SCRIPT = '''
import os, sys, json

PROJECT_ROOT = {project_root!r}
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import register_modules
from ultralytics_modules.nwd import patch_all_nwd
patch_all_nwd(c_base={c_base}, k={k}, alpha={alpha}, use_sa={use_sa}, use_nwd_nms={use_nwd_nms})
print("[SA-NWD] Patched: c_base={c_base}, k={k}, alpha={alpha}, nms={use_nwd_nms}")

from ultralytics import YOLO

weight_path = {weight_path!r}
train_args = {train_args!r}

if os.path.exists(weight_path):
    print(f"[SKIP] {{weight_path}} exists, running val only")
    model = YOLO(weight_path)
    metrics = model.val(data="configs/data.yaml", imgsz=1280, split="val")
else:
    model = YOLO({model_cfg!r})
    model.train(
        pretrained="yolo11n.pt",
        project={project!r},
        name={exp_name!r},
        exist_ok=True,
        **train_args,
    )
    metrics = model.val(data="configs/data.yaml", imgsz=1280, split="val")

result = {{
    "name": {display_name!r},
    "map50": round(float(metrics.box.map50), 4),
    "map75": round(float(metrics.box.map75), 4),
    "map": round(float(metrics.box.map), 4),
    "precision": round(float(metrics.box.mp), 4),
    "recall": round(float(metrics.box.mr), 4),
}}
result["f1"] = round(2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"] + 1e-8), 4)
print("RESULT:", json.dumps(result))
'''


def run_experiment(display_name, model_cfg, exp_name,
                   c_base=12.0, k=2.0, alpha=0.5,
                   use_sa=True, use_nwd_nms=False):
    """Run a single SA-NWD comparison experiment."""
    weight_path = os.path.join(SA_NWD_PROJECT, exp_name, "weights", "best.pt")

    script = SINGLE_EXP_SCRIPT.format(
        project_root=PROJECT_ROOT,
        weight_path=weight_path,
        model_cfg=model_cfg,
        exp_name=exp_name,
        display_name=display_name,
        train_args=TRAIN_ARGS,
        project=SA_NWD_PROJECT,
        c_base=c_base,
        k=k,
        alpha=alpha,
        use_sa=use_sa,
        use_nwd_nms=use_nwd_nms,
    )

    print(f"\n{'='*70}")
    print(f"  {display_name}")
    print(f"  SA-NWD: c_base={c_base}, k={k}, alpha={alpha}, nms={use_nwd_nms}")
    print(f"{'='*70}")

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=PROJECT_ROOT,
    )

    result_line = None
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("RESULT:"):
            result_line = line[len("RESULT:"):].strip()

    proc.wait()
    if proc.returncode != 0:
        print(f"[ERROR] {exp_name} exited with code {proc.returncode}")
        return {"name": display_name, "map50": 0.0, "map75": 0.0,
                "map": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    if result_line:
        return json.loads(result_line)
    return {"name": display_name, "map50": 0.0, "map75": 0.0,
            "map": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def main():
    print("=" * 70)
    print("  SA-NWD Comparison Experiments")
    print("=" * 70)

    experiments = [
        # (display_name, model_cfg, exp_name, c_base, k, alpha, use_sa, use_nwd_nms)
        ("Full (hybrid α=0.5)",
         "configs/yolo11n-improved.yaml", "hybrid_full",
         12.0, 2.0, 0.5, True, False),

        ("Full (hybrid α=0.5 + NMS)",
         "configs/yolo11n-improved.yaml", "hybrid_full_nms",
         12.0, 2.0, 0.5, True, True),
    ]

    results = []
    for display_name, model_cfg, exp_name, c_base, k, alpha, use_sa, use_nwd_nms in experiments:
        r = run_experiment(display_name, model_cfg, exp_name,
                           c_base, k, alpha, use_sa, use_nwd_nms)
        results.append(r)

        os.makedirs("runs/sa_nwd", exist_ok=True)
        with open("runs/sa_nwd/sa_nwd_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SA-NWD Comparison Results")
    print(f"{'='*70}")
    print(f"  {'Experiment':<30} {'mAP50':>7} {'mAP75':>7} {'mAP50-95':>9} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['name']:<30} {r['map50']:>7.4f} {r['map75']:>7.4f} "
              f"{r['map']:>9.4f} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}")
    print(f"  {'='*70}")
    print(f"\n  Results: runs/sa_nwd/sa_nwd_results.json")


if __name__ == "__main__":
    main()
