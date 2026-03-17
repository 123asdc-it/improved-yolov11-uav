"""
Ablation study: incrementally add each improvement module to verify individual contribution.

Each experiment runs in an independent subprocess to avoid CUDA context pollution.

Experimental groups (6 total):
  0. Baseline         - stock yolo11n.pt, no modifications, no NWD
  1. +NWD             - stock yolo11n.pt with NWD loss + NWD-TAL patches
  2. +NWD+P2          - P2 detection head (4-head Detect) + NWD
  3. +NWD+P2+SimAM    - P2 + SimAM attention at P3/P4 + NWD
  4. +NWD+P2+SimAM+PConv - P2 + SimAM + PConv_C3k2 in neck + NWD
  5. Full (Ours)      - P2 + SimAM + PConv + BiFPN_Concat + NWD (yolo11n-improved.yaml)

Usage: cd project root, then  python scripts/ablation.py
"""

import os
import sys
import json
import subprocess
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

# Absolute path for training output — avoids nested project dirs
ABLATION_PROJECT = "/root/drone_detection/runs/ablation"

# ============================================================
# Common training parameters
# ============================================================
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


# ============================================================
# YAML config generation for ablation experiments 2-4
# ============================================================
def create_ablation_yamls():
    """Generate intermediate architecture YAMLs for ablation experiments 2-4.

    Experiment 5 (Full) uses configs/yolo11n-improved.yaml directly.
    Experiments 0-1 use stock yolo11n.pt (no custom YAML needed).

    Layer index verification for each config is documented inline.
    """
    os.makedirs("configs/ablation", exist_ok=True)

    # ----------------------------------------------------------
    # Experiment 2: +NWD+P2
    # Backbone: stock YOLOv11n (11 layers, indices 0-10)
    #   0:  Conv [64,3,2]           P1/2
    #   1:  Conv [128,3,2]          P2/4
    #   2:  C3k2 [256,F,0.25]      ← P2 feature tap
    #   3:  Conv [256,3,2]          P3/8
    #   4:  C3k2 [512,F,0.25]      ← P3 feature tap
    #   5:  Conv [512,3,2]          P4/16
    #   6:  C3k2 [512,T]           ← P4 feature tap
    #   7:  Conv [1024,3,2]         P5/32
    #   8:  C3k2 [1024,T]
    #   9:  SPPF [1024,5]
    #   10: C2PSA [1024]            ← P5 feature tap
    #
    # Head top-down (P5→P4→P3→P2):
    #   11: Upsample               (P4 spatial size)
    #   12: Concat[-1, 6]          (P5_up + backbone_P4)
    #   13: C3k2 [512,F]           head P4 features
    #   14: Upsample               (P3 spatial size)
    #   15: Concat[-1, 4]          (P4_up + backbone_P3)
    #   16: C3k2 [256,F]           head P3 features
    #   17: Upsample               (P2 spatial size)
    #   18: Concat[-1, 2]          (P3_up + backbone_P2)
    #   19: C3k2 [128,F]           → P2 output for Detect
    #
    # Head bottom-up (P2→P3→P4→P5):
    #   20: Conv [128,3,2]         downsample P2
    #   21: Concat[-1, 16]         (P2_down + head_P3=layer16)
    #   22: C3k2 [256,F]           → P3 output for Detect
    #   23: Conv [256,3,2]         downsample P3
    #   24: Concat[-1, 13]         (P3_down + head_P4=layer13)
    #   25: C3k2 [512,F]           → P4 output for Detect
    #   26: Conv [512,3,2]         downsample P4
    #   27: Concat[-1, 10]         (P4_down + backbone_P5=layer10)
    #   28: C3k2 [1024,T]          → P5 output for Detect
    #   29: Detect[19,22,25,28]    P2, P3, P4, P5
    # ----------------------------------------------------------
    ablation_nwd_p2 = {
        "nc": 1,
        "scales": {"n": [0.50, 0.25, 1024]},
        "backbone": [
            [-1, 1, "Conv", [64, 3, 2]],           # 0  P1/2
            [-1, 1, "Conv", [128, 3, 2]],           # 1  P2/4
            [-1, 2, "C3k2", [256, False, 0.25]],    # 2  P2 features
            [-1, 1, "Conv", [256, 3, 2]],           # 3  P3/8
            [-1, 2, "C3k2", [512, False, 0.25]],    # 4  P3 features
            [-1, 1, "Conv", [512, 3, 2]],           # 5  P4/16
            [-1, 2, "C3k2", [512, True]],           # 6  P4 features
            [-1, 1, "Conv", [1024, 3, 2]],          # 7  P5/32
            [-1, 2, "C3k2", [1024, True]],          # 8
            [-1, 1, "SPPF", [1024, 5]],             # 9
            [-1, 2, "C2PSA", [1024]],               # 10 P5 features
        ],
        "head": [
            # --- Top-down ---
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 11
            [[-1, 6], 1, "Concat", [1]],                    # 12
            [-1, 2, "C3k2", [512, False]],                  # 13 head P4
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 14
            [[-1, 4], 1, "Concat", [1]],                    # 15
            [-1, 2, "C3k2", [256, False]],                  # 16 head P3
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 17
            [[-1, 2], 1, "Concat", [1]],                    # 18
            [-1, 2, "C3k2", [128, False]],                  # 19 → P2 detect
            # --- Bottom-up ---
            [-1, 1, "Conv", [128, 3, 2]],                   # 20
            [[-1, 16], 1, "Concat", [1]],                   # 21
            [-1, 2, "C3k2", [256, False]],                  # 22 → P3 detect
            [-1, 1, "Conv", [256, 3, 2]],                   # 23
            [[-1, 13], 1, "Concat", [1]],                   # 24
            [-1, 2, "C3k2", [512, False]],                  # 25 → P4 detect
            [-1, 1, "Conv", [512, 3, 2]],                   # 26
            [[-1, 10], 1, "Concat", [1]],                   # 27
            [-1, 2, "C3k2", [1024, True]],                  # 28 → P5 detect
            [[19, 22, 25, 28], 1, "Detect", ["nc"]],        # 29
        ],
    }

    # ----------------------------------------------------------
    # Experiment 3: +NWD+P2+SimAM
    # Backbone: insert SimAM after C3k2 at P3 (layer 5) and P4 (layer 8)
    #   0:  Conv [64,3,2]           P1/2
    #   1:  Conv [128,3,2]          P2/4
    #   2:  C3k2 [256,F,0.25]      ← P2 feature tap
    #   3:  Conv [256,3,2]          P3/8
    #   4:  C3k2 [512,F,0.25]
    #   5:  SimAM []                ← P3 feature tap (post-attention)
    #   6:  Conv [512,3,2]          P4/16
    #   7:  C3k2 [512,T]
    #   8:  SimAM []                ← P4 feature tap (post-attention)
    #   9:  Conv [1024,3,2]         P5/32
    #   10: C3k2 [1024,T]
    #   11: SPPF [1024,5]
    #   12: C2PSA [1024]            ← P5 feature tap
    #
    # Head top-down (concat refs → backbone P4=8, P3=5, P2=2):
    #   13: Upsample
    #   14: Concat[-1, 8]
    #   15: C3k2 [512,F]           head P4
    #   16: Upsample
    #   17: Concat[-1, 5]
    #   18: C3k2 [256,F]           head P3
    #   19: Upsample
    #   20: Concat[-1, 2]
    #   21: C3k2 [128,F]           → P2 detect
    #
    # Head bottom-up (concat refs → head P3=18, head P4=15, backbone P5=12):
    #   22: Conv [128,3,2]
    #   23: Concat[-1, 18]
    #   24: C3k2 [256,F]           → P3 detect
    #   25: Conv [256,3,2]
    #   26: Concat[-1, 15]
    #   27: C3k2 [512,F]           → P4 detect
    #   28: Conv [512,3,2]
    #   29: Concat[-1, 12]
    #   30: C3k2 [1024,T]          → P5 detect
    #   31: Detect[21,24,27,30]
    # ----------------------------------------------------------
    ablation_nwd_p2_simam = {
        "nc": 1,
        "scales": {"n": [0.50, 0.25, 1024]},
        "backbone": [
            [-1, 1, "Conv", [64, 3, 2]],           # 0  P1/2
            [-1, 1, "Conv", [128, 3, 2]],           # 1  P2/4
            [-1, 2, "C3k2", [256, False, 0.25]],    # 2  P2 features
            [-1, 1, "Conv", [256, 3, 2]],           # 3  P3/8
            [-1, 2, "C3k2", [512, False, 0.25]],    # 4
            [-1, 1, "SimAM", []],                   # 5  P3 features (SimAM)
            [-1, 1, "Conv", [512, 3, 2]],           # 6  P4/16
            [-1, 2, "C3k2", [512, True]],           # 7
            [-1, 1, "SimAM", []],                   # 8  P4 features (SimAM)
            [-1, 1, "Conv", [1024, 3, 2]],          # 9  P5/32
            [-1, 2, "C3k2", [1024, True]],          # 10
            [-1, 1, "SPPF", [1024, 5]],             # 11
            [-1, 2, "C2PSA", [1024]],               # 12 P5 features
        ],
        "head": [
            # --- Top-down ---
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 13
            [[-1, 8], 1, "Concat", [1]],                    # 14
            [-1, 2, "C3k2", [512, False]],                  # 15 head P4
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 16
            [[-1, 5], 1, "Concat", [1]],                    # 17
            [-1, 2, "C3k2", [256, False]],                  # 18 head P3
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 19
            [[-1, 2], 1, "Concat", [1]],                    # 20
            [-1, 2, "C3k2", [128, False]],                  # 21 → P2 detect
            # --- Bottom-up ---
            [-1, 1, "Conv", [128, 3, 2]],                   # 22
            [[-1, 18], 1, "Concat", [1]],                   # 23
            [-1, 2, "C3k2", [256, False]],                  # 24 → P3 detect
            [-1, 1, "Conv", [256, 3, 2]],                   # 25
            [[-1, 15], 1, "Concat", [1]],                   # 26
            [-1, 2, "C3k2", [512, False]],                  # 27 → P4 detect
            [-1, 1, "Conv", [512, 3, 2]],                   # 28
            [[-1, 12], 1, "Concat", [1]],                   # 29
            [-1, 2, "C3k2", [1024, True]],                  # 30 → P5 detect
            [[21, 24, 27, 30], 1, "Detect", ["nc"]],        # 31
        ],
    }

    # ----------------------------------------------------------
    # Experiment 4: +NWD+P2+SimAM+PConv
    # Backbone: identical to experiment 3 (indices 0-12)
    # Head: replace C3k2 with PConv_C3k2 in neck (keep C3k2 at P5)
    #   Layer indices identical to experiment 3 — only module types change.
    #   13-21: top-down with PConv_C3k2
    #   22-30: bottom-up with PConv_C3k2 (except layer 30 = C3k2)
    #   31: Detect[21,24,27,30]
    # ----------------------------------------------------------
    ablation_nwd_p2_simam_pconv = {
        "nc": 1,
        "scales": {"n": [0.50, 0.25, 1024]},
        "backbone": [
            [-1, 1, "Conv", [64, 3, 2]],           # 0  P1/2
            [-1, 1, "Conv", [128, 3, 2]],           # 1  P2/4
            [-1, 2, "C3k2", [256, False, 0.25]],    # 2  P2 features
            [-1, 1, "Conv", [256, 3, 2]],           # 3  P3/8
            [-1, 2, "C3k2", [512, False, 0.25]],    # 4
            [-1, 1, "SimAM", []],                   # 5  P3 features (SimAM)
            [-1, 1, "Conv", [512, 3, 2]],           # 6  P4/16
            [-1, 2, "C3k2", [512, True]],           # 7
            [-1, 1, "SimAM", []],                   # 8  P4 features (SimAM)
            [-1, 1, "Conv", [1024, 3, 2]],          # 9  P5/32
            [-1, 2, "C3k2", [1024, True]],          # 10
            [-1, 1, "SPPF", [1024, 5]],             # 11
            [-1, 2, "C2PSA", [1024]],               # 12 P5 features
        ],
        "head": [
            # --- Top-down ---
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 13
            [[-1, 8], 1, "Concat", [1]],                    # 14
            [-1, 2, "PConv_C3k2", [512]],                   # 15 head P4
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 16
            [[-1, 5], 1, "Concat", [1]],                    # 17
            [-1, 2, "PConv_C3k2", [256]],                   # 18 head P3
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 19
            [[-1, 2], 1, "Concat", [1]],                    # 20
            [-1, 2, "PConv_C3k2", [128]],                   # 21 → P2 detect
            # --- Bottom-up ---
            [-1, 1, "Conv", [128, 3, 2]],                   # 22
            [[-1, 18], 1, "Concat", [1]],                   # 23
            [-1, 2, "PConv_C3k2", [256]],                   # 24 → P3 detect
            [-1, 1, "Conv", [256, 3, 2]],                   # 25
            [[-1, 15], 1, "Concat", [1]],                   # 26
            [-1, 2, "PConv_C3k2", [512]],                   # 27 → P4 detect
            [-1, 1, "Conv", [512, 3, 2]],                   # 28
            [[-1, 12], 1, "Concat", [1]],                   # 29
            [-1, 2, "C3k2", [1024, True]],                  # 30 → P5 detect
            [[21, 24, 27, 30], 1, "Detect", ["nc"]],        # 31
        ],
    }

    for name, cfg in [
        ("ablation_nwd_p2", ablation_nwd_p2),
        ("ablation_nwd_p2_simam", ablation_nwd_p2_simam),
        ("ablation_nwd_p2_simam_pconv", ablation_nwd_p2_simam_pconv),
    ]:
        path = f"configs/ablation/{name}.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=None, allow_unicode=True, sort_keys=False)
        print(f"  Created: {path}")


# ============================================================
# Subprocess script template
# ============================================================
# Placeholders filled per-experiment:
#   {project_root}  - absolute path to repo root
#   {weight_path}   - expected best.pt location
#   {model_cfg}     - YAML path or "yolo11n.pt"
#   {pretrained}    - pretrained weight path
#   {exp_name}      - run name under project dir
#   {display_name}  - human-readable experiment name
#   {use_nwd}       - "True" or "False"
#   {train_args}    - repr of TRAIN_ARGS dict
#   {ablation_project} - absolute path for project= arg
SINGLE_EXP_SCRIPT = '''
import os, sys, json

PROJECT_ROOT = {project_root!r}
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

# Step 1: register custom modules (SimAM, PConv_C3k2, BiFPN_Concat, etc.)
import register_modules

# Step 2: conditionally apply NWD loss + NWD-TAL patches
if {use_nwd!r}:
    from ultralytics_modules.nwd import patch_all_nwd
    patch_all_nwd(loss_constant=12.0, tal_constant=12.0)
    print("[NWD] Patched loss and TAL with NWD (constant=12.0)")
else:
    print("[NWD] Skipped — baseline without NWD")

from ultralytics import YOLO

weight_path = {weight_path!r}
train_args = {train_args!r}

if os.path.exists(weight_path):
    print(f"[SKIP] {{weight_path}} already exists, running val only")
    model = YOLO(weight_path)
    metrics = model.val(data="configs/data.yaml", imgsz=1280, split="val")
else:
    model = YOLO({model_cfg!r})
    model.train(
        pretrained={pretrained!r},
        project={ablation_project!r},
        name={exp_name!r},
        exist_ok=True,
        **train_args,
    )
    metrics = model.val(data="configs/data.yaml", imgsz=1280, split="val")

result = {{
    "name": {display_name!r},
    "map50": round(float(metrics.box.map50), 4),
    "map": round(float(metrics.box.map), 4),
}}
print("RESULT:", json.dumps(result))
'''


def run_experiment_subprocess(display_name, model_cfg, exp_name,
                              pretrained="yolo11n.pt", use_nwd=True):
    """Run a single ablation experiment in a fresh subprocess (clean CUDA context)."""
    weight_path = os.path.join(ABLATION_PROJECT, exp_name, "weights", "best.pt")

    script = SINGLE_EXP_SCRIPT.format(
        project_root=PROJECT_ROOT,
        weight_path=weight_path,
        model_cfg=model_cfg,
        pretrained=pretrained,
        exp_name=exp_name,
        display_name=display_name,
        use_nwd=use_nwd,
        train_args=TRAIN_ARGS,
        ablation_project=ABLATION_PROJECT,
    )

    print(f"\n{'='*70}")
    print(f"  Experiment: {exp_name}")
    print(f"  Config:     {display_name}")
    print(f"  NWD:        {'ON' if use_nwd else 'OFF'}")
    print(f"  Model:      {model_cfg}")
    print(f"{'='*70}")

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=PROJECT_ROOT,
    )

    result_line = None
    assert proc.stdout is not None  # guaranteed by stdout=PIPE
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("RESULT:"):
            result_line = line[len("RESULT:"):].strip()

    proc.wait()
    if proc.returncode != 0:
        print(f"[ERROR] {exp_name} exited with code {proc.returncode}")
        return {"name": display_name, "map50": 0.0, "map": 0.0}

    if result_line:
        return json.loads(result_line)
    return {"name": display_name, "map50": 0.0, "map": 0.0}


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  Ablation Study: YOLOv11n Improvements for Small-Object Detection")
    print("=" * 70)

    # Generate intermediate YAML configs for experiments 2-4
    print("\nGenerating ablation YAML configs...")
    create_ablation_yamls()

    # ----------------------------------------------------------
    # Experiment definitions
    # (display_name, model_cfg, exp_name, pretrained, use_nwd)
    # ----------------------------------------------------------
    experiments = [
        # 0. Baseline: stock model, NO NWD
        ("Baseline",
         "yolo11n.pt",
         "baseline",
         "yolo11n.pt",
         False),

        # 1. +NWD: stock model WITH NWD loss patch (no structural change)
        ("+NWD",
         "yolo11n.pt",
         "nwd_only",
         "yolo11n.pt",
         True),

        # 2. +NWD+P2: P2 detection head, standard Concat/C3k2, NWD
        ("+NWD+P2",
         "configs/ablation/ablation_nwd_p2.yaml",
         "nwd_p2",
         "yolo11n.pt",
         True),

        # 3. +NWD+P2+SimAM: add SimAM at P3/P4, NWD
        ("+NWD+P2+SimAM",
         "configs/ablation/ablation_nwd_p2_simam.yaml",
         "nwd_p2_simam",
         "yolo11n.pt",
         True),

        # 4. +NWD+P2+SimAM+PConv: add PConv_C3k2 in neck, NWD
        ("+NWD+P2+SimAM+PConv",
         "configs/ablation/ablation_nwd_p2_simam_pconv.yaml",
         "nwd_p2_simam_pconv",
         "yolo11n.pt",
         True),

        # 5. Full (Ours): all improvements + BiFPN, NWD
        ("Full (Ours)",
         "configs/yolo11n-improved.yaml",
         "full_improved",
         "yolo11n.pt",
         True),
    ]

    results = []
    for display_name, model_cfg, exp_name, pretrained, use_nwd in experiments:
        r = run_experiment_subprocess(
            display_name, model_cfg, exp_name, pretrained, use_nwd
        )
        results.append(r)

        # Save intermediate results after each experiment
        os.makedirs("runs/ablation", exist_ok=True)
        with open("runs/ablation/ablation_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------------
    # Final summary table
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Ablation Results Summary")
    print(f"{'='*70}")
    print(f"  {'#':<4}{'Experiment':<30}{'mAP50':>8}{'mAP50-95':>10}{'delta50':>9}{'delta':>9}")
    print(f"  {'-'*70}")

    base_map50 = results[0]["map50"] if results else 0.0
    base_map = results[0]["map"] if results else 0.0

    for i, r in enumerate(results):
        d50 = r["map50"] - base_map50
        d = r["map"] - base_map
        sign50 = "+" if d50 >= 0 else ""
        sign = "+" if d >= 0 else ""
        if i == 0:
            print(f"  {i:<4}{r['name']:<30}{r['map50']:>8.4f}{r['map']:>10.4f}{'—':>9}{'—':>9}")
        else:
            print(f"  {i:<4}{r['name']:<30}{r['map50']:>8.4f}{r['map']:>10.4f}"
                  f"{sign50}{d50:>7.4f}{sign}{d:>7.4f}")

    print(f"  {'='*70}")
    print(f"\n  Results saved to: runs/ablation/ablation_results.json")


if __name__ == "__main__":
    main()
