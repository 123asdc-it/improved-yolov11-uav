"""
run_k_sensitivity.py — k hyperparameter sensitivity sweep

Trains three variants of SA-NWD with different k values on nwd_p2 architecture.
k=1.0 is already available (nwd_p2, mAP50=0.9781) and is NOT re-run here.

Experiments:
  k=0.5: moderate adaptation  C(median) ≈ 137
  k=2.0: stronger adaptation  C(median) ≈ 511
  k=3.0: aggressive adaptation C(median) ≈ 760

Together with existing results:
  k=0.0 (Exp D, Fixed NWD)
  k=0.5 (this script)
  k=1.0 (existing nwd_p2, mAP50=0.9781)  ← NOT re-run
  k=2.0 (this script)
  k=3.0 (this script)

Architecture: nwd_p2 (P2 head, same as best config)
All with alpha=0.5, nwd_min=0.0, seed=0, 300 epochs

Server path: /root/drone_detection/
Output:      runs/ablation/nwd_p2_k{value}/
"""

import os
import sys
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

ABLATION_PROJECT = '/root/drone_detection/runs/ablation'
NWD_P2_YAML = 'configs/ablation/ablation_nwd_p2.yaml'

TRAIN_ARGS = dict(
    data='configs/data.yaml',
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
    seed=0,
)

# k=1.0 already done (nwd_p2), skip it
# GRSL 裁剪版可通过环境变量只跑部分点：GRSL_K_VALUES="0.5,2.0"
_env_k = os.environ.get('GRSL_K_VALUES', '').strip()
if _env_k:
    K_VALUES = [float(x) for x in _env_k.split(',') if x.strip()]
    print(f'[k-sweep] 从环境变量 GRSL_K_VALUES 读取 K_VALUES={K_VALUES}')
else:
    K_VALUES = [0.5, 2.0, 3.0]

# C(median) reference values for logging
import math
REF_MEDIAN = 0.002315
C_BASE = 12.0
for k in K_VALUES:
    c_med = C_BASE * (1 + k / math.sqrt(REF_MEDIAN))
    print(f'k={k}: C(median)={c_med:.1f}')

SINGLE_EXP_SCRIPT = '''
import os, sys, json, math
from pathlib import Path

PROJECT_ROOT = {project_root!r}
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

import register_modules

from ultralytics_modules.nwd import patch_all_nwd
k_val = {k_val!r}
patch_all_nwd(c_base=12.0, k=k_val, alpha=0.5, use_sa=True, nwd_min=0.0)

ref = 0.002315
c_med = 12.0 * (1 + k_val / (ref ** 0.5))
print(f"[k={k_val:.1f}] C(median)={{c_med:.1f}}, architecture: nwd_p2")

from ultralytics import YOLO

exp_name = {exp_name!r}
weight_path = {weight_path!r}

if Path(weight_path).exists():
    print(f"[SKIP] {{weight_path}} exists, val only")
    model = YOLO(weight_path)
    metrics = model.val(data="configs/data.yaml", imgsz=1280, split="val")
else:
    model = YOLO({model_cfg!r})
    model.train(
        pretrained="yolo11n.pt",
        project={ablation_project!r},
        name=exp_name,
        exist_ok=True,
        **{train_args!r},
    )
    metrics = model.val(data="configs/data.yaml", imgsz=1280, split="val")

result = {{
    "name": f"k-sweep k={{k_val:.1f}}",
    "exp": exp_name,
    "k": k_val,
    "map50":     round(float(metrics.box.map50), 4),
    "map":       round(float(metrics.box.map), 4),
    "precision": round(float(metrics.box.mp), 4),
    "recall":    round(float(metrics.box.mr), 4),
}}
result["f1"] = round(
    2 * result["precision"] * result["recall"] /
    (result["precision"] + result["recall"] + 1e-8), 4
)
print("RESULT: " + json.dumps(result))

out_path = Path({ablation_project!r}) / exp_name / "result.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Result saved to {{out_path}}")
'''

all_results = []

for k_val in K_VALUES:
    k_str = str(k_val).replace('.', 'p')   # e.g. 0.5 → 0p5
    exp_name = f'nwd_p2_k{k_str}'
    weight_path = f'{ABLATION_PROJECT}/{exp_name}/weights/best.pt'

    script = SINGLE_EXP_SCRIPT.format(
        project_root=str(PROJECT_ROOT),
        k_val=k_val,
        exp_name=exp_name,
        weight_path=weight_path,
        model_cfg=NWD_P2_YAML,
        ablation_project=ABLATION_PROJECT,
        train_args=TRAIN_ARGS,
    )

    print(f'\n{"="*60}')
    print(f'  k-sweep: k={k_val}  →  {exp_name}')
    print(f'{"="*60}')

    proc = subprocess.Popen(
        [sys.executable, '-c', script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    result_line = None
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end='', flush=True)
        if line.startswith('RESULT:'):
            result_line = line[len('RESULT:'):].strip()

    proc.wait()
    if proc.returncode != 0:
        print(f'[ERROR] k={k_val} exited with code {proc.returncode}')
        all_results.append({'k': k_val, 'exp': exp_name, 'map50': 0.0, 'map': 0.0})
    elif result_line:
        all_results.append(json.loads(result_line))

# Append known k=1.0 result
all_results.append({
    'name': 'k-sweep k=1.0 (existing)',
    'exp': 'nwd_p2',
    'k': 1.0,
    'map50': 0.9781,
    'note': 'existing result, not re-run',
})

all_results.sort(key=lambda x: x.get('k', 0))

print(f'\n{"="*60}')
print('  k-Sensitivity Summary')
print(f'{"="*60}')
print(f'  {"k":<8} {"exp":<25} {"mAP50":>8}')
print(f'  {"-"*45}')
for r in all_results:
    print(f'  {r.get("k", "?"):<8} {r.get("exp", "?"):<25} {r.get("map50", 0):>8.4f}')

# Save summary
summary_path = Path(ABLATION_PROJECT) / 'k_sensitivity_results.json'
with open(summary_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'\nSummary saved to {summary_path}')
