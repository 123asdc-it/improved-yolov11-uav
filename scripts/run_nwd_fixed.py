"""
run_nwd_fixed.py — Experiment D: Fixed NWD (k=0, C=const)

Ablation: isolates the contribution of adaptive C vs. fixed C.
  loss = 0.5 * NWD(k=0, C=c_base) + 0.5 * CIoU
  k=0 → c_adapt = c_base = 12 for ALL object scales (standard NWD)

Architecture: nwd_p2 (same as best configuration, adds P2 head)
Baseline for comparison: nwd_p2 with k=1.0 → mAP50=0.9781

If Exp D ≈ 0.9781: adaptive C has no independent contribution;
                    NWD itself is the main driver.
If Exp D < 0.9781:  adaptive C provides measurable benefit,
                    supporting the SA-NWD design choice.

Server path: /root/drone_detection/
Output:      runs/ablation/nwd_p2_fixed_c/
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules  # noqa: F401

# k=0: c_adapt = c_base * (1 + 0/sqrt(S)) = c_base = 12 (fixed constant)
# Note: use_sa=True with k=0 intentionally reuses the SA-NWD code path
# with k=0, which mathematically reduces to standard NWD (fixed C=c_base).
# This is equivalent to use_sa=False but keeps identical code paths for fairness.
from ultralytics_modules.nwd import patch_all_nwd
patch_all_nwd(c_base=12.0, k=0.0, alpha=0.5, use_sa=True, nwd_min=0.0)
print('[Exp D] Fixed NWD: k=0, C=12 (constant), alpha=0.5')
print('[Exp D] Architecture: nwd_p2 (P2 head, same as best config)')

from ultralytics import YOLO

EXP_NAME = 'nwd_p2_fixed_c'
ABLATION_PROJECT = '/root/drone_detection/runs/ablation'
WEIGHT_PATH = f'{ABLATION_PROJECT}/{EXP_NAME}/weights/best.pt'
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

if Path(WEIGHT_PATH).exists():
    print(f'[SKIP] {WEIGHT_PATH} already exists, running val only')
    model = YOLO(WEIGHT_PATH)
    metrics = model.val(data='configs/data.yaml', imgsz=1280, split='val')
else:
    model = YOLO(NWD_P2_YAML)
    model.train(
        pretrained='yolo11n.pt',
        project=ABLATION_PROJECT,
        name=EXP_NAME,
        exist_ok=True,
        **TRAIN_ARGS,
    )
    metrics = model.val(data='configs/data.yaml', imgsz=1280, split='val')

result = {
    'name': 'Exp D: Fixed NWD (k=0, C=const)',
    'exp': EXP_NAME,
    'config': 'nwd_p2 arch, k=0, alpha=0.5',
    'map50':     round(float(metrics.box.map50), 4),
    'map':       round(float(metrics.box.map), 4),
    'precision': round(float(metrics.box.mp), 4),
    'recall':    round(float(metrics.box.mr), 4),
}
result['f1'] = round(
    2 * result['precision'] * result['recall'] /
    (result['precision'] + result['recall'] + 1e-8), 4
)
print('RESULT: ' + json.dumps(result))

# Save to file
out_path = Path(ABLATION_PROJECT) / EXP_NAME / 'result.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Exp D] Result saved to {out_path}')
