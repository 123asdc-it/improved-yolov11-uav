"""
run_p2_only.py — Experiment E_p2only: P2 Head Only (no SA-NWD)

Ablation: isolates the independent contribution of the P2 detection head.
  loss = CIoU (standard, no NWD patch)
  architecture = P2+P3+P4+P5 (4-scale, same as nwd_p2 architecture)

Comparison chain:
  Baseline (no P2, CIoU):  mAP50=0.9600
  +P2 only (this exp):     mAP50=?         ← P2 head independent contribution
  +SA-NWD (no P2):         mAP50=0.9705    ← SA-NWD independent contribution
  +SA-NWD+P2 (best):       mAP50=0.9781    ← combined

Server path: /root/drone_detection/
Output:      runs/ablation/p2_only/
Config:      configs/ablation/ablation_p2.yaml
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

# NO NWD patch — standard CIoU loss, stock TAL
# register_modules still needed for any custom YAML references
import register_modules  # noqa: F401

from ultralytics import YOLO

EXP_NAME = 'p2_only'
ABLATION_PROJECT = '/root/drone_detection/runs/ablation'
WEIGHT_PATH = f'{ABLATION_PROJECT}/{EXP_NAME}/weights/best.pt'
P2_YAML = 'configs/ablation/ablation_p2.yaml'

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

print('[E_p2only] P2 Head Only, standard CIoU loss, no NWD patch')
print('[E_p2only] Architecture: P2+P3+P4+P5 (ablation_p2.yaml)')

if Path(WEIGHT_PATH).exists():
    print(f'[SKIP] {WEIGHT_PATH} already exists, running val only')
    model = YOLO(WEIGHT_PATH)
    metrics = model.val(data='configs/data.yaml', imgsz=1280, split='val')
else:
    model = YOLO(P2_YAML)
    model.train(
        pretrained='yolo11n.pt',
        project=ABLATION_PROJECT,
        name=EXP_NAME,
        exist_ok=True,
        **TRAIN_ARGS,
    )
    metrics = model.val(data='configs/data.yaml', imgsz=1280, split='val')

result = {
    'name': 'E_p2only: P2 Head only (no SA-NWD)',
    'exp': EXP_NAME,
    'config': 'ablation_p2.yaml, CIoU loss, standard TAL',
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

out_path = Path(ABLATION_PROJECT) / EXP_NAME / 'result.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f'[E_p2only] Result saved to {out_path}')
