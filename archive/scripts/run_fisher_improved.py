"""
run_fisher_improved.py — Fisher-Guided Improved Model Training

Architecture (all derived from Fisher information analysis I_IoU ∝ 1/s^2):
  - P2 detection head (stride=4): insufficient feature resolution for tiny objects
  - SimAM zero-param attention:   low discriminability of tiny object features
  - PConv_C3k2 lightweight neck:  compensates P2 head computational cost
  - NO BiFPN (ablation shows negative contribution: 0.9373 < 0.9666)

Loss (Fisher-Guided):
  Stage 1 (50 ep): Standard CIoU — stabilize pretrained weights
  Stage 2 (250ep): Scale-Aware CIoU (w ∝ 1/sqrt(s)) — Fisher compensation
                   + SA-NWD for P2 head (via SA-NWD-TAL, nwd_min=0.3)

Usage: python scripts/run_fisher_improved.py
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules
from ultralytics import YOLO

COMMON = dict(
    data='configs/data.yaml',
    imgsz=1280,
    batch=8,
    cos_lr=True,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    workers=4,
    cache=False,
)

# -------------------------------------------------------
# Stage 1: Standard CIoU — stabilize pretrained weights
# -------------------------------------------------------
print('\n' + '=' * 60)
print('  Stage 1: Standard CIoU (50 epochs)')
print('  Purpose: stabilize pretrained weights before scale-aware loss')
print('=' * 60)

model = YOLO('configs/yolo11n-improved.yaml')
model.train(
    pretrained='yolo11n.pt',
    epochs=50,
    patience=60,
    lr0=0.01,
    warmup_epochs=5,
    project='runs/detect',
    name='fisher_s1',
    exist_ok=True,
    **COMMON,
)
r1 = model.val(data='configs/data.yaml', imgsz=1280, split='val')
s1_map50 = round(float(r1.box.map50), 4)
print(f'  Stage 1 done: val mAP50 = {s1_map50}')

best1 = Path('runs/detect/runs/detect/fisher_s1/weights/best.pt')
if not best1.exists():
    best1 = Path('runs/detect/fisher_s1/weights/best.pt')
if not best1.exists():
    # fallback: find any best.pt under fisher_s1
    candidates = list(Path('runs').rglob('fisher_s1/weights/best.pt'))
    best1 = candidates[0] if candidates else Path('runs/detect/fisher_s1/weights/last.pt')
print(f'  Loading from: {best1}')

# -------------------------------------------------------
# Stage 2: Fisher-Guided Loss
#   - Scale-Aware CIoU: w(s) = sqrt(ref_area/s) (Fisher compensation)
#   - SA-NWD-TAL: better positive assignment for tiny objects
# -------------------------------------------------------
print('\n' + '=' * 60)
print('  Stage 2: Fisher-Guided Loss (250 epochs)')
print('  Scale-Aware CIoU + SA-NWD-TAL (nwd_min=0.3)')
print('=' * 60)

from ultralytics_modules.nwd import patch_scale_aware_loss, patch_sa_nwd_tal
patch_scale_aware_loss(ref_area=0.002, max_scale=2.0, min_scale=0.5)
patch_sa_nwd_tal(c_base=12.0, k=1.0, nwd_min=0.3)

model2 = YOLO(str(best1))
model2.train(
    epochs=250,
    patience=100,
    lr0=0.001,
    warmup_epochs=3,
    project='runs/detect',
    name='fisher_s2',
    exist_ok=True,
    **COMMON,
)
r2 = model2.val(data='configs/data.yaml', imgsz=1280, split='val')

result = {
    'name': 'Fisher-Guided Improved (Scale-Aware CIoU + SA-NWD-TAL)',
    'stage1_map50': s1_map50,
    'map50': round(float(r2.box.map50), 4),
    'map75': round(float(r2.box.map75), 4),
    'map': round(float(r2.box.map), 4),
    'precision': round(float(r2.box.mp), 4),
    'recall': round(float(r2.box.mr), 4),
}
result['f1'] = round(
    2 * result['precision'] * result['recall'] /
    (result['precision'] + result['recall'] + 1e-8), 4
)

print('\n' + '=' * 60)
print(f"  Stage 1 mAP50: {s1_map50}")
print(f"  Stage 2 mAP50: {result['map50']}")
print('=' * 60)
print(f'RESULT: {json.dumps(result)}')

os.makedirs('runs/fisher', exist_ok=True)
with open('runs/fisher/result.json', 'w') as f:
    json.dump(result, f, indent=2)
