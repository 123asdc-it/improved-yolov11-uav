"""
run_nwd_reverse.py — Experiment E: Reverse SA-NWD (C proportional to sqrt(S))

Ablation: tests the Fisher-theoretically-motivated direction.
  C_adapt = c_base * (1 + k * sqrt(avg_area / ref_area))
  small objects → C < 2*c_base (stricter loss, larger gradient)
  large objects → C > 2*c_base (more lenient loss)

This is the OPPOSITE direction of the current SA-NWD design.
Expected outcome: if this performs worse than k=1.0 (SA-NWD),
it confirms that the regularization interpretation is correct
and the Fisher-compensation direction is not beneficial in practice.

Architecture: nwd_p2 (same as best configuration)
Comparison baseline: nwd_p2 k=1.0 → mAP50=0.9781

C values at k=1, c_base=12 (drone dataset, ref_area=0.002315):
  p10  (tiny):   C ≈ 16.7  (stricter than fixed C=12)
  median:        C ≈ 24.0
  p90  (large):  C ≈ 29.0  (more lenient)

Server path: /root/drone_detection/
Output:      runs/ablation/nwd_p2_reverse_c/
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules  # noqa: F401

from ultralytics_modules.nwd import patch_sa_nwd_loss_reverse, patch_sa_nwd_tal

# Reverse loss: C grows with scale
patch_sa_nwd_loss_reverse(c_base=12.0, k=1.0, alpha=0.5, ref_area=0.002315)
# TAL: keep standard SA-NWD (k=1.0) — only loss direction changes
patch_sa_nwd_tal(c_base=12.0, k=1.0, nwd_min=0.0)
print('[Exp E] Reverse SA-NWD: C = 12*(1 + sqrt(S/0.002315)), alpha=0.5')
print('[Exp E] Architecture: nwd_p2 (P2 head)')

from ultralytics import YOLO

EXP_NAME = 'nwd_p2_reverse_c'
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
    'name': 'Exp E: Reverse SA-NWD (C grows with scale)',
    'exp': EXP_NAME,
    'config': 'nwd_p2 arch, C=12*(1+sqrt(S/ref)), alpha=0.5',
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
print(f'[Exp E] Result saved to {out_path}')
