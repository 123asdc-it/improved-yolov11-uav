"""
run_alpha_0p3.py — SA-NWD alpha=0.3 消融实验

hybrid loss = 0.3 * SA-NWD + 0.7 * CIoU
对比 alpha=0.5（当前最优）和 alpha=0.7，验证混合权重对结果的影响。

Architecture: nwd_p2（P2 head，与主消融一致）
Baseline for comparison: nwd_p2 alpha=0.5 → mAP50=0.9781

Server path: /root/drone_detection/
Output:      runs/ablation/nwd_p2_alpha03/
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules  # noqa: F401

from ultralytics_modules.nwd import patch_all_nwd
patch_all_nwd(c_base=12.0, k=1.0, alpha=0.3, use_sa=True, nwd_min=0.0)
print('[alpha=0.3] loss = 0.3 * SA-NWD + 0.7 * CIoU, architecture: nwd_p2')

from ultralytics import YOLO

EXP_NAME = 'nwd_p2_alpha03'
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
    'name': 'alpha=0.3 (0.3*SA-NWD + 0.7*CIoU)',
    'exp': EXP_NAME,
    'alpha': 0.3,
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
print(f'Result saved to {out_path}')
