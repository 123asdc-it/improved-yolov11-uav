import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules
from ultralytics_modules.nwd import patch_scale_aware_loss
patch_scale_aware_loss(ref_area=0.002, max_scale=2.0, min_scale=0.5)
print('[Fisher] Scale-Aware CIoU only. TAL unchanged (IoU-based).')

from ultralytics import YOLO

best1 = 'runs/detect/runs/detect/two_stage_s1/weights/best.pt'
print('From: ' + best1 + '  (Stage1 mAP50=0.920)')

model = YOLO(best1)
model.train(
    epochs=250,
    patience=100,
    lr0=0.001,
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
    warmup_epochs=3,
    workers=4,
    cache=False,
    project='runs/detect',
    name='scale_aware_only',
    exist_ok=True,
)
r = model.val(data='configs/data.yaml', imgsz=1280, split='val')

result = {
    'name': 'Scale-Aware CIoU (Fisher-Guided)',
    'stage1_map50': 0.920,
    'map50': round(float(r.box.map50), 4),
    'map75': round(float(r.box.map75), 4),
    'map': round(float(r.box.map), 4),
    'precision': round(float(r.box.mp), 4),
    'recall': round(float(r.box.mr), 4),
}
result['f1'] = round(
    2 * result['precision'] * result['recall'] /
    (result['precision'] + result['recall'] + 1e-8), 4
)
print('RESULT: ' + json.dumps(result))
