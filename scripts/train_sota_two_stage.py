"""
SOTA Two-stage: Stage1=CIoU, Stage2=hybrid SA-NWD+CIoU
"""
import os, sys, json
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules
from ultralytics import YOLO

COMMON = dict(
    data='configs/data.yaml', imgsz=1280, batch=2,
    cos_lr=True, mosaic=1.0, mixup=0.15, copy_paste=0.2,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    warmup_epochs=5, workers=4, cache=False,
)

print('=== SOTA Stage 1: Pure CIoU (50 epochs) ===')
model = YOLO('configs/yolo11n-sota.yaml')
model.train(pretrained='yolo11n.pt', epochs=50, patience=60, lr0=0.01,
            project='runs/detect', name='sota_s1', exist_ok=True, **COMMON)
r1 = model.val(data='configs/data.yaml', imgsz=1280, split='val')
print(f'Stage 1 mAP50: {float(r1.box.map50):.4f}')

best1 = 'runs/detect/sota_s1/weights/best.pt'
if not Path(best1).exists():
    best1 = 'runs/detect/sota_s1/weights/last.pt'

print(f'=== SOTA Stage 2: Hybrid SA-NWD+CIoU+NMS (250 epochs) ===')
from ultralytics_modules.nwd import patch_all_nwd
patch_all_nwd(c_base=12.0, k=2.0, alpha=0.5, use_sa=True, use_nwd_nms=True)

model2 = YOLO(best1)
model2.train(epochs=250, patience=100, lr0=0.001,
             project='runs/detect', name='sota_s2', exist_ok=True, **COMMON)
r2 = model2.val(data='configs/data.yaml', imgsz=1280, split='val')

result = {'name': 'SOTA Two-Stage',
          'map50': round(float(r2.box.map50), 4),
          'map': round(float(r2.box.map), 4),
          'precision': round(float(r2.box.mp), 4),
          'recall': round(float(r2.box.mr), 4)}
print(f'RESULT: {json.dumps(result)}')
