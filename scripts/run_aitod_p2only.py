"""run_aitod_p2only.py — E12: AI-TOD + P2 only (no SA-NWD, CIoU loss)"""
import os, sys, json
from pathlib import Path
PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
import register_modules  # noqa
# NO NWD patch — stock CIoU loss
print('[E12] AI-TOD + P2 only (CIoU loss, no NWD patch)')
from ultralytics import YOLO

EXP_NAME = 'aitod_p2only'
ABLATION_PROJECT = '/root/drone_detection/runs/ablation'
WEIGHT_PATH = f'{ABLATION_PROJECT}/{EXP_NAME}/weights/best.pt'
AITOD_DATA = 'datasets/aitod_converted/aitod_data.yaml'
P2_YAML = 'configs/ablation/ablation_p2.yaml'

TRAIN_ARGS = dict(data=AITOD_DATA, imgsz=800, epochs=300, patience=100,
                  batch=8, lr0=0.01, cos_lr=True, mosaic=1.0, mixup=0.1,
                  copy_paste=0.1, warmup_epochs=5, workers=4, cache=False, seed=0)

if Path(WEIGHT_PATH).exists():
    model = YOLO(WEIGHT_PATH)
    metrics = model.val(data=AITOD_DATA, imgsz=800, split='val')
else:
    model = YOLO(P2_YAML)
    model.train(pretrained='yolo11n.pt', project=ABLATION_PROJECT,
                name=EXP_NAME, exist_ok=True, **TRAIN_ARGS)
    metrics = model.val(data=AITOD_DATA, imgsz=800, split='val')

result = {'name': 'E12: AI-TOD + P2 only', 'exp': EXP_NAME,
          'map50': round(float(metrics.box.map50), 4), 'map': round(float(metrics.box.map), 4),
          'precision': round(float(metrics.box.mp), 4), 'recall': round(float(metrics.box.mr), 4)}
print('RESULT: ' + json.dumps(result))
out = Path(ABLATION_PROJECT) / EXP_NAME / 'result.json'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f: json.dump(result, f, indent=2)
