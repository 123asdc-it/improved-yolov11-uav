"""run_aitod_nwd_fixed.py — E9: AI-TOD + NWD k=0 (fixed C, standard NWD)

TODO(c_base tuning): k=0 固定 C=12 来自私有集校准。AI-TOD 物体小 9 倍，
C=12 对 AI-TOD 仍可能偏大；但本实验的目的是"和 SA-NWD 同 c_base 对照"，
所以跟其他 AI-TOD 实验保持 c_base=12 一致（参见 plan Part 8 Risk A）。
"""
import os, sys, json
from pathlib import Path
PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
import register_modules  # noqa
from ultralytics_modules.nwd import patch_all_nwd
# c_base=12 跟 SA-NWD 实验一致以保证对照（见顶部 TODO）
patch_all_nwd(c_base=12.0, k=0.0, alpha=0.5, nwd_min=0.0)
print('[E9] AI-TOD + NWD k=0 (fixed C=12, standard NWD hybrid)')
from ultralytics import YOLO

EXP_NAME = 'aitod_nwd_fixed'
ABLATION_PROJECT = '/root/drone_detection/runs/ablation'
WEIGHT_PATH = f'{ABLATION_PROJECT}/{EXP_NAME}/weights/best.pt'
AITOD_DATA = 'datasets/aitod/aitod_data.yaml'

TRAIN_ARGS = dict(data=AITOD_DATA, imgsz=800, epochs=300, patience=100,
                  batch=8, lr0=0.01, cos_lr=True, mosaic=1.0, mixup=0.1,
                  copy_paste=0.1, warmup_epochs=5, workers=4, cache=False, seed=0)

if Path(WEIGHT_PATH).exists():
    model = YOLO(WEIGHT_PATH)
    metrics = model.val(data=AITOD_DATA, imgsz=800, split='val')
else:
    model = YOLO('yolo11n.pt')
    model.train(project=ABLATION_PROJECT, name=EXP_NAME, exist_ok=True, **TRAIN_ARGS)
    metrics = model.val(data=AITOD_DATA, imgsz=800, split='val')

result = {'name': 'E9: AI-TOD + NWD k=0', 'exp': EXP_NAME,
          'map50': round(float(metrics.box.map50), 4), 'map': round(float(metrics.box.map), 4),
          'precision': round(float(metrics.box.mp), 4), 'recall': round(float(metrics.box.mr), 4)}
print('RESULT: ' + json.dumps(result))
out = Path(ABLATION_PROJECT) / EXP_NAME / 'result.json'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f: json.dump(result, f, indent=2)
