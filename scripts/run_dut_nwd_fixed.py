"""run_dut_nwd_fixed.py — DUT Anti-UAV + NWD k=0 (fixed C, standard NWD)"""
import os, sys, json
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
import register_modules  # noqa

from ultralytics_modules.nwd import patch_all_nwd
# k=0: C_adapt = c_base * (1 + 0/sqrt(S)) = c_base = 12 (fixed constant = standard NWD)
patch_all_nwd(c_base=12.0, k=0.0, alpha=0.5, nwd_min=0.0)
print('[DUT NWD k=0] Fixed C=12, standard NWD hybrid, no P2')

from ultralytics import YOLO

EXP_NAME = 'dut_nwd_fixed'
ABLATION_PROJECT = '/root/drone_detection/runs/ablation'
WEIGHT_PATH = f'{ABLATION_PROJECT}/{EXP_NAME}/weights/best.pt'
DUT_DATA = 'datasets/dut/dut_data.yaml'

TRAIN_ARGS = dict(
    data=DUT_DATA, imgsz=1280, epochs=300, patience=100,
    batch=8, lr0=0.01, cos_lr=True, mosaic=1.0, mixup=0.15,
    copy_paste=0.2, warmup_epochs=5, workers=4, cache=False, seed=0,
)

if Path(WEIGHT_PATH).exists():
    print(f'[SKIP] Running val only')
    model = YOLO(WEIGHT_PATH)
    metrics = model.val(data=DUT_DATA, imgsz=1280, split='val')
else:
    model = YOLO('yolo11n.pt')
    model.train(project=ABLATION_PROJECT, name=EXP_NAME, exist_ok=True, **TRAIN_ARGS)
    metrics = model.val(data=DUT_DATA, imgsz=1280, split='val')

result = {
    'name': 'DUT + NWD k=0 (fixed C)', 'exp': EXP_NAME,
    'dataset': 'DUT Anti-UAV',
    'map50': round(float(metrics.box.map50), 4),
    'map':   round(float(metrics.box.map), 4),
    'precision': round(float(metrics.box.mp), 4),
    'recall':    round(float(metrics.box.mr), 4),
}
print('RESULT: ' + json.dumps(result))
out = Path(ABLATION_PROJECT) / EXP_NAME / 'result.json'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f:
    json.dump(result, f, indent=2)
