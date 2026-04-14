"""run_aitod_sanwd_p2.py — E11: AI-TOD + SA-NWD + P2 Head (best config)

TODO(c_base tuning): c_base=12 是私有集校准值，AI-TOD 物体小 9 倍可能需要 c_base≈4-6。
决策推迟到结果出来后再定（参见 plan Part 8 Risk A 和 run_aitod_sanwd.py 顶部注释）。
"""
import os, sys, json
from pathlib import Path
PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
import register_modules  # noqa
from ultralytics_modules.nwd import patch_all_nwd
# c_base=12 见顶部 TODO 说明
patch_all_nwd(c_base=12.0, k=1.0, alpha=0.5, nwd_min=0.0)
print('[E11] AI-TOD + SA-NWD + P2 Head (best config: k=1.0, alpha=0.5)')
from ultralytics import YOLO

EXP_NAME = 'aitod_sanwd_p2'
ABLATION_PROJECT = '/root/drone_detection/runs/ablation'
WEIGHT_PATH = f'{ABLATION_PROJECT}/{EXP_NAME}/weights/best.pt'
AITOD_DATA = 'datasets/aitod/aitod_data.yaml'
NWD_P2_YAML = 'configs/ablation/ablation_nwd_p2.yaml'

TRAIN_ARGS = dict(data=AITOD_DATA, imgsz=800, epochs=300, patience=100,
                  batch=8, lr0=0.01, cos_lr=True, mosaic=1.0, mixup=0.1,
                  copy_paste=0.1, warmup_epochs=5, workers=4, cache=False, seed=0)

if Path(WEIGHT_PATH).exists():
    model = YOLO(WEIGHT_PATH)
    metrics = model.val(data=AITOD_DATA, imgsz=800, split='val')
else:
    model = YOLO(NWD_P2_YAML)
    model.train(pretrained='yolo11n.pt', project=ABLATION_PROJECT,
                name=EXP_NAME, exist_ok=True, **TRAIN_ARGS)
    metrics = model.val(data=AITOD_DATA, imgsz=800, split='val')

result = {'name': 'E11: AI-TOD + SA-NWD + P2', 'exp': EXP_NAME,
          'map50': round(float(metrics.box.map50), 4), 'map': round(float(metrics.box.map), 4),
          'precision': round(float(metrics.box.mp), 4), 'recall': round(float(metrics.box.mr), 4)}
print('RESULT: ' + json.dumps(result))
out = Path(ABLATION_PROJECT) / EXP_NAME / 'result.json'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f: json.dump(result, f, indent=2)
