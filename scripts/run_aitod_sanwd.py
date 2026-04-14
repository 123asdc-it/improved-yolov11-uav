"""run_aitod_sanwd.py — E10: AI-TOD + SA-NWD (k=1.0, no P2)

TODO(c_base tuning): c_base=12 是私有集校准值（median normalized area ~0.0023）。
AI-TOD 物体约 12.8 px（imgsz=800 下 normalized area ~0.000256，比私有集小 9 倍），
理论上 c_base 应该约 4-6。固定 c_base=12 会让 C_adapt 过大、梯度信号过软，
SA-NWD 提升可能从预期 +5~10% 降到 +1~3%。
决策推迟到 baseline + SA-NWD 数字出来后再定（参见 plan Part 8 Risk A）。
"""
import os, sys, json
from pathlib import Path
PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
import register_modules  # noqa
from ultralytics_modules.nwd import patch_all_nwd
# c_base=12 见顶部 TODO 说明，跨数据集可能需要调整
patch_all_nwd(c_base=12.0, k=1.0, alpha=0.5, nwd_min=0.0)
print('[E10] AI-TOD + SA-NWD (k=1.0, alpha=0.5, no P2 head)')
from ultralytics import YOLO

EXP_NAME = 'aitod_sanwd'
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

result = {'name': 'E10: AI-TOD + SA-NWD (k=1.0)', 'exp': EXP_NAME,
          'map50': round(float(metrics.box.map50), 4), 'map': round(float(metrics.box.map), 4),
          'precision': round(float(metrics.box.mp), 4), 'recall': round(float(metrics.box.mr), 4)}
print('RESULT: ' + json.dumps(result))
out = Path(ABLATION_PROJECT) / EXP_NAME / 'result.json'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f: json.dump(result, f, indent=2)
