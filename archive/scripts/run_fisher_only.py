"""
run_fisher_only.py — 纯 Fisher-Guided Scale-Aware CIoU，不含 SA-NWD

对比实验 A：单独验证 Fisher 补偿的贡献
  loss = scale_aware_CIoU，w(s) = sqrt(ref_area/s)，无 SA-NWD

起点：runs/ablation/nwd_p2_simam_pconv/weights/best.pt（mAP50=0.9666）
    架构完全一致：P2+SimAM+PConv，4-head，stride=[4,8,16,32]
    跳过冷启动，直接从收敛权重微调，100~150 epoch 即可出结果

论文用途：
    消融表中作为 "+Fisher-CIoU (no SA-NWD)" 行
    对比 run_clean（无Fisher）和 run_clean_fisher（SA-NWD+Fisher）
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules
from ultralytics_modules.nwd import patch_scale_aware_loss

patch_scale_aware_loss(ref_area=0.002, max_scale=1.3, min_scale=0.8)
print('[Fisher-Only] Scale-Aware CIoU, w(s)=sqrt(ref_area/s). No SA-NWD.')

from ultralytics import YOLO

BEST_PT = 'runs/ablation/nwd_p2_simam_pconv/weights/best.pt'
print(f'[Fisher-Only] Starting from: {BEST_PT}  (mAP50=0.9666, same arch)')

model = YOLO(BEST_PT)
model.train(
    data='configs/data.yaml',
    imgsz=1280,
    epochs=150,
    patience=50,
    batch=8,
    lr0=0.001,
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
    name='fisher_only',
    exist_ok=True,
)

r = model.val(data='configs/data.yaml', imgsz=1280, split='val')
result = {
    'name': 'Fisher-Only (Scale-Aware CIoU, no SA-NWD)',
    'base_map50': 0.9666,
    'map50':     round(float(r.box.map50), 4),
    'map75':     round(float(r.box.map75), 4),
    'map':       round(float(r.box.map), 4),
    'precision': round(float(r.box.mp), 4),
    'recall':    round(float(r.box.mr), 4),
}
result['f1'] = round(
    2 * result['precision'] * result['recall'] /
    (result['precision'] + result['recall'] + 1e-8), 4
)
print('RESULT: ' + json.dumps(result))
