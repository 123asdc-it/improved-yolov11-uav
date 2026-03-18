"""
run_clean_fisher.py — SA-NWD + Fisher-Guided Scale-Aware CIoU（两个贡献叠加）

对比实验 B：验证 SA-NWD 与 Fisher 补偿的联合贡献
  loss = 0.5 * SA-NWD  +  0.5 * scale_aware_CIoU
  SA-NWD:           Wasserstein 回归，scale-adaptive C
  scale_aware_CIoU: CIoU × w(s)，w(s) = sqrt(ref_area/s)

起点：runs/ablation/nwd_p2_simam_pconv/weights/best.pt（mAP50=0.9666）
    架构完全一致：P2+SimAM+PConv，4-head，stride=[4,8,16,32]

论文用途：
    消融表最终行 "+SA-NWD + Fisher-CIoU"
    期望：在 run_fisher_only 基础上进一步提升
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules
from ultralytics_modules.nwd import patch_sa_nwd_fisher_loss

patch_sa_nwd_fisher_loss(c_base=12.0, k=1.0, alpha=0.5,
                          ref_area=0.002, max_scale=1.3, min_scale=0.8)
print('[SA-NWD+Fisher] Combined: 0.5*SA-NWD + 0.5*Fisher-CIoU. No TAL patch.')

from ultralytics import YOLO

BEST_PT = 'runs/ablation/nwd_p2_simam_pconv/weights/best.pt'
print(f'[SA-NWD+Fisher] Starting from: {BEST_PT}  (mAP50=0.9666, same arch)')

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
    name='clean_fisher',
    exist_ok=True,
)

r = model.val(data='configs/data.yaml', imgsz=1280, split='val')
result = {
    'name': 'SA-NWD + Fisher-CIoU (combined)',
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
