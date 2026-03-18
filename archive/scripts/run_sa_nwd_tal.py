"""
run_sa_nwd_tal.py — 验证 SA-NWD 全流程（Loss + TAL），修复版

验证的内容：
  - SA-NWD Loss: alpha * SA-NWD + (1-alpha) * CIoU
  - SA-NWD-TAL:  nwd_min=0.3 正确生效（修复了之前写死 0.01 的 bug）
  - 无两阶段：直接从 best.pt 微调，消除 loss 切换震荡
  - 无 Fisher：单独验证 TAL 的贡献

起点：runs/ablation/nwd_p2_simam_pconv/weights/best.pt（mAP50=0.9666）
    架构完全一致：P2+SimAM+PConv，4-head

论文用途：
    消融表中 "+SA-NWD-TAL" 行
    期望：比 0.9666（无 TAL）更高，若稳定则说明 TAL 有贡献
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules
from ultralytics_modules.nwd import patch_sa_nwd_loss, patch_sa_nwd_tal

# Loss: SA-NWD hybrid (同 run_clean)
patch_sa_nwd_loss(c_base=12.0, k=1.0, alpha=0.5)
# TAL: SA-NWD 替代 IoU，nwd_min=0.3 正确过滤低质量 anchor（bug 已修复）
patch_sa_nwd_tal(c_base=12.0, k=1.0, nwd_min=0.3)
print('[SA-NWD-TAL] Loss: SA-NWD+CIoU hybrid. TAL: SA-NWD with nwd_min=0.3 (fixed).')

from ultralytics import YOLO

BEST_PT = 'runs/ablation/nwd_p2_simam_pconv/weights/best.pt'
print(f'[SA-NWD-TAL] Starting from: {BEST_PT}  (mAP50=0.9666, same arch)')

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
    name='sa_nwd_tal',
    exist_ok=True,
)

r = model.val(data='configs/data.yaml', imgsz=1280, split='val')
result = {
    'name': 'SA-NWD Loss + SA-NWD-TAL (full pipeline, no Fisher)',
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
