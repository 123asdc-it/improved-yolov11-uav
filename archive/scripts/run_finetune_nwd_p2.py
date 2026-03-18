"""
run_finetune_nwd_p2.py — 路径二+三结合：从 nwd_p2/best.pt 微调，改用纯标准 NWD

起点：runs/ablation/nwd_p2/weights/best.pt（mAP50=0.9781）

改进后策略（修复4个问题）：
  问题1：lr 过高打破收敛 → lr0=0.0001（比原来低 100 倍）
  问题2：warmup 扰动收敛点 → warmup_epochs=0（无热身，直接从 lr0 开始）
  问题3：close_mosaic 引入噪声 → close_mosaic=0（不关闭，保持训练一致性）
  问题4：强增强阻碍精细收敛 → mixup=0.05, copy_paste=0.05（大幅减弱）

路径三结合：改用纯标准 NWD（k=0, alpha=1.0），给 loss landscape 轻微变化，
  比同 loss 微调更有可能找到新的更高极值。

预期：突破 0.9781，目标 0.980+
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules
from ultralytics_modules.nwd import patch_all_nwd

# 路径三结合：纯标准 NWD（k=0, alpha=1.0）
# 相比 nwd_p2 的 SA-NWD hybrid，loss landscape 有轻微变化
# 给模型一个新的优化方向，同时保持 Wasserstein 度量
patch_all_nwd(c_base=12.0, k=0.0, alpha=1.0, use_sa=False,
              use_nwd_nms=False, nwd_min=0.3)
print('[Finetune-NWD-P2-v2] Pure standard NWD (k=0, alpha=1.0). No warmup, lr=0.0001.')

from ultralytics import YOLO

BEST_PT = 'runs/ablation/nwd_p2/weights/best.pt'
print(f'[Finetune-NWD-P2-v2] Starting from: {BEST_PT}  (mAP50=0.9781)')

model = YOLO(BEST_PT)
model.train(
    data='configs/data.yaml',
    imgsz=1280,
    epochs=50,
    patience=30,
    batch=8,
    optimizer='SGD',
    lr0=0.0001,
    momentum=0.937,
    weight_decay=0.0005,
    lrf=0.1,
    cos_lr=True,
    warmup_epochs=0,      # 已收敛权重，无需热身
    warmup_bias_lr=0.0,
    mosaic=1.0,
    mixup=0.05,           # 大幅减弱：0.15 → 0.05
    copy_paste=0.05,      # 大幅减弱：0.20 → 0.05
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    close_mosaic=0,       # 不关闭 mosaic，保持一致性
    workers=4,
    cache=False,
    project='runs/detect',
    name='finetune_nwd_p2_v2',
    exist_ok=True,
)

r = model.val(data='configs/data.yaml', imgsz=1280, split='val')
result = {
    'name': 'Finetune NWD+P2 v2 (pure NWD k=0, lr=0.0001, from best.pt 0.9781)',
    'base_map50': 0.9781,
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
