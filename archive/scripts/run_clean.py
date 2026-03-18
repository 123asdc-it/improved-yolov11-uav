"""
run_clean.py — 干净的单阶段训练，论文核心贡献最小集

移除的组件（保留为可选，代码不删）：
  - 两阶段训练：直接一阶段从 yolo11n.pt 出发，消除 loss 切换引起的震荡
  - SA-NWD-TAL：不 patch TaskAlignedAssigner，标准 IoU-TAL 稳定可靠
  - Scale-Aware loss batch-normalize：整体不用 Scale-Aware，直接用纯 SA-NWD loss
  - BiFPN：yaml 已移除（消融实验确认负贡献）
  - CARAFE / RepVGG：不用 sota.yaml，避免小数据集过拟合

保留的核心贡献（与消融最优组 0.9666 完全一致）：
  - P2 检测头（stride=4）：yolo11n-improved.yaml
  - SimAM 零参数注意力：yolo11n-improved.yaml
  - PConv_C3k2 轻量化 neck：yolo11n-improved.yaml
  - SA-NWD loss（hybrid alpha=0.5）：patch_sa_nwd_loss，不含 TAL patch

使用方法：python scripts/run_clean.py
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules

# ── 核心贡献：只用 SA-NWD loss，不动 TAL ──────────────────────
from ultralytics_modules.nwd import patch_sa_nwd_loss
patch_sa_nwd_loss(c_base=12.0, k=1.0, alpha=0.5)
print('[Clean] SA-NWD loss (hybrid alpha=0.5). TAL unchanged (standard IoU).')
print('[Clean] No two-stage, no BiFPN, no SA-NWD-TAL, no Scale-Aware.')

from ultralytics import YOLO

model = YOLO('configs/yolo11n-improved.yaml')
model.train(
    pretrained='yolo11n.pt',
    data='configs/data.yaml',
    imgsz=1280,
    epochs=300,
    patience=100,
    batch=8,
    lr0=0.01,
    cos_lr=True,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    warmup_epochs=5,
    workers=4,
    cache=False,
    project='runs/detect',
    name='clean',
    exist_ok=True,
)

r = model.val(data='configs/data.yaml', imgsz=1280, split='val')
result = {
    'name': 'Clean (P2+SimAM+PConv+SA-NWD, single-stage)',
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
