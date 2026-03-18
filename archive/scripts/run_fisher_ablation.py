"""
Fisher-CIoU 修正实验：公平验证 Fisher 补偿效果。

核心修正：
  1. 从 yolo11n.pt 从头训（不是微调），和消融主线完全一致
  2. lr0=0.01, 300 epochs, patience=100（同 ablation.py）
  3. 放宽 clamp 范围 [0.5, 3.0]（原 [0.8, 1.3] 几乎无效果）
  4. DFL loss 不乘 scale_w（修正逻辑错误）
  5. 使用 ablation_nwd_p2 架构（消融最优架构）

实验组：
  A. nwd_p2 + Fisher-CIoU（修正版）:
     loss = α × SA-NWD + (1-α) × Fisher-Scale-Aware-CIoU
     Fisher 只加在 CIoU 上，DFL 不加
  B. nwd_p2 + Fisher-CIoU（宽参数）:
     同 A，但 ref_area=0.0005（数据集最小面积），clamp=[0.3, 5.0]
  C. nwd_p2 + 纯 Fisher-CIoU（无 SA-NWD）:
     loss = Fisher-Scale-Aware-CIoU（对照组）

对比基线：nwd_p2 消融实验 mAP50=0.9781（α×SA-NWD + (1-α)×普通CIoU）

Usage: python scripts/run_fisher_ablation.py --exp A|B|C
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, choices=['A', 'B', 'C'])
args = parser.parse_args()

import register_modules

# ============================================================
# 修正版 Fisher-CIoU patch（DFL 不乘 scale_w）
# ============================================================
def patch_fisher_ciou_fixed(c_base=12.0, k=1.0, alpha=0.5,
                             ref_area=0.002, max_scale=3.0, min_scale=0.5):
    """SA-NWD + Fisher-CIoU（修正版：DFL 不加 Fisher 权重）。"""
    from ultralytics.utils.loss import BboxLoss
    from ultralytics.utils.tal import bbox2dist
    from ultralytics.utils.metrics import bbox_iou
    from ultralytics_modules.nwd import sa_nwd

    def _forward(self, pred_dist, pred_bboxes, anchor_points,
                 target_bboxes, target_scores, target_scores_sum,
                 fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_fg = pred_bboxes[fg_mask]
        tb = target_bboxes[fg_mask]

        # Fisher scale weight: w(s) = sqrt(ref_area / s)
        img_area = float(imgsz[0]) * float(imgsz[1]) + 1e-8
        gt_areas = ((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])) / img_area
        scale_w = torch.sqrt(
            torch.tensor(ref_area, dtype=tb.dtype, device=tb.device) /
            gt_areas.clamp(min=1e-6)
        ).clamp(min=min_scale, max=max_scale).unsqueeze(-1)

        if alpha > 0:
            # SA-NWD loss（不乘 scale_w，SA-NWD 有自己的尺度自适应）
            sa_score = sa_nwd(pred_fg, tb, c_base=c_base, k=k)
            loss_sa = ((1.0 - sa_score).unsqueeze(-1) * weight).sum() / target_scores_sum
        else:
            loss_sa = torch.tensor(0.0, device=tb.device)

        # Fisher-compensated CIoU loss（scale_w 只加在这里）
        ciou = bbox_iou(pred_fg, tb, xywh=False, CIoU=True)
        loss_ciou = ((1.0 - ciou).unsqueeze(-1) * weight * scale_w).sum() / target_scores_sum

        loss_iou = alpha * loss_sa + (1.0 - alpha) * loss_ciou

        # DFL loss：不乘 scale_w（修正原 bug）
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes,
                                    self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist_s = pred_dist * stride
            pred_dist_s[..., 0::2] /= imgsz[1]
            pred_dist_s[..., 1::2] /= imgsz[0]
            import torch.nn.functional as F
            loss_dfl = (
                F.l1_loss(pred_dist_s[fg_mask], target_ltrb[fg_mask],
                          reduction="none").mean(-1, keepdim=True) * weight
            ).sum() / target_scores_sum

        return loss_iou, loss_dfl

    BboxLoss.forward = _forward
    tag = f"alpha={alpha}, ref={ref_area}, clamp=[{min_scale},{max_scale}]"
    print(f"✓ Fisher-CIoU (fixed, DFL clean) applied ({tag})")


def patch_fisher_ciou_only(ref_area=0.002, max_scale=3.0, min_scale=0.5):
    """纯 Fisher-CIoU（无 SA-NWD），对照组。"""
    patch_fisher_ciou_fixed(c_base=12.0, k=1.0, alpha=0.0,
                             ref_area=ref_area, max_scale=max_scale, min_scale=min_scale)


# ============================================================
# 实验配置
# ============================================================
from ultralytics_modules.nwd import patch_sa_nwd_tal

EXP_CONFIG = {
    'A': {
        'name': 'fisher_fixed_A',
        'desc': 'SA-NWD + Fisher-CIoU (fixed, standard params)',
        'setup': lambda: (
            patch_fisher_ciou_fixed(c_base=12.0, k=1.0, alpha=0.5,
                                     ref_area=0.002, max_scale=3.0, min_scale=0.5),
            patch_sa_nwd_tal(c_base=12.0, k=1.0, nwd_min=0.3),
        ),
    },
    'B': {
        'name': 'fisher_fixed_B',
        'desc': 'SA-NWD + Fisher-CIoU (wide params, ref=min_area)',
        'setup': lambda: (
            patch_fisher_ciou_fixed(c_base=12.0, k=1.0, alpha=0.5,
                                     ref_area=0.0005, max_scale=5.0, min_scale=0.3),
            patch_sa_nwd_tal(c_base=12.0, k=1.0, nwd_min=0.3),
        ),
    },
    'C': {
        'name': 'fisher_fixed_C',
        'desc': 'Pure Fisher-CIoU only (no SA-NWD, control group)',
        'setup': lambda: (
            patch_fisher_ciou_only(ref_area=0.002, max_scale=3.0, min_scale=0.5),
        ),
    },
}

cfg = EXP_CONFIG[args.exp]
print(f'\n{"="*60}')
print(f'  Experiment {args.exp}: {cfg["desc"]}')
print(f'  FROM SCRATCH: yolo11n.pt → ablation_nwd_p2, 300 epochs')
print(f'{"="*60}\n')

cfg['setup']()

from ultralytics import YOLO
model = YOLO('configs/ablation/ablation_nwd_p2.yaml')
model.train(
    pretrained='yolo11n.pt',
    data='configs/data.yaml',
    imgsz=1280, epochs=300, patience=100, batch=8,
    lr0=0.01, cos_lr=True, mosaic=1.0, mixup=0.15, copy_paste=0.2,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    warmup_epochs=5, workers=4, cache=False,
    project='/root/drone_detection/runs/ablation',
    name=cfg['name'], exist_ok=True,
)

r = model.val(data='configs/data.yaml', imgsz=1280, split='val')
result = {
    'name': cfg['desc'],
    'exp': args.exp,
    'map50': round(float(r.box.map50), 4),
    'map': round(float(r.box.map), 4),
    'precision': round(float(r.box.mp), 4),
    'recall': round(float(r.box.mr), 4),
}
result['f1'] = round(2*result['precision']*result['recall']/(result['precision']+result['recall']+1e-8), 4)
print(f'RESULT: {json.dumps(result)}')
