"""
run_working_point_log.py — Experiment E1: W₂/C Working Point Distribution

Purpose:
  Statistics the distribution of x = W₂(pred, gt) / C_adapt during training
  to determine Proposition 2 narrative angle for the paper.

  x ∈ [0.3, 1.5]  → working-point equalization narrative
  x ∈ [0.01, 0.1] → training stability narrative (preventing explosion)
  x ≈ 0.003       → need to rethink theoretical angle

Method:
  Monkey-patches SA-NWD to log (W₂, C_adapt, x=W₂/C) statistics every 10 epochs
  using a forward hook on BboxLoss. Statistics written to a JSON log file.

  Uses the same nwd_p2 configuration as the best model (~6h training).

Output:
  runs/ablation/nwd_p2_w2log/result.json           (final metrics)
  runs/ablation/nwd_p2_w2log/w2_distribution.json  (per-epoch W₂/C stats)

Server path: /root/drone_detection/
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules  # noqa: F401

import torch
from ultralytics_modules.nwd import (
    patch_sa_nwd_tal,
    bbox_to_gaussian, wasserstein_2d, bbox_area, sa_nwd
)

# ── Logging state ──────────────────────────────────────────────
_w2_log = []           # list of per-batch stats dicts
_epoch_stats = []      # list of per-epoch aggregated stats
_current_epoch = [0]   # mutable epoch counter

def _patched_sa_nwd_loss_with_logging(c_base=12.0, k=1.0, alpha=0.5, log_every=10):
    """Patch BboxLoss to use SA-NWD AND log W₂/C statistics."""
    try:
        from ultralytics.utils.loss import BboxLoss
        from ultralytics.utils.tal import bbox2dist
        from ultralytics.utils.metrics import bbox_iou

        def _patched_forward(self, pred_dist, pred_bboxes, anchor_points,
                             target_bboxes, target_scores, target_scores_sum,
                             fg_mask, imgsz, stride):
            """Hybrid SA-NWD + CIoU with W₂/C logging."""
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            pred_fg = pred_bboxes[fg_mask]
            target_fg = target_bboxes[fg_mask]

            eps = 1e-7
            mu1, sigma1 = bbox_to_gaussian(pred_fg, eps)
            mu2, sigma2 = bbox_to_gaussian(target_fg, eps)
            w2_sq = wasserstein_2d(mu1, sigma1, mu2, sigma2)
            w2 = torch.sqrt(w2_sq + eps)

            area1 = bbox_area(pred_fg, eps)
            area2 = bbox_area(target_fg, eps)
            avg_area = ((area1 + area2) / 2.0).clamp(min=1e-4)
            c_adapt = c_base * (1.0 + k / torch.sqrt(avg_area))

            x = w2 / c_adapt  # the working point

            # Log every log_every epochs
            ep = _current_epoch[0]
            if ep % log_every == 0 and len(pred_fg) > 0:
                x_np = x.detach().cpu().float().numpy()
                _w2_log.append({
                    'epoch': ep,
                    'x_mean': float(np.mean(x_np)),
                    'x_median': float(np.median(x_np)),
                    'x_p10': float(np.percentile(x_np, 10)),
                    'x_p90': float(np.percentile(x_np, 90)),
                    'x_min': float(np.min(x_np)),
                    'x_max': float(np.max(x_np)),
                    'n_pairs': int(len(x_np)),
                    'c_adapt_mean': float(c_adapt.detach().cpu().float().mean()),
                    'w2_mean': float(w2.detach().cpu().float().mean()),
                    'avg_area_mean': float(avg_area.detach().cpu().float().mean()),
                })

            score = torch.exp(-x)
            loss_sa = ((1.0 - score).unsqueeze(-1) * weight).sum() / target_scores_sum

            ciou = bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)
            loss_ciou = ((1.0 - ciou).unsqueeze(-1) * weight).sum() / target_scores_sum
            loss_iou = alpha * loss_sa + (1.0 - alpha) * loss_ciou

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
                )
                loss_dfl = loss_dfl.sum() / target_scores_sum

            return loss_iou, loss_dfl

        BboxLoss.forward = _patched_forward
        print(f'✓ SA-NWD+logging patch applied (c_base={c_base}, k={k}, alpha={alpha}, log_every={log_every})')
    except Exception as e:
        print(f'⚠ Logging patch failed: {e}')


# Apply patches
_patched_sa_nwd_loss_with_logging(c_base=12.0, k=1.0, alpha=0.5, log_every=10)
patch_sa_nwd_tal(c_base=12.0, k=1.0, nwd_min=0.0)
print('[E1] W₂/C working point logging: SA-NWD k=1.0, alpha=0.5')

from ultralytics import YOLO
from ultralytics.utils import callbacks as cb_module

EXP_NAME = 'nwd_p2_w2log'
ABLATION_PROJECT = '/root/drone_detection/runs/ablation'
WEIGHT_PATH = f'{ABLATION_PROJECT}/{EXP_NAME}/weights/best.pt'
NWD_P2_YAML = 'configs/ablation/ablation_nwd_p2.yaml'
LOG_PATH = Path(ABLATION_PROJECT) / EXP_NAME / 'w2_distribution.json'

TRAIN_ARGS = dict(
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
    seed=0,
)

# Register epoch counter callback
def on_train_epoch_end(trainer):
    ep = trainer.epoch
    _current_epoch[0] = ep
    # Aggregate stats collected during this epoch
    epoch_batches = [d for d in _w2_log if d['epoch'] == ep]
    if epoch_batches:
        _epoch_stats.append({
            'epoch': ep,
            'x_mean': float(np.mean([d['x_mean'] for d in epoch_batches])),
            'x_median': float(np.mean([d['x_median'] for d in epoch_batches])),
            'x_p10': float(np.mean([d['x_p10'] for d in epoch_batches])),
            'x_p90': float(np.mean([d['x_p90'] for d in epoch_batches])),
            'n_batches': len(epoch_batches),
        })
        # Incremental save every 10 epochs
        if ep % 10 == 0:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_PATH, 'w') as f:
                json.dump(_epoch_stats, f, indent=2)
            print(f'[E1] ep={ep} x_mean={_epoch_stats[-1]["x_mean"]:.4f} '
                  f'x_p10={_epoch_stats[-1]["x_p10"]:.4f} '
                  f'x_p90={_epoch_stats[-1]["x_p90"]:.4f}')


if Path(WEIGHT_PATH).exists():
    print(f'[SKIP] {WEIGHT_PATH} already exists, running val only')
    model = YOLO(WEIGHT_PATH)
    metrics = model.val(data='configs/data.yaml', imgsz=1280, split='val')
else:
    model = YOLO(NWD_P2_YAML)
    model.add_callback('on_train_epoch_end', on_train_epoch_end)
    model.train(
        pretrained='yolo11n.pt',
        project=ABLATION_PROJECT,
        name=EXP_NAME,
        exist_ok=True,
        **TRAIN_ARGS,
    )
    metrics = model.val(data='configs/data.yaml', imgsz=1280, split='val')

# Final save of epoch stats
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(LOG_PATH, 'w') as f:
    json.dump(_epoch_stats, f, indent=2)
print(f'[E1] W₂/C distribution saved to {LOG_PATH} ({len(_epoch_stats)} epochs logged)')

# Print summary
if _epoch_stats:
    mid = _epoch_stats[len(_epoch_stats) // 2]
    last = _epoch_stats[-1]
    print(f'[E1] Midpoint (ep {mid["epoch"]}): x_mean={mid["x_mean"]:.4f}')
    print(f'[E1] Final    (ep {last["epoch"]}): x_mean={last["x_mean"]:.4f}, '
          f'x_p10={last["x_p10"]:.4f}, x_p90={last["x_p90"]:.4f}')
    # Determine narrative angle
    x_final = last["x_mean"]
    if 0.3 <= x_final <= 1.5:
        print('[E1] NARRATIVE: working-point equalization (x in [0.3, 1.5])')
    elif 0.01 <= x_final < 0.3:
        print('[E1] NARRATIVE: training stability (x in [0.01, 0.3])')
    else:
        print(f'[E1] NARRATIVE: unexpected range (x={x_final:.4f}), rethink theory')

result = {
    'name': 'E1: W₂/C working point distribution logging',
    'exp': EXP_NAME,
    'config': 'nwd_p2 arch, k=1.0, alpha=0.5, +W₂/C logging',
    'map50':     round(float(metrics.box.map50), 4),
    'map':       round(float(metrics.box.map), 4),
    'precision': round(float(metrics.box.mp), 4),
    'recall':    round(float(metrics.box.mr), 4),
    'w2_log_path': str(LOG_PATH),
    'n_epochs_logged': len(_epoch_stats),
}
result['f1'] = round(
    2 * result['precision'] * result['recall'] /
    (result['precision'] + result['recall'] + 1e-8), 4
)
print('RESULT: ' + json.dumps(result))

out_path = Path(ABLATION_PROJECT) / EXP_NAME / 'result.json'
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f'[E1] Result saved to {out_path}')
