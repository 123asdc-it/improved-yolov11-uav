"""
run_asanwd_compare.py — Experiment E4: Asa-NWD Batch-Level vs SA-NWD Per-Object

Purpose:
  Compare our per-object adaptive C (SA-NWD) against the batch-level heuristic
  used by Asa-NWD (SMTrack, arXiv:2508.14607).

  Asa-NWD uses the mean area of ALL ground-truth boxes in the batch as the
  scale normalization, rather than individual per-box areas.

  SA-NWD:    C_adapt = C_base * (1 + k / sqrt(S_i))    per-object
  Asa-NWD:   C_adapt = C_base * (1 + k / sqrt(mean_S)) batch-level

If Asa-NWD ≈ SA-NWD:
  → Key distinction can be softened: "simpler design, no batch statistics"
If SA-NWD > Asa-NWD by Δ > 0.3%:
  → Key distinction is real and should be emphasized in the paper

Server path: /root/drone_detection/
Output:      runs/ablation/asa_nwd_batchlevel/
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules  # noqa: F401

import torch
from ultralytics_modules.nwd import patch_sa_nwd_tal, bbox_to_gaussian, wasserstein_2d, bbox_area

# ── Asa-NWD: batch-level adaptive C ────────────────────────────

def _patch_asanwd_loss(c_base=12.0, k=1.0, alpha=0.5):
    """Patch BboxLoss with batch-level adaptive C (Asa-NWD style).

    C_adapt = c_base * (1 + k / sqrt(mean_gt_area_in_batch))
    This is a batch-level heuristic, as opposed to SA-NWD's per-object C.
    """
    try:
        from ultralytics.utils.loss import BboxLoss
        from ultralytics.utils.tal import bbox2dist
        from ultralytics.utils.metrics import bbox_iou

        def _patched_forward(self, pred_dist, pred_bboxes, anchor_points,
                             target_bboxes, target_scores, target_scores_sum,
                             fg_mask, imgsz, stride):
            """Asa-NWD: batch-level adaptive C."""
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            pred_fg = pred_bboxes[fg_mask]
            target_fg = target_bboxes[fg_mask]

            eps = 1e-7
            mu1, sigma1 = bbox_to_gaussian(pred_fg, eps)
            mu2, sigma2 = bbox_to_gaussian(target_fg, eps)
            w2_sq = wasserstein_2d(mu1, sigma1, mu2, sigma2)
            w2 = torch.sqrt(w2_sq + eps)

            # Batch-level: use mean area of ALL gt boxes in this batch
            # (Asa-NWD uses entire batch GT statistics, not per-pair)
            all_areas = bbox_area(target_bboxes.reshape(-1, 4), eps)
            batch_mean_area = all_areas.mean().clamp(min=1e-4)
            # Single scalar C for all pairs in this batch
            c_adapt_scalar = c_base * (1.0 + k / torch.sqrt(batch_mean_area))

            score = torch.exp(-w2 / c_adapt_scalar)
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
        print(f'✓ Asa-NWD batch-level patch applied (c_base={c_base}, k={k}, alpha={alpha})')
    except Exception as e:
        print(f'⚠ Asa-NWD patch failed: {e}')


# Apply Asa-NWD (batch-level) loss + SA-NWD TAL (keep TAL as SA-NWD for fair comparison)
_patch_asanwd_loss(c_base=12.0, k=1.0, alpha=0.5)
patch_sa_nwd_tal(c_base=12.0, k=1.0, nwd_min=0.3)
print('[E4] Asa-NWD: batch-level C, SA-NWD TAL, k=1.0, alpha=0.5')
print('[E4] Architecture: nwd_p2 (same as best config)')

from ultralytics import YOLO

EXP_NAME = 'asa_nwd_batchlevel'
ABLATION_PROJECT = '/root/drone_detection/runs/ablation'
WEIGHT_PATH = f'{ABLATION_PROJECT}/{EXP_NAME}/weights/best.pt'
NWD_P2_YAML = 'configs/ablation/ablation_nwd_p2.yaml'

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

if Path(WEIGHT_PATH).exists():
    print(f'[SKIP] {WEIGHT_PATH} already exists, running val only')
    model = YOLO(WEIGHT_PATH)
    metrics = model.val(data='configs/data.yaml', imgsz=1280, split='val')
else:
    model = YOLO(NWD_P2_YAML)
    model.train(
        pretrained='yolo11n.pt',
        project=ABLATION_PROJECT,
        name=EXP_NAME,
        exist_ok=True,
        **TRAIN_ARGS,
    )
    metrics = model.val(data='configs/data.yaml', imgsz=1280, split='val')

result = {
    'name': 'E4: Asa-NWD batch-level C (vs SA-NWD per-object)',
    'exp': EXP_NAME,
    'config': 'nwd_p2 arch, batch-level C, k=1.0, alpha=0.5',
    'map50':     round(float(metrics.box.map50), 4),
    'map':       round(float(metrics.box.map), 4),
    'precision': round(float(metrics.box.mp), 4),
    'recall':    round(float(metrics.box.mr), 4),
}
result['f1'] = round(
    2 * result['precision'] * result['recall'] /
    (result['precision'] + result['recall'] + 1e-8), 4
)
sa_nwd_map50 = 0.9781  # nwd_p2 per-object SA-NWD
delta = sa_nwd_map50 - result['map50']
result['delta_vs_sanwd'] = round(delta, 4)
print('RESULT: ' + json.dumps(result))
print(f'[E4] SA-NWD per-object ({sa_nwd_map50}) vs Asa-NWD batch-level ({result["map50"]}): '
      f'Δ={delta:+.4f}')
if abs(delta) < 0.003:
    print('[E4] → Difference <0.3%: soften Key distinction claim → "simpler design"')
else:
    print('[E4] → Difference ≥0.3%: per-object advantage is real, keep Key distinction')

out_path = Path(ABLATION_PROJECT) / EXP_NAME / 'result.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f'[E4] Result saved to {out_path}')
