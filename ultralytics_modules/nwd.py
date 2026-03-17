"""
NWD: Normalized Wasserstein Distance for Tiny Object Detection
Paper: A Normalized Gaussian Wasserstein Distance for Tiny Object Detection (ISPRS 2022)

Core idea: IoU is fundamentally flawed for small objects because:
  - A 2-pixel shift on a 10x10 box drops IoU by ~35%
  - A 2-pixel shift on a 100x100 box drops IoU by ~4%
  NWD models each bbox as a 2D Gaussian and computes Wasserstein distance,
  which is smooth, continuous, and scale-insensitive.

Two integration points:
  1. patch_loss(): Replace CIoU loss with NWD-based loss
  2. patch_tal():  Replace IoU in label assignment with NWD (more positive samples for small objects)
"""

import torch
import math


def bbox_to_gaussian(bboxes, eps=1e-7):
    """Convert bboxes (xyxy) to 2D Gaussian parameters.

    Each bbox is modeled as N(cx, cy, w/2C, h/2C) where C is a constant.
    We use C=6 so that 3-sigma covers the bbox extent (99.7% of the distribution).

    Args:
        bboxes: (N, 4) tensor in xyxy format
    Returns:
        mu: (N, 2) centers
        sigma: (N, 2) standard deviations
    """
    x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1).clamp(min=eps)
    h = (y2 - y1).clamp(min=eps)
    # C=6: 3-sigma coverage
    sigma_x = w / 6.0
    sigma_y = h / 6.0
    mu = torch.stack([cx, cy], dim=-1)
    sigma = torch.stack([sigma_x, sigma_y], dim=-1)
    return mu, sigma


def wasserstein_2d(mu1, sigma1, mu2, sigma2):
    """Compute squared 2D Wasserstein distance between two sets of Gaussians.

    For diagonal covariance Gaussians, the closed-form is:
      W^2 = ||mu1 - mu2||^2 + ||sigma1 - sigma2||^2

    Args:
        mu1, mu2: (N, 2) center coordinates
        sigma1, sigma2: (N, 2) standard deviations
    Returns:
        w2: (N,) squared Wasserstein distances
    """
    return ((mu1 - mu2) ** 2).sum(dim=-1) + ((sigma1 - sigma2) ** 2).sum(dim=-1)


def nwd(box1, box2, eps=1e-7, constant=12.0):
    """Compute Normalized Wasserstein Distance between two sets of bboxes.

    Args:
        box1, box2: (N, 4) tensors in xyxy format
        constant: Normalization constant C. Controls sensitivity.
                  Smaller C = more sensitive to positional differences.
                  Recommended: 12.0 for general, 8.0 for very tiny objects.
    Returns:
        nwd_score: (N,) tensor in [0, 1], higher = more similar (like IoU)
    """
    mu1, sigma1 = bbox_to_gaussian(box1, eps)
    mu2, sigma2 = bbox_to_gaussian(box2, eps)
    w2 = wasserstein_2d(mu1, sigma1, mu2, sigma2)
    # Normalize: NWD = exp(-sqrt(W^2) / C)
    nwd_score = torch.exp(-torch.sqrt(w2 + eps) / constant)
    return nwd_score


def nwd_loss(pred_bboxes, target_bboxes, weight, target_scores_sum,
             constant=12.0, eps=1e-7):
    """NWD-based bounding box regression loss.

    Replaces CIoU loss. Returns scalar loss value.

    Args:
        pred_bboxes: (M, 4) predicted bboxes (xyxy, fg_mask applied)
        target_bboxes: (M, 4) target bboxes (xyxy, fg_mask applied)
        weight: (M, 1) per-sample weights from target scores
        target_scores_sum: scalar normalization factor
        constant: NWD normalization constant
    """
    nwd_score = nwd(pred_bboxes, target_bboxes, eps=eps, constant=constant)
    loss = ((1.0 - nwd_score).unsqueeze(-1) * weight).sum() / target_scores_sum
    return loss


def patch_nwd_loss(constant=12.0):
    """Monkey-patch ultralytics BboxLoss to use NWD instead of CIoU.

    Call this BEFORE model.train().
    """
    try:
        from ultralytics.utils.loss import BboxLoss
        from ultralytics.utils.tal import bbox2dist

        _orig_forward = BboxLoss.forward

        def _patched_forward(self, pred_dist, pred_bboxes, anchor_points,
                             target_bboxes, target_scores, target_scores_sum,
                             fg_mask, imgsz, stride):
            """NWD-based bbox loss (replaces CIoU)."""
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

            # NWD loss instead of CIoU
            loss_iou = nwd_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                                weight, target_scores_sum, constant=constant)

            # DFL loss (keep original ultralytics logic)
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
        print(f"\u2713 NWD loss patch applied (constant={constant})")
    except Exception as e:
        print(f"\u26a0 NWD loss patch failed: {e}, falling back to default CIoU")


def patch_nwd_tal(constant=12.0):
    """Monkey-patch ultralytics TaskAlignedAssigner to use NWD instead of IoU.

    This is the HIGHEST IMPACT change for small object detection:
    standard IoU-based assignment gives few positive samples to small objects,
    because their IoU is inherently low. NWD-based assignment treats small
    objects fairly, providing more training signal.

    Call this BEFORE model.train().
    """
    try:
        from ultralytics.utils.tal import TaskAlignedAssigner, bbox_iou

        _orig_get_box_metrics = TaskAlignedAssigner.get_box_metrics

        def _patched_get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
            """Replace IoU with NWD in label assignment metrics."""
            na = pd_bboxes.shape[-2]
            mask_gt = mask_gt.bool()
            overlaps = torch.zeros(
                [self.bs, self.n_max_boxes, na],
                dtype=pd_bboxes.dtype, device=pd_bboxes.device
            )
            bbox_scores = torch.zeros(
                [self.bs, self.n_max_boxes, na],
                dtype=pd_scores.dtype, device=pd_scores.device
            )

            ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
            ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
            ind[1] = gt_labels.long().squeeze(-1)
            bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

            # ★ Key change: use NWD instead of IoU for overlap computation
            pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
            gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
            overlaps[mask_gt] = nwd(pd_boxes, gt_boxes, constant=constant)

            align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
            return align_metric, overlaps

        TaskAlignedAssigner.get_box_metrics = _patched_get_box_metrics
        print(f"\u2713 NWD-TAL patch applied (constant={constant})")
    except Exception as e:
        print(f"\u26a0 NWD-TAL patch failed: {e}, falling back to default IoU-TAL")


def patch_all_nwd(loss_constant=12.0, tal_constant=12.0):
    """Apply both NWD loss and NWD-TAL patches.

    Call this once before model.train().
    """
    patch_nwd_loss(constant=loss_constant)
    patch_nwd_tal(constant=tal_constant)
