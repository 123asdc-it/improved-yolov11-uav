"""
Inner-IoU Loss (2023)
Paper: Inner-IoU: More Effective Intersection over Union Loss with Auxiliary Bounding Box

For small objects, standard IoU suffers from sparse gradients when boxes
hardly overlap. Inner-IoU computes IoU on scaled-down auxiliary boxes,
providing denser gradients for small target regression.

Integration: monkey-patch ultralytics loss to replace CIoU with Inner-CIoU.
"""

import torch
import math


def inner_iou(box1, box2, ratio=0.7, eps=1e-7, xywh=True):
    """
    Compute Inner-IoU between box1 and box2.

    Args:
        box1, box2: (N,4) tensors in xywh or xyxy format.
        ratio: Scale ratio for inner (auxiliary) boxes. Smaller -> denser gradients.
               Recommended 0.5-0.8 for small objects.
        xywh: If True, input is (cx,cy,w,h); else (x1,y1,x2,y2).
    Returns:
        inner_iou: (N,) tensor
    """
    if xywh:
        # Convert to xyxy
        b1x1, b1x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1y1, b1y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2x1, b2x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2y1, b2y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        b1x1, b1y1, b1x2, b1y2 = box1[...,0], box1[...,1], box1[...,2], box1[...,3]
        b2x1, b2y1, b2x2, b2y2 = box2[...,0], box2[...,1], box2[...,2], box2[...,3]

    # Inner (auxiliary) boxes — scaled down around center
    b1_cx = (b1x1 + b1x2) / 2
    b1_cy = (b1y1 + b1y2) / 2
    b1_w  = (b1x2 - b1x1) * ratio
    b1_h  = (b1y2 - b1y1) * ratio
    ib1x1, ib1x2 = b1_cx - b1_w/2, b1_cx + b1_w/2
    ib1y1, ib1y2 = b1_cy - b1_h/2, b1_cy + b1_h/2

    b2_cx = (b2x1 + b2x2) / 2
    b2_cy = (b2y1 + b2y2) / 2
    b2_w  = (b2x2 - b2x1) * ratio
    b2_h  = (b2y2 - b2y1) * ratio
    ib2x1, ib2x2 = b2_cx - b2_w/2, b2_cx + b2_w/2
    ib2y1, ib2y2 = b2_cy - b2_h/2, b2_cy + b2_h/2

    # Intersection of inner boxes
    inter_w = (torch.min(ib1x2, ib2x2) - torch.max(ib1x1, ib2x1)).clamp(0)
    inter_h = (torch.min(ib1y2, ib2y2) - torch.max(ib1y1, ib2y1)).clamp(0)
    inter = inter_w * inter_h

    # Union of inner boxes
    area1 = b1_w * b1_h
    area2 = b2_w * b2_h
    union = area1 + area2 - inter + eps

    return inter / union


def inner_ciou_loss(box1, box2, ratio=0.7, eps=1e-7):
    """
    Inner-CIoU loss: CIoU on original boxes + Inner-IoU term.

    Combines the complete geometric penalty of CIoU with the dense
    gradient property of Inner-IoU for small objects.
    """
    # ---- Standard CIoU components ----
    # box1, box2: (N, 4) in xywh
    b1x1 = box1[..., 0] - box1[..., 2] / 2
    b1x2 = box1[..., 0] + box1[..., 2] / 2
    b1y1 = box1[..., 1] - box1[..., 3] / 2
    b1y2 = box1[..., 1] + box1[..., 3] / 2
    b2x1 = box2[..., 0] - box2[..., 2] / 2
    b2x2 = box2[..., 0] + box2[..., 2] / 2
    b2y1 = box2[..., 1] - box2[..., 3] / 2
    b2y2 = box2[..., 1] + box2[..., 3] / 2

    inter_w = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0)
    inter_h = (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
    inter = inter_w * inter_h
    area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    area2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    union = area1 + area2 - inter + eps
    iou = inter / union

    # Enclosing box
    cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
    ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
    c2 = cw**2 + ch**2 + eps

    # Center distance
    rho2 = ((box2[..., 0] - box1[..., 0])**2 +
             (box2[..., 1] - box1[..., 1])**2)

    # Aspect ratio consistency
    v = (4 / math.pi**2) * (
        torch.atan(box2[..., 2] / (box2[..., 3] + eps)) -
        torch.atan(box1[..., 2] / (box1[..., 3] + eps))
    )**2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2 + alpha * v)

    # ---- Inner-IoU term ----
    iiou = inner_iou(box1, box2, ratio=ratio, eps=eps, xywh=True)

    # Loss = (1 - CIoU) combined with (1 - Inner-IoU)
    # Weight inner_iou equally — tunable if needed
    return 1 - (ciou + iiou) / 2


def patch_ultralytics_loss(ratio=0.7):
    """
    Monkey-patch ultralytics BboxLoss to use Inner-CIoU instead of CIoU.

    Call this after import register_modules and before model.train().
    """
    try:
        from ultralytics.utils.loss import BboxLoss
        import torch

        _orig_forward = BboxLoss.forward

        def _patched_forward(self, pred_dist, pred_bboxes, anchor_points,
                             target_bboxes, target_scores, target_scores_sum,
                             fg_mask, imgsz, stride):
            """Replace CIoU with Inner-CIoU in bbox loss."""
            from ultralytics.utils.metrics import bbox_iou
            from ultralytics.utils.tal import bbox2dist

            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

            # Inner-IoU replaces standard CIoU
            iiou = inner_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                             ratio=ratio, xywh=False)
            loss_iou = ((1.0 - iiou) * weight).sum() / target_scores_sum

            # DFL loss (keep original logic)
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
        print("✓ Inner-IoU loss patch applied (ratio=%.1f)" % ratio)
    except Exception as e:
        print(f"⚠ Inner-IoU patch failed: {e}, falling back to default CIoU")
