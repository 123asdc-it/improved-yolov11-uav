"""
SA-NWD: Scale-Adaptive Normalized Wasserstein Distance for Small Object Detection

Extends NWD (ISPRS 2022) with two key innovations:

1. Scale-Adaptive constant:
     C_adapt = C_base × (1 + k / √S̄)
   where S̄ is the mean area of two boxes. Small objects get larger C (more tolerant
   to positional offset), large objects use near-standard C.

2. Wasserstein-consistent detection framework:
   - TAL:  SA-NWD replaces IoU in label assignment → more positive samples for small objects
   - Loss: SA-NWD replaces CIoU in bbox regression → smoother gradients for small objects
   - NMS:  IoU + NWD hybrid → catches near-duplicate predictions that IoU misses

References:
  - NWD: Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection", ISPRS 2022
  - SA-NWD: Our contribution (scale-adaptive extension)
"""

import torch
import torchvision


# ============================================================
# Core: Gaussian modeling and Wasserstein distance
# ============================================================

def bbox_to_gaussian(bboxes, eps=1e-7):
    """Convert bboxes (xyxy) to 2D Gaussian parameters.

    Each bbox is modeled as N(cx, cy, w/6, h/6) where 6 = 2×3-sigma
    so that 99.7% of the distribution falls within the bbox.

    Args:
        bboxes: (..., 4) tensor in xyxy format
    Returns:
        mu: (..., 2) centers
        sigma: (..., 2) standard deviations
    """
    x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1).clamp(min=eps)
    h = (y2 - y1).clamp(min=eps)
    sigma_x = w / 6.0
    sigma_y = h / 6.0
    mu = torch.stack([cx, cy], dim=-1)
    sigma = torch.stack([sigma_x, sigma_y], dim=-1)
    return mu, sigma


def wasserstein_2d(mu1, sigma1, mu2, sigma2):
    """Squared 2D Wasserstein distance between diagonal-covariance Gaussians.

    W²(A, B) = ||μ₁ - μ₂||² + ||σ₁ - σ₂||²
    """
    return ((mu1 - mu2) ** 2).sum(dim=-1) + ((sigma1 - sigma2) ** 2).sum(dim=-1)


def bbox_area(bboxes, eps=1e-7):
    """Compute area of xyxy bboxes. Returns (...,) tensor."""
    w = (bboxes[..., 2] - bboxes[..., 0]).clamp(min=eps)
    h = (bboxes[..., 3] - bboxes[..., 1]).clamp(min=eps)
    return w * h


# ============================================================
# SA-NWD: Scale-Adaptive Normalized Wasserstein Distance
# ============================================================

def sa_nwd(box1, box2, c_base=12.0, k=2.0, eps=1e-7):
    """Scale-Adaptive Normalized Wasserstein Distance (SA-NWD).

    Core innovation: the normalization constant C adapts to object scale.
      C_adapt = c_base × (1 + k / √S̄)
    where S̄ = mean area of box1 and box2.

    Small objects (small S̄) → large C → more tolerant to offset → stabler training
    Large objects (large S̄) → C ≈ c_base → standard sensitivity

    Args:
        box1, box2: (..., 4) tensors in xyxy format
        c_base: Base normalization constant (default 12.0)
        k: Scale adaptation factor (default 2.0, range 1.0-3.0)
    Returns:
        score: (...,) tensor in [0, 1], higher = more similar
    """
    mu1, sigma1 = bbox_to_gaussian(box1, eps)
    mu2, sigma2 = bbox_to_gaussian(box2, eps)
    w2 = wasserstein_2d(mu1, sigma1, mu2, sigma2)

    # Scale-adaptive constant
    area1 = bbox_area(box1, eps)
    area2 = bbox_area(box2, eps)
    avg_area = (area1 + area2) / 2.0
    c_adapt = c_base * (1.0 + k / torch.sqrt(avg_area + eps))

    score = torch.exp(-torch.sqrt(w2 + eps) / c_adapt)
    return score


def nwd(box1, box2, eps=1e-7, constant=12.0):
    """Standard NWD (fixed constant). Kept for backward compatibility.

    Use sa_nwd() for the scale-adaptive version.
    """
    mu1, sigma1 = bbox_to_gaussian(box1, eps)
    mu2, sigma2 = bbox_to_gaussian(box2, eps)
    w2 = wasserstein_2d(mu1, sigma1, mu2, sigma2)
    return torch.exp(-torch.sqrt(w2 + eps) / constant)


# ============================================================
# SA-NWD Loss
# ============================================================

def sa_nwd_loss(pred_bboxes, target_bboxes, weight, target_scores_sum,
                c_base=12.0, k=2.0, eps=1e-7):
    """SA-NWD-based bounding box regression loss.

    Args:
        pred_bboxes: (M, 4) predicted bboxes (xyxy, fg_mask applied)
        target_bboxes: (M, 4) target bboxes (xyxy, fg_mask applied)
        weight: (M, 1) per-sample weights
        target_scores_sum: scalar normalization factor
    """
    score = sa_nwd(pred_bboxes, target_bboxes, c_base=c_base, k=k, eps=eps)
    loss = ((1.0 - score).unsqueeze(-1) * weight).sum() / target_scores_sum
    return loss


# ============================================================
# NWD-NMS: Hybrid IoU + NWD Non-Maximum Suppression
# ============================================================

def nwd_nms(boxes, scores, iou_threshold=0.7, nwd_threshold=0.8,
            c_base=12.0, k=2.0):
    """Hybrid NMS using both IoU and SA-NWD.

    Logic:
      1. Standard IoU-NMS first (removes high-IoU duplicates)
      2. Among survivors, further suppress if SA-NWD > nwd_threshold
         (catches near-duplicate small-object predictions that IoU misses)

    Args:
        boxes: (N, 4) tensor in xyxy format
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for standard NMS
        nwd_threshold: SA-NWD threshold for additional suppression
    Returns:
        keep: indices of kept boxes
    """
    # Step 1: Standard IoU NMS
    keep_iou = torchvision.ops.nms(boxes, scores, iou_threshold)
    if len(keep_iou) <= 1:
        return keep_iou

    # Step 2: NWD-based suppression among IoU-NMS survivors
    kept_boxes = boxes[keep_iou]
    kept_scores = scores[keep_iou]
    n = len(keep_iou)

    # Sort by score (descending) - already sorted by torchvision.ops.nms
    suppressed = torch.zeros(n, dtype=torch.bool, device=boxes.device)
    final_keep = []

    for i in range(n):
        if suppressed[i]:
            continue
        final_keep.append(keep_iou[i])

        if i + 1 < n:
            # Compute SA-NWD between current box and all remaining
            remaining_mask = ~suppressed & (torch.arange(n, device=boxes.device) > i)
            if remaining_mask.any():
                curr_box = kept_boxes[i].unsqueeze(0).expand(remaining_mask.sum(), -1)
                rem_boxes = kept_boxes[remaining_mask]
                nwd_scores = sa_nwd(curr_box, rem_boxes, c_base=c_base, k=k)
                # Suppress high-NWD boxes (near-duplicates)
                suppress_idx = torch.where(remaining_mask)[0][nwd_scores > nwd_threshold]
                suppressed[suppress_idx] = True

    return torch.tensor(final_keep, dtype=torch.long, device=boxes.device)


# ============================================================
# Monkey patches for ultralytics integration
# ============================================================

def patch_sa_nwd_loss(c_base=12.0, k=2.0, alpha=0.5):
    """Monkey-patch ultralytics BboxLoss to use hybrid SA-NWD + CIoU loss.

    Hybrid loss = alpha * SA-NWD + (1-alpha) * CIoU
    - SA-NWD provides scale-adaptive, smooth gradients for small objects
    - CIoU preserves explicit center distance + aspect ratio constraints
    - Together they are more robust than either alone

    Args:
        c_base: SA-NWD base constant
        k: SA-NWD scale adaptation factor (0 = standard NWD)
        alpha: Blending weight. 0.5 = equal blend. 1.0 = pure SA-NWD.
    """
    try:
        from ultralytics.utils.loss import BboxLoss
        from ultralytics.utils.tal import bbox2dist
        from ultralytics.utils.metrics import bbox_iou

        def _patched_forward(self, pred_dist, pred_bboxes, anchor_points,
                             target_bboxes, target_scores, target_scores_sum,
                             fg_mask, imgsz, stride):
            """Hybrid SA-NWD + CIoU bbox loss."""
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            pred_fg = pred_bboxes[fg_mask]
            target_fg = target_bboxes[fg_mask]

            # Component 1: SA-NWD loss (scale-adaptive, smooth for small objects)
            sa_score = sa_nwd(pred_fg, target_fg, c_base=c_base, k=k)
            loss_sa = ((1.0 - sa_score).unsqueeze(-1) * weight).sum() / target_scores_sum

            # Component 2: CIoU loss (precise geometric constraints)
            ciou = bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)
            loss_ciou = ((1.0 - ciou).unsqueeze(-1) * weight).sum() / target_scores_sum

            # Hybrid: alpha * SA-NWD + (1-alpha) * CIoU
            loss_iou = alpha * loss_sa + (1.0 - alpha) * loss_ciou

            # DFL loss (unchanged from ultralytics)
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
        print(f"\u2713 Hybrid SA-NWD+CIoU loss patch applied (c_base={c_base}, k={k}, alpha={alpha})")
    except Exception as e:
        print(f"\u26a0 SA-NWD loss patch failed: {e}")


def patch_sa_nwd_tal(c_base=12.0, k=1.0, nwd_min=0.3):
    """Monkey-patch ultralytics TaskAlignedAssigner to use SA-NWD.

    nwd_min controls the minimum SA-NWD score for a positive anchor.
    Scores below nwd_min are zeroed out, preventing low-quality anchors
    from becoming positives (which caused precision collapse in earlier versions).

    Typical range: 0.2~0.4. Default 0.3 is stable for drone dataset.
    Lower values give more positive samples but risk false positives.

    Bug fix: previous version had nwd_min parameter but used hardcoded 0.01,
    which let almost all anchors pass and caused P to collapse to ~0.1.
    Now nwd_min is correctly applied.
    """
    try:
        from ultralytics.utils.tal import TaskAlignedAssigner

        def _patched_get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
            """SA-NWD-based label assignment."""
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

            # SA-NWD instead of IoU for label assignment
            pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
            gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
            nwd_scores = sa_nwd(pd_boxes, gt_boxes, c_base=c_base, k=k)
            # Apply nwd_min threshold: zero out low-quality matches
            # This is the key fix — previously hardcoded to 0.01 which let all anchors pass
            nwd_scores = nwd_scores * (nwd_scores >= nwd_min).float()
            overlaps[mask_gt] = nwd_scores

            align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
            return align_metric, overlaps

        TaskAlignedAssigner.get_box_metrics = _patched_get_box_metrics
        print(f"\u2713 SA-NWD-TAL patch applied (c_base={c_base}, k={k}, nwd_min={nwd_min})")
    except Exception as e:
        print(f"\u26a0 SA-NWD-TAL patch failed: {e}")


def patch_nwd_nms(iou_threshold=0.7, nwd_threshold=0.8, c_base=12.0, k=2.0):
    """Monkey-patch ultralytics NMS to use hybrid IoU+NWD NMS."""
    try:
        import ultralytics.utils.nms as nms_module
        _orig_nms = nms_module.non_max_suppression

        def _patched_nms(*args, **kwargs):
            """NMS with NWD hybrid suppression for small objects."""
            results = _orig_nms(*args, **kwargs)

            # Apply NWD suppression on each image's results
            for i, det in enumerate(results):
                if det is not None and len(det) > 1:
                    boxes = det[:, :4]
                    scores = det[:, 4]
                    keep = nwd_nms(boxes, scores,
                                   iou_threshold=iou_threshold,
                                   nwd_threshold=nwd_threshold,
                                   c_base=c_base, k=k)
                    results[i] = det[keep]

            return results

        nms_module.non_max_suppression = _patched_nms
        print(f"\u2713 NWD-NMS patch applied (iou={iou_threshold}, nwd={nwd_threshold})")
    except Exception as e:
        print(f"\u26a0 NWD-NMS patch failed: {e}")


# ============================================================
# Convenience: apply all patches at once
# ============================================================

def patch_all_nwd(c_base=12.0, k=1.0, alpha=0.5, use_sa=True, use_nwd_nms=False,
                  nms_iou_threshold=0.7, nms_nwd_threshold=0.8, nwd_min=0.3):
    """Apply SA-NWD patches to ultralytics.

    Args:
        c_base:  Base normalization constant for NWD (default 12.0)
        k:       Scale adaptation factor. 0 = standard NWD, 1.0 = SA-NWD (default).
        alpha:   Loss blending weight. 0.5 = hybrid SA-NWD+CIoU (default).
                 1.0 = pure SA-NWD. 0.0 = pure CIoU.
        use_sa:  If True (default), use SA-NWD; if False, use standard NWD (k ignored)
        use_nwd_nms: If True, also patch NMS with hybrid IoU+NWD (default False)
        nms_iou_threshold: IoU threshold for NMS
        nms_nwd_threshold: NWD threshold for additional NMS suppression
        nwd_min: Minimum SA-NWD score for TAL positive anchors (default 0.3).
                 Values below this are zeroed out. Prevents precision collapse.
    """
    if use_sa:
        patch_sa_nwd_loss(c_base=c_base, k=k, alpha=alpha)
        patch_sa_nwd_tal(c_base=c_base, k=k, nwd_min=nwd_min)
    else:
        # Fall back to standard NWD (fixed constant)
        patch_sa_nwd_loss(c_base=c_base, k=0.0, alpha=alpha)
        patch_sa_nwd_tal(c_base=c_base, k=0.0, nwd_min=nwd_min)

    if use_nwd_nms:
        patch_nwd_nms(iou_threshold=nms_iou_threshold,
                      nwd_threshold=nms_nwd_threshold,
                      c_base=c_base, k=k)


# ============================================================
# Legacy: standard NWD (fixed constant) — kept for compatibility
# ============================================================

def nwd_loss(pred_bboxes, target_bboxes, weight, target_scores_sum,
             constant=12.0, eps=1e-7):
    """Standard NWD loss (fixed constant). Use sa_nwd_loss() for SA-NWD."""
    score = nwd(pred_bboxes, target_bboxes, eps=eps, constant=constant)
    loss = ((1.0 - score).unsqueeze(-1) * weight).sum() / target_scores_sum
    return loss


def patch_nwd_loss(constant=12.0, alpha=1.0):
    """Patch BboxLoss with standard NWD (fixed constant, pure NWD loss)."""
    patch_sa_nwd_loss(c_base=constant, k=0.0, alpha=alpha)


def patch_nwd_tal(constant=12.0):
    """Patch TAL with standard NWD (fixed constant)."""
    patch_sa_nwd_tal(c_base=constant, k=0.0)


# ============================================================
# Scale-Aware Loss Weighting (Fisher-Guided)
# ============================================================

def patch_scale_aware_loss(ref_area=0.002, max_scale=1.3, min_scale=0.8):
    """Patch BboxLoss with Fisher-Guided Scale-Aware CIoU loss.

    Directly derived from Fisher information analysis:
      I_IoU(s) proportional to 1/s^2  =>  gradient signal vanishes for small objects

    Compensation: w(s) = sqrt(ref_area / s)
    Applied as a direct coefficient on the per-sample loss, NOT multiplied on top
    of target_scores (which would cause double amplification).
    No batch normalization (which was incorrectly cancelling the compensation).

    For drone dataset (median area ~0.0023):
      w(0.0023) = sqrt(0.002/0.0023) ≈ 0.93  (near-average object, slight reduction)
      w(0.0005) = sqrt(0.002/0.0005) = 2.0 -> clamped to 1.3  (very small object)
      w(0.010)  = sqrt(0.002/0.010)  ≈ 0.45 -> clamped to 0.8  (relatively large)

    Args:
        ref_area: Reference normalized object area. Default 0.002.
        max_scale: Upper clamp. 1.3 = max 30% amplification for tiny objects.
        min_scale: Lower clamp. 0.8 = max 20% reduction for larger objects.
    """
    try:
        from ultralytics.utils.loss import BboxLoss
        from ultralytics.utils.tal import bbox2dist
        from ultralytics.utils.metrics import bbox_iou

        def _patched_forward(self, pred_dist, pred_bboxes, anchor_points,
                             target_bboxes, target_scores, target_scores_sum,
                             fg_mask, imgsz, stride):
            """Fisher-Guided Scale-Aware CIoU loss."""
            # Standard target_scores weight — unchanged, no double amplification
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

            tb = target_bboxes[fg_mask]
            img_area = float(imgsz[0]) * float(imgsz[1]) + 1e-8
            gt_areas = ((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])) / img_area

            # Fisher compensation: w(s) = sqrt(ref_area / s), no batch-normalize
            scale_w = torch.sqrt(
                torch.tensor(ref_area, dtype=tb.dtype, device=tb.device) /
                gt_areas.clamp(min=1e-6)
            ).clamp(min=min_scale, max=max_scale).unsqueeze(-1)

            # CIoU loss: scale_w applied directly on (1-ciou), independent of weight
            ciou = bbox_iou(pred_bboxes[fg_mask], tb, xywh=False, CIoU=True)
            loss_iou = ((1.0 - ciou).unsqueeze(-1) * weight * scale_w).sum() / target_scores_sum

            # DFL loss: same scale_w applied for consistency
            if self.dfl_loss:
                target_ltrb = bbox2dist(anchor_points, target_bboxes,
                                        self.dfl_loss.reg_max - 1)
                loss_dfl = (
                    self.dfl_loss(
                        pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                        target_ltrb[fg_mask]
                    ) * weight * scale_w
                ).sum() / target_scores_sum
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
                              reduction='none').mean(-1, keepdim=True) * weight * scale_w
                ).sum() / target_scores_sum

            return loss_iou, loss_dfl

        BboxLoss.forward = _patched_forward
        print(f"\u2713 Fisher Scale-Aware CIoU applied (ref_area={ref_area}, scale=[{min_scale},{max_scale}])")
    except Exception as e:
        print(f"\u26a0 Scale-Aware loss patch failed: {e}")


def patch_sa_nwd_fisher_loss(c_base=12.0, k=1.0, alpha=0.5,
                              ref_area=0.002, max_scale=1.3, min_scale=0.8):
    """Patch BboxLoss with SA-NWD + Fisher-Guided Scale-Aware CIoU (combined).

    loss = alpha * SA-NWD  +  (1-alpha) * scale_aware_CIoU

    SA-NWD:           smooth Wasserstein-based regression, scale-adaptive C
    scale_aware_CIoU: standard CIoU amplified by w(s)=sqrt(ref_area/s)
                      — Fisher compensation for vanishing gradients on small objects

    This is the full combined contribution: two complementary improvements.
    scale_w has NO batch normalization and does NOT double-amplify target_scores.

    Args:
        c_base:     SA-NWD base constant (default 12.0)
        k:          SA-NWD scale factor (default 1.0)
        alpha:      SA-NWD blend weight (default 0.5)
        ref_area:   Fisher reference area (default 0.002)
        max_scale:  Fisher upper clamp (default 1.3)
        min_scale:  Fisher lower clamp (default 0.8)
    """
    try:
        from ultralytics.utils.loss import BboxLoss
        from ultralytics.utils.tal import bbox2dist
        from ultralytics.utils.metrics import bbox_iou

        def _patched_forward(self, pred_dist, pred_bboxes, anchor_points,
                             target_bboxes, target_scores, target_scores_sum,
                             fg_mask, imgsz, stride):
            """SA-NWD + Fisher Scale-Aware CIoU combined loss."""
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

            # Component 1: SA-NWD loss (scale-adaptive Wasserstein)
            sa_score = sa_nwd(pred_fg, tb, c_base=c_base, k=k)
            loss_sa = ((1.0 - sa_score).unsqueeze(-1) * weight).sum() / target_scores_sum

            # Component 2: Fisher-compensated CIoU loss
            ciou = bbox_iou(pred_fg, tb, xywh=False, CIoU=True)
            loss_ciou = ((1.0 - ciou).unsqueeze(-1) * weight * scale_w).sum() / target_scores_sum

            # Combined: alpha * SA-NWD + (1-alpha) * Fisher-CIoU
            loss_iou = alpha * loss_sa + (1.0 - alpha) * loss_ciou

            # DFL loss: Fisher scale_w applied
            if self.dfl_loss:
                target_ltrb = bbox2dist(anchor_points, target_bboxes,
                                        self.dfl_loss.reg_max - 1)
                loss_dfl = (
                    self.dfl_loss(
                        pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                        target_ltrb[fg_mask]
                    ) * weight * scale_w
                ).sum() / target_scores_sum
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
                              reduction='none').mean(-1, keepdim=True) * weight * scale_w
                ).sum() / target_scores_sum

            return loss_iou, loss_dfl

        BboxLoss.forward = _patched_forward
        print(f"\u2713 SA-NWD + Fisher CIoU loss applied "
              f"(c_base={c_base}, k={k}, alpha={alpha}, "
              f"ref_area={ref_area}, scale=[{min_scale},{max_scale}])")
    except Exception as e:
        print(f"\u26a0 SA-NWD+Fisher loss patch failed: {e}")
