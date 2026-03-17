"""
verify_error_distribution.py

验证核心理论假设：sigma(s) = sigma_0 * s^(-b)，理论预期 b ≈ 0.5

用法：
    python scripts/verify_error_distribution.py
    python scripts/verify_error_distribution.py --weights /path/to/best.pt
    python scripts/verify_error_distribution.py --weights /path/to/best.pt --imgsz 1280
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import register_modules  # noqa: F401
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def box_iou_numpy(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) -> (N,M) IoU matrix, xyxy format."""
    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    a2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / np.maximum(a1[:, None] + a2[None, :] - inter, 1e-8)


def load_gt_boxes(label_path: str, img_w: int, img_h: int) -> np.ndarray:
    """Load YOLO format labels, return xyxy pixel coords."""
    boxes = []
    if not Path(label_path).exists():
        return np.zeros((0, 4), dtype=np.float32)
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts[:5])
            boxes.append([
                (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                (cx + w / 2) * img_w, (cy + h / 2) * img_h,
            ])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def collect_errors(
    weights_path: str,
    val_img_dir: str,
    imgsz: int = 1280,
    conf: float = 0.25,
    iou_thres: float = 0.45,
) -> tuple:
    """Run inference on val set, return (areas, center_errors) arrays."""
    from ultralytics import YOLO
    model = YOLO(weights_path)

    img_dir = Path(val_img_dir)
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"[INFO] Found {len(img_files)} validation images")

    areas, deltas = [], []
    for i, img_path in enumerate(img_files):
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(img_files)} ...")

        results = model(str(img_path), imgsz=imgsz, conf=conf, iou=iou_thres, verbose=False)
        r = results[0]
        img_h, img_w = r.orig_shape[:2]

        if r.boxes is not None and len(r.boxes) > 0:
            pred_xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        else:
            pred_xyxy = np.zeros((0, 4), dtype=np.float32)

        # Derive label path from image path
        lbl = str(img_path).replace("/images/", "/labels/")
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            lbl = lbl.replace(ext, ".txt")
        lbl = str(Path(lbl).with_suffix(".txt"))
        gt_xyxy = load_gt_boxes(lbl, img_w, img_h)

        if len(gt_xyxy) == 0 or len(pred_xyxy) == 0:
            continue

        iou_mat = box_iou_numpy(gt_xyxy, pred_xyxy)
        for gi in range(len(gt_xyxy)):
            j = int(np.argmax(iou_mat[gi]))
            if iou_mat[gi, j] < 0.1:
                continue
            gt_cx = (gt_xyxy[gi, 0] + gt_xyxy[gi, 2]) / 2
            gt_cy = (gt_xyxy[gi, 1] + gt_xyxy[gi, 3]) / 2
            pr_cx = (pred_xyxy[j, 0] + pred_xyxy[j, 2]) / 2
            pr_cy = (pred_xyxy[j, 1] + pred_xyxy[j, 3]) / 2
            delta = float(np.sqrt((pr_cx - gt_cx) ** 2 + (pr_cy - gt_cy) ** 2))
            area = float(max(
                (gt_xyxy[gi, 2] - gt_xyxy[gi, 0]) * (gt_xyxy[gi, 3] - gt_xyxy[gi, 1]),
                1.0
            ))
            areas.append(area)
            deltas.append(delta)

    return np.array(areas, dtype=np.float64), np.array(deltas, dtype=np.float64)


def fit_and_plot(
    areas: np.ndarray,
    deltas: np.ndarray,
    output_path: str,
    n_bins: int = 10,
) -> None:
    """Bin by area, compute sigma per bin, fit power law sigma = sigma0 * s^(-b), plot."""
    log_areas = np.log10(areas)
    edges = np.linspace(log_areas.min(), log_areas.max(), n_bins + 1)

    centers, sigmas, valid, bin_deltas_list = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (log_areas >= lo) & (log_areas <= (hi if i < n_bins - 1 else hi + 1e-9))
        d = deltas[mask]
        bin_deltas_list.append(d)
        c = 10 ** ((lo + hi) / 2)
        centers.append(c)
        if len(d) < 5:
            sigmas.append(np.nan)
            valid.append(False)
        else:
            sigmas.append(float(np.std(d, ddof=1)))
            valid.append(True)

    centers_arr = np.array(centers)
    sigmas_arr = np.array(sigmas)
    valid_arr = np.array(valid)

    vc = centers_arr[valid_arr]
    vs = sigmas_arr[valid_arr]

    if len(vc) < 2:
        print("[WARN] Not enough valid bins to fit. Consider lowering --conf.")
        return

    # OLS in log-log space: log(sigma) = log(sigma0) + (-b)*log(s)
    ls = np.log(vc)
    lsig = np.log(vs)
    A = np.column_stack([np.ones_like(ls), ls])
    coeffs, *_ = np.linalg.lstsq(A, lsig, rcond=None)
    log_sigma0, neg_b = coeffs
    b = -neg_b
    sigma0 = np.exp(log_sigma0)
    lsig_pred = A @ coeffs
    ss_res = np.sum((lsig - lsig_pred) ** 2)
    ss_tot = np.sum((lsig - lsig.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    print("\n" + "=" * 60)
    print("  Error Distribution Analysis")
    print("=" * 60)
    print(f"  Total matched pairs : {len(areas)}")
    print(f"  Valid bins          : {int(valid_arr.sum())} / {n_bins}")
    print(f"  Fitted b            : {b:.4f}  (theory: 0.5)")
    print(f"  Fitted sigma_0      : {sigma0:.4f}")
    print(f"  R squared           : {r2:.4f}")

    if abs(b - 0.5) < 0.15:
        conclusion = "PASS: assumption sigma proportional to 1/sqrt(s) is well-supported"
    elif abs(b - 0.5) < 0.25:
        conclusion = f"MARGINAL: use sigma proportional to s^(-{b:.2f}) in theory"
    else:
        conclusion = f"WARN: b={b:.2f} deviates from 0.5, revise C_adapt formula to use s^(-{b:.2f})"
    print(f"  Conclusion          : {conclusion}")
    print("=" * 60)

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Error Distribution: sigma(s) = {sigma0:.3f} * s^(-{b:.3f}),  R2={r2:.3f}",
        fontsize=12
    )

    # Panel 1: log-log scatter + fit line
    ax1.scatter(vc, vs, c="steelblue", s=80, zorder=5, label="Measured sigma per bin")
    s_fit = np.logspace(np.log10(centers_arr.min()), np.log10(centers_arr.max()), 200)
    ax1.plot(s_fit, sigma0 * s_fit ** (-b), "r-", lw=2,
             label=f"Fit: {sigma0:.2f}*s^(-{b:.2f})")
    ax1.plot(s_fit, sigma0 * s_fit ** (-0.5), "g--", lw=1.5, alpha=0.7,
             label="Theory: s^(-0.5)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("GT Box Area (pixels sq)")
    ax1.set_ylabel("Center Error Std Dev (pixels)")
    ax1.set_title("Log-Log: sigma vs area")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: per-bin error histograms
    colors = plt.cm.viridis(np.linspace(0, 1, n_bins))
    max_delta = float(np.percentile(deltas, 95))
    for i, (d, v) in enumerate(zip(bin_deltas_list, valid_arr)):
        if v and len(d) > 0:
            ax2.hist(d, bins=20, range=(0, max_delta), alpha=0.4,
                     color=colors[i], label=f"bin{i+1} (n={len(d)})")
    ax2.set_xlabel("Center Error (pixels)")
    ax2.set_ylabel("Count")
    ax2.set_title("Error Distribution per Scale Bin")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify error distribution hypothesis")
    parser.add_argument("--weights", type=str, default="",
                        help="Path to best.pt (auto-detected if empty)")
    parser.add_argument("--data", type=str, default="configs/data.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", type=str, default="outputs/error_distribution.png")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # Auto-detect weights
    weights = args.weights
    if not weights:
        candidates = [
            "/root/drone_detection/runs/ablation/baseline/weights/best.pt",
            str(PROJECT_ROOT / "runs" / "ablation" / "baseline" / "weights" / "best.pt"),
        ]
        for c in candidates:
            if Path(c).exists():
                weights = c
                break
    if not weights or not Path(weights).exists():
        raise FileNotFoundError(
            "No best.pt found. Pass --weights /path/to/best.pt"
        )
    print(f"[INFO] Using weights: {weights}")

    # Load data config
    import yaml
    with open(args.data) as f:
        cfg = yaml.safe_load(f)
    val_key = cfg.get("val", "datasets/images/val")
    val_dir = Path(val_key) if Path(val_key).is_absolute() else PROJECT_ROOT / val_key
    print(f"[INFO] Val dir: {val_dir}")

    areas, deltas = collect_errors(
        weights, str(val_dir), imgsz=args.imgsz, conf=args.conf
    )
    print(f"[INFO] Collected {len(areas)} matched pairs")

    if len(areas) < 10:
        raise RuntimeError(
            f"Only {len(areas)} matched pairs. Lower --conf or check data."
        )

    fit_and_plot(areas, deltas, args.output, n_bins=args.n_bins)


if __name__ == "__main__":
    main()
