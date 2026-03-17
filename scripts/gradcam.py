"""
gradcam.py — Grad-CAM visualization for paper figures

生成 Baseline vs Ours 的热力图对比，用于论文 Figure (Grad-CAM Visualization)

依赖：
    pip install grad-cam  # pytorch-grad-cam

用法：
    python scripts/gradcam.py \
        --baseline runs/.../baseline/weights/best.pt \
        --ours     runs/.../two_stage_s2/weights/best.pt \
        --images   datasets/images/val \
        --output   paper/figs/gradcam.pdf
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def load_model(weights_path):
    """Load YOLO model."""
    try:
        import register_modules  # noqa
    except ImportError:
        pass
    from ultralytics import YOLO
    return YOLO(weights_path)


def get_target_layer(model):
    """Get last backbone conv layer for Grad-CAM target."""
    # For YOLOv11, the last C3k2 block before SPPF is a good target
    m = model.model
    # Walk the model to find a suitable conv layer
    target = None
    for name, module in m.named_modules():
        if hasattr(module, 'cv2') or 'c3k2' in name.lower() or 'c2psa' in name.lower():
            target = module
    if target is None:
        # Fallback: last Conv with weight
        for module in m.modules():
            if isinstance(module, torch.nn.Conv2d) and module.weight.shape[0] > 32:
                target = module
    return target


def compute_gradcam(model_yolo, img_path, target_layer, imgsz=640):
    """Compute Grad-CAM heatmap for one image."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("  [WARN] pytorch-grad-cam not installed. pip install grad-cam")
        return None, None

    import torchvision.transforms.functional as TF
    from PIL import Image

    device = next(model_yolo.model.parameters()).device
    m = model_yolo.model
    m.eval()

    # Load and preprocess image
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((imgsz, imgsz))
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)
    img_float = np.array(img_pil) / 255.0

    # Target: maximize sum of all activations (unsupervised Grad-CAM)
    class SumTarget:
        def __call__(self, output):
            if isinstance(output, (list, tuple)):
                return output[0].sum()
            return output.sum()

    try:
        cam = GradCAM(model=m, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=img_tensor, targets=[SumTarget()])
        grayscale_cam = grayscale_cam[0]
        visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        return img_rgb, visualization
    except Exception as e:
        print(f"  [WARN] Grad-CAM failed: {e}")
        return img_rgb, None


def select_images(img_dir, n=4):
    """Select representative images from val set."""
    img_dir = Path(img_dir)
    imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if len(imgs) <= n:
        return imgs
    # Spread selection across the set
    step = len(imgs) // n
    return [imgs[i * step] for i in range(n)]


def plot_gradcam_comparison(baseline_path, ours_path, img_dir, output_path,
                            n_images=3, imgsz=640):
    """Generate side-by-side Grad-CAM comparison: Input | Baseline | Ours."""
    print("Loading models...")
    baseline_model = load_model(baseline_path)
    ours_model = load_model(ours_path)

    baseline_layer = get_target_layer(baseline_model)
    ours_layer = get_target_layer(ours_model)

    if baseline_layer is None or ours_layer is None:
        print("  [ERROR] Could not find target layer")
        return

    images = select_images(img_dir, n_images)
    if not images:
        print(f"  [ERROR] No images found in {img_dir}")
        return

    n_cols = 3  # Input | Baseline CAM | Ours CAM
    fig, axes = plt.subplots(n_images, n_cols,
                             figsize=(n_cols * 3, n_images * 3))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Baseline (YOLOv11n)", "Ours (SA-NWD)"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontweight="bold", fontsize=9)

    for i, img_path in enumerate(images):
        print(f"  Processing {img_path.name}...")
        img_rgb, baseline_cam = compute_gradcam(
            baseline_model, img_path, baseline_layer, imgsz)
        _, ours_cam = compute_gradcam(
            ours_model, img_path, ours_layer, imgsz)

        if img_rgb is None:
            continue

        img_show = cv2.resize(img_rgb, (imgsz, imgsz))
        axes[i, 0].imshow(img_show)
        axes[i, 0].axis("off")

        if baseline_cam is not None:
            axes[i, 1].imshow(baseline_cam)
        else:
            axes[i, 1].imshow(img_show)
            axes[i, 1].text(0.5, 0.5, "N/A", transform=axes[i, 1].transAxes,
                            ha="center", va="center")
        axes[i, 1].axis("off")

        if ours_cam is not None:
            axes[i, 2].imshow(ours_cam)
        else:
            axes[i, 2].imshow(img_show)
            axes[i, 2].text(0.5, 0.5, "N/A", transform=axes[i, 2].transAxes,
                            ha="center", va="center")
        axes[i, 2].axis("off")

    plt.tight_layout(pad=0.5)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM comparison")
    parser.add_argument("--baseline", type=str, required=True,
                        help="Path to baseline best.pt")
    parser.add_argument("--ours", type=str, required=True,
                        help="Path to our model best.pt")
    parser.add_argument("--images", type=str, default="datasets/images/val",
                        help="Validation images directory")
    parser.add_argument("--output", type=str, default="paper/figs/gradcam.pdf")
    parser.add_argument("--n", type=int, default=3, help="Number of images")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    plot_gradcam_comparison(
        baseline_path=args.baseline,
        ours_path=args.ours,
        img_dir=args.images,
        output_path=args.output,
        n_images=args.n,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
