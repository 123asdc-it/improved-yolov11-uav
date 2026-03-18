"""
SAHI-style Training Slice: 将训练集图像切成重叠小块，提升小目标检测效果。

核心思路：
  原图 1920x1080 + 50x50 目标 → 目标占 0.15% 面积（模型难学）
  640x640 切片 + 50x50 目标 → 目标占 0.6% 面积（提升 4 倍）
  391 张图 → ~2500 切片 → 数据量增加 6 倍

注意：
  - 只切训练集，val/test 保持原图（用 SAHI 推理评估）
  - 切片训练必须配合 SAHI 推理，否则 domain gap 会导致指标下降
  - 生成 configs/data_sliced.yaml，训练时 imgsz=640

用法：cd 项目根目录，然后 python scripts/slice_dataset.py
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# ============================================================
# 配置
# ============================================================
SLICE_SIZE = 640           # 切片尺寸
OVERLAP = 0.2              # 重叠率（20%）
MIN_OVERLAP_RATIO = 0.5    # 目标至少 50% 在切片内才保留
MAX_BG_RATIO = 0.2         # 空背景切片最多保留 20%（控制正负样本比例）

SRC_IMG_DIR = PROJECT_ROOT / "datasets" / "images" / "train"
SRC_LBL_DIR = PROJECT_ROOT / "datasets" / "labels" / "train"
DST_IMG_DIR = PROJECT_ROOT / "datasets_sliced" / "images" / "train"
DST_LBL_DIR = PROJECT_ROOT / "datasets_sliced" / "labels" / "train"


def compute_slice_coords(img_w, img_h, slice_size, overlap):
    """计算所有切片的左上角坐标。

    Returns:
        list of (x1, y1, x2, y2) in pixel coordinates
    """
    stride = int(slice_size * (1 - overlap))
    coords = []

    if img_w <= 0 or img_h <= 0:
        return coords

    y = 0
    y2 = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x2 = min(x + slice_size, img_w)
            y2 = min(y + slice_size, img_h)
            # 右/下边缘对齐，确保切片完整
            x1 = max(0, x2 - slice_size)
            y1 = max(0, y2 - slice_size)
            coords.append((x1, y1, x2, y2))
            if x2 == img_w:
                break
            x += stride
        if y2 == img_h:
            break
        y += stride

    return coords


def load_yolo_labels(label_path, img_w, img_h):
    """读取 YOLO 格式标注，返回像素坐标 [cls, x1, y1, x2, y2]。"""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            px1 = (cx - w / 2) * img_w
            py1 = (cy - h / 2) * img_h
            px2 = (cx + w / 2) * img_w
            py2 = (cy + h / 2) * img_h
            boxes.append([cls, px1, py1, px2, py2])
    return boxes


def clip_boxes_to_slice(boxes, sx1, sy1, sx2, sy2, min_overlap_ratio):
    """裁剪并过滤目标框到切片范围内。

    只保留与切片重叠面积 >= 原框面积 * min_overlap_ratio 的框。
    返回 YOLO 格式的归一化坐标列表。
    """
    sw = sx2 - sx1
    sh = sy2 - sy1
    result = []

    for cls, bx1, by1, bx2, by2 in boxes:
        ix1 = max(bx1, sx1)
        iy1 = max(by1, sy1)
        ix2 = min(bx2, sx2)
        iy2 = min(by2, sy2)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        orig_area = (bx2 - bx1) * (by2 - by1)
        inter_area = (ix2 - ix1) * (iy2 - iy1)

        if orig_area <= 0 or inter_area / orig_area < min_overlap_ratio:
            continue

        # 裁剪到切片内并归一化
        cx_new = ((ix1 + ix2) / 2 - sx1) / sw
        cy_new = ((iy1 + iy2) / 2 - sy1) / sh
        w_new = (ix2 - ix1) / sw
        h_new = (iy2 - iy1) / sh

        cx_new = np.clip(cx_new, 0.0, 1.0)
        cy_new = np.clip(cy_new, 0.0, 1.0)
        w_new = np.clip(w_new, 0.001, 1.0)
        h_new = np.clip(h_new, 0.001, 1.0)

        result.append((cls, cx_new, cy_new, w_new, h_new))

    return result


def slice_image(img_path, lbl_path, slice_size, overlap, min_overlap_ratio):
    """切片单张图像，返回有目标切片和空背景切片。"""
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    boxes = load_yolo_labels(lbl_path, img_w, img_h)
    coords = compute_slice_coords(img_w, img_h, slice_size, overlap)

    stem = img_path.stem
    positive_slices = []
    bg_slices = []

    for i, (sx1, sy1, sx2, sy2) in enumerate(coords):
        clipped = clip_boxes_to_slice(boxes, sx1, sy1, sx2, sy2, min_overlap_ratio)

        slice_img = img.crop((sx1, sy1, sx2, sy2))
        # 边缘切片可能不满 slice_size，pad 到统一尺寸
        if slice_img.size != (slice_size, slice_size):
            padded = Image.new("RGB", (slice_size, slice_size), (114, 114, 114))
            padded.paste(slice_img, (0, 0))
            slice_img = padded

        slice_name = f"{stem}_s{i:04d}"

        if clipped:
            positive_slices.append((slice_img, clipped, slice_name))
        else:
            bg_slices.append((slice_img, slice_name))

    return positive_slices, bg_slices


def process_train_split():
    """处理训练集：切片 + 过滤空背景切片。"""
    DST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    DST_LBL_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(SRC_IMG_DIR.glob("*.jpg")) + sorted(SRC_IMG_DIR.glob("*.png"))
    print(f"  训练集图像数: {len(img_paths)}")

    all_positive = []
    all_bg = []

    for img_path in img_paths:
        lbl_path = SRC_LBL_DIR / (img_path.stem + ".txt")
        pos, bg = slice_image(
            img_path, lbl_path, SLICE_SIZE, OVERLAP, MIN_OVERLAP_RATIO
        )
        all_positive.extend(pos)
        all_bg.extend(bg)

    # 保存有目标切片
    for slice_img, clipped, name in all_positive:
        slice_img.save(DST_IMG_DIR / f"{name}.jpg", quality=95)
        with open(DST_LBL_DIR / f"{name}.txt", "w") as f:
            for cls, cx, cy, w, h in clipped:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # 按比例保留空背景切片
    n_bg_keep = int(len(all_positive) * MAX_BG_RATIO)
    rng = np.random.RandomState(42)
    bg_indices = rng.choice(len(all_bg), min(n_bg_keep, len(all_bg)), replace=False)

    for idx in bg_indices:
        slice_img, name = all_bg[idx]
        slice_img.save(DST_IMG_DIR / f"{name}.jpg", quality=95)
        open(DST_LBL_DIR / f"{name}.txt", "w").close()

    total = len(all_positive) + len(bg_indices)
    print(f"  有目标切片: {len(all_positive)}")
    print(f"  空背景切片: 保留 {len(bg_indices)}/{len(all_bg)} ({MAX_BG_RATIO:.0%})")
    print(f"  训练切片总计: {total}")
    return total


def copy_split(src_img, src_lbl, dst_img, dst_lbl, split_name):
    """直接复制 val/test 到切片数据集目录（不切片）。"""
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    imgs = sorted(src_img.glob("*.jpg")) + sorted(src_img.glob("*.png"))
    for f in imgs:
        shutil.copy2(f, dst_img / f.name)

    lbls = sorted(src_lbl.glob("*.txt"))
    for f in lbls:
        shutil.copy2(f, dst_lbl / f.name)

    print(f"  {split_name} 集: 直接复制 {len(imgs)} 张图")


def write_data_yaml(n_train):
    """生成切片数据集的 data.yaml。"""
    yaml_path = PROJECT_ROOT / "configs" / "data_sliced.yaml"
    content = f"""# SAHI 切片训练数据集配置
# 训练集: 640x640 切片（共 {n_train} 张），val/test 为原图
# 训练时使用 imgsz=640（与切片尺寸一致）
# 推理时使用 SAHI 切片推理

path: datasets_sliced
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: drone
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"\n  数据配置: {yaml_path}")


def main():
    print("=" * 60)
    print("  SAHI Training Slice")
    print(f"  切片: {SLICE_SIZE}x{SLICE_SIZE}, 重叠: {OVERLAP:.0%}")
    print(f"  最小重叠比例: {MIN_OVERLAP_RATIO:.0%}, 背景保留: {MAX_BG_RATIO:.0%}")
    print("=" * 60)

    # 清理旧切片
    dst_root = PROJECT_ROOT / "datasets_sliced"
    if dst_root.exists():
        print("\n  清理旧的切片数据集...")
        shutil.rmtree(dst_root)

    # 训练集切片
    print("\n[1/3] 切片训练集...")
    n_train = process_train_split()

    # 验证集复制
    print("\n[2/3] 复制验证集...")
    copy_split(
        PROJECT_ROOT / "datasets" / "images" / "val",
        PROJECT_ROOT / "datasets" / "labels" / "val",
        PROJECT_ROOT / "datasets_sliced" / "images" / "val",
        PROJECT_ROOT / "datasets_sliced" / "labels" / "val",
        "val"
    )

    # 测试集复制
    print("\n[3/3] 复制测试集...")
    copy_split(
        PROJECT_ROOT / "datasets" / "images" / "test",
        PROJECT_ROOT / "datasets" / "labels" / "test",
        PROJECT_ROOT / "datasets_sliced" / "images" / "test",
        PROJECT_ROOT / "datasets_sliced" / "labels" / "test",
        "test"
    )

    # 写配置
    write_data_yaml(n_train)

    print("\n" + "=" * 60)
    print("  切片完成！训练命令参考:")
    print("  python -c \"")
    print("    import register_modules")
    print("    from ultralytics_modules.nwd import patch_all_nwd")
    print("    from ultralytics import YOLO")
    print("    patch_all_nwd()")
    print("    model = YOLO('configs/yolo11n-improved.yaml')")
    print("    model.train(data='configs/data_sliced.yaml', imgsz=640, batch=16, ...)")
    print("  \"")
    print("=" * 60)


if __name__ == "__main__":
    main()
