"""
小目标 Copy-Paste 增强：从训练集裁剪目标，粘贴到其他图像中增加目标密度。

核心思路：
  当前训练集每张图平均 ~1 个目标，目标密度太低
  通过 copy-paste，每张图增加 2-4 个目标 → 模型每步能看到更多正样本

方案：
  1. 建立目标库：从训练图中裁剪所有目标图像块
  2. 对每张训练图，从目标库随机取 2-4 个目标粘贴到空白区域
  3. 生成增强图像 + 更新标注（391 → 782 张，原图保留）
  4. 可与切片脚本联合使用：先 copy-paste → 再切片

用法：cd 项目根目录，然后 python scripts/augment_copy_paste.py
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# ============================================================
# 配置
# ============================================================
PASTE_COUNT_MIN = 2       # 每张图最少粘贴几个目标
PASTE_COUNT_MAX = 4       # 最多粘贴几个
SCALE_RANGE = (0.8, 1.2)  # 粘贴时随机缩放范围
MARGIN = 10               # 距图像边缘的最小距离（像素）
MAX_RETRY = 50            # 寻找不重叠位置的最大尝试次数

SRC_IMG_DIR = PROJECT_ROOT / "datasets" / "images" / "train"
SRC_LBL_DIR = PROJECT_ROOT / "datasets" / "labels" / "train"
DST_IMG_DIR = PROJECT_ROOT / "datasets_augmented" / "images" / "train"
DST_LBL_DIR = PROJECT_ROOT / "datasets_augmented" / "labels" / "train"


def load_yolo_labels(label_path, img_w, img_h):
    """读取 YOLO 标注，返回像素坐标 [(cls, x1, y1, x2, y2), ...]。"""
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
            boxes.append((cls, px1, py1, px2, py2))
    return boxes


def save_yolo_labels(label_path, boxes, img_w, img_h):
    """保存 YOLO 格式标注。boxes: [(cls, x1, y1, x2, y2), ...]"""
    with open(label_path, "w") as f:
        for cls, x1, y1, x2, y2 in boxes:
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            cx = np.clip(cx, 0.0, 1.0)
            cy = np.clip(cy, 0.0, 1.0)
            w = np.clip(w, 0.001, 1.0)
            h = np.clip(h, 0.001, 1.0)
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def build_object_bank(img_dir, lbl_dir):
    """从训练集中裁剪所有目标，建立目标库。

    Returns:
        list of (crop_image: PIL.Image, cls: int, orig_w: int, orig_h: int)
    """
    bank = []
    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    for img_path in img_paths:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        boxes = load_yolo_labels(lbl_path, img_w, img_h)

        for cls, x1, y1, x2, y2 in boxes:
            # 稍微扩大裁剪区域（加 2 像素边距），避免边缘裁切
            x1c = max(0, int(x1) - 2)
            y1c = max(0, int(y1) - 2)
            x2c = min(img_w, int(x2) + 2)
            y2c = min(img_h, int(y2) + 2)
            crop_w = x2c - x1c
            crop_h = y2c - y1c
            if crop_w < 5 or crop_h < 5:
                continue
            crop = img.crop((x1c, y1c, x2c, y2c))
            bank.append((crop, cls, crop_w, crop_h))

    return bank


def boxes_overlap(box_a, box_b):
    """检查两个框是否重叠。box: (x1, y1, x2, y2)"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def find_paste_position(img_w, img_h, paste_w, paste_h, existing_boxes, rng):
    """寻找一个不与已有目标重叠的粘贴位置。

    Returns:
        (x1, y1, x2, y2) or None if no valid position found
    """
    for _ in range(MAX_RETRY):
        x1 = rng.randint(MARGIN, max(MARGIN + 1, img_w - paste_w - MARGIN))
        y1 = rng.randint(MARGIN, max(MARGIN + 1, img_h - paste_h - MARGIN))
        x2 = x1 + paste_w
        y2 = y1 + paste_h

        if x2 > img_w - MARGIN or y2 > img_h - MARGIN:
            continue

        overlap = False
        for box in existing_boxes:
            if boxes_overlap((x1, y1, x2, y2), box[1:]):
                overlap = True
                break
        if not overlap:
            return (x1, y1, x2, y2)

    return None


def augment_image(img_path, lbl_path, obj_bank, rng):
    """对单张图像进行 copy-paste 增强。

    Returns:
        (augmented_image, augmented_boxes) or None if no augmentation possible
    """
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    boxes = load_yolo_labels(lbl_path, img_w, img_h)

    # 转成 (cls, x1, y1, x2, y2) 格式的列表
    existing = list(boxes)
    n_paste = rng.randint(PASTE_COUNT_MIN, PASTE_COUNT_MAX + 1)
    pasted = 0

    for _ in range(n_paste):
        # 随机选目标
        crop_img, cls, orig_w, orig_h = obj_bank[rng.randint(len(obj_bank))]

        # 随机缩放
        scale = rng.uniform(*SCALE_RANGE)
        new_w = max(5, int(orig_w * scale))
        new_h = max(5, int(orig_h * scale))
        crop_resized = crop_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # 寻找不重叠位置
        pos = find_paste_position(img_w, img_h, new_w, new_h, existing, rng)
        if pos is None:
            continue

        x1, y1, x2, y2 = pos

        # 粘贴（直接覆盖，简单有效）
        img.paste(crop_resized, (x1, y1))
        existing.append((cls, x1, y1, x2, y2))
        pasted += 1

    if pasted == 0:
        return None

    return img, existing


def main():
    print("=" * 60)
    print("  Small Object Copy-Paste Augmentation")
    print(f"  每张图粘贴 {PASTE_COUNT_MIN}-{PASTE_COUNT_MAX} 个目标")
    print(f"  缩放范围: {SCALE_RANGE}")
    print("=" * 60)

    # 建立目标库
    print("\n[1/3] 建立目标库...")
    obj_bank = build_object_bank(SRC_IMG_DIR, SRC_LBL_DIR)
    print(f"  目标库大小: {len(obj_bank)} 个目标")

    if len(obj_bank) == 0:
        print("  目标库为空，退出")
        return

    # 复制原始数据
    DST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    DST_LBL_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(SRC_IMG_DIR.glob("*.jpg")) + sorted(SRC_IMG_DIR.glob("*.png"))

    print(f"\n[2/3] 复制原始训练集 ({len(img_paths)} 张)...")
    for img_path in img_paths:
        lbl_path = SRC_LBL_DIR / (img_path.stem + ".txt")
        # 复制原图
        Image.open(img_path).save(DST_IMG_DIR / img_path.name, quality=95)
        if lbl_path.exists():
            with open(lbl_path) as f_in, open(DST_LBL_DIR / lbl_path.name, "w") as f_out:
                f_out.write(f_in.read())
        else:
            open(DST_LBL_DIR / (img_path.stem + ".txt"), "w").close()

    # 生成增强图像
    print(f"\n[3/3] 生成增强图像...")
    rng = np.random.RandomState(42)
    augmented_count = 0

    for img_path in img_paths:
        lbl_path = SRC_LBL_DIR / (img_path.stem + ".txt")
        result = augment_image(img_path, lbl_path, obj_bank, rng)

        if result is None:
            continue

        aug_img, aug_boxes = result
        aug_name = f"{img_path.stem}_aug"
        aug_img.save(DST_IMG_DIR / f"{aug_name}.jpg", quality=95)
        save_yolo_labels(
            DST_LBL_DIR / f"{aug_name}.txt",
            aug_boxes,
            aug_img.size[0], aug_img.size[1]
        )
        augmented_count += 1

    total = len(img_paths) + augmented_count
    print(f"\n  原始图像: {len(img_paths)}")
    print(f"  增强图像: {augmented_count}")
    print(f"  总计: {total}")

    # 复制 val/test
    for split in ["val", "test"]:
        src_img = PROJECT_ROOT / "datasets" / "images" / split
        dst_img = PROJECT_ROOT / "datasets_augmented" / "images" / split
        src_lbl = PROJECT_ROOT / "datasets" / "labels" / split
        dst_lbl = PROJECT_ROOT / "datasets_augmented" / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for f in sorted(src_img.glob("*")):
            Image.open(f).save(dst_img / f.name, quality=95)
        for f in sorted(src_lbl.glob("*.txt")):
            with open(f) as fi, open(dst_lbl / f.name, "w") as fo:
                fo.write(fi.read())
        count = len(list(src_img.glob("*")))
        print(f"  {split} 集: 复制 {count} 张")

    # 写 data.yaml
    yaml_path = PROJECT_ROOT / "configs" / "data_augmented.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""# Copy-Paste 增强训练数据集
# 训练集: 原始 {len(img_paths)} 张 + 增强 {augmented_count} 张 = {total} 张
# val/test 不增强

path: datasets_augmented
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: drone
""")
    print(f"\n  数据配置: {yaml_path}")
    print("\n" + "=" * 60)
    print("  增强完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
