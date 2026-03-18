"""
download_visdrone.py — 下载并转换 VisDrone2019-DET 数据集

运行方式（在项目根目录执行）：
    python scripts/download_visdrone.py

下载目标：datasets/VisDrone/
配置文件：configs/data_visdrone.yaml（自动生成）

数据集规模：
    train: 6471 images
    val:   548  images
    test:  1610 images（test-dev）
    大小：约 2.3 GB（下载），解压后约 4 GB

原始格式：VisDrone csv 标注（x,y,w,h,score,cls,trunc,occ）
转换后：YOLO 格式（cls cx cy w h，归一化）
忽略：score=0 的标注（ignored regions）
类别（10类）：
    0:pedestrian 1:people 2:bicycle 3:car 4:van
    5:truck 6:tricycle 7:awning-tricycle 8:bus 9:motor
"""

import os
import sys
import shutil
from pathlib import Path

# 确保在项目根目录运行
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
print(f"[VisDrone] Project root: {PROJECT_ROOT}")

DATASET_DIR = PROJECT_ROOT / "datasets" / "VisDrone"
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_visdrone.yaml"

# ── 下载 ──────────────────────────────────────────────────────────────
def download_visdrone():
    from ultralytics.utils.downloads import download

    # GitHub Releases 直链（ultralytics/assets）
    ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
    files = [
        "VisDrone2019-DET-train.zip",
        "VisDrone2019-DET-val.zip",
        "VisDrone2019-DET-test-dev.zip",
    ]

    print(f"[VisDrone] Downloading to: {DATASET_DIR}")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    import subprocess
    for fname in files:
        url = f"{ASSETS_URL}/{fname}"
        dest = DATASET_DIR / fname
        if dest.exists():
            print(f"[VisDrone] Already exists, skipping: {fname}")
            continue
        print(f"[VisDrone] Downloading {fname} ...")
        ret = subprocess.run(
            ["curl", "-L", "--retry", "3", "--retry-delay", "2",
             "-o", str(dest), url],
            check=True
        )
        print(f"[VisDrone] Saved: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    print("[VisDrone] Download complete. Extracting...")
    import zipfile
    for fname in files:
        dest = DATASET_DIR / fname
        print(f"[VisDrone] Extracting {fname} ...")
        with zipfile.ZipFile(dest, "r") as zf:
            zf.extractall(DATASET_DIR)
        dest.unlink()
        print(f"[VisDrone] Extracted and removed zip: {fname}")


# ── 转换 ──────────────────────────────────────────────────────────────
def visdrone2yolo(split: str, source_name: str):
    """将 VisDrone 原始标注转换为 YOLO 格式。"""
    from PIL import Image
    from tqdm import tqdm

    source_dir  = DATASET_DIR / source_name
    images_dir  = DATASET_DIR / "images" / split
    labels_dir  = DATASET_DIR / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 将图片移动到标准目录
    src_images = source_dir / "images"
    if src_images.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        for img in src_images.glob("*.jpg"):
            img.rename(images_dir / img.name)
        print(f"[VisDrone] Moved images → {images_dir}")

    # 转换标注
    ann_dir = source_dir / "annotations"
    ann_files = list(ann_dir.glob("*.txt"))
    converted, skipped = 0, 0

    for f in tqdm(ann_files, desc=f"Converting {split}"):
        img_path = images_dir / f.with_suffix(".jpg").name
        if not img_path.exists():
            skipped += 1
            continue

        img_w, img_h = Image.open(img_path).size
        dw, dh = 1.0 / img_w, 1.0 / img_h
        lines = []

        with open(f, encoding="utf-8") as fp:
            for row in [x.split(",") for x in fp.read().strip().splitlines()]:
                if len(row) < 6:
                    continue
                if row[4] == "0":           # ignored region
                    continue
                x, y, w, h = map(int, row[:4])
                cls = int(row[5]) - 1       # VisDrone 类别从 1 开始
                if cls < 0 or cls > 9:
                    continue
                cx = (x + w / 2) * dw
                cy = (y + h / 2) * dh
                wn = w * dw
                hn = h * dh
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")

        (labels_dir / f.name).write_text("".join(lines), encoding="utf-8")
        converted += 1

    print(f"[VisDrone] {split}: {converted} labels converted, {skipped} skipped.")

    # 清理原始目录
    if source_dir.exists():
        shutil.rmtree(source_dir)
        print(f"[VisDrone] Cleaned up: {source_dir}")


# ── 生成 data_visdrone.yaml ───────────────────────────────────────────
def write_config():
    content = """# VisDrone2019-DET 数据集配置
# Ultralytics YOLO format
# 下载并转换：python scripts/download_visdrone.py

path: datasets/VisDrone
train: images/train   # 6471 images
val:   images/val     # 548  images
test:  images/test    # 1610 images (test-dev)

nc: 10
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
    CONFIG_PATH.write_text(content, encoding="utf-8")
    print(f"[VisDrone] Config written: {CONFIG_PATH}")


# ── 统计验证 ──────────────────────────────────────────────────────────
def verify():
    print("\n[VisDrone] Dataset summary:")
    for split in ("train", "val", "test"):
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        n_img = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  {split:6s}: {n_img:5d} images, {n_lbl:5d} labels")


# ── 主流程 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 检查是否已存在
    already = (DATASET_DIR / "images" / "train").exists()
    if already:
        print("[VisDrone] Dataset already exists, skipping download.")
        print("[VisDrone] Delete datasets/VisDrone/ to re-download.")
        verify()
        write_config()
        sys.exit(0)

    download_visdrone()

    splits = {
        "VisDrone2019-DET-train":    "train",
        "VisDrone2019-DET-val":      "val",
        "VisDrone2019-DET-test-dev": "test",
    }
    for folder, split in splits.items():
        visdrone2yolo(split, folder)

    write_config()
    verify()
    print("\n[VisDrone] Done. Use configs/data_visdrone.yaml for training.")
