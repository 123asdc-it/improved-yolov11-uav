"""
convert_dut_to_yolo.py — Convert DUT Anti-UAV dataset to Ultralytics YOLO format

DUT Anti-UAV:
  Paper: "DUT Anti-UAV: A Large-Scale Benchmark for Vision-Based UAV Detection"
         IEEE TITS 2022, arXiv:2205.10851, Dalian University of Technology
  GitHub: https://github.com/wangdongdut/DUT-Anti-UAV
  Format: Pascal VOC XML (xmin/ymin/xmax/ymax, absolute pixels)
  Class:  single class "UAV"
  Splits: train 5200 / val 2600 / test 2200 = 10,000 images
  Median bbox area: 0.000472  (88.3% of targets < 1% image area — extreme tiny)

Input structure (as downloaded):
  <src>/
  ├── train/img/*.jpg  +  train/xml/*.xml
  ├── val/img/*.jpg    +  val/xml/*.xml
  └── test/img/*.jpg   +  test/xml/*.xml

Output structure:
  <dst>/
  ├── images/train/  val/  test/
  ├── labels/train/  val/  test/   (YOLO txt: "0 cx cy w h")
  └── dut_data.yaml

Usage (local, from project root):
  python scripts/convert_dut_to_yolo.py
  # default: --src datasets  --dst datasets/dut

Usage (server):
  python scripts/convert_dut_to_yolo.py \\
      --src /root/drone_detection/datasets \\
      --dst /root/drone_detection/datasets/dut
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


CLASS_IDX = 0  # single class: UAV


def voc_to_yolo(xml_path: Path):
    """Parse one Pascal VOC XML, return list of YOLO-format strings."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_w = int(root.find('size/width').text)
    img_h = int(root.find('size/height').text)

    lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip().upper()
        if name not in ('UAV', 'DRONE', 'UAS'):
            continue

        b = obj.find('bndbox')
        xmin = float(b.find('xmin').text)
        ymin = float(b.find('ymin').text)
        xmax = float(b.find('xmax').text)
        ymax = float(b.find('ymax').text)

        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h

        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.0, min(1.0, w))
        h  = max(0.0, min(1.0, h))

        if w < 1e-5 or h < 1e-5:
            continue

        lines.append(f"{CLASS_IDX} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def convert_split(src_dir: Path, dst_dir: Path, split: str, copy_images: bool):
    img_src = src_dir / split / 'img'
    xml_src = src_dir / split / 'xml'

    if not img_src.exists():
        print(f'[WARN] {img_src} not found, skipping {split}')
        return 0, 0

    img_dst = dst_dir / 'images' / split
    lbl_dst = dst_dir / 'labels' / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_src.glob('*.jpg')) + sorted(img_src.glob('*.png'))
    n_images = n_boxes = n_empty = 0

    for img_path in img_files:
        stem = img_path.stem
        xml_path = xml_src / f'{stem}.xml'

        yolo_lines = voc_to_yolo(xml_path) if xml_path.exists() else []
        if not yolo_lines:
            n_empty += 1

        (lbl_dst / f'{stem}.txt').write_text('\n'.join(yolo_lines))

        if copy_images:
            dst_img = img_dst / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

        n_images += 1
        n_boxes += len(yolo_lines)

    print(f'[{split:5s}] {n_images} images, {n_boxes} boxes, {n_empty} empty')
    return n_images, n_boxes


def write_data_yaml(dst_dir: Path):
    content = f"""# DUT Anti-UAV — Ultralytics YOLO format
# Paper: IEEE TITS 2022, arXiv:2205.10851
# GitHub: https://github.com/wangdongdut/DUT-Anti-UAV
# Converted by scripts/convert_dut_to_yolo.py

path: {dst_dir.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names: ['UAV']
"""
    out = dst_dir / 'dut_data.yaml'
    out.write_text(content)
    print(f'[YAML] {out}')
    return out


def print_stats(dst_dir: Path):
    areas = []
    for lbl in (dst_dir / 'labels' / 'train').glob('*.txt'):
        for line in lbl.read_text().strip().splitlines():
            if not line:
                continue
            parts = line.split()
            areas.append(float(parts[3]) * float(parts[4]))
    if not areas:
        return
    areas.sort()
    n = len(areas)
    tiny  = sum(1 for a in areas if a < 0.01)
    small = sum(1 for a in areas if 0.01 <= a < 0.05)
    large = sum(1 for a in areas if a >= 0.05)
    print(f'\nBBox area stats (train, {n} boxes):')
    print(f'  min={areas[0]:.6f}  median={areas[n//2]:.6f}  max={areas[-1]:.6f}')
    print(f'  tiny(<1%):  {tiny}  ({100*tiny/n:.1f}%)')
    print(f'  small(1-5%): {small} ({100*small/n:.1f}%)')
    print(f'  large(>5%):  {large} ({100*large/n:.1f}%)')
    print(f'  [ref] private dataset median: 0.002300')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert DUT Anti-UAV (VOC XML) to Ultralytics YOLO format')
    parser.add_argument('--src', default='datasets',
                        help='Root dir containing train/val/test subdirs (default: datasets)')
    parser.add_argument('--dst', default='datasets/dut',
                        help='Output directory (default: datasets/dut)')
    parser.add_argument('--no-copy', action='store_true',
                        help='Skip copying images (if already in dst)')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    print(f'Converting DUT Anti-UAV: {src} → {dst}')

    total_img = total_box = 0
    for split in ['train', 'val', 'test']:
        n_img, n_box = convert_split(src, dst, split, not args.no_copy)
        total_img += n_img
        total_box += n_box

    write_data_yaml(dst)
    print_stats(dst)
    print(f'\nDone: {total_img} images, {total_box} boxes')
    print(f'Next: rsync datasets/dut/ to server, then bash scripts/run_dut_queue.sh')
