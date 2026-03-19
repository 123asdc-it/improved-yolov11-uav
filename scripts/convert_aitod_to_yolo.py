"""
convert_aitod_to_yolo.py — Convert AI-TOD dataset to Ultralytics YOLO format

AI-TOD dataset:
  - Paper: "Tiny Object Detection in Aerial Images" (ICPR 2021)
  - URL: https://github.com/jwwangchn/AI-TOD
  - Format: COCO JSON (train/val/test splits)
  - 8 categories: airplane, bridge, storage-tank, ship, swimming-pool,
                  vehicle, person, wind-mill
  - v1: ~18K images (DOTA source only)
  - v2: ~28K images (DOTA + xView); xView images require separate license
  - Image size: 800×800 (already cropped)

Output structure (Ultralytics YOLO format):
  datasets/aitod/
  ├── images/
  │   ├── train/   *.png
  │   ├── val/     *.png
  │   └── test/    *.png
  └── labels/
      ├── train/   *.txt  (class cx cy w h, normalized [0,1])
      ├── val/     *.txt
      └── test/    *.txt

Usage:
  # v1 (no xView license needed):
  python scripts/convert_aitod_to_yolo.py \\
      --src /path/to/aitod_raw \\
      --dst datasets/aitod

  # v2 with xView filtering (only use DOTA-sourced images):
  python scripts/convert_aitod_to_yolo.py \\
      --src /path/to/aitod_raw \\
      --dst datasets/aitod \\
      --filter-xview

  # v2 use all images (requires xView license):
  python scripts/convert_aitod_to_yolo.py \\
      --src /path/to/aitod_raw \\
      --dst datasets/aitod \\
      --use-all

xView filtering:
  AI-TOD v2 image filenames from xView start with 'xview_' or contain
  a numeric prefix > 1000 (DOTA images use P####.png naming).
  With --filter-xview (default for v2), only DOTA-sourced images are kept.
  This reduces ~28K → ~18K images but requires no xView license.
"""

import json
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


# AI-TOD category mapping (category_id → class_idx, class_name)
AITOD_CATEGORIES = {
    1: (0, 'airplane'),
    2: (1, 'bridge'),
    3: (2, 'storage-tank'),
    4: (3, 'ship'),
    5: (4, 'swimming-pool'),
    6: (5, 'vehicle'),
    7: (6, 'person'),
    8: (7, 'wind-mill'),
}


def is_xview_image(filename: str) -> bool:
    """Return True if this image is from xView source (not DOTA).

    AI-TOD v2 xView images are named like:
      xview_XXXXXX.png  or  XXXXXX.png (6-digit numeric, no 'P' prefix)
    DOTA images are named like:
      P0001__1__0___0.png  (start with 'P' followed by 4 digits)
    """
    stem = Path(filename).stem
    # DOTA pattern: starts with P + digits
    if stem.upper().startswith('P') and stem[1:].split('__')[0].isdigit():
        return False
    # xView pattern: purely numeric or starts with 'xview'
    if stem.lower().startswith('xview'):
        return True
    if stem.split('__')[0].isdigit():
        return True
    return False


def coco_to_yolo(annotation: dict, img_w: int, img_h: int):
    """Convert COCO annotation to YOLO format.

    COCO: [x_min, y_min, width, height] (absolute pixels)
    YOLO: [class_id, cx, cy, w, h] (normalized [0,1])
    """
    cat_id = annotation['category_id']
    class_idx = AITOD_CATEGORIES.get(cat_id, (None,))[0]
    if class_idx is None:
        return None

    x_min, y_min, w, h = annotation['bbox']
    cx = (x_min + w / 2) / img_w
    cy = (y_min + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    # Skip degenerate boxes
    if w_norm < 1e-5 or h_norm < 1e-5:
        return None

    return f"{class_idx} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}"


def convert_split(src_dir: Path, dst_dir: Path, split: str,
                  copy_images: bool = True, filter_xview: bool = True):
    """Convert one split (train/val/test) from COCO to YOLO format."""
    # Try multiple annotation filename conventions
    ann_candidates = [
        src_dir / 'annotations' / f'aitod_{split}_v2.json',
        src_dir / 'annotations' / f'aitod_{split}_v1.json',
        src_dir / 'annotations' / f'ai_tod_{split}.json',
        src_dir / 'annotations' / f'{split}.json',
    ]
    ann_file = None
    for c in ann_candidates:
        if c.exists():
            ann_file = c
            break
    if ann_file is None:
        print(f'[WARN] No annotation file found for split={split}, tried:')
        for c in ann_candidates:
            print(f'       {c}')
        return 0, 0

    print(f'[{split}] Loading {ann_file}...')
    with open(ann_file) as f:
        coco = json.load(f)

    img_info = {img['id']: img for img in coco['images']}
    ann_by_image = {}
    for ann in coco['annotations']:
        ann_by_image.setdefault(ann['image_id'], []).append(ann)

    out_img_dir = dst_dir / 'images' / split
    out_lbl_dir = dst_dir / 'labels' / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_images = n_boxes = n_skipped_xview = 0

    for img_id, img in tqdm(img_info.items(), desc=f'[{split}]'):
        img_filename = img['file_name']

        # Filter xView images if requested
        if filter_xview and is_xview_image(img_filename):
            n_skipped_xview += 1
            continue

        img_w = img['width']
        img_h = img['height']

        # Locate source image
        src_img = src_dir / 'images' / img_filename
        if not src_img.exists():
            src_img = src_dir / 'images' / split / img_filename
        if not src_img.exists():
            continue

        anns = ann_by_image.get(img_id, [])
        yolo_lines = []
        for ann in anns:
            line = coco_to_yolo(ann, img_w, img_h)
            if line is not None:
                yolo_lines.append(line)
                n_boxes += 1

        stem = Path(img_filename).stem
        lbl_path = out_lbl_dir / f'{stem}.txt'
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        if copy_images:
            dst_img = out_img_dir / img_filename
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

        n_images += 1

    msg = f'[{split}] Done: {n_images} images, {n_boxes} boxes'
    if n_skipped_xview:
        msg += f', {n_skipped_xview} xView images skipped'
    print(msg)
    return n_images, n_boxes


def write_data_yaml(dst_dir: Path, splits: list):
    """Write aitod_data.yaml for Ultralytics training."""
    class_names = [v[1] for v in sorted(AITOD_CATEGORIES.values(), key=lambda x: x[0])]
    yaml_content = f"""# AI-TOD dataset configuration for Ultralytics YOLO
# Converted by scripts/convert_aitod_to_yolo.py
# Paper: "Tiny Object Detection in Aerial Images" (ICPR 2021)

path: {dst_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    out = dst_dir / 'aitod_data.yaml'
    with open(out, 'w') as f:
        f.write(yaml_content)
    print(f'[YAML] Written: {out}')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert AI-TOD to Ultralytics YOLO format')
    parser.add_argument('--src', required=True, help='Path to raw AI-TOD directory')
    parser.add_argument('--dst', default='datasets/aitod', help='Output directory')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--no-copy', action='store_true',
                        help='Skip copying images (use when images already in dst)')
    parser.add_argument('--filter-xview', action='store_true', default=True,
                        help='Filter out xView-sourced images (default: True for v2 compliance)')
    parser.add_argument('--use-all', action='store_true',
                        help='Use all images including xView (requires xView license)')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    copy_images = not args.no_copy
    filter_xview = not args.use_all  # --use-all overrides --filter-xview

    print(f'Converting AI-TOD: {src} → {dst}')
    print(f'Splits: {args.splits}')
    print(f'xView filter: {"ON (DOTA-only)" if filter_xview else "OFF (all images)"}')

    total_images = 0
    total_boxes = 0
    for split in args.splits:
        n_img, n_box = convert_split(src, dst, split,
                                     copy_images=copy_images,
                                     filter_xview=filter_xview)
        total_images += n_img
        total_boxes += n_box

    yaml_path = write_data_yaml(dst, args.splits)

    print(f'\nConversion complete:')
    print(f'  Total images: {total_images}')
    print(f'  Total boxes:  {total_boxes}')
    print(f'  Data YAML:    {yaml_path}')
    if filter_xview:
        print(f'  Note: xView images excluded. Use --use-all if you have xView license.')
    print(f'\nNext: rsync datasets/aitod/ to server, then bash scripts/run_aitod_queue.sh')
