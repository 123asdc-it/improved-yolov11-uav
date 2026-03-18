"""
run_tal_ablation.py — Experiments E2/E3: TAL Ablation

Isolates the independent contribution of SA-NWD Loss vs SA-NWD TAL.

  E2 (loss_only):  SA-NWD loss (alpha=0.5) + stock CIoU-based TAL
                   → architecture: stock yolo11n (no P2)
  E3 (tal_only):   standard CIoU loss + SA-NWD TAL
                   → architecture: stock yolo11n (no P2)

Combined with existing results:
  Baseline:   CIoU loss + CIoU TAL          → mAP50=0.9600
  E2:         SA-NWD loss + CIoU TAL        → mAP50=?
  E3:         CIoU loss + SA-NWD TAL        → mAP50=?
  nwd_only:   SA-NWD loss + SA-NWD TAL      → mAP50=0.9705

Usage: python scripts/run_tal_ablation.py [--exp {e2,e3,both}]

Server path: /root/drone_detection/
Output:      runs/ablation/nwd_loss_only/ and runs/ablation/nwd_tal_only/
"""

import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules  # noqa: F401

from ultralytics import YOLO

ABLATION_PROJECT = '/root/drone_detection/runs/ablation'
STOCK_YAML = 'yolo11n.pt'  # stock architecture (no P2)

TRAIN_ARGS = dict(
    data='configs/data.yaml',
    imgsz=1280,
    epochs=300,
    patience=100,
    batch=8,
    lr0=0.01,
    cos_lr=True,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    warmup_epochs=5,
    workers=4,
    cache=False,
    seed=0,
)


def run_e2_loss_only():
    """E2: SA-NWD Loss only, standard CIoU TAL (no SA-NWD TAL patch)."""
    print('\n' + '='*60)
    print('[E2] SA-NWD Loss only (stock TAL, no P2 head)')
    print('='*60)

    # Patch only the loss, NOT the TAL
    from ultralytics_modules.nwd import patch_sa_nwd_loss
    patch_sa_nwd_loss(c_base=12.0, k=1.0, alpha=0.5)

    exp_name = 'nwd_loss_only'
    weight_path = f'{ABLATION_PROJECT}/{exp_name}/weights/best.pt'

    if Path(weight_path).exists():
        print(f'[SKIP] {weight_path} already exists, running val only')
        model = YOLO(weight_path)
        metrics = model.val(data='configs/data.yaml', imgsz=1280, split='val')
    else:
        model = YOLO(STOCK_YAML)
        model.train(
            project=ABLATION_PROJECT,
            name=exp_name,
            exist_ok=True,
            **TRAIN_ARGS,
        )
        metrics = model.val(data='configs/data.yaml', imgsz=1280, split='val')

    result = {
        'name': 'E2: SA-NWD Loss only (stock TAL)',
        'exp': exp_name,
        'config': 'stock arch, SA-NWD loss (k=1.0, alpha=0.5), CIoU TAL',
        'map50':     round(float(metrics.box.map50), 4),
        'map':       round(float(metrics.box.map), 4),
        'precision': round(float(metrics.box.mp), 4),
        'recall':    round(float(metrics.box.mr), 4),
    }
    result['f1'] = round(
        2 * result['precision'] * result['recall'] /
        (result['precision'] + result['recall'] + 1e-8), 4
    )
    print('RESULT E2: ' + json.dumps(result))

    out_path = Path(ABLATION_PROJECT) / exp_name / 'result.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[E2] Saved to {out_path}')
    return result


def run_e3_tal_only():
    """E3: SA-NWD TAL only, standard CIoU loss."""
    print('\n' + '='*60)
    print('[E3] SA-NWD TAL only (stock CIoU loss, no P2 head)')
    print('='*60)

    # Patch only the TAL, NOT the loss
    from ultralytics_modules.nwd import patch_sa_nwd_tal
    patch_sa_nwd_tal(c_base=12.0, k=1.0, nwd_min=0.3)

    exp_name = 'nwd_tal_only'
    weight_path = f'{ABLATION_PROJECT}/{exp_name}/weights/best.pt'

    if Path(weight_path).exists():
        print(f'[SKIP] {weight_path} already exists, running val only')
        model = YOLO(weight_path)
        metrics = model.val(data='configs/data.yaml', imgsz=1280, split='val')
    else:
        model = YOLO(STOCK_YAML)
        model.train(
            project=ABLATION_PROJECT,
            name=exp_name,
            exist_ok=True,
            **TRAIN_ARGS,
        )
        metrics = model.val(data='configs/data.yaml', imgsz=1280, split='val')

    result = {
        'name': 'E3: SA-NWD TAL only (CIoU loss)',
        'exp': exp_name,
        'config': 'stock arch, CIoU loss, SA-NWD TAL (k=1.0, nwd_min=0.3)',
        'map50':     round(float(metrics.box.map50), 4),
        'map':       round(float(metrics.box.map), 4),
        'precision': round(float(metrics.box.mp), 4),
        'recall':    round(float(metrics.box.mr), 4),
    }
    result['f1'] = round(
        2 * result['precision'] * result['recall'] /
        (result['precision'] + result['recall'] + 1e-8), 4
    )
    print('RESULT E3: ' + json.dumps(result))

    out_path = Path(ABLATION_PROJECT) / exp_name / 'result.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[E3] Saved to {out_path}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['e2', 'e3', 'both'], default='both',
                        help='Which experiment to run (default: both)')
    args = parser.parse_args()

    results = {}
    if args.exp in ('e2', 'both'):
        results['e2'] = run_e2_loss_only()
    if args.exp in ('e3', 'both'):
        results['e3'] = run_e3_tal_only()

    print('\n' + '='*60)
    print('TAL ABLATION SUMMARY:')
    print('  Baseline (CIoU loss + CIoU TAL): mAP50=0.9600')
    if 'e2' in results:
        print(f'  E2 (SA-NWD loss + CIoU TAL):    mAP50={results["e2"]["map50"]}')
    if 'e3' in results:
        print(f'  E3 (CIoU loss + SA-NWD TAL):    mAP50={results["e3"]["map50"]}')
    print('  nwd_only (both):                 mAP50=0.9705')
    print('='*60)
