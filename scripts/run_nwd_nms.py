"""
run_nwd_nms.py — Experiment E5: SA-NWD-NMS Validation

Purpose:
  Validate whether SA-NWD-based NMS provides measurable improvement
  over standard IoU-NMS for tiny object detection.

  Configuration:
    - SA-NWD Loss + SA-NWD TAL (same as best nwd_p2)
    - NMS: hybrid IoU + SA-NWD (use_nwd_nms=True)
    - Architecture: nwd_p2 (P2 head)

Decision rule (from docs/CLAUDE.md):
  If NMS boost > 0 AND FPS drop ≤ 5%: keep NMS as optional contribution
  If FPS drop > 5%: mention efficiency tradeoff in paper even if metric improves
  If NMS boost ≤ 0: drop NMS from contributions entirely

Server path: /root/drone_detection/
Output:      runs/ablation/nwd_p2_nms/
"""

import os
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path('/root/drone_detection')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import register_modules  # noqa: F401

from ultralytics_modules.nwd import patch_all_nwd
# Enable NWD-NMS: IoU NMS first, then SA-NWD suppression among survivors
patch_all_nwd(
    c_base=12.0,
    k=1.0,
    alpha=0.5,
    nwd_min=0.3,
    use_nwd_nms=True,
    nms_iou_threshold=0.7,
    nms_nwd_threshold=0.8,
)
print('[E5] SA-NWD-NMS enabled: IoU NMS + SA-NWD suppression')
print('[E5] Architecture: nwd_p2 (same as best config, + NMS patch)')

from ultralytics import YOLO
import torch

EXP_NAME = 'nwd_p2_nms'
ABLATION_PROJECT = '/root/drone_detection/runs/ablation'
WEIGHT_PATH = f'{ABLATION_PROJECT}/{EXP_NAME}/weights/best.pt'
NWD_P2_YAML = 'configs/ablation/ablation_nwd_p2.yaml'

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

if Path(WEIGHT_PATH).exists():
    print(f'[SKIP] {WEIGHT_PATH} already exists, running val only')
    model = YOLO(WEIGHT_PATH)
    metrics = model.val(data='configs/data.yaml', imgsz=1280, split='val')
else:
    model = YOLO(NWD_P2_YAML)
    model.train(
        pretrained='yolo11n.pt',
        project=ABLATION_PROJECT,
        name=EXP_NAME,
        exist_ok=True,
        **TRAIN_ARGS,
    )
    metrics = model.val(data='configs/data.yaml', imgsz=1280, split='val')

# Measure FPS with NWD-NMS
print('[E5] Measuring inference FPS with NWD-NMS...')
model_for_fps = YOLO(WEIGHT_PATH)
dummy_imgs = ['/root/drone_detection/datasets/images/val'] * 1  # use val dir

# Warmup
_ = model_for_fps.predict(source=dummy_imgs[0], imgsz=1280, verbose=False, save=False)

# Timed run (50 images)
t0 = time.perf_counter()
results_pred = model_for_fps.predict(
    source=dummy_imgs[0], imgsz=1280, verbose=False, save=False,
    stream=True
)
n_imgs = 0
for _ in results_pred:
    n_imgs += 1
elapsed = time.perf_counter() - t0
fps_nms = n_imgs / elapsed if elapsed > 0 else 0.0

# Baseline FPS (from eval_results.json, best.pt without NWD-NMS)
baseline_fps = 59.4  # from nwd_p2 eval; will be updated by E_fps

result = {
    'name': 'E5: SA-NWD-NMS (hybrid IoU + SA-NWD suppression)',
    'exp': EXP_NAME,
    'config': 'nwd_p2 arch, NWD-NMS enabled (iou=0.7, nwd=0.8)',
    'map50':        round(float(metrics.box.map50), 4),
    'map':          round(float(metrics.box.map), 4),
    'precision':    round(float(metrics.box.mp), 4),
    'recall':       round(float(metrics.box.mr), 4),
    'fps_nms':      round(fps_nms, 1),
    'fps_baseline': baseline_fps,
    'fps_drop_pct': round((baseline_fps - fps_nms) / baseline_fps * 100, 1),
}
result['f1'] = round(
    2 * result['precision'] * result['recall'] /
    (result['precision'] + result['recall'] + 1e-8), 4
)
nwd_p2_map50 = 0.9781
delta = result['map50'] - nwd_p2_map50
result['delta_vs_no_nms'] = round(delta, 4)

print('RESULT: ' + json.dumps(result))
print(f'[E5] mAP50: {result["map50"]} (vs {nwd_p2_map50} without NMS), Δ={delta:+.4f}')
print(f'[E5] FPS: {fps_nms:.1f} (baseline {baseline_fps}), drop={result["fps_drop_pct"]}%')

if delta > 0 and result['fps_drop_pct'] <= 5.0:
    print('[E5] → KEEP NMS: positive contribution, FPS drop acceptable')
elif delta > 0 and result['fps_drop_pct'] > 5.0:
    print('[E5] → MENTION TRADEOFF: NMS improves mAP but >5% FPS drop')
else:
    print('[E5] → DROP NMS: no mAP benefit, not worth the complexity')

out_path = Path(ABLATION_PROJECT) / EXP_NAME / 'result.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f'[E5] Result saved to {out_path}')
