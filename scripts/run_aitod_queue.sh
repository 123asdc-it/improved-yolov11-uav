#!/bin/bash
# run_aitod_queue.sh — AI-TOD Cross-Dataset Validation Queue (E7-E12)
#
# Queue order (serial, ~24-36h each, total ~6-8 days):
#   E7:  Dry run (1 epoch, OOM check)             ~30min
#   E8:  AI-TOD Baseline (YOLOv11n + CIoU)        ~24-36h
#   E9:  AI-TOD + NWD k=0 (fixed C)               ~24-36h
#   E10: AI-TOD + SA-NWD (k=1.0, c_base=12)       ~24-36h
#   E11: AI-TOD + SA-NWD + P2 Head                ~24-36h
#   E12: AI-TOD + P2 only (no NWD)                ~24-36h
#
# Prerequisites:
#   1. Convert AI-TOD: python scripts/convert_aitod_to_yolo.py ...
#   2. Confirm datasets/aitod_converted/aitod_data.yaml has correct absolute path
#   3. Confirm E7 dry run succeeds (no OOM)
#
# Usage (on server):
#   screen -S aitod
#   cd /root/drone_detection
#   bash scripts/run_aitod_queue.sh 2>&1 | tee logs/aitod_queue.log

# Note: no set -e — individual experiment failures handled by run_exp()
cd /root/drone_detection

LOG_DIR="logs"
AITOD_DATA="datasets/aitod_converted/aitod_data.yaml"
ABLATION_PROJECT="runs/ablation"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  AI-TOD Cross-Dataset Validation Queue"
echo "  Start: $(date)"
echo "  Data: $AITOD_DATA"
echo "============================================================"

# Check data yaml exists
if [ ! -f "$AITOD_DATA" ]; then
    echo "[ERROR] $AITOD_DATA not found. Run convert_aitod_to_yolo.py first."
    exit 1
fi

run_exp() {
    local name="$1"
    local script="$2"
    echo ""
    echo "------------------------------------------------------------"
    echo "  START: $name  [$(date)]"
    echo "------------------------------------------------------------"
    python "$script" 2>&1 | tee "$LOG_DIR/${name}.log"
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo "[ERROR] $name failed with exit code $exit_code"
        echo "[WARN]  Continuing with next experiment..."
    else
        echo "[OK] $name finished at $(date)"
    fi
}

# ── E7: Dry run ─────────────────────────────────────────────────
echo ""
echo "  E7: Dry run (1 epoch, OOM check)"
python - <<'PYEOF' 2>&1 | tee logs/aitod_dryrun.log
import os, sys
from pathlib import Path
os.chdir('/root/drone_detection')
sys.path.insert(0, 'scripts')
import register_modules
from ultralytics_modules.nwd import patch_all_nwd
patch_all_nwd(c_base=12.0, k=1.0, alpha=0.5, nwd_min=0.3)
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(
    data='datasets/aitod_converted/aitod_data.yaml',
    imgsz=800,
    epochs=1,
    batch=8,
    workers=2,
    cache=False,
    project='runs/ablation',
    name='aitod_dryrun',
    exist_ok=True,
)
print('[E7] Dry run OK — no OOM')
PYEOF
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] E7 dry run failed (OOM or other error). Check batch size."
    echo "[ABORT] Stopping AI-TOD queue."
    exit 1
fi
echo "[E7] Dry run passed. Starting full queue..."

# ── E8: AI-TOD Baseline ─────────────────────────────────────────
run_exp "aitod_baseline" "scripts/run_aitod_baseline.py"

# ── E9: AI-TOD + NWD k=0 ───────────────────────────────────────
run_exp "aitod_nwd_fixed" "scripts/run_aitod_nwd_fixed.py"

# ── E10: AI-TOD + SA-NWD ───────────────────────────────────────
run_exp "aitod_sanwd" "scripts/run_aitod_sanwd.py"

# ── E11: AI-TOD + SA-NWD + P2 ──────────────────────────────────
run_exp "aitod_sanwd_p2" "scripts/run_aitod_sanwd_p2.py"

# ── E12: AI-TOD + P2 only ──────────────────────────────────────
run_exp "aitod_p2only" "scripts/run_aitod_p2only.py"

echo ""
echo "============================================================"
echo "  AI-TOD Queue Complete: $(date)"
echo "============================================================"
echo ""
echo "  Results summary:"
echo "  ----------------"
for result_file in runs/ablation/aitod_*/result.json; do
    exp=$(basename $(dirname "$result_file"))
    map50=$(python3 -c "import json; d=json.load(open('$result_file')); print(d.get('map50','?'))" 2>/dev/null || echo "?")
    echo "  $exp: mAP50=$map50"
done
