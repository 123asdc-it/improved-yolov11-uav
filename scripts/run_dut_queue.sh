#!/bin/bash
# run_dut_queue.sh — DUT Anti-UAV experiment queue
#
# Queue (serial, ~8h each, total ~32h):
#   1. dut_baseline    YOLOv11n + CIoU (no NWD)
#   2. dut_nwd_fixed   + NWD k=0 (fixed C, standard NWD)
#   3. dut_sanwd       + SA-NWD k=1.0 (no P2)
#   4. dut_sanwd_p2    + SA-NWD k=1.0 + P2 head  ← final method
#
# Prerequisites:
#   datasets/dut/ must exist (run convert_dut_to_yolo.py first)
#
# Usage (on server):
#   screen -S dut_queue
#   cd /root/drone_detection
#   bash scripts/run_dut_queue.sh 2>&1 | tee logs/dut_queue.log

# Note: no set -e — individual failures handled by run_exp()
cd /root/drone_detection

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Check dataset exists
if [ ! -f "datasets/dut/dut_data.yaml" ]; then
    echo "[ERROR] datasets/dut/dut_data.yaml not found."
    echo "        Run: python scripts/convert_dut_to_yolo.py first."
    exit 1
fi

echo "============================================================"
echo "  DUT Anti-UAV Experiment Queue"
echo "  Start: $(date)"
echo "============================================================"

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
        echo "[ERROR] $name failed (exit $exit_code), continuing..."
    else
        echo "[OK] $name done at $(date)"
    fi
}

run_exp "dut_baseline"   "scripts/run_dut_baseline.py"
run_exp "dut_nwd_fixed"  "scripts/run_dut_nwd_fixed.py"
run_exp "dut_sanwd"      "scripts/run_dut_sanwd.py"
run_exp "dut_sanwd_p2"   "scripts/run_dut_sanwd_p2.py"

echo ""
echo "============================================================"
echo "  DUT Queue Complete: $(date)"
echo "============================================================"
echo ""
echo "  Results:"
for f in runs/ablation/dut_*/result.json; do
    exp=$(basename $(dirname "$f"))
    map50=$(python3 -c "import json; print(json.load(open('$f')).get('map50','?'))" 2>/dev/null || echo "?")
    echo "  $exp: mAP50=$map50"
done
