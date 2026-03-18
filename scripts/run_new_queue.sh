#!/bin/bash
# run_new_queue.sh — New experiment queue (replaces Fisher A/B/C)
#
# Queue order (serial, ~6h each, total ~42h):
#   [already running] nwd_p2_pconv     (~4h remaining)
#   [kept]  alpha=0.3                  300ep
#   [kept]  alpha=0.7                  300ep
#   [new]   D: Fixed NWD (k=0)         300ep  ← replaces Fisher A
#   [new]   E: Reverse C               300ep  ← replaces Fisher B
#   [new]   k=0.5                      300ep  ← replaces Fisher C
#   [new]   k=2.0                      300ep
#   [new]   k=3.0                      300ep
#
# k=1.0 (nwd_p2, mAP50=0.9781) already done — NOT re-run.
# alpha sweep uses nwd_p2 architecture (ablation_nwd_p2.yaml).
#
# Usage (on server, after killing old screen session):
#   screen -S queue
#   cd /root/drone_detection
#   bash scripts/run_new_queue.sh 2>&1 | tee logs/new_queue.log
#
# Kill old session first:
#   screen -S full_queue -X quit   (or whatever the session name is)

set -e
cd /root/drone_detection

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  New Experiment Queue"
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
        echo "[ERROR] $name failed with exit code $exit_code"
        echo "[WARN]  Continuing with next experiment..."
    else
        echo "[OK] $name finished at $(date)"
    fi
}

# ── Alpha sweep (kept from original queue) ──────────────────────
# These use nwd_p2 architecture with different alpha blend weights.
# Scripts should already exist from previous queue planning.
# If not present, they are skipped with a warning.

for alpha_script in scripts/run_alpha_0p3.py scripts/run_alpha_0p7.py; do
    if [ -f "$alpha_script" ]; then
        name=$(basename "$alpha_script" .py)
        run_exp "$name" "$alpha_script"
    else
        echo "[SKIP] $alpha_script not found, skipping alpha sweep entry"
    fi
done

# ── New ablation experiments ────────────────────────────────────

# Experiment D: Fixed NWD (k=0)
run_exp "nwd_p2_fixed_c" "scripts/run_nwd_fixed.py"

# Experiment E: Reverse SA-NWD (C grows with scale)
run_exp "nwd_p2_reverse_c" "scripts/run_nwd_reverse.py"

# k sensitivity: k=0.5, 2.0, 3.0 (k=1.0 already done)
run_exp "k_sensitivity" "scripts/run_k_sensitivity.py"

echo ""
echo "============================================================"
echo "  All experiments done: $(date)"
echo "============================================================"

# Print summary of all result.json files
echo ""
echo "  Results summary:"
echo "  ----------------"
for result_file in runs/ablation/*/result.json; do
    exp=$(basename $(dirname "$result_file"))
    map50=$(python3 -c "import json; d=json.load(open('$result_file')); print(d.get('map50','?'))" 2>/dev/null || echo "?")
    echo "  $exp: mAP50=$map50"
done
