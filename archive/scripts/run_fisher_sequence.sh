#!/bin/bash
# run_fisher_sequence.sh
# 等待 run_clean 结束后，串行执行所有实验：
#   1. fisher_only    — 纯 Fisher-CIoU，无 SA-NWD
#   2. clean_fisher   — SA-NWD + Fisher-CIoU 叠加
#   3. sa_nwd_tal     — SA-NWD Loss + SA-NWD-TAL（TAL bug 已修复）
# 在服务器上后台运行：nohup bash scripts/run_fisher_sequence.sh >> fisher_seq_log.txt 2>&1 &

set -e
cd /root/drone_detection

echo "=== [$(date)] Waiting for run_clean to finish ==="
while pgrep -f "run_clean.py" > /dev/null 2>&1; do
    sleep 60
done

echo "=== [$(date)] run_clean finished. Starting fisher_only ==="
python /root/drone_detection/scripts/run_fisher_only.py >> /root/drone_detection/fisher_only_log.txt 2>&1
echo "=== [$(date)] fisher_only done ==="

echo "=== [$(date)] Starting clean_fisher ==="
python /root/drone_detection/scripts/run_clean_fisher.py >> /root/drone_detection/clean_fisher_log.txt 2>&1
echo "=== [$(date)] clean_fisher done ==="

echo "=== [$(date)] Starting sa_nwd_tal (full pipeline, fixed nwd_min) ==="
python /root/drone_detection/scripts/run_sa_nwd_tal.py >> /root/drone_detection/sa_nwd_tal_log.txt 2>&1
echo "=== [$(date)] sa_nwd_tal done. All experiments complete ==="
