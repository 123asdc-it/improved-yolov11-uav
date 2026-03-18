#!/bin/bash
# Fisher-CIoU 修正实验队列
# 等待前面的实验全部完成后，依次跑 A → B → C

cd /root/drone_detection

echo "=== [$(date)] Waiting for alpha_sweep to finish ==="
while pgrep -f 'run_alpha_exp' > /dev/null 2>&1; do sleep 120; done
while pgrep -f 'alpha_sweep' > /dev/null 2>&1; do sleep 60; done
echo "=== [$(date)] All previous experiments done ==="

echo "=== [$(date)] Starting Fisher Exp A: SA-NWD + Fisher-CIoU (fixed) ==="
python scripts/run_fisher_ablation.py --exp A >> fisher_ablation_log.txt 2>&1
echo "=== [$(date)] Exp A done ==="

echo "=== [$(date)] Starting Fisher Exp B: SA-NWD + Fisher-CIoU (wide params) ==="
python scripts/run_fisher_ablation.py --exp B >> fisher_ablation_log.txt 2>&1
echo "=== [$(date)] Exp B done ==="

echo "=== [$(date)] Starting Fisher Exp C: Pure Fisher-CIoU (no SA-NWD) ==="
python scripts/run_fisher_ablation.py --exp C >> fisher_ablation_log.txt 2>&1
echo "=== [$(date)] Exp C done ==="

echo "=== [$(date)] ALL FISHER EXPERIMENTS DONE ===" >> fisher_ablation_log.txt

# 提取结果
echo ""
echo "=== FISHER ABLATION RESULTS ==="
grep "^RESULT:" fisher_ablation_log.txt
