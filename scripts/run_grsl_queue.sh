#!/bin/bash
# run_grsl_queue.sh — GRSL 最小补丁队列（服务器丢失后重跑）
#
# 对应：docs/06_GRSL最小补丁方案.md 的套餐 B（推荐版）
#
# 跑什么（串行，单卡）：
#   Tier 1（必须，~12h）：
#     1. Exp D — Fixed NWD (k=0)                  ~6h
#     2. Exp E — Reverse C                        ~6h
#     3. FPS 重测（用现有 nwd_p2/best.pt）         ~5min
#
#   Tier 2（强烈建议，~18h）：
#     4. k-sensitivity — 只跑 k=0.5 和 k=2.0        ~12h
#     5. E_p2only — P2 only 无 SA-NWD              ~6h
#
#   总时长：~30h（含 FPS 测量 5 分钟，可忽略）
#
# 使用方法（已 setup_new_server.sh 跑过）：
#   screen -S grsl
#   cd /root/drone_detection
#   conda activate drone
#   bash scripts/run_grsl_queue.sh 2>&1 | tee logs/grsl_queue.log
#   # Ctrl-A, D 分离；重新连接：screen -r grsl
#
# 中途只想跑 Tier 1（套餐 A，省钱版）：
#   TIER=1 bash scripts/run_grsl_queue.sh
#
# 跳过某个实验（例如已跑过 Exp D）：
#   SKIP_D=1 bash scripts/run_grsl_queue.sh
# 或直接让 run_nwd_fixed.py 里的 WEIGHT_PATH 检查逻辑自动 skip

set -u
cd /root/drone_detection

TIER="${TIER:-2}"           # 默认跑 Tier 1+2（套餐 B）；设 TIER=1 只跑 Tier 1（套餐 A）
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  GRSL 最小补丁队列"
echo "  TIER=$TIER（1=仅 Tier 1, 2=Tier 1+2）"
echo "  开始：$(date)"
echo "============================================================"

# 统一的实验执行函数（继承自 run_new_queue.sh 的风格）
run_exp() {
    local name="$1"
    local script="$2"
    local skip_var="${3:-}"

    if [ -n "$skip_var" ] && [ "${!skip_var:-}" = "1" ]; then
        echo ""
        echo "[SKIP] $name（环境变量 $skip_var=1）"
        return 0
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "  START: $name  [$(date)]"
    echo "------------------------------------------------------------"
    python "$script" 2>&1 | tee "$LOG_DIR/${name}.log"
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo "[ERROR] $name 失败（exit=$exit_code），继续下一个"
    else
        echo "[OK] $name 完成于 $(date)"
    fi
}

# ╔═══════════════════════════════════════════════════════════╗
# ║                   Tier 1（必须）                           ║
# ╚═══════════════════════════════════════════════════════════╝

# 1. Exp D：Fixed NWD (k=0)
run_exp "nwd_p2_fixed_c" "scripts/run_nwd_fixed.py" "SKIP_D"

# 2. Exp E：Reverse C
run_exp "nwd_p2_reverse_c" "scripts/run_nwd_reverse.py" "SKIP_E"

# 3. FPS 重测（用现有最优权重，不需要重训练）
echo ""
echo "------------------------------------------------------------"
echo "  START: FPS 重测（nwd_p2/best.pt）  [$(date)]"
echo "------------------------------------------------------------"
if [ -f "runs/ablation/nwd_p2/weights/best.pt" ]; then
    python scripts/eval.py \
        --weights runs/ablation/nwd_p2/weights/best.pt \
        --names "SA-NWD+P2" \
        --imgsz 1280 \
        2>&1 | tee "$LOG_DIR/fps_eval.log"
    echo "[OK] FPS 测量完成"
else
    echo "[WARN] runs/ablation/nwd_p2/weights/best.pt 不存在，本地 rsync 是否包含 runs/？"
    echo "       备用方案：本地 Mac 用 MPS 测近似 FPS，论文标注 GPU 型号待补"
fi

# ╔═══════════════════════════════════════════════════════════╗
# ║                   Tier 2（强烈建议）                       ║
# ╚═══════════════════════════════════════════════════════════╝

if [ "$TIER" -ge 2 ]; then
    # 4. k-sensitivity，只跑 k=0.5 和 k=2.0（GRSL 裁剪版）
    #    通过环境变量 GRSL_K_VALUES 控制，避免修改 run_k_sensitivity.py 本身
    echo ""
    echo "------------------------------------------------------------"
    echo "  START: k-sensitivity (k=0.5, 2.0)  [$(date)]"
    echo "------------------------------------------------------------"
    GRSL_K_VALUES="0.5,2.0" python scripts/run_k_sensitivity.py 2>&1 | tee "$LOG_DIR/k_sensitivity.log"
    echo "[OK] k-sensitivity 完成于 $(date)"

    # 5. E_p2only：P2 only，无 SA-NWD
    run_exp "p2_only" "scripts/run_p2_only.py" "SKIP_P2ONLY"
fi

# ╔═══════════════════════════════════════════════════════════╗
# ║                   结果汇总                                  ║
# ╚═══════════════════════════════════════════════════════════╝

echo ""
echo "============================================================"
echo "  队列完成于 $(date)"
echo "============================================================"
echo ""
echo "  结果汇总："
echo "  ----------------"
for result_file in runs/ablation/*/result.json; do
    if [ -f "$result_file" ]; then
        exp=$(basename $(dirname "$result_file"))
        map50=$(python -c "import json; d=json.load(open('$result_file')); print(d.get('map50','?'))" 2>/dev/null || echo "?")
        map=$(python -c "import json; d=json.load(open('$result_file')); print(d.get('map','?'))" 2>/dev/null || echo "?")
        echo "  $exp: mAP50=$map50 mAP50-95=$map"
    fi
done

if [ -f "runs/ablation/k_sensitivity_results.json" ]; then
    echo ""
    echo "  k-sensitivity："
    python -c "
import json
with open('runs/ablation/k_sensitivity_results.json') as f:
    for r in json.load(f):
        print(f\"    k={r.get('k','?'):<5} mAP50={r.get('map50','?')}\")
" 2>/dev/null || echo "    (无法解析 k_sensitivity_results.json)"
fi

echo ""
echo "  下一步：rsync runs/ablation/ 回本地，填入 paper/main.tex Table 1"
echo "  rsync 命令（本地执行）："
echo "    rsync -avz --include='*/' --include='result.json' --include='results.csv' \\"
echo "        --include='weights/best.pt' --exclude='*' \\"
echo "        'root@server:/root/drone_detection/runs/ablation/' ./runs/ablation/"
