#!/bin/bash
# setup_new_server.sh — 新服务器一键环境准备（GRSL 最小补丁方案 Day 1 上午）
#
# 前置要求：
#   - 本地已恢复 datasets/ 目录（从学长/外接盘）
#   - 服务器已开机，ssh 能通
#   - 服务器上 /root/drone_detection 不存在或为空
#
# 使用流程（两端配合）：
#
# 【本地执行（Mac 端）】— 先上传代码和数据：
#   cd ~/Work/Project/无人机
#   SERVER="root@i-X.gpushare.com -p <port>"
#   rsync -avz --exclude runs --exclude datasets.zip --exclude archive \
#       ./{configs,scripts,ultralytics_modules,yolo11n.pt,paper,docs} \
#       "$SERVER:/root/drone_detection/"
#   rsync -avz ./datasets/ "$SERVER:/root/drone_detection/datasets/"
#
# 【服务器执行】— 然后 ssh 过去跑这个脚本：
#   ssh "$SERVER"
#   cd /root/drone_detection
#   bash scripts/setup_new_server.sh
#
# 整个流程预计 30 分钟（含数据上传时间，取决于网速）。
# 完成后可以直接 bash scripts/run_grsl_queue.sh 启动 Tier 1+2 实验。

set -u  # 不用 set -e，允许个别步骤失败时继续，由人工判断
cd /root/drone_detection

echo "============================================================"
echo "  SA-NWD Drone Detection — 新服务器环境准备"
echo "  时间: $(date)"
echo "============================================================"

# ── 0. 前置检查 ─────────────────────────────────────────────
echo ""
echo "[0/5] 前置检查..."
if [ ! -f yolo11n.pt ]; then
    echo "[ERROR] yolo11n.pt 不存在，本地 rsync 是否成功？"
    exit 1
fi
if [ ! -d datasets/images/train ]; then
    echo "[ERROR] datasets/images/train 不存在，数据集 rsync 是否完成？"
    exit 1
fi
if [ ! -f configs/data.yaml ]; then
    echo "[ERROR] configs/data.yaml 不存在"
    exit 1
fi
if [ ! -f ultralytics_modules/nwd.py ]; then
    echo "[ERROR] ultralytics_modules/nwd.py 不存在"
    exit 1
fi

# 统计数据集
TRAIN_COUNT=$(ls datasets/images/train 2>/dev/null | wc -l)
VAL_COUNT=$(ls datasets/images/val 2>/dev/null | wc -l)
TEST_COUNT=$(ls datasets/images/test 2>/dev/null | wc -l)
echo "  数据集图片数：train=$TRAIN_COUNT, val=$VAL_COUNT, test=$TEST_COUNT"
echo "  预期：train=391, val=111, test=57"
echo "  前置检查通过 ✓"

# ── 1. GPU 检查 ─────────────────────────────────────────────
echo ""
echo "[1/5] GPU 检查..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[WARN] nvidia-smi 不可用，确认是否在 GPU 实例上"
fi

# ── 2. Conda 环境 ────────────────────────────────────────────
echo ""
echo "[2/5] Conda 环境准备..."
ENV_NAME="drone"

# 初始化 conda（某些实例需要）
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda 不可用，请先安装 Miniconda 或 Anaconda"
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "  环境 $ENV_NAME 已存在，激活"
else
    echo "  创建环境 $ENV_NAME (python=3.11)"
    conda create -n "$ENV_NAME" python=3.11 -y
fi
conda activate "$ENV_NAME"
echo "  当前环境：$CONDA_DEFAULT_ENV"
echo "  Python：$(python --version)"

# ── 3. 依赖安装 ─────────────────────────────────────────────
echo ""
echo "[3/5] 依赖安装..."
# ultralytics 版本锁定 8.4.22（与原服务器一致，避免 API 漂移）
pip install ultralytics==8.4.22 -q
# torch 通常随 ultralytics 装好；若 CUDA 版本不对手动重装
python -c "import torch; print(f'  torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import ultralytics; print(f'  ultralytics: {ultralytics.__version__}')"

# ── 4. 日志目录 ─────────────────────────────────────────────
echo ""
echo "[4/5] 建日志和实验输出目录..."
mkdir -p logs runs/ablation
echo "  logs/ 和 runs/ablation/ 已就绪"

# ── 5. 1-epoch 冒烟测试（可选，用 --skip-smoke 跳过）───────────
echo ""
echo "[5/5] 1-epoch 冒烟测试（确认训练流程能跑通）..."
if [ "${1:-}" = "--skip-smoke" ]; then
    echo "  [SKIP] 用户要求跳过冒烟测试"
else
    # 用 nwd_p2 架构跑 1 epoch，看 loss 是否正常下降
    python -c "
import sys
sys.path.insert(0, 'scripts')
import register_modules
from ultralytics_modules.nwd import patch_all_nwd
patch_all_nwd(c_base=12.0, k=1.0, alpha=0.5, nwd_min=0.3)
from ultralytics import YOLO
m = YOLO('configs/ablation/ablation_nwd_p2.yaml')
m.train(
    data='configs/data.yaml',
    imgsz=1280, epochs=1, batch=8, lr0=0.01,
    pretrained='yolo11n.pt',
    project='runs/ablation', name='smoke_test', exist_ok=True,
    seed=0, workers=4, cache=False,
)
print('[SMOKE OK] 1-epoch 训练完成，可以启动完整实验')
" 2>&1 | tail -20
    # 冒烟测试完成后删除 smoke_test 目录避免污染
    rm -rf runs/ablation/smoke_test
fi

# ── 结束 ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  环境准备完成 ✓"
echo "  下一步：screen -S grsl && bash scripts/run_grsl_queue.sh"
echo "============================================================"
