# 无人机小目标检测 — YOLOv11n + SA-NWD

基于 YOLOv11n 的**反无人机**极小目标检测改进方案。
核心贡献：**SA-NWD（Scale-Adaptive Normalized Wasserstein Distance）** + **P2 小目标检测头**。
目标期刊：IEEE GRSL / IEEE TGRS（三区）。

---

## 快速上手

### 1. 环境安装

```bash
pip install ultralytics
```

### 2. 修改数据集路径

编辑 `configs/data.yaml`，将 `path` 改为数据集实际位置：

```yaml
path: /root/drone_detection/datasets  # 服务器绝对路径
```

### 3. 运行（所有命令在项目根目录执行）

```bash
# 基线训练
python scripts/train_baseline.py

# 主消融实验（6组，串行子进程，~36h）
python scripts/ablation.py

# 服务器完整队列（Exp D/E + k-sweep，~42h）
bash scripts/run_new_queue.sh 2>&1 | tee logs/new_queue.log

# 统一评估
python scripts/eval.py --weights \
    runs/ablation/baseline/weights/best.pt \
    runs/ablation/nwd_p2/weights/best.pt \
    --names "Baseline" "SA-NWD+P2"
```

---

## 项目结构

```
无人机/
├── CLAUDE.md                          # 全局工作规范（归档/Git/README 约定）
├── configs/
│   ├── data.yaml                      # 数据集配置（单类 drone，559 张）
│   └── yolo11n-improved.yaml          # 改进架构（P2+SimAM+PConv，参考用）
│
├── ultralytics_modules/               # 自定义模块（monkey-patch 注入）
│   ├── nwd.py                         # SA-NWD 核心实现 ★
│   ├── simam.py                       # SimAM 零参数注意力
│   ├── pconv.py                       # PConv + PConv_C3k2
│   └── __init__.py                    # 模块导出
│
├── scripts/                           # 活跃实验脚本（14 个）
│   ├── train_baseline.py              # 基线训练
│   ├── ablation.py                    # 主消融实验（6 组）
│   ├── run_nwd_fixed.py               # Exp D：Fixed NWD (k=0)
│   ├── run_nwd_reverse.py             # Exp E：Reverse C 方向验证
│   ├── run_k_sensitivity.py           # k 超参扫描（k=0.5/2.0/3.0）
│   ├── run_alpha_0p3.py               # alpha=0.3 消融
│   ├── run_alpha_0p7.py               # alpha=0.7 消融
│   ├── run_new_queue.sh               # 服务器串行队列
│   ├── eval.py                        # 统一评估
│   ├── collect_results.py             # 汇总结果 → JSON + LaTeX
│   ├── generate_paper_figs.py         # 论文图表生成
│   ├── plot_results.py                # 消融柱状图 + PR 曲线
│   ├── plot_training_curves.py        # 训练曲线
│   ├── gradcam.py                     # Grad-CAM 热力图
│   ├── register_modules.py            # 自定义模块注册（必须最先 import）
│   └── verify_error_distribution.py   # 理论假设验证（有 bug 待修）
│
├── paper/
│   ├── main.tex                       # IEEE 论文正文（IEEEtran）
│   ├── refs.bib                       # 参考文献
│   └── figs/                          # 图表文件（pdf/png/csv）
│
├── docs/                              # 项目文档
│   ├── CLAUDE.md                      # 技术上下文（实验结果/待完成清单）
│   ├── 操作指南.md                     # 详细操作手册
│   └── 04_投稿改进计划.md              # 投稿任务清单（当前状态）
│
├── datasets/                          # 数据集（不上传 git）
├── runs/                              # 训练输出（不上传 git）
│
└── archive/                           # 归档（失败/过时实验）
    ├── scripts/                       # 17 个归档脚本（含 Fisher 系列）
    ├── modules/                       # bifpn/carafe/repvgg/legacy
    ├── sessions/                      # AI 决策记录（含 Fisher 失败分析）
    └── results/                       # 历史评估结果
```

---

## 核心方案

### SA-NWD（最终方案）

将 bbox 建模为 2D 高斯，用 Wasserstein-2 距离替代 IoU，自适应常数 C 基于标注噪声正则化设计：

```
C_adapt = C_base × (1 + k / √S̄)
```

小目标 S 小 → C 大 → loss 对位置误差更宽松 → 防止对标注噪声过拟合。

混合损失：`L = α · (1 − SA-NWD) + (1−α) · CIoU`

### 消融实验结果

| 实验 | mAP50 | 说明 |
|------|:-----:|------|
| Baseline（YOLOv11n + CIoU） | 0.9600 | 对比基准 |
| +SA-NWD（k=1, α=0.5） | 0.9705 | +1.05% |
| **+SA-NWD + P2 Head** | **0.9781** | **+1.81%，当前最优** |
| +SimAM | 0.9519 | 负贡献（小数据集过拟合） |
| +SimAM+PConv | 0.9749 | 仍低于纯 nwd_p2 |
| +BiFPN（Full） | 0.9365 | 负贡献 -4.16% |
| Exp D: Fixed NWD (k=0) | 待填 | 验证自适应 C 贡献 |
| Exp E: Reverse C | 待填 | 验证方向选择合理性 |

### 核心函数

```python
from ultralytics_modules.nwd import patch_all_nwd

# 推荐用法（主实验）
patch_all_nwd(c_base=12.0, k=1.0, alpha=0.5, nwd_min=0.3)
```

> `patch_scale_aware_loss` 和 `patch_sa_nwd_fisher_loss` 已废弃（实验证实负贡献），禁止在新实验中使用。

---

## 数据集

- **任务**：检测空中飞行的无人机（反无人机场景，非 VisDrone 的从无人机俯拍地面）
- **格式**：Ultralytics YOLO（归一化 xywh）
- **类别**：1 类（drone）
- **划分**：train 391 / val 111 / test 57
- **分辨率**：1920×1080
- **特点**：极小目标，归一化面积中位数 0.0023（约 50×50 像素）

---

## 论文编译

上传 `paper/` 文件夹到 [Overleaf](https://www.overleaf.com/)，选择 **pdfLaTeX** 编译器。

详细操作见 `paper/README.md`。

---

## 注意事项

- `register_modules.py` **必须在 `import ultralytics` 之前执行**，各训练脚本已自动 import
- `nwd_min=0.3` 必须设置，默认值已修复（之前写死 0.01 导致 P 崩溃）
- GPU 内存：batch=8 需要约 8.1GB，**不能并行两个训练进程**
- 服务器 `configs/data.yaml` 的 `path` 必须改为绝对路径
