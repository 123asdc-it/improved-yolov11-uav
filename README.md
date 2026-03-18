# 无人机小目标检测 — YOLOv11n 改进方案

基于 YOLOv11n 的无人机极小目标检测改进算法。核心贡献：**SA-NWD（Scale-Adaptive Normalized Wasserstein Distance）** + Fisher 信息理论推导 + SimAM 零参数注意力 + P2 检测头 + PConv 轻量化。包含完整的训练、评估、消融实验和 IEEE 论文模板。

---

## 快速上手

### 1. 环境安装

```bash
pip install ultralytics
```

### 2. 修改数据集路径

编辑 `configs/data.yaml`，将 `path` 改为数据集实际位置：

```yaml
path: /root/drone_detection/datasets  # 改为你的实际路径
```

### 3. 运行（所有命令在项目根目录执行）

```bash
# 基线训练
python scripts/train_baseline.py

# 主实验：单阶段 SA-NWD（无 Fisher，无 TAL）
python scripts/run_clean.py

# 消融实验（6组，自动顺序执行）
python scripts/ablation.py

# 统一评估
python scripts/eval.py --weights runs/ablation/baseline/weights/best.pt \
                                 runs/detect/clean/weights/best.pt \
                       --names "Baseline" "Ours"
```

---

## 项目结构

```
无人机/
├── configs/
│   ├── data.yaml                     # 数据集配置（单类 drone）
│   ├── yolo11n-improved.yaml         # 改进模型（P2+SimAM+PConv，无 BiFPN）
│   ├── yolo11n-sota.yaml             # SOTA 模型（+RepVGG+CARAFE，备用）
│   └── ablation/                     # 消融 YAML（自动生成）
│
├── ultralytics_modules/              # 自定义网络模块
│   ├── simam.py                      #   SimAM 零参数注意力（ICML 2021）[主用]
│   ├── nwd.py                        #   SA-NWD 全套 patch [核心]
│   ├── pconv.py                      #   PConv + PConv_C3k2（CVPR 2023）
│   ├── bifpn.py                      #   BiFPN_Concat [保留，sota 用]
│   ├── repvgg.py                     #   RepVGGBlock（CVPR 2021）[sota 用]
│   ├── carafe.py                     #   CARAFE（ICCV 2019）[sota 用]
│   ├── attention.py                  #   EMA/CA [旧方案，归档]
│   └── inner_iou.py                  #   Inner-IoU [旧方案，归档]
│
├── scripts/
│   ├── register_modules.py           # 自定义模块注册（必须最先 import）
│   ├── train_baseline.py             # 基线训练
│   ├── train_improved.py             # 改进模型（patch_all_nwd）
│   ├── train_sota.py                 # SOTA 模型
│   ├── ablation.py                   # 消融实验（6组）
│   │
│   ├── run_clean.py                  # [主实验] 单阶段 SA-NWD，从 yolo11n.pt
│   ├── run_fisher_only.py            # Fisher-CIoU only，从 best.pt 微调
│   ├── run_clean_fisher.py           # SA-NWD + Fisher-CIoU，从 best.pt 微调
│   ├── run_sa_nwd_tal.py             # SA-NWD + TAL（nwd_min bug 已修复）
│   ├── run_fisher_sequence.sh        # 自动串行等待队列
│   │
│   ├── eval.py                       # 统一评估（mAP/Params/FPS/FLOPs）
│   ├── collect_results.py            # 汇总实验结果 → JSON + LaTeX
│   ├── plot_results.py               # 生成论文图表
│   ├── gradcam.py                    # Grad-CAM 热力图
│   ├── slice_dataset.py              # SAHI 切片增强
│   ├── augment_copy_paste.py         # Copy-Paste 增强
│   └── verify_error_distribution.py  # 验证 σ(s) ∝ √s 理论假设
│
├── paper/
│   ├── main.tex                      # IEEE Journal 论文
│   └── refs.bib                      # 参考文献
│
├── docs/
│   ├── CLAUDE.md                     # 技术上下文（内部）
│   └── 操作指南.md                    # 详细操作手册
│
└── datasets/
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/
```

---

## 改进方案

### 理论基础

我们从 Fisher 信息角度分析了 IoU loss 对小目标的梯度消失问题：

```
I_IoU(s) ∝ s⁻²   →  小目标梯度信号消失
    ↓ 理论推导
C*(s) ∝ √s        →  Fisher 等变的最优 NWD 自适应常数
    ↓ 工程实现
SA-NWD + Fisher-CIoU
```

### 核心组件（yolo11n-improved.yaml）

| 组件 | 模块 | 论文 | 作用 |
|------|------|------|------|
| SA-NWD Loss | `patch_sa_nwd_loss` | ISPRS 2022 + 本文 | Wasserstein 度量，scale-adaptive C |
| SA-NWD-TAL | `patch_sa_nwd_tal` | TOOD 2021 + 本文 | scale-equivariant 标签分配 |
| Fisher-Guided CIoU | `patch_scale_aware_loss` | 本文 | w(s)=√(ref/s) 补偿小目标梯度 |
| P2 检测头 | 4-head Detect | — | stride=4，覆盖极小目标 |
| SimAM 注意力 | `SimAM` | ICML 2021 | 零参数，不增加过拟合风险 |
| PConv 轻量化 | `PConv_C3k2` | CVPR 2023 | neck 计算量降低 ~20% |

> **BiFPN 已移除**：消融实验（0.9373 vs 0.9666）证实负贡献，当前 improved.yaml 使用标准 Concat。

### 消融实验结果

| 实验 | mAP50 | 说明 |
|------|:-----:|------|
| Baseline（YOLOv11n） | 0.929 | 官方预训练权重 fine-tune |
| +NWD | ~0.94 | 仅换 loss，无结构改变 |
| +NWD+P2 | ~0.95 | 加 P2 检测头 |
| +NWD+P2+SimAM | ~0.96 | 加零参数注意力 |
| **+NWD+P2+SimAM+PConv** | **0.9666** | 当前最优 |
| +Fisher-CIoU | 待定 | run_fisher_only 进行中 |
| +SA-NWD+Fisher | 待定 | run_clean_fisher 进行中 |
| +SA-NWD+TAL | 待定 | run_sa_nwd_tal 进行中 |

### nwd.py 核心函数

```python
from ultralytics_modules.nwd import (
    patch_sa_nwd_loss,         # SA-NWD hybrid loss
    patch_sa_nwd_tal,          # SA-NWD 标签分配（nwd_min=0.3）
    patch_scale_aware_loss,    # Fisher-Guided CIoU
    patch_sa_nwd_fisher_loss,  # SA-NWD + Fisher 合并
    patch_all_nwd,             # 便捷调用（loss + TAL）
)

# 推荐用法（主实验）
patch_sa_nwd_loss(c_base=12.0, k=1.0, alpha=0.5)

# 全流程（含 TAL）
patch_all_nwd(c_base=12.0, k=1.0, alpha=0.5, nwd_min=0.3)
```

---

## 数据集

- **格式**：Ultralytics YOLO（归一化 xywh）
- **类别**：1 类（drone）
- **划分**：train 391 / val 111 / test 57
- **分辨率**：1920×1080
- **特点**：极小目标，归一化面积中位数 0.0023（约 50×50 像素）

---

## 数据增强策略

### SAHI 切片训练

```bash
python scripts/slice_dataset.py
# 生成 datasets_sliced/ + configs/data_sliced.yaml
```

### Copy-Paste 增强

```bash
python scripts/augment_copy_paste.py
# 生成 datasets_augmented/
```

---

## 注意事项

- `register_modules.py` 在各脚本头部自动 import，**必须在 ultralytics 之前 import**
- 消融实验使用独立子进程，避免 CUDA 上下文污染
- `patch_sa_nwd_tal` 必须设置 `nwd_min=0.3`，默认值已修复（之前写死 0.01 导致 P 崩溃）
- GPU 内存：batch=8 需要 ~8.1GB，不能并行两个训练进程

---

## 论文编译

上传 `paper/` 文件夹到 [Overleaf](https://www.overleaf.com/)，选择 pdfLaTeX 编译器即可。

论文标题：*SA-NWD: Scale-Adaptive Wasserstein Distance with Fisher-Guided Training for UAV-Based Small Object Detection*
