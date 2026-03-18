# 无人机小目标检测项目 — 技术上下文

## 项目概述

团队实习任务：基于 YOLOv11n 的无人机极小目标检测改进算法，含论文撰写。
- 阶段1（3.14-3.21）：复现 YOLOv11n 基线 ✅
- 阶段2（3.21-4.21）：改进算法 + IEEE 论文
- 目标期刊：IEEE TGRS（3区）

## 数据集

- 位置：`datasets/`（ultralytics 格式，单类别 `drone`）
- 划分：train 391 / val 111 / test 57，分辨率 1920×1080
- 核心特征：**极小目标**，归一化面积中位数 0.0023（~50×50 像素）
- 已知问题：train/test 有相邻帧交叉污染（视频抽帧），所有方法同等受影响

## 最终方案（实习版）

### 核心改进

| 组件 | 模块 | 效果 |
|------|------|------|
| SA-NWD Loss（hybrid alpha=0.5） | `patch_sa_nwd_loss` | +1.05% mAP50 |
| P2 小目标检测头（stride=4） | yaml 架构 | +0.76% mAP50 |

**最佳配置：NWD + P2 → mAP50 = 0.9781**

### 已验证为负贡献（已归档）

| 组件 | mAP50 | 变化 |
|------|:-----:|------|
| +SimAM | 0.9519 | -2.62% |
| +SimAM+PConv | 0.9749 | PConv 补回部分 |
| +BiFPN | 0.9365 | -3.84% |
| Fisher-CIoU 系列 | 0.827~0.937 | 全部低于基线 |
| SA-NWD-TAL 微调 | 震荡崩溃 | 训练不稳定 |

## 消融实验结果（runs/ablation/，统一 seed=0，300ep）

| 步骤 | 配置 | mAP50 | P | R |
|------|------|:-----:|:---:|:---:|
| Baseline | YOLOv11n | 0.9600 | 0.967 | 0.869 |
| +NWD | 仅换 loss+TAL | 0.9705 | 0.985 | 0.925 |
| **+NWD+P2** | **加 P2 检测头** | **0.9781** | 0.968 | 0.935 |
| +NWD+P2+SimAM | 加 SimAM | 0.9519 | 0.961 | 0.915 |
| +NWD+P2+SimAM+PConv | 加 PConv | 0.9749 | 0.956 | 0.935 |
| Full (+BiFPN) | 加 BiFPN | 0.9365 | 0.949 | 0.872 |

## 文件结构

```
无人机/
├── configs/
│   ├── data.yaml                      # 数据集配置（单类 drone）
│   └── yolo11n-improved.yaml          # 改进模型架构（P2+SimAM+PConv）
├── ultralytics_modules/
│   ├── __init__.py                    # 模块导出（仅活跃模块）
│   ├── nwd.py                         # SA-NWD 全套 patch [核心]
│   ├── simam.py                       # SimAM 零参数注意力
│   └── pconv.py                       # PConv + PConv_C3k2
├── scripts/
│   ├── register_modules.py            # 自定义模块注册（必须最先 import）
│   ├── train_baseline.py              # 基线训练
│   ├── ablation.py                    # 消融实验（6组，子进程隔离）
│   ├── eval.py                        # 统一评估（精度+效率全指标）
│   ├── collect_results.py             # 汇总结果 → JSON + LaTeX
│   ├── plot_results.py                # 论文图表
│   ├── plot_training_curves.py        # 训练曲线可视化
│   ├── gradcam.py                     # Grad-CAM 热力图
│   └── verify_error_distribution.py   # 验证 σ(s) ∝ √s（有 bug 待修）
├── paper/
│   ├── main.tex                       # IEEE 论文（占位符待填）
│   ├── refs.bib                       # 参考文献（3 条假引用待替换）
│   └── figs/                          # 待生成图表
├── docs/
│   ├── CLAUDE.md                      # 本文件
│   ├── 操作指南.md                     # 操作手册
│   ├── 01_审稿回复模板.md
│   ├── 02_修稿与补实验计划.md
│   ├── 03_论文贡献表述模板.md
│   ├── 04_投稿改进计划.md
│   └── 04_按章节改稿提纲.md
└── archive/                           # 已归档（失败/未使用的实验代码）
    ├── scripts/                       # 15 个脚本
    ├── modules/                       # bifpn, repvgg, carafe, legacy/
    └── yolo11n-sota.yaml              # SOTA 配置
```

## nwd.py 函数速查

| 函数 | 用途 | 关键参数 |
|------|------|---------|
| `sa_nwd(box1, box2)` | 计算 SA-NWD 相似度 [0,1] | `c_base=12.0, k=1.0` |
| `patch_sa_nwd_loss()` | 替换 BboxLoss：`α·SA-NWD + (1-α)·CIoU` | `alpha=0.5` |
| `patch_sa_nwd_tal()` | 替换 TAL：SA-NWD 替代 IoU | `nwd_min=0.3` |
| `patch_all_nwd()` | 便捷调用：loss + TAL | `k=1.0, alpha=0.5, nwd_min=0.3` |

## 恒源云环境

- RTX 3060 12GB，PyTorch 2.4.0+cu121，Python 3.11，ultralytics 8.4.22
- SSH：`ssh -p 21635 root@i-1.gpushare.com`
- 项目路径：`/root/drone_detection/`
- 单个训练 batch=8 需 ~8.1GB，不能并行两个训练
- 路径嵌套：`project='runs/detect'` 生成 `runs/detect/runs/detect/exp_name/`

### 服务器实验目录

- 消融实验：`/root/drone_detection/runs/ablation/{baseline,nwd_only,nwd_p2,...}/`
- 独立实验：`/root/drone_detection/runs/detect/runs/detect/{clean,fisher_only,...}/`

## 训练参数（标准配置）

```python
# 消融实验（从 yolo11n.pt）
lr0=0.01, epochs=300, patience=100, warmup_epochs=5, batch=8,
imgsz=1280, cos_lr=True, mosaic=1.0, mixup=0.15, copy_paste=0.2
```

## 待完成

- [ ] 修复 verify_error_distribution.py（line 267: `cfg.get(path, ...)` → `cfg.get('path', ...)`）
- [ ] 统一 eval 所有关键 best.pt
- [ ] TTA 评估（零成本试 nwd_p2 best.pt）
- [ ] 生成论文图表 → paper/figs/
- [ ] 论文填数（替换 XX.X 占位符）
- [ ] 替换 3 条假引用
- [ ] 删除论文中 BiFPN/SimAM 正面描述
- [ ] ONNX 导出 + FPS 测试
