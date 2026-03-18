# 无人机小目标检测项目 — 技术上下文

## 项目概述

实习任务：基于 YOLOv11n 的**反无人机**小目标检测改进算法，含论文撰写。
- 阶段1（3.14-3.21）：复现 YOLOv11n 基线 ✅
- 阶段2（3.21-4.21）：改进算法 + IEEE 论文
- 目标期刊：IEEE TGRS / IEEE GRSL（三区）

## 数据集

- 位置：`datasets/`（ultralytics 格式，单类别 `drone`）
- 划分：train 391 / val 111 / test 57，分辨率 1920×1080
- 核心特征：**极小目标**，归一化面积中位数 0.0023（~50×50 像素）
- 任务：检测空中飞行的无人机（非 VisDrone 那种从无人机俯拍地面）
- 已知问题：train/test 有相邻帧交叉污染（视频抽帧），所有方法同等受影响
- 数据集来源：**待确认**（审稿人必问，需在论文 §IV 补 citation）

## 最终方案

### 核心改进（有实验数据支撑）

| 组件 | 模块 | mAP50 贡献 |
|------|------|-----------|
| SA-NWD Loss（hybrid α=0.5） | `patch_sa_nwd_loss` | +1.05% |
| P2 小目标检测头（stride=4） | yaml 架构 | +0.76% |

**最佳配置：SA-NWD + P2 → mAP50 = 0.9781**

### 已验证为负贡献（已归档）

| 组件 | mAP50 | 结论 |
|------|:-----:|------|
| +SimAM | 0.9519 | 391 张数据集过拟合 |
| +SimAM+PConv | 0.9749 | PConv 补回部分 |
| +BiFPN | 0.9365 | 过拟合 |
| Fisher-CIoU 系列 | 0.827~0.937 | 全部低于基线，SA-NWD 与 Fisher-CIoU 方向相反导致训练信号混乱 |
| SA-NWD-TAL 微调 | 震荡崩溃 | 在已收敛权重上不稳定 |

## 消融实验结果（runs/ablation/，统一 seed=0，300ep）

### 主消融链

| 步骤 | 配置 | mAP50 | P | R |
|------|------|:-----:|:---:|:---:|
| Baseline | YOLOv11n + CIoU | 0.9600 | 0.967 | 0.869 |
| +SA-NWD | 仅换 loss（k=1, α=0.5） | 0.9705 | 0.985 | 0.925 |
| **+SA-NWD+P2** | **加 P2 检测头** | **0.9781** | 0.968 | 0.935 |
| +SA-NWD+P2+SimAM | 加零参数注意力 | 0.9519 | 0.961 | 0.915 |
| +SA-NWD+P2+SimAM+PConv | 加轻量化 neck | 0.9749 | 0.956 | 0.935 |
| Full (+BiFPN) | 加特征融合 | 0.9365 | 0.949 | 0.872 |

### 自适应 C 消融（nwd_p2 架构，待填）

| 实验 | 配置 | mAP50 | P | R |
|------|------|:-----:|:---:|:---:|
| Exp D | Fixed NWD (k=0, C=const) | 待填 | — | — |
| Exp E | Reverse C（C 随尺度增大） | 待填 | — | — |

### k 超参数扫描（nwd_p2 架构，待填）

| k | mAP50 |
|---|:-----:|
| 0.0 | 待填（= Exp D） |
| 0.5 | 待填 |
| **1.0** | **0.9781**（已有） |
| 2.0 | 待填 |
| 3.0 | 待填 |

k-sweep 结果将以折线图（Figure）呈现于论文 Analysis 小节，不加入主消融表。

## 文件结构

```
无人机/
├── configs/
│   ├── data.yaml                        # 数据集配置（单类 drone）
│   └── yolo11n-improved.yaml            # 改进模型架构（P2+SimAM+PConv，已弃用）
│
├── ultralytics_modules/
│   ├── __init__.py                      # 模块导出（含新 reverse 函数）
│   ├── nwd.py                           # SA-NWD 全套实现 [核心]
│   ├── simam.py                         # SimAM 零参数注意力
│   └── pconv.py                         # PConv + PConv_C3k2
│
├── scripts/
│   ├── register_modules.py              # 自定义模块注册（必须最先 import）
│   ├── train_baseline.py                # 基线训练
│   ├── ablation.py                      # 主消融实验（6 组，子进程隔离）
│   ├── eval.py                          # 统一评估（精度+效率全指标）
│   ├── collect_results.py               # 汇总结果 → JSON + LaTeX
│   ├── plot_results.py                  # 论文图表
│   ├── plot_training_curves.py          # 训练曲线可视化
│   ├── generate_paper_figs.py           # 论文所需图表生成
│   ├── gradcam.py                       # Grad-CAM 热力图
│   ├── verify_error_distribution.py     # 验证 σ(s) ∝ √s（有 bug 待修）
│   ├── run_nwd_fixed.py                 # 实验 D：Fixed NWD (k=0)
│   ├── run_nwd_reverse.py               # 实验 E：Reverse C（Fisher 方向）
│   ├── run_k_sensitivity.py             # k 超参扫描（k=0.5/2.0/3.0）
│   └── run_new_queue.sh                 # 服务器串行队列脚本
│
├── paper/
│   ├── main.tex                         # IEEE 论文正文
│   ├── refs.bib                         # 参考文献（已清理假引用）
│   └── figs/                            # 已生成图表（pdf/png/csv）
│
├── docs/
│   ├── CLAUDE.md                        # 本文件：技术上下文
│   ├── 操作指南.md                       # 环境配置与实验操作
│   ├── 01_审稿回复模板.md
│   ├── 02_修稿与补实验计划.md
│   ├── 03_论文贡献表述模板.md
│   ├── 04_投稿改进计划.md               # 当前投稿任务清单
│   └── 04_按章节改稿提纲.md
│
├── datasets/                            # 数据集（不上传 git）
├── datasets.zip                         # 原始压缩包（不动）
├── runs/                                # 训练输出（不上传 git）
│
└── archive/                             # 已归档（失败/过时内容）
    ├── scripts/                         # 17 个过时脚本（含 Fisher 系列）
    ├── modules/                         # bifpn, repvgg, carafe, legacy/
    ├── sessions/                        # AI 会话记录（含决策上下文）
    │   ├── session-ses_2fff.md          # Fisher 失败分析、nwd_p2 选择过程
    │   └── session-ses_3068.md
    ├── results/
    │   └── baseline_640_eval.json
    ├── 远程GPU操作手册.md
    └── yolo11n-sota.yaml
```

## nwd.py 函数速查

### 活跃函数

| 函数 | 用途 | 关键参数 |
|------|------|---------|
| `sa_nwd(box1, box2)` | 计算 SA-NWD 相似度 [0,1] | `c_base=12.0, k=1.0` |
| `patch_sa_nwd_loss()` | 替换 BboxLoss：`α·SA-NWD + (1-α)·CIoU` | `c_base=12, k=1.0, alpha=0.5` |
| `patch_sa_nwd_tal()` | 替换 TAL：SA-NWD 替代 IoU | `c_base=12, k=1.0, nwd_min=0.3` |
| `patch_all_nwd()` | 便捷调用：loss + TAL | `k=1.0, alpha=0.5, nwd_min=0.3` |
| `sa_nwd_reverse(box1, box2)` | **Exp E**：C 随尺度增大（Fisher 理论方向对照） | `c_base=12.0, k=1.0, ref_area=0.002315` |
| `patch_sa_nwd_loss_reverse()` | **Exp E**：patch BboxLoss 用 reverse C | `c_base=12, k=1.0, alpha=0.5, ref_area=0.002315` |

### 已废弃函数（DEPRECATED）

| 函数 | 原用途 | 废弃原因 |
|------|--------|---------|
| `patch_scale_aware_loss()` | Fisher-Guided CIoU | 所有实验低于基线（0.827~0.937） |
| `patch_sa_nwd_fisher_loss()` | SA-NWD + Fisher-CIoU 组合 | 两者方向相反，组合更差 |

## SA-NWD 自适应 C 的设计逻辑

```
C_adapt = c_base * (1 + k / sqrt(avg_area))
```

**不是** Fisher 等变最优推导（那会给出 C∝s，方向相反）。

**实际动机**：标注噪声正则化。小目标标注不确定性大（1px 误差对 20px 目标影响远大于 200px），
大 C 防止 loss 过度惩罚位置噪声，起到尺度自适应正则化效果。

- 小目标（S 小）→ C 大 → loss 对位置误差更宽松 → 防止对标注噪声过拟合
- 大目标（S 大）→ C 趋近 c_base → 标准 NWD 行为

Exp D（k=0，固定 C）和 Exp E（反向 C）是验证这个设计选择的消融实验。

## 恒源云服务器环境

- RTX 3060 12GB，PyTorch 2.4.0+cu121，Python 3.11，ultralytics 8.4.22
- SSH：`ssh -p 21635 root@i-1.gpushare.com`（端口号可能变，以恒源云控制台为准）
- 项目路径：`/root/drone_detection/`
- 单个训练 batch=8 需 ~8.1GB，**不能并行两个训练**

### 服务器实验目录

```
/root/drone_detection/runs/ablation/
├── baseline/          mAP50=0.9600
├── nwd_only/          mAP50=0.9705  （stock arch，k=1）
├── nwd_p2/            mAP50=0.9781  ★ 当前最优
├── nwd_p2_simam/      mAP50=0.9519
├── nwd_p2_simam_pconv/ mAP50=0.9749
├── nwd_p2_pconv/      运行中（~ep80/300，alpha sweep 系列）
├── nwd_p2_fixed_c/    待运行（Exp D）
├── nwd_p2_reverse_c/  待运行（Exp E）
├── nwd_p2_k0p5/       待运行
├── nwd_p2_k2p0/       待运行
└── nwd_p2_k3p0/       待运行
```

### 新实验队列部署

当前服务器上旧 Fisher 队列（full_queue.sh screen session）需要先停掉，再部署新队列：

```bash
# 1. 杀掉旧 screen session（名称以实际为准）
screen -ls                        # 查看当前 session 名
screen -S full_queue -X quit      # 停掉旧队列

# 2. 同步代码到服务器
# （rsync 或 git pull，确保 scripts/ 和 ultralytics_modules/ 已更新）

# 3. 启动新队列（在 screen 里运行，约 42h）
screen -S queue
cd /root/drone_detection
bash scripts/run_new_queue.sh 2>&1 | tee logs/new_queue.log
```

## 训练参数（标准配置）

```python
# 所有消融实验统一配置
lr0=0.01, epochs=300, patience=100, warmup_epochs=5, batch=8,
imgsz=1280, cos_lr=True, mosaic=1.0, mixup=0.15, copy_paste=0.2,
seed=0, pretrained='yolo11n.pt'
```

## 论文修改记录（已完成）

| 问题 | 修改内容 | 状态 |
|------|---------|------|
| Theorem 2 跳步推导（C∝s→C∝√s 无依据） | 删除，改为正则化视角叙述 + Design Validation remark | ✅ |
| 摘要"providing stronger gradients" | 改为"scale-adaptive regularization" | ✅ |
| Introduction/Related Work 声称"唯一最优 C∝1/√s" | 改为"annotation-noise regularization 动机 + 实验验证" | ✅ |
| Theorem 1 术语"Fisher information" | 改为"expected squared gradient G_IoU" | ✅ |
| TAL 公式 α 符号冲突 | TAL 改用 γ/δ | ✅ |
| 消融表 BiFPN 数据矛盾 | 重组消融表，加 Exp D/E 占位行 | ✅ |
| SOTA 表假引用/占位符 | 改为"loss function comparison"，加 NWD 固定 C 行占位 | ✅ |
| Future Work 引用 VisDrone | 改为 Anti-UAV / Det-Fly | ✅ |
| refs.bib 3 条假引用 | 删除，新增 Anti-UAV 引用 | ✅ |

## 上下文快照（2026-03-19，第 1 轮）

- 当前进展：完成论文 9 处理论/数据矛盾修复；完成项目结构整理（归档 Fisher 系列脚本、session 文件、历史结果）；新建根目录 CLAUDE.md（全局工作规范）；新建 10 个目录 README；新增 Exp D/E/k-sweep 三组实验脚本；__init__.py 补充新函数导出；nwd.py 废弃函数加 DEPRECATED 注释
- 最新决定：论文理论从"Fisher 等变最优"改为"标注噪声正则化"视角；不使用 VisDrone（任务不匹配）；k-sweep 以折线图展示于 Analysis 小节，不加主消融表；Fisher-CIoU 全部归档
- 服务器/实验状态：新队列脚本（run_new_queue.sh）已就绪，待部署；旧 Fisher 队列需先 kill；nwd_p2_pconv 约 ep80/300 仍在运行
- 待完成：（1）服务器部署新队列；（2）Exp D/E/k-sweep 结果填入消融表；（3）调研公开数据集（Anti-UAV/Det-Fly）格式，确定一个进行跨数据集验证；（4）数据集来源 citation 补入论文 §IV

## 待完成清单

```
论文数据填入：
  [ ] Exp D 结果出来后填入消融表（Fixed NWD 行）
  [ ] Exp E 结果出来后填入消融表（Reverse C 行）
  [ ] SOTA 表 NWD 固定 C 行（= Exp D 结果）
  [ ] k-sweep 结果 → generate_paper_figs.py 生成折线图 → 加入 Analysis 小节

论文内容补充：
  [ ] 数据集来源 citation：确认数据集出处，§IV 补 \cite{} 引用（审稿人必问）
  [ ] Limitation 段：补充数据集规模说明 + 跨数据集泛化说明

公开数据集实验（投三区必须）：
  [ ] 调研候选数据集（Anti-UAV / Det-Fly / Drone-vs-Bird）
      → 确认标注格式（需 bbox，非 mask/track-only）
      → 确认数据规模和可下载性
  [ ] 确定一个数据集，转换为 ultralytics YOLO 格式
  [ ] 服务器：Baseline + SA-NWD (Ours)，各 300ep
  [ ] 论文 §V 新增"Cross-Dataset Generalization"小节

代码维护：
  [ ] 修复 verify_error_distribution.py（line 267: cfg.get(path,...) → cfg.get('path',...)）
```
