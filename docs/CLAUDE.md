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
├── runs/                                # 训练输出（不入 git，本地备份）
│   └── ablation/                        # 7 个主消融实验（含权重和图表）
│       ├── baseline/weights/best.pt
│       ├── nwd_only/weights/best.pt
│       ├── nwd_p2/weights/best.pt       ★ 最优模型（5.9MB）
│       ├── nwd_p2_simam/weights/best.pt
│       ├── nwd_p2_simam_pconv/weights/best.pt
│       ├── full_improved/weights/best.pt
│       └── nwd_p2_pconv/weights/best.pt
├── eval_results.json                    # 效率指标（FPS=59.4，Params），入 git
├── yolo11n.pt                           # 预训练权重（不入 git）
├── yolo26n.pt                           # 预训练权重（不入 git）
│
└── archive/                             # 已归档（失败/过时内容）
    ├── scripts/                         # 17 个过时脚本（含 Fisher 系列）
    ├── modules/                         # bifpn, repvgg, carafe, legacy/
    │   └── legacy/
    │       ├── __init__.py
    │       ├── attention.py             # EMA / CA 注意力（已归档）
    │       └── inner_iou.py             # Inner-IoU 损失（已归档）
    ├── runs/                            # 归档历史实验（详见 archive/runs/README.md）
    │   ├── detect/                      # 早期 detect 实验（20 个）
    │   ├── early_ablation/              # 早期架构消融（8 个）
    │   ├── sa_nwd/                      # 早期 SA-NWD 实验（2 个）
    │   ├── val_outputs/                 # 48 个评估输出
    │   └── eval_640/                    # 640 分辨率基线评估
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

## 对话计数器

```
当前计数：1 / 5
上次快照：2026-03-19，第 2 轮

每 5 轮对话后触发快照，计数归零。
```

## 重要注意事项

- `runs/` 和 `*.pt` **不入 git**（`.gitignore` 规则覆盖），本地是唯一副本，服务器是训练来源
- `configs/ablation/` 和 `eval_results.json` 已加 `.gitignore` 例外，**入 git**
- 服务器数据同步到本地用 rsync，本地代码更新同步到服务器也用 rsync，两端保持一致
- 服务器无 git，代码版本管理只在本地进行

## 上下文快照（2026-03-19，第 2 轮）

- 当前进展：论文定位已从"反无人机检测"扩展为"极小目标检测通用方法（以反无人机为主要应用）"；公开数据集选定 AI-TOD（NWD 原文基准，需合成 xView 部分，COCO 格式）；服务器已部署新队列，alpha=0.3 实验正在运行（ep ~10/300）；完成代码/项目结构整理，本地与服务器通过 rsync 同步；发现并确认多个论文/代码问题（见代码问题/论文问题两节）
- 最新决定：
  目标期刊：先投 IEEE TGRS（三区），审稿期间并行准备 TCSVT（二区）升刊版
  AI-TOD 实验固定超参（c_base=12, k=1.0, alpha=0.5，私有集调出，不重新调参）
  RT-DETR 跨检测器实验降为 P2（TGRS 版本不含，TCSVT 升刊时补）
  Exp E 设计为 partial reverse（只改 Loss 方向，TAL 保持 forward），论文里加一句说明
  SA-NWD-NMS：补实验 E5 验证，根据结果决定是否保留为贡献（方案 B）
  AI-TOD 使用 imgsz=800（与 NWD 原文对齐），batch=8（dry run 先确认不 OOM）
  S̄ 描述改为 "detection head coordinate frame"（stride-normalized，非像素面积）
  论文理论：Theorem 1 补完整证明（附录）+ 新增 Proposition 2（NWD 梯度工作点分析）
  Proposition 2 叙事角度：必须等 E1（x=W₂/C 分布统计）结果出来后再定
- 服务器/实验状态：
  运行中：alpha=0.3（ep ~10/300），screen session 165438.queue
  队列中（run_new_queue.sh 串行）：alpha=0.7 → Exp D → Exp E → k-sweep
  nwd_p2/weights/ 只有 best.pt 和 last.pt（无中间 epoch checkpoint）
  待提交（Day 0）：E1（x=W₂/C 分布统计，~6h 重跑，最高优先级）→ E2/E3/E4/E5/E6
  待确认：AI-TOD xView 账号申请（需去 xviewdataset.org 注册）
- 关键技术发现：
  BboxLoss.forward 传入坐标是 stride-normalized（target_bboxes/stride_tensor）
  P3头(stride=8): 50px目标 S̄≈39，C_adapt≈14；P2头(stride=4): S̄≈156，C_adapt≈13
  W₂ 实际量级和 x=W₂/C 真实分布未知，必须等 E1 结果
  之前某轮反馈中"C≈650"的估算是错的（用了错误的坐标系假设）
- 待完成（下一步最重要的事）：
  1. 今天：C1-C4 代码修复 + P0-P4 论文紧急修改 + 提交 E1 到服务器（最高优先级）
  2. Day 3：E1 结果 → 确定 Proposition 2 叙事角度 → 开始理论写作
  3. xView 账号审核后：E7 dry run → 启动 AI-TOD 实验队列（E8-E11）
  4. Day 5-10：Theorem 1 完整证明（附录） + Proposition 2 正文
  5. Day 14：投稿 IEEE TGRS

## 待完成清单

```
代码修复（Day 0，最高优先级）：
  [ ] C1: nwd.py:72 sa_nwd() 默认 k=2.0 → k=1.0
  [ ] C2: nwd.py:~191 patch_sa_nwd_loss() 默认 k=2.0 → k=1.0（之前漏掉）
  [ ] C3: verify_error_distribution.py L267 cfg.get(path,...) → cfg.get('path',...)
  [ ] C4: run_new_queue.sh 去掉 set -e

论文修改（Day 0-3）：
  [ ] P0（最高）: NMS 从贡献降级为"可选组件，待 E5 验证"
  [ ] P1: σ=w/2 → σ=w/6，加脚注说明与 NWD 原文实现对齐
  [ ] P2: 加 W₂=√(W₂²) 的一句说明
  [ ] P3: S̄ 描述改为 detection head coordinate frame（不是像素面积）
  [ ] P4: Exp E 描述加一句：只改 Loss 方向，TAL 保持 forward
  [ ] P5: 标题去掉 Fisher-Guided
  [ ] P6: 全文 scale-equivariant → scale-adaptive
  [ ] P7: Introduction 第一段明确 anti-UAV + 极小目标检测定位
  [ ] P8: 贡献合并为 3 条，加 "one metric fixes all" 叙事框架
  [ ] P9: Key distinction 加一层：NWD 原文只改 label assignment，本文全链路替换
  [ ] P10: refs.bib 删掉 zhong2025asanwd 和 lei2025rfwnet 的主观 note 字段
  [ ] P11: 消融表格式统一（小数），加 mAP50-95，拆 SA-NWD 列为 Loss/TAL 两列
  [ ] P12: Fig.1 caption 说明负贡献曲线是消融的一部分
  [ ] P13: §IV Dataset 补充采集方式（设备/时间/抽帧/标注协议）
  [ ] P14: 新增 Limitation 段

新增实验（按优先级）：
  P0（必须）：
  [ ] E1: x=W₂/C 分布统计（~6h 重跑带 logging 的完整训练，最高优先级）
  [ ] E2: TAL 消融 Loss-only（stock yolo11n，只 patch loss，~6h）
  [ ] E3: TAL 消融 TAL-only（stock yolo11n，只 patch TAL，~6h）
  [ ] E4: Asa-NWD batch-level 对比（全 batch GT 均值面积，~6h）
  [ ] E5: SA-NWD-NMS 验证（use_nwd_nms=True，~6h，视结果决定是否保留 NMS 贡献）
  [ ] E6: 私有集 3 seeds（seed=1,2，~12h）
  [ ] E7: AI-TOD dry run（1 epoch，确认 batch=8 不 OOM，~30min）
  [ ] E8-E11: AI-TOD 完整消融（Baseline/NWD/SA-NWD/+P2，各~24-36h）

  P1（强烈建议）：
  [ ] E12-E13: RT-DETR 跨检测器（AI-TOD，降为 P2，TCSVT 版本补）

理论工作（等 E1 结果后开始）：
  [ ] T1: 确定 Proposition 2 叙事角度（基于 E1 x 分布）
  [ ] T2: Theorem 1 完整证明（附录，注意指数是 ⁴ 不是 ²）
  [ ] T3: Proposition 2 正文（显式引用 Theorem 1 假设作为前提）
  [ ] T4: §III 叙事重构（Theorem 1 → Prop.2 → SA-NWD 设计）

新增论文内容（等实验完成后）：
  [ ] 消融表填入 Exp D/E 行、TAL 单独消融行、Asa-NWD 行、NMS 行（视 E5）
  [ ] SOTA 表填入 NWD (k=0) 行
  [ ] k-sweep 折线图 + alpha sweep 结果（§IV-C）
  [ ] AI-TOD 实验表格（§IV-B，含 NWD 原文 Faster R-CNN 参考行†）
  [ ] AP 按尺寸分桶分析（tiny/small/medium，AI-TOD 官方评估工具）
  [ ] x=W₂/C 分布图（E1 结果，验证 Proposition 2）
  [ ] Failure Mode Analysis（测试集漏检/误检案例，使用 gradcam.py）
  [ ] 私有集 3 seeds 标准差加入主结果表
  [ ] §IV Dataset 数据集来源说明（自采，IP 摄像头，4个采集批次）

需要新建的脚本（执行时写）：
  [ ] scripts/run_working_point_log.py（E1，logging 版训练）
  [ ] scripts/run_tal_ablation.py（E2/E3）
  [ ] scripts/run_asanwd_compare.py（E4，batch-level 均值）
  [ ] scripts/run_nwd_nms.py（E5）
  [ ] scripts/convert_aitod_to_yolo.py（含自动生成 configs/aitod_data.yaml）
  [ ] scripts/run_aitod_queue.sh（E7-E11 串行队列）
  [ ] scripts/analyze_failure_cases.py
  [ ] scripts/analyze_ap_by_size.py
```
