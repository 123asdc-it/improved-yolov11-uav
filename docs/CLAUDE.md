# 无人机小目标检测项目 — 技术上下文

## 项目概述

团队实习任务：基于 YOLOv11n 的无人机极小目标检测改进算法，含论文撰写。
- 阶段1（3.14-3.21）：复现 YOLOv11n 基线 ✅
- 阶段2（3.21-4.21）：改进算法 + IEEE 论文

## 数据集

- 位置：`datasets/`（ultralytics 格式，单类别 `drone`）
- 划分：train 391 / val 111 / test 57，分辨率 1920×1080
- 核心特征：**极小目标**，归一化面积中位数 0.0023（~50×50 像素）

## 当前方案（SimAM + NWD）

### 核心理念
传统改进（加注意力、换卷积）= 增加参数 → 391 张小数据集容易过拟合。
本方案：**改度量（NWD）+ 零参数注意力（SimAM）+ 结构优化**，专门解决"小目标 + 小数据集"问题。

### 改进模型（论文主体）
`configs/yolo11n-improved.yaml` + NWD patch
- **NWD 损失 + NWD-TAL**（ISPRS 2022）：Wasserstein 距离替代 IoU，小目标度量更稳定
- **P2 小目标检测头**（stride=4，4-head Detect）
- **SimAM 零参数注意力**（ICML 2021）：backbone P3/P4，不增加任何参数
- **PConv_C3k2 轻量化 neck**
- **BiFPN 加权特征融合**

### SOTA 模型（追求极致指标）
`configs/yolo11n-sota.yaml` + NWD patch
- RepVGG backbone（训练多分支，推理单分支）
- SimAM 注意力（替代旧方案的 CA）
- CARAFE 内容感知上采样（P2 分支专用）
- PConv_C3k2 + BiFPN（同改进模型）

## 文件结构

```
无人机/
├── configs/
│   ├── data.yaml                     # 数据集配置
│   ├── data_sliced.yaml              # 切片数据集配置（由 slice_dataset.py 生成）
│   ├── yolo11n-improved.yaml         # 改进模型（NWD+P2+SimAM+PConv+BiFPN）
│   ├── yolo11n-sota.yaml             # SOTA 模型（+RepVGG+CARAFE）
│   └── ablation/                     # 消融 YAML（自动生成）
├── ultralytics_modules/
│   ├── simam.py                      # SimAM 零参数注意力 [主用]
│   ├── nwd.py                        # NWD 损失 + NWD-TAL patch [主用]
│   ├── pconv.py                      # PConv + PConv_C3k2
│   ├── bifpn.py                      # BiFPN_Concat
│   ├── repvgg.py                     # RepVGGBlock（已修复 1x1 分支）
│   ├── carafe.py                     # CARAFE
│   ├── attention.py                  # EMA + CA [旧方案，保留参考]
│   └── inner_iou.py                  # Inner-IoU [旧方案，保留参考]
├── scripts/
│   ├── register_modules.py           # 模块注册 + parse_model patch
│   ├── train_baseline.py             # 基线训练
│   ├── train_improved.py             # 改进模型训练（自动应用 NWD patch）
│   ├── train_sota.py                 # SOTA 训练（自动应用 NWD patch）
│   ├── ablation.py                   # 消融实验（6 组，subprocess 独立进程）
│   ├── eval.py                       # 统一评估（FPS/Params/ModelSize）
│   ├── slice_dataset.py              # SAHI 训练切片（391图→~2500切片）
│   └── augment_copy_paste.py         # 小目标 Copy-Paste 增强
├── paper/
│   ├── main.tex                      # IEEE Journal 论文（待更新为新方案）
│   └── refs.bib                      # BibTeX 参考文献
└── docs/
    ├── CLAUDE.md                     # 本文件
    └── 操作指南.md                    # 操作手册（待更新为新方案）
```

## 关键技术细节

### 模块注册机制
- `register_modules.py` 通过 monkey-patch `tasks.parse_model` 注入自定义模块
- 各脚本头部已自动 `import register_modules`
- SimAM/EMA/CA：pass-through，`c2 = ch[f]`，args 传 `[c2]`
- RepVGGBlock：args 传 `[c1, c2, 3, stride]`
- CARAFE：args 前置 channels，即 `[c2, scale_factor]`

### NWD 机制
- `nwd.py` 提供 `patch_all_nwd()` 函数，在 `model.train()` 前调用
- 同时 patch BboxLoss.forward（NWD 替代 CIoU）和 TaskAlignedAssigner.get_box_metrics（NWD 替代 IoU）
- 常数 C=12.0 控制灵敏度，小目标建议 8.0-12.0

### 消融实验设计（6 组）
```
0. Baseline          — YOLOv11n 原版，无 NWD
1. +NWD              — 仅换损失和标签分配（无结构改变）
2. +NWD+P2           — 加 P2 小目标检测头
3. +NWD+P2+SimAM     — 加零参数注意力
4. +NWD+P2+SimAM+PConv — 轻量化 neck
5. Full (Ours)       — + BiFPN 加权融合
```
训练参数：patience=100, warmup_epochs=5, mixup=0.15, copy_paste=0.2

### 恒源云环境
- RTX 3060 12GB，PyTorch 2.4.0+cu121，Python 3.11，ultralytics 8.4.22
- SSH：`ssh -p 21635 root@i-1.gpushare.com`
- 项目路径：`/root/drone_detection/`

## 已完成修复

- [x] `repvgg.py:41` 1x1 分支条件修复
- [x] `inner_iou.py` 参数签名修复（服务器已部署）
- [x] EMA → SimAM 替换（所有配置和脚本）
- [x] Inner-IoU → NWD 替换（损失 + 标签分配）
- [x] 消融实验重写（5 组 → 6 组，含 NWD 独立实验）
- [x] 消融训练参数调优（patience=100, warmup=5, 增强数据增强）
- [x] Git 仓库初始化 + GitHub 推送
- [x] session 文件从 git 移除（含密码）

## 旧方案（EMA + Inner-IoU）— 归档

旧方案的代码保留在 `attention.py` 和 `inner_iou.py` 中供参考。
旧消融结果（仅供对比，非论文数据）：
| 实验 | mAP50 | 早停 epoch |
|------|:-----:|:----------:|
| Baseline | 0.9617 | 123 |
| +P2 | 0.9515 | 176 |
| +P2+EMA | 0.9328 | 43 |
| +P2+EMA+PConv | 0.9360 | 181 |
**问题**：EMA 导致 epoch 43 就早停，消融趋势下降。决定全面切换到 SimAM + NWD。

## 待完成

- [ ] 消融 v2 完成 → 收集 6 组结果
- [ ] SOTA 模型训练（train_sota.py）
- [ ] eval.py 统一评估（FPS/Params/ModelSize）
- [ ] 可视化（Grad-CAM、NWD 标签分配对比、检测结果对比、速度-精度图）
- [ ] SAHI 切片训练（可选，视消融结果决定）
- [ ] VisDrone 交叉验证（可选）
- [ ] ONNX 导出 + 推理速度
- [ ] 更新 paper/main.tex（方法章节重写，数据填入）
- [ ] 更新 paper/refs.bib（添加 SimAM、NWD 引用）
- [ ] 更新 docs/操作指南.md
