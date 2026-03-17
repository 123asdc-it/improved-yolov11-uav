# 无人机小目标检测项目 — 技术上下文

## 项目概述

团队实习任务：基于 YOLOv11n 的无人机极小目标检测改进算法，含论文撰写。
- 阶段1（3.14-3.21）：复现 YOLOv11n 基线 ✅
- 阶段2（3.21-4.21）：改进算法 + IEEE 论文

## 数据集

- 位置：`datasets/`（ultralytics 格式，单类别 `drone`）
- 划分：train 391 / val 111 / test 57，分辨率 1920×1080
- 核心特征：**极小目标**，归一化面积中位数 0.0023（~50×50 像素）

## 模型方案

### 方案一：改进模型（论文主体）
`configs/yolo11n-improved.yaml`
- P2 小目标检测头（stride=4，4-head Detect）
- EMA 注意力（backbone P3/P4）
- PConv_C3k2 轻量化 neck
- BiFPN 加权特征融合

### 方案二：SOTA 模型（追求极致指标）
`configs/yolo11n-sota.yaml`
- RepVGG backbone（训练多分支，推理单分支）
- CA（Coordinate Attention，比 EMA 更强）
- CARAFE 内容感知上采样（P2 分支专用）
- PConv_C3k2 + BiFPN（同改进模型）
- Inner-IoU 损失函数（训练时 patch）

## 文件结构

```
无人机/
├── README.md                         # 阅读指南（新增）
├── configs/
│   ├── data.yaml                     # 数据集配置
│   ├── yolo11n-improved.yaml         # 改进模型
│   ├── yolo11n-sota.yaml             # SOTA 模型
│   └── ablation/                     # 消融 YAML（自动生成）
├── ultralytics_modules/
│   ├── attention.py                  # EMA + CA
│   ├── pconv.py                      # PConv + PConv_C3k2
│   ├── bifpn.py                      # BiFPN_Concat（已修复懒初始化）
│   ├── repvgg.py                     # RepVGGBlock
│   ├── carafe.py                     # CARAFE
│   └── inner_iou.py                  # Inner-IoU + ultralytics patch
├── scripts/
│   ├── register_modules.py           # 模块注册 + parse_model patch
│   ├── train_baseline.py             # 基线训练（含 test 集评估）
│   ├── train_improved.py             # 改进模型训练（含 val/test/TTA）
│   ├── train_sota.py                 # SOTA 训练（含 Inner-IoU + TTA）
│   ├── ablation.py                   # 消融实验（subprocess 独立进程）
│   └── eval.py                       # 统一评估（end-to-end FPS + ModelSize）
├── paper/
│   ├── main.tex                      # IEEE Journal 论文（IEEEtran）
│   └── refs.bib                      # 22 篇 BibTeX 参考文献
└── docs/
    ├── CLAUDE.md                     # 本文件
    └── 操作指南.md                    # 详细操作手册
```

## 运行方式

```bash
# 所有命令在项目根目录执行
python scripts/train_baseline.py
python scripts/train_improved.py
python scripts/train_sota.py
python scripts/ablation.py
python scripts/eval.py --weights <weights> --names <names>
```

## 已完成实验结果

| 模型 | val mAP50 | val mAP50-95 | Recall | 状态 |
|------|:---------:|:------------:|:------:|:----:|
| YOLOv11n baseline | 0.929 | 0.473 | 0.860 | ✅ 完成 |
| YOLOv11n improved | 0.944 | 0.464 | 0.906 | ✅ 完成 |
| 消融实验（5组） | — | — | — | ⏳ 进行中 |
| SOTA 模型 | — | — | — | ⏳ 待运行 |

## 关键技术细节

### 模块注册机制
- `register_modules.py` 通过 monkey-patch `tasks.parse_model` 注入自定义模块
- 各脚本头部已自动 `import register_modules`，无需手动调用
- BiFPN_Concat：`parse_model` 中自动传 `num_inputs=len(f)`，已修复懒初始化
- EMA/CA：pass-through，`c2 = ch[f]`，args 传 `[c2]`
- RepVGGBlock：args 传 `[c1, c2, 3, stride]`
- CARAFE：args 前置 channels，即 `[c2, scale_factor]`

### 已知问题与修复
- **BiFPN 懒初始化（已修复）**：权重改为在 `__init__` 预分配
- **ablation.py DataLoader 卡死（已修复）**：改用 subprocess 独立进程
- **Inner-IoU patch 参数签名（已修复）**：适配 ultralytics 8.4.22 的 9 参数签名
- **ablation baseline 路径重复**：`runs/detect/runs/ablation/baseline` 是 ultralytics 自动追加前缀的行为，不影响权重读取

### 恒源云环境
- RTX 3060 12GB，PyTorch 2.4.0+cu121，Python 3.11，ultralytics 8.4.22
- SSH：`ssh -p 21635 root@i-1.gpushare.com`
- 项目路径：`/root/drone_detection/`
- `make_divisible` 在此版本位于 `ultralytics.utils.ops`（已做兼容处理）

## 已完成修复

- [x] `repvgg.py:41` 1x1 分支条件修复（去掉 `if out_channels == in_channels`，让所有 RepVGGBlock 都有多分支训练）
- [x] `inner_iou.py` 参数签名修复（适配 ultralytics 8.4.22 的 9 参数）—— 服务器已是修复版
- [x] `ablation.py` 层索引修复 —— 当前消融运行版本已验证正确

## 消融实验已知问题

- **+P2+EMA 组 mAP50=0.9328**：EMA 模块导致 epoch 43 就早停，严重欠训练
- **原因**：patience=50 对新增参数的架构太短，391 张小数据集上 EMA 容易过拟合
- **修复计划**：用 `patience=100, warmup_epochs=5, copy_paste=0.2, mixup=0.15` 重跑这两组

## 执行策略

### 第一步：当前方案跑通
1. 等消融 Full 组跑完 → 收集 5 组结果
2. 修 repvgg.py（已完成）→ 上传服务器 → 跑 SOTA
3. 用 patience=100 重跑 +P2+EMA 和 +P2+EMA+PConv
4. 看消融趋势是否递增

### 第二步：Plan B（如果重跑后消融仍不行）

| 替换 | 当前 | 换成 | 理由 |
|------|------|------|------|
| 注意力 | EMA（有参数，过拟合） | **SimAM**（零参数，ICML 2021） | 391 张图不应加参数 |
| 损失函数 | Inner-IoU | **NWD**（Wasserstein 距离，ISPRS 2022） | IoU 对小目标天然不稳定，NWD 在 AI-TOD 上 +6.7 AP |
| 标签分配 | 默认 TAL (IoU-based) | **NWD-based TAL** | 让小目标获得更多正样本 |
| 推理增强 | 无 | **SAHI 切片推理** | 不需重训，pip install sahi，+3~7 AP |

### 不换的
- YOLO 框架（论文结构已定，ultralytics 生态好）
- P2 检测头（小目标必备）
- PConv + BiFPN（轻量化 + 融合，机制合理）

## 待完成

- [ ] 收集消融 5 组完整结果
- [ ] 上传修复版 repvgg.py → 跑 SOTA
- [ ] 重跑 +P2+EMA / +P2+EMA+PConv（patience=100）
- [ ] 评估消融趋势，决定是否启用 Plan B
- [ ] 运行 eval.py 获取 FPS/Params/ModelSize 数据
- [ ] 填写论文所有 XX.X 占位符
- [ ] 绘制图表（架构图、训练曲线、PR 曲线、检测对比、Grad-CAM）
- [ ] 补充作者信息，Overleaf 编译
