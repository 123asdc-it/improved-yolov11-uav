# 无人机小目标检测 — YOLOv11n 改进方案

基于 YOLOv11n 的无人机极小目标检测改进算法。核心改进：NWD 损失函数 + SimAM 零参数注意力 + P2 检测头 + PConv 轻量化 + BiFPN 加权融合。包含完整的训练、评估、消融实验、数据增强和 IEEE 论文模板。

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
# 阶段1：基线训练
python scripts/train_baseline.py

# 阶段2：改进模型训练（NWD + SimAM + P2 + PConv + BiFPN）
python scripts/train_improved.py

# 阶段3：SOTA 模型训练（+ RepVGG + CARAFE）
python scripts/train_sota.py

# 消融实验（6组，自动顺序执行）
python scripts/ablation.py

# 统一评估（mAP50 / Params / FPS / 模型大小）
python scripts/eval.py --weights runs/detect/yolo11n_baseline/weights/best.pt \
                                runs/detect/yolo11n_improved/weights/best.pt \
                       --names "Baseline" "Improved"

# 可选：SAHI 切片训练数据生成
python scripts/slice_dataset.py

# 可选：Copy-Paste 小目标增强
python scripts/augment_copy_paste.py
```

---

## 项目结构

```
无人机/
├── configs/                          # 模型与数据集配置
│   ├── data.yaml                     #   数据集路径配置（需修改 path）
│   ├── data_sliced.yaml              #   切片数据集配置（由 slice_dataset.py 生成）
│   ├── yolo11n-improved.yaml         #   改进模型（NWD+P2+SimAM+PConv+BiFPN）
│   ├── yolo11n-sota.yaml             #   SOTA 模型（+RepVGG+CARAFE）
│   └── ablation/                     #   消融实验自动生成的 YAML
│
├── ultralytics_modules/              # 自定义网络模块
│   ├── __init__.py
│   ├── simam.py                      #   SimAM 零参数注意力（ICML 2021）[主用]
│   ├── nwd.py                        #   NWD 损失 + NWD-TAL 标签分配（ISPRS 2022）[主用]
│   ├── pconv.py                      #   PConv + PConv_C3k2 轻量化卷积
│   ├── bifpn.py                      #   BiFPN_Concat 加权特征融合
│   ├── repvgg.py                     #   RepVGGBlock 结构重参数化
│   ├── carafe.py                     #   CARAFE 内容感知上采样
│   ├── attention.py                  #   EMA / CA 注意力 [旧方案，保留参考]
│   └── inner_iou.py                  #   Inner-IoU 损失 [旧方案，保留参考]
│
├── scripts/                          # 训练与评估脚本
│   ├── register_modules.py           #   模块注册（必须最先导入）
│   ├── train_baseline.py             #   YOLOv11n 基线训练
│   ├── train_improved.py             #   改进模型训练（自动应用 NWD patch）
│   ├── train_sota.py                 #   SOTA 模型训练（自动应用 NWD patch）
│   ├── ablation.py                   #   消融实验（6组，独立子进程）
│   ├── eval.py                       #   统一评估（含 FPS/Params/ModelSize）
│   ├── slice_dataset.py              #   SAHI 训练切片（391图→~2500切片）
│   └── augment_copy_paste.py         #   小目标 Copy-Paste 增强（391→782图）
│
├── paper/                            # IEEE 论文
│   ├── main.tex                      #   论文正文（IEEEtran Journal 格式）
│   ├── refs.bib                      #   BibTeX 参考文献
│   └── figs/                         #   图片目录（待填充）
│
├── docs/                             # 文档
│   ├── CLAUDE.md                     #   项目技术上下文
│   └── 操作指南.md                    #   详细操作手册
│
└── datasets/                         # 数据集（ultralytics 格式）
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/
```

---

## 改进方案

### 核心思路

传统改进（加注意力、换卷积）= 增加参数 → 391 张小数据集容易过拟合。
本方案：**改度量（NWD）+ 零参数注意力（SimAM）+ 结构优化（P2/PConv/BiFPN）**，
专门解决"小目标 + 小数据集"的核心矛盾。

### 改进模型（yolo11n-improved.yaml）

| 改进 | 模块 | 论文 | 作用 |
|------|------|------|------|
| NWD 损失 + 标签分配 | `patch_all_nwd()` | ISPRS 2022 | Wasserstein 距离替代 IoU，小目标度量更稳定 |
| P2 检测头 | 4-head Detect | — | stride=4 高分辨率特征图，覆盖极小目标 |
| SimAM 注意力 | `SimAM` | ICML 2021 | 零参数注意力，增强判别性特征，不过拟合 |
| PConv 轻量化 | `PConv_C3k2` | CVPR 2023 | neck 计算量降低 ~20% |
| BiFPN 融合 | `BiFPN_Concat` | CVPR 2020 | 可学习权重自适应特征融合 |

### SOTA 模型（yolo11n-sota.yaml）

在改进模型基础上额外添加：

| 改进 | 模块 | 论文 |
|------|------|------|
| RepVGG backbone | `RepVGGBlock` | CVPR 2021 |
| CARAFE 上采样 | `CARAFE` | ICCV 2019 |

### 消融实验设计（6组）

```
0. Baseline          — YOLOv11n 原版
1. +NWD              — 仅换损失函数和标签分配
2. +NWD+P2           — 加 P2 小目标检测头
3. +NWD+P2+SimAM     — 加零参数注意力
4. +NWD+P2+SimAM+PConv — 轻量化 neck
5. Full (Ours)       — + BiFPN 加权融合
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

将 1920×1080 图切成 640×640 重叠小块，小目标在切片中"变大" 4 倍：

```bash
python scripts/slice_dataset.py
# 生成 datasets_sliced/ + configs/data_sliced.yaml
# 训练时用 imgsz=640, batch=16
```

### Copy-Paste 增强

从训练集裁剪目标粘贴到其他图中，每图增加 2-4 个目标：

```bash
python scripts/augment_copy_paste.py
# 生成 datasets_augmented/ + configs/data_augmented.yaml
```

---

## 注意事项

- `register_modules.py` 已在各脚本头部自动导入，**无需手动调用**
- `train_improved.py` 和 `train_sota.py` 会自动调用 `patch_all_nwd()` 应用 NWD 损失
- 消融实验使用独立子进程运行每组，避免 CUDA 上下文污染
- BiFPN_Concat 的 `num_inputs` 在 `parse_model` 中自动推断
- RepVGGBlock 训练后可用 `repvgg_model_convert()` 合并分支加速推理

---

## 论文编译

上传 `paper/` 文件夹到 [Overleaf](https://www.overleaf.com/)，选择 pdfLaTeX 编译器即可。
