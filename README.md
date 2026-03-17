# 无人机小目标检测 — YOLOv11n 改进方案

基于 YOLOv11n 的无人机极小目标检测改进算法，包含完整的训练、评估、消融实验和 IEEE 论文模板。

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

# 阶段2：改进模型训练（四项改进）
python scripts/train_improved.py

# 阶段2：SOTA 模型训练（全部改进）
python scripts/train_sota.py

# 消融实验（5组，自动顺序执行）
python scripts/ablation.py

# 统一评估（mAP50 / Params / FPS / 模型大小）
python scripts/eval.py --weights runs/detect/yolo11n_baseline/weights/best.pt \
                                runs/detect/yolo11n_improved/weights/best.pt \
                       --names "Baseline" "Improved"
```

---

## 项目结构

```
无人机/
├── configs/                          # 模型与数据集配置
│   ├── data.yaml                     #   数据集路径配置（需修改 path）
│   ├── yolo11n-improved.yaml         #   改进模型（P2+EMA+PConv+BiFPN）
│   ├── yolo11n-sota.yaml             #   SOTA 模型（RepVGG+CA+CARAFE+PConv+BiFPN）
│   └── ablation/                     #   消融实验自动生成的 YAML
│
├── ultralytics_modules/              # 自定义网络模块
│   ├── __init__.py
│   ├── attention.py                  #   EMA（消融用）/ CA（Coordinate Attention）
│   ├── pconv.py                      #   PConv + PConv_C3k2 轻量化卷积
│   ├── bifpn.py                      #   BiFPN_Concat 加权特征融合
│   ├── repvgg.py                     #   RepVGGBlock 结构重参数化
│   ├── carafe.py                     #   CARAFE 内容感知上采样
│   └── inner_iou.py                  #   Inner-IoU 损失函数
│
├── scripts/                          # 训练与评估脚本
│   ├── register_modules.py           #   模块注册（必须最先导入）
│   ├── train_baseline.py             #   YOLOv11n 基线训练
│   ├── train_improved.py             #   四项改进模型训练
│   ├── train_sota.py                 #   全改进 SOTA 模型训练
│   ├── ablation.py                   #   消融实验（独立子进程，5组）
│   └── eval.py                       #   统一评估（含 FPS/Params/ModelSize）
│
├── paper/                            # IEEE 论文
│   ├── main.tex                      #   论文正文（IEEEtran Journal 格式）
│   ├── refs.bib                      #   BibTeX 参考文献（22 篇）
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

### 改进模型（yolo11n-improved.yaml）

| 改进 | 模块 | 作用 |
|------|------|------|
| P2 检测头 | 4-head Detect | stride=4 特征图，检测极小目标 |
| EMA 注意力 | `EMA` | backbone 特征增强，位置感知 |
| PConv 轻量化 | `PConv_C3k2` | neck 计算量降低 ~20% |
| BiFPN 融合 | `BiFPN_Concat` | 可学习权重自适应特征融合 |

### SOTA 模型（yolo11n-sota.yaml）

| 改进 | 模块 | 论文 |
|------|------|------|
| RepVGG backbone | `RepVGGBlock` | CVPR 2021 |
| CA 注意力 | `CA` | CVPR 2021 |
| CARAFE 上采样 | `CARAFE` | ICCV 2019 |
| PConv 轻量化 | `PConv_C3k2` | CVPR 2023 |
| BiFPN 融合 | `BiFPN_Concat` | CVPR 2020 |
| Inner-IoU 损失 | `patch_ultralytics_loss` | 2023 |

---

## 已完成实验

| 模型 | val mAP50 | val mAP50-95 | Recall |
|------|:---------:|:------------:|:------:|
| YOLOv11n baseline | 0.929 | 0.473 | 0.860 |
| YOLOv11n improved | 0.944 | 0.464 | 0.906 |
| SOTA（训练中） | — | — | — |

---

## 数据集

- **格式**：Ultralytics YOLO（归一化 xywh）
- **类别**：1 类（drone）
- **划分**：train 391 / val 111 / test 57
- **分辨率**：1920×1080
- **特点**：极小目标，归一化面积中位数 0.0023（约 50×50 像素）

---

## 注意事项

- `register_modules.py` 已在各脚本头部自动导入，**无需手动调用**
- 消融实验使用独立子进程运行每组，避免 CUDA 上下文污染
- BiFPN_Concat 的 `num_inputs` 在 `parse_model` 中自动推断，无需手动指定
- RepVGGBlock 训练后可用 `repvgg_model_convert()` 合并分支加速推理

---

## 论文编译

上传 `paper/` 文件夹到 [Overleaf](https://www.overleaf.com/)，选择 pdfLaTeX 编译器即可。
