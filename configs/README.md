# configs/ — 配置文件目录

存放模型架构和数据集配置文件。

---

## 文件说明

### `data.yaml` — 数据集配置（活跃，每次部署必须检查 path）

```yaml
path: datasets          # 本地相对路径（本地开发用）
# 服务器部署必须改为绝对路径：
# path: /root/drone_detection/datasets

train: images/train     # 391 张训练图
val:   images/val       # 111 张验证图
test:  images/test      # 57 张测试图
nc: 1
names: ['drone']
```

> **注意**：每次同步代码到服务器后必须确认 `path` 已改为绝对路径，否则训练时找不到数据集。

数据集特征：
- 单类别（`drone`），反无人机检测场景
- 归一化面积中位数 0.0023（约 50×50 像素），属于极小目标
- 已知问题：train/test 有相邻帧交叉污染（视频抽帧），所有方法同等受影响

---

### `yolo11n-improved.yaml` — 改进模型架构（参考用，非当前推荐）

包含 P2 检测头 + SimAM + PConv_C3k2 的完整架构定义。

**当前状态**：消融实验证实 SimAM（-2.62%）和 PConv 对本数据集为负贡献。
**实际训练推荐**：直接用 `yolo11n.pt`（内置架构）+ `patch_all_nwd()` 组合，不使用此 yaml。
**保留原因**：完整记录改进尝试，含详细注释，可作为未来架构修改的参考模板。

主要模块（yaml 内部已标 ★）：
- ★ `SimAM` — backbone P3/P4 后，零参数注意力
- ★ `PConv_C3k2` — neck 中替换标准 C3k2，降低 FLOPs
- ★ P2 检测头 — stride=4，提供 320×320 特征图覆盖极小目标

---

## 服务器上的额外配置（本地未同步）

`configs/ablation/ablation_nwd_p2.yaml` — 消融实验专用架构（P2 头，无 SimAM/PConv）

此文件由 `ablation.py` 在服务器上自动生成，不需要手动维护或上传。
