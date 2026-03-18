# ultralytics_modules/ — 核心自定义模块

通过 monkey-patch 注入到 ultralytics 框架的自定义模块。
`register_modules.py` 负责将自定义层注册到 ultralytics 命名空间；
`patch_*` 函数直接替换 `BboxLoss.forward` 和 `TaskAlignedAssigner` 的行为，无需修改 ultralytics 源码。

---

## 工作原理

```python
# 标准使用流程（所有训练脚本的开头）
import sys; sys.path.insert(0, '/root/drone_detection/scripts')
import register_modules                          # 注册 SimAM、PConv 等自定义层

from ultralytics_modules.nwd import patch_all_nwd
patch_all_nwd(c_base=12.0, k=1.0, alpha=0.5, nwd_min=0.3)  # 替换 BboxLoss + TAL

from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(...)
```

---

## 文件说明

### `nwd.py` ★ 核心文件（活跃）

SA-NWD（Scale-Adaptive Normalized Wasserstein Distance）完整实现。

#### 活跃函数（可在新实验中使用）

| 函数 | 用途 | 关键参数 |
|------|------|---------|
| `sa_nwd(box1, box2)` | 计算单对 bbox 的 SA-NWD 相似度，返回 [0,1] | `c_base=12.0, k=1.0` |
| `sa_nwd_loss(pred, target, weight, sum)` | batch 级 SA-NWD loss | `c_base=12.0, k=1.0` |
| `patch_sa_nwd_loss()` | 替换 BboxLoss：`α·SA-NWD + (1-α)·CIoU` | `c_base=12, k=1.0, alpha=0.5` |
| `patch_sa_nwd_tal()` | 替换 TAL：SA-NWD 代替 IoU 做标签分配 | `c_base=12, k=1.0, nwd_min=0.3` |
| `patch_all_nwd()` | 便捷调用：同时 patch loss 和 TAL | `k=1.0, alpha=0.5, nwd_min=0.3` |
| `sa_nwd_reverse(box1, box2)` | **Exp E 专用**：C 随尺度增大（Fisher 理论方向对照） | `c_base=12.0, k=1.0, ref_area=0.002315` |
| `patch_sa_nwd_loss_reverse()` | **Exp E 专用**：patch BboxLoss 用 reverse C | `c_base=12, k=1.0, alpha=0.5` |
| `nwd() / nwd_loss()` | 标准 NWD（固定 C），兼容旧代码 | `constant=12.0` |
| `patch_nwd_loss() / patch_nwd_tal()` | 标准 NWD patch，k=0 等价 | `constant=12.0` |

#### 废弃函数（禁止在新实验中使用）

| 函数 | 废弃原因 | 实验结果 | 废弃时间 |
|------|---------|---------|---------|
| `patch_scale_aware_loss()` | Fisher-CIoU：所有实验低于基线 | mAP50 0.827~0.937 | 2026-03 |
| `patch_sa_nwd_fisher_loss()` | SA-NWD+Fisher-CIoU 组合：两者方向相反，组合更差 | 低于单独 SA-NWD | 2026-03 |

#### C_adapt 设计说明

```
C_adapt = C_base × (1 + k / √S̄)

动机：标注噪声正则化（非 Fisher 等变最优推导）
  小目标 S 小 → C 大 → loss 对位置误差宽松 → 防止对标注噪声过拟合
  大目标 S 大 → C 趋近 C_base → 标准 NWD 行为
  k=0 → 退化为固定 C 的标准 NWD（Exp D 基准）

数值参考（c_base=12, k=1）：
  p10（极小目标 S≈0.00035）：C ≈ 650
  median（S≈0.002315）：       C ≈ 261
  p90（较大目标 S≈0.00465）：  C ≈ 188
```

---

### `simam.py`（活跃，用于 yaml 配置）

SimAM：Simple Attention Module（ICML 2021），零参数注意力，基于能量理论推导封闭解。

- 消融结果：单独加 SimAM mAP50 下降 2.62%（391 张小数据集，分布改变导致退化）
- 保留原因：yolo11n-improved.yaml 中使用，记录改进尝试历史

### `pconv.py`（活跃，用于 yaml 配置）

PConv + PConv_C3k2：Partial Convolution（CVPR 2023 FasterNet），只对 1/4 通道做卷积降低 FLOPs。

- 消融结果：与 SimAM 合用 mAP50=0.9749，略低于纯 nwd_p2 的 0.9781
- 保留原因：yaml 配置中使用，记录改进尝试历史

### `__init__.py`（活跃）

导出所有活跃函数。新增函数后必须同步更新此文件的 `import` 和 `__all__`。

当前导出：`SimAM`, `PConv`, `PConv_C3k2`, `sa_nwd`, `sa_nwd_loss`, `nwd_nms`,
`patch_sa_nwd_loss`, `patch_sa_nwd_tal`, `patch_nwd_nms`, `nwd`, `nwd_loss`,
`patch_nwd_loss`, `patch_nwd_tal`, `patch_all_nwd`,
`sa_nwd_reverse`, `patch_sa_nwd_loss_reverse`
