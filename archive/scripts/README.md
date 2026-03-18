# archive/scripts/ — 归档脚本目录

以下脚本不再用于当前实验，保留作为历史记录和失败案例参考。
如需了解某个脚本的实现细节，直接查看文件内的 docstring。

---

## 分类一：Fisher 系列（实验证伪，2026-03）

所有 Fisher-Guided CIoU 相关实验 mAP50 全部低于基线（0.827~0.937 vs 基线 0.960）。

**根本原因**：SA-NWD 用大 C 放松小目标 loss（正则化方向），Fisher-CIoU 用大权重加强小目标 loss（补偿方向），两者方向完全相反，组合使用导致训练信号混乱，效果比单独使用任一方法都差。

| 脚本 | 原用途 | 实验结果 | 归档时间 |
|------|--------|---------|---------|
| `run_fisher_only.py` | 纯 Fisher-CIoU loss（无 SA-NWD），从 best.pt 微调 | mAP50 ≈ 0.937 | 2026-03 |
| `run_clean_fisher.py` | SA-NWD + Fisher-CIoU 组合，从 best.pt 微调 | mAP50 ≈ 0.869（比单独更差）| 2026-03 |
| `run_fisher_ablation.py` | Fisher-CIoU 修正实验（放宽 clamp [0.5,3.0]，从头训） | 低于基线 | 2026-03 |
| `run_fisher_improved.py` | Fisher 改进版（修正 DFL 逻辑错误后重跑） | 低于基线 | 2026-03 |
| `run_fisher_queue.sh` | 串行调度 Fisher A/B/C 实验的队列脚本 | — | 2026-03 |
| `run_fisher_sequence.sh` | 等待消融完成后自动跑 Fisher 系列的 shell 脚本 | — | 2026-03 |
| `run_scale_aware_only.py` | 纯 Scale-Aware CIoU（无任何 NWD 组件） | 低于基线 | 2026-03 |

---

## 分类二：SA-NWD 变体（功能已合并或方案迭代，2026-03）

| 脚本 | 原用途 | 归档原因 | 归档时间 |
|------|--------|---------|---------|
| `run_clean.py` | 单阶段 SA-NWD 训练，消融主链的早期版本 | 功能已合并进 `ablation.py` | 2026-03 |
| `run_sa_nwd_comparison.py` | SA-NWD 与 baseline 的早期对比实验 | 被正式消融取代 | 2026-03 |
| `run_sa_nwd_tal.py` | SA-NWD + TAL 组合验证（修复 nwd_min bug 后） | 训练震荡崩溃，已放弃 TAL 微调路线 | 2026-03 |
| `run_finetune_nwd_p2.py` | 从 nwd_p2/best.pt 微调，尝试进一步提升 | 微调不稳定，低于原始 nwd_p2（0.9781） | 2026-03 |

---

## 分类三：旧架构训练（方案迭代，2026-03 早期）

| 脚本 | 原用途 | 归档原因 | 归档时间 |
|------|--------|---------|---------|
| `train_improved.py` | YOLOv11n-improved 架构训练（含 BiFPN） | BiFPN 消融证实负贡献（-3.84%） | 2026-03 |
| `train_sota.py` | SOTA 配置训练（RepVGG+CARAFE+BiFPN+NWD） | 训练不稳定，全面低于基线 | 2026-03 |
| `train_sota_two_stage.py` | SOTA 配置两阶段训练（CIoU 预热→NWD 微调） | 两阶段无明显优势，不值得额外复杂度 | 2026-03 |

---

## 分类四：数据处理工具（一次性任务完成）

| 脚本 | 原用途 | 归档原因 | 归档时间 |
|------|--------|---------|---------|
| `slice_dataset.py` | SAHI 风格切片：将 1920×1080 原图切为重叠小块 | 实验证实对单目标稀疏场景无提升，任务完成 | 2026-03 |
| `augment_copy_paste.py` | 从训练集裁剪小目标粘贴到其他图像，增加目标密度 | 功能已集成为 ultralytics 内置增强，脚本不再需要 | 2026-03 |
| `download_visdrone.py` | 下载并转换 VisDrone2019-DET 数据集为 YOLO 格式 | VisDrone 任务不匹配（从无人机俯拍地面目标，非检测无人机），已确认不需要 | 2026-03 |
