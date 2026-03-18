# archive/ — 归档目录

存放所有不再活跃使用但保留参考价值的内容。
**禁止直接删除项目中的任何文件**，废弃内容一律移此归档。

---

## 子目录索引

### `scripts/` — 17 个归档脚本

详见 `archive/scripts/README.md`。按归档原因分为 4 类：

| 类别 | 脚本数 | 归档原因 |
|------|:----:|---------|
| Fisher 系列 | 7 | 实验证伪，mAP50 全部低于基线（0.827~0.937 vs 0.960） |
| SA-NWD 变体 | 4 | 功能已合并至主消融链，或方案迭代后被替换 |
| 旧架构训练 | 3 | BiFPN/RepVGG/CARAFE 消融证实负贡献 |
| 数据处理工具 | 3 | 一次性任务完成，含 VisDrone 下载（任务不匹配已确认） |

### `modules/` — 5 个归档模块

详见 `archive/modules/README.md`。

| 模块 | 归档原因 |
|------|---------|
| `bifpn.py` | 消融 mAP50=0.9365，负贡献 -3.84% |
| `carafe.py` | 与其他模块组合时训练不稳定 |
| `repvgg.py` | 小数据集过拟合，全面低于基线 |
| `legacy/attention.py` | EMA/CA 被零参数 SimAM 替代 |
| `legacy/inner_iou.py` | 被 SA-NWD 替代，未进入消融实验 |

### `sessions/` — 2 个 AI 决策记录

详见 `archive/sessions/README.md`。
**不得删除**，含实验选型分析、Fisher 失败原因、理论修改讨论等关键决策上下文，可用于论文 Discussion 写作参考。

### `results/` — 历史评估结果

详见 `archive/results/README.md`。

| 文件 | 内容 |
|------|------|
| `baseline_640_eval.json` | YOLOv11n baseline 在 640 分辨率下的评估结果（论文用 1280，此文件供参考） |

---

## 其他归档文件

| 文件 | 原用途 | 归档原因 | 归档时间 |
|------|--------|---------|---------|
| `yolo11n-sota.yaml` | 早期 SOTA 方案架构（RepVGG+CARAFE+BiFPN+NWD） | 训练不稳定，全面低于基线 | 2026-03 |
| `远程GPU操作手册.md` | 连接哥哥 Windows 电脑作为远程 GPU 的操作说明 | 已改用恒源云，不再需要 | 2026-03 |
