# archive/results/ — 历史评估结果

存放不再用于当前论文但有参考价值的历史实验结果文件。

---

## 文件说明

| 文件 | 内容 | 实验配置 | 备注 |
|------|------|---------|------|
| `baseline_640_eval.json` | YOLOv11n baseline 完整评估指标（mAP50/mAP50-95/P/R/F1/Params/FLOPs/FPS） | imgsz=640，CIoU loss，标准 3-head | 论文用 imgsz=1280，此文件仅供对比参考 |
| `ablation_results_server.json` | 主消融实验汇总（`collect_results.py` 生成） | 6 组消融实验 | **注意**：数字来自训练中途采样，和 `results.csv` best epoch 有差异，不用于论文 |
| `eval_results_server.json` | 效率指标专用结果（`batch_eval.py` 重新验证） | 统一重跑验证，含 FPS(59.4)、Params | **注意**：mAP50 数字与 `results.csv` best epoch 略有差异（重新验证 vs 训练中最优），FPS 数字是权威来源 |

### 两个 JSON 数字差异说明

- **论文使用的数字来源**：`runs/ablation/{exp}/results.csv` 的 best epoch 行（如 nwd_p2 mAP50=0.9781）
- `ablation_results_server.json`：`collect_results.py` 从 CSV 采样的中间结果，用于早期快速汇总，数字偏小
- `eval_results_server.json`：`batch_eval.py` 重新 `model.val()` 的结果，数字接近但不完全等于 best epoch（验证集随机性导致±0.001左右波动）
- **FPS 数字（59.4）**：只在 `eval_results_server.json` 里有，是论文效率指标的唯一来源

---

## `server_logs/` 子目录

历史训练日志（从服务器同步，**不入 git**，本地备份）。

| 内容 | 大小 | 说明 |
|------|------|------|
| `*_log.txt`（30 个） | ~54MB | 各实验的完整训练输出，含 loss 曲线原始数字 |
| `queue_logs/` | ~20KB | 新队列（Exp D/E/k-sweep）的失败日志（`register_modules.py` BiFPN_Concat 错误，已修复） |
