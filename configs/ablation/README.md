# configs/ablation/ — 消融实验架构配置

本目录存放消融实验专用的模型架构 YAML 文件。
这些文件是实验脚本（`scripts/run_nwd_fixed.py` 等）的直接依赖，已加入 git 版本管理。

---

## 文件说明

| 文件 | 用途 | 状态 |
|------|------|------|
| `ablation_nwd_p2.yaml` | **主消融架构**：YOLOv11n + P2 检测头（stride=4），无 SimAM/PConv。所有 Exp D/E/k-sweep/alpha-sweep 均使用此 YAML | 活跃 |
| `ablation_nwd_p2_simam.yaml` | nwd_p2 + SimAM 注意力 | 参考（消融已跑完） |
| `ablation_nwd_p2_simam_pconv.yaml` | nwd_p2 + SimAM + PConv_C3k2 | 参考（消融已跑完） |
| `ablation_nwd_p2_pconv.yaml` | nwd_p2 + PConv_C3k2（无 SimAM） | 参考（消融已跑完） |
| `ablation_p2.yaml` | 早期 P2 头架构（无 NWD，用于早期消融） | 归档参考 |
| `ablation_p2_ema.yaml` | 早期 P2 + EMA 注意力（EMA 已被 SimAM 替代） | 归档参考 |
| `ablation_p2_ema_pconv.yaml` | 早期 P2 + EMA + PConv | 归档参考 |

---

## 注意事项

- `ablation_nwd_p2.yaml` 中未指定 `scale` 参数，ultralytics 会警告 `no model scale passed, assuming scale='n'`，这是正常行为（等价于 yolo11n 规模）
- 这些 YAML 文件由早期实验自动生成后手动整理，服务器路径为 `/root/drone_detection/configs/ablation/`
- 新实验需要新架构时，在此目录添加新 YAML 并同步更新本文件
