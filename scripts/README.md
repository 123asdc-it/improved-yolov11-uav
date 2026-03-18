# scripts/ — 活跃脚本目录

本目录只存放当前项目中活跃使用或计划使用的脚本。
过时/失败的脚本移至 `archive/scripts/`。

---

## 使用前提

所有训练脚本必须在项目根目录运行：

```bash
cd /root/drone_detection        # 服务器路径
python scripts/<脚本名>.py
```

`register_modules.py` 会被各训练脚本自动 import，无需手动调用。

---

## 7 个功能组（共 27 个文件）

### A. 基础训练（2 个）

| 脚本 | 用途 | 状态 |
|------|------|------|
| `train_baseline.py` | YOLOv11n + CIoU 基线训练，300ep，建立对比基准 | 活跃 |
| `register_modules.py` | 注册自定义模块（SimAM/PConv/NWD）到 ultralytics 命名空间，所有训练前自动调用 | 活跃 |

### B. 主消融实验（7 个）

| 脚本 | 用途 | 架构 | 预计时间 |
|------|------|------|---------|
| `ablation.py` | 主消融链：6 组实验串行子进程（Baseline→+NWD→+P2→+SimAM→+PConv→Full） | stock→nwd_p2 | ~36h |
| `run_nwd_fixed.py` | **Exp D**：Fixed NWD k=0，隔离自适应 C 的独立贡献 | nwd_p2 | ~6h |
| `run_nwd_reverse.py` | **Exp E**：Reverse C（C 随尺度增大），验证方向选择（partial: 只改 Loss） | nwd_p2 | ~6h |
| `run_k_sensitivity.py` | k 超参扫描（k=0.5/2.0/3.0），k=1.0 复用已有 nwd_p2 结果不重跑 | nwd_p2 | ~18h |
| `run_alpha_0p3.py` | alpha=0.3（0.3×SA-NWD + 0.7×CIoU） | nwd_p2 | ~6h |
| `run_alpha_0p7.py` | alpha=0.7（0.7×SA-NWD + 0.3×CIoU） | nwd_p2 | ~6h |
| `run_p2_only.py` | **E_p2only**：P2 head 独立贡献（无 NWD patch），分离 P2 贡献 | ablation_p2 | ~6h |

### C. 补充消融实验（5 个）

| 脚本 | 用途 | 实验ID |
|------|------|--------|
| `run_working_point_log.py` | **E1**：统计训练中 x=W₂/C 分布（logging patch），确定 Prop.2 叙事角度 | E1 |
| `run_tal_ablation.py` | **E2/E3**：TAL 消融（Loss-only / TAL-only），隔离各 pipeline 阶段贡献 | E2,E3 |
| `run_asanwd_compare.py` | **E4**：Asa-NWD batch-level vs SA-NWD per-object，验证 Key distinction | E4 |
| `run_nwd_nms.py` | **E5**：SA-NWD-NMS 验证，决定 NMS 是否作为贡献保留（FPS vs mAP 权衡） | E5 |

### D. AI-TOD 跨数据集验证（7 个）

| 脚本 | 用途 | 实验ID |
|------|------|--------|
| `convert_aitod_to_yolo.py` | AI-TOD COCO JSON → Ultralytics YOLO 格式转换，生成 aitod_data.yaml | — |
| `run_aitod_queue.sh` | AI-TOD 串行队列：E7 dry run → E8-E12，约 6-8 天 | E7-E12 |
| `run_aitod_baseline.py` | **E8**：AI-TOD Baseline（YOLOv11n + CIoU） | E8 |
| `run_aitod_nwd_fixed.py` | **E9**：AI-TOD + NWD k=0（固定 C） | E9 |
| `run_aitod_sanwd.py` | **E10**：AI-TOD + SA-NWD（k=1.0，无 P2） | E10 |
| `run_aitod_sanwd_p2.py` | **E11**：AI-TOD + SA-NWD + P2（best config） | E11 |
| `run_aitod_p2only.py` | **E12**：AI-TOD + P2 only（无 NWD） | E12 |

### E. 队列调度（2 个）

| 脚本 | 用途 |
|------|------|
| `run_new_queue.sh` | 私有集串行队列：alpha sweep → Exp D → Exp E → k-sweep，约 42h |
| `run_aitod_queue.sh` | AI-TOD 串行队列（包含 dry run），约 6-8 天 |

### F. 评估与结果收集（2 个）

| 脚本 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `eval.py` | 统一评估：mAP50/mAP50-95/P/R/F1/Params/FLOPs/FPS/ModelSize | `best.pt` 路径列表 | 终端格式化表格 |
| `collect_results.py` | 汇总所有 `runs/ablation/*/result.json` | `runs/` 目录 | JSON + LaTeX 表格片段 |

### G. 可视化与工具（5 个）

| 脚本 | 用途 | 输出 | 状态 |
|------|------|------|------|
| `plot_results.py` | 精度对比柱状图 | `paper/figs/ablation_bar.pdf` | 活跃 |
| `plot_training_curves.py` | 训练曲线（mAP50 + box loss per epoch） | `paper/figs/training_curves.pdf` | 活跃 |
| `generate_paper_figs.py` | 论文所需图表，含 k vs mAP50 折线图（k-sweep 完成后实现） | `paper/figs/` | 部分待完成 |
| `gradcam.py` | Grad-CAM 热力图，baseline vs ours 对比 | `runs/vis/` | 活跃 |
| `verify_error_distribution.py` | 验证理论假设 σ(s) ∝ √s，检验误差分布与尺度的关系 | — | 活跃（bug 已修） |

---

## 依赖关系

```
register_modules.py
    ↑ 被所有训练脚本自动 import（必须最先执行）

ablation.py              → runs/ablation/{baseline, nwd_only, nwd_p2, ...}/
run_nwd_fixed.py         → runs/ablation/nwd_p2_fixed_c/result.json
run_nwd_reverse.py       → runs/ablation/nwd_p2_reverse_c/result.json
run_k_sensitivity.py     → runs/ablation/nwd_p2_k{0p5,2p0,3p0}/
run_p2_only.py           → runs/ablation/p2_only/result.json
run_working_point_log.py → runs/ablation/nwd_p2_w2log/w2_distribution.json
run_tal_ablation.py      → runs/ablation/{nwd_loss_only, nwd_tal_only}/
run_asanwd_compare.py    → runs/ablation/asa_nwd_batchlevel/result.json
run_nwd_nms.py           → runs/ablation/nwd_p2_nms/result.json

convert_aitod_to_yolo.py → datasets/aitod/  (需先运行)
run_aitod_queue.sh       → runs/ablation/aitod_*/

collect_results.py       ← 读取所有 runs/ablation/*/result.json
generate_paper_figs.py   ← 读取 runs/ablation/ + paper/figs/*.csv
eval.py                  ← 读取指定 best.pt 文件
```

---

## 建议运行顺序（服务器）

```
Phase 1（当前）:
  run_new_queue.sh          私有集补充消融（alpha/D/E/k-sweep，~42h）

Phase 2（并行，E1-E6）:
  run_working_point_log.py  E1：W₂/C 分布统计（~6h）
  run_tal_ablation.py       E2/E3：TAL 消融（~12h）
  run_asanwd_compare.py     E4：Asa-NWD 对比（~6h）
  run_nwd_nms.py            E5：NMS 验证（~6h）
  run_p2_only.py            E_p2only：P2 独立贡献（~6h）

Phase 3（AI-TOD）:
  python convert_aitod_to_yolo.py --src /path/to/aitod_raw --dst datasets/aitod
  bash run_aitod_queue.sh   E7-E12（~6-8 天）

Phase 4（论文最终化）:
  collect_results.py        汇总所有结果
  eval.py                   FPS/Params 效率指标
  generate_paper_figs.py    生成论文图表
```
