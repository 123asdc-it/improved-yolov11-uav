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

## 6 个功能组（共 14 个文件）

### A. 基础训练（2 个）

| 脚本 | 用途 | 状态 |
|------|------|------|
| `train_baseline.py` | YOLOv11n + CIoU 基线训练，300ep，建立对比基准 | 活跃 |
| `register_modules.py` | 注册自定义模块（SimAM/PConv/NWD）到 ultralytics 命名空间，所有训练前自动调用 | 活跃 |

### B. 消融实验（5 个）

| 脚本 | 用途 | 架构 | 预计时间 |
|------|------|------|---------|
| `ablation.py` | 主消融链：6 组实验串行子进程（Baseline→+NWD→+P2→+SimAM→+PConv→Full） | stock→nwd_p2 | ~36h |
| `run_nwd_fixed.py` | **Exp D**：Fixed NWD k=0，隔离自适应 C 的独立贡献 | nwd_p2 | ~6h |
| `run_nwd_reverse.py` | **Exp E**：Reverse C（C 随尺度增大），验证方向选择 | nwd_p2 | ~6h |
| `run_k_sensitivity.py` | k 超参扫描（k=0.5/2.0/3.0），k=1.0 复用已有 nwd_p2 结果不重跑 | nwd_p2 | ~18h |
| `run_alpha_0p3.py` | alpha=0.3（0.3×SA-NWD + 0.7×CIoU），验证混合权重影响 | nwd_p2 | ~6h |
| `run_alpha_0p7.py` | alpha=0.7（0.7×SA-NWD + 0.3×CIoU），验证混合权重影响 | nwd_p2 | ~6h |

> Exp D/E/k-sweep 是自适应 C 方向验证实验，结果填入论文消融表和 Analysis 折线图。

### C. 队列调度（1 个）

| 脚本 | 用途 |
|------|------|
| `run_new_queue.sh` | 服务器串行队列：alpha sweep → Exp D → Exp E → k-sweep，约 42h |

服务器运行方式：

```bash
# 先停旧队列
screen -ls
screen -S <旧session名> -X quit

# 启动新队列
screen -S queue
cd /root/drone_detection
mkdir -p logs
bash scripts/run_new_queue.sh 2>&1 | tee logs/new_queue.log
```

### D. 评估与结果收集（2 个）

| 脚本 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `eval.py` | 统一评估：mAP50/mAP50-95/P/R/F1/Params/FLOPs/FPS/Latency/ModelSize | `best.pt` 路径列表 | 终端格式化表格 |
| `collect_results.py` | 汇总所有 `runs/ablation/*/result.json` | `runs/` 目录 | JSON + LaTeX 表格片段 |

### E. 可视化与论文图表（4 个）

| 脚本 | 用途 | 输出 | 状态 |
|------|------|------|------|
| `plot_results.py` | 精度对比柱状图 | `paper/figs/ablation_bar.pdf` | 活跃 |
| `plot_training_curves.py` | 训练曲线（mAP50 + box loss per epoch） | `paper/figs/training_curves.pdf` | 活跃 |
| `generate_paper_figs.py` | 论文所需图表，含 k vs mAP50 折线图（k-sweep 完成后实现） | `paper/figs/` | 部分待完成 |
| `gradcam.py` | Grad-CAM 热力图，baseline vs ours 对比 | `runs/vis/` | 活跃 |

### F. 工具与验证（1 个）

| 脚本 | 用途 | 状态 |
|------|------|------|
| `verify_error_distribution.py` | 验证理论假设 σ(s) ∝ √s，检验误差分布与尺度的关系 | **有 bug**：L267 `cfg.get(path,...)` → `cfg.get('path',...)` 待修 |

---

## 依赖关系

```
register_modules.py
    ↑ 被所有训练脚本自动 import（必须最先执行）

ablation.py          → runs/ablation/{baseline, nwd_only, nwd_p2, ...}/
run_nwd_fixed.py     → runs/ablation/nwd_p2_fixed_c/result.json
run_nwd_reverse.py   → runs/ablation/nwd_p2_reverse_c/result.json
run_k_sensitivity.py → runs/ablation/nwd_p2_k{0p5,2p0,3p0}/ + k_sensitivity_results.json

collect_results.py   ← 读取所有 runs/ablation/*/result.json
generate_paper_figs.py ← 读取 runs/ablation/ + paper/figs/*.csv
eval.py              ← 读取指定 best.pt 文件
```

---

## 建议运行顺序（服务器）

```
1. train_baseline.py          建立基线（runs/ablation/baseline/）
2. ablation.py                主消融链（~36h）
3. run_new_queue.sh           Exp D/E/k-sweep（~42h，与 ablation 串行）
4. collect_results.py         汇总结果
5. eval.py                    补充效率指标
6. generate_paper_figs.py     生成论文图表
```
