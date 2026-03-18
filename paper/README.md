# paper/ — 论文文件目录

目标期刊：IEEE GRSL（短文，优先）/ IEEE TGRS（长文）
模板：IEEEtran journal 格式

---

## A. 当前状态（动态，随实验结果更新）

> 最近更新：2026-03-19

### 基本信息

| 项目 | 内容 |
|------|------|
| 目标期刊 | IEEE GRSL（5页短文，优先）/ IEEE TGRS（长文，需补公开数据集实验） |
| 当前页数 | 约 8 页（不含参考文献） |
| 论文标题 | SA-NWD: Scale-Adaptive Wasserstein Distance with Fisher-Guided Training for UAV-Based Small Object Detection |

### 占位符状态

| 位置 | 内容 | 状态 |
|------|------|------|
| 消融表 Exp D 行 | Fixed NWD (k=0) mAP50/P/R | ⏳ 待填（实验进行中） |
| 消融表 Exp E 行 | Reverse C mAP50/P/R | ⏳ 待填（实验进行中） |
| SOTA 表 NWD 行 | Fixed NWD mAP50（= Exp D 结果） | ⏳ 待填 |
| Analysis 小节 | k vs mAP50 折线图（Figure） | ⏳ 待 k-sweep 完成后生成 |
| §IV Dataset | 数据集来源 `\cite{}` | ❌ 缺失（审稿人必问） |
| Limitation 段 | 数据集规模 + 跨域泛化说明 | ❌ 缺失 |

### 已完成的论文修改（2026-03-19）

| 问题 | 修改方式 |
|------|---------|
| Theorem 2 跳步推导（C∝s→C∝√s 无依据） | 删除，改为正则化视角 + Design Validation remark |
| 摘要"providing stronger gradients" | 改为"scale-adaptive regularization" |
| Introduction/Related Work 声称唯一最优 C∝1/√s | 改为 annotation-noise 动机 + 实验验证叙述 |
| Theorem 1 术语"Fisher information" | 改为"expected squared gradient G_IoU" |
| TAL 公式 α 符号冲突 | TAL 改用 γ/δ，与混合损失的 α 区分 |
| refs.bib 3 条假引用（example2023 系列） | 删除 |
| BiFPN 在 abstract/related work 正面描述 | 删除，改为"消融证实负贡献" |
| Future Work 引用 VisDrone（任务不匹配） | 改为 Anti-UAV / Det-Fly |
| 消融表结构混乱，缺 Exp D/E 行 | 重组表格，加占位行 |
| SOTA 表仅有自己的方法 | 改为"loss function comparison"，加 NWD 固定 C 行 |

---

## B. 使用说明（静态）

### 文件说明

| 文件/目录 | 说明 | 状态 |
|---------|------|------|
| `main.tex` | 论文正文，IEEEtran journal 格式 | 活跃 |
| `refs.bib` | 参考文献库，已清理 3 条假引用 | 活跃 |
| `figs/` | 图表文件，pdf/png 双格式 + csv 原始数据 | 部分待更新 |

### figs/ 文件说明

| 文件 | 生成脚本 | 用于论文何处 | 状态 |
|------|---------|------------|------|
| `training_curves.{pdf,png}` | `plot_training_curves.py` | Figure：训练曲线（mAP50 + loss） | 已生成 |
| `ablation_bar.{pdf,png}` | `plot_results.py` | Figure：消融实验柱状图 | 已生成 |
| `pr_comparison.{pdf,png}` | `plot_results.py` | Figure：PR 曲线对比 | 已生成 |
| `error_distribution.png` | `verify_error_distribution.py` | 理论验证图（暂未入论文） | 已生成 |
| `baseline.csv` | ultralytics 训练日志 | 图表数据源 | 已有 |
| `nwd_only.csv` | ultralytics 训练日志 | 图表数据源 | 已有 |
| `nwd_p2.csv` | ultralytics 训练日志 | 图表数据源 | 已有 |
| `nwd_p2_simam_pconv.csv` | ultralytics 训练日志 | 图表数据源 | 已有 |
| k vs mAP50 折线图 | `generate_paper_figs.py`（待实现） | Figure：k 超参敏感性 | ⏳ 待生成 |

### 编译方法（Overleaf，推荐）

1. 打开 [overleaf.com](https://www.overleaf.com/)
2. New Project → Upload Project → 上传 `paper/` 目录下全部文件
3. 编译器选择 **pdfLaTeX**
4. 点击 Recompile

### 本地编译

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex   # 第三次确保交叉引用正确
open main.pdf
```

### 更新/重新生成图表

```bash
# 从项目根目录运行
python scripts/plot_training_curves.py   # 训练曲线
python scripts/plot_results.py           # 消融柱状图 + PR 曲线
python scripts/generate_paper_figs.py    # 所有论文图表（含 k-sweep 折线图，k-sweep 完成后）
# 输出均到 paper/figs/
```

### 投稿前检查清单

- [ ] 全文无 `--`、`待填` 占位符
- [ ] Table 所有数字与 `runs/ablation/*/result.json` 可对应
- [ ] Exp D/E 数据已填入消融表
- [ ] k-sweep 折线图已加入 Analysis 小节
- [ ] §IV Dataset 有数据集来源 `\cite{}`
- [ ] Limitation 段已添加
- [ ] refs.bib 所有条目可在公开数据库查到
- [ ] Abstract 与 Conclusion 口径一致
- [ ] 符号 α/γ/δ 全文唯一含义
