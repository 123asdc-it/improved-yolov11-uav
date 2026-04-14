# GRSL 最小补丁方案

> 生成日期：2026-04-14
> 前提：服务器已丢失，选择路径 2（重租短期 GPU 补关键实验）
> 目标：用最少成本把论文从"消融表一堆 `--`"修补到"能投 IEEE GRSL（三区 letter）"
> 上游依赖：详见 `docs/05_投稿前体检报告.md`

---

## 一、目标期刊定位

**首选：IEEE GRSL（Geoscience and Remote Sensing Letters）**

- 三区，letter 格式 5 页（含参考文献）
- 不要求跨数据集验证（TGRS 才要求）
- 可以聚焦一个核心贡献
- 审稿周期通常 2-3 个月

**备选：Sensors / Remote Sensing（开放获取，MDPI）**

- 三区边缘 / 四区（视年份）
- 要求完整论文格式，但审稿速度更快（1-2 个月）
- 有 APC（版面费），~1000-2000 CHF

---

## 二、前提条件（必须先解决）

### 2.1 数据集恢复

- [ ] 从外接盘 / 学校电脑 / 学长处取回私有集 559 张原始图片 + 标注
- [ ] 本地恢复 `datasets/` 目录结构：`datasets/images/{train,val,test}` + `datasets/labels/{train,val,test}`
- [ ] 验证：用本机现有 `runs/ablation/baseline/weights/best.pt` 跑一次 val，确认 mAP50 能复现 0.9600（验证数据完整性）

### 2.2 服务器重租

- **推荐**：恒源云 RTX 3060 12GB（跟之前环境一致，避免环境适配折腾），¥1/h 左右
- **可选**：AutoDL 3060/3070 等同档 GPU 平台
- **不推荐**：4090/A100（显存过剩反而浪费钱，SA-NWD batch=8 只需 8.1GB）

### 2.3 环境准备（重租后 ~2 小时）

```bash
# 服务器端
mkdir -p /root/drone_detection
# 从本地 rsync 代码
rsync -avz ~/Work/Project/无人机/{configs,scripts,ultralytics_modules,yolo11n.pt} \
    root@server:/root/drone_detection/
# 上传数据集
rsync -avz ~/Work/Project/无人机/datasets/ \
    root@server:/root/drone_detection/datasets/
# Python 环境
conda create -n drone python=3.11 -y && conda activate drone
pip install ultralytics==8.4.22 torch torchvision
# 验证环境：跑 1 epoch baseline 确认 mAP 能起来
# (10 分钟，不是完整训练)
```

> **注意**：所有 `run_*.py` 脚本里 `PROJECT_ROOT = Path('/root/drone_detection')` 是硬编码的。新服务器必须用同样路径，否则改脚本。
>
> `configs/data.yaml` 里 `path: datasets` 是相对路径，只要 cwd 正确就能 work。

---

## 三、实验优先级分档

> 总原则：**越靠前的实验，不做越会让论文论点塌**。每项都标注了"不补怎么降级"的备选话术。

### Tier 1 — GRSL 绝对必须（不补不能投）

| # | 实验 | 脚本 | GPU 时长 | 意义 | 不补的后果 |
|:-:|------|------|:--------:|------|------|
| 1 | **Exp D** Fixed NWD (k=0) | `run_nwd_fixed.py` | ~6h | 证明自适应 C 不等于 NWD 加常数 | Remark 1 的 "validated" 就是谎言，审稿直拒 |
| 2 | **Exp E** Reverse C | `run_nwd_reverse.py` | ~6h | 证明 C 递减方向是对的，不是 Fisher 递增 | 同上，主论点失去支撑 |
| 3 | **FPS 重测** | `eval.py --weights nwd_p2/best.pt` | 5 分钟 | 修正摘要 FPS 占位 | 摘要注脚"待重测"投不出去 |

**Tier 1 总成本**：~12.5 GPU·h ≈ **¥15**

**Tier 1 产出**：Table 1 的 Exp D 行 + Exp E 行填完 + 摘要 FPS 填数。这就足够把论文从"不能投"推到"能投 GRSL"。

### Tier 2 — 强烈建议（审稿人会追问，不补要改话术）

| # | 实验 | 脚本 | GPU 时长 | 意义 | 不补的备选话术 |
|:-:|------|------|:--------:|------|------|
| 4 | **k-sweep** 两点（k=0.5, 2.0） | `run_k_sensitivity.py` 裁剪 | ~12h | Figure: k vs mAP50 折线图 | 删 Figure，加一句"we fix k=1.0 based on preliminary validation" |
| 5 | **E_p2only** P2 only 无 SA-NWD | `run_p2_only.py` | ~6h | 分离 P2 独立贡献（+0.76%） | 改 §III-C 贡献表述为"combined SA-NWD+P2 yields +1.81%" 不拆开 |

**Tier 2 总成本**：~18 GPU·h ≈ **¥18**

**Tier 2 产出**：k 折线图 + P2 独立贡献消融。让论文从"能投 GRSL" 推到"被接受概率 +20%"。

### Tier 3 — GRSL 可以砍（TGRS 才需要）

以下全部**不做**，用论文话术绕过：

- **alpha sweep** (0.3/0.7)：改为 "we adopt α=0.5 as a balanced choice following common practice"
- **E1** (W₂/C 分布统计)：GRSL 不写 Proposition 2，只保留 Theorem 1 够用
- **E2/E3** (TAL-only / Loss-only)：改为 "we apply SA-NWD to both TAL and loss, extending NWD-RKA~\cite{wang2022nwd}"
- **E4** (per-object vs batch-level Δ)：改 Key distinction 表述为 "per-object formulation is architecturally simpler, requiring no batch statistics"
- **E5** (NWD-NMS)：整节删除，NMS 不作为贡献
- **E6** (多种子)：改为 "we report single-seed results; multi-seed analysis is deferred to future work"
- **DUT Anti-UAV / AI-TOD 跨数据集**：GRSL 5 页装不下 §IV-D

---

## 四、三个套餐方案

### 📦 套餐 A：省钱版（仅 Tier 1）

- 成本：**¥15** + 服务器购买最低时长（大部分平台 ¥5/天）≈ **¥25-30 总花费**
- 时长：2 天（包含环境搭建 + 2 组实验串行 + FPS 测）
- 产出：Table 1 Exp D/E 行、摘要 FPS 填数
- 论文话术调整：砍 k-sensitivity 图、砍 P2-only 行、其余 Tier 3 按备选话术改
- **GRSL 接受概率：55-65%**

### 📦 套餐 B：推荐版（Tier 1 + Tier 2）

- 成本：**¥35** + 最低时长 ≈ **¥50-60 总花费**
- 时长：3-4 天（Tier 1 串行 + Tier 2 并行或串行）
- 产出：Table 1 完整填满 + k-sensitivity Figure + E_p2only 消融
- 论文话术调整：Tier 3 按备选话术改（不影响核心论点）
- **GRSL 接受概率：70-80%**

### 📦 套餐 C：稳健版（Tier 1 + Tier 2 + 部分 Tier 3）

- 额外加：alpha sweep（0.3/0.7）+ E2/E3（TAL/Loss-only）= 再 ~30 GPU·h
- 成本：**¥65** + 时长 ≈ **¥90-100 总花费**
- 时长：5-7 天
- 产出：接近当初 TGRS 版本的消融完整性
- **GRSL 接受概率：80-85%；Sensors 70-75%**（Sensors 对消融完整性更看重）

### 💡 我的建议：套餐 B

套餐 A 省¥20，但砍 k-sensitivity 图是硬伤（审稿人常常会专门问）；
套餐 C 多花¥40，但额外的 alpha/TAL 实验对 GRSL 是过度投入。
**套餐 B（¥50-60）是拐点最优**。

---

## 五、执行时间线（以套餐 B 为例）

```
Day 0（今天）
 ├─ 联系学长/找外接盘，取回私有集数据
 ├─ 在恒源云/AutoDL 下单 3060 GPU（~¥5 日租）
 └─ 等服务器开机通知

Day 1（上午 2h + 下午实验）
 ├─ 上午：rsync 代码 + 数据，conda 装环境，跑 1 epoch baseline 验证（10min）
 ├─ 下午：启动 Exp D（k=0, Fixed NWD）— 后台 6h
 └─ 本地：用 nwd_p2/best.pt 测 FPS（5 分钟）→ 填摘要

Day 2
 ├─ 上午：Exp D 完成，rsync result.json 回本地，填 Table 1
 ├─ 启动 Exp E（Reverse C）— 后台 6h
 └─ Tier 1 结束

Day 3
 ├─ 上午：Exp E 完成，填 Table 1
 ├─ 启动 k-sweep（k=0.5, 2.0 串行）— 后台 12h
 └─ 本地修改论文：清 main.tex 的 pending/TODO，填 FPS

Day 4
 ├─ 上午：k-sweep 完成，生成 k-sensitivity Figure（generate_paper_figs.py）
 ├─ 启动 E_p2only（P2 only）— 后台 6h
 └─ 本地：写 Remark 1 的验证段（引用 Exp D/E 结果）

Day 5
 ├─ 上午：E_p2only 完成，填最后一行 Table 1
 ├─ 服务器停机退租
 └─ 本地：Tier 3 实验全部按备选话术改 §IV

Day 6-7
 ├─ 全文 GRSL 5 页格式裁剪：
 │    - 删 §IV-D Cross-Dataset（TGRS 专用）
 │    - 压缩 §II Related Work
 │    - Theorem 1 证明放正文不放附录
 │    - 删除 Proposition 2 占位
 ├─ 作者信息填入
 ├─ Cover Letter 起草
 └─ 投稿前自检（参照 docs/04_按章节改稿提纲.md 的 15 项清单）
```

**总耗时：7 天，总花费 ~¥55**（含服务器 + 实验）

---

## 六、潜在风险与坑

| 风险 | 概率 | 应对 |
|------|:----:|------|
| 数据集找不回（学长失联 / 外接盘损坏） | 低 | 如真发生，切路径 1（砍 `--` 占位投 Access） |
| 重租服务器 Python 环境不兼容 ultralytics 8.4.22 | 中 | 下好镜像前测试 `pip install ultralytics==8.4.22` 能否跑通；如不能跑，改 script 适配新版 |
| `PROJECT_ROOT = /root/drone_detection` 路径不一致 | 中 | 新服务器用同路径 OR 用 sed 批量替换 |
| Exp D/E 结果"太接近" nwd_p2（< 0.5% Δ） | 中 | 说明自适应 C 贡献本就小；此时降调 Remark 1 的声称，改为 "SA-NWD 的主要价值在于 NWD 本身，adaptive C 提供额外鲁棒性"；不致命 |
| Exp E (Reverse C) 反而**超过** nwd_p2 | 低但严重 | Fisher 方向反而更好意味着论文核心叙事崩塌，此时必须重写 §III；**建议 Exp E 先跑，如果出这种结果提前决策** |
| k-sweep 显示 k=0.5 或 2.0 更优 | 低 | 论文主结果改为对应 k 值，重新评估 nwd_p2 不现实；此时保留 k=1.0 叙事，k-sweep 作为"鲁棒性"而非"最优点"展示 |
| 连续训练 3060 过热降频 | 低 | 恒源云有散热，单卡跑 ~30h 没问题；一次只跑一组实验，不并行 |

---

## 七、Tier 1 脚本快速参照

现有脚本都已经写好，只需确保服务器路径正确：

| 实验 | 脚本路径 | 输出目录 | 关键参数 |
|------|---------|---------|---------|
| Exp D | `scripts/run_nwd_fixed.py` | `runs/ablation/nwd_p2_fixed_c/` | k=0.0, alpha=0.5, nwd_min=0.0 |
| Exp E | `scripts/run_nwd_reverse.py` | `runs/ablation/nwd_p2_reverse_c/` | k=1.0（reverse 方向）, alpha=0.5, ref_area=0.002315 |
| k-sweep 裁剪版 | 改 `run_k_sensitivity.py` 的 `K_VALUES = [0.5, 2.0]` | `runs/ablation/nwd_p2_k{value}/` | 各自 alpha=0.5 |
| E_p2only | `scripts/run_p2_only.py` | `runs/ablation/p2_only/` | 无 patch_all_nwd，纯 P2 架构 |
| FPS 本地测 | `scripts/eval.py` | stdout | `--weights runs/ablation/nwd_p2/weights/best.pt --device mps`（Mac）或 `--device 0`（GPU） |

---

## 八、投稿前最终清单（纯文档层面）

完成套餐 B 所有实验后，论文还要做：

- [ ] 清 main.tex 所有 `--` 占位 → 真数字或删行
- [ ] 清所有 `pending`、`TODO`、`camera-ready version` 元标记
- [ ] 作者信息填入（L29-33）
- [ ] FPS 注脚（L42）改为具体数字 + GPU 型号
- [ ] Remark 1 (L171-173) 根据 Exp D/E 结果重写"validated"段
- [ ] 砍 §IV-D Cross-Dataset（TGRS 专用）
- [ ] 砍 §IV-E 里的 Proposition 2 占位段
- [ ] 压缩 §II Related Work（GRSL 5 页不够用）
- [ ] Cover Letter（贡献 3 句 + 适合 GRSL 理由 + 无一稿多投声明）
- [ ] 所有图表用 PDF 格式
- [ ] `paper/figs/` 里过时的图（generate_paper_figs.py 产物）重新生成

---

## 九、关键文件索引

| 文件 | 作用 |
|------|------|
| `docs/05_投稿前体检报告.md` | 上游诊断报告，列出所有红线/建议修/润色 |
| `docs/04_按章节改稿提纲.md` | 投稿前 15 项自检清单 |
| `scripts/run_nwd_fixed.py` | Exp D（Tier 1） |
| `scripts/run_nwd_reverse.py` | Exp E（Tier 1） |
| `scripts/run_k_sensitivity.py` | k-sweep（Tier 2） |
| `scripts/run_p2_only.py` | E_p2only（Tier 2） |
| `scripts/eval.py` | FPS 测量 |
| `configs/ablation/ablation_nwd_p2.yaml` | P2 架构配置（Exp D/E/k-sweep 都用） |
| `configs/ablation/ablation_p2.yaml` | 纯 P2 架构（E_p2only 用） |
