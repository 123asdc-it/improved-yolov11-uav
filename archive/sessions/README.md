# archive/sessions/ — AI 会话决策记录

存放与 AI 的完整对话记录，含实验决策、理论分析、方案选择等重要上下文。
**禁止删除**，可用于论文 Discussion 写作、方案回溯、向他人解释决策过程。

---

## `session-ses_3068.md`

- **日期**：2026-03-17（09:52 — 15:50）
- **主要内容**：

  1. **ultralytics 框架解释**：为什么必须用它，monkey-patch 的工作方式
  2. **早期实验结果梳理**：消融链路 baseline(0.9600) → nwd_only(0.9705) → nwd_p2(0.9781) 的数据和分析
  3. **SA-NWD vs Fisher-CIoU 第一轮分析**：为什么 Wasserstein 距离比 Fisher 补偿更有效
  4. **nwd_p2 确认为最优配置**：mAP50=0.9781，放弃 SimAM(-2.62%)/PConv/BiFPN(-3.84%)

- **关键决策**：确定 nwd_p2（SA-NWD + P2 头）为最终方案，Fisher-CIoU 系列列为待验证实验

---

## `session-ses_2fff.md`

- **日期**：2026-03-18
- **主要内容**：

  1. **Theorem 2 推导问题发现**：C∝s → C∝√s 的跳步无依据，与代码实现方向（C∝1/s）相反
  2. **Fisher-CIoU 失败的根本原因**：SA-NWD（大 C 放松小目标）与 Fisher-CIoU（大权重加强小目标）方向相反，含完整数值推导（W₂/C 工作点分析）
  3. **Exp D/E 设计**：如何通过对照实验验证 C 方向选择的合理性；ref_area 归一化方案的数值推导（为什么选 c_base×(1+k×√(S/ref)) 而非其他公式）
  4. **论文理论叙述重构**：从"Fisher 等变最优"改为"标注噪声正则化"视角的完整讨论过程
  5. **旧队列替换决策**：杀掉 Fisher A/B/C 队列，重新部署 Exp D/E/k-sweep 队列；7 个实验的完整参数表

- **关键决策**：
  - 删除 Theorem 2（改正则化视角）
  - 设计 Exp D（固定 C 对照）和 Exp E（反向 C 对照）
  - k-sweep 结果以折线图形式展示于 Analysis 小节（不加主消融表）
  - 不使用 VisDrone（任务不匹配，已确认）
