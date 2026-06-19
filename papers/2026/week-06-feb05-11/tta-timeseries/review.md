# Test-Time Adaptation for Non-stationary Time Series: From Synthetic Regime Shifts to Financial Markets

> **arXiv:** [2602.00073](https://arxiv.org/abs/2602.00073) | **日期 / Date:** 2026-02-05 | **作者 / Authors:** Yurui Wu, Qingying Deng, Wonou Chung, Mairui Li

---

## 关键摘录 / Key Excerpts

> 1. "We investigate a lightweight adaptation technique during inference for handling distribution shifts in time series — only modifying scaling and shifting parameters while keeping the main network frozen."
>    / "我们研究了一种推理时的轻量级适应技术来处理时序分布漂移——仅修改缩放和偏移参数，保持主网络冻结。"

> 2. "We discover that simple numerical updates perform optimally in genuine financial climates, whereas aggressive modifications can degrade predictions."
>    / "我们发现简单的数值更新在真实金融环境中表现最优，而激进的修改反而可能降低预测质量。"

---

## Q1: 核心问题 / Core Problem

**中文：**
金融时序数据具有强非平稳性——市场regime（牛市/熊市/震荡市）频繁切换，导致预训练模型的预测性能急剧下降。传统方法要么：
1. 完全重训模型（计算成本高，滞后性大）
2. 使用固定模型（无法适应新regime）

本文提出Test-Time Adaptation (TTA)：在推理时仅调整BatchNorm的缩放/偏移参数（γ, β），不修改主网络权重，用极低的计算成本实现对新regime的快速适应。

**English:**
Financial time series is strongly non-stationary — market regimes (bull/bear/sideways) switch frequently, causing pretrained model prediction performance to degrade sharply. Traditional methods either:
1. Fully retrain models (high compute cost, significant lag)
2. Use fixed models (cannot adapt to new regimes)

This paper proposes Test-Time Adaptation (TTA): at inference time, only adjust BatchNorm scale/shift parameters (γ, β) without modifying main network weights, achieving rapid adaptation to new regimes with minimal compute cost.

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **轻量级推理时适应 (Lightweight Inference-Time Adaptation)**：
   - 仅调整γ, β参数（<1%参数量）
   - 无需标签：利用无标注新数据的自监督信号
   - 实时更新，适应流式数据

2. **分类任务的熵最小化+时序一致性 (Classification: Entropy Min + Temporal Consistency)**：
   - 最小化预测熵增强方向确定性
   - 强制相邻时间步预测一致

3. **回归任务的EMA教师+小扰动 (Regression: EMA Teacher + Small Perturbation)**：
   - 指数移动平均教师模型提供伪标签
   - 轻微时序扰动保持输出稳定性

4. **关键发现 (Key Finding)**：
   - 在合成漂移场景中，激进适应有效
   - 在真实金融数据中，简单数值更新（仅调整统计量）最优
   - 过度适应在噪声大的金融数据中反而有害

**English:**

1. **Lightweight Inference-Time Adaptation**: Only adjusts γ, β parameters (<1% params); no labels needed: self-supervised signal from unlabeled new data; real-time updates for streaming data.

2. **Classification: Entropy Min + Temporal Consistency**: Minimize prediction entropy for directional certainty; enforce adjacent timestep prediction consistency.

3. **Regression: EMA Teacher + Small Perturbation**: Exponential moving average teacher provides pseudo-labels; slight temporal perturbation maintains output stability.

4. **Key Finding**: Aggressive adaptation works in synthetic shift scenarios; simple numerical updates (only adjusting statistics) are optimal in real financial data; over-adaptation is harmful in noisy financial data.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **适应的稳定性保证**：如何防止适应过程中参数发散？
2. **多资产场景**：不同资产可能有不同regime，如何处理？
3. **与在线学习的对比**：与标准在线梯度下降相比的优势？
4. **超参数敏感性**：EMA衰减率、扰动幅度等参数的选择

**English:**

1. **Adaptation Stability Guarantee**: How to prevent parameter divergence during adaptation?
2. **Multi-Asset Scenarios**: Different assets may have different regimes — how to handle?
3. **vs Online Learning**: Advantages over standard online gradient descent?
4. **Hyperparameter Sensitivity**: EMA decay rate, perturbation magnitude parameter choices

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**推荐阅读顺序：**
1. 第3节（方法）——TTA的具体实现
2. 第5节（金融实验）——真实金融数据的评估
3. 第6节（分析）——为什么简单方法在金融中更好

**量化金融映射方向：**
- 量化策略的在线自适应：不重训模型但适应新市场状态
- Regime检测：通过TTA损失变化检测regime切换
- 组合管理：不同策略模块的独立适应

**English:**

**Recommended Reading Order:**
1. Section 3 (Method) — TTA specific implementation
2. Section 5 (Financial Experiments) — real financial data evaluation
3. Section 6 (Analysis) — why simple methods work better in finance

**Quant Finance Mapping:**
- Quant strategy online self-adaptation: adapt to new market states without retraining
- Regime detection: detect regime switches via TTA loss changes
- Portfolio management: independent adaptation of different strategy modules
