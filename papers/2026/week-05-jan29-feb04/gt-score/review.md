# GT-Score: A Robust Objective Function for Reducing Overfitting in Data-Driven Trading Strategies

> **arXiv:** [2602.00080](https://arxiv.org/abs/2602.00080) | **日期 / Date:** 2026-01-29 | **作者 / Authors:** (Quantitative Finance Researchers)

---

## 关键摘录 / Key Excerpts

> 1. "Data-driven trading strategies are notoriously prone to overfitting — the GT-Score introduces a structural penalty that separates genuine predictive signal from noise memorization."
>    / "数据驱动的交易策略极易过拟合——GT-Score引入结构性惩罚，将真实预测信号与噪声记忆区分开来。"

> 2. "Unlike traditional cross-validation, GT-Score accounts for the temporal dependence structure of financial returns and the multiple-testing nature of strategy selection."
>    / "与传统交叉验证不同，GT-Score考虑了金融收益的时序依赖性和策略选择的多重检验本质。"

---

## Q1: 核心问题 / Core Problem

**中文：**
量化交易策略开发中的过拟合是最致命的问题之一：
1. **数据窥探偏差 (Data Snooping)**：在大量策略变体中挑选历史表现最好的
2. **非平稳性**：金融数据的分布随时间变化，历史好的策略未来未必好
3. **多重检验**：测试1000个策略，即使全是随机的也会有几个看起来很好
4. **传统CV失效**：标准K折交叉验证假设数据i.i.d.，违反时序依赖

GT-Score提出一个鲁棒目标函数，在训练过程中内嵌防过拟合机制，而非依赖事后检验。

**English:**
Overfitting is the most critical problem in quant trading strategy development:
1. **Data Snooping**: Selecting the best historical performer from many strategy variants
2. **Non-stationarity**: Financial data distributions change over time; past performance ≠ future
3. **Multiple Testing**: Testing 1000 strategies, even all random, a few will look good
4. **Traditional CV Failure**: Standard K-fold CV assumes i.i.d. data, violating temporal dependence

GT-Score proposes a robust objective function with built-in anti-overfitting during training, not post-hoc testing.

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **GT-Score目标函数 (GT-Score Objective)**：
   - 结合收益预测准确性和策略泛化能力
   - 内嵌时序稳定性惩罚：要求策略在滚动窗口上表现一致
   - 多重检验校正：基于策略搜索空间大小自动调整惩罚强度

2. **时序感知交叉验证 (Time-Aware Cross-Validation)**：
   - Purged K-Fold: 在训练和验证之间添加"清洗期"
   - Embargo CV: 禁止使用验证期后的任何信息
   - 自适应K值：根据自相关长度选择最优折叠数

3. **策略复杂度度量 (Strategy Complexity Measure)**：
   - 参数量 + 特征数 + 条件分支数的综合度量
   - 复杂度越高，GT-Score惩罚越大
   - 自动平衡收益和简洁性 (Occam's Razor)

4. **实验验证 (Empirical Validation)**：
   - 在美股、加密货币、外汇数据上验证
   - GT-Score vs 标准MSE vs Sharpe优化
   - 样本外表现的显著提升

**English:**

1. **GT-Score Objective**: Combines return prediction accuracy with strategy generalization; temporal stability penalty requiring consistent rolling-window performance; multiple testing correction auto-adjusting penalty by search space size.

2. **Time-Aware Cross-Validation**: Purged K-Fold with "washout period" between train/validation; Embargo CV forbidding post-validation info; adaptive K based on autocorrelation length.

3. **Strategy Complexity Measure**: Combined parameter count + feature count + conditional branch count; higher complexity → stronger GT-Score penalty; automatic balance of returns and simplicity (Occam's Razor).

4. **Empirical Validation**: US stocks, crypto, forex; GT-Score vs standard MSE vs Sharpe optimization; significant out-of-sample improvement.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **GT-Score超参数选择**：惩罚强度如何设定？是否存在过正则化风险？
2. **计算成本**：时序感知CV的计算开销是否可接受？
3. **策略空间定义**：如何准确度量策略搜索空间的大小？
4. **实盘验证缺失**：是否考虑了交易成本、滑点、流动性？

**English:**

1. **GT-Score Hyperparameters**: How to set penalty strength? Risk of over-regularization?
2. **Computational Cost**: Is time-aware CV overhead acceptable?
3. **Strategy Space Definition**: How to accurately measure strategy search space size?
4. **Missing Live Validation**: Transaction costs, slippage, liquidity considered?

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**推荐阅读顺序：**
1. 第3节（GT-Score定义）——理解目标函数的数学构造
2. 第4节（时序CV）——掌握Purged/Embargo CV的实现
3. 第5节（实验）——重点关注样本外表现的改善幅度
4. 附录（复杂度度量）——了解策略空间的形式化定义

**量化金融映射方向：**
- 所有量化策略开发的默认损失函数替换
- 因子挖掘中的防过拟合框架
- 策略组合的鲁棒优化

**English:**

**Recommended Reading Order:**
1. Section 3 (GT-Score Definition) — mathematical construction of the objective
2. Section 4 (Temporal CV) — Purged/Embargo CV implementation
3. Section 5 (Experiments) — focus on out-of-sample improvement magnitude
4. Appendix (Complexity Measure) — formal strategy space definition

**Quant Finance Mapping:**
- Default loss function replacement for all quant strategy development
- Anti-overfitting framework for factor mining
- Robust optimization for strategy portfolios
