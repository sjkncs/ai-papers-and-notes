# Insider Purchase Signals in Microcap Equities: Gradient Boosting Detection of Abnormal Returns

> **arXiv:** [2602.06198](https://arxiv.org/abs/2602.06198) | **日期 / Date:** 2026-02 | **作者 / Authors:** Hangyi Zhao

---

## 关键摘录 / Key Excerpts

> 1. "Distance from the 52-week high dominates feature importance — contrary to mean-reversion expectations, buying after significant price surges produced the best outcomes."
>    / "距52周高点的距离主导特征重要性——与均值回归预期相反，在显著价格上涨后买入产生了最好的结果。"

> 2. "Trend validation identifies stronger executive confidence in less liquid trading environments of microcap equities."
>    / "趋势验证在流动性较低的小盘股交易环境中识别出更强的管理层信心。"

---

## Q1: 核心问题 / Core Problem

**中文：**
内部人士交易(insider trading)是合法的、公开披露的交易行为，但：
1. 小盘股(microcap, $30M-$500M)的内部人交易是否包含有价值的信息？
2. 如何从海量内部人交易中区分"信号"和"噪声"？
3. 传统的均值回归逻辑在小盘股中是否适用？

本文用6年数据、数千笔交易、梯度提升分类器回答这些问题。

**English:**
Insider trading (legally disclosed) raises questions:
1. Do insider transactions in microcaps ($30M-$500M) contain valuable information?
2. How to distinguish "signal" from "noise" among massive insider transactions?
3. Does traditional mean-reversion logic apply in microcaps?

This paper answers using 6 years of data, thousands of trades, and gradient boosting classification.

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **Gradient Boosting分类器**：基于内部人身份、交易历史、市场条件预测异常收益
2. **特征重要性发现**："距52周高点距离"是主导特征，而非传统认为的交易量或内部人级别
3. **反直觉发现**：在显著上涨后内部人买入 = 最强信号（趋势验证 > 均值回归）
4. **小盘股特异性**：流动性低的环境中，内部人信息优势被放大

**English:**

1. **Gradient Boosting Classifier**: Predicts abnormal returns based on insider identity, transaction history, market conditions
2. **Feature Importance Discovery**: "Distance from 52-week high" is the dominant feature, not volume or insider rank
3. **Counter-Intuitive Finding**: Insider buying after significant price surges = strongest signal (trend validation > mean reversion)
4. **Microcap Specificity**: Insider information advantage is amplified in low-liquidity environments

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **交易成本**：小盘股的高买卖价差是否侵蚀了alpha？
2. **流动性约束**：策略的实际可投资金额上限？
3. **法规变化**：SEC规则变化对内部人交易披露的影响？
4. **样本量**：6年数据中"信号"交易有多少？是否存在小样本过拟合？

**English:**

1. **Transaction Costs**: Do high bid-ask spreads in microcaps erode alpha?
2. **Liquidity Constraints**: Practical investable amount upper limit?
3. **Regulatory Changes**: Impact of SEC rule changes on insider trading disclosure?
4. **Sample Size**: How many "signal" trades in 6 years? Small-sample overfitting risk?

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**推荐阅读顺序：**
1. 第4节（特征工程）——理解哪些特征最重要
2. 第5节（结果）——重点关注反直觉发现
3. 第6节（稳健性检验）——交易成本和流动性敏感性

**量化金融映射方向：**
- 另类数据策略：SEC Form 4数据作为alpha信号源
- 小盘股alpha：流动性溢价的微观结构解释
- 梯度提升特征工程：金融特征重要性的方法论

**English:**

**Recommended Reading Order:**
1. Section 4 (Feature Engineering) — understand which features matter most
2. Section 5 (Results) — focus on counter-intuitive findings
3. Section 6 (Robustness Checks) — transaction cost and liquidity sensitivity

**Quant Finance Mapping:**
- Alternative data strategy: SEC Form 4 data as alpha signal source
- Microcap alpha: Microstructural explanation for liquidity premium
- Gradient boosting feature engineering: Methodology for financial feature importance
