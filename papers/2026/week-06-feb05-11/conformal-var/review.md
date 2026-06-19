# Taming Tail Risk in Financial Markets: Conformal Risk Control for Nonstationary Portfolio VaR

> **arXiv:** [2602.03903](https://arxiv.org/abs/2602.03903) | **日期 / Date:** 2026-02-07 | **作者 / Authors:** Marc Schmitt

---

## 关键摘录 / Key Excerpts

> 1. "Financial losses are non-stationary and regime-dependent, making static VaR estimates unreliable. We introduce regime-weighted conformal risk control that adapts protective margins using historical errors with exponential time decay and regime-similarity metrics."
>    / "金融损失是非平稳的且依赖regime，使静态VaR估计不可靠。我们引入regime加权保形风险控制，使用指数时间衰减和regime相似性度量的历史误差来自适应保护边际。"

> 2. "Our method functions independently of specific architectures, wrapping any conditional quantile forecaster to achieve targeted breach frequencies with finite-sample guarantees under weighted exchangeability."
>    / "我们的方法独立于特定架构运行，包裹任何条件分位数预测器，在加权可交换性下以有限样本保证实现目标突破频率。"

---

## Q1: 核心问题 / Core Problem

**中文：**
Value-at-Risk (VaR) 是金融监管和风险管理的核心指标，但面临：
1. **非平稳性**：市场regime切换导致VaR估计失效
2. **模型依赖性**：不同预测模型给出不同的VaR估计
3. **缺乏保证**：传统方法无法保证VaR的实际覆盖率

保形预测(Conformal Prediction)提供了一种分布无关的有限样本保证框架，但标准的保形预测假设数据可交换(i.i.d.)，这在金融时序中不成立。

本文提出Regime-Weighted Conformal Risk Control，通过：
- 指数时间衰减降低旧误差的权重
- Regime相似性度量重新加权历史残差
- 在任何条件分位数预测器上提供"黑盒"包装

**English:**
Value-at-Risk (VaR) is core to financial regulation and risk management, but faces:
1. **Non-stationarity**: Market regime switches invalidate VaR estimates
2. **Model dependence**: Different prediction models give different VaR estimates
3. **Lack of guarantees**: Traditional methods cannot guarantee actual VaR coverage

Conformal Prediction provides a distribution-free finite-sample guarantee framework, but standard conformal prediction assumes data exchangeability (i.i.d.), which doesn't hold in financial time series.

This paper proposes Regime-Weighted Conformal Risk Control via:
- Exponential time decay to reduce weight of old errors
- Regime similarity metrics to reweight historical residuals
- "Black-box" wrapper on any conditional quantile forecaster

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **Regime加权保形预测 (Regime-Weighted Conformal Prediction)**：
   - 定义regime相似度函数（基于市场状态特征向量）
   - 加权经验分位数：相近regime的残差获得更高权重
   - 证明加权可交换性条件下的有限样本覆盖保证

2. **模型无关的包装器 (Model-Agnostic Wrapper)**：
   - 可包裹任何分位数回归模型（QRNN, Quantile Forest等）
   - 不修改内部模型结构
   - 即插即用的VaR校正

3. **自适应边际调整 (Adaptive Margin Adjustment)**：
   - 指数衰减率自动调整
   - 市场高波动期自动扩大保护边际
   - 低波动期收窄边际提高资本效率

4. **实证验证 (Empirical Validation)**：
   - 多种资产类别（股票、债券、商品、加密）
   - 覆盖频率精确匹配目标（如95% VaR实际突破≤5%）
   - 与传统GARCH-VaR、历史模拟法的对比

**English:**

1. **Regime-Weighted Conformal Prediction**: Define regime similarity function (based on market state feature vector); weighted empirical quantiles: residuals from similar regimes get higher weights; prove finite-sample coverage guarantee under weighted exchangeability.

2. **Model-Agnostic Wrapper**: Can wrap any quantile regression model (QRNN, Quantile Forest, etc.); doesn't modify internal model structure; plug-and-play VaR correction.

3. **Adaptive Margin Adjustment**: Automatic exponential decay rate adjustment; automatically widen protective margins in high volatility; narrow margins in low volatility for capital efficiency.

4. **Empirical Validation**: Multiple asset classes (stocks, bonds, commodities, crypto); coverage frequency precisely matches target (e.g., 95% VaR actual breach ≤5%); comparison with traditional GARCH-VaR, historical simulation.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **Regime定义的主观性**：regime相似性函数如何定义？是否引入人为偏差？
2. **加权可交换性的现实性**：金融数据在多大程度上满足此假设？
3. **计算实时性**：对高频交易场景的适用性？
4. **极端尾部事件**：黑天鹅事件中保形预测的保证是否仍然有效？

**English:**

1. **Regime Definition Subjectivity**: How is regime similarity function defined? Does it introduce human bias?
2. **Weighted Exchangeability Realism**: To what extent do financial data satisfy this assumption?
3. **Computational Real-Time**: Applicability to high-frequency trading scenarios?
4. **Extreme Tail Events**: Do conformal prediction guarantees still hold in black swan events?

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**推荐阅读顺序：**
1. 第2节（保形预测回顾）——理解理论基础
2. 第3节（方法）——regime加权机制的设计
3. 第4节（理论保证）——有限样本覆盖证明
4. 第5节（实验）——多资产类别的VaR评估

**量化金融映射方向：**
- 组合风险管理系统中的VaR校正模块
- 压力测试：生成极端regime下的VaR估计
- 监管资本计算：满足Basel III/IV的模型验证要求

**English:**

**Recommended Reading Order:**
1. Section 2 (Conformal Prediction Review) — theoretical foundations
2. Section 3 (Method) — regime weighting mechanism design
3. Section 4 (Theoretical Guarantees) — finite-sample coverage proof
4. Section 5 (Experiments) — multi-asset class VaR evaluation

**Quant Finance Mapping:**
- VaR correction module in portfolio risk management systems
- Stress testing: VaR estimates under extreme regimes
- Regulatory capital calculation: meeting Basel III/IV model validation requirements
