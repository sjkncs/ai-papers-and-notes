# Autonomous Factor Investing via Agentic AI

> **arXiv:** [2603.14288](https://arxiv.org/abs/2603.14288) | **日期 / Date:** 2026-03-14 | **作者 / Authors:** Allen Yikuan Huang, Zheqi Fan

---

## 关键摘录 / Key Excerpts

> 1. "We present an autonomous framework for systematic factor investing via agentic AI, replacing manual input with a self-directed engine that generates signals independently."
>    / "我们提出了一个通过Agentic AI进行系统化因子投资的自主框架，用自主引擎替代人工输入，独立生成投资信号。"

> 2. "Out-of-sample validation shows an annualized Sharpe ratio of 3.11, demonstrating a scalable and interpretable paradigm."
>    / "样本外验证显示年化夏普比率为3.11，展示了一个可扩展且可解释的范式。"

---

## Q1: 核心问题 / Core Problem

**中文：**
传统因子投资依赖人工因子构建、回测和选择，流程缓慢且受限于人类认知偏差。核心问题：
1. 因子发现高度依赖专家经验和直觉
2. 因子衰减(alpha decay)需要持续人工监控和更新
3. 因子组合的再平衡依赖手动决策
4. 从数据到交易信号的端到端自动化程度低

本文用Agentic AI构建自主因子投资引擎，实现从数据摄取→因子挖掘→组合构建→交易执行的端到端自主化。

**English:**
Traditional factor investing relies on manual factor construction, backtesting, and selection — slow and limited by human cognitive bias. Core problems:
1. Factor discovery highly dependent on expert experience and intuition
2. Alpha decay requires continuous manual monitoring and updates
3. Factor portfolio rebalancing relies on manual decisions
4. Low end-to-end automation from data to trading signals

This paper uses Agentic AI to build an autonomous factor investing engine, achieving end-to-end autonomy from data ingestion → factor mining → portfolio construction → trade execution.

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **自主因子发现引擎 (Autonomous Factor Discovery Engine)**：
   - LLM驱动的因子假设生成
   - 自动回测验证和筛选
   - 因子组合的自适应权重调整

2. **样本外Sharpe 3.11**：
   - 严格样本外验证（无数据窥探）
   - 覆盖多个资产类别和市场regime
   - 经济意义检验（因子暴露分析）

3. **可解释性 (Interpretability)**：
   - 每个因子的经济学逻辑可追溯
   - 因子贡献归因透明
   - 组合决策过程可审计

4. **可扩展架构 (Scalable Architecture)**：
   - 模块化设计支持新增因子类别
   - 多资产类别统一框架
   - 低延迟信号生成

**English:**

1. **Autonomous Factor Discovery Engine**: LLM-driven factor hypothesis generation; automatic backtest validation and screening; adaptive weight adjustment for factor portfolios.

2. **Out-of-Sample Sharpe 3.11**: Strict out-of-sample validation (no data snooping); covers multiple asset classes and market regimes; economic significance tests (factor exposure analysis).

3. **Interpretability**: Each factor's economic logic is traceable; transparent factor contribution attribution; auditable portfolio decision process.

4. **Scalable Architecture**: Modular design supports new factor categories; unified multi-asset framework; low-latency signal generation.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **Sharpe 3.11的合理性**：如此高的Sharpe是否暗示过拟合或数据问题？
2. **交易成本**：因子换手率和实际交易成本的影响？
3. **因子衰减**：自主引擎如何应对因子alpha的自然衰减？
4. **LLM幻觉风险**：AI生成的因子假设是否可能基于错误的经济逻辑？

**English:**

1. **Sharpe 3.11 Plausibility**: Does such high Sharpe suggest overfitting or data issues?
2. **Transaction Costs**: Factor turnover and actual trading cost impact?
3. **Alpha Decay**: How does the autonomous engine handle natural factor alpha decay?
4. **LLM Hallucination Risk**: Could AI-generated factor hypotheses be based on flawed economic logic?

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**推荐阅读顺序：**
1. 第3节（框架架构）——理解自主引擎的完整流程
2. 第4节（因子发现）——LLM如何生成和验证因子
3. 第5节（实验结果）——重点关注样本外表现和稳健性
4. 第6节（可解释性）——因子经济学逻辑的追溯

**量化金融映射方向：**
- 直接构建自主因子挖掘系统
- LLM驱动的alpha研究自动化
- 因子组合的实时自适应管理

**English:**

**Recommended Reading Order:**
1. Section 3 (Framework Architecture) — full autonomous engine pipeline
2. Section 4 (Factor Discovery) — how LLM generates and validates factors
3. Section 5 (Experimental Results) — focus on out-of-sample and robustness
4. Section 6 (Interpretability) — tracing factor economic logic

**Quant Finance Mapping:**
- Directly build autonomous factor mining systems
- LLM-driven alpha research automation
- Real-time adaptive factor portfolio management
