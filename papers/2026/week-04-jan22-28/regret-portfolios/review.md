# Regret-Driven Portfolios: LLM-Guided Smart Clustering for Optimal Allocation

> **arXiv:** [2601.17021](https://arxiv.org/abs/2601.17021) | **日期 / Date:** 2026-01-22 | **作者 / Authors:** Muhammad Abro, Hassan Jaleel

---

## 关键摘录 / Key Excerpts

> 1. "We introduce a regret-minimization portfolio framework that merges continuous algorithmic adaptation with LLM-derived market sentiment metrics to build high-yield, low-volatility portfolios."
>    / "我们引入一个遗憾最小化组合框架，将持续算法适应与LLM衍生的市场情绪指标相结合，构建高收益、低波动组合。"

> 2. "The LLM-guided smart clustering groups assets by semantic similarity in their news narratives, enabling diversification that goes beyond traditional return-based correlation."
>    / "LLM引导的智能聚类按新闻叙事的语义相似性对资产分组，实现超越传统基于收益相关性的分散化。"

> 3. "Empirical tests show the strategy outperforms SPY buy-and-hold by substantial margins in both annual returns and Sharpe ratio."
>    / "实证测试显示该策略在年化收益和夏普比率上均显著跑赢SPY买入持有基准。"

---

## Q1: 核心问题 / Core Problem

**中文：**
传统组合优化（如均值-方差）面临两大挑战：
1. **输入敏感性**：对预期收益和协方差矩阵的估计误差极度敏感
2. **信息维度有限**：仅使用历史收益数据，忽略了文本、情绪等另类数据

本文提出利用LLM的语义理解能力来增强组合构建：
- LLM对金融新闻进行语义聚类，识别"叙事相关"资产组
- 在叙事相关组内进行遗憾最小化权重分配
- 动态跟踪市场情绪变化，调整聚类结构

核心问题：如何将LLM的非结构化文本理解能力转化为可操作的投资信号？

**English:**
Traditional portfolio optimization (e.g., mean-variance) faces two challenges:
1. **Input sensitivity**: Extremely sensitive to estimation errors in expected returns and covariance
2. **Limited information dimension**: Uses only historical returns, ignoring alternative data like text/sentiment

This paper leverages LLM semantic understanding to enhance portfolio construction:
- LLM performs semantic clustering on financial news, identifying "narrative-correlated" asset groups
- Regret-minimization weight allocation within narrative-correlated groups
- Dynamic tracking of sentiment changes to adjust clustering structure

Core question: How to transform LLM's unstructured text understanding into actionable investment signals?

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **LLM引导智能聚类 (LLM-Guided Smart Clustering)**：
   - 使用LLM embedding对金融新闻进行语义向量化
   - 层次聚类将语义相似的资产分组
   - 与传统行业分类和收益相关性聚类的对比优势
   - 动态更新聚类结构（月度/事件驱动）

2. **遗憾最小化权重分配 (Regret-Minimization Allocation)**：
   - Follow-the-Leader策略在每个聚类内进行资产配置
   - 后悔值作为动态风险度量
   - 结合下行保护机制（最大回撤约束）

3. **情绪增强信号 (Sentiment-Enhanced Signals)**：
   - LLM生成的市场情绪分数
   - 情绪极端值触发再平衡
   - 情绪趋势作为动量信号的补充

4. **实验验证 (Empirical Validation)**：
   - 回测期：2015-2025（覆盖多次市场危机）
   - 年化超额收益、夏普比率、最大回撤全面优于SPY
   - 交易成本敏感性分析

**English:**

1. **LLM-Guided Smart Clustering**: LLM embeddings for financial news vectorization; hierarchical clustering of semantically similar assets; advantages over traditional industry classification and return-based clustering; dynamic clustering updates (monthly/event-driven).

2. **Regret-Minimization Allocation**: Follow-the-Leader strategy within each cluster; regret value as dynamic risk measure; downside protection (max drawdown constraint).

3. **Sentiment-Enhanced Signals**: LLM-generated market sentiment scores; extreme sentiment triggers rebalancing; sentiment trend complements momentum signals.

4. **Empirical Validation**: Backtest 2015-2025 (covering multiple market crises); superior annual excess return, Sharpe ratio, max drawdown vs SPY; transaction cost sensitivity analysis.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **LLM幻觉与偏差 (LLM Hallucination & Bias)**：
   - LLM对金融文本的理解可能产生系统性偏差
   - 情绪评分的一致性和稳定性如何保证？
   - 模型版本更新对策略的影响（可复现性问题）

2. **前视偏差与实现细节 (Look-Ahead Bias)**：
   - 新闻发布时间与交易执行的时间差
   - 聚类更新是否使用了未来信息？
   - 交易成本和滑点的实际影响

3. **过拟合风险 (Overfitting Risk)**：
   - 超参数（聚类数、再平衡频率等）的鲁棒性
   - 样本内vs样本外性能差距
   - 不同LLM模型间的一致性

4. **基准选择的公平性 (Benchmark Fairness)**：
   - 仅与SPY买入持有对比是否足够？
   - 需要与成熟的量化策略（如动量、价值因子）对比
   - 风险调整后收益的统计显著性检验

**English:**

1. **LLM Hallucination & Bias**: Systematic bias in financial text understanding? Sentiment score consistency and stability? Model version update impact (reproducibility)?
2. **Look-Ahead Bias**: Time gap between news publication and trade execution? Does clustering update use future information? Actual impact of transaction costs and slippage?
3. **Overfitting Risk**: Robustness of hyperparameters (cluster count, rebalance frequency)? In-sample vs out-of-sample gap? Consistency across different LLM models?
4. **Benchmark Fairness**: SPY buy-hold only sufficient? Need comparison with established quant strategies (momentum, value factors)? Statistical significance of risk-adjusted returns?

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**推荐阅读顺序 / Recommended Reading Order:**
1. 第3节（方法）——理解LLM聚类+遗憾最小化的完整流程
2. 第4节（实验）——关注回测结果的统计显著性
3. 第2节（相关工作）——LLM在金融中的应用综述
4. 附录（超参数敏感性）——鲁棒性分析

**关键方法论需要掌握 / Key Methodology to Master:**
- LLM embedding在金融文本中的应用
- 在线学习中的遗憾最小化（Follow-the-Leader, Hedge等）
- 层次聚类与语义相似性度量
- 组合风险度量（VaR, CVaR, Max Drawdown）

**潜在研究方向 / Potential Research Directions:**
- 多LLM集成投票提高情绪信号质量
- 将聚类从语义扩展到因果推理（因果图组合）
- 实时新闻流的在线聚类与交易执行
- 结合知识图谱的结构化情绪分析

**English:**

**Recommended Reading Order:**
1. Section 3 (Method) — full pipeline of LLM clustering + regret minimization
2. Section 4 (Experiments) — focus on statistical significance of backtest results
3. Section 2 (Related Work) — LLM applications in finance survey
4. Appendix (Hyperparameter Sensitivity) — robustness analysis

**Key Methodology to Master:**
- LLM embeddings for financial text applications
- Online learning regret minimization (Follow-the-Leader, Hedge, etc.)
- Hierarchical clustering with semantic similarity measures
- Portfolio risk measures (VaR, CVaR, Max Drawdown)

**Potential Research Directions:**
- Multi-LLM ensemble voting to improve sentiment signal quality
- Extend clustering from semantic to causal reasoning (causal graph portfolios)
- Online clustering and trade execution with real-time news streams
- Structured sentiment analysis combined with knowledge graphs
