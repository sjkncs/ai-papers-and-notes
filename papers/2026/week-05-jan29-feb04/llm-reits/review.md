# LLM Multi-Agent Investment System for Chinese Public REITs

> **arXiv:** [2602.00082](https://arxiv.org/abs/2602.00082) | **日期 / Date:** 2026-01-30 | **作者 / Authors:** (Multi-Agent Finance Researchers)

---

## 关键摘录 / Key Excerpts

> 1. "We design a collaborative AI framework where specialized LLM agents — acting as fundamental analyst, technical analyst, risk manager, and portfolio optimizer — collectively evaluate Chinese public REITs."
>    / "我们设计了一个协作AI框架，其中专业化的LLM代理——分别扮演基本面分析师、技术分析师、风险经理和组合优化师——共同评估中国公募REITs。"

> 2. "The multi-agent debate mechanism allows agents to challenge each other's assessments, reducing individual bias and producing more balanced investment recommendations."
>    / "多代理辩论机制允许代理互相挑战评估，减少个体偏差并产生更平衡的投资建议。"

---

## Q1: 核心问题 / Core Problem

**中文：**
中国公募REITs市场面临独特挑战：
1. **信息碎片化**：REITs涉及底层资产（高速公路、产业园、仓储物流等），信息分散
2. **定价复杂**：需同时考虑资产估值、现金流分配、政策影响
3. **分析师覆盖不足**：相比A股，REITs的卖方研报覆盖率低
4. **投资者教育不足**：散户对REITs的估值逻辑理解有限

本文提出LLM多代理系统，让不同角色的AI分析师协作决策，弥补人工覆盖不足。

**English:**
Chinese public REITs market faces unique challenges:
1. **Information fragmentation**: REITs involve underlying assets (highways, industrial parks, warehouses), with scattered info
2. **Pricing complexity**: Requires simultaneous consideration of asset valuation, cash flow distribution, policy impacts
3. **Insufficient analyst coverage**: Lower sell-side research coverage vs A-shares
4. **Limited investor education**: Retail investors have limited understanding of REITs valuation logic

This paper proposes an LLM multi-agent system where different AI analyst roles collaborate to compensate for insufficient human coverage.

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **四角色代理架构 (Four-Role Agent Architecture)**：
   - 基本面分析师：分析REITs底层资产的现金流、出租率、资本开支
   - 技术分析师：分析价格走势、成交量模式、市场情绪
   - 风险经理：评估利率风险、政策风险、集中度风险
   - 组合优化师：综合前三者观点，输出最优配置建议

2. **多代理辩论机制 (Multi-Agent Debate Mechanism)**：
   - 每个代理独立评估后公开观点
   - 其他代理可质疑并反驳
   - 多轮辩论后达成共识
   - 减少单一代理的幻觉和偏差

3. **中国REITs特化知识 (China REITs-Specific Knowledge)**：
   - 嵌入中国REITs估值框架（DCF、NAV、FFO倍数）
   - 理解中国政策环境（土地制度、税收优惠）
   - 处理中文财报和公告

4. **回测验证 (Backtest Validation)**：
   - 2021-2025中国公募REITs数据
   - 与单一LLM和人工基准对比
   - 风险调整收益的显著提升

**English:**

1. **Four-Role Agent Architecture**: Fundamental analyst (cash flows, occupancy, capex); Technical analyst (price trends, volume patterns, sentiment); Risk manager (interest rate, policy, concentration risk); Portfolio optimizer (synthesize all views, output allocation).

2. **Multi-Agent Debate Mechanism**: Independent assessment → public opinion → cross-challenge → multi-round consensus → reduced hallucination and bias.

3. **China REITs-Specific Knowledge**: Embedded China REITs valuation (DCF, NAV, FFO multiples); China policy context (land system, tax incentives); Chinese financial report processing.

4. **Backtest Validation**: 2021-2025 China public REITs data; comparison with single LLM and human benchmarks; significant risk-adjusted return improvement.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **LLM幻觉在金融场景的风险**：代理间的辩论能否真正消除幻觉？
2. **数据可得性**：中国REITs历史数据短（2021年才启动），样本量是否足够？
3. **前视偏差**：代理是否使用了未来信息？
4. **计算成本**：多代理辩论的token消耗和延迟

**English:**

1. **LLM Hallucination in Finance**: Can inter-agent debate truly eliminate hallucinations?
2. **Data Availability**: China REITs history is short (launched 2021), sufficient sample size?
3. **Look-Ahead Bias**: Do agents use future information?
4. **Compute Cost**: Token consumption and latency of multi-agent debate

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**推荐阅读顺序：**
1. 第3节（代理架构）——理解四角色的分工和交互协议
2. 第4节（辩论机制）——多代理共识的收敛过程
3. 第5节（实验）——中国REITs回测结果

**量化金融映射方向：**
- 多代理系统的量化投资决策框架
- LLM代理角色设计的方法论
- 辩论机制在策略评审中的应用

**English:**

**Recommended Reading Order:**
1. Section 3 (Agent Architecture) — role division and interaction protocol
2. Section 4 (Debate Mechanism) — multi-agent consensus convergence
3. Section 5 (Experiments) — China REITs backtest results

**Quant Finance Mapping:**
- Multi-agent quant investment decision framework
- LLM agent role design methodology
- Debate mechanism application in strategy review
