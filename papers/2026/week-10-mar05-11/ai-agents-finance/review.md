# AI Agents in Financial Markets: Architecture, Applications, and Systemic Implications

> **arXiv:** [2603.13942](https://arxiv.org/abs/2603.13942) | **日期 / Date:** 2026-03-13 | **作者 / Authors:** Hui Gong et al.

---

## 关键摘录 / Key Excerpts

> 1. "Autonomous systems are shifting financial automation from isolated prediction tasks to comprehensive agentic workflows."
>    / "自主系统正在将金融自动化从孤立的预测任务转向全面的代理工作流。"

> 2. "The four-layer architecture connects design traits to market-level outcomes including efficiency, liquidity, and resilience."
>    / "四层架构将设计特征与市场级结果（包括效率、流动性和韧性）联系起来。"

---

## Q1: 核心问题 / Core Problem

**中文：**
AI在金融中的应用正从单一预测任务向复杂自主决策转变，但缺乏系统性的架构框架。现有研究分散在：
- 价格预测（点状任务）
- 情绪分析（单一模态）
- 组合优化（独立模块）

缺少一个统一的架构将感知、推理、决策和执行整合为端到端的自主金融代理。本文提出四层架构来系统化这一领域。

**English:**
AI in finance is shifting from single prediction tasks to complex autonomous decision-making, but lacks a systematic architectural framework. Existing research is scattered across:
- Price prediction (point tasks)
- Sentiment analysis (single modality)
- Portfolio optimization (independent modules)

A unified architecture integrating perception, reasoning, decision-making, and execution into end-to-end autonomous financial agents is missing. This paper proposes a four-layer architecture.

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **四层架构 (Four-Layer Architecture)**：
   - 感知层 (Perception): 多模态数据摄取（价格、新闻、另类数据）
   - 推理层 (Reasoning): LLM驱动的分析和推理
   - 决策层 (Decision): 策略生成和组合优化
   - 执行层 (Execution): 订单执行和风险管理

2. **有界自主 (Bounded Autonomy) 概念**：
   - 定义AI代理的自主性边界
   - 人类监督与自主决策的平衡框架
   - 安全护栏和风险控制机制

3. **系统性影响分析 (Systemic Implications)**：
   - AI代理对市场效率的影响
   - 流动性影响分析
   - 系统性风险评估

**English:**

1. **Four-Layer Architecture**: Perception (multimodal data ingestion) → Reasoning (LLM-driven analysis) → Decision (strategy generation) → Execution (order execution and risk management)

2. **Bounded Autonomy Concept**: Defining autonomy boundaries for AI agents; balance framework between human oversight and autonomous decisions; safety guardrails and risk control mechanisms.

3. **Systemic Implications Analysis**: Impact of AI agents on market efficiency; liquidity impact analysis; systemic risk assessment.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **综述的原创性**：架构框架是否为真正的贡献还是仅仅是分类学？
2. **实证验证不足**：四层架构是否在真实系统中验证？
3. **系统性风险的量化**：AI代理对市场的系统性影响如何量化？
4. **监管合规**：框架是否考虑了不同司法管辖区的监管要求？

**English:**

1. **Survey Originality**: Is the architecture framework a genuine contribution or just taxonomy?
2. **Insufficient Empirical Validation**: Is the four-layer architecture validated in real systems?
3. **Quantifying Systemic Risk**: How to quantify systemic impact of AI agents on markets?
4. **Regulatory Compliance**: Does the framework consider regulatory requirements across jurisdictions?

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**推荐阅读顺序：**
1. 第2节（架构）——理解四层设计
2. 第3节（应用）——各层的具体实现案例
3. 第4节（系统性影响）——市场级影响分析
4. 第5节（未来方向）——有界自主和研究机会

**量化金融映射方向：**
- 构建自己的多代理量化系统架构
- 有界自主概念用于实盘策略的风险控制
- 系统性影响分析用于策略容量评估

**English:**

**Recommended Reading Order:**
1. Section 2 (Architecture) — four-layer design
2. Section 3 (Applications) — implementation examples per layer
3. Section 4 (Systemic Implications) — market-level impact analysis
4. Section 5 (Future Directions) — bounded autonomy and research opportunities

**Quant Finance Mapping:**
- Build your own multi-agent quant system architecture
- Bounded autonomy concept for live strategy risk control
- Systemic impact analysis for strategy capacity assessment
