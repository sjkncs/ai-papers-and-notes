# Impact of LLMs News Sentiment Analysis on Stock Price Movement Prediction

> **arXiv:** [2602.00086](https://arxiv.org/abs/2602.00086) | **日期 / Date:** 2026-01-30 | **会议 / Venue:** ICLR 2026 Workshop on Advances in Financial AI

---

## 关键摘录 / Key Excerpts

> 1. "We systematically evaluate how different LLM architectures interpret financial news sentiment and quantify their marginal contribution to stock price movement prediction."
>    / "我们系统评估了不同LLM架构如何解读金融新闻情绪，并量化其对股价预测的边际贡献。"

> 2. "LLM sentiment signals show the strongest predictive power during regime transitions and earnings announcement periods, while contributing minimally during low-information regimes."
>    / "LLM情绪信号在regime切换和财报发布期间展现最强预测力，而在低信息regime期间贡献最小。"

---

## Q1: 核心问题 / Core Problem

**中文：**
LLM情绪分析已被广泛用于金融文本处理，但关键问题尚未充分回答：
1. LLM情绪信号的**边际贡献**到底有多大？（相对于传统情绪词典和简单统计方法）
2. 不同LLM架构（GPT, LLaMA, BERT类）的情绪理解能力有何差异？
3. 情绪信号在什么市场状态下最有效？
4. 如何将情绪信号与价格/量数据融合？

本文系统回答这些问题，为LLM情绪在量化投资中的实际应用提供循证指导。

**English:**
LLM sentiment analysis is widely used in financial text processing, but key questions remain:
1. What is the **marginal contribution** of LLM sentiment signals? (vs traditional sentiment lexicons and simple statistics)
2. How do different LLM architectures (GPT, LLaMA, BERT) differ in sentiment understanding?
3. When is sentiment signal most effective?
4. How to fuse sentiment with price/volume data?

This paper systematically answers these questions, providing evidence-based guidance for LLM sentiment in quant investing.

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **多模型情绪对比 (Multi-Model Sentiment Comparison)**：
   - GPT-4o, LLaMA-3, FinBERT, VADER的对比
   - 情绪一致性分析（模型间agree率）
   - 不同新闻类型的情绪准确度

2. **边际贡献量化 (Marginal Contribution Quantification)**：
   - 基线：仅价格/量特征
   - 增强：价格/量 + LLM情绪
   - IC提升、方向准确率改善、组合收益增量

3. **Regime-Conditioned分析 (Regime-Conditioned Analysis)**：
   - 高波动期 vs 低波动期
   - 财报季 vs 非财报季
   - 宏观事件日 vs 普通日

4. **信号融合策略 (Signal Fusion Strategies)**：
   - 早期融合：情绪作为特征输入
   - 晚期融合：独立信号加权
   - 动态融合：根据regime调整权重

**English:**

1. **Multi-Model Sentiment Comparison**: GPT-4o, LLaMA-3, FinBERT, VADER comparison; sentiment consistency analysis; accuracy across news types.

2. **Marginal Contribution Quantification**: Baseline (price/volume only) vs enhanced (+LLM sentiment); IC improvement, directional accuracy gain, portfolio return increment.

3. **Regime-Conditioned Analysis**: High vs low volatility; earnings season vs non-earnings; macro event days vs normal days.

4. **Signal Fusion Strategies**: Early fusion (sentiment as feature); Late fusion (independent signal weighting); Dynamic fusion (regime-adjusted weights).

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **LLM推理成本**：GPT-4o API调用成本 vs 情绪信号的增量收益
2. **时效性问题**：新闻发布到LLM处理到信号产生的延迟
3. **情绪信号的衰减**：情绪信号的有效半衰期
4. **模型依赖风险**：LLM版本更新导致的情绪评分变化

**English:**

1. **LLM Inference Cost**: GPT-4o API cost vs sentiment signal incremental returns
2. **Timeliness**: Latency from news publication → LLM processing → signal generation
3. **Signal Decay**: Effective half-life of sentiment signals
4. **Model Dependency Risk**: Sentiment score changes from LLM version updates

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**推荐阅读顺序：**
1. 第4节（边际贡献实验）——核心结果表
2. 第5节（Regime分析）——何时使用情绪信号
3. 第6节（融合策略）——工程实现指南

**量化金融映射方向：**
- LLM情绪因子的实盘部署方案
- 情绪信号的regime-adaptive权重
- 成本-收益最优的LLM调用策略

**English:**

**Recommended Reading Order:**
1. Section 4 (Marginal Contribution Experiments) — core results table
2. Section 5 (Regime Analysis) — when to use sentiment signals
3. Section 6 (Fusion Strategies) — engineering implementation guide

**Quant Finance Mapping:**
- LLM sentiment factor live deployment scheme
- Regime-adaptive sentiment signal weights
- Cost-revenue optimal LLM calling strategy
