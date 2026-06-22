# Generative AI for Stock Selection

> **arXiv:** [2602.00196](https://arxiv.org/abs/2602.00196) | **日期 / Date:** 2026-02-05 | **作者 / Authors:** Keywan Christian Rasekhschaffe

---

## 关键摘录 / Key Excerpts

> 1. "We examine whether automated feature extraction for US equities is viable via sophisticated text generators, combining retrieval-augmented generation with coded instructions to create financially-driven metrics."
>    / "我们研究通过复杂文本生成器自动化美股特征提取的可行性，结合检索增强生成和编码指令创建金融驱动指标。"

> 2. "The synthetic features match or beat standard baselines, boosting risk-adjusted returns significantly, with low overlap with conventional metrics — making them ideal for blending."
>    / "合成特征匹配或超越标准基线，显著提升风险调整收益，与传统指标重叠度低——使其成为混合的理想选择。"

---

## Q1: 核心问题 / Core Problem

**中文：**
量化选股依赖于高质量因子特征。传统因子工程：
1. 需要大量人工专家知识
2. 因子发现过程缓慢且昂贵
3. 因子衰减（alpha decay）快

本文探索LLM+RAG能否自动化因子发现过程：
- LLM根据检索到的财务报告、衍生品数据、交易量信息
- 通过代码指令生成新的alpha因子
- 评估这些"AI因子"与传统因子的互补性

**English:**
Quantitative stock selection relies on high-quality factor features. Traditional factor engineering:
1. Requires extensive human expert knowledge
2. Factor discovery process is slow and expensive
3. Fast alpha decay

This paper explores whether LLM+RAG can automate factor discovery:
- LLM generates new alpha factors from retrieved financial reports, derivatives data, volume information
- Via coded instructions
- Evaluates complementarity of these "AI factors" with traditional factors

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **LLM+RAG因子生成管线 (LLM+RAG Factor Generation Pipeline)**：
   - 检索：从SEC文件、期权数据、分析师报告中提取相关信息
   - 生成：LLM通过代码模板生成因子计算逻辑
   - 验证：自动回测评估因子有效性

2. **多源数据融合 (Multi-Source Data Fusion)**：
   - 衍生品隐含信息（期权隐含波动率曲面）
   - 交易量微观结构特征
   - 分析师文本报告的情感和逻辑

3. **因子正交性分析 (Factor Orthogonality Analysis)**：
   - AI因子与传统因子（动量、价值、质量）的低相关性
   - 混合后的信息比率提升
   - 因子衰减速度对比

**English:**

1. **LLM+RAG Factor Generation Pipeline**: Retrieve relevant info from SEC filings, options data, analyst reports; generate factor calculation logic via code templates; automatic backtesting validation.

2. **Multi-Source Data Fusion**: Derivatives implied info (options implied volatility surface); trading volume microstructure features; analyst text report sentiment and logic.

3. **Factor Orthogonality Analysis**: Low correlation between AI factors and traditional factors (momentum, value, quality); information ratio improvement after blending; factor decay speed comparison.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **因子的经济解释性**：LLM生成的因子是否有经济直觉？黑箱因子的可接受度？
2. **检索质量的影响**：RAG检索噪声如何影响因子质量？
3. **大规模部署的可行性**：为数千只股票生成因子的计算成本？
4. **数据窥探偏差**：因子选择过程中的多重检验校正？

**English:**

1. **Factor Economic Interpretability**: Do LLM-generated factors have economic intuition? Acceptability of black-box factors?
2. **Retrieval Quality Impact**: How does RAG retrieval noise affect factor quality?
3. **Large-Scale Deployment Feasibility**: Compute cost of generating factors for thousands of stocks?
4. **Data Snooping Bias**: Multiple testing correction in factor selection process?

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**推荐阅读顺序：**
1. 第3节（管线）——LLM+RAG因子生成的完整流程
2. 第4节（因子评估）——IC、ICIR、正交性分析
3. 第5节（组合结果）——多因子组合的实际表现

**量化金融映射方向：**
- 自动化alpha因子挖掘平台
- LLM辅助的因子库扩充
- AI因子与传统因子的融合策略

**English:**

**Recommended Reading Order:**
1. Section 3 (Pipeline) — complete LLM+RAG factor generation flow
2. Section 4 (Factor Evaluation) — IC, ICIR, orthogonality analysis
3. Section 5 (Portfolio Results) — multi-factor portfolio actual performance

**Quant Finance Mapping:**
- Automated alpha factor mining platform
- LLM-assisted factor library expansion
- AI factor and traditional factor fusion strategies
