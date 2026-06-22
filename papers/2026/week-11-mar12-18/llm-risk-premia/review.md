# LLM Risk Premia Modeling

> **arXiv:** 基于2603系列量化金融论文综合 | **日期 / Date:** 2026-03

---

## Q1: 核心问题

**中文：** 传统资产定价模型(Fama-French等)使用线性因子模型解释风险溢价，但：
1. 因子暴露是非线性的且时变的
2. 新的风险维度（ESG、地缘政治、供应链）难以纳入线性框架
3. 文本信息（财报电话会、新闻）包含线性模型无法捕获的风险溢价信息

本文探索LLM作为非线性风险溢价建模工具的能力。

**English:** Traditional asset pricing models (Fama-French) use linear factor models to explain risk premia, but:
1. Factor exposures are nonlinear and time-varying
2. New risk dimensions (ESG, geopolitics, supply chain) are hard to fit into linear frameworks
3. Textual information (earnings calls, news) contains risk premium information that linear models cannot capture

This paper explores LLMs as nonlinear risk premium modeling tools.

## Q2: 核心贡献

1. **LLM风险溢价因子**: 用LLM从非结构化文本中提取风险溢价信号
2. **非线性因子暴露**: LLM自然捕获因子间的非线性交互
3. **时变风险定价**: 通过prompt工程实现regime-dependent风险定价
4. **与传统因子融合**: LLM因子与Fama-French因子的正交性和增量贡献

## Q3: 审稿攻击点

1. LLM推理成本 vs 因子信号增量价值？ 2) 模型更新导致因子定义漂移？ 3) 可解释性？

## Q4: 量化金融映射

- 直接构建LLM风险溢价因子
- 与现有alpha因子组合的增量分析
- Regime-adaptive风险定价模型
