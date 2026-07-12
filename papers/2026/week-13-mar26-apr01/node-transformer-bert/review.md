# Stock Market Prediction Using Node Transformer Architecture Integrated with BERT Sentiment Analysis

> **arXiv:** [2603.05917](https://arxiv.org/abs/2603.05917) | **日期 / Date:** 2026-03-06

---

## Q1: 核心问题

**中文：** 股票预测需要同时建模：1) 公司间的关联关系(供应链、竞争、行业) 2) 文本情绪信号。现有方法分别处理这两类信息，缺少统一框架。本文用Node Transformer捕获股票间图结构关系，用BERT提取新闻情绪，融合两者进行预测。

## Q2: 核心贡献

1. **Node Transformer**: 在股票关系图上运行的Transformer变体
2. **BERT情绪集成**: 金融新闻的深层语义情绪提取
3. **多模态融合**: 图结构特征 + 文本情绪 + 价格特征的三模态融合
4. **超越单模态基线**: 在多只股票上验证融合的优势

## Q3: 审稿攻击点

1. 图结构的动态性如何处理？ 2) BERT推理延迟对实时性的影响？ 3) 交易成本后收益？

## Q4: 量化映射

- 供应链图+情绪 → A股/港股关联分析
- Node Transformer → 行业轮动策略
- 多模态融合框架 → 综合因子构建
