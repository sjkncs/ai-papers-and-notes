# Explainable Regime Aware Investing

> **arXiv:** [2603.04441](https://arxiv.org/abs/2603.04441) | **日期 / Date:** 2026-02 | **作者 / Authors:** Amine Boukardagha

---

## Q1: 核心问题 / Core Problem

**中文：** 市场regime切换使静态配置策略失效，但现有regime模型存在：1) 不可解释（黑箱HMM状态含义不明）2) 前视偏差（非严格因果）3) 高换手率 4) regime标签在时间上不一致。本文用Wasserstein HMM+严格因果推理解决这些问题。

**English:** Market regime switches invalidate static allocation, but existing regime models suffer from: 1) Uninterpretability 2) Look-ahead bias 3) High turnover 4) Inconsistent regime labels. This paper uses Wasserstein HMM + strictly causal inference.

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**
1. **严格因果Wasserstein HMM**: 滚动高斯HMM推理，无未来信息泄露
2. **2-Wasserstein模板跟踪**: 保持regime标签在时间上的语义一致性
3. **交易成本感知均值-方差优化**: regime概率直接输入优化器，控制换手率
4. **2025年熊市验证**: 在市场下跌时自动转入防御仓位

**English:**
1. Strictly causal Wasserstein HMM: Rolling Gaussian HMM inference, no future leakage
2. 2-Wasserstein template tracking: Maintains regime label semantic consistency over time
3. Transaction-cost-aware mean-variance optimization: Regime probabilities feed directly into optimizer
4. 2025 bear market validation: Automatically shifts to defensive positions during market decline

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**
1. HMM状态数的选择对结果敏感性？ 2) Wasserstein距离计算成本？ 3) 仅用日频数据的局限性？ 4) 与GARCH-regime模型的公平对比？

---

## Q4: 量化金融映射

- Wasserstein HMM → A股/港股regime检测
- 严格因果 → 实盘可用的regime信号
- 模板跟踪 → regime标签一致性（对回测报告很重要）
- 交易成本感知 → 适合高频再平衡场景
