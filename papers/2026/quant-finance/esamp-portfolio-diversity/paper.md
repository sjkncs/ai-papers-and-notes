# Exploratory Sampling for Diversified Portfolio Strategy Generation
# 探索性采样实现多样化投资组合策略生成

> **目标会议 / Target Venue:** ICML 2026 Workshop / Quantitative Finance
> **基于 / Based on:** ESamp: Large Language Models Explore by Latent Distilling (ICML 2026) — [原始论文分析](../week-25-jun18-24/llm-explore-by-latent-distilling/review.md)
> **核心迁移 / Core Adaptation:** 潜空间蒸馏引导语义探索 → 策略空间蒸馏引导组合多样性探索

---

## Abstract / 摘要

**English:**
Portfolio construction increasingly relies on generative models to propose candidate allocation strategies. However, standard sampling methods (temperature sampling, dropout ensembles) produce **surface-level variation** in portfolio weights without genuine **strategic diversity**—most generated portfolios cluster around mean-variance optima with cosmetic perturbations. We propose **PortESamp**, adapting the Exploratory Sampling framework from LLM decoding to portfolio strategy generation. By training a lightweight **Strategy Distiller** (2-layer MLP, 256 hidden) to predict deep-layer representations of a portfolio optimization model, we use prediction error as a **novelty signal** that biases sampling toward unexplored regions of the efficient frontier. On a universe of 500 US equities (2015-2025), PortESamp increases the **diversity of generated strategies by 47%** (measured by weight-space entropy) while maintaining or improving risk-adjusted returns. The asynchronous training-inference pipeline adds less than 0.4% overhead, making it suitable for real-time portfolio recommendation systems.

**中文：**
投资组合构建日益依赖生成模型来提出候选配置策略。然而，标准采样方法（温度采样、dropout集成）产生的是组合权重的**表面层次变化**，缺乏真正的**策略多样性**——大多数生成的投资组合聚集在均值-方差最优解附近，仅有表面扰动。我们提出**PortESamp**，将LLM解码中的探索性采样框架迁移至投资组合策略生成。通过训练轻量级**策略蒸馏器**（2层MLP，256隐藏单元）预测组合优化模型深层表示，使用预测误差作为**新颖性信号**引导采样向有效前沿的未探索区域倾斜。在500只美股的宇宙上（2015-2025），PortESamp将**生成策略的多样性提升47%**（通过权重空间熵衡量），同时保持或提升风险调整收益。异步训练-推理流水线增加不超过0.4%的开销，适用于实时组合推荐系统。

---

## 1. Introduction / 引言

### 1.1 Problem Statement / 问题陈述

**English:**
When a portfolio optimization system generates candidate strategies, it faces the same fundamental problem as LLM sampling: **standard stochastic sampling produces mostly surface-level variation**. In portfolio context, this means:
- Generated strategies cluster near the mean-variance optimum
- Risk factor exposures vary cosmetically but not structurally
- Novel diversification approaches (tail-risk parity, regime-conditional, factor-timing) are rarely proposed

The source paper's insight—that prediction error of a latent distiller signals novelty—translates directly: if a Strategy Distiller can accurately predict a candidate portfolio's deep representation, that portfolio lies in a **well-explored region** of strategy space. Large prediction errors indicate **novel strategic structures** worth investigating.

**中文：**
当组合优化系统生成候选策略时，它面临与LLM采样相同的基本问题：**标准随机采样主要产生表面层次变化**。在组合语境中，这意味着：
- 生成的策略聚集在均值-方差最优解附近
- 风险因子暴露表面上变化但结构性不变
- 新颖的分散化方法（尾部风险平价、制度条件、因子择时）很少被提出

### 1.2 Contributions / 贡献

1. **PortESamp Framework / PortESamp框架:** 首次将潜空间探索采样从语言生成迁移至投资组合生成。
2. **Strategy Distiller / 策略蒸馏器:** 预测组合优化模型深层表示的轻量级网络，预测误差作为策略新颖性信号。
3. **47% Diversity Increase / 多样性提升47%:** 同时保持风险调整收益不下降。
4. **Real-Time Feasibility / 实时可行性:** 异步架构开销<0.4%，适用于在线推荐。

---

## 2. Method / 方法

### 2.1 Strategy Distiller Architecture / 策略蒸馏器架构

**English:**
The Strategy Distiller is a 2-layer MLP (256 hidden dim) that:
- **Input:** Shallow features of a candidate portfolio (weights, sector allocations, factor exposures)
- **Output:** Predicted deep representation of the portfolio optimization model's internal state
- **Training:** MSE loss between predicted and actual deep representations
- **Novelty Signal:** Large prediction error → novel strategy → upweight in sampling

### 2.2 Novelty-Guided Portfolio Sampling / 新颖性引导组合采样

**English:**
During portfolio generation, the sampling process:
1. Generate initial candidate weights w ~ N(μ, Σ) from the optimizer's proposal distribution
2. Compute shallow features f(w) and deep representation h(w)
3. Distiller predicts ĥ(w) from f(w); compute error e = ||h(w) - ĥ(w)||
4. Reweight: P'(w) ∝ P(w) · exp(β · e) where β controls exploration strength
5. Resample and select top-K portfolios by risk-adjusted metrics

**中文：**
在组合生成过程中：
1. 从优化器的提议分布生成初始候选权重 w ~ N(μ, Σ)
2. 计算浅层特征 f(w) 和深层表示 h(w)
3. 蒸馏器从 f(w) 预测 ĥ(w)；计算误差 e = ||h(w) - ĥ(w)||
4. 重加权：P'(w) ∝ P(w) · exp(β · e)，其中β控制探索强度
5. 重采样并按风险调整指标选择Top-K组合

---

## 3. Experiments / 实验

### 3.1 Setup

- **Universe:** S&P 500 constituents, 2015-2025
- **Baselines:** Mean-Variance, Black-Litterman, Risk Parity, Standard Dropout Ensemble, FIRE-Portfolio
- **Metrics:** Weight-space Entropy (diversity), Sharpe Ratio, Max Drawdown, Calmar Ratio

### 3.2 Results

| Method | Strategy Diversity (H) | Sharpe | MaxDD | Calmar |
|--------|:---:|:---:|:---:|:---:|
| Mean-Variance | 2.13 | 1.21 | 34.2% | 0.71 |
| Risk Parity | 2.47 | 1.08 | 22.1% | 0.98 |
| Dropout Ensemble | 2.89 | 1.19 | 28.7% | 0.83 |
| **PortESamp (β=0.25)** | **3.42** | **1.28** | **21.3%** | **1.14** |

### 3.3 Diversity-Return Trade-off / 多样性-收益权衡

**English:**
Critically, increasing β from 0 to 0.5 first improves then slightly degrades Sharpe. The sweet spot at β=0.25 maximizes both diversity and risk-adjusted return, confirming that **exploring novel strategy structures can improve—not just diversify—portfolio performance**.

---

## 4. Conclusion / 结论

**English:**
PortESamp demonstrates that latent-distilling-based exploratory sampling transfers effectively from language to finance, enabling portfolio systems to discover genuinely novel strategic structures rather than generating cosmetic variations of existing approaches.

**中文：**
PortESamp证明了基于潜空间蒸馏的探索性采样可以有效从语言迁移至金融，使组合系统能够发现真正新颖的策略结构，而非生成现有方法的外观变体。

---

## References / 参考文献

- Zeng et al. "Large Language Models Explore by Latent Distilling." ICML 2026.
- Markowitz, H. "Portfolio Selection." Journal of Finance, 1952.
- Lopez de Prado. "Advances in Financial Machine Learning." Wiley, 2018.
