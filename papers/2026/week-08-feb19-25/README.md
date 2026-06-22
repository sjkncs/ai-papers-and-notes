# Week 8: 2026-02-19 — 2026-02-25 / 第8周

## 本周学术动态 / Weekly Highlights

**中文：** 2026年第八周，Transformer内部机制解析与Regime-aware量化成为焦点。LeCun团队揭示"Massive Activations"和"Attention Sinks"的重叠只是pre-norm架构的副作用而非功能必需；Explainable Regime-Aware Investing用Wasserstein HMM实现可解释的regime自适应资产配置；Adaptive Window Selection提出动态窗口选择优化风险预测；RL Risk-Sensitive提出基准化投资的探索随机化方法。

**English:** Week 8 of 2026 featured Transformer internal mechanism analysis and regime-aware quant as focal points. LeCun's team revealed that "Massive Activations" and "Attention Sinks" overlap is merely a pre-norm architectural artifact; Explainable Regime-Aware Investing used Wasserstein HMM for interpretable regime-adaptive allocation; Adaptive Window Selection proposed dynamic window optimization for risk forecasting; RL Risk-Sensitive introduced exploratory randomization for benchmarked investment.

## 重点论文 / Featured Papers

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [Spike, Sparse, Sink: Attention Anatomy](https://arxiv.org/abs/2603.05498) | Transformer Analysis | [review.md](attention-sinks/review.md) | [code/](attention-sinks/code/) |
| 2 | [Explainable Regime Aware Investing](https://arxiv.org/abs/2603.04441) | Portfolio Strategy | [review.md](regime-aware-investing/review.md) | [code/](regime-aware-investing/code/) |
| 3 | [Adaptive Window Risk Forecasting](https://arxiv.org/abs/2603.01157) | Risk Management | [review.md](adaptive-window/review.md) | [code/](adaptive-window/code/) |
| 4 | [RL Risk-Sensitive Investment](https://arxiv.org/abs/2603.00738) | RL Portfolio | [review.md](rl-risk-sensitive/review.md) | [code/](rl-risk-sensitive/code/) |

## 领域趋势 / Trends

1. Transformer机制解剖 / Transformer mechanism dissection — 分离架构artifact与功能组件
2. Regime-aware配置 / Regime-aware allocation — Wasserstein距离驱动的自适应组合
3. 动态窗口风险 / Dynamic window risk — 自适应时间尺度选择
4. RL探索策略 / RL exploration strategies — 风险敏感基准化投资的随机化探索
