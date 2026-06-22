# Week 11: 2026-03-12 — 2026-03-18 / 第11周

## 本周学术动态 / Weekly Highlights

**中文：** 2026年第十一周，序列建模架构革新与注意力机制改进并行推进。Mamba-3引入复值状态更新和MIMO公式，在状态规模减半的情况下追平Mamba-2性能；Attention Residuals提出残差连接增强注意力机制；LLM风险溢价建模探索大语言模型在资产定价中的新应用。

**English:** Week 11 featured parallel advances in sequence modeling architecture and attention mechanisms. Mamba-3 introduces complex-valued state updates and MIMO formulation, matching Mamba-2 performance with half the state size; Attention Residuals proposes residual-enhanced attention; LLM risk premium modeling explores new applications of LLMs in asset pricing.

## 重点论文 / Featured Papers

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [Mamba-3: Improved SSM Sequence Modeling](https://arxiv.org/abs/2603.15569) | Architecture | [review.md](mamba3/review.md) | [code/](mamba3/code/) |
| 2 | [Attention Residuals](https://arxiv.org/abs/2603.15031) | Attention Mechanism | [review.md](attention-residuals/review.md) | — |
| 3 | [LLM Risk Premia Modeling](https://arxiv.org/abs/2603.14288) | Asset Pricing | [review.md](llm-risk-premia/review.md) | [code/](llm-risk-premia/code/) |

## 其他论文 / Other Papers

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| AI Agents in Financial Markets (2603.13942) | AI Finance Survey | 四层AI代理金融架构 (Week 10已覆盖) |
| Autonomous Factor Investing (2603.14288) | Quant Factor | Agentic AI自主因子挖掘, Sharpe 3.11 |

## 领域趋势 / Trends

1. SSM架构进化 / SSM architecture evolution — Mamba-3复值状态+MIMO实现效率飞跃
2. 注意力残差 / Attention residuals — 残差连接增强注意力表达力
3. LLM资产定价 / LLM asset pricing — 大语言模型建模风险溢价
4. 架构效率化 / Architecture efficiency — 更小状态规模，同等性能
