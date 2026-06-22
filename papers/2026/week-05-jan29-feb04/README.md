# Week 5: 2026-01-29 — 2026-02-04 / 第5周

## 本周学术动态 / Weekly Highlights

**中文：** 2026年第五周，MoE效率优化与LLM金融应用成为焦点。《Scaling Embeddings Outperforms Scaling Experts》证明扩大embedding维度比增加专家数量更高效，催生了仅3B激活参数的68.5B模型LongCat-Flash-Lite；LatentLens揭示LLM中视觉token的可解释性，发现大部分视觉token在所有层都是可解读的；GT-Score提出减少交易策略过拟合的鲁棒目标函数；LLM多代理系统首次应用于中国公募REITs投资；LLM情绪分析对股价预测的影响在ICLR 2026 Workshop发表。

**English:** Week 5 of 2026 featured MoE efficiency optimization and LLM financial applications as focal points. "Scaling Embeddings Outperforms Scaling Experts" proved expanding embedding dimensions is more efficient than adding experts, yielding LongCat-Flash-Lite (68.5B params, ~3B active); LatentLens revealed visual token interpretability in LLMs, finding most visual tokens are interpretable across all layers; GT-Score proposed a robust objective function reducing overfitting in trading strategies; LLM multi-agent system was first applied to Chinese public REITs investment; LLM sentiment analysis impact on stock prediction was presented at ICLR 2026 Workshop.

## 重点论文 / Featured Papers

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [Scaling Embeddings > Scaling Experts](https://arxiv.org/abs/2601.21204) | MoE Efficiency | [review.md](scaling-embeddings/review.md) | — |
| 2 | [LatentLens: Interpretable Visual Tokens](https://arxiv.org/abs/2602.00462) | Multimodal AI | [review.md](latent-lens/review.md) | — |
| 3 | [GT-Score: Robust Trading Objective](https://arxiv.org/abs/2602.00080) | Trading ML | [review.md](gt-score/review.md) | [code/](gt-score/code/) |
| 4 | [LLM Multi-Agent for Chinese REITs](https://arxiv.org/abs/2602.00082) | AI Investment | [review.md](llm-reits/review.md) | [code/](llm-reits/code/) |
| 5 | [LLM Sentiment for Stock Prediction](https://arxiv.org/abs/2602.00086) | Sentiment Analysis | [review.md](llm-sentiment-stock/review.md) | [code/](llm-sentiment-stock/code/) |

## 其他论文 / Other Papers

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| Bitcoin Price with CFA (2602.00037) | Crypto Prediction | 组合融合分析用于比特币预测 |
| Rough Martingale Optimal Transport (2602.00097) | Risk Management | 非可建模风险因子的监管资本计算 |
| PredictionMarketBench (2602.00133) | AI Trading Agents | SWE-bench风格的交易代理回测框架 |
| ERNIE 5.0 Technical Report (2602.04705) | LLM Architecture | 百度统一多模态基础模型 |

## 领域趋势 / Trends

1. MoE效率突破 / MoE efficiency breakthrough — embedding扩展替代专家扩展
2. 视觉token可解释性 / Visual token interpretability — LLM内部表征的透明化
3. 交易策略防过拟合 / Trading strategy anti-overfitting — 鲁棒目标函数设计
4. LLM多代理投资 / LLM multi-agent investment — 中国REITs的AI协作决策
5. 情绪驱动预测 / Sentiment-driven prediction — LLM情绪信号的边际贡献

## 重要时间节点 / Key Dates

| 事件 / Event | 日期 / Date |
|---|---|
| ICLR 2026 Workshop Submissions | Feb 2026 |
| AAAI 2026 Conference | Feb–Mar 2026 |
| NeurIPS 2026 Submissions Open | Apr 2026 |
