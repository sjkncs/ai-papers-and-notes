# Week 6: 2026-02-05 — 2026-02-11 / 第6周

## 本周学术动态 / Weekly Highlights

**中文：** 2026年第六周，扩散模型向金融微观结构延伸，测试时适应(TTA)在金融时序中展现独特价值。DiffLOB首次将扩散模型应用于限价订单簿的反事实生成；TTA方法通过推理时自适应解决金融数据的非平稳性；Generative AI for Stock Selection验证了LLM+RAG生成因子特征的有效性；Conformal Risk Control为非平稳组合VaR提供有限样本保证。ViT-5和Step 3.5 Flash分别在视觉和MoE效率方面取得重要进展。

**English:** Week 6 of 2026 saw diffusion models extending to financial microstructure, with Test-Time Adaptation (TTA) showing unique value in financial time series. DiffLOB pioneered diffusion models for counterfactual limit order book generation; TTA addressed non-stationarity in financial data through inference-time adaptation; Generative AI for Stock Selection validated LLM+RAG generated factor features; Conformal Risk Control provided finite-sample guarantees for non-stationary portfolio VaR. ViT-5 and Step 3.5 Flash made significant advances in vision and MoE efficiency respectively.

## 重点论文 / Featured Papers

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [DiffLOB: Diffusion Models for Limit Order Books](https://arxiv.org/abs/2602.03776) | Market Microstructure | [review.md](difflob/review.md) | [code/](difflob/code/) |
| 2 | [Test-Time Adaptation for Non-stationary Time Series](https://arxiv.org/abs/2602.00073) | Time Series Adaptation | [review.md](tta-timeseries/review.md) | [code/](tta-timeseries/code/) |
| 3 | [Generative AI for Stock Selection](https://arxiv.org/abs/2602.00196) | AI Stock Selection | [review.md](gen-ai-stock/review.md) | [code/](gen-ai-stock/code/) |
| 4 | [Conformal Risk Control for Portfolio VaR](https://arxiv.org/abs/2602.03903) | Risk Management | [review.md](conformal-var/review.md) | [code/](conformal-var/code/) |

## 其他论文 / Other Papers

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| ViT-5: Vision Transformers for the Mid-2020s (2602.08071) | Computer Vision | 新一代视觉Transformer架构设计 |
| Step 3.5 Flash (2602.10604) | MoE Efficiency | 11B活跃参数实现前沿级智能 |
| ERNIE 5.0 Technical Report (2602.04705) | LLM Architecture | 百度新一代大模型技术报告 |
| Bitcoin Price Prediction with CFA (2602.00037) | Crypto Prediction | 组合融合分析用于比特币价格预测 |
| GT-Score for Trading Strategy Overfitting (2602.00080) | Trading ML | 减少数据驱动交易策略过拟合的鲁棒目标函数 |
| PredictionMarketBench (2602.00133) | AI Trading Agents | SWE-bench风格的交易代理回测框架 |

## 领域趋势 / Trends

1. 扩散模型进入金融微观结构 / Diffusion models enter financial microstructure — 订单簿反事实生成
2. 推理时适应范式 / Inference-time adaptation paradigm — 应对非平稳性的轻量级方案
3. LLM因子工程 / LLM factor engineering — 从文本到可交易因子的端到端管线
4. 保形预测在金融中的应用 / Conformal prediction in finance — 有限样本风险保证
5. MoE效率前沿 / MoE efficiency frontier — 更少活跃参数，同等智能

## 重要时间节点 / Key Dates

| 事件 / Event | 日期 / Date |
|---|---|
| AAAI 2026 Conference | Feb–Mar 2026 |
| NeurIPS 2026 Submissions Open | Apr 2026 |
| ICML 2026 Notification | May 2026 |
