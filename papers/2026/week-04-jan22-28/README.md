# Week 4: 2026-01-22 — 2026-01-28 / 第4周

## 本周学术动态 / Weekly Highlights

**中文：** 2026年第四周，LLM强化学习训练范式与金融AI应用成为双主线。POPE通过"特权信息引导探索"解决RL在困难问题上的稀疏奖励瓶颈；R³提出Replay/Reflection/Ranking三重奖励机制替代逐步标注；MarketGANs首次将因子模型嵌入GAN架构实现多资产联合时序生成；Regret-Driven Portfolios将LLM引导的智能聚类应用于组合优化，显著跑赢SPY基准。ICLR 2026通知日（1月25日）公布结果，社区进入论文发表密集期。

**English:** Week 4 of 2026 featured dual themes: LLM RL training paradigms and financial AI applications. POPE solved sparse reward bottlenecks in hard RL problems via "privileged information-guided exploration"; R³ proposed triple reward mechanisms (Replay/Reflection/Ranking) replacing step-by-step labeling; MarketGANs embedded factor models into GAN architecture for joint multi-asset time series generation; Regret-Driven Portfolios applied LLM-guided smart clustering to portfolio optimization, significantly outperforming SPY benchmark. ICLR 2026 notification day (Jan 25) triggered a surge of published papers.

## 重点论文 / Featured Papers

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [POPE: Privileged On-Policy Exploration](https://arxiv.org/abs/2601.18779) | LLM RL Training | [review.md](pope/review.md) | [code/](pope/code/) |
| 2 | [R³: Replay, Reflection, Ranking Rewards](https://arxiv.org/abs/2601.19620) | LLM RL Rewards | [review.md](rl-rewards/review.md) | — |
| 3 | [MarketGANs: Multivariate Financial Time-Series](https://arxiv.org/abs/2601.17773) | Financial Data Gen | [review.md](marketgans/review.md) | [code/](marketgans/code/) |
| 4 | [Regret-Driven Portfolios: LLM-Guided Clustering](https://arxiv.org/abs/2601.17021) | Portfolio Optimization | [review.md](regret-portfolios/review.md) | [code/](regret-portfolios/code/) |

## 其他论文 / Other Papers

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| Latent-Space Contrastive RL (2601.17275) | LLM Reasoning | 潜空间对比学习实现稳定高效推理 |
| Reuse your FLOPs (2601.18795) | RL Efficiency | Off-policy prefix conditioning扩展RL到困难问题 |
| Teaching Models to Teach Themselves (2601.18778) | Self-Improvement | 模型自教学推理在可学习性边缘 |
| Scaling Embeddings > Scaling Experts (2601.21204) | MoE Efficiency | 扩大embedding维度优于增加专家数量 |

## 领域趋势 / Trends

1. RL训练范式进化 / RL training evolution — 从PPO到特权信息引导，解决稀疏奖励
2. 金融数据增强 / Financial data augmentation — GAN+因子模型生成多资产联合时序
3. LLM量化融合 / LLM-Quant integration — 大语言模型直接引导组合构建
4. 奖励工程创新 / Reward engineering innovation — 结构化熵替代逐步标注
5. 非平稳适应 / Non-stationary adaptation — 推理时适应应对regime切换

## 重要时间节点 / Key Dates

| 事件 / Event | 日期 / Date |
|---|---|
| ICLR 2026 Notification Day | Jan 25, 2026 |
| AAAI 2026 Conference | Feb–Mar 2026 |
| NeurIPS 2026 Submissions Open | Apr 2026 |
