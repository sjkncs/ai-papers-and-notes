# Week 13: 2026-03-26 — 2026-04-01 / 第13周

## 本周学术动态 / Weekly Highlights

**中文：** 2026年第十三周，跨架构蒸馏与金融AI代理成为焦点。Attention to Mamba提出从Transformer到Mamba的原则化蒸馏方案；Node Transformer+BERT融合图神经网络和NLP情绪进行股票预测；自主AI代理用强化学习进行期权对冲；随机注意力通过Langevin动力学为金融计算提供物理启发的注意力机制。

**English:** Week 13 featured cross-architecture distillation and financial AI agents. Attention to Mamba proposes principled distillation from Transformer to Mamba; Node Transformer+BERT fuses GNN and NLP sentiment for stock prediction; Autonomous AI agents use RL for option hedging; Stochastic Attention via Langevin Dynamics provides physics-inspired attention for financial computation.

## 重点论文 / Featured Papers

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [Attention to Mamba: Cross-Architecture Distillation](https://arxiv.org/abs/2604.14191) | Architecture | [review.md](attention-to-mamba/review.md) | [code/](attention-to-mamba/code/) |
| 2 | [Node Transformer + BERT Stock Prediction](https://arxiv.org/abs/2603.xxxx) | Stock Prediction | [review.md](node-transformer-bert/review.md) | [code/](node-transformer-bert/code/) |
| 3 | [Autonomous AI Agents for Option Hedging](https://arxiv.org/abs/2603.xxxx) | Options RL | [review.md](autonomous-option-hedging/review.md) | [code/](autonomous-option-hedging/code/) |
| 4 | [Stochastic Attention via Langevin Dynamics](https://arxiv.org/abs/2603.xxxx) | Attention Mechanism | [review.md](stochastic-attention/review.md) | [code/](stochastic-attention/code/) |

## 其他论文 / Other Papers

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| Quantile-based VaR/ES Forecasting | Risk | 分位数建模增强尾部风险估计 |
| Optimizer as Implicit Prior | ML Finance | 优化器偏差作为金融时序的隐式先验 |
| GAR: Generative Adversarial Regression | Stress Testing | GAN生成压力测试风险场景 |
| Bipartite Graph Cross-Market Forecasting | Cross-Market | 二部图建模美中跨市场收益预测 |
| DatedGPT: Preventing Lookahead Bias | LLM Finance | 时间感知预训练防止前视偏差 |

## 领域趋势 / Trends

1. 跨架构蒸馏 / Cross-architecture distillation — Transformer知识迁移到SSM
2. GNN+NLP融合 / GNN+NLP fusion — 图结构+文本情绪的股票预测
3. 自主期权对冲 / Autonomous option hedging — RL代理管理衍生品风险
4. 物理启发注意力 / Physics-inspired attention — Langevin动力学增强注意力
