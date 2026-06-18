# 量化金融领域论文创作 / Quantitative Finance Paper Adaptations

> **基于已推送的顶会论文，将其核心方法论迁移至量化金融领域**
> Adapting core methodologies from featured conference papers to quantitative finance domains

---

## 论文映射总览 / Paper Mapping Overview

| # | 原始顶会论文 / Source Paper | 会议 / Venue | 核心方法 / Core Method | 量化金融对应论文 / Quant Finance Paper | 量化方向 / Quant Area |
|---|---|---|---|---|---|
| 1 | [Thinking with Video](../week-24-jun10-16/thinking-with-video/review.md) | CVPR 2026 | 视频生成作为推理范式 | [Thinking with Time-Series](thinking-with-time-series/paper.md) | 多周期市场推理 |
| 2 | [Markovian Scale Prediction](../week-24-jun10-16/markovian-scale-prediction/review.md) | CVPR 2026 | 滑动窗口多尺度预测 | [Markovian Multi-Resolution Forecasting](markovian-price-prediction/paper.md) | 多分辨率价格预测 |
| 3 | [SAM 3D](../week-24-jun10-16/sam-3d/review.md) | CVPR 2026 | 单图3D重建 | [SAM-Vol: Segment Anything for Volatility](sam-volatility/paper.md) | 波动率曲面重建 |
| 4 | [GRPO is Secretly a PRM](../week-25-jun18-24/grpo-is-prm/review.md) | ICML 2026 | 隐式过程奖励模型 | [GRPO for Trading Execution](grpo-trading-execution/paper.md) | 交易执行优化 |
| 5 | [ESamp: Latent Distilling](../week-25-jun18-24/llm-explore-by-latent-distilling/review.md) | ICML 2026 | 潜空间探索采样 | [ESamp for Portfolio Diversification](esamp-portfolio-diversity/paper.md) | 组合多样性探索 |
| 6 | [CEO-Bench](../week-25-jun18-24/ceo-bench/review.md) | arXiv 2026 | 长程Agent决策评测 | [QuantBench: Long-Horizon Market Regimes](quantbench-long-horizon/paper.md) | 长周期交易评测 |
| 7 | [TROLL Trust Regions](../week-04-jan22-28/troll-trust-regions/review.md) | ICLR 2026 | 离散可微信任区域 | [TROLL for Portfolio Risk](troll-portfolio-risk/paper.md) | 组合风险管理 |
| 8 | [Deep Ignorance](../week-04-jan22-28/deep-ignorance/review.md) | ICLR 2026 | 预训练数据防篡改 | [Financial Deep Ignorance](financial-deep-ignorance/paper.md) | 金融模型安全 |
| 9 | [Societies of Thought](../week-04-jan22-28/societies-of-thought/review.md) | Google 2026 | 内部多智能体推理 | [Investment Societies of Thought](investment-societies/paper.md) | 多专家投资决策 |

---

## 创作方法论 / Creation Methodology

每篇量化金融论文遵循以下迁移逻辑：

1. **提取源论文核心机制** — 剥离具体应用场景，抽象出数学/算法本质
2. **映射到量化金融问题** — 找到金融领域中结构类似的未解决问题
3. **领域特定适配** — 引入金融特有的约束（流动性、交易成本、监管、非平稳性）
4. **设计金融实验** — 使用金融数据集（S&P 500、加密货币、期权链）、金融指标（Sharpe、MaxDD、Calmar）
5. **中英双语撰写** — 桥接ML社区和量化金融社区的术语体系

---

## 目录结构 / Directory Structure

```
papers/2026/quant-finance/
├── README.md                          ← 本文件
├── thinking-with-time-series/         ← 对应 Thinking with Video
│   └── paper.md
├── markovian-price-prediction/        ← 对应 Markovian Scale Prediction
│   └── paper.md
├── sam-volatility/                    ← 对应 SAM 3D
│   └── paper.md
├── grpo-trading-execution/            ← 对应 GRPO is PRM
│   └── paper.md
├── esamp-portfolio-diversity/         ← 对应 ESamp
│   └── paper.md
├── quantbench-long-horizon/           ← 对应 CEO-Bench
│   └── paper.md
├── troll-portfolio-risk/              ← 对应 TROLL
│   └── paper.md
├── financial-deep-ignorance/          ← 对应 Deep Ignorance
│   └── paper.md
└── investment-societies/              ← 对应 Societies of Thought
    └── paper.md
```
