# Week 13: 2026-03-26 — 2026-04-01 / 第13周

---

## 本周学术动态 / Weekly Highlights

**中文：** 2026 年第 13 周（3 月 26 日 — 4 月 1 日），arXiv 核心类别（cs.CV / cs.LG / cs.AI / cs.CL / stat.ML）共新增约 **2,520** 篇论文。本周焦点集中在“跨架构蒸馏”“图神经网络与 NLP 融合”“深度对冲与衍生品定价”以及“具身世界模型与 VLA”四条主线：
1. **Transformer → Mamba 的知识迁移**：*Attention to Mamba* 提出从 Transformer 到 Mamba 的原则化蒸馏方案，让 SSM 在保持线性推理复杂度的同时继承 Transformer 的表达能力；
2. **GNN 与语言模型的金融融合**：*Node Transformer + BERT Stock Prediction* 将图神经网络的关系建模与 BERT 情绪提取结合，用于股票预测；
3. **强化学习进入期权风险管理**：*Autonomous AI Agents for Option Hedging* 用短fall感知的 RL 代理自主学习对冲策略，弱化对 Black-Scholes 模型假设的依赖；
4. **随机化注意力机制**：*Stochastic Attention via Langevin Dynamics* 将 Langevin 动力学引入 Modern Hopfield 能量模型，为注意力提供不确定性量化；
5. **具身智能与世界模型交汇**：AutoWorld、DIAL 等工作将多智能体交通模拟、潜在世界建模与 VLA 结合，推动机器人与自动驾驶系统的可泛化决策。

**English:** Week 13 of 2026 (March 26 — April 1) saw approximately **2,520** new papers across the core arXiv categories (cs.CV / cs.LG / cs.AI / cs.CL / stat.ML). The week centered on four threads: cross-architecture distillation, GNN-NLP fusion for finance, deep hedging and derivatives pricing, and embodied world models/VLAs:
1. **Transformer-to-Mamba knowledge transfer**: *Attention to Mamba* proposes a principled distillation recipe from Transformer to Mamba, letting SSMs inherit Transformer expressiveness while preserving linear inference complexity;
2. **GNN and language-model fusion in finance**: *Node Transformer + BERT Stock Prediction* combines graph-neural relationship modeling with BERT sentiment extraction for stock forecasting;
3. **RL enters option risk management**: *Autonomous AI Agents for Option Hedging* uses shortfall-aware RL agents to learn hedging policies with weaker reliance on Black-Scholes assumptions;
4. **Stochastic attention mechanisms**: *Stochastic Attention via Langevin Dynamics* introduces Langevin dynamics into the Modern Hopfield energy model to provide uncertainty quantification for attention;
5. **Embodied AI meets world models**: Works such as AutoWorld and DIAL combine multi-agent traffic simulation, latent world modeling, and VLAs to push generalizable decision-making in robotics and autonomous driving.

---

## 重点论文深度分析 / Featured Papers with Deep Reviews

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [Attention to Mamba: A Recipe for Cross-Architecture Distillation](https://arxiv.org/abs/2604.14191) | Architecture | [review.md](attention-to-mamba/review.md) | [code/](attention-to-mamba/code/) |
| 2 | [Stock Market Prediction Using Node Transformer Architecture Integrated with BERT Sentiment Analysis](https://arxiv.org/abs/2603.05917) | Stock Prediction | [review.md](node-transformer-bert/review.md) | [code/](node-transformer-bert/code/) |
| 3 | [Autonomous AI Agents for Option Hedging: Enhancing Financial Stability through Shortfall Aware Reinforcement Learning](https://arxiv.org/abs/2603.06587) | Options RL | [review.md](autonomous-option-hedging/review.md) | [code/](autonomous-option-hedging/code/) |
| 4 | [Stochastic Attention via Langevin Dynamics on the Modern Hopfield Energy](https://arxiv.org/abs/2603.06875) | Attention Mechanism | [review.md](stochastic-attention/review.md) | [code/](stochastic-attention/code/) |

---

## 本周论文列表（按领域分类）/ Weekly Papers by Area

> 本周 arXiv 五个主要类别新增约 **2,520** 篇论文（submitted dates: 2026-03-26 to 2026-04-01）。以下按领域精选代表性工作，并附简短双语要点。
>
> This week added approximately **2,520** papers across five main arXiv categories (submitted dates: 2026-03-26 to 2026-04-01). Representative works by area are listed below with brief bilingual key points.

### 计算机视觉 / Computer Vision

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [Hierarchical Pre-Training of Vision Encoders with Large Language Models](https://arxiv.org/abs/2604.00086v1) | CV / Vision-Language | 提出分层预训练框架，利用大语言模型监督视觉编码器学习从局部到全局的语义层次，提升下游视觉任务性能。It proposes a hierarchical pre-training framework that uses large language models to supervise vision encoders in learning local-to-global semantic hierarchies, improving downstream vision tasks. |
| [$R_\text{dm}$: Re-conceptualizing Distribution Matching as a Reward for Diffusion Distillation](https://arxiv.org/abs/2603.28460v2) | CV / Diffusion | 将分布匹配重新概念化为扩散蒸馏的奖励函数，在保持生成质量的同时加速采样。It re-conceptualizes distribution matching as a reward for diffusion distillation, accelerating sampling while preserving generation quality. |

### 自然语言处理与大模型 / NLP & LLM

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning](https://arxiv.org/abs/2604.01170v1) | NLP / Reasoning | 通过测试时训练实现在线推理校准，使大语言模型在满足覆盖保证的同时泛化到分布外推理任务。It enables online reasoning calibration via test-time training, allowing LLMs to generalize to out-of-distribution reasoning tasks while satisfying coverage guarantees. |
| [Stochastic Attention: Connectome-Inspired Randomized Routing for Expressive Linear-Time Attention](https://arxiv.org/abs/2604.00754v2) | NLP / Attention | 受连接组学启发引入随机化路由，在线性时间复杂度下增强注意力的表达能力。It introduces connectome-inspired randomized routing to enhance the expressiveness of linear-time attention mechanisms. |
| [MF-QAT: Multi-Format Quantization-Aware Training for Elastic Inference](https://arxiv.org/abs/2604.00529v1) | NLP / Efficiency | 提出多格式量化感知训练，让单一模型在推理时根据硬件约束弹性选择精度格式。It proposes multi-format quantization-aware training, allowing a single model to elastically select precision formats at inference time based on hardware constraints. |

### 图神经网络与结构化学习 / Graph Neural Networks

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [On the Complexity of Optimal Graph Rewiring for Oversmoothing and Oversquashing in Graph Neural Networks](https://arxiv.org/abs/2603.26140v1) | GNN / Theory | 从计算复杂性角度分析图重连缓解过平滑与过挤压问题的难度，为可扩展图学习方法提供理论边界。It analyzes the computational complexity of optimal graph rewiring for mitigating oversmoothing and oversquashing, providing theoretical limits for scalable graph learning. |
| [Cross-attentive Cohesive Subgraph Embedding to Mitigate Oversquashing in GNNs](https://arxiv.org/abs/2603.27529v3) | GNN / Method | 利用交叉注意力学习内聚子图嵌入，缓解消息传递中的过挤压问题并提升图分类性能。It uses cross-attentive cohesive subgraph embedding to mitigate oversquashing in message passing and improve graph classification. |

### 智能体与强化学习 / Agents & RL

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [AutoWorld: Scaling Multi-Agent Traffic Simulation with Self-Supervised World Models](https://arxiv.org/abs/2603.28963v1) | Agents / World Models | 用自监督世界模型扩展多智能体交通仿真，生成高质量长尾场景以支持自动驾驶策略训练。It scales multi-agent traffic simulation with self-supervised world models, generating high-quality long-tail scenarios for autonomous-driving policy training. |
| [DIAL: Decoupling Intent and Action via Latent World Modeling for End-to-End VLA](https://arxiv.org/abs/2603.29844v2) | Agents / VLA | 通过潜在世界模型解耦意图与动作，提升端到端视觉-语言-动作模型在复杂指令下的泛化能力。It decouples intent and action via latent world modeling to improve end-to-end vision-language-action models under complex instructions. |

### 机器学习与 AI 基础 / ML & AI Foundations

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [KVSculpt: KV Cache Compression as Distillation](https://arxiv.org/abs/2603.27819v1) | ML / Inference | 将 KV 缓存压缩重新建模为知识蒸馏问题，在显著压缩缓存的同时保持长上下文生成质量。It reframes KV-cache compression as knowledge distillation, significantly compressing caches while maintaining long-context generation quality. |
| [Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes](https://arxiv.org/abs/2603.25562v2) | ML / Distillation | 重新审视同策略蒸馏的失败模式，并提出简单修复方案以提升策略蒸馏稳定性。It revisits on-policy distillation, identifies empirical failure modes, and proposes simple fixes to improve policy distillation stability. |

### 金融与量化 / Finance & Quant

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [Bridging Stochastic Control and Deep Hedging: Structural Priors for No-Transaction Band Networks](https://arxiv.org/abs/2603.29994v1) | Quant / Hedging | 将随机控制理论的结构先验融入深度对冲网络，学习无交易区间策略以降低交易成本。It incorporates structural priors from stochastic control theory into deep hedging networks to learn no-transaction-band policies and reduce transaction costs. |
| [Option Pricing on Automated Market Maker Tokens](https://arxiv.org/abs/2603.29763v1) | Quant / Derivatives | 为自动做市商代币建立期权定价框架，连接 DeFi 流动性机制与经典衍生品定价理论。It develops an option-pricing framework for automated market maker tokens, connecting DeFi liquidity mechanisms with classical derivatives pricing theory. |
| [Bridging Structured Knowledge and Data: A Unified Framework with Finance Applications](https://arxiv.org/abs/2604.00987v1) | Finance / ML | 统一结构化知识与数据驱动的学习框架，在金融场景中融合专家规则与统计模型。It unifies structured-knowledge and data-driven learning, integrating expert rules with statistical models for finance applications. |

---

## 领域趋势 / Trends

1. **跨架构蒸馏成为主流 / Cross-architecture distillation goes mainstream** — 从 Attention to Mamba 到 KVSculpt、On-Policy Distillation，研究者将蒸馏思想从模型压缩推广到异构架构迁移与缓存压缩。
2. **图神经网络进入金融核心场景 / GNNs enter core financial scenarios** — Node Transformer、GNN 过挤压理论、跨市场二部图预测等工作显示，图结构学习正从分子/推荐走向资产关系与风险网络建模。
3. **强化学习重塑衍生品风险管理 / RL reshapes derivative risk management** — 自主期权对冲、深度对冲结构先验、无交易区间网络等工作将 RL 与传统随机控制结合，推动衍生品风控从模型驱动走向数据驱动。
4. **随机化与不确定性量化注意力 / Stochastic and uncertainty-aware attention** — Stochastic Attention、Langevin 动力学注意力等研究探索注意力机制的概率化扩展，为高风险决策提供不确定性估计。
5. **具身世界模型加速落地 / Embodied world models accelerate deployment** — AutoWorld、DIAL 等将世界模型与多智能体仿真、VLA 结合，使机器人与自动驾驶系统从模仿走向可组合、可解释的决策。
6. **推理效率的弹性化 / Elastic inference efficiency** — MF-QAT、KV 缓存压缩、TurboAngle 等工作让模型在单一权重下支持多精度、多压缩比的弹性部署。

---

## PhD 阅读路线图 / PhD Reading Roadmap

### 阶段一：建立本周全景（30 分钟）
- 通读 Weekly Highlights 与 Trends，明确 4 条主线：跨架构蒸馏、GNN+NLP 金融、RL 衍生品风控、具身世界模型。

### 阶段二：精读重点论文（2–3 小时）
1. **Attention to Mamba**：聚焦 Transformer 到 Mamba 的初始化与蒸馏损失设计，思考如何在金融长序列模型中复用已训练 Transformer 的知识。
2. **Node Transformer + BERT**：阅读 [review.md](node-transformer-bert/review.md)，理解图关系、文本情绪与价格特征的三模态融合策略。
3. **Autonomous AI Agents for Option Hedging**：结合 [option_hedging.py](autonomous-option-hedging/code/option_hedging.py) 理解短fall感知奖励如何塑造对冲行为。
4. **Stochastic Attention via Langevin Dynamics**：分析 Langevin 采样如何替代确定性 softmax，并评估在金融风险估计中的潜在应用。

### 阶段三：按兴趣深入子领域（各 1–2 小时）
- **蒸馏与高效推理**：Attention to Mamba → KVSculpt → Revisiting On-Policy Distillation → MF-QAT（从架构迁移到缓存压缩与弹性量化）。
- **GNN 金融应用**：Node Transformer → Cross-attentive Cohesive Subgraph → Graph Rewiring Complexity（从应用到理论）。
- **衍生品与风控**：Autonomous Option Hedging → Bridging Stochastic Control and Deep Hedging → Option Pricing on AMM Tokens（从 RL 对冲到 DeFi 定价）。
- **具身智能**：DIAL → AutoWorld（从 VLA 意图解耦到多智能体交通仿真）。

### 阶段四：动手实验（2–4 小时）
- 运行 [attention_to_mamba.py](attention-to-mamba/code/attention_to_mamba.py) 与 [option_hedging.py](autonomous-option-hedging/code/option_hedging.py)，分别对比蒸馏前后 Mamba 的困惑度与 RL 对冲的 P&L 分布。
- 在公开股票关系图与新闻数据上复现 Node Transformer + BERT 的多模态融合思路，观察图结构对预测稳定性的贡献。

---

## 数据来源 / Sources

- arXiv API (`export.arxiv.org`)，查询类别 cs.CV / cs.LG / cs.AI / cs.CL / stat.ML / q-fin.*，提交日期范围 `2026-03-26` 至 `2026-04-01`。
- 本仓库 Week 13 原始 featured reviews（Attention to Mamba、Node Transformer+BERT、Autonomous Option Hedging、Stochastic Attention）由先前会话整理，本次补充完整分类列表、趋势、路线图与真实 arXiv 链接。
