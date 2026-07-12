# Week 11: 2026-03-12 — 2026-03-18 / 第11周

---

## 本周学术动态 / Weekly Highlights

**中文：** 2026 年第 11 周（3 月 12–18 日），arXiv 核心类别围绕“序列架构效率跃迁”“注意力内部机制再设计”“Agent 安全与宪法化”以及“LLM 驱动的量化金融”四条主线展开：
1. **状态空间模型再进化**：Mamba-3 引入复值状态更新与 MIMO 公式，以一半状态规模追平 Mamba-2，展示 SSM 架构仍有显著效率红利；
2. **注意力机制的“内部残差”化**：Attention Residuals 与 Residual Stream Duality 等工作将残差思想注入注意力计算或隐藏流，试图在零额外参数或少额外开销下提升表达能力；
3. **研究型 Agent 的验证驱动**：MiroThinker-1.7 & H1 等论文将“验证”作为重载科研 Agent 的核心组件，推动 Agent 从生成走向可验证推理；
4. **LLM 进入资产定价与组合优化**：从匿名化投资组合优化到高阶矩原生组合搜索，再到用户整合的风险溢价建模，金融 AI 正从单点预测走向端到端决策；
5. **高效采样与推理并行推进**：FlashSampling、Unified Spatio-Temporal Token Scoring 等工作在采样效率和视频 VLM 推理压缩上给出具体方案。

**English:** Week 11 of 2026 (March 12–18) centered on four threads: efficient sequence-architecture evolution, internal attention redesign, agent safety/constitutionalization, and LLM-driven quantitative finance:
1. **State-space model evolution**: Mamba-3 introduces complex-valued state updates and a MIMO formulation, matching Mamba-2 with half the state size and showing that SSM architectures still have significant efficiency headroom;
2. **Internal residual attention**: Attention Residuals and Residual Stream Duality inject residual ideas into attention computation or hidden streams, aiming to improve expressiveness with zero or minimal extra parameters;
3. **Verification-driven research agents**: Works such as MiroThinker-1.7 & H1 make verification a core component of heavy-duty research agents, pushing agents from generation toward verifiable reasoning;
4. **LLMs in asset pricing and portfolio optimization**: From anonymized portfolio optimization to native higher-moment portfolio search and user-integrated risk-premium modeling, financial AI is moving from isolated prediction to end-to-end decision-making;
5. **Efficient sampling and inference**: FlashSampling and Unified Spatio-Temporal Token Scoring offer concrete advances in sampling efficiency and video-VLM inference compression.

---

## 重点论文深度分析 / Featured Papers with Deep Reviews

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569v1) | Architecture / SSM | [review.md](mamba3/review.md) | [code/](mamba3/code/) |
| 2 | [Attention Residuals](https://arxiv.org/abs/2603.15031v1) | Attention Mechanism | [review.md](attention-residuals/review.md) | — |
| 3 | LLM Risk Premia Modeling | Asset Pricing | [review.md](llm-risk-premia/review.md) | [code/](llm-risk-premia/code/) |

---

## 本周论文列表（按领域分类）/ Weekly Papers by Area

> 以下精选自 2026-03-12 至 2026-03-18 提交于 arXiv 的论文，按领域归类并附简短双语要点。
> Representative works submitted to arXiv between 2026-03-12 and 2026-03-18, grouped by area with brief bilingual key points.

### 计算机视觉 / Computer Vision

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [Unified Spatio-Temporal Token Scoring for Efficient Video VLMs](https://arxiv.org/abs/2603.18004v1) | CV / Video Understanding | 提出统一的时空 token 评分框架，通过显式建模视频帧内与帧间的信息冗余，在保持性能的同时压缩视频 VLM 的推理开销。It proposes a unified spatio-temporal token-scoring framework that explicitly models intra- and inter-frame redundancy to reduce video-VLM inference cost while preserving performance. |
| [When Generative Augmentation Hurts: A Benchmark Study of GAN and Diffusion Models for Bias Correction in AI Classification Systems](https://arxiv.org/abs/2603.16134v1) | CV / Generative Augmentation | 系统评测生成式数据增强对分类偏差修正的真实效果，发现 GAN 与扩散模型增强在某些场景下会放大而非缓解子群体偏差。It benchmarks generative augmentation for bias correction in classification, finding that GAN- and diffusion-based augmentation can amplify rather than mitigate subgroup bias in some settings. |

### 自然语言处理与大模型 / NLP & LLM

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [MiroThinker-1.7 & H1: Towards Heavy-Duty Research Agents via Verification](https://arxiv.org/abs/2603.15726v1) | NLP / LLM Agents | 面向重载科研 Agent 构建以验证为核心的训练与推理框架，提升长程研究任务中的事实一致性与结果可信度。It builds a verification-centric training and inference framework for heavy-duty research agents, improving factual consistency and result reliability on long-horizon scientific tasks. |
| [Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs](https://arxiv.org/abs/2603.15051v1) | NLP / Reasoning | 提出隐式推理的自适应锚点精炼机制，让模型在潜在空间中迭代修正推理表示，减少显式链式思维长度。It proposes adaptive anchor refinement for implicit reasoning, letting models iteratively refine representations in latent space and shorten explicit chain-of-thought traces. |

### 智能体与强化学习 / Agents & RL

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [MAC: Multi-Agent Constitution Learning](https://arxiv.org/abs/2603.15968v1) | Agents / Multi-Agent RL | 提出多智能体宪法学习框架，让一组 Agent 通过显式宪法约束协同演化行为策略，提升开放环境中的集体安全性。It proposes multi-agent constitution learning, enabling a group of agents to co-evolve behavioral policies under explicit constitutional constraints for improved collective safety in open environments. |
| [FlashSampling: Fast and Memory-Efficient Exact Sampling](https://arxiv.org/abs/2603.15854v2) | ML / Sampling | 设计快速且内存高效的精确采样算法，在扩散模型等生成流程中降低采样延迟与显存峰值。It designs a fast, memory-efficient exact sampling algorithm that reduces latency and peak memory in generative pipelines such as diffusion models. |

### 机器学习与 AI 基础 / ML & AI Foundations

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [Residual Stream Duality in Modern Transformer Architectures](https://arxiv.org/abs/2603.16039v2) | ML / Transformer Internals | 揭示现代 Transformer 中残差流的对偶结构，为注意力残差等机制提供表示空间层面的理论解释。It reveals a duality structure in the residual stream of modern Transformers, providing a representation-space explanation for mechanisms such as attention residuals. |
| [Linearized Attention Cannot Enter the Kernel Regime at Any Practical Width](https://arxiv.org/abs/2603.13085v2) | ML Theory | 从理论上证明线性化注意力在实际宽度下无法进入核机制，为线性注意力的表达能力边界提供严格分析。It proves that linearized attention cannot enter the kernel regime at practically feasible widths, giving a rigorous analysis of the expressive limits of linear attention. |

### 金融与量化 / Finance & Quant

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [Can Blindfolded LLMs Still Trade? An Anonymization-First Framework for Portfolio Optimization](https://arxiv.org/abs/2603.17692v1) | Quant Finance | 构建“先匿名化”的投资组合优化框架，检验大语言模型在去除敏感标识信息后是否仍能生成有效交易信号。It builds an anonymization-first portfolio-optimization framework to test whether LLMs can still produce effective trading signals after sensitive identifiers are removed. |
| [Hyper-Adaptive Momentum Dynamics for Native Cubic Portfolio Optimization: Avoiding Quadratization Distortion in Higher-Order Cardinality-Constrained Search](https://arxiv.org/abs/2603.15947v1) | Quant Finance | 提出原生三次组合优化方法，用超自适应动量动态避免高阶矩信息在二次化过程中的扭曲。It proposes a native cubic portfolio-optimization approach that uses hyper-adaptive momentum dynamics to avoid distortion of higher-moment information during quadratization. |

### 系统与高效训练 / Systems

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [Federated Learning for Privacy-Preserving Medical AI](https://arxiv.org/abs/2603.15901v1) | Systems / Privacy-Preserving ML | 综述并实证联邦学习在医疗 AI 中的隐私保护训练方案，讨论非独立同分布数据与模型异构性的工程对策。It surveys and empirically evaluates federated learning for privacy-preserving medical AI, discussing engineering strategies for non-IID data and model heterogeneity. |

---

## 领域趋势 / Trends

1. **SSM 架构的数学驱动再设计 / Math-driven SSM redesign** — Mamba-3 从连续 SSM 离散化、复值状态到 MIMO 公式，显示序列模型效率提升仍可从第一性原理中挖掘。
2. **注意力“内部结构”的精细化 / Finer internal attention structures** — Attention Residuals、Residual Stream Duality、Mixture-of-Depths Attention 等尝试在注意力块内部或隐藏流上做文章，以低代价改善表达与推理。
3. **科研 Agent 的验证闭环 / Verification loops for research agents** — MiroThinker 等重载 Agent 将“生成-验证-修正”闭环作为核心，强调结果可信度而非单纯的输出量。
4. **LLM 进入量化投资决策核心 / LLMs at the core of quant investment decisions** — 本周多篇 q-fin 论文将 LLM 用于组合优化、期权策略、风险溢价与投资者行为建模，金融 AI 从信号生成走向决策辅助。
5. **生成模型的高效采样与忠实生成 / Efficient sampling and faithful generation** — FlashSampling、扩散模型对齐与视频生成控制等研究在追求更快推理的同时，也开始关注物理合理性与偏差修正。
6. **多智能体系统的安全宪法化 / Constitutional safety for multi-agent systems** — MAC、TrinityGuard 等研究为多个 Agent 的协作引入显式规则与记忆，试图在开放环境中建立可审计的安全边界。

---

## PhD 阅读路线图 / PhD Reading Roadmap

### 阶段一：建立本周全景（20 分钟）
- 通读 Weekly Highlights 与 Trends，抓住四条主线：SSM 效率跃迁、注意力内部机制、Agent 验证闭环、LLM 量化金融。

### 阶段二：精读重点论文（2–3 小时）
1. **Mamba-3**：聚焦“连续 SSM 离散化 → 复值状态 → MIMO”的三层改进，思考金融时序中的周期性多变量建模映射。
2. **Attention Residuals**：对比标准注意力与残差增强注意力的信息流差异，评估在现有金融 Transformer 中零参数升级的可行性。
3. **LLM Risk Premia Modeling**：阅读 [review.md](llm-risk-premia/review.md)，梳理 LLM 因子与传统 Fama-French 因子的正交性与增量贡献。

### 阶段三：按兴趣深入子领域（各 1–2 小时）
- **序列建模与效率**：Mamba-3 → Linearized Attention 理论 → FlashSampling（从架构到采样）。
- **Transformer 内部机制**：Attention Residuals → Residual Stream Duality → Mixture-of-Depths Attention（由现象到理论）。
- **Agent 安全与科研**：MAC → MiroThinker-1.7 & H1 → TrinityGuard（多 Agent 宪法到科研验证）。
- **量化金融 LLM 应用**：LLM Risk Premia → Blindfolded LLMs Trade → Hyper-Adaptive Momentum（从风险溢价到组合优化）。

### 阶段四：动手实验（2–4 小时）
- 复现或调试 [mamba3/code/](mamba3/code/) 中的示例，尝试在合成正弦/季节性序列上对比复值状态与实值状态的拟合差异。
- 用 LLM Risk Premia 的思路，在公开财报电话会文本上提取风险情绪信号，并与传统市场因子做回归增量检验。

---

## 重要时间节点 / Key Dates

| 事件 / Event | 日期 / Date |
|---|---|
| AAAI 2026 Conference | Feb–Mar 2026 |
| ICLR 2026 Camera Ready | Mar 2026 |
| NeurIPS 2026 Submissions Open | Apr 2026 |

---

## 数据来源 / Sources

- arXiv API (`export.arxiv.org`)，查询类别 cs.CV / cs.LG / cs.AI / cs.CL / stat.ML / q-fin.*，提交日期范围 `2026-03-12` 至 `2026-03-18`。
- 本仓库 Week 11 的 featured reviews（Mamba-3、Attention Residuals、LLM Risk Premia）由先前会话整理，本次补充完整分类列表、趋势、路线图与数据来源说明。
