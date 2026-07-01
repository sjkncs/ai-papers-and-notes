# Week 26: 2026-06-25 — 2026-07-01 / 第 26 周

> **日期说明 / Date note**: 当前运行日期为 2026-07-01，对应 2026 年第 26 周（6 月 25 日–7 月 1 日）。本报告优先覆盖 **2026-06-25 至 2026-07-01** 期间公开的论文；6 月 24 日及此前发布但仍具时效性的工作亦作为补充列入。arXiv 在 6 月 27–28 日（周末）无新公告，6 月 30 日与 7 月 1 日集中释放了 ICML 2026 与 ECCV 2026 的一批重要预印本。
> **Current run date is 2026-07-01**, corresponding to Week 26 of 2026 (June 25 – July 1). This report primarily covers papers publicly released **2026-06-25 through 2026-07-01**; a few still-relevant works released on June 24 or earlier are included as supplementary entries. arXiv had no announcements on June 27–28 (weekend), while June 30 and July 1 saw a concentrated release of ICML 2026 and ECCV 2026 preprints.

---

## 本周学术动态 / Weekly Highlights

**中文：** 2026 年第 26 周是 ICML 2026 预印本持续释放、ECCV 2026 与 ACL 2026 接受论文密集上传的一周。虽然受周末影响实际公告日只有 6 月 24–26 日，但这两天涌现出多个高影响力方向：

1. **递归自我改进的评估器演化**：*The Red Queen Gödel Machine* 打破“固定评估器”假设，提出让 Agent 与评估器共同演化，在代码、学术论文审稿、数学证明三个任务上超越固定评估器 baseline。
2. **实时语音 AI 的安全鸿沟**：*Real-Time Voice AI Hears but Does Not Listen* 对 OpenAI、Google、阿里巴巴四款 production 级语音系统做 red-teaming，揭示“能识别情绪却不依据情绪行动”的情感智能鸿沟。
3. **机器人 VLA 的上下文世界模型**：*In-Context World Modeling for Robotic Control* 把系统辨识变成上下文适应问题，让机器人在新相机视角/新形态下无需微调即可泛化。
4. **三值后训练量化的工程突破**：ICML 2026 Oral *CAT-Q* 用 512 条校准样本把 1.7B–235B 参数 LLM 量化为三值模型，训练 token 比 BitNet 1.58 少约 10 万倍。
5. **LLM 在线 RL 的方向一致性**：*GEOALIGN* 发现高奖励 rollout 可能与批次内大多数 rollout 的方向冲突，提出几何 rollout 策划来稳定训练。
6. **多模态大模型的视觉懒惰**：*Staying VIGILant* 用反事实视觉对齐和强化学习，迫使 MLLM 更依赖视觉证据而非语言先验。
7. **分子系统采样的自回归生成器**：*Autoregressive Boltzmann Generators* 跳出归一化流范式，用自回归模型生成肽链等分子平衡样本。
8. **端到端离散 token 图像生成**：*GEAR* 首次联合端到端训练 VQ tokenizer 与自回归生成器，通过 hard/soft 双读码机制绕过不可微 argmax，ImageNet gFID 收敛速度最高提升 10 倍。
9. **控制中的测试时自适应世界模型**：*AdaJEPA* 在 MPC 闭环中利用执行动作后观测到的真实状态转移做自监督在线更新，无需额外标注即可重新校准 JEPA 世界模型。
10. **面向 Agentic RL 的角色型信用分配**：*TRIAGE* 为 GRPO 的 outcome credit 增加语义角色轴（决定性 / 探索 / 无进展 / 回归），修正均匀轨迹级 advantage 的两个结构性盲区。

**English:** Week 26 of 2026 is a period of continued ICML 2026 preprint releases and dense uploads of ECCV 2026 and ACL 2026 accepted papers. Although the effective arXiv announcement days are only June 24–26 due to the weekend, these two days produced several high-impact directions:

1. **Evolving evaluators for recursive self-improvement**: *The Red Queen Gödel Machine* breaks the assumption of a fixed evaluator and proposes co-evolving agents and evaluators, surpassing fixed-evaluator baselines on code, academic paper reviewing, and mathematical proof tasks.
2. **Safety gap in real-time voice AI**: *Real-Time Voice AI Hears but Does Not Listen* red-teams four production voice systems from OpenAI, Google, and Alibaba, revealing an emotional intelligence gap where systems perceive emotion but fail to act on it.
3. **In-context world models for robot VLAs**: *In-Context World Modeling for Robotic Control* reframes system identification as an in-context adaptation problem, enabling zero-shot generalization to new camera viewpoints and morphologies.
4. **Engineering breakthrough in ternary post-training quantization**: The ICML 2026 Oral *CAT-Q* quantizes 1.7B–235B parameter LLMs into ternary models using only 512 calibration samples, reducing training tokens by roughly 100,000× compared to BitNet 1.58.
5. **Directional consistency in online LLM RL**: *GEOALIGN* finds that high-reward rollouts can conflict with the majority direction in a batch, and proposes geometric rollout curation to stabilize training.
6. **Visual laziness in multimodal LLMs**: *Staying VIGILant* uses counterfactual visual alignment and reinforcement learning to force MLLMs to rely more on visual evidence and less on language priors.
7. **Autoregressive generators for molecular equilibrium sampling**: *Autoregressive Boltzmann Generators* depart from the normalizing-flow paradigm and use autoregressive models to sample equilibrium states of peptides and other molecular systems.
8. **End-to-end discrete-token image generation**: *GEAR* jointly trains a VQ tokenizer and an autoregressive generator, using a dual hard/soft codebook read-out to bypass the non-differentiable argmax and speed up ImageNet gFID convergence by up to 10×.
9. **Test-time adaptive world models for control**: *AdaJEPA* performs online self-supervised updates inside the MPC loop, using observed transitions to recalibrate a JEPA world model without extra labels.
10. **Role-typed credit assignment for agentic RL**: *TRIAGE* adds a semantic role axis (decisive / exploration / no-progress / regression) to GRPO outcome credit, correcting two structural blind spots of uniform trajectory-level advantages.

---

## 重点论文深度分析 / Featured Papers with Deep Reviews

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [GEAR: Guided End-to-End AutoRegression for Image Synthesis](https://arxiv.org/abs/2606.32039) | CV / Generative Models | [review.md](gear/review.md) | [code/](gear/code/) |
| 2 | [AdaJEPA: An Adaptive Latent World Model](https://arxiv.org/abs/2606.32026) | Robotics / World Models | [review.md](adajepa/review.md) | — |
| 3 | [TRIAGE: Role-Typed Credit Assignment for Agentic Reinforcement Learning](https://arxiv.org/abs/2606.32017) | Agents / RL | [review.md](triage/review.md) | — |
| 4 | [The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators](https://arxiv.org/abs/2606.26294) | Agents / RL / ML Theory | [review.md](red-queen-godel-machine/review.md) | — |
| 5 | [Real-Time Voice AI Hears but Does Not Listen](https://arxiv.org/abs/2606.26083) | NLP / LLM / Multimodal Safety | [review.md](real-time-voice-ai/review.md) | — |
| 6 | [In-Context World Modeling for Robotic Control](https://arxiv.org/abs/2606.26025) | Robotics / VLA / World Models | [review.md](in-context-world-modeling/review.md) | — |
| 7 | [CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs](https://arxiv.org/abs/2606.26650) | Systems / Efficient LLM | [review.md](cat-q/review.md) | [code/](cat-q/code/) |

---

## 本周论文列表（按领域分类）/ Weekly Papers by Area

> 本周 arXiv 主要类别新增论文集中在 6 月 24–26 日，覆盖 CV、NLP/LLM、ML Theory、Systems、Agents/RL、Robotics 与 Multimodal。以下按领域精选代表性工作，并附简短要点。  
> New arXiv submissions this week concentrate on June 24–26 across CV, NLP/LLM, ML Theory, Systems, Agents/RL, Robotics, and Multimodal. Representative works are listed below with brief key points.

### 计算机视觉 / CV

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **[GEAR: Guided End-to-End AutoRegression for Image Synthesis](https://arxiv.org/abs/2606.32039)** | 联合端到端训练 VQ tokenizer 与 AR 生成器；hard/soft 双读码 + REPA 对齐，ImageNet gFID 收敛速度提升最高 10 倍。 |
| **[Paying More Attention to Visual Tokens in Self-Evolving Large Multimodal Models](https://arxiv.org/abs/2606.27373)** | ECCV 2026。自演进多模态模型中加强对视觉 token 的关注。 |
| **[Staying VIGILant: Mitigating Visual Laziness via Counterfactual Visual Alignment in MLLMs](https://arxiv.org/abs/2606.26387)** | ECCV 2026。通过反事实视觉对齐与 RL 惩罚“视觉懒惰”，减少幻觉。 |
| **[LCG: Long-Context Consistent Image Generation with Sparse Relational Attention](https://arxiv.org/abs/2606.26171)** | 用稀疏关系注意力实现多图序列一致性生成，无需 per-subject 微调。 |
| **[GeMoE: Gating Entropy is All You Need for Uncertainty-aware Adaptive Routing in MoE-based Large Vision-Language Models](https://arxiv.org/abs/2606.26287)** | 利用门控熵动态选择专家数量，降低 MoE-VLM 推理成本。 |
| **[Neural Voxel Dynamics: Learning Implicit 3D Physics via Volumetric Feature Advection](https://arxiv.org/abs/2606.26410)** | 从单目视频学习体素级隐式 3D 物理动态，无需显式 3D 监督。 |
| **[LiveEdit: Towards Real-Time Diffusion-Based Streaming Video Editing](https://arxiv.org/abs/2606.26740)** | 将扩散视频编辑蒸馏到轻量网络，实现接近实时的流式编辑。 |
| **[ResilPhase: Plug-and-Play Phase Mapping for Fast Diffusion Acceleration](https://arxiv.org/abs/2606.26769)** | ECCV 2026。通过相位映射插件加速扩散模型。 |
| **[DanceOPD: On-Policy Generative Field Distillation](https://arxiv.org/abs/2606.27377)** | 在线策略生成场蒸馏。 |
| **[Ask, Solve, Generate: Self-Evolving Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2606.27376)** | 通过自一致性奖励实现自演进统一多模态理解与生成。 |
| **[Safe Autoregressive Image Generation with Iterative Self-Improving Codebooks](https://arxiv.org/abs/2606.27147)** | ICML 2026。迭代自改进码本的安全自回归图像生成。 |
| **[See & Sniff: Learning Visuo-Olfactory Representations](https://arxiv.org/abs/2606.27307)** | ECCV 2026。学习视觉-嗅觉联合表征。 |
| **[NaviCache: Test-Time Self-Calibration Caching for Video Generation](https://arxiv.org/abs/2606.26795)** | ICML 2026。视频生成的测试时自校准缓存。 |

### 自然语言处理与大模型 / NLP & LLM

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **[Real-Time Voice AI Hears but Does Not Listen](https://arxiv.org/abs/2606.26083)** | 对四款实时语音 AI 做 red-teaming，揭示情感智能鸿沟。 |
| **[Empowering GUI Agents via Autonomous Experience Exploration and Hindsight Experience Utilization for Task Planning](https://arxiv.org/abs/2606.27330)** | ACL 2026 Main。通过自主经验探索与事后经验利用增强 GUI Agent 任务规划。 |
| **[Prompt Injection in Automated Résumé Screening](https://arxiv.org/abs/2606.27287)** | ACL 2026 Findings。自动简历筛选中的提示注入攻击。 |
| **[Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation](https://arxiv.org/abs/2606.27226)** | ICML 2026 Workshop。用二元问题提升 LLM 评估可解释性。 |
| **[CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs](https://arxiv.org/abs/2606.26650)** | ICML 2026 Oral。后训练三值量化 1.7B–235B 参数 LLM。 |
| **[Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding](https://arxiv.org/abs/2606.26120)** | 扩散语言模型的训练无关加速。 |
| **[Nemotron-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context](https://arxiv.org/abs/2606.26493)** | 双塔架构融合自回归上下文与扩散去噪。 |
| **[Speaking Numbers to LLMs](https://arxiv.org/abs/2606.26487)** | IJCAI 2026。用小波嵌入做时间序列预测。 |
| **[Know2Guess: A Contamination-Aware Multi-Zone Benchmark for Knowledge-Boundary Evaluation](https://arxiv.org/abs/2606.26101)** | 无数据污染的知识边界评测基准。 |
| **[Information-Aware KV Cache Compression for Long Reasoning](https://arxiv.org/abs/2606.26875)** | 基于预测不确定性的长推理 KV Cache 压缩。 |
| **[ConflictScore: Identifying and Measuring How Language Models Handle Conflicting Evidence](https://arxiv.org/abs/2606.26437)** | 量化 LLM 处理冲突证据的能力。 |

### 机器学习理论 / ML Theory

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **[SemRF: A Semantic Reference Frame for Residual-Stream Dynamics in Language Models](https://arxiv.org/abs/2606.32022)** | 为语言模型残差流分析建立语义参考框架，通过固定锚点分离“测量漂移”与“真实计算动态”，并定义语义 Voronoi 轨迹与知识密度。 |
| **[The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators](https://arxiv.org/abs/2606.26294)** | 让评估器与 Agent 共同演化，实现非平稳效用下的递归自我改进。 |
| **[Blackwell Approachability and Gradient Equilibrium are Equivalent](https://arxiv.org/abs/2606.27315)** | COLT 2026。证明 Blackwell 可达性与梯度均衡等价。 |
| **[The Geometry of Updates: Fisher Alignment at Vocabulary Scale](https://arxiv.org/abs/2606.27242)** | ICML 2026。词汇表尺度上的 Fisher 对齐分析。 |
| **[Finding Stationary Points by Comparisons](https://arxiv.org/abs/2606.27032)** | ICML 2026。通过比较寻找稳定点。 |
| **[High-Probability PL-SGD with Markovian Noise](https://arxiv.org/abs/2606.26316)** | 马尔可夫噪声下 PL-SGD 的高概率收敛保证。 |
| **[Learning Probabilistic Filters with Strictly Proper Scoring Rules](https://arxiv.org/abs/2606.26497)** | 用严格适当评分规则训练概率滤波器。 |
| **[Natural Ungrokking: Asymmetric Control of Which Rules Survive Pretraining](https://arxiv.org/abs/2606.26050)** | 研究预训练中哪些规则会被保留或遗忘。 |
| **[Model Forensics: Investigating Whether Concerning Behavior Reflects Misalignment](https://arxiv.org/abs/2606.26071)** | 通过 CoT 与反事实实验区分模型恶意行为与混淆。 |

### 系统 / Systems

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **[CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs](https://arxiv.org/abs/2606.26650)** | ICML 2026 Oral。PTQ 三值量化，8×A100 上 8–60 小时完成。 |
| **[SOLAR: AI-Powered Speed-of-Light Performance Analysis](https://arxiv.org/abs/2606.26383)** | 用 LLM 把 PyTorch/JAX 代码翻译为性能分析模型。 |
| **[Moebius: Serving Mixture-of-Expert Models with Seamless Runtime Parallelism Switch](https://arxiv.org/abs/2606.26607)** | MoE 服务时无缝切换专家并行与张量并行。 |
| **[DMuon: Efficient Distributed Muon Training with Near-Adam Overhead](https://arxiv.org/abs/2606.27153)** | 首个开源分布式 Muon 优化器实现。 |
| **[Optimizing CUDA like a Human: Micro-Profiling Tools as Expert Surrogates for LLM-Based GPU Kernel Optimization](https://arxiv.org/abs/2606.26453)** | LLM + 微分析器迭代优化 CUDA kernel。 |
| **[RolloutPipe: Overlapping Pipelined Rollout and Training in Disaggregated On-Policy LLM RL](https://arxiv.org/abs/2606.26997)** | RLVR 系统中 rollout 与训练流水线重叠。 |
| **[Speculation at a Distance: Where Edge-Cloud Speculative Decoding Actually Pays Off](https://arxiv.org/abs/2606.25091)** | 边缘-云端投机解码的适用场景理论分析。 |

### Agent / 强化学习 / Agents / RL

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **[TRIAGE: Role-Typed Credit Assignment for Agentic Reinforcement Learning](https://arxiv.org/abs/2606.32017)** | 用结构化 LLM judge 为 GRPO 的 segment 分配语义角色（D/E/N/R），修正 outcome-only credit 的两个结构性盲区。 |
| **[The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators](https://arxiv.org/abs/2606.26294)** | Agent 与评估器共同演化的递归自我改进框架。 |
| **[GEOALIGN: Geometric Rollout Curation for Robust LLM Reinforcement Learning](https://arxiv.org/abs/2606.26917)** | ICML 2026。通过几何一致性策划 rollout，稳定在线 RL。 |
| **[The Verification Horizon: No Silver Bullet for Coding Agent Rewards](https://arxiv.org/abs/2606.26300)** | 编码 Agent 的奖励验证困境与奖励黑客抑制。 |
| **[SKILL-DISCO: Distilling and Compiling Agent Traces into Reusable Procedural Skills](https://arxiv.org/abs/2606.26669)** | 将成功 Agent 轨迹蒸馏为可复用程序化技能。 |
| **[Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents](https://arxiv.org/abs/2606.26080)** | 发现后训练策略与参考策略的对数比可作为免费过程奖励信号。 |
| **[The Unfireable Safety Kernel: Execution-Time AI Alignment for AI Agents](https://arxiv.org/abs/2606.26057)** | 用 Rust 实现不可被绕过的 Agent 运行时安全内核。 |
| **[Autoformalization of Agent Instructions into Policy-as-Code](https://arxiv.org/abs/2606.26649)** | ICML 2026 Workshop。将 Agent 指令自动形式化为策略代码。 |
| **[Reasoning Quality Emerges Early: Data Curation for Reasoning Models](https://arxiv.org/abs/2606.26797)** | ICML 2026。推理模型的数据策划。 |

### 机器人学 / Robotics

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **[AdaJEPA: An Adaptive Latent World Model](https://arxiv.org/abs/2606.32026)** | 在 MPC 闭环中做测试时自适应：每次执行动作后用观测到的真实转移自监督更新 JEPA 世界模型，提升 ID/OOD 规划成功率。 |
| **[In-Context World Modeling for Robotic Control](https://arxiv.org/abs/2606.26025)** | 将系统辨识视为上下文适应，无需微调适应新相机视角/形态。 |
| **[FORCE: Efficient VLA Reinforcement Fine-Tuning via Value-Calibrated Warm-up and Self-Distillation](https://arxiv.org/abs/2606.26006)** | 通过价值校准预热与自蒸馏高效微调 VLA。 |
| **[OmniContact: Chaining Meta-Skills via Contact Flow for Generalizable Humanoid Loco-Manipulation](https://arxiv.org/abs/2606.26201)** | 用 contact flow 串联元技能实现人形机器人移动-操作泛化。 |
| **[Tactile-WAM: Touch-Aware World Action Model with Tactile Asymmetric Attention](https://arxiv.org/abs/2606.26801)** | 在世界动作模型中引入触觉非对称注意力。 |
| **[Humanoid-DART: Humanoid Loco-Manipulation using Diffusion-guided Augmentation](https://arxiv.org/abs/2606.26955)** | 扩散引导的人形机器人移动-操作数据增强。 |
| **[RobOralScan: Learning Active Intraoral Scanning for Robotic Dental Reconstruction](https://arxiv.org/abs/2606.27123)** | 机器人主动口腔扫描 RL 流水线。 |
| **[RelAfford6D: Relational 6D Affordance Graphs for Constraint-Driven Robotic Manipulation](https://arxiv.org/abs/2606.27146)** | 关系化 6D affordance 图用于约束驱动操作。 |
| **[FAR-LIO: Fast, Accurate, and Robust LiDAR-Inertial Odometry](https://arxiv.org/abs/2606.26010)** | 高速自主导航的雷达惯性里程计。 |

### 多模态 / Multimodal

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **[Same Evidence, Different Answer: Auditing Order Sensitivity in Multimodal Large Language Models](https://arxiv.org/abs/2606.26079)** | 审计 MLLM 对输入顺序的敏感性，揭示答案稳定性缺陷。 |
| **[What We are Missing in Multimodal LLM Evaluation?](https://arxiv.org/abs/2606.26348)** | 综述 MLLM 评测盲区（时序-空间一致性、物理世界理解）。 |
| **[MKG-RAG-Bench: Benchmarking Retrieval in Multimodal Knowledge Graph-Augmented Generation](https://arxiv.org/abs/2606.26458)** | 多模态知识图增强生成检索基准。 |
| **[Unison: Benchmarking Unified Multimodal Models via Synergistic Understanding and Generation](https://arxiv.org/abs/2606.26984)** | ICML 2026。统一多模态模型评测基准。 |

### 科学计算 / Scientific Computing

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **[Autoregressive Boltzmann Generators](https://arxiv.org/abs/2606.27361)** | ICML 2026 Spotlight。跳出归一化流，用自回归模型采样分子平衡态；在 8 残基系统上 zero-shot 能量误差降低 60% 以上。 |
| **[LithoDreamer: A Physics-Informed World Model for Computational Lithography](https://arxiv.org/abs/2606.26713)** | ICML 2026。计算光刻的物理信息世界模型。 |
| **[Target-Aware Bandit Allocation for Scalable Surrogate Optimization in Chemical Space](https://arxiv.org/abs/2606.26657)** | ICML 2026。化学空间代理优化中的目标感知 bandit 分配。 |

---

## 趋势观察 / Trend Observations

**中文：**

1. **从“固定评估”到“共同演化”**：RQGM 代表 Agent 自我改进研究的一个转折点——不再把评估器当作外部常量，而是让其与策略同步演化。这既是能力上的扩展，也是安全上的新风险点。
2. **多模态系统的“感知-行动鸿沟”**：语音 AI 的“情感智能鸿沟”和 MLLM 的“视觉懒惰”共同揭示：多模态模型往往能“感知”到正确信息，但决策/生成时却被单一模态（通常是文本）主导。如何关闭这一鸿沟将成为未来一年多模态安全研究的核心。
3. **机器人学习进入“上下文适应”时代**：ICWM 把系统辨识做成上下文学习，与 VLA 的“微调泛化”路线形成对比。未来可能出现更多“零参数适应”的机器人策略，降低部署成本。
4. **量化从“训练感知”走向“后训练极致压缩”**：CAT-Q 证明 PTQ 三值化也能达到甚至超越 QAT 方法，配合硬件优化可能显著降低端侧/边缘推理成本。
5. **在线 RL 的稳定性问题被重新重视**：GEOALIGN 从几何一致性角度切入 rollout curation，说明随着 LLM RL 规模扩大，高方差、奖励黑客和方向不一致将成为工程与理论的双重焦点。
6. **端到端训练重新进入离散 AR 生成**：GEAR 证明在 VQ-AR pipeline 中，tokenizer 与 AR 生成器可以联合训练；关键在于把对齐损失与 NTP 损失解耦，避免 straight-through estimator 崩溃。
7. **世界模型从“训练完冻结”走向“部署中自适应”**：AdaJEPA 把测试时训练引入 MPC 闭环，预示着未来控制算法将同时具备预训练先验与在线适应能力。
8. **Agentic RL 需要更细粒度的信用语言**：TRIAGE 用结构化角色标签替代单一 outcome score，说明在复杂交互任务中，segment-level 的语义诊断将成为 reward engineering 的新层次。

**English:**

1. **From fixed evaluation to co-evolution**: RQGM represents a turning point in agent self-improvement research—evaluators are no longer treated as external constants but co-evolve with the policy. This is both a capability expansion and a new safety risk.
2. **The perception-action gap in multimodal systems**: The emotional intelligence gap in voice AI and visual laziness in MLLMs both reveal that multimodal models often perceive correct information but are dominated by a single modality (usually text) during decision-making or generation. Closing this gap will be central to multimodal safety research in the coming year.
3. **Robotic learning enters the in-context adaptation era**: ICWM reframes system identification as in-context learning, contrasting with the fine-tuning-for-generalization paradigm in VLAs. We may see more zero-parameter adaptation methods for robot policies, reducing deployment costs.
4. **Quantization moves from training-aware to extreme post-training compression**: CAT-Q shows that PTQ ternarization can match or surpass QAT methods, and with hardware optimization could significantly reduce edge inference costs.
5. **Stability of online RL is re-emphasized**: GEOALIGN approaches rollout curation from a geometric-consistency perspective, showing that as LLM RL scales, high variance, reward hacking, and directional inconsistency will become dual engineering and theoretical focuses.
6. **End-to-end training re-enters discrete AR generation**: GEAR shows that in VQ-AR pipelines the tokenizer and AR generator can be jointly trained; the key is decoupling the alignment loss from the NTP loss to avoid straight-through estimator collapse.
7. **World models move from frozen-after-training to adaptive-during-deployment**: AdaJEPA brings test-time training into the MPC closed loop, suggesting future control algorithms will combine pretrained priors with online adaptation.
8. **Agentic RL needs finer-grained credit languages**: TRIAGE replaces a single outcome score with structured role labels, showing that segment-level semantic diagnosis will become a new layer of reward engineering in complex interactive tasks.

---

## 博士生阅读路线建议 / PhD Reading Roadmap

**中文：**

| 方向 / Direction | 本周必读 / Must-read this week | 延伸阅读 / Further reading | 建议跳过 / Skip for now |
|---|---|---|---|
| **Agent / 自我改进** | The Red Queen Gödel Machine | Gödel Machine 原始论文； verifier/reward model 演化相关文献 | 纯实现细节若代码未开源 |
| **多模态安全 / 语音** | Real-Time Voice AI Hears but Does Not Listen | SER/副语言理解；MLLM 模态偏见研究 | 若你不做语音，可只看结论 |
| **机器人 / VLA** | In-Context World Modeling for Robotic Control | 系统辨识教材；OpenVLA / π0；domain randomization | 纯视觉或纯语言研究者 |
| **高效 LLM / 系统** | CAT-Q | BitNet b1.58；GPTQ / AWQ / SmoothQuant 等 PTQ 方法 | 若只关心算法理论 |
| **RL / 对齐** | GEOALIGN + VIGILant | GRPO / PPO / DPO 稳定性分析；MLLM 对齐 | 若只关心传统 CV/NLP benchmark |
| **ML 理论** | Blackwell Approachability and Gradient Equilibrium are Equivalent | 博弈论与在线学习经典教材 | 若只做应用工程 |
| **生成模型 / AR** | GEAR | REPA / REPA-E；VQ-VAE / LFQ / IBQ；LlamaGen | 若不做视觉生成 |
| **机器人 / 世界模型** | AdaJEPA | JEPA / H-JEPA / V-JEPA；MPC；online system identification | 若不做机器人或控制 |
| **Agent / RL 信用分配** | TRIAGE | GRPO / PPO / PRM；outcome-supervised vs process reward | 若不做长程交互 Agent |

**English:**

| Direction | Must-read this week | Further reading | Skip for now |
|---|---|---|---|
| **Agent / self-improvement** | The Red Queen Gödel Machine | Original Gödel Machine paper; verifier/reward-model evolution literature | Implementation details if code is not open |
| **Multimodal safety / speech** | Real-Time Voice AI Hears but Does Not Listen | SER / paralinguistic understanding; modality bias in MLLMs | Conclusions-only if you do not work on speech |
| **Robotics / VLA** | In-Context World Modeling for Robotic Control | System identification textbooks; OpenVLA / π0; domain randomization | Pure vision or pure language researchers |
| **Efficient LLM / systems** | CAT-Q | BitNet b1.58; GPTQ / AWQ / SmoothQuant PTQ methods | If you only care about algorithm theory |
| **RL / alignment** | GEOALIGN + Staying VIGILant | GRPO / PPO / DPO stability; MLLM alignment | If you only care about traditional CV/NLP benchmarks |
| **ML theory** | Blackwell Approachability and Gradient Equilibrium are Equivalent | Game theory and online learning classics | If you only do applied engineering |
| **Generative models / AR** | GEAR | REPA / REPA-E; VQ-VAE / LFQ / IBQ; LlamaGen | If you do not work on visual generation |
| **Robotics / world models** | AdaJEPA | JEPA / H-JEPA / V-JEPA; MPC; online system identification | If you do not work on robotics or control |
| **Agents / RL credit assignment** | TRIAGE | GRPO / PPO / PRM; outcome-supervised vs process reward | If you do not work on long-horizon interactive agents |

---

## 本周推荐行动 / Recommended Actions for This Week

**中文：**
- 如果你在做 **Agent 自我改进**，重点理解 RQGM 的“受控效用演化”设计，并思考它与你当前固定评估器 pipeline 的结合点。
- 如果你在做 **语音或多模态产品**，请阅读 *Real-Time Voice AI Hears but Does Not Listen* 的实验设计，考虑在自己的产品中加入情绪-决策一致性检查。
- 如果你在做 **机器人 VLA 部署**，尝试把 ICWM 的思想融入你的系统：在任务执行前增加一小段“系统探索”上下文，观察泛化性能变化。
- 如果你在做 **端侧 LLM**，运行本目录 `cat-q/code/cat_q_demo.py`，理解三值 PTQ 的核心机制，并评估是否适用于你的模型规模。
- 如果你在做 **自回归图像生成**，运行本目录 `gear/code/gear_demo.py`，观察 dual hard/soft read-out 如何在不触发 STE 崩溃的情况下实现 tokenizer 端到端引导。

**English:**
- If you work on **agent self-improvement**, focus on understanding RQGM's "controlled utility evolution" design and consider how it might integrate with your current fixed-evaluator pipeline.
- If you work on **speech or multimodal products**, read the experimental design of *Real-Time Voice AI Hears but Does Not Listen* and consider adding emotion-decision consistency checks to your product.
- If you work on **robot VLA deployment**, try incorporating the ICWM idea into your system: add a short "system exploration" context before task execution and observe generalization improvements.
- If you work on **on-device LLMs**, run `cat-q/code/cat_q_demo.py` in this directory to understand the core mechanism of ternary PTQ and evaluate whether it fits your model scale.
- If you work on **autoregressive image generation**, run `gear/code/gear_demo.py` to see how the dual hard/soft read-out enables end-to-end guidance of the tokenizer without STE collapse.
