# Week 10: 2026-03-05 — 2026-03-11 / 第10周

---

## 本周学术动态 / Weekly Highlights

**中文：** 2026 年第 10 周（3 月 5–11 日），arXiv 五个核心类别（cs.CV / cs.LG / cs.AI / cs.CL / stat.ML）共新增约 **2,610** 篇论文。本周核心主题围绕“Agentic AI 的工程化落地”与“推理效率的视觉-语言协同”展开：
1. **小模型多语言深度覆盖**：*Tiny Aya* 以 3.35B 参数覆盖 70 种语言，展示“用更聪明的训练替代简单缩放”的新路径；
2. **金融 Agent 的系统化架构**：*AI Agents in Financial Markets* 提出感知-推理-决策-执行四层架构，将金融自动化从孤立预测任务推向端到端代理工作流；
3. **自主因子投资的 Agentic AI 实践**：*Autonomous Factor Investing via Agentic AI* 用 LLM 驱动的因子假设、自动回测与自适应组合，报告样本外 Sharpe 3.11；
4. **推理效率与可信性并重**：从 SPOT 的潜在停顿推理、SmartThinker 的 CoT 长度校准，到 C2-Faith 对 CoT 忠实度的评测，研究者在压缩推理成本的同时追问推理是否可信；
5. **具身智能与世界模型交汇**：MetaWorld-X、PlayWorld、AtomicVLA 等工作将 VLA、世界模型与原子技能学习结合，推动机器人从模仿走向可组合泛化。

**English:** Week 10 of 2026 (March 5–11) saw approximately **2,610** new papers across the five core arXiv categories (cs.CV / cs.LG / cs.AI / cs.CL / stat.ML). The week centered on the engineering of Agentic AI and vision-language synergy for reasoning efficiency:
1. **Small-model multilingual depth**: *Tiny Aya* covers 70 languages with only 3.35B parameters, showing a new path of "smarter training instead of simple scaling";
2. **Systematic financial-agent architecture**: *AI Agents in Financial Markets* proposes a four-layer perception-reasoning-decision-execution architecture, moving financial automation from isolated prediction to end-to-end agentic workflows;
3. **Agentic AI for autonomous factor investing**: *Autonomous Factor Investing via Agentic AI* uses LLM-driven factor hypotheses, automatic backtesting, and adaptive portfolio construction, reporting an out-of-sample Sharpe of 3.11;
4. **Reasoning efficiency plus trustworthiness**: SPOT compresses explicit CoT into latent reasoning, SmartThinker calibrates CoT length, and C2-Faith benchmarks causal/coverage faithfulness, asking whether cheaper reasoning remains reliable;
5. **Embodied AI meets world models**: MetaWorld-X, PlayWorld, and AtomicVLA combine VLA, world models, and atomic skill learning to push robots from imitation toward compositional generalization.

---

## 重点论文深度分析 / Featured Papers with Deep Reviews

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [Tiny Aya: Bridging Scale and Multilingual Depth](https://arxiv.org/abs/2603.11510) | Multilingual LLM | [review.md](tiny-aya/review.md) | — |
| 2 | [AI Agents in Financial Markets: Architecture, Applications, and Systemic Implications](https://arxiv.org/abs/2603.13942) | AI Finance Survey | [review.md](ai-agents-finance/review.md) | — |
| 3 | [Autonomous Factor Investing via Agentic AI](https://arxiv.org/abs/2603.14288) | Quant Factor Investing | [review.md](autonomous-factor/review.md) | [numpy code](autonomous-factor/code/autonomous_factor.py) / [PyTorch code](autonomous-factor/code/autonomous_factor_pytorch.py) |

---

## 本周论文列表（按领域分类）/ Weekly Papers by Area

> 本周 arXiv 五个主要类别新增约 **2,610** 篇论文（submitted dates: 2026-03-05 to 2026-03-11）。以下按领域精选代表性工作，并附简短双语要点。

> This week added approximately **2,610** papers across five main arXiv categories (submitted dates: 2026-03-05 to 2026-03-11). Representative works by area are listed below with brief bilingual key points.

### 计算机视觉 / Computer Vision

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [EVLF: Early Vision-Language Fusion for Generative Dataset Distillation](https://arxiv.org/abs/2603.07476v1) | CV | 提出早期视觉-语言融合方法，通过在编码器与生成主干之间对齐文本和视觉嵌入，改善扩散式数据集蒸馏的语义忠实度与视觉一致性。It proposes an early vision-language fusion method that aligns text and visual embeddings between the encoder and generative backbone to improve semantic fidelity and visual coherence in diffusion-based dataset distillation. |
| [DyQ-VLA: Temporal-Dynamic-Aware Quantization for Embodied Vision-Language-Action Models](https://arxiv.org/abs/2603.07904v2) | CV | 提出面向具身 VLA 模型的时序动态感知量化框架，基于运动学代理动态切换并分配比特宽度，在保持 99.5% 性能的同时将内存占用降至 30.9%。It proposes a temporal-dynamic-aware quantization framework for embodied VLAs that uses kinematic proxies to dynamically switch and allocate bit-widths, reducing memory footprint to 30.9% while retaining 99.5% performance. |
| [SAMoE-VLA: A Scene Adaptive Mixture-of-Experts Vision-Language-Action Model for Autonomous Driving](https://arxiv.org/abs/2603.08113v1) | CV | 提出场景自适应 MoE VLA 模型，用鸟瞰图特征替代 token 嵌入进行专家路由，并引入条件跨模态因果注意力，在自动驾驶开环与闭环基准上达到最先进性能。It proposes a scene-adaptive MoE VLA that routes experts with bird's-eye-view features and introduces conditional cross-modal causal attention, achieving state-of-the-art performance on open- and closed-loop autonomous-driving benchmarks. |
| [LiveWorld: Simulating Out-of-Sight Dynamics in Generative Video World Models](https://arxiv.org/abs/2603.07145v2) | CV | 提出让生成式视频世界模型模拟视野外动态持续演化的框架，通过持久全局状态和监控机制保证物体在不可见时仍持续演化。It proposes a framework that enables generative video world models to simulate persistent out-of-sight dynamics via a persistent global state and monitor mechanism. |
| [HALP: Detecting Hallucinations in Vision-Language Models without Generating a Single Token](https://arxiv.org/abs/2603.05465v1) | CV | 在不生成任何 token 的情况下，通过探测 VLM 内部表示预测幻觉风险，在多款现代 VLM 上达到最高 0.93 AUROC，支持早期干预与选择性解码。It predicts hallucination risk in VLMs without generating any tokens by probing internal representations, reaching up to 0.93 AUROC across modern VLMs to enable early intervention and selective decoding. |
| [MultiHaystack: Benchmarking Multimodal Retrieval and Reasoning over 40K Images, Videos, and Documents](https://arxiv.org/abs/2603.05697v1) | CV | 推出首个大规模跨模态检索与推理基准，涵盖 46,000 余个图像、视频和文档候选，揭示跨模态检索是 MLLM 端到端可靠性的主要瓶颈。It introduces the first large-scale cross-modal retrieval-and-reasoning benchmark with over 46,000 candidates, revealing multimodal retrieval as a key bottleneck for end-to-end MLLM reliability. |
| [PatchCue: Enhancing Vision-Language Model Reasoning with Patch-Based Visual Cues](https://arxiv.org/abs/2603.05869v2) | CV | 提出基于图像块的视觉提示范式，通过冷启动监督微调与过程奖励强化学习增强 VLM 视觉推理，性能优于像素级边界框与点提示。It proposes a patch-based visual cue paradigm that enhances VLM reasoning via supervised fine-tuning and process-reward RL, outperforming pixel-level bounding boxes and point cues. |

### 自然语言处理与大模型 / NLP & LLM

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [C2-Faith: Benchmarking LLM Judges for Causal and Coverage Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2603.05167v2) | NLP/LLM | 提出评估链式思维推理中因果忠实度与覆盖忠实度的基准，发现 LLM 裁判能检测错误但难以定位，并系统性地高估推理完整性。It introduces a benchmark for causal and coverage faithfulness in CoT reasoning, finding that LLM judges detect errors but struggle to localize them and systematically overestimate reasoning completeness. |
| [SmartThinker: Progressive Chain-of-Thought Length Calibration for Efficient Large Language Model Reasoning](https://arxiv.org/abs/2603.08000v2) | NLP/LLM | 提出基于 GRPO 的思维链长度渐进校准方法，动态估计最佳长度并调节长度奖励系数，实现最高 52.5% 的平均长度压缩同时提升准确率。It proposes a GRPO-based progressive CoT length calibration method that dynamically estimates the optimal length and modulates length-reward coefficients, achieving up to 52.5% length compression while improving accuracy. |
| [BandPO: Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds for LLM Reinforcement Learning](https://arxiv.org/abs/2603.04918v1) | NLP/LLM | 提出带约束策略优化，将 f-散度信任区域投影为动态概率感知裁剪区间，缓解 PPO 式裁剪对低概率高优势动作过度抑制导致的熵崩溃。It introduces Band-constrained Policy Optimization that projects f-divergence trust regions into dynamic probability-aware clipping intervals, mitigating entropy collapse caused by PPO-style clipping over-suppressing low-probability high-advantage actions. |
| [Safer Reasoning Traces: Measuring and Mitigating Chain-of-Thought Leakage in LLMs](https://arxiv.org/abs/2603.05618v1) | NLP/LLM | 系统测量链式思维提示中的个人信息泄漏风险，发现 CoT 持续增加高隐私类别泄漏，并评估多种轻量级推理时守门员策略。It systematically measures PII leakage risk in CoT prompting, finds that CoT consistently increases leakage especially for high-risk categories, and evaluates lightweight inference-time gatekeepers. |
| [Re^2: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving](https://arxiv.org/abs/2603.07197v1) | NLP/LLM | 提出通过强化学习让 LLM 学会重新求解，当推理路径不佳时主动放弃并重启，在纯 RL 下将“重做”行为从 0.5% 提升至 30% 以上。It proposes teaching LLMs to re-solve via RL, enabling them to abandon unproductive reasoning paths and restart, amplifying redo behavior from 0.5% to over 30% with pure RL. |
| [Ares: Adaptive Reasoning Effort Selection for Efficient LLM Agents](https://arxiv.org/abs/2603.07915v1) | NLP/LLM | 提出面向多步 Agent 任务的每步推理 effort 动态选择框架，通过轻量路由器根据交互历史选择最低适宜推理级别，减少高达 52.7% 的推理 token 消耗。It proposes a per-step dynamic reasoning-effort selection framework for multi-step agents, using a lightweight router to choose the lowest adequate reasoning level based on interaction history, reducing reasoning token usage by up to 52.7%. |
| [ReflexiCoder: Teaching Large Language Models to Self-Reflect on Generated Code and Self-Correct It via Reinforcement Learning](https://arxiv.org/abs/2603.05863v2) | NLP/LLM | 提出将代码生成、反思与自纠错内化为模型权重的强化学习框架，无需执行反馈即可在推理时自主调试，在多项代码基准上达到开源模型最优。It proposes an RL framework that internalizes code generation, reflection, and self-correction into model weights, enabling autonomous debugging at inference without execution feedback and achieving state-of-the-art results among open-source models on multiple coding benchmarks. |
| [SPOT: Span-level Pause-of-Thought for Efficient and Interpretable Latent Reasoning in Large Language Models](https://arxiv.org/abs/2603.06222v1) | NLP/LLM | 提出基于 Sinkhorn 跨层语义对齐的潜在停顿 token 框架，将显式思维链压缩为可解释潜在推理，平均提升 2.3 点准确率并减少 37.5% 生成 token。It proposes a latent pause-token framework with span-level semantic alignment that compresses explicit CoT into interpretable latent reasoning, improving accuracy by 2.3 points on average while reducing generated tokens by 37.5%. |

### 智能体与强化学习 / Agents & RL

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [NePPO: Near-Potential Policy Optimization for General-Sum Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2603.06977v2) | Agents/RL | 提出近似势函数策略优化方法，通过学习玩家无关势函数将一般和博弈的纳什均衡近似转化为合作博弈求解，并在混合合作竞争环境中验证。It proposes Near-Potential Policy Optimization that learns a player-independent potential function to approximate Nash equilibria of general-sum games via cooperative-game solutions, validated in mixed cooperative-competitive environments. |
| [KARL: Knowledge Agents via Reinforcement Learning](https://arxiv.org/abs/2603.05218v1) | Agents/RL | 提出面向企业搜索 Agent 的强化学习训练系统，构建多能力评估套件 KARLBench，并通过迭代大批量离线 RL 实现成本-质量帕累托最优。It presents an RL training system for enterprise search agents, building the multi-capability KARLBench and achieving Pareto-optimal cost-quality trade-offs via iterative large-batch off-policy RL. |
| [AutoResearch-RL: Perpetual Self-Evaluating Reinforcement Learning Agents for Autonomous Neural Architecture Discovery](https://arxiv.org/abs/2603.07300v2) | Agents/RL | 提出让 RL 智能体自主进行神经网络架构与超参数研究的框架，在固定时间预算内迭代修改训练脚本并以验证 bits-per-byte 为奖励，无需人工干预。It presents a framework where an RL agent autonomously conducts neural architecture and hyperparameter research by iteratively modifying training scripts under a fixed time budget with validation bits-per-byte as reward, without human intervention. |
| [Graph-GRPO: Training Graph Flow Models with Reinforcement Learning](https://arxiv.org/abs/2603.10395v2) | Agents/RL | 提出用于图流模型的在线强化学习框架，解析推导转移概率实现可微 rollout，并引入局部扰动重生成策略，在分子优化任务上达到最先进性能。It proposes an online RL framework for graph flow models that analytically derives transition probabilities for differentiable rollouts and introduces a local perturbation-regeneration strategy, achieving state-of-the-art performance on molecular optimization. |
| [Generalization in Online Reinforcement Learning for Mobile Agents](https://arxiv.org/abs/2603.07432v1) | Agents/RL | 提出移动 GUI Agent 在线强化学习的泛化基准 AndroidWorld-Generalization，并开源完整 RL 训练系统，揭示对未见实例、模板和应用的泛化挑战。It proposes AndroidWorld-Generalization, a benchmark for generalization in online RL for mobile GUI agents, and open-sources the full RL training system, revealing challenges in generalizing to unseen instances, templates, and apps. |

### 系统与高效训练 / Systems

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [HIERAMP: Coarse-to-Fine Autoregressive Amplification for Generative Dataset Distillation](https://arxiv.org/abs/2603.06932v1) | Systems | 提出面向生成式数据集蒸馏的粗到细自回归放大方法，在不同 VAR 尺度注入类别 token 引导关注判别性区域，持续提升蒸馏基准性能。It proposes a coarse-to-fine autoregressive amplification method for generative dataset distillation that injects class tokens at different VAR scales to guide focus on discriminative regions, consistently improving distillation benchmarks. |
| [FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](https://arxiv.org/abs/2603.04890v1) | Systems | 提出多模态联邦学习统一框架，通过双层对抗对齐、粒度感知融合和相似性引导集成蒸馏处理模态差异与模型异构性。It proposes a unified multimodal federated learning framework that addresses modality gaps and model heterogeneity via bi-level adversarial alignment, granularity-aware fusion, and similarity-guided ensemble distillation. |
| [Training Flow Matching: The Role of Weighting and Parameterization](https://arxiv.org/abs/2603.06454v2) | Systems | 通过系统数值研究探讨去噪生成模型中损失加权和输出参数化与数据流形内在维度、模型架构及数据集大小的交互作用。It conducts a systematic numerical study of loss weighting and output parameterization in denoising generative models and their interaction with data-manifold intrinsic dimensionality, architecture, and dataset size. |
| [The Coupling Within: Flow Matching via Distilled Normalizing Flows](https://arxiv.org/abs/2603.09014v1) | Systems | 提出归一化流匹配方法，利用预训练归一化流的可逆双射蒸馏自适应耦合来训练流模型，性能优于独立耦合和最优传输耦合。It proposes Normalized Flow Matching, which distills adaptive couplings from pretrained invertible normalizing flows to train flow models, outperforming independent and optimal-transport couplings. |

### 机器人学与具身智能 / Robotics & Embodied AI

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [MetaWorld-X: Hierarchical World Modeling via VLM-Orchestrated Experts for Humanoid Loco-Manipulation](https://arxiv.org/abs/2603.08572v1) | Robotics | 提出分层世界模型框架，通过 VLM 监督的智能路由机制组合专门化专家策略，实现人形机器人 loco-manipulation 的组合泛化。It proposes a hierarchical world-model framework that composes specialized expert policies via a VLM-supervised intelligent routing mechanism, enabling compositional generalization for humanoid loco-manipulation. |
| [PlayWorld: Learning Robot World Models from Autonomous Play](https://arxiv.org/abs/2603.09030v3) | Robotics | 提出从机器人自主玩耍中训练高保真视频世界模拟器的可扩展流程，无需人类演示即可捕捉复杂长尾物理交互，并在真实世界 RL 中提升 65% 成功率。It presents a scalable pipeline for training high-fidelity video world simulators from autonomous robot play, capturing complex long-tail physical interactions without human demonstrations and improving real-world RL success rates by 65%. |
| [AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots](https://arxiv.org/abs/2603.07648v1) | Robotics | 提出统一规划-执行框架，通过技能引导的 MoE 构建可扩展原子技能库并实现持续学习，在长程任务和持续学习中显著超越基线。It proposes a unified planning-and-execution framework that builds a scalable atomic skill library via a skill-guided MoE and enables continual learning, significantly outperforming baselines on long-horizon tasks and continual learning. |

### 机器学习理论 / ML Theory

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [CompassDPO: Dynamics-Controlled Direct Preference Optimization for Robust Safety Alignment](https://arxiv.org/abs/2603.07211v2) | ML Theory | 提出动态控制直接偏好优化框架，通过调节更新方向与幅度提升对噪声偏好标注的鲁棒性，在 PKU-SafeRLHF 上持续改进安全性。It proposes a dynamics-controlled DPO framework that improves robustness to noisy preference labels by regulating update direction and magnitude, consistently improving safety on PKU-SafeRLHF. |
| [The Norm-Separation Delay Law of Grokking: A First-Principles Theory of Delayed Generalization](https://arxiv.org/abs/2603.13331v2) | ML Theory | 建立 Grokking 延迟的“范数分离延迟定律”，将延迟泛化重新解释为规范驱动的表示相变，并在 293 次训练运行中得到验证。It establishes the Norm-Separation Delay Law for grokking, reframing delayed generalization as a norm-driven representational phase transition and validating it across 293 training runs. |

### 其他交叉方向 / Other

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark](https://arxiv.org/abs/2603.05075v1) | Other | 推出首个统一任意到任意交错多模态基准，涵盖 7 种模态和 30 个领域，并配套评估套件与 Agent 基线，推动统一多模态理解与生成的研究。It introduces the first unified any-to-any interleaved multimodal benchmark spanning 7 modalities and 30 domains, along with an evaluation suite and agentic baseline, advancing research in unified multimodal understanding and generation. |

---

## 领域趋势 / Trends

1. **Agentic AI 的工程化 / Agentic AI engineering** — 从单点工具到感知-推理-决策-执行的端到端系统，金融、搜索、科研、机器人等领域同时出现四层架构范式。
2. **推理成本的主动控制 / Active reasoning-cost control** — SPOT、Ares、SmartThinker 等工作在 token 级或步骤级动态决定“想多少”，在压缩 37–53% 推理开销的同时保持或提升准确率。
3. **视觉-语言-动作模型的场景自适应 / VLA scene adaptation** — SAMoE-VLA、DyQ-VLA、AtomicVLA 用 MoE、量化和原子技能让 VLA 在自动驾驶与机器人长程任务中更可扩展。
4. **世界模型的“视野外”一致性 / Out-of-sight world-model consistency** — LiveWorld 与 PlayWorld 关注物体离开视野后状态是否持续演化，这是具身智能可信部署的关键瓶颈。
5. **CoT 的可信性审计 / CoT trustworthiness audit** — C2-Faith、Safer Reasoning Traces 从因果忠实度、隐私泄漏、幻觉检测三个维度建立评测，让“思维链”不只是性能工具。
6. **小模型的专业化后训练 / Specialized post-training for small models** — Tiny Aya 通过区域化后训练在 3.35B 参数上实现 70 语言深度覆盖，提示“小模型+专业化”是高部署价值场景的可行路线。

---

## PhD 阅读路线图 / PhD Reading Roadmap

### 阶段一：建立本周全景（30 分钟）
- 通读本 README 的 Weekly Highlights 与 Trends，明确 3 个核心叙事：Agentic AI 工程化、推理效率控制、VLA/世界模型。

### 阶段二：精读重点论文（2–3 小时）
1. **Tiny Aya**：聚焦“全球平衡 + 区域后训练”的训练策略，思考低资源语言与小容量的 trade-off。
2. **AI Agents in Financial Markets**：从四层架构出发，画出自己研究领域的 Agent 架构草图。
3. **Autonomous Factor Investing**：结合 [PyTorch 复现](autonomous-factor/code/autonomous_factor_pytorch.py) 理解 LLM 因子假设 → 自动回测 → 自适应权重的闭环。

### 阶段三：按兴趣深入子领域（各 1–2 小时）
- **推理效率**：SPOT → Ares → SmartThinker（由潜在推理到动态 effort 选择）。
- **VLA/机器人**：AtomicVLA → SAMoE-VLA → PlayWorld（从技能学习到世界模型）。
- **可信 CoT**：C2-Faith → Safer Reasoning Traces → HALP（跨文本与视觉）。
- **RL 理论**：BandPO → NePPO → Graph-GRPO（信任区域、多智能体、图生成）。

### 阶段四：动手实验（2–4 小时）
- 运行 [autonomous_factor_pytorch.py](autonomous-factor/code/autonomous_factor_pytorch.py)，尝试修改因子库或权重衰减窗口，观察 Sharpe 与有效因子数变化。
- 用本周论文中 1–2 个开源基准（如 AndroidWorld-Generalization、MultiHaystack）快速跑通官方 demo，建立直觉。

---

## 重要时间节点 / Key Dates

| 事件 / Event | 日期 / Date |
|---|---|
| AAAI 2026 Conference | Feb–Mar 2026 |
| ICLR 2026 Camera Ready | Mar 2026 |
| NeurIPS 2026 Submissions Open | Apr 2026 |

---

## 数据来源 / Sources

- arXiv API (`export.arxiv.org`)，查询类别 cs.CV / cs.LG / cs.AI / cs.CL / stat.ML，提交日期范围 `2026-03-05` 至 `2026-03-11`。
- 本仓库 Week 10 原始 featured reviews 由先前会话整理，本次补充完整分类列表、趋势、路线图与 PyTorch 复现代码。
