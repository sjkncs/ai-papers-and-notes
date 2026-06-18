# Week 4: 2026-01-22 — 2026-01-28 / 第4周

---

## 本周学术动态 / Weekly Highlights

**中文：** 2026年第4周是ICLR 2026接受论文通知周（1月25日），大量高质量预印本集中释放。本周核心亮点包括：(1) ICLR 2026理论突破——TROLL用可微信任区域投影替代PPO clipping，在训练速度和稳定性上全面超越标准PPO；(2) Deep Ignorance揭示预训练数据筛选可构建比后训练方法强一个数量级的防篡改安全屏障，且仅消耗不到1%的总训练FLOPS；(3) Google Research发现推理模型的能力提升并非来自"算得更久"，而是来自内部"思维社会"的多智能体模拟；(4) LongRLVR证明长上下文RL需要可验证的上下文奖励，14B模型RULER-QA从73.17提升至88.90；(5) AAAI 2026收到创纪录投稿量，会议即将于2-3月召开。arXiv方面，推理模型 societies of thought、Kolmogorov复杂度与深度学习桥接、以及扩散模型理论成为热点。

**English:** Week 4 of 2026 was ICLR 2026 acceptance notification week (Jan 25), with a concentrated release of high-quality preprints. Key highlights: (1) ICLR 2026 theoretical breakthrough — TROLL replaces PPO clipping with differentiable trust region projection, comprehensively surpassing standard PPO in training speed and stability; (2) Deep Ignorance reveals that pretraining data filtering can build tamper-resistant safety barriers over an order of magnitude stronger than post-training methods, consuming less than 1% of total training FLOPS; (3) Google Research finds that reasoning models' capability improvements come not from "computing longer" but from internal "societies of thought" multi-agent simulation; (4) LongRLVR proves long-context RL requires verifiable context rewards, boosting 14B model RULER-QA from 73.17 to 88.90; (5) AAAI 2026 received record submissions, conference upcoming in Feb–Mar. On arXiv, reasoning model societies of thought, bridging Kolmogorov complexity and deep learning, and diffusion model theory are trending.

---

## 重点论文深度分析 / Featured Papers with Deep Reviews

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [TROLL: Trust Regions improve RL for LLMs](https://arxiv.org/abs/2510.03817) | RL Theory | [review.md](troll-trust-regions/review.md) | [code/](troll-trust-regions/code/) |
| 2 | [Deep Ignorance](https://arxiv.org/abs/2508.06601) | AI Safety | [review.md](deep-ignorance/review.md) | — |
| 3 | [Reasoning Models Generate Societies of Thought](https://arxiv.org/abs/2601.10825) | LLM Reasoning | [review.md](societies-of-thought/review.md) | — |

---

## 本周论文列表（按领域分类）/ Weekly Papers by Area

### 机器学习理论 / ML Theory

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **TROLL: Trust Regions improve RL for Large Language Models** | ICLR 2026。离散可微信任区域投影替代PPO clipping，token级KL约束，训练速度和稳定性一致超越PPO。 |
| **Bridging Kolmogorov Complexity and Deep Learning** | ICLR 2026。基于Kolmogorov复杂度为Transformer提出渐近最优描述长度目标，标准优化器无法找到此类解。 |
| **From REINFORCE to Dr. GRPO: A Unified Perspective on LLM Post-Training** | ICLR 2026。统一视角比较REINFORCE到GRPO的梯度估计技术 |
| **Tracing the Principles Behind Modern Diffusion Models** | ICLR 2026。探索扩散模型的理论基础 |
| **Flow Straight and Fast in Hilbert Space: Functional Rectified Flow** | ICLR 2026。Hilbert空间中的函数式整流流，去除限制性测度论假设 |
| **Low-dimensional Topology of Deep Neural Networks** | ICLR 2026。深度神经网络的低维拓扑结构 |
| **Pre-training Under Infinite Compute** | ICLR 2026。无限计算下的扩展律，最优权重衰减比标准实践大30倍 |
| **Riemannian Federated Learning via Averaging Gradient Streams** | ICLR 2026。几何方法用于联邦学习 |

### 大语言模型与推理 / LLM & Reasoning

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **Reasoning Models Generate Societies of Thought** | Google Research。推理模型能力提升来自内部多智能体模拟而非单纯计算时间，对话式支架加速推理。 |
| **LongRLVR: Long-Context RL Requires Verifiable Context Rewards** | ICLR 2026。长上下文RL需要可验证上下文奖励，14B模型RULER-QA从73.17→88.90，LongBench v2从39.8→46.5。 |
| **JustRL: Scaling a 1.5B LLM with a Simple RL Recipe** | ICLR 2026。极简RL配方扩展1.5B模型 |
| **WebArbiter: A Generative Reasoning Process Reward Model for Web Agents** | ICLR 2026。WebArbiter-7B超越GPT-5达9.1点，轨迹搜索超越最佳WebPRM 6.4点。 |
| **On the Eligibility of LLMs for Counterfactual Reasoning** | ICLR 2026。分解式研究LLM处理反事实推理的能力 |
| **Ice Cream Doesn't Cause Drowning: Benchmarking LLMs Against Statistical Pitfalls in Causal Inference** | ICLR 2026。测试LLM在因果推断中的统计陷阱 |
| **Safety at One Shot: Patching Fine-Tuned LLMs with A Single Instance** | ICLR 2026。单实例快速修正微调LLM的不安全行为 |
| **What Characterizes Effective Reasoning? Revisiting CoT** | ICLR 2026。重新审视CoT长度、回顾和结构 |
| **Reliability-Aware LLM Alignment from Inconsistent Human Feedback** | ICLR 2026。从不一致人类反馈中进行可靠性感知对齐 |

### 智能体与系统 / Agents & Systems

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **AgentGym-RL: Open-Source Framework for Long-Horizon Agent RL** | ICLR 2026。多轮RL训练LLM Agent的开源框架 |
| **Breaking Barriers: Do RL Post-Training Gains Transfer To Unseen Domains?** | ICLR 2026。RL后训练收益是否迁移到未见领域 |
| **The Ideation-Execution Gap: Human ideas score higher than LLM ideas** | ICLR 2026。人类想法在质量上优于LLM想法 |
| **RoboCasa365: Comprehensive Robot Simulation Benchmark** | ICLR 2026。日常任务机器人模拟综合基准 |
| **SNaX: Accelerated Mixture-of-Experts via Joint Algorithm-Kernel Design** | ICLR 2026。联合算法-内核设计加速MoE |

### 计算机视觉 / Computer Vision

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **SAM 3: Segment Anything with Concepts** | ICLR 2026 Highlight。统一检测、分割和跟踪。 |
| **SANA-Video: Efficient Video Generation with Linear Transformers** | ICLR 2026。线性Transformer高效生成720×1280分钟级视频。 |
| **Lyra: 3D Reconstruction via Self-Distillation** | ICLR 2026。自蒸馏将扩散知识转化为显式3DGS表征。 |
| **OmniWorld: Multi-Domain Multi-Modal Dataset for 4D World Modeling** | ICLR 2026。多样化4D世界建模数据集 |
| **LikePhys: Evaluating Intuitive Physics in Video Diffusion** | ICLR 2026。通过似然偏好评估视频扩散模型的物理直觉 |

### 安全与隐私 / Safety & Privacy

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards** | ICLR 2026。预训练数据筛选防篡改，6.9B模型套件开源，抵抗力超后训练方法一个数量级。 |
| **Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards** | ICLR 2026。预训练数据筛选构建防篡改安全屏障 |
| **Mechanistic Anomaly Detection via Functional Attribution** | ICML 2026(early)。可解释异常检测 |

### 多模态与音频 / Multimodal & Audio

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **Kimi-Dev: Software Engineering Agent via Agentless Methods** | ICLR 2026。无agent方法训练软件工程agent，SWE-bench Verified达60.4%。 |
| **Multi-state Protein Sequence Design with DynamicMPNN** | ICLR 2026。动态MPNN用于蛋白质序列设计 |
| **Discrete Audio Tokens: More Than a Survey!** | ICLR 2026。音频信号的离散表示 |

---

## 趋势观察 / Trends

**中文：**

1. **PPO Clipping的终结者出现 / The End of PPO Clipping?**
   TROLL以理论优雅的信任区域投影直接替代了PPO使用近8年的clipping启发式。如果这一方法在更大规模上得到验证，可能会引发LLM RL训练范式的更新潮。值得关注的是，TROLL与GRPO等无critic方法的兼容性——如果GRPO+TROLL组合被证明有效，将是reasoning模型训练的重大升级。

2. **"预防优于治疗"的AI安全新范式 / "Prevention Over Cure" in AI Safety**
   Deep Ignorance代表了一种根本性的安全思维转变：从"训练后加固"转向"训练前预防"。这一思路与疫苗接种的逻辑相似——与其在感染后治疗，不如在感染前建立免疫。但dual-use知识的定义难题仍是悬而未决的伦理挑战。

3. **推理即社会模拟 / Reasoning as Social Simulation**
   Google的"Societies of Thought"发现暗示，人类引以为傲的推理能力可能只是大脑中多个子系统"协商"的结果。如果这一机制被充分理解和工程化，未来可能会出现显式多角色架构的推理模型，其性能可能远超单角色长思考链。

4. **ICLR 2026的"去炒作化"信号 / ICLR 2026's "De-Hype" Signals**
   多个高影响力工作（如Ideation-Execution Gap发现人类想法优于LLM、Pre-training Under Infinite Compute质疑标准扩展律）表明顶级会议开始更重视对现有范式的批判性反思，而非单纯的性能提升。

5. **Agent评测的多样化爆发 / Diversified Explosion in Agent Evaluation**
   本周出现了WebArbiter（Web Agent PRM）、AgentGym-RL（多轮RL框架）、RoboCasa365（日常机器人任务）等多个agent基础设施工作，标志着agent评测正在从单一任务完成度向多维度能力评估演进。

**English:**

1. **The End of PPO Clipping?**
   TROLL directly replaces the clipping heuristic that PPO has used for nearly 8 years with a theoretically elegant trust region projection. If validated at larger scales, this could trigger a wave of updates to LLM RL training paradigms. Notably, TROLL's compatibility with critic-free methods like GRPO — if GRPO+TROLL is proven effective, it would be a major upgrade for reasoning model training.

2. **"Prevention Over Cure" in AI Safety**
   Deep Ignorance represents a fundamental shift in safety thinking: from "post-training fortification" to "pre-training prevention." This logic resembles vaccination — rather than treating after infection, build immunity before infection. But the definitional challenge of dual-use knowledge remains an unresolved ethical problem.

3. **Reasoning as Social Simulation**
   Google's "Societies of Thought" discovery implies that human reasoning — our proudest cognitive capability — may simply be the result of "negotiation" among multiple subsystems in the brain. If this mechanism is fully understood and engineered, future models with explicit multi-role architectures may far outperform single-role long-thought-chains.

4. **ICLR 2026's "De-Hype" Signals**
   Multiple high-impact works (e.g., Ideation-Execution Gap finding human ideas superior to LLMs, Pre-training Under Infinite Compute questioning standard scaling laws) suggest top-tier conferences are beginning to value critical reflection on existing paradigms more than pure performance gains.

5. **Diversified Explosion in Agent Evaluation**
   This week saw multiple agent infrastructure works including WebArbiter (Web Agent PRM), AgentGym-RL (multi-turn RL framework), and RoboCasa365 (daily robot tasks), marking agent evaluation's evolution from single-task completion to multi-dimensional capability assessment.

---

## 博士生本周阅读路线建议 / PhD Reading Roadmap for This Week

**中文：**

**如果你做RL/LLM后训练 / If you work on RL or LLM post-training:**
> 精读 TROLL (ICLR 2026) → 运行本仓库中的 [troll_ppo.py](troll-trust-regions/code/troll_ppo.py) → 在自有GRPO代码中替换clipping为TROLL投影 → 对比训练稳定性

**如果你做AI安全/对齐 / If you work on AI safety or alignment:**
> 精读 Deep Ignorance (ICLR 2026) → 分析其多阶段筛选流水线设计 → 思考：你的场景中dual-use知识的边界在哪里？→ 延伸阅读: Circuit-Breakers, Sleeper Agents

**如果你做LLM推理/思维链 / If you work on LLM reasoning or CoT:**
> 精读 Societies of Thought (Google, Jan 2026) → 理解机制可解释性分析方法 → 思考：你的推理模型是否也展现多视角内部结构？→ 延伸阅读: Multi-Agent Debate, Self-Consistency

**如果你做扩散模型/生成模型 / If you work on diffusion or generative models:**
> 精读 Flow Straight and Fast in Hilbert Space (ICLR 2026) 或 Tracing Principles of Diffusion (ICLR 2026) → 理解生成模型的理论前沿

**本周必读 (All areas) / Must-read this week (All areas):**
1. TROLL — 有代码复现，可能改变你对PPO的认知
2. Deep Ignorance — 安全范式转变的代表作
3. Reasoning Models Generate Societies of Thought — Google的推理机制新发现

---

## 重要时间节点 / Key Dates

| 事件 / Event | 日期 / Date | 状态 / Status |
|---|---|---|
| ICLR 2026 Notification | Jan 25, 2026 | 已通知 / Notified |
| AAAI 2026 Conference | Feb–Mar 2026 | 即将召开 / Upcoming |
| NeurIPS 2026 CFP | Early 2026 | 已发布 / Published |
| ICML 2026 Submission Deadline | Jan/Feb 2026 (已截止 / Closed) | 审稿中 / Under review |

---

## 数据来源 / Sources

- [ICLR 2026 Papers](https://iclr.cc/virtual/2026/papers.html)
- [ICLR 2026 Paper Digest Highlights](https://www.paperdigest.org/2026/02/iclr-2026-papers-highlights/)
- [ICLR 2026 Accepted Paper List - Paper Copilot](https://papercopilot.com/paper-list/iclr-paper-list/iclr-2026-paper-list/)
- [arXiv cs.AI](https://arxiv.org/list/cs.AI/recent)
- [arXiv cs.LG](https://arxiv.org/list/cs.LG/recent)
- [Hugging Face Daily Papers](https://huggingface.co/papers)
