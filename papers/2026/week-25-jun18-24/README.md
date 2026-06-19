# Week 25: 2026-06-17 — 2026-06-23 / 第25周

---

## 本周学术动态 / Weekly Highlights

**中文：** 2026年第25周正值ICML 2026论文预览高峰期（会议将于7月召开），大量接受论文的预印本开始在arXiv上密集释放。本周核心亮点包括：(1) ICML 2026理论突破——GRPO被严格证明等价于隐式过程奖励模型(PRM)，并提出λ-GRPO改进；(2) ESamp解码方法通过潜空间蒸馏实现test-time语义探索，在AIME25上将Pass@16提升1.4个百分点；(3) CEO-Bench揭示当前SOTA agent在长期战略决策中的严重缺陷；(4) 数据驱动的长上下文RL新范式——仅需14K示例和极简GRPO即可显著提升长上下文推理；(5) GraphPO将推理路径建模为DAG，在Qwen2.5-7B上平均准确率达40.9%。arXiv方面，JetFlow突破推测解码的扩展瓶颈、Sumi开源扩散语言模型、以及多模态agent评测密集出现。

**English:** Week 25 of 2026 was the peak preview period for ICML 2026 (conference in July), with accepted papers densely releasing preprints on arXiv. Key highlights: (1) ICML 2026 theoretical breakthrough — GRPO rigorously proven equivalent to an implicit Process Reward Model (PRM), with λ-GRPO improvement proposed; (2) ESamp decoding achieves test-time semantic exploration via latent distilling, improving Pass@16 on AIME25 by 1.4 points; (3) CEO-Bench reveals severe deficiencies in current SOTA agents' long-term strategic decision-making; (4) Data-driven long-context RL paradigm — only 14K examples with minimal GRPO substantially improves long-context reasoning; (5) GraphPO models reasoning paths as DAGs, achieving 40.9% average accuracy on Qwen2.5-7B. On arXiv, JetFlow breaks speculative decoding scaling ceilings, Sumi open-source diffusion language model, and multimodal agent benchmarks densely appear.

---

## 重点论文深度分析 / Featured Papers with Deep Reviews

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [GRPO is Secretly a Process Reward Model](https://arxiv.org/abs/2509.21154) | RL Theory | [review.md](grpo-is-prm/review.md) | — |
| 2 | [Large Language Models Explore by Latent Distilling](https://arxiv.org/abs/2604.24927) | LLM Decoding | [review.md](llm-explore-by-latent-distilling/review.md) | [code/](llm-explore-by-latent-distilling/code/) |
| 3 | [CEO-Bench: Can Agents Play the Long Game?](https://arxiv.org/abs/2606.18543) | Agents Eval | [review.md](ceo-bench/review.md) | — |

---

## 本周论文列表（按领域分类）/ Weekly Papers by Area

### 机器学习理论 / ML Theory

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **GRPO is Secretly a Process Reward Model** | ICML 2026。严格证明GRPO等价于隐式蒙特卡洛PRM，提出λ-GRPO解决步骤频率不平衡。Qwen-1.5B准确率从48.37%提升至55.76%。 |
| **Large Language Models Explore by Latent Distilling** | ICML 2026。ESamp通过潜空间蒸馏预测误差引导语义探索，AIME25 Pass@16达31.7%，开销<5%。 |
| **What Characterizes Effective Reasoning? Revisiting CoT** | ICML 2026。重新审视思维链长度、回顾和结构对推理效果的影响 |
| **Position: We Need A Unified Definition of Hallucination** | ICML 2026。主张幻觉的本质是世界模型缺陷 |
| **Low-dimensional Topology of Deep Neural Networks** | ICML 2026。深度神经网络的低维拓扑结构分析 |
| **On the Residual Scaling of Looped Transformers** | arXiv。循环Transformer的稳定性与可迁移性 |
| **Effects of Sparsity and Superposition on Loss** | arXiv。稀疏性与叠加对简单自编码器损失的影响 |
| **A Link between Shock-wave Theory and SGD** | arXiv。激波理论与SGD的对称约化联系 |

### 大语言模型与推理 / LLM & Reasoning

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **GraphPO: Graph-based Policy Optimization for Reasoning Models** | arXiv。将推理路径建模为DAG，语义等价路径共享后缀。Qwen2.5-7B数学基准达40.9%。 |
| **Beyond Reward Engineering: A Data Recipe for Long-Context RL** | arXiv。14K示例极简GRPO提升长上下文推理，Qwen3平均提升+7.2/+3.2/+6.4点。 |
| **JetFlow: Breaking the Scaling Ceiling of Speculative Decoding** | arXiv。并行树状起草突破推测解码扩展瓶颈 |
| **Self-CTRL: Self-Consistency Training with RL** | arXiv。自一致性训练的强化学习方法 |
| **LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents** | arXiv。LLM Agent自动发现RL后训练策略 |
| **RegMix-D: Dynamic Data Mixing via Proxy Training Trajectories** | arXiv。基于代理训练轨迹的动态数据混合 |
| **Sumi: Open Uniform Diffusion Language Model from Scratch** | arXiv。从零训练的开源统一扩散语言模型 |
| **Good SFT Optimizes for SFT, Better SFT Prepares for RL** | ICML 2026。SFT与RL的关系再思考 |
| **Reliability-Aware LLM Alignment from Inconsistent Human Feedback** | ICML 2026。从不一致人类反馈中进行可靠性感知对齐 |

### 智能体与评测 / Agents & Evaluation

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **CEO-Bench: Can Agents Play the Long Game?** | arXiv。CEO战略决策模拟评测，仅Claude Opus 4.8和GPT-5.5能保持不亏损。 |
| **Decoupling Search from Reasoning: Vendor-Agnostic Grounding for LLM Agents** | arXiv。搜索与推理解耦的LLM Agent架构 |
| **Towards an Agent-First Web: Redesigning the Web for AI Agents** | arXiv。为AI Agent重新设计Web的愿景 |
| **CEO-Bench: Can Agents Play the Long Game?** | arXiv。CEO战略决策模拟评测 |
| **VISUALSKILL: Multimodal Skills for Computer-Use Agents** | arXiv。多模态技能赋能电脑使用agent |
| **Skill-Guided Continuation Distillation for GUI Agents** | arXiv。技能引导的GUI Agent连续蒸馏 |
| **SAGE: Stochastic Prompt Optimization via Agent-Guided Exploration** | arXiv。Agent引导探索的随机提示优化 |
| **CoreMem: Riemannian Retrieval for Long-Term Memory in Dialogue Agents** | arXiv。黎曼检索实现对话agent长期记忆 |
| **ProfiLLM: Utility-Aligned Agentic User Profiling** | arXiv。效用对齐的agent用户画像 |
| **Towards Scalable Customization of Multi-Agent Systems** | arXiv。多智能体系统的可扩展定制 |

### 计算机视觉 / Computer Vision

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **End-to-End Training for Unified Tokenization and Latent Denoising (UNITE)** | ICML 2026。统一tokenization和潜在去噪的端到端训练，ImageNet FID 2.12/1.73，成本比DINO方法低15倍。 |
| **Self-Supervised Flow Matching (Self-Flow)** | ICML 2026。自监督流匹配范式，将表示学习融入生成框架 |
| **Pixel MeanFlow (pMF)** | ICML 2026。像素均值流直接图像合成 |
| **Data-Forcing Distillation: Restoring Diversity in Few-Step Video Gen** | arXiv。数据强制蒸馏恢复少步视频生成多样性 |
| **UniTemp: Unlocking Video Generation in Any Temporal Order** | arXiv。双向蒸馏实现任意时序视频生成 |
| **MolmoMotion: Forecasting 3D Point Trajectories with Language** | arXiv。语言指令引导的3D点轨迹预测 |
| **SpectralDiT: Timestep-Conditioned Spectral Residual Correction** | arXiv。时间步条件谱残差校正流匹配DiT |
| **Real-Time Physics Simulation with Dynamic Mesh-Gaussian Reconstructions** | arXiv。动态网格-高斯重建实时物理模拟 |
| **HiGS: Hierarchical Rendering for Real-Time 3D Gaussian Splatting** | arXiv。分层实时3DGS渲染架构 |

### 多模态与系统 / Multimodal & Systems

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **NAVI-Orbital: Zero-Shot VLM for Autonomous Earth Observation** | arXiv。首个在轨零样本视觉语言模型自主地球观测 |
| **Beyond Prediction: Tail-Aware Scheduling for LLM Inference** | arXiv。尾部感知LLM推理调度 |
| **Continuous Audio Thinking for Large Audio Language Models** | arXiv。大型音频语言模型的连续音频思维 |
| **Hyper-ICL: Hyperbolic Anchor Distillation for Multimodal ICL** | ICML 2026。双曲锚点蒸馏提升多模态上下文学习 |
| **PragReST: Self-Reinforcing Counterfactual Reasoning for Pragmatic Language Understanding** | arXiv。自增强反事实推理用于语用语言理解 |

---

## 趋势观察 / Trends

**中文：**

1. **GRPO的理论解密时代来临 / The Era of GRPO Theoretical Decryption Arrives**
   本文证明GRPO隐式实现了PRM，这是一个颠覆性发现——社区一直以为GRPO"不需要过程标注"，但事实证明它只是以非平凡方式自动构建了过程监督。这意味着：(a) GRPO的上限可能受限于其蒙特卡洛PRM的估计质量；(b) 显式PRM和GRPO之间的选择不再是"标注成本 vs. 性能"的简单权衡，而是"隐式估计质量 vs. 显式标注精度"的技术判断。

2. **Test-Time Compute 的"探索-利用"新维度 / New "Explore-Exploit" Dimension for Test-Time Compute**
   ESamp揭示了test-time scaling的一个关键盲区：现有的方法（Best-of-N、MCTS、Speculative Decoding）都在"如何利用已有知识"上做文章，而ESamp首次系统性地解决了"如何在生成过程中发现新知识"。将Distiller的在线训练与主模型推理解耦的架构设计，对test-time系统方向的工程实践具有直接参考价值。

3. **Agent评测的"长期主义"转向 / The "Long-termism" Turn in Agent Evaluation**
   CEO-Bench与GAIA、SWE-bench等基准形成鲜明对照——后者问"agent能否完成这个任务"，前者问"agent能否在动态环境中持续做出有利决策"。这标志着agent评测从"任务完成度"向"生存适应性"的范式转移。如果这一方向得到扩展，可能会出现类似"Agent Turing Test"的长期生存挑战。

4. **数据配方（Data Recipe）取代奖励工程 / Data Recipes Replacing Reward Engineering**
   "Beyond Reward Engineering"和CEO-Bench共同指向一个趋势：RL训练的效果瓶颈可能不在奖励函数设计，而在训练数据的构造。14K精心筛选的示例就能超越复杂的奖励工程，这对RL研究的方法论有深远影响。

5. **推理路径的图结构化表示兴起 / Rise of Graph-Structured Reasoning Path Representations**
   GraphPO将推理过程表示为DAG，通过语义等价类合并和双组图优势估计优化策略。这代表了从"链式思维"到"图式思维"的重要演进，可能启发后续工作探索更复杂的推理拓扑（如循环推理、概率图推理）。

**English:**

1. **The Era of GRPO Theoretical Decryption Arrives**
   This paper proves GRPO implicitly implements PRM, a disruptive finding — the community had assumed GRPO "doesn't need process annotations," but in fact it automatically constructs process supervision in a non-trivial way. This means: (a) GRPO's ceiling may be limited by its Monte-Carlo PRM estimation quality; (b) the choice between explicit PRM and GRPO is no longer a simple "annotation cost vs. performance" trade-off, but a technical judgment of "implicit estimation quality vs. explicit annotation precision."

2. **New "Explore-Exploit" Dimension for Test-Time Compute**
   ESamp reveals a critical blind spot in test-time scaling: existing methods (Best-of-N, MCTS, Speculative Decoding) all focus on "how to leverage existing knowledge," while ESamp is the first to systematically address "how to discover new knowledge during generation." The architectural design decoupling Distiller's online training from main model inference has direct reference value for test-time systems engineering.

3. **The "Long-termism" Turn in Agent Evaluation**
   CEO-Bench contrasts sharply with GAIA, SWE-bench, and other benchmarks — the latter asks "can the agent complete this task," while the former asks "can the agent continuously make favorable decisions in a dynamic environment." This marks a paradigm shift in agent evaluation from "task completion" to "survival adaptability." If this direction expands, we may see long-term survival challenges akin to an "Agent Turing Test."

4. **Data Recipes Replacing Reward Engineering**
   "Beyond Reward Engineering" and CEO-Bench together point to a trend: RL training bottlenecks may lie not in reward function design but in training data construction. 14K carefully curated examples can surpass complex reward engineering, with profound methodological implications for RL research.

5. **Rise of Graph-Structured Reasoning Path Representations**
   GraphPO represents reasoning as a DAG, optimizing policy through semantic equivalence class merging and dual-group graph advantage estimation. This represents an important evolution from "chain-of-thought" to "graph-of-thought," potentially inspiring follow-up work exploring more complex reasoning topologies (e.g., cyclic reasoning, probabilistic graph reasoning).

---

## 博士生本周阅读路线建议 / PhD Reading Roadmap for This Week

**中文：**

**如果你做RL/LLM后训练 / If you work on RL or LLM post-training:**
> 精读 GRPO is Secretly a PRM (ICML 2026) → 手推 Theorem 1 → 在自有GRPO代码中实现λ-GRPO → 对比原版GRPO → 思考：你的任务中哪些步骤是高频/低频的？

**如果你做Test-Time Compute/解码 / If you work on test-time compute or decoding:**
> 精读 LLMs Explore by Latent Distilling (ICML 2026) → 运行本仓库中的 [esamp.py](llm-explore-by-latent-distilling/code/esamp.py) → 将ESamp集成到你的推理框架中 → 在自有任务上测试β敏感度

**如果你做Agent评测/规划 / If you work on agent evaluation or planning:**
> 精读 CEO-Bench → 分析其环境设计抽象方法 → 思考：你的领域是否有类似需要"长期适应"的任务？→ 延伸阅读: GAIA, AgentBench

**如果你做推理路径优化 / If you work on reasoning path optimization:**
> 精读 GraphPO (arXiv) → 对比链式/树式/图式推理的优势 → 思考：你的任务中是否存在语义等价路径合并的机会？

**本周必读 (All areas) / Must-read this week (All areas):**
1. GRPO is Secretly a Process Reward Model — 理论优雅，可能改变你对GRPO的认知
2. LLMs Explore by Latent Distilling — 有代码复现，test-time方向的创意之作
3. Beyond Reward Engineering: A Data Recipe for Long-Context RL — 数据-centric RL的新标杆

---

## 重要时间节点 / Key Dates

| 事件 / Event | 日期 / Date | 状态 / Status |
|---|---|---|
| ICML 2026 Conference | Jul 2026 | 即将召开 / Upcoming |
| NeurIPS 2026 Review Period | Jun–Jul 2026 | 审稿中 / Under review |
| ECCV 2026 Review Period | Jun–Jul 2026 | 审稿中 / Under review |
| AAAI 2027 Submission Deadline | Aug 2026 | 准备中 / Preparation |
| CVPR 2026 Code Release Wave | Jun 2026 | 进行中 / Ongoing |

---

## 数据来源 / Sources

- [ICML 2026 Papers](https://icml.cc/virtual/2026/papers.html)
- [ICML 2026 Paper Digest Highlights](https://www.paperdigest.org/2026/05/icml-2026-papers-highlights/)
- [arXiv cs.CV Recent](https://arxiv.org/list/cs.CV/recent)
- [arXiv cs.LG Recent](https://arxiv.org/list/cs.LG/recent)
- [arXiv cs.AI Recent](https://arxiv.org/list/cs.AI/recent)
- [arXiv cs.CL Recent](https://arxiv.org/list/cs.CL/recent)
- [Hugging Face Daily Papers](https://huggingface.co/papers)
