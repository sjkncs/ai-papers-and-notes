# Week 25: 2026-06-18 — 2026-06-24 / 第25周

---

## 本周学术动态 / Weekly Highlights

**中文：** 2026 年第 25 周是 ICML 2026 预印本预览与 CVPR 2026 会后 arXiv 释放叠加的高峰期。仅 6 月 18–19 日两天，cs.CV / cs.LG / cs.AI / cs.CL / stat.ML 五个类别就新增约 480 篇论文。本周核心亮点包括：
1. **GRPO 的理论解密**：ICML 2026 论文 *GRPO is Secretly a Process Reward Model* 严格证明 GRPO 等价于隐式蒙特卡洛 PRM，并提出 λ-GRPO 改进；
2. **Test-time 语义探索**：ICML 2026 论文 *Large Language Models Explore by Latent Distilling* 提出 ESamp，用浅层到深层的预测误差作为新颖性信号引导解码，AIME25 Pass@16 提升 1.4 个百分点；
3. **长期战略 Agent 评测**：*CEO-Bench* 模拟企业 CEO 决策，揭示当前 SOTA Agent 在长期动态环境中的严重缺陷；
4. **视觉 AR 生成的空间并行突破**：*SSD* 提出空间投机解码，把自回归图像生成的理论复杂度从 O(n²) 降到 O(n)，最高 13.3× 加速；
5. **世界模型持久状态诊断**：*Current World Models Lack a Persistent State Core* 发布 WRBench，发现当前模型在目标离开视野后不会继续演化其状态；
6. **推理效率的 token 级解耦**：*ADaPT* 用 `<think>` / `<answer>` mode-selection token 将效率与正确性信号解耦，实现连续可控的快慢思考。

**English:** Week 25 of 2026 is a superposition of the ICML 2026 preprint preview peak and the post-CVPR 2026 arXiv release wave. In just two days (June 18–19), approximately 480 new papers appeared across cs.CV, cs.LG, cs.AI, cs.CL, and stat.ML. Key highlights:
1. **Theoretical decryption of GRPO**: the ICML 2026 paper *GRPO is Secretly a Process Reward Model* rigorously proves GRPO is equivalent to an implicit Monte-Carlo PRM and proposes λ-GRPO;
2. **Test-time semantic exploration**: the ICML 2026 paper *Large Language Models Explore by Latent Distilling* proposes ESamp, using shallow-to-deep prediction error as a novelty signal to guide decoding, improving AIME25 Pass@16 by 1.4 points;
3. **Long-horizon strategic agent evaluation**: *CEO-Bench* simulates corporate CEO decision-making and reveals severe deficits in current SOTA agents in dynamic long-term environments;
4. **Spatial-parallel breakthrough for visual AR generation**: *SSD* proposes spatial speculative decoding, reducing the theoretical complexity of autoregressive image generation from O(n²) to O(n) with up to 13.3× speedup;
5. **Persistent-state diagnosis for world models**: *Current World Models Lack a Persistent State Core* introduces WRBench and finds that current models do not evolve object states while unobserved;
6. **Token-level decoupling for reasoning efficiency**: *ADaPT* uses `<think>` / `<answer>` mode-selection tokens to decouple efficiency and correctness signals, enabling continuously controllable fast/slow thinking.

---

## 重点论文深度分析 / Featured Papers with Deep Reviews

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [GRPO is Secretly a Process Reward Model](https://arxiv.org/abs/2509.21154) | ML Theory / RL | [review.md](grpo-is-prm/review.md) | — |
| 2 | [Large Language Models Explore by Latent Distilling](https://arxiv.org/abs/2604.24927) | LLM Decoding / Test-Time Compute | [review.md](llm-explore-by-latent-distilling/review.md) | [code/](llm-explore-by-latent-distilling/code/) |
| 3 | [CEO-Bench: Can Agents Play the Long Game?](https://arxiv.org/abs/2606.18543) | Agents / Evaluation | [review.md](ceo-bench/review.md) | — |
| 4 | [SSD: Spatially Speculative Decoding Accelerates Autoregressive Image Generation](https://arxiv.org/abs/2606.20543) | CV / Systems | [review.md](ssd-spatially-speculative-decoding/review.md) | [code/](ssd-spatially-speculative-decoding/code/) |
| 5 | [Current World Models Lack a Persistent State Core](https://arxiv.org/abs/2606.20545) | CV / World Models | [review.md](current-world-models-persistent-state/review.md) | — |
| 6 | [ADaPT: Token-Level Decoupling for Efficient Large Reasoning Models](https://arxiv.org/abs/2606.19919) | NLP & LLM / Efficient Reasoning | [review.md](adapt-token-level-decoupling/review.md) | — |

---

## 本周论文列表（按领域分类）/ Weekly Papers by Area

> 本周 arXiv 五个主要类别新增约 480 篇论文（announcement dates: 18–19 Jun 2026），叠加 ICML 2026 预印本释放。以下按领域精选代表性工作，并附简短要点。

> This week saw approximately 480 new papers across five main arXiv categories (announcement dates: 18–19 Jun 2026),叠加 the ICML 2026 preprint release. Representative works by area are listed below with brief key points.

### 机器学习理论 / ML Theory

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **[GRPO is Secretly a Process Reward Model](https://arxiv.org/abs/2509.21154)** | ICML 2026。严格证明 GRPO 等价于隐式蒙特卡洛 PRM，提出 λ-GRPO 解决步骤频率不平衡。Qwen-1.5B 准确率从 48.37% 提升至 55.76%。 |
| **What Characterizes Effective Reasoning? Revisiting CoT** | ICML 2026。重新审视思维链长度、回顾和结构对推理效果的影响。 |
| **Position: We Need A Unified Definition of Hallucination** | ICML 2026。主张幻觉的本质是世界模型缺陷。 |
| **Low-dimensional Topology of Deep Neural Networks** | ICML 2026。深度神经网络的低维拓扑结构分析。 |
| **Optimal Deterministic Multicalibration and Omniprediction** | arXiv。最优确定性多校准与全能预测。 |
| **On the Oracle Complexity of Interpolation-Based Gradient Descent** | arXiv。基于插值的梯度下降 oracle 复杂度。 |
| **Quantile of Means: A Bonus-Free Ensemble Method for Minimax Optimal Reinforcement Learning** | arXiv。均值分位数：无 bonus 的集成方法达到极小极大最优 RL。 |
| **Fisher-Geometric Sharpness and the Implicit Bias of SGD toward Flat Minima** | arXiv。Fisher-几何锐度与 SGD 向平坦极小值的隐式偏置。 |
| **Sparsity, Superposition, and Forgetting: A Mechanistic Study of Representation Retention in Continual Learning** | arXiv。持续学习表示保留的机制研究。 |

### 自然语言处理与大模型 / NLP & LLM

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **[Large Language Models Explore by Latent Distilling](https://arxiv.org/abs/2604.24927)** | ICML 2026。ESamp 通过潜空间蒸馏预测误差引导语义探索，AIME25 Pass@16 达 31.7%，开销 <5%。 |
| **[ADaPT: Token-Level Decoupling for Efficient Large Reasoning Models](https://arxiv.org/abs/2606.19919)** | token 级快慢思考解耦，推理时连续控制效率-性能权衡。 |
| **GraphPO: Graph-based Policy Optimization for Reasoning Models** | 将推理路径建模为 DAG，语义等价路径共享后缀。Qwen2.5-7B 数学基准达 40.9%。 |
| **Beyond Reward Engineering: A Data Recipe for Long-Context RL** | 14K 示例极简 GRPO 提升长上下文推理，Qwen3 平均提升 +7.2/+3.2/+6.4 点。 |
| **Multi-Task Bayesian In-Context Learning** | 多任务贝叶斯上下文学习框架。 |
| **UltraQuant: 4-bit KV Caching for Context-Heavy Agents** | 面向长上下文 Agent 的 4-bit KV Cache 量化。 |
| **VIMPO: Value-Implicit Policy Optimization for LLMs** | 隐式值策略优化用于 LLM。 |
| **Manifold Bandits: Bayesian Curriculum Learning over the Latent Geometry of LLMs** | 流形 bandit 做 LLM 潜在几何上的课程学习。 |
| **Algebraic Dead Directions in LayerNorm Transformers** | LayerNorm Transformer 中的代数死亡方向。 |
| **Concept Flow Models: Anchoring Concept-Based Reasoning with Hierarchical Bottlenecks** | 概念流模型：用层次瓶颈锚定概念推理。 |

### 智能体与强化学习 / Agents & RL

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **[CEO-Bench: Can Agents Play the Long Game?](https://arxiv.org/abs/2606.18543)** | CEO 战略决策模拟评测，仅 Claude Opus 4.8 和 GPT-5.5 能保持不亏损。 |
| **MetaResearcher: Scaling Deep Research via Self-Reflective Reinforcement Learning** | 对抗虚拟环境中的自反思 RL 扩展深度研究 Agent。 |
| **ScaffoldAgent: Utility-Guided Dynamic Outline Optimization for Open-Ended Deep Research** | 效用引导的动态大纲优化用于开放式深度研究。 |
| **Process-Verified Reinforcement Learning for Theorem Proving via Lean** | 通过 Lean 进行过程验证的定理证明 RL。 |
| **AgentArmor: A Framework, Evaluation, & Mitigation of Coding Agent Failures** | 编码 Agent 故障的框架、评测与缓解。 |
| **LedgerAgent: Structured State for Policy-Adherent Tool-Calling Agents** | 结构化状态保证工具调用 Agent 遵循策略。 |
| **Sovereign Execution Brokers: Enforcing Certificate-Bound Authority in Agentic Control Planes** | 在 Agent 控制平面强制执行证书绑定权限。 |
| **Benchmarking Agentic Review Systems** | Agentic 审稿系统基准。 |
| **Decoupling Search from Reasoning: Vendor-Agnostic Grounding for LLM Agents** | 搜索与推理解耦的 LLM Agent 架构。 |
| **VISUALSKILL: Multimodal Skills for Computer-Use Agents** | 多模态技能赋能电脑使用 agent。 |

### 计算机视觉 / Computer Vision

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **[SSD: Spatially Speculative Decoding Accelerates Autoregressive Image Generation](https://arxiv.org/abs/2606.20543)** | 空间投机解码，AR 图像生成最高 13.3× 加速。 |
| **[Current World Models Lack a Persistent State Core](https://arxiv.org/abs/2606.20545)** | WRBench 诊断世界模型持久状态核心，揭示目标离屏后状态不演化。 |
| **JanusMesh: Fast and Zero-Shot 3D Visual Illusion Generation via Cross-Space Denoising** | ECCV 2026。跨空间去噪实现零样本 3D 视觉错觉生成。 |
| **End-to-End Training for Unified Tokenization and Latent Denoising (UNITE)** | ICML 2026。统一 tokenization 和潜在去噪的端到端训练，ImageNet FID 2.12/1.73。 |
| **Self-Supervised Flow Matching (Self-Flow)** | ICML 2026。自监督流匹配范式，将表示学习融入生成框架。 |
| **TimeProVe: Propose, then Verify for Efficient Long Video Temporal Reasoning** | 长视频时序推理：先提议再验证。 |
| **Thinking in Boxes: 3D Editing in Real Images Made Easy** | 真实图像 3D 编辑。 |
| **HumanScale: Egocentric Human Video Can Outperform Real-Robot Data for Embodied Pretraining** | 第一人称人体视频可超越真机数据做具身预训练。 |
| **The FID Lottery: Quantifying Hidden Randomness in Generative-Model Evaluation** | 量化生成模型评估中 FID 的隐藏随机性。 |
| **HiGS: Hierarchical Rendering for Real-Time 3D Gaussian Splatting** | 分层实时 3DGS 渲染架构。 |

### 系统与高效推理 / Systems

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **JetFlow: Breaking the Scaling Ceiling of Speculative Decoding** | 并行树状起草突破推测解码扩展瓶颈。 |
| **Execution-State Capsules: Graph-Bound Execution-State Checkpoint and Restore** | 物理 AI 端侧低延迟服务的执行状态胶囊。 |
| **StreamKL: Fast and Memory-Efficient KL Divergence for Boosting Attention Distillation** | 快速内存高效的 KL 散度用于注意力蒸馏。 |
| **LOKI: Memory-Free Null-Space Constrained Lifelong Knowledge Editing** | 零内存零空间约束的终身知识编辑。 |
| **CacheWeaver: Cache-Aware Evidence Ordering for Efficient Grounded RAG Inference** | 缓存感知的证据排序用于高效 Grounded RAG。 |
| **Quantum ring all-reduce: communication and privacy advantages for distributed learning** | 量子环 all-reduce 的通信与隐私优势。 |
| **Beyond Prediction: Tail-Aware Scheduling for LLM Inference** | 尾部感知 LLM 推理调度。 |

### 机器人学与具身智能 / Robotics & Embodied AI

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **ENPIRE: Agentic Robot Policy Self-Improvement in the Real World** | 真实世界中的 Agentic 机器人策略自改进。 |
| **Reward as An Agent for Embodied World Models** | 将奖励建模为具身世界模型中的 Agent。 |
| **HumanScale: Egocentric Human Video Can Outperform Real-Robot Data for Embodied Pretraining** | 第一人称人体视频可超越真机数据做具身预训练。 |
| **Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation** | 频率感知的流匹配用于连续一致机器人动作生成。 |
| **FlowMaps: Modeling Long-Term Multimodal Object Dynamics with Flow Matching** | 用流匹配建模长期多模态物体动力学。 |
| **Data Standards for Humanoid Robotics: The Missing Infrastructure for Physical AI** | 人形机器人数据标准：物理 AI 缺失的基础设施。 |
| **Physical Atari: A Robust and Accessible Platform for Real-time RL on Robots** | 物理 Atari：机器人实时 RL 的鲁棒可及平台。 |

### 多模态 / Multimodal

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **NAVI-Orbital: Zero-Shot VLM for Autonomous Earth Observation** | 首个在轨零样本视觉语言模型自主地球观测。 |
| **Continuous Audio Thinking for Large Audio Language Models** | 大型音频语言模型的连续音频思维。 |
| **Hyper-ICL: Hyperbolic Anchor Distillation for Multimodal ICL** | ICML 2026。双曲锚点蒸馏提升多模态上下文学习。 |

---

## 趋势观察 / Trends

**中文：**

1. **GRPO 的理论解密时代来临 / The Era of GRPO Theoretical Decryption Arrives**
   *GRPO is Secretly a PRM* 证明 GRPO 隐式实现了 PRM，这意味着 GRPO 的上限可能受限于其蒙特卡洛 PRM 估计质量，而显式 PRM 与 GRPO 的选择将变成“隐式估计质量 vs. 显式标注精度”的技术判断。

2. **Test-Time Compute 的“探索-利用”新维度 / New Explore-Exploit Dimension for Test-Time Compute**
   ESamp 与 ADaPT 共同指向一个趋势：test-time scaling 不再只是“生成更多候选并挑选”，而是“在生成过程中动态决定思考深度”和“引导模型探索语义上未见的区域”。

3. **Agent 评测从任务完成走向长期生存 / Agent Evaluation Shifts from Task Completion to Long-Term Survival**
   CEO-Bench 与本周大量 Agent 评测论文说明，社区正从“能否完成这个任务”转向“能否在动态环境中持续做出有利决策”。

4. **视觉 AR 生成进入空间并行时代 / Visual AR Generation Enters the Spatial-Parallel Era**
   SSD 把自回归视觉生成从 1D raster-scan 转向 2D 空间投机解码。如果与 VAR/MAR 等原生 2D 表示结合，扩散模型在图像生成中的主导地位可能真正受到挑战。

5. **世界模型从“看起来像”走向“演得像” / World Models Move from Looking Right to Evolving Right**
   WRBench 把“持久状态核心”推上议程，世界模型评测将从帧级 fidelity 转向状态级因果一致性，催生更多显式状态记忆与物理引擎耦合的工作。

**English:**

1. **The Era of GRPO Theoretical Decryption Arrives**
   *GRPO is Secretly a PRM* proves GRPO implicitly implements a PRM, implying GRPO's ceiling may be limited by Monte-Carlo PRM estimation quality and that the choice between explicit PRM and GRPO becomes a technical judgment of implicit estimation quality vs. explicit annotation precision.

2. **New Explore-Exploit Dimension for Test-Time Compute**
   ESamp and ADaPT together point to a trend: test-time scaling is no longer just "generate more candidates and select," but "dynamically decide thinking depth during generation" and "guide the model toward semantically unexplored regions."

3. **Agent Evaluation Shifts from Task Completion to Long-Term Survival**
   CEO-Bench and this week's many agent-evaluation papers show the community moving from "can it complete this task?" to "can it continuously make favorable decisions in dynamic environments?"

4. **Visual AR Generation Enters the Spatial-Parallel Era**
   SSD shifts autoregressive visual generation from 1D raster-scan to 2D spatial speculative decoding. Combined with native 2D representations such as VAR/MAR, diffusion models' dominance in image generation may be genuinely challenged.

5. **World Models Move from Looking Right to Evolving Right**
   WRBench puts the persistent-state core on the agenda, shifting world-model evaluation from frame-level fidelity to state-level causal consistency and inspiring more work on explicit state memory and physics-engine coupling.

---

## 博士生本周阅读路线建议 / PhD Reading Roadmap for This Week

**中文：**

**如果你做 RL / LLM 后训练 / If you work on RL or LLM post-training:**
> 精读 *GRPO is Secretly a PRM* → 手推 Theorem 1 → 在自有 GRPO 代码中实现 λ-GRPO → 对比原版 GRPO → 思考：你的任务中哪些步骤是高频/低频的？

**如果你做 Test-Time Compute / 解码 / If you work on test-time compute or decoding:**
> 精读 *Large Language Models Explore by Latent Distilling* → 运行本仓库中的 [esamp.py](llm-explore-by-latent-distilling/code/) → 将 ESamp 集成到你的推理框架中 → 在自有任务上测试 β 敏感度；同时浏览 ADaPT 的 mode-selection 思路。

**如果你做 Agent 评测 / 规划 / If you work on agent evaluation or planning:**
> 精读 *CEO-Bench* → 分析其环境设计抽象方法 → 思考：你的领域是否有类似需要“长期适应”的任务？→ 延伸阅读: GAIA, AgentBench。

**如果你做视觉生成 / If you work on visual generation:**
> 精读 *SSD*（含本仓库 [ssd.py](ssd-spatially-speculative-decoding/code/ssd.py) 复现代码）→ 理解水平/垂直 draft 与隐藏状态蒸馏 → 思考：你的 AR 生成任务能否从空间投机中受益？

**如果你做世界模型 / 具身智能 / If you work on world models or embodied AI:**
> 精读 *Current World Models Lack a Persistent State Core* → 理解 WRBench 的三个评测问题 → 思考：你的生成模型是否也只是“跟踪镜头”？

**本周必读（全领域）/ Must-read this week (All areas):**
1. *GRPO is Secretly a Process Reward Model* — 理论优雅，可能改变你对 GRPO 的认知。
2. *Large Language Models Explore by Latent Distilling* — 有代码复现，test-time 方向的创意之作。
3. *SSD* — 有代码复现，视觉 AR 生成的前沿加速思路。
4. *Current World Models Lack a Persistent State Core* — 范式级问题定义，可能重塑世界模型评测。

---

## 重要时间节点 / Key Dates

| 事件 / Event | 日期 / Date | 状态 / Status |
|---|---|---|
| CVPR 2026 Conference | Jun 8–15, 2026 | 已结束 / Concluded |
| CVPR 2026 Awards | Jun 2026 | 已公布 / Announced |
| ICML 2026 Conference | Jul 2026 | 即将召开 / Upcoming |
| NeurIPS 2026 Review Period | Jun–Jul 2026 | 审稿中 / Under review |
| ECCV 2026 Review Period / Final Decisions | Jun 2026 | 陆续放出 / Rolling out |
| AAAI 2027 Submission Deadline | Aug 2026 | 准备中 / Preparation |

---

## 数据来源 / Sources

- [ICML 2026 Papers](https://icml.cc/virtual/2026/papers.html)
- [ICML 2026 Paper Digest Highlights](https://www.paperdigest.org/2026/05/icml-2026-papers-highlights/)
- [CVPR 2026 Best Papers](https://cvpr.thecvf.com/Conferences/2026/News/Best_Papers)
- [arXiv cs.CV Recent](https://arxiv.org/list/cs.CV/recent)
- [arXiv cs.LG Recent](https://arxiv.org/list/cs.LG/recent)
- [arXiv cs.AI Recent](https://arxiv.org/list/cs.AI/recent)
- [arXiv cs.CL Recent](https://arxiv.org/list/cs.CL/recent)
- [arXiv stat.ML Recent](https://arxiv.org/list/stat.ML/recent)
- [Hugging Face Daily Papers](https://huggingface.co/papers)
