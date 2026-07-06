# Week 17: 2026-04-23 — 2026-04-29 / 第 17 周

> **本周范围 / Scope:** 2026-04-23 (Thu) — 2026-04-29 (Wed)  
> **本周主题 / Theme:** 长上下文证据利用、可控世界模拟器、视觉-触觉世界模型与参数高效可解释性。

---

## 本周学术动态 / Weekly Highlights

- **WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory**  
  提出将语义运动编排与视觉生成解耦的视频世界模型，并引入持久动态对象记忆，实现长时间遮挡后仍保持对象身份一致。  
  A video world model that decouples semantic motion orchestration from visual generation and introduces persistent dynamic object memory, preserving object identity after long occlusions.

- **ReContext: Recursive Evidence Replay as LLM Harness for Long-Context Reasoning**  
  免训练地利用模型内部相关性信号构建查询条件化证据池，并在生成前递归回放证据，在八个长上下文基准上取得最佳平均排名。  
  A training-free method that builds a query-conditioned evidence pool from internal relevance signals and recursively replays it before generation, achieving the best average rank across eight long-context benchmarks.

- **VT-WAM: Visual-Tactile World Action Model for Contact-Rich Manipulation**  
  将视觉与触觉观测统一为世界行动模型，用于接触丰富的机器人操作，使策略能够预测接触力并泛化到新物体。  
  Unifies visual and tactile observations into a world action model for contact-rich manipulation, enabling policies to predict contact forces and generalize to novel objects.

- **Expander Sparse Autoencoders: Parameter-Efficient Dictionaries for Mechanistic Interpretability**  
  用左 d-正则扩展图掩码替代密集 SAE 解码器，将学习值减少 293 倍，同时保留 84% 的密集交叉熵损失恢复能力。  
  Replaces dense SAE decoders with left-d-regular expander-graph masks, reducing learned values by 293× while retaining 84% of dense CE-loss recovery.

- **Distributed Attacks in Persistent-State AI Control**  
  研究在多智能体持续状态控制场景下的分布式对抗攻击，揭示当前 LLM 智能体在状态记忆与协调机制上的脆弱性。  
  Studies distributed adversarial attacks in persistent-state multi-agent control, revealing vulnerabilities in state memory and coordination mechanisms of current LLM agents.

- **Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots** *(开源项目发布 / Open-source release)*  
  发布面向异构机器人平台的轻量级嵌入式推理运行时，支持 VLA 模型在 CPU/GPU/NPU 上低延迟部署。  
  Releases a lightweight inference runtime for heterogeneous robot platforms, enabling low-latency deployment of VLA models on CPU/GPU/NPU.

- **Beyond Adam: SOAP and Muon for Faster, Label-Efficient Training of Machine Learning Interatomic Potentials**  
  比较 SOAP 与 Muon 等新型优化器在训练机器学习原子势时的收敛速度与标注效率，为科学机器学习优化器选择提供实证依据。  
  Compares SOAP and Muon optimizers for training machine-learning interatomic potentials, providing empirical guidance on optimizer choice for scientific ML.

- **Program-as-Weights: A Programming Paradigm for Fuzzy Functions**  
  将程序编码为神经网络权重，实现可微分的模糊函数编程，为神经符号推理提供新的表达形式。  
  Encodes programs as neural network weights to enable differentiable programming of fuzzy functions, offering a new representation for neuro-symbolic reasoning.

---

## 重点论文深度分析 / Featured Papers with Deep Reviews

| 论文 / Paper | 领域 / Area | 深度评论 / Deep Review | 代码 / Code |
|---|---|---|---|
| ReContext: Recursive Evidence Replay as LLM Harness for Long-Context Reasoning | NLP & LLM | [review.md](recontext/review.md) | — |
| WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory | CV | [review.md](worlddirector/review.md) | — |
| Expander Sparse Autoencoders: Parameter-Efficient Dictionaries for Mechanistic Interpretability | ML Theory | [review.md](expander-sparse-autoencoders/review.md) | [expander_sae.py](expander-sparse-autoencoders/code/expander_sae.py) |

---

## 本周论文列表（按领域分类）/ Weekly Papers by Area

### CV / 计算机视觉

- **WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory** — [arXiv:2604.02517](https://arxiv.org/abs/2604.02517)  
  可控世界模拟器，持久动态记忆，语义运动与视觉生成解耦。
- **PointDiT: Pixel-Space Diffusion for Monocular Geometry Estimation** — [arXiv:2604.02515](https://arxiv.org/abs/2604.02515)  
  像素空间扩散模型用于单目几何估计。
- **VT-WAM: Visual-Tactile World Action Model for Contact-Rich Manipulation** — [arXiv:2604.02503](https://arxiv.org/abs/2604.02503)  
  视觉-触觉世界行动模型，面向接触丰富操作。
- **DisciplineGen-1M: A Large-Scale Dataset for Multidisciplinary Visual Generation and Editing** — [arXiv:2604.02290](https://arxiv.org/abs/2604.02290)  
  多学科视觉生成与编辑的大规模数据集。

### NLP & LLM / 自然语言处理与大语言模型

- **ReContext: Recursive Evidence Replay as LLM Harness for Long-Context Reasoning** — [arXiv:2604.02509](https://arxiv.org/abs/2604.02509)  
  递归证据回放，提升长上下文推理证据利用率。
- **Online Safety Monitoring for LLMs** — [arXiv:2604.02510](https://arxiv.org/abs/2604.02510)  
  在线安全监控框架。
- **LACUNA: A Testbed for Evaluating Localization Precision for LLM Unlearning** — [arXiv:2604.02513](https://arxiv.org/abs/2604.02513)  
  评估 LLM 遗忘定位精度的测试平台。
- **Visually Grounded Self-Reflection for Vision-Language Models via Reinforcement Learning** — [arXiv:2604.02490](https://arxiv.org/abs/2604.02490)  
  基于强化学习的视觉基础自反思，用于视觉-语言模型。

### ML Theory / 机器学习理论

- **Expander Sparse Autoencoders: Parameter-Efficient Dictionaries for Mechanistic Interpretability** — [arXiv:2604.01799](https://arxiv.org/abs/2604.01799)  
  扩展图稀疏自编码器，参数高效可解释性字典。
- **Beyond Adam: SOAP and Muon for Faster, Label-Efficient Training of Machine Learning Interatomic Potentials** — [arXiv:2604.02499](https://arxiv.org/abs/2604.02499)  
  超越 Adam 的新型优化器在原子势训练中的实证比较。
- **Generalization in Offline RL: The Structure is More Important than the Amount of Pessimism** — [arXiv:2604.02288](https://arxiv.org/abs/2604.02288)  
  离线强化学习泛化：结构比悲观程度更重要。
- **An Optimisation Framework for the Well-Conditioned Training of Physics-Informed Neural Networks** — [arXiv:2604.02194](https://arxiv.org/abs/2604.02194)  
  物理信息神经网络良态训练的优化框架。

### Systems / 系统

- **Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots** — [arXiv:2604.02501](https://arxiv.org/abs/2604.02501)  
  面向异构机器人的具身 AI 可移植推理运行时。
- **Efficient PEFT Methods with Adaptive Checkpointing for Vision Models and VLMs on Resource Constrained Consumer-GPUs** — [arXiv:2604.02158](https://arxiv.org/abs/2604.02158)  
  消费级 GPU 上自适应检查点的高效 PEFT 方法。
- **WBMM: Windowed Batch Matrix Multiplication for Efficient Large Receptive Field Convolution** — [arXiv:2604.02097](https://arxiv.org/abs/2604.02097)  
  窗口化批矩阵乘法，实现大感受野卷积。
- **Privacy-Preserving and Verifiable Approximate Distributed Coded Computing** — [arXiv:2604.02187](https://arxiv.org/abs/2604.02187)  
  隐私保护且可验证的近似分布式编码计算。

### Agents & RL / 智能体与强化学习

- **Distributed Attacks in Persistent-State AI Control** — [arXiv:2604.02514](https://arxiv.org/abs/2604.02514)  
  持续状态 AI 控制中的分布式攻击。
- **EvoPolicyGym: Evaluating Autonomous Policy Evolution in Interactive Environments** — [arXiv:2604.02440](https://arxiv.org/abs/2604.02440)  
  交互式环境中自主策略进化的评估平台。
- **DecompRL: Solving Harder Problems by Learning Modular Code Generation** — [arXiv:2604.02390](https://arxiv.org/abs/2604.02390)  
  通过学习模块化代码生成解决更难问题。
- **Rank-Then-Act: Reward-Free Control from Frame-Order Progress** — [arXiv:2604.01897](https://arxiv.org/abs/2604.01897)  
  无需奖励的帧顺序推进控制。

### Robotics / 机器人学

- **VT-WAM: Visual-Tactile World Action Model for Contact-Rich Manipulation** — [arXiv:2604.02503](https://arxiv.org/abs/2604.02503)  
  视觉-触觉世界行动模型。
- **WorldSample: Closed-loop Real-robot RL with World Modelling** — [arXiv:2604.02431](https://arxiv.org/abs/2604.02431)  
  基于世界模型的真实机器人闭环强化学习。
- **HEFT: Heavy-Payload Full-size Humanoid Teleoperation with Privileged Motion Guidance** — [arXiv:2604.02332](https://arxiv.org/abs/2604.02332)  
  重载全尺寸人形机器人遥操作。
- **ACID: Action Consistency via Inverse Dynamics for Planning with World Models** — [arXiv:2604.02403](https://arxiv.org/abs/2604.02403)  
  利用逆动力学实现与世界模型规划的动作一致性。

---

## 趋势观察 / Trends

**1. 从“更长上下文”到“更会使用证据” / From longer contexts to better evidence usage**

长上下文模型已经能够容纳 128K 甚至更多 token，但关键问题正从“能装多少”转向“能用多少”。ReContext 等工作尝试用模型内部信号重新组织证据，使生成过程更高效地利用已有上下文。未来半年，预计将出现更多免训练的上下文重排序与证据回放方法，并与检索增强生成（RAG）形成互补。

LLMs can now hold 128K tokens or more, but the key question is shifting from "how much can fit" to "how much can be used." Works like ReContext reorganize evidence using internal signals, helping generation better exploit the available context. In the next six months, expect more training-free context reranking and evidence-replay methods that complement retrieval-augmented generation (RAG).

**2. 可控世界模拟器进入“对象级持久记忆”阶段 / Controllable world simulators move to object-level persistent memory**

WorldDirector 等研究将运动编排与视觉生成分离，并引入持久动态记忆，使视频世界模型在遮挡、重入等复杂情境下保持一致性。这标志着世界模型正从“生成合理视频”向“支持长期可控交互”演进，有望为机器人仿真与影视内容创作提供统一基础设施。

WorldDirector decouples motion orchestration from visual generation and introduces persistent dynamic memory, allowing video world models to remain consistent through occlusions and re-entries. This signals a shift from merely generating plausible video toward enabling long-term controllable interaction, with implications for robot simulation and cinematic content creation.

**3. 视觉-触觉联合建模成为机器人操作新基线 / Visual-tactile joint modeling becomes a new baseline for robot manipulation**

VT-WAM 将视觉与触觉统一为单一世界行动模型，使机器人策略能够预测接触力并泛化到新物体。随着低成本触觉传感器和 VLA 模型的发展，视觉-触觉融合预计将在接触丰富的装配、抓取与灵巧操作中成为标准配置。

VT-WAM unifies vision and touch into a single world action model, letting policies predict contact forces and generalize to novel objects. With cheaper tactile sensors and VLA models, visual-tactile fusion is poised to become standard for contact-rich assembly, grasping, and dexterous manipulation.

**4. 参数高效的 mechanistic interpretability 走向实用 / Parameter-efficient mechanistic interpretability becomes practical**

Expander SAE 通过扩展图掩码将稀疏自编码器的解码器参数压缩两个数量级，同时保留大部分解释能力。这表明“可解释性”不必以巨大存储为代价，未来可能出现面向大模型的轻量级特征字典库，支持在线编辑与安全审计。

Expander SAEs compress decoder parameters by two orders of magnitude using expander-graph masks while retaining most explanatory power. This shows interpretability need not come at huge storage cost, and lightweight feature dictionaries for large models may soon enable online editing and safety auditing.

---

## 博士生本周阅读路线建议 / PhD Reading Roadmap

**通用建议 / General advice**

先浏览本周亮点，选择与你研究方向最接近的 2–3 篇论文精读；对核心方法进行小规模复现；将论文中的评估指标与你自己的基准对齐。

Skim the weekly highlights, pick 2–3 papers closest to your research, read them in depth, reproduce the core method at small scale, and align their evaluation metrics with your own benchmarks.

**CV / 视频生成与世界模型 / Video generation & world models**

精读 WorldDirector，重点理解“持久动态记忆”与“运动-生成分解”；若资源允许，尝试用简单 2D 场景复现对象身份保持机制。

Read WorldDirector carefully, focusing on persistent dynamic memory and motion-generation decoupling. If resources allow, try reproducing object-identity preservation in a simple 2D scene.

**NLP & LLM / 长上下文与推理 / Long context & reasoning**

精读 ReContext，实现递归证据回放的最小版本；对比其与 prompt compression、RAG 在相同长文本 QA 数据集上的效果。

Read ReContext and implement a minimal recursive evidence-replay loop; compare it with prompt compression and RAG on the same long-document QA dataset.

**ML Theory / 可解释性与优化 / Interpretability & optimization**

精读 Expander SAE 并在小型 transformer 或合成数据上复现；同时阅读 SOAP/Muon 优化器论文，思考其在科学机器学习任务中的适用性。

Read Expander SAE and reproduce it on a small transformer or synthetic data; also read the SOAP/Muon optimizer paper and consider its applicability to scientific ML tasks.

**Systems / 高效推理与训练 / Efficient inference & training**

关注 Embodied.cpp 与 WBMM，理解端侧/机器人推理的优化点；尝试将一个 VLA 模型用 ONNX/TensorRT 或 vLLM/SGLang 部署到本地 GPU，记录吞吐与延迟。

Focus on Embodied.cpp and WBMM to understand edge/robot inference optimizations; try deploying a VLA model with ONNX/TensorRT or vLLM/SGLang on a local GPU and record throughput and latency.

**Agents & RL / 智能体安全与代码生成 / Agent safety & code generation**

精读 Distributed Attacks in Persistent-State AI Control 与 DecompRL，思考多智能体状态的对抗脆弱性，以及模块化代码生成对 RL 策略结构的启发。

Read Distributed Attacks in Persistent-State AI Control and DecompRL, reflecting on adversarial vulnerabilities of multi-agent state and the implications of modular code generation for RL policy structures.

**Robotics / 视觉-触觉与真实机器人学习 / Visuotactile & real-robot learning**

精读 VT-WAM 与 WorldSample，理解视觉-触觉世界模型如何支持闭环真实机器人 RL；若有触觉设备，可收集小数据集训练简单抓取策略。

Read VT-WAM and WorldSample to understand how visuotactile world models support closed-loop real-robot RL; if you have a tactile sensor, collect a small dataset and train a simple grasping policy.

---

## 重要时间节点 / Key Dates

| 会议 / Conference | 事件 / Event | 日期 / Date |
|---|---|---|
| ICLR 2026 | 主会 / Main conference | 2026-04-23 — 2026-04-25 |
| ICLR 2026 | 研讨会 / Workshops | 2026-04-26 — 2026-04-27 |
| CVPR 2026 | 早鸟注册截止 / Early registration deadline | 2026-04-23 |
| CVPR 2026 | 主会 / Main conference | June 8–14, 2026 |
| CVPR 2026 | 研讨会与教程 / Workshops & tutorials | 2026-06-03 — 2026-06-04 |
| ICML 2026 | 论文接收通知 / Author notification | 2026-04-30 |
| ICML 2026 | 主会 / Main conference | July 6–11, 2026 |
| ICML 2026 | 研讨会 / Workshops | 2026-07-10 — 2026-07-11 |
| NeurIPS 2026 | 摘要截止 / Abstract deadline | 2026-05-04 (AoE) |
| NeurIPS 2026 | 全文截止 / Full paper deadline | 2026-05-06 (AoE) |
| NeurIPS 2026 | 研讨会申请截止 / Workshop application deadline | 2026-06-06 (AoE) |
| NeurIPS 2026 | 论文接收通知 / Author notification | 2026-09-24 |
| NeurIPS 2026 | 主会 / Main conference | 2026-12-08 — 2026-12-10 |

---

## 数据来源 / Sources

- arXiv 分类订阅 / arXiv category feeds:  
  - [cs.CV - Computer Vision and Pattern Recognition](https://arxiv.org/list/cs.CV/recent)  
  - [cs.LG - Machine Learning](https://arxiv.org/list/cs.LG/recent)  
  - [cs.CL - Computation and Language](https://arxiv.org/list/cs.CL/recent)  
  - [cs.AI - Artificial Intelligence](https://arxiv.org/list/cs.AI/recent)  
  - [cs.RO - Robotics](https://arxiv.org/list/cs.RO/recent)  
  - [cs.IT - Information Theory](https://arxiv.org/list/cs.IT/recent)  
  - [cs.DC - Distributed, Parallel, and Cluster Computing](https://arxiv.org/list/cs.DC/recent)
- [NeurIPS 2026 Dates and Deadlines](https://neurips.cc/Conferences/2026/Dates)
- [CVPR 2026 Dates](https://cvpr.thecvf.com/Conferences/2026/Dates)
- [ICML 2026 Dates](https://icml.cc/Conferences/2026/Dates)
- [ICLR 2026 Dates](https://iclr.cc/Conferences/2026/Dates)
- [SGLang GitHub Repository](https://github.com/sgl-project/sglang)
- [vLLM vs SGLang 2026 Benchmarks](https://techsy.io/en/blog/vllm-vs-sglang)

---

*本简报由双语 AI 研究策展人自动生成，论文标题与摘要基于 arXiv 最新投稿并映射至 2026 年第 17 周。*/
