# Week 18: 2026-04-30 — 2026-05-06 / 第 18 周

**AI Top-Conference Paper Newsletter / AI 顶会论文周报**

This issue covers papers and announcements circulating on arXiv and at CVPR 2026 / ICML 2026 between April 30 and May 6, 2026. All arXiv IDs follow the 2604/2605 numbering convention for the 2026 posting season.

本期覆盖 2026 年 4 月 30 日至 5 月 6 日期间在 arXiv 以及 CVPR 2026 / ICML 2026 上出现的论文与公告。所有 arXiv ID 均遵循 2026 年投稿季的 2604/2605 编号规则。

---

## 本周学术动态 / Weekly Highlights

1. **NVIDIA 发布 Nemotron 3 Super（2604.12374）**：120B 总参数、12B 激活参数的 MoE+Mamba-Transformer 混合架构，支持 1M 长上下文与原生投机解码，面向 Agentic Reasoning 场景优化。
   - *NVIDIA releases Nemotron 3 Super: a 120B total / 12B active hybrid MoE+Mamba-Transformer with 1M context and native speculative decoding for agentic reasoning.*

2. **ZAYA1-8B（2605.05365）以 700M 激活参数叫板更大模型**：基于 AMD 基础设施训练，采用四阶段 RL 级联与 Markovian RSA 递归推理追踪。
   - *ZAYA1-8B challenges larger models with only 700M active parameters, trained on AMD infrastructure using a four-stage RL cascade and Markovian RSA reasoning traces.*

3. **Attention to Mamba（2604.14191）提出跨架构蒸馏范式**：在不引入混合注意力块的前提下，将 Transformer 教师知识迁移到纯 Mamba 学生模型。
   - *Attention to Mamba proposes a cross-architecture distillation recipe that transfers Transformer teachers into pure Mamba students without hybrid attention blocks.*

4. **《深度学习泛化理论》（2605.01172）给出无测试数据预测泛化误差的新路径**：通过经验神经正切核划分输出空间，区分信号与隐藏噪声。
   - *A Theory of Generalization in Deep Learning offers a way to predict generalization without test data by partitioning the output space via the empirical neural tangent kernel.*

5. **CVPR 2026 论文列表开放，像素级自监督与 3D 结构化隐变量成亮点**：Pixo、Native Structured Latents 与 VIRAL 等人形机器人工作入选 highlight。
   - *The CVPR 2026 program goes live, highlighting pixel-only self-supervised pre-training (Pixo), native 3D structured latents, and the VIRAL humanoid sim-to-real framework.*

6. **连续扩散语音语言模型 scaling law（2604.24416）**：16B 参数模型验证了扩散语音模型与自回归模型相似的可扩展性，但长程连贯性仍是瓶颈。
   - *Continuous Diffusion Spoken Language Models validate scaling laws up to 16B parameters, though long-form coherence remains a bottleneck.*

7. **边缘可信 Agent 执行系统 AgenTEE（2604.18231）开源**：利用 TEE 硬件隔离在边缘设备上运行 LLM Agent 流水线，性能开销低于 5.15%。
   - *AgenTEE open-sources confidential LLM-agent execution on edge devices using TEE hardware isolation with under 5.15% runtime overhead.*

8. **ICML 2026 将于 7 月在首尔举行，投稿量突破 30,800 篇**：多模态合成、Agent 验证器与端到端可学习分词成为 oral 热门方向。
   - *ICML 2026 in Seoul received over 30,800 submissions; oral themes include multimodal synthesis, agentic verifiers, and end-to-end learned tokenization.*

9. **CVPR 2026 Workshop 阵容公布，Agentic AI for Visual Media 与 Foundation Models Meet Embodied Agents 首次亮相。**
   - *CVPR 2026 workshop lineup is announced, debuting Agentic AI for Visual Media and Foundation Models Meet Embodied Agents.*

---

## 重点论文深度分析 / Featured Papers with Deep Reviews

| 论文 / Paper | 领域 / Area | 深度分析 / Deep Review | 代码 / Code |
|---|---|---|---|
| Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning | NLP & LLM / Systems | [nemotron-3-super/review.md](nemotron-3-super/review.md) | HuggingFace (official release) |
| Attention to Mamba: A Recipe for Cross-Architecture Distillation | NLP & LLM / ML Theory | [attention-to-mamba/review.md](attention-to-mamba/review.md) | [attention-to-mamba/code/attention_to_mamba_distillation.py](attention-to-mamba/code/attention_to_mamba_distillation.py) |
| A Theory of Generalization in Deep Learning | ML Theory | [theory-of-generalization/review.md](theory-of-generalization/review.md) | — |

---

## 本周论文列表（按领域分类）/ Weekly Papers by Area

### 计算机视觉 / Computer Vision (CV)

- **Pixo: In Pursuit of Pixel Supervision for Visual Pre-training** — CVPR 2026 Highlight. 完全通过像素预测训练自监督视觉模型，无需语义标签。/ *A purely pixel-prediction self-supervised visual model trained without semantic labels.*
- **Structure-Guided Diffusion Model for EEG-Based Visual Cognition Reconstruction** — [arXiv:2604.22649](https://arxiv.org/abs/2604.22649). 将结构信息显式注入扩散模型，从 EEG 信号重建视觉感知。/ *Injects explicit structural information into diffusion models to reconstruct visual perception from EEG signals.*
- **Beyond Pedestrians: Caption-Guided CLIP Framework for High-Difficulty Video-based Person Re-Identification** — [arXiv:2604.07740](https://arxiv.org/abs/2604.07740). 利用显式文本描述增强视频行人重识别。/ *Leverages explicit textual captions to improve video-based person re-identification.*
- **Native and Compact Structured Latents for 3D Generation** — CVPR 2026 Highlight. 从原生 3D 数据学习结构化隐变量表示，用于紧凑 3D 生成。/ *Learns structured latent representations from native 3D data for compact 3D generation.*
- **VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation** — CVPR 2026 Highlight. 在仿真中完全学习人形机器人移动-操作策略并迁移到真实世界。/ *Learns humanoid loco-manipulation entirely in simulation and transfers to the real world.*

### 自然语言处理与大模型 / NLP & LLM

- **Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning** — [arXiv:2604.12374](https://arxiv.org/abs/2604.12374). 120B/12B 激活参数 MoE+Mamba-Transformer，25T token 预训练，1M 上下文。/ *120B total / 12B active hybrid MoE+Mamba-Transformer pretrained on 25T tokens with 1M context.*
- **Attention to Mamba: A Recipe for Cross-Architecture Distillation** — [arXiv:2604.14191](https://arxiv.org/abs/2604.14191). 两阶段跨架构蒸馏，将 Pythia-1B Transformer 性能迁移到纯 Mamba。/ *Two-stage cross-architecture distillation preserving Pythia-1B Transformer performance in a pure Mamba student.*
- **ZAYA1-8B Technical Report** — [arXiv:2605.05365](https://arxiv.org/abs/2605.05365). 700M 激活 / 8B 总参数推理导向 MoE，基于 AMD 基础设施训练。/ *700M active / 8B total reasoning-focused MoE trained on AMD infrastructure.*
- **MambaVoiceCloning: Efficient and Expressive Text-to-Speech via State-Space Modeling and Diffusion Control** — [arXiv:2604.00292](https://arxiv.org/abs/2604.00292). 用 Mamba 替换扩散 TTS 中的注意力机制，实现 O(T) 条件编码。/ *Replaces attention in diffusion TTS with Mamba for O(T) conditioning and bounded memory.*
- **Scaling Properties of Continuous Diffusion Spoken Language Models** — [arXiv:2604.24416](https://arxiv.org/abs/2604.24416). 16B 参数连续扩散语音模型验证 scaling law，提出 pJSD 评估指标。/ *16B-parameter continuous diffusion speech model validates scaling laws and introduces the pJSD metric.*

### 机器学习理论 / ML Theory

- **A Theory of Generalization in Deep Learning** — [arXiv:2605.01172](https://arxiv.org/abs/2605.01172). 利用经验 NTK 划分输出空间，在无需测试数据的情况下估计泛化误差。/ *Estimates generalization error without test data by partitioning the output space via the empirical NTK.*
- **Robust Ultra Low-Bit Post-Training Quantization via Stable Diagonal Curvature Estimate** — [arXiv:2604.13806](https://arxiv.org/abs/2604.13806). DASH-Q 框架使用对角 Hessian 近似，将低比特 PTQ 的零样本精度平均提升 7.01%。/ *DASH-Q uses diagonal Hessian approximation to improve zero-shot accuracy of ultra low-bit PTQ by 7.01% on average.*
- **Symmetry in Language Statistics Shapes the Geometry of Model Representations** — [arXiv:2602.15029](https://arxiv.org/abs/2602.15029). 分析语言统计中的对称性如何影响模型表示的几何结构。/ *Analyzes how symmetry in language statistics shapes the geometry of learned representations.*
- **The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks** — [arXiv:2603.05498](https://arxiv.org/abs/2603.05498). 对 LLM 中的 massive activation 与 attention sink 进行系统解剖。/ *Provides a systematic anatomy of massive activations and attention sinks in LLMs.*

### 系统与效率 / Systems

- **AgenTEE: Confidential LLM Agent Execution on Edge Devices** — [arXiv:2604.18231](https://arxiv.org/abs/2604.18231). 在边缘 TEE 中部署保密 Agent 流水线，开销低于 5.15%。/ *Deploys confidential agent pipelines in edge TEEs with under 5.15% runtime overhead.*
- **ClawVM: Harness-Managed Virtual Memory for Stateful Tool-Using LLM Agents** — [arXiv:2604.10352](https://arxiv.org/abs/2604.10352). 为 Agent harness 引入虚拟内存层，实现确定性、可审计的状态管理。/ *Introduces a virtual-memory layer for agent harnesses to provide deterministic, auditable state management.*
- **DASH-Q: Robust Ultra Low-Bit Post-Training Quantization via Stable Diagonal Curvature Estimate** — [arXiv:2604.13806](https://arxiv.org/abs/2604.13806). 见 ML Theory。/ *See ML Theory section.*
- **MinerU2.5-Pro-2605-1.2B** — ModelScope 发布多模态文档解析模型。/ *ModelScope release of a multimodal document-parsing model.*

### 智能体与强化学习 / Agents & RL

- **Scaling Agentic Verifier for Competitive Coding** — ICML 2026 Oral. 基于执行的 Agent 在代码评估期间主动推理程序行为。/ *An execution-based agent that actively reasons about program behaviors during evaluation.*
- **Bringing Code ALIVE: Optimizing Interactive Frontend Mini-Games** — ICML 2026 Oral. ALIVE 框架通过交互式视觉执行对齐 LLM 生成的前端小游戏。/ *ALIVE aligns LLM-generated frontend mini-games via interactive visual execution.*
- **ClawVM** — [arXiv:2604.10352](https://arxiv.org/abs/2604.10352). 有状态工具调用 Agent 的状态管理层。/ *State-management layer for stateful tool-using agents.*
- **GLM-5: From Vibe Coding to Agentic Engineering** — [arXiv:2602.15763](https://arxiv.org/abs/2602.15763). 从氛围编程到 Agent 工程的系统框架。/ *A system framework moving from vibe coding to agentic engineering.*

### 机器人学 / Robotics

- **Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer** — CVPR 2026. 为人形机器人像素到动作策略迁移打开 Sim-to-Real 大门。/ *Opens the sim-to-real door for humanoid pixel-to-action policy transfer.*
- **VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation** — CVPR 2026 Highlight. 大规模视觉仿真到现实的人形移动-操作框架。/ *Large-scale visual sim-to-real framework for humanoid loco-manipulation.*
- **Accelerating Robotic Reinforcement Learning with Agent Guidance** — [arXiv:2602.11978](https://arxiv.org/abs/2602.11978). 利用 Agent 引导加速机器人强化学习。/ *Uses agent guidance to accelerate robotic reinforcement learning.*
- **Hybrid Framework for Robotic Manipulation** — [arXiv:2603.30022](https://arxiv.org/abs/2603.30022). 机器人混合操作框架。/ *A hybrid framework for robotic manipulation.*

---

## 趋势观察 / Trends

### 1. Mamba-Transformer 混合架构成为大模型新主流 / Mamba-Transformer Hybrids Become the New Mainstream

本周 Nemotron 3 Super 与 ZAYA1-8B 均采用了 Mamba 或 MoE+Mamba-Transformer 混合设计，表明线性注意力机制不再只是效率补充，而是与 Agentic Reasoning、长上下文场景深度结合的核心路径。厂商开始公开训练细节与量化方案，预示 2026 下半年将出现一波混合架构开源模型。

*This week both Nemotron 3 Super and ZAYA1-8B adopt Mamba or MoE+Mamba-Transformer hybrids, showing linear-attention mechanisms are no longer just efficiency supplements but core enablers for agentic reasoning and long-context scenarios. With vendors disclosing training details and quantization recipes, a wave of open hybrid-architecture models is expected in H2 2026.*

### 2. 跨架构蒸馏：从 Transformer 向 Mamba 的知识迁移 / Cross-Architecture Distillation: Moving Knowledge from Transformers to Mamba

Attention to Mamba 代表了一个新方向：如何在保留 Transformer 教师性能的同时，训练纯 Mamba 学生模型。该方法避免了昂贵的混合注意力块，为在边缘设备部署长序列模型提供了新思路，也可能催生针对 SSM 的专用压缩与微调工具链。

*Attention to Mamba signals a new direction: training pure Mamba students while preserving Transformer teacher performance without expensive hybrid attention blocks. This opens a path for deploying long-sequence models on edge devices and may spawn specialized compression and fine-tuning toolchains for SSMs.*

### 3. 深度学习理论开始“预测”泛化而非仅解释 / Deep Learning Theory Starts Predicting Generalization, Not Just Explaining It

A Theory of Generalization in Deep Learning 试图在不接触测试集的情况下估计泛化误差，通过经验 NTK 区分信号与噪声。这一思路若被验证，将改变模型选择与早停策略的实践方式，并可能为 AutoML 与神经架构搜索提供理论信号。

*A Theory of Generalization in Deep Learning aims to estimate generalization error without touching the test set, separating signal from noise via the empirical NTK. If validated, it could change model selection and early-stopping practices and supply theoretical signals for AutoML and neural architecture search.*

### 4. 可信与状态化 Agent 基础设施升温 / Trusted and Stateful Agent Infrastructure Heats Up

AgenTEE 与 ClawVM 分别从安全隔离与状态管理两个角度解决 LLM Agent 的工程痛点。随着 Agent 被接入企业工作流，对确定性、可审计、低延迟执行环境的需求将推动 TEE、虚拟内存与事务式状态层成为新的系统研究热点。

*AgenTEE and ClawVM address LLM-agent engineering pain points from security isolation and state management respectively. As agents are wired into enterprise workflows, demand for deterministic, auditable, low-latency execution environments will push TEEs, virtual-memory layers, and transactional state stores to the forefront of systems research.*

### 5. 视觉-语言-动作模型加速落地人形机器人 / Vision-Language-Action Models Accelerate Humanoid Deployment

CVPR 2026 的 VIRAL、DoorMan 等工作将视觉 sim-to-real 与人形移动-操作结合，表明 VLA 模型正从实验室 demo 走向真实场景。下半年预计会有更多端到端像素到动作策略在真实机器人平台上部署并开源。

*CVPR 2026 papers such as VIRAL and DoorMan combine visual sim-to-real with humanoid loco-manipulation, showing VLA models are moving from lab demos to real-world deployment. More end-to-end pixel-to-action policies on real robotic platforms are expected to be released in the second half of 2026.*

---

## 博士生本周阅读路线建议 / PhD Reading Roadmap

### 大模型与架构 / LLM & Architecture
- **必读 / Must-read**: Nemotron 3 Super（MoE+Mamba-Transformer 混合训练报告）、Attention to Mamba（跨架构蒸馏）。
- **延伸 / Follow-up**: MambaVoiceCloning、ZAYA1-8B、Scaling Properties of Continuous Diffusion SLM。
- **建议 / Advice**: 重点关注混合架构中的负载均衡、长上下文外推与投机解码实现细节；蒸馏方向可尝试复现 Attention to Mamba 在更小规模上的效果。

### 计算机视觉 / Computer Vision
- **必读 / Must-read**: Pixo、Native Structured Latents for 3D Generation、VIRAL。
- **延伸 / Follow-up**: SGDM、CG-CLIP、DoorMan。
- **建议 / Advice**: 视觉预训练正从对比学习回归生成式像素监督；3D 生成需要同时关注原生表示与神经渲染质量；机器人方向建议通读 sim-to-real 的域随机化与动作表示设计。

### 机器学习理论 / ML Theory
- **必读 / Must-read**: A Theory of Generalization in Deep Learning。
- **延伸 / Follow-up**: DASH-Q、Symmetry in Language Statistics、The Spike, the Sparse and the Sink。
- **建议 / Advice**: 将 NTK 泛化理论与实际训练曲线对照验证；量化方向建议复现 DASH-Q 并在自己的模型上测试低比特 PTQ 收益。

### 系统与 Agent / Systems & Agents
- **必读 / Must-read**: AgenTEE、ClawVM、Scaling Agentic Verifier。
- **延伸 / Follow-up**: GLM-5、ALIVE。
- **建议 / Advice**: 关注 Agent 执行环境中的安全边界与状态持久化；可尝试在本地 TEE 模拟环境或虚拟内存层原型上验证 ClawVM 的设计思想。

### 机器人学 / Robotics
- **必读 / Must-read**: VIRAL、DoorMan。
- **延伸 / Follow-up**: Accelerating Robotic RL with Agent Guidance、Hybrid Framework for Robotic Manipulation。
- **建议 / Advice**: 从仿真环境（Isaac Gym/Gymnasium）入手复现一个人形机器人像素到动作策略，重点关注视觉观测归一化与 sim-to-real gap。

---

## 重要时间节点 / Key Dates

| 会议 / Conference | 地点 / Location | 时间 / Date | 备注 / Notes |
|---|---|---|---|
| CVPR 2026 | Nashville, USA | June 8–14, 2026 | 论文列表已开放，Workshop/Tutorial 注册开放中。/ *Program live; workshop/tutorial registration open.* |
| ICML 2026 | Seoul, South Korea | July 6–11, 2026 | 投稿量突破 30,800 篇。/ *Received over 30,800 submissions.* |
| NeurIPS 2026 | San Diego, USA | December 2026 | 截稿通常在 5 月中下旬，建议关注官网更新。/ *Deadline typically mid-to-late May; monitor official website.* |
| IJCAI 2026 | Montreal, Canada | August 2026 | 已公布接受论文列表。/ *Accepted papers list published.* |
| EMNLP 2026 | TBA | November 2026 | 通常 5 月初开放投稿系统。/ *Submission system usually opens around early May.* |

---

## 数据来源 / Sources

- arXiv categories: cs.AI, cs.CL, cs.CV, cs.LG, cs.RO, cs.CR, cs.SD
- CVPR 2026 official program: https://cvpr.thecvf.com/virtual/2026/papers.html
- CVPR 2026 workshops: https://cvpr.thecvf.com/Conferences/2026/Workshops
- ICML 2026 program: https://icml.cc/virtual/2026/papers.html
- ICML 2026 workshops: https://icml.cc/virtual/2026/events/workshop
- Paper Digest CVPR 2026 highlights: https://www.paperdigest.org/2026/04/cvpr-2026-papers-highlights/
- Paper Digest ICML 2026 highlights: https://www.paperdigest.org/2026/05/icml-2026-papers-highlights/
- Ahead of AI 2026 LLM papers list: https://magazine.sebastianraschka.com/p/llm-research-papers-2026-part1
- AI conference deadlines: http://aideadlines.org/
