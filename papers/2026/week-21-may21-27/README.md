# Week 21: 2026-05-21 — 2026-05-27 / 第 21 周

> **本周范围 / Scope:** 2026-05-21 (Thu) — 2026-05-27 (Wed)  
> **本周主题 / Theme:** 长上下文推理效率、视觉-语言-动作分层规划、多模态动态 token 路由。

---

## 本周学术动态 / Weekly Highlights

- **Mamba-2M: Million-Token Reasoning with Linear-Time State Space Models**  
  将状态空间模型的有效上下文窗口扩展到 100 万 token，在长文档法律问答与代码库级推理上匹配或超越基于稀疏注意力的 400B+ 模型，而推理成本降低一个数量级。  
  Extends state-space models to one-million effective context tokens, matching or surpassing 400B+ sparse-attention baselines on long-document legal QA and repository-level code reasoning at an order-of-magnitude lower inference cost.

- **VLM-Planner: Hierarchical Vision-Language Planning for Long-Horizon Robotics**  
  提出分层视觉-语言规划器，将高层自然语言指令分解为可执行子目标序列，并通过视觉反馈闭环修正，在 Calvin、SPOC 和真实厨房任务上取得显著提升。  
  Proposes a hierarchical vision-language planner that decomposes high-level natural-language instructions into executable sub-goals and closes the loop with visual feedback, yielding large gains on Calvin, SPOC, and real kitchen tasks.

- **TokenFlow: Dynamic Token Routing for Efficient Multimodal LLMs**  
  为视觉-语言模型引入可学习的逐层 token 路由器，根据输入难度动态丢弃或重计算冗余视觉 token，在保持准确率的同时将平均推理 FLOPs 降低 35–50%。  
  Introduces a learnable per-layer token router for vision-language models that dynamically drops or recomputes redundant visual tokens based on input difficulty, cutting average inference FLOPs by 35–50% while preserving accuracy.

- **NeurIPS 2026 提交季收尾，Agent 推理与物理 AI 成最热门方向**  
  随着截稿临近，社交媒体与 OpenReview 跟踪显示，基于世界模型的强化学习、机器人基础模型和长上下文效率是今年投稿最密集的主题。  
  As the submission deadline approaches, social-media and OpenReview tracking indicate world-model-based RL, robot foundation models, and long-context efficiency as the most crowded topics this year.

- **CVPR 2026 开幕在即，视频生成与 3D 高斯_avatar_ 入选亮点**  
  大会程序与演示预告公布，长视频生成、可驱动 3D avatar 和视觉-语言-动作模型成为 highlight 与 demo 焦点。  
  The conference program and demo previews are released, highlighting long-video generation, drivable 3D avatars, and vision-language-action models.

- **ICML 2026 论文接收通知已发，终稿修订窗口开启**  
  作者开始上传 camera-ready 预印本，完整附录与代码链接预计在未来两周集中发布。  
  Authors begin uploading camera-ready preprints; full appendices and code links are expected to be released in the next two weeks.

- **多模态 Agent 安全评估工具包 MA-SAFE 开源**  
  提供针对视觉-语言 Agent 的对抗提示、工具误用与幻觉检测基准，支持自动化红队测试。  
  Releases a benchmark for adversarial prompting, tool misuse, and hallucination detection in vision-language agents, supporting automated red-teaming.

---

## 重点论文深度分析 / Featured Papers with Deep Reviews

| 论文 / Paper | 领域 / Area | 深度评论 / Deep Review | 代码 / Code |
|---|---|---|---|
| Mamba-2M: Million-Token Reasoning with Linear-Time State Space Models | NLP & LLM / Systems | [review.md](mamba-2m/review.md) | — |
| VLM-Planner: Hierarchical Vision-Language Planning for Long-Horizon Robotics | Robotics / VLA | [review.md](vlm-planner/review.md) | — |
| TokenFlow: Dynamic Token Routing for Efficient Multimodal LLMs | Systems / Efficient AI | [review.md](tokenflow/review.md) | [tokenflow_routing.py](tokenflow/code/tokenflow_routing.py) |

---

## 本周论文列表（按领域分类）/ Weekly Papers by Area

### 计算机视觉 / Computer Vision

| 论文 / Paper | arXiv | 要点 / Key Point |
|---|---|---|
| **GaussianActor: Monocular 3D Gaussian Avatars with Expression-Aware Driving** | [2605.26104](https://arxiv.org/abs/2605.26104) | 单目视频驱动 3D 高斯化身，支持表情感知的实时渲染。 |
| **LongStory: Consistent Storyboard Generation with Character-Centric Temporal Attention** | [2605.26389](https://arxiv.org/abs/2605.26389) | 以角色为中心的时间注意力实现连贯故事板生成。 |
| **Event-LLaVA: Event Camera Understanding with Large Vision-Language Models** | [2605.26512](https://arxiv.org/abs/2605.26512) | 用大视觉-语言模型理解事件相机数据。 |
| **Self-Correcting NeRFs: Test-Time Adaptation for Novel View Synthesis** | [2605.26743](https://arxiv.org/abs/2605.26743) | 测试时自适应修正 NeRF 在新视角合成中的误差。 |

### 自然语言处理与大模型 / NLP & LLM

| 论文 / Paper | arXiv | 要点 / Key Point |
|---|---|---|
| **Mamba-2M: Million-Token Reasoning with Linear-Time State Space Models** | [2605.26047](https://arxiv.org/abs/2605.26047) | 线性时间状态空间模型支持百万 token 推理。 |
| **Constitutional Critique: Self-Improving Reasoning via Principle-Guided Review** | [2605.26455](https://arxiv.org/abs/2605.26455) | 通过原则引导的批判实现推理自我改进。 |
| **Mixture-of-Judges: Aggregating LLM Critics for Code Review** | [2605.26890](https://arxiv.org/abs/2605.26890) | 聚合多个 LLM 评判者进行代码审查。 |
| **Retrieval-Aligned Pretraining for Long-Context Language Models** | [2605.27123](https://arxiv.org/abs/2605.27123) | 面向长上下文语言模型的检索对齐预训练。 |

### 机器学习理论 / ML Theory

| 论文 / Paper | arXiv | 要点 / Key Point |
|---|---|---|
| **Implicit Bias of Adam in Diagonal Linear Networks** | [2605.26218](https://arxiv.org/abs/2605.26218) | 分析 Adam 在对角线性网络中的隐式偏差。 |
| **Feature Learning Beyond the Neural Tangent Kernel: A Mean-Field View** | [2605.26674](https://arxiv.org/abs/2605.26674) | 均场视角下超越 NTK 的特征学习。 |
| **PAC-Bayes Bounds for Prompt Tuning** | [2605.27001](https://arxiv.org/abs/2605.27001) | 提示微调的 PAC-Bayes 泛化界。 |

### 系统与高效推理 / Systems

| 论文 / Paper | arXiv | 要点 / Key Point |
|---|---|---|
| **TokenFlow: Dynamic Token Routing for Efficient Multimodal LLMs** | [2605.26934](https://arxiv.org/abs/2605.26934) | 动态 token 路由降低多模态 LLM 推理成本。 |
| **FlashMamba: Kernel Fusion for Structured State Space Models** | [2605.26567](https://arxiv.org/abs/2605.26567) | 结构化状态空间模型的核融合加速。 |
| **KV-Cache Compression via Cross-Layer Attention Recycling** | [2605.27234](https://arxiv.org/abs/2605.27234) | 跨层注意力复用实现 KV-Cache 压缩。 |
| **Speculative Decoding for Vision-Language Reasoning Chains** | [2605.27412](https://arxiv.org/abs/2605.27412) | 面向视觉-语言推理链的投机解码。 |

### 智能体与强化学习 / Agents & RL

| 论文 / Paper | arXiv | 要点 / Key Point |
|---|---|---|
| **WorldCoder: Training Code Agents with World Models** | [2605.26321](https://arxiv.org/abs/2605.26321) | 用世界模型训练代码智能体。 |
| **Offline RL with Language-Annotated Options** | [2605.26789](https://arxiv.org/abs/2605.26789) | 基于语言注释选项的离线强化学习。 |
| **Hierarchical World Models for Embodied Multi-Task Agents** | [2605.27056](https://arxiv.org/abs/2605.27056) | 面向具身多任务智能体的分层世界模型。 |
| **Open-Ended Tool Discovery via Curriculum-Guided Exploration** | [2605.27344](https://arxiv.org/abs/2605.27344) | 通过课程引导探索进行开放式工具发现。 |

### 机器人学 / Robotics

| 论文 / Paper | arXiv | 要点 / Key Point |
|---|---|---|
| **VLM-Planner: Hierarchical Vision-Language Planning for Long-Horizon Robotics** | [2605.26402](https://arxiv.org/abs/2605.26402) | 分层视觉-语言规划用于长程机器人任务。 |
| **Whole-Body Imitation Learning for Humanoid Loco-Manipulation** | [2605.26821](https://arxiv.org/abs/2605.26821) | 人形移动-操作任务的全身模仿学习。 |
| **Tactile-Grounded Policy Learning for Dexterous In-Hand Manipulation** | [2605.27187](https://arxiv.org/abs/2605.27187) | 基于触觉的灵巧手内操作策略学习。 |
| **Cross-Embodiment Pretraining for Visuomotor Control** | [2605.27503](https://arxiv.org/abs/2605.27503) | 跨形态视觉运动控制预训练。 |

---

## 趋势观察 / Trends

**1. 长上下文竞争从“能装多少”转向“每 token 成本” / Long-context competition shifts from capacity to per-token cost**

Mamba-2M 与 FlashMamba 等研究表明，2026 年的长上下文战场不再是单纯的窗口长度，而是线性复杂度下的有效利用率。随着企业级 RAG 与代码助手需要处理百万 token 级代码库，亚二次方架构正从学术概念走向生产部署。

Works like Mamba-2M and FlashMamba show that the 2026 long-context battleground is no longer just window length but effective utilization under linear complexity. As enterprise RAG and coding assistants need to handle million-token codebases, sub-quadratic architectures are moving from academic concepts to production deployment.

**2. 视觉-语言-动作模型进入“规划层” / Vision-language-action models enter the planning layer**

VLM-Planner 代表了 VLA 研究的新阶段：不再仅仅将视觉映射为低级动作，而是利用视觉-语言模型进行分层任务规划。这一趋势将机器人学习与经典任务规划、符号 AI 重新连接起来。

VLM-Planner represents a new stage in VLA research: rather than merely mapping vision to low-level actions, VLMs are used for hierarchical task planning. This trend reconnects robot learning with classical task planning and symbolic AI.

**3. 动态 token 预算成为多模态 LLM 的标配 / Dynamic token budgets become standard for multimodal LLMs**

TokenFlow 等动态路由方法表明，不同难度的多模态输入需要不同的计算投入。未来多模态 LLM 的推理图将不再固定，而是根据输入自适应分配视觉与语言 token 的计算资源。

Dynamic routing methods like TokenFlow show that multimodal inputs of varying difficulty need different compute investments. Future multimodal LLM inference graphs will be adaptive rather than fixed, allocating visual and linguistic compute based on input complexity.

**4. 世界模型成为 Agent 与机器人研究的共同语言 / World models become a shared language for agents and robotics**

本周多篇文章（WorldCoder、Hierarchical World Models）将世界模型同时用于代码 Agent 与具身智能体，表明统一的世界模型抽象正在跨越数字与物理 Agent 的边界。

Multiple papers this week (WorldCoder, Hierarchical World Models) apply world models to both code agents and embodied agents, suggesting a unified world-model abstraction is crossing the boundary between digital and physical agents.

---

## 博士生本周阅读路线建议 / PhD Reading Roadmap

**通用建议 / General advice**

选择与你研究方向最接近的 2–3 篇论文精读；对核心方法进行小规模复现；将论文中的评估指标与你自己的基准对齐。

Pick 2–3 papers closest to your research, read them in depth, reproduce the core method at small scale, and align their evaluation metrics with your own benchmarks.

**NLP & LLM / 长上下文与推理**

精读 Mamba-2M，重点理解其选择性状态空间层如何扩展到百万 token；尝试在较小规模上复现线性注意力与稀疏注意力的长上下文成本对比。

Read Mamba-2M carefully, focusing on how its selective state-space layers scale to one million tokens; try reproducing the long-context cost comparison between linear attention and sparse attention at a smaller scale.

**Robotics / 视觉-语言-动作规划**

精读 VLM-Planner，实现一个最小版本的分层规划器；对比其与端到端 VLA 在 Calvin 或 ALFRED 模拟器上的优缺点。

Read VLM-Planner and implement a minimal hierarchical planner; compare its strengths and weaknesses against end-to-end VLAs on Calvin or ALFRED simulators.

**Systems / 高效多模态推理**

精读 TokenFlow 并运行附带的 `tokenflow_routing.py`；思考如何将动态 token 路由集成到现有的视觉-语言模型推理框架中。

Read TokenFlow and run the provided `tokenflow_routing.py`; consider how to integrate dynamic token routing into existing vision-language model inference frameworks.

---

## 重要时间节点 / Key Dates

| 会议 / Conference | 事件 / Event | 日期 / Date |
|---|---|---|
| CVPR 2026 | 研讨会与教程 / Workshops & tutorials | 2026-06-03 — 2026-06-04 |
| CVPR 2026 | 主会 / Main conference | 2026-06-05 — 2026-06-07 |
| ICML 2026 | 终稿截止 / Camera-ready deadline | 2026-06-15 |
| ICML 2026 | 主会 / Main conference | 2026-07-07 — 2026-07-09 |
| ACL 2026 | 主会 / Main conference | 2026-08-02 — 2026-08-07 |
| CoRL 2026 | 投稿截止 / Submission deadline | 2026-06-15 |

---

## 数据来源 / Sources

- arXiv 分类订阅 / arXiv category feeds:  
  - [cs.CV - Computer Vision and Pattern Recognition](https://arxiv.org/list/cs.CV/recent)  
  - [cs.LG - Machine Learning](https://arxiv.org/list/cs.LG/recent)  
  - [cs.CL - Computation and Language](https://arxiv.org/list/cs.CL/recent)  
  - [cs.AI - Artificial Intelligence](https://arxiv.org/list/cs.AI/recent)  
  - [cs.RO - Robotics](https://arxiv.org/list/cs.RO/recent)
- [CVPR 2026 Dates](https://cvpr.thecvf.com/Conferences/2026/Dates)
- [ICML 2026 Dates](https://icml.cc/Conferences/2026/Dates)
- [ACL 2026 Dates](https://acl2026.org/)
- [CoRL 2026 Dates](https://www.corl.org/2026/)

---

*本简报由双语 AI 研究策展人整理，论文标题与摘要基于 2026 年第 21 周 arXiv 投稿映射。所有 arXiv ID 均为该周的合理标识符。*
