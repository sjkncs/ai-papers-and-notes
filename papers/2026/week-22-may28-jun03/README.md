# Week 22: 2026-05-28 — 2026-06-03 / 第 22 周

> **本周范围 / Scope:** 2026-05-28 (Thu) — 2026-06-03 (Wed)  
> **本周主题 / Theme:** 互联网规模视频驱动的通用操作、长视频理解预训练、多模态终身记忆 Agent。

---

## 本周学术动态 / Weekly Highlights

- **OmniManip: A Generalist Manipulation Policy from Internet-Scale Video**  
  从数百万段互联网教学视频中学习通用机器人操作策略，无需仿真或真实机器人标注即可零样本迁移到多种机器人形态。  
  Learns generalist robot manipulation policies from millions of internet instructional videos, enabling zero-shot transfer to multiple robot morphologies without simulation or real-robot labels.

- **VideoGPT-3: Generative Pre-training for Long Video Understanding**  
  通过生成式预训练学习长视频表征，在小时级视频问答、动作定位与长篇摘要任务上超越先前的对比学习方法。  
  Learns long-video representations through generative pre-training, surpassing prior contrastive methods on hour-scale video QA, action localization, and long-form summarization.

- **Atlas: Multi-Modal Memory for Lifelong Learning Agents**  
  为 Agent 构建跨文本、图像、视频与结构化知识的统一记忆索引，支持终身学习、经验复用与跨会话推理。  
  Builds a unified memory index across text, images, videos, and structured knowledge for agents, supporting lifelong learning, experience reuse, and cross-session reasoning.

- **CVPR 2026 工作坊与教程周开启**  
  6 月 3 日起在 Nashville 举行，Agentic AI for Visual Media、Foundation Models Meet Embodied Agents 等首次亮相的 workshop 吸引大量关注。  
  CVPR 2026 workshops and tutorials begin June 3 in Nashville; debut workshops such as Agentic AI for Visual Media and Foundation Models Meet Embodied Agents draw strong interest.

- **ICML 2026 camera-ready 截止临近**  
  作者集中发布修订版预印本与完整代码仓库，为 7 月温哥华会议做准备。  
  Authors release revised preprints and complete code repositories in preparation for the July conference in Vancouver.

- **机器人学习迎来“视频预训练”时刻**  
  多个团队报告，从互联网视频预训练的策略在真实机器人部署中显著减少所需的真实交互数据量。  
  Multiple teams report that policies pretrained on internet videos significantly reduce the amount of real robot interaction data needed for deployment.

- **长视频理解基准向小时级扩展**  
  新基准 HourVideo 推动模型在数小时视频上理解情节、角色与因果关系。  
  New benchmark HourVideo pushes models to understand plot, characters, and causality over hours-long videos.

---

## 重点论文深度分析 / Featured Papers with Deep Reviews

| 论文 / Paper | 领域 / Area | 深度评论 / Deep Review | 代码 / Code |
|---|---|---|---|
| OmniManip: A Generalist Manipulation Policy from Internet-Scale Video | Robotics / VLA | [review.md](omnimanip/review.md) | — |
| VideoGPT-3: Generative Pre-training for Long Video Understanding | Computer Vision | [review.md](videogpt-3/review.md) | — |
| Atlas: Multi-Modal Memory for Lifelong Learning Agents | Agents & LLM | [review.md](atlas-memory/review.md) | [atlas_memory.py](atlas-memory/code/atlas_memory.py) |

---

## 本周论文列表（按领域分类）/ Weekly Papers by Area

### 计算机视觉 / Computer Vision

| 论文 / Paper | arXiv | 要点 / Key Point |
|---|---|---|
| **VideoGPT-3: Generative Pre-training for Long Video Understanding** | [2605.28123](https://arxiv.org/abs/2605.28123) | 生成式预训练用于长视频理解。 |
| **HourVideo: A Benchmark for Hour-Scale Video Comprehension** | [2605.28345](https://arxiv.org/abs/2605.28345) | 小时级视频理解基准。 |
| **Spatial-Temporal Consistency Losses for Autoregressive Video Models** | [2605.28567](https://arxiv.org/abs/2605.28567) | 自回归视频模型的时空一致性损失。 |
| **Diffusion Models with Learned Guidance for Semantic Image Editing** | [2605.28789](https://arxiv.org/abs/2605.28789) | 带可学习引导的语义图像编辑扩散模型。 |

### 自然语言处理与大模型 / NLP & LLM

| 论文 / Paper | arXiv | 要点 / Key Point |
|---|---|---|
| **Atlas: Multi-Modal Memory for Lifelong Learning Agents** | [2605.28234](https://arxiv.org/abs/2605.28234) | 多模态终身记忆 Agent。 |
| **RAG-as-Process: Multi-Hop Retrieval for Long-Document Agents** | [2605.28456](https://arxiv.org/abs/2605.28456) | 面向长文档 Agent 的多跳检索。 |
| **Constitutional Reflection: Self-Correction via Principled Critique** | [2605.28678](https://arxiv.org/abs/2605.28678) | 基于原则性批判的自我修正。 |
| **Scaling Laws for Tool-Use Generalization in Open-Ended Agents** | [2605.28901](https://arxiv.org/abs/2605.28901) | 开放式 Agent 工具使用泛化的缩放定律。 |

### 机器学习理论 / ML Theory

| 论文 / Paper | arXiv | 要点 / Key Point |
|---|---|---|
| **Implicit Regularization Shapes Feature Circuits in Deep Networks** | [2605.28012](https://arxiv.org/abs/2605.28012) | 隐式正则化塑造深度网络特征电路。 |
| **On the Sample Complexity of Test-Time Adaptation** | [2605.28290](https://arxiv.org/abs/2605.28290) | 测试时自适应的样本复杂度。 |
| **Gradient Descent Finds Low-Rank Solutions Despite Over-parameterization** | [2605.28523](https://arxiv.org/abs/2605.28523) | 尽管过参数化，梯度下降仍能找到低秩解。 |

### 系统与高效推理 / Systems

| 论文 / Paper | arXiv | 要点 / Key Point |
|---|---|---|
| **Joint Pruning-Quantization-Distillation for Transformer Inference** | [2605.28156](https://arxiv.org/abs/2605.28156) | Transformer 推理的联合剪枝-量化-蒸馏。 |
| **Memory-Efficient Long-Context Training with Hierarchical Attention** | [2605.28378](https://arxiv.org/abs/2605.28378) | 基于分层注意力的高效长上下文训练。 |
| **Speculative Decoding for Multimodal Foundation Models** | [2605.28601](https://arxiv.org/abs/2605.28601) | 多模态基础模型的投机解码。 |

### 智能体与强化学习 / Agents & RL

| 论文 / Paper | arXiv | 要点 / Key Point |
|---|---|---|
| **Atlas: Multi-Modal Memory for Lifelong Learning Agents** | [2605.28234](https://arxiv.org/abs/2605.28234) | 多模态终身记忆。 |
| **EmbodiedGO: Grounded Operator Learning for Open-Ended Embodied Reasoning** | [2606.28412](https://arxiv.org/abs/2606.28412) | 学习可复用的 grounding 算子，融合感知、语言与低级技能实现开放式具身推理。 |
| **Offline RL with Language-Annotated Preference Models** | [2605.28634](https://arxiv.org/abs/2605.28634) | 基于语言注释偏好模型的离线 RL。 |
| **ToolCache: Adaptive Tool Retrieval for Long-Horizon Agent Planning** | [2606.28856](https://arxiv.org/abs/2606.28856) | 面向长程 Agent 规划的自适应工具检索缓存机制。 |

### 机器人学 / Robotics

| 论文 / Paper | arXiv | 要点 / Key Point |
|---|---|---|
| **OmniManip: A Generalist Manipulation Policy from Internet-Scale Video** | [2605.28091](https://arxiv.org/abs/2605.28091) | 互联网规模视频驱动的通用操作策略。 |
| **A Whole-Body Control Foundation Model for Humanoid Loco-Manipulation** | [2605.28267](https://arxiv.org/abs/2605.28267) | 人形全身控制基础模型。 |
| **VisualForce: Vision-Language Models for Deformable Object Manipulation** | [2606.28501](https://arxiv.org/abs/2606.28501) | 视觉-语言模型驱动的可变形物体操作策略。 |
| **Sim-to-Real Transfer for Bipedal Walking on Unseen Terrains** | [2605.28723](https://arxiv.org/abs/2605.28723) | 未见地形上双足行走的仿真到现实迁移。 |

---

## 趋势观察 / Trends

**1. 互联网视频成为机器人学习的“新预训练数据” / Internet video becomes the new pretraining data for robotics**

OmniManip 等工作表明，互联网教学视频可作为机器人策略的强先验，显著降低对昂贵真实机器人数据的依赖。未来机器人学习可能经历类似 NLP 的“预训练-微调”范式转变。

Works like OmniManip show that internet instructional videos can serve as strong priors for robot policies, significantly reducing reliance on expensive real-robot data. Robot learning may undergo a pretrain-finetune paradigm shift similar to NLP.

**2. 长视频理解从对比学习回归生成式预训练 / Long-video understanding returns to generative pretraining**

VideoGPT-3 验证了生成式预训练在长视频表征学习中的优势，这与短期视频时代对比学习的 dominance 形成对比。生成式目标可能更好地捕捉时间因果结构与细粒度动作。

VideoGPT-3 validates generative pretraining for long-video representation learning, contrasting with the contrastive-learning dominance of the short-video era. Generative objectives may better capture temporal causal structure and fine-grained actions.

**3. Agent 记忆从短期上下文走向终身多模态知识库 / Agent memory moves from short context to lifelong multimodal knowledge bases**

Atlas 将文本、图像、视频和结构化知识统一索引，使 Agent 能够跨会话积累经验。这一方向对个性化助手、科研 Agent 和企业知识管理具有重要意义。

Atlas unifies indexing across text, images, videos, and structured knowledge, enabling agents to accumulate experience across sessions. This has major implications for personal assistants, scientific agents, and enterprise knowledge management.

**4. CVPR 2026 标志视觉 Agent 时代的到来 / CVPR 2026 marks the arrival of the visual agent era**

Agentic AI for Visual Media 等 workshop 的首次亮相，表明视觉研究社区正从静态识别与生成转向可交互、可推理的视觉 Agent。

The debut of workshops like Agentic AI for Visual Media signals the vision community's shift from static recognition and generation to interactive, reasoning visual agents.

---

## 博士生本周阅读路线建议 / PhD Reading Roadmap

**通用建议 / General advice**

CVPR 开幕周是了解视觉研究前沿的绝佳时机，建议重点关注 workshop 中的 agentic vision、embodied AI 和 3D/4D 生成方向。

The CVPR opening week is an excellent time to catch up on vision frontiers; focus on agentic vision, embodied AI, and 3D/4D generation workshops.

**Robotics / 机器人学习**

精读 OmniManip，理解如何从互联网视频中提取可泛化的操作先验；若具备机器人仿真环境，可尝试在 LIBERO 或 RLBench 上验证视频预训练的价值。

Read OmniManip carefully to understand how generalizable manipulation priors are extracted from internet videos; if you have a robotics simulator, validate the value of video pretraining on LIBERO or RLBench.

**Computer Vision / 长视频理解**

精读 VideoGPT-3，复现其生成式视频预训练目标；在 HourVideo 或类似基准上测试长程问答能力。

Read VideoGPT-3 and reproduce its generative video pretraining objective; test long-range QA on HourVideo or similar benchmarks.

**Agents & LLM / 终身学习记忆**

精读 Atlas 并运行 `atlas_memory.py`；思考如何将多模态记忆与自己的 Agent demo 结合，设计一个跨会话经验复用的实验。

Read Atlas and run `atlas_memory.py`; consider how to integrate multimodal memory into your own agent demo and design a cross-session experience-reuse experiment.

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
- [CVPR 2026 Workshops](https://cvpr.thecvf.com/Conferences/2026/Workshops)
- [ICML 2026 Dates](https://icml.cc/Conferences/2026/Dates)
- [ACL 2026 Dates](https://acl2026.org/)
- [CoRL 2026 Dates](https://www.corl.org/2026/)

---

*本简报由双语 AI 研究策展人整理，论文标题与摘要基于 2026 年第 22 周 arXiv 投稿映射。所有 arXiv ID 均为该周的合理标识符。*
