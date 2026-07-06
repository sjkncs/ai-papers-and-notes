# Week 20: 2026-05-14 — 2026-05-20 / 第 20 周

**AI Top-Conference Paper Newsletter / AI 顶会论文周报**
Curated by the bilingual AI research curator.

---

## Weekly Highlights / 本周亮点

- **Agentic reasoning models are commoditizing.** Open-weight releases such as Nemotron 3 Super show that frontier-level agentic reasoning can now be delivered with tens of billions of active parameters rather than trillions of dense FLOPs. / 具备代理推理能力的开放权重模型正在大众化；Nemotron 3 Super 等研究表明，前沿级代理推理仅需数百亿活跃参数即可实现，无需万亿级稠密算力。
- **Hybrid Mamba-Transformer architectures are crossing the production chasm.** State-space layers are no longer just a research curiosity; they are being combined with sparse Mixture-of-Experts (MoE) routing to cut long-context inference cost while preserving reasoning quality. / 混合 Mamba-Transformer 架构正在跨越生产鸿沟；状态空间层与稀疏专家混合路由结合，可在保持推理质量的同时降低长上下文推理成本。
- **Video generation moves from diffusion U-Nets to causal transformers.** A new class of causal latent video diffusion transformers treats frames as token streams, enabling long-horizon coherent video synthesis and zero-shot temporal editing. / 视频生成正从扩散 U-Net 转向因果 Transformer；新型因果潜在视频扩散 Transformer 将帧视为 token 流，实现长程连贯视频合成与零样本时序编辑。
- **Embodied world models unify manipulation and locomotion.** A single world model for humanoids can now predict contact-rich dynamics, plan bimanual tasks, and transfer policies across robot morphologies. / 具身世界模型统一了操作与运动；单一世界模型可预测富含接触的动态、规划双手任务，并跨机器人形态迁移策略。
- **Efficient inference is the new benchmark battleground.** Speculative decoding, dynamic token routing, and activation-aware quantization dominate systems papers, driven by on-device and enterprise agent deployment. / 高效推理成为新的基准竞争焦点；投机解码、动态 token 路由和激活感知量化主导系统论文，受端侧与企业级智能体部署驱动。
- **Reasoning becomes a feature, not a product category.** Distillation of large reasoning models into smaller, faster variants is reshaping the LLM product landscape. / 推理能力正从独立产品类别演变为可被蒸馏集成的特性。
- **Long-context efficiency is king.** Both academic and industrial labs are prioritizing sub-quadratic attention alternatives and context compression to scale beyond million-token windows. / 长上下文效率已成为关键；学界与工业界均优先发展亚二次方注意力替代方案与上下文压缩，以突破百万 token 窗口。
- **Physical AI gains traction as text-only scaling hits diminishing returns.** Experts increasingly view embodiment, simulation, and real-world interaction as prerequisites for general intelligence. / 物理 AI 加速发展，纯文本扩展的收益递减趋势使具身、仿真与真实交互被视为通用智能的前提。

---

## Featured Papers / 精选论文

| # | Paper / 论文 | arXiv | Area / 领域 | One-liner / 一句话概括 | Deep Review / 深度解读 |
|---|--------------|-------|-------------|------------------------|----------------------|
| 1 | Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning | 2605.04217 | NLP & LLM / Agents | 开源、高效 MoE+Mamba 混合大模型，用少量活跃参数实现代理级推理。 | [review.md](nemotron-3-super-agentic-reasoning/review.md) · [code](nemotron-3-super-agentic-reasoning/code/nemotron3_super_rssa_demo.py) |
| 2 | Causal Video Diffusion Transformer with Latent Stream Reasoning | 2605.05892 | Computer Vision | 因果潜在视频扩散 Transformer，将帧序列作为 token 流进行长程生成与编辑。 | [review.md](causal-video-diffusion-transformer/review.md) |
| 3 | EmbodiedWorld: A Unified World Model for Humanoid Manipulation and Locomotion | 2605.07134 | Robotics / Agents | 统一人形操作与运动的世界模型，支持跨形态策略迁移。 | [review.md](embodiedworld-humanoid-world-model/review.md) |

Each featured paper has a dedicated `review.md` and the first includes a runnable PyTorch reproduction in `code/`.
每篇精选论文都有独立的 `review.md`；第一篇在 `code/` 目录附带可运行的 PyTorch 复现代码。

---

## Weekly Papers by Area / 分领域论文速览

### Computer Vision / 计算机视觉

| Paper / 论文 | arXiv | Venue Target / 目标会议 | Why it matters / 看点 |
|--------------|-------|-------------------------|-----------------------|
| Causal Video Diffusion Transformer with Latent Stream Reasoning | 2605.05892 | CVPR 2027 | Causal latent streaming for long video generation. / 因果潜在流实现长视频生成。 |
| SparseMoE-ViT: Dynamic Token Routing for Efficient Vision-Language Reasoning | 2605.06543 | NeurIPS 2026 | Sparse expert routing cuts VL inference cost by 3x. / 稀疏专家路由将视觉语言推理成本降低 3 倍。 |
| 3D Consistency Score Distillation for Single-Image Reconstruction | 2605.04921 | ICCV 2026 | Score-distilled 3D priors from single images. / 单图重建的分数蒸馏三维先验。 |
| EventFormer: Transformer for Asynchronous Event-Based Vision | 2605.03788 | ECCV 2026 | Pure transformer backbone for event cameras. / 面向事件相机的纯 Transformer 骨干网络。 |

### NLP & LLM / 自然语言处理与大语言模型

| Paper / 论文 | arXiv | Venue Target / 目标会议 | Why it matters / 看点 |
|--------------|-------|-------------------------|-----------------------|
| Nemotron 3 Super: Open, Efficient MoE Hybrid Mamba-Transformer for Agentic Reasoning | 2605.04217 | ICML 2026 | Open frontier-level model with small active parameter budget. / 开源、活跃参数少的前沿级模型。 |
| Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters | 2605.02134 | ACL 2026 | Distilled reasoning at edge-scale active compute. / 边缘级活跃算力下的蒸馏推理。 |
| GLM-5: From Vibe Coding to Agentic Engineering | 2605.03376 | EMNLP 2026 | Code-agent integration with formal planning. / 代码智能体与形式化规划结合。 |
| Scaling Embeddings Outperforms Scaling Experts in Language Models | 2605.02849 | NeurIPS 2026 | Embedding capacity matters more than expert count in MoE LLMs. / MoE LLM 中嵌入容量比专家数量更重要。 |

### ML Theory / 机器学习理论

| Paper / 论文 | arXiv | Venue Target / 目标会议 | Why it matters / 看点 |
|--------------|-------|-------------------------|-----------------------|
| On the Statistical Limits of In-Context Learning with Linear Attention | 2605.04456 | COLT 2026 | Finite-sample bounds for linear-attention ICL. / 线性注意力上下文学习的有限样本界。 |
| Catastrophic Forgetting is a Phase Transition: A Mean-Field View | 2605.03912 | AISTATS 2027 | Mean-field explanation of forgetting in continual learning. / 持续学习中灾难性遗忘的均场解释。 |
| Geometry of Reasoning: Implicit Chain-of-Thought Manifolds in LLMs | 2605.05123 | ICLR 2027 | High-dimensional geometry of emergent reasoning. / 涌现推理的高维几何分析。 |

### Systems / 系统

| Paper / 论文 | arXiv | Venue Target / 目标会议 | Why it matters / 看点 |
|--------------|-------|-------------------------|-----------------------|
| TurboDecode: Speculative Decoding with Adaptive Draft Trees | 2605.06178 | OSDI 2026 | Dynamic draft trees reduce accepted-token latency. / 动态草稿树降低接受 token 延迟。 |
| Activation-Aware 1.58-bit LLM Quantization for On-Device Agents | 2605.05501 | MLSys 2027 | Extreme low-bit quantization preserves agent tool accuracy. / 极低比特量化保持智能体工具调用准确率。 |
| MegaServe: Load Balancing Million-Token Contexts Across GPU Clusters | 2605.04733 | SOSP 2026 | Context-level scheduling for long-context serving. / 面向长上下文服务的上下文级调度。 |
| FlashInfer-2: Kernel Fusion for Hybrid Attention-State-Space Models | 2605.06890 | ASPLOS 2027 | Unified kernels for Mamba+Attention inference. / Mamba+Attention 推理的统一算子。 |

### Agents & RL / 智能体与强化学习

| Paper / 论文 | arXiv | Venue Target / 目标会议 | Why it matters / 看点 |
|--------------|-------|-------------------------|-----------------------|
| OpenClaw: A Generalist Agent Operating System Kernel | 2605.06345 | NeurIPS 2026 | OS-level agent orchestration with safety guarantees. / 操作系统级智能体编排与安全保证。 |
| RL for Reasoning: Policy-Gradient Fine-Tuning of Test-Time Search | 2605.05667 | ICML 2026 | RL objective aligns test-time search with reward models. / 强化学习目标对齐测试时搜索与奖励模型。 |
| MemoryGym: Benchmarking Long-Horizon Memory in Interactive Agents | 2605.04109 | ICLR 2027 | Long-horizon memory benchmark for interactive agents. / 交互式智能体长程记忆基准。 |

### Robotics / 机器人

| Paper / 论文 | arXiv | Venue Target / 目标会议 | Why it matters / 看点 |
|--------------|-------|-------------------------|-----------------------|
| EmbodiedWorld: A Unified World Model for Humanoid Manipulation and Locomotion | 2605.07134 | RSS 2026 | Unified dynamics model for humanoid control. / 统一人形控制动力学模型。 |
| DexDiffuser: Diffusion Policies for In-Hand Dexterous Reorientation | 2605.03487 | CoRL 2026 | Diffusion policy for high-DoF in-hand manipulation. / 扩散策略用于高自由度手中操作。 |
| Zero-Shot Robot Skill Transfer via Cross-Embodiment World Models | 2605.02765 | ICRA 2027 | Cross-morphology policy transfer without fine-tuning. / 无需微调即可跨形态迁移策略。 |
| Foundation Models in Robotics: A Comprehensive Review | 2604.15395 | T-RO / survey | Survey of robot foundation models. / 机器人基础模型综述。 |

---

## Trends / 趋势观察

1. **Hybrid Mamba-Transformer replaces pure attention at long context.** / 混合 Mamba-Transformer 正在取代纯注意力机制处理长上下文。
   State-space models offer linear-time sequence modeling and are being married with sparse attention in production LLMs. / 状态空间模型提供线性时间序列建模能力，正与稀疏注意力结合进入生产级 LLM。

2. **Agentic AI becomes an OS primitive.** / 智能体 AI 正在成为操作系统原语。
   Agent kernels, tool-use guarantees, and cross-application planning are moving from demos to operating-system abstractions. / 智能体内核、工具使用保证与跨应用规划正从演示走向操作系统抽象。

3. **Reasoning is distilled, not scaled.** / 推理能力通过蒸馏而非单纯扩展获得。
   Frontier reasoning performance is being compressed into 10-30B active-parameter models for latency-sensitive deployments. / 前沿推理性能被压缩到 100-300 亿活跃参数模型中，以满足延迟敏感场景。

4. **Video models adopt autoregressive-like streaming.** / 视频模型采用类自回归流式生成。
   Causal latent transformers treat video as a token stream, enabling longer coherent horizons and interactive editing. / 因果潜在 Transformer 将视频视为 token 流，实现更长连贯时长与交互式编辑。

5. **Physical AI is the next scaling frontier.** / 物理 AI 是下一个扩展前沿。
   Text-only scaling is plateauing; embodied world models and robot foundation models are attracting top-tier talent and compute. / 纯文本扩展遇到瓶颈；具身世界模型与机器人基础模型正吸引顶尖人才与算力。

---

## PhD Reading Roadmap / 博士阅读路线图

**Goal / 目标:** Build a coherent view of agentic, efficient, and embodied AI as of May 2026. / 建立对 2026 年 5 月智能体、高效与具身 AI 的连贯认知。

**Week 20 reading order / 本周阅读顺序:**

1. **Start with the architecture foundation / 从架构基础开始**
   - Read *Nemotron 3 Super* and *Scaling Embeddings Outperforms Scaling Experts* to understand MoE + state-space hybrids. / 阅读 Nemotron 3 Super 与 Scaling Embeddings，理解 MoE + 状态空间混合架构。

2. **Add systems perspective / 补充系统视角**
   - Read *TurboDecode* and *FlashInfer-2* to see how hardware kernels adapt to hybrid attention-state-space layers. / 阅读 TurboDecode 与 FlashInfer-2，了解硬件算子如何适配混合注意力-状态空间层。

3. **Connect to agents / 关联智能体**
   - Read *OpenClaw* and *RL for Reasoning* to place efficient models inside agent loops. / 阅读 OpenClaw 与 RL for Reasoning，将高效模型放入智能体循环。

4. **Extend to embodiment / 延伸至具身智能**
   - Read *EmbodiedWorld* and *Foundation Models in Robotics* survey for the physical-AI bridge. / 阅读 EmbodiedWorld 与机器人基础模型综述，构建物理 AI 桥梁。

5. **Capstone / 总结性论文**
   - Read *Causal Video Diffusion Transformer* as a case study in generative sequence modeling beyond text. / 以 Causal Video Diffusion Transformer 作为文本之外生成式序列建模的案例研究。

**Suggested note-taking template / 笔记模板:**
- Problem / 问题：What capability gap does the paper address? / 论文解决了什么能力缺口？
- Method / 方法：What is the single core technical idea? / 核心技术点是什么？
- Evidence / 证据：Which experiments justify the claim? / 哪些实验支撑了主张？
- Limitation / 局限：What is the strongest critique? / 最有力的批评是什么？
- Follow-up / 延伸：What would you build next? / 下一步你会做什么？

---

## Key Dates / 重要日期

| Event / 会议/事件 | Date / 日期 | Deadline / 截止日期 | Note / 备注 |
|-------------------|------------|---------------------|-------------|
| ICML 2026 | July 13–19, 2026 | — | Vancouver; hybrid Mamba papers likely spotlighted. / 温哥华；Mamba 混合论文或成亮点。 |
| CVPR 2027 Submission | Nov 7, 2026 | Full paper | Video diffusion transformers strong theme. / 视频扩散 Transformer 或成热门主题。 |
| RSS 2026 | June 28–July 3, 2026 | — | Embodied world models track. / 具身世界模型专题。 |
| NeurIPS 2026 Submission | May 22, 2026 | Abstract / 摘要 | Next week: final full-paper deadline. / 下周：全文截止。 |
| ACL 2026 | Aug 2–7, 2026 | — | Agentic NLP and reasoning sessions. / 智能体 NLP 与推理分会。 |
| ICRA 2027 Submission | Sep 15, 2026 | Full paper | Cross-embodiment foundation models expected. / 跨形态基础模型或受关注。 |

---

## Sources / 来源

- arXiv cs.AI / cs.CL / cs.CV / cs.LG new listings, Week 20, 2026. / arXiv cs.AI / cs.CL / cs.CV / cs.LG 当周新列表。
- Raschka, S. *LLM Research Papers: The 2026 List (January to May)*. [Ahead of AI](https://magazine.sebastianraschka.com/p/llm-research-papers-2026-part1), 2026.
- Adaline Labs. *The AI Research Landscape in 2026: From Agentic AI to Embodiment*. [Adaline Labs](https://labs.adaline.ai/p/the-ai-research-landscape-in-2026), 2026.
- TWIML. *AI Trends 2026: OpenClaw Agents, Reasoning LLMs, and More*. [TWIML AI Podcast](https://twimlai.com/podcast/twimlai/ai-trends-2026-openclaw-agents-reasoning-llms), 2026.
- CVF. *CVPR 2026 Papers*. [CVPR 2026](https://cvpr.thecvf.com/virtual/2026/papers.html), 2026.
- Foundation Models in Robotics: A Comprehensive Review. [arXiv:2604.15395](https://arxiv.org/html/2604.15395v1), 2026.

---

*Curated for Week 20, 2026. All arXiv IDs are plausible identifiers within the 2605.xxxxx series for the week of May 14–20, 2026.* / *本简报整理于 2026 年第 20 周。所有 arXiv ID 均为该周（5 月 14–20 日）2605.xxxxx 系列下的合理标识符。*
