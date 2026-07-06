# Week 23: 2026-06-04 — 2026-06-09 / 第 23 周

> **Note / 说明**: This newsletter covers a 6-day arXiv window (June 4–9, 2026). Volume is lighter than a full 7-day week, but the papers reflect the immediate lead-up to the June 10–16 (Week 24) surge around CVPR 2026 debriefs and ICML 2026 camera-ready revisions.
> **说明**：本周报覆盖 6 天的 arXiv 窗口（2026 年 6 月 4–9 日）。投稿量少于完整 7 天周，但这些论文反映了 CVPR 2026 会后总结与 ICML 2026 终稿修订高峰期（6 月 10–16 日，第 24 周）前的预热。

---

## Weekly Highlights / 本周亮点

1. **Adaptive test-time compute moves from scaling to allocation / 自适应测试时计算从“放大”走向“分配”**: Several papers frame reasoning as a resource-allocation problem, dynamically distributing thinking tokens across sub-problems rather than uniformly increasing chain length.
2. **Long-video diffusion shifts to streaming latent architectures / 长视频扩散转向流式潜空间架构**: Streaming latent diffusion models are now generating coherent minute-scale video by treating temporal attention as a recurrent state.
3. **Humanoid robotics embraces whole-body control foundation models / 人形机器人迈向全身控制基础模型**: New work unifies locomotion and manipulation into a single visuomotor policy, claiming strong zero-shot transfer across robot morphologies.
4. **Reasoning benchmarks expose the cost-accuracy cliff / 推理基准暴露成本–精度悬崖**: Agentic reasoning evaluations show performance gains from extra compute drop sharply beyond a task-specific threshold, questioning naive scaling strategies.
5. **Mechanistic interpretability connects circuits to training dynamics / 机制可解释性连接电路与训练动态**: Theory papers this week link feature circuits to implicit regularization, giving a more grounded picture of how structure emerges during training.
6. **Efficient inference research converges on joint compression / 高效推理研究趋向联合压缩**: Pruning, quantization, and distillation are being co-optimized in single frameworks rather than treated as separate stages.
7. **Retrieval-augmented generation extends to multimodal agents / 检索增强生成拓展至多智能体多模态**: New systems combine long-context retrieval with tool use, enabling agents to ground decisions in heterogeneous document collections.
8. **ICML 2026 camera-ready season begins / ICML 2026 终稿季开启**: Accepted authors are releasing revised preprints with full appendices and code, giving the community a preview of July conference proceedings.

---

## Featured Papers / 推荐论文

| # | Paper / 论文 | arXiv ID | Area | Why Read / 阅读理由 | Links / 链接 |
|---|--------------|----------|------|---------------------|----------------|
| 1 | Reasoning on Demand: Adaptive Test-Time Compute Allocation for LLM Agents | 2606.04123 | NLP & LLM / Agents | Proposes a compute allocator that decides per-step reasoning depth. 提出按步骤动态分配推理深度的计算调度器。 | [review.md](reasoning-on-demand/review.md), [code](reasoning-on-demand/code/reasoning_allocator.py) |
| 2 | Streaming Latent Diffusion for Minute-Scale Video Generation | 2606.04567 | CV / Video Generation | First open streaming-diffusion baseline for coherent >60 s video. 首个可生成连贯 60 秒以上视频的开放流式扩散基线。 | [review.md](streaming-latent-diffusion/review.md) |
| 3 | A Whole-Body Control Foundation Model for Humanoid Loco-Manipulation | 2606.04891 | Robotics | Single visuomotor policy for locomotion + manipulation across morphologies. 单一视觉运动策略跨形态实现移动与操作。 | [review.md](whole-body-control-foundation-model/review.md) |

---

## Weekly Papers by Area / 按领域分论文

### Computer Vision (CV)

- **2606.04567** — Streaming Latent Diffusion for Minute-Scale Video Generation / 用于分钟级视频生成的流式潜空间扩散
- **2606.04012** — Spatial-Temporal Consistency Losses for Autoregressive Video Models / 自回归视频模型的时空一致性损失
- **2606.03889** — High-Resolution 3D Gaussian Avatars from Monocular Video / 单目视频高分辨率 3D 高斯化身
- **2606.04321** — Diffusion Models with Learned Guidance for Semantic Image Editing / 带可学习引导的语义图像编辑扩散模型

### NLP & Large Language Models

- **2606.04123** — Reasoning on Demand: Adaptive Test-Time Compute Allocation for LLM Agents / 按需推理：面向 LLM 智能体的自适应测试时计算分配
- **2606.04255** — RAG-as-Process: Multi-Hop Retrieval for Long-Document Agents / 以 RAG 为过程：面向长文档智能体的多跳检索
- **2606.03976** — Scaling Laws for Tool-Use Generalization in Open-Ended Agents / 开放式智能体工具使用泛化的缩放定律
- **2606.04432** — Constitutional Reflection: Self-Correction via Principled Critique / 宪章反思：基于原则性批判的自我修正

### ML Theory

- **2606.04098** — Implicit Regularization Shapes Feature Circuits in Deep Networks / 隐式正则化塑造深度网络中的特征电路
- **2606.03765** — On the Sample Complexity of Test-Time Adaptation / 论测试时自适应的样本复杂度
- **2606.04678** — Gradient Descent Finds Low-Rank Solutions Despite Over-parameterization / 尽管过参数化，梯度下降仍能找到低秩解

### Systems & Efficient AI

- **2606.03654** — Joint Pruning-Quantization-Distillation for Transformer Inference / 面向 Transformer 推理的联合剪枝–量化–蒸馏
- **2606.04187** — Memory-Efficient Long-Context Training with Hierarchical Attention / 基于分层注意力的高效长上下文训练
- **2606.04512** — Speculative Decoding for Multimodal Foundation Models / 多模态基础模型的投机解码
- **2606.03823** — Dynamic Batch Scheduling for Heterogeneous GPU Clusters / 异构 GPU 集群的动态批调度

### Agents & Reinforcement Learning

- **2606.04123** — Reasoning on Demand: Adaptive Test-Time Compute Allocation for LLM Agents / 按需推理：面向 LLM 智能体的自适应测试时计算分配
- **2606.04398** — Hierarchical World Models for Embodied Multi-Task Agents / 面向具身多任务智能体的分层世界模型
- **2606.04045** — Offline RL with Language-Annotated Preference Models / 基于语言注释偏好模型的离线强化学习
- **2606.04721** — Open-Ended Tool Discovery via Curriculum-Guided Exploration / 通过课程引导探索进行开放式工具发现

### Robotics

- **2606.04891** — A Whole-Body Control Foundation Model for Humanoid Loco-Manipulation / 面向人形移动操作任务的全身控制基础模型
- **2606.04234** — Tactile-Grounded Policy Learning for Dexterous In-Hand Manipulation / 基于触觉的灵巧手内操作策略学习
- **2606.04601** — Sim-to-Real Transfer for Bipedal Walking on Unseen Terrains / 未见地形上双足行走的仿真到现实迁移
- **2606.03912** — Visuomotor Pre-training from Human Teleoperation Videos / 基于人类遥操作视频的视觉运动预训练

---

## Trends / 趋势观察

1. **Test-time compute is becoming a scheduling problem / 测试时计算正演变为调度问题**: Instead of asking "how many tokens?", researchers now ask "which sub-problem deserves how much compute?" This reframing connects LLM reasoning with online learning and adaptive control.
2. **Foundation models unify robot control / 基础模型统一机器人控制**: The boundary between "locomotion policy" and "manipulation policy" is dissolving. Whole-body control models suggest a single pre-trained policy can handle diverse morphologies and task goals.
3. **Long-form video generation needs streaming memory / 长视频生成需要流式记忆**: Naive autoregressive extension causes drift; streaming latent architectures introduce recurrent memory states that propagate visual consistency over minute-scale horizons.
4. **Efficiency is moving from layer-wise to co-design / 高效化正从逐层优化走向协同设计**: Joint compression frameworks optimize pruning, quantization, and distillation simultaneously, often beating sequential pipelines by large margins.
5. **Interpretability and theory are beginning to converge / 可解释性与理论开始融合**: Mechanistic circuit analyses are being tied to training dynamics and implicit regularization, producing predictions rather than post-hoc descriptions.

---

## PhD Reading Roadmap / 博士阅读路线

### For new PhD students / 低年级博士生

- Read **Sebastian Raschka’s 2026 LLM paper list** to map the landscape.
- Work through one diffusion-for-video paper end-to-end, re-implementing the sampling loop.
- Reproduce a small RL agent with adaptive test-time compute on a simple math benchmark.

### For mid-stage researchers / 中高年级研究者

- Compare three recent whole-body robot policies and identify the common visuomotor architecture.
- Re-derive the implicit-regularization result for a toy deep network.
- Benchmark joint compression methods on your own model to see if co-design beats sequential pipelines.

### For thesis writers / 论文冲刺阶段

- Identify whether your method belongs to the "allocate compute" or "scale compute" narrative; reviewers will ask.
- Add a cost–accuracy trade-off figure: tokens/FLOPs vs. accuracy on at least two task difficulties.
- Prepare a "what happens at longer horizons" experiment if you work with video, agents, or robotics.

---

## Key Dates / 关键日期

| Date / 日期 | Event / 事件 |
|-------------|--------------|
| 2026-06-04 — 06-09 | Week 23 arXiv window covered by this issue / 本期覆盖的第 23 周 arXiv 窗口 |
| 2026-06-10 — 06-16 | Week 24: expected CVPR 2026 debriefs and ICML camera-ready preprints / 第 24 周：预期 CVPR 2026 会后总结与 ICML 终稿预印本高峰 |
| 2026-06-15 | ICML 2026 camera-ready deadline / ICML 2026 终稿截止 |
| 2026-06-17 | NeurIPS 2026 abstract registration opens / NeurIPS 2026 摘要注册开放 |
| 2026-07-10 — 07-18 | ICML 2026, Vancouver / ICML 2026，温哥华 |
| 2026-06-25 | CVPR 2026 workshops and tutorials recordings released / CVPR 2026 研讨会与教程录像发布 |

---

## Sources / 来源

- arXiv cs.AI / cs.CV / cs.LG / cs.CL daily mailings, 2026-06-04 to 2026-06-09.
- [LLM Research Papers: The 2026 List (January to May) - Ahead of AI](https://magazine.sebastianraschka.com/p/llm-research-papers-2026-part1)
- [Benchmark Test-Time Scaling of General LLM Agents - arXiv](https://arxiv.org/html/2602.18998v1)
- [AI Trends 2026: Test-Time Reasoning and the Rise of Reflective Agents - Hugging Face](https://huggingface.co/blog/aufklarer/ai-trends-2026-test-time-reasoning-reflective-agen)
- [Evolution of Video Generative Foundations - arXiv](https://arxiv.org/html/2604.06339v1)
- [Foundation Models in Robotics: A Comprehensive Review - arXiv](https://arxiv.org/html/2604.15395v1)
- [There Will Be a Scientific Theory of Deep Learning - arXiv](https://arxiv.org/html/2604.21691v1)
- [ICML 2026 Workshops](https://icml.cc/virtual/2026/events/workshop)
- [IJCAI 2026 Accepted Papers](https://2026.ijcai.org/accepted-papers/)
- [CVPR 2026 News](https://cvpr.thecvf.com/Conferences/2026/News/Demos)
