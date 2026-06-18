# Week 24: 2026-06-10 — 2026-06-16 / 第24周

---

## 本周学术动态 / Weekly Highlights

**中文：** 2026年第24周是CVPR 2026会议周（6月初于纳什维尔召开），最佳论文与奖项正式公布。本周核心亮点包括：(1) CVPR 2026最佳论文——Google DeepMind的D4RT动态4D场景重建；(2) "Thinking with Video"提出视频生成作为统一多模态推理新范式，Sora-2在推理基准上超越GPT-5；(3) Markovian Scale Prediction将VAR峰值内存降低83.8%；(4) SAM 3D获得最佳论文荣誉提名，实现单图到完整3D资产的一体化重建；(5) NitroGen开源游戏agent基础模型，在1000+游戏上训练。arXiv方面，世界模型（World Model）、VLA（Vision-Language-Action）和具身智能持续高热。

**English:** Week 24 of 2026 was CVPR 2026 conference week (early June in Nashville), with best paper awards officially announced. Key highlights: (1) CVPR 2026 Best Paper — D4RT dynamic 4D scene reconstruction from Google DeepMind; (2) "Thinking with Video" proposes video generation as a unified multimodal reasoning paradigm, with Sora-2 outperforming GPT-5 on reasoning benchmarks; (3) Markovian Scale Prediction reduces VAR peak memory by 83.8%; (4) SAM 3D receives Best Paper Honorable Mention, enabling end-to-end single-image-to-3D-asset reconstruction; (5) NitroGen open-source gaming agent foundation model trained on 1,000+ games. On arXiv, world models, VLA, and embodied intelligence remain highly active.

---

## 重点论文深度分析 / Featured Papers with Deep Reviews

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [Thinking with Video](https://arxiv.org/abs/2511.04570) | Multimodal Reasoning | [review.md](thinking-with-video/review.md) | — |
| 2 | [Markovian Scale Prediction](https://arxiv.org/abs/2511.23334) | Visual Generation | [review.md](markovian-scale-prediction/review.md) | [code/](markovian-scale-prediction/code/) |
| 3 | [SAM 3D](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_SAM_3D_3Dfy_Anything_in_Images_CVPR_2026_paper.html) | 3D Vision | [review.md](sam-3d/review.md) | — |

---

## 本周论文列表（按领域分类）/ Weekly Papers by Area

### 计算机视觉 / Computer Vision (CV)

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **D4RT: Efficiently Reconstructing Dynamic Scenes One D4RT at a Time** | CVPR 2026 Best Paper。轻量Transformer从视频重建动态4D场景几何与运动。 |
| **SAM 3D: 3Dfy Anything in Images** | CVPR 2026 Best Paper HM。单图预测3D形状、纹理、姿态，人工偏好5:1胜率。 |
| **Markovian Scale Prediction** | CVPR 2026。滑动窗口压缩VAR上下文，内存降83.8%，FID降10.5%。 |
| **Thinking with Video** | CVPR 2026。视频生成作为多模态推理范式，Sora-2 MMMU达69.2%。 |
| **UniTEX: Universal High Fidelity Generative Texturing for 3D Shapes** | 通用3D形状生成纹理 |
| **BrepGaussian: CAD reconstruction from Multi-View Images** | 多视图CAD重建+高斯溅射 |
| **GaussianZoom: Progressive Zoom-in Generative 3D Gaussian Splatting** | 渐进式变焦3DGS生成 |
| **Learning 3D Representations for Spatial Intelligence** | 无姿态多视图图像学3D表征 |
| **PhysicsGM: Large Physical Gaussian Model for 4D Synthesis** | 物理高斯模型前馈4D合成 |
| **BRDFusion: Physics Meets Generation for Urban Scene Inverse Rendering** | 城市场景逆渲染 |
| **Local-GS: Accelerating 3D Gaussian Splatting** | 瓦片局部warp加速3DGS |
| **HiGS: Hierarchical Rendering for Real-Time 3D Gaussian Splatting** | 分层实时3DGS渲染架构 |
| **Molmo2: Open Weights VLM with Video Understanding** | CVPR 2026 Best Paper HM。开源视频理解VLM |
| **ChordEdit: One-Step Low-Energy Transport for Image Editing** | CVPR 2026 Student HM。免训练反演快速图像编辑 |
| **A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens** | CVPR 2026 Best Paper。Delta Token高效世界建模 |

### 自然语言处理与大模型 / NLP & LLM

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **Latent Thought Flow: Efficient Latent Reasoning in LLMs** | 潜在空间高效推理 |
| **Scaling LLM Reasoning from Minimal Labels** | 半监督+轻量验证器扩展LLM推理 |
| **The Value Axis: Language Models Encode Whether They're on the Right Track** | 语言模型内部价值轴编码 |
| **KVEraser: Learning to Steer KV Cache for Efficient Localized Context Erasing** | KV Cache定向擦除 |
| **TokenPilot: Cache-Efficient Context Management for LLM Agents** | LLM Agent缓存高效上下文管理 |
| **DEEPRUBRIC: Evidence-Tree Rubric Supervision for Deep Research Agents** | 证据树rubric监督深度研究agent |
| **LESS Is More: Mutual-Stability Sampling for Diffusion Language Models** | 扩散语言模型互稳定采样 |
| **Speaking the Language of Science: General-Purpose Generative Foundation Model for Natural Sciences** | 自然科学通用生成基础模型 |

### 机器学习理论 / ML Theory

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **ExpRL: Exploratory RL for LLM Mid-Training** | 探索式RL用于LLM中期训练，发现紧凑动作集 |
| **Scalable Circuit Learning for Interpreting LLMs** | 可扩展电路学习解释大模型 |
| **Taming Curvature: Architecture Warm-Up for Stable Transformer Training** | Transformer架构预热稳定训练 |
| **RepNet: Tackling Spectral Bias via Parameter Reparameterization** | 参数重参数化解决谱偏置 |
| **Upper Bounds on Generalization Error via Local Robustness** | 局部鲁棒性泛化误差上界 |
| **Fantastic Pretraining Optimizers and Where to Find Them II** | 预训练优化器新进展 |

### 智能体与强化学习 / Agents & RL

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **NitroGen: Open Foundation Model for Generalist Gaming Agents** | CVPR 2026 Best Paper HM。1000+游戏40000小时训练，任务成功率提升52%。 |
| **HiF-VLA: Hindsight, Insight and Foresight for VLA Models** | VLA模型 hindsight+insight+foresight |
| **SAGE: Training Smart Any-Horizon Agents for Long Video Reasoning with RL** | RL训练长视频推理任意 horizon agent |
| **WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning** | 动态多模态记忆agent长视频推理 |
| **MyPCBench: Benchmark for Personally Intelligent Computer-Use Agents** | 个性化智能电脑使用agent基准 |
| **CoffeeBench: Benchmarking Long-Horizon LLM Agents in Multi-Agent Economies** | 多智能体经济长程LLM agent基准 |
| **OpenClaw-Skill: Collective Skill Tree Search for Agentic LLMs** | 集体技能树搜索 |
| **AgentFairBench: Do LLM Agents Discriminate When They Act?** | LLM Agent公平性评测 |
| **When in Doubt, Plan It Out: SLM Deliberation for Reactive RL** | 小语言模型deliberation用于反应式RL |

### 机器人学与具身智能 / Robotics & Embodied AI

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **Qwen-RobotWorld Technical Report** | 通义千问机器人世界模型，语言条件视频生成统一具身世界建模 |
| **DreamX-World 1.0: General-Purpose Interactive World Model** | 通用交互式世界模型 |
| **MeshLoom: Feed-Forward Non-Rigid Registration of Mesh Sequences** | 前馈非刚性网格序列配准 |
| **SurroundNEXO: Ego-Centric Metric Bridging for Autonomous Driving** | 自动驾驶自中心度量桥接 |
| **StressDream: Steering Video World Models for Policy Evaluation** | 引导视频世界模型进行策略评估 |

### 系统与多模态 / Systems & Multimodal

| 论文 / Paper | 要点 / Key Point |
|---|---|
| **OmniSonic: Universal Audio Generation from Video and Text** | 视频+文本统一音频生成 |
| **MMDiff: Extending Diffusion Transformers for Multi-Modal Generation** | 扩散Transformer多模态生成扩展 |
| **UniM: Unified Any-to-Any Interleaved Multimodal Benchmark** | 统一任意模态交错基准 |
| **Zamba2-VL Technical Report** | Zamba2-VL技术报告 |

---

## 趋势观察 / Trends

**中文：**

1. **视频生成=推理引擎? / Video Generation = Reasoning Engine?**
   "Thinking with Video"代表了多模态推理范式的重大转变——从将视频视为"需要理解的内容"，转向将视频视为"思考过程本身的载体"。如果这一范式得到验证，将对视觉基础模型的训练目标产生深远影响。但成本与可及性仍是最大障碍。

2. **VAR加速进入实用阶段 / VAR Acceleration Enters Practical Stage**
   Markovian Scale Prediction将VAR的内存开销降到消费级GPU可承受范围，这意味着自回归视觉生成可能真正挑战扩散模型在图像生成领域的主导地位。关注后续是否会出现基于MSP的开源高分辨率图像生成器。

3. **3D重建的"SAM时刻" / The "SAM Moment" for 3D Reconstruction**
   SAM 3D试图成为3D领域的SAM——任意图像输入，统一3D输出。与SAM类似，其核心挑战在于开放世界的泛化能力和细粒度可控性。若开源生态建设成功，将极大降低3D内容创作门槛。

4. **通用游戏agent浮出水面 / Generalist Gaming Agents Emerge**
   NitroGen在1000+游戏上的训练证明了跨游戏泛化的可行性。游戏作为具身智能的沙盒，其agent能力可能快速迁移到真实世界机器人控制。

5. **世界模型竞赛白热化 / World Model Competition Intensifies**
   Qwen-RobotWorld、DreamX-World、PhysicsGM、StressDream等世界模型工作密集出现，表明世界建模正从研究概念走向可部署系统。

**English:**

1. **Video Generation = Reasoning Engine?**
   "Thinking with Video" represents a paradigm shift in multimodal reasoning — from viewing video as "content to be understood" to viewing video as "the carrier of the thinking process itself." If validated, this will profoundly impact the training objectives of visual foundation models. Cost and accessibility remain the biggest barriers.

2. **VAR Acceleration Enters Practical Stage**
   Markovian Scale Prediction reduces VAR memory overhead to consumer-GPU-tolerable levels, meaning autoregressive visual generation may genuinely challenge diffusion models' dominance in image generation. Watch for open-source high-resolution image generators based on MSP.

3. **The "SAM Moment" for 3D Reconstruction**
   SAM 3D aims to become the SAM of 3D — any image in, unified 3D out. Like SAM, its core challenges are open-world generalization and fine-grained controllability. If the open-source ecosystem succeeds, it will dramatically lower the barrier to 3D content creation.

4. **Generalist Gaming Agents Emerge**
   NitroGen's training on 1,000+ games demonstrates the feasibility of cross-game generalization. As embodied intelligence sandboxes, gaming agent capabilities may rapidly transfer to real-world robot control.

5. **World Model Competition Intensifies**
   Dense appearance of world model works (Qwen-RobotWorld, DreamX-World, PhysicsGM, StressDream) shows world modeling is moving from research concept to deployable systems.

---

## 博士生本周阅读路线建议 / PhD Reading Roadmap for This Week

**中文：**

**如果你做视觉生成 / If you work on visual generation:**
> 精读 Markovian Scale Prediction (CVPR 2026) → 运行本仓库中的 [markovian_var.py](markovian-scale-prediction/code/markovian_var.py) 复现代码 → 对比原始VAR论文中的实现细节 → 思考：你的生成任务是否可以从马尔可夫近似中受益？

**如果你做多模态推理/LLM / If you work on multimodal reasoning or LLM:**
> 精读 Thinking with Video (CVPR 2026) → 查看 VideoThinkBench 的构造逻辑 → 思考：视频推理的 cost-benefit 分析 → 延伸阅读: Visual CoT, Sora技术报告

**如果你做3D视觉/具身智能 / If you work on 3D vision or embodied AI:**
> 精读 SAM 3D (CVPR 2026) + NitroGen (CVPR 2026) → 对比单图3D重建的不同技术路线 (SAM 3D vs. Zero-1-to-3 vs. Shap-E) → 关注 Qwen-RobotWorld 和 DreamX-World 中世界模型的表示学习方案

**如果你做ML理论 / If you work on ML theory:**
> 精读 ExpRL (arXiv) 或 Scalable Circuit Learning → 前者关注RL mid-training的理论分解，后者关注LLM可解释性的电路学习方法

**本周必读 (All areas) / Must-read this week (All areas):**
1. Markovian Scale Prediction — 有代码复现，理解成本低，技术优雅
2. Thinking with Video —  paradigm-level 的工作，可能定义未来一年的研究方向
3. SAM 3D 或 NitroGen (二选一，根据你的细分方向) — 了解CVPR最佳论文的水准线

---

## 重要时间节点 / Key Dates

| 事件 / Event | 日期 / Date | 状态 / Status |
|---|---|---|
| CVPR 2026 Conference | Jun 8–15, 2026 | 刚结束 / Just concluded |
| ICML 2026 Conference | Jul 2026 | 即将召开 / Upcoming |
| NeurIPS 2026 Submission Deadline | May 2026 (已截止 / Closed) | 审稿中 / Under review |
| ECCV 2026 Submission Deadline | Mar 2026 (已截止 / Closed) | 审稿中 / Under review |
| AAAI 2027 Submission Deadline | Aug 2026 | 准备中 / Preparation |

---

## 数据来源 / Sources

- [CVPR 2026 Papers](https://cvpr.thecvf.com/virtual/2026/papers.html)
- [CVPR 2026 Best Papers](https://cvpr.thecvf.com/Conferences/2026/News/Best_Papers)
- [arXiv cs.CV Recent](https://arxiv.org/list/cs.CV/recent)
- [arXiv cs.LG Recent](https://arxiv.org/list/cs.LG/recent)
- [arXiv cs.AI Recent](https://arxiv.org/list/cs.AI/recent)
- [Paper Digest: CVPR 2026 Highlights](https://www.paperdigest.org/2026/04/cvpr-2026-papers-highlights/)
- [SkalskiP/top-cvpr-2026-papers](https://github.com/SkalskiP/top-cvpr-2026-papers)
