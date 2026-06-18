# DepthDirector: Beyond Inpainting — Unleash 3D Understanding for Precise Camera-Controlled Video Generation

> **arXiv:** [2601.10214](https://arxiv.org/abs/2601.10214) | **日期 / Date:** 2026-01-15 | **作者 / Authors:** Dong-Yu Chen, Yixin Guo, Shuojin Yang, Tai-Jiang Mu, Shi-Min Hu

---

## 关键摘录 / Key Excerpts

> 1. "Current camera-controlled video generation methods treat the problem as inpainting — filling in disoccluded regions as new viewpoints are revealed. We argue this is fundamentally insufficient: true camera control requires genuine 3D understanding."
>    / "当前的相机控制视频生成方法将问题视为修补——在新视角揭示时填补被遮挡区域。我们认为这根本不够：真正的相机控制需要真正的3D理解。"

> 2. "Our dual-stream motion perception architecture separately models camera ego-motion and object dynamics, enabling precise control that is impossible when the two are entangled in a single representation."
>    / "我们的双流运动感知架构分别建模相机自身运动和物体动态，实现了在两种运动纠缠于单一表征时无法实现的精准控制。"

> 3. "The 8K synthetic dataset generated from Unreal Engine provides the first large-scale, geometrically-accurate training resource for camera-controlled video generation, with ground-truth camera parameters and depth maps."
>    / "从Unreal Engine生成的8K合成数据集是首个大规模、几何精确的相机控制视频生成训练资源，提供了真实的相机参数和深度图。"

---

## Q1: 核心问题 / Core Problem

**中文：**
本文要解决的核心问题是：如何在视频生成中实现精确的相机轨迹控制？

现有方法的根本局限在于将相机控制视为2D修补问题——当相机移动时，只需要"填补"新出现的区域。这种方法在以下场景失败：
- 大范围相机移动导致的严重遮挡/去遮挡
- 需要一致性3D结构的场景旋转
- 复杂的相机轨迹（如环绕、俯仰组合运动）

DepthDirector提出：真正的相机控制需要显式的3D理解，即模型需要理解场景的几何结构，而不仅仅是像素级的模式匹配。

这涉及几个子问题：
- 如何在扩散模型中有效利用深度信息？
- 如何分离相机运动和物体运动？
- 如何获取大规模、几何精确的训练数据？

**English:**
The core problem: how to achieve precise camera trajectory control in video generation?

The fundamental limitation of existing methods is treating camera control as a 2D inpainting problem — when the camera moves, only "fill in" newly revealed regions. This approach fails in scenarios involving:
- Severe occlusion/disocclusion from large camera movements
- Scene rotations requiring consistent 3D structure
- Complex camera trajectories (e.g., combined orbit and pitch movements)

DepthDirector argues: true camera control requires explicit 3D understanding — the model must understand scene geometry, not just pixel-level pattern matching.

This involves several sub-problems:
- How to effectively leverage depth information in diffusion models?
- How to disentangle camera motion from object motion?
- How to obtain large-scale, geometrically accurate training data?

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **双流运动感知架构 (Dual-Stream Motion Perception Architecture)**：
   - 独立的相机运动流：专门处理相机位姿变化对视觉内容的影响
   - 独立的物体动态流：处理场景中物体的自主运动
   - 交叉注意力融合机制：在需要时允许两个流交换信息
   - 这种分离使模型能够独立控制相机而不影响物体运动的连贯性

2. **深度引导生成管线 (Depth-Guided Generation Pipeline)**：
   - 显式深度图作为条件输入，而非隐式学习
   - 深度一致性约束确保生成视频在3D空间中的连贯性
   - 深度感知注意力机制：利用深度信息引导空间注意力的分配

3. **LoRA高效微调模块 (LoRA Efficient Fine-tuning Module)**：
   - 针对视频生成模型的定制LoRA设计
   - 选择性适配器注入：仅在运动相关层添加LoRA
   - 显著降低训练成本（据报告减少约60%的GPU时间）

4. **8K合成数据集 (8K Synthetic Dataset)**：
   - 使用Unreal Engine 5生成
   - 包含真实相机参数、深度图和法线图
   - 涵盖多样化的场景类型和相机轨迹
   - 据作者称是同类数据集中最大规模

**English:**

1. **Dual-Stream Motion Perception Architecture**:
   - Independent camera motion stream: handles camera pose changes' effect on visual content
   - Independent object dynamics stream: handles autonomous object motion in the scene
   - Cross-attention fusion mechanism: allows information exchange between streams when needed
   - This separation enables independent camera control without affecting object motion coherence

2. **Depth-Guided Generation Pipeline**:
   - Explicit depth maps as conditioning input rather than implicit learning
   - Depth consistency constraints ensure 3D spatial coherence in generated video
   - Depth-aware attention: uses depth information to guide spatial attention allocation

3. **LoRA Efficient Fine-tuning Module**:
   - Custom LoRA design for video generation models
   - Selective adapter injection: only adds LoRA to motion-relevant layers
   - Significantly reduces training cost (reported ~60% GPU time reduction)

4. **8K Synthetic Dataset**:
   - Generated using Unreal Engine 5
   - Includes ground-truth camera parameters, depth maps, and normal maps
   - Covers diverse scene types and camera trajectories
   - Claimed to be the largest dataset of its kind

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **合成数据到真实视频的迁移 (Synthetic-to-Real Transfer)**：
   - 在Unreal Engine合成数据上训练的模型，在真实世界视频上的泛化能力如何？
   - 合成数据的域差距是否限制了实际应用场景？
   - 是否需要在真实数据上进行额外的微调？成本如何？

2. **深度估计的误差传播 (Depth Estimation Error Propagation)**：
   - 在实际应用中，深度图通常来自估计而非真实值
   - 深度估计误差如何影响最终视频质量？
   - 模型对深度噪声的鲁棒性如何？

3. **与最新方法的对比公平性 (Fairness of Comparisons)**：
   - 比较的方法是否使用了类似的预训练权重？
   - 合成数据优势是否来自数据量而非方法创新？
   - 是否在多个基准上进行了评估？

4. **计算成本 (Computational Cost)**：
   - 双流架构是否显著增加了推理时间？
   - 深度图计算/估计的额外开销
   - LoRA虽然减少了训练成本，但推理效率如何？

5. **相机控制的精度量化 (Quantification of Camera Control Precision)**：
   - 如何客观评估相机控制的精确度？
   - 仅靠视觉质量和FID/FVD分数是否足够？
   - 需要更细粒度的相机参数误差度量

**English:**

1. **Synthetic-to-Real Transfer**: How well does the model trained on Unreal Engine synthetic data generalize to real-world video? Does the synthetic data domain gap limit practical applications? Is additional fine-tuning on real data needed, and at what cost?

2. **Depth Estimation Error Propagation**: In practice, depth maps come from estimation, not ground truth. How do depth estimation errors affect final video quality? How robust is the model to depth noise?

3. **Fairness of Comparisons**: Do compared methods use similar pretrained weights? Is the synthetic data advantage from data volume rather than methodological innovation? Evaluation on multiple benchmarks?

4. **Computational Cost**: Does the dual-stream architecture significantly increase inference time? Additional overhead of depth map computation/estimation? LoRA reduces training cost but what about inference efficiency?

5. **Camera Control Precision Quantification**: How to objectively evaluate camera control precision? Are visual quality and FID/FVD scores sufficient? Finer-grained camera parameter error metrics needed.

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**推荐阅读顺序 / Recommended Reading Order:**
1. 第3节（方法）——理解双流架构和深度引导机制
2. 第4节（数据集）——了解合成数据生成流程
3. 第5节（实验）——重点关注消融实验
4. 第2节（相关工作）——理解相机控制视频生成的技术脉络

**关键方法论需要掌握 / Key Methodology to Master:**
- 扩散模型条件生成：如何将深度图和相机参数作为条件注入
- 双流架构设计：解耦运动表征的通用方法
- LoRA微调：参数高效微调的原理和实践
- Unreal Engine数据管线：合成数据生成的工程实践

**潜在研究方向 / Potential Research Directions:**
- 将深度引导扩展到多视角一致性视频生成
- 研究自监督深度估计与视频生成的联合训练
- 探索深度引导在3D高斯泼溅 (3D Gaussian Splatting) 中的应用
- 将双流运动分离思想扩展到人机交互场景

**English:**

**Recommended Reading Order:**
1. Section 3 (Method) — understand dual-stream architecture and depth-guidance mechanism
2. Section 4 (Dataset) — synthetic data generation pipeline
3. Section 5 (Experiments) — focus on ablation studies
4. Section 2 (Related Work) — technical lineage of camera-controlled video generation

**Key Methodology to Master:**
- Conditional generation in diffusion models: injecting depth maps and camera parameters
- Dual-stream architecture design: general approach to disentangling motion representations
- LoRA fine-tuning: principles and practice of parameter-efficient fine-tuning
- Unreal Engine data pipeline: engineering practices for synthetic data generation

**Potential Research Directions:**
- Extend depth guidance to multi-view consistent video generation
- Joint training of self-supervised depth estimation and video generation
- Explore depth guidance in 3D Gaussian Splatting
- Extend dual-stream motion separation to human-robot interaction scenarios
