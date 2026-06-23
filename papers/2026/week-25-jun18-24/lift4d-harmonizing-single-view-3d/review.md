# Lift4D: Harmonizing Single-View 3D Estimation for 4D Reconstruction In-the-Wild

> **arXiv:** [2606.23688](https://arxiv.org/abs/2606.23688) | **日期 / Date:** 2026-06-22 | **作者 / Authors:** Yehonathan Litman, Xiaoxuan Ma, Manan Shah, Nicolas Ugrinovic, Kris Kitani, Fernando De la Torre, Shubham Tulsiani

---

## 关键摘录 / Key Excerpts

> 1. "Reconstructing dynamic non-rigid objects from monocular video requires integrating visual cues from direct observations with data-driven priors over geometry and appearance."
>    / "从单目视频中重建动态非刚性物体，需要将直接观察到的视觉线索与关于几何和外观的数据驱动先验相结合。"

> 2. "Prior approaches either learn to directly predict 4D representations from visual input or initialize a 3D representation that is subsequently deformed and refined based on video evidence."
>    / "先前的方法要么学习直接从视觉输入预测 4D 表示，要么初始化一个 3D 表示，随后根据视频证据进行变形和细化。"

> 3. "However, the former are constrained by the scarcity of 4D training data, while the latter leverage priors only for the initial reconstruction and rely solely on video supervision thereafter; neither handles complex in-the-wild scenarios with large deformations and occlusions well."
>    / "然而，前者受限于 4D 训练数据的稀缺，后者仅在初始重建时利用先验，之后完全依赖视频监督；两者都无法很好地处理具有大形变和遮挡的复杂野外场景。"

> 4. "We present Lift4D, a test-time optimization framework that addresses both limitations."
>    / "我们提出 Lift4D，一种在测试时优化的框架，同时解决了上述两个局限性。"

> 5. "First, we adapt an existing single-view 3D reconstruction model to yield temporally consistent per-frame predictions via causal latent conditioning, providing a coherent initialization for a deformable 3D Gaussian Splatting representation."
>    / "首先，我们调整现有的单视图 3D 重建模型，通过因果潜在条件化产生时间一致的逐帧预测，为可变形 3D 高斯溅射表示提供连贯的初始化。"

> 6. "We then 'sculpt' this representation to match the input video through an occlusion-aware optimization that faithfully recovers visible surface details while completing unobserved regions using a view-conditioned diffusion prior."
>    / "然后，我们通过一种遮挡感知的优化来‘雕刻’这一表示以匹配输入视频，在忠实地恢复可见表面细节的同时，利用视图条件化的扩散先验补全未观察区域。"

> 7. "Lift4D clearly improves over prior 4D reconstruction methods."
>    / "Lift4D 明显优于先前的 4D 重建方法。"

---

## Q1: 核心问题 / Core Problem

**中文：**

从单目视频中重建动态非刚性物体的 4D（3D + 时间）几何与外观，是计算机视觉中长期存在的难题。现有两条技术路线各有致命弱点：

1. **端到端 4D 预测方法**：直接从视频学习 4D 表示。这类方法受限于 4D 训练数据的稀缺性，难以泛化到野外（in-the-wild）真实视频。
2. **3D 初始化 + 视频驱动变形方法**：先用单帧或多帧初始化 3D 表示，再用视频监督进行变形和细化。这类方法只在初始化时使用数据驱动先验，之后完全依赖视频监督，无法处理大形变、遮挡和未观察区域。

因此，核心问题是：**如何在测试时联合利用单视图 3D 重建先验、视频时序一致性和扩散模型补全能力，实现野外动态物体的真实感 4D 重建？**

**English:**

Reconstructing the 4D (3D + time) geometry and appearance of dynamic non-rigid objects from monocular video is a long-standing challenge in computer vision. Existing technical routes each have critical weaknesses:

1. **End-to-end 4D prediction methods**: learn 4D representations directly from video. These are constrained by the scarcity of 4D training data and struggle to generalize to real in-the-wild videos.
2. **3D initialization + video-driven deformation methods**: initialize a 3D representation from one or a few frames, then deform and refine it using video supervision. These leverage data-driven priors only at initialization and rely entirely on video supervision afterward, failing to handle large deformations, occlusions, and unobserved regions.

The core problem is therefore: **how can we jointly leverage single-view 3D reconstruction priors, temporal video consistency, and diffusion-model completion capabilities at test time to achieve photorealistic 4D reconstruction of dynamic objects in the wild?**

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **Lift4D 测试时优化框架**：首次将单视图 3D 先验、时序一致性和扩散补全统一在一个测试时优化框架中。
2. **因果潜在条件化**：改造单视图 3D 重建模型，使其在逐帧预测时利用因果潜在条件，保证时间连贯性。
3. **可变形 3D 高斯溅射表示**：用稀疏控制节点参数化时变可变形 3D 高斯，兼顾表达能力和计算效率。
4. **遮挡感知渲染监督**：在优化过程中正确处理遮挡区域，避免用被遮挡像素的错误监督破坏几何。
5. **视图条件扩散先验**：利用图像扩散模型补全未观察区域，解决大形变和遮挡导致的几何缺失问题。
6. **SOTA 4D 重建质量**：在 Consistent4D 合成数据集和 Pexels/DAVIS 野外视频上取得领先的 LPIPS / FVD / CLIP / EPE 指标。

**English:**

1. **Lift4D test-time optimization framework**: The first to unify single-view 3D priors, temporal consistency, and diffusion completion in a single test-time optimization framework.
2. **Causal latent conditioning**: Adapts a single-view 3D reconstruction model to use causal latent conditioning for per-frame predictions, ensuring temporal coherence.
3. **Deformable 3D Gaussian Splatting representation**: Parameterizes time-varying deformable 3D Gaussians with sparse control nodes, balancing expressiveness and computational efficiency.
4. **Occlusion-aware rendering supervision**: Correctly handles occluded regions during optimization, preventing erroneous supervision from corrupted pixels.
5. **View-conditioned diffusion prior**: Uses an image diffusion model to complete unobserved regions, addressing geometry loss caused by large deformations and occlusions.
6. **SOTA 4D reconstruction quality**: Achieves leading LPIPS / FVD / CLIP / EPE metrics on the Consistent4D synthetic dataset and on in-the-wild Pexels/DAVIS videos.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **优化时间与计算成本**：测试时优化通常很慢，Lift4D 每段视频需要优化多久？是否比先前方法慢一个数量级？
2. **扩散先验的忠实性**：视图条件扩散先验可能生成视觉上合理但几何不正确的内容，如何验证补全区域的真实 3D 一致性？
3. **对单视图重建模型的依赖**：如果底层单视图模型在野外数据上失败（如 unusual pose、极端遮挡），Lift4D 的初始化质量会严重下降。
4. **训练数据与评估的覆盖度**：Pexels/DAVIS 视频虽然 wild，但是否包含足够多样的运动模式（如快速旋转、非人类物体、流体）？
5. **稀疏控制节点的表达能力**：对于高度复杂的非刚性形变（如衣服褶皱、头发），稀疏控制节点是否足以表达细节？
6. **与 video diffusion / 4D Gaussian 方法的公平比较**： reviewer 会质疑与 DreamGaussian4D、4DGen 等最新方法的比较是否使用了相同的数据预处理和评估协议。

**English:**

1. **Optimization time and computational cost**: Test-time optimization is usually slow. How long does Lift4D take per video? Is it an order of magnitude slower than prior methods?
2. **Fidelity of the diffusion prior**: The view-conditioned diffusion prior may generate visually plausible but geometrically incorrect content. How is true 3D consistency of completed regions validated?
3. **Dependence on the single-view reconstruction model**: If the underlying single-view model fails on wild data (e.g., unusual poses, extreme occlusion), Lift4D's initialization quality drops sharply.
4. **Coverage of training data and evaluation**: Although Pexels/DAVIS videos are in-the-wild, do they contain sufficiently diverse motion patterns (fast rotation, non-human objects, fluids)?
5. **Expressiveness of sparse control nodes**: For highly complex non-rigid deformations (clothing wrinkles, hair), are sparse control nodes expressive enough?
6. **Fair comparison with video diffusion / 4D Gaussian methods**: Reviewers will question whether comparisons with recent methods such as DreamGaussian4D and 4DGen use identical data preprocessing and evaluation protocols.

---

## Q4: 博士生阅读指南 / PhD Reading Guide

**中文：**

**如果你是做 3D 视觉 / 神经渲染 / 视频理解的博士生：**

- **精读部分**：
  - 第 1–2 页的问题定义与两种 prior approach 的对比；
  - 方法部分：因果潜在条件化的具体实现、可变形 3DGS 的参数化、遮挡感知损失、扩散先验的引入方式；
  - 实验部分：定量指标定义、与 Consistent4D / Pexels / DAVIS 上 baselines 的对比。
- **跳过部分**：
  - 若已熟悉 3D Gaussian Splatting，可跳过相关背景；
  - 若对单视图重建（如 LRM、InstantMesh）不熟悉，可先阅读这些前置工作。
- **复现建议**：
  - 先用现有单视图重建模型（如 LRM、Wonder3D）生成逐帧 3D 初始化；
  - 实现一个简化的可变形 3DGS，用相邻帧的位姿和光度损失做时序约束；
  - 引入一个预训练的图像扩散模型（如 Stable Diffusion）作为视图条件先验；
  - 重点观察：去掉扩散先验后，遮挡区域是否出现明显伪影？
- **可延伸的研究点**：
  - 将 Lift4D 的测试时优化思想扩展到多视角视频输入；
  - 用 video diffusion model（如 SVD、CogVideo）替代图像扩散先验，探索时间一致性更强的补全。

**English:**

**If you are a PhD student working on 3D vision, neural rendering, or video understanding:**

- **Read carefully**:
  - Pages 1–2: problem definition and comparison of prior approaches;
  - Method section: implementation of causal latent conditioning, parameterization of deformable 3DGS, occlusion-aware loss, and how the diffusion prior is introduced;
  - Experiments: metric definitions and quantitative comparisons with baselines on Consistent4D, Pexels, and DAVIS.
- **Skip if familiar**:
  - Background on 3D Gaussian Splatting if already known;
  - Single-view reconstruction preliminaries (e.g., LRM, InstantMesh) if already familiar.
- **Reproduction suggestions**:
  - First use an existing single-view reconstruction model (e.g., LRM, Wonder3D) to generate per-frame 3D initializations;
  - Implement a simplified deformable 3DGS with temporal constraints from neighboring frames via pose and photometric losses;
  - Introduce a pretrained image diffusion model (e.g., Stable Diffusion) as a view-conditioned prior;
  - Key observation: when the diffusion prior is removed, do occluded regions show obvious artifacts?
- **Potential research extensions**:
  - Extend Lift4D's test-time optimization idea to multi-view video input;
  - Replace the image diffusion prior with a video diffusion model (e.g., SVD, CogVideo) to explore stronger temporal consistency in completion.
