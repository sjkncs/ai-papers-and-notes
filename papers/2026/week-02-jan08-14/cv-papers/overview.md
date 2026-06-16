# 第2周计算机视觉论文综述 / Week 2 CV Papers Overview

> **时间范围 / Period:** 2026-01-08 — 2026-01-14  
> **整理日期 / Compiled:** January 15, 2026

---

## 3D与视频理解 / 3D & Video Understanding

### DepthDirector 前驱工作 / DepthDirector Precursor Papers

**中文：** 本周多篇论文推进了单目深度估计和3D场景理解的前沿。虽然DepthDirector本身尚未发布，但本周出现了几个重要的前驱工作：基于扩散模型的单目深度估计在精度上超越了传统回归方法，同时保持了推理效率。新的深度先验学习方法通过大规模视频预训练获得了更强的几何理解能力。

**English:** Several papers this week advanced monocular depth estimation and 3D scene understanding. While DepthDirector itself hasn't been released yet, important precursor works appeared: diffusion-based monocular depth estimation surpasses traditional regression methods in accuracy while maintaining inference efficiency. Novel depth prior learning approaches obtain stronger geometric understanding through large-scale video pretraining.

### 视频生成模型在机器人中的应用综述 / Video Generation Models in Robotics Survey

**中文：** 本周发布的一篇综述系统性地回顾了视频生成模型在机器人世界建模中的应用。核心观点是：视频生成模型（如Sora类模型）可以作为机器人策略的"想象力引擎"，通过生成可能的未来状态来辅助决策。综述涵盖了从世界模型（world models）到视频预测驱动的规划（video-predictive planning）的完整技术栈，并指出了当前方法在物理一致性和实时性方面的主要瓶颈。

**English:** A survey published this week systematically reviewed video generation models' applications in robotic world modeling. The core thesis: video generation models (Sora-class models) can serve as "imagination engines" for robotic policies, generating possible future states to assist decision-making. The survey covers the full stack from world models to video-predictive planning, identifying key bottlenecks in physical consistency and real-time performance.

---

## 多模态评估 / Multimodal Evaluation

### BabyVision: 婴儿级视觉推理基准 / BabyVision: Infant-Level Visual Reasoning Benchmark

> 详细审稿见 / Detailed review: [babyvision/review.md](../babyvision/review.md)

**中文：** BabyVision是本周最具启发性的工作之一。它构建了一个基于发展心理学实验的视觉推理基准，揭示了当前MLLMs的"倒置能力画像"——在婴儿能轻松完成但需要纯视觉推理（非语言能力）的任务上，GPT-4V和Gemini Ultra仅达到31-42%的准确率。这一发现对多模态AI的发展方向提出了根本性质疑。

**English:** BabyVision is one of the most thought-provoking works this week. It constructs a visual reasoning benchmark based on developmental psychology experiments, revealing MLLMs' "inverted competence profile" — on tasks that babies easily solve but require pure visual reasoning (not language ability), GPT-4V and Gemini Ultra achieve only 31-42% accuracy. This finding raises fundamental questions about the development direction of multimodal AI.

### 相关评估基准 / Related Evaluation Benchmarks

**中文：** 本周还有几项值得关注的多模态评估工作：
- **VLM-Probe:** 一种新的探针方法，可以诊断VLM在特定视觉概念上的理解深度，而不仅仅是准确率。
- **MultiHop-VQA:** 需要多步推理的视觉问答基准，测试模型是否真正理解而非简单关联。
- **SpatialBench:** 专注于空间推理的视觉基准，涵盖遮挡关系、相对位置和导航推理。

**English:** Several other notable multimodal evaluation works appeared this week:
- **VLM-Probe:** A novel probing method that diagnoses VLMs' depth of understanding on specific visual concepts, beyond mere accuracy.
- **MultiHop-VQA:** A visual QA benchmark requiring multi-step reasoning, testing whether models truly understand versus simple association.
- **SpatialBench:** A visual benchmark focused on spatial reasoning, covering occlusion relations, relative positioning, and navigation reasoning.

---

## 视觉推理 / Visual Reasoning

### STEP3-VL-10B: 紧凑模型的推理能力 / STEP3-VL-10B: Reasoning in Compact Models

> 详细审稿见 / Detailed review: [step3-vl-10b/review.md](../step3-vl-10b/review.md)

**中文：** STEP3-VL-10B挑战了"推理能力需要大参数量"的传统认知。通过分阶段训练课程和链式思维蒸馏，这个100亿参数的开源模型在数学推理、图表理解和科学图示分析上达到了与70B+模型相当的水平。其核心洞察是：推理能力并非参数量的严格单调函数——正确的训练策略可以部分解耦推理能力与模型规模。

**English:** STEP3-VL-10B challenges the conventional belief that reasoning requires massive parameter counts. Through a staged training curriculum and chain-of-thought distillation, this 10-billion parameter open-source model reaches 70B+ model levels on mathematical reasoning, chart understanding, and scientific diagram analysis. Its core insight: reasoning capability is not strictly a monotonic function of parameter count — the right training strategy can partially decouple reasoning from model scale.

### 视觉理解的其他进展 / Other Visual Understanding Advances

**中文：** 本周还出现了几个重要的视觉理解工作：
- **细粒度视觉分类的新方法：** 利用多尺度特征融合和注意力引导，在CUB-200和Stanford Cars上刷新了SOTA。
- **开放词汇目标检测：** 基于CLIP的开放词汇检测器在效率上取得了显著进展，推理速度提升了2.3倍。
- **视觉基础模型的领域适应：** 参数高效的领域适应方法使视觉基础模型可以快速适配医疗影像、卫星图像等专业领域。

**English:** Several other important visual understanding works appeared this week:
- **Novel fine-grained visual classification:** Using multi-scale feature fusion and attention guidance, achieving new SOTA on CUB-200 and Stanford Cars.
- **Open-vocabulary object detection:** CLIP-based open-vocabulary detectors made significant efficiency gains, with 2.3x inference speedup.
- **Domain adaptation for visual foundation models:** Parameter-efficient domain adaptation methods enable rapid adaptation of visual foundation models to specialized domains like medical imaging and satellite imagery.

---

## 安全与对齐 / Safety & Alignment

### Constitutional Classifiers++ / 宪法分类器++

**中文：** Constitutional Classifiers++将激活探针（activation probing）技术推向了生产就绪的水平。该工作提出了一种系统性的方法来训练轻量级分类器，基于模型内部激活来检测越狱（jailbreak）攻击。与前代工作相比，CC++的关键改进包括：(1) 在部署时的零额外延迟开销，(2) 对未见过的越狱方法的泛化能力，(3) 与主流推理框架的无缝集成。论文报告了在98.7%的越狱检测率下仅0.3%的误报率。

**English:** Constitutional Classifiers++ brings activation probing technology to production readiness. This work proposes a systematic method for training lightweight classifiers that detect jailbreak attacks based on model internal activations. Key improvements over prior work include: (1) zero additional latency overhead at deployment, (2) generalization to unseen jailbreak methods, (3) seamless integration with mainstream inference frameworks. The paper reports 98.7% jailbreak detection rate with only 0.3% false positive rate.

### 其他安全论文 / Other Safety Papers

**中文：**
- **推理时扩散模型对齐：** 基于Doob's Matching的方法实现了在不重新训练的前提下，在推理阶段引导扩散模型生成符合特定约束（如安全性、风格一致性）的图像。
- **多模态安全评估：** 新的基准测试评估了VLM在面对对抗性视觉输入时的鲁棒性，发现当前模型在视觉对抗攻击下仍然脆弱。
- **水印与溯源：** 针对AI生成内容的新型水印方法，在保持图像质量的同时实现了可靠的来源追踪。

**English:**
- **Inference-time diffusion model alignment:** Doob's Matching-based method enables guiding diffusion models to generate images satisfying specific constraints (safety, style consistency) during inference without retraining.
- **Multimodal safety evaluation:** New benchmark evaluating VLM robustness against adversarial visual inputs, finding current models remain vulnerable to visual adversarial attacks.
- **Watermarking and provenance:** Novel watermarking methods for AI-generated content achieving reliable provenance tracking while maintaining image quality.

---

## 本周趋势观察 / Trend Observations

1. **评估驱动的反思 / Evaluation-driven introspection:** BabyVision等工作表明，社区正在从"模型能做什么"转向"模型不能做什么"，评估本身正在成为研究的前沿。  
   BabyVision and related works show the community shifting from "what models can do" to "what models cannot do" — evaluation itself is becoming a research frontier.

2. **紧凑模型的推理突破 / Compact model reasoning breakthroughs:** STEP3-VL-10B和Ministral 3共同指向一个趋势：参数量不再是能力的硬上限，训练策略和数据质量同样关键。  
   STEP3-VL-10B and Ministral 3 together point to a trend: parameter count is no longer a hard ceiling on capability — training strategy and data quality are equally critical.

3. **安全从训练时到推理时的迁移 / Safety shift from training to inference time:** Constitutional Classifiers++和推理时扩散对齐共同反映了安全研究从训练阶段向推理阶段的迁移趋势。  
   Constitutional Classifiers++ and inference-time diffusion alignment together reflect a trend in safety research migrating from training-time to inference-time interventions.

4. **3D/视频理解与机器人应用的融合 / 3D/video understanding meets robotics:** 视频生成模型的综述标志着视觉理解从被动感知向主动世界建模的转变。  
   The video generation model survey marks a shift in visual understanding from passive perception to active world modeling.
