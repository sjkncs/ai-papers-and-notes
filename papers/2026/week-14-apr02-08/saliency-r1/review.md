# Saliency-R1: Enforcing Interpretable and Faithful Vision-language Reasoning via Saliency-map Alignment Reward

> **arXiv:** [2604.04500](https://arxiv.org/abs/2604.04500) | **日期 / Date:** 2026-04-06 | **作者 / Authors:** Shizhan Gong, Minda Hu, Qiyuan Zhang, Chen Ma, Qi Dou

---

## 关键摘录 / Key Excerpts

> 1. "Concerns about their trustworthiness persist, particularly regarding tendencies to lean more on textual cues than visual evidence and the risk of producing ungrounded or fabricated responses."
>    / "关于其可信度的担忧持续存在，特别是倾向于依赖文本线索而非视觉证据，以及产生无根据或捏造响应的风险。"

> 2. "We introduce a novel saliency map technique that efficiently highlights critical image regions contributing to generated tokens without additional computational overhead."
>    / "我们引入了一种新颖的显著性图技术，高效地突出对生成 token 有贡献的关键图像区域，无需额外计算开销。"

> 3. "We use the overlap between the saliency maps and human-annotated bounding boxes as the reward function, and apply GRPO to align the salient parts and critical regions."
>    / "我们使用显著性图与人工标注边界框之间的重叠作为奖励函数，并应用 GRPO 对齐显著部分和关键区域。"

---

## Q1: 核心问题 / Core Problem

**中文：**

VLM 在推理时经常依赖文本先验而非真正的视觉证据，导致"幻觉"和不忠实的推理链。现有可解释性方法要么计算昂贵，要么事后归因（post-hoc attribution）与模型实际推理过程不一致。本文核心问题：**能否以零额外计算开销的方式，让 VLM 在推理时真正"看"到正确的视觉区域，并用 GRPO 对齐显著性图与人工标注区域？**

**English:**

VLMs often rely on textual priors rather than genuine visual evidence during reasoning, leading to hallucination and unfaithful reasoning chains. Existing interpretability methods are either computationally expensive or post-hoc attributions inconsistent with actual reasoning. Core question: **Can we make VLMs truly "look at" the correct visual regions during reasoning with zero additional computational overhead, using GRPO to align saliency maps with human annotations?**

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **高效显著性图**：无需额外前向传播即可提取 token 级视觉显著性图。
2. **视觉信息流追踪**：可扩展到追踪视觉信息如何从图像流经推理过程到达最终答案。
3. **GRPO 对齐**：用显著性图与人工标注的重叠作为奖励，通过 GRPO 训练模型关注正确区域。
4. **三维度提升**：在推理忠实度、可解释性和整体任务性能上同时提升。

**English:**

1. **Efficient saliency maps**: Extract token-level visual saliency maps without additional forward passes.
2. **Visual information flow tracing**: Extensible to tracing how visual information flows from image through reasoning to final answer.
3. **GRPO alignment**: Use overlap between saliency maps and human annotations as reward, training via GRPO to focus on correct regions.
4. **Three-dimensional improvement**: Simultaneously improves reasoning faithfulness, interpretability, and task performance.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **显著性图质量**：无需额外计算的显著性图是否足够精确？与 GradCAM、Attention Rollout 等方法对比如何？
2. **人工标注的可扩展性**：依赖人工标注边界框作为奖励信号，是否限制了大规模应用？
3. **GRPO 训练稳定性**：显著性对齐奖励是否与其他奖励（如正确答案奖励）产生冲突？
4. **跨任务泛化**：在 VQA 上训练的对齐能否迁移到图像描述或视觉推理等任务？

**English:**

1. **Saliency map quality**: Are zero-cost saliency maps precise enough? How do they compare to GradCAM, Attention Rollout?
2. **Human annotation scalability**: Relying on human bounding box annotations as reward limits large-scale application?
3. **GRPO training stability**: Does saliency alignment reward conflict with other rewards (e.g., correct answer reward)?
4. **Cross-task generalization**: Can alignment trained on VQA transfer to image captioning or visual reasoning?

---

## Q4: 博士生阅读指南 / PhD Reading Guide

**中文：**

**如果你做 VLM 可解释性 / 对齐 / GRPO：**
- **精读**：显著性图提取方法、GRPO 奖励设计、视觉信息流追踪。
- **复现**：在 LLaVA 或 Qwen-VL 上实现显著性图提取 + GRPO 训练流程。
- **延伸**：将显著性对齐与 HALP（生成前幻觉检测）结合，构建"先看对再回答"的安全 VLM。

**English:**

**If you work on VLM interpretability, alignment, or GRPO:**
- **Read carefully**: Saliency map extraction, GRPO reward design, visual information flow tracing.
- **Reproduce**: Implement saliency extraction + GRPO training on LLaVA or Qwen-VL.
- **Extend**: Combine saliency alignment with HALP (pre-generation hallucination detection) to build a "look first, answer second" safe VLM.
