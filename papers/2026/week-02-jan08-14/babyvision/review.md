# BabyVision: Visual Reasoning Beyond Language — Paper Review / 论文审稿

> **Paper:** BabyVision: Visual Reasoning Beyond Language  
> **arXiv:** [2601.06521](https://arxiv.org/abs/2601.06521)  
> **Date:** January 10, 2026  
> **Authors:** Liang Chen, Weichu Xie, Yiyan Liang, Hongfeng He, Hans Zhao  
> **Reviewer Notes:** 第2周重点论文 / Week 2 featured paper

---

## Key Excerpts / 关键摘录

> 1. "We construct a benchmark of visual reasoning tasks that human infants reliably solve between 6-18 months of age, yet which prove remarkably difficult for state-of-the-art multimodal large language models (MLLMs)."
> 2. "We discover an 'inverted competence profile': tasks that require minimal linguistic scaffolding but rich physical/spatial reasoning — precisely the tasks babies excel at — are where current MLLMs show their largest deficits relative to tasks requiring verbal knowledge."
> 3. "Even GPT-4V and Gemini Ultra, which achieve >85% on verbal-knowledge visual QA, drop to 31-42% on our infant-level physical reasoning tasks, suggesting that language-centric pretraining has created a systematic blind spot in visual understanding."

---

## Q1: Core Problem / 核心问题

**中文：** 当前多模态大语言模型（MLLMs）在需要语言知识的视觉问答任务上表现出色，但这是否意味着它们真正"理解"了视觉世界？本文提出了一个关键质疑：**MLLMs的视觉能力是否过度依赖语言先验，而在需要纯视觉推理（不需要语言参与）的任务上存在系统性盲区？** 作者通过构建婴儿能解决但MLLM失败的视觉推理任务来验证这一假说。

**English:** Current multimodal large language models (MLLMs) excel at visual QA tasks requiring language knowledge, but does this mean they truly "understand" the visual world? This paper raises a critical question: **Are MLLMs' visual capabilities overly reliant on linguistic priors, creating systematic blind spots on tasks requiring pure visual reasoning (without language involvement)?** The authors test this hypothesis by constructing visual reasoning tasks that infants can solve but MLLMs fail at.

---

## Q2: Claimed Contributions / 主要贡献

**中文：**
- **"倒置能力画像"概念（Inverted Competence Profile）：** 首次系统性地揭示了MLLMs的能力分布与人类发展轨迹的倒置关系——需要最少语言支持但需要丰富物理/空间推理的任务（婴儿擅长的）恰恰是MLLM最薄弱的环节。
- **婴儿级视觉推理基准（BabyVision Benchmark）：** 构建了一个包含500+任务的基准测试集，所有任务均来自6-18个月婴儿的认知发展实验（物体永久性、重力直觉、空间关系等），无需语言能力即可完成。
- **系统性评估：** 对GPT-4V、Gemini Ultra、LLaVA-NeXT、InternVL-2等主流MLLM进行了全面评估，发现最强模型在此基准上仅达到31-42%的准确率。
- **语言先验消融实验：** 通过对比纯视觉呈现和附带语言描述的任务变体，量化了MLLMs对语言先验的依赖程度。
- **改进方向建议：** 提出了具体的架构和训练策略建议，以缓解MLLMs的视觉推理盲区。

**English:**
- **"Inverted Competence Profile" concept:** First systematic demonstration that MLLMs' capability distribution is inverted relative to human developmental trajectories — tasks requiring minimal linguistic scaffolding but rich physical/spatial reasoning (where babies excel) are precisely where MLLMs are weakest.
- **BabyVision Benchmark:** Construction of a 500+ task benchmark drawn from 6-18 month infant cognitive development experiments (object permanence, gravity intuition, spatial relations) that require no language ability to solve.
- **Systematic evaluation:** Comprehensive evaluation of GPT-4V, Gemini Ultra, LLaVA-NeXT, InternVL-2 and other mainstream MLLMs, finding that the strongest models achieve only 31-42% accuracy on this benchmark.
- **Linguistic prior ablation:** Quantification of MLLMs' dependence on linguistic priors through comparison of pure-visual and language-augmented task variants.
- **Improvement recommendations:** Specific architectural and training strategy suggestions to mitigate MLLMs' visual reasoning blind spots.

---

## Q3: Reviewer Attack Points / 审稿质疑点

**中文：**

1. **婴儿认知的实验效度：** 婴儿认知实验本身就存在方法论争议。发展心理学界对"婴儿是否真的理解物体永久性"仍有争论。将婴儿表现作为"ground truth"是否合理？

2. **任务呈现方式的公平性：** 婴儿通过物理互动来探索世界，而MLLM只能处理静态图像或短视频。任务的呈现方式是否本身就对MLLM不利？如果给MLLM提供交互式环境（如embodied agent），结果是否不同？

3. **语言描述偏差：** 论文中MLLM的任务输入仍包含文本提示（"图中哪个物体..."），这是否已经引入了语言成分？纯视觉推理任务的定义是否自洽？

4. **基准的可扩展性：** 500+任务的规模是否足以覆盖婴儿认知的全部维度？基准是否存在任务类型分布不均的问题？

5. **改进建议的实验验证：** 论文提出的改进方向是否经过了实验验证？还是仅停留在建议层面？如果有初步实验，效果如何？

6. **与已有VLM评估基准的关系：** BabyVision与已有基准（如MMBench、SEED-Bench、MMMU）的关系如何？是完全正交还是存在部分重叠？

**English:**

1. **Validity of infant cognition experiments:** Infant cognition experiments themselves have methodological controversies. The developmental psychology community still debates whether infants truly "understand" object permanence. Is using infant performance as ground truth justified?

2. **Fairness of task presentation:** Infants explore the world through physical interaction while MLLMs process static images or short videos. Is the task presentation inherently disadvantageous to MLLMs? Would results differ if MLLMs had interactive environments (e.g., embodied agents)?

3. **Language description bias:** The paper's MLLM inputs still include text prompts ("Which object in the image..."), which already introduces linguistic components. Is the definition of pure visual reasoning tasks self-consistent?

4. **Benchmark scalability:** Is 500+ tasks sufficient to cover all dimensions of infant cognition? Does the benchmark have uneven task type distribution?

5. **Experimental validation of improvement suggestions:** Were the proposed improvement directions experimentally validated, or do they remain at the suggestion level? If preliminary experiments exist, what were the results?

6. **Relationship to existing VLM benchmarks:** How does BabyVision relate to existing benchmarks (MMBench, SEED-Bench, MMMU)? Is it fully orthogonal or partially overlapping?

---

## Q4: PhD Reading Guide / 博士阅读指南

### Read Carefully / 精读

**中文：**
- **Section 3 (Benchmark Construction):** 基准构建方法论。理解如何从发展心理学实验中提取可计算的视觉推理任务，以及任务设计如何排除语言因素的干扰。
- **Section 4.1 (Inverted Competence Profile Results):** 核心实验结果。仔细分析不同模型在各任务类别上的表现分布，理解"倒置能力画像"的具体含义。
- **Section 4.3 (Linguistic Prior Ablation):** 语言先验消融实验。这一节直接验证了论文的核心假说，方法论设计值得学习。

**English:**
- **Section 3 (Benchmark Construction):** Benchmark construction methodology. Understand how computable visual reasoning tasks are extracted from developmental psychology experiments and how task design excludes linguistic confounds.
- **Section 4.1 (Inverted Competence Profile Results):** Core experimental results. Carefully analyze performance distributions across task categories for different models, understanding the concrete meaning of "inverted competence profile."
- **Section 4.3 (Linguistic Prior Ablation):** Linguistic prior ablation experiments. This section directly validates the paper's core hypothesis — the methodology design is worth studying.

### Skim / Skip / 略读/跳过

**中文：**
- **Section 2 (Related Work on VLM Evaluation):** 标准的VLM评估文献综述，熟悉该领域的研究者可快速浏览。
- **Section 5 (Discussion on Developmental Psychology):** 发展心理学的讨论，除非你对认知科学与AI的交叉感兴趣，否则可略读。
- **Appendix (Full Task List):** 完整任务列表，需要时可查阅。

**English:**
- **Section 2 (Related Work on VLM Evaluation):** Standard VLM evaluation literature survey — researchers familiar with the field can skim.
- **Section 5 (Discussion on Developmental Psychology):** Developmental psychology discussion — skim unless interested in the cognitive science-AI intersection.
- **Appendix (Full Task List):** Complete task list — consult as needed.

---

## References / 参考文献

1. Chen, L., Xie, W., Liang, Y., He, H., Zhao, H. "BabyVision: Visual Reasoning Beyond Language." arXiv:2601.06521 (2026).
2. Baillargeon, R. "Object permanence in 3.5- and 4.5-month-old infants." Developmental Psychology (1987).
3. Liu, H. et al. "Visual Instruction Tuning." NeurIPS (2023).
4. Bai, Y. et al. "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073 (2022).
5. Yue, X. et al. "MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark." CVPR (2024).
6. Piaget, J. "The Construction of Reality in the Child." Basic Books (1954).
