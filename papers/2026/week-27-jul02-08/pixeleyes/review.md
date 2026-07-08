# PixelEyes: Decoupling Perception and Reasoning for Pinpoint Visual Evidence Seeking

**论文链接 / Paper:** https://arxiv.org/abs/2607.00115  
**类型 / Type:** arXiv preprint, July 2026  
**领域 / Area:** Multimodal Large Language Models (MLLMs), Visual Reasoning, Perception-Reasoning Decoupling  

---

## Q1. 这篇论文试图解决什么问题？/ What problem does this paper address?

**中文：**

当前多模态大语言模型（MLLM）在多轮视觉推理中反复出现**定位失败**的问题：模型无法精确找到问题所指的视觉目标，于是被迫生成越来越长的推理轨迹来“补偿”错误定位，最终形成冗余、低效的推理链。作者把根因归结为**感知与推理在单一模型中高度耦合**——MLLM 同时负责“想什么”和“在哪里看”，一旦定位出错就会引发额外的推理轮次。论文试图用系统化的方式把二者解耦：让推理器决定“找什么”，让专门的感知工具回答“在哪里”。

> This paper explores multi-turn visual reasoning and observes that MLLMs repeatedly fail to localize the target, leading to long, redundant trajectories. We attribute this failure to the entanglement of reasoning and perception within a single model, the MLLM reasons and localizes simultaneously, and inaccurate localization triggers additional reasoning turns that bloat the trajectory.

**English:**

Current MLLMs repeatedly suffer from **localization failures** during multi-turn visual reasoning: the model cannot pinpoint the visual target referred to by the question, so it generates increasingly long reasoning trajectories to compensate for the wrong grounding, resulting in redundant and inefficient chains. The authors trace the root cause to the **entanglement of reasoning and perception inside a single model** — the MLLM is responsible for both "what to think" and "where to look" at the same time, so inaccurate localization triggers extra reasoning turns. The paper aims to systematically decouple them: the reasoner decides *what* to look for, while a specialized perception tool answers *where* it is.

---

## Q2. 核心贡献是什么？/ What are the core contributions?

**中文：**

1. **Mask 引导的视觉搜索（Mask-guided Visual Search）**：调用指代分割模型提供像素级精度的 Mask 定位，让推理器摆脱对不精确接地的补偿。
2. **语义区域广度优先搜索（Semantic-region BFS）**：把图像划分为语义区域并以 BFS 方式组织探索，消除因反复裁剪错误子区域造成的冗余循环。
3. **PixelEyes-6K 数据集**：通过重合成现有数据的专家轨迹，将 Mask 引导搜索与 BFS 逻辑显式嵌入训练数据。
4. **Pinpoint-Bench 基准**：一个零提示（zero-hint）视觉搜索基准，问题中不含任何位置线索，并带有实例级 Mask 与边界框，可把定位失败与推理失败分开诊断。

> To solve this problem, we propose PixelEyes, a multi-turn visual reasoning agent that explicitly decouples reasoning from perception, i.e., the reasoner decides what to look for, while a specialized perception tool answers where it is. Specifically, PixelEyes introduces 1) Mask-guided Visual Search... 2) Semantic-region Breadth-first Search (BFS).

**English:**

1. **Mask-guided Visual Search**: invokes a referring segmentation model to provide mask-precise localization, freeing the reasoner from compensating for imprecise grounding.
2. **Semantic-region Breadth-First Search (BFS)**: organizes exploration as a BFS over semantic regions, eliminating redundant loops caused by repeatedly cropping incorrect sub-regions.
3. **PixelEyes-6K dataset**: resynthesizes expert trajectories from existing data to explicitly embed mask-guided search and BFS logic into the model.
4. **Pinpoint-Bench benchmark**: a zero-hint visual-search benchmark in which questions contain no location cues, with instance-level masks and bounding boxes that separate localization failures from reasoning failures.

---

## Q3. 审稿人会从哪些角度攻击它？/ What would a reviewer attack?

**中文：**

1. **泛化性与语言先验依赖**：BFS 的区域划分和查询解析依赖语言描述与预训练视觉先验，面对未见领域、特殊语言风格或低资源语言时可能失效。
2. **端到端训练困境**：感知工具与推理器之间的接口若是符号化/不可微的，则难以联合优化；若全部可微，通信成本与梯度稳定性需要验证。
3. **延迟与交互成本**：多轮 Mask 调用和区域裁剪增加了 wall-clock 延迟，对实时交互应用不友好，需要与一次性编码方法做严格的延迟-准确率权衡分析。
4. **与已有工作的边界模糊**：需要更明确地区分于 Q-Former、Flamingo、LLaVA-Plus、MiniGPT-v2 以及最近的 visual search / tool-use 方法。
5. **基准覆盖有限**：目前公开信息集中在 VQA 与幻觉检测，复杂多图推理、视频时序、3D 空间等场景尚未充分验证。

**English:**

1. **Generalization and linguistic-prior dependence**: BFS region decomposition and query parsing rely on linguistic descriptions and pretrained visual priors; they may fail on unseen domains, low-resource languages, or unusual language styles.
2. **End-to-end training dilemma**: if the perception-tool-to-reasoner interface is symbolic/non-differentiable, joint optimization is hard; if fully differentiable, communication cost and gradient stability need verification.
3. **Latency and interaction cost**: multi-round mask invocation and region cropping increase wall-clock latency, which is unfriendly to real-time interactive applications; a strict latency-vs-accuracy trade-off with single-pass methods is needed.
4. **Unclear boundary with prior work**: clearer differentiation is needed from Q-Former, Flamingo, LLaVA-Plus, MiniGPT-v2, and recent visual-search / tool-use methods.
5. **Limited benchmark coverage**: public evidence focuses mainly on VQA and hallucination detection; complex multi-image reasoning, video temporal reasoning, and 3D spatial reasoning remain underexplored.

---

## Q4. 博士生应该精读还是跳过？/ Should a PhD student read or skip?

**中文：**

**建议精读，尤其适合做 MLLM 视觉推理、可解释性与主动感知的博士生。** PixelEyes 把“感知-推理解耦”从一句口号变成了可训练的模块化系统，并提供了专门的数据集与诊断基准。如果你正在考虑如何让 MLLM 像人类一样主动“看”图、如何减少幻觉、或者如何设计可审计的多轮视觉 Agent，这篇论文会给出很清晰的架构参考。阅读时重点跟踪：感知工具与 LLM 的接口是否可微、训练数据如何构造、以及 Pinpoint-Bench 与传统 VQA 基准的差异。

**English:**

**Worth a close read, especially for PhD students working on MLLM visual reasoning, interpretability, and active perception.** PixelEyes turns the slogan of "perception-reasoning decoupling" into a trainable modular system, complete with a dedicated dataset and diagnostic benchmark. If you are thinking about how to make MLLMs actively "look" at images like humans, how to reduce hallucinations, or how to design auditable multi-turn visual agents, this paper provides a clear architectural reference. When reading, focus on whether the perception-tool-to-LLM interface is differentiable, how the training data is constructed, and how Pinpoint-Bench differs from conventional VQA benchmarks.
