# The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks

> **arXiv:** [2603.05498](https://arxiv.org/abs/2603.05498) | **日期 / Date:** 2026-03-05 | **作者 / Authors:** Shangwen Sun, Alfredo Canziani, Yann LeCun, Jiachen Zhu

---

## 关键摘录 / Key Excerpts

> 1. "We study two recurring phenomena in Transformer language models: massive activations, in which a small number of tokens exhibit extreme outliers in a few channels, and attention sinks, in which certain tokens attract disproportionate attention mass regardless of semantic relevance."
>    / "我们研究 Transformer 语言模型中两个反复出现的现象：大规模激活——少量 token 在少数通道中呈现极端离群值；注意力汇聚——某些 token 无论语义相关性如何，都吸引不成比例的注意力质量。"

> 2. "Through systematic experiments, we show that the co-occurrence is largely an architectural artifact of modern Transformer design, and that the two phenomena serve related but distinct functions."
>    / "通过系统实验，我们证明两者的共现主要是现代 Transformer 设计的架构假象，且两种现象承担相关但不同的功能。"

> 3. "Massive activations operate globally: they induce near-constant hidden representations that persist across layers, effectively functioning as implicit parameters of the model."
>    / "大规模激活全局运作：它们诱导跨层持续的近常数隐表示，实际上充当模型的隐式参数。"

> 4. "Attention sinks operate locally: they modulate attention outputs across heads and bias individual heads toward short-range dependencies."
>    / "注意力汇聚局部运作：它们跨注意力头调制输出，并将单个头偏向短距离依赖。"

> 5. "We identify the pre-norm configuration as the key choice that enables the co-occurrence, and show that ablating it causes the two phenomena to decouple."
>    / "我们发现 pre-norm 配置是实现共现的关键设计选择，消融它会导致两种现象解耦。"

---

## Q1: 核心问题 / Core Problem

**中文：**

Transformer 语言模型中存在两种被广泛观察但理解不足的现象：
1. **大规模激活（Massive Activations）**：少量 token 在少数通道上产生极端离群值（spike），这些离群值可能影响数值稳定性和模型可解释性。
2. **注意力汇聚（Attention Sinks）**：某些 token（通常是 BOS 或标点符号）无论语义如何，都吸引大量注意力质量，导致注意力分布偏离语义。

先前工作观察到这两种现象经常共现且涉及相同 token，但它们的功能角色和因果关系尚不清楚。本文真正想回答的问题是：**这两种现象是同一个潜在机制的不同表现，还是承担不同功能的独立现象？它们为什么会共现？**

**English:**

Two widely observed but poorly understood phenomena exist in Transformer language models:
1. **Massive Activations**: a small number of tokens produce extreme outliers in a few channels (spikes), which may affect numerical stability and interpretability.
2. **Attention Sinks**: certain tokens (typically BOS or punctuation) attract disproportionate attention mass regardless of semantic relevance, causing attention distributions to deviate from semantics.

Prior work observes that these phenomena frequently co-occur and involve the same tokens, but their functional roles and causal relationship remain unclear. The core question this paper answers is: **Are these two phenomena different manifestations of the same underlying mechanism, or are they independent phenomena serving different functions? Why do they co-occur?**

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **解耦大规模激活与注意力汇聚**：通过系统实验证明两者的共现主要是 pre-norm 架构设计的假象，而非因果关系。
2. **功能分析**：
   - 大规模激活全局运作，产生跨层持续的近常数隐表示，充当模型的隐式参数；
   - 注意力汇聚局部运作，跨头调制输出并偏向短距离依赖。
3. **Pre-norm 的关键作用**：识别出 pre-norm（LayerNorm 在注意力/FFN 之前）是导致两者共现的关键设计选择，消融它会使两种现象解耦。
4. **对模型设计的启示**：为理解 Transformer 内部工作机制和架构改进提供了理论基础。

**English:**

1. **Decoupling massive activations and attention sinks**: Systematic experiments show that co-occurrence is largely an architectural artifact of pre-norm design, not a causal relationship.
2. **Functional analysis**:
   - Massive activations operate globally, producing near-constant hidden representations that persist across layers, acting as implicit parameters;
   - Attention sinks operate locally, modulating outputs across heads and biasing toward short-range dependencies.
3. **Key role of pre-norm**: Identifies pre-norm (LayerNorm before attention/FFN) as the key design choice enabling co-occurrence; ablating it decouples the two phenomena.
4. **Implications for model design**: Provides theoretical foundation for understanding Transformer internals and architectural improvements.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **实验覆盖度**：论文是否在足够多的模型规模（从小到超大）和架构变体上验证？pre-norm vs post-norm 的对比是否充分？
2. **功能归因的可靠性**：如何确保"隐式参数"和"短距离依赖偏向"的功能归因是因果的而非相关的？
3. **实际影响**：理解这些现象对实际模型训练或推理有什么具体改进？论文是否提出了可操作的架构修改建议？
4. **与现有工作的区分**：与 StreamingLLM、Massive Activations in LLMs 等先前工作的区别和增量贡献是否清晰？
5. **消融实验的充分性**：pre-norm 消融是否在所有模型上都导致解耦？是否存在例外情况？

**English:**

1. **Experimental coverage**: Are experiments conducted on sufficient model scales and architectural variants? Is the pre-norm vs post-norm comparison thorough?
2. **Reliability of functional attribution**: How is it ensured that the functional attributions ("implicit parameters" and "short-range dependency bias") are causal rather than correlational?
3. **Practical impact**: What specific improvements to model training or inference does understanding these phenomena enable? Does the paper propose actionable architectural modifications?
4. **Differentiation from prior work**: Are distinctions from StreamingLLM, Massive Activations in LLMs, and other prior work clearly articulated?
5. **Sufficiency of ablation**: Does pre-norm ablation cause decoupling on all models? Are there exceptions?

---

## Q4: 博士生阅读指南 / PhD Reading Guide

**中文：**

**如果你是做 Transformer 架构 / 模型可解释性 / 高效推理的博士生：**

- **精读部分**：
  - 第 1–2 页的现象定义与问题提出；
  - 第 3–4 页的实验方法论（如何检测大规模激活和注意力汇聚）；
  - 第 5 页的 pre-norm 消融实验和功能归因分析。
- **跳过部分**：
  - 若已熟悉 Transformer 架构和 LayerNorm，可跳过相关背景；
  - 若对 attention sink 的先前工作（StreamingLLM 等）不熟悉，可先阅读。
- **复现建议**（本仓库提供 PyTorch 分析代码）：
  - 在 LLaMA / GPT-2 等模型上运行 attention sink 和 massive activation 检测脚本；
  - 比较 pre-norm vs post-norm 模型中两种现象的共现程度；
  - 重点观察：消融 pre-norm 后，两种现象是否真的解耦？
- **可延伸的研究点**：
  - 设计不依赖 pre-norm 的新架构，同时保留两种现象的有益功能；
  - 将大规模激活和注意力汇聚的知识用于模型压缩或推理加速。

**English:**

**If you are a PhD student working on Transformer architecture, model interpretability, or efficient inference:**

- **Read carefully**:
  - Pages 1–2: phenomenon definition and problem formulation;
  - Pages 3–4: experimental methodology (how to detect massive activations and attention sinks);
  - Page 5: pre-norm ablation experiments and functional attribution analysis.
- **Skip if familiar**:
  - Transformer architecture and LayerNorm background if already known;
  - Prior work on attention sinks (StreamingLLM, etc.) if already familiar.
- **Reproduction suggestions** (this repo provides PyTorch analysis code):
  - Run the attention sink and massive activation detection script on LLaMA / GPT-2 models;
  - Compare co-occurrence in pre-norm vs post-norm models;
  - Key observation: after ablating pre-norm, do the two phenomena truly decouple?
- **Potential research extensions**:
  - Design new architectures that don't rely on pre-norm while preserving beneficial functions of both phenomena;
  - Use knowledge of massive activations and attention sinks for model compression or inference acceleration.
