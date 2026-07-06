# ReContext: Recursive Evidence Replay as LLM Harness for Long-Context Reasoning / ReContext：面向长上下文推理的递归证据回放

> **arXiv:** [2604.02509](https://arxiv.org/abs/2604.02509) | **日期 / Date:** 2026-04-24 | **作者 / Authors:** Yanjun Zhao, Ruizhong Qiu, Tianxin Wei, Yuanchen Bei, Zhining Liu, Lingjie Chen, Ismini Lourentzou, Hanghang Tong, Jingrui He

---

## 关键摘录 / Key Excerpts

> “大型语言模型虽然能够访问极长上下文，但往往无法有效利用其中的证据；ReContext 通过模型内部相关性信号构建查询条件化的证据池，并在不改变原始输入的情况下进行证据回放。”  
> “LLMs can access very long contexts yet often fail to exploit the evidence within; ReContext builds a query-conditioned evidence pool from internal relevance signals and replays evidence without altering the original input.”

> “ReContext 是一个免训练的推理方法，将证据组织与生成过程解耦。”  
> “ReContext is a training-free inference method that decouples evidence organization from generation.”

> “在 128K 上下文长度下，该方法在八个长上下文数据集上取得最佳平均排名，并在 Qwen 与 Llama 系列模型上验证有效。”  
> “At 128K context length, the method achieves the best average rank across eight long-context datasets and proves effective on both Qwen and Llama families.”

> “理论分析借助联想记忆（associative memory）概念解释注意力机制中的证据检索行为。”  
> “The theoretical analysis uses associative-memory concepts to explain evidence-retrieval behavior in attention mechanisms.”

---

## Q1: 核心问题 / Core Problem

**中文**  
当前的大语言模型能够将数十万 token 的上下文一次性读入，但“可读”不等于“会用”。许多长上下文任务失败的原因并非上下文长度不足，而是模型难以从海量文本中定位并持续利用与查询相关的证据。ReContext 试图解决的核心问题是：在不需要额外训练或改变原始输入的前提下，如何让 LLM 在生成答案前主动组织并回放相关证据？

**English**  
Modern LLMs can ingest hundreds of thousands of tokens at once, but being able to read a long context is not the same as being able to use it. Many long-context tasks fail not because the window is too small, but because the model cannot locate and persistently exploit evidence relevant to the query. ReContext addresses the core question: without additional training or altering the original input, how can an LLM actively organize and replay relevant evidence before generating an answer?

---

## Q2: 核心贡献 / Claimed Contributions

1. **提出递归证据回放框架 / A recursive evidence-replay framework**  
   设计了一种免训练的推理流程，在生成前反复构造并回放查询条件化证据池。  
   Designs a training-free inference pipeline that repeatedly constructs and replays a query-conditioned evidence pool before generation.

2. **利用模型内部相关性信号 / Uses internal relevance signals**  
   不依赖外部检索器或显式记忆模块，而是从模型自身的注意力与隐藏状态中提取相关性信号。  
   Avoids external retrievers or explicit memory modules by deriving relevance signals from the model's own attention and hidden states.

3. **证据组织与生成分离 / Decouples evidence organization from generation**  
   将“找证据”与“写答案”分为两个阶段，使生成模块可以专注于基于已组织证据的推理。  
   Separates evidence gathering from answer generation, letting the generation module focus on reasoning over already organized evidence.

4. **广泛的实证验证 / Broad empirical validation**  
   在八个长上下文数据集、128K 上下文长度、多种主流模型上取得最佳平均排名。  
   Achieves the best average rank across eight long-context datasets, 128K context length, and multiple mainstream models.

5. **理论视角 / Theoretical perspective**  
   引入联想记忆视角分析注意力机制中的证据检索，为方法提供了概念上的解释。  
   Introduces an associative-memory perspective to analyze evidence retrieval in attention, providing conceptual justification.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

1. **“递归”定义模糊 / The term "recursive" is vague**  
   论文未清晰说明递归的停止条件、收敛性以及每一步如何改变证据池。  
   The paper does not clearly state stopping conditions, convergence properties, or how each step changes the evidence pool.

2. **内部信号的可迁移性存疑 / Generalizability of internal signals**  
   基于注意力/隐藏状态的相关性信号可能在不同模型架构或语言上差异很大，缺乏跨模态验证。  
   Relevance signals based on attention or hidden states may vary greatly across architectures and languages, with little cross-modal validation.

3. **超参数敏感 / Hyperparameter sensitivity**  
   证据池大小、回放次数、阈值等超参数对效果影响可能很大，论文未充分消融。  
   Pool size, replay steps, and thresholds likely heavily influence performance, but the paper lacks thorough ablations.

4. **与现有方法的比较不足 / Limited comparison with existing methods**  
   未与 RAG、prompt compression、instruction-based 重排序等方法在同一数据集上做全面对比。  
   Does not provide a full comparison with RAG, prompt compression, or instruction-based reranking on the same datasets.

5. **理论分析较浅 / Thin theoretical analysis**  
   联想记忆类比提供了直觉，但未能给出严格的误差界或样本复杂度分析。  
   The associative-memory analogy gives intuition but offers no rigorous error bounds or sample-complexity analysis.

---

## Q4: 博士生阅读指南 / PhD Reading Guide

**中文**  
建议先精读摘要、第 2–3 节的方法描述以及实验表格，理解“证据池”是如何从模型内部信号中构造的。接下来可用一个 7B 级别的开源模型（如 Qwen2.5-7B-Instruct 或 Llama-3.1-8B-Instruct）在 LongBench 或自建的 32K/128K 长文档 QA 数据上复现。重点比较：原始输入直接生成、ReContext 证据回放、以及简单的基于余弦相似度的段落重排序三种策略。可延伸方向包括：将证据回放与 RAG 结合、把方法扩展到多模态长上下文（长视频/长文档图文）、以及在检索增强生成中用它替代重排序模块。

**English**  
Start with the abstract, Sections 2–3 on methodology, and the experiment tables to understand how the evidence pool is built from internal signals. Then reproduce it with a 7B-level open model such as Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct on LongBench or a custom 32K/128K long-document QA dataset. Compare three strategies: direct generation, ReContext evidence replay, and simple cosine-similarity paragraph reranking. Possible extensions include combining evidence replay with RAG, extending it to multimodal long contexts (long video / long illustrated documents), or using it as a reranking module in retrieval-augmented generation.
