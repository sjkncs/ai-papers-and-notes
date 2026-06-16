# MHLA: Restoring Expressivity of Linear Attention — Paper Review / 论文审稿

> **Paper:** MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head  
> **arXiv:** [2601.07832](https://arxiv.org/abs/2601.07832)  
> **Date:** January 12, 2026  
> **Authors:** Kewei Zhang, Ye Huang, Yufan Deng, Jincheng Yu, Junsong Chen, Huan Ling, Enze Xie, Daquan Zhou  
> **Reviewer Notes:** 第2周重点论文 / Week 2 featured paper

---

## Key Excerpts / 关键摘录

> 1. "We identify that the expressivity gap between linear and softmax attention stems from linear attention's inability to form token-specific attention distributions — a limitation we address by introducing multi-head projection at the token level rather than the sequence level."
> 2. "MHLA achieves this without increasing runtime complexity: by reorganizing the computation graph to apply multiple projection heads per-token before the linear recurrence, we recover 93% of the accuracy gap to softmax attention on standard language modeling benchmarks."
> 3. "Across visual understanding (ImageNet-1K), language modeling (WikiText-103, C4), and image generation (CelebA-HQ), MHLA consistently outperforms prior linear attention variants and approaches softmax attention quality at O(n) complexity."

---

## Q1: Core Problem / 核心问题

**中文：** Softmax注意力机制虽然是Transformer的核心，但其O(n²)的时间复杂度限制了长序列处理的效率。线性注意力通过核近似将复杂度降至O(n)，但引入了显著的表达能力损失——线性注意力对所有token使用相同的全局注意力模式，无法像softmax那样生成token-specific的注意力分布。本文试图解决：**如何在不增加运行时开销的前提下，恢复线性注意力相对于softmax注意力的表达能力差距？**

**English:** While softmax attention is central to Transformers, its O(n²) time complexity limits efficient long-sequence processing. Linear attention reduces complexity to O(n) via kernel approximation but introduces significant expressivity loss — linear attention applies the same global attention pattern to all tokens, unable to generate token-specific attention distributions like softmax. This paper addresses: **How can we recover the expressivity gap between linear and softmax attention without increasing runtime overhead?**

---

## Q2: Claimed Contributions / 主要贡献

**中文：**
- **Token级多头机制（Token-Level Multi-Head）：** 将多头投影从序列级别移至token级别，使每个token拥有独立的注意力子空间，从而恢复线性注意力的token-specific表达能力。
- **零额外运行时开销：** 通过重组计算图，在token维度上并行应用多头投影，MHLA不引入额外的运行时复杂度——理论上保持O(n)，实际上与标准线性注意力延迟相当。
- **统一的表达力恢复框架：** 提供了理论分析证明token-level多头投影可以将线性注意力的表达力提升至接近softmax注意力的水平，并量化了恢复程度与头数的关系。
- **跨模态一致性验证：** 在视觉（ImageNet-1K分类）、语言（WikiText-103/C4语言建模）和生成（CelebA-HQ图像生成）三个模态上一致展示了性能提升。
- **恢复93%的精度差距：** 在标准语言建模基准上，MHLA恢复了线性注意力与softmax注意力之间93%的精度差距。

**English:**
- **Token-Level Multi-Head mechanism:** Moving multi-head projection from sequence-level to token-level, giving each token its own attention subspace and restoring token-specific expressivity for linear attention.
- **Zero additional runtime overhead:** By reorganizing the computation graph to apply multi-head projections in parallel across the token dimension, MHLA introduces no additional runtime complexity — theoretically maintaining O(n), practically matching standard linear attention latency.
- **Unified expressivity restoration framework:** Theoretical analysis proving that token-level multi-head projection can elevate linear attention expressivity to near-softmax levels, with quantified relationship between restoration degree and number of heads.
- **Cross-modal consistency validation:** Consistent performance improvements demonstrated across vision (ImageNet-1K classification), language (WikiText-103/C4 language modeling), and generation (CelebA-HQ image generation).
- **93% accuracy gap recovery:** MHLA recovers 93% of the accuracy gap between linear and softmax attention on standard language modeling benchmarks.

---

## Q3: Reviewer Attack Points / 审稿质疑点

**中文：**

1. **"93%恢复"的基准依赖性：** 93%这一数字高度依赖于所选择的基准和评估指标。在更具挑战性的任务（如长文本推理、代码生成）上，恢复率是否同样高？论文应报告更多基准上的结果分布。

2. **Token级多头的参数量增长：** 虽然论文声称"零额外运行时开销"，但token级多头投影是否增加了模型的参数量？如果参数量增加了，那么性能提升是否部分来自于更多的参数而非架构创新？

3. **与FlashAttention等高效softmax实现的对比：** 即使MHLA保持了O(n)复杂度，现代FlashAttention实现在实际硬件上已经非常高效。在相同wall-clock时间下，MHLA是否真的优于优化后的softmax注意力？

4. **长序列上的实际表现：** 论文的理论分析假设了理想的核近似。在非常长的序列（>32K tokens）上，累积的近似误差是否会导致性能退化？

5. **核函数的选择敏感性：** MHLA的性能是否对底层线性注意力核函数的选择敏感？论文使用了哪种核函数（ELU+, ReLU, 随机傅里叶特征）？不同核函数之间的性能差异如何？

6. **理论分析与实验的对齐：** 理论上的表达力恢复上界是否与实验观察到的性能提升一致？是否存在理论预测与实验结果不匹配的情况？

7. **与其他线性注意力改进方法的对比：** 论文应与RetNet、Mamba、RWKV等近期线性/状态空间模型进行对比，说明MHLA的独特优势。

**English:**

1. **Benchmark dependency of "93% recovery":** This number is highly dependent on chosen benchmarks and metrics. On more challenging tasks (long-text reasoning, code generation), is recovery equally high? The paper should report result distributions across more benchmarks.

2. **Parameter count increase from token-level multi-head:** While claiming "zero additional runtime overhead," does token-level multi-head projection increase model parameter count? If so, are performance gains partially from more parameters rather than architectural innovation?

3. **Comparison with efficient softmax implementations:** Even with O(n) complexity, modern FlashAttention implementations are highly optimized on actual hardware. Under equal wall-clock time, does MHLA truly outperform optimized softmax attention?

4. **Performance on very long sequences:** The theoretical analysis assumes ideal kernel approximation. On very long sequences (>32K tokens), does accumulated approximation error cause performance degradation?

5. **Kernel function selection sensitivity:** Is MHLA's performance sensitive to the choice of underlying linear attention kernel? Which kernel (ELU+, ReLU, random Fourier features) does the paper use? What are the performance differences across kernels?

6. **Theory-experiment alignment:** Does the theoretical expressivity restoration upper bound align with experimentally observed performance gains? Are there cases where theoretical predictions mismatch experimental results?

7. **Comparison with other linear attention improvements:** The paper should compare with RetNet, Mamba, RWKV and other recent linear/state-space models, articulating MHLA's unique advantages.

---

## Q4: PhD Reading Guide / 博士阅读指南

### Read Carefully / 精读

**中文：**
- **Section 3 (Token-Level Multi-Head Formulation):** 核心方法论。仔细理解token-level与sequence-level多头的数学区别，以及为什么token级别能恢复表达力。这一节的理论分析是全文的基石。
- **Section 4.2 (Computation Graph Reorganization):** 如何实现零额外开销的计算图重组。这对工程实现至关重要，也是论文的关键创新之一。
- **Figure 3-5 (Attention Pattern Visualization):** 注意力模式的可视化对比，直观展示MHLA如何恢复了token-specific的注意力分布。

**English:**
- **Section 3 (Token-Level Multi-Head Formulation):** Core methodology. Carefully understand the mathematical distinction between token-level and sequence-level multi-head, and why token-level restores expressivity. This section's theoretical analysis is the paper's foundation.
- **Section 4.2 (Computation Graph Reorganization):** How zero-overhead computation graph reorganization is achieved. Critical for engineering implementation and a key innovation.
- **Figure 3-5 (Attention Pattern Visualization):** Visual comparison of attention patterns, intuitively showing how MHLA restores token-specific attention distributions.

### Skim / Skip / 略读/跳过

**中文：**
- **Section 2 (Background on Linear Attention):** 线性注意力的标准背景介绍，熟悉该领域的研究者可快速浏览。
- **Section 6 (Image Generation Results):** 图像生成实验，除非你的研究方向涉及生成模型，否则可略读结论。
- **Appendix A (Proofs):** 详细数学证明，需要时可查阅。

**English:**
- **Section 2 (Background on Linear Attention):** Standard background on linear attention — researchers familiar with the field can skim.
- **Section 6 (Image Generation Results):** Image generation experiments — skim conclusions unless your research involves generative models.
- **Appendix A (Proofs):** Detailed mathematical proofs — consult as needed.

---

## References / 参考文献

1. Zhang, K., Huang, Y., Deng, Y. et al. "MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head." arXiv:2601.07832 (2026).
2. Katharopoulos, A. et al. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML (2020).
3. Choromanski, K. et al. "Rethinking Attention with Performers." ICLR (2021).
4. Dao, T. et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS (2022).
5. Sun, Y. et al. "RetNet: Retentive Network: A Successor to Transformer for Large Language Models." arXiv:2307.08621 (2023).
6. Gu, A. and Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752 (2023).
