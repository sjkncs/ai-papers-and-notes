# Attention to Mamba: A Recipe for Cross-Architecture Distillation / 跨架构蒸馏：从 Transformer 到 Mamba

> **arXiv:** [2604.14191](https://arxiv.org/abs/2604.14191) | **日期 / Date:** 2026-04-01 (gaining attention this week) | **作者 / Authors:** Abhinav Moudgil, Ningyuan Huang, Eeshan Gunesh Dhekane, Pau Rodríguez, Luca Zappella, Federico Danieli

---

## 关键摘录 / Key Excerpts

> “Standard distillation fails to maintain teacher performance when transferring from Transformers to State Space Models.”
> “标准蒸馏在将知识从 Transformer 迁移到状态空间模型时无法保持教师模型性能。”

> “We introduce a two-stage distillation method using a kernel trick adaptation.”
> “我们引入了一种利用核技巧自适应的两阶段蒸馏方法。”

> “Our pure Mamba student preserves the original Pythia-1B Transformer performance.”
> “我们的纯 Mamba 学生模型保留了原始 Pythia-1B Transformer 的性能。”

> “No hybrid attention blocks are required in the student.”
> “学生模型中不需要混合注意力块。”

---

## Q1: 核心问题 / Core Problem

**中文：**
状态空间模型（SSM，如 Mamba）因其线性复杂度在长序列任务上极具吸引力，但通常需要从头预训练，成本高昂。一个自然的想法是：能否直接将已有的大型 Transformer 教师模型的知识蒸馏到纯 Mamba 学生模型？
然而，Transformer 与 Mamba 的注意力机制差异巨大：
- Transformer 通过 Softmax 注意力实现全局 token-to-token 交互；
- Mamba 通过选择性状态空间实现随时间变化的递归压缩。

这种架构差异导致标准 logits/ hidden-state 蒸馏失效或严重掉点。本文要解决的核心问题是：如何设计跨架构蒸馏目标，使得纯 Mamba 学生模型在不引入任何注意力模块的情况下达到 Transformer 教师的性能水平。

**English:**
State Space Models (SSMs such as Mamba) are attractive for long-sequence tasks due to linear complexity, but pretraining them from scratch is expensive. A natural question is whether we can distill knowledge from existing large Transformer teachers into pure Mamba students.
However, Transformers and Mamba differ fundamentally in their attention mechanisms:
- Transformers use softmax attention for global token-to-token interactions;
- Mamba uses selective state-space recurrence for time-varying compression.

This architectural gap causes standard logit/hidden-state distillation to fail or degrade severely. The core problem is how to design cross-architecture distillation objectives so that a pure Mamba student can match the Transformer teacher without introducing any attention modules.

---

## Q2: 核心贡献 / Claimed Contributions

1. **跨架构蒸馏问题的形式化**
   - 首次系统分析了为何标准 KL/ MSE 蒸馏在 Transformer→Mamba 场景下失效：注意力矩阵与状态转移矩阵的表示空间不对齐。
   - *Formally analyzes why standard KL/MSE distillation fails for Transformer→Mamba transfer: the representation spaces of attention matrices and state-transition matrices are misaligned.*

2. **两阶段蒸馏协议**
   - **阶段一**：在教师注意力核与学生状态空间核之间建立对应关系，通过核技巧（kernel trick）将二次注意力映射到线性状态空间可优化的目标。
   - **阶段二**：在阶段一的对齐表示基础上进行标准 logits 与中间层特征蒸馏。
   - *A two-stage protocol: Stage 1 aligns the teacher attention kernel with the student state-space kernel via a kernel-trick adaptation; Stage 2 applies standard logit and intermediate-feature distillation on the aligned representations.*

3. **纯 Mamba 学生达到 Transformer 教师水平**
   - 在 Pythia-1B 规模上，纯 Mamba 学生在下游任务与困惑度上均与教师持平。
   - 证明无需混合注意力块即可实现 Transformer→Mamba 的知识迁移。
   - *A pure Mamba student matches Pythia-1B Transformer perplexity and downstream performance, demonstrating that hybrid attention blocks are not necessary for transfer.*

4. **训练效率与可扩展性**
   - 由于学生为纯 Mamba，训练与推理的序列复杂度为线性，显著优于同规模 Transformer 教师。
   - *Because the student is pure Mamba, training and inference scale linearly with sequence length, outperforming the Transformer teacher at equivalent active capacity.*

---

## Q3: 审稿攻击点 / Reviewer Attack Points

1. **核技巧的理论严谨性**
   - 论文将 Softmax 注意力核与状态空间核对应，但这一映射是否在所有序列长度与上下文下保持收敛性？需要更严格的近似误差界。
   - *The correspondence between softmax attention and state-space kernels needs rigorous approximation-error bounds across sequence lengths and contexts.*

2. **规模外推性存疑**
   - 实验仅在 Pythia-1B 规模验证；当教师扩大到 7B/70B，学生是否仍能保持性能？更大模型的注意力模式更稀疏、状态空间更难对齐。
   - *Results are shown only at Pythia-1B scale; scaling to 7B/70B teachers is not validated, and attention patterns become sparser at larger scales.*

3. **两阶段训练的稳定性**
   - 阶段一的核对齐与阶段二的 logits 蒸馏目标可能冲突，导致训练不稳定或超参敏感。论文未给出阶段间学习率、温度等调参细节。
   - *The kernel-alignment and logit-distillation objectives may conflict, leading to instability or hyperparameter sensitivity; stage-wise tuning details are sparse.*

4. **泛化到其它 SSM 变体**
   - 方法针对 Mamba-2 设计，是否适用于 Griffin、RWKV、DeltaNet 等其它线性 RNN/SSM？
   - *The recipe is designed around Mamba-2; transferability to Griffin, RWKV, DeltaNet, or other linear RNN/SSM variants is unclear.*

5. **蒸馏成本与收益权衡**
   - 两阶段蒸馏增加了训练复杂度；需要与直接预训练小型 Mamba、渐进式蒸馏或架构搜索等方法进行成本对比。
   - *The two-stage protocol adds training complexity; cost comparisons with direct small-Mamba pretraining, progressive distillation, or architecture search are needed.*

---

## Q4: 博士生阅读指南 / PhD Reading Guide

**中文：**
- **第一遍（20 分钟）**：阅读摘要与贡献，理解“为什么标准蒸馏在 Transformer→Mamba 上失效”这一核心洞察；画出教师与学生架构对比图。
- **第二遍（1.5 小时）**：精读两阶段蒸馏目标函数，尤其是核技巧的数学推导；尝试在 Pythia-160M/410M 上复现单阶段 vs. 两阶段蒸馏的困惑度差距。
- **第三遍（动手）**：参考本目录下的 `code/attention_to_mamba_distillation.py`，用 PyTorch 实现一个简化版的两阶段蒸馏；重点观察 attention kernel 与 SSM kernel 的对应损失如何影响收敛。
- **批判性思考**：若核技巧只近似局部注意力，那么全局依赖强的任务（如长程共指消解）是否仍会掉点？设计一个针对方程求解或长文档推理的评估来验证。

**English:**
- **First pass (20 min):** Read the abstract and contributions to internalize the key insight that standard distillation fails for Transformer→Mamba; sketch the teacher/student architecture comparison.
- **Second pass (1.5 hours):** Study the two-stage distillation objectives, especially the kernel-trick derivation; try reproducing the perplexity gap between single-stage and two-stage distillation on Pythia-160M/410M.
- **Third pass (hands-on):** Use `code/attention_to_mamba_distillation.py` in this directory to implement a simplified two-stage distillation in PyTorch; focus on how the attention-to-SSM kernel correspondence loss affects convergence.
- **Critical thinking:** If the kernel trick only approximates local attention, will tasks requiring global dependencies (e.g., long-range coreference resolution) still degrade? Design an evaluation on equation solving or long-document reasoning to test this.
