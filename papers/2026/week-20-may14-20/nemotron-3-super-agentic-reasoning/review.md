# Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning / Nemotron 3 Super：面向代理推理的开源高效 MoE-Mamba 混合 Transformer 模型

- **arXiv:** [2605.04217](https://arxiv.org/abs/2605.04217)
- **Date:** 2026-05-16
- **Authors:** NVIDIA Research, Meta GenAI, Stanford HAI (authors listed in supplementary material)
- **Target venues:** ICML 2026, NeurIPS 2026

---

## Key Excerpts / 关键摘录

> "We present Nemotron 3 Super, an open-weights model that achieves frontier-level agentic reasoning with only 37 billion active parameters by interleaving sparse Mixture-of-Experts (MoE) layers with structured state-space (Mamba-3) blocks." / “我们提出 Nemotron 3 Super，一种开放权重模型，通过交替稀疏专家混合（MoE）层与结构化状态空间（Mamba-3）模块，仅以 370 亿活跃参数达到前沿级代理推理能力。”

> "The hybrid architecture reduces long-context inference latency by 4.2x compared with a dense transformer of equivalent quality on a 128K-token benchmark, while improving tool-use accuracy by 6.3% on AgentBench-2026." / “在 128K token 基准上，混合架构相比同等质量的稠密 Transformer 将长上下文推理延迟降低 4.2 倍，同时在 AgentBench-2026 上将工具调用准确率提升 6.3%。”

> "We introduce routed state-space attention (RSSA), which dynamically dispatches tokens to either a local Mamba recurrence or a sparse attention expert based on a lightweight router trained with load-balancing auxiliary losses." / “我们提出路由式状态空间注意力（RSSA），通过带有负载均衡辅助损失的轻量路由器，动态将 token 分派给局部 Mamba 递归或稀疏注意力专家。”

> "A comprehensive open-source release includes model weights, training data recipes, and inference kernels optimized for both NVIDIA Blackwell and consumer RTX GPUs." / “全面的开源发布包含模型权重、训练数据配方以及针对 NVIDIA Blackwell 和消费级 RTX GPU 优化的推理内核。”

---

## Q1. Core Problem / 核心问题

**English:**
Frontier AI capabilities are increasingly gated by inference cost, latency, and energy consumption. Dense transformer models with hundreds of billions of parameters can reason and act as agents, but they require massive activation budgets and quadratic attention costs that make long-context agentic workflows economically impractical. The core problem is therefore to preserve frontier-level reasoning, tool use, and long-context comprehension while operating at a small active-parameter budget and near-linear sequence complexity.

**中文：**
前沿 AI 能力正日益受到推理成本、延迟和能耗的限制。拥有数千亿参数的稠密 Transformer 模型能够进行推理并作为智能体行动，但它们需要巨大的激活预算和二次方注意力成本，使得长上下文代理工作流在经济上难以承受。因此，核心问题在于：如何在较小的活跃参数预算和接近线性的序列复杂度下，保持前沿级推理、工具使用与长上下文理解能力。

---

## Q2. Claimed Contributions / 声称贡献

**English:**
1. **Hybrid MoE + Mamba-3 architecture:** Interleaves sparse MoE feed-forward layers with Mamba-3 state-space blocks and occasional sparse attention layers, achieving high quality at low active-parameter count.
2. **Routed State-Space Attention (RSSA):** A token-level router decides, for each layer and each token, whether to use local recurrence (Mamba) or sparse attention, with load-balancing losses to avoid expert collapse.
3. **Frontier agentic performance at 37B active parameters:** Matches or exceeds GPT-5-mini and Claude-4-Sonnet on AgentBench-2026, SWE-bench Pro, and multi-turn function-calling benchmarks.
4. **128K-context efficiency:** Sub-quadratic scaling enables 4.2x latency reduction and 3.1x lower memory footprint than dense equivalents at 128K tokens.
5. **Full open-source stack:** Releases weights, data-mixture recipes, training code, and fused kernels for community reproducibility and downstream adaptation.

**中文：**
1. **混合 MoE + Mamba-3 架构：** 将稀疏 MoE 前馈层与 Mamba-3 状态空间模块及少量稀疏注意力层交错堆叠，以较少的活跃参数实现高质量。
2. **路由式状态空间注意力（RSSA）：** 逐层、逐 token 决定使用局部递归（Mamba）还是稀疏注意力，并通过负载均衡损失避免专家崩塌。
3. **370 亿活跃参数下的前沿代理性能：** 在 AgentBench-2026、SWE-bench Pro 和多轮函数调用基准上达到或超越 GPT-5-mini 与 Claude-4-Sonnet。
4. **128K 上下文效率：** 亚二次方扩展使 128K token 下的延迟降低 4.2 倍，内存占用减少 3.1 倍。
5. **完整开源栈：** 发布权重、数据配比配方、训练代码与融合内核，支持社区复现与下游适配。

---

## Q3. Reviewer Attack Points / 审稿人攻击点

**English:**
1. **Router collapse and load imbalance:** While auxiliary losses are reported, the paper does not provide a full ablation showing that RSSA routing remains stable across diverse long-context distributions; small shifts in workload could degrade load balance.
2. **Evaluation cherry-picking concern:** AgentBench-2026 and SWE-bench Pro are leaderboard benchmarks heavily optimized by the open community; stronger evidence would include held-out, adversarial tool-use tasks not in public training corpora.
3. **Comparison with dense models:** The dense baseline is described as "equivalent quality" but it is unclear whether it was trained with the same data, compute, and hyperparameters; fair comparison may require iso-FLOP or iso-training-time controls.
4. **State-space limitations on long-range dependencies:** Mamba variants historically struggle with selective copying and associative recall tasks compared with full attention; the paper should report results on standard synthetic long-context tasks (e.g., Needle-in-Haystack, PassKey).
5. **Hardware-specific kernel gains:** The latency claims rely heavily on custom CUDA kernels for Blackwell; generalization to other accelerators (AMD MI300, Apple Silicon, Qualcomm) is not demonstrated.

**中文：**
1. **路由器崩塌与负载不均衡：** 虽然报告了辅助损失，但论文未提供完整消融实验，证明 RSSA 路由在多样化长上下文分布上保持稳定；工作负载的小幅偏移可能破坏负载均衡。
2. **评测选择偏差：** AgentBench-2026 与 SWE-bench Pro 是被开源社区大量优化的排行榜基准；更有力的证据应包含未出现在公开训练语料中的对抗性工具使用任务。
3. **与稠密模型的比较：** “同等质量”的稠密基线是否使用相同数据、算力与超参数训练尚不明确；公平的比较需要 iso-FLOP 或 iso-training-time 控制。
4. **状态空间在长程依赖上的局限：** 相比完整注意力，Mamba 变体在选择性复制和关联回忆任务上历来存在困难；论文应报告标准合成长上下文任务（如 Needle-in-Haystack、PassKey）的结果。
5. **硬件相关内核收益：** 延迟声明高度依赖面向 Blackwell 的定制 CUDA 内核；未展示向其他加速器（AMD MI300、Apple Silicon、Qualcomm）的泛化能力。

---

## Q4. PhD Reading Guide / 博士阅读指南

**English:**
- **Pre-requisites:** Read the original Mamba paper (Gu & Dao, 2023) and the Mixtral-8x7B MoE paper to understand the two building blocks. Then review a recent efficient-attention survey to place sparse attention alternatives in context.
- **Study sequence:** Start with Section 3 (architecture) and Section 4.1 (RSSA routing). Then read the training recipe in Section 5 before looking at the benchmark tables in Section 6.
- **Critical reading questions:** (1) How does the router generalize to context lengths not seen during training? (2) What happens to tool-use accuracy if you replace RSSA with pure Mamba or pure sparse attention? (3) Are the reported gains primarily from architecture, data mixture, or post-training?
- **Reproduction tips:** The released kernels are optimized for Blackwell; for non-NVIDIA hardware, start with the reference PyTorch implementation in `code/` and scale down to a 1B-parameter toy model on OpenHermes or a synthetic agent dataset.
- **Follow-up ideas:** Combine RSSA with KV-cache compression, apply the hybrid block to vision-language sequences, or study the theoretical expressivity of MoE+Mamba mixtures compared with standard transformers.

**中文：**
- **前置知识：** 阅读原始 Mamba 论文（Gu & Dao, 2023）与 Mixtral-8x7B MoE 论文以理解两大构件；再阅读一篇高效注意力综述，将稀疏注意力替代方案置于上下文中。
- **阅读顺序：** 从第 3 节（架构）和第 4.1 节（RSSA 路由）开始；然后阅读第 5 节的训练配方，再看第 6 节的基准表。
- **批判性问题：** (1) 路由器如何泛化到训练时未见过的上下文长度？(2) 若将 RSSA 替换为纯 Mamba 或纯稀疏注意力，工具调用准确率如何变化？(3) 报告的收益主要来自架构、数据配比还是后训练？
- **复现建议：** 发布的内核针对 Blackwell 优化；对于非 NVIDIA 硬件，可从 `code/` 中的参考 PyTorch 实现开始，在 OpenHermes 或合成智能体数据集上缩小到 10 亿参数玩具模型。
- **延伸思路：** 将 RSSA 与 KV 缓存压缩结合、将混合块应用于视觉语言序列，或研究 MoE+Mamba 混合相比标准 Transformer 的理论表达能力。
