# Scaling Embeddings Outperforms Scaling Experts in Language Models

> **arXiv:** [2601.21204](https://arxiv.org/abs/2601.21204) | **日期 / Date:** 2026-01-29 | **作者 / Authors:** Hong Liu, Jiaqi Zhang, Chao Wang, Xing Hu et al.

---

## 关键摘录 / Key Excerpts

> 1. "We identify specific regimes where embedding scaling achieves a superior Pareto frontier compared to adding more experts in MoE architectures."
>    / "我们识别出在MoE架构中，embedding扩展相比增加专家数量能实现更优Pareto前沿的特定区域。"

> 2. "LongCat-Flash-Lite, a 68.5B parameter model with ~3B activated, dedicates over 30 billion weights to token representations while outperforming standard MoE baselines."
>    / "LongCat-Flash-Lite，一个68.5B参数、~3B激活的模型，将超过300亿权重分配给token表示，同时超越标准MoE基线。"

---

## Q1: 核心问题 / Core Problem

**中文：**
MoE（Mixture-of-Experts）架构通过稀疏激活降低计算成本，但面临两个瓶颈：
1. **专家数量天花板**：增加专家导致路由困难、专家利用率下降
2. **硬件适配差**：大量小专家不匹配GPU/TPU的计算模式

本文探索了一条替代路径：不增加专家，而是扩大embedding维度。将更多容量放在token表示中，而非更多子网络中。

**English:**
MoE architectures reduce compute via sparse activation but face two bottlenecks:
1. **Expert count ceiling**: More experts cause routing difficulties and low utilization
2. **Hardware mismatch**: Many small experts don't match GPU/TPU compute patterns

This paper explores an alternative: instead of more experts, expand embedding dimensions. Put more capacity in token representations rather than more sub-networks.

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **Embedding Scaling Law**: 证明在特定参数预算下，embedding维度扩展比专家数量扩展更高效
2. **LongCat-Flash-Lite模型**: 68.5B总参数/3B激活，30B+用于embedding，在代码和agent任务上超越同规模MoE
3. **硬件协同设计**: 自定义硬件适配 + 预测性token生成，将理论稀疏性转化为实际加速

**English:**

1. **Embedding Scaling Law**: Proves embedding dimension scaling is more efficient than expert count scaling under specific parameter budgets
2. **LongCat-Flash-Lite**: 68.5B total / 3B active, 30B+ in embeddings, outperforms same-scale MoE on code and agent tasks
3. **Hardware Co-design**: Custom hardware adaptation + predictive token generation, converting theoretical sparsity to actual speedup

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **Pareto前沿的适用范围**：embedding扩展是否在所有模型规模上都有效？
2. **推理效率**：大embedding是否增加推理内存（KV cache）？
3. **与其他效率方法的对比**：与GQA, KV cache压缩等的协同效果

**English:**

1. **Pareto Frontier Scope**: Is embedding scaling effective at all model scales?
2. **Inference Efficiency**: Do large embeddings increase inference memory (KV cache)?
3. **Comparison with Other Efficiency Methods**: Synergy with GQA, KV cache compression, etc.

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**量化金融映射方向：**
- 因子模型中的"专家vs维度"权衡：更多因子 vs 更丰富的因子表示
- 交易策略的容量分配：更多策略 vs 更好的特征工程
- MoE思想在量化组合中的应用：专家=子策略，路由器=市场状态

**English:**

**Quant Finance Mapping:**
- "Experts vs dimensions" tradeoff in factor models: more factors vs richer factor representations
- Trading strategy capacity allocation: more strategies vs better feature engineering
- MoE ideas in quant portfolios: experts = sub-strategies, router = market state
