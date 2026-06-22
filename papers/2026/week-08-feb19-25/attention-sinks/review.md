# Spike, Sparse, Sink: Anatomy of Massive Activations and Attention Sinks

> **arXiv:** [2603.05498](https://arxiv.org/abs/2603.05498) | **日期 / Date:** 2026-03-04 | **作者 / Authors:** Shangwen Sun, Alfredo Canziani, Yann LeCun, Jiachen Zhu

---

## Q1: 核心问题 / Core Problem

**中文：** Transformer中反复出现两个现象：1) "Massive Activations"——特定token在某些通道上出现极端异常值；2) "Attention Sinks"——某些token不管语义如何都吸引过多注意力。二者经常重叠，但本文证明这只是pre-norm架构的副作用，而非功能必需。

**English:** Two recurring Transformer phenomena: "Massive Activations" (extreme channel outliers) and "Attention Sinks" (tokens attracting excessive attention regardless of meaning). Their overlap is merely a pre-norm architectural artifact, not functional necessity.

## Q2: 核心贡献

1. **Massive Activations全局作用**：异常激活充当"隐式参数"，跨层维持隐藏状态稳定
2. **Attention Sinks局部作用**：引导单个注意力头捕获短程依赖
3. **Pre-norm是罪魁祸首**：正是pre-norm配置迫使这两个独立机制重叠
4. **解耦验证**：移除pre-norm后成功分离两个现象

## Q3: 审稿攻击点

1. 仅分析了标准Transformer，对MoE/SSM等新架构是否适用？
2. 解耦后性能是否下降？ 3) 对量化/剪枝的影响？

## Q4: 量化金融映射

- Attention Sinks → 金融时序中的"锚定token"：首个时间步是否总是吸引过多注意力？
- Massive Activations → 因子模型中的极端值：异常激活对应市场极端事件？
- Pre-norm → 归一化设计对量化模型的影响：BatchNorm vs LayerNorm在金融预测中的差异
