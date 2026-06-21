# Attention Residuals

> **arXiv:** [2603.15031](https://arxiv.org/abs/2603.15031) | **日期 / Date:** 2026-03-15 | **作者 / Authors:** (Attention Mechanism Researchers)

---

## Q1: 核心问题

**中文：** 标准自注意力机制缺乏残差连接的显式设计。虽然Transformer整体有残差连接，但注意力模块内部的信息流没有残差路径。本文探索在注意力计算内部引入残差连接是否能增强表达力。

**English:** Standard self-attention lacks explicit residual connections within the attention computation. While Transformers have residual connections at the block level, the internal attention information flow has no residual path. This paper explores whether introducing residuals within attention computation enhances expressiveness.

## Q2: 核心贡献

1. **Attention Residual Connection**: 在Q·K^T计算后添加残差，保留原始注意力模式的同时学习增量修正
2. **理论分析**: 证明注意力残差等价于在注意力矩阵上添加低秩修正
3. **实验验证**: 在语言建模和视觉任务上均有稳定提升
4. **零额外参数**: 残差连接不引入新参数，仅需修改前向传播

## Q3: 审稿攻击点

1. 与已有残差变体的区别？ 2) 在长序列中的梯度流动影响？ 3) 与FlashAttention的兼容性？

## Q4: 量化金融映射

- 金融时序注意力中保留历史模式信息
- 零参数改进 → 直接增强现有金融Transformer模型
- 注意力残差 → 因子模型中的增量alpha信号
