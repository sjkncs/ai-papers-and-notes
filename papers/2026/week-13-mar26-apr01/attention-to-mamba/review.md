# Attention to Mamba: A Recipe for Cross-Architecture Distillation

> **arXiv:** [2604.14191](https://arxiv.org/abs/2604.14191) | **日期 / Date:** 2026-03-31 | **作者 / Authors:** Abhinav Moudgil, Ningyuan Huang, Eeshan Gunesh Dhekane, Pau Rodríguez, Luca Zappella, Federico Danieli

---

## Q1: 核心问题

**中文：** SSM(Mamba)在推理效率上优于Transformer，但从头训练SSM难以达到同等性能。如何从已训练的Transformer中蒸馏知识到Mamba，让Mamba在保持效率优势的同时获得Transformer的表达能力？

**English:** SSMs (Mamba) outperform Transformers in inference efficiency, but training SSMs from scratch struggles to match Transformer performance. How to distill knowledge from trained Transformers into Mamba, maintaining efficiency while gaining Transformer-level expressiveness?

## Q2: 核心贡献

1. **原则化初始化**: 从Transformer线性化版本推导出Mamba的初始化方案
2. **跨架构蒸馏管线**: 完整的Teacher(Transformer)→Student(Mamba)蒸馏流程
3. **性能保持**: 蒸馏后Mamba的困惑度接近Teacher
4. **效率优势保持**: 蒸馏不损害Mamba的推理速度优势

## Q3: 审稿攻击点

1. 蒸馏在不同规模模型上的泛化性？ 2) 长序列上蒸馏效果是否衰减？ 3) 金融时序的特殊性？

## Q4: 量化金融映射

- 从大型金融Transformer蒸馏到轻量Mamba → 低延迟交易部署
- 跨架构知识迁移 → 用离线大模型指导在线小模型
- 长序列效率优势 → 多年日频数据的端到端处理
