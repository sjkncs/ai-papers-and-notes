# Symmetry in Language Statistics Shapes the Geometry of Model Representations

> **arXiv:** [2602.15029](https://arxiv.org/abs/2602.15029) | **日期 / Date:** 2026-02-16 | **作者 / Authors:** Dhruva Karkada, Daniel J. Korchinski, Andres Nava, Matthieu Wyart, Yasaman Bahri

---

## 关键摘录 / Key Excerpts

> 1. "Translational invariance — where co-occurrences depend solely on relative distances rather than absolute positions — forces representation spaces to adopt specific manifold topologies."
>    / "平移不变性——共现仅依赖相对距离而非绝对位置——迫使表征空间采用特定的流形拓扑。"

> 2. "We mathematically demonstrate that translation symmetry fundamentally sculpts the geometric architecture of learned neural representations."
>    / "我们从数学上证明了平移对称性从根本上塑造了学习到的神经表征的几何架构。"

---

## Q1: 核心问题 / Core Problem

**中文：**
神经网络为何将概念组织成特定的空间模式（如时间单位排成圆形或直线）？本文揭示：人类语言的统计规律性直接决定了向量空间中的结构形态。

核心发现：当共现统计具有平移不变性时，表征空间必然采用圆/环面等对称拓扑。当潜在变量控制共现时，表征获得对应的几何结构。

**English:**
Why do neural networks organize concepts into specific spatial patterns (temporal units in circles or lines)? This paper reveals: statistical regularities in human language directly dictate structural forms in vector spaces.

Core finding: When co-occurrence statistics have translational invariance, representation spaces necessarily adopt symmetric topologies like circles/tori. When latent variables control co-occurrences, representations acquire corresponding geometric structures.

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **理论证明**：平移不变性 → 圆/环面拓扑的严格数学推导
2. **实验验证**：在大规模AI系统的表征中验证理论预测
3. **物理视角**：将统计物理的对称性分析框架引入表征学习
4. **潜在变量发现**：当隐藏变量控制信息时，表征结构自然涌现

**English:**

1. **Theoretical Proof**: Rigorous mathematical derivation of translational invariance → circle/torus topology
2. **Empirical Validation**: Theory predictions verified in large-scale AI system representations
3. **Physics Perspective**: Statistical physics symmetry analysis framework introduced to representation learning
4. **Latent Variable Discovery**: Representation structures naturally emerge when hidden variables control information

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **模型规模依赖**：理论预测是否在所有模型规模上都成立？
2. **任务特异性**：不同训练任务是否影响对称性结构？
3. **实际利用**：如何利用已知的几何结构改进模型？
4. **金融时序适用性**：时序数据的平移不变性假设在金融中是否合理？

**English:**

1. **Model Scale Dependence**: Do theoretical predictions hold at all model scales?
2. **Task Specificity**: Do different training tasks affect symmetry structures?
3. **Practical Exploitation**: How to leverage known geometric structures to improve models?
4. **Financial Time Series Applicability**: Is translational invariance assumption reasonable for financial time series?

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**量化金融映射方向：**
- 因子表征几何：市场因子（动量、价值等）是否形成特定的几何拓扑？
- 时序对称性：金融时序中是否存在近似的平移不变性？(regime-dependent)
- 流形学习：在金融数据的低维流形上发现alpha信号
- 对称性破缺：市场regime切换对应表征几何的拓扑变化

**English:**

**Quant Finance Mapping:**
- Factor representation geometry: Do market factors (momentum, value, etc.) form specific geometric topologies?
- Temporal symmetry: Does approximate translational invariance exist in financial time series? (regime-dependent)
- Manifold learning: Discovering alpha signals on low-dimensional manifolds of financial data
- Symmetry breaking: Market regime switches correspond to topological changes in representation geometry
