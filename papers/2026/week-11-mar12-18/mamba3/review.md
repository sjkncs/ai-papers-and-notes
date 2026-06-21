# Mamba-3: Improved Sequence Modeling Using State Space Principles

> **arXiv:** [2603.15569](https://arxiv.org/abs/2603.15569) | **日期 / Date:** 2026-03-16 | **作者 / Authors:** Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu

---

## 关键摘录 / Key Excerpts

> 1. "Mamba-3 introduces three improvements guided by state space model principles: a more expressive recurrence derived from SSM discretization, a complex-valued state update rule, and a multi-input multi-output (MIMO) formulation."
>    / "Mamba-3引入三项由状态空间模型原理指导的改进：源自SSM离散化的更具表达力的递推、复值状态更新规则和多输入多输出(MIMO)公式。"

> 2. "Results show 1.8 point gain in average downstream accuracy, matching Mamba-2 with half of its predecessor's state size."
>    / "结果显示下游平均准确率提升1.8个百分点，以一半的状态规模追平Mamba-2。"

---

## Q1: 核心问题 / Core Problem

**中文：**
SSM(状态空间模型)如Mamba-1/2在长序列建模上展现优势，但仍有提升空间：
1. 实值状态限制了表达能力（无法自然表达振荡/旋转模式）
2. SISO(单输入单输出)限制了通道间信息交互
3. 离散化方法的选择影响模型的理论保证

Mamba-3从SSM的数学原理出发，系统解决这三个问题。

**English:**
SSMs like Mamba-1/2 show advantages in long-sequence modeling, but room for improvement remains:
1. Real-valued states limit expressiveness (cannot naturally represent oscillation/rotation patterns)
2. SISO limits inter-channel information exchange
3. Discretization method choices affect theoretical guarantees

Mamba-3 systematically addresses these from SSM mathematical principles.

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **SSM离散化递推 (SSM Discretization Recurrence)**：
   - 从连续SSM的精确离散化推导出更具表达力的递推公式
   - 理论上保证了连续-离散等价性

2. **复值状态更新 (Complex-Valued State Update)**：
   - 状态用复数表示，自然编码振荡和旋转模式
   - 对周期性序列（如季节性金融数据）特别有效

3. **MIMO公式 (Multi-Input Multi-Output)**：
   - 从SISO扩展到MIMO，允许通道间信息交互
   - 在多变量时序预测中优势明显

4. **效率提升**：以一半的状态规模达到Mamba-2的性能

**English:**

1. **SSM Discretization Recurrence**: More expressive recurrence from exact continuous SSM discretization; theoretical continuous-discrete equivalence guarantee.

2. **Complex-Valued State Update**: States in complex numbers naturally encode oscillation and rotation; especially effective for periodic sequences (e.g., seasonal financial data).

3. **MIMO Formulation**: SISO→MIMO enables inter-channel information exchange; clear advantage in multivariate time series prediction.

4. **Efficiency**: Matches Mamba-2 performance with half the state size.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **复值计算开销**：复数运算是否增加实际推理时间？
2. **MIMO通道扩展性**：通道数增加时MIMO的计算复杂度？
3. **与Transformer的公平对比**：是否在相同的FLOPs预算下对比？
4. **金融时序特化**：复值状态对金融周期性数据的实际提升幅度？

**English:**

1. **Complex Computation Overhead**: Does complex arithmetic increase actual inference time?
2. **MIMO Channel Scalability**: MIMO computational complexity with increasing channels?
3. **Fair Transformer Comparison**: Same FLOPs budget comparison?
4. **Financial Time Series Specialization**: Actual improvement of complex-valued states on financial periodic data?

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**量化金融映射方向：**
- Mamba-3用于高频金融序列建模（tick数据、订单簿）
- 复值状态捕获金融周期（日/周/月/年季节性）
- MIMO建模多资产联动（跨资产相关性捕获）
- 长序列效率优势→多年日频数据的端到端处理

**English:**

**Quant Finance Mapping:**
- Mamba-3 for high-frequency financial sequence modeling (tick data, order books)
- Complex-valued states capture financial cycles (daily/weekly/monthly/yearly seasonality)
- MIMO models multi-asset co-movement (cross-asset correlation capture)
- Long-sequence efficiency → end-to-end processing of multi-year daily data
