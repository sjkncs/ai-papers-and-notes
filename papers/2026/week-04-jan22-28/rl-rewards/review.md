# R³: Replay, Reflection, and Ranking Rewards for LLM Reinforcement Learning

> **arXiv:** [2601.19620](https://arxiv.org/abs/2601.19620) | **日期 / Date:** 2026-01-26 | **作者 / Authors:** Zhizheng Jiang, Kang Zhao, Weikai Xu, Xinkui Lin, Wei Liu, Jian Luan, Shuo Shang, Peng Han

---

## 关键摘录 / Key Excerpts

> 1. "When intra-group advantages collapse under challenging tasks, standard RL optimization becomes nearly random — we need structural alternatives to step-by-step reward labeling."
>    / "当困难任务下组内优势坍塌时，标准RL优化近乎随机——我们需要逐步奖励标注的结构性替代方案。"

> 2. "Our tripartite reward system — Replay for cross-context advantage, Reflection for in-context refinement, and Ranking for structural entropy-based scoring — eliminates the need for expensive step-by-step annotations."
>    / "我们的三重奖励系统——Replay用于跨上下文优势、Reflection用于上下文内精炼、Ranking用于基于结构熵的评分——消除了昂贵的逐步标注需求。"

---

## Q1: 核心问题 / Core Problem

**中文：**
当前LLM的RL训练（如GRPO）面临"优势坍塌"问题：当一组采样回答中大部分质量相近时，优势值接近零，梯度信号消失。这在困难数学推理任务中尤为严重。

R³提出三种互补的奖励机制来替代逐步标注：
- **Replay**：跨上下文重用历史轨迹中的有价值样本
- **Reflection**：让模型从自身失败中学习并改进
- **Ranking**：基于token级熵模式对回答进行相对排序

**English:**
Current LLM RL training (e.g., GRPO) faces "advantage collapse": when most sampled responses in a group are similar quality, advantage values approach zero and gradient signals vanish. This is especially severe in hard math reasoning tasks.

R³ proposes three complementary reward mechanisms replacing step-by-step labeling:
- **Replay**: Cross-context reuse of valuable historical trajectory samples
- **Reflection**: Model learns from its own failures and self-improves
- **Ranking**: Relative response scoring based on token-level entropy patterns

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **Replay机制**：维护历史轨迹缓存，在新训练中重新利用过去的高质量样本，维持跨批次的优势差异
2. **Reflection机制**：模型生成初步回答后，提供自身错误信息让其自我修正，修正后的回答获得更高奖励
3. **Ranking机制**：引入"结构熵"度量——低熵token占比反映回答的确定性/信心，用于自动排序
4. **组合效果**：三个机制在数学推理benchmark上达到SoTA，同时减少了推理token数

**English:**

1. **Replay**: Historical trajectory cache, reusing high-quality past samples in new training to maintain cross-batch advantage differences
2. **Reflection**: After initial response, model receives its own error info for self-correction; corrected response gets higher reward
3. **Ranking**: "Structural entropy" metric — proportion of low-entropy tokens reflects response certainty/confidence, used for automatic ranking
4. **Combined Effect**: All three achieve SoTA on math reasoning benchmarks while reducing reasoning token count

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **Replay缓存管理**：缓存大小、更新策略、过时数据的处理
2. **Reflection的自举风险**：模型自我修正是否引入了确认偏差？
3. **结构熵的泛化性**：在非数学任务（如创意写作）中是否有效？
4. **计算开销**：三重奖励系统的额外计算成本

**English:**

1. **Replay Cache Management**: Cache size, update policy, stale data handling
2. **Reflection Bootstrapping Risk**: Does self-correction introduce confirmation bias?
3. **Structural Entropy Generalization**: Effective in non-math tasks (e.g., creative writing)?
4. **Computational Overhead**: Extra cost of triple reward system

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**推荐阅读顺序：**
1. 第3节——三重奖励机制的技术细节
2. 第4节——结构熵的数学定义和直觉
3. 第5节——消融实验（每个组件的独立贡献）

**量化金融映射方向：**
- Replay机制 → 历史交易轨迹重放用于策略RL训练
- Reflection → 交易决策自我审视：模型回顾并修正交易信号
- Ranking → 多策略输出的自动排序与选择

**English:**

**Recommended Reading Order:**
1. Section 3 — technical details of triple reward mechanisms
2. Section 4 — structural entropy mathematical definition and intuition
3. Section 5 — ablation studies (each component's independent contribution)

**Quant Finance Mapping:**
- Replay → Historical trade trajectory replay for strategy RL training
- Reflection → Trade decision self-review: model reviews and corrects trade signals
- Ranking → Automatic ranking and selection of multi-strategy outputs
