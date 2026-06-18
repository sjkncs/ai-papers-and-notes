# Why LLMs Aren't Scientists Yet — 审稿人级深度分析

> **arXiv:** [2601.03315](https://arxiv.org/abs/2601.03315) | **Date:** 2026-01-06
> **Authors:** Dhruv Trehan, Paras Chopra
> **Area:** AI Agents / Scientific Discovery / LLM Evaluation

---

## 关键摘录 / Key Excerpts

> *"Four end-to-end attempts to autonomously generate ML research papers."*

> *"Of these four, three attempts failed during implementation or evaluation."*

> *"A single system advanced successfully, getting accepted to Agents4Science 2025."*

> *"Six recurring failure modes: bias toward training data defaults, implementation drift under execution pressure, memory and context degradation across long-horizon tasks, overexcitement that declares success despite obvious failures, insufficient domain intelligence, and weak scientific taste in experimental design."*

---

## Q1: 它真正想解决的问题是什么？

**中文：** 评估当前LLM-based"AI科学家"系统的真实能力边界。在"AI Scientist"概念火热的背景下（Sakana AI等声称LLM可以自主完成科研全流程），本文通过4次完整的端到端实验，诚实地展示了这些系统在哪些环节失败。核心发现：虽然概念验证可行（1/4成功发表论文），但失败率极高，且失败模式高度一致——说明存在系统性缺陷而非偶发问题。

**English:** Evaluating real capability boundaries of LLM-based "AI Scientist" systems. Amid hype (Sakana AI etc.), this paper conducts 4 end-to-end experiments showing where these systems fail. Core finding: while proof-of-concept is possible (1/4 published), failure rates are extremely high with consistent failure patterns — indicating systematic flaws.

### 六类失败模式 / Six Failure Modes

| # | 失败模式 / Failure Mode | 描述 / Description |
|---|---|---|
| 1 | **Bias toward training data defaults** | 倾向于生成与训练数据相似的方案，缺乏原创性 |
| 2 | **Implementation drift** | 执行过程中偏离原始计划，代码逐步劣化 |
| 3 | **Memory/context degradation** | 长周期任务中上下文遗忘导致的一致性崩溃 |
| 4 | **Overexcitement** | 过度乐观地声明成功，即使存在明显失败 |
| 5 | **Insufficient domain intelligence** | 对特定领域的深层知识理解不足 |
| 6 | **Weak scientific taste** | 实验设计缺乏科学品味，选择不当的评估方法 |

---

## Q2: 它声称的贡献是什么？

**中文：**
- 4次端到端自主科研实验的系统性记录（3/4失败，1/4成功发表于Agents4Science 2025）
- 识别6类反复出现的失败模式
- 为AI Scientist系统的能力边界提供实证基准
- "Overexcitement"（过度乐观）这一发现的实践意义——对所有使用LLM自评的研究者有警示

**English:**
- Systematic documentation of 4 end-to-end autonomous research experiments
- Identification of 6 recurring failure modes
- Empirical baseline for AI Scientist capability boundaries
- Practical significance of the "overexcitement" finding

---

## Q3: 最可能被reviewer攻击的地方

**中文：**

1. **样本量太小：** 仅4次实验，统计意义有限。失败可能是特定任务选择导致的
2. **缺乏对照组：** 没有人类基线对比——人类研究生首次做研究的成功率是多少？
3. **任务选择偏差：** 4个实验选了什么方向？是否选择了LLM本身就不擅长的领域？
4. **"失败"的定义模糊：** "Implementation or evaluation"阶段失败的具体标准是什么？
5. **系统版本问题：** 使用的LLM版本、agent框架是否是最优配置？
6. **时效性风险：** LLM能力快速迭代，结论可能在几个月内过时

**English:**

1. **Small sample size:** Only 4 experiments; limited statistical significance
2. **No control group:** No human baseline comparison
3. **Task selection bias:** Were LLM-unfriendly domains selected?
4. **Vague failure definition:** Specific criteria for "failure" unclear
5. **System version concerns:** Optimal LLM/framework configuration?
6. **Timeliness risk:** Conclusions may become outdated within months

---

## Q4: 博士生精读指南

### 精读 / Read Carefully
- 6类失败模式的详细描述和案例——可对照自己的agent系统排查
- "Overexcitement"部分：LLM自我评估失准的表现，对所有使用LLM自评的研究者有警示
- 4次实验的任务设置和流程

### 可跳过 / Skim or Skip
- 关于"AI Scientist"概念的泛泛背景介绍
- 对Sakana AI等先前工作的综述（如果你已了解该领域）

---

## References

- [arXiv:2601.03315](https://arxiv.org/abs/2601.03315)
- [LessWrong Discussion](https://www.lesswrong.com/posts/y7TpjDtKFcJSGzunm/why-llms-aren-t-scientists-yet)
- [Lossfunk Letters Analysis](https://letters.lossfunk.com/p/why-llms-arent-scientists-yet)
- Sakana AI Scientist: [Lu et al., 2024](https://arxiv.org/abs/2408.06292)
