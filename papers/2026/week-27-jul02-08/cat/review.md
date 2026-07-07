# CAT: Confidence-Adaptive Tree Search for Efficient Large Reasoning Models

**论文链接 / Paper:** https://arxiv.org/abs/2607.00862  
**类型 / Type:** arXiv preprint, July 2026  
**领域 / Area:** Large Reasoning Models (LRMs), Efficient Inference, Test-Time Compute, Tree Search  

---

## 1. 一句话总结 / One-Sentence Summary

CAT 提出了一种**置信度自适应树搜索**算法，使大推理模型（LRM）能够根据中间步骤的置信度动态决定是继续深入搜索还是提前终止/回退，从而在数学、代码与逻辑推理任务上以显著更低的测试时计算量达到与完整树搜索相近甚至更高的准确率。

CAT proposes a **confidence-adaptive tree search** algorithm that lets large reasoning models dynamically decide whether to keep searching, stop early, or backtrack based on the confidence of intermediate steps, achieving comparable or better accuracy than full tree search with substantially lower test-time compute on mathematical, coding, and logical reasoning tasks.

---

## 2. 研究背景与动机 / Background and Motivation

### 中文
随着 OpenAI o1/o3、DeepSeek-R1、Kimi k1.5 等“大推理模型”（LRM）的兴起，测试时计算（test-time compute）成为提升推理能力的关键杠杆。主流方法包括：

1. **Best-of-N / Majority Voting**：生成多个答案并选择最优或最一致的答案。
2. **蒙特卡洛树搜索（MCTS）**：显式构建推理树，通过价值函数或过程奖励模型（PRM）引导搜索。
3. **逐步验证与回退**：让模型在每一步判断置信度，低置信度时重新思考。

这些方法普遍存在一个效率问题：**它们往往对“简单步骤”和“困难步骤”分配相同的计算预算**。人类解题时，会在有把握的步骤快速推进，在不确定的地方才放慢思考；CAT 试图让 LRM 模仿这种策略。

### English
With the rise of “large reasoning models” (LRMs) such as OpenAI o1/o3, DeepSeek-R1, and Kimi k1.5, test-time compute has become the key lever for improving reasoning ability. Mainstream methods include:

1. **Best-of-N / majority voting**: generate multiple answers and select the best or most consistent one.
2. **Monte Carlo Tree Search (MCTS)**: explicitly build a reasoning tree guided by a value function or process reward model (PRM).
3. **Step-by-step verification and backtracking**: let the model judge confidence at each step and reconsider when confidence is low.

These methods share an efficiency problem: **they often allocate the same compute budget to “easy steps” and “hard steps.”** Humans solve problems by moving quickly through confident steps and slowing down only when uncertain. CAT aims to let LRMs mimic this strategy.

---

## 3. 方法概述 / Method Overview

### 中文
CAT 的核心思想是：**将推理过程建模为树搜索，但每个节点的扩展预算由该节点处的置信度动态决定**。

具体组件包括：

1. **步骤级置信度估计（Step-Level Confidence Estimation）**
   - 利用模型自身的 token 概率、内部隐藏状态，或一个轻量级验证器，估计当前推理步骤的正确概率。
   - 置信度可以来自 logits 的熵、答案一致性、或专门的 PRM/ORM。

2. **置信度驱动的扩展策略（Confidence-Driven Expansion）**
   - 高置信度节点：减少分支数，甚至直接单路径推进（greedy rollout）。
   - 低置信度节点：增加分支数，进行更广泛的搜索与验证。
   - 极低置信度节点：触发回退（backtracking）或重新生成前置步骤。

3. **自适应搜索深度与宽度（Adaptive Depth and Width）**
   - 整体计算预算固定时，CAT 将预算重新分配给“值得思考”的子树，而不是均匀展开。
   - 可结合 beam search、MCTS、或 lookahead planning 实现。

4. **终止与聚合（Termination and Aggregation）**
   - 当某条路径达到足够高的终局置信度时提前终止搜索。
   - 最终答案通过路径聚合（如加权投票、PRM 打分）产生。

### English
The core idea of CAT is: **model reasoning as tree search, but let the expansion budget at each node be dynamically determined by confidence at that node**.

Specific components include:

1. **Step-level confidence estimation**
   - Use the model’s own token probabilities, internal hidden states, or a lightweight verifier to estimate the probability that the current reasoning step is correct.
   - Confidence can come from logit entropy, answer consistency, or dedicated PRM/ORM models.

2. **Confidence-driven expansion policy**
   - High-confidence nodes: reduce branching or even proceed with a single greedy rollout.
   - Low-confidence nodes: increase branching for broader search and verification.
   - Very-low-confidence nodes: trigger backtracking or regeneration of preceding steps.

3. **Adaptive depth and width**
   - With a fixed total compute budget, CAT reallocates budget to subtrees that are “worth thinking about” instead of expanding uniformly.
   - Can be combined with beam search, MCTS, or lookahead planning.

4. **Termination and aggregation**
   - Search terminates early when a path reaches a sufficiently high terminal confidence.
   - The final answer is produced by aggregating paths, e.g., weighted voting or PRM scoring.

---

## 4. 实验与结果 / Experiments and Results

### 中文
CAT 在多个推理基准上进行了评估：

- **数学推理**：GSM8K、MATH-500、AIME、OlympiadBench。
- **代码生成**：HumanEval、MBPP、LiveCodeBench。
- **逻辑与常识推理**：ARC、StrategyQA、GPQA 等。

对照方法包括：

- 贪婪解码（Greedy Decoding）
- Best-of-N / Self-Consistency
- 标准 MCTS / Beam Search
- 过程奖励模型引导搜索（PRM search）

主要发现：

1. **准确率提升或持平**：在 MATH-500 上，CAT 以约 50% 的测试时调用次数达到与标准 MCTS 相当的准确率。
2. **计算效率显著改善**：在固定预算下，CAT 的 pass@1 与多数投票表现优于均匀搜索。
3. **回退机制有效**：在步骤级错误传播严重的数学证明题中，低置信度回退可减少级联错误。
4. **模型无关性**：CAT 可应用于不同规模的 LRM，包括 7B 到 70B+ 的模型。

### English
CAT was evaluated on multiple reasoning benchmarks:

- **Mathematical reasoning**: GSM8K, MATH-500, AIME, OlympiadBench.
- **Code generation**: HumanEval, MBPP, LiveCodeBench.
- **Logical and commonsense reasoning**: ARC, StrategyQA, GPQA, etc.

Baselines include:

- Greedy decoding
- Best-of-N / self-consistency
- Standard MCTS / beam search
- PRM-guided search

Key findings:

1. **Improved or comparable accuracy**: on MATH-500, CAT achieves comparable accuracy to standard MCTS with roughly 50% fewer test-time calls.
2. **Significantly better compute efficiency**: under a fixed budget, CAT’s pass@1 and majority-voting performance exceed uniform search.
3. **Backtracking is effective**: in mathematical proof problems where step-level errors propagate, low-confidence backtracking reduces cascading errors.
4. **Model-agnostic**: CAT can be applied to LRMs ranging from 7B to 70B+ parameters.

---

## 5. 审稿评价 / Critical Evaluation

### 优点 / Strengths

**中文：**

1. **问题极具现实意义**： LRM 的推理成本正在爆炸式增长，CAT 直击“如何聪明地花钱（计算）”这一核心问题。
2. **方法简洁优雅**： 置信度自适应扩展是一种通用机制，可插入现有 MCTS、Beam Search、Verifier 框架中。
3. **与人类认知一致**： “该快则快、该慢则慢”的非均匀思考策略具有直观的认知合理性。
4. **可解释性增强**： 置信度轨迹可作为推理过程的“元认知”信号，帮助调试错误步骤。

**English:**

1. **Highly practical problem**: LRM inference costs are exploding, and CAT directly attacks the core question of how to spend compute intelligently.
2. **Simple and elegant method**: confidence-adaptive expansion is a general mechanism that can be plugged into existing MCTS, beam search, and verifier frameworks.
3. **Aligned with human cognition**: the non-uniform thinking strategy of “fast when confident, slow when uncertain” has intuitive cognitive plausibility.
4. **Enhanced interpretability**: confidence traces serve as a “metacognitive” signal of the reasoning process, helping debug erroneous steps.

### 局限与问题 / Limitations and Concerns

**中文：**

1. **置信度校准问题**： 语言模型的 softmax 概率并不总是良好校准的置信度估计。如果置信度本身不可靠，自适应策略可能适得其反。
2. **奖励 hacking**： 如果置信度与最终答案正确性之间的相关性不强，模型可能学会“自信地犯错”。
3. **超参数敏感**： 分支阈值、回退阈值、预算分配函数等超参数可能需要针对任务调优。
4. **与 PRM/ORM 的边界**： 如果 CAT 使用额外的验证模型来估计置信度，其实际开销与收益需要与专用 PRM 方法仔细对比。
5. **理论保证有限**： 目前主要为经验性结果，缺少关于搜索复杂度、近似最优性、或收敛性的理论分析。

**English:**

1. **Confidence calibration issues**: language-model softmax probabilities are not always well-calibrated confidence estimates. If confidence itself is unreliable, adaptive strategies can backfire.
2. **Reward hacking**: if the correlation between confidence and final answer correctness is weak, the model may learn to “be confidently wrong.”
3. **Hyperparameter sensitivity**: branching thresholds, backtracking thresholds, and budget allocation functions may require task-specific tuning.
4. **Boundary with PRM/ORM**: if CAT uses an additional verifier model to estimate confidence, its actual overhead and benefits need careful comparison with dedicated PRM methods.
5. **Limited theoretical guarantees**: current results are largely empirical; there is little theoretical analysis of search complexity, approximate optimality, or convergence.

---

## 6. 对领域的影响 / Impact on the Field

### 中文
CAT 代表了大推理模型“测试时计算优化”方向的重要尝试。它提示我们：推理能力的提升不一定来自更多的计算，而可能来自**更聪明的计算分配**。在 o1/R1 之后，社区已经开始讨论“推理即搜索”的范式；CAT 进一步细化了这一范式，将“置信度”作为搜索控制的中心变量。未来，CAT 可能与以下方向深度融合：

- 过程奖励模型（PRM）与结果奖励模型（ORM）
- 强化学习中的信用分配与探索
- 长思维链（long CoT）的自动压缩与展开
- 多智能体辩论与验证系统

### English
CAT represents an important attempt to optimize **test-time compute** for large reasoning models. It suggests that improved reasoning may come not from more compute but from **smarter allocation of compute**. After o1/R1, the community is already discussing the paradigm of “reasoning as search”; CAT refines this paradigm by placing **confidence** at the center of search control. In the future, CAT may deeply integrate with:

- Process reward models (PRMs) and outcome reward models (ORMs)
- Credit assignment and exploration in reinforcement learning
- Automatic compression and expansion of long chains of thought
- Multi-agent debate and verification systems

---

## 7. 结论与建议 / Conclusion and Recommendation

### 中文
CAT 是本周最值得深入阅读的算法论文之一，尤其适合关注 LRM 推理效率的博士生和工程师。建议重点跟踪：

- 置信度估计器是否开源、是否可插拔；
- 在不同难度分布（简单题 vs 竞赛题）上的表现差异；
- 与现有 PRM、MCTS、Best-of-N 库（如 rStar、OpenR）的兼容性；
- 是否存在理论分析或最坏情况保证。

### English
CAT is one of the most worthwhile algorithmic papers to read deeply this week, especially for PhD students and engineers focused on LRM inference efficiency. Key points to track:

- Whether the confidence estimator is open-sourced and pluggable;
- Performance differences across difficulty distributions (easy vs. competition problems);
- Compatibility with existing PRM, MCTS, and Best-of-N libraries such as rStar and OpenR;
- Whether theoretical analysis or worst-case guarantees are provided.
