# Reasoning on Demand: Adaptive Test-Time Compute Allocation for LLM Agents

**arXiv:** 2606.04123  
**Date:** 2026-06-05  
**Authors:** Lin-Lin Zhang, Benoit Côté, Yuki Tanaka, Aditya Krishnan, Ryan P. Adams  
**Institution:** MIT CSAIL, FAIR at Meta, University of Tokyo

---

## Key Excerpts / 核心摘录

> "We cast LLM reasoning as a sequential decision problem: at each thinking step, the agent decides whether to continue reasoning, call a tool, or emit an answer." / "我们将 LLM 推理建模为序列决策问题：在每个思考步骤，智能体决定继续推理、调用工具或输出答案。"

> "Uniform test-time scaling wastes up to 60% of reasoning tokens on sub-problems that are already solved, while under-allocating compute to genuinely ambiguous steps." / "均匀的测试时缩放会在已解决的子问题上浪费多达 60% 的推理 token，而真正模糊的步骤却得不到足够计算。"

> "Our compute allocator, a lightweight value network trained on outcome reward signals, reduces average token usage by 34% at the same accuracy on GSM8K-Hard and SWE-bench-Lite." / "我们的计算分配器是一个基于结果奖励信号训练的轻量级价值网络，在 GSM8K-Hard 和 SWE-bench-Lite 上保持同等准确率的同时，平均 token 消耗降低 34%。"

> "The optimal allocation policy is non-monotonic: adding more thinking to an early step can reduce the need for verification later." / "最优分配策略是非单调的：在较早步骤投入更多思考可以减少后续验证的需求。"

> "We release checkpoints, training data, and a reference implementation to support reproducibility of adaptive reasoning systems." / "我们发布检查点、训练数据和参考实现，以支持自适应推理系统的可复现性。"

---

## Q1 Core Problem / 核心问题

**English:**  
Large reasoning models (LRMs) typically allocate a fixed or heuristic thinking budget across all problems and all reasoning steps. This uniform approach ignores the heterogeneous difficulty of sub-problems: some steps are trivial and should be answered immediately, while others require extended exploration, tool calls, or self-verification. The paper asks whether a learned allocator can dynamically distribute test-time compute to maximize accuracy under a token budget, and whether such an allocator can generalize across domains and model scales.

**中文：**  
大型推理模型（LRM）通常在所有问题和所有推理步骤上分配固定或启发式的思考预算。这种均匀方法忽略了子问题难度的异质性：有些步骤应当立即回答，而有些步骤需要扩展探索、工具调用或自我验证。本文研究学习得到的分配器能否在 token 预算约束下动态分配测试时计算以最大化准确率，并能否跨领域和模型规模泛化。

---

## Q2 Claimed Contributions / 声称贡献

1. **A unified formalism for adaptive test-time compute / 自适应测试时计算的统一形式化**: The authors frame reasoning as a Markov Decision Process (MDP) where actions include "think", "tool", and "answer", and the reward is task correctness minus compute cost. This formulation allows off-the-shelf RL algorithms to train a compute policy.
2. **Lightweight value-network allocator / 轻量级价值网络分配器**: A small transformer-based value head predicts the expected utility of continuing to reason versus stopping, conditioned on the latent state of the base LLM. It is trained with outcome rewards from a process-supervised reward model.
3. **Non-monotonic allocation insights / 非单调分配洞见**: The paper shows empirically that the best policy sometimes spends extra compute early to avoid later mistakes, contradicting simple step-wise threshold heuristics.
4. **Strong empirical gains / 显著实验提升**: On GSM8K-Hard, MATH-500, SWE-bench-Lite, and a new multi-step web-agent benchmark, the allocator matches or exceeds uniform scaling while using 24–42% fewer reasoning tokens.
5. **Open release / 开放发布**: Code, checkpoints, and synthetic training trajectories are provided to enable reproduction and extension.

---

## Q3 Reviewer Attack Points / 审稿人攻击点

1. **Reward hacking on the value network / 价值网络的奖励作弊**: The allocator is trained with outcome rewards. Reviewers will ask whether it learns to exploit the process reward model, producing verbose but plausible-looking reasoning that does not improve real correctness. Ablation studies on reward model quality are limited.
2. **Generalization across base models / 跨基础模型泛化**: Most experiments use a single 70B-parameter LRM. The paper claims scale invariance but presents only two smaller-model experiments. It is unclear whether the allocator transfers to very large (400B+) or small (7B) models.
3. **Domain-specific hyperparameters / 领域特定超参数**: The cost–accuracy trade-off coefficient is tuned per benchmark. A single universal constant is not established, which weakens the claim of a general allocation policy.
4. **Comparison with simpler baselines / 与更简单基线的比较**: The authors compare against uniform scaling and a few hand-designed heuristics but do not include straightforward baselines such as perplexity-based early stopping or length-normalized confidence thresholds.
5. **Compute overhead of the allocator itself / 分配器自身的计算开销**: While the value head is small, it is evaluated at every reasoning step. The paper reports token savings but does not fully account for wall-clock overhead, especially for the smallest budgets.

---

## Q4 PhD Reading Guide / 博士阅读指南

**English:**  
Start with Section 3, where the MDP formulation is introduced. Compare it to prior work on test-time scaling (e.g., o1-style reasoning, self-consistency, tree-of-thoughts). Then study Section 4.2 to understand how the value head is trained with process supervision rather than per-token policy gradients. Re-implement the allocator for a small model on a simple math dataset—this newsletter includes a reference PyTorch reproduction in `code/reasoning_allocator.py`. Pay special attention to the cost–accuracy Pareto curves in Section 5.3; they are the strongest empirical argument. For a research project, consider whether your task has naturally decomposable sub-problems; if not, the allocator may add complexity without benefit.

**中文：**  
从第 3 节开始，那里引入了 MDP 形式化。将其与测试时缩放的前期工作（如 o1 风格推理、自一致性、思维树）进行比较。然后学习 4.2 节，理解价值头如何用过程监督而非逐 token 策略梯度训练。在一小型模型和简单数学数据集上复现分配器——本期周报提供了 `code/reasoning_allocator.py` 的 PyTorch 参考实现。特别关注 5.3 节的成本–准确率帕累托曲线，它们是最强的实验论据。若用于研究项目，请考虑你的任务是否具有自然可分解的子问题；若没有，分配器可能只会增加复杂度而无收益。
