# Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning

**arXiv:** [2605.06130](https://arxiv.org/abs/2605.06130)  
**Date:** 2026-05-07  
**Authors:** Yaorui Shi, Yuxin Chen, Zhengxi Lu, Yuchun Miao, Shugui Liu, Qi GU, Xunliang Cai, Xiang Wang, An Zhang

---

## Key Excerpts / 核心摘录

> "A persistent skill library allows language model agents to reuse successful strategies across tasks."
>
> 持久的技能库让语言模型智能体能够跨任务复用成功策略。

> "We consolidate skill internalization and utilization into a single agent loop, unifying the previously separated processes of skill discovery and skill application."
>
> 我们将技能内化与利用整合进单一智能体循环，统一了此前分离的技能发现与技能应用过程。

> "Skill1 introduces skill tokens that are stored, retrieved, and updated through reinforcement learning, enabling end-to-end evolution of the skill library."
>
> Skill1 引入可存储、检索并通过强化学习更新的技能 token，实现技能库的端到端演化。

> "Empirical gains are reported on web navigation and interactive decision-making benchmarks when compared with both vanilla LLM agents and prior skill-augmented baselines."
>
> 在网页导航与交互式决策基准上，相比原始 LLM 智能体与早期技能增强基线均取得经验性提升。

---

## Q1 Core Problem / 核心问题

Current LLM-based agents can solve isolated tasks via in-context reasoning, but they fail to accumulate reusable knowledge across episodes. Earlier skill-based systems usually split the pipeline into a skill-discovery phase and a skill-application phase, which causes inconsistent credit assignment and makes it hard to improve the library from task feedback.

当前基于 LLM 的智能体可以通过上下文推理解决孤立任务，但无法在多个回合之间积累可复用知识。早期的技能系统通常将流程划分为技能发现与技能应用两个阶段，导致信用分配不一致，并难以根据任务反馈改进技能库。

---

## Q2 Claimed Contributions / 声称贡献

1. **Unified evolution framework.** Skill1 trains a single agent that both internalizes new skills into a library and retrieves existing skills during action selection.
   **统一演化框架。** Skill1 训练一个既能将新技能内化入库、又能在动作选择时检索已有技能的单一智能体。

2. **Skill tokens as first-class objects.** Skills are represented as differentiable tokens that can be stored, updated, and composed through gradient-based RL.
   **技能 token 作为一等对象。** 技能以可微 token 表示，可通过基于梯度的 RL 进行存储、更新与组合。

3. **End-to-end library learning.** The skill library is not handcrafted or frozen after discovery; it co-evolves with the policy via environment reward.
   **端到端库学习。** 技能库不是手工设计或在发现后冻结，而是随策略通过环境奖励共同演化。

4. **Strong empirical results.** The authors report improved success rates on interactive decision-making benchmarks over both vanilla LLM agents and modular skill-based baselines.
   **较强的经验结果。** 作者在交互式决策基准上报告了相比原始 LLM 智能体与模块化技能基线更高的成功率。

---

## Q3 Reviewer Attack Points / 审稿人攻击点

1. **Skill representation is under-motivated.** The paper does not fully explain why skill tokens are the right abstraction, or how they compare to program-like skill descriptions used in Voyager / Synapse.
   **技能表示的动机不足。** 论文未充分解释为何 skill token 是正确的抽象，也未与 Voyager / Synapse 中类程序的技能描述进行充分比较。

2. **Limited benchmark coverage.** The evaluation tasks may be too narrow to claim general cross-task transfer; a broader suite across coding, web, and embodied domains is missing.
   **基准覆盖有限。** 评估任务可能过于狭窄，难以支撑通用跨任务迁移的断言；缺少横跨代码、网页与具身场景的更广泛测试集。

3. **Retrieval mechanism details are sparse.** It is unclear how the agent decides when to retrieve a skill versus relying on base LLM reasoning, and how retrieval errors are corrected.
   **检索机制细节不足。** 尚不清楚智能体何时检索技能、何时依赖基础 LLM 推理，以及检索错误如何修正。

4. **Catastrophic forgetting risk.** Continuously updating a shared skill library may overwrite old skills; the paper does not report forgetting curves or ablations on library size.
   **灾难性遗忘风险。** 持续更新共享技能库可能覆盖旧技能；论文未报告遗忘曲线或库规模的消融实验。

5. **Reproducibility of the library.** Without released code and a deterministic initialization of the skill vocabulary, it may be hard to reproduce the reported gains.
   **技能库的可复现性。** 若未发布代码且技能词表的初始化不确定，报告的提升可能难以复现。

---

## Q4 PhD Reading Guide / 博士阅读指南

**Time budget / 时间预算:** 2 reading sessions of 45 minutes each.

1. **Session 1: Motivation and architecture.**
   Focus on Sections 1-3. Map Skill1 onto prior work (Voyager, Synapse, AdaPlanner). Ask: what exactly is "unified" here, and why does it matter?
   **第一节：动机与架构。** 重点阅读第 1-3 节。将 Skill1 与先前工作（Voyager、Synapse、AdaPlanner）对比。思考：这里“统一”具体指什么？为何重要？

2. **Session 2: Training objective and experiments.**
   Read the RL formulation and the skill update rule carefully. Then check the appendix for implementation details and baseline hyperparameters.
   **第二节：训练目标与实验。** 仔细阅读 RL 形式化与技能更新规则，然后查看附录中的实现细节与基线超参数。

3. **Hands-on exercise / 动手练习:**
   Implement a minimal skill-augmented agent in a gridworld or text-based game. Maintain a small set of skill embeddings, train the agent with policy gradients, and measure whether retrieval improves over a baseline with only task instructions.
   在网格世界或文字游戏中实现一个最小化的技能增强智能体。维护少量技能嵌入，用策略梯度训练，并比较仅使用任务指令的基线，衡量检索是否带来提升。

4. **Extension idea / 拓展方向:**
   Add an explicit forgetting-guarded update rule (e.g., regularize old skill embeddings) and test whether it improves transfer across a sequence of diverse tasks.
   加入显式的防遗忘更新规则（例如正则化旧技能嵌入），测试其在多样化任务序列上的迁移效果是否提升。

---
