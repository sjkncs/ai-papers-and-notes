# Learning Process Rewards via Success Visitation Matching for Efficient RL

> **arXiv:** [2606.23640](https://arxiv.org/abs/2606.23640) | **日期 / Date:** 2026-06-22 | **作者 / Authors:** Raymond Tsao, Andrew Wagenmaker, Sergey Levine

---

## 关键摘录 / Key Excerpts

> 1. "In many modern applications of reinforcement learning (RL), the natural reward for a task of interest is inherently sparse: a reward of 0 is given everywhere except when the task is completed, when a reward of +1 is given."
>    / "在许多现代强化学习应用中，任务的自然奖励本质上是稀疏的：除了任务完成时获得 +1 奖励外，其余所有时刻的奖励均为 0。"

> 2. "Training a policy to maximize such a sparse reward requires solving a challenging credit assignment problem, leading to slow or ineffective RL improvement."
>    / "训练策略以最大化这种稀疏奖励需要解决一个困难的信用分配问题，导致强化学习改进缓慢或无效。"

> 3. "We propose a simple approach to transform a sparse outcome reward into a dense process reward."
>    / "我们提出一种简单的方法，将稀疏的结果奖励转化为密集的过程奖励。"

> 4. "Our approach relies on training a discriminator to distinguish between previous successful and unsuccessful episodes, and using this discriminator to incentivize the RL-learned policy to match the state-action visitations of successful episodes, while avoiding those of unsuccessful episodes."
>    / "我们的方法依赖于训练一个判别器来区分历史成功和失败的片段，并利用该判别器激励 RL 学习到的策略去匹配成功片段的状态-动作访问分布，同时避免失败片段的访问分布。"

> 5. "By incentivizing the policy to match the visitations over all states, not just those that correspond to task success, this reward provides dense feedback on whether progress is being made towards task completion, and, we show, provably achieves this without changing the optimal policy."
>    / "通过激励策略匹配所有状态上的访问分布，而不仅仅是与任务成功对应的状态，这种奖励为是否正在向任务完成取得进展提供了密集反馈，并且我们证明这可以在不改变最优策略的前提下实现。"

> 6. "SVM rewards consistently improve both the final converged success rate and the sample efficiency... reduced the necessary interaction steps to achieve target performance levels by approximately half."
>    / "SVM 奖励持续提升最终收敛成功率和样本效率……将达到目标性能所需的交互步数减少了约一半。"

---

## Q1: 核心问题 / Core Problem

**中文：**

稀疏奖励是强化学习在真实世界任务中的核心瓶颈之一。机器人操作、游戏通关、导航等任务的天然奖励往往是二元的：成功时 +1，其余时刻 0。这种稀疏性导致：

1. **信用分配困难**：策略无法判断哪些中间动作对最终成功有贡献；
2. **探索效率低下**：大量随机探索无法获得任何学习信号；
3. **真实世界样本成本高昂**：机器人每个失败 episode 都意味着物理时间和硬件磨损。

传统解决方法包括手工设计密集奖励、逆强化学习（IRL）和过程奖励模型（PRM）。但手工奖励需要领域专家，IRL 计算昂贵，PRM 通常需要大量标注或依赖 outcome-supervised 的近似。

本文真正想解决的问题是：**能否用历史成功与失败片段自动学习一个密集的过程奖励函数，既不需要人工设计奖励，也不改变原始任务的最优策略？**

**English:**

Sparse rewards are one of the core bottlenecks of reinforcement learning in real-world tasks. In robotic manipulation, game playing, navigation, and similar domains, the natural reward is often binary: +1 upon success and 0 everywhere else. This sparsity leads to:

1. **Difficult credit assignment**: the policy cannot judge which intermediate actions contributed to eventual success;
2. **Inefficient exploration**: large amounts of random exploration receive no learning signal;
3. **High real-world sample cost**: every failed robot episode consumes physical time and hardware wear.

Traditional remedies include hand-crafted dense rewards, inverse reinforcement learning (IRL), and process reward models (PRM). However, hand-crafted rewards require domain experts, IRL is computationally expensive, and PRMs usually need extensive annotation or rely on outcome-supervised approximations.

The core problem this paper addresses is: **can we automatically learn a dense process reward function from historical successful and failed episodes without hand-crafted rewards and without changing the optimal policy of the original task?**

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **Success Visitation Matching (SVM)**：提出通过训练判别器区分历史成功/失败片段，并据此构造密集过程奖励的方法。
2. **理论保证**：证明在适当条件下，最大化 SVM 奖励的策略同样最大化原始稀疏奖励（最优策略不变性）。
3. **自动化与简单性**：相比 IRL 或显式 PRM，只需要解决一个二分类问题，避免了密度估计或大量人工标注。
4. **机器人微调实验**：在 LIBERO-90 仿真基准（90 个任务、20 个场景）和 WidowX 真实机器人上验证，SVM 奖励使样本效率提升约一倍，最终成功率超过 80%。
5. **VLA 模型微调**：在视觉-语言-动作（VLA）模型微调中，SVM 过程奖励显著加速收敛。

**English:**

1. **Success Visitation Matching (SVM)**: A method that constructs dense process rewards by training a discriminator to distinguish historical successful from unsuccessful episodes.
2. **Theoretical guarantee**: Proves that, under appropriate conditions, any policy maximizing the SVM reward also maximizes the original sparse reward (policy invariance).
3. **Automation and simplicity**: Compared to IRL or explicit PRMs, it only requires solving a binary classification problem, avoiding density estimation or extensive human annotation.
4. **Robot fine-tuning experiments**: Validated on the LIBERO-90 simulation benchmark (90 tasks across 20 scenes) and on a real WidowX robot, SVM rewards approximately double sample efficiency and achieve final success rates above 80%.
5. **VLA model fine-tuning**: SVM process rewards significantly accelerate convergence when fine-tuning vision-language-action (VLA) models.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **判别器分布外泛化**：判别器在训练早期成功/失败数据上学习，但随着策略改进，新访问的状态可能超出判别器训练分布，导致奖励信号失效（reward hacking）。
2. **非平稳性**：历史成功/失败缓冲区需要不断更新，旧的成功 episode 可能对新策略不再具有参考价值，论文是否处理了这种非平稳性？
3. **多模态成功路径**：如果成功轨迹的访问分布是多模态的，判别器可能只捕获主要模态，导致策略被迫模仿“平均”成功行为而丢失更优路径。
4. **理论条件的现实性**：最优策略不变性的证明可能依赖于确定性转移或完整状态覆盖等强假设，真实机器人任务中这些假设是否成立？
5. **与 advantage/return 分解方法的比较**：审稿人会要求与 GAE、PER、RUDDER、o3 风格 PRM 等方法进行更全面的比较。
6. **失败样本的利用**：论文强调避免失败访问，但失败轨迹中的“接近成功”状态其实很有价值，简单惩罚是否会浪费这些样本？

**English:**

1. **Discriminator out-of-distribution generalization**: The discriminator is trained on early successful/failed data; as the policy improves, newly visited states may fall outside the discriminator's training distribution, causing reward hacking.
2. **Non-stationarity**: Historical success/failure buffers must be continually updated; older successful episodes may no longer be representative for the current policy. Does the paper handle this non-stationarity?
3. **Multimodal success paths**: If successful trajectory visitations are multimodal, the discriminator may capture only the dominant mode, forcing the policy to imitate an "average" successful behavior and miss better paths.
4. **Realism of theoretical conditions**: The policy-invariance proof may rely on strong assumptions such as deterministic transitions or full state coverage. Do these hold in real robot tasks?
5. **Comparison with advantage/return decomposition methods**: Reviewers will demand more comprehensive comparisons with GAE, PER, RUDDER, o3-style PRMs, etc.
6. **Utilization of failure samples**: The paper emphasizes avoiding failure visitations, but states near success in failed trajectories are valuable. Does simple penalization waste these samples?

---

## Q4: 博士生阅读指南 / PhD Reading Guide

**中文：**

**如果你是做 RL / 机器人学习 / VLA 的博士生：**

- **精读部分**：
  - 第 1–2 页的问题定义与 SVM 的核心直觉；
  - 方法部分：判别器的损失函数、SVM 奖励的数学形式、最优策略不变性证明；
  - 实验部分：LIBERO-90 的多任务结果、WidowX 真实机器人结果、VLA 微调对比。
- **跳过部分**：
  - 若熟悉稀疏奖励、过程奖励模型等背景，可跳过相关 related work；
  - 若对机器人 benchmark 不熟悉，可先阅读 LIBERO 论文了解任务设置。
- **复现建议**（本仓库提供简化 PyTorch 复现）：
  - 在 Gymnasium/MuJoCo 的简单稀疏奖励任务（如 MountainCar 稀疏版、FetchReach）上实现 SVM；
  - 维护一个成功/失败回放缓冲区，训练一个小的 MLP 判别器；
  - 将判别器输出转换为过程奖励，叠加到原始稀疏奖励上；
  - 重点观察：判别器是否需要频繁重新训练？奖励是否随策略改进而退化？
- **可延伸的研究点**：
  - 在 LLM Agent 任务中使用 SVM：将成功/失败 episode 的“状态”定义为 LLM 的轨迹历史，训练判别器判断当前历史更接近成功还是失败；
  - 结合 contrastive learning：不仅区分成功/失败，还学习成功轨迹之间的阶段相似性，构造更细粒度的过程奖励。

**English:**

**If you are a PhD student working on RL, robot learning, or VLA:**

- **Read carefully**:
  - Pages 1–2: problem definition and the core intuition behind SVM;
  - Method section: discriminator loss, mathematical form of SVM reward, and policy-invariance proof;
  - Experiments: multi-task results on LIBERO-90, real WidowX results, and VLA fine-tuning comparisons.
- **Skip if familiar**:
  - Related work on sparse rewards and process reward models if already known;
  - Robot benchmark details if you are already familiar with LIBERO.
- **Reproduction suggestions** (this repo provides a simplified PyTorch reproduction):
  - Implement SVM on simple sparse-reward Gymnasium/MuJoCo tasks (e.g., sparse MountainCar, FetchReach);
  - Maintain success/failure replay buffers and train a small MLP discriminator;
  - Convert discriminator outputs into process rewards added to the original sparse reward;
  - Key observations: does the discriminator need frequent retraining? Does the reward degrade as the policy improves?
- **Potential research extensions**:
  - Apply SVM to LLM agent tasks: treat an agent's trajectory history as the "state," train a discriminator to classify whether the current history is closer to success or failure;
  - Combine with contrastive learning: distinguish success/failure while also learning stage similarities among successful trajectories to construct finer-grained process rewards.
