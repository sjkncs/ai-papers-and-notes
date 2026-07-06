# Learning Agentic Policy from Action Guidance

**arXiv:** [2605.12004](https://arxiv.org/abs/2605.12004)  
**Date:** 2026-05-12  
**Authors:** Yuxiang Ji, Zengbin Wang, Yong Wang, Shidong Yang, Ziyu Ma, Guanhua Chen, Zonghua Sun, Liaoni Wu, Xiangxiang Chu

---

## Key Excerpts / 核心摘录

> "ActGuide-RL guides LLMs using external behavioral inputs without heavy supervised fine-tuning."
>
> ActGuide-RL 利用外部行为输入指导大语言模型，而无需大量监督微调。

> "The core idea is to treat action guidance as a soft prior over the action space and fold it into the RL objective through a guidance-aware divergence term."
>
> 核心思想是将动作指导视为动作空间上的软先验，并通过指导感知的散度项将其融入 RL 目标。

> "Our method reduces the sample complexity of agent fine-tuning while preserving the base model's general reasoning capabilities."
>
> 我们的方法降低了智能体微调的样本复杂度，同时保留了基础模型的通用推理能力。

> "Experiments on tool-use and web-navigation tasks show that even noisy guidance leads to faster convergence than pure RL or vanilla supervised fine-tuning."
>
> 在工具使用与网页导航任务上的实验表明，即使是带有噪声的指导也能带来比纯 RL 或普通监督微调更快的收敛。

---

## Q1 Core Problem / 核心问题

Training LLM agents with only sparse environment rewards is sample inefficient and often destabilizes the base model's language abilities. Supervised fine-tuning on expert trajectories avoids instability but requires large, high-quality datasets. The paper asks whether external action guidance, cheaper to obtain than full trajectories, can be converted into an RL signal that improves learning without heavy SFT.

仅使用稀疏环境奖励训练 LLM 智能体样本效率低，且常破坏基础模型的语言能力。基于专家轨迹的监督微调可避免不稳定，但需要大量高质量数据。本文探讨：是否可以将比完整轨迹更廉价的外部动作指导转化为 RL 信号，在避免大量 SFT 的同时提升学习效果。

---

## Q2 Claimed Contributions / 声称贡献

1. **Action-guidance-to-policy distillation.** ActGuide-RL converts guidance distributions into a KL-style regularizer inside a policy-gradient objective.
   **动作指导到策略的蒸馏。** ActGuide-RL 将指导分布转化为策略梯度目标中的 KL 风格正则项。

2. **No heavy SFT required.** The base LLM is kept mostly frozen; only a lightweight adapter or value head is updated during RL.
   **无需大量 SFT。** 基础 LLM 大部分保持冻结，仅在 RL 期间更新轻量适配器或价值头。

3. **Robustness to noisy guidance.** The loss down-weights guidance when it conflicts with high-return environment feedback.
   **对噪声指导的鲁棒性。** 当指导与高回报环境反馈冲突时，损失函数会降低该指导的权重。

4. **Strong empirical gains.** The method converges faster than PPO alone and outperforms SFT baselines on tool-use and web-navigation tasks.
   **较强的经验提升。** 该方法比单独 PPO 收敛更快，并在工具使用与网页导航任务上超越 SFT 基线。

---

## Q3 Reviewer Attack Points / 审稿人攻击点

1. **Quality of guidance is assumed, not controlled.** The paper relies on guidance that may be expensive or unavailable in many real domains; an ablation on guidance source diversity is needed.
   **指导质量被假设而非控制。** 论文依赖的指导在许多真实领域可能昂贵或不可用；需要关于指导来源多样性的消融实验。

2. **Suboptimality risk from biased guidance.** If the guidance systematically points to a local optimum, the KL regularizer may lock the policy into that suboptimum rather than explore.
   **有偏指导导致的次优风险。** 若指导系统性地指向局部最优，KL 正则项可能将策略锁定在该局部最优，而非探索。

3. **Comparison with IL+RL literature is thin.** There is a long history of combining imitation learning and RL; the paper should position itself against DAgger, SQIL, and recent LLM-agent IL methods.
   **与 IL+RL 文献的比较不足。** 结合模仿学习与 RL 的研究历史悠久；论文应与 DAgger、SQIL 及近期 LLM-agent IL 方法进行对比。

4. **Evaluation uses simulated guidance.** It is unclear whether gains hold when guidance comes from real human annotators or from imperfect retrieval systems.
   **评估使用模拟指导。** 尚不清楚当指导来自真实人类标注者或不完美检索系统时，提升是否仍然成立。

5. **Hyperparameter sensitivity.** The weight of the guidance term is crucial but the paper provides limited analysis of how sensitive final performance is to this weight.
   **超参数敏感性。** 指导项的权重至关重要，但论文对该权重敏感性的分析有限。

---

## Q4 PhD Reading Guide / 博士阅读指南

**Time budget / 时间预算:** 1.5 hours, split into two sessions.

1. **Session 1: Problem framing and loss design.**
   Read Sections 1-3. Draw the training loop: where does guidance enter, and how does it interact with advantage estimates?
   **第一节：问题设定与损失设计。** 阅读第 1-3 节。画出训练循环：指导在何处进入，如何与优势估计交互？

2. **Session 2: Experiments and limitations.**
   Focus on the task definitions, baseline choices, and error analysis. Pay attention to how the authors define "noisy guidance."
   **第二节：实验与局限。** 关注任务定义、基线选择与错误分析。注意作者如何定义“噪声指导”。

3. **Hands-on exercise / 动手练习:**
   Run the provided `code/actguide_rl.py` toy reproduction. Then replace the synthetic guidance with a hand-coded heuristic for the gridworld and measure the impact of guidance weight on convergence.
   运行附带的 `code/actguide_rl.py` 玩具复现。然后将合成指导替换为手写网格世界启发式，测量指导权重对收敛的影响。

4. **Extension idea / 拓展方向:**
   Implement an LLM-as-guidance module that generates action priors from task descriptions, and test whether ActGuide-RL reduces the number of environment interactions needed compared to a pure RL baseline.
   实现一个 LLM-as-guidance 模块，根据任务描述生成动作先验，测试 ActGuide-RL 是否能减少相比纯 RL 基线所需的环境交互次数。

---
