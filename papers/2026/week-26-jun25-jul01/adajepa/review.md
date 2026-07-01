# AdaJEPA: An Adaptive Latent World Model

**Authors:** Ying Wang, Oumayma Bounou, Yann LeCun, Mengye Ren  
**Affiliations:** New York University, AMI Labs  
**arXiv:** [2606.32026](https://arxiv.org/abs/2606.32026)  
**Date:** June 30, 2026  
**Project Page:** [agenticlearning.ai/adajepa](https://agenticlearning.ai/adajepa)

---

## Q1: 它真正想解决的问题是什么？ / What problem does it really solve?

**中文：** 现有的latent world model（尤其是JEPA类模型）在训练完成后通常在测试时保持冻结。然而，在真实部署环境中，即使是很小的分布偏移（如视觉噪声、光照变化、背景干扰，或物理参数变化如摩擦、质量、接触动力学）也会导致模型的一步预测误差在MPC的规划horizon上累积，最终使规划失败。AdaJEPA的核心问题是：**如何在世界模型已经部署后，仍能在MPC的闭环中持续自适应地校准模型，而不需要额外的专家标注或目标域微调数据？**

**English:** Existing latent world models, especially JEPA-style models, are typically kept frozen at test time. Yet even small distribution shifts in deployment—visual corruptions such as noise, lighting, or background distractors, or physical changes such as friction, mass, or contact dynamics—can cause one-step prediction errors to compound over the MPC planning horizon and lead to planning failure. AdaJEPA asks: **how can a world model continue to calibrate itself inside the closed loop of MPC after deployment, without extra expert labels or target-domain fine-tuning data?**

> "However, these models are typically kept frozen at test time: when their predictions become inaccurate, planning can fail, especially under test-time distribution shift."

> "Even small one-step errors can compound over the planning horizon, causing actions that appear effective in latent imagination to fail in the real environment."

---

## Q2: 它声称的贡献是什么？ / What are the claimed contributions?

**中文：**
1. **测试时自适应的JEPA世界模型：** 提出AdaJEPA，在MPC的Plan-Execute-Adapt-Replan闭环中，利用每次执行动作后观察到的真实状态转移作为自监督信号，对模型进行在线更新。
2. **轻量化且样本高效：** 每次MPC replanning step只需1个gradient step，且只更新encoder/predictor的一小部分参数，即可显著提升ID和OOD任务的成功率。
3. **跨分布偏移的鲁棒性提升：** 在PushT和PointMaze上验证了shape shift、visual shift（高斯模糊、椒盐噪声、暗光、颜色变化）等多种OOD场景下的性能提升。

**English:**
1. **Test-time adaptive JEPA world model:** AdaJEPA performs online model updates inside the Plan-Execute-Adapt-Replan loop of MPC, using the observed next-state transition after each executed action as a self-supervised signal.
2. **Lightweight and sample-efficient:** Only one gradient step per MPC replanning step, and only a small subset of encoder/predictor parameters are updated, yet it substantially improves success on both in-distribution and out-of-distribution tasks.
3. **Robustness across distribution shifts:** Evaluated on PushT and PointMaze under shape shifts, visual shifts (Gaussian blur, salt-and-pepper noise, dark lighting, color changes), and other OOD variations.

> "After training, AdaJEPA plans and executes the first action chunk, uses the observed next-state transition as a self-supervised adaptation signal, and replans with the updated model."

> "Across a range of goal-reaching tasks, AdaJEPA substantially improves planning success with as few as one gradient step per MPC replanning step."

---

## Q3: 最可能被reviewer攻击的地方在哪里？ / Most likely reviewer attack points?

**中文：**
1. **自适应会不会导致catastrophic forgetting或collapse？** 虽然论文提到只更新最后一层且使用stop-gradient，但在线自适应长期运行后是否会让模型偏离预训练表示？Reviewer可能要求更长的episode长度和渐进式退化分析。
2. **与online model-based RL的边界模糊：** 论文声称与Sutton的online MBRL不同，因为不学习value/policy。但实际上这仍是online dynamics adaptation，与MPC中的online system identification高度相关。需要更清楚地界定novelty。
3. **Buffer策略的选择影响：** recent-N vs hard-N的选择、buffer size N的设定，对结果影响有多大？如果hyperparameter敏感，则实际部署价值会打折扣。
4. **缺乏真实机器人实验：** 所有实验都在simulated PushT/PointMaze上。Reviewer可能会问：这种每步gradient update在真实机器人上是否实时可行？

**English:**
1. **Does adaptation cause catastrophic forgetting or collapse?** Although the paper notes that only last layers are updated and stop-gradient is used, reviewers may ask whether long-term online adaptation drifts the model away from pretrained representations. Longer episodes and progressive degradation analysis would help.
2. **Blurry boundary with online model-based RL:** The paper distinguishes itself from Sutton's online MBRL by not learning value/policy, but this is still online dynamics adaptation, closely related to online system identification in MPC. The novelty needs sharper articulation.
3. **Buffer strategy impact:** How sensitive are results to recent-N vs hard-N, or to buffer size N? If hyperparameters are sensitive, practical deployment value diminishes.
4. **No real robot experiments:** All experiments are in simulated PushT/PointMaze. Reviewers may question whether per-step gradient updates are real-time feasible on physical robots.

> "The adaptation is both sample- and compute-efficient, as each update modifies only a small subset of parameters and can be performed with a single gradient step on the latest transition of the current episode."

---

## Q4: 同方向博士生应精读哪些、跳过哪些？ / PhD reading guide: read carefully vs. skip?

**中文：**
- **必读 / Read carefully:**
  - **Section 1:** 对test-time distribution shift和MPC error compounding的motivation写得非常具体。
  - **Section 3.2:** Plan-Execute-Adapt-Replan loop和Algorithm 1是方法核心。
  - **Section 4.1:** OOD设置（shape shift / visual shift）定义清晰，可直接用于自己的实验设计。
  - **Figure 1:** 一张图讲清楚整体框架。
- **可跳 / Skim or skip:**
  - **Appendix B.2:** buffer策略的对比，除非要做在线自适应研究。
  - **Related Work中TTA分类部分:** 如果已经熟悉TTT/TTA，可快速浏览。
  - **完整的PushT数据生成细节:** 除非要复现环境。

**English:**
- **Read carefully:**
  - **Section 1:** The motivation on test-time distribution shift and MPC error compounding is very concrete.
  - **Section 3.2:** The Plan-Execute-Adapt-Replan loop and Algorithm 1 are the core method.
  - **Section 4.1:** The OOD setup definitions (shape shift / visual shift) are clear and reusable for your own experiments.
  - **Figure 1:** One figure clearly explains the whole framework.
- **Skim or skip:**
  - **Appendix B.2:** Buffer strategy comparison, unless you work on online adaptation.
  - **Related Work TTA taxonomy:** Skim if you already know TTT/TTA.
  - **Detailed PushT data generation:** Skip unless you reproduce the environment.

---

## Key Original Quotes / 关键原文摘录

> "To adapt world models during deployment, we propose AdaJEPA, a test-time adaptation framework that operates within the closed loop of MPC. Rather than treating the model as fixed after training, AdaJEPA uses the transition observed after each executed action as a self-supervised signal to adapt the world model before the next replan."

> "This makes learning and planning tightly coupled: the model optimizes actions using its predictions, the consequence of the actions provides the adaptation signal, and the adapted model improves prediction and thus planning for the next iteration."

> "We humans also continually update internal world models with new experience, and these updates shape subsequent decisions."
