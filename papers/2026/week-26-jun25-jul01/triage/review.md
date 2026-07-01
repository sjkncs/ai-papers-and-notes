# TRIAGE: Role-Typed Credit Assignment for Agentic Reinforcement Learning

**Authors:** Yuanda Xu, Zhengze Zhou, Hejian Sang, Xiaomin Li, Jiaxin Zhang, Xinchen Du, Zhipeng Wang, Alborz Geramifard  
**Affiliations:** LinkedIn Corporation, Harvard University, Johns Hopkins University  
**arXiv:** [2606.32017](https://arxiv.org/abs/2606.32017)  
**Date:** June 30, 2026

---

## Q1: 它真正想解决的问题是什么？ / What problem does it really solve?

**中文：** 在agentic reinforcement learning中，标准GRPO只使用最终的verifier outcome作为整条trajectory的advantage信号，并均匀广播给所有action token。这种outcome-only credit存在两个结构性盲区：第一，失败的trajectory中可能包含有用的探索动作（如搜索、检查），这些动作不应被完全惩罚；第二，成功的trajectory中可能包含冗余甚至有害的动作（如错误编辑、重复点击），这些动作不应因为是成功轨迹的一部分就获得正credit。TRIAGE要解决的问题是：**如何为agentic trajectory中的每个environment-facing segment分配与其语义角色相匹配的credit，而不是简单地继承整条轨迹的outcome。**

**English:** In agentic reinforcement learning, standard GRPO uses only the final verifier outcome as the advantage signal for the entire trajectory and broadcasts it uniformly to all action tokens. This outcome-only credit has two structural blind spots: first, failed trajectories may contain useful exploratory actions (searches, inspections) that should not be fully punished; second, successful trajectories may contain redundant or harmful actions (wrong edits, repeated clicks) that should not receive positive credit merely because the trajectory eventually succeeds. TRIAGE asks: **how can we assign credit to each environment-facing segment in an agentic trajectory according to its semantic role, rather than simply inheriting the trajectory-level outcome?**

> "Standard GRPO uses the final verifier outcome as a uniform advantage over all action tokens. This outcome signal is useful but structurally incomplete: it punishes useful exploration in failed rollouts and reinforces redundant or regressive actions in successful rollouts."

---

## Q2: 它声称的贡献是什么？ / What are the claimed contributions?

**中文：**
1. **语义角色分类法：** 提出四角色分类（Decisive / Exploration / No-progress / Regression），将agentic segment credit从单一outcome轴扩展为outcome + role双轴。
2. **TRIAGE框架：** 使用结构化的LLM judge作为role classifier（而非无约束的reward model），将每个segment归类后，通过固定的role-conditioned规则映射到bounded segment-level process rewards。
3. **理论保证：** 证明role-conditioned credit是从role labels alone可表达的最小均方误差（MSE-optimal）segment-level correction，能降低advantage estimation error和policy gradient variance。
4. **多基准验证：** 在ALFWorld、Search-QA、WebShop上均超越GRPO baseline，并且在completed rollouts中减少environment-facing turns 10.4%-14.8%。

**English:**
1. **Semantic role taxonomy:** A four-role taxonomy (Decisive / Exploration / No-progress / Regression) expands segment credit from a single outcome axis to an outcome + role dual axis.
2. **TRIAGE framework:** Uses a structured LLM judge as a role classifier (not an unconstrained reward model), maps each segment to fixed role-conditioned rules that produce bounded segment-level process rewards.
3. **Theoretical justification:** Proves that role-conditioned credit is the MSE-optimal segment-level correction expressible from role labels alone, reducing advantage-estimation error and policy-gradient variance.
4. **Multi-benchmark validation:** Outperforms GRPO on ALFWorld, Search-QA, and WebShop, and reduces environment-facing turns by 10.4%-14.8% on completed rollouts.

> "We propose TRIAGE, a role-typed credit assignment framework that adds a semantic role axis to outcome credit. A structured judge classifies each segment as decisive progress, useful exploration, no-progress infrastructure, or regression, and a fixed role-conditioned rule maps these labels to bounded segment-level process rewards."

---

## Q3: 最可能被reviewer攻击的地方在哪里？ / Most likely reviewer attack points?

**中文：**
1. **Judge的可靠性与成本：** TRIAGE依赖LLM judge对每个segment进行角色分类。如果judge本身出错（尤其D/E边界模糊），可能引入系统性噪声。此外，在线训练时每步都调用LLM judge的成本是否可接受？论文未充分讨论inference开销。
2. **角色分类的主观性：** “探索” vs “无进展” vs “回归”的边界在某些任务中可能非常模糊。Reviewer可能质疑这种分类是否足够robust，以及是否需要大量人工设计的rubric。
3. **与PRM/VLMs的对比不够充分：** 论文对比了scalar judge-derived PRM和outcome-supervised value baseline，但没有与近期更强的PRM方法（如MCTS-based PRM、self-consistency PRM）对比。
4. **理论结果的假设过强：** MSE-optimal correction的证明基于role labels可靠的假设，但现实中judge有噪声，此时最优性是否仍成立？

**English:**
1. **Judge reliability and cost:** TRIAGE relies on an LLM judge to classify every segment. If the judge errs—especially at the D/E boundary—it introduces systematic noise. The inference cost of calling an LLM judge at every training step is also not thoroughly discussed.
2. **Subjectivity of role taxonomy:** The boundaries between exploration, no-progress, and regression can be fuzzy in some tasks. Reviewers may question whether the taxonomy is robust and whether it requires heavy hand-crafted rubrics.
3. **Insufficient comparison with stronger PRMs:** The paper compares against a scalar judge-derived PRM and an outcome-supervised value baseline, but not against stronger recent PRM methods such as MCTS-based PRMs or self-consistency PRMs.
4. **Strong assumptions in theory:** The MSE-optimality proof assumes reliable role labels, which may not hold in practice when the judge is noisy.

> "The judge does not need perfect D/E boundary agreement. Its key capability is asymmetric error correction: in successful rollouts, find local regressions that should not inherit positive credit; in failed rollouts, find locally useful segments that should not inherit full negative credit."

---

## Q4: 同方向博士生应精读哪些、跳过哪些？ / PhD reading guide: read carefully vs. skip?

**中文：**
- **必读 / Read carefully:**
  - **Section 1 & 3:** 对outcome-only credit两个结构性盲区的分析非常到位。
  - **Table 1 & Table 2:** 四角色分类法和默认credit rule，一目了然。
  - **Section 4.1:** 理论证明连接了role-conditioned credit与lower-variance policy gradients。
  - **Section 5:** 实验设计，特别是ablations显示regression suppression是主导贡献。
- **可跳 / Skim or skip:**
  - **完整的LLM judge prompt/rubric:** 除非要复现，否则浏览即可。
  - **Appendix中各数据集的具体设置:** 按需查看。
  - **Related Work中PRM综述:** 若已熟悉，可快速浏览。

**English:**
- **Read carefully:**
  - **Sections 1 & 3:** The analysis of the two structural blind spots of outcome-only credit is excellent.
  - **Tables 1 & 2:** The four-role taxonomy and default credit rules are very clear.
  - **Section 4.1:** The theory connecting role-conditioned credit to lower-variance policy gradients.
  - **Section 5:** Experiments, especially ablations showing regression suppression is the dominant contributor.
- **Skim or skip:**
  - **Full LLM judge prompt/rubric:** Skim unless reproducing.
  - **Dataset-specific setup in appendix:** Check as needed.
  - **PRM survey in related work:** Skim if already familiar.

---

## Key Original Quotes / 关键原文摘录

> "Agentic trajectories need a second axis: the local semantic role of each segment."

> "Our central claim is therefore: agentic RL needs a role axis in addition to an outcome axis. The most important distinction is that exploration is not no-progress."

> "Like medical triage, which sorts patients by the kind of attention they need before allocating treatment, TRIAGE first sorts each environment-facing segment into a semantic role before deciding how much credit it should inherit from the trajectory outcome."
