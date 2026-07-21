# When Model Merging Rivals Joint Multi-Task Reinforcement Learning: A Task-Vector Geometry Analysis

**论文链接 / Paper:** https://arxiv.org/abs/2607.16062
**作者 / Author:** S. Aaron McClendon
**类型 / Type:** arXiv preprint, July 2026 (2607.16062)
**领域 / Area:** Model Merging, Multi-Task Reinforcement Learning, Task Vectors, LLM Agents

---

## Q1. 这篇论文试图解决什么问题？/ What problem does this paper address?

**中文：**

模型合并（model merging）常被宣传为联合多任务训练（joint multi-task training）的替代方案：与其把多个任务的数据混在一起重新训练一个模型，不如把已经独立训练好的多个专家模型的参数直接"加"在一起。然而在强化学习（RL）场景下，这个"替代"几乎从未与它声称要替代的真正基线做过对比——人们之所以合并独立发布的智能体，恰恰是因为拿不到一个联合训练的模型。因此本文要回答的核心问题是：**在 RL 后训练场景中，模型合并是否真的能匹敌联合多任务 RL？如果能，为什么不同的合并方法之间几乎没有差别？**

> Model merging is promoted as a substitute for joint multi-task training, yet in the reinforcement-learning setting this substitution is essentially never tested against the baseline it claims to replace: methods merge independently released agents precisely because a joint model is unavailable. We build the missing comparison.

**English:**

Model merging is widely promoted as a cheap substitute for joint multi-task training: instead of pooling multi-task data and retraining, one simply adds together the parameters of several independently trained expert models. But in the reinforcement-learning (RL) setting this "substitution" is almost never tested against the very baseline it claims to replace, because a jointly trained model is usually unavailable. The paper builds this missing comparison and asks two questions: (1) does merging RL-tuned specialists actually rival joint multi-task RL on the same data, and (2) why do very different merge algorithms (TIES, RAM+) end up statistically indistinguishable?

---

## Q2. 核心贡献是什么？/ What are the core contributions?

**中文：**

1. **补齐缺失的对照实验**：在 AppWorld 智能体基准上，用 LOOP 分别训练 difficulty-1 与 difficulty-2 的 Qwen3-8B 专家，然后把它们合并（TIES、RAM+），并与在**同样数据**上联合训练的模型正面对比。结论：在 task-goal completion 指标上，合并**匹敌**联合 RL，且每一种合并变体之间在统计上无法区分。
2. **用任务向量几何（task-vector geometry）解释"为什么合并方法无所谓"**：作者测量了专家们的任务向量（微调后参数减去基座参数），发现它们**近乎正交**（余弦相似度仅 0.06–0.10），尽管两者约有 65% 的支撑集（非零参数位置）重叠。
3. **方向与支撑集解耦（decoupling of direction and support）**：由于任务向量的方向近乎正交、而支撑集大量重叠，基于符号/支撑集裁剪的合并方法（RAM、TIES）实际上"坍缩"为近乎均匀平均——这正是各方法无差别的根本原因。
4. **严谨的几何标定**：用随机初始化的"地板"（random-init floor）和同一次训练的"天花板"（same-run ceiling）来标定那一小段共享方向，确认它反映的是"学习"而非低秩参数化的假象。
5. **开源全部代码与统计**。

> On task-goal completion, merging matches joint RL -- and every merge variant is statistically indistinguishable. [...] they are near-orthogonal (cosine 0.06 - 0.10) despite ~65% support overlap [...]. Because direction and support are decoupled, support and sign-based merging (RAM, TIES) collapse to near-uniform averaging.

**English:**

1. **The missing head-to-head comparison.** On the AppWorld agent benchmark, difficulty-1 and difficulty-2 Qwen3-8B specialists are trained with LOOP, merged (TIES, RAM+), and pitted against a jointly trained model on the *same* data. Merging *matches* joint RL on task-goal completion, and every merge variant is statistically indistinguishable.
2. **A task-vector-geometry explanation for why the merge method does not matter.** The specialists' task vectors are *near-orthogonal* (cosine 0.06–0.10) despite ~65% support overlap.
3. **Direction–support decoupling.** Because direction and support are decoupled, sign/support-based pruning methods (RAM, TIES) collapse to near-uniform averaging — explaining their indistinguishability.
4. **Careful geometric calibration** against a random-init floor and a same-run ceiling, confirming the small shared direction reflects learning rather than the low-rank parameterization.
5. **Full release of code and statistics.**

---

## Q3. 审稿人会从哪些角度攻击它？/ What would a reviewer attack?

**中文：**

1. **单种子、单基准的证据强度**：论文自承是 "exploratory"，只在 AppWorld 上、用一个模型族（Qwen3-8B）、有限种子做的匹配对比。审稿人会质疑"合并匹敌联合 RL"是否可跨基准、跨模型规模复现，统计功效是否足够支撑"统计上无法区分"的结论（无差别可能只是样本量不足）。
2. **只有两个专家、两个难度层级**：任务向量近乎正交的结论建立在 difficulty-1 vs difficulty-2 这一"同域不同难度"设定上。当任务真正异构（例如代码 vs 数学 vs 对话）或专家数量增多、任务向量出现显著干扰时，结论是否还成立？
3. **"匹敌"的天花板本身可能不高**：如果联合 RL 在 AppWorld 上的绝对表现有限，那么"合并匹敌联合 RL"也可能只是"两者都不够强"。需要与更强的多任务基线对比。
4. **近正交是否为高维必然**：在超高维参数空间中，两个方向随机也倾向于近正交（余弦接近 0）。作者虽用 random-init floor 标定，但审稿人会追问：0.06–0.10 相对随机地板的提升幅度在统计与实际意义上是否足够显著。
5. **合并权重系数的敏感性**：TIES/RAM+ 都涉及缩放/裁剪超参，"坍缩为均匀平均"的结论对这些超参是否稳健，论文披露是否充分。

**English:**

1. **Single-seed, single-benchmark evidence.** The paper is explicitly "exploratory": one benchmark (AppWorld), one model family (Qwen3-8B), limited seeds. Reviewers will question whether "merging matches joint RL" replicates across benchmarks and scales, and whether the statistical power is sufficient to support "indistinguishable" (which may merely reflect small sample size).
2. **Only two specialists, two difficulty tiers.** Near-orthogonality is measured on a same-domain/different-difficulty setup. Does it hold when tasks are genuinely heterogeneous (code vs math vs dialogue), or when more specialists introduce real task-vector interference?
3. **The ceiling may itself be low.** If joint RL's absolute AppWorld performance is limited, "merging rivals joint RL" could just mean "both are weak." A stronger multi-task baseline is needed.
4. **Is near-orthogonality just a high-dimensional artifact?** In very high-dimensional parameter space, random directions are already near-orthogonal. Even with a random-init floor, reviewers will press on whether the 0.06–0.10 lift over that floor is statistically and practically significant.
5. **Sensitivity to merge coefficients.** TIES/RAM+ involve scaling/pruning hyperparameters; how robust is the "collapse to uniform averaging" claim to these choices?

---

## Q4. 博士生应该精读还是跳过？/ Should a PhD student read or skip?

**中文：**

**建议精读方法与分析部分，尤其适合研究模型合并、多任务学习、参数空间几何与 RL 后训练的博士生；实验规模较小，可略读其绝对指标。** 这篇论文最大的价值不在"刷点"，而在于提供了一个**可迁移的分析工具箱**：把"为什么合并有效/为什么方法无差别"这类模糊直觉，转化为可测量的几何量（任务向量的余弦相似度、支撑集重叠、方向-支撑解耦），并用 random-init floor / same-run ceiling 做标定。对于同方向博士生，应精读：任务向量的定义与计算、near-orthogonality 的度量与标定方法、以及"方向-支撑解耦 ⇒ 合并坍缩为均匀平均"的推理链条。可以跳过：AppWorld 具体任务设置的工程细节。**下节的 PyTorch 复现正是把这套几何分析工具最小化实现出来。**

**English:**

**Read the method and analysis closely — especially if you work on model merging, multi-task learning, parameter-space geometry, or RL post-training — but skim the (small-scale) absolute numbers.** The paper's real value is a *transferable analysis toolkit*: it turns vague intuitions ("why does merging work / why are methods indistinguishable?") into measurable geometric quantities (task-vector cosine, support overlap, direction–support decoupling), calibrated against a random-init floor and a same-run ceiling. Focus on the definition and computation of task vectors, the near-orthogonality measurement and its calibration, and the reasoning chain "direction–support decoupling ⇒ merging collapses to uniform averaging." You can skip the AppWorld engineering details. **The PyTorch reproduction below implements exactly this geometry toolkit in minimal form.**
