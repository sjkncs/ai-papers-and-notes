# CAT: Confidence-Adaptive Thinking for Efficient Reasoning of Large Reasoning Models

**论文链接 / Paper:** https://arxiv.org/abs/2607.00862  
**类型 / Type:** arXiv preprint, July 2026  
**领域 / Area:** Large Reasoning Models (LRMs), Efficient Inference, Preference Optimization, Test-Time Compute  

---

## Q1. 这篇论文试图解决什么问题？/ What problem does this paper address?

**中文：**

大推理模型（LRM）依靠长思维链（CoT）在复杂任务上取得了显著成功，但它们常常在简单查询上**过度思考（overthinking）**，造成巨大的 Token 开销和推理效率下降。现有压缩方法大多采用统一的长度裁剪或粗粒度难度估计，往往在难题上导致性能退化。论文要解决的是：如何让 LRM **根据问题难度自适应地调节推理长度**——在自信的步骤上压缩，在不确定的步骤上深入思考，从而兼顾准确率与延迟。

> Large Reasoning Models (LRMs) have achieved remarkable success on complex tasks by leveraging long chain-of-thought (CoT) trajectories, yet they frequently exhibit overthinking on simple queries, resulting in significant token overhead and reduced inference efficiency. However, existing compression methods predominantly apply uniform length reduction or rely on coarse-grained difficulty estimation, often leading to performance degradation on difficult problems.

**English:**

Large Reasoning Models (LRMs) have achieved remarkable success on complex tasks by leveraging long chain-of-thought (CoT) trajectories, but they often **overthink on simple queries**, causing significant token overhead and reduced inference efficiency. Existing compression methods mostly apply uniform length reduction or coarse-grained difficulty estimation, frequently degrading performance on difficult problems. The paper asks how to let LRMs **adaptively adjust reasoning length based on problem difficulty** — compressing confident responses while deliberating on uncertain ones, so as to balance accuracy and latency.

---

## Q2. 核心贡献是什么？/ What are the core contributions?

**中文：**

1. **自信度自适应思考框架（CAT）**：将模型内在的**自我确信信号（self-certainty）**作为自信度，引入偏好优化过程，使模型自主学习何时该长思考、何时可短回答。
2. **基于问题难度的动态推理长度调制**：不同于统一裁剪，CAT 在难题上保留甚至延长推理，在简单题上压缩，避免“一刀切”带来的性能损失。
3. **跨模型与跨基准的泛化验证**：在多个不同基座模型和多个推理基准上，CAT  consistently 优于当前最优基线。
4. **工业场景的实用潜力**：论文强调该方法为实际工业应用中平衡准确率与延迟提供了稳健方案。

> To address this limitation, we propose Confidence-Adaptive Thinking (CAT), a framework that incorporates the model's intrinsic self-certainty signals as confidence into the preference optimization process, which autonomously modulates reasoning lengths based on problem difficulty. Experimental results show that CAT consistently outperforms state-of-the-art baselines on reasoning accuracy across multiple benchmarks on different base models.

**English:**

1. **Confidence-Adaptive Thinking (CAT) framework**: incorporates the model's **intrinsic self-certainty signals** as confidence into the preference optimization process, enabling the model to learn autonomously when to think longer and when to answer briefly.
2. **Dynamic reasoning-length modulation based on problem difficulty**: unlike uniform truncation, CAT preserves or even extends reasoning on hard problems while compressing on easy ones, avoiding one-size-fits-all performance loss.
3. **Cross-model and cross-benchmark generalization**: consistently outperforms state-of-the-art baselines across multiple base models and multiple reasoning benchmarks.
4. **Industrial practicality**: the paper emphasizes that the method offers a robust solution for balancing accuracy and latency in real-world industrial scenarios.

---

## Q3. 审稿人会从哪些角度攻击它？/ What would a reviewer attack?

**中文：**

1. **自信度校准风险**：语言模型的 softmax 概率或内部状态未必是良好校准的自信度估计。如果自信度本身不可靠，偏好优化可能学到错误的停止/继续策略，甚至“自信地犯错”。
2. **偏好数据的构造成本**：要训练模型区分“长而正确”“短而正确”“长而错误”“短而错误”的回复，需要大量成对偏好数据，论文对数据构造过程披露不足。
3. **与 PRM/ORM 的边界**：如果自信度信号来自额外验证器，则与过程/结果奖励模型方法高度重叠；如果仅来自模型自身，则其可信度有限。
4. **任务难度估计的粒度**：论文强调“基于问题难度”，但难度是在序列级别、步骤级别还是 Token 级别估计？粒度选择会直接影响方法效果。
5. **理论保证缺失**：目前主要是经验性结果，缺少关于收敛性、最优停止策略或最坏情况下性能保证的分析。

**English:**

1. **Confidence calibration risk**: softmax probabilities or internal states of language models are not necessarily well-calibrated confidence estimates. If confidence itself is unreliable, preference optimization may learn wrong stop/continue strategies, or even "be confidently wrong."
2. **Cost of preference-data construction**: training the model to distinguish long-correct, short-correct, long-wrong, and short-wrong responses requires substantial pairwise preference data; the paper does not disclose this process in enough detail.
3. **Boundary with PRM/ORM methods**: if the confidence signal comes from an external verifier, the method overlaps heavily with process/outcome reward models; if it comes only from the model itself, its reliability is limited.
4. **Granularity of difficulty estimation**: the paper emphasizes "based on problem difficulty," but is difficulty estimated at the sequence, step, or token level? The granularity choice directly affects effectiveness.
5. **Lack of theoretical guarantees**: current results are largely empirical; there is little analysis of convergence, optimal stopping policy, or worst-case performance guarantees.

---

## Q4. 博士生应该精读还是跳过？/ Should a PhD student read or skip?

**中文：**

**建议精读，尤其适合关注大推理模型推理效率、测试时计算优化与偏好优化的博士生。** CAT 把“自信度”从一种可解释性副产品提升为控制推理长度的核心信号，并尝试用偏好优化让模型自主学习这种控制。这是 LRM 从“堆更多计算”走向“更聪明地分配计算”的代表性工作。阅读时重点跟踪：自信度具体如何计算、偏好对如何构造、训练过程是否稳定、以及短回答是否真的在简单题上没有性能损失。

**English:**

**Worth a close read, especially for PhD students working on large-reasoning-model efficiency, test-time compute optimization, and preference optimization.** CAT elevates "confidence" from an interpretability byproduct to a core signal for controlling reasoning length, and attempts to use preference optimization to let the model learn this control autonomously. It is a representative work in the shift of LRMs from "throwing more compute" to "allocating compute more intelligently." When reading, focus on how confidence is computed, how preference pairs are constructed, whether training is stable, and whether short responses truly incur no performance loss on easy problems.
