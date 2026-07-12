# Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade

**论文链接 / Paper:** https://arxiv.org/abs/2607.06503  
**类型 / Type:** arXiv preprint, July 2026  
**领域 / Area:** LLM Agents, Efficient Inference, Representation Probing, Cascade Control  

---

## Q1. 这篇论文试图解决什么问题？/ What problem does this paper address?

**中文：**

大语言模型智能体在解决多步任务时，常常会陷入**注定要失败的轨迹**，但它们在失败变得可观测之前仍会持续消耗大量推理算力。作者发现，失败其实可以从智能体的内部表示中很早预测出来：仅读取可观测行为的评分器在第一步几乎只能随机猜测，而基于隐藏层激活的轻量级探针就能预测最终 episode 失败。论文要解决的问题是：如何利用这种早期可预测性，在保留绝大多数成功 episode 的前提下，尽早中止注定失败的轨迹，从而节省推理成本。

> Large language model (LLM) agents solving multi-step tasks frequently commit to trajectories that are doomed to fail, yet continue to consume substantial inference compute before the failure becomes observable. We show that failure is predictable early from the agent's internal representations: lightweight per-round probes on hidden activations anticipate eventual episode failure as early as the first interaction round, where scorers reading only the agent's observable behavior are barely better than chance.

**English:**

LLM agents solving multi-step tasks frequently commit to trajectories that are doomed to fail, yet continue to consume substantial inference compute before the failure becomes observable. The authors show that failure is predictable early from the agent's internal representations: lightweight per-round probes on hidden activations can anticipate eventual episode failure as early as the first interaction round, while scorers reading only the agent's observable behavior are barely better than chance. The paper asks how to exploit this early predictability to abort doomed trajectories as early as possible while preserving most successful episodes, thereby saving inference cost.

---

## Q2. 核心贡献是什么？/ What are the core contributions?

**中文：**

1. **隐藏状态可早期预测失败**：提出并验证 hidden-state probe 在首轮即可预测 episode 最终失败，而行为-only probe 效果接近随机。
2. **Recall-Controlled Probe Cascade**：每轮设置一个无分布假设的校准门控，按用户指定的全局 recall 目标联合搜索各轮 recall 预算，保证最终成功的 episode 能以指定比例存活过所有门。
3. **显著推理开销节省**：在 TextCraft 上，90% recall 目标下可节省 37%-47% 推理算力，比最佳单门策略提升 1.6-1.7 倍。
4. **样本复杂度刻画**：给出如何根据数据量判断能“安全承诺”多高的 recall 目标，为部署者提供可操作的指导。

> We turn this signal into a practical abort cascade: one distribution-free calibrated gate per round, with per-round recall budgets jointly searched so that eventually-successful episodes survive all gates at a user-specified global rate... Across two agent models on TextCraft, the cascade meets every recall target from 90% to 97% and, at the 90% target, saves 47.1% +/- 10.3% (Qwen-2.5-7B) and 37.2% +/- 8.8% (Llama-3.2-3B) of inference compute, 1.6--1.7x the best single-gate policy.

**English:**

1. **Hidden states predict failure early**: propose and validate that hidden-state probes can predict episode failure from the first round, while behavior-only probes are near chance.
2. **Recall-controlled probe cascade**: uses one distribution-free calibrated gate per round, with per-round recall budgets jointly searched so that eventually-successful episodes survive all gates at a user-specified global rate.
3. **Significant inference savings**: on TextCraft, at a 90% recall target, saves 47.1% ± 10.3% (Qwen-2.5-7B) and 37.2% ± 8.8% (Llama-3.2-3B) of inference compute, 1.6–1.7× better than the best single-gate policy.
4. **Sample-complexity characterization**: tells practitioners which recall promises their data can and provably cannot back, providing actionable deployment guidance.

---

## Q3. 审稿人会从哪些角度攻击它？/ What would a reviewer attack?

**中文：**

1. **任务覆盖有限**：实验主要在 TextCraft 上进行，需要更多任务（如网页导航、代码生成、多智能体协作）验证方法普适性。
2. **探针与模型耦合**：不同架构或规模的模型需要重新训练探针，论文对跨模型迁移能力讨论不足。
3. **Recall 与最终性能权衡**：虽然 recall 目标保证成功 episode 不被误杀，但 cascade 本身是否会影响 agent 的搜索/探索行为？如果 agent 知道可能被中止，是否会改变策略？
4. **假阳性的下游影响**：被中止的 episode 是否完全浪费？能否利用部分轨迹进行在线学习或重启，而不是简单丢弃？
5. **校准方法的稳健性**：无分布假设的校准门在分布外任务或模型更新后是否仍可靠？需要更长期的验证。

**English:**

1. **Limited task coverage**: experiments are mainly on TextCraft; more tasks (web navigation, code generation, multi-agent collaboration) are needed to validate generality.
2. **Probe-model coupling**: different architectures or model sizes require retraining probes; the paper does not discuss cross-model transfer enough.
3. **Recall vs. final performance trade-off**: while the recall target ensures successful episodes are not killed, does the cascade itself affect the agent's search/exploration behavior? Might the agent change strategy if it knows it can be aborted?
4. **Downstream impact of false positives**: are aborted episodes completely wasted? Could partial trajectories be used for online learning or restart rather than simple discard?
5. **Robustness of calibration**: are distribution-free calibrated gates still reliable under distribution-shift or model updates? Longer-term validation is needed.

---

## Q4. 博士生应该精读还是跳过？/ Should a PhD student read or skip?

**中文：**

**建议精读，尤其适合从事 LLM Agent、测试时计算优化、可解释性与表示学习的博士生。** 这篇论文把“内部表示包含任务失败信号”这一直觉变成了可部署的级联控制机制，并提供 recall 保证与样本复杂度分析，是非常扎实的算法工作。如果你关心如何让 Agent 少做无效计算、如何从隐藏状态中提取元认知信号、或者如何设计带统计保证的早停策略，Doomed from the Start 会给出很多启发。阅读时重点跟踪：探针架构与训练数据构造、cascade 门控的校准方法、以及在不同 recall 目标下的算力节省曲线。

**English:**

**Worth a close read, especially for PhD students in LLM agents, test-time compute optimization, interpretability, and representation learning.** This paper turns the intuition that "internal representations contain task-failure signals" into a deployable cascade control mechanism with recall guarantees and sample-complexity analysis — a solid algorithmic work. If you care about reducing wasted computation in agents, extracting metacognitive signals from hidden states, or designing early-stopping strategies with statistical guarantees, Doomed from the Start offers many insights. When reading, focus on the probe architecture and training-data construction, the calibration method for cascade gates, and the compute-savings curve under different recall targets.
