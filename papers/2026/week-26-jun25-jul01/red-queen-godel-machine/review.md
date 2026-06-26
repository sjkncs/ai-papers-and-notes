# The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators

**Authors / 作者：** Alex Iacob, Andrej Jovanović, William F. Shen, Daniel Burkhardt, Meghdad Kurmanji, Nurbek Tastan, Lorenzo Sani, Niccolò Alberto Elia Venanzi, Ambroise Odonnat, Zeyu Cao, Bill Marino, Xinchi Qiu, Nicholas D. Lane  
**arXiv / 链接：** [https://arxiv.org/abs/2606.26294](https://arxiv.org/abs/2606.26294)  
**Published / 发表时间：** 2026-06-24 (arXiv announcement ~2026-06-26)  
**Venue / 会议：** Pre-print (work in progress)

---

## 摘要 / Abstract（原文 verbatim）

> Self-improving agents are state-of-the-art (SOTA) on agentic coding benchmarks and have recently been extended to general domains. However, their search methods generally assume a stationary evaluation criterion: a fixed verifier, benchmark, or labeled dataset that remains valid as the agent improves. This ignores a central feature of evolution: species adapt as their environments change with them. We aim to bring the same principle to recursive self-improvement, making evaluation part of the improvement loop and opening search to evolving evaluators, adversarial objectives, and dynamic utilities that may surpass static benchmarks. We introduce the Red Queen Godel Machine (RQGM), an evolutionary framework for recursive self-improvement under non-stationary utilities. The RQGM makes this possible through controlled utility evolution: search is organized into epochs with a fixed within-epoch evaluation criterion, while the utility can be updated at epoch boundaries, so self-improvement guarantees hold per epoch as the objective evolves across them. We begin by showing that even on verifiable coding tasks, the RQGM improves test pass rate over the prior SOTA by adding a complementary agent-as-a-judge code-review signal. This signal is cheaper and the RQGM uses 1.35x-1.72x fewer tokens. We then turn to scientific paper writing and reviewing, and Olympiad-level proof writing and grading, where the RQGM improves performance over prior self-improving agents: co-evolved writers reach 1.78x-1.86x higher acceptance rates under a diverse agent-as-a-judge panel, while co-evolved graders reach 9% higher ground-truth accuracy. In paper reviewing, the strongest baseline reviewer over-accepts AI-generated papers at up to 1.91x the human rate. The RQGM corrects this by introducing an adversarial objective that discovers reviewers equally stringent on AI and human work.

---

## Q1：它真正想解决的问题是什么？ / What problem does it really want to solve?

**中文：** 当前最前沿的“自我改进 Agent”（self-improving agents）——无论是代码生成 Agent 还是数学证明 Agent——几乎都依赖一个**静态评估器**：固定的 verifier、固定的 benchmark、固定的人工标注数据集。这个假设忽略了进化论的核心洞察：环境本身会随着物种的演化而共同演化。RQGM 要打破的正是这种“评估标准一成不变”的隐含假设，研究如何在**效用函数（utility）本身也随时间变化**的情况下，仍然实现递归自我改进。

具体而言，它试图回答：当 Agent 变得越来越强时，原有的测试集会失效、人类标注会饱和、静态 verifier 会被 exploit，我们该如何让评估机制本身也参与演化，从而避免自我改进陷入局部最优或奖励黑客（reward hacking）？

**English:** State-of-the-art self-improving agents—whether for code generation or theorem proving—almost universally rely on a **stationary evaluator**: a fixed verifier, benchmark, or labeled dataset. This ignores a central insight from evolution: environments co-evolve with the species that inhabit them. RQGM aims to break this implicit assumption of a fixed evaluation criterion and asks how recursive self-improvement can continue when the **utility function itself changes over time**.

Specifically, it asks: as agents become stronger, test sets saturate, human labels become insufficient, and static verifiers are exploited. How can the evaluation mechanism itself evolve so that self-improvement does not collapse into local optima or reward hacking?

---

## Q2：它声称的贡献是什么？ / What does it claim to contribute?

**中文：**

1. **概念框架**：提出“红皇后哥德尔机”（Red Queen Gödel Machine, RQGM），将递归自我改进从“固定评估器下的搜索”扩展为“评估器与 Agent 共同演化”的框架。
2. **受控效用演化（controlled utility evolution）**：把训练划分为若干 epoch，每个 epoch 内部评估标准固定（保证可证明性），但在 epoch 边界允许效用函数更新，从而在非平稳环境中维持自我改进保证。
3. **三任务实证**：
   - **代码生成**：加入“Agent 作为代码评审者”信号，用更少 token（1.35–1.72× 减少）取得比先前 SOTA 更高的测试通过率。
   - **学术论文写作与审稿**：共同演化的写作者在多样化 Agent 评审团下接受率提升 1.78–1.86×；共同演化的评分员在 ground-truth 准确率上提升 9%。
   - **审稿偏见修正**：引入对抗目标，发现对 AI 生成论文和人类论文同等严格的审稿人，修正 baseline 审稿员对 AI 论文的过度接受（高达 1.91×）。

**English:**

1. **Conceptual framework**: Introduces the Red Queen Gödel Machine (RQGM), extending recursive self-improvement from search under a fixed evaluator to a setting where evaluators and agents co-evolve.
2. **Controlled utility evolution**: Organizes training into epochs. Within each epoch the evaluation criterion is fixed (preserving guarantees), but the utility can be updated at epoch boundaries, maintaining self-improvement guarantees in non-stationary environments.
3. **Empirical validation on three tasks**:
   - **Code generation**: Adds an agent-as-code-reviewer signal, achieving higher test-pass rates than prior SOTA with fewer tokens (1.35–1.72× reduction).
   - **Scientific writing and reviewing**: Co-evolved writers improve acceptance rates by 1.78–1.86× under a diverse agent reviewer panel; co-evolved graders improve ground-truth accuracy by 9%.
   - **Reviewer bias correction**: Introduces an adversarial objective that discovers reviewers equally stringent on AI-generated and human-written papers, correcting baseline over-acceptance of AI papers by up to 1.91×.

---

## Q3：最可能被 reviewer 攻击的地方在哪里？ / Where are reviewers most likely to attack?

**中文：**

1. **循环论证与目标漂移（goal drift）**：让评估器与 Agent 共同演化，本质上是用一个变化的尺子量自己。Reviewer 会质疑：如果效用函数每个 epoch 都在变，那么如何定义“改进”？是否存在隐性的目标漂移，使得系统只是在追逐不断变化的评估器，而非真正提升能力？
2. **经验证据的因果性不足**：论文展示 RQGM 在多个任务上优于 baseline，但提升可能来自“更多计算 / 更多模型调用 / 更复杂的 prompt engineering”，而非“共同演化”本身。需要更严格的消融实验来分离“Agent-as-judge 信号”与“评估器共同演化”各自的贡献。
3. **可复现性与闭源细节**：涉及多个 Agent 角色、动态效用更新、对抗目标设计，实现细节复杂。如果代码、prompt 或效用更新规则未完全公开，reviewer 会质疑结果的可复现性。
4. **安全与对齐风险**：共同演化的评估器可能收敛到人类无法理解的评估标准，或被策略性 exploit。论文虽然提出“受控演化”，但对如何避免失控（runaway）讨论不足。
5. **benchmark 的代表性**：代码、学术论文、数学证明三个任务都偏“可验证”领域，难以推广到开放域（如对话、创意写作、伦理判断），而这些领域恰恰是评估器演化最危险也最有价值的地方。

**English:**

1. **Circularity and goal drift**: Letting the evaluator co-evolve with the agent means measuring progress with a moving ruler. Reviewers will ask: if the utility function changes every epoch, how is "improvement" defined? Is there hidden goal drift, where the system merely chases an evolving evaluator rather than genuinely improving?
2. **Weak causal evidence for co-evolution**: RQGM outperforms baselines on several tasks, but the gains may come from more compute, more model calls, or more sophisticated prompting rather than from co-evolution per se. Stricter ablations are needed to isolate the contribution of evaluator co-evolution from the agent-as-judge signal.
3. **Reproducibility and closed details**: With multiple agent roles, dynamic utility updates, and adversarial objectives, implementation details matter. If code, prompts, or utility-update rules are not fully disclosed, reviewers will question reproducibility.
4. **Safety and alignment risks**: Co-evolved evaluators may converge to standards humans no longer understand, or be strategically exploited. The paper proposes "controlled evolution" but does not deeply address how to prevent runaway dynamics.
5. **Limited domain generality**: The three tasks—code, academic papers, and proofs—are all highly verifiable. It remains unclear whether RQGM generalizes to open-ended domains such as dialogue, creative writing, or ethical judgment, where evaluator evolution is both most valuable and most dangerous.

---

## Q4：同方向博士生应精读哪些、跳过哪些？ / What should PhD students in this direction read carefully vs. skip?

**中文：**

**必读（精读）：**
- **本文引言与 Related Work**：它把自我改进、演化计算、AI 安全、评估器学习四个方向编织在一起，是快速进入该交叉领域的优质入口。
- **“受控效用演化”的形式化或算法描述**（若论文正文有详细描述）：这是 RQGM 的核心创新，理解 epoch 边界如何更新 utility 对设计自己的系统至关重要。
- **实验中的消融与失败案例**：尤其关注“RQGM 何时不如固定评估器？”这能帮你判断该方法的真实适用范围。

**延伸阅读（按兴趣）：**
- **Gödel Machine 原始论文**（Schmidhuber, 2005）：理解递归自我改进的理论源头。
- **OpenAI / DeepMind 关于 verifier / reward model 演化的工作**：如 *Let's Verify Step by Step*、*Self-Improving LLMs via Direct Preference Optimization* 等，建立评估器学习的背景知识。
- **AI 安全文献中关于 goal drift 与 specification gaming 的讨论**：如 *Risks from Learned Optimization*、*Scalable Agent Alignment via Reward Modeling*。

**可以跳过：**
- 如果只做**纯代码生成或数学证明**，而不关心评估器演化，那么 RQGM 的“共同演化”部分可能过于抽象；直接看它的 Agent-as-judge 实现即可。
- 如果关心**可复现工程系统**，而论文未开源完整代码，则具体数字可以暂时不记，重点学习其设计思想。

**English:**

**Must-read (close reading):**
- **The introduction and related work**: It weaves together self-improvement, evolutionary computation, AI safety, and learned evaluators, making it an excellent entry point to this interdisciplinary area.
- **The formalization/algorithm of controlled utility evolution** (if detailed in the main text): This is the core innovation; understanding how utilities are updated at epoch boundaries is essential for designing your own systems.
- **Ablations and failure cases in the experiments**: Pay special attention to "When does RQGM underperform a fixed evaluator?" to judge the true applicability of the method.

**Further reading (by interest):**
- **The original Gödel Machine paper** (Schmidhuber, 2005): To understand the theoretical roots of recursive self-improvement.
- **OpenAI / DeepMind work on verifier/reward-model evolution**: e.g., *Let's Verify Step by Step*, *Self-Improving LLMs via Direct Preference Optimization*.
- **AI safety literature on goal drift and specification gaming**: e.g., *Risks from Learned Optimization*, *Scalable Agent Alignment via Reward Modeling*.

**Can skip:**
- If you work purely on **code generation or theorem proving** and are not interested in evaluator evolution, the co-evolution part of RQGM may be too abstract; focus on its agent-as-judge implementation instead.
- If you care about **reproducible engineering systems** and the paper does not release full code, you need not memorize the exact numbers; focus on the design ideas.
