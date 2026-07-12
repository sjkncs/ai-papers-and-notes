# VLM-Planner: Hierarchical Vision-Language Planning for Long-Horizon Robotics

> **arXiv:** [2605.26402](https://arxiv.org/abs/2605.26402)  
> **领域 / Area:** Robotics / Vision-Language-Action  
> **一句话概括 / One-liner:** 用视觉-语言模型将高层指令分层分解为子目标，并通过视觉反馈闭环完成长程机器人任务。

---

## 关键摘录 / Key Excerpts

> "VLM-Planner treats long-horizon manipulation as a two-level planning problem: a high-level vision-language planner proposes sub-goal configurations, and a low-level visuomotor policy executes them while a visual verifier decides when to replan."
>
> "VLM-Planner 将长程操作视为两层规划问题：高层视觉-语言规划器提出子目标配置，低层视觉运动策略执行，视觉验证器决定何时重新规划。"

> "By grounding sub-goals in visual observations rather than language alone, the planner can recover from partial execution failures without human intervention."
>
> "通过将子目标扎根于视觉观测而非仅依赖语言，规划器能够在部分执行失败时无需人工干预即可恢复。"

---

## Q1: 核心问题 / Core Problem

**中文：** 端到端视觉-语言-动作（VLA）模型虽然能直接将自然语言指令映射为低级动作，但在长程、多步骤任务中容易累积错误，且难以利用高层任务结构。VLM-Planner 探索：能否将经典任务规划与现代视觉-语言模型结合，使机器人既能理解高层目标，又能通过视觉反馈闭环修正执行过程？

**English:** End-to-end vision-language-action (VLA) models can map natural-language instructions directly to low-level actions, but they easily accumulate errors in long-horizon, multi-step tasks and struggle to exploit high-level task structure. VLM-Planner explores whether classical task planning can be combined with modern vision-language models so that robots understand high-level goals and correct execution through visual feedback loops.

---

## Q2: 自称贡献 / Claimed Contributions

1. **分层规划框架 / Hierarchical planning framework**  
   提出高层 VLM 规划器 + 低层 VLA 执行器 + 视觉验证器的三段式架构，将长程任务分解为可复用的子目标。  
   Proposes a three-stage architecture of high-level VLM planner, low-level VLA executor, and visual verifier, decomposing long-horizon tasks into reusable sub-goals.

2. **视觉 grounding 的子目标表示 / Visually grounded sub-goal representation**  
   子目标以视觉状态描述（visual state description）表示，而非抽象符号，使规划器能直接利用环境观测。  
   Sub-goals are represented as visual state descriptions rather than abstract symbols, letting the planner directly use environmental observations.

3. **闭环 replanning 机制 / Closed-loop replanning mechanism**  
   视觉验证器检测子目标是否达成，失败时触发高层规划器重新生成后续计划。  
   A visual verifier detects whether sub-goals are achieved and triggers the high-level planner to regenerate subsequent plans upon failure.

4. **跨模拟器与真实机器人实验 / Cross-simulator and real-robot experiments**  
   在 Calvin、SPOC 仿真环境和真实厨房场景中进行评估，平均任务完成率提升 18–24%。  
   Evaluates on Calvin, SPOC simulators, and real kitchen scenes, improving average task completion by 18–24%.

---

## Q3: 审稿人攻击点 / Reviewer Attack Points

- **规划延迟高 / High planning latency**  
  每次 replan 都需要调用大型 VLM，导致执行间隔可达数秒，难以应用于高速操作场景。  
  Each replan requires calling a large VLM, causing execution intervals of several seconds, limiting use in high-speed manipulation.

- **子目标粒度依赖人工设计 / Sub-goal granularity relies on human design**  
  论文未明确说明子目标分解的粒度如何选择，不同任务可能需要领域专家手动调整。  
  The paper does not clarify how sub-goal granularity is chosen; different tasks may require manual tuning by domain experts.

- **视觉验证器的错误模式 / Failure modes of the visual verifier**  
  当子目标部分达成但视觉状态与预期有细微差异时，验证器容易误判，导致不必要的 replan 或遗漏失败。  
  The visual verifier may misjudge when a sub-goal is partially achieved but the visual state differs subtly from expectations, causing unnecessary replanning or missed failures.

- **泛化到未见环境的证据有限 / Limited evidence for unseen environments**  
  实验主要在与训练数据分布相似的厨房和操作任务上进行，未见场景（如户外、工业环境）的泛化能力未经验证。  
  Experiments are mainly on kitchen and manipulation tasks similar to training data; generalization to unseen scenarios (outdoor, industrial) is not validated.

---

## Q4: PhD 阅读指南 / PhD Reading Guide

**阅读顺序 / Reading order:**
1. 先读摘要与图 1 的系统架构，理解三段式分工。  
   Read the abstract and Figure 1 system architecture to understand the three-stage division of labor.
2. 精读第 3 节关于视觉子目标表示与验证器的设计。  
   Read Section 3 carefully on visual sub-goal representation and verifier design.
3. 查看附录中的 prompt 模板，思考 VLM prompt 工程对结果的影响。  
   Check the prompt templates in the appendix and consider how VLM prompt engineering affects results.
4. 分析实验表 3 中失败案例的分类，识别系统瓶颈。  
   Analyze the failure-case taxonomy in Table 3 to identify system bottlenecks.

**复现建议 / Reproduction advice:**  
在 ALFRED 或 Calvin 仿真环境中实现一个最小版本：用现成 VLM（GPT-4V / LLaVA）生成子目标，用开源 VLA 策略执行，用简单的视觉匹配判断是否需要 replan。  
Implement a minimal version in ALFRED or Calvin: use an off-the-shelf VLM (GPT-4V / LLaVA) to generate sub-goals, an open VLA policy to execute, and simple visual matching to decide whether to replan.

**延伸问题 / Follow-up questions:**  
- 能否将子目标生成蒸馏为一个小型视觉-语言模型，以降低延迟？  
  Can sub-goal generation be distilled into a small vision-language model to reduce latency?
- 如果高层规划器能访问环境的符号模型（如 PDDL），是否能进一步提升可解释性与效率？  
  If the high-level planner had access to a symbolic model of the environment (e.g., PDDL), could interpretability and efficiency be further improved?
