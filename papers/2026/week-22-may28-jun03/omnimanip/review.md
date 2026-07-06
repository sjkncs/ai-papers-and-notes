# OmniManip: A Generalist Manipulation Policy from Internet-Scale Video

> **arXiv:** [2605.28091](https://arxiv.org/abs/2605.28091)  
> **领域 / Area:** Robotics / Vision-Language-Action  
> **一句话概括 / One-liner:** 从数百万段互联网教学视频中学习通用机器人操作策略，实现跨形态零样本迁移。

---

## 关键摘录 / Key Excerpts

> "We pre-train a visuomotor transformer on 2.4 million instructional videos, then distill the resulting policy into cross-embodiment action primitives that transfer zero-shot to 7 robot platforms."
>
> "我们在 240 万段教学视频上预训练视觉运动 Transformer，然后将所得策略蒸馏为可跨形态迁移的动作原语，在 7 种机器人平台上实现零样本迁移。"

> "The key insight is that internet videos contain rich, task-agnostic contact and motion priors that can be decoded into embodiment-invariant action representations."
>
> "核心洞见在于，互联网视频蕴含丰富的任务无关接触与运动先验，可被解码为与机器人形态无关的动作表示。"

---

## Q1: 核心问题 / Core Problem

**中文：** 真实机器人数据收集成本高昂，限制了机器人学习策略的泛化能力。OmniManip 研究：能否仅利用互联网上的教学视频，学习通用的操作先验，并将其迁移到未见过的机器人形态与任务上？

**English:** Real robot data collection is expensive, limiting the generalization of robot learning policies. OmniManip asks: can general manipulation priors be learned from internet instructional videos alone and transferred to unseen robot morphologies and tasks?

---

## Q2: 自称贡献 / Claimed Contributions

1. **互联网规模视频预训练 / Internet-scale video pretraining**  
   在 240 万段教学视频上预训练视觉运动 Transformer，学习任务无关的接触与运动先验。  
   Pretrains a visuomotor Transformer on 2.4M instructional videos to learn task-agnostic contact and motion priors.

2. **形态无关动作表示 / Embodiment-invariant action representation**  
   将视频先验解码为与具体机器人形态无关的动作原语（action primitives），支持跨平台迁移。  
   Decodes video priors into embodiment-invariant action primitives that support cross-platform transfer.

3. **零样本跨形态迁移 / Zero-shot cross-embodiment transfer**  
   在 7 种不同机器人平台上进行零样本评估，涵盖单臂、双臂、人形手等多种形态。  
   Evaluates zero-shot on 7 different robot platforms, including single-arm, dual-arm, and humanoid hands.

4. **少量真实数据微调即可超过专有方法 / Few-shot real-data finetuning surpasses specialized methods**  
   在特定机器人上仅用 50 条真实轨迹微调即可超越从头训练的专有策略。  
   With only 50 real trajectories of robot-specific finetuning, surpasses specialized policies trained from scratch.

---

## Q3: 审稿人攻击点 / Reviewer Attack Points

- **视频与真实机器人之间的域差距 / Domain gap between videos and real robots**  
  互联网视频的拍摄视角、物体外观与真实机器人平台差异巨大，论文需要更强的域自适应分析。  
  Internet videos differ greatly from real robot platforms in camera viewpoint and object appearance; stronger domain-adaptation analysis is needed.

- **动作原语的粒度与可组合性 / Granularity and composability of action primitives**  
  形态无关原语的定义与组合方式未完全明确，复杂长程任务可能需要更高级别的规划器。  
  The definition and composition of embodiment-invariant primitives are not fully clarified; complex long-horizon tasks may need a higher-level planner.

- **零样本性能仍有限 / Limited zero-shot performance**  
  虽然能完成简单任务，但在需要精细接触力的复杂操作上，零样本成功率仍低于专门训练的策略。  
  While simple tasks succeed, zero-shot success on fine contact-heavy operations remains below specially trained policies.

- **数据规模与质量的权衡 / Trade-off between data scale and quality**  
  240 万段视频的数据清洗过程未详细披露，低质量或错误标注视频可能影响策略学习。  
  The curation process for 2.4M videos is not detailed; low-quality or mislabeled videos may affect policy learning.

---

## Q4: PhD 阅读指南 / PhD Reading Guide

**阅读顺序 / Reading order:**
1. 先读摘要与图 1，理解视频预训练到动作原语的流程。  
   Read the abstract and Figure 1 to understand the pipeline from video pretraining to action primitives.
2. 精读第 3 节动作表示学习，尤其是形态无关编码器的设计。  
   Read Section 3 on action representation learning, especially the embodiment-invariant encoder design.
3. 查看附录中的跨平台实验细节，分析失败案例。  
   Check the appendix for cross-platform experiment details and analyze failure cases.
4. 思考视频预训练与仿真到现实迁移的关系。  
   Reflect on the relationship between video pretraining and sim-to-real transfer.

**复现建议 / Reproduction advice:**  
在 Something-Something v2 或 Epic-Kitchens 视频数据上预训练一个小型视觉运动模型，然后在仿真机器人环境（PyBullet、Isaac Gym）上评估零样本迁移能力。  
Pretrain a small visuomotor model on Something-Something v2 or Epic-Kitchens, then evaluate zero-shot transfer in a simulated robot environment (PyBullet, Isaac Gym).

**延伸问题 / Follow-up questions:**  
- 互联网视频预训练是否能取代仿真预训练，或两者应如何结合？  
  Can internet video pretraining replace simulation pretraining, or how should they be combined?
- 如何设计可解释的形态无关动作原语，使人类工程师能够理解和调试？  
  How can interpretable embodiment-invariant action primitives be designed so engineers can understand and debug them?
