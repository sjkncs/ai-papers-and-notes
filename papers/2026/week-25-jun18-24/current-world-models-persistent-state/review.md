# Current World Models Lack a Persistent State Core

- **arXiv:** [2606.20545](https://arxiv.org/abs/2606.20545)
- **Authors:** Jinpeng Lu, Dexu Zhu, Haoyuan Shi, Linghan Cai, Guo Tang, Yinda Chen, Jie Cao, Duyu Tang, Yi Zhang, Yong Dai, Xiaozhu Ju
- **Category:** Computer Vision (CV) / World Models / Embodied AI
- **Code Reproduction:** — (benchmark paper, no algorithm to reproduce)

---

## Q1: 它真正想解决的问题是什么？ / What problem does it really solve?

**中文：** 世界模型被越来越多地视为通往 AGI 的关键一步，但当前主流评测只关心“生成帧是否逼真、运动是否连贯、相机是否可控”等表面属性，而从未追问：**当相机移开后，世界是否仍在按照物理规律演化？** 本文指出，现有世界模型本质上是在做“跟踪镜头”式的生成——目标离开视野后再返回时，模型只是把它恢复到离开时的状态，而不是让该目标在不可见期间继续演化。论文真正想解决的问题是：**为世界模型建立一套诊断持久世界状态（persistent world state）的评测体系，并证明当前主流模型普遍缺乏稳定演化的内部状态核心。**

**English:** World models are increasingly seen as a crucial step toward AGI, but mainstream benchmarks focus only on surface-level properties such as frame fidelity, motion coherence, and camera controllability. They never ask: **does the world keep evolving according to physical laws when the camera looks away?** This paper argues that current world models essentially produce "tracking-shot" generation — when a target leaves the field of view and returns, the model restores it to the state at which it was abandoned rather than simulating its evolution while unobserved. The core problem addressed is: **how to build a diagnostic benchmark for persistent world states and demonstrate that mainstream models lack a stable, evolving internal state core.**

> "World models are increasingly regarded as a decisive step toward artificial general intelligence, yet modeling the physical world demands more than rendering convincing frames on demand: it requires an internal world state that keeps evolving over time."

---

## Q2: 它声称的贡献是什么？ / What does it claim to contribute?

**中文：**
1. **WRBench 诊断基准**：首个系统评估世界模型“持久状态核心”的 benchmark。它将相机运动视为对可观测性的干预，把评测拆解为一条人工校准的链条：相机是否执行了请求的互动？目标在视野内是否保持连续可识别？返回的目标状态是否与离开时触发的事件一致？
2. **大规模跨模型诊断**：在 23 个模型、9600 个视频、4 种控制范式（text-to-world、world-to-world、video-to-world、action-to-world）上统一测试，证明“状态不演化”是跨范式、跨模型家族、跨规模的顽固缺陷。
3. **新的设计目标倡导**：作者主张把“物理状态核的稳定性”和“视角干预下世界线的一致性”提升为世界模型设计的一等目标，而不是继续只优化像素质量。

**English:**
1. **WRBench diagnostic benchmark**: the first systematic benchmark that evaluates the "persistent state core" of world models. It treats camera motion as an intervention on observability and decomposes evaluation into a human-calibrated chain: does the camera execute the requested interaction? does the scene stay continuous and identifiable while in view? does the returning target remain consistent with the event set in motion?
2. **Large-scale cross-model diagnosis**: tests across 23 models, 9,600 videos, and four control paradigms (text-to-world, world-to-world, video-to-world, action-to-world), showing that the failure to evolve state is stubbornly consistent across paradigms, model families, and scales.
3. **New design objective advocacy**: the authors argue that the stability of the physical state kernel and the consistency of worldlines under viewpoint intervention should become first-class objectives of world-model design rather than continuing to optimize pixel quality alone.

> "We introduce WRBench, the first systematic diagnostic benchmark that treats camera motion as an intervention on observability and resolves evaluation into a human-calibrated chain..."

---

## Q3: 最可能被 reviewer 攻击的地方在哪里？ / Where is it most vulnerable to reviewer attacks?

**中文：**
1. **“没有持久状态”是否只是训练数据/提示词的 artifact？**  reviewer 可能质疑：如果模型在训练数据中见过大量“离开再返回”的视频，是否就能学会持久状态？作者需要更严格地证明这是架构/目标函数的结构性缺陷，而非数据分布问题。
2. **WRBench 的人工校准链是否过于复杂？** 三个问题的标注成本高，且涉及主观判断。 reviewer 会关心 inter-annotator agreement、自动评测指标与人工判断的一致性。
3. **对比对象是否公平？** 论文比较了 23 个模型，但它们的能力范围差异很大（有的只做视频生成，有的做动作条件世界模型）。把不同控制范式放在同一榜单上是否合理？
4. **“物理状态核”的因果定义不够形式化。** 作者大量借用物理术语，但并未给出可被严格检验的数学定义。对ML社区而言，持久状态、世界线、事件等概念需要更清晰的 operationalization。

**English:**
1. **Is the lack of persistent state merely a data/prompt artifact?** Reviewers may argue that if models were trained on more "leave-and-return" videos they would learn persistent states. The authors need to more rigorously show this is a structural flaw in architecture/objectives rather than a data-distribution issue.
2. **Is WRBench's human-calibrated chain too complex?** The three-question annotation is costly and involves subjective judgment. Reviewers will care about inter-annotator agreement and the correlation between automatic metrics and human judgments.
3. **Is the comparison fair?** The paper compares 23 models with very different capabilities (some only do video generation, others action-conditioned world models). Is it legitimate to rank them on the same leaderboard?
4. **The causal definition of "physical state kernel" is not formalized.** The paper borrows heavily from physics terminology but does not provide a mathematically crisp definition that can be strictly tested. For the ML community, concepts like persistent state, worldline, and event need clearer operationalization.

> "Current systems maintain the observed world as a tracking shot, resuming a returning target in the state at which it was abandoned rather than advancing the event while it went unseen."

> *Reviewer concern:* This framing is compelling but qualitative; the paper must show that the failure cannot be fixed simply by scaling dataset coverage or model size.

---

## Q4: 同方向博士生应精读哪些、跳过哪些？ / What should PhD students in this direction read carefully vs skip?

**中文：**
- **必读（精读）：**
  - **WRBench 的构造逻辑与三个评测问题**：这是本文核心方法学贡献，适合借鉴到其他需要“反事实/持久状态”评测的任务。
  - **结果部分关于 scale 的实验**：作者声称扩大规模不能解决该问题，这是与传统 scaling law 叙事对抗的强论断，必须看证据。
  - 相关背景：视频世界模型（Sora、Gen-3、VideoPoet）、物理仿真（Physion、VirtualHome）、以及因果推断中的“未观测混杂”概念。
- **可选读（了解即可）：**
  - 若你不是做世界模型或具身智能，只需读 abstract + figure 1 + 结论，把握“持久状态核心”这一概念即可。
  - 论文中 39 页的大量定性视频分析可快速浏览。
- **不建议投入：** 若你的研究方向是 2D 图像生成或纯 NLP，本文与你的直接技术路线较远，不必精读实现细节。

**English:**
- **Must-read (in depth):**
  - **WRBench construction and the three evaluation questions**: this is the core methodological contribution and is valuable for any task requiring counterfactual/persistent-state evaluation.
  - **Scale-related experiments in the results**: the strong claim that scaling does not fix the problem contradicts conventional scaling-law narratives and must be evaluated carefully.
  - Background: video world models (Sora, Gen-3, VideoPoet), physical simulation (Physion, VirtualHome), and the causal-inference notion of unobserved confounding.
- **Optional (skim):**
  - If you do not work on world models or embodied AI, read the abstract, figure 1, and conclusion to grasp the "persistent state core" concept.
  - The 39-page qualitative video analysis can be browsed quickly.
- **Not recommended:** If your research is 2D image generation or pure NLP, this paper is not directly on your technical path; do not spend time on implementation details.

---

## 关键原文摘录 / Key Excerpts

> "World models are increasingly regarded as a decisive step toward artificial general intelligence, yet modeling the physical world demands more than rendering convincing frames on demand: it requires an internal world state that keeps evolving over time."

> "This requirement is a blind spot of existing benchmarks, which reward surface properties such as fidelity, motion, and camera controllability while never asking whether a generated world keeps evolving once it is unobserved."

> "Across 9,600 videos from 23 models spanning four control paradigms, one finding proves stubborn: current systems maintain the observed world as a tracking shot, resuming a returning target in the state at which it was abandoned rather than advancing the event while it went unseen."

> "Because this failure recurs across control paradigms, model families, and increments of scale, robust world-state evolution does not follow from cleaner imagery, tighter control, richer geometric priors, or sheer parameter count."

> "We therefore argue that the stability of the physical state kernel and the consistency of worldlines under viewpoint intervention should become first-class objectives of world-model design, so that a world model captures how the world will unfold rather than how the next frame appears."
