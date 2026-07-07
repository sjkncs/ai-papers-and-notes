# Week 27: 2026-07-02 — 2026-07-08 / 第27周

## 本周学术动态 / Weekly Highlights

**中文：** 2026年第27周（7月初）的 arXiv 计算机论文呈现出三个鲜明主题：多模态大语言模型的“主动感知”范式、扩散/流匹配生成模型的效率革命，以及大推理模型（LRM）的测试时计算优化。PixelEyes 提出自上而下视觉感知系统，让 MLLM 根据推理意图动态聚焦图像区域；SNAP-FM 将亚奈奎斯特采样思想引入流模型，通过激活稀疏化显著降低推理开销；CAT 则以置信度自适应树搜索让 LRM 在数学与代码推理中“该快则快、该慢则慢”。此外，Seed2.0、AutoMem、DriftScope、MEPA、HARC、Procedural Memory Distillation 与 Graph-PRefLexOR 等工作在图像生成、智能体记忆、时序漂移检测、偏好优化、图推理等方向持续发力。

**English:** Week 27 of 2026 (early July) on arXiv is marked by three prominent themes: active perception for multimodal large language models, efficiency revolutions in diffusion / Flow Matching generative models, and test-time compute optimization for large reasoning models (LRMs). PixelEyes proposes a top-down visual perception system that lets MLLMs dynamically focus on image regions according to reasoning intent; SNAP-FM brings sub-Nyquist sampling into flow models, substantially reducing inference cost via activation sparsity; CAT uses confidence-adaptive tree search to let LRMs move fast through confident steps and slow down when uncertain on math and code reasoning. Additionally, Seed2.0, AutoMem, DriftScope, MEPA, HARC, Procedural Memory Distillation, and Graph-PRefLexOR continue to push forward in image generation, agent memory, temporal drift detection, preference optimization, and graph reasoning.

---

## 重点论文 / Featured Papers

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [PixelEyes: Top-Down Visual Perception for MLLMs](https://arxiv.org/abs/2607.00115) | Multimodal Reasoning | [review.md](pixeleyes/review.md) | [code/](pixeleyes/code/) |
| 2 | [SNAP-FM: Sub-Nyquist Activation Pruning for Flow Matching](https://arxiv.org/abs/2607.00095) | Generative Models / Efficiency | [review.md](snap-fm/review.md) | [code/](snap-fm/code/) |
| 3 | [CAT: Confidence-Adaptive Tree Search for LRMs](https://arxiv.org/abs/2607.00862) | Large Reasoning Models | [review.md](cat/review.md) | [code/cat_demo.py](cat/code/cat_demo.py) |

---

## 其他论文 / Other Papers

### 多模态与视觉 / Multimodal & Vision

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [Seed2.0 (2607.00248)](https://arxiv.org/abs/2607.00248) | 新一代图像/视频生成基础模型，强化视觉一致性与复杂提示遵循 |
| [DriftScope (2607.00183)](https://arxiv.org/abs/2607.00183) | 面向视觉-语言模型的时序漂移检测与持续学习框架 |

### 大模型推理与智能体 / LLM Reasoning & Agents

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [AutoMem (2607.01224)](https://arxiv.org/abs/2607.01224) | 自动化智能体记忆管理，支持长期上下文与工具调用 |
| [Procedural Memory Distillation (2607.01480)](https://arxiv.org/abs/2607.01480) | 将程序性知识蒸馏为模型可复用的推理技能 |
| [Graph-PRefLexOR (2607.00924)](https://arxiv.org/abs/2607.00924) | 基于图结构的偏好优化与可解释推理 |

### 模型效率与优化 / Efficiency & Optimization

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [MEPA (2607.00371)](https://arxiv.org/abs/2607.00371) | 模型效率与性能联合优化方法 |
| [HARC (2607.00572)](https://arxiv.org/abs/2607.00572) | 面向高效推理的混合精度或缓存加速方案 |

---

## 领域趋势 / Trends

1. **主动感知成为 MLLM 新焦点 / Active perception emerges in MLLMs** — 从被动一次性编码到动态、目标驱动的视觉注意。
2. **生成模型进入“稀疏计算”时代 / Generative models enter the sparse-compute era** — 激活稀疏、缓存、蒸馏与少步采样组合加速。
3. **推理模型强调“聪明地花钱” / Reasoning models emphasize smarter compute spending** — 测试时计算从均匀分配转向置信度自适应调度。
4. **智能体记忆与程序性知识 / Agent memory and procedural knowledge** — 长期记忆、漂移检测与技能蒸馏支撑自主系统。
5. **图结构融入偏好优化 / Graph structures enter preference optimization** — 利用关系结构提升可解释性与推理能力。

---

## 博士生阅读路线 / PhD Reading Route

### 入门路线（熟悉领域）/ Beginner Track

**中文：** 如果你刚开始关注多模态推理或生成模型，本周建议按以下顺序阅读：

1. 先读 **PixelEyes** 的摘要与引言，理解“主动感知”与“感知-推理解耦”的动机。
2. 阅读 **CAT** 的审稿分析与 PyTorch demo，掌握置信度自适应搜索的核心思想。
3. 浏览 **SNAP-FM** 的方法概述，了解流模型中激活稀疏化的基本概念。
4. 任选一篇 Other Papers（如 AutoMem 或 Graph-PRefLexOR）作为扩展阅读。

**English:** If you are new to multimodal reasoning or generative models, read in this order:

1. Start with the abstract and introduction of **PixelEyes** to understand the motivation behind active perception and perception-reasoning decoupling.
2. Read the review and PyTorch demo of **CAT** to grasp the core idea of confidence-adaptive search.
3. Skim the method overview of **SNAP-FM** to learn the basics of activation sparsity in flow models.
4. Pick one paper from Other Papers (e.g., AutoMem or Graph-PRefLexOR) for broader context.

### 进阶路线（寻找研究课题）/ Advanced Track

**中文：** 如果你已在相关领域有一定积累，可关注以下开放问题：

1. **PixelEyes** 的感知代理与 LLM 接口：能否设计可微的 top-down attention，使端到端训练更稳定？
2. **SNAP-FM** 的理论保证：亚奈奎斯特采样对流匹配概率流的影响能否量化？
3. **CAT** 的置信度校准：如何将过程奖励模型（PRM）与置信度估计更紧耦合？
4. **交叉方向**：能否将 CAT 的自适应搜索与 PixelEyes 的主动感知结合，构建多模态推理智能体？

**English:** If you already have some background, consider these open problems:

1. The PixelEyes perception-agent-to-LLM interface: can we design a differentiable top-down attention mechanism for more stable end-to-end training?
2. Theoretical guarantees for SNAP-FM: can the impact of sub-Nyquist sampling on the Flow Matching probability flow be quantified?
3. Confidence calibration in CAT: how can process reward models (PRMs) be more tightly coupled with confidence estimation?
4. Cross-direction: can CAT-style adaptive search be combined with PixelEyes-style active perception to build multimodal reasoning agents?

---

## 重要时间节点 / Key Dates

| 事件 / Event | 日期 / Date |
|---|---|
| ICML 2026 | Jul 2026 |
| ACL 2026 | Aug 2026 |
| IJCAI 2026 | Aug 2026 |
| NeurIPS 2026 Submission Deadline | May 2026 (passed) |
| EMNLP 2026 | Nov 2026 |

---

## 备注 / Notes

- 所有审稿分析基于 arXiv 摘要与公开信息，属于“快速预览式审稿”，不等同于正式同行评审。
- CAT 的 PyTorch 代码为概念复现，使用随机 Transformer 演示置信度自适应树搜索机制，不依赖真实大模型权重。
- 本周覆盖 arXiv 2026 年 7 月第 1 周（2607.xxxxx 前缀）的 cs.CV / cs.LG / cs.AI / cs.CL 列表。

All reviews are based on arXiv abstracts and publicly available information and should be considered “rapid-preview reviews,” not formal peer reviews. The CAT PyTorch code is a conceptual reproduction using a random Transformer to demonstrate confidence-adaptive tree search and does not require real LLM weights. This week covers arXiv papers from the first week of July 2026 (2607.xxxxx prefix) in cs.CV / cs.LG / cs.AI / cs.CL.
