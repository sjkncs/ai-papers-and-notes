# Week 29: 2026-07-16 — 2026-07-22 / 第 29 周

## 本周学术动态 / Weekly Highlights

**中文：** 2026 年第 29 周，"智能体系统的理论化"与"模型合并 / 蒸馏的机理分析"成为主线。两篇最受关注的论文都在做同一件事——把长期停留在工程直觉层面的问题拉回可证伪的理论：*When Do Multi-Agent Systems Help?* 用信息瓶颈（Information Bottleneck）刻画多智能体（MAS）相对单智能体（SAS）的优势边界，指出 MAS 的非平凡收益只能来自"有界中继信道"下的压缩权衡；*When Model Merging Rivals Joint Multi-Task RL* 则补齐了强化学习后训练里长期缺失的对照实验，用任务向量几何（近乎正交 + 支撑集重叠）解释了"为什么不同合并方法几乎没有区别"。与此同时，本周 arXiv 与 Hugging Face 榜单被 on-policy 蒸馏（SEED、ShortOPD、Demystifying On-Policy Distillation）、长上下文 RL（LongStraw）、万亿参数 Zero-RL 推理（Ring-Zero）、世界-动作模型（BadWAM、GigaWorld-Policy）、以及一波具身智能 / 机器人基座模型（ABot-N1、Xiaomi-Robotics-U0、RoboTTT、EgoSteer）刷屏。

**English:** Week 29 of 2026 is dominated by two threads: *theorizing agentic systems* and *mechanistic analysis of model merging / distillation*. The two most-discussed papers share one ambition — dragging long-standing engineering intuitions back onto falsifiable theoretical ground. *When Do Multi-Agent Systems Help?* uses an Information Bottleneck view to delimit when multi-agent (MAS) beats single-agent (SAS), arguing the nontrivial MAS advantage can only come from the compression trade-off under bounded relay channels. *When Model Merging Rivals Joint Multi-Task RL* builds the head-to-head baseline long missing in RL post-training and explains — via task-vector geometry (near-orthogonal directions + overlapping support) — why different merge methods are statistically indistinguishable. Meanwhile the arXiv / Hugging Face charts are flooded with on-policy distillation (SEED, ShortOPD, Demystifying On-Policy Distillation), long-context RL (LongStraw), trillion-parameter Zero-RL reasoning (Ring-Zero), world-action models (BadWAM, GigaWorld-Policy), and a wave of embodied / robotics foundation models (ABot-N1, Xiaomi-Robotics-U0, RoboTTT, EgoSteer).

---

## 重点论文 / Featured Papers

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [When Do Multi-Agent Systems Help? An Information Bottleneck Perspective](https://arxiv.org/abs/2607.16133) | Agents / ML Theory | [review.md](mas-information-bottleneck/review.md) | [code/mas_ib_demo.py](mas-information-bottleneck/code/mas_ib_demo.py) |
| 2 | [When Model Merging Rivals Joint Multi-Task Reinforcement Learning: A Task-Vector Geometry Analysis](https://arxiv.org/abs/2607.16062) | Model Merging / Multi-Task RL | [review.md](task-vector-geometry/review.md) | [code/task_vector_geometry_demo.py](task-vector-geometry/code/task_vector_geometry_demo.py) |

**审稿四问 / Four review questions（每篇 review 均覆盖）：** Q1 真正想解决的问题 / the real problem；Q2 声称的贡献 / claimed contributions；Q3 最可能被 reviewer 攻击之处 / likely reviewer attacks；Q4 博士生应精读 or 跳过 / read-or-skip guidance。

---

## 其他论文 / Other Papers

### 计算机视觉 / Computer Vision

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [VideoChat3 (2607.14935)](https://arxiv.org/abs/2607.14935) | 全开源的通用视频多模态大模型，兼顾效率与视频理解泛化 |
| [Boogu-Image-0.1 (2607.13125)](https://arxiv.org/abs/2607.13125) | 增强开源统一多模态理解与生成的基座 |
| [UniVR (2607.12800)](https://arxiv.org/abs/2607.12800) | "在视觉空间中思考"的统一视觉推理框架 |
| [Registers Matter for Pixel-Space Diffusion Transformers (2605.16147)](https://arxiv.org/abs/2605.16147) | register token 对像素空间扩散 Transformer 的作用 |
| [4D Human-Scene Reconstruction from Low-Overlap Captures (2607.09125)](https://arxiv.org/abs/2607.09125) | 低重叠采集下的 4D 人-场景重建 |
| [AsySplat (2607.10995)](https://arxiv.org/abs/2607.10995) | 面向长序列场景的高效非对称 3D 高斯泼溅 |

### 自然语言处理与大语言模型 / NLP & LLMs

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [LongStraw (2607.14952)](https://arxiv.org/abs/2607.14952) | 固定 GPU 预算下超越 2M token 的长上下文 RL |
| [Ring-Zero (2607.12395)](https://arxiv.org/abs/2607.12395) | 把 Zero-RL 扩展到万亿参数以激发涌现推理 |
| [Metacognition in LLMs (2607.11881)](https://arxiv.org/abs/2607.11881) | LLM 元认知的基础、进展与机会（综述） |
| [Token Time Continuous Diffusion (2607.14106)](https://arxiv.org/abs/2607.14106) | 面向语言建模的 token 时间连续扩散 |
| [Partition, Prompt, Aggregate (2607.15277)](https://arxiv.org/abs/2607.15277) | 语言模型的统计式自一致性 |

### 机器学习理论 / ML Theory

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [Task-Vector Geometry Analysis (2607.16062)](https://arxiv.org/abs/2607.16062) | 任务向量近乎正交 + 支撑集重叠解释合并方法无差别（本周重点） |
| [A Theory of Contrastive Learning with Natural Images (2607.07470)](https://arxiv.org/abs/2607.07470) | 自然图像上对比学习的理论刻画 |
| [DeepLoop: Depth Scaling for Looped Transformers (2607.13491)](https://arxiv.org/abs/2607.13491) | 循环 Transformer 的深度缩放 |
| [Demystifying On-Policy Distillation (2607.13399)](https://arxiv.org/abs/2607.13399) | On-policy 蒸馏的角色、病理与调控 |
| [Spectral Rewiring for Exploration, Purification, and Model Merging (2607.03065)](https://arxiv.org/abs/2607.03065) | 谱重连视角下的探索、净化与模型合并 |

### 系统与高效推理 / Systems & Efficient Inference

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [Byte-Exact KV-Cache Grafting (2607.14431)](https://arxiv.org/abs/2607.14431) | 字节级精确的 KV-Cache 嫁接，同时更快更省 |
| [KronQ (2607.07964)](https://arxiv.org/abs/2607.07964) | 基于 Kronecker 分解 Hessian 的 LLM 量化 |
| [ShortOPD (2607.13124)](https://arxiv.org/abs/2607.13124) | 短到长 on-policy 蒸馏恢复被剪枝的 LLM |
| [PalmClaw (2607.13027)](https://arxiv.org/abs/2607.13027) | 面向手机的原生端侧智能体框架 |

### 智能体与强化学习 / Agents & RL

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [Multi-Agent Systems Information Bottleneck (2607.16133)](https://arxiv.org/abs/2607.16133) | MAS 何时有帮助的信息瓶颈判据（本周重点） |
| [SEED (2607.14777)](https://arxiv.org/abs/2607.14777) | 面向智能体 RL 的自演化 on-policy 蒸馏 |
| [SearchOS-V1 (2607.15257)](https://arxiv.org/abs/2607.15257) | 鲁棒的开放域信息搜寻智能体协作 |
| [Multi-Agent LLMs Fail to Explore Each Other (2607.11250)](https://arxiv.org/abs/2607.11250) | 多智能体 LLM 之间探索不足的实证分析 |
| [Self-Improvements in Modern Agentic Systems: A Survey (2607.13104)](https://arxiv.org/abs/2607.13104) | 现代智能体系统自我改进综述 |
| [GRASP (2607.10463)](https://arxiv.org/abs/2607.10463) | 面向 Agentic RAG 的粒度感知搜索策略 |

### 机器人学与具身智能 / Robotics & Embodied AI

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [ABot-N1 (2607.10383)](https://arxiv.org/abs/2607.10383) | 通用视觉语言导航基座模型 |
| [Xiaomi-Robotics-U0 (2607.11643)](https://arxiv.org/abs/2607.11643) | 由世界基座模型驱动的统一具身合成 |
| [RoboTTT (2607.15275)](https://arxiv.org/abs/2607.15275) | 机器人策略的上下文扩展 |
| [EgoSteer (2607.09701)](https://arxiv.org/abs/2607.09701) | 从第一视角视频学习可操控的灵巧操作 |
| [BadWAM (2607.15207)](https://arxiv.org/abs/2607.15207) | 世界-动作模型"梦对了却做错了"的分析 |

---

## 领域趋势 / Trends

1. **智能体研究从"更强的流水线"转向"可证伪的理论"/ Agent research shifts from stronger pipelines to falsifiable theory** — MAS-IB 用信息瓶颈给出"何时有帮助"的判据，Multi-Agent LLMs Fail to Explore 则从反面提供实证边界。
2. **模型合并进入机理分析时代 / Model merging enters the mechanistic era** — 任务向量几何（近正交 + 支撑集重叠）解释了不同合并算法为何无差别，Spectral Rewiring 从谱视角切入合并/净化。
3. **On-policy 蒸馏成为后训练主力 / On-policy distillation becomes the post-training workhorse** — SEED、ShortOPD、Demystifying 三篇同周出现，覆盖自演化、剪枝恢复与病理分析。
4. **长上下文与超大规模 RL 并行推进 / Long-context and ultra-scale RL advance in parallel** — LongStraw 在固定预算下突破 2M token，Ring-Zero 把 Zero-RL 推到万亿参数。
5. **具身智能基座模型密集发布 / Embodied foundation models proliferate** — 导航（ABot-N1）、统一合成（Xiaomi-U0）、上下文扩展（RoboTTT）、灵巧操作（EgoSteer）齐头并进。

---

## 博士生阅读路线 / PhD Reading Route

### 入门路线（熟悉领域）/ Beginner Track

**中文：** 如果你刚开始关注智能体系统与模型合并，本周建议按以下顺序：

1. 先读 **MAS Information Bottleneck** 的审稿分析，理解"MAS 优势只能来自有界中继压缩权衡"这一核心直觉，并运行 `mas_ib_demo.py` 直观看到 MAS 增益随瓶颈宽度 b 的变化。
2. 阅读 **Task-Vector Geometry** 的审稿，掌握任务向量、近正交、支撑集重叠与"方向-支撑解耦"这套几何语言，运行 `task_vector_geometry_demo.py` 复现"合并方法坍缩为均匀平均"。
3. 浏览 **Demystifying On-Policy Distillation (2607.13399)** 与 **Self-Improvements Survey (2607.13104)**，建立本周两大热点（蒸馏 / 智能体自改进）的全景。

**English:** If you are new to agentic systems and model merging, read in this order:

1. Start with the MAS Information Bottleneck review to grasp "the MAS advantage can only come from the bounded-relay compression trade-off," then run `mas_ib_demo.py` to see the MAS gain vary with bottleneck width b.
2. Read the Task-Vector Geometry review to learn the geometric vocabulary (task vectors, near-orthogonality, support overlap, direction–support decoupling), then run `task_vector_geometry_demo.py` to reproduce "merge methods collapse to uniform averaging."
3. Skim Demystifying On-Policy Distillation (2607.13399) and the Self-Improvements Survey (2607.13104) for a panorama of this week's two hot threads (distillation / agentic self-improvement).

### 进阶路线（寻找研究课题）/ Advanced Track

**中文：** 如果你已有一定积累，可关注以下开放问题：

1. **MAS-IB 的可测量性**：有效参数 β 与"中继信息损失"在真实 LLM 系统里如何估计而不陷入循环论证（用下游准确率反推）？
2. **任务向量几何的可扩展性**：当专家数量 > 2、任务真正异构（代码 / 数学 / 对话）时，近正交与"合并=均匀平均"是否还成立？干扰何时出现？
3. **蒸馏 × 合并的交叉**：on-policy 蒸馏得到的学生模型，其任务向量几何是否与独立 RL 专家不同？能否用几何指标预测"哪些专家适合合并"？
4. **理论 × 具身**：MAS-IB 的中继瓶颈能否解释多机器人 / 多模块具身系统中的通信约束？

**English:** If you already have some background, consider these open problems:

1. **Measurability of MAS-IB.** How can the effective β and "relay information loss" be estimated in real LLM systems without circularity (back-solving from downstream accuracy)?
2. **Scalability of task-vector geometry.** With > 2 specialists and genuinely heterogeneous tasks (code / math / dialogue), do near-orthogonality and "merge = uniform averaging" still hold? When does interference appear?
3. **Distillation × merging.** Do students from on-policy distillation exhibit different task-vector geometry than independent RL specialists? Can geometric metrics predict which specialists are mergeable?
4. **Theory × embodiment.** Can the MAS-IB relay bottleneck explain communication constraints in multi-robot / multi-module embodied systems?

---

## 重要时间节点 / Key Dates

| 事件 / Event | 日期 / Date |
|---|---|
| ICML 2026 | Jul 2026 |
| ACL 2026 | Aug 2026 |
| IJCAI 2026 | Aug 2026 |
| COLM 2026 | Oct 2026 |
| EMNLP 2026 | Nov 2026 |
| NeurIPS 2026 | Dec 2026 |

---

## 备注 / Notes

- 本周两篇重点论文（arXiv 2607.16133 与 2607.16062）均已核对 arXiv 页面确认标题、作者与摘要；"其他论文"列表取自 2026 年第 29 周（Jul 12–18）arXiv / Hugging Face 公开榜单的真实条目。
- 所有审稿分析基于 arXiv 摘要与公开信息，属"快速预览式审稿"，不等同于正式同行评审。
- 两份 PyTorch 代码均为概念复现，用于演示论文核心机制：`mas_ib_demo.py` 用可控合成信道再现"中继瓶颈宽度 → MAS 增益"曲线；`task_vector_geometry_demo.py` 复现"任务向量近正交 + 支撑集重叠 → 合并坍缩为均匀平均"。不依赖真实 LLM、机器人或检索系统。

The two featured papers (arXiv 2607.16133 and 2607.16062) were verified against their arXiv pages for title, authors, and abstract; the "Other Papers" list is drawn from real entries on the arXiv / Hugging Face charts for Week 29 of 2026 (Jul 12–18). All reviews are rapid-preview reviews based on public abstracts, not formal peer reviews. The two PyTorch scripts are conceptual reproductions of the core mechanisms and depend on no real LLM, robot, or retrieval system.
