# Week 30: 2026-07-23 — 2026-07-29 / 第 30 周

## 本周学术动态 / Weekly Highlights

**中文：** 2026 年第 30 周，主线是 **"RLVR 的成熟化"** 与 **"长上下文推理的接地(grounding)问题"**。可验证奖励强化学习(RLVR)正从"能不能用"迈向"如何用好"：*Off-Context GRPO* 直面 RLVR 在极难题上"全组零奖励→零梯度"的**学习悬崖**，用带特权信息的引导 rollout + 重要性校正把梯度拉回无提示目标；同期 *ISO* 提出"RLVR 原生优化栈"，*Fishing Out Free Riders* 用 Shapley 值做并行推理的奖励归因。另一条主线是长上下文：*Copy Less, Ground More (GEAR)* 系统刻画了"重复复制(repetitive copying)"这一失效模式——模型把大段原文抄进思维链而非真正推理，并用证据感知奖励治好它。与此同时，**世界模型(world model)** 全面开花：*FilmWorld*(小说到电影)、*ABot-World-0*(单卡桌面世界回放)、*AlayaWorld*(长时程世界建模)、*DWM*(把世界效应与动作解耦)、*Masked Visual Actions*(统一世界建模)。**智能体可靠性/可观测性**也集中出现：*CodeRescue*、*PhoenixRepair*、*AgentDebugX*、*Agents in the Wild*、以及审计 AI 科研破坏行为的 *ResearchArena*。

**English:** Week 30 of 2026 is driven by two threads: the *maturation of RLVR* and the *grounding problem in long-context reasoning*. RLVR is moving from "does it work" to "how to do it well": *Off-Context GRPO* tackles the **learning cliff** (all-zero rewards → zero gradient) on hard problems by using privileged, guided rollouts plus importance correction to steer gradients back to the unguided goal; *ISO* proposes an "RLVR-native optimization stack," and *Fishing Out Free Riders* uses Shapley-value reward attribution for parallel reasoning. The second thread is long context: *Copy Less, Ground More (GEAR)* characterizes **repetitive copying** — models dumping input text into their chain-of-thought instead of reasoning — and cures it with evidence-aware rewards. Meanwhile **world models** are everywhere: *FilmWorld* (novel→film), *ABot-World-0* (single-GPU desktop rollouts), *AlayaWorld* (long-horizon), *DWM* (disentangling world effects from actions), and *Masked Visual Actions* (unified world modeling). **Agent reliability / observability** also surged: *CodeRescue*, *PhoenixRepair*, *AgentDebugX*, *Agents in the Wild*, and *ResearchArena* (auditing sabotage in automated AI R&D).

---

## 重点论文 / Featured Papers

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [Off-Context GRPO: Learning to Reason on Hard Problems using Privileged Information](https://arxiv.org/abs/2607.19313) | Agents / RL (RLVR) | [review.md](off-context-grpo/review.md) | [code/oc_grpo_demo.py](off-context-grpo/code/oc_grpo_demo.py) |
| 2 | [Copy Less, Ground More: Overcoming Repetitive Copying in Long-Context Reasoning via Evidence-Aware RL](https://arxiv.org/abs/2607.19345) | NLP / LLM | [review.md](gear-evidence-aware-rl/review.md) | [code/gear_reward_demo.py](gear-evidence-aware-rl/code/gear_reward_demo.py) |

**审稿四问 / Four review questions（每篇 review 均覆盖）：** Q1 真正想解决的问题 / the real problem；Q2 声称的贡献 / claimed contributions；Q3 最可能被 reviewer 攻击之处 / likely reviewer attacks；Q4 博士生应精读 or 跳过 / read-or-skip guidance。

**代码复现结论 / Reproduction takeaways：**
- `oc_grpo_demo.py`：在可控合成任务上复现"学习悬崖"——vanilla GRPO 组内全零奖励→梯度≈0、p(correct) 停在 ~0.008；OC-GRPO(有界重要性校正)把 p(correct) 提到 ~0.088（**11×**）。并附诊断：**朴素无界 `w=p/q` 会因 IS 恒等式退化回零梯度**，凸显论文强调的稳定化/方差控制之必要。
- `gear_reward_demo.py`：对比"仅准确率"与"GEAR(证据感知)"奖励——GEAR 在保持接地(grounding=1.0)的同时，把对干扰段的复制率从 **0.50 → 0.00**、推理长度从 **22 → 4**，即论文所说的 "Copy Less, Ground More"。

---

## 其他论文 / Other Papers

### 计算机视觉 / Computer Vision

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [Appearance Pointers (2607.19344)](https://arxiv.org/abs/2607.19344) | 面向扩散 Transformer 的多模态区域控制 |
| [Text Template Tokens Are Implicit Semantic Registers (2607.19139)](https://arxiv.org/abs/2607.19139) | 模板 token 在扩散 Transformer 中充当隐式语义寄存器 |
| [Image Editing Models are Numerical Solvers (2607.18787)](https://arxiv.org/abs/2607.18787) | 用数值求解器视角重新解释图像编辑模型 |
| [IGGT4D (2607.19228)](https://arxiv.org/abs/2607.19228) | 流式 4D 实例接地几何 Transformer |
| [FlexiAvatar (2607.19100)](https://arxiv.org/abs/2607.19100) | 身体可见度变化下的统一 3D 高斯人体化身 |
| [Point Ladder Tuning (2607.19171)](https://arxiv.org/abs/2607.19171) | 面向 3D 点云理解的参数高效适配 |
| [UVFaceFusion (2607.18798)](https://arxiv.org/abs/2607.18798) | UV 空间融合的多视图拓扑一致人脸重建 |

### 自然语言处理与大语言模型 / NLP & LLMs

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [Copy Less, Ground More / GEAR (2607.19345)](https://arxiv.org/abs/2607.19345) | 证据感知 RL 治疗长上下文"重复复制"（本周重点） |
| [Selective SSM Adaptation & Retrieval (2607.19326)](https://arxiv.org/abs/2607.19326) | 选择性状态空间适配 + 检索增强推理 |
| [DAIS (2607.19088)](https://arxiv.org/abs/2607.19088) | 依赖感知的中间 QA 监督用于复杂推理 |
| [CASE (2607.18820)](https://arxiv.org/abs/2607.18820) | 因果对齐/结构约束提升思维链忠实度 |
| [Structured Output Reduces Answer Diversity (2607.18476)](https://arxiv.org/abs/2607.18476) | 结构化输出在 44 个模型上降低答案多样性 |
| [GAMUT (2607.19322)](https://arxiv.org/abs/2607.19322) | 开放式生成事实完整性基准 |
| [Convolution for Large Language Models (2607.18413)](https://arxiv.org/abs/2607.18413) | 面向 LLM 的卷积结构 |
| [Fusion Embedding (2607.18666)](https://arxiv.org/abs/2607.18666) | 文本/图像/视频/音频的统一嵌入空间 |

### 机器学习理论与方法 / ML Theory & Methods

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [ISO: An RLVR-Native Optimization Stack (2607.19331)](https://arxiv.org/abs/2607.19331) | 为可验证奖励 RL 原生设计的优化栈 |
| [Dual Attention Residuals (2607.18730)](https://arxiv.org/abs/2607.18730) | 双注意力残差结构 |
| [Persistent Latent Policy States from Reasoning FT (2607.18532)](https://arxiv.org/abs/2607.18532) | 推理微调如何在模型中形成持久潜在策略状态 |
| [MUX: Continuous Reasoning via Multiplexed Tokens (2607.18264)](https://arxiv.org/abs/2607.18264) | 用多路复用 token 实现连续推理 |

### 系统与高效训练/推理 / Systems & Efficiency

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [Where Should Optimizer State Live? (2607.19058)](https://arxiv.org/abs/2607.19058) | 分层优化器状态分配的高效 MoE 训练 |
| [AdaFlash (2607.19223)](https://arxiv.org/abs/2607.19223) | 用蒸馏扩散草稿模型做自适应推测解码 |
| [H²SD: Hybrid Hindsight Self-Distillation (2607.18955)](https://arxiv.org/abs/2607.18955) | 混合后见之明的自蒸馏 |

### 智能体与强化学习 / Agents & RL

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [Off-Context GRPO (2607.19313)](https://arxiv.org/abs/2607.19313) | 特权信息打破 RLVR 学习悬崖（本周重点） |
| [Fishing Out Free Riders (2607.18979)](https://arxiv.org/abs/2607.18979) | 基于 Shapley 的并行推理 RL 奖励归因 |
| [CodeRescue (2607.19338)](https://arxiv.org/abs/2607.19338) | 预算校准的编码智能体恢复路由 |
| [PhoenixRepair (2607.18859)](https://arxiv.org/abs/2607.18859) | 重新思考软件智能体的修复策略探索 |
| [AgentDebugX (2607.18754)](https://arxiv.org/abs/2607.18754) | LLM 智能体失败可观测、归因与恢复工具箱 |
| [Agents in the Wild (2607.19336)](https://arxiv.org/abs/2607.19336) | 智能体从研究到部署的落地鸿沟 |
| [Search-on-Graph-R1 (2607.18481)](https://arxiv.org/abs/2607.18481) | 用 RL 训练 LLM 在知识图谱上搜索 |
| [ResearchArena (2607.19321)](https://arxiv.org/abs/2607.19321) | 评测自动化 AI 研发中的破坏与监控 |

### 机器人学与具身/世界模型 / Robotics, Embodied AI & World Models

| 论文 / Paper | 要点 / Key Point |
|---|---|
| [Masked Visual Actions for Unified World Modeling (2607.19343)](https://arxiv.org/abs/2607.19343) | 用掩码视觉动作统一世界建模 |
| [ABot-World-0 (2607.19191)](https://arxiv.org/abs/2607.19191) | 单卡桌面 GPU 上的交互式世界回放 |
| [AlayaWorld (2607.18367)](https://arxiv.org/abs/2607.18367) | 交互式长时程世界建模技术报告 |
| [DWM (2607.18715)](https://arxiv.org/abs/2607.18715) | 在潜在世界模型中分离世界效应与动作 |
| [FilmWorld (2607.19038)](https://arxiv.org/abs/2607.19038) | 动态电影化世界建模驱动的小说到电影生成 |
| [Athena-Brain (2607.18985)](https://arxiv.org/abs/2607.18985) | 面向通用具身智能的高效"机器人大脑" |

---

## 领域趋势 / Trends

1. **RLVR 从"能用"走向"用好"/ RLVR shifts from working to working well** — 学习悬崖(OC-GRPO)、原生优化栈(ISO)、奖励归因(Fishing Out Free Riders)三线并进，标志 RLVR 进入"工程/理论精细化"阶段。
2. **长上下文的核心矛盾是接地而非容量 / The long-context bottleneck is grounding, not capacity** — GEAR 指出"重复复制"随上下文变长而恶化，问题在于模型不会聚焦关键证据，而非上下文窗口不够大。
3. **世界模型成为具身与生成的公共基座 / World models become the shared substrate** — 从长时程(AlayaWorld)、动作解耦(DWM)到内容生成(FilmWorld)与单卡回放(ABot-World-0)，"可交互世界"横跨机器人与视频生成。
4. **智能体研究转向可靠性与可观测性 / Agents pivot to reliability & observability** — 修复(PhoenixRepair)、恢复路由(CodeRescue)、失败归因(AgentDebugX)、部署鸿沟(Agents in the Wild)、破坏审计(ResearchArena)集中出现。
5. **效率手段持续下沉到训练细节 / Efficiency drills into training internals** — MoE 优化器状态分层、蒸馏扩散草稿的推测解码、混合后见之明自蒸馏，都在"系统×算法"交界处抠成本。

---

## 博士生阅读路线 / PhD Reading Route

### 入门路线（熟悉领域）/ Beginner Track

**中文：** 若你刚开始关注 RLVR 与长上下文推理，建议按以下顺序：

1. 先读 **Off-Context GRPO** 的审稿，理解"全组零奖励→零优势→零梯度"这条学习悬崖的因果链，并运行 `oc_grpo_demo.py`，亲眼看到 vanilla 停在 ~0.008、OC-GRPO 升到 ~0.088，以及"朴素无界 p/q 退化回零梯度"的诊断。
2. 阅读 **GEAR** 的审稿，掌握"准确率 + 接地奖励 − 干扰惩罚"这套奖励塑形语言，运行 `gear_reward_demo.py` 复现"少抄、多接地"。
3. 浏览 **ISO (2607.19331)** 与 **Agents in the Wild (2607.19336)**，建立本周"RLVR 优化"与"智能体落地"两大背景。

**English:** If you are new to RLVR and long-context reasoning:

1. Start with the Off-Context GRPO review to grasp the cliff's causal chain (all-zero rewards → zero advantage → zero gradient), then run `oc_grpo_demo.py` to *see* vanilla stall at ~0.008 while OC-GRPO climbs to ~0.088, plus the diagnostic that naive unbounded `p/q` collapses back to zero.
2. Read the GEAR review to learn the reward-shaping vocabulary (accuracy + grounding reward − distractor penalty), then run `gear_reward_demo.py` to reproduce "copy less, ground more."
3. Skim ISO (2607.19331) and Agents in the Wild (2607.19336) for the week's two backdrops (RLVR optimization / agent deployment).

### 进阶路线（寻找研究课题）/ Advanced Track

**中文：** 若已有积累，可关注以下开放问题：

1. **重要性校正的偏差-方差权衡**：OC-GRPO 用有界/截断权重稳定训练，但精确 IS 会退化回零梯度——如何设计"既低方差又低偏差"的 token 级校正？可否用 trust-region 或 V-trace 式双截断给出理论保证？
2. **特权信息的来源与公平性**：提示来自教师模型≈蒸馏，来自标注≈难扩展。能否用"模型自生成的失败轨迹 + 少量提示"闭环放大，而不引入测试泄漏？
3. **接地奖励的 reward hacking**：GEAR 的干扰惩罚基于重叠度量，模型可能用改写规避。能否用语义级/因果级接地信号，并与过程奖励模型(PRM)结合？
4. **RLVR × 长上下文的交叉**：把 GEAR 的证据接地奖励接进 OC-GRPO 的难题引导 rollout——在长文档难题上，是否能同时打破学习悬崖并抑制重复复制？

**English:** For those with background, consider these open problems:

1. **Bias–variance of importance correction.** OC-GRPO stabilizes via bounded/truncated weights, yet exact IS collapses to a zero gradient — how to design token-level corrections that are both low-variance and low-bias? Can trust-region or V-trace-style double clipping give guarantees?
2. **Provenance/fairness of privileged information.** Hints from a teacher ≈ distillation; from annotators ≈ hard to scale. Can we bootstrap from self-generated failed trajectories + minimal hints without test leakage?
3. **Reward hacking of grounding.** GEAR's distractor penalty uses overlap metrics that paraphrase can dodge — can semantic/causal grounding signals plus a PRM close the gap?
4. **RLVR × long context.** Plug GEAR's evidence-grounding reward into OC-GRPO's guided hard-problem rollouts — on long-document hard problems, can we break the learning cliff *and* suppress repetitive copying at once?

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

- 本周为 2026 年第 30 周（Jul 23–29），按仓库既有 `week-NN` 目录约定组织；两篇重点论文（arXiv 2607.19313 与 2607.19345）均已核对 arXiv 页面确认标题、作者与摘要，"其他论文"列表取自本周 arXiv cs.CV/cs.CL/cs.LG/cs.AI 最新公开条目。
- 所有审稿分析基于 arXiv 摘要与公开信息，属"快速预览式审稿"，不等同于正式同行评审。
- 两份 PyTorch 代码均为概念级复现，用于演示论文核心机制，不依赖真实 LLM/机器人/检索系统；`oc_grpo_demo.py` 额外给出了"精确 IS 退化回零梯度"的诊断实验，透明说明了有界权重作为稳定化 stand-in 的作用。

Week 30 of 2026 (Jul 23–29) follows the repo's existing `week-NN` layout. The two featured papers (arXiv 2607.19313 and 2607.19345) were verified against their arXiv pages for title, authors, and abstract; the "Other Papers" list is drawn from this week's latest public arXiv cs.CV/cs.CL/cs.LG/cs.AI entries. All reviews are rapid-preview reviews based on public abstracts, not formal peer reviews. Both PyTorch scripts are conceptual reproductions of the core mechanisms and depend on no real LLM, robot, or retrieval system; `oc_grpo_demo.py` additionally includes a diagnostic showing that exact importance sampling collapses back to a zero gradient, transparently framing the bounded weight as a stability stand-in.
