# Week 19: 2026-05-07 — 2026-05-13 / 第 19 周

> AI top-conference paper newsletter curated for researchers and PhD students.
> 面向研究者与博士生的 AI 顶会论文周报。

---

## Weekly Highlights / 本周亮点

1. **Skill libraries enter the mainstream.** Two independent works (Skill1 and AHD Agent) argue that reusable skill memory is the missing layer between LLM reasoning and long-horizon agency.
   **技能库正在成为主流。** Skill1 与 AHD Agent 两篇文章指出，可复用的技能记忆是大模型推理与长程智能体之间的缺失层。

2. **Action guidance can replace expensive SFT for agents.** ActGuide-RL shows how external behavioral hints can be folded into an RL objective without heavy supervised fine-tuning.
   **动作指导有望替代昂贵的监督微调。** ActGuide-RL 将外部行为提示融入强化学习目标，无需大量 SFT 数据。

3. **Bigger vision models are not automatically more explainable.** A careful scaling study finds no consistent improvement in localization-based explanations as model size grows.
   **更大的视觉模型未必更可解释。** 一项 scaling 研究发现，基于定位的解释质量并未随模型规模稳定提升。

4. **Test-time adaptation keeps appearing.** Linearizing Vision Transformer with Test-Time Training and MicroViTv2 show that inference-time tuning remains a hot design axis for efficient CV.
   **测试时适应持续升温。** 线性化 ViT 与 MicroViTv2 表明，推理时调优仍是高效 CV 的重要设计维度。

5. **Multi-agent RL moves toward orchestration traces.** Researchers are now treating agent coordination as a learnable scheduling problem rather than hand-written protocols.
   **多智能体 RL 转向编排轨迹。** 研究者将智能体协作视为可学习的调度问题，而非手写协议。

6. **Durable intermediate artifacts become a first-class systems concern.** The Rosen & Rosen data-model paper makes the case for persistent, inspectable agent state.
   **持久化的中间产物成为系统级一等公民。** Rosen 等人提出的数据模型主张智能体状态应可持久化、可审查。

7. **Wheeled-legged robot control gets a unified architecture.** MUJICA reports real-robot gains on Unitree Go2-W by integrating multiple skills into a single joint controller.
   **轮腿机器人控制迎来统一架构。** MUJICA 在 Unitree Go2-W 上通过统一多技能的关节控制器取得实机提升。

8. **Generalization theory catches up with feature learning.** A new non-asymptotic NTK analysis explains benign overfitting, double descent, and grokking in a single framework.
   **泛化理论正在追赶特征学习。** 新的非渐近 NTK 分析在一个框架内解释了良性过拟合、双下降与 grokking。

---

## Featured Papers / 精选论文

| Paper / 论文 | arXiv | Area / 领域 | Why read / 阅读理由 |
|---|---|---|---|
| Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning | [2605.06130](https://arxiv.org/abs/2605.06130) | Agents & RL | First end-to-end skill library plus RL in one loop. / 首个将技能库与 RL 统一训练的框架。 |
| Learning Agentic Policy from Action Guidance | [2605.12004](https://arxiv.org/abs/2605.12004) | Agents & RL | Avoids heavy SFT by distilling guidance into RL. / 通过将指导蒸馏到 RL，避免大量 SFT。 |
| Scaling Vision Models Does Not Consistently Improve Localisation-Based Explanation Quality | [2605.10142](https://arxiv.org/abs/2605.10142) | CV / XAI | Cautions against scaling-for-safety assumptions. / 对“越大越安全/可解释”的假设提出警告。 |
| MicroViTv2: Beyond the FLOPS for Edge Energy-Friendly Vision Transformers | [2605.10148](https://arxiv.org/abs/2605.10148) | CV / Systems | Real-device efficient ViT with SDTA attention. / 面向真实设备的低能耗 ViT，采用 SDTA 注意力。 |

Detailed reviews are in the corresponding sub-folders:
- `skill1-unified-evolution-of-skill-augmented-agents/review.md`
- `learning-agentic-policy-from-action-guidance/review.md`
- `scaling-vision-models-does-not-consistently-improve-localisation-based-explanation-quality/review.md`

A runnable PyTorch reproduction for **Learning Agentic Policy from Action Guidance** is provided in:
- `learning-agentic-policy-from-action-guidance/code/actguide_rl.py`

---

## Weekly Papers by Area / 按领域分类

### Computer Vision / 计算机视觉

| Title / 标题 | arXiv | Date / 日期 |
|---|---|---|
| Scaling Vision Models Does Not Consistently Improve Localisation-Based Explanation Quality | [2605.10142](https://arxiv.org/abs/2605.10142) | 2026-05-11 |
| MicroViTv2: Beyond the FLOPS for Edge Energy-Friendly Vision Transformers | [2605.10148](https://arxiv.org/abs/2605.10148) | 2026-05-11 |
| Representative Attention For Vision Transformers | [2605.14913](https://arxiv.org/abs/2605.14913) | 2026-05-14 |
| Vision Transformers Need Better Token Interaction | [2605.23868](https://arxiv.org/abs/2605.23868) | 2026-05-22 |

### NLP & LLM / 自然语言处理与大模型

| Title / 标题 | arXiv | Date / 日期 |
|---|---|---|
| Can LLMs Take Retrieved Information with a Grain of Salt? | [2605.06919](https://arxiv.org/abs/2605.06919) | 2026-05-07 |
| Shaping Schema via Language Representation as the Next Frontier for LLM Intelligence Expanding | [2605.09271](https://arxiv.org/abs/2605.09271) | 2026-05-10 |
| ProgramBench: Can Language Models Rebuild Programs From Scratch? | [2605.03546](https://arxiv.org/abs/2605.03546) | 2026-05-05 |
| Customizing an LLM for Enterprise Software Engineering | [2605.16517](https://arxiv.org/abs/2605.16517) | 2026-05-15 |

### ML Theory / 机器学习理论

| Title / 标题 | arXiv | Date / 日期 |
|---|---|---|
| A Theory of Generalization in Deep Learning | [2605.01172](https://arxiv.org/abs/2605.01172) | 2026-05-02 |
| Extrapolation in Statistical Learning with Extreme Value Theory | [2605.01909](https://arxiv.org/abs/2605.01909) | 2026-05-03 |
| Enhanced and Efficient Reasoning in Large Learning Models | [2605.14036](https://arxiv.org/abs/2605.14036) | 2026-05-13 |

### Systems / 系统

| Title / 标题 | arXiv | Date / 日期 |
|---|---|---|
| Intermediate Artifacts as First-Class Citizens: A Data Model for Durable Intermediate Artifacts in Agentic Systems | [2605.12087](https://arxiv.org/abs/2605.12087) | 2026-05-12 |
| AiraXiv: An AI-Driven Open-Access Platform for Human and AI Scientists | [2605.21481](https://arxiv.org/abs/2605.21481) | 2026-05-20 |

### Agents & RL / 智能体与强化学习

| Title / 标题 | arXiv | Date / 日期 |
|---|---|---|
| Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning | [2605.06130](https://arxiv.org/abs/2605.06130) | 2026-05-07 |
| AHD Agent: Agentic Reinforcement Learning for Automatic Heuristic Design | [2605.08756](https://arxiv.org/abs/2605.08756) | 2026-05-09 |
| Learning Agentic Policy from Action Guidance | [2605.12004](https://arxiv.org/abs/2605.12004) | 2026-05-12 |
| Reinforcement Learning for LLM-based Multi-Agent Systems through Orchestration Traces | [2605.02801](https://arxiv.org/abs/2605.02801) | 2026-05-04 |

### Robotics / 机器人学

| Title / 标题 | arXiv | Date / 日期 |
|---|---|---|
| Embodied AI in Action: Insights from SAE World Congress 2026 on Safety, Trust, Robotics, and Real-World Deployment | [2605.10653](https://arxiv.org/abs/2605.10653) | 2026-05-11 |
| MUJICA: Multi-skill Unified Joint Integration of Control Architecture for Wheeled-Legged Robots | [2605.13058](https://arxiv.org/abs/2605.13058) | 2026-05-13 |
| Hybrid LLM-based Intelligent Framework for Robot Task Scheduling | [2605.15486](https://arxiv.org/abs/2605.15486) | 2026-05-15 |
| World Model for Robot Learning: A Comprehensive Survey | [2605.00080](https://arxiv.org/abs/2605.00080) | 2026-05-01 |

---

## Trends / 趋势

1. **From few-shot prompting to persistent skill memory.** Frontier agents are being redesigned around read-write memory that survives across episodes, not just in-context examples.
   **从少样本提示到持久技能记忆。** 前沿智能体正围绕可跨回合读写的记忆重新设计，而非仅依赖上下文示例。

2. **Guidance-to-policy distillation is emerging as an alternative to SFT.** Action guidance, preferences, and heuristics are being converted into soft RL objectives to reduce annotation cost.
   **指导到策略的蒸馏正在成为 SFT 的替代方案。** 动作指导、偏好与启发式规则被转化为软 RL 目标，以降低标注成本。

3. **Efficient vision is shifting from FLOP-counting to real-device metrics.** Throughput, latency, and energy on actual edge hardware are becoming the primary optimization targets.
   **高效视觉正从 FLOP 统计转向真实设备指标。** 实际边缘硬件上的吞吐、延迟与能耗成为主要优化目标。

4. **Agentic systems are becoming inspectable infrastructure.** Papers now argue for durable intermediate artifacts, provenance tracking, and institutional readiness before deployment.
   **智能体系统正成为可审查的基础设施。** 文献开始主张持久化中间产物、溯源追踪以及部署前的机构就绪评估。

5. **Theory is reconnecting with empirical phenomena.** New non-asymptotic analyses of generalization aim to explain grokking, double descent, and benign overfitting in feature-learning regimes.
   **理论正在重新对接经验现象。** 新的非渐近泛化分析试图解释特征学习场景下的 grokking、双下降与良性过拟合。

---

## PhD Reading Roadmap / 博士阅读路线图

### For first-year PhDs / 一年级博士生
1. Start with the **Skill1** and **ActGuide-RL** reviews to understand the modern agent training stack.
   从 **Skill1** 与 **ActGuide-RL** 的评述入手，理解现代智能体训练栈。
2. Read **MicroViTv2** as an example of how to write an efficient-model paper with real-device evaluation.
   阅读 **MicroViTv2**，学习如何撰写基于真实设备评估的高效模型论文。
3. Run the provided `actguide_rl.py` toy code to internalize the guidance-distillation idea.
   运行附带的 `actguide_rl.py` 玩具代码，内化指导蒸馏思想。

### For advanced students / 高年级博士生
1. Dive into **A Theory of Generalization in Deep Learning** and relate it to your own overparameterized experiments.
   深入阅读 **A Theory of Generalization in Deep Learning**，并联系自己的过参数化实验。
2. Compare **Scaling Vision Models** with prior XAI scaling studies and design a follow-up metric study.
   将 **Scaling Vision Models** 与已有 XAI scaling 研究对比，设计后续指标研究。
3. Use **Intermediate Artifacts as First-Class Citizens** to think about the systems paper you could build around your agent demo.
   借助 **Intermediate Artifacts as First-Class Citizens** 思考如何围绕自己的智能体 demo 撰写系统论文。

---

## Key Dates / 重要日期

| Conference / 会议 | Deadline / 截止日期 | Location / 地点 |
|---|---|---|
| NeurIPS 2026 abstract | 2026-05-11 | Vancouver |
| ICML 2026 | 2026-02 (accepted papers announced) | Rio de Janeiro, July |
| ACL 2026 | 2026-07-15 (main) | Prague |
| ECCV 2026 | 2026-03 (reviews) | Tel Aviv, Sept |
| CoRL 2026 | 2026-06-15 | Daejeon, Korea |
| ICRA 2027 | 2026-09-15 | Atlanta |

> Dates are approximate and should be confirmed on official websites.
> 日期为近似值，请以官网为准。

---

## Sources / 来源

All papers were identified from the arXiv `cs.AI`, `cs.CV`, `cs.LG`, and `cs.RO` announcement mailing lists for the period 2026-05-01 to 2026-05-26. Additional context was drawn from:

- arXiv Artificial Intelligence listings: https://arxiv.org/list/cs.AI/recent
- arXiv Computer Vision listings: https://arxiv.org/list/cs.CV/recent
- arXiv Learning listings: https://arxiv.org/list/cs.LG/recent
- arXiv Robotics listings: https://arxiv.org/list/cs.RO/recent
- LinkedIn AI News Week 19: https://www.linkedin.com/pulse/ai-news-week-19-may-4-10-2026-mantas-lukauskas-phd-vyjnc

Individual paper links are listed in the tables above.
