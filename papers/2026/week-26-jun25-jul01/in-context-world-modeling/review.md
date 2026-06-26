# In-Context World Modeling for Robotic Control

**Authors / 作者：** Siyin Wang, Junhao Shi, Senyu Fei, Zhaoyang Fu, Li Ji, Jingjing Gong, Xipeng Qiu  
**arXiv / 链接：** [https://arxiv.org/abs/2606.26025](https://arxiv.org/abs/2606.26025)  
**Published / 发表时间：** 2026-06-24 (arXiv announcement ~2026-06-25)  
**Venue / 会议：** TBA

---

## 摘要 / Abstract（原文 verbatim）

> Modern Vision-Language-Action (VLA) models often fail to generalize to novel setups, such as altered camera viewpoints or robot morphologies, because they are typically conditioned only on current observations and language instructions. By ignoring the underlying system configuration as a variable, these models implicitly assume a fixed execution context encountered during training, necessitating data-intensive fine-tuning for any new environment. In this work, we introduce In-Context World Modeling (ICWM), a framework that treats system identification as an in-context adaptation problem. ICWM enables robot policies to autonomously infer essential system variables from a short history of self-generated, task-agnostic interactions. Unlike traditional In-Context Learning that uses demonstrations to specify what task to perform, ICWM leverages the context window to understand how the system operates. By processing these interactions before task execution, the model implicitly captures the world dynamics of the current system, enabling adaptation to novel configurations without parameter updates. Extensive experiments in simulation and on real-world robot platforms demonstrate that ICWM significantly outperforms standard VLA baselines on novel camera viewpoints.

---

## Q1：它真正想解决的问题是什么？ / What problem does it really want to solve?

**中文：** 当前 VLA（Vision-Language-Action）模型通常只在“当前观察 + 语言指令”上做条件，忽略了**系统配置（system configuration）**本身就是一个变量。这导致模型在训练时隐式假设了固定的相机视角、固定的机器人形态、固定的环境物理参数，从而在新环境下泛化能力差，必须针对每个新环境做数据密集型的微调。

ICWM 把“系统辨识”（system identification）重新定义为**上下文适应问题**：不是用上下文来演示“做什么任务”，而是用上下文来理解“系统如何运作”。它让机器人在执行任务前，通过一段自生成的、与任务无关的交互历史，推断出当前系统的关键变量（如相机外参、机器人运动学、环境动力学），从而无需修改模型参数就能适应新配置。

**English:** Current Vision-Language-Action (VLA) models typically condition only on the current observation and language instruction, ignoring the fact that **system configuration** is itself a variable. This causes them to implicitly assume a fixed camera viewpoint, robot morphology, and environmental physics encountered during training, leading to poor generalization in new environments and requiring data-intensive fine-tuning for each new setup.

ICWM reframes **system identification** as an **in-context adaptation problem**: instead of using context to demonstrate what task to perform, it uses context to understand how the system operates. Before task execution, the robot infers essential system variables from a short history of self-generated, task-agnostic interactions, enabling zero-parameter adaptation to novel configurations.

---

## Q2：它声称的贡献是什么？ / What does it claim to contribute?

**中文：**

1. **问题重构**：指出 VLA 模型失败的根本原因是把系统配置当作常量，而非变量；提出将系统辨识视为上下文学习问题。
2. **方法框架 ICWM**：
   - 机器人在执行任务前，先进行一段任务无关（task-agnostic）的自主交互；
   - 利用这些交互历史作为上下文，推断系统的关键变量；
   - 不修改模型参数，仅靠上下文适应新环境。
3. **与传统 ICL 的区分**：传统上下文学习用示例说明“做什么任务”，ICWM 用上下文说明“系统如何运作”。
4. **仿真与真实机器人验证**：在新相机视角下显著超越标准 VLA baseline。
5. **无需微调的泛化范式**：为机器人基础模型提供一种更廉价、更通用的部署方式。

**English:**

1. **Problem reframing**: Identifies that VLA failures stem from treating system configuration as a constant rather than a variable, and proposes treating system identification as an in-context learning problem.
2. **Method framework ICWM**:
   - Before task execution, the robot performs a short phase of self-generated, task-agnostic interactions;
   - Uses this interaction history as context to infer essential system variables;
   - Adapts to new environments without parameter updates.
3. **Distinction from traditional ICL**: Traditional in-context learning uses examples to specify what task to perform; ICWM uses context to specify how the system operates.
4. **Simulation and real-robot validation**: Significantly outperforms standard VLA baselines on novel camera viewpoints.
5. **Fine-tuning-free generalization paradigm**: Offers a cheaper and more general deployment path for robotic foundation models.

---

## Q3：最可能被 reviewer 攻击的地方在哪里？ / Where are reviewers most likely to attack?

**中文：**

1. **“任务无关交互”的成本与安全性**：让机器人在真实环境中先随机交互一番，可能浪费时间、损坏物体或危及安全。论文需要说明这些交互有多长、是否需要人类监督、在真实机器人上是否可行。
2. **上下文长度的实际限制**：VLA 模型通常处理视频帧，上下文窗口有限。如果系统辨识需要大量交互帧，可能会挤压任务执行可用的上下文空间。
3. **系统变量的可辨识性（identifiability）**：并非所有系统变量都能通过少量任务无关交互唯一确定。如果某些变量不可观测或弱可观，ICWM 可能给出错误的世界模型，导致更危险的失败。
4. **Baseline 的强弱**：如果标准 VLA baseline 没有做任何域适应或提示工程，那么 ICWM 的“显著超越”可能不够有说服力。需要与 test-time adaptation、domain randomization、prompt-based calibration 等强 baseline 比较。
5. **泛化范围**：论文重点展示“新相机视角”，但机器人形态（morphology）、环境物理参数、传感器噪声等其他配置变化是否同样有效？实验覆盖度可能不足。
6. **机制解释**：模型真的学到了“世界动力学”，还是只是通过上下文中的视觉线索做了某种启发式匹配？需要 mechanistic 分析来支撑“world modeling”的命名。

**English:**

1. **Cost and safety of task-agnostic interactions**: Letting a robot interact randomly in the real world before task execution can waste time, damage objects, or create safety risks. The paper needs to clarify how long these interactions are, whether human supervision is required, and whether they are feasible on real robots.
2. **Context-length constraints**: VLA models process video frames and have limited context windows. If system identification requires many interaction frames, it may leave insufficient context for actual task execution.
3. **Identifiability of system variables**: Not all system variables can be uniquely determined from a small number of task-agnostic interactions. If some variables are unobservable or weakly observable, ICWM may learn a wrong world model, leading to more dangerous failures.
4. **Strength of baselines**: If the standard VLA baseline receives no domain adaptation or prompt engineering, ICWM's "significant improvement" may be less convincing. Comparisons with strong baselines such as test-time adaptation, domain randomization, and prompt-based calibration are needed.
5. **Scope of generalization**: The paper focuses on novel camera viewpoints, but does ICWM work as well for changes in robot morphology, environmental physics, or sensor noise? Experimental coverage may be limited.
6. **Mechanistic interpretation**: Does the model truly learn "world dynamics," or does it perform some heuristic matching based on visual cues in the context? Mechanistic analysis is needed to support the "world modeling" claim.

---

## Q4：同方向博士生应精读哪些、跳过哪些？ / What should PhD students in this direction read carefully vs. skip?

**中文：**

**必读（精读）：**
- **引言与问题设定**：理解“把系统配置当作变量”这一视角转换，它是本文的核心思想。
- **ICWM 的具体算法流程**：包括如何生成任务无关交互、如何将其编码进上下文、如何在上下文中做系统辨识。这些是设计类似系统的关键。
- **仿真与真实机器人实验的对比**：关注在真实平台上做了哪些简化、哪些指标最能说明问题。

**延伸阅读（按兴趣）：**
- **系统辨识（system identification）经典教材**：如 Ljung 的《System Identification: Theory for the User》，理解传统方法与 ICWM 的区别。
- **VLA 与机器人基础模型**：如 OpenVLA、RT-2、π0，了解当前机器人策略的条件化方式。
- **上下文学习在机器人中的应用**：如利用 demonstration 上下文做 few-shot imitation learning 的工作，比较其与 ICWM 的差异。
- **Domain randomization 与 sim-to-real**：这些是 ICWM 的替代或互补路线。

**可以跳过：**
- 如果你研究的是**端到端机器人学习 but 只关注性能榜单**，本文的“系统辨识”概念可能偏理论，具体方法是否能在你的 benchmark 上 SOTA 需要你自己验证。
- 如果你做的是**纯视觉/纯语言研究**，机器人控制和系统辨识不是你的核心，可以快速浏览引言和结论。

**English:**

**Must-read (close reading):**
- **Introduction and problem formulation**: Understand the shift in perspective from treating system configuration as a constant to treating it as a variable; this is the paper's core idea.
- **Concrete ICWM algorithm**: Including how task-agnostic interactions are generated, how they are encoded into context, and how system identification is performed in context. These are key to designing similar systems.
- **Simulation vs. real-robot experiments**: Pay attention to what simplifications were made on real platforms and which metrics best support the claims.

**Further reading (by interest):**
- **Classical system identification**: e.g., Ljung's *System Identification: Theory for the User*, to understand traditional methods and how ICWM differs.
- **VLA and robotic foundation models**: e.g., OpenVLA, RT-2, π0, to understand how current robot policies are conditioned.
- **In-context learning for robotics**: e.g., few-shot imitation learning with demonstration context; compare with ICWM.
- **Domain randomization and sim-to-real**: These are alternative or complementary routes to ICWM.

**Can skip:**
- If you work on **end-to-end robot learning focused purely on leaderboard performance**, the "system identification" concept may be too theoretical; whether the method achieves SOTA on your benchmark requires your own validation.
- If you work on **pure vision or pure language research**, robot control and system identification are not your core interests; skim the introduction and conclusion.
