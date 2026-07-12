# EmbodiedWorld: A Unified World Model for Humanoid Manipulation and Locomotion / EmbodiedWorld：统一人形操作与运动的世界模型

- **arXiv:** [2605.07134](https://arxiv.org/abs/2605.07134)
- **Date:** 2026-05-20
- **Authors:** UC Berkeley BAIR, Boston Dynamics AI Institute, University of Tokyo, ETH Zurich
- **Target venues:** RSS 2026, CoRL 2026

---

## Key Excerpts / 关键摘录

> "EmbodiedWorld trains a single transformer-based world model on Internet-scale human motion data, large-scale physics simulation, and real robot teleoperation trajectories to jointly predict future proprioception, visual observations, and contact forces." / “EmbodiedWorld 在互联网规模的人体运动数据、大规模物理仿真以及真实机器人遥操作轨迹上训练单一基于 Transformer 的世界模型，联合预测未来本体感觉、视觉观测与接触力。”

> "The model enables zero-shot transfer of bimanual manipulation policies from a 7-DoF fixed-base manipulator to a 33-DoF humanoid without any real-world fine-tuning on the target platform." / “该模型可将双手操作策略从 7 自由度固定基座机械臂零样本迁移到 33 自由度人形机器人，无需在目标平台上进行任何真实世界微调。”

> "By conditioning future prediction on latent action tokens, EmbodiedWorld supports both model-predictive control and reinforcement-learning fine-tuning, bridging the gap between world-model imagination and closed-loop robot execution." / “通过以潜在动作 token 为条件预测未来，EmbodiedWorld 同时支持模型预测控制与强化学习微调，弥合世界模型想象与闭环机器人执行之间的鸿沟。”

> "On a suite of contact-rich tasks including peg insertion, drawer opening, and stair ascent, EmbodiedWorld reduces real-world sample complexity by an order of magnitude compared with behavior-cloning baselines." / “在一组富含接触的任务（包括插销、开抽屉、上楼梯）上，EmbodiedWorld 相比行为克隆基线将真实世界样本复杂度降低一个数量级。”

---

## Q1. Core Problem / 核心问题

**English:**
Humanoid robots promise general-purpose physical assistance, but learning policies for high-degree-of-freedom (DoF) bodies remains data-hungry, fragile, and platform-specific. Separate manipulation and locomotion models fail to exploit shared physical priors, and simulation-to-reality gaps often require expensive real-world fine-tuning. The core problem is to learn a single, general-purpose world model of physical interaction that supports both manipulation and locomotion, transfers across robot morphologies, and reduces real-world data requirements.

**中文：**
人形机器人有望提供通用物理辅助，但学习高自由度（DoF）身体的策略仍需要大量数据、脆弱且平台相关。分开的操作与运动模型无法利用共享的物理先验，仿真到现实的差距通常需要昂贵的真实世界微调。核心问题在于：学习一个通用的物理交互世界模型，同时支持操作与运动、跨机器人形态迁移，并降低真实世界数据需求。

---

## Q2. Claimed Contributions / 声称贡献

**English:**
1. **Unified humanoid world model:** A single transformer predicts future states across vision, proprioception, and contact forces for both manipulation and locomotion.
2. **Cross-embodiment policy transfer:** Policies trained on one morphology transfer zero-shot to different morphologies (e.g., fixed arm to humanoid) by mapping actions into a shared latent action space.
3. **Latent action-token conditioning:** Actions are represented as discrete tokens, enabling the model to be used both as a simulator for planning and as an initialization for RL fine-tuning.
4. **Multimodal training mixture:** Combines human video, physics simulation, and real robot trajectories into a unified pre-training objective.
5. **Real-world sample efficiency:** Achieves strong performance on contact-rich tasks with an order of magnitude fewer real-world episodes than behavior cloning.

**中文：**
1. **统一人形世界模型：** 单一 Transformer 同时预测操作与运动任务下的视觉、本体感觉与接触力未来状态。
2. **跨形态策略迁移：** 通过将动作映射到共享潜在动作空间，在一种形态上训练的策略可零样本迁移到不同形态（如固定机械臂到人形）。
3. **潜在动作 token 条件化：** 动作被表示为离散 token，使模型既可作为规划仿真器，也可作为强化学习微调的初始化。
4. **多模态训练混合：** 将人体视频、物理仿真与真实机器人轨迹整合到统一预训练目标中。
5. **真实世界样本效率：** 在富含接触的任务上，以比行为克隆少一个数量级的真实世界回合数取得强劲性能。

---

## Q3. Reviewer Attack Points / 审稿人攻击点

**English:**
1. **Domain gap between human video and robot embodiment:** Internet human videos lack robot morphology and dynamics; the paper must explain how the shared latent action space avoids absorbing human-specific biomechanics that do not transfer.
2. **Contact-force prediction accuracy:** Accurate contact-force prediction is crucial for contact-rich manipulation, but force labels are noisy or absent in human videos and many simulators; evaluation of force accuracy is limited.
3. **Task diversity and generalization:** The evaluated tasks (peg insertion, drawer opening, stair ascent) are still relatively constrained; claims of "general-purpose" world modeling require broader open-ended household or industrial tasks.
4. **Sim-to-real fine print:** Zero-shot transfer is impressive, but the paper should clarify whether the target humanoid was seen during training in simulation or whether the morphology is truly held out.
5. **Compute and data scale:** Training a single transformer on Internet-scale video plus large-scale simulation is expensive; the carbon cost, wall-clock time, and exact data scale are not detailed, limiting reproducibility.

**中文：**
1. **人体视频与机器人形态之间的域差距：** 互联网人体视频缺少机器人形态与动态；论文需解释共享潜在动作空间如何避免吸收无法迁移的人体特定生物力学。
2. **接触力预测精度：** 准确的接触力预测对富含接触的操作至关重要，但人体视频和许多仿真器中力的标注缺失或嘈杂；力精度评估有限。
3. **任务多样性与泛化：** 评估任务（插销、开抽屉、上楼梯）仍相对受限；“通用”世界模型的主张需要更广泛的开放式家庭或工业任务支持。
4. **仿真到现实的细节：** 零样本迁移令人印象深刻，但论文应说明目标人形是否在仿真训练时出现过，还是形态真正被留出。
5. **计算与数据规模：** 在互联网规模视频加大规模仿真上训练单一 Transformer 成本高昂；碳成本、墙钟时间与确切数据规模未详细说明，限制了可复现性。

---

## Q4. PhD Reading Guide / 博士阅读指南

**English:**
- **Pre-requisites:** Read RT-2, RT-X, and Octo for robot foundation models; read DreamerV3 and UniPi for world-model-based RL; review humanoid simulators such as Isaac Gym, MuJoCo, or Genesis.
- **Study sequence:** Start with Section 3 (problem formulation and data mixture), then Section 4 (model architecture), Section 5 (latent action space and transfer), and Section 6 (experiments).
- **Critical reading questions:** (1) How is the shared latent action space trained, and what prevents human-video biases from leaking into robot policies? (2) Does zero-shot transfer work for dynamic, contact-rich tasks or only quasi-static ones? (3) How much does real-world data still help if fine-tuning is allowed?
- **Reproduction tips:** Begin with a small embodiment-transfer experiment in MuJoCo using two morphologies (e.g., 7-DoF arm and a simple humanoid torso); verify that latent action tokens align across bodies.
- **Follow-up ideas:** Integrate tactile sensing into the world model, combine EmbodiedWorld with language-conditioned task planners, or study safety guarantees when the world model is used in closed-loop MPC.

**中文：**
- **前置知识：** 阅读 RT-2、RT-X、Octo 等机器人基础模型；阅读 DreamerV3 与 UniPi 等基于世界模型的强化学习；回顾 Isaac Gym、MuJoCo 或 Genesis 等人形仿真器。
- **阅读顺序：** 从第 3 节（问题形式化与数据混合）开始，然后读第 4 节（模型架构）、第 5 节（潜在动作空间与迁移）、第 6 节（实验）。
- **批判性问题：** (1) 共享潜在动作空间如何训练，如何防止人体视频偏差泄漏到机器人策略？(2) 零样本迁移是否适用于动态、富含接触的任务，还是仅适用于准静态任务？(3) 若允许微调，真实世界数据仍能带来多大提升？
- **复现建议：** 在 MuJoCo 中从两个形态（如 7 自由度机械臂与简单人形躯干）的小型迁移实验开始，验证潜在动作 token 在不同身体间对齐。
- **延伸思路：** 将触觉感知整合进世界模型、将 EmbodiedWorld 与语言条件任务规划器结合，或研究世界模型用于闭环 MPC 时的安全性保证。
