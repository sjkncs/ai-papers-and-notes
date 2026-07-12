# WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory / WorldDirector：具有持久动态记忆的可控世界模拟器

> **arXiv:** [2604.02517](https://arxiv.org/abs/2604.02517) | **日期 / Date:** 2026-04-23 | **作者 / Authors:** Hanlin Wang, Hao Ouyang, Qiuyu Wang, Wen Wang, Qingyan Bai, Ka Leong Cheng, Yue Yu, Yixuan Li, Yihao Meng, Zichen Liu, Yanhong Zeng, Yujun Shen, Qifeng Chen

---

## 关键摘录 / Key Excerpts

> “现有的世界模型往往将物理动态与视觉生成纠缠在一起，从而限制了可控性。”  
> “Existing world models often entangle physical dynamics and visual generation, limiting controllability.”

> “WorldDirector 显式解耦语义运动编排与视觉生成，并引入持久动态对象记忆，实现无限制视角探索与动态实体身份保持。”  
> “WorldDirector explicitly decouples semantic motion orchestration from visual generation and introduces persistent dynamic object memory, enabling unrestricted viewpoint exploration and persistent identities for dynamic entities.”

> “通过大型语言模型管理 3D 路径与相机角度，系统能够在复杂长时长事件中维持严格的物理逻辑与外观稳定。”  
> “By using large language models to manage 3D paths and camera angles, the system maintains strict physical logic and appearance stability throughout complex long-duration events.”

> “项目主页展示了在遮挡与重入场景下仍保持对象身份一致的视频结果。”  
> “The project page demonstrates video results that preserve object identity consistently through occlusion and re-entry scenarios.”

---

## Q1: 核心问题 / Core Problem

**中文**  
生成式世界模型通常能合成视觉上逼真的短视频片段，但难以在长时间、多事件、多视角的场景中保持一致的对象身份和物理逻辑。当动态物体被遮挡、离开视野再返回时，传统方法往往出现外观漂移或身份丢失。WorldDirector 要解决的核心问题是：如何构建一个既可控又能在长时间跨度内保持动态对象身份一致的世界模拟器？

**English**  
Generative world models can synthesize visually plausible short video clips, but struggle to maintain consistent object identity and physical logic across long, multi-event, multi-view scenarios. When dynamic objects are occluded, leave the frame, and later return, traditional methods often suffer from appearance drift or identity loss. The core problem WorldDirector tackles is: how to build a controllable world simulator that preserves dynamic object identity over long horizons?

---

## Q2: 核心贡献 / Claimed Contributions

1. **持久动态对象记忆 / Persistent dynamic object memory**  
   为场景中的动态实体维护长期记忆，使其在离开视野后仍能被准确恢复。  
   Maintains long-term memory for dynamic entities so they can be accurately restored after leaving the field of view.

2. **语义运动编排与视觉生成解耦 / Decoupled semantic motion orchestration and visual generation**  
   将“物体如何运动”与“画面如何渲染”分开，便于独立控制。  
   Separates how objects move from how frames are rendered, enabling independent control.

3. **LLM 驱动的 3D 路径与相机控制 / LLM-driven 3D path and camera control**  
   利用大型语言模型规划复杂的 3D 轨迹和相机运动，实现高层语义控制。  
   Uses large language models to plan complex 3D trajectories and camera motions for high-level semantic control.

4. **严格物理逻辑与外观稳定性 / Strict physical logic and appearance stability**  
   在长时间复杂事件中保持物理一致性和视觉外观稳定。  
   Maintains physical consistency and visual stability throughout long and complex events.

5. **项目页面与视频演示 / Project page and video demonstrations**  
   提供了项目主页，展示遮挡、重入和多视角探索等挑战场景下的生成结果。  
   Provides a project page with generated results in challenging scenarios such as occlusion, re-entry, and multi-view exploration.

---

## Q3: 审稿攻击点 / Reviewer Attack Points

1. **对 LLM 规划器的依赖 / Reliance on an LLM planner**  
   运动编排由 LLM 完成，规划错误会沿时间传播，导致后续帧出现物理不一致。  
   Motion orchestration relies on an LLM, and planning errors can propagate over time, causing physical inconsistencies in later frames.

2. **定量评估不足 / Insufficient quantitative evaluation**  
   身份保持、物理一致性等多依赖主观视频质量或用户研究，缺乏标准化指标。  
   Identity preservation and physical consistency may rely on subjective video quality or user studies rather than standardized metrics.

3. **模型规模与训练成本 / Model scale and training cost**  
   解耦架构可能引入额外模块，训练与推理成本未详细披露。  
   The decoupled architecture may introduce additional modules; training and inference costs are not detailed.

4. **开放世界泛化能力 / Open-world generalization**  
   演示场景可能集中在有限类别上，能否泛化到任意新物体与复杂交互未知。  
   Demonstrations may focus on limited categories; generalization to arbitrary novel objects and complex interactions is unclear.

5. **与现有视频生成模型的对比 / Comparison with existing video generators**  
   未与 Sora、Runway Gen-4 等工业级模型在相同提示下做系统对比。  
   Does not systematically compare with industrial models such as Sora or Runway Gen-4 under identical prompts.

---

## Q4: 博士生阅读指南 / PhD Reading Guide

**中文**  
建议先观看项目主页视频，建立对“持久动态记忆”的直观理解；然后阅读方法部分，重点跟踪对象记忆如何编码、更新与检索。若无法复现完整视频生成 pipeline，可从简化任务入手：在一个 2D 网格世界中实现“对象被遮挡后仍保持身份”的模拟。可延伸方向包括：将 WorldDirector 的记忆机制迁移到机器人仿真器、与 NeRF/3DGS 结合实现可编辑 3D 场景、以及设计面向长视频一致性的定量基准。

**English**  
Start by watching the videos on the project page to build intuition for persistent dynamic memory. Then read the methodology, focusing on how object memory is encoded, updated, and retrieved. If the full video-generation pipeline is too costly to reproduce, begin with a simplified task: implement a 2D grid world where objects retain identity after occlusion. Possible extensions include migrating WorldDirector's memory mechanism to robot simulators, combining it with NeRF/3DGS for editable 3D scenes, and designing quantitative benchmarks for long-video consistency.
