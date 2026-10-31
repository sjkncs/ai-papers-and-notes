# SceneEvolve: Learning to Code the World — Harness Evolution for Compositional 3D Reconstruction

**arXiv:** [2610.37046](https://arxiv.org/abs/2610.37046)
**Authors:** Haoyu Yao, Chenguo Lin, Ruize Du, Yadong Mu
**Categories:** cs.CV

---

## Q1: 它真正想解决的问题是什么？ / What problem does it really solve?

**核心问题：** 如何从单张图像重建可编辑、可组合的 3D 场景——不仅恢复几何，还要理解物体结构和空间关系？

**深层动机：** 现有 3D 场景重建方法要么生成不可编辑的隐式表示（如 NeRF/3DGS），要么依赖大型闭源模型（如 GPT-5.6）做代码化重建但成本高昂。SceneEvolve 提出了一条新路径：用小模型 + 可进化的"脚手架"（harness）来实现大模型级别的重建质量。

---

## Q2: 它声称的贡献是什么？ / What are the claimed contributions?

1. **SceneAct：** 组合式语义动作接口，基于参数化原语、层次组合和语义空间关系
2. **SceneHarness-RSI：** 利用教师 Agent 迭代进化脚手架，小模型在部署时无需教师辅助
3. **惊人的对比结果：**
   - Qwen3.8-27B + 进化脚手架 vs GPT-5.6-Sol：重建质量高 32.3%，部署成本低 59.8%
   - 推理时间减少 47.0%
   - 脚手架可迁移到其他模型（Kimi-K3），达到 GPT-6-Astra 水平

---

## Q3: 最可能被 reviewer 攻击的地方在哪里？ / Most likely reviewer attack points?

1. **脚手架的通用性：** 在特定开发集上进化的脚手架是否能泛化到全新的场景类型？
2. **教师-学生框架的假设：** 教师 Agent 的质量直接决定脚手架上限，如果教师本身在某些场景上表现差怎么办？
3. **可编辑性的量化评估：** "可编辑"是核心卖点之一，但如何客观评估编辑的易用性和正确性？
4. **代码化表示的局限：** 将场景表示为代码是否限制了连续几何的表达精度？

---

## Q4: 同方向博士生应精读哪些、跳过哪些？ / PhD reading guide

**精读：**
- SceneAct 的动作接口设计——如何定义参数化原语和组合操作
- 脚手架进化过程——理解"小模型如何超越大模型"的关键机制
- 成本对比分析——对实际部署决策有直接参考价值

**跳读：**
- 具体的重建指标数值表（关注相对提升幅度）

**对 Agent 研究的启示：** "进化脚手架"的思路——用强模型生成可复用的工作流程，让小模型执行——与 Agent/Skills 体系的理念高度一致。

---

## 与本研究方向的关联

双重相关：
- **3D 视觉：** 提供了一种代码化的组合式 3D 场景重建方法
- **Agent/Skills：** 脚手架进化思想可迁移到 Skill 编排——用强 Agent 生成 Skill 模板，弱 Agent 执行
