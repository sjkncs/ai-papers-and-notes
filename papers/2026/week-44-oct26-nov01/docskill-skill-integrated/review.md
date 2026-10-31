# DocSkill: Skill-Integrated Agentic Reasoning for Multilingual Multi-Hop Document Question Answering

**arXiv:** [2610.36759](https://arxiv.org/abs/2610.36759)
**Authors:** Jing Huang, Zhiya Tan, Changtao Miao, Shutao Gong, Yang Wang, Joey Tianyi Zhou, Jianshu Li, Ying Yan
**Categories:** cs.CL, cs.CV

---

## Q1: 它真正想解决的问题是什么？ / What problem does it really solve?

**核心问题：** 多语言多跳文档问答中，如何让 Agent 学会自适应地选择和编排技能（skill）——知道何时该检索、何时该跨页整合、何时该验证答案？

**深层动机：** 现有多跳文档 QA 方法要么单次处理所有文档（效率低），要么自适应使用工具但缺乏有效的技能编排策略。DocSkill 的核心洞察是：技能编排本身是一个可学习的策略，可以通过 MCTS 探索 + 偏好优化来训练。

---

## Q2: 它声称的贡献是什么？ / What are the claimed contributions?

1. **技能编排策略学习：** 有界 MCTS 离线探索技能轨迹 → SFT + DPO 训练编排策略
2. **MMHop-DocBench：** 10 种语言、8 个文档领域、60K QA 对的多语言多跳基准
3. **HopGen 流水线：** 基于层次化页面-元素图构建可追溯的跨页证据路径
4. **实验结果：** Qwen3-VL-8B + DocSkill 在 MMHop-DocBench 上准确率达 41.0、F1 达 49.2，超越最强基线 5.5/6.3 个点

---

## Q3: 最可能被 reviewer 攻击的地方在哪里？ / Most likely reviewer attack points?

1. **MCTS 搜索的效率和可扩展性：** 有界 MCTS 的搜索深度和宽度限制如何影响技能轨迹质量？
2. **技能定义的粒度：** 技能的划分方式是否影响最终性能？不同技能粒度下的鲁棒性如何？
3. **多语言覆盖的深度：** 10 种语言中是否包含低资源语言？在低资源语言上的表现是否与高资源语言一致？
4. **DPO 偏好对的构造：** MCTS 轨迹到偏好对的转换方式是否引入偏差？

---

## Q4: 同方向博士生应精读哪些、跳过哪些？ / PhD reading guide

**精读：**
- 技能编排策略的 MCTS 探索和训练流程——核心方法论
- HopGen 流水线的证据路径构造——对构建可追溯 QA 系统有参考价值
- 技能选择和跳过的分析——理解 Agent 何时决定"不需要更多信息"

**跳读：**
- 各语言各领域的详细性能分解表

**对 Agent/Skills 的启示：** MCTS 探索技能轨迹 → DPO 训练编排策略，这是 Skill 自动化编排的一条可行路径。

---

## 与本研究方向的关联

直接相关：Agent/Skills 体系的核心问题之一就是技能编排。DocSkill 提供了一个"学习何时使用哪个 Skill"的完整范式，从 MCTS 探索到 DPO 训练的策略学习流程可迁移到其他 Skill 编排场景。
