# Proactive Memory for Event-Driven LLM Agents

**arXiv:** [2610.36967](https://arxiv.org/abs/2610.36967)
**Authors:** Jinyang Li, Zichuan Fu, Xian Wu, Wenlin Zhang, Yimin Deng, Kaifeng Guo, Guojing Li, Junjia QI, Xiangyu Zhao
**Categories:** cs.AI

---

## Q1: 它真正想解决的问题是什么？ / What problem does it really solve?

**核心问题：** 主动型 LLM Agent 如何从持续积累的历史中识别和利用重复出现的行为规律，以预判当前事件中的服务机会？

**深层动机：** 现有 Agent 记忆系统要么依赖近期上下文，要么基于语义相似性检索相关记忆，但都缺乏对"历史规律在当前是否仍然适用"的验证能力。一个真正主动的 Agent 需要能够：(1) 从历史中提炼行为规律，(2) 在遇到新事件时验证规律的可适用性，(3) 据此采取预防性或前瞻性行动。

---

## Q2: 它声称的贡献是什么？ / What are the claimed contributions?

1. **ProactMem 三层记忆架构：**
   - 近期事件层（recent events）
   - 时间轨迹层（temporal trajectories）：可从源事件验证
   - 行为规律层（behavioral regularities）：通过历史回放评估适用性
2. **ProactiveStream 基准：** 纵向基准，覆盖三个应用范围，Agent 按时间顺序处理事件流
3. **实验结果：** F1 = 52.6%，比五个外部基线中最强者提升 21.4%（相对）

---

## Q3: 最可能被 reviewer 攻击的地方在哪里？ / Most likely reviewer attack points?

1. **基准构造的生态效度：** ProactiveStream 的事件流是否反映了真实应用场景的复杂性和噪声水平？
2. **行为规律的泛化性：** 从有限历史中提炼的规律是否会过拟合特定场景？在分布外事件上的鲁棒性如何？
3. **F1 = 52.6% 的绝对水平：** 虽然相对提升显著，但绝对值不到 53%，实用性可能被质疑
4. **计算开销：** 历史回放在大规模记忆库上的延迟和成本未充分讨论
5. **与 TrawMem 的区别：** 同期出现的多篇记忆架构论文，需要清晰区分贡献边界

---

## Q4: 同方向博士生应精读哪些、跳过哪些？ / PhD reading guide

**精读：**
- 三层记忆的组织方式和验证机制——这是核心创新
- ProactiveStream 基准的设计思路——对评估主动型 Agent 有参考价值
- "历史回放"验证规律适用性的具体实现

**跳读：**
- 与 SimpleMem 等基线的详细数值对比表（关注趋势而非具体数字）

**记忆架构研究对比：** 与同期的 TrawMem (2610.36918) 对比——ProactMem 关注"规律发现与验证"（前瞻性），TrawMem 关注"任务角色分配"（组织性），两者互补。建议精读一篇后略读另一篇的方法部分。

---

## 与本研究方向的关联

直接相关：Agent/Skills 体系中的长期记忆管理。主动记忆机制可以帮助 Agent 预判何时需要调用特定 Skill，而非被动响应用户指令。
