# Concordat: Typed Sessions for LLM Agents

**arXiv:** [2610.37461](https://arxiv.org/abs/2610.37461)
**Authors:** Milad Sikaroudi
**Categories:** cs.AI, cs.MA

---

## Q1: 它真正想解决的问题是什么？ / What problem does it really solve?

**核心问题：** LLM Agent 在执行有后果的操作（取消预订、退款、修改账户）时，如何保证运行时行为符合预定义的业务规则——尤其是那些"必须获得用户确认才能执行"类的安全约束？

**深层动机：** 当前 Agent 的规则写在 prompt 里，依赖模型的"自觉遵守"。但实验显示，标准 Agent 在 gpt-4.1 上有 13/50 个任务、Claude Haiku 上有 29/128 个任务执行了未经确认的操作。更令人担忧的是，tau-bench 的自带奖励机制也没检测到大多数违规。

---

## Q2: 它声称的贡献是什么？ / What are the claimed contributions?

1. **将 Agent 建模为有限状态协议对象：** 借鉴 typestate 和 session types，每个 Agent 的接口是一个有限状态机
2. **运行时强制机制：**
   - 每轮只提供当前状态合法的工具集
   - 非法调用在产生副作用前被拒绝
   - 人类保留的决策只能被"请求"而非"执行"
   - Token 和工具调用预算在手off间线性分配
3. **Liskov 替换检查：** 子类 Agent 不能绕过父类协议
4. **零违规实验结果：** 在 tau-bench 的 50 个航空任务上，Concordat Agent 完全消除了未经确认的操作

---

## Q3: 最可能被 reviewer 攻击的地方在哪里？ / Most likely reviewer attack points?

1. **表达能力限制：** 有限状态协议能否覆盖所有现实业务规则？对于"如果客户是 VIP 且投诉过 3 次以上则可以自动退款"这类条件分支规则，状态机是否会爆炸？
2. **性能开销：** 每轮的合法性检查和工具过滤对延迟的影响未量化
3. **协议设计负担：** 用户需要手动编写有限状态协议，这对非专业开发者的门槛较高
4. **tau-bench 局限性：** 仅在一个基准（航空客服）上验证，跨领域泛化性未证明
5. **单人工作：** 单一作者的工程完整性和可复现性可能被质疑

---

## Q4: 同方向博士生应精读哪些、跳过哪些？ / PhD reading guide

**精读：**
- Section 3 (Protocol Design)：如何将业务规则映射到有限状态协议——这是核心思想
- Section 5 (Evaluation)：特别是 tau-bench 自身奖励未检测到违规的发现，对 Agent 评估领域有重要启示
- Liskov 替换原则在 Agent 继承中的应用

**跳读：**
- Python 实现细节（直接看开源代码更好）

**对 Agent 安全研究的启示：** 本文证明了 prompt-level 的安全约束不可靠，需要 structure-level 的强制机制。这与 AI 对齐中的"能力控制"思路一致。

---

## 与本研究方向的关联

直接相关：Agent/Skills 体系中，Skill 的调用顺序和前置条件天然适合用会话类型建模。Concordat 的方法可以直接用于 Skill 编排的运行时安全保障。
