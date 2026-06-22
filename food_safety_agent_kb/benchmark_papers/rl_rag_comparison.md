# Graph-R1, KG-R1, MMOA-RAG: RL × RAG 前沿论文对比

> 深度对比：2025 年三篇代表性的 **用强化学习优化 RAG / KG-RAG** 工作
> Side-by-side analysis: 3 representative works on RL-optimized RAG / KG-RAG in 2025

---

## 一、总览 / Overview

| 维度 | Graph-R1 | KG-R1 | MMOA-RAG |
|---|---|---|---|
| **会议** | ACL 2025 | arXiv 2025 | arXiv 2025 |
| **知识结构** | 知识超图 (hypergraph) | 知识图谱 (KG) | 文档集合 (vector DB) |
| **Agent 数量** | 单 Agent | 单 Agent | 多 Agent (4 个) |
| **RL 算法** | End-to-end RL | GRPO | Multi-Agent MARL |
| **关键创新** | 轻量超图 + 多轮交互 | Schema-agnostic 1-hop 接口 | 联合优化全 pipeline |
| **可迁移性** | 中 | 高（plug-and-play） | 低（pipeline-specific） |
| **效率** | 中 | 高（小模型 + 少 token） | 低（多模块） |

---

## 二、Graph-R1: Agentic GraphRAG via End-to-End RL

### 2.1 问题

传统 RAG 的痛点：
- **Chunk-based 检索** 缺乏结构语义
- **GraphRAG** 构建成本高
- **固定一次性检索** 不灵活
- 依赖长上下文 + prompt 设计

### 2.2 方法

```text
┌─────────────────────────────────────────────────┐
│  Graph-R1                                      │
├─────────────────────────────────────────────────┤
│  1. 轻量级知识超图构建                           │
│     - 实体 + 关系 → 超图                        │
│     - 比传统 GraphRAG 节省 80% 构建成本          │
│                                                 │
│  2. 多轮 Agent-环境交互                         │
│     - Agent 提出 query                          │
│     - 超图环境返回相关子图                       │
│     - Agent 决定下一步                           │
│                                                 │
│  3. 端到端 RL 奖励                              │
│     - 最终答案准确性作为 reward                  │
│     - 反向传播优化策略                           │
└─────────────────────────────────────────────────┘
```

### 2.3 优势

- ✅ 显式建模知识结构
- ✅ 自适应多轮检索
- ✅ 端到端训练

### 2.4 适用场景
- 关系密集的知识（医疗、食品安全法规图谱）
- 需要推理链条的任务

---

## 三、KG-R1: Efficient and Transferable Agentic KG-RAG

### 3.1 问题

现有 KG-RAG 的痛点：
- 多 LLM 模块（planning / reasoning / responding）
- 推理成本高
- 绑定到特定 KG schema

### 3.2 方法

**核心**: 用 **单 Agent + KG server**，端到端 RL 训练

```text
┌─────────────────────────────────────────────────┐
│  KG-R1                                         │
├─────────────────────────────────────────────────┤
│  ┌──────────┐         ┌──────────────────┐      │
│  │  Agent    │ ←────→ │  KG Server       │      │
│  │  (Qwen    │  1-hop │  (Schema-agnostic)│      │
│  │   2.5-3B) │ queries│                  │      │
│  └──────────┘         └──────────────────┘      │
│       ↑                              ↑           │
│       └────── End-to-End RL ─────────┘           │
└─────────────────────────────────────────────────┘
```

### 3.3 Schema-Agnostic KG Server

```python
# 4 个 1-hop 操作 — 对任何 directed KG 充分
get_head_relations(entity)       # 某实体的所有关系 (head→?)
get_tail_relations(entity)       # 某实体的所有关系 (?→entity)
get_head_entities(relation, tail) # 找到 (?, relation, tail)
get_tail_entities(head, relation) # 找到 (head, relation, ?)
```

**数学保证**: 任何 directed KG 的推理路径都可以通过这些操作遍历。

### 3.4 GRPO-Style 训练

```python
reward = per_turn_reward + global_reward
loss = GRPO(policy, trajectories, reward)
```

### 3.5 三大优势

| 优势 | 说明 |
|---|---|
| **效率** | Qwen-2.5-3B 即可超越多模块大模型 |
| **可迁移** | 训练后即插即用到任何 KG |
| **少 token** | 比多模块 workflow 节省 50%+ tokens |

### 3.6 适用场景
- 已有知识图谱（企业 KG）
- 关心推理成本
- 想跨 KG 部署

---

## 四、MMOA-RAG: Multi-Module joint Optimization Algorithm

### 4.1 问题

现有 RAG 训练的痛点：
- 各模块 (query rewrite / retrieval / filter / generation) **分开 SFT**
- 目标不一致 → 整体效果次优
- 简单两模块 RL 忽略复杂依赖

### 4.2 方法：把 RAG Pipeline 视为多 Agent 协作

```text
┌──────────────────────────────────────────────────┐
│  MMOA-RAG — Multi-Agent Cooperative RAG          │
├──────────────────────────────────────────────────┤
│                                                  │
│  Agent 1: Query Rewriter                         │
│     ↕                                            │
│  Agent 2: Retriever                              │
│     ↕                                            │
│  Agent 3: Filter                                 │
│     ↕                                            │
│  Agent 4: Generator                              │
│     ↕                                            │
│  Joint Reward: 答案 F1                            │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 4.3 联合优化

```python
# 每个 agent 独立 actor-critic
for agent in [rewriter, retriever, filter, generator]:
    agent.policy = PolicyNetwork(state_dim, action_dim)
    agent.critic = ValueNetwork(state_dim)

# 共享 reward
reward = compute_f1(final_answer, ground_truth)

# 联合更新
for agent in agents:
    agent.update(trajectory, reward)
```

### 4.4 关键特性

| 特性 | 说明 |
|---|---|
| **统一 reward** | 最终答案 F1，所有 agent 共享 |
| **避免目标不一致** | 各 agent 不会各自优化错的目标 |
| **可适配** | 换不同 RAG pipeline 只需重新训练 |
| **消融清晰** | 每个 agent 贡献可量化 |

### 4.5 适用场景
- 复杂多模块 RAG 系统
- 已有 baseline 想联合优化
- 想量化每个模块贡献

---

## 五、对比矩阵 / Comparison Matrix

### 5.1 任务维度

| 维度 | Graph-R1 | KG-R1 | MMOA-RAG |
|---|---|---|---|
| **知识结构** | 知识超图 | 知识图谱 | 文档 |
| **构建成本** | 低 (轻量) | 中 (需 KG) | 极低 (vector DB) |
| **推理效率** | 中 | 高 | 低 (多模块) |
| **可迁移性** | 中 | 高 | 低 |
| **调优难度** | 高 | 中 | 中 |
| **冷启动** | 中 | 易 (plug-and-play) | 易 |

### 5.2 适用场景

| 场景 | 推荐 |
|---|---|
| 食品安全法规图谱推理 | **KG-R1** (有 KG 即插即用) |
| 食品安全知识超图 | **Graph-R1** (关系密集) |
| 多模块 RAG pipeline 优化 | **MMOA-RAG** (联合训练) |
| 资源受限 | **KG-R1** (小模型 + 少 token) |
| 探索性研究 | **Graph-R1** (新颖超图) |

---

## 六、对食品安全的应用启示

### 6.1 食品安全法规知识图谱

构建：
```text
Entity: 21 CFR 117 (FSMA)
  ├─ requires: HACCP plan
  ├─ applies_to: food facility
  ├─ citation_of: regulation 117.80
  └─ related_to: 21 CFR 123 (Seafood HACCP)
```

使用 **KG-R1**:
- Agent 接收 query: "冷链食品需要什么计划？"
- 1-hop KG 检索: 找到 21 CFR 117
- LLM 推理回答

### 6.2 食品危害超图

```text
超图节点 (实体): food, hazard, regulation, product, symptom
超图边 (高阶关系): 
  [food, contains, hazard, triggers, symptom]
  [food, regulated_by, regulation, monitored_by, agency]
```

使用 **Graph-R1**:
- 多跳推理
- 显式建模高阶关系

### 6.3 食品安全事件 RAG 系统

```text
事件文档 (vector DB)
   ↓
Multi-Agent RAG
├─ Agent 1: Hazard Classifier (PubMedBERT)
├─ Agent 2: Severity Estimator
├─ Agent 3: Affected Product Extractor
└─ Agent 4: Regulatory Mapper (RAG)
```

使用 **MMOA-RAG**:
- 联合优化 4 个 agent
- 共享最终答案 F1 reward

---

## 七、复现路线 / Reproduction

### 7.1 Graph-R1

```bash
# 论文主页 + 代码 (见 arXiv)
git clone <graph_r1_repo>
cd graph_r1
python train.py --dataset 2wikimultihopqa
```

### 7.2 KG-R1

```bash
git clone https://github.com/Jinyeop3110/KG-R1
cd KG-R1
pip install -r requirements.txt

# 训练
python train.py \
  --model Qwen/Qwen2.5-3B \
  --kg_dataset WebQSP \
  --algorithm GRPO

# 评估
python evaluate.py --model_path checkpoints/kg_r1
```

### 7.3 MMOA-RAG

```bash
git clone https://github.com/chenyiqun/MMOA-RAG
cd MMOA-RAG
pip install -r requirements.txt

# 训练多 agent
python train_marl.py \
  --pipeline modular_rag \
  --reward_metric f1 \
  --n_episodes 10000
```

---

## 八、未来趋势 / Future Trends

### 8.1 三大方向

1. **End-to-End RL** 成为主流（DeepResearcher, KG-R1, Graph-R1）
2. **Multi-Agent Cooperative** 解决复杂 pipeline（MMOA-RAG）
3. **小模型 + 强 workflow** 反超大模型（KG-R1 + DesignX）

### 8.2 食品安全应用展望

- 法规知识图谱 + KG-R1 → 智能法规问答
- 危害超图 + Graph-R1 → 危害推理链
- 多 Agent RAG + MMOA-RAG → 召回决策自动化

---

## 九、一句话总结

> **Graph-R1 / KG-R1 / MMOA-RAG 代表了 2025 年 RL × RAG 三大方向：End-to-End RL、可迁移的 plug-and-play、多 Agent 协同优化。食品安全的知识密集型场景天然适合这些技术。**

---

### 参考文献 / References

1. **Graph-R1**: arXiv 2507.21892 — "Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning"
2. **KG-R1**: arXiv 2509.26383 — "Efficient and Transferable Agentic Knowledge Graph RAG via Reinforcement Learning"
3. **MMOA-RAG**: OpenReview [9Ia0KiVAut](https://openreview.net/forum?id=9Ia0KiVAut) — "Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning"
