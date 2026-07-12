# 3️⃣ RL × 知识库 / Reinforcement Learning × Knowledge Base

> 用强化学习训练 RAG / Agent / KG-Reasoning 系统的前沿工作。

## 概览 / Overview

| 项目 | 会议/年份 | 关键创新 | 代码 |
|---|---|---|---|
| [Graph-R1](#1-graph-r1) | ACL 2025 | End-to-end RL over 知识超图 | - |
| [KG-R1](#2-kg-r1) | arXiv 2025 | 可插拔 KG + GRPO | [Link](https://github.com/Jinyeop3110/KG-R1) |
| [MMOA-RAG](#3-mmoa-rag) | arXiv 2025 | 多 Agent 协同优化 RAG 全流程 | [Link](https://github.com/chenyiqun/MMOA-RAG) |
| [DeepResearcher](#4-deepresearcher) | EMNLP 2025 | Real-web 端到端 RL | [Link](https://github.com/GAIR-NLP/DeepResearcher) |
| [Smart-Searcher](#5-smart-searcher) | EMNLP 2025 Findings | 内/外部知识动态平衡 | [Link](https://github.com/RUCAIBox/R1-Searcher-plus) |
| [Experiential RL (ERL)](#6-experiential-rl) | Microsoft Research | 经验-反思-固化三阶段 | [Link](https://github.com/microsoft/experiential_rl) |
| [SafeOR-Gym](#7-safeor-gym) | arXiv 2025 | 9 个 OR 场景 Safe RL | [Link](https://github.com/li-group/SafeOR-Gym) |

---

## 1. Graph-R1

- **Paper**: https://www.emergentmind.com/papers/2507.21892
- **arXiv**: 2507.21892

### 解决的问题
- 传统 RAG: chunk-based 缺乏结构语义
- GraphRAG: 构建成本高、固定一次性检索、依赖长上下文

### 方法
- 轻量级知识超图构建
- 将检索建模为 **多轮 Agent-环境交互**
- 端到端奖励机制优化

### 结果
- 优于传统 GraphRAG 和 RL-enhanced RAG
- 推理准确率、检索效率、生成质量全面提升

---

## 2. KG-R1 (Efficient and Transferable Agentic KG-RAG)

- **Paper**: https://api.emergentmind.com/papers/2509.26383
- **Repo**: https://github.com/Jinyeop3110/KG-R1

### 核心创新
- **单 Agent** 与 KG 作为环境交互
- 端到端 RL 训练
- GRPO-style 目标 (per-turn + global reward)

### KG 服务器接口 (schema-agnostic)
```python
get_head_relations(entity)
get_tail_relations(entity)
get_head_entities(relation, tail)
get_tail_entities(head, relation)
```
> 1-hop 检索操作对任何 directed KG 充分

### 优势
- ✅ 使用 Qwen-2.5-3B 即可超越多模块大模型方法
- ✅ 训练后可即插即用到新 KG
- ✅ 更少 generation tokens

---

## 3. MMOA-RAG (Multi-Module joint Optimization Algorithm)

- **Paper**: https://openreview.net/forum?id=9Ia0KiVAut
- **Repo**: https://github.com/chenyiqun/MMOA-RAG

### 范式转变
- 把 RAG pipeline 视为 **多 Agent 协作任务**
- 每个组件（query rewrite / retrieval / filter / generation）= 一个 RL agent

### 联合优化
- 统一 reward: 最终答案 F1
- 避免各模块单独 SFT 时的目标不一致

### 消融
- 验证每个 agent 的贡献
- 可适配不同 RAG pipelines 和 benchmarks

---

## 4. DeepResearcher

- **Paper**: https://aclanthology.org/2025.emnlp-main.22.pdf
- **Repo**: https://github.com/GAIR-NLP/DeepResearcher

### 关键贡献
- 首个 **真实 Web 环境** 端到端 RL 训练 deep research agent
- 多 Agent 架构 (browsing agents 应对不同网页结构)

### 实验结果
- 超越 prompt engineering baseline **+28.9 分**
- 超越 RAG-based RL agents **+7.2 分**

### 涌现能力
- 规划 (Planning)
- 交叉验证 (Cross-validation)
- 自我反思 (Self-reflection)
- 找不到答案时 **保持诚实**

---

## 5. Smart-Searcher (R1-Searcher-plus)

- **Paper**: https://aclanthology.org/2025.findings-emnlp.731.pdf
- **Repo**: https://github.com/RUCAIBox/R1-Searcher-plus

### 两阶段训练
```text
Stage 1: SFT Cold-start     (格式学习)
   ↓
Stage 2: RL for Dynamic Knowledge Acquisition
   ├── outcome-supervision
   ├── internal knowledge reward
   └── memorization mechanism
```

### 创新
- 内部 + 外部知识动态平衡
- 检索次数降低 **42.9%**
- 性能超越强 baseline **+4.3%**

---

## 6. Experiential RL (ERL) by Microsoft Research

- **Repo**: https://github.com/microsoft/experiential_rl
- **Paper**: https://arxiv.org/abs/2602.13949

### 核心机制：经验-反思-固化循环
```text
Attempt → Experience
   ↓
Reflection (结构化反思)
   ↓
Consolidation (转化为持久行为)
   ↓
Next Attempt (改进)
```

### 与 LangGraph / AutoGen 集成
```python
from rllm.sdk import ErlHotpotSearchAgent
agent = ErlHotpotSearchAgent(
    base_agent="langgraph",
    env="hotpot_qa"
)
# ERL first-attempt → reflection → second-attempt
```

### 适用场景
- 已经用 LangGraph / AutoGen 搭好 agent
- 想加入 **从经验中学习** 而无需重训模型
- 长链路、难任务、需多轮反思的场景

---

## 7. SafeOR-Gym

- **Paper**: https://arxiv.org/html/2506.02255v1
- **Repo**: https://github.com/li-group/SafeOR-Gym
- **集成**: OmniSafe CMDP interface

### 9 个 OR 环境
涵盖能源、制造、供应链场景：

| 环境 | 类别 | 食品安全相关？ |
|---|---|---|
| Inventory Management | 库存 | ✅ 冷链/食品库存 |
| Energy Systems | 能源 | - |
| Manufacturing | 制造 | ✅ 食品加工排程 |
| Supply Chains | 供应链 | ✅ 食品供应链 |

### 特色
- **Cost-based 约束违反**
- **混合离散-连续动作空间**
- 与多种 Safe RL 算法兼容

### 应用
食品安全约束（HACCP 阈值、温度上限、召回触发条件）可编码为 cost-based constraint。

---

## 🎯 实战选型 / Practical Selection

```
目标
│
├─ 想训练 agent 检索 KG 回答问题 → KG-R1 (轻量、可迁移)
├─ 想优化整个 RAG pipeline → MMOA-RAG (联合优化)
├─ 想做 deep web research → DeepResearcher (real web RL)
├─ 想让 LangGraph agent 从经验学习 → ERL (post-training)
├─ 想用 GRPO over 知识图 → Graph-R1
└─ 想在食品安全 OR 场景验证 Safe RL → SafeOR-Gym
```
