# 📅 2026 顶会周报：食品安全 × AI Agent / 2026 Top-Conference Weekly Report

> 2026 年食品安全 × AI Agent 顶会精选论文摘要
> Curated 2026 papers on Food-Safety × AI Agent from top conferences (NeurIPS / ICLR / ICML / SemEval / EMNLP / ACL)

---

## 📊 趋势概览 / Trend Overview

### 2025-2026 三大趋势

1. **RL × Agent 全面融合**
   - 从 prompt engineering → 端到端 RL 训练 agent
   - GRPO / XPO / Flow-GRPO 成为主流算法
   - 代表：DeepResearcher / Smart-Searcher / Graph-R1 / KG-R1 / AgentFlow

2. **食品安全 LLM Safety 觉醒**
   - FoodGuardBench (3,339 FDA-grounded queries)
   - FoodGuard-4B guardrail 模型
   - FoodSafeSum / KAD-FoodHazard 等领域数据集涌现

3. **Agent 工作流自动化**
   - AFlow (ICLR 2025 Oral) — MCTS 自动发现 workflow
   - DesignX (NeurIPS 2025) — 双 Agent RL 算法设计
   - OptiMindTune — 多 Agent HPO

---

## 🎯 按会议分类 / By Conference

### NeurIPS 2025

| 论文 | 主题 | 与本知识库关系 |
|---|---|---|
| [DesignX](https://arxiv.org/pdf/2505.17866) | Dual-Agent RL Algorithm Design | 调优 / 算法设计 |
| [AGENTFLOW / Flow-GRPO](https://openreview.net/pdf?id=ddabb524f1439770dfe2d5bad51cf9a57d2a6ecd.pdf) | In-the-Flow Agentic System Optimization | Agent RL 训练 |
| [Trajectory Bellman Residual Minimization](https://mdp.sh/) | Value-based LLM Reasoning | RL 算法 |
| [Outcome-Based Online RL](https://mdp.sh/) | 算法 + 理论界限 | RL 算法 |

### ICLR 2025

| 论文 | 主题 | 与本知识库关系 |
|---|---|---|
| [AFlow](https://arxiv.org/abs/2410.10762) (Oral) | MCTS Agentic Workflow Optimization | 工作流自动发现 |
| [XPO](https://openreview.net/pdf?id=QYigQ6gXNw) | Exploratory Preference Optimization | RLHF / 在线探索 |

### ICML 2025

| 论文 | 主题 | 与本知识库关系 |
|---|---|---|
| [Reinforce LLM Reasoning with Multi-Agent Reflection](https://mdp.sh/) | Multi-Agent Reasoning | 多 Agent RL |

### SemEval 2025

| 论文 | 主题 | 与本知识库关系 |
|---|---|---|
| [Task 9: Food Hazard Detection](https://aclanthology.org/2025.semeval-1.325/) | 食品危害检测 benchmark | 食品安全核心 |

### EMNLP 2025

| 论文 | 主题 | 与本知识库关系 |
|---|---|---|
| [DeepResearcher](https://aclanthology.org/2025.emnlp-main.22.pdf) | Real-web Deep Research RL | Agent + RL |
| [Smart-Searcher](https://aclanthology.org/2025.findings-emnlp.731.pdf) | Dynamic Knowledge Acquisition RL | Agent + RL |
| [FoodSafeSum](https://aclanthology.org/2025.findings-emnlp.911.pdf) | 食品安全文档摘要数据集 | 食品安全数据集 |

### ACL 2025

| 论文 | 主题 | 与本知识库关系 |
|---|---|---|
| [Graph-R1](https://www.emergentmind.com/papers/2507.21892) | Agentic GraphRAG via RL | RAG + RL |

---

## 📈 食品安全 LLM/Agent Timeline

```text
2024 ────────────────────────────
  │
  ├─ Food Hazard Detection 数据集 (Randl et al. 2024)
  │
2025 ────────────────────────────
  │
  ├─ SemEval-2025 Task 9 (食品危害检测 challenge)
  │   ├─ KAD-FoodHazard (Top-2)
  │   └─ LLM 合成数据对长尾分布有效
  │
  ├─ FoodSafeSum (EMNLP Findings) — 数据集
  │
  ├─ AFrexAI HACCP Skill (1933★)
  │
  └─ LangGraph 生态成熟 (35k★)
  │
2026 ────────────────────────────
  │
  └─ FoodGuardBench + FoodGuard-4B
      └─ 首个 FDA-grounded 3339 queries LLM safety benchmark
```

---

## 🧠 Agent × RL Timeline

```text
2024 ────────────────────────────
  │
  ├─ AFlow (Oct 2024) — 早期 MCTS workflow 优化
  │
2025 ────────────────────────────
  │
  ├─ ICLR 2025: AFlow Oral / XPO
  ├─ ICML 2025: Multi-Agent Reflection
  ├─ NeurIPS 2025: DesignX / AgentFlow
  ├─ EMNLP 2025: DeepResearcher / Smart-Searcher
  └─ ACL 2025: Graph-R1
  │
2026 ────────────────────────────
  │
  └─ KG-R1 (高效可迁移) / MMOA-RAG (联合优化)
```

---

## 🎓 跨领域交叉点 / Cross-Domain Insights

### RL × RAG (3 大趋势)
1. **End-to-End RL** (DeepResearcher, KG-R1, Graph-R1)
2. **Multi-Agent Cooperative** (MMOA-RAG, Multi-Agent Reflection)
3. **Exploration + Memorization** (Smart-Searcher, XPO)

### Workflow × MCTS (2 大流派)
1. **MCTS over Code** (AFlow)
2. **Dual-Agent RL** (DesignX)

### Food Safety × LLM Safety (3 个新方向)
1. **Benchmarks**: FoodGuardBench, SemEval-2025 Task 9
2. **Datasets**: FoodSafeSum, KAD-FoodHazard
3. **Guardrails**: FoodGuard-4B, afrexai-food-safety Skill

---

## 📝 论文速读建议 / Reading Recommendations

### 必读 (Foundational)

| 论文 | 一句话总结 |
|---|---|
| **AFlow** | MCTS 在 code-represented workflow 空间搜索 |
| **DesignX** | 双 Agent RL 联合学习工作流结构 + 超参 |
| **DeepResearcher** | Real-web RL 涌现规划/反思/诚实能力 |
| **KG-R1** | 单 Agent + GRPO + 可插拔 KG |

### 选读 (Domain-Specific)

| 论文 | 适用场景 |
|---|---|
| FoodGuardBench | LLM 食品安全对齐 |
| SemEval-2025 Task 9 | 食品危害分类 |
| OptiMindTune | 业务侧多 Agent 调优 |
| ERL | 已搭好 LangGraph 想增强学习 |

### 跟踪 (Watch List)

- AgentFlow 系列（NeurIPS Workshop on Efficient Reasoning）
- MMOA-RAG 系列（多模块联合 RL）
- ERL 系列（Microsoft Research）
- FoodGuard 后续工作（食品领域 guardrail）

---

*Last Updated: 2026-06-22*
*Coverage: NeurIPS 2025, ICLR 2025, ICML 2025, ACL 2025, EMNLP 2025, SemEval 2025, arXiv 2025-2026*
