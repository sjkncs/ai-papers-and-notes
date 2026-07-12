# Food Safety AI Agent Knowledge Base | 食品安全 AI Agent 知识库

> **聚焦食品安全的 AI Agent 调优参数设计、知识库、强化学习、agent 搭建与工作流的高星仓库与顶会论文**
>
> Curated high-star GitHub repositories and top-conference papers for **food-safety–oriented AI agents**: tuning/parameter design, knowledge base construction, reinforcement learning, agent construction, and workflow orchestration. Bilingual CN/EN.

## 📋 主题概览 / Topic Index

| # | 主题 / Topic | 路径 / Path | 说明 / Description |
|---|---|---|---|
| 1 | **食品安全 Agent 框架 / Food-Safety Agent Frameworks** | [`agents/`](./agents/) | SafeGuard AI、SFPPy、FoodSafe、KAD-FoodHazard 等 |
| 2 | **调优与参数设计 / Tuning & Parameter Design** | [`optimization/`](./optimization/) | AutoRAG、OptiMindTune、Metaflow-Optuna、HPO for RAG |
| 3 | **RL × RAG / 知识库 × 强化学习** | [`rl_rag/`](./rl_rag/) | Graph-R1、KG-R1、MMOA-RAG、DeepResearcher、Smart-Searcher |
| 4 | **工作流与多 Agent 编排 / Workflow & Multi-Agent Orchestration** | [`workflow/`](./workflow/) | LangGraph、AFlow、AFlow、AutoGen、ChatFood、WhatsEat |
| 5 | **顶会论文 / Top-Conference Papers** | [`benchmark_papers/`](./benchmark_papers/) | NeurIPS/ICLR/ICML/SemEval/EMNLP 2024-2026 |
| 6 | **会议周报 / Conference Reports** | [`top-conferences/`](./top-conferences/) | 周报形式的精选论文摘要 |

---

## 🏆 高星仓库速查表 / High-Star Repository Cheat Sheet

| ⭐ | 仓库 / Repo | 简介 / Description | 链接 |
|---:|---|---|---|
| 35k+ | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | 主流 stateful agent 工作流编排框架 | [link](https://github.com/langchain-ai/langgraph) |
| 4.8k | [Marker-Inc-Korea/AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) | RAG 流水线 AutoML 自动调优 | [link](https://github.com/Marker-Inc-Korea/AutoRAG) |
| 796 | [chauncygu/Safe-Reinforcement-Learning-Baselines](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baselines) | 安全 RL 基线（含 supply chain inventory） | [link](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baselines) |
| 579 | [xuhongzuo/DeepOD](https://github.com/xuhongzuo/deepod) | 27 个时序/表格异常检测模型（含 Transformer） | [link](https://github.com/xuhongzuo/deepod) |
| 536 | [FoundationAgents/AFlow](https://github.com/FoundationAgents/AFLOW) | ICLR 2025 Oral：MCTS 自动优化 Agent 工作流 | [link](https://github.com/FoundationAgents/AFLOW) |
| 1.9k★ | [afrexai-cto/afrexai-food-safety](https://afrexai-cto.github.io/context-packs/) | 食品行业 HACCP 合规 AI Agent Skill | [link](https://level8.bg/tools/skills/afrexai-food-safety) |

---

## 🔬 学术会议高引论文 / Top Conference Papers (2024-2026)

| 会议 | 论文 | 链接 |
|---|---|---|
| **NeurIPS 2025** | DesignX: Human-Competitive Algorithm Designer for Black-Box Optimization | [arXiv](https://arxiv.org/pdf/2505.17866) / [Code](https://github.com/MetaEvo/DesignX) |
| **NeurIPS 2025** | AGENTFLOW: In-the-Flow Agentic System Optimization | [OpenReview](https://openreview.net/pdf?id=ddabb524f1439770dfe2d5bad51cf9a57d2a6ecd.pdf) |
| **ICLR 2025 Oral** | AFlow: Automating Agentic Workflow Generation | [GitHub](https://github.com/FoundationAgents/AFLOW) |
| **ICLR 2025** | XPO: Exploratory Preference Optimization for RLHF | [OpenReview](https://openreview.net/pdf?id=QYigQ6gXNw) |
| **ICML 2025** | Reinforce LLM Reasoning with Multi-Agent Reflection | [PDF](https://mdp.sh/) |
| **ACL/EMNLP 2025** | Graph-R1: Agentic GraphRAG via End-to-end RL | [arXiv](https://www.emergentmind.com/papers/2507.21892) |
| **EMNLP 2025** | DeepResearcher: Scaling Deep Research via RL in Real-world Environments | [ACL](https://aclanthology.org/2025.emnlp-main.22.pdf) |
| **EMNLP 2025** | Smart-Searcher: Dynamic Knowledge Acquisition via RL | [ACL](https://aclanthology.org/2025.findings-emnlp.731.pdf) |
| **SemEval 2025** | Task 9: Food Hazard Detection Challenge | [arXiv](https://arxiv.org/html/2503.19800) |
| **EMNLP 2025 Findings** | FoodSafeSum: Food Safety Document Summarization | [ACL](https://aclanthology.org/2025.findings-emnlp.911.pdf) |
| **2026** | FoodGuardBench: Benchmarking Food-Safety Risks in LLMs | [arXiv](https://arxiv.org/html/2604.01444v2) |
| **arXiv 2025** | KG-R1: Efficient Agentic KG-RAG via RL | [GitHub](https://github.com/Jinyeop3110/KG-R1) |
| **arXiv 2025** | MMOA-RAG: Multi-Agent RL for RAG | [GitHub](https://github.com/chenyiqun/MMOA-RAG) |
| **arXiv 2025** | OptiMindTune: Multi-Agent HPO Framework | [arXiv](https://arxiv.org/html/2505.19205v2) |
| **arXiv 2025** | HPO Analysis for RAG (162 configurations) | [arXiv](https://arxiv.org/html/2505.03452v2) |

---

## 🧭 主题矩阵 / Topic Matrix

### 1️⃣ 食品安全 Agent 框架 / Food-Safety Agent Frameworks

- **SafeGuard AI** ([Sharma-Aditya7/multiagent-food-compliance](https://github.com/Sharma-Aditya7/multiagent-food-compliance)) — Multi-agent + CrewAI + Groq 实现 HACCP 自动审计
- **SFPPy** ([ovitrac/SFPPy](https://github.com/ovitrac/SFPPy)) — 食品接触合规 + 迁移建模 + agentic SAG
- **FoodSafe** ([AtharvaVavhal/FoodSafe](https://github.com/AtharvaVavhal/FoodSafe)) — YOLOv8 + IndicBERT + Prophet 多模态掺假检测
- **KAD-FoodHazard** ([phanben110/KAD-FoodHazard](https://github.com/phanben110/KAD-FoodHazard)) — SemEval-2025 Task 9 Top-2 方案，RAG + LLM 数据增强
- **SERSFormer** ([BioinfoMachineLearning/SERSFormer](https://github.com/BioinfoMachineLearning/SERSFormer)) — Transformer 检测食品农药残留
- **afrexai-food-safety** — 1933★ 行业 Skill，HACCP/FDA/USDA 法规映射

### 2️⃣ 调优与参数设计 / Tuning & Parameter Design

- **AutoRAG** ([Marker-Inc-Korea/AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG), 4.8k★) — AutoML-style 自动搜索最优 RAG 流水线
- **rag7 auto-tuning** ([bmsuisse/rag7](https://github.com/bmsuisse/rag7)) — 21 维检索 knob Optuna 自动寻优
- **OptiMindTune** ([arXiv](https://arxiv.org/html/2505.19205v2)) — 多 Agent (Recommender/Evaluator/Decision) 协同 HPO
- **DesignX** ([MetaEvo/DesignX](https://github.com/MetaEvo/DesignX)) — 双 Agent RL：Agent-1 选算子、Agent-2 调超参
- **HPO for RAG Survey** ([arXiv](https://arxiv.org/html/2505.03452v2)) — 162 维 RAG 配置搜索空间系统研究

### 3️⃣ RL × 知识库 / Reinforcement Learning × Knowledge Base

- **Graph-R1** — End-to-end RL over 知识超图
- **KG-R1** ([Jinyeop3110/KG-R1](https://github.com/Jinyeop3110/KG-R1)) — GRPO 训练单 Agent + 可插拔 KG
- **MMOA-RAG** ([chenyiqun/MMOA-RAG](https://github.com/chenyiqun/MMOA-RAG)) — 多 Agent 协同优化 query rewrite/retrieve/filter/generate
- **DeepResearcher** ([GAIR-NLP/DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)) — Real-web 端到端 RL 训练 Deep Research Agent
- **Smart-Searcher** ([RUCAIBox/R1-Searcher-plus](https://github.com/RUCAIBox/R1-Searcher-plus)) — 内/外部知识动态平衡
- **Experiential RL (ERL)** ([microsoft/experiential_rl](https://github.com/microsoft/experiential_rl)) — 经验-反思-固化三阶段训练
- **SafeOR-Gym** ([li-group/SafeOR-Gym](https://github.com/li-group/SafeOR-Gym)) — 9 个 OR 场景 Safe RL benchmark

### 4️⃣ 工作流与多 Agent 编排 / Workflow & Multi-Agent Orchestration

- **LangGraph** ([langchain-ai/langgraph](https://github.com/langchain-ai/langgraph), 35k★) — 工业级 stateful agent 图编排
- **AFlow** ([FoundationAgents/AFLOW](https://github.com/FoundationAgents/AFLOW), ICLR 2025 Oral) — MCTS 自动发现 agent 工作流
- **awesome-LangGraph** ([vonzosten/awesome-LangGraph](https://github.com/vonzosten/awesome-LangGraph)) — LangChain 生态资源索引
- **FoodCatalyst** ([akshaykarthicks/FoodCatalyst](https://github.com/akshaykarthicks/FoodCatalyst)) — CrewAI 三 Agent (Scout/Critic/Planner) 餐厅发现
- **WhatsEat** ([NUS-AIS-Practice-Modules/WhatsEat-backend-LangGraph-supervisor-py](https://github.com/NUS-AIS-Practice-Modules/WhatsEat-backend-LangGraph-supervisor-py)) — LangGraph supervisor 多 Agent
- **Annapurna** ([shashanksrajak/chatbot-agent-food-ordering](https://github.com/shashanksrajak/chatbot-agent-food-ordering)) — LangGraph 食品订购 Agent
- **ChatFood** ([mohammadi-milad-mim/ChatFood](https://github.com/mohammadi-milad-mim/ChatFood)) — LangGraph + LanceDB + ReAct 多 Agent
- **MASala** ([Naman009/MASala](https://github.com/Naman009/MASala)) — CrewAI Analyzer/Nutritionist/Chef/Presenter 食谱生成
- **ChatLearn GRPO Tutorial** ([alibaba/ChatLearn](https://github.com/alibaba/ChatLearn/blob/main/docs/en/tutorial/tutorial_grpo_fsdp_sglang_agent.md)) — LangGraph + GRPO Agent 训练

---

## 📂 目录结构 / Directory Structure

```
food_safety_agent_kb/
├── README.md                          # 本文件 / Index
├── agents/                            # 食品安全 Agent 框架
│   ├── safeguard_ai.md
│   ├── sfppy.md
│   ├── foodsafe.md
│   ├── kad_foodhazard.md
│   └── sersformer.md
├── optimization/                      # 调优与参数设计
│   ├── autorag.md
│   ├── optimindtune.md
│   ├── designx.md
│   ├── rag7_auto_tuning.md
│   └── hpo_for_rag_survey.md
├── rl_rag/                            # RL × 知识库
│   ├── graph_r1.md
│   ├── kg_r1.md
│   ├── mmoa_rag.md
│   ├── deepresearcher.md
│   ├── smart_searcher.md
│   ├── experiential_rl.md
│   └── safeor_gym.md
├── workflow/                          # 工作流与多 Agent 编排
│   ├── langgraph.md
│   ├── aflow.md
│   ├── foodcatalyst.md
│   ├── chatfood.md
│   ├── whatseat.md
│   ├── annapurna.md
│   └── masala.md
├── benchmark_papers/                  # 顶会论文详细解读
│   ├── semeval2025_task9_foodhazard.md
│   ├── foodsafesum_emnlp2025.md
│   ├── foodguardbench.md
│   ├── xpo_iclr2025.md
│   └── ...
└── top-conferences/                   # 会议周报
    └── 2026-nips-iclr-icml-food-safety-agent-report.md
```

---

## 🛠️ 关键工程建议 / Engineering Recommendations

### RAG 调优顺序（基于 [HPO Analysis](https://arxiv.org/html/2505.03452v2) 与 [rag7 auto-tuning](https://github.com/bmsuisse/rag7/blob/main/docs/auto-tuning.md)）

1. **Embedding Model** — 选型先于所有参数
2. **Chunk size + Overlap** — 控制上下文粒度
3. **Top-k / Retrieval factor** — 决定召回-精度
4. **Reranker / Fusion (RRF K=60, per-source weights)** — 后期精排
5. **Generator / Temperature** — 法律、医疗 (食品安全) 推荐 0.0–0.2

### 多 Agent 编排选型

| 需求 | 推荐框架 |
|---|---|
| 状态化、循环、复杂分支 | **LangGraph** (35k★) |
| 角色化快速原型 | **CrewAI** (FoodCatalyst/MASala) |
| Agent 工作流自动发现 | **AFlow** (ICLR 2025 Oral) |
| Tool-Integrated RL 训练 | **ChatLearn + LangGraph + GRPO** |

### Safe RL 食品安全应用

- 用 [SafeOR-Gym](https://github.com/li-group/SafeOR-Gym) 验证 HACCP 约束下的库存/排程策略
- 用 [Safe-Reinforcement-Learning-Baselines](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baselines) 对比 CMDP 算法
- 用 [Hybrid-RL-SNO-Restaurant-Inventory](https://github.com/ShafakatArnob/Hybrid-RL-SNO-Restaurant-Inventory) 验证冷链/餐饮库存 RL

---

## 📅 更新 / Last Updated

- **2026-06-22** — 首次创建，覆盖 2024-2026 顶会论文 + 高星仓库
- 计划每周二、周五 10:00 CST 更新

---

*本仓库聚焦食品安全场景的 AI Agent 技术栈（调优参数 / 知识库 / RL / 工作流），所有资源均为开源或可公开访问。*
*This knowledge base curates open-source resources for AI agents in food-safety scenarios, covering parameter tuning, knowledge bases, reinforcement learning, and workflow orchestration.*
