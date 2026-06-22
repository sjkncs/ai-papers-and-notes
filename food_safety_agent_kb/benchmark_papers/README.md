# 5️⃣ 顶会论文 / Top-Conference Papers

> 食品安全 × AI Agent 相关顶会论文详细解读（2024-2026）

## 📚 论文清单 / Paper List

### 🎯 直接相关：食品安全 / Directly Food-Safety

| 论文 | 会议 | 日期 | 主题 |
|---|---|---|---|
| [SemEval-2025 Task 9: Food Hazard Detection](#1-semeval2025-task9) | SemEval | 2025 | 食品危害检测 benchmark |
| [FoodSafeSum](#2-foodsafesum) | EMNLP 2025 Findings | 2025 | 食品安全文档摘要数据集 |
| [FoodGuardBench](#3-foodguardbench) | arXiv 2026 | 2026 | LLM 食品安全 jailbreak benchmark |
| [KAD-FoodHazard (MyMy @ SemEval-2025 Task 9)](#4-kad-foodhazard) | SemEval 2025 | 2025 | RAG + LLM 数据增强 |

### 🤖 AI Agent / Workflow / RL

| 论文 | 会议 | 日期 | 主题 |
|---|---|---|---|
| [DesignX](#5-designx) | NeurIPS 2025 | 2025 | 双 Agent RL 算法设计 |
| [AGENTFLOW / Flow-GRPO](#6-agentflow) | NeurIPS 2025 Workshop | 2025 | In-the-Flow Agent RL |
| [AFlow](#7-aflow) | ICLR 2025 Oral | 2025 | MCTS 自动优化 agent 工作流 |
| [XPO (Exploratory Preference Optimization)](#8-xpo) | ICLR 2025 | 2025 | 在线探索 RLHF |
| [Multi-Agent Reflection](#9-multi-agent-reflection) | ICML 2025 | 2025 | 多 Agent 反思强化推理 |
| [Graph-R1](#10-graph-r1) | ACL 2025 | 2025 | Agentic GraphRAG + RL |
| [DeepResearcher](#11-deepresearcher) | EMNLP 2025 | 2025 | Real-web 端到端 RL |
| [Smart-Searcher](#12-smart-searcher) | EMNLP 2025 Findings | 2025 | 动态知识获取 RL |
| [MMOA-RAG](#13-mmoa-rag) | arXiv 2025 | 2025 | 多 Agent 协同 RAG |
| [KG-R1](#14-kg-r1) | arXiv 2025 | 2025 | 高效 KG-RAG via RL |
| [OptiMindTune](#15-optimindtune) | arXiv 2025 | 2025 | 多 Agent HPO |
| [HPO for RAG Analysis](#16-hpo-rag) | arXiv 2025 | 2025 | 162 维 RAG 配置搜索 |

---

## 1. SemEval-2025 Task 9: Food Hazard Detection Challenge

- **Paper**: https://arxiv.org/html/2503.19800
- **ACL Anthology**: https://aclanthology.org/2025.semeval-1.325/
- **数据规模**: 6,644 手工标注食品事件报告

### 两个子任务
- **Subtask 1**: 判断是否隐含 10 类食品危害 + 关联食品类别
- **Subtask 2**: 精细化分类（具体危害标签 + 具体产品标签）

### 数据集
- 来源：web (社交媒体、新闻)
- 极端类别不平衡
- **CC BY-NC-SA 4.0**

### 关键发现
1. **LLM 合成数据** 对长尾分布过采样非常有效
2. 集成策略显著提升分类性能
3. encoder-only / encoder-decoder / decoder-only 三类架构表现 **没有明显 winner**
4. **KAD-FoodHazard** (PubMed RAG) Top-2

### 应用启示
- 食品召回事件自动分类
- 食品安全舆情监控
- 供应链风险预警

---

## 2. FoodSafeSum (EMNLP 2025 Findings)

- **Paper**: https://aclanthology.org/2025.findings-emnlp.911.pdf

### 数据集
- 人工 + LLM 生成的食品安全文档摘要
- 食品安全相关 metadata

### 三个 NLP 任务
1. **多标签分类** (MLC): 按领域特定类别组织文档
2. **文档检索**: 获取法规和科学证据
3. **RAG QA**: 提升事实准确性

### 关键发现
- LLM 摘要与人工摘要 **表现相当或更好**
- 摘要聚类支持事件追踪 + 合规监控

---

## 3. FoodGuardBench

- **Paper**: https://arxiv.org/html/2604.01444v2
- **规模**: 3,339 FDA-grounded queries

### 三大贡献
1. **FoodGuardBench** — 首个 FDA 法规基础的 benchmark
2. **可扩展对抗性 query 生成 pipeline** — AutoDAN / PAP 等 jailbreak
3. **FoodGuard-4B** — 4B 参数的领域 guardrail 模型

### 揭示的三个关键漏洞
1. LLM 在食品领域 **safety alignment 稀疏**
2. 被攻破后 **生成可操作的危险指导**
3. 现有 LLM-based guardrails **系统性地忽视** 领域威胁

---

## 4. KAD-FoodHazard (MyMy @ SemEval-2025 Task 9)

- **Repo**: https://github.com/phanben110/KAD-FoodHazard
- **License**: GPL-3.0
- **排名**: Top-2 Subtask 1 + Top-2 Subtask 2

### Pipeline
```text
PubMed API → Document Retrieval (RAG)
       ↓
   LLM Generation (Llama 3.1 / Mixtral)
       ↓
   Validation Filter (scoring)
       ↓
   Fine-tune (PubMedBERT / Gemini Flash)
       ↓
   Ensemble Prediction
```

### 关键模块
- **KnowledgeAugmentedData**: 用 RAG + 领域知识增强训练数据
- **FineTuneFoodHazard**: 在增强数据上微调

---

## 5. DesignX (NeurIPS 2025)

- **Paper**: https://arxiv.org/pdf/2505.17866
- **Repo**: https://github.com/MetaEvo/DesignX

### 创新
- 首个 **端到端联合学习** 算法结构 + 超参
- Modular-EC：116 个算子模块库
- 双 Transformer Agent + 协同训练 reward

### 应用
- Protein-docking
- AutoML
- UAV path planning

### 食品安全启示
- 冷链调度算法自动设计
- 库存策略自动发现

---

## 6. AGENTFLOW / Flow-GRPO (NeurIPS 2025 Workshop: Efficient Reasoning)

- **Paper**: https://openreview.net/pdf?id=ddabb524f1439770dfe2d5bad51cf9a57d2a6ecd.pdf

### 核心创新
- **In-the-Flow 学习**：在执行流中优化 planner
- **Flow-GRPO**：on-policy 算法，单次轨迹广播奖励到所有 turn

### 框架
```text
┌──────────────────────────────────────┐
│ Action Planner P  (Trainable)        │
│ Tool Executor E                      │
│ Execution Verifier V                 │
│ Solution Generator G                 │
│ Shared Memory M + Toolset K          │
└──────────────────────────────────────┘
```

### 关键技术
- 长链路 credit assignment 简化
- 收敛性证明

---

## 7. AFlow (ICLR 2025 Oral)

- **Paper**: https://arxiv.org/abs/2410.10762
- **Repo**: https://github.com/FoundationAgents/AFLOW

### 关键结果
- 6 benchmark 平均 **+5.7%** vs SOTA
- 小模型以 **4.55% GPT-4o 成本** 超越 GPT-4o (在特定任务)

### 架构
- Node / Operator / Workflow / Optimizer / Evaluator
- MCTS over code-represented workflows

---

## 8. XPO (Exploratory Preference Optimization, ICLR 2025)

- **Paper**: https://openreview.net/pdf?id=QYigQ6gXNw
- **arXiv**: https://arxiv.org/html/2405.21046
- **Code**: Hugging Face TRL — XPOTrainer

### 核心创新
- 在线 RLHF **探索** 算法
- 仅需 **一行修改** 在线 DPO
- 提供 **principled exploration bonus**
- 理论证明 sample-efficient

### 关键观察
DPO 隐式执行 Bellman error 最小化（implicit Q*-approximation）

### 实践
- HuggingFace TRL 已集成 [XPO Trainer](https://huggingface.co/docs/trl/xpo_trainer)
- 应用到安全对齐、有害 query 拒答

---

## 9. Reinforce LLM Reasoning with Multi-Agent Reflection (ICML 2025)

- **Authors**: Yurun Yuan, Tengyang Xie
- **arXiv**: https://arxiv.org/abs/...

### 范式
- 多 Agent 相互反思强化推理
- 比单 Agent 更稳定

---

## 10. Graph-R1 (ACL 2025)

- **arXiv**: 2507.21892
- **核心**: Agentic GraphRAG via End-to-end RL

### 创新
- 知识超图构建
- 多轮 Agent-环境交互
- 端到端奖励

---

## 11. DeepResearcher (EMNLP 2025)

- **Paper**: https://aclanthology.org/2025.emnlp-main.22.pdf
- **Repo**: https://github.com/GAIR-NLP/DeepResearcher

### 关键结果
- +28.9 over prompt engineering baseline
- +7.2 over RAG-based RL agents

### 涌现能力
- Planning
- Cross-validation
- Self-reflection
- Honesty (找不到答案时)

---

## 12. Smart-Searcher (EMNLP 2025 Findings)

- **Paper**: https://aclanthology.org/2025.findings-emnlp.731.pdf
- **Repo**: https://github.com/RUCAIBox/R1-Searcher-plus

### 创新
- 内部 + 外部知识动态平衡
- RL 训练 Dynamic Knowledge Acquisition
- 检索次数 -42.9%
- 性能 +4.3% over strong baseline

---

## 13. MMOA-RAG

- **Paper**: https://openreview.net/forum?id=9Ia0KiVAut
- **Repo**: https://github.com/chenyiqun/MMOA-RAG

### 范式
- 多 Agent 协同 (query rewrite / retrieval / filter / generation)
- 统一 reward (答案 F1)
- 联合训练避免目标不一致

---

## 14. KG-R1

- **arXiv**: 2509.26383
- **Repo**: https://github.com/Jinyeop3110/KG-R1

### 优势
- Qwen-2.5-3B 即可超越大模型方法
- 训练后 **即插即用** 到新 KG
- GRPO-style 训练

---

## 15. OptiMindTune

- **arXiv**: https://arxiv.org/html/2505.19205v2
- **框架**: Recommender + Evaluator + Decision 三 Agent 协同 HPO

### 实验
- Breast Cancer / Iris / Wine
- Best: RandomForestClassifier @ Breast Cancer = 96.14%

---

## 16. HPO for RAG Analysis (arXiv 2025)

- **Paper**: https://arxiv.org/html/2505.03452v2

### 搜索空间
- **162 维** RAG 配置
- 5 个核心参数

### 推荐顺序 (Greedy-R-CC)
1. Embedding Model
2. Chunk Size
3. Chunk Overlap
4. Generative Model
5. Top-k

### 配套工作 (arXiv 2505.08445)
- Chroma 快 13%，Faiss 精度高
- naive chunking 最佳
- 法律/医疗推荐：naive + reranker + Faiss + T=0.0-0.2
- CRAG 可达 99% context precision
