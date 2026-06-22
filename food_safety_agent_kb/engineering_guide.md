# 食品安全 AI Agent 系统搭建实战指南 / Food-Safety AI Agent Construction Guide

> 把本知识库的仓库、论文、框架整合成可落地的工程方案
> Integration of repositories, papers, and frameworks into deployable engineering solutions

---

## 一、整体技术栈 / Recommended Stack

### 1.1 基础层 / Base Layer

| 组件 | 推荐 | 理由 |
|---|---|---|
| **LLM** | GPT-4o / Qwen-2.5-72B / Claude 3.5 | 领域推理 + 法规理解 |
| **Embedding** | bge-m3 / text-embedding-3-large | 中英双语 + 长文档 |
| **Vector DB** | Faiss (精度优先) / Chroma (速度优先) | 见 HPO 论文 |
| **Workflow** | LangGraph (35k★) | 工业级 |

### 1.2 智能层 / Intelligence Layer

| 组件 | 推荐 | 理由 |
|---|---|---|
| **Agent Orchestration** | LangGraph | 状态化、复杂分支 |
| **Workflow Optimization** | AFlow (ICLR 2025 Oral) | 自动发现工作流 |
| **HPO** | AutoRAG (4.8k★) + Optuna | RAG 自动调优 |
| **RL Training** | ChatLearn + GRPO | Agent 训练 |

### 1.3 安全层 / Safety Layer

| 组件 | 推荐 | 理由 |
|---|---|---|
| **Guardrail** | FoodGuard-4B | 食品安全专用 |
| **Benchmark** | FoodGuardBench (3339 queries) | 验证 LLM safety |
| **Compliance Check** | afrexai-food-safety Skill | HACCP/FDA 法规 |

### 1.4 知识层 / Knowledge Layer

| 组件 | 推荐 | 理由 |
|---|---|---|
| **RAG Framework** | LangGraph + RAG | 状态化 RAG |
| **KG-RAG** | KG-R1 (Qwen-2.5-3B) | 可迁移、低成本 |
| **Multi-Agent RAG** | MMOA-RAG | 联合优化 |
| **Hypergraph RAG** | Graph-R1 | 关系推理 |

---

## 二、四类典型场景 / 4 Typical Scenarios

### 场景 A: 食品安全法规问答

```text
[User Query] → LangGraph Workflow
                    ↓
            Query Rewriter (LLM)
                    ↓
            Retriever (KG-R1 over 法规KG)
                    ↓
            Reviewer (LLM 验证)
                    ↓
            Generator (LLM + 引用)
                    ↓
            Guardrail (FoodGuard-4B)
                    ↓
[Answer + Citations]
```

**核心仓库**:
- [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
- [Jinyeop3110/KG-R1](https://github.com/Jinyeop3110/KG-R1)
- [chauncygu/Safe-Reinforcement-Learning-Baselines](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baselines)

### 场景 B: 食品危害事件自动分类

```text
[事件文本] → KAD-FoodHazard Pipeline
                    ↓
            PubMed RAG (领域知识)
                    ↓
            LLM 增强 (Llama 3.1)
                    ↓
            Validation Filter
                    ↓
            PubMedBERT Fine-tune
                    ↓
            Ensemble
                    ↓
[Hazard Label + Product Label]
```

**核心仓库**:
- [phanben110/KAD-FoodHazard](https://github.com/phanben110/KAD-FoodHazard)

### 场景 C: 食品掺假检测

```text
[食品图像] → FoodSafe Pipeline
                    ↓
            YOLOv8 (实时检测)
                    ↓
            IndicBERT / MuRIL (NLP)
                    ↓
            Prophet (时序)
                    ↓
            Personalized Risk Score
                    ↓
[Adulteration Alert + 风险等级]
```

**核心仓库**:
- [AtharvaVavhal/FoodSafe](https://github.com/AtharvaVavhal/FoodSafe)
- [BioinfoMachineLearning/SERSFormer](https://github.com/BioinfoMachineLearning/SERSFormer) (SERS 农药残留)

### 场景 D: HACCP 自动审计

```text
[工厂 SOP 文档] → SafeGuard AI Workflow
                    ↓
            Multi-Agent (CrewAI):
            ├─ Risk Identification Agent
            ├─ HACCP Compliance Agent
            └─ Documentation Agent
                    ↓
            [Audit Trail + Risk Report]
```

**核心仓库**:
- [Sharma-Aditya7/multiagent-food-compliance](https://github.com/Sharma-Aditya7/multiagent-food-compliance)

---

## 三、参数调优实战 / Parameter Tuning Practice

### 3.1 顺序（按重要性）

```text
1. Embedding Model
   ├─ 中英: bge-m3
   ├─ 纯英: text-embedding-3-large
   └─ 食品安全: bge-m3 + 领域 fine-tune

2. Chunk Size + Overlap
   ├─ 一般: 512 + 50
   ├─ 长文: 1024 + 100
   └─ 短 query: 256 + 25

3. Top-k
   ├─ Reranker 强: 20-30
   └─ 无 Reranker: 5-10

4. Reranker / Fusion (RRF K=60)
   ├─ Per-source weights:
   │   dense=1.0, BM25=1.0, sparse=0.6
   └─ 食品安全文档: BM25 权重提到 1.2

5. Generator / Temperature
   └─ 食品安全: T=0.0-0.2 (高 factuality)
```

### 3.2 自动调优工具

| 工具 | 用途 | Star |
|---|---|---|
| [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) | 全 pipeline 自动搜索 | 4.8k★ |
| [rag7 auto-tuning](https://github.com/bmsuisse/rag7) | 21 维 retrieval 调优 | - |
| [Optuna](https://optuna.org/) | 通用 HPO 库 | - |

---

## 四、强化学习训练 / RL Training

### 4.1 推荐路径

```text
已有 LangGraph Agent?
├─ 是 → ERL (experience + reflection + consolidation)
└─ 否 → 先搭 LangGraph，再上 ERL

想训练 multi-agent 协同？
├─ 是 → MMOA-RAG 或 ChatLearn GRPO
└─ 否 → 单 Agent (KG-R1)

想做 deep research？
└─ DeepResearcher (real-web RL)
```

### 4.2 训练资源

| 资源 | 链接 |
|---|---|
| ChatLearn (Alibaba) | https://github.com/alibaba/ChatLearn |
| microsoft/experiential_rl | https://github.com/microsoft/experiential_rl |
| Pearl (Meta) | https://github.com/facebookresearch/Pearl |

---

## 五、生产部署清单 / Production Deployment Checklist

### 5.1 安全 / Safety

- [ ] FoodGuard-4B guardrail 部署
- [ ] FoodGuardBench 评估 LLM safety
- [ ] 输入侧 jailbreak 检测
- [ ] 输出侧有害内容过滤

### 5.2 性能 / Performance

- [ ] Embedding 缓存
- [ ] Reranker skip（高置信不调用）
- [ ] Fast path（短 query 跳过 LLM 预处理）
- [ ] HyDE 启用阈值

### 5.3 监控 / Monitoring

- [ ] LangSmith (LangGraph 集成)
- [ ] MLflow (模型版本)
- [ ] Drift detection (输入分布变化)
- [ ] Audit log (法规要求)

### 5.4 评估 / Evaluation

- [ ] RAGAS (faithfulness / context precision)
- [ ] SemEval-2025 Task 9 (食品危害)
- [ ] FoodGuardBench (safety)
- [ ] 业务侧 A/B test

---

## 六、常见误区 / Common Pitfalls

### 6.1 调参误区

| 误区 | 正确做法 |
|---|---|
| ❌ 不建 eval set 就调 | ✅ 先建 ground-truth QA pair |
| ❌ 默认用 K=10 当 RRF | ✅ RRF K=60 + per-source weights |
| ❌ 同时调 embedding 和 chunk | ✅ 先定 embedding 再调 chunk |
| ❌ 用 aggregate metric 决策 | ✅ 按 query 类型细分看 |

### 6.2 Agent 误区

| 误区 | 正确做法 |
|---|---|
| ❌ 直接套 LangGraph 模板 | ✅ 先梳理业务状态机 |
| ❌ 每个 node 都用最强 LLM | ✅ 简单 node 用小模型 |
| ❌ 一次性发布完整 pipeline | ✅ 渐进式 rollout |
| ❌ 没有 fallback 路径 | ✅ 设计降级策略 |

### 6.3 RL 误区

| 误区 | 正确做法 |
|---|---|
| ❌ 上来就上 GRPO | ✅ 先用 SFT 建立 baseline |
| ❌ reward 全靠最终答案 | ✅ 设计 per-turn reward |
| ❌ 训练在合成问题 | ✅ 接近真实分布 |
| ❌ 没有 safety constraint | ✅ 用 Safe RL 算法 |

---

## 七、学习路径 / Learning Path

### 7.1 入门 (1-2 周)

1. 阅读 [langgraph README](https://github.com/langchain-ai/langgraph) — 理解 stateful agent
2. 跑通 [FoodCatalyst](https://github.com/akshaykarthicks/FoodCatalyst) — CrewAI 三 Agent
3. 阅读 [AutoRAG 文档](https://marker-inc-korea.github.io/AutoRAG/) — RAG 自动调优

### 7.2 进阶 (1-2 月)

1. 复现 [DesignX](https://github.com/MetaEvo/DesignX) — 双 Agent RL
2. 跑通 [KAD-FoodHazard](https://github.com/phanben110/KAD-FoodHazard) — 食品安全 RAG
3. 读 [AFlow paper](https://arxiv.org/abs/2410.10762) — MCTS workflow 优化
4. 部署 [KG-R1](https://github.com/Jinyeop3110/KG-R1) — 知识图谱 RAG

### 7.3 高级 (持续)

1. 跟踪 [NeurIPS / ICLR / ICML](https://paperswithcode.com/) 最新工作
2. 关注 [awesome-LangGraph](https://github.com/vonzosten/awesome-LangGraph) 生态
3. 参与 [SemEval 食品安全任务](https://aclanthology.org/2025.semeval-1.325/)

---

## 八、一句话总结

> **食品安全 AI Agent 系统 = LangGraph 工作流 + 领域 KG-RAG (KG-R1) + 自动调优 (AutoRAG) + 安全 Guardrail (FoodGuard-4B) + 持续 RL 训练 (ERL/GRPO)。技术栈完整、论文可复现、仓库高 star，可以直接作为落地方案。**

---

*Last Updated: 2026-06-22*
*参考本知识库的所有仓库与论文*
