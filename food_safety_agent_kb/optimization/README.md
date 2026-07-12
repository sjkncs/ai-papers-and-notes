# 2️⃣ 调优与参数设计 / Tuning & Parameter Design

> AI Agent / RAG / LLM 系统调优、超参数设计、HPO 框架。

## 概览 / Overview

| 仓库 / 论文 | ⭐ | 类别 | 说明 |
|---|---:|---|---|
| [Marker-Inc-Korea/AutoRAG](#1-autorag) | 4815 | AutoML RAG | 自动搜索最优 RAG pipeline |
| [bmsuisse/rag7](#2-rag7-auto-tuning) | - | RAG Tuner | 21 维 retrieval knob Optuna 自动寻优 |
| [OptiMindTune](#3-optimindtune) | paper | Multi-Agent HPO | Recommender/Evaluator/Decision 三 Agent |
| [MetaEvo/DesignX](#4-designx) | NeurIPS 2025 | Dual-Agent RL | 双 Agent：工作流生成 + 超参控制 |
| [HPO for RAG Survey](#5-hpo-for-rag-analysis) | arXiv 2025 | Survey | 162 维配置搜索空间 |

---

## 1. AutoRAG (4.8k★)

- **Repo**: https://github.com/Marker-Inc-Korea/AutoRAG
- **Stars**: 4815 | **License**: Apache-2.0
- **Homepage**: https://marker-inc-korea.github.io/AutoRAG/

### 核心理念
**AutoML 思想用于 RAG** — 自动评估并选择最优 RAG pipeline。

### 用法 / Usage
```yaml
# config.yaml
nodes:
  - node_type: query_decomposer       # 拆解 query
  - node_type: retrieval              # 检索
  - node_type: reranker               # 重排
  - node_type: prompt_maker           # prompt 构造
    strategy:
      metrics:
        - metric_name: meteor
        - metric_name: rouge
        - metric_name: sem_score
          embedding_model: openai
  - node_type: generator              # 生成
    strategy:
      metrics:
        - metric_name: bleu
        - metric_name: g_eval
```

```bash
autorag evaluate --config config.yaml --dataset your_qa.parquet
# → 自动遍历模块组合，输出最优 pipeline
```

### 适用场景
- 不知道哪种 RAG 配置适合自己的数据
- 想避免手动调 chunk_size / embedding / top-k / prompt 等
- 食品安全文档（异构、术语多）调优

---

## 2. rag7 auto-tuning

- **Repo**: https://github.com/bmsuisse/rag7/blob/main/docs/auto-tuning.md
- **机制**: `RAGTuner.tune()` 使用 **Optuna TPE sampler** 搜索 17 维

### 关键参数范围

| Parameter | Range | 含义 |
|---|---:|---|
| `top_k` | 5–20 | 最终返回文档数 |
| `retrieval_factor` | 2–8 | 检索放大倍数 |
| `rerank_top_n` | 3–10 | rerank 后保留数 |
| `semantic_ratio` | 0.3–0.9 | BM25 ⇄ vector 平衡 |
| `query_min_words` | 1–8 | 短 query 跳过 LLM 预处理 |
| `bm25_fallback_semantic` | 0.7–1.0 | BM25 弱时语义增强 |
| `rerank_skip_gap` | 0.05–0.3 | 跳过 rerank 的分数阈值 |
| `hyde_min_words` | 4–12 | HyDE 启用阈值 |
| `bm25_fallback_threshold` | 0.2–0.6 | 触发语义回退阈值 |
| `fast_accept_score` | 0.5–0.95 | 快速路径置信度 |
| `fast_accept_confidence` | 0.6–0.95 | 跳过 LLM 确认的阈值 |
| `rerank_skip_dominance` | 0.6–0.95 | rerank 跳过阈值 |
| `expert_threshold` | 0.05–0.3 | 升级 expert reranker 阈值 |

### None-able 字段
> `None` = 完全禁用该阶段，是搜索的一等假设

```python
from rag7 import RAGTuner
tuner = RAGTuner(study_name="food_safety_v1", direction="maximize")
best = tuner.tune(eval_fn=eval_pipeline, n_trials=200)
```

---

## 3. OptiMindTune

- **Paper**: https://arxiv.org/html/2505.19205v2
- **框架**: 多 Agent HPO

### 三个 Agent 角色
```text
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Recommender     │ →  │  Evaluator       │ →  │  Decision        │
│  (LLM)           │    │  (LLM)           │    │  (LLM)           │
│  生成候选超参    │    │  评估模型表现    │    │  调度搜索过程    │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

### 实验 / Experiments
- 数据集：Breast Cancer / Iris / Wine
- Baseline：Optuna
- Best：RandomForestClassifier @ Breast Cancer = 96.14%

---

## 4. DesignX (NeurIPS 2025)

- **Paper**: https://arxiv.org/pdf/2505.17866
- **Repo**: https://github.com/MetaEvo/DesignX
- **Code**: https://github.com/MetaEvo/DesignX

### 核心创新
- 首次 **端到端联合学习** 算法结构 + 超参
- Modular-EC：116 个算子模块库

### 双 Agent 架构
```text
Agent-1 (π_φ, Transformer)    Agent-2 (π_θ, Transformer)
  ↓                                ↓
自回归生成算子工作流         动态调整超参
Module IDs → 116 种选择      μ, Σ → Normal 分布 → C_t^m ~ N(μ, Σ)
                                ↓
              协同训练 reward: J_φ + J_θ
```

### 训练规模
- 10,000 合成问题
- 应用于 Protein-docking / AutoML / UAV path planning

### 应用启示
- 食品安全优化场景（冷链调度、库存优化）可借鉴
- 双 Agent 思路适合 "模型选型 + 超参" 联合优化

---

## 5. HPO for RAG Analysis

- **Paper**: https://arxiv.org/html/2505.03452v2

### 搜索空间
- **162 维 RAG 配置**
- 5 个核心参数：chunk_size, overlap, embedding_model, top_k, generator

### 现有工具
| 工具 | 策略 | 链接 |
|---|---|---|
| AutoRAG | Greedy | https://github.com/Marker-Inc-Korea/AutoRAG |
| RAGBuilder | TPE | https://github.com/codertemplat/RAGBuilder |
| LlamaIndex | RayTune / Optuna / Hyperopt | https://github.com/run-llama/llama_index |

### 推荐顺序 (Greedy-R-CC)
1. Embedding Model (context correctness metric)
2. Chunk Size
3. Chunk Overlap
4. Generative Model
5. Top-k

### 关键发现 (arXiv:2505.08445)
| Finding | 数值 |
|---|---|
| Chroma vs Faiss | Chroma 快 13%，Faiss 精度高 |
| Chunking | naive 固定长度 + 小窗口 + 最小 overlap 最佳 |
| Re-ranking | 提升有限，runtime ×5 |
| 法律/医疗 (食品安全) | Faithfulness + Context Precision 优先 |
| 推荐配置 | naive chunking + reranker + Faiss + T=0.0–0.2 |
| CRAG (Corrective) | context precision 达 99% |

---

## 🎯 实战决策树 / Practical Decision Tree

```
你是哪种场景？
│
├─ 不知道 RAG 哪部分该调 → AutoRAG (greedy)
├─ 知道范围，想自动寻最优 → Optuna + rag7 (TPE)
├─ 想联合结构 + 超参 → DesignX (dual-agent RL)
├─ 想加 LLM reasoning → OptiMindTune (multi-agent)
└─ 想做严肃学术对比 → 复现 HPO for RAG 论文 (162 维 sweep)
```
