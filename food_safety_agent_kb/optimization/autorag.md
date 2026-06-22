# AutoRAG: AutoML for RAG Pipeline (4.8k★)

> 用 AutoML 思想自动寻找最优 RAG pipeline 配置

## 📦 仓库信息

- **Repo**: https://github.com/Marker-Inc-Korea/AutoRAG
- **Stars**: 4,815 | **Forks**: 403 | **License**: Apache-2.0
- **Releases**: 65
- **Homepage**: https://marker-inc-korea.github.io/AutoRAG/

## 🎯 核心理念

> AutoML 思想用于 RAG：自动评估并选择最优 RAG pipeline

RAG 系统有无数组合方式：

| 维度 | 选项 |
|---|---|
| Query Preprocessing | query_decomposer, hyde, query_rewrite |
| Retrieval | BM25, dense, hybrid, multi_query |
| Reranker | cohere, bge-reranker, cross-encoder |
| Prompt Maker | fstring, chat, long_context |
| Generator | openai, claude, llama, mixtral |
| Chunk Strategy | token, sentence, semantic |

**人工尝试 → AutoRAG 自动搜索**

## 🚀 快速开始

### 安装

```bash
pip install AutoRAG
```

### 1. 准备数据

```python
# qa.parquet — query + ground truth
import pandas as pd

qa_data = pd.DataFrame({
    "query": [...],
    "ground_truth": [...],
    "ground_truth_context": [...]
})
qa_data.to_parquet("qa.parquet")
```

### 2. 配置 YAML

```yaml
# config.yaml
nodes:
  - node_type: query_decomposer
  - node_type: retrieval
    strategy:
      metrics:
        - metric_name: retrieval_f1
        - metric_name: retrieval_recall
      modules:
        - module_type: bm25
        - module_type: vectordb
          embedding_model: openai
          top_k: [5, 10, 20]
  - node_type: reranker
    strategy:
      metrics:
        - metric_name: retrieval_f1
      modules:
        - module_type: cohere_reranker
        - module_type: bge_reranker
  - node_type: prompt_maker
    strategy:
      metrics:
        - metric_name: meteor
        - metric_name: rouge
      modules:
        - module_type: fstring
          prompt: |
            Question: {query}
            Context: {retrieved_context}
            Answer:
  - node_type: generator
    strategy:
      metrics:
        - metric_name: bleu
        - metric_name: g_eval
      modules:
        - module_type: openai
          llm: [gpt-4o-mini, gpt-4o]
          temperature: [0.0, 0.2, 0.5]
```

### 3. 运行

```bash
autorag evaluate \
  --config config.yaml \
  --qa_data_path qa.parquet \
  --corpus_path corpus.parquet \
  --output_dir ./result
```

### 4. 输出

```text
result/
├── best_pipeline.yaml      # 最优 pipeline 配置
├── summary.csv             # 所有尝试的评分
└── trial_logs/             # 每个 trial 详细日志
```

## 🧠 核心算法：Greedy Search

AutoRAG 使用 **Greedy Sequential Optimization**：

```text
Step 1: 优化 query_decomposer (在所有 metric 上评估)
Step 2: 用 Step 1 最优配置 → 优化 retrieval
Step 3: 用 Step 1-2 → 优化 reranker
Step 4: 用 Step 1-3 → 优化 prompt_maker
Step 5: 用 Step 1-4 → 优化 generator
```

### Greedy 三种顺序

| Order | 描述 |
|---|---|
| **Greedy-M** | Model 先: Generative → Embedding → Chunk → Top-k |
| **Greedy-R** | Retrieval 先: Embedding → Chunk → Generative → Top-k |
| **Greedy-R-CC** | Retrieval + Context Correctness: 先优化 retrieval 再调生成 |

## 📊 关键能力

| 能力 | 说明 |
|---|---|
| **自动评估** | 内置 retrieval_f1, bleu, rouge, g_eval 等 |
| **模块组合** | 数十种 RAG 模块自由组合 |
| **最优选择** | 自动输出最优 pipeline |
| **可复现** | YAML 配置 → 完全可复现 |
| **可视化** | summary.csv 方便分析 |

## 🔧 模块清单（部分）

### Query Preprocessing
- `query_decomposer` — 拆解复杂 query
- `query_rewrite` — 改写 query
- `hyde` — Hypothetical Document Embeddings

### Retrieval
- `bm25` — 稀疏检索
- `vectordb` — 稠密检索
- `hybrid` — 混合检索

### Reranker
- `cohere_reranker`
- `bge_reranker`
- `cross_encoder`

### Prompt Maker
- `fstring` — 简单模板
- `chat` — 对话模板
- `long_context` — 长上下文

### Generator
- `openai`, `claude`, `llama`, `mixtral`, `bedrock`

## 💡 实战应用

### 食品安全文档 RAG

```yaml
nodes:
  - node_type: retrieval
    strategy:
      modules:
        - module_type: hybrid
          embedding_model: bge-m3
          top_k: [10, 20, 50]
          bm25_weight: [0.3, 0.5, 0.7]
  - node_type: reranker
    strategy:
      modules:
        - module_type: bge_reranker
  - node_type: generator
    strategy:
      modules:
        - module_type: openai
          llm: gpt-4o-mini
          temperature: [0.0, 0.1]  # 食品安全要低温度
```

## 🔗 关联项目

- [bmsuisse/rag7](https://github.com/bmsuisse/rag7) — 21 维 Optuna 自动调优
- [RAGBuilder](https://github.com/codertemplat/RAGBuilder) — TPE 策略
- [LlamaIndex](https://github.com/run-llama/llama_index) — 集成 RayTune/Optuna
- [OptiMindTune](../optimization/README.md#3-optimindtune) — 多 Agent HPO

## 📚 论文依据

- arXiv:2505.03452 — "An Analysis of Hyper-Parameter Optimization Methods for Retrieval Augmented Generation"
- arXiv:2505.08445 — "Optimizing Retrieval-Augmented Generation: Analysis of Hyperparameter Impact"

## ⚠️ 限制

- Greedy 不保证全局最优
- 需要 ground-truth QA 数据
- 评估耗时（每个 trial 都要跑 LLM）

## 🌟 为什么是 4.8k★？

1. **降低 RAG 调优门槛**
2. **生产级可靠**
3. **模块化设计**
4. **持续活跃维护**（最新 v0.3.22）
