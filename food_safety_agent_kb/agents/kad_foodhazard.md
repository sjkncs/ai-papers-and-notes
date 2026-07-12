# KAD-FoodHazard: SemEval-2025 Task 9 Top-2 方案

> Knowledge-Augmented Data 增强方法解决食品危害分类的极端类别不平衡问题

## 📦 仓库信息

- **Repo**: https://github.com/phanben110/KAD-FoodHazard
- **Stars**: 5 | **Forks**: 3 | **License**: GPL-3.0
- **创建**: 2025-02-27
- **成绩**: SemEval-2025 Task 9 Subtask 1 & 2 **Top-2**

## 🎯 任务背景

**SemEval-2025 Task 9**: Food Hazard Detection Challenge
- 6,644 手工标注食品事件报告
- 长尾分布（10 类危害）
- CC BY-NC-SA 4.0 数据

### 两个子任务
- **Subtask 1**: Hazard Category (10 类) + Product Category
- **Subtask 2**: Hazard Label (细) + Product Label (细)

## 💡 核心创新：Knowledge-Augmented Data (KAD)

> 不是直接用 RAG 增强推理，而是 **用 RAG 增强训练数据**！

```text
┌──────────────────────────────────────────────────┐
│  KAD Pipeline                                    │
├──────────────────────────────────────────────────┤
│  1. PubMed API → 检索相关领域文献                │
│  2. LLM (Llama 3.1 / Mixtral) → 生成增强样本    │
│  3. Validation Filter → 过滤低质样本            │
│  4. PubMedBERT / Gemini Flash Fine-tune         │
│  5. Ensemble → 最终预测                          │
└──────────────────────────────────────────────────┘
```

## 🛠️ 技术栈

| 组件 | 用途 |
|---|---|
| **PubMed API** | 领域知识检索 |
| **Llama 3.1 / Mixtral** | 数据生成 |
| **PubMedBERT** | 领域微调 backbone |
| **Gemini Flash** | 替代 backbone |
| **RAG paradigm** | 检索增强 |

## 🚀 复现

```bash
git clone https://github.com/phanben110/KAD-FoodHazard
cd KAD-FoodHazard
pip install -r requirements.txt

# 设置 API Keys
export PUBMED_API_KEY=<key>
export OPENAI_API_KEY=<key>  # 如果用 OpenAI 模型
export GOOGLE_API_KEY=<key>  # 如果用 Gemini

# 1. 知识检索 + 数据生成
python KnowledgeAugmentedData.py \
  --input data/train.csv \
  --pubmed-api-key $PUBMED_API_KEY \
  --llm llama-3.1

# 2. 数据验证过滤
python ValidationFilter.py --threshold 0.7

# 3. 微调
python FineTuneFoodHazard.py \
  --model PubMedBERT \
  --data augmented_train.csv \
  --epochs 10

# 4. 集成 + 评估
python Ensemble.py --models [pubmedbert, gemini_flash]
```

## 📊 关键模块详解

### 1. KnowledgeAugmentedData

```python
from langchain.llms import OpenAI
from pubmed_lookup import PubMedAPI

llm = OpenAI(model="llama-3.1")
pubmed = PubMedAPI(api_key=PUBMED_KEY)

def augment_sample(text, label):
    # 1. PubMed 检索
    docs = pubmed.search(text, top_k=5)

    # 2. LLM 生成增强样本
    context = "\n".join([doc.abstract for doc in docs])
    prompt = f"""
    Context from PubMed:
    {context}

    Original sample (label: {label}):
    {text}

    Generate 5 diverse paraphrases that maintain the same label.
    """
    augmented = llm.generate(prompt)
    return augmented
```

### 2. Validation Filter

```python
def filter_low_quality(samples, threshold=0.7):
    """用 LLM 评分过滤低质样本"""
    scored = []
    for sample in samples:
        score = llm.evaluate(
            f"Is this sample high-quality and consistent with label?\n{sample}",
            rubric="quality 0-1"
        )
        if score >= threshold:
            scored.append(sample)
    return scored
```

### 3. FineTuneFoodHazard

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    num_labels=10  # 10 类危害
)
# 在 augmented_train.csv 上 fine-tune
```

### 4. Ensemble

```python
def ensemble_predict(text):
    preds = []
    for model in [pubmedbert_model, gemini_model, ...]:
        preds.append(model.predict(text))

    # 投票 / 加权融合
    final = weighted_vote(preds, weights=[0.4, 0.3, 0.3])
    return final
```

## 📈 性能 (Top-2 关键)

- **Subtask 1**: Top-2
- **Subtask 2**: Top-2
- 比 baseline 提升 ~10-15%
- 长尾类别改善最显著

## 🔗 关联论文

- SemEval-2025 Task 9 Paper: https://aclanthology.org/2025.semeval-1.325/
- KAD-FoodHazard Paper: 同 ACL Anthology
- FoodSafeSum (相关): https://aclanthology.org/2025.findings-emnlp.911.pdf

## 💡 启示

1. **数据增强比模型调优更有效**（尤其长尾）
2. **RAG 用于训练**（而非仅用于推理）值得探索
3. **领域 backbone**（PubMedBERT）比通用 LLM 更稳
4. **集成**几乎总能提升 3-5%

## ⚠️ 限制

- ⭐ 仅 5 stars，但学术价值高
- GPL-3.0 许可（注意商用）
- 依赖外部 API（PubMed、OpenAI）
