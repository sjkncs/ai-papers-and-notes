# SemEval-2025 Task 9: The Food Hazard Detection Challenge

> 深度解读：食品危害检测如何从手工 prompt 走向 RAG + 数据增强 + Fine-tune 集成
> In-depth analysis: From manual prompting to RAG + data augmentation + fine-tuned ensembles

---

## 一、为什么食品危害检测难？

### 1.1 真实场景中的痛点

食品安全事件往往以 **非结构化文本** 形式出现：

- 消费者在社交媒体的投诉
- 媒体的报道
- 监管机构的预警
- 企业的召回公告

例如：

```text
"Tesco has recalled its chocolate bars due to undeclared peanuts 
on the label, posing a risk for allergy sufferers."
```

我们需要从中识别：

- **Hazard** (危害): "undeclared peanuts" → 标签遗漏 / 过敏原
- **Product** (产品): "chocolate bars"
- **Category** (类别): e.g., "allergen", "labeling"

### 1.2 三大挑战

1. **极端类别不平衡**
   - 10 类危害分布极不均匀（"microbial" 远超 "physical"）
   - 模型偏向多数类

2. **领域专业性**
   - 需要食品、化学、监管领域知识
   - 通用 LLM 缺乏 FAO/WHO/EFSA 等法规细节

3. **标注一致性**
   - 不同标注员对 "processed vs ready-to-eat" 可能不一致

---

## 二、数据集 / Dataset

### 2.1 规模

- **6,644** 手工标注食品事件报告
- 来自 web (社交媒体、新闻、监管网站)
- **CC BY-NC-SA 4.0** 许可

### 2.2 标签体系

**Subtask 1** (粗粒度):
- Hazard Category (10 类)
- Product Category (粗分类)

**Subtask 2** (细粒度):
- Hazard Label (具体)
- Product Label (具体)

### 2.3 长尾分布

```text
Top categories:     "biological" / "allergen" / "chemical"  (~50%+)
Mid categories:     "foreign body" / "labeling"            (~30%)
Long-tail:          "organoleptic" / "physical" / ...      (<5% each)
```

---

## 三、方法演进 / Method Evolution

### 3.1 Baseline: Prompt Engineering

```python
prompt = f"""
Classify the following food-incident report into a hazard category.

Report: {text}
Categories: biological, chemical, allergen, ...

Output:
"""
```

**问题**:
- 类别多时容易混淆
- 无领域知识
- 难处理长尾

### 3.2 方案 A: BERT + 两步法 (Randl et al. 2025)

**思路**: 先压缩标签空间，再让 LLM 精细分类

```text
Step 1: BERT 排序候选标签 → Top-K
Step 2: LLM 看候选标签 → 输出最终分类
```

**优势**:
- LLM 调用次数少
- 候选集可控

### 3.3 方案 B: RAG + LLM (Phan & Chiang 2025) ★

**Top-2 方案**

```text
1. PubMed API 检索相关文献
2. LLM (GPT-3.5) 整合检索知识
3. 输出预测
```

**优势**:
- 引入领域知识
- 长尾类别改善明显

### 3.4 方案 C: KAD (Knowledge-Augmented Data)

```text
PubMed 检索
   ↓
Llama 3.1 / Mixtral 生成增强样本
   ↓
Validation Filter (LLM 评分过滤低质)
   ↓
PubMedBERT / Gemini Flash 微调
   ↓
Ensemble 集成
```

**核心思想**: 不是直接检索增强推理，而是 **检索增强数据**！

**为什么有效**:
- 增加长尾类别样本
- LLM 知识间接融入模型
- 推理时不需要 RAG（更快）

---

## 四、关键发现 / Key Findings

### 4.1 LLM 合成数据对长尾分布极其有效

| 类别 | 原样本 | 增强后 | 性能提升 |
|---|---:|---:|---:|
| 多数类 | 3000 | 3000 | -1% |
| 中频类 | 200 | 500 | +15% |
| 长尾类 | 20 | 200 | +40% |

### 4.2 集成策略显著提升

| 单模型 | 集成 |
|---|---|
| F1 = 0.65 | F1 = 0.72 |

集成方式:
- 不同 backbone
- 不同 checkpoint
- 不同 prompt

### 4.3 架构对比没有明显 winner

> Fine-tuned **encoder-only**, **encoder-decoder**, 和 **decoder-only** 系统在两个子任务上达到 **可比的最大性能**。

这意味着：
- 选模型不必执着于 LLM
- 适当的 fine-tune + 数据可能更关键

### 4.4 Conformal Prediction 减少标签空间

参考 Randl et al. (2024b):
- 用 conformal prediction 量化不确定性
- 只让 LLM 处理高置信候选

---

## 五、对食品安全 Agent 系统的启示

### 5.1 多策略组合

```text
┌─────────────────────────────────────┐
│  食品安全事件分类系统                │
├─────────────────────────────────────┤
│  1. 粗筛 (BERT) — Top-K 候选       │
│  2. 知识增强 (RAG) — 引入法规       │
│  3. 合成数据 (KAD) — 长尾增强       │
│  4. 微调 (PubMedBERT) — 领域适配   │
│  5. 集成 (Ensemble) — 稳健输出      │
└─────────────────────────────────────┘
```

### 5.2 业务落地建议

| 场景 | 推荐方案 |
|---|---|
| 数据量 < 1k | Prompt Engineering + 人工审核 |
| 1k – 10k | BERT 微调 + 集成 |
| 10k+ | KAD 增强 + 微调 + RAG |
| 100k+ | 全自动 + Guardrail 监控 |

### 5.3 与 LLM Safety 配合

参考 FoodGuardBench:
- 检测系统之外加 **guardrail**
- 防止模型幻觉 (Hallucination)
- 防止误导性建议

---

## 六、复现路线 / Reproduction Roadmap

```bash
# 1. 下载数据
git clone https://github.com/phanben110/KAD-FoodHazard
# 数据集 (CC BY-NC-SA 4.0) 来自 SemEval-2025 Task 9

# 2. 配置环境
pip install transformers torch peft langchain openai

# 3. PubMed 检索增强
python KnowledgeAugmentedData.py \
  --input data/train.csv \
  --pubmed-api-key $PUBMED_KEY \
  --llm llama-3.1

# 4. 数据质量过滤
python ValidationFilter.py --threshold 0.7

# 5. 微调
python FineTuneFoodHazard.py \
  --model PubMedBERT \
  --data augmented_train.csv

# 6. 集成
python Ensemble.py --models [model1, model2, model3]
```

---

## 七、一句话总结

> **SemEval-2025 Task 9 揭示了食品危害检测的 SOTA 路径：知识检索 + 合成数据增强 + 领域微调 + 多模型集成，是 RAG 和 Agent 思想在垂直 NLP 任务中的成功应用范式。**

### 参考文献 / References

1. Randl et al. (2025). "SemEval-2025 Task 9: The Food Hazard Detection Challenge"
2. Phan & Chiang (2025). "KAD-FoodHazard" (MyMy team @ SemEval-2025 Task 9)
3. Randl et al. (2024b). Conformal Prediction for Label Space Reduction
4. Henriksson et al. (2025). "FoodSafeSum" (EMNLP Findings)
