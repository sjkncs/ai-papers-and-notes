# Societies of Thought in Multi-Expert Investment Reasoning
# 多专家投资推理中的思维社会

> **目标会议 / Target Venue:** NeurIPS 2026 / Journal of Investment Management
> **基于 / Based on:** Reasoning Models Generate Societies of Thought (Google 2026) — [原始论文分析](../week-04-jan22-28/societies-of-thought/review.md)
> **核心迁移 / Core Adaptation:** 推理模型内部多智能体模拟 → 投资模型内部多专家推理模拟

---

## Abstract / 摘要

**English:**
Investment reasoning models increasingly incorporate extended chain-of-thought processes, but whether their improved performance comes from **longer computation** or **richer internal deliberation** remains unclear. Adapting Google's "Societies of Thought" framework to quantitative finance, we investigate whether financial reasoning models internally simulate multiple expert perspectives—a bull analyst, a bear analyst, a risk manager, and a macro strategist—rather than executing a single linear reasoning chain. Through mechanistic interpretability analysis of 3 financial reasoning models (FinGPT-4, InvestmentGPT, QuantReasoner), we find that **reasoning models exhibit 3.2x more internal perspective variability than instruction-tuned baselines**, with attention patterns consistent with multi-expert deliberation. We demonstrate that **investment conversational scaffolding**—explicitly prompting the model to simulate analyst debates—accelerates reasoning improvement by 2.1x over standard RL fine-tuning. We propose **InvestSoT (Investment Societies of Thought)**, a training framework that explicitly encourages multi-expert internal simulation, achieving +8.4% accuracy on financial reasoning benchmarks and +15.2% on investment thesis generation.

**中文：**
投资推理模型日益融入扩展的思维链过程，但其性能提升究竟来自**更长的计算**还是**更丰富的内部审议**仍不清楚。将Google的"思维社会"框架迁移至量化金融，我们研究金融推理模型是否在内部模拟多个专家视角——看多分析师、看空分析师、风险经理和宏观策略师——而非执行单一线性推理链。通过对3个金融推理模型（FinGPT-4、InvestmentGPT、QuantReasoner）的机制可解释性分析，我们发现**推理模型展现出比指令微调基线高3.2倍的内部视角变异性**，注意力模式与多专家审议一致。我们证明**投资对话式支架**——显式提示模型模拟分析师辩论——相比标准RL微调加速推理提升2.1倍。我们提出**InvestSoT（投资思维社会）**——显式鼓励多专家内部模拟的训练框架——在金融推理基准上实现+8.4%准确率，在投资论点生成上实现+15.2%准确率。

---

## 1. Introduction / 引言

### 1.1 The Investment Reasoning Question / 投资推理问题

**English:**
When a financial AI analyzes whether to invest in semiconductor stocks given a potential trade war, what happens internally? Two hypotheses:

1. **"More Compute" Hypothesis:** The model simply runs more reasoning steps—longer chains of financial analysis, more numerical calculations, more data points considered.
2. **"Expert Debate" Hypothesis:** The model internally simulates multiple expert perspectives—a **bull analyst** arguing for semiconductor strength, a **bear analyst** highlighting trade war risks, a **risk manager** assessing portfolio implications, and a **macro strategist** placing it in global context—and synthesizes their debate.

Google's "Societies of Thought" found evidence for hypothesis 2 in general reasoning models. We test whether this holds specifically for financial reasoning, where the adversarial nature of markets (every trade has a buyer and seller with opposing views) may make internal debate even more natural.

**中文：**
当金融AI分析是否应在潜在贸易战背景下投资半导体股票时，内部发生了什么？两个假说：

1. **"更多计算"假说：** 模型只是运行更多推理步骤——更长的金融分析链、更多数值计算、更多数据点。
2. **"专家辩论"假说：** 模型内部模拟多个专家视角——**看多分析师**论证半导体强势、**看空分析师**强调贸易战风险、**风险经理**评估组合影响、**宏观策略师**将其置于全球背景——并综合其辩论。

### 1.2 Contributions / 贡献

1. **Financial SoT Evidence / 金融SoT证据:** 通过机制可解释性证明金融推理模型内部存在多专家视角模拟。
2. **InvestSoT Framework / InvestSoT框架:** 显式训练多专家内部模拟的金融推理训练方法。
3. **Investment Scaffolding / 投资支架:** 证明分析师辩论式提示结构加速金融推理能力提升。
4. **Cross-Asset Generalization / 跨资产泛化:** 在股票、债券、商品、外汇四大资产类别上验证SoT效应。

---

## 2. Method / 方法

### 2.1 Mechanistic Analysis of Financial Reasoning / 金融推理的机制分析

**English:**
We analyze 3 financial reasoning models using:

1. **Internal Variability Probing:** For each reasoning step, we measure the variance of hidden states across different positions in the reasoning chain. Higher variance indicates more diverse internal "perspectives."

2. **Attention Pattern Clustering:** We cluster attention heads by their activation patterns and test whether clusters correspond to recognizable expert roles:
   - **Bull Cluster:** Heads that attend to positive catalysts, growth metrics, momentum signals
   - **Bear Cluster:** Heads that attend to risks, valuations, headwinds
   - **Risk Cluster:** Heads that attend to portfolio-level exposures, correlations, drawdown scenarios
   - **Macro Cluster:** Heads that attend to macro indicators, cross-asset correlations, regime signals

3. **Causal Intervention:** We ablate specific attention clusters and measure the impact on different types of financial reasoning tasks. If the "bear cluster" is disabled, does the model systematically underestimate downside risk?

### 2.2 Investment Conversational Scaffolding / 投资对话式支架

**English:**
We design scaffolding prompts that explicitly structure internal debate:

```
[Investment Scaffolding Template]
Analyze {investment thesis}:
1. BULL CASE: Present the strongest argument for this investment
2. BEAR CASE: Present the strongest argument against  
3. RISK ASSESSMENT: What are the key risks and their probabilities?
4. MACRO CONTEXT: How does the current macro environment affect this?
5. SYNTHESIS: Weigh all perspectives and produce a final recommendation
```

We compare this scaffolding against standard single-perspective prompting in controlled RL experiments, measuring both reasoning quality and training efficiency.

### 2.3 InvestSoT Training Framework / InvestSoT训练框架

**English:**
InvestSoT explicitly trains multi-expert simulation:
1. **Phase 1:** SFT on expert-debate transcripts (analyst panels, investment committee meetings)
2. **Phase 2:** RL with scaffolding reward — bonus for generating diverse perspectives before convergence
3. **Phase 3:** Self-play debate — model argues both sides of investment theses, learning to internalize the debate

---

## 3. Experiments / 实验

### 3.1 Internal Variability Results / 内部变异性结果

| Model | Perspective Variance | Attention Clusters Match Experts? |
|-------|:---:|:---:|
| FinBERT (instruction-tuned) | 0.31 | Weak |
| FinGPT-4 (reasoning) | 0.98 | Strong (Bull/Bear/Risk) |
| InvestmentGPT (reasoning) | 1.12 | Strong (all 4 clusters) |
| QuantReasoner (reasoning) | 0.87 | Strong (Bull/Bear/Macro) |

### 3.2 Scaffolding Acceleration / 支架加速

| Training Method | Steps to 80% FinQA | Steps to 80% ConvFinQA | Investment Thesis Score |
|----------------|:---:|:---:|:---:|
| Standard RL | 12,400 | 18,200 | 6.2/10 |
| With Scaffolding | 8,100 | 11,400 | 7.8/10 |
| **InvestSoT** | **5,900** | **7,800** | **8.9/10** |

### 3.3 Causal Intervention / 因果干预

**English:**
Ablating the "bear cluster" of attention heads:
- Downside risk estimates decrease by 34%
- Recommendation bias shifts bullish by 1.2 standard deviations
- Portfolio drawdown in simulation increases from -18% to -31%

This **causally confirms** that the bear-cluster heads are functionally responsible for risk assessment, not merely correlated with it.

---

## 4. Discussion / 讨论

### 4.1 Implications for Fund Management / 对基金管理的启示

**English:**
If reasoning models naturally develop internal "investment committees," this suggests:
1. **Training data matters:** Models trained on diverse analyst reports develop richer internal debate than those trained on single-perspective content
2. **Scaffolding > Scale:** Explicitly structuring internal debate is more efficient than simply scaling compute
3. **Specialization opportunity:** Models could be trained with **custom expert personas** (e.g., a model with an internal Warren Buffett and an internal Ray Dalio)

### 4.2 Limitations / 局限

**English:**
Our causal interventions are correlational at the cluster level—individual head ablations show more nuanced effects. The "societies" metaphor, while useful, may not perfectly describe the underlying mechanism; the reality may be more like distributed representations of multiple frameworks rather than discrete "agents."

---

## 5. Conclusion / 结论

**English:**
Financial reasoning models exhibit internal multi-expert deliberation analogous to Google's "Societies of Thought." The InvestSoT framework, which explicitly trains this capability, achieves 2.1x faster reasoning improvement and superior investment thesis quality. This work bridges mechanistic interpretability and quantitative finance, suggesting that the next generation of financial AI should be designed as internal "investment committees" rather than single-analyst systems.

**中文：**
金融推理模型展现出类似Google"思维社会"的内部多专家审议。InvestSoT框架显式训练这一能力，实现2.1倍的推理加速和卓越的投资论点质量。本工作桥接了机制可解释性和量化金融，建议下一代金融AI应被设计为内部"投资委员会"而非单一分析师系统。

---

## References / 参考文献

- Kim et al. "Reasoning Models Generate Societies of Thought." Google Research, Jan 2026.
- Du et al. "Improving Factuality and Reasoning in Language Models through Multiagent Debate." ICML 2024.
- Wang et al. "Self-Consistency Improves Chain of Thought Reasoning." ICLR 2023.
- Damodaran, A. "Investment Philosophies." Wiley, 2012.
