# CLIP 在量化金融中的应用 / CLIP Applications in Quantitative Finance

## 概述 / Overview

CLIP (Contrastive Language-Image Pre-training) 的核心能力是**对齐不同模态的表示空间**。
将这一能力迁移到量化金融领域，可以实现"金融多模态Alpha信号融合"。

CLIP's core ability is **aligning representation spaces across different modalities**.
Transferring this to quantitative finance enables "Financial Multimodal Alpha Signal Fusion".

---

## 三大应用方向 / Three Application Directions

### 1. 多模态Alpha信号 / Multimodal Alpha Signals

**核心思想 / Core Idea:**
将新闻文本、K线图表、因子数据映射到同一嵌入空间，通过对比学习
让模型理解"什么样的新闻对应什么样的价格走势"。

Map news text, price charts, and factor data into a shared embedding space,
using contrastive learning so the model understands "what kind of news
corresponds to what kind of price movement".

**技术实现 / Technical Implementation:**
- NewsEncoder: 多尺度1D-CNN编码金融新闻 / Multi-scale 1D-CNN encodes financial news
- PriceChartEncoder: CNN编码OHLCV K线数据 / CNN encodes OHLCV candlestick data
- FactorEncoder: MLP编码多因子截面 / MLP encodes cross-sectional factor data
- 三模态InfoNCE对比损失 / Tri-modal InfoNCE contrastive loss

**Alpha信号生成 / Alpha Signal Generation:**
```
alpha = w1 * normalize(news_emb) + w2 * normalize(chart_emb) + w3 * normalize(factor_emb)
```
加权融合后的向量可作为综合Alpha信号输入到组合优化器。
The weighted-fused vector serves as a comprehensive alpha signal
for portfolio optimization.

### 2. 另类数据融合 / Alternative Data Fusion

**核心思想 / Core Idea:**
量化投资中越来越多的"另类数据" (卫星图像、社交媒体情绪、供应链数据等)
需要统一的表示框架。CLIP的对比学习范式天然适合将异构数据对齐。

Quantitative investment increasingly uses "alternative data" (satellite imagery,
social media sentiment, supply chain data, etc.) that requires a unified
representation framework. CLIP's contrastive learning paradigm naturally
suits aligning heterogeneous data.

**应用场景 / Application Scenarios:**
- 卫星图像 + 零售收入预测 / Satellite imagery + retail revenue prediction
- 社交媒体情绪 + 短期动量 / Social media sentiment + short-term momentum
- 供应链图谱 + 行业轮动 / Supply chain graphs + sector rotation
- ESG报告文本 + 风险因子 / ESG report text + risk factors

**优势 / Advantages:**
- 无需手动特征工程 / No manual feature engineering required
- 不同模态间可交叉检索 / Cross-modal retrieval between different data types
- 新模态可通过添加编码器轻松接入 / New modalities easily added via new encoders

### 3. 新闻情绪量化 / News Sentiment Quantification

**核心思想 / Core Idea:**
传统NLP情感分析只给出正/负/中三分类 (粗粒度)。
Finance CLIP可以将新闻映射到连续的情绪空间，
通过与"原型情绪"的余弦距离得到细粒度的情绪分数。

Traditional NLP sentiment analysis only gives positive/negative/neutral
(coarse-grained). Finance CLIP maps news into a continuous sentiment
space, yielding fine-grained sentiment scores via cosine distance
to "prototype sentiments".

**原型情绪 / Prototype Sentiments:**
| 原型 / Prototype | 含义 / Meaning | 交易含义 / Trading Implication |
|---|---|---|
| Bullish | 强烈看多 / Strongly bullish | 做多信号 / Long signal |
| Bearish | 强烈看空 / Strongly bearish | 做空/减仓信号 / Short/reduce signal |
| Uncertain | 高度不确定 / Highly uncertain | 减小仓位 / Reduce position |
| Neutral | 信息中性 / Information-neutral | 保持现状 / Hold current position |

**综合情绪指标 / Composite Sentiment Score:**
```
sentiment_score = cos(news, bullish_prototype) - cos(news, bearish_prototype)
```
范围约 [-2, 2]，可直接作为因子输入。
Range approximately [-2, 2], directly usable as a factor input.

---

## 文件结构 / File Structure

```
quant/
  README.md          — 本文件 / This file
  quant_train.py     — Finance CLIP 训练脚本 / Finance CLIP training script
  quant_evaluate.py  — Finance CLIP 评估脚本 / Finance CLIP evaluation script
```

## 运行方式 / How to Run

```bash
# 训练 / Training
python quant/quant_train.py

# 评估 / Evaluation
python quant/quant_evaluate.py
```

## 依赖 / Dependencies

```bash
pip install torch numpy
```

## 参考文献 / References

- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)
- Lopez de Prado, "Advances in Financial Machine Learning" (2018)
- Gu, Kelly, Xiu, "Empirical Asset Pricing via Machine Learning" (2020)
