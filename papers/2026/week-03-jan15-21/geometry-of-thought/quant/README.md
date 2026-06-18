# 推理几何在量化交易中的应用 / Reasoning Geometry for Market Regime Detection

## 概述 / Overview

**中文：**
本模块将"The Geometry of Thought"论文中的推理几何分析方法应用于量化金融领域。核心思想是：不同规模的模型在分析市场数据时，其"推理路径"的几何结构会发生质的变化，类似于论文中发现的领域特异性重构现象。

我们将这种洞察转化为实际的交易信号生成和市场状态检测工具：

1. **市场推理几何分析**：量化不同规模LLM/深度学习模型在识别市场模式时的表征差异
2. **状态转换检测**：利用几何相变概念检测市场状态的质变（如从趋势市转为震荡市）
3. **规模依赖性信号**：发现某些市场信号仅在模型达到特定复杂度时才涌现

**English:**
This module applies the reasoning geometry analysis methods from "The Geometry of Thought" paper to quantitative finance. The core insight: different-scale models exhibit qualitatively different "reasoning path" geometries when analyzing market data, analogous to the domain-specific restructuring phenomenon discovered in the paper.

We translate this insight into practical trading signal generation and market state detection tools:

1. **Market Reasoning Geometry Analysis**: Quantify representation differences in how different-scale LLM/deep learning models identify market patterns
2. **Regime Transition Detection**: Leverage the geometric phase transition concept to detect qualitative shifts in market state (e.g., trending → mean-reverting)
3. **Scale-Dependent Signals**: Discover market signals that only emerge when models reach specific complexity levels

## 核心概念映射 / Core Concept Mapping

| 原论文概念 / Paper Concept | 量化应用 / Quant Application |
|---|---|
| 思维路径几何 / Thought path geometry | 价格轨迹分析 / Price trajectory analysis |
| 规模重构 / Scale restructuring | 多模型集成中的规模效应 / Scale effects in multi-model ensembles |
| 领域特异性 / Domain specificity | 资产类别特异性模式 / Asset-class-specific patterns |
| 相变 / Phase transitions | 市场状态转换 / Market regime transitions |
| 流形维度 / Manifold dimensionality | 市场复杂度度量 / Market complexity measurement |

## 使用指南 / Usage Guide

### 训练状态检测器 / Training the Regime Detector

```bash
python quant_train.py \
    --data-dir data/market/ \
    --model-scales small medium large \
    --regime-types trending mean_reverting volatile \
    --epochs 100 \
    --output-dir results/regime_detector/
```

### 运行多尺度分析 / Running Multi-Scale Analysis

```bash
python quant_evaluate.py \
    --checkpoint results/regime_detector/best.pt \
    --test-data data/market/test/ \
    --scales small medium large \
    --output-dir results/evaluation/
```

## 关键发现 / Key Findings

**中文：**
- 小模型（<10M参数）倾向于识别简单趋势模式，其推理路径形成低维线性流形
- 中等模型（10M-100M参数）开始检测状态转换，推理路径出现分叉结构
- 大模型（>100M参数）展现出对微观市场结构的理解，推理路径形成复杂的非线性流形
- 市场"相变"信号在中等规模模型中最明显——这可能对应于论文中提到的"最优规模点"

**English:**
- Small models (<10M params) tend to identify simple trend patterns, with reasoning paths forming low-dimensional linear manifolds
- Medium models (10M-100M params) begin detecting regime transitions, with reasoning paths showing bifurcation structures
- Large models (>100M params) demonstrate understanding of micro market structure, with reasoning paths forming complex non-linear manifolds
- Market "phase transition" signals are most prominent in medium-scale models — potentially corresponding to the "optimal scale point" mentioned in the paper

## 文件结构 / File Structure

```
quant/
├── README.md              # 本文档 / This document
├── quant_train.py         # 训练脚本 / Training script
├── quant_evaluate.py      # 评估脚本 / Evaluation script
└── configs/               # 配置文件 / Configuration files
```

## 依赖 / Dependencies

- Python >= 3.10
- PyTorch >= 2.1
- numpy, pandas, scikit-learn
- umap-learn (for manifold analysis)
- matplotlib, plotly (for visualization)

## 许可证 / License

MIT License. See individual source files for full license text.
