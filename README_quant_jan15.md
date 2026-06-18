# AI Papers and Notes (2026-01-15 ~ 2026-01-21)

> 每周 AI 论文追踪 + 量化金融代码复现

## 本周更新概览 (2026-01-15 ~ 2026-01-21)

### 通用 AI 论文
| 日期 | 论文 | 方向 | 量化相关度 |
|------|------|------|----------|
| 01-13 | [Ministral 3](papers/2601.08584_ministral3.md) | 高效密集LLM | ★★ |
| 01-15 | [Are Reasoning Models Reasoning or Guessing?](papers/2601.10679_reasoning_or_guessing.md) | 推理模型分析 | ★★ |
| 01-16 | [MARO: Reasoning from Social Interaction](papers/2601.12323_maro.md) | 社交推理学习 | ★ |
| 01-17 | [Teaching LRMs Effective Reflection](papers/2601.12720_effective_reflection.md) | 推理反思训练 | ★★ |

### 量化金融论文
| 日期 | 论文 | 子领域 | 代码复现 |
|------|------|--------|----------|
| 01-15 | [AlphaCFG: Alpha Discovery via Grammar-Guided Search](papers/2601.22119_alphacfg.md) | 因子挖掘 | [code/](code/alpha_factor_mining/alphacfg.py) |
| 01-08 | [Hybrid Transformer GNN for Correlation Forecasting](papers/2601.04602_correlation_gnn.md) | 时序预测 | [code/](code/time_series/correlation_gnn.py) |
| 01-06 | [PriceSeer: LLMs for Stock Prediction](papers/2601.06088_priceseer.md) | 时序预测 | [code/](code/time_series/llm_stock_predict.py) |
| 01-07 | [Non-Convex Portfolio via Energy-Based Models](papers/2601.07792_ebm_portfolio.md) | 组合优化 | [code/](code/portfolio_optimization/ebm_portfolio.py) |
| 01-12 | [Enhancing Portfolio with Deep Learning](papers/2601.07942_dl_portfolio.md) | 组合优化 | [code/](code/portfolio_optimization/transformer_portfolio.py) |
| 01-09 | [Smart Predict-then-Optimize for Portfolios](papers/2601.04062_spo_portfolio.md) | 组合优化 | [code/](code/portfolio_optimization/spo_portfolio.py) |
| 01-15 | [Deep g-Pricing for CSI 300 Options](papers/2601.18804_deep_g_pricing.md) | 衍生品定价 | [code/](code/portfolio_optimization/deep_option_pricing.py) |
| 01-15 | [Regret-Driven LLM Portfolio Allocation](papers/2601.17021_llm_portfolio.md) | NLP/组合 | [code/](code/nlp_finance/llm_portfolio.py) |
| 01-11 | [LLaMA-3-8B for Financial NER](papers/2601.10043_financial_ner.md) | NLP | [code/](code/nlp_finance/financial_ner.py) |
| 01-07 | [Feature Engineering Beats DL in Flow Prediction](papers/2601.07131_feature_vs_dl.md) | 因子工程 | [code/](code/alpha_factor_mining/feature_beats_dl.py) |

## 项目结构
```
ai-papers-and-notes/
├── README.md
├── requirements.txt
├── papers/                          # 论文阅读笔记
│   ├── 2601.22119_alphacfg.md
│   ├── 2601.04602_correlation_gnn.md
│   ├── ...
├── code/                            # 代码复现
│   ├── alpha_factor_mining/         # 因子挖掘
│   │   ├── alphacfg.py             # AlphaCFG 复现
│   │   └── feature_beats_dl.py     # 特征工程 vs DL
│   ├── time_series/                 # 时序预测
│   │   ├── correlation_gnn.py      # Transformer+GNN 相关性预测
│   │   └── llm_stock_predict.py    # LLM 股票预测
│   ├── portfolio_optimization/      # 组合优化
│   │   ├── ebm_portfolio.py        # 能量模型组合优化
│   │   ├── transformer_portfolio.py# Transformer 资产配置
│   │   ├── spo_portfolio.py        # Predict-then-Optimize
│   │   └── deep_option_pricing.py  # 深度学习期权定价
│   └── nlp_finance/                 # 金融NLP
│       ├── financial_ner.py        # 金融命名实体识别
│       └── llm_portfolio.py        # LLM驱动组合分配
└── quant_utils/                     # 量化工具库
    ├── data_loader.py              # 数据加载
    └── backtest.py                 # 回测框架
```

## 环境配置
```bash
pip install -r requirements.txt
```

## 参考来源
- [arXiv cs.AI](https://arxiv.org/list/cs.AI/2026)
- [arXiv q-fin](https://arxiv.org/list/q-fin/2026)
- [dair-ai/AI-Papers-of-the-Week](https://github.com/dair-ai/AI-Papers-of-the-Week)
- [Sebastian Raschka: LLM Research Papers 2026](https://magazine.sebastianraschka.com/p/llm-research-papers-2026-part1)
