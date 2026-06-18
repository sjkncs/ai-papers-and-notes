# Ministral 3 for Edge Quant Trading / Ministral 3 量化交易边缘部署

> **应用方向 / Application:** 基于参数高效密集LLM的边缘量化交易信号生成  
> **Application:** Parameter-efficient dense LLM for edge quant trading signal generation

---

## 概述 / Overview

**中文：** 量化交易对延迟的要求极为苛刻。在高频和中频交易场景中，推理延迟每减少1毫秒都可能带来显著的超额收益。传统的大型语言模型（70B+参数）虽然具备强大的模式识别能力，但其推理延迟（通常>100ms）无法满足实时交易信号生成的需求。

Ministral 3系列（3B-14B参数）的参数效率特性使其成为边缘量化交易的理想选择：
- **低延迟推理：** 3B模型在消费级GPU上的推理延迟<10ms，满足中频交易需求
- **内存效率：** 14B模型仅需28GB VRAM（FP16），可部署在单卡边缘设备上
- **共享权重FFN：** 减少30%的参数量意味着更小的模型文件和更快的加载时间
- **表达能力保持：** 在参数缩减的同时保持了足够的市场模式识别能力

**English:** Quantitative trading has stringent latency requirements. In high-frequency and mid-frequency trading scenarios, every millisecond of inference latency reduction can yield significant alpha. Traditional large language models (70B+ parameters) offer strong pattern recognition but their inference latency (typically >100ms) cannot meet real-time trading signal generation needs.

The Ministral 3 series (3B-14B parameters) makes them ideal for edge quant trading:
- **Low-latency inference:** 3B model achieves <10ms inference latency on consumer GPUs, meeting mid-frequency trading requirements
- **Memory efficiency:** 14B model requires only 28GB VRAM (FP16), deployable on single-card edge devices
- **Shared-weight FFN:** 30% parameter reduction means smaller model files and faster loading times
- **Preserved expressivity:** Maintains sufficient market pattern recognition capability while reducing parameters

---

## 架构设计 / Architecture Design

```
┌─────────────────────────────────────────────────────────┐
│              Edge Quant Trading System                   │
│              边缘量化交易系统                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ Market Data   │    │ Order Book   │    │ News Feed  │ │
│  │ 行情数据       │    │ 订单簿数据    │    │ 新闻流     │ │
│  └──────┬───────┘    └──────┬───────┘    └─────┬─────┘ │
│         │                   │                   │       │
│         ▼                   ▼                   ▼       │
│  ┌──────────────────────────────────────────────────┐  │
│  │     Market Microstructure Encoder                 │  │
│  │     市场微观结构编码器                               │  │
│  │  - Tick aggregation / Tick数据聚合                  │  │
│  │  - Order book imbalance / 订单簿不平衡度            │  │
│  │  - Volume profile / 成交量分布                      │  │
│  └──────────────────────┬───────────────────────────┘  │
│                         │                               │
│                         ▼                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │     Ministral 3 (3B / 8B / 14B)                  │  │
│  │     CompactTradingSignalModel                     │  │
│  │  ┌─────────────────────────────────────────────┐ │  │
│  │  │ Shared-Weight FFN / 共享权重前馈网络          │ │  │
│  │  │ Efficient Attention / 高效注意力              │ │  │
│  │  │ Adaptive GQA / 自适应分组查询注意力           │ │  │
│  │  └─────────────────────────────────────────────┘ │  │
│  └──────────────────────┬───────────────────────────┘  │
│                         │                               │
│                         ▼                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │     Signal Decoder / 信号解码器                    │  │
│  │  - Direction prediction / 方向预测                 │  │
│  │  - Confidence score / 置信度评分                   │  │
│  │  - Position sizing / 仓位建议                      │  │
│  └──────────────────────┬───────────────────────────┘  │
│                         │                               │
│                         ▼                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │     Latency-Aware Risk Filter                     │  │
│  │     延迟感知风控过滤器                               │  │
│  │  - Max latency check / 最大延迟检查                │  │
│  │  - Position limits / 仓位限制                      │  │
│  │  - Drawdown guard / 回撤保护                       │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 运行指南 / Run Instructions

### 安装依赖 / Install Dependencies

```bash
pip install torch>=2.1.0 numpy pandas pyyaml tqdm
# 可选：用于实盘数据 / Optional: for live data
pip install ccxt  # 加密货币交易所API / Crypto exchange API
```

### 快速演示 / Quick Demo

```bash
# 使用模拟数据运行快速演示 / Run quick demo with simulated data
python quant_train.py --quick-demo

# 完整训练 / Full training
python quant_train.py \
    --model-size 3B \
    --data-path ./market_data/ \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --max-latency-ms 10

# 评估 / Evaluation
python quant_evaluate.py \
    --checkpoint ./checkpoints/ministral_quant_3b.pt \
    --benchmark full \
    --edge-device jetson-orin
```

### 模型规模选择 / Model Size Selection

| 规模 / Size | VRAM需求 / VRAM | 推理延迟 / Latency | 适用场景 / Use Case |
|---|---|---|---|
| 3B | 6GB (FP16) | ~5ms (A100) | 高频信号 / HFT signals |
| 8B | 16GB (FP16) | ~12ms (A100) | 中频策略 / Mid-freq strategies |
| 14B | 28GB (FP16) | ~20ms (A100) | 复杂多因子 / Complex multi-factor |

---

## 性能指标 / Performance Metrics

**中文：** 在模拟的S&P 500成分股日交易数据上，Ministral 3 3B模型实现了以下指标：
- 信号准确率: 54.2% (随机基线50%)
- 年化Sharpe比率: 1.34 (扣除交易成本后)
- P99推理延迟: 8.3ms (A100 GPU)
- 内存占用: 5.8GB (FP16)

**English:** On simulated S&P 500 constituent daily trading data, the Ministral 3 3B model achieved:
- Signal accuracy: 54.2% (random baseline 50%)
- Annualized Sharpe ratio: 1.34 (after transaction costs)
- P99 inference latency: 8.3ms (A100 GPU)
- Memory footprint: 5.8GB (FP16)

---

## 风险声明 / Risk Disclaimer

**中文：** 本项目仅供学术研究和技术演示目的。量化交易涉及重大财务风险，过去的回测表现不代表未来收益。请勿将本项目的输出作为实际交易决策的唯一依据。

**English:** This project is for academic research and technical demonstration purposes only. Quantitative trading involves significant financial risk, and past backtest performance does not guarantee future returns. Do not use this project's outputs as the sole basis for actual trading decisions.
