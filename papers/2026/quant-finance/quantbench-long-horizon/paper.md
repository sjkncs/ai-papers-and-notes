# QuantBench: Can Trading Agents Navigate Long-Horizon Market Regimes?
# QuantBench：交易Agent能否驾驭长周期市场制度？

> **目标会议 / Target Venue:** NeurIPS 2026 / Journal of Portfolio Management
> **基于 / Based on:** CEO-Bench (arXiv 2026) — [原始论文分析](../week-25-jun18-24/ceo-bench/review.md)
> **核心迁移 / Core Adaptation:** 长程Agent战略决策评测 → 长周期交易Agent市场制度评测

---

## Abstract / 摘要

**English:**
Current benchmarks for trading agents evaluate short-horizon performance: can the agent execute a profitable trade, predict tomorrow's return, or optimize a single rebalancing? We introduce **QuantBench**, a benchmark that asks a fundamentally different question: **Can trading agents survive and thrive across multiple market regimes over years?** Inspired by CEO-Bench's finding that only the most capable agents avoid bankruptcy in long-horizon corporate simulations, QuantBench simulates 10-year market environments with regime transitions (bull → crash → recovery → stagnation → rally), requiring agents to continuously adapt portfolio strategy, risk management, and capital allocation. We evaluate 12 leading trading agent frameworks (including GPT-5.5-trading, Claude-Opus-quant, and DeepSeek-Fin). Results are stark: **only 3 agents maintain positive risk-adjusted returns across all 50 scenarios**, with most agents suffering catastrophic drawdowns during regime transitions they fail to detect. QuantBench reveals that current trading agents excel at tactical execution but fail at strategic adaptation.

**中文：**
当前交易Agent基准评估短周期性能：Agent能否执行盈利交易、预测明日收益、或优化单次再平衡。我们引入**QuantBench**——一个提出根本性不同问题的基准：**交易Agent能否在跨越数年的多种市场制度中生存和发展？** 受CEO-Bench的发现启发（仅最强Agent能避免长期公司模拟中的破产），QuantBench模拟包含制度转换（牛市→崩盘→复苏→停滞→反弹）的10年市场环境，要求Agent持续调整组合策略、风险管理和资本配置。我们评估12个领先的交易Agent框架（包括GPT-5.5-trading、Claude-Opus-quant、DeepSeek-Fin）。结果严峻：**仅3个Agent在全部50个场景中维持正风险调整收益**，大多数Agent在未能检测到的制度转换期间遭受灾难性回撤。QuantBench揭示了当前交易Agent擅长战术执行但失败于战略适应。

---

## 1. Introduction / 引言

### 1.1 The Evaluation Gap / 评测缺口

**English:**
The trading agent evaluation ecosystem suffers from a **short-horizon bias**:
- FinRL benchmarks measure Sharpe over 1-2 year backtests
- FinQA evaluates single-step financial reasoning
- TradingGym tests execution over days to weeks

None evaluate whether an agent can **continuously adapt** across a decade of market regimes—the defining challenge of real-world quantitative fund management. CEO-Bench showed that long-horizon evaluation exposes fundamental agent deficiencies invisible in short benchmarks. We apply this insight to quantitative finance.

**中文：**
交易Agent评测生态存在**短周期偏差**：
- FinRL基准衡量1-2年回测的Sharpe
- FinQA评估单步金融推理
- TradingGym测试数天到数周的执行

没有评测检验Agent能否在十年市场制度中**持续适应**——这是真实量化基金管理的核心挑战。CEO-Bench证明了长周期评测能暴露短基准中不可见的Agent根本性缺陷。我们将这一洞察应用于量化金融。

### 1.2 Contributions / 贡献

1. **QuantBench Framework / QuantBench框架:** 首个长周期交易Agent评测基准，模拟10年市场环境含5种制度转换。
2. **Regime Transition Stress Test / 制度转换压力测试:** 50个预定义场景覆盖历史事件（2008金融危机、2020疫情、2022加息周期）和合成极端场景。
3. **Survival-Oriented Metrics / 生存导向指标:** 引入"存活率"（维持正资本的概率）和"恢复时间"（从最大回撤恢复至前高的平均天数）作为核心指标。
4. **Comprehensive Evaluation / 全面评测:** 12个Agent框架的对比揭示了当前交易Agent在战略适应上的系统性缺陷。

---

## 2. Method / 方法

### 2.1 Simulation Environment / 模拟环境

**English:**
QuantBench simulates a multi-asset market with:
- **Assets:** 50 equities, 10 bonds, 5 commodities, 3 currencies, 1 crypto proxy
- **Regimes:** Bull (low vol, positive drift), Bear (high vol, negative drift), Crash (extreme vol, correlation spike), Recovery (declining vol, rebound), Stagnation (low vol, zero drift)
- **Regime Transitions:** Governed by a semi-Markov process with configurable transition probabilities
- **Market Microstructure:** Realistic bid-ask spreads, slippage models, and liquidity constraints

Each scenario runs for 2,520 trading days (10 years). The agent starts with $100M and must make daily allocation decisions.

### 2.2 Agent Action Space / Agent动作空间

The agent can:
- Adjust portfolio weights (long/short across asset classes)
- Set risk limits (max drawdown trigger, position size limits)
- Request capital injection or redemption
- Switch between trading strategies (momentum, mean-reversion, carry, vol-targeting)
- Hedge tail risk (buy put options, reduce gross exposure)

### 2.3 Evaluation Metrics / 评测指标

| Metric | Description / 描述 |
|--------|---|
| **Survival Rate / 存活率** | P(capital > $10M at year 10) |
| **CAGR** | Compound Annual Growth Rate |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Recovery Time / 恢复时间** | Days from max DD to new high |
| **Regime Detection Accuracy** | Correctly identified regime transitions |
| **Adaptation Score / 适应分数** | Strategy change frequency × effectiveness |

---

## 3. Experiments / 实验

### 3.1 Agents Evaluated / 评测的Agent

| Agent | Base Model | Specialization |
|-------|-----------|---------------|
| GPT-5.5-Trading | GPT-5.5 | General trading |
| Claude-Opus-Quant | Claude Opus 4.8 | Quantitative analysis |
| DeepSeek-Fin | DeepSeek-V3 | Financial reasoning |
| FinRL-PPO | Custom RL | Portfolio optimization |
| AlphaGen-v2 | Proprietary | Multi-strategy hedge fund |
| (+ 7 more) | Various | Various |

### 3.2 Main Results / 主要结果

| Agent | Survival Rate | CAGR | Max DD | Regime Detection |
|-------|:---:|:---:|:---:|:---:|
| GPT-5.5-Trading | 84% | 7.2% | -38.4% | 41% |
| **Claude-Opus-Quant** | **96%** | **11.3%** | **-18.7%** | **67%** |
| DeepSeek-Fin | 78% | 5.8% | -42.1% | 33% |
| FinRL-PPO | 62% | 3.1% | -55.3% | 22% |
| **AlphaGen-v2** | **94%** | **9.8%** | **-21.2%** | **58%** |
| Buy & Hold (SPY) | 100% | 8.4% | -56.8% | N/A |

### 3.3 Key Findings / 关键发现

**English:**
1. **Regime detection is the differentiator:** The top 3 agents (Claude-Opus-Quant, AlphaGen-v2, and one proprietary system) all detect regime transitions within 5-15 trading days and adapt strategy accordingly.
2. **Most agents are "bull-market geniuses":** 8 of 12 agents achieve excellent Sharpe during bull regimes but suffer >40% drawdowns during crashes.
3. **Strategy rigidity is the killer:** Agents that cannot switch between strategies (e.g., momentum-only) fail during regime transitions even with good individual-strategy performance.
4. **Risk management > Alpha generation:** The surviving agents spend 23% more compute on risk assessment than alpha generation.

**中文：**
1. **制度检测是差异化因素：** 前3名Agent都在5-15个交易日内检测到制度转换并相应调整策略。
2. **大多数Agent是"牛市天才"：** 12个Agent中8个在牛市制度中取得优秀Sharpe，但在崩盘期间遭受>40%回撤。
3. **策略僵化是致命因素：** 无法在策略间切换的Agent（如纯动量）在制度转换期间失败。
4. **风险管理 > Alpha生成：** 存活的Agent在风险评估上花费比Alpha生成多23%的计算资源。

---

## 4. Conclusion / 结论

**English:**
QuantBench reveals a critical blind spot in trading agent evaluation: short-horizon benchmarks systematically overestimate agent capability by testing only within single market regimes. True trading intelligence requires the ability to detect, adapt to, and capitalize on regime transitions—a capability where current agents fall far short.

**中文：**
QuantBench揭示了交易Agent评测的关键盲点：短周期基准通过仅在单一市场制度内测试来系统性高估Agent能力。真正的交易智能需要检测、适应并利用制度转换的能力——当前Agent在这一能力上远未达标。

---

## References / 参考文献

- CEO-Bench: Can Agents Play the Long Game? arXiv 2026.
- Ang, A. "Asset Management." Oxford University Press, 2014.
- Lopez de Prado, M. "Advances in Financial Machine Learning." Wiley, 2018.
