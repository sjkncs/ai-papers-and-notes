# Deep Delta Learning x 量化金融 / DDL for Quantitative Finance

> **论文 / Paper:** "Deep Delta Learning" (arXiv:2601.00417)
> **作者 / Authors:** Yifan Zhang, Yifeng Liu, Mengdi Wang, Quanquan Gu

---

## 核心理念 / Core Philosophy

Deep Delta Learning (DDL) 将残差连接从"只能累加"升级为"可选择性重写"。
这一特性在量化金融中有独特的应用价值——金融市场本质上是regime-switching的系统，
需要模型能够主动"遗忘"过时的市场状态信号。

DDL upgrades residual connections from "append-only" to "selective rewrite."
This has unique value in quantitative finance — financial markets are inherently
regime-switching systems that require models to actively "forget" obsolete
market state signals.

**标准残差 / Standard Residual:**
```
x_{l+1} = x_l + f_l(x_l)         # 只能累加 / Append only
```

**DDL残差 / DDL Residual:**
```
x_{l+1} = x_l + g_l(x_l) * (target_l(x_l) - x_l) + f_l(x_l)
                                         # 可重写+累加 / Rewrite + append
```

---

## 三大应用方向 / Three Application Areas

### 1. Regime-Aware 时序预测 / Regime-Aware Forecasting

**问题 / Problem:**
金融市场在不同regime之间切换（动量、均值回归、高波动），标准Transformer的
残差累加机制无法主动清除过时regime的信号。

Financial markets switch between regimes (momentum, mean-reversion, high-volatility).
Standard Transformer residual accumulation cannot actively clear stale regime signals.

**DDL解决方案 / DDL Solution:**
- 门控函数 `g(x)` 学习识别regime切换 / Gate function learns to detect regime switches
- 当regime切换时，`g(x) -> 1`，将隐状态向新regime的target值拉近
- When regime switches, `g(x) -> 1`, pulling hidden state toward new regime's target
- 当regime稳定时，`g(x) -> 0`，保留当前特征不变
- When regime is stable, `g(x) -> 0`, preserving current features

**量化意义 / Quant Significance:**
```
动量regime → 均值回归regime 切换时:
  旧动量因子信号需要被"覆写"而非继续累加
  Old momentum signals need to be "overwritten" rather than accumulated

DDL门控自动学习这种切换:
  g(动量特征) ≈ 1 → 主动清除旧动量信号
  g(均值回归特征) ≈ 0 → 保留均值回归信号
```

**实现 / Implementation:** `quant_train.py` 中的 `DDLTimeSeriesForecaster`

---

### 2. MoE因子路由 / MoE Factor Routing

**问题 / Problem:**
不同市场状态下，有效的因子组合不同。传统方法对所有regime使用相同的因子权重，
或需要手动切换因子策略。

Different factor combinations are effective under different market conditions.
Traditional methods use the same factor weights for all regimes, or require
manual strategy switching.

**DDL启发 / DDL Inspiration:**
借鉴DDL的"选择性重写"思想:
- 多个因子expert (动量专家、价值专家、波动率专家)
- Router根据当前市场状态（类似DDL的gate）动态选择激活哪些expert
- 未激活的expert不参与计算 (类似DDL中gate≈0时不做重写)

Borrowing DDL's "selective rewrite" concept:
- Multiple factor experts (momentum, value, volatility experts)
- Router dynamically selects experts based on current market state (analogous to DDL's gate)
- Inactive experts don't participate in computation (like DDL when gate ≈ 0)

**量化意义 / Quant Significance:**
```
Expert 1: 动量因子专家 (趋势市) → 趋势延续信号
Expert 2: 价值因子专家 (均值回归市) → 反转信号
Expert 3: 波动率因子专家 (高波动市) → 风险控制信号
Router: 根据市场状态动态选择 → 自适应因子组合
```

**实现 / Implementation:** `quant_train.py` 中的 `MoEFactorRouter`

---

### 3. 端到端投资组合优化 / End-to-End Portfolio Optimization

**问题 / Problem:**
传统投资组合优化分两步: (1) 预测收益 (2) 均值-方差优化。
两步分离导致预测目标与最终投资目标不一致。

Traditional portfolio optimization is two-step: (1) predict returns
(2) mean-variance optimization. The separation causes prediction objective
misalignment with the ultimate investment objective.

**DDL解决方案 / DDL Solution:**
- DDL时序编码器作为收益预测模块 (利用regime-aware特性)
- 端到端联合优化: 因子数据 → DDL编码 → 组合权重
- 约束投影层确保权重满足投资约束 (非负、和为1、最大仓位)

- DDL time-series encoder as return prediction module (leveraging regime-awareness)
- End-to-end joint optimization: factor data → DDL encoding → portfolio weights
- Constraint projection layer ensures weights satisfy investment constraints

**量化意义 / Quant Significance:**
```
传统方法:
  因子数据 → [收益预测模型] → 预测收益 → [优化器] → 组合权重
  (两步分离，目标不一致)

DDL端到端:
  因子数据 → [DDL编码器 + 权重网络] → 组合权重
  (联合优化，目标一致)

关键优势: DDL的regime感知能力使得预测在不同市场状态下都有效
Key advantage: DDL's regime awareness keeps predictions valid across market states
```

**实现 / Implementation:** `quant_train.py` 中的 `DDLPortfolioOptimizer`

---

## 文件结构 / File Structure

```
quant/
├── README.md              # 本文件 / This file
├── quant_train.py         # 完整训练脚本 / Full training script
│   ├── DDLTimeSeriesForecaster  # DDL时序预测器
│   ├── MoEFactorRouter          # MoE因子路由器
│   ├── DDLPortfolioOptimizer    # DDL投资组合优化器
│   └── simulate_market_data()   # 模拟市场数据生成
└── quant_evaluate.py      # 回测评估脚本 / Backtest evaluation
    ├── Sharpe ratio 计算
    ├── Max drawdown 计算
    ├── Regime检测准确率
    └── DDL vs 标准Transformer对比
```

## 运行 / Run

```bash
# 训练 / Train
python quant/quant_train.py

# 评估 / Evaluate
python quant/quant_evaluate.py

# 或快速演示 / Or quick demo
python quant/quant_train.py --quick-demo
python quant/quant_evaluate.py --quick-demo
```

## 依赖 / Dependencies

```bash
pip install torch numpy
```

---

## 关键参考文献 / Key References

- Deep Delta Learning: [arXiv:2601.00417](https://arxiv.org/abs/2601.00417)
- DeltaNet (related work): [Schlag et al., 2021](https://arxiv.org/abs/2102.11174)
- Mixture of Experts in Finance: MiMo-V2-Flash architecture
- Regime-Switching Models: Hamilton (1989)
