# Autonomous AI Agents for Option Hedging: Enhancing Financial Stability through Shortfall Aware Reinforcement Learning

> **arXiv:** [2603.06587](https://arxiv.org/abs/2603.06587) | **日期 / Date:** 2026-02-01

---

## Q1: 核心问题

**中文：** 期权对冲需要实时调整Delta/Gamma/Vega等Greeks敞口，传统方法依赖Black-Scholes等模型的假设。RL代理能否在不依赖特定模型假设的情况下，自主学习最优对冲策略？

## Q2: 核心贡献

1. **Shortfall-Aware RL**: 引入对冲缺口感知的奖励函数
2. **无模型假设**: 不依赖BS公式，直接从市场数据学习
3. **交易成本内嵌**: 将交易成本纳入RL环境
4. **多资产对冲**: 同时管理多个期权头寸的Greeks

## Q3: 审稿攻击点

1. RL训练稳定性？ 2) 极端市场条件下的鲁棒性？ 3) 与Delta对冲的公平对比？

## Q4: 量化映射

- RL对冲 → 结构化产品的风险管理
- Shortfall-aware奖励 → 尾部风险控制
- 无模型方法 → 奇异期权对冲
