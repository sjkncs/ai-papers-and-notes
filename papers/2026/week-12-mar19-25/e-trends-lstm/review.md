# E-TRENDS: Enhanced LSTM Trend Forecasting for Equities

> **arXiv:** [2603.14453](https://arxiv.org/abs/2603.14453) | **日期 / Date:** 2026-03-15 | **作者 / Authors:** Buchanan, Benhamou

---

## Q1: 核心问题

**中文：** 标准LSTM在股票预测中面临：1) 梯度消失导致长期依赖捕获不足 2) 趋势vs噪声难以区分 3) 单一时间尺度的局限性。E-TRENDS通过增强LSTM架构解决这些问题。

## Q2: 核心贡献

1. **趋势增强门控**: 专门的门控机制分离趋势信号和噪声
2. **多尺度输入**: 同时输入多个时间尺度的特征
3. **趋势损失函数**: 直接优化趋势预测准确性而非MSE
4. **在 equities 上超越标准LSTM/GRU/Transformer基线**

## Q3: 审稿攻击点

1. 趋势定义的主观性？ 2) 交易成本后的实际收益？ 3) 与TCN的对比？

## Q4: 量化映射

- 增强LSTM用于股票趋势跟踪策略
- 趋势门控机制可迁移到其他RNN架构
- 多尺度输入框架用于多频率因子融合
