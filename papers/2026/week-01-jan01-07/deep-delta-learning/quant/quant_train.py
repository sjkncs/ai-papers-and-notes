#!/usr/bin/env python3
"""
Deep Delta Learning x 量化金融 — 完整训练脚本
DDL for Quantitative Finance — Full Training Script
=====================================================

基于论文 "Deep Delta Learning" (arXiv:2601.00417) 的量化金融应用训练脚本。
Training script for DDL quantitative finance applications.

三大应用 / Three Applications:
  1. DDL时序预测器 — Regime-aware金融时序预测
     DDL Time-Series Forecaster — Regime-aware financial time-series forecasting
  2. MoE因子路由器 — 动态因子组合
     MoE Factor Router — Dynamic factor combination
  3. DDL投资组合优化器 — 端到端组合构建
     DDL Portfolio Optimizer — End-to-end portfolio construction

包含 / Includes:
  - 模拟市场数据生成 / Simulated market data generation
  - 完整训练循环 / Complete training loop
  - 初步回测评估 / Preliminary backtest evaluation

用法 / Usage:
    python quant/quant_train.py
    python quant/quant_train.py --quick-demo
    python quant/quant_train.py --epochs 50 --lr 1e-4

Author: Auto-generated from paper analysis
License: MIT
"""

import os
import sys
import math
import time
import json
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 1. DDL时序预测器 / DDL Time-Series Forecaster
# ============================================================

class MarketRegimeGate(nn.Module):
    """
    市场状态门控 / Market Regime Gate

    学习识别当前市场状态(动量/均值回归/高波动)，
    并选择性重写隐状态中过时的regime信号。

    Learns to detect current market regime (momentum/mean-reversion/high-vol)
    and selectively rewrites stale regime signals in hidden state.

    量化意义 / Quant Significance:
      当市场从动量切换到均值回归时，旧的动量特征需要被主动覆写而非继续累加。
      When market switches from momentum to mean-reversion, old momentum
      features need active overwriting rather than continued accumulation.
    """

    def __init__(self, d_model: int, num_regimes: int = 3):
        """
        Args:
            d_model: 隐藏维度 / Hidden dimension
            num_regimes: 市场状态数 / Number of market regimes
        """
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # 门控: 决定哪些维度需要被重写 / Gate: which dimensions to rewrite
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # 目标值: 当前regime下的"正确"特征值
        # Target: the "correct" feature value under current regime
        self.target = nn.Linear(d_model, d_model)

        # Regime分类头 (辅助训练) / Regime classification head (auxiliary)
        self.regime_head = nn.Linear(d_model, num_regimes)

        # 初始化 / Initialization
        nn.init.normal_(self.target.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.target.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model] 时序隐状态 / Time-series hidden state

        Returns:
            delta_correction: [batch, seq_len, d_model] 重写修正量 / Rewrite correction
            regime_logits: [batch, seq_len, num_regimes] 状态分类 / Regime classification
        """
        normed = self.norm(x)
        g = self.gate(normed)           # [B, T, D] 门控值 / Gate values in [0,1]
        t = self.target(normed)         # [B, T, D] 目标值 / Target values

        # Delta修正 = gate * (target - current)
        # 含义: 按gate的开度，将current向target拉近
        # Meaning: pull current toward target proportional to gate opening
        delta = g * (t - x)

        regime_logits = self.regime_head(normed)

        return delta, regime_logits


class DDLTimeSeriesForecaster(nn.Module):
    """
    DDL时序预测模型 / DDL Time-Series Forecasting Model

    用于金融时序预测，核心改进 / For financial time-series, core improvements:
    1. 每层Transformer Block使用DDL残差替代标准残差
       Each Transformer Block uses DDL residual instead of standard
    2. 加入MarketRegimeGate进行regime-aware重写
       MarketRegimeGate for regime-aware rewriting
    3. 输出层预测未来收益率
       Output layer predicts future returns

    与标准时序Transformer对比 / Comparison with standard time-series Transformer:
      标准 / Standard: x_{l+1} = x_l + f_l(x_l)
        (只能累加，regime信号过时后无法消除 / append-only, stale regime signals persist)
      DDL: x_{l+1} = x_l + g*(target-x_l) + f_l(x_l)
        (可重写，主动适应regime切换 / rewrite-capable, adapts to regime switches)
    """

    def __init__(
        self,
        num_features: int = 50,       # 输入因子数 / Number of input factors
        d_model: int = 128,           # 隐藏维度 / Hidden dimension
        n_heads: int = 4,             # 注意力头数 / Attention heads
        n_layers: int = 3,            # Transformer层数 / Transformer layers
        num_regimes: int = 3,         # 市场状态数 / Number of regimes
        forecast_horizon: int = 5,    # 预测未来N天 / Forecast N days ahead
        max_seq_len: int = 252,       # 最大序列长度 / Max sequence length
        dropout: float = 0.1,
    ):
        super().__init__()

        # 因子嵌入 / Factor embedding
        self.factor_proj = nn.Linear(num_features, d_model)

        # 时间位置编码 / Temporal positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # DDL Transformer Blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                # 层归一化和注意力 / LayerNorm and attention
                'norm1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                ),
                # FFN
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                ),
                # DDL门控修正 / DDL gated correction
                'regime_gate': MarketRegimeGate(d_model, num_regimes),
            }))

        # 预测头 / Forecast head
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, forecast_horizon),
        )

        self.num_features = num_features
        self.forecast_horizon = forecast_horizon
        self.max_seq_len = max_seq_len

    def forward(
        self,
        factor_data: torch.Tensor,
        regime_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            factor_data: [batch, seq_len, num_features] 因子数据 / Factor data
            regime_labels: [batch, seq_len] 可选的regime标签 / Optional regime labels

        Returns:
            dict with:
                'returns': [batch, forecast_horizon] 预测的未来收益
                'regime_logits': [batch, seq_len, num_regimes] regime预测
                'regime_loss': regime分类损失 (如有标签)
                'all_hidden': [batch, seq_len, d_model] 所有层最终隐状态
        """
        B, T, num_feat = factor_data.shape
        assert T <= self.max_seq_len, f"Sequence {T} exceeds max {self.max_seq_len}"

        # 因子投影 / Factor projection
        x = self.factor_proj(factor_data)  # [B, T, d_model]

        # 加位置编码 / Add positional encoding
        x = x + self.pos_embed[:, :T, :]

        all_regime_logits = []

        for block in self.blocks:
            # --- DDL Step 1: Regime-aware delta correction ---
            delta, regime_logits = block['regime_gate'](x)
            all_regime_logits.append(regime_logits)

            # --- DDL Step 2: Standard attention ---
            normed = block['norm1'](x)
            attn_out, _ = block['attn'](normed, normed, normed)

            # --- DDL Step 3: Combine (delta + attention) ---
            # DDL: x = x + delta + attention_output
            x = x + delta + attn_out

            # --- Standard FFN with residual ---
            x = x + block['ffn'](block['norm2'](x))

        # 取最后一个时间步做预测 / Use last timestep for forecasting
        last_hidden = x[:, -1, :]  # [B, d_model]
        returns = self.forecast_head(last_hidden)  # [B, forecast_horizon]

        # 汇总regime预测 / Aggregate regime predictions
        regime_stack = torch.stack(all_regime_logits, dim=0)
        avg_regime = regime_stack.mean(dim=0)  # [B, T, num_regimes]

        # Regime分类损失 / Regime classification loss
        regime_loss = None
        if regime_labels is not None:
            regime_loss = F.cross_entropy(
                avg_regime.view(-1, avg_regime.size(-1)),
                regime_labels.view(-1),
            )

        return {
            'returns': returns,
            'regime_logits': avg_regime,
            'regime_loss': regime_loss,
            'all_hidden': x,
        }


# ============================================================
# 2. MoE因子路由器 / MoE Factor Router
# ============================================================

class MoEFactorRouter(nn.Module):
    """
    MoE因子路由器 / MoE Factor Router

    借鉴DDL"选择性重写"和MiMo-V2-Flash的MoE设计:
    - 多个因子expert但每次只激活top-K (sparse activation)
    - Routing network学习根据市场状态动态选择
    - 未激活的expert不参与计算 (类似DDL中gate≈0)

    Borrowing DDL's "selective rewrite" and MiMo's MoE design:
    - Multiple factor experts but only top-K activated (sparse activation)
    - Routing network learns dynamic selection based on market state
    - Inactive experts don't compute (like DDL when gate ≈ 0)

    Expert分配 / Expert assignment:
      Expert 0: 动量因子 / Momentum factors
      Expert 1: 价值因子 / Value factors
      Expert 2: 波动率因子 / Volatility factors
      Expert 3-7: 其他因子 / Other factors
    """

    def __init__(
        self,
        num_factors: int = 50,
        num_experts: int = 8,
        top_k: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 路由网络 / Routing network
        self.router = nn.Sequential(
            nn.Linear(num_factors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

        # 因子专家 / Factor experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_factors, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),  # 输出: 单因子收益预测 / Output: factor return
            )
            for _ in range(num_experts)
        ])

    def forward(self, factor_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            factor_input: [batch, num_factors] 当前因子快照 / Current factor snapshot

        Returns:
            dict with:
                'prediction': [batch, 1] 综合预测 / Combined prediction
                'expert_weights': [batch, num_experts] expert权重 / Expert weights
                'load_balance_loss': 负载均衡损失 / Load balancing loss
        """
        B = factor_input.shape[0]

        # 路由: 计算每个expert的激活概率
        # Routing: compute activation probability for each expert
        router_logits = self.router(factor_input)  # [B, num_experts]

        # Top-K选择 / Top-K selection
        top_k_logits, top_k_indices = router_logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [B, top_k]

        # 构建稀疏权重矩阵 / Build sparse weight matrix
        sparse_weights = torch.zeros(B, self.num_experts, device=factor_input.device)
        expert_outputs = torch.zeros(B, self.num_experts, device=factor_input.device)

        # 只激活选中的expert / Only activate selected experts
        for k in range(self.top_k):
            idx = top_k_indices[:, k]  # [B]
            for b in range(B):
                expert_idx = idx[b].item()
                expert_out = self.experts[expert_idx](factor_input[b:b+1])
                expert_outputs[b, expert_idx] = expert_out.squeeze()
            sparse_weights.scatter_(
                1, top_k_indices[:, k:k+1], top_k_weights[:, k:k+1]
            )

        # 加权聚合 / Weighted aggregation
        prediction = (expert_outputs * sparse_weights).sum(dim=-1, keepdim=True)

        # 负载均衡损失 / Load balancing loss
        expert_freq = sparse_weights.mean(dim=0)
        ideal_freq = torch.ones_like(expert_freq) / self.num_experts
        load_balance_loss = ((expert_freq - ideal_freq) ** 2).sum()

        return {
            'prediction': prediction,
            'expert_weights': sparse_weights,
            'load_balance_loss': load_balance_loss,
        }


# ============================================================
# 3. DDL投资组合优化器 / DDL Portfolio Optimizer
# ============================================================

class DDLPortfolioOptimizer(nn.Module):
    """
    DDL端到端投资组合优化器 / DDL End-to-End Portfolio Optimizer

    输入: 过去N天的因子数据 / Input: factor data from past N days
    输出: 投资组合权重 / Output: portfolio weights
      满足 / Satisfying: 权重之和=1, 权重>=0, 单资产不超过max_position

    使用DDL时序编码器 + 约束投影层
    Uses DDL time-series encoder + constraint projection layer

    传统方法 vs DDL方法 / Traditional vs DDL approach:
      传统: 因子→预测收益→均值方差优化→权重 (两步分离)
      DDL:  因子→DDL编码→权重网络→权重 (端到端)
    """

    def __init__(
        self,
        num_assets: int = 30,         # 资产数量 / Number of assets
        num_factors: int = 50,        # 每资产因子数 / Factors per asset
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        forecast_horizon: int = 1,    # 预测1天 / Forecast 1 day
        max_seq_len: int = 252,
        max_position: float = 0.1,    # 单资产最大仓位 / Max position per asset
    ):
        super().__init__()
        self.num_assets = num_assets
        self.max_position = max_position

        # DDL时序编码器 / DDL time-series encoder
        self.forecaster = DDLTimeSeriesForecaster(
            num_features=num_factors,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            forecast_horizon=forecast_horizon,
            max_seq_len=max_seq_len,
        )

        # 组合权重网络 / Portfolio weight network
        self.weight_net = nn.Sequential(
            nn.Linear(d_model + num_assets, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_assets),
        )

    def forward(
        self,
        factor_data: torch.Tensor,
        cov_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            factor_data: [batch, lookback, num_assets, num_factors]
            cov_matrix: [batch, num_assets, num_assets] 协方差矩阵 (可选)

        Returns:
            dict with:
                'weights': [batch, num_assets] 投资组合权重
                'expected_return': [batch, 1] 预期组合收益
                'risk': [batch, 1] 组合风险 (如有cov_matrix)
                'predicted_returns': [batch, num_assets] 预测的各资产收益
        """
        B, T, A, num_feat = factor_data.shape

        # 对每个资产独立预测 / Predict independently for each asset
        asset_returns = []
        asset_hidden = []
        for a in range(A):
            asset_factors = factor_data[:, :, a, :]  # [B, T, F]
            result = self.forecaster(asset_factors)
            asset_returns.append(result['returns'])    # [B, 1]
            asset_hidden.append(result['all_hidden'][:, -1, :])  # [B, d_model]

        predicted_returns = torch.cat(asset_returns, dim=-1)  # [B, num_assets]

        # 使用最后一个资产的隐状态 + 预测收益构建权重
        # Use last asset's hidden state + predicted returns for weight construction
        combined_input = torch.cat([asset_hidden[-1], predicted_returns], dim=-1)
        raw_weights = self.weight_net(combined_input)

        # 约束投影 / Constraint projection
        # Softmax确保权重之和=1且非负 / Softmax ensures weights sum to 1 and non-negative
        weights = F.softmax(raw_weights, dim=-1)
        # 截断: 单资产不超过max_position / Clamp: max position per asset
        weights = torch.clamp(weights, max=self.max_position)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # 重新归一化 / Re-normalize

        # 组合预期收益 / Portfolio expected return
        expected_return = (weights * predicted_returns).sum(dim=-1, keepdim=True)

        # 组合风险 / Portfolio risk
        risk = None
        if cov_matrix is not None:
            risk = torch.bmm(
                weights.unsqueeze(1),
                torch.bmm(cov_matrix, weights.unsqueeze(-1)),
            ).squeeze(-1)

        return {
            'weights': weights,
            'expected_return': expected_return,
            'risk': risk,
            'predicted_returns': predicted_returns,
        }


# ============================================================
# 4. 模拟市场数据生成 / Simulated Market Data Generation
# ============================================================

def simulate_market_data(
    num_samples: int = 1000,
    seq_len: int = 60,
    num_assets: int = 10,
    num_factors: int = 20,
    regime_transition_prob: float = 0.05,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    生成模拟金融数据 / Generate simulated financial data

    模拟三种市场regime (带Markov链转换):
    Simulates three market regimes (with Markov chain transitions):
      - Regime 0: 动量 (趋势延续) / Momentum (trend continuation)
      - Regime 1: 均值回归 (震荡) / Mean-reversion (oscillation)
      - Regime 2: 高波动 (crash/rally) / High-volatility (crash/rally)

    Args:
        num_samples: 样本数量 / Number of samples
        seq_len: 序列长度 / Sequence length
        num_assets: 资产数量 / Number of assets
        num_factors: 因子数量 / Number of factors
        regime_transition_prob: regime切换概率 / Regime transition probability
        seed: 随机种子 / Random seed

    Returns:
        dict with:
            'factors': [num_samples, seq_len, num_assets, num_factors]
            'returns': [num_samples, seq_len, num_assets]
            'regimes': [num_samples, seq_len]
    """
    rng = np.random.RandomState(seed)

    # 生成Markov链regime序列 / Generate Markov chain regime sequences
    regimes = np.zeros((num_samples, seq_len), dtype=np.int64)
    for s in range(num_samples):
        regimes[s, 0] = rng.randint(0, 3)
        for t in range(1, seq_len):
            if rng.random() < regime_transition_prob:
                # 切换到不同regime / Switch to different regime
                options = [r for r in range(3) if r != regimes[s, t-1]]
                regimes[s, t] = rng.choice(options)
            else:
                regimes[s, t] = regimes[s, t-1]

    regimes_tensor = torch.from_numpy(regimes)

    # 生成因子数据 / Generate factor data
    # 基础因子 + regime-dependent偏移
    # Base factors + regime-dependent offsets
    base_factors = torch.randn(num_samples, seq_len, num_assets, num_factors)

    # Regime-dependent因子偏移 / Regime-dependent factor offsets
    regime_offsets = torch.zeros(3, num_factors)
    regime_offsets[0, :num_factors//3] = 0.5       # 动量因子偏移
    regime_offsets[1, num_factors//3:2*num_factors//3] = 0.5  # 价值因子偏移
    regime_offsets[2, 2*num_factors//3:] = 0.5      # 波动率因子偏移

    for s in range(num_samples):
        for t in range(seq_len):
            r = regimes[s, t]
            base_factors[s, t] += regime_offsets[r].unsqueeze(0)

    # 生成收益数据 (regime-dependent统计特征)
    # Generate returns (regime-dependent statistics)
    returns = torch.zeros(num_samples, seq_len, num_assets)
    for s in range(num_samples):
        for t in range(seq_len):
            r = regimes[s, t]
            if r == 0:  # 动量: 正均值，低波动 / Momentum: positive mean, low vol
                returns[s, t] = torch.randn(num_assets) * 0.01 + 0.001
            elif r == 1:  # 均值回归: 零均值，低波动 / Mean-rev: zero mean, low vol
                returns[s, t] = torch.randn(num_assets) * 0.005
            else:  # 高波动: 零均值，高波动 / High-vol: zero mean, high vol
                returns[s, t] = torch.randn(num_assets) * 0.03

    return {
        'factors': base_factors,
        'returns': returns,
        'regimes': regimes_tensor,
    }


def create_forecaster_batch(
    data: Dict[str, torch.Tensor],
    batch_size: int,
    start_idx: int = 0,
    device: torch.device = torch.device("cpu"),
    asset_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从数据集中创建训练batch / Create training batch from dataset

    Args:
        data: 模拟数据 / Simulated data
        batch_size: 批次大小 / Batch size
        start_idx: 起始索引 / Start index
        device: 设备 / Device
        asset_idx: 使用的资产索引 / Asset index to use

    Returns:
        (factor_data, return_targets, regime_labels) all on device
    """
    end_idx = min(start_idx + batch_size, data['factors'].shape[0])

    factor_data = data['factors'][start_idx:end_idx, :, asset_idx, :].to(device)
    # 目标: 未来N天收益的平均值 / Target: average of future N-day returns
    return_targets = data['returns'][start_idx:end_idx, -5:, asset_idx].mean(dim=-1).unsqueeze(-1).to(device)
    regime_labels = data['regimes'][start_idx:end_idx].to(device)

    return factor_data, return_targets, regime_labels


# ============================================================
# 5. 训练函数 / Training Functions
# ============================================================

def train_forecaster(
    model: DDLTimeSeriesForecaster,
    data: Dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    regime_weight: float = 0.1,
    print_interval: int = 10,
) -> List[Dict]:
    """
    训练DDL时序预测器 / Train DDL time-series forecaster

    Args:
        model: DDL预测模型 / DDL forecasting model
        data: 模拟数据 / Simulated data
        device: 设备 / Device
        epochs: 训练轮数 / Training epochs
        batch_size: 批次大小 / Batch size
        lr: 学习率 / Learning rate
        regime_weight: regime损失权重 / Regime loss weight
        print_interval: 打印间隔 / Print interval

    Returns:
        list of training metrics per epoch
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num_samples = data['factors'].shape[0]
    num_batches = num_samples // batch_size

    history = []

    print(f"\n{'='*60}")
    print(f"  Training DDL Time-Series Forecaster")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_pred_loss = 0.0
        epoch_regime_loss = 0.0
        num_steps = 0

        # 打乱数据 / Shuffle data
        perm = torch.randperm(num_samples)
        shuffled_factors = data['factors'][perm]
        shuffled_returns = data['returns'][perm]
        shuffled_regimes = data['regimes'][perm]

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            # 使用第一个资产做训练 / Use first asset for training
            factor_batch = shuffled_factors[start:end, :, 0, :].to(device)
            return_target = shuffled_returns[start:end, -5:, 0].mean(dim=-1).unsqueeze(-1).to(device)
            regime_batch = shuffled_regimes[start:end].to(device)

            # 前向传播 / Forward pass
            result = model(factor_batch, regime_labels=regime_batch)

            # 复合损失 / Composite loss
            pred_loss = F.mse_loss(result['returns'], return_target)
            regime_loss = result['regime_loss'] if result['regime_loss'] is not None else torch.tensor(0.0)
            total_loss = pred_loss + regime_weight * regime_loss

            # 反向传播 / Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_pred_loss += pred_loss.item()
            epoch_regime_loss += regime_loss.item() if isinstance(regime_loss, torch.Tensor) else regime_loss
            num_steps += 1

        # 计算平均 / Compute averages
        avg_loss = epoch_loss / max(num_steps, 1)
        avg_pred = epoch_pred_loss / max(num_steps, 1)
        avg_regime = epoch_regime_loss / max(num_steps, 1)

        history.append({
            "epoch": epoch + 1,
            "total_loss": avg_loss,
            "prediction_loss": avg_pred,
            "regime_loss": avg_regime,
        })

        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | Pred: {avg_pred:.4f} | "
                  f"Regime: {avg_regime:.4f}")

    return history


def train_moe_router(
    model: MoEFactorRouter,
    data: Dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    load_balance_weight: float = 0.01,
    print_interval: int = 10,
) -> List[Dict]:
    """
    训练MoE因子路由器 / Train MoE factor router

    Args:
        model: MoE路由器模型 / MoE router model
        data: 模拟数据 / Simulated data
        device: 设备 / Device
        epochs, batch_size, lr: 训练参数 / Training parameters
        load_balance_weight: 负载均衡损失权重 / Load balance loss weight
        print_interval: 打印间隔 / Print interval

    Returns:
        list of training metrics per epoch
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num_samples = data['factors'].shape[0]
    num_batches = num_samples // batch_size

    history = []

    print(f"\n{'='*60}")
    print(f"  Training MoE Factor Router")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_lb_loss = 0.0
        num_steps = 0

        perm = torch.randperm(num_samples)

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            # 取最新时间步的因子快照 / Take latest timestep factor snapshot
            factor_snapshot = data['factors'][perm[start:end], -1, 0, :].to(device)
            return_target = data['returns'][perm[start:end], -1, 0:1].to(device)

            # 前向传播 / Forward pass
            result = model(factor_snapshot)

            # 损失 / Loss
            pred_loss = F.mse_loss(result['prediction'], return_target)
            lb_loss = result['load_balance_loss']
            total_loss = pred_loss + load_balance_weight * lb_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_lb_loss += lb_loss.item()
            num_steps += 1

        avg_loss = epoch_loss / max(num_steps, 1)
        avg_lb = epoch_lb_loss / max(num_steps, 1)

        history.append({
            "epoch": epoch + 1,
            "total_loss": avg_loss,
            "load_balance_loss": avg_lb,
        })

        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | LB: {avg_lb:.6f}")

    return history


def train_portfolio_optimizer(
    model: DDLPortfolioOptimizer,
    data: Dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    risk_weight: float = 0.5,
    print_interval: int = 10,
) -> List[Dict]:
    """
    训练DDL投资组合优化器 / Train DDL portfolio optimizer

    损失函数: -预期收益 + risk_weight * 风险 + 集中度惩罚
    Loss function: -expected_return + risk_weight * risk + concentration penalty

    Args:
        model: 组合优化器模型 / Portfolio optimizer model
        data: 模拟数据 / Simulated data
        device: 设备 / Device
        epochs, batch_size, lr: 训练参数 / Training parameters
        risk_weight: 风险惩罚权重 / Risk penalty weight
        print_interval: 打印间隔 / Print interval

    Returns:
        list of training metrics per epoch
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num_samples = data['factors'].shape[0]
    num_batches = num_samples // batch_size

    history = []

    print(f"\n{'='*60}")
    print(f"  Training DDL Portfolio Optimizer")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Risk weight: {risk_weight}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_return = 0.0
        epoch_risk = 0.0
        num_steps = 0

        perm = torch.randperm(num_samples)

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            indices = perm[start:end]

            # 构建投资组合输入 / Build portfolio input
            # [batch, seq_len, num_assets, num_factors]
            factor_batch = data['factors'][indices].to(device)

            # 计算经验协方差矩阵 / Compute empirical covariance matrix
            returns_batch = data['returns'][indices].to(device)  # [B, T, A]
            returns_mean = returns_batch.mean(dim=1)  # [B, A]
            returns_centered = returns_batch - returns_mean.unsqueeze(1)
            cov_matrix = torch.bmm(
                returns_centered.transpose(1, 2), returns_centered
            ) / (returns_batch.shape[1] - 1)  # [B, A, A]

            # 前向传播 / Forward pass
            result = model(factor_batch, cov_matrix=cov_matrix)

            # 组合优化损失 / Portfolio optimization loss
            # 最大化收益 - 最小化风险 - 最小化集中度
            # Maximize return - minimize risk - minimize concentration
            neg_return = -result['expected_return'].mean()
            risk = result['risk'].mean() if result['risk'] is not None else torch.tensor(0.0)

            # 集中度惩罚: 权重越均匀越好 / Concentration penalty: prefer uniform weights
            weights = result['weights']
            uniform = torch.ones_like(weights) / weights.shape[-1]
            concentration = ((weights - uniform) ** 2).sum(dim=-1).mean()

            total_loss = neg_return + risk_weight * risk + 0.1 * concentration

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_return += result['expected_return'].mean().item()
            epoch_risk += risk.item() if isinstance(risk, torch.Tensor) else risk
            num_steps += 1

        avg_loss = epoch_loss / max(num_steps, 1)
        avg_return = epoch_return / max(num_steps, 1)
        avg_risk = epoch_risk / max(num_steps, 1)

        history.append({
            "epoch": epoch + 1,
            "total_loss": avg_loss,
            "expected_return": avg_return,
            "risk": avg_risk,
        })

        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | Return: {avg_return:.4f} | "
                  f"Risk: {avg_risk:.6f}")

    return history


# ============================================================
# 6. 初步回测 / Preliminary Backtest
# ============================================================

def simple_backtest(
    model: DDLTimeSeriesForecaster,
    data: Dict[str, torch.Tensor],
    device: torch.device,
    asset_idx: int = 0,
) -> Dict[str, float]:
    """
    简单回测评估 / Simple backtest evaluation

    使用模型预测做简单的交易信号回测。
    Simple trading signal backtest using model predictions.

    策略 / Strategy:
      - 预测收益 > 0 → 做多 / Predicted return > 0 → long
      - 预测收益 <= 0 → 空仓 / Predicted return <= 0 → flat

    Args:
        model: DDL预测模型 / DDL forecasting model
        data: 测试数据 / Test data
        device: 设备 / Device
        asset_idx: 资产索引 / Asset index

    Returns:
        dict: 回测指标 / Backtest metrics
    """
    model.eval()
    num_samples = data['factors'].shape[0]

    all_predictions = []
    all_actual_returns = []
    all_regime_preds = []

    with torch.no_grad():
        batch_size = 64
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)

            factor_batch = data['factors'][start:end, :, asset_idx, :].to(device)
            result = model(factor_batch)

            # 预测收益 (取第一个预测) / Predicted return (first forecast)
            pred = result['returns'][:, 0].cpu()
            all_predictions.append(pred)

            # 实际收益 / Actual returns
            actual = data['returns'][start:end, -1, asset_idx].cpu()
            all_actual_returns.append(actual)

            # Regime预测 / Regime predictions
            regime_probs = F.softmax(result['regime_logits'][:, -1, :], dim=-1)
            regime_pred = regime_probs.argmax(dim=-1).cpu()
            all_regime_preds.append(regime_pred)

    predictions = torch.cat(all_predictions)
    actuals = torch.cat(all_actual_returns)
    regime_preds = torch.cat(all_regime_preds)
    actual_regimes = data['regimes'][:len(predictions), -1]

    # 交易信号 / Trading signals
    signals = (predictions > 0).float()  # 1 = long, 0 = flat

    # 策略收益 / Strategy returns
    strategy_returns = signals * actuals

    # 计算指标 / Compute metrics
    total_return = strategy_returns.sum().item()
    avg_return = strategy_returns.mean().item()
    std_return = strategy_returns.std().item()

    # Sharpe比率 (年化, 假设252个交易日)
    # Sharpe ratio (annualized, assuming 252 trading days)
    sharpe = (avg_return / max(std_return, 1e-8)) * math.sqrt(252) if std_return > 0 else 0.0

    # 最大回撤 / Maximum drawdown
    cumulative = strategy_returns.cumsum(dim=0)
    running_max = cumulative.cummax(dim=0).values
    drawdowns = running_max - cumulative
    max_drawdown = drawdowns.max().item()

    # Regime检测准确率 / Regime detection accuracy
    regime_accuracy = (regime_preds == actual_regimes[:len(regime_preds)]).float().mean().item()

    # 胜率 / Win rate
    trades = strategy_returns[strategy_returns != 0]
    win_rate = (trades > 0).float().mean().item() if len(trades) > 0 else 0.0

    return {
        "total_return": total_return,
        "avg_daily_return": avg_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "regime_accuracy": regime_accuracy,
        "win_rate": win_rate,
        "num_trades": int(signals.sum().item()),
        "avg_prediction": predictions.mean().item(),
    }


# ============================================================
# 7. 主函数 / Main Function
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DDL Quant Finance Training / DDL量化金融训练"
    )
    parser.add_argument("--quick-demo", action="store_true",
                        help="运行快速演示 / Run quick demo")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--num-assets", type=int, default=10)
    parser.add_argument("--num-factors", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="quant_results")
    parser.add_argument("--skip-portfolio", action="store_true",
                        help="跳过组合优化训练 (较慢) / Skip portfolio optimization")
    return parser.parse_args()


def main():
    args = parse_args()

    # 快速演示模式 / Quick demo mode
    if args.quick_demo:
        args.epochs = 5
        args.num_samples = 200
        args.seq_len = 30
        args.num_assets = 5
        args.num_factors = 10
        args.d_model = 32
        args.n_layers = 1
        args.skip_portfolio = True

    # 设置设备 / Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 设置随机种子 / Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 生成模拟数据 / Generate simulated data
    print(f"\nGenerating simulated market data...")
    data = simulate_market_data(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        num_assets=args.num_assets,
        num_factors=args.num_factors,
        seed=args.seed,
    )
    print(f"  Samples: {args.num_samples}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Assets: {args.num_assets}")
    print(f"  Factors: {args.num_factors}")

    # Regime分布 / Regime distribution
    regime_counts = [
        (data['regimes'] == r).sum().item() for r in range(3)
    ]
    total = sum(regime_counts)
    regime_names = ["Momentum/动量", "Mean-Reversion/均值回归", "High-Vol/高波动"]
    print(f"  Regime distribution:")
    for i, name in enumerate(regime_names):
        print(f"    {name}: {regime_counts[i]:>6d} ({regime_counts[i]/total:.1%})")

    # --- 训练1: DDL时序预测器 / Train 1: DDL Forecaster ---
    forecaster = DDLTimeSeriesForecaster(
        num_features=args.num_factors,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        forecast_horizon=5,
        max_seq_len=args.seq_len,
    ).to(device)

    forecaster_history = train_forecaster(
        forecaster, data, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )

    # 回测 / Backtest
    backtest_results = simple_backtest(forecaster, data, device)
    print(f"\n{'='*60}")
    print(f"  DDL Forecaster Backtest Results / 回测结果")
    print(f"{'='*60}")
    for k, v in backtest_results.items():
        print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")

    # --- 训练2: MoE因子路由器 / Train 2: MoE Router ---
    moe_router = MoEFactorRouter(
        num_factors=args.num_factors,
        num_experts=6,
        top_k=2,
        hidden_dim=args.d_model,
    ).to(device)

    moe_history = train_moe_router(
        moe_router, data, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )

    # --- 训练3: DDL组合优化器 / Train 3: Portfolio Optimizer ---
    portfolio_history = []
    if not args.skip_portfolio:
        portfolio_opt = DDLPortfolioOptimizer(
            num_assets=args.num_assets,
            num_factors=args.num_factors,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_seq_len=args.seq_len,
        ).to(device)

        portfolio_history = train_portfolio_optimizer(
            portfolio_opt, data, device,
            epochs=args.epochs, batch_size=min(args.batch_size, 16), lr=args.lr,
        )
    else:
        print("\n  Skipping portfolio optimization (use --skip-portfolio=false to enable)")

    # --- 保存结果 / Save results ---
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "forecaster_history": forecaster_history,
        "moe_history": moe_history,
        "portfolio_history": portfolio_history,
        "backtest_results": backtest_results,
        "config": vars(args),
    }

    results_path = os.path.join(args.output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # 保存模型 / Save models
    torch.save(forecaster.state_dict(),
               os.path.join(args.output_dir, "forecaster_model.pt"))
    torch.save(moe_router.state_dict(),
               os.path.join(args.output_dir, "moe_router_model.pt"))
    print(f"Models saved to: {args.output_dir}/")

    print(f"\n{'='*60}")
    print(f"  Training Complete / 训练完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
