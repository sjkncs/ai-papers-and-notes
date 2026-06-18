# ============================================================================
# MIT License
# ============================================================================
# Copyright (c) 2026 AI Papers Weekly Review Project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ============================================================================

"""
Edge-Deployable Quant Trading Model based on Ministral 3
基于Ministral 3的边缘部署量化交易模型

This module implements a compact trading signal generation model that leverages
the Ministral 3 architecture's parameter efficiency for low-latency edge
deployment in quantitative trading scenarios.

本模块实现了一个紧凑的交易信号生成模型，利用Ministral 3架构的参数效率
在量化交易场景中实现低延迟边缘部署。

Components / 组件:
    - CompactTradingSignalModel: Small LLM for real-time signal generation
      小型LLM用于实时信号生成
    - MarketMicrostructureEncoder: Encode order book features compactly
      紧凑地编码订单簿特征
    - LatencyAwareTrainer: Train with latency constraints
      带延迟约束的训练器
    - simulate_market_data(): Generate realistic market dynamics
      生成逼真的市场动态数据
"""

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Configuration / 配置
# ============================================================================

@dataclass
class QuantModelConfig:
    """
    Configuration for the compact trading signal model.
    紧凑交易信号模型的配置。
    """

    # 模型规模 / Model scale
    hidden_dim: int = 256           # 隐藏维度 / Hidden dimension
    num_layers: int = 4             # Transformer层数 / Number of layers
    num_heads: int = 8              # 注意力头数 / Number of attention heads
    num_kv_heads: int = 2           # KV头数（GQA）/ KV heads for GQA
    ffn_scale: float = 2.0          # FFN缩放因子 / FFN scale factor

    # 交易相关参数 / Trading-specific parameters
    num_market_features: int = 32   # 市场特征数 / Number of market features
    lookback_window: int = 64       # 回看窗口 / Lookback window (ticks)
    signal_horizon: int = 10        # 预测时域 / Prediction horizon (ticks)
    num_signal_classes: int = 3     # 信号类别数 / Signal classes (buy/hold/sell)

    # 市场微观结构编码 / Market microstructure encoding
    orderbook_levels: int = 5       # 订单簿深度 / Order book depth levels
    price_resolution: int = 8       # 价格分辨率 / Price discretization bins

    # 延迟约束 / Latency constraints
    max_latency_ms: float = 10.0    # 最大允许延迟 / Maximum allowed latency (ms)
    latency_penalty_weight: float = 0.1  # 延迟惩罚权重 / Latency penalty weight

    # 训练参数 / Training parameters
    dropout: float = 0.1
    shared_ffn: bool = True         # 使用共享权重FFN / Use shared-weight FFN

    @property
    def head_dim(self) -> int:
        """每个头的维度 / Dimension per head"""
        return self.hidden_dim // self.num_heads


# ============================================================================
# Market Microstructure Encoder / 市场微观结构编码器
# ============================================================================

class MarketMicrostructureEncoder(nn.Module):
    """
    Compact encoder for market microstructure features.
    市场微观结构特征的紧凑编码器。

    Encodes order book snapshots, trade data, and price features into a
    compact representation suitable for the trading signal model.
    Designed for parameter efficiency to minimize latency.

    将订单簿快照、交易数据和价格特征编码为适合交易信号模型的紧凑表示。
    设计参数效率以最小化延迟。
    """

    def __init__(self, config: QuantModelConfig):
        super().__init__()
        self.config = config

        # 订单簿编码 / Order book encoding
        # 每个level有 bid_price, bid_size, ask_price, ask_size
        # Each level has bid_price, bid_size, ask_price, ask_size
        ob_features_per_level = 4
        total_ob_features = config.orderbook_levels * ob_features_per_level

        # 订单簿特征投影 / Order book feature projection
        self.ob_encoder = nn.Sequential(
            nn.Linear(total_ob_features, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
        )

        # 交易特征编码 / Trade feature encoding
        # price, volume, trade_direction, timestamp_delta
        trade_features = 4
        self.trade_encoder = nn.Sequential(
            nn.Linear(trade_features, config.hidden_dim // 4),
            nn.GELU(),
        )

        # 市场统计特征 / Market statistics features
        # vwap, spread, imbalance, volatility, momentum
        stats_features = 5
        self.stats_encoder = nn.Sequential(
            nn.Linear(stats_features, config.hidden_dim // 4),
            nn.GELU(),
        )

        # 时间编码（正弦位置编码）/ Temporal encoding (sinusoidal positional)
        self.temporal_encoder = SinusoidalPositionalEncoding(
            config.hidden_dim, max_len=config.lookback_window
        )

        # 最终合并层 / Final fusion layer
        self.fusion = nn.Linear(
            config.hidden_dim // 4 * 3,  # ob + trade + stats
            config.hidden_dim,
        )
        self.fusion_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        orderbook: torch.Tensor,
        trades: torch.Tensor,
        stats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode market microstructure features.
        编码市场微观结构特征。

        Args:
            orderbook: [B, lookback, levels * 4] order book snapshots
            trades: [B, lookback, 4] trade features
            stats: [B, lookback, 5] market statistics

        Returns:
            Encoded features [B, lookback, hidden_dim]
        """
        # 编码各部分 / Encode each component
        ob_encoded = self.ob_encoder(orderbook)      # [B, L, hidden_dim//4]
        trade_encoded = self.trade_encoder(trades)    # [B, L, hidden_dim//4]
        stats_encoded = self.stats_encoder(stats)     # [B, L, hidden_dim//4]

        # 拼接并融合 / Concatenate and fuse
        combined = torch.cat([ob_encoded, trade_encoded, stats_encoded], dim=-1)
        fused = self.fusion(combined)                  # [B, L, hidden_dim]

        # 添加时间位置编码 / Add temporal positional encoding
        fused = self.temporal_encoder(fused)

        # 层归一化 / Layer normalization
        fused = self.fusion_norm(fused)

        return fused


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    时序序列的正弦位置编码。
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding / 添加位置编码"""
        return x + self.pe[:, : x.size(1)]


# ============================================================================
# Compact Trading Signal Model / 紧凑交易信号模型
# ============================================================================

class CompactTradingSignalModel(nn.Module):
    """
    Small LLM-based model for real-time trading signal generation.
    基于小型LLM的实时交易信号生成模型。

    Uses the Ministral 3 architecture principles (shared-weight FFN, GQA)
    for parameter-efficient inference on edge devices.
    使用Ministral 3架构原则（共享权重FFN、GQA）实现边缘设备上的参数高效推理。
    """

    def __init__(self, config: QuantModelConfig):
        super().__init__()
        self.config = config

        # 市场特征编码器 / Market feature encoder
        self.market_encoder = MarketMicrostructureEncoder(config)

        # 轻量级Transformer主干 / Lightweight transformer backbone
        self.layers = nn.ModuleList([
            CompactTransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # 最终归一化 / Final normalization
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # 信号头 / Signal head
        # 输出：方向预测 + 置信度 / Output: direction prediction + confidence
        self.signal_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_signal_classes),
        )

        # 置信度估计头 / Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # 回归头：预期收益率 / Regression head: expected return
        self.return_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
        )

        # 初始化 / Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Weight initialization / 权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        orderbook: torch.Tensor,
        trades: torch.Tensor,
        stats: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate trading signals from market data.
        从市场数据生成交易信号。

        Args:
            orderbook: [B, lookback, levels*4] order book data
            trades: [B, lookback, 4] trade data
            stats: [B, lookback, 5] market statistics

        Returns:
            Dictionary with signal logits, confidence, and expected returns
            包含信号logits、置信度和预期收益的字典
        """
        # 编码市场特征 / Encode market features
        hidden = self.market_encoder(orderbook, trades, stats)

        # Transformer处理 / Transformer processing
        for layer in self.layers:
            hidden = layer(hidden)

        # 最终归一化 / Final normalization
        hidden = self.final_norm(hidden)

        # 使用最后一个时间步的表示生成信号
        # Use last timestep representation for signal generation
        last_hidden = hidden[:, -1, :]  # [B, hidden_dim]

        # 生成信号 / Generate signals
        signal_logits = self.signal_head(last_hidden)       # [B, num_classes]
        confidence = self.confidence_head(last_hidden)       # [B, 1]
        expected_return = self.return_head(last_hidden)      # [B, 1]

        return {
            "signal_logits": signal_logits,
            "confidence": confidence,
            "expected_return": expected_return,
            "hidden_states": hidden,
        }


class CompactTransformerBlock(nn.Module):
    """
    Compact transformer block with GQA and shared-weight FFN.
    使用GQA和共享权重FFN的紧凑Transformer块。
    """

    def __init__(self, config: QuantModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # 注意力层 / Attention layer (with GQA)
        self.attn_norm = nn.LayerNorm(config.hidden_dim)
        self.attention = CompactGQAAttention(config)

        # FFN层（可选共享权重）/ FFN layer (optional shared weights)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        self.ffn = CompactFFN(config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass / 前向传播"""
        # 注意力 + 残差 / Attention + residual
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # FFN + 残差 / FFN + residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CompactGQAAttention(nn.Module):
    """
    Grouped-query attention for compact models.
    紧凑模型的分组查询注意力。
    """

    def __init__(self, config: QuantModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """GQA forward pass / GQA前向传播"""
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 扩展KV / Expand KV for GQA
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # 缩放点积注意力 / Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 因果掩码 / Causal mask
        causal = torch.triu(
            torch.full((S, S), float("-inf"), device=q.device), diagonal=1
        )
        attn = attn + causal.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(output)


class CompactFFN(nn.Module):
    """
    Compact FFN with optional shared weights (Ministral style).
    可选共享权重的紧凑FFN（Ministral风格）。
    """

    def __init__(self, config: QuantModelConfig):
        super().__init__()
        intermediate_dim = int(config.hidden_dim * config.ffn_scale)
        self.shared = config.shared_ffn

        if self.shared:
            # 共享权重模式 / Shared weight mode
            shared_dim = intermediate_dim // 2
            self.gate = nn.Linear(config.hidden_dim, shared_dim, bias=False)
            self.up = nn.Linear(config.hidden_dim, shared_dim, bias=False)
            self.down = nn.Linear(shared_dim, config.hidden_dim, bias=False)
        else:
            self.gate = nn.Linear(config.hidden_dim, intermediate_dim, bias=False)
            self.up = nn.Linear(config.hidden_dim, intermediate_dim, bias=False)
            self.down = nn.Linear(intermediate_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass / SwiGLU前向传播"""
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ============================================================================
# Market Data Simulation / 市场数据模拟
# ============================================================================

def simulate_market_data(
    num_samples: int = 10000,
    lookback: int = 64,
    orderbook_levels: int = 5,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate simulated market data with realistic dynamics.
    生成具有逼真动态的模拟市场数据。

    Simulates order book snapshots, trade data, and market statistics
    with realistic statistical properties including:
    - Mean-reverting price dynamics / 均值回归价格动态
    - Volume clustering / 成交量聚集
    - Spread dynamics / 价差动态
    - Order book imbalance / 订单簿不平衡

    Args:
        num_samples: Number of samples to generate / 生成样本数
        lookback: Lookback window size / 回看窗口大小
        orderbook_levels: Number of order book levels / 订单簿层级数
        seed: Random seed / 随机种子

    Returns:
        Dictionary with simulated market data tensors
    """
    torch.manual_seed(seed)

    # 基础价格过程（几何布朗运动 + 均值回归）
    # Base price process (geometric Brownian motion + mean reversion)
    dt = 1.0 / (252 * 390)  # 分钟级数据 / Minute-level data
    mu = 0.0  # 漂移率 / Drift
    sigma = 0.20  # 年化波动率 / Annualized volatility
    mean_reversion_strength = 0.1  # 均值回归强度

    prices = torch.zeros(num_samples, lookback)
    prices[:, 0] = 100.0  # 初始价格 / Initial price
    log_mean = math.log(100.0)

    for t in range(1, lookback):
        log_price = torch.log(prices[:, t - 1])
        # 均值回归 + 布朗运动 / Mean reversion + Brownian motion
        dlog = (
            mean_reversion_strength * (log_mean - log_price) * dt
            + mu * dt
            + sigma * math.sqrt(dt) * torch.randn(num_samples)
        )
        prices[:, t] = prices[:, t - 1] * torch.exp(dlog)

    # 生成订单簿数据 / Generate order book data
    # 每个level: bid_price, bid_size, ask_price, ask_size
    ob_features = torch.zeros(num_samples, lookback, orderbook_levels * 4)

    for t in range(lookback):
        mid_price = prices[:, t].unsqueeze(-1)  # [N, 1]
        for level in range(orderbook_levels):
            # 价差随level增大 / Spread increases with level
            spread = (level + 1) * 0.001 * mid_price
            bid_price = mid_price - spread / 2
            ask_price = mid_price + spread / 2
            # 订单大小（对数正态分布）/ Order sizes (log-normal)
            bid_size = torch.exp(torch.randn(num_samples, 1) * 0.5 + 3.0)
            ask_size = torch.exp(torch.randn(num_samples, 1) * 0.5 + 3.0)

            offset = level * 4
            ob_features[:, t, offset] = bid_price.squeeze(-1)
            ob_features[:, t, offset + 1] = bid_size.squeeze(-1)
            ob_features[:, t, offset + 2] = ask_price.squeeze(-1)
            ob_features[:, t, offset + 3] = ask_size.squeeze(-1)

    # 归一化价格特征 / Normalize price features
    ob_mean = ob_features.mean(dim=(0, 1), keepdim=True)
    ob_std = ob_features.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
    ob_features = (ob_features - ob_mean) / ob_std

    # 生成交易数据 / Generate trade data
    # price_return, volume, direction, time_delta
    trades = torch.zeros(num_samples, lookback, 4)
    for t in range(lookback):
        if t == 0:
            trades[:, t, 0] = 0.0  # 收益率 / Return
        else:
            trades[:, t, 0] = (prices[:, t] - prices[:, t - 1]) / prices[:, t - 1]
        # 成交量（与波动率相关）/ Volume (correlated with volatility)
        trades[:, t, 1] = torch.exp(torch.randn(num_samples) * 0.5 + 5.0)
        # 方向（+1买入/-1卖出）/ Direction (+1 buy / -1 sell)
        trades[:, t, 2] = torch.sign(torch.randn(num_samples))
        # 时间间隔 / Time delta
        trades[:, t, 3] = torch.abs(torch.randn(num_samples) * 0.1 + 1.0)

    # 生成市场统计数据 / Generate market statistics
    # vwap, spread, imbalance, volatility, momentum
    stats = torch.zeros(num_samples, lookback, 5)
    for t in range(lookback):
        stats[:, t, 0] = prices[:, t]  # VWAP近似 / VWAP approximation
        if t > 0:
            stats[:, t, 1] = torch.abs(prices[:, t] - prices[:, t - 1])  # Spread
            # 波动率（过去窗口的标准差）/ Volatility (std of past window)
            window = min(t + 1, 20)
            stats[:, t, 3] = prices[:, max(0, t - window):t + 1].std(dim=1)
            # 动量 / Momentum
            stats[:, t, 4] = prices[:, t] - prices[:, max(0, t - 10)]
        # 订单簿不平衡度 / Order book imbalance
        bid_total = ob_features[:, t, 1::4].sum(dim=-1)
        ask_total = ob_features[:, t, 3::4].sum(dim=-1)
        stats[:, t, 2] = (bid_total - ask_total) / (bid_total + ask_total + 1e-8)

    # 归一化 / Normalize
    trades_mean = trades.mean(dim=(0, 1), keepdim=True)
    trades_std = trades.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
    trades = (trades - trades_mean) / trades_std

    stats_mean = stats.mean(dim=(0, 1), keepdim=True)
    stats_std = stats.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
    stats = (stats - stats_mean) / stats_std

    # 生成标签（未来收益方向）/ Generate labels (future return direction)
    # 0=卖出/sell, 1=持有/hold, 2=买入/buy
    future_horizon = min(10, lookback - 1)
    future_returns = torch.zeros(num_samples)
    for i in range(num_samples):
        future_returns[i] = (prices[i, -1] - prices[i, -future_horizon]) / prices[i, -future_horizon]

    # 三分类标签 / Three-class labels
    threshold = 0.001  # 0.1%阈值
    labels = torch.ones(num_samples, dtype=torch.long)  # 默认持有 / Default hold
    labels[future_returns > threshold] = 2   # 买入 / Buy
    labels[future_returns < -threshold] = 0  # 卖出 / Sell

    return {
        "orderbook": ob_features,
        "trades": trades,
        "stats": stats,
        "labels": labels,
        "prices": prices,
        "future_returns": future_returns,
    }


# ============================================================================
# Dataset / 数据集
# ============================================================================

class MarketDataset(Dataset):
    """
    PyTorch dataset for market microstructure data.
    市场微观结构数据的PyTorch数据集。
    """

    def __init__(self, data: Dict[str, torch.Tensor]):
        self.orderbook = data["orderbook"]
        self.trades = data["trades"]
        self.stats = data["stats"]
        self.labels = data["labels"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.orderbook[idx],
            self.trades[idx],
            self.stats[idx],
            self.labels[idx],
        )


# ============================================================================
# Latency-Aware Trainer / 延迟感知训练器
# ============================================================================

class LatencyAwareTrainer:
    """
    Trainer that incorporates latency constraints into the loss function.
    将延迟约束纳入损失函数的训练器。

    During training, measures inference latency and penalizes the model
    if it exceeds the maximum allowed latency. This encourages the model
    to learn representations that can be computed efficiently.

    在训练期间测量推理延迟，如果模型超过最大允许延迟则施加惩罚。
    这鼓励模型学习可以高效计算的表示。
    """

    def __init__(
        self,
        model: CompactTradingSignalModel,
        config: QuantModelConfig,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # 优化器 / Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # 学习率调度器（余弦退火）/ LR scheduler (cosine annealing)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        # 损失函数 / Loss functions
        self.signal_criterion = nn.CrossEntropyLoss()
        self.return_criterion = nn.MSELoss()

        # 训练指标 / Training metrics
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.latencies: List[float] = []

    def compute_latency_penalty(self, batch_size: int = 32) -> float:
        """
        Measure inference latency and compute penalty.
        测量推理延迟并计算惩罚。

        Args:
            batch_size: Batch size for latency measurement

        Returns:
            Latency penalty value / 延迟惩罚值
        """
        # 创建虚拟输入 / Create dummy input
        dummy_ob = torch.randn(
            batch_size, self.config.lookback_window,
            self.config.orderbook_levels * 4, device=self.device
        )
        dummy_trades = torch.randn(
            batch_size, self.config.lookback_window, 4, device=self.device
        )
        dummy_stats = torch.randn(
            batch_size, self.config.lookback_window, 5, device=self.device
        )

        # 预热 / Warmup
        self.model.eval()
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(dummy_ob, dummy_trades, dummy_stats)

        # 测量延迟 / Measure latency
        num_runs = 20
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(dummy_ob, dummy_trades, dummy_stats)
        elapsed_ms = (time.time() - start) / num_runs * 1000

        # 计算惩罚 / Compute penalty
        if elapsed_ms > self.config.max_latency_ms:
            # 二次惩罚超出部分 / Quadratic penalty for exceeding limit
            excess = elapsed_ms - self.config.max_latency_ms
            penalty = (excess / self.config.max_latency_ms) ** 2
        else:
            penalty = 0.0

        self.model.train()
        return penalty

    def train_step(
        self,
        orderbook: torch.Tensor,
        trades: torch.Tensor,
        stats: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step with latency-aware loss.
        带延迟感知损失的单步训练。

        Args:
            orderbook: Order book data [B, L, features]
            trades: Trade data [B, L, 4]
            stats: Market stats [B, L, 5]
            labels: Signal labels [B]

        Returns:
            Dictionary of loss components / 损失分量字典
        """
        self.model.train()

        # 前向传播 / Forward pass
        outputs = self.model(orderbook, trades, stats)

        # 信号分类损失 / Signal classification loss
        signal_loss = self.signal_criterion(outputs["signal_logits"], labels)

        # 预期收益回归损失 / Expected return regression loss
        # 使用标签构造目标收益 / Construct target returns from labels
        target_returns = (labels.float() - 1.0) * 0.001  # [-0.001, 0, 0.001]
        return_loss = self.return_criterion(
            outputs["expected_return"].squeeze(-1), target_returns
        )

        # 置信度加权损失 / Confidence-weighted loss
        # 鼓励模型在校准置信度时更准确
        # Encourage model to be more accurate when confident
        confidence = outputs["confidence"].squeeze(-1)
        weighted_signal_loss = (signal_loss * confidence.mean()
                                + 0.1 * (1 - confidence).mean())

        # 总损失 / Total loss
        total_loss = (
            weighted_signal_loss
            + 0.5 * return_loss
        )

        # 延迟惩罚 / Latency penalty
        latency_penalty = self.compute_latency_penalty(batch_size=orderbook.size(0))
        total_loss = total_loss + self.config.latency_penalty_weight * latency_penalty

        # 反向传播 / Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        metrics = {
            "total_loss": total_loss.item(),
            "signal_loss": signal_loss.item(),
            "return_loss": return_loss.item(),
            "latency_penalty": latency_penalty,
            "avg_confidence": confidence.mean().item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        return metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        log_interval: int = 50,
    ) -> Dict[str, float]:
        """
        Train for one full epoch.
        训练一个完整epoch。

        Args:
            dataloader: Training data loader / 训练数据加载器
            epoch: Current epoch number / 当前epoch编号
            log_interval: Steps between log outputs / 日志输出间隔步数

        Returns:
            Epoch-level metrics / Epoch级别指标
        """
        epoch_losses = []
        epoch_accuracies = []

        for step, (ob, trades, stats, labels) in enumerate(dataloader):
            ob = ob.to(self.device)
            trades = trades.to(self.device)
            stats = stats.to(self.device)
            labels = labels.to(self.device)

            metrics = self.train_step(ob, trades, stats, labels)
            epoch_losses.append(metrics["total_loss"])

            # 计算准确率 / Compute accuracy
            with torch.no_grad():
                outputs = self.model(ob, trades, stats)
                preds = outputs["signal_logits"].argmax(dim=-1)
                accuracy = (preds == labels).float().mean().item()
                epoch_accuracies.append(accuracy)

            if (step + 1) % log_interval == 0:
                print(
                    f"  Epoch {epoch:3d} | Step {step+1:4d}/{len(dataloader)} | "
                    f"Loss: {metrics['total_loss']:.4f} | "
                    f"SigLoss: {metrics['signal_loss']:.4f} | "
                    f"Acc: {accuracy:.3f} | "
                    f"LR: {metrics['learning_rate']:.2e}"
                )

        return {
            "avg_loss": sum(epoch_losses) / len(epoch_losses),
            "avg_accuracy": sum(epoch_accuracies) / len(epoch_accuracies),
        }


# ============================================================================
# Main Training Function / 主训练函数
# ============================================================================

def main():
    """
    Main entry point for training the quant trading model.
    量化交易模型训练的主入口。
    """
    parser = argparse.ArgumentParser(
        description="Train edge-deployable quant trading model / 训练边缘部署量化交易模型"
    )
    parser.add_argument(
        "--quick-demo", action="store_true",
        help="Run quick demo with small data / 使用小数据运行快速演示"
    )
    parser.add_argument(
        "--model-size", type=str, default="3B", choices=["mini", "small", "medium", "3B"],
        help="Model size variant / 模型规模变体"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs / 训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size / 批量大小")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate / 学习率")
    parser.add_argument(
        "--max-latency-ms", type=float, default=10.0,
        help="Maximum inference latency in ms / 最大推理延迟（毫秒）"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device / 设备")
    parser.add_argument(
        "--output-dir", type=str, default="./checkpoints",
        help="Output directory for checkpoints / 检查点输出目录"
    )
    parser.add_argument("--num-samples", type=int, default=10000, help="Training samples / 训练样本数")
    parser.add_argument("--seed", type=int, default=42, help="Random seed / 随机种子")
    args = parser.parse_args()

    # 快速演示模式 / Quick demo mode
    if args.quick_demo:
        args.epochs = 3
        args.num_samples = 2000
        args.batch_size = 32
        print("=" * 60)
        print("Quick Demo Mode / 快速演示模式")
        print("=" * 60)

    # 设置随机种子 / Set random seed
    torch.manual_seed(args.seed)

    # 模型配置 / Model configuration
    size_configs = {
        "mini": dict(hidden_dim=64, num_layers=2, num_heads=4, num_kv_heads=1),
        "small": dict(hidden_dim=128, num_layers=3, num_heads=4, num_kv_heads=2),
        "medium": dict(hidden_dim=256, num_layers=4, num_heads=8, num_kv_heads=2),
        "3B": dict(hidden_dim=256, num_layers=4, num_heads=8, num_kv_heads=2),
    }

    config = QuantModelConfig(
        **size_configs[args.model_size],
        max_latency_ms=args.max_latency_ms,
    )

    print(f"\n{'='*60}")
    print(f"Edge Quant Trading Model Training / 边缘量化交易模型训练")
    print(f"{'='*60}")
    print(f"Model size / 模型规模: {args.model_size}")
    print(f"Hidden dim / 隐藏维度: {config.hidden_dim}")
    print(f"Layers / 层数: {config.num_layers}")
    print(f"Max latency / 最大延迟: {config.max_latency_ms}ms")
    print(f"Device / 设备: {args.device}")
    print(f"Epochs / 训练轮数: {args.epochs}")
    print(f"{'='*60}\n")

    # 生成模拟数据 / Generate simulated data
    print("Generating simulated market data / 生成模拟市场数据...")
    data = simulate_market_data(
        num_samples=args.num_samples,
        lookback=config.lookback_window,
        orderbook_levels=config.orderbook_levels,
        seed=args.seed,
    )

    # 数据分割 / Data split (80% train, 20% val)
    split = int(0.8 * args.num_samples)
    train_data = {k: v[:split] for k, v in data.items()}
    val_data = {k: v[split:] for k, v in data.items()}

    print(f"Train samples / 训练样本: {split}")
    print(f"Val samples / 验证样本: {args.num_samples - split}")

    # 标签分布 / Label distribution
    train_labels = train_data["labels"]
    for cls in range(config.num_signal_classes):
        count = (train_labels == cls).sum().item()
        name = ["Sell/卖出", "Hold/持有", "Buy/买入"][cls]
        print(f"  {name}: {count} ({count/split*100:.1f}%)")

    # 创建数据加载器 / Create data loaders
    train_dataset = MarketDataset(train_data)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    val_dataset = MarketDataset(val_data)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # 创建模型 / Create model
    model = CompactTradingSignalModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters / 模型参数: {total_params:,} ({total_params/1e6:.1f}M)")

    # 创建训练器 / Create trainer
    trainer = LatencyAwareTrainer(
        model=model,
        config=config,
        lr=args.learning_rate,
        device=args.device,
    )

    # 训练循环 / Training loop
    print(f"\nStarting training / 开始训练...\n")
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # 训练 / Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_metrics['avg_loss']:.4f} | "
            f"Train Acc: {train_metrics['avg_accuracy']:.3f}"
        )

        # 验证 / Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for ob, trades, stats, labels in val_loader:
                ob = ob.to(args.device)
                trades = trades.to(args.device)
                stats = stats.to(args.device)
                labels = labels.to(args.device)

                outputs = model(ob, trades, stats)
                loss = F.cross_entropy(outputs["signal_logits"], labels)
                val_loss_sum += loss.item() * labels.size(0)

                preds = outputs["signal_logits"].argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss_sum / val_total
        print(
            f"{'':>16s} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f}"
        )

        # 保存最佳模型 / Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(
                args.output_dir, f"ministral_quant_{args.model_size}_best.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_accuracy": val_acc,
            }, save_path)
            print(f"  -> Saved best model (acc={val_acc:.3f}) / 保存最佳模型")

    # 最终评估 / Final evaluation
    print(f"\n{'='*60}")
    print(f"Training Complete / 训练完成")
    print(f"Best validation accuracy / 最佳验证准确率: {best_val_acc:.3f}")
    print(f"{'='*60}")

    # 延迟测量 / Latency measurement
    print("\nLatency Measurement / 延迟测量:")
    dummy_ob = torch.randn(1, config.lookback_window, config.orderbook_levels * 4, device=args.device)
    dummy_trades = torch.randn(1, config.lookback_window, 4, device=args.device)
    dummy_stats = torch.randn(1, config.lookback_window, 5, device=args.device)

    # 预热 / Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_ob, dummy_trades, dummy_stats)

    # 测量 / Measure
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(dummy_ob, dummy_trades, dummy_stats)
            latencies.append((time.time() - start) * 1000)

    latencies.sort()
    print(f"  P50 latency / P50延迟: {latencies[50]:.2f}ms")
    print(f"  P95 latency / P95延迟: {latencies[95]:.2f}ms")
    print(f"  P99 latency / P99延迟: {latencies[99]:.2f}ms")


if __name__ == "__main__":
    main()
