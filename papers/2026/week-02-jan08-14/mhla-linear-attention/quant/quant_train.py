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
MHLA for High-Frequency Market Prediction
MHLA用于高频市场预测

Applies the Multi-Head Linear Attention (MHLA) mechanism to high-frequency
time series prediction in financial markets. The O(n) complexity of linear
attention is particularly advantageous for processing long tick-level
sequences in real-time trading systems.

将多头线性注意力（MHLA）机制应用于金融市场的高频时间序列预测。
线性注意力的O(n)复杂度对于实时交易系统中处理长tick级序列特别有利。

Components / 组件:
    - EfficientMarketAttention: MHLA applied to market microstructure
      MHLA应用于市场微观结构
    - TokenLevelMarketEncoder: Token-level encoding of tick data
      tick数据的token级编码
    - HighFrequencyForecaster: Real-time prediction model
      实时预测模型
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
class MHLAMarketConfig:
    """
    Configuration for MHLA-based market prediction model.
    基于MHLA的市场预测模型配置。
    """

    # 模型架构 / Model architecture
    hidden_dim: int = 128           # 隐藏维度 / Hidden dimension
    num_layers: int = 4             # Transformer层数 / Number of layers
    num_heads: int = 4              # 注意力头数 / Number of attention heads
    token_heads: int = 4            # Token级多头数 / Token-level heads (MHLA core)
    feature_dim: int = 32           # 线性注意力特征维度 / Linear attention feature dim

    # 线性注意力核函数 / Linear attention kernel
    # 可选 / Options: "elu_plus", "relu", "rff"
    kernel_type: str = "elu_plus"

    # 市场数据参数 / Market data parameters
    num_tick_features: int = 16     # 每个tick的特征数 / Features per tick
    lookback_ticks: int = 256       # 回看tick数 / Lookback ticks
    prediction_horizon: int = 50    # 预测时域（ticks）/ Prediction horizon (ticks)
    num_output_classes: int = 5     # 输出类别数 / Output classes (quintile returns)

    # 高频特有参数 / High-frequency specific parameters
    price_levels: int = 10          # 价格 discretization levels
    volume_buckets: int = 8         # 成交量分桶数 / Volume buckets
    time_encoding_dim: int = 16     # 时间编码维度 / Time encoding dimension

    # 延迟约束 / Latency constraints
    max_latency_ms: float = 5.0    # 最大延迟 / Maximum latency (ms)
    latency_penalty: float = 0.05  # 延迟惩罚系数 / Latency penalty coefficient

    # 训练参数 / Training parameters
    dropout: float = 0.1
    ffn_scale: float = 3.0         # FFN缩放 / FFN scale factor


# ============================================================================
# Linear Attention Kernel / 线性注意力核函数
# ============================================================================

class MarketKernel(nn.Module):
    """
    Kernel function adapted for market microstructure features.
    适应市场微观结构特征的核函数。

    Maps market features into a feature space where linear attention
    can effectively capture temporal dependencies.
    将市场特征映射到线性注意力可以有效捕获时间依赖关系的特征空间。
    """

    def __init__(self, input_dim: int, feature_dim: int, kernel_type: str = "elu_plus"):
        super().__init__()
        self.kernel_type = kernel_type
        self.feature_dim = feature_dim

        # 可学习的核投影 / Learnable kernel projection
        self.projection = nn.Linear(input_dim, feature_dim, bias=False)
        nn.init.normal_(self.projection.weight, std=0.02)

        if kernel_type == "rff":
            # 随机傅里叶特征的频率参数
            # Frequency parameters for random Fourier features
            self.register_buffer(
                "freq",
                torch.randn(feature_dim // 2) * math.sqrt(2.0 / input_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply kernel transformation.
        应用核函数变换。

        Args:
            x: Input features [..., input_dim]

        Returns:
            Kernel-mapped features [..., feature_dim]
        """
        projected = self.projection(x)

        if self.kernel_type == "elu_plus":
            # ELU+1: 保证非负性，适合注意力权重
            # ELU+1: Ensures non-negativity, suitable for attention weights
            return F.elu(projected) + 1.0

        elif self.kernel_type == "relu":
            return F.relu(projected) + 1e-6

        elif self.kernel_type == "rff":
            # cos(Wx) / sqrt(d) approximation
            cos_features = torch.cos(projected[..., : self.feature_dim // 2])
            sin_features = torch.sin(projected[..., : self.feature_dim // 2])
            return torch.cat([cos_features, sin_features], dim=-1) / math.sqrt(
                self.feature_dim
            )
        else:
            raise ValueError(f"Unknown kernel / 未知核函数: {self.kernel_type}")


# ============================================================================
# Token-Level Market Encoder / Token级市场编码器
# ============================================================================

class TokenLevelMarketEncoder(nn.Module):
    """
    Token-level encoding of tick data using MHLA mechanism.
    使用MHLA机制对tick数据进行token级编码。

    Each tick is encoded with multiple projection heads, enabling
    the model to capture different aspects of market microstructure
    (price dynamics, volume patterns, order flow imbalance) simultaneously.

    每个tick使用多个投影头编码，使模型能够同时捕获市场微观结构的
    不同方面（价格动态、成交量模式、订单流不平衡）。
    """

    def __init__(self, config: MHLAMarketConfig):
        super().__init__()
        self.config = config

        # Tick特征投影 / Tick feature projection
        self.tick_proj = nn.Linear(config.num_tick_features, config.hidden_dim)

        # Token级多头投影 / Token-level multi-head projection
        # 每个token（tick）有token_heads个独立的投影子空间
        # Each token (tick) has token_heads independent projection subspaces
        self.token_projections = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.feature_dim, bias=False)
            for _ in range(config.token_heads)
        ])

        # 每个头的核函数 / Kernel function for each head
        self.kernels = nn.ModuleList([
            MarketKernel(config.feature_dim, config.feature_dim, config.kernel_type)
            for _ in range(config.token_heads)
        ])

        # 时间编码 / Temporal encoding
        self.time_encoder = TimeEncoding(config.time_encoding_dim, config.hidden_dim)

        # 合并层 / Fusion layer
        self.fusion = nn.Linear(
            config.token_heads * config.feature_dim + config.hidden_dim,
            config.hidden_dim,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, tick_data: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode tick data with token-level multi-head mechanism.
        使用token级多头机制编码tick数据。

        Args:
            tick_data: [B, T, num_tick_features] raw tick data
            timestamps: [B, T] timestamp for each tick

        Returns:
            Encoded representations [B, T, hidden_dim]
        """
        B, T, _ = tick_data.shape

        # 基础投影 / Base projection
        base_features = self.tick_proj(tick_data)  # [B, T, hidden_dim]

        # 添加时间编码 / Add time encoding
        time_features = self.time_encoder(timestamps)  # [B, T, hidden_dim]
        base_features = base_features + time_features

        # Token级多头编码 / Token-level multi-head encoding
        token_head_outputs = []
        for head_idx in range(self.config.token_heads):
            # 每个头独立投影 / Each head projects independently
            projected = self.token_projections[head_idx](base_features)
            # 应用核函数 / Apply kernel
            kernel_output = self.kernels[head_idx](projected)
            token_head_outputs.append(kernel_output)

        # 拼接所有token头的输出 / Concatenate all token head outputs
        multi_head_features = torch.cat(token_head_outputs, dim=-1)
        # [B, T, token_heads * feature_dim]

        # 与基础特征合并 / Merge with base features
        combined = torch.cat([multi_head_features, base_features], dim=-1)
        encoded = self.fusion(combined)  # [B, T, hidden_dim]

        return self.layer_norm(encoded)


class TimeEncoding(nn.Module):
    """
    Continuous time encoding for irregular tick intervals.
    不规则tick间隔的连续时间编码。

    Uses sinusoidal encoding adapted for financial time series where
    tick intervals are irregular and carry information about market activity.
    使用适合金融时间序列的正弦编码，其中tick间隔不规则
    且携带关于市场活跃度的信息。
    """

    def __init__(self, encoding_dim: int, output_dim: int):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.projection = nn.Linear(encoding_dim, output_dim)

        # 多尺度频率 / Multi-scale frequencies
        # 从tick级到日级的时间尺度 / Time scales from tick to daily
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),  # 最快频率 / Fastest frequency
                math.log(1.0 / 23400),  # 日级 / Daily (6.5h * 3600s)
                encoding_dim,
            )
        )
        self.register_buffer("freqs", freqs)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode timestamps into continuous embeddings.
        将时间戳编码为连续嵌入。

        Args:
            timestamps: [B, T] timestamp values (seconds from market open)

        Returns:
            Time embeddings [B, T, output_dim]
        """
        # 正弦时间编码 / Sinusoidal time encoding
        angles = timestamps.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        # 截断或填充到encoding_dim / Truncate or pad to encoding_dim
        encoding = encoding[..., : self.encoding_dim]
        return self.projection(encoding)


# ============================================================================
# Efficient Market Attention / 高效市场注意力
# ============================================================================

class EfficientMarketAttention(nn.Module):
    """
    MHLA applied to market microstructure attention.
    将MHLA应用于市场微观结构注意力。

    Uses linear attention with token-level multi-head for O(n) complexity
    while maintaining the expressivity needed to capture complex
    market microstructure patterns.
    使用带token级多头的线性注意力实现O(n)复杂度，
    同时保持捕获复杂市场微观结构模式所需的表达能力。
    """

    def __init__(self, config: MHLAMarketConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.feature_dim = config.feature_dim
        self.token_heads = config.token_heads

        # QKV投影 / QKV projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # 核函数 / Kernel functions
        self.q_kernel = MarketKernel(self.head_dim, self.feature_dim, config.kernel_type)
        self.k_kernel = MarketKernel(self.head_dim, self.feature_dim, config.kernel_type)

        # Token级多头投影 / Token-level multi-head projections
        self.token_q_proj = nn.ModuleList([
            nn.Linear(self.feature_dim, self.feature_dim, bias=False)
            for _ in range(config.token_heads)
        ])
        self.token_k_proj = nn.ModuleList([
            nn.Linear(self.feature_dim, self.feature_dim, bias=False)
            for _ in range(config.token_heads)
        ])

        # Token级多头输出合并 / Token-level multi-head output fusion
        self.token_fusion = nn.Linear(
            config.token_heads * self.feature_dim, self.feature_dim
        )

    def forward(self, hidden_states: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Forward pass with MHLA linear attention on market data.
        在市场数据上使用MHLA线性注意力的前向传播。

        Args:
            hidden_states: [B, T, hidden_dim]
            causal: Apply causal masking / 应用因果掩码

        Returns:
            Attention output [B, T, hidden_dim]
        """
        B, T, _ = hidden_states.shape

        # QKV投影并重塑 / QKV projection and reshape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, H, T, head_dim]

        # 核映射 / Kernel mapping
        q_feat = self.q_kernel(q)  # [B, H, T, feature_dim]
        k_feat = self.k_kernel(k)

        # Token级多头增强 / Token-level multi-head enhancement
        q_enhanced_list = []
        k_enhanced_list = []
        for th in range(self.config.token_heads):
            q_th = self.token_q_proj[th](q_feat)
            k_th = self.token_k_proj[th](k_feat)
            # ReLU确保非负性 / ReLU ensures non-negativity
            q_enhanced_list.append(F.relu(q_th) + 1e-6)
            k_enhanced_list.append(F.relu(k_th) + 1e-6)

        # 合并token级多头 / Fuse token-level heads
        q_enhanced = self.token_fusion(torch.cat(q_enhanced_list, dim=-1))
        k_enhanced = self.token_fusion(torch.cat(k_enhanced_list, dim=-1))

        # 线性注意力计算 / Linear attention computation
        if causal:
            # 因果线性注意力（前缀和）/ Causal linear attention (prefix sum)
            kv = k_enhanced.unsqueeze(-1) * v.unsqueeze(-2)
            # [B, H, T, feature_dim, head_dim]
            cumulative_kv = torch.cumsum(kv, dim=2)
            cumulative_k = torch.cumsum(k_enhanced, dim=2)

            numerator = torch.einsum("bhte,bhtev->bhtv", q_enhanced, cumulative_kv)
            denominator = torch.einsum("bhte,bhte->bht", q_enhanced, cumulative_k)
            output = numerator / denominator.unsqueeze(-1).clamp(min=1e-6)
        else:
            # 双向线性注意力 / Bidirectional linear attention
            kv_context = torch.einsum("bhte,bhtv->bhev", k_enhanced, v)
            k_sum = k_enhanced.sum(dim=2, keepdim=True)

            numerator = torch.einsum("bhte,bhev->bhtv", q_enhanced, kv_context)
            denominator = torch.einsum("bhte,bhse->bht", q_enhanced, k_sum)
            output = numerator / denominator.unsqueeze(-1).clamp(min=1e-6)

        # 重塑并输出投影 / Reshape and output projection
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(output)


# ============================================================================
# High-Frequency Forecaster / 高频预测器
# ============================================================================

class HighFrequencyForecaster(nn.Module):
    """
    Real-time high-frequency market prediction model using MHLA.
    使用MHLA的实时高频市场预测模型。

    Architecture:
    架构:
        Tick Data -> TokenLevelMarketEncoder -> MHLA Transformer Layers
        -> Prediction Head (return quintiles + confidence)
    """

    def __init__(self, config: MHLAMarketConfig):
        super().__init__()
        self.config = config

        # 市场编码器 / Market encoder
        self.encoder = TokenLevelMarketEncoder(config)

        # MHLA Transformer层 / MHLA Transformer layers
        self.layers = nn.ModuleList([
            MHLAMarketBlock(config) for _ in range(config.num_layers)
        ])

        # 最终归一化 / Final normalization
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # 预测头 / Prediction heads
        # 收益五分位预测 / Return quintile prediction
        self.quintile_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_output_classes),
        )

        # 波动率预测 / Volatility prediction
        self.volatility_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Softplus(),
        )

        # 置信度估计 / Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Weight initialization / 权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tick_data: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for high-frequency prediction.
        高频预测的前向传播。

        Args:
            tick_data: [B, T, num_tick_features] raw tick features
            timestamps: [B, T] timestamps for each tick

        Returns:
            Dictionary with quintile logits, volatility, and confidence
            包含五分位logits、波动率和置信度的字典
        """
        # 编码 / Encode
        hidden = self.encoder(tick_data, timestamps)

        # Transformer处理 / Transformer processing
        for layer in self.layers:
            hidden = layer(hidden)

        hidden = self.final_norm(hidden)

        # 使用最后一个时间步 / Use last time step
        last_hidden = hidden[:, -1, :]

        return {
            "quintile_logits": self.quintile_head(last_hidden),
            "volatility": self.volatility_head(last_hidden),
            "confidence": self.confidence_head(last_hidden),
            "all_hidden": hidden,
        }


class MHLAMarketBlock(nn.Module):
    """
    Transformer block with MHLA market attention.
    使用MHLA市场注意力的Transformer块。
    """

    def __init__(self, config: MHLAMarketConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_dim)
        self.attention = EfficientMarketAttention(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)

        # SwiGLU FFN / SwiGLU FFN
        intermediate = int(config.hidden_dim * config.ffn_scale)
        self.ffn_gate = nn.Linear(config.hidden_dim, intermediate, bias=False)
        self.ffn_up = nn.Linear(config.hidden_dim, intermediate, bias=False)
        self.ffn_down = nn.Linear(intermediate, config.hidden_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass / 前向传播"""
        # 注意力 + 残差 / Attention + residual
        residual = x
        x = self.attn_norm(x)
        x = self.attention(x, causal=True)
        x = self.dropout(x)
        x = residual + x

        # FFN + 残差 / FFN + residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))
        x = self.dropout(x)
        x = residual + x

        return x


# ============================================================================
# Simulate High-Frequency Data / 模拟高频数据
# ============================================================================

def simulate_hf_market_data(
    num_samples: int = 5000,
    num_ticks: int = 256,
    num_features: int = 16,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate simulated high-frequency market data.
    生成模拟的高频市场数据。

    Simulates realistic tick-level data including:
    模拟逼真的tick级数据，包括：
    - Price movements with microstructure noise / 带微观结构噪声的价格变动
    - Volume patterns with clustering / 带聚集效应的成交量模式
    - Bid-ask spread dynamics / 买卖价差动态
    - Order flow imbalance / 订单流不平衡

    Args:
        num_samples: Number of sequences / 序列数
        num_ticks: Ticks per sequence / 每序列tick数
        num_features: Features per tick / 每tick特征数
        seed: Random seed / 随机种子

    Returns:
        Dictionary with tick_data, timestamps, labels, and prices
    """
    torch.manual_seed(seed)

    # 价格过程 / Price process
    prices = torch.zeros(num_samples, num_ticks)
    prices[:, 0] = 100.0

    # 使用Hawkes过程模拟到达率 / Use Hawkes process for arrival rates
    base_intensity = 10.0  # 基础到达率 / Base arrival rate
    excitation = 0.5       # 激发参数 / Excitation parameter
    decay = 5.0            # 衰减率 / Decay rate

    tick_data = torch.zeros(num_samples, num_ticks, num_features)
    timestamps = torch.zeros(num_samples, num_ticks)

    for i in range(num_samples):
        current_time = 0.0
        intensity = base_intensity

        for t in range(num_ticks):
            # 模拟tick间隔 / Simulate tick interval
            dt = torch.exp(torch.randn() * 0.3) / intensity
            current_time += dt
            timestamps[i, t] = current_time

            # 价格变动 / Price movement
            if t > 0:
                vol = 0.0001 * (1 + 0.5 * abs(tick_data[i, t-1, 1]))  # Vol dependent on volume
                dprice = vol * torch.randn() * prices[i, t-1]
                prices[i, t] = prices[i, t-1] + dprice
            else:
                prices[i, t] = 100.0

            # 更新Hawkes强度 / Update Hawkes intensity
            intensity = base_intensity + excitation * math.exp(-decay * dt)

            # 填充tick特征 / Fill tick features
            tick_data[i, t, 0] = (prices[i, t] - 100.0) / 100.0  # Normalized price
            tick_data[i, t, 1] = torch.exp(torch.randn() * 0.5 + 2.0)  # Volume
            tick_data[i, t, 2] = torch.sign(torch.randn())  # Direction
            tick_data[i, t, 3] = abs(tick_data[i, t, 0] - (tick_data[i, t-1, 0] if t > 0 else 0))  # Abs return
            # Spread proxy / 价差代理
            tick_data[i, t, 4] = abs(torch.randn()) * 0.001
            # Order imbalance / 订单不平衡
            tick_data[i, t, 5] = torch.tanh(torch.randn())
            # VWAP deviation / VWAP偏离
            tick_data[i, t, 6] = tick_data[i, t, 0] - (
                prices[i, :t+1].mean() - 100.0
            ) / 100.0
            # Remaining features: random market noise
            # 剩余特征：随机市场噪声
            for f in range(7, num_features):
                tick_data[i, t, f] = torch.randn() * 0.1

    # 归一化 / Normalize
    tick_mean = tick_data.mean(dim=(0, 1), keepdim=True)
    tick_std = tick_data.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
    tick_data = (tick_data - tick_mean) / tick_std

    # 生成标签：未来收益五分位 / Generate labels: future return quintiles
    horizon = min(50, num_ticks - 1)
    future_returns = (prices[:, -1] - prices[:, -horizon]) / prices[:, -horizon]
    # 五分位 / Quintiles
    _, thresholds = torch.sort(future_returns)
    q20 = thresholds[num_samples // 5]
    q40 = thresholds[2 * num_samples // 5]
    q60 = thresholds[3 * num_samples // 5]
    q80 = thresholds[4 * num_samples // 5]

    labels = torch.zeros(num_samples, dtype=torch.long)
    labels[future_returns >= q20] = 1
    labels[future_returns >= q40] = 2
    labels[future_returns >= q60] = 3
    labels[future_returns >= q80] = 4

    return {
        "tick_data": tick_data,
        "timestamps": timestamps,
        "labels": labels,
        "prices": prices,
        "future_returns": future_returns,
    }


# ============================================================================
# Dataset / 数据集
# ============================================================================

class HFMarketDataset(Dataset):
    """High-frequency market dataset / 高频市场数据集"""

    def __init__(self, data: Dict[str, torch.Tensor]):
        self.tick_data = data["tick_data"]
        self.timestamps = data["timestamps"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tick_data[idx], self.timestamps[idx], self.labels[idx]


# ============================================================================
# Training with Latency Constraints / 带延迟约束的训练
# ============================================================================

def train_hf_model(
    config: MHLAMarketConfig,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    num_samples: int = 5000,
    device: str = "cpu",
) -> Tuple[HighFrequencyForecaster, Dict]:
    """
    Train the high-frequency forecaster with latency constraints.
    训练带延迟约束的高频预测器。

    Args:
        config: Model configuration / 模型配置
        epochs: Number of training epochs / 训练轮数
        batch_size: Training batch size / 训练批量大小
        lr: Learning rate / 学习率
        num_samples: Number of training samples / 训练样本数
        device: Compute device / 计算设备

    Returns:
        Tuple of (trained model, training metrics)
    """
    print(f"\n{'='*60}")
    print(f"Training High-Frequency MHLA Forecaster / 训练高频MHLA预测器")
    print(f"{'='*60}")

    # 生成数据 / Generate data
    print("Generating HF market data / 生成高频市场数据...")
    data = simulate_hf_market_data(
        num_samples=num_samples,
        num_ticks=config.lookback_ticks,
        num_features=config.num_tick_features,
    )

    # 数据分割 / Data split
    split = int(0.8 * num_samples)
    train_data = {k: v[:split] for k, v in data.items()}
    val_data = {k: v[split:] for k, v in data.items()}

    train_dataset = HFMarketDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = HFMarketDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型 / Create model
    model = HighFrequencyForecaster(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters / 模型参数: {total_params:,} ({total_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    all_metrics = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for step, (ticks, ts, labels) in enumerate(train_loader):
            ticks = ticks.to(device)
            ts = ts.to(device)
            labels = labels.to(device)

            outputs = model(ticks, ts)
            logits = outputs["quintile_logits"]

            # 分类损失 / Classification loss
            cls_loss = criterion(logits, labels)

            # 延迟惩罚 / Latency penalty
            latency_penalty = 0.0
            if config.max_latency_ms > 0:
                # 估算延迟 / Estimate latency
                start = time.time()
                with torch.no_grad():
                    _ = model(ticks[:4], ts[:4])
                est_latency = (time.time() - start) * 1000
                if est_latency > config.max_latency_ms:
                    latency_penalty = config.latency_penalty * (
                        (est_latency - config.max_latency_ms) / config.max_latency_ms
                    ) ** 2

            total_loss = cls_loss + latency_penalty

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)
            epoch_loss += total_loss.item() * labels.size(0)

        scheduler.step()

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total

        # 验证 / Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for ticks, ts, labels in val_loader:
                ticks = ticks.to(device)
                ts = ts.to(device)
                labels = labels.to(device)

                outputs = model(ticks, ts)
                logits = outputs["quintile_logits"]
                loss = criterion(logits, labels)

                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item() * labels.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss / val_total

        metrics = {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        }
        all_metrics.append(metrics)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}"
        )

    print(f"\nBest validation accuracy / 最佳验证准确率: {best_val_acc:.3f}")
    return model, {"metrics": all_metrics, "best_val_acc": best_val_acc}


# ============================================================================
# Main / 主程序
# ============================================================================

def main():
    """Main entry point / 主入口"""
    parser = argparse.ArgumentParser(
        description="Train MHLA high-frequency market forecaster / 训练MHLA高频市场预测器"
    )
    parser.add_argument("--quick-demo", action="store_true", help="Quick demo / 快速演示")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs / 训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size / 批量大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate / 学习率")
    parser.add_argument("--device", type=str, default="cpu", help="Device / 设备")
    parser.add_argument("--num-samples", type=int, default=5000, help="Samples / 样本数")
    parser.add_argument("--seed", type=int, default=42, help="Random seed / 随机种子")
    args = parser.parse_args()

    if args.quick_demo:
        args.epochs = 3
        args.num_samples = 1000
        args.batch_size = 32
        print("Quick Demo Mode / 快速演示模式")

    torch.manual_seed(args.seed)

    config = MHLAMarketConfig()
    model, results = train_hf_model(
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.num_samples,
        device=args.device,
    )

    # 保存模型 / Save model
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "best_val_acc": results["best_val_acc"],
    }, "./checkpoints/mhla_hf_forecaster.pt")
    print("Model saved / 模型已保存: ./checkpoints/mhla_hf_forecaster.pt")


if __name__ == "__main__":
    main()
