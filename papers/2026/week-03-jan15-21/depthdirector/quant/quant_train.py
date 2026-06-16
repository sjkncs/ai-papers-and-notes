# ==============================================================================
# MIT License
#
# Copyright (c) 2026 AI Papers and Notes
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
# ==============================================================================

"""
DepthDirector Quantitative Finance Training / 深度导演量化金融训练

Applies DepthDirector's depth-guided generation concepts to quantitative finance:
- Model price "depth" as multi-layer order book
- Generate market scenarios with depth awareness
- "Camera control" = perspective shift in market analysis

将DepthDirector的深度引导生成概念应用于量化金融：
- 将价格"深度"建模为多层订单簿
- 生成具有深度感知能力的市场情景
- "相机控制" = 市场分析中的视角转换

Usage / 用法:
    python quant_train.py --data-dir data/ --epochs 100
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 配置日志 / Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ============================================================================
# 数据类 / Data Classes
# ============================================================================


class MarketPerspective(Enum):
    """
    市场视角枚举 / Market perspective enumeration

    类比DepthDirector中的"相机控制"——不同的分析视角揭示不同的市场信息。
    Analogous to "camera control" in DepthDirector — different analytical
    perspectives reveal different market information.
    """

    MACRO = "macro"           # 宏观视角 / Macro perspective
    SECTOR = "sector"         # 板块视角 / Sector perspective
    INDIVIDUAL = "individual" # 个股视角 / Individual stock perspective
    MICRO = "micro"           # 微观结构视角 / Micro-structure perspective


@dataclass
class OrderBookSnapshot:
    """
    订单簿快照 / Order book snapshot

    模拟市场深度数据，类似DepthDirector中的深度图。
    Simulates market depth data, analogous to depth maps in DepthDirector.
    """

    bid_prices: torch.Tensor      # 买入价格 / Bid prices
    bid_sizes: torch.Tensor       # 买入数量 / Bid sizes
    ask_prices: torch.Tensor      # 卖出价格 / Ask prices
    ask_sizes: torch.Tensor       # 卖出数量 / Ask sizes
    mid_price: float = 0.0       # 中间价 / Mid price
    spread: float = 0.0          # 价差 / Spread
    imbalance: float = 0.0       # 买卖不平衡 / Bid-ask imbalance


@dataclass
class TrainingConfig:
    """训练配置 / Training configuration"""

    data_dir: str = "data/market"
    output_dir: str = "results/depth_quant"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # 模型参数 / Model parameters
    price_features: int = 64
    depth_levels: int = 20
    hidden_dim: int = 256
    embed_dim: int = 128
    n_heads: int = 8
    n_layers: int = 4
    # 深度感知参数 / Depth-aware parameters
    n_perspectives: int = 4
    trajectory_length: int = 50
    depth_consistency_weight: float = 0.3
    # 其他 / Other
    seed: int = 42
    device: str = "auto"
    log_interval: int = 10


# ============================================================================
# 价格轨迹深度模型 / Price Trajectory Depth Model
# ============================================================================


class PriceTrajectoryDepthModel(nn.Module):
    """
    价格轨迹深度模型 / Price trajectory depth model

    将订单簿建模为多层"深度图"，类似DepthDirector中使用显式深度信息。
    每一层订单簿代表市场的一个"深度层"。

    Models order book as multi-layer "depth maps", similar to how DepthDirector
    uses explicit depth information. Each layer of the order book represents
    a "depth layer" of the market.

    Args:
        price_features: 价格特征维度 / Price feature dimension
        depth_levels: 订单簿深度层数 / Order book depth levels
        hidden_dim: 隐藏层维度 / Hidden dimension
        embed_dim: 嵌入维度 / Embedding dimension
        n_heads: 注意力头数 / Number of attention heads
    """

    def __init__(
        self,
        price_features: int = 64,
        depth_levels: int = 20,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.price_features = price_features
        self.depth_levels = depth_levels
        self.hidden_dim = hidden_dim

        # 价格序列编码器 / Price sequence encoder
        self.price_encoder = nn.Sequential(
            nn.Linear(price_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # 深度编码器：将订单簿深度编码为特征
        # Depth encoder: encode order book depth into features
        # 类比DepthDirector的深度图编码器 / Analogous to DepthDirector's depth map encoder
        self.depth_encoder = nn.Sequential(
            nn.Linear(depth_levels * 4, hidden_dim),  # 4 = bid_price, bid_size, ask_price, ask_size
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # 深度感知卷积 / Depth-aware convolution
        # 利用深度信息引导卷积的注意力 / Use depth info to guide convolution attention
        self.depth_aware_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.BatchNorm1d(hidden_dim),
            )
            for _ in range(3)
        ])

        # 深度-价格融合注意力 / Depth-price fusion attention
        # 双流交叉注意力：价格流和深度流
        # Dual-stream cross-attention: price stream and depth stream
        self.fusion_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=0.1
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # 轨迹预测头 / Trajectory prediction head
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, embed_dim),
        )

        # 价格方向预测 / Price direction prediction
        self.direction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 3),  # 上涨/下跌/平稳 / up/down/flat
        )

        # 深度一致性预测 / Depth consistency prediction
        self.depth_pred_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, depth_levels * 4),
        )

    def forward(
        self,
        price_seq: torch.Tensor,
        depth_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 / Forward pass

        Args:
            price_seq: [B, T, price_features] 价格序列
            depth_seq: [B, T, depth_levels, 4] 订单簿深度序列

        Returns:
            包含嵌入、预测和损失项的字典
        """
        B, T, _ = price_seq.shape

        # 编码价格序列 / Encode price sequence
        price_emb = self.price_encoder(price_seq)  # [B, T, hidden]

        # 编码深度序列 / Encode depth sequence
        # 展平深度维度 / Flatten depth dimensions
        depth_flat = depth_seq.reshape(B, T, -1)  # [B, T, depth_levels * 4]
        depth_emb = self.depth_encoder(depth_flat)  # [B, T, hidden]

        # 深度感知卷积 / Depth-aware convolution
        # 价格特征通过深度感知卷积 / Price features through depth-aware conv
        price_conv = price_emb.permute(0, 2, 1)  # [B, hidden, T]
        for conv_layer in self.depth_aware_conv:
            price_conv = conv_layer(price_conv)
        price_conv = price_conv.permute(0, 2, 1)  # [B, T, hidden]

        # 双流融合（价格流 × 深度流）
        # Dual-stream fusion (price stream × depth stream)
        fused, _ = self.fusion_attn(
            query=price_conv, key=depth_emb, value=depth_emb
        )
        fused = self.fusion_norm(price_conv + fused)  # [B, T, hidden]

        # 时序池化 / Temporal pooling
        pooled = fused.mean(dim=1)  # [B, hidden]

        # 轨迹嵌入 / Trajectory embedding
        trajectory_emb = self.trajectory_head(pooled)  # [B, embed_dim]

        # 预测 / Predictions
        direction_pred = self.direction_head(trajectory_emb)  # [B, 3]
        depth_pred = self.depth_pred_head(trajectory_emb)  # [B, depth_levels * 4]
        depth_pred = depth_pred.reshape(B, self.depth_levels, 4)

        return {
            "trajectory_embedding": trajectory_emb,
            "direction_pred": direction_pred,
            "depth_pred": depth_pred,
            "fused_features": fused,
        }

    def compute_depth_consistency_loss(
        self,
        depth_pred: torch.Tensor,
        depth_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算深度一致性损失 / Compute depth consistency loss

        确保预测的订单簿深度与目标一致，包括价格排序约束和数量非负约束。
        Ensures predicted order book depth matches target, including price
        ordering constraints and non-negative size constraints.

        Args:
            depth_pred: [B, depth_levels, 4] 预测深度
            depth_target: [B, depth_levels, 4] 目标深度

        Returns:
            深度一致性损失
        """
        # L1重建损失 / L1 reconstruction loss
        recon_loss = F.l1_loss(depth_pred, depth_target)

        # 价格排序约束（bid递减，ask递增）
        # Price ordering constraints (bids descending, asks ascending)
        pred_bids = depth_pred[:, :, 0]  # [B, depth_levels]
        pred_asks = depth_pred[:, :, 2]  # [B, depth_levels]

        # bid价格应该递减 / Bid prices should be descending
        bid_order_loss = F.relu(pred_bids[:, 1:] - pred_bids[:, :-1]).mean()
        # ask价格应该递增 / Ask prices should be ascending
        ask_order_loss = F.relu(pred_asks[:, :-1] - pred_asks[:, 1:]).mean()

        # 数量非负约束 / Non-negative size constraint
        pred_bid_sizes = depth_pred[:, :, 1]
        pred_ask_sizes = depth_pred[:, :, 3]
        size_nonneg_loss = F.relu(-pred_bid_sizes).mean() + F.relu(-pred_ask_sizes).mean()

        total = recon_loss + 0.5 * (bid_order_loss + ask_order_loss) + 0.1 * size_nonneg_loss
        return total


# ============================================================================
# 深度感知市场生成器 / Depth-Aware Market Generator
# ============================================================================


class DepthAwareMarketGenerator(nn.Module):
    """
    深度感知市场情景生成器 / Depth-aware market scenario generator

    类比DepthDirector的视频生成：生成具有深度感知（订单簿结构）的市场情景。
    Analogous to DepthDirector's video generation: generates market scenarios
    with depth awareness (order book structure).

    Args:
        embed_dim: 嵌入维度 / Embedding dimension
        n_depth_levels: 深度层数 / Number of depth levels
        seq_length: 生成长度 / Generation sequence length
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_depth_levels: int = 20,
        seq_length: int = 50,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_depth_levels = n_depth_levels
        self.seq_length = seq_length

        # 条件编码器 / Conditioning encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * 4),
        )

        # 自回归生成器 / Autoregressive generator
        self.gen_rnn = nn.GRU(
            input_size=embed_dim + n_depth_levels * 4,
            hidden_size=embed_dim * 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # 价格输出头 / Price output head
        self.price_head = nn.Linear(embed_dim * 2, 5)  # OHLCV

        # 深度输出头 / Depth output head
        self.depth_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_depth_levels * 4),
        )

    def forward(
        self,
        condition: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        生成市场情景 / Generate market scenario

        Args:
            condition: [B, embed_dim] 条件向量
            n_steps: 生成步数（默认使用self.seq_length）

        Returns:
            生成的价格和深度序列
        """
        B = condition.shape[0]
        steps = n_steps or self.seq_length

        # 编码条件 / Encode condition
        cond = self.condition_encoder(condition)
        cond = cond.reshape(B, 1, -1)  # [B, 1, embed_dim*4]

        # 初始输入 / Initial input
        gen_input = cond[:, :, :self.embed_dim]
        depth_init = torch.zeros(B, 1, self.n_depth_levels * 4, device=condition.device)
        gen_input = torch.cat([gen_input, depth_init], dim=-1)

        # 自回归生成 / Autoregressive generation
        all_prices = []
        all_depths = []
        hidden = None

        for t in range(steps):
            output, hidden = self.gen_rnn(gen_input, hidden)

            # 价格预测 / Price prediction
            price = self.price_head(output)
            all_prices.append(price)

            # 深度预测 / Depth prediction
            depth = self.depth_head(output)
            depth = depth.reshape(B, 1, self.n_depth_levels, 4)
            all_depths.append(depth)

            # 下一步输入 / Next step input
            if t < steps - 1:
                gen_input = torch.cat([output, depth.reshape(B, 1, -1)], dim=-1)

        prices = torch.cat(all_prices, dim=1)  # [B, steps, 5]
        depths = torch.cat(all_depths, dim=1)  # [B, steps, depth_levels, 4]

        return {"prices": prices, "depths": depths}


# ============================================================================
# 相机控制类比 / Camera Control Analogy
# ============================================================================


class CameraControlAnalogy(nn.Module):
    """
    相机控制类比模块 / Camera control analogy module

    在DepthDirector中，"相机控制"允许从不同视角观察场景。
    在量化应用中，"视角控制"允许从不同分析维度观察市场。

    In DepthDirector, "camera control" allows observing scenes from different
    perspectives. In the quant application, "perspective control" allows
    observing the market from different analytical dimensions.

    Args:
        embed_dim: 嵌入维度 / Embedding dimension
        n_perspectives: 分析视角数量 / Number of analytical perspectives
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_perspectives: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_perspectives = n_perspectives

        # 视角编码器 / Perspective encoder
        self.perspective_encoder = nn.Embedding(n_perspectives, embed_dim)

        # 视角条件化注意力 / Perspective-conditioned attention
        self.perspective_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True
        )

        # 视角投影 / Perspective projection
        # 每个视角关注市场数据的不同方面
        # Each perspective focuses on different aspects of market data
        self.perspective_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
            )
            for _ in range(n_perspectives)
        ])

        # 视角融合 / Perspective fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * n_perspectives, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        market_features: torch.Tensor,
        perspective_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        应用视角控制 / Apply perspective control

        Args:
            market_features: [B, T, embed_dim] 市场特征序列
            perspective_ids: [B] 视角ID（可选，默认使用所有视角）

        Returns:
            视角条件化的特征
        """
        B, T, D = market_features.shape

        if perspective_ids is None:
            # 使用所有视角并融合 / Use all perspectives and fuse
            perspective_outputs = []
            for p_id in range(self.n_perspectives):
                p_emb = self.perspective_encoder(
                    torch.tensor(p_id, device=market_features.device).unsqueeze(0)
                )  # [1, embed_dim]
                p_emb = p_emb.unsqueeze(0).expand(B, T, -1)  # [B, T, embed_dim]

                # 视角条件化注意力 / Perspective-conditioned attention
                p_out, _ = self.perspective_attn(
                    query=market_features, key=p_emb, value=market_features
                )
                p_out = self.perspective_projections[p_id](p_out)
                perspective_outputs.append(p_out.mean(dim=1))  # [B, embed_dim]

            # 拼接所有视角 / Concatenate all perspectives
            all_perspectives = torch.cat(perspective_outputs, dim=-1)
            fused = self.fusion(all_perspectives)  # [B, embed_dim]

            return {
                "fused": fused,
                "per_perspective": perspective_outputs,
            }
        else:
            # 使用指定视角 / Use specified perspective
            p_emb = self.perspective_encoder(perspective_ids)  # [B, embed_dim]
            p_emb = p_emb.unsqueeze(1).expand(-1, T, -1)

            p_out, _ = self.perspective_attn(
                query=market_features, key=p_emb, value=market_features
            )

            # 使用对应视角的投影 / Use corresponding perspective projection
            outputs = []
            for i in range(B):
                p_id = perspective_ids[i].item()
                outputs.append(
                    self.perspective_projections[p_id](p_out[i : i + 1])
                )
            p_out = torch.cat(outputs, dim=0)

            return {
                "fused": p_out.mean(dim=1),
                "per_perspective": [p_out.mean(dim=1)],
            }


# ============================================================================
# 合成市场数据生成 / Synthetic Market Data Generation
# ============================================================================


def generate_synthetic_market_data(
    n_samples: int = 2000,
    seq_length: int = 50,
    price_features: int = 64,
    depth_levels: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成合成市场训练数据 / Generate synthetic market training data

    创建包含价格序列和对应订单簿深度的合成数据。
    Creates synthetic data with price sequences and corresponding order book depth.

    Args:
        n_samples: 样本数量
        seq_length: 序列长度
        price_features: 价格特征维度
        depth_levels: 订单簿深度层数

    Returns:
        (price_sequences, depth_sequences, labels) 三元组
    """
    logger.info(
        "Generating %d synthetic market samples / 生成 %d 个合成市场样本",
        n_samples,
        n_samples,
    )

    price_seqs = []
    depth_seqs = []
    labels = []

    for i in range(n_samples):
        # 随机选择市场状态 / Randomly select market regime
        regime = np.random.choice(["trending", "volatile", "calm"], p=[0.3, 0.3, 0.4])

        # 生成价格序列 / Generate price sequence
        if regime == "trending":
            drift = np.random.choice([-1, 1]) * np.random.uniform(0.001, 0.005)
            vol = np.random.uniform(0.005, 0.015)
            label = 0 if drift > 0 else 1  # 上涨/下跌
        elif regime == "volatile":
            drift = 0.0
            vol = np.random.uniform(0.02, 0.05)
            label = 2  # 高波动
        else:
            drift = 0.0
            vol = np.random.uniform(0.001, 0.005)
            label = 3  # 平静

        returns = np.random.randn(seq_length) * vol + drift
        prices = 100.0 + np.cumsum(returns)

        # 构建价格特征 / Build price features
        features = np.zeros((seq_length, price_features))
        features[:, 0] = prices
        features[:, 1] = returns
        # 移动平均 / Moving averages
        for w in [5, 10, 20]:
            ma = np.convolve(prices, np.ones(w) / w, mode="same")
            features[:, 2 + w // 5] = ma
        # 波动率 / Volatility
        features[:, 6] = np.convolve(returns ** 2, np.ones(10) / 10, mode="same")
        # 剩余特征用随机填充 / Fill remaining with random
        for f in range(7, price_features):
            features[:, f] = np.random.randn(seq_length) * 0.1

        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        # 生成订单簿深度 / Generate order book depth
        depth = np.zeros((seq_length, depth_levels, 4))
        for t in range(seq_length):
            mid = prices[t]
            spread = vol * mid * np.random.uniform(0.5, 2.0)

            # bid价格（递减）/ Bid prices (descending)
            depth[t, :, 0] = mid - spread / 2 - np.arange(depth_levels) * spread * 0.5
            # bid数量（随机）/ Bid sizes
            depth[t, :, 1] = np.random.exponential(100, depth_levels) * (1 + np.random.randn() * 0.1)
            # ask价格（递增）/ Ask prices (ascending)
            depth[t, :, 2] = mid + spread / 2 + np.arange(depth_levels) * spread * 0.5
            # ask数量（随机）/ Ask sizes
            depth[t, :, 3] = np.random.exponential(100, depth_levels) * (1 + np.random.randn() * 0.1)

            # 在高波动市场中增加深度不均匀性
            # Increase depth imbalance in volatile markets
            if regime == "volatile":
                imbalance = np.random.randn() * 0.3
                depth[t, :, 1] *= (1 + imbalance)
                depth[t, :, 3] *= (1 - imbalance)

        price_seqs.append(torch.tensor(features, dtype=torch.float32))
        depth_seqs.append(torch.tensor(depth, dtype=torch.float32))
        labels.append(label)

    return (
        torch.stack(price_seqs),
        torch.stack(depth_seqs),
        torch.tensor(labels, dtype=torch.long),
    )


# ============================================================================
# 训练函数 / Training Function
# ============================================================================


def train_depth_market_model(config: TrainingConfig) -> Dict[str, Any]:
    """
    训练深度感知市场模型 / Train depth-aware market model

    完整训练流程：
    1. 准备合成/真实市场数据 / Prepare synthetic/real market data
    2. 训练价格轨迹深度模型 / Train price trajectory depth model
    3. 训练市场情景生成器 / Train market scenario generator
    4. 训练视角控制模块 / Train perspective control module
    5. 保存和评估 / Save and evaluate

    Args:
        config: 训练配置

    Returns:
        训练结果
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    logger.info("Device / 设备: %s", device)

    # === 1. 准备数据 / Prepare data ===
    logger.info("Preparing data / 准备数据...")
    price_seqs, depth_seqs, labels = generate_synthetic_market_data(
        n_samples=2000,
        seq_length=config.trajectory_length,
        price_features=config.price_features,
        depth_levels=config.depth_levels,
    )

    # 划分训练/验证集 / Split train/validation
    n_train = int(0.8 * len(price_seqs))
    indices = torch.randperm(len(price_seqs))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_prices = price_seqs[train_idx].to(device)
    train_depths = depth_seqs[train_idx].to(device)
    train_labels = labels[train_idx].to(device)
    val_prices = price_seqs[val_idx].to(device)
    val_depths = depth_seqs[val_idx].to(device)
    val_labels = labels[val_idx].to(device)

    logger.info(
        "Train: %d samples, Val: %d samples / 训练: %d 样本, 验证: %d 样本",
        n_train,
        len(val_idx),
        n_train,
        len(val_idx),
    )

    # === 2. 训练价格轨迹深度模型 / Train price trajectory depth model ===
    logger.info("=" * 60)
    logger.info("Training Price Trajectory Depth Model / 训练价格轨迹深度模型")
    logger.info("=" * 60)

    depth_model = PriceTrajectoryDepthModel(
        price_features=config.price_features,
        depth_levels=config.depth_levels,
        hidden_dim=config.hidden_dim,
        embed_dim=config.embed_dim,
        n_heads=config.n_heads,
    ).to(device)

    optimizer = optim.AdamW(
        depth_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    n_train_samples = train_prices.shape[0]
    best_val_acc = 0.0

    for epoch in range(config.epochs):
        depth_model.train()
        epoch_loss = 0.0
        n_batches = 0

        # 小批量训练 / Mini-batch training
        perm = torch.randperm(n_train_samples)
        for start in range(0, n_train_samples, config.batch_size):
            end = min(start + config.batch_size, n_train_samples)
            batch_idx = perm[start:end]

            batch_prices = train_prices[batch_idx]
            batch_depths = train_depths[batch_idx]
            batch_labels = train_labels[batch_idx]

            optimizer.zero_grad()
            output = depth_model(batch_prices, batch_depths)

            # 分类损失 / Classification loss
            cls_loss = F.cross_entropy(output["direction_pred"], batch_labels)

            # 深度一致性损失 / Depth consistency loss
            # 使用最后一个时间步的深度作为目标
            # Use last timestep's depth as target
            last_depth = batch_depths[:, -1, :, :]
            depth_loss = depth_model.compute_depth_consistency_loss(
                output["depth_pred"], last_depth
            )

            total_loss = cls_loss + config.depth_consistency_weight * depth_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(depth_model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % config.log_interval == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                "Epoch %d/%d | Loss: %.4f / 第%d轮 | 损失: %.4f",
                epoch + 1,
                config.epochs,
                avg_loss,
                epoch + 1,
                avg_loss,
            )

        # 验证 / Validation
        if (epoch + 1) % (config.log_interval * 2) == 0:
            depth_model.eval()
            with torch.no_grad():
                val_output = depth_model(val_prices, val_depths)
                val_preds = val_output["direction_pred"].argmax(dim=1)
                val_acc = float((val_preds == val_labels).float().mean().item())
                logger.info("Val accuracy / 验证准确率: %.4f (best: %.4f)", val_acc, best_val_acc)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

    # === 3. 训练市场情景生成器 / Train market scenario generator ===
    logger.info("=" * 60)
    logger.info("Training Depth-Aware Market Generator / 训练深度感知市场生成器")
    logger.info("=" * 60)

    generator = DepthAwareMarketGenerator(
        embed_dim=config.embed_dim,
        n_depth_levels=config.depth_levels,
        seq_length=config.trajectory_length,
    ).to(device)

    gen_optimizer = optim.AdamW(generator.parameters(), lr=1e-3)

    # 使用深度模型的嵌入训练生成器
    # Train generator using depth model's embeddings
    depth_model.eval()
    with torch.no_grad():
        train_output = depth_model(train_prices, train_depths)
        train_embeddings = train_output["trajectory_embedding"]

    for epoch in range(50):
        generator.train()
        perm = torch.randperm(n_train_samples)
        gen_loss_total = 0.0
        n_gen_batches = 0

        for start in range(0, n_train_samples, config.batch_size):
            end = min(start + config.batch_size, n_train_samples)
            batch_idx = perm[start:end]
            batch_emb = train_embeddings[batch_idx]

            gen_optimizer.zero_grad()
            gen_output = generator(batch_emb)

            # 重建损失 / Reconstruction loss
            target_prices = train_prices[batch_idx]
            target_depths = train_depths[batch_idx]

            # 价格重建 / Price reconstruction
            price_recon = F.mse_loss(
                gen_output["prices"][:, :target_prices.shape[1], :1],
                target_prices[:, :, :1],
            )
            # 深度重建 / Depth reconstruction
            depth_recon = F.l1_loss(
                gen_output["depths"][:, :target_depths.shape[1]],
                target_depths,
            )

            gen_loss = price_recon + 0.5 * depth_recon
            gen_loss.backward()
            gen_optimizer.step()
            gen_loss_total += gen_loss.item()
            n_gen_batches += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                "Generator epoch %d: loss=%.4f / 生成器第%d轮: 损失=%.4f",
                epoch + 1,
                gen_loss_total / max(n_gen_batches, 1),
                epoch + 1,
                gen_loss_total / max(n_gen_batches, 1),
            )

    # === 4. 训练视角控制 / Train perspective control ===
    logger.info("=" * 60)
    logger.info("Training Perspective Control / 训练视角控制")
    logger.info("=" * 60)

    perspective_ctrl = CameraControlAnalogy(
        embed_dim=config.embed_dim,
        n_perspectives=config.n_perspectives,
    ).to(device)

    # 简单训练：学习不同视角的区分能力
    # Simple training: learn to distinguish perspectives
    persp_optimizer = optim.AdamW(perspective_ctrl.parameters(), lr=1e-3)

    for epoch in range(30):
        perspective_ctrl.train()
        perm = torch.randperm(n_train_samples)
        persp_loss_total = 0.0

        for start in range(0, n_train_samples, config.batch_size):
            end = min(start + config.batch_size, n_train_samples)
            batch_idx = perm[start:end]
            batch_emb = train_embeddings[batch_idx].unsqueeze(1)  # [B, 1, D]

            persp_optimizer.zero_grad()
            persp_output = perspective_ctrl(batch_emb)
            # 简单对比损失 / Simple contrastive loss
            fused = persp_output["fused"]
            loss = F.mse_loss(fused, batch_emb.squeeze(1))
            loss.backward()
            persp_optimizer.step()
            persp_loss_total += loss.item()

        if (epoch + 1) % 10 == 0:
            logger.info(
                "Perspective epoch %d: loss=%.4f / 视角第%d轮: 损失=%.4f",
                epoch + 1,
                persp_loss_total,
                epoch + 1,
                persp_loss_total,
            )

    # === 5. 保存模型 / Save models ===
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"model_state_dict": depth_model.state_dict(), "best_val_acc": best_val_acc},
        output_dir / "price_depth_model.pt",
    )
    torch.save(generator.state_dict(), output_dir / "market_generator.pt")
    torch.save(perspective_ctrl.state_dict(), output_dir / "perspective_control.pt")

    logger.info(
        "All models saved to / 所有模型保存到: %s",
        output_dir,
    )

    return {
        "depth_model": depth_model,
        "generator": generator,
        "perspective_control": perspective_ctrl,
        "best_val_acc": best_val_acc,
    }


# ============================================================================
# 命令行接口 / CLI
# ============================================================================


def parse_args() -> TrainingConfig:
    """解析命令行参数 / Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="DepthDirector Quant Training / 深度导演量化训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="data/market")
    parser.add_argument("--output-dir", type=str, default="results/depth_quant")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--price-features", type=int, default=64)
    parser.add_argument("--depth-levels", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-perspectives", type=int, default=4)
    parser.add_argument("--trajectory-length", type=int, default=50)
    parser.add_argument("--depth-consistency-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-interval", type=int, default=10)

    args = parser.parse_args()

    return TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        price_features=args.price_features,
        depth_levels=args.depth_levels,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_perspectives=args.n_perspectives,
        trajectory_length=args.trajectory_length,
        depth_consistency_weight=args.depth_consistency_weight,
        seed=args.seed,
        device=args.device,
        log_interval=args.log_interval,
    )


# ============================================================================
# 入口 / Entry Point
# ============================================================================


if __name__ == "__main__":
    config = parse_args()
    logger.info("Starting DepthDirector quant training / 开始深度导演量化训练")
    results = train_depth_market_model(config)
    logger.info("Training complete / 训练完成")
