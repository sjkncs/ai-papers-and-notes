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
Market Reasoning Geometry Training / 市场推理几何训练

Applies the "Geometry of Thought" framework to quantitative finance:
- Analyze how model scale qualitatively changes market pattern recognition
- Detect regime transitions using geometric phase transitions
- Identify which trading signals emerge at which model scales

将《The Geometry of Thought》框架应用于量化金融：
- 分析模型规模如何定性地改变市场模式识别
- 利用几何相变检测市场状态转换
- 识别哪些交易信号在哪个模型规模涌现

Usage / 用法:
    python quant_train.py --data-dir data/ --epochs 100 --output-dir results/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
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
# 枚举和数据类 / Enums and Data Classes
# ============================================================================


class MarketRegime(Enum):
    """
    市场状态枚举 / Market regime enumeration

    市场可能处于的不同定性状态，对应论文中的"推理领域"。
    Different qualitative states the market may be in, analogous to
    "reasoning domains" in the paper.
    """

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class ModelScale(Enum):
    """
    模型规模枚举 / Model scale enumeration

    不同复杂度的模型配置，对应论文中的"规模"。
    Different model complexity configurations, analogous to "scale" in the paper.
    """

    SMALL = "small"       # <10M参数 / <10M parameters
    MEDIUM = "medium"     # 10M-100M参数 / 10M-100M parameters
    LARGE = "large"       # >100M参数 / >100M parameters


@dataclass
class TrainingConfig:
    """
    训练配置 / Training configuration

    包含所有训练超参数和路径设置。
    Contains all training hyperparameters and path settings.
    """

    data_dir: str = "data/market"
    output_dir: str = "results/regime_detector"
    # 训练参数 / Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    # 模型参数 / Model parameters
    input_features: int = 128
    hidden_dim: int = 256
    embed_dim: int = 128
    n_heads: int = 8
    n_layers: int = 4
    # 几何分析参数 / Geometry analysis parameters
    regime_types: List[str] = field(
        default_factory=lambda: ["trending", "mean_reverting", "volatile"]
    )
    model_scales: List[str] = field(
        default_factory=lambda: ["small", "medium", "large"]
    )
    # 其他 / Other
    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    log_interval: int = 10
    save_interval: int = 10


# ============================================================================
# 市场数据集 / Market Dataset
# ============================================================================


class MarketSequenceDataset(Dataset):
    """
    市场序列数据集 / Market sequence dataset

    加载和预处理时间序列市场数据，标注市场状态。
    支持合成数据生成用于测试。

    Loads and preprocesses time-series market data with regime labels.
    Supports synthetic data generation for testing.

    Args:
        data_dir: 数据目录 / Data directory
        regime_types: 要包含的市场状态 / Regime types to include
        sequence_length: 序列长度 / Sequence length
        n_features: 特征数量 / Number of features
    """

    def __init__(
        self,
        data_dir: str = "data/market",
        regime_types: Optional[List[str]] = None,
        sequence_length: int = 64,
        n_features: int = 128,
        synthetic: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.regime_types = regime_types or [r.value for r in MarketRegime]
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.synthetic = synthetic

        self.sequences: List[torch.Tensor] = []
        self.regime_labels: List[int] = []
        self.metadata: List[Dict[str, Any]] = []

        self._load_or_generate()

    def _load_or_generate(self) -> None:
        """
        加载真实数据或生成合成数据 / Load real data or generate synthetic

        如果数据目录存在，尝试加载CSV/Parquet文件；
        否则生成模拟不同市场状态的合成数据。

        If data directory exists, try loading CSV/Parquet files;
        otherwise generate synthetic data simulating different market regimes.
        """
        if not self.synthetic and self.data_dir.exists():
            self._load_real_data()
        else:
            logger.info(
                "Generating synthetic market data / 生成合成市场数据"
            )
            self._generate_synthetic_data()

    def _load_real_data(self) -> None:
        """
        从CSV/Parquet加载真实市场数据 / Load real market data from CSV/Parquet

        预期文件格式：
        - 时间序列价格数据（OHLCV）
        - 预计算的技术指标特征
        - 市场状态标签列
        """
        try:
            import pandas as pd

            for f in sorted(self.data_dir.glob("*.csv")):
                df = pd.read_csv(f)
                logger.info("Loaded %s: %d rows / 加载 %s: %d 行", f.name, len(df), f.name, len(df))
                # 滑动窗口切分 / Sliding window segmentation
                features = df.drop(columns=["regime", "date"], errors="ignore").values
                labels = df["regime"].values if "regime" in df.columns else np.zeros(len(df))
                for i in range(len(features) - self.sequence_length):
                    seq = torch.tensor(
                        features[i : i + self.sequence_length], dtype=torch.float32
                    )
                    # 使用窗口中最后一个标签作为整段的状态 / Use last label in window
                    label = int(labels[i + self.sequence_length - 1])
                    self.sequences.append(seq)
                    self.regime_labels.append(label)
        except ImportError:
            logger.warning("pandas not available, falling back to synthetic / pandas不可用，回退到合成数据")
            self._generate_synthetic_data()

    def _generate_synthetic_data(
        self, n_samples_per_regime: int = 500
    ) -> None:
        """
        生成模拟不同市场状态的合成数据
        Generate synthetic data simulating different market regimes

        每种市场状态具有独特的统计特性：
        - 趋势上涨：正漂移 + 低波动
        - 趋势下跌：负漂移 + 中等波动
        - 均值回归：零漂移 + 强自相关
        - 高波动：大标准差 + 肥尾
        - 危机：极端负偏斜 + 高相关性

        Each regime has unique statistical properties:
        - Trending up: positive drift + low volatility
        - Trending down: negative drift + medium volatility
        - Mean reverting: zero drift + strong autocorrelation
        - High volatility: large std + fat tails
        - Crisis: extreme negative skew + high correlation
        """
        regime_configs = {
            0: {"drift": 0.001, "vol": 0.01, "name": "trending_up"},
            1: {"drift": -0.001, "vol": 0.015, "name": "trending_down"},
            2: {"drift": 0.0, "vol": 0.008, "name": "mean_reverting", "ar_coeff": 0.9},
            3: {"drift": 0.0, "vol": 0.04, "name": "high_volatility"},
            4: {"drift": -0.003, "vol": 0.05, "name": "crisis"},
            5: {"drift": 0.002, "vol": 0.02, "name": "recovery"},
        }

        for regime_id, config in regime_configs.items():
            for _ in range(n_samples_per_regime):
                seq = self._generate_regime_sequence(config)
                self.sequences.append(seq)
                self.regime_labels.append(regime_id)

        logger.info(
            "Generated %d synthetic sequences across %d regimes / "
            "生成了 %d 个合成序列，覆盖 %d 个状态",
            len(self.sequences),
            len(regime_configs),
            len(self.sequences),
            len(regime_configs),
        )

    def _generate_regime_sequence(self, config: Dict[str, Any]) -> torch.Tensor:
        """
        根据配置生成单条市场序列 / Generate a single market sequence from config

        Args:
            config: 市场状态配置（drift, vol, etc.）

        Returns:
            [sequence_length, n_features] 特征张量
        """
        T = self.sequence_length
        F = self.n_features
        drift = config.get("drift", 0.0)
        vol = config.get("vol", 0.01)
        ar_coeff = config.get("ar_coeff", 0.0)

        # 生成基础价格路径 / Generate base price path
        if ar_coeff > 0:
            # 均值回归路径（AR过程）/ Mean-reverting path (AR process)
            returns = np.zeros(T)
            for t in range(1, T):
                returns[t] = ar_coeff * returns[t - 1] + np.random.randn() * vol
        elif config.get("name") == "crisis":
            # 危机：肥尾分布 / Crisis: fat-tailed distribution
            returns = np.random.standard_t(df=3, size=T) * vol + drift
        else:
            # 普通布朗运动 / Standard Brownian motion
            returns = np.random.randn(T) * vol + drift

        # 构建多特征表示 / Build multi-feature representation
        features = np.zeros((T, F))

        # 前几个特征：价格和衍生指标 / First features: price and derived metrics
        prices = np.cumsum(returns) + 100.0
        features[:, 0] = prices  # 价格 / Price
        features[:, 1] = returns  # 收益率 / Returns
        features[:, 2] = np.convolve(returns, np.ones(5) / 5, mode="same")  # 5日MA
        features[:, 3] = np.convolve(returns ** 2, np.ones(10) / 10, mode="same")  # 波动率

        # 技术指标特征 / Technical indicator features
        # RSI-like feature
        pos_returns = np.maximum(returns, 0)
        neg_returns = np.maximum(-returns, 0)
        avg_gain = np.convolve(pos_returns, np.ones(14) / 14, mode="same")
        avg_loss = np.convolve(neg_returns, np.ones(14) / 14, mode="same") + 1e-10
        features[:, 4] = avg_gain / (avg_gain + avg_loss)

        # MACD-like feature
        ema_short = np.convolve(prices, np.exp(-np.arange(12) / 12), mode="same")
        ema_long = np.convolve(prices, np.exp(-np.arange(26) / 26), mode="same")
        features[:, 5] = ema_short - ema_long

        # 填充剩余特征（模拟多资产/多频率信号）
        # Fill remaining features (simulating multi-asset/multi-frequency signals)
        for f_idx in range(6, F):
            freq = 0.1 + 0.05 * (f_idx % 10)
            phase = np.random.uniform(0, 2 * np.pi)
            features[:, f_idx] = (
                np.sin(2 * np.pi * freq * np.arange(T) + phase) * vol * 10
                + np.random.randn(T) * vol * 0.5
            )

        # 标准化 / Standardize
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        return torch.tensor(features, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.sequences[idx], self.regime_labels[idx]


# ============================================================================
# 市场推理几何模型 / Market Reasoning Geometry Model
# ============================================================================


class MarketReasoningGeometry(nn.Module):
    """
    市场推理几何模型 / Market reasoning geometry model

    核心模型：分析不同规模模型如何理解市场模式。
    通过将市场数据嵌入几何空间，检测不同规模下的表征差异。

    Core model: analyzes how different-scale models understand market patterns.
    Embeds market data into geometric space and detects representation differences
    across scales.

    Args:
        input_features: 输入特征维度 / Input feature dimension
        hidden_dim: 隐藏层维度 / Hidden dimension
        embed_dim: 几何嵌入维度 / Geometric embedding dimension
        n_regimes: 市场状态数量 / Number of market regimes
        n_heads: 注意力头数 / Number of attention heads
        n_layers: Transformer层数 / Number of Transformer layers
        scale: 模型规模标识 / Model scale identifier
    """

    def __init__(
        self,
        input_features: int = 128,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        n_regimes: int = 6,
        n_heads: int = 8,
        n_layers: int = 4,
        scale: str = "medium",
    ) -> None:
        super().__init__()
        self.scale = scale
        self.embed_dim = embed_dim

        # 根据规模调整维度 / Scale-dependent dimension adjustment
        scale_factor = {"small": 0.5, "medium": 1.0, "large": 2.0}.get(scale, 1.0)
        actual_hidden = int(hidden_dim * scale_factor)
        actual_embed = int(embed_dim * scale_factor)
        actual_heads = max(1, int(n_heads * scale_factor))
        actual_layers = max(1, int(n_layers * scale_factor))

        # 输入投影 / Input projection
        self.input_proj = nn.Linear(input_features, actual_hidden)

        # 时序位置编码 / Temporal positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(256, actual_hidden) * 0.02
        )

        # Transformer编码器 / Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=actual_hidden,
            nhead=actual_heads,
            dim_feedforward=actual_hidden * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=actual_layers
        )

        # 几何嵌入头 / Geometric embedding head
        self.embed_head = nn.Sequential(
            nn.Linear(actual_hidden, actual_embed),
            nn.GELU(),
            nn.LayerNorm(actual_embed),
        )

        # 状态分类头 / Regime classification head
        self.classifier = nn.Sequential(
            nn.Linear(actual_embed, actual_embed // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(actual_embed // 2, n_regimes),
        )

        # 内在维度估计辅助网络 / Intrinsic dimension estimation auxiliary network
        self.id_estimator = nn.Sequential(
            nn.Linear(actual_embed, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        self.actual_embed_dim = actual_embed

    def forward(
        self, x: torch.Tensor, return_geometry: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 / Forward pass

        Args:
            x: [B, T, F] 市场序列
            return_geometry: 是否返回几何分析结果

        Returns:
            包含嵌入、分类logits和几何度量的字典
        """
        B, T, _ = x.shape

        # 输入投影 + 位置编码 / Input projection + positional encoding
        h = self.input_proj(x)
        h = h + self.pos_encoding[:T].unsqueeze(0)

        # Transformer编码 / Transformer encoding
        h = self.transformer(h)

        # 时序池化 / Temporal pooling
        h_pooled = h.mean(dim=1)  # [B, hidden]

        # 几何嵌入 / Geometric embedding
        embeddings = self.embed_head(h_pooled)  # [B, embed_dim]

        # 状态分类 / Regime classification
        logits = self.classifier(embeddings)  # [B, n_regimes]

        result: Dict[str, torch.Tensor] = {
            "embeddings": embeddings,
            "logits": logits,
            "hidden_states": h,
        }

        if return_geometry:
            # 内在维度估计 / Intrinsic dimension estimation
            id_est = self.id_estimator(embeddings)
            result["intrinsic_dim"] = id_est

            # 路径曲率（从隐藏状态计算）
            # Path curvature (computed from hidden states)
            if T >= 3:
                curvature = self._compute_path_curvature(h)
                result["curvature"] = curvature

        return result

    def _compute_path_curvature(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        计算推理路径曲率 / Compute reasoning path curvature

        使用连续三点的Menger曲率。
        Uses Menger curvature from consecutive triplets.

        Args:
            hidden_states: [B, T, D] 隐藏状态序列

        Returns:
            [B] 每样本平均曲率
        """
        B, T, D = hidden_states.shape
        if T < 3:
            return torch.zeros(B, device=hidden_states.device)

        # 差分 / Differences
        d1 = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
        d2 = hidden_states[:, 2:, :] - hidden_states[:, 1:-1, :]

        # 余弦距离（曲率的代理）/ Cosine distance (proxy for curvature)
        cos_sim = F.cosine_similarity(
            d1[:, :-1, :3], d2[:, :, :3], dim=-1
        )
        curvature = 1.0 - cos_sim  # 越偏离直线越弯曲

        return curvature.mean(dim=1)


# ============================================================================
# 状态转换检测器 / Regime Transition Detector
# ============================================================================


class RegimeTransitionDetector(nn.Module):
    """
    状态转换检测器 / Regime transition detector

    基于几何相变概念，检测市场状态的质变。
    当推理路径的几何结构发生急剧变化时，触发转换信号。

    Based on the geometric phase transition concept, detects qualitative shifts
    in market state. Triggers transition signals when reasoning path geometry
    changes abruptly.

    Args:
        embed_dim: 嵌入维度 / Embedding dimension
        window_size: 检测窗口大小 / Detection window size
    """

    def __init__(
        self,
        embed_dim: int = 128,
        window_size: int = 10,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size

        # 嵌入变化编码器 / Embedding change encoder
        self.change_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
        )

        # 时序卷积（检测突变）/ Temporal convolution (detect abrupt changes)
        self.temporal_conv = nn.Conv1d(
            embed_dim // 2, 64, kernel_size=5, padding=2
        )

        # 转换概率头 / Transition probability head
        self.transition_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 新状态预测头 / New regime prediction head
        self.regime_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 6),  # 6种市场状态 / 6 market regimes
        )

    def forward(
        self, embedding_sequence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        检测状态转换 / Detect regime transitions

        Args:
            embedding_sequence: [B, T, embed_dim] 时间序列嵌入

        Returns:
            转换概率和新状态预测
        """
        B, T, D = embedding_sequence.shape

        # 计算相邻嵌入的差异 / Compute differences between consecutive embeddings
        diffs = embedding_sequence[:, 1:, :] - embedding_sequence[:, :-1, :]

        # 构建变化对 / Build change pairs
        pairs = torch.cat(
            [embedding_sequence[:, :-1, :], diffs], dim=-1
        )

        # 编码变化 / Encode changes
        change_feat = self.change_encoder(pairs)  # [B, T-1, embed_dim//2]

        # 时序卷积 / Temporal convolution
        change_feat_t = change_feat.permute(0, 2, 1)  # [B, D, T-1]
        temporal_feat = F.gelu(self.temporal_conv(change_feat_t))
        temporal_feat = temporal_feat.permute(0, 2, 1)  # [B, T-1, 64]

        # 转换概率 / Transition probability
        transition_prob = self.transition_head(temporal_feat)  # [B, T-1, 1]

        # 新状态预测 / New regime prediction
        regime_pred = self.regime_predictor(temporal_feat)  # [B, T-1, n_regimes]

        return {
            "transition_prob": transition_prob.squeeze(-1),
            "regime_pred": regime_pred,
        }


# ============================================================================
# 规模依赖性信号分析器 / Scale-Dependent Signal Analyzer
# ============================================================================


class ScaleDependentSignalAnalyzer(nn.Module):
    """
    规模依赖性信号分析器 / Scale-dependent signal analyzer

    分析哪些市场信号在不同模型规模下涌现。
    对应论文中"相变"现象——某些能力仅在特定规模才出现。

    Analyzes which market signals emerge at different model scales.
    Corresponds to "phase transition" phenomenon in the paper — certain
    capabilities only emerge at specific scales.

    Args:
        embed_dim: 嵌入维度 / Embedding dimension
        n_scales: 模型规模数量 / Number of model scales
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_scales: int = 3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_scales = n_scales

        # 跨尺度注意力 / Cross-scale attention
        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True
        )

        # 涌现检测器 / Emergence detector
        # 检测信号在哪个规模首次出现
        # Detects at which scale a signal first appears
        self.emergence_detector = nn.Sequential(
            nn.Linear(embed_dim * n_scales, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_scales),
            nn.Sigmoid(),
        )

        # 信号质量评估器 / Signal quality evaluator
        self.signal_quality = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        scale_embeddings: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        分析规模依赖性信号 / Analyze scale-dependent signals

        Args:
            scale_embeddings: 每个规模的嵌入列表, 每个[B, embed_dim]

        Returns:
            涌现模式和信号质量分析
        """
        # 确保所有规模嵌入维度一致 / Ensure consistent embed dim across scales
        # 使用最小维度进行对齐 / Use minimum dim for alignment
        min_dim = min(e.shape[-1] for e in scale_embeddings)
        aligned = []
        for emb in scale_embeddings:
            if emb.shape[-1] > min_dim:
                emb = emb[..., :min_dim]
            elif emb.shape[-1] < min_dim:
                pad = torch.zeros(
                    *emb.shape[:-1], min_dim - emb.shape[-1],
                    device=emb.device,
                )
                emb = torch.cat([emb, pad], dim=-1)
            aligned.append(emb)

        # 拼接跨尺度嵌入 / Concatenate cross-scale embeddings
        combined = torch.cat(aligned, dim=-1)  # [B, embed_dim * n_scales]

        # 涌现检测 / Emergence detection
        emergence = self.emergence_detector(combined)  # [B, n_scales]

        # 信号质量（每个规模）/ Signal quality (per scale)
        quality_scores = []
        for emb in aligned:
            q = self.signal_quality(emb)
            quality_scores.append(q)
        quality = torch.cat(quality_scores, dim=-1)  # [B, n_scales]

        return {
            "emergence": emergence,
            "signal_quality": quality,
            "aligned_embeddings": aligned,
        }


# ============================================================================
# 训练函数 / Training Functions
# ============================================================================


def compute_geometry_loss(
    model_output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    regime_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    计算综合几何损失 / Compute comprehensive geometry loss

    结合分类损失、几何正则化和对比学习损失。
    Combines classification loss, geometric regularization, and contrastive loss.

    Args:
        model_output: 模型输出字典
        labels: 真实标签
        regime_weights: 状态权重（处理不平衡）

    Returns:
        各项损失的字典
    """
    logits = model_output["logits"]
    embeddings = model_output["embeddings"]

    # 分类损失 / Classification loss
    if regime_weights is not None:
        cls_loss = F.cross_entropy(logits, labels, weight=regime_weights)
    else:
        cls_loss = F.cross_entropy(logits, labels)

    # 几何正则化：鼓励同类状态形成紧凑簇
    # Geometric regularization: encourage compact clusters for same regime
    reg_loss = torch.tensor(0.0, device=logits.device)
    unique_labels = labels.unique()
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 1:
            cluster_emb = embeddings[mask]
            centroid = cluster_emb.mean(dim=0)
            reg_loss = reg_loss + F.mse_loss(cluster_emb, centroid.expand_as(cluster_emb))
    reg_loss = reg_loss / max(len(unique_labels), 1)

    # 对比损失：鼓励不同状态形成分离的簇
    # Contrastive loss: encourage separation between different regimes
    contrastive_loss = torch.tensor(0.0, device=logits.device)
    if len(unique_labels) >= 2:
        centroids = []
        for label in unique_labels:
            mask = labels == label
            centroids.append(embeddings[mask].mean(dim=0))
        centroids = torch.stack(centroids)
        # 最大化质心间距离（负距离损失）
        # Maximize inter-centroid distance (negative distance loss)
        dist_matrix = torch.cdist(centroids, centroids, p=2)
        dist_matrix.fill_diagonal_(float("inf"))
        contrastive_loss = -dist_matrix.min() * 0.1

    # 内在维度正则化 / Intrinsic dimension regularization
    id_loss = torch.tensor(0.0, device=logits.device)
    if "intrinsic_dim" in model_output:
        # 鼓励合理的内在维度（不要太低也不要太高）
        # Encourage reasonable intrinsic dimensionality
        target_id = torch.tensor(8.0, device=logits.device)
        id_loss = F.mse_loss(
            model_output["intrinsic_dim"].mean(), target_id
        ) * 0.01

    total_loss = cls_loss + 0.1 * reg_loss + contrastive_loss + id_loss

    return {
        "total": total_loss,
        "classification": cls_loss,
        "geometric_reg": reg_loss,
        "contrastive": contrastive_loss,
        "id_reg": id_loss,
    }


def train_regime_detector(
    config: TrainingConfig,
) -> Dict[str, Any]:
    """
    训练状态检测器（多尺度比较）
    Train regime detector with multi-scale comparison

    完整训练流程：
    1. 准备数据 / Prepare data
    2. 训练多个规模的模型 / Train models at multiple scales
    3. 分析规模间的几何差异 / Analyze geometric differences across scales
    4. 训练状态转换检测器 / Train regime transition detector
    5. 分析规模依赖性信号 / Analyze scale-dependent signals

    Args:
        config: 训练配置

    Returns:
        训练结果字典
    """
    # 设置随机种子 / Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 设备选择 / Device selection
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    logger.info("Using device / 使用设备: %s", device)

    # 准备数据 / Prepare data
    logger.info("Preparing data / 准备数据...")
    dataset = MarketSequenceDataset(
        data_dir=config.data_dir,
        regime_types=config.regime_types,
        n_features=config.input_features,
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 计算状态权重（处理不平衡）/ Compute regime weights
    label_counts = np.bincount(dataset.regime_labels, minlength=6)
    regime_weights = torch.tensor(
        1.0 / (label_counts + 1), dtype=torch.float32, device=device
    )
    regime_weights = regime_weights / regime_weights.sum() * len(regime_weights)

    results: Dict[str, Any] = {"scale_models": {}, "comparisons": []}

    # === 多尺度训练 / Multi-scale training ===
    for scale_name in config.model_scales:
        logger.info("=" * 60)
        logger.info(
            "Training %s scale model / 训练 %s 规模模型",
            scale_name,
            scale_name,
        )
        logger.info("=" * 60)

        model = MarketReasoningGeometry(
            input_features=config.input_features,
            hidden_dim=config.hidden_dim,
            embed_dim=config.embed_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            scale=scale_name,
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )

        best_val_acc = 0.0
        scale_embeddings: List[torch.Tensor] = []

        for epoch in range(config.epochs):
            model.train()
            epoch_losses: Dict[str, float] = {}

            for batch_idx, (sequences, labels) in enumerate(train_loader):
                sequences = sequences.to(device)
                labels = torch.tensor(labels, dtype=torch.long, device=device)

                optimizer.zero_grad()
                output = model(sequences, return_geometry=True)
                losses = compute_geometry_loss(output, labels, regime_weights)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            scheduler.step()

            # 日志 / Logging
            if (epoch + 1) % config.log_interval == 0:
                n_batches = len(train_loader)
                logger.info(
                    "Epoch %d/%d [%s] | Loss: %.4f | Cls: %.4f | GeoReg: %.4f",
                    epoch + 1,
                    config.epochs,
                    scale_name,
                    epoch_losses.get("total", 0) / n_batches,
                    epoch_losses.get("classification", 0) / n_batches,
                    epoch_losses.get("geometric_reg", 0) / n_batches,
                )

            # 验证 / Validation
            if (epoch + 1) % (config.log_interval * 2) == 0:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for sequences, labels in val_loader:
                        sequences = sequences.to(device)
                        labels = torch.tensor(labels, dtype=torch.long, device=device)
                        output = model(sequences, return_geometry=False)
                        preds = output["logits"].argmax(dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                val_acc = correct / max(total, 1)
                logger.info(
                    "Validation accuracy / 验证准确率: %.4f (best: %.4f)",
                    val_acc,
                    best_val_acc,
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

        # 收集最终嵌入用于跨尺度比较
        # Collect final embeddings for cross-scale comparison
        model.eval()
        all_embs = []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                output = model(sequences, return_geometry=False)
                all_embs.append(output["embeddings"].cpu())
        if all_embs:
            scale_embeddings = torch.cat(all_embs, dim=0)
        else:
            scale_embeddings = torch.randn(100, config.embed_dim)

        results["scale_models"][scale_name] = {
            "model": model,
            "best_val_acc": best_val_acc,
            "embeddings": scale_embeddings,
        }

        logger.info(
            "Scale %s training complete. Best val acc: %.4f / "
            "规模 %s 训练完成。最佳验证准确率: %.4f",
            scale_name,
            best_val_acc,
            scale_name,
            best_val_acc,
        )

    # === 跨尺度比较 / Cross-scale comparison ===
    logger.info("=" * 60)
    logger.info("Cross-scale comparison / 跨尺度比较")
    logger.info("=" * 60)

    scale_names = list(results["scale_models"].keys())
    for i in range(len(scale_names)):
        for j in range(i + 1, len(scale_names)):
            emb_i = results["scale_models"][scale_names[i]]["embeddings"]
            emb_j = results["scale_models"][scale_names[j]]["embeddings"]

            # 对齐维度 / Align dimensions
            min_dim = min(emb_i.shape[-1], emb_j.shape[-1])
            emb_i_aligned = emb_i[..., :min_dim]
            emb_j_aligned = emb_j[..., :min_dim]

            # Procrustes距离 / Procrustes distance
            centroid_i = emb_i_aligned.mean(dim=0)
            centroid_j = emb_j_aligned.mean(dim=0)
            proc_dist = torch.norm(centroid_i - centroid_j).item()

            # 内在维度差异 / Intrinsic dimensionality difference
            dim_i = emb_i_aligned.shape[-1]
            dim_j = emb_j_aligned.shape[-1]

            logger.info(
                "Comparison %s vs %s: Procrustes=%.4f | Dim diff=%d / "
                "比较 %s vs %s: Procrustes=%.4f | 维度差=%d",
                scale_names[i],
                scale_names[j],
                proc_dist,
                dim_j - dim_i,
                scale_names[i],
                scale_names[j],
                proc_dist,
                dim_j - dim_i,
            )

            results["comparisons"].append(
                {
                    "scale_a": scale_names[i],
                    "scale_b": scale_names[j],
                    "procrustes_distance": proc_dist,
                    "dim_difference": dim_j - dim_i,
                }
            )

    # === 训练状态转换检测器 / Train regime transition detector ===
    logger.info("=" * 60)
    logger.info("Training regime transition detector / 训练状态转换检测器")
    logger.info("=" * 60)

    # 使用中等规模模型的嵌入 / Use medium-scale model embeddings
    medium_key = "medium" if "medium" in results["scale_models"] else scale_names[0]
    medium_model = results["scale_models"][medium_key]["model"]

    transition_detector = RegimeTransitionDetector(
        embed_dim=config.embed_dim,
    ).to(device)

    td_optimizer = optim.AdamW(transition_detector.parameters(), lr=1e-3)

    # 生成转换训练数据 / Generate transition training data
    model = medium_model
    model.eval()
    transition_data = []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            output = model(sequences, return_geometry=False)
            # 使用隐藏状态作为序列嵌入 / Use hidden states as sequence embeddings
            emb_seq = output["hidden_states"][:, :, :config.embed_dim]
            transition_data.append((emb_seq.cpu(), torch.tensor(labels)))

    # 简单训练 / Simple training
    for epoch in range(20):
        transition_detector.train()
        total_loss = 0.0
        for emb_seq, labels in transition_data:
            emb_seq = emb_seq.to(device)
            td_optimizer.zero_grad()
            td_output = transition_detector(emb_seq)
            # 简单损失：转换概率的稀疏性（大多数时间点不应有转换）
            # Simple loss: sparsity of transition probabilities
            sparsity_loss = td_output["transition_prob"].mean()
            sparsity_loss.backward()
            td_optimizer.step()
            total_loss += sparsity_loss.item()

        if (epoch + 1) % 5 == 0:
            logger.info(
                "Transition detector epoch %d: loss=%.4f / "
                "转换检测器第 %d 轮: 损失=%.4f",
                epoch + 1,
                total_loss / len(transition_data),
                epoch + 1,
                total_loss / len(transition_data),
            )

    # === 规模依赖性信号分析 / Scale-dependent signal analysis ===
    logger.info("=" * 60)
    logger.info("Analyzing scale-dependent signals / 分析规模依赖性信号")
    logger.info("=" * 60)

    signal_analyzer = ScaleDependentSignalAnalyzer(
        embed_dim=config.embed_dim,
        n_scales=len(scale_names),
    ).to(device)

    # 收集每个规模的平均嵌入 / Collect mean embeddings per scale
    scale_embs_for_analysis = []
    for sn in scale_names:
        emb = results["scale_models"][sn]["embeddings"]
        mean_emb = emb.mean(dim=0, keepdim=True)  # [1, D]
        scale_embs_for_analysis.append(mean_emb.to(device))

    # 扩展为batch / Expand to batch
    batch_size_for_analysis = 32
    scale_embs_batched = [
        e.expand(batch_size_for_analysis, -1) for e in scale_embs_for_analysis
    ]

    signal_analyzer.eval()
    with torch.no_grad():
        signal_result = signal_analyzer(scale_embs_batched)
        emergence = signal_result["emergence"]  # [B, n_scales]
        quality = signal_result["signal_quality"]  # [B, n_scales]

    logger.info(
        "Emergence pattern (avg across batch) / 涌现模式（batch平均）: %s",
        emergence.mean(dim=0).tolist(),
    )
    logger.info(
        "Signal quality (avg across batch) / 信号质量（batch平均）: %s",
        quality.mean(dim=0).tolist(),
    )

    # === 保存结果 / Save results ===
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存最佳模型 / Save best model
    for scale_name, scale_data in results["scale_models"].items():
        model_path = output_dir / f"model_{scale_name}.pt"
        torch.save(
            {
                "model_state_dict": scale_data["model"].state_dict(),
                "best_val_acc": scale_data["best_val_acc"],
                "scale": scale_name,
            },
            model_path,
        )
        logger.info("Saved model to / 保存模型到: %s", model_path)

    # 保存转换检测器 / Save transition detector
    td_path = output_dir / "transition_detector.pt"
    torch.save(transition_detector.state_dict(), td_path)
    logger.info("Saved transition detector to / 保存转换检测器到: %s", td_path)

    # 保存比较结果 / Save comparison results
    comparison_path = output_dir / "scale_comparisons.pt"
    torch.save(results["comparisons"], comparison_path)

    logger.info(
        "Training complete! Results saved to / 训练完成！结果保存到: %s",
        output_dir,
    )

    return results


# ============================================================================
# 命令行接口 / CLI
# ============================================================================


def parse_args() -> TrainingConfig:
    """解析命令行参数 / Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Market Reasoning Geometry Training / 市场推理几何训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="data/market",
                        help="数据目录 / Data directory")
    parser.add_argument("--output-dir", type=str, default="results/regime_detector",
                        help="输出目录 / Output directory")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 / Training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批次大小 / Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="学习率 / Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="权重衰减 / Weight decay")
    parser.add_argument("--input-features", type=int, default=128,
                        help="输入特征数 / Input features")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="隐藏维度 / Hidden dimension")
    parser.add_argument("--embed-dim", type=int, default=128,
                        help="嵌入维度 / Embedding dimension")
    parser.add_argument("--n-heads", type=int, default=8,
                        help="注意力头数 / Attention heads")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="Transformer层数 / Transformer layers")
    parser.add_argument("--model-scales", nargs="+", default=["small", "medium", "large"],
                        help="模型规模列表 / Model scales")
    parser.add_argument("--regime-types", nargs="+",
                        default=["trending", "mean_reverting", "volatile"],
                        help="市场状态类型 / Regime types")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 / Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备 / Device")

    args = parser.parse_args()

    return TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        input_features=args.input_features,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        model_scales=args.model_scales,
        regime_types=args.regime_types,
        seed=args.seed,
        device=args.device,
    )


# ============================================================================
# 入口 / Entry Point
# ============================================================================


if __name__ == "__main__":
    config = parse_args()
    logger.info(
        "Configuration / 配置: %s",
        {k: v for k, v in config.__dict__.items() if k not in ("data_dir",)},
    )
    results = train_regime_detector(config)
    logger.info("All done / 全部完成")
