"""
Finance CLIP 训练脚本 / Finance CLIP Training Script
=====================================================

用法 / Usage:
    python quant/quant_train.py
    python quant/quant_train.py --batch_size 64 --max_steps 5000

本脚本训练一个三模态金融CLIP模型 / This script trains a tri-modal Finance CLIP model:
  - 新闻文本 (News)       ← NewsEncoder (多尺度CNN)
  - K线图表 (Price Chart) ← PriceChartEncoder (CNN)
  - 因子数据 (Factors)    ← FactorEncoder (MLP)

使用模拟金融数据进行演示，可替换为真实数据。
Uses simulated financial data for demonstration; can be replaced with real data.

依赖 / Dependencies: pip install torch numpy
"""

import os
import sys
import argparse
import logging
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

# 添加父目录到路径以便导入 / Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 金融模态编码器 / Financial Modality Encoders
# ============================================================

class NewsEncoder(nn.Module):
    """
    金融新闻编码器 / Financial News Encoder.

    将新闻文本编码为固定维度向量。
    Encodes news text into a fixed-dimension vector.

    架构 / Architecture:
      1. Token Embedding: 词嵌入 / Word embedding
      2. 多尺度1D-CNN: 捕获短词组(3)、短语(5)、句子级(7)语义
         Multi-scale 1D-CNN: captures n-gram(3), phrase(5), sentence-level(7) semantics
      3. 时间维度池化: 对每个卷积核的输出做平均池化
         Temporal pooling: average pool each kernel's output
      4. 投影 + LayerNorm: 输出最终表示 / Project + LayerNorm for final representation

    量化意义 / Quant Significance:
      低延迟处理实时新闻流，适合高频策略中的新闻驱动信号。
      Low-latency processing of real-time news streams, suitable for
      news-driven signals in high-frequency strategies.

    Args:
        vocab_size (int): 词表大小 / Vocabulary size.
        embed_dim (int):  嵌入维度 / Embedding dimension.
        hidden_dim (int): CNN隐藏维度 / CNN hidden dimension.
    """

    def __init__(self, vocab_size: int = 30000, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 多尺度卷积: 不同kernel大小捕获不同长度的金融语义模式
        # Multi-scale convolution: different kernel sizes capture financial
        # semantic patterns of different lengths
        # kernel=3: 短词组如"earnings beat" / Short n-grams like "earnings beat"
        # kernel=5: 短语如"stock price surges on" / Phrases like "stock price surges on"
        # kernel=7: 句子级如完整的新闻标题 / Sentence-level like full news headlines
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7]
        ])

        self.proj = nn.Linear(hidden_dim * 3, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward pass.

        Args:
            text_ids: [B, T] 新闻文本token IDs / News text token IDs.

        Returns:
            [B, embed_dim] 新闻表示向量 / News representation vector.
        """
        # Token嵌入 / Token embedding
        x = self.embedding(text_ids)    # [B, T, embed_dim]
        x = x.transpose(1, 2)           # [B, embed_dim, T]

        # 多尺度卷积 + 时间池化 / Multi-scale conv + temporal pooling
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(x))         # [B, hidden_dim, T]
            c = c.mean(dim=-1)          # [B, hidden_dim] 时间维度平均 / Temporal average
            conv_outs.append(c)

        # 拼接多尺度特征 / Concatenate multi-scale features
        x = torch.cat(conv_outs, dim=-1)  # [B, hidden_dim * 3]

        # 投影 + 归一化 / Project + normalize
        x = self.proj(x)
        x = self.norm(x)
        return x


class PriceChartEncoder(nn.Module):
    """
    K线图表编码器 / Price Chart (Candlestick) Encoder.

    将K线数据(OHLCV)编码为向量。
    Encodes OHLCV candlestick data into a vector.

    输入格式 / Input format:
      [batch, lookback, 5] — Open, High, Low, Close, Volume

    架构 / Architecture:
      两层1D-CNN + 自适应池化 + 投影
      Two-layer 1D-CNN + adaptive pooling + projection

    量化意义 / Quant Significance:
      将技术分析的"图形识别"能力向量化。传统技术分析依赖人工识别
      K线形态(头肩顶、双底等)，本编码器自动学习这些模式的向量表示。
      与新闻文本对齐后可做跨模态检索: 给定一个K线图形，找到历史上
      类似的新闻事件。

      Vectorizes the "pattern recognition" capability of technical analysis.
      Traditional TA relies on manual identification of candlestick patterns
      (head & shoulders, double bottom, etc.); this encoder automatically
      learns vector representations of these patterns. After alignment with
      news text, enables cross-modal retrieval: given a chart pattern, find
      historically similar news events.

    Args:
        lookback (int): 回溯天数 / Lookback window in days.
        embed_dim (int): 嵌入维度 / Embedding dimension.
    """

    def __init__(self, lookback: int = 20, embed_dim: int = 256):
        super().__init__()
        # OHLCV = 5个通道 / 5 channels
        self.conv_layers = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 自适应池化到固定长度 / Adaptive pool to fixed length
        )
        self.proj = nn.Linear(64, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ohlcv: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward pass.

        Args:
            ohlcv: [B, lookback, 5] K线数据 (O, H, L, C, V).

        Returns:
            [B, embed_dim] 价格图表表示 / Price chart representation.
        """
        x = ohlcv.transpose(1, 2)   # [B, 5, lookback] 通道维度在前 / Channel-first
        x = self.conv_layers(x)     # [B, 64, 1]
        x = x.squeeze(-1)           # [B, 64]
        x = self.proj(x)
        x = self.norm(x)
        return x


class FactorEncoder(nn.Module):
    """
    因子数据编码器 / Factor Data Encoder.

    将多因子截面数据编码为向量。
    Encodes cross-sectional multi-factor data into a vector.

    输入 / Input:
      [batch, num_factors] — 各因子暴露值 / Factor exposure values
      例如 / e.g.: 估值因子(PE), 动量因子, 波动率因子, 质量因子...

    架构 / Architecture:
      两层MLP + ReLU + LayerNorm

    量化意义 / Quant Significance:
      因子投资是量化的核心方法之一。将因子数据嵌入到与新闻和图表
      共享的空间，可以发现因子值与事件/形态之间的隐含关系。
      例如: 低估值 + 利好新闻 → 更强的做多信号。

      Factor investing is a core quantitative method. Embedding factor data
      into a space shared with news and charts reveals implicit relationships
      between factor values and events/patterns.
      E.g.: low valuation + positive news -> stronger long signal.

    Args:
        num_factors (int): 因子数量 / Number of factors.
        embed_dim (int):   嵌入维度 / Embedding dimension.
    """

    def __init__(self, num_factors: int = 50, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_factors, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, factors: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward pass.

        Args:
            factors: [B, num_factors] 因子截面数据 / Factor cross-sectional data.

        Returns:
            [B, embed_dim] 因子表示 / Factor representation.
        """
        return self.net(factors)


# ============================================================
# Finance CLIP 模型 / Finance CLIP Model
# ============================================================

class FinanceCLIP(nn.Module):
    """
    金融CLIP模型 — 三模态对比学习
    Finance CLIP Model — Tri-modal Contrastive Learning.

    对齐三种金融模态到同一嵌入空间 / Aligns three financial modalities into shared space:
      1. 新闻文本 (News)
      2. K线图表 (Price Charts)
      3. 因子数据 (Factors)

    对比学习策略 / Contrastive Learning Strategy:
      - 同一时间点 (stock_id, timestamp) 的 (新闻, K线, 因子) 是正样本对
        (News, Chart, Factor) from the same (stock_id, timestamp) are positive pairs
      - 不同时间点的组合是负样本 (batch内的其他样本)
        Combinations from different timestamps are negatives (other samples in batch)

    损失函数 / Loss:
      三组对称InfoNCE损失之和:
      Sum of three symmetric InfoNCE losses:
        L = L(news, chart) + L(news, factor) + L(chart, factor)

    Args:
        vocab_size (int):   词表大小 / Vocabulary size.
        embed_dim (int):    编码器维度 / Encoder dimension.
        projection_dim (int): 投影维度 / Projection dimension.
        lookback (int):     K线回溯天数 / Price chart lookback days.
        num_factors (int):  因子数量 / Number of factors.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 256,
        projection_dim: int = 128,
        lookback: int = 20,
        num_factors: int = 50,
    ):
        super().__init__()
        self.projection_dim = projection_dim

        # 三个模态编码器 / Three modality encoders
        self.news_encoder = NewsEncoder(vocab_size, embed_dim)
        self.chart_encoder = PriceChartEncoder(lookback, embed_dim)
        self.factor_encoder = FactorEncoder(num_factors, embed_dim)

        # 投影到共享空间 / Project to shared space
        self.news_proj = nn.Linear(embed_dim, projection_dim)
        self.chart_proj = nn.Linear(embed_dim, projection_dim)
        self.factor_proj = nn.Linear(embed_dim, projection_dim)

        # 可学习温度参数 / Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

    def encode_news(self, text_ids: torch.Tensor) -> torch.Tensor:
        """编码新闻并归一化 / Encode news and normalize."""
        features = self.news_proj(self.news_encoder(text_ids))
        return F.normalize(features, dim=-1)

    def encode_chart(self, ohlcv: torch.Tensor) -> torch.Tensor:
        """编码K线并归一化 / Encode chart and normalize."""
        features = self.chart_proj(self.chart_encoder(ohlcv))
        return F.normalize(features, dim=-1)

    def encode_factors(self, factors: torch.Tensor) -> torch.Tensor:
        """编码因子并归一化 / Encode factors and normalize."""
        features = self.factor_proj(self.factor_encoder(factors))
        return F.normalize(features, dim=-1)

    def contrastive_loss(
        self,
        text_ids: torch.Tensor,
        ohlcv: torch.Tensor,
        factors: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        三模态对比损失 / Tri-modal Contrastive Loss.

        计算所有模态对之间的对称InfoNCE损失:
        Computes symmetric InfoNCE loss for all modality pairs:
          - 新闻↔K线 / News<->Chart
          - 新闻↔因子 / News<->Factor (如果有因子数据 / if factor data available)
          - K线↔因子 / Chart<->Factor (如果有因子数据 / if factor data available)

        Args:
            text_ids: [B, T] 新闻文本 / News text.
            ohlcv:    [B, lookback, 5] K线数据 / Chart data.
            factors:  [B, num_factors] 因子数据 / Factor data (optional).

        Returns:
            dict: 各项损失 / Dictionary of individual and total losses.
        """
        news_emb = self.encode_news(text_ids)     # [B, D]
        chart_emb = self.encode_chart(ohlcv)      # [B, D]

        scale = self.logit_scale.exp().clamp(max=100.0)
        B = text_ids.shape[0]
        labels = torch.arange(B, device=text_ids.device)

        # ---- 新闻 ↔ K线 / News <-> Chart ----
        sim_nc = scale * (news_emb @ chart_emb.T)
        loss_nc = (F.cross_entropy(sim_nc, labels) + F.cross_entropy(sim_nc.T, labels)) / 2

        total_loss = loss_nc
        losses = {"news_chart": loss_nc}

        # ---- 如果有因子数据，加入更多对比对 / If factor data, add more contrastive pairs ----
        if factors is not None:
            factor_emb = self.encode_factors(factors)  # [B, D]

            # 新闻 ↔ 因子 / News <-> Factor
            sim_nf = scale * (news_emb @ factor_emb.T)
            loss_nf = (F.cross_entropy(sim_nf, labels) + F.cross_entropy(sim_nf.T, labels)) / 2

            # K线 ↔ 因子 / Chart <-> Factor
            sim_cf = scale * (chart_emb @ factor_emb.T)
            loss_cf = (F.cross_entropy(sim_cf, labels) + F.cross_entropy(sim_cf.T, labels)) / 2

            total_loss = total_loss + loss_nf + loss_cf
            losses["news_factor"] = loss_nf
            losses["chart_factor"] = loss_cf

        losses["total"] = total_loss
        return losses

    def multimodal_alpha_signal(
        self,
        text_ids: torch.Tensor,
        ohlcv: torch.Tensor,
        factors: torch.Tensor,
        weights: Optional[Tuple[float, float, float]] = None,
    ) -> torch.Tensor:
        """
        多模态Alpha信号 / Multimodal Alpha Signal.

        融合三种模态的归一化表示作为综合Alpha信号。
        Fuses normalized representations from all three modalities
        as a comprehensive alpha signal.

        Args:
            text_ids: [B, T] 新闻 / News.
            ohlcv:    [B, lookback, 5] K线 / Chart.
            factors:  [B, num_factors] 因子 / Factors.
            weights:  (新闻权重, K线权重, 因子权重) / (news_w, chart_w, factor_w).

        Returns:
            [B, projection_dim] 融合Alpha信号 / Fused alpha signal.
        """
        if weights is None:
            weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)

        news_emb = self.encode_news(text_ids)
        chart_emb = self.encode_chart(ohlcv)
        factor_emb = self.encode_factors(factors)

        alpha = (
            weights[0] * news_emb +
            weights[1] * chart_emb +
            weights[2] * factor_emb
        )
        return alpha

    @torch.no_grad()
    def zero_shot_event_classify(
        self,
        ohlcv: torch.Tensor,
        event_text_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        零样本事件分类 / Zero-Shot Event Classification.

        用自然语言描述金融事件，无需重新训练即可分类K线形态。
        Classify chart patterns using natural language event descriptions,
        without retraining.

        Args:
            ohlcv:          [B, lookback, 5] K线数据 / Chart data.
            event_text_ids: [C, T] 事件描述文本的token IDs / Event description token IDs.

        Returns:
            [B, C] 各事件的概率 / Probability for each event.
        """
        chart_emb = self.encode_chart(ohlcv)       # [B, D]
        text_emb = self.encode_news(event_text_ids)  # [C, D]

        similarity = chart_emb @ text_emb.T         # [B, C]
        probs = similarity.softmax(dim=-1)
        return probs


# ============================================================
# 模拟金融数据生成 / Simulated Financial Data Generation
# ============================================================

def generate_mock_financial_data(
    batch_size: int,
    vocab_size: int,
    seq_len: int,
    lookback: int,
    num_factors: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    生成模拟金融数据用于训练演示。
    Generate simulated financial data for training demonstration.

    数据结构 / Data structure:
      - text_ids: 模拟新闻token / Simulated news tokens
      - ohlcv: 模拟K线 (正态分布) / Simulated OHLCV (normal distribution)
      - factors: 模拟因子暴露 / Simulated factor exposures
      - returns: 模拟未来收益 (用于评估) / Simulated future returns (for evaluation)
    """
    data = {
        "text_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        "ohlcv": torch.randn(batch_size, lookback, 5, device=device),
        "factors": torch.randn(batch_size, num_factors, device=device),
        "returns": torch.randn(batch_size, device=device),  # 未来N日收益 / Future N-day returns
    }
    return data


# ============================================================
# 训练循环 / Training Loop
# ============================================================

def parse_args():
    """解析命令行参数 / Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Finance CLIP 训练 / Finance CLIP Training"
    )
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--num_factors", type=int, default=30)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--experiment_name", type=str, default="finance_clip")
    return parser.parse_args()


def main():
    """主训练函数 / Main training function."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Finance CLIP 训练 / Finance CLIP Training")
    print("=" * 60)
    print(f"设备 / Device: {device}")

    # ---- 模型 / Model ----
    model = FinanceCLIP(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        projection_dim=args.projection_dim,
        lookback=args.lookback,
        num_factors=args.num_factors,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 / Total parameters: {total_params:,}")
    print(f"  NewsEncoder: {sum(p.numel() for p in model.news_encoder.parameters()):,}")
    print(f"  ChartEncoder: {sum(p.numel() for p in model.chart_encoder.parameters()):,}")
    print(f"  FactorEncoder: {sum(p.numel() for p in model.factor_encoder.parameters()):,}")

    # ---- 优化器 / Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    # ---- 学习率调度 (线性预热 + 余弦退火) ----
    # ---- LR schedule (linear warmup + cosine annealing) ----
    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.learning_rate * step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        progress = min(progress, 1.0)
        min_lr = 1e-6
        return min_lr + (args.learning_rate - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    # ---- 训练循环 / Training Loop ----
    model.train()
    running_losses: Dict[str, float] = {}

    print(f"\n开始训练 / Starting training: {args.max_steps} steps...")

    for step in range(1, args.max_steps + 1):
        # 生成模拟数据 / Generate mock data
        batch = generate_mock_financial_data(
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            lookback=args.lookback,
            num_factors=args.num_factors,
            device=device,
        )

        # 更新学习率 / Update learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 前向传播 + 计算损失 / Forward pass + compute loss
        losses = model.contrastive_loss(
            text_ids=batch["text_ids"],
            ohlcv=batch["ohlcv"],
            factors=batch["factors"],
        )

        # 反向传播 / Backward pass
        optimizer.zero_grad()
        losses["total"].backward()

        # 梯度裁剪 / Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 参数更新 / Parameter update
        optimizer.step()

        # 累计损失 / Accumulate losses
        for key, val in losses.items():
            if key not in running_losses:
                running_losses[key] = 0.0
            running_losses[key] += val.item()

        # 日志 / Logging
        if step % args.log_every == 0:
            avg_losses = {k: v / args.log_every for k, v in running_losses.items()}
            running_losses = {}

            temperature = 1.0 / model.logit_scale.exp().item()
            loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in avg_losses.items())
            print(f"  Step {step}/{args.max_steps} | {loss_str} | LR: {lr:.6f} | Temp: {temperature:.4f}")

    # ---- 保存检查点 / Save checkpoint ----
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": args.max_steps,
        "args": vars(args),
    }, save_path)

    print(f"\n检查点已保存 / Checkpoint saved: {save_path}")

    # ---- 训练后简单验证 / Post-training quick validation ----
    print("\n" + "-" * 60)
    print("训练后快速验证 / Post-Training Quick Validation")
    print("-" * 60)

    model.eval()
    with torch.no_grad():
        test_batch = generate_mock_financial_data(
            batch_size=16,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            lookback=args.lookback,
            num_factors=args.num_factors,
            device=device,
        )

        # 多模态Alpha信号 / Multimodal alpha signal
        alpha = model.multimodal_alpha_signal(
            test_batch["text_ids"],
            test_batch["ohlcv"],
            test_batch["factors"],
            weights=(0.3, 0.3, 0.4),
        )
        print(f"  Alpha信号形状 / Alpha signal shape: {alpha.shape}")
        print(f"  Alpha信号L2范数 / Alpha signal L2 norm: {alpha.norm(dim=-1).mean().item():.4f}")

        # 温度 / Temperature
        print(f"  最终温度 / Final temperature: {1.0 / model.logit_scale.exp().item():.4f}")

    print("\n" + "=" * 60)
    print("训练完成 / Training Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
