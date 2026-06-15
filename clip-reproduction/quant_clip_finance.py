"""
CLIP × 量化金融: 多模态Alpha信号融合
CLIP for Quantitative Finance: Multimodal Alpha Signal Fusion
=============================================================

基于CLIP (Radford et al., 2021) 的金融多模态对齐实现。

核心思想: 将CLIP的图文对齐能力迁移到金融领域，
对齐 "新闻文本"、"K线图表"、"因子数据" 到同一嵌入空间。

三个应用:
1. 金融CLIP: 新闻-价格图表对齐 → 零样本事件分类
2. 多模态因子融合: 另类数据统一表示
3. 新闻情绪量化: 基于嵌入距离的细粒度情感

运行: python quant_clip_finance.py
依赖: pip install torch numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, List


# ============================================================
# 1. 金融CLIP / Finance CLIP
# ============================================================

class NewsEncoder(nn.Module):
    """
    金融新闻编码器 / Financial News Encoder

    将新闻文本编码为固定维度向量。
    简化版: 使用1D-CNN + Attention池化 (比Transformer更轻量)

    量化意义: 快速编码大量新闻流，低延迟处理实时新闻。
    """

    def __init__(self, vocab_size: int = 30000, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 多尺度CNN捕获不同长度的金融语义
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]  # 短词组、短语、句子级
        ])

        self.proj = nn.Linear(hidden_dim * 3, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_ids: [batch, seq_len] 新闻文本token IDs

        Returns:
            [batch, embed_dim] 新闻表示向量
        """
        x = self.embedding(text_ids)  # [B, T, embed_dim]
        x = x.transpose(1, 2)        # [B, embed_dim, T]

        # 多尺度卷积
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(x))  # [B, hidden_dim, T]
            c = c.mean(dim=-1)   # [B, hidden_dim] 时间维度池化
            conv_outs.append(c)

        x = torch.cat(conv_outs, dim=-1)  # [B, hidden_dim * 3]
        x = self.proj(x)
        x = self.norm(x)
        return x


class PriceChartEncoder(nn.Module):
    """
    K线图表编码器 / Price Chart (Candlestick) Encoder

    将K线数据(OHLCV)编码为向量。
    输入: 过去N天的Open/High/Low/Close/Volume数据。

    量化意义: 将技术分析的"图形识别"能力向量化，
    与新闻文本对齐后可做跨模态检索。
    """

    def __init__(self, lookback: int = 20, embed_dim: int = 256):
        super().__init__()
        # OHLCV = 5 channels
        self.conv_layers = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ohlcv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ohlcv: [batch, lookback, 5] K线数据 (Open, High, Low, Close, Volume)

        Returns:
            [batch, embed_dim] 价格图表表示
        """
        x = ohlcv.transpose(1, 2)   # [B, 5, lookback]
        x = self.conv_layers(x)     # [B, 64, 1]
        x = x.squeeze(-1)           # [B, 64]
        x = self.proj(x)
        x = self.norm(x)
        return x


class FactorEncoder(nn.Module):
    """
    因子数据编码器 / Factor Data Encoder

    将多因子数据编码为向量。
    输入: 多因子截面数据 (如: 估值因子, 动量因子, 波动率因子等)
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
        Args:
            factors: [batch, num_factors] 因子截面数据

        Returns:
            [batch, embed_dim] 因子表示
        """
        return self.net(factors)


class FinanceCLIP(nn.Module):
    """
    金融CLIP / Finance CLIP

    对齐三种金融模态到同一嵌入空间:
    1. 新闻文本 (News)
    2. K线图表 (Price Charts)
    3. 因子数据 (Factors)

    使用InfoNCE对比学习:
    - 同一时间点的 (新闻, K线) 是正样本对
    - 不同时间点的组合是负样本

    应用:
    - 零样本事件分类: 用自然语言描述查询相似历史模式
    - 多模态检索: 给定一张K线图，检索相关新闻
    - 信号融合: 多模态表示的加权平均作为Alpha信号
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

        # 三个模态编码器
        self.news_encoder = NewsEncoder(vocab_size, embed_dim)
        self.chart_encoder = PriceChartEncoder(lookback, embed_dim)
        self.factor_encoder = FactorEncoder(num_factors, embed_dim)

        # 共享投影空间
        self.news_proj = nn.Linear(embed_dim, projection_dim)
        self.chart_proj = nn.Linear(embed_dim, projection_dim)
        self.factor_proj = nn.Linear(embed_dim, projection_dim)

        # 可学习温度
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def encode_news(self, text_ids: torch.Tensor) -> torch.Tensor:
        features = self.news_proj(self.news_encoder(text_ids))
        return F.normalize(features, dim=-1)

    def encode_chart(self, ohlcv: torch.Tensor) -> torch.Tensor:
        features = self.chart_proj(self.chart_encoder(ohlcv))
        return F.normalize(features, dim=-1)

    def encode_factors(self, factors: torch.Tensor) -> torch.Tensor:
        features = self.factor_proj(self.factor_encoder(factors))
        return F.normalize(features, dim=-1)

    def contrastive_loss(
        self,
        text_ids: torch.Tensor,
        ohlcv: torch.Tensor,
        factors: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        三模态对比损失 / Tri-modal Contrastive Loss

        同时对齐: 新闻↔K线, 新闻↔因子, K线↔因子
        """
        news_emb = self.encode_news(text_ids)
        chart_emb = self.encode_chart(ohlcv)

        scale = self.logit_scale.exp().clamp(max=100.0)
        B = text_ids.shape[0]
        labels = torch.arange(B, device=text_ids.device)

        # 新闻 ↔ K线
        sim_nc = scale * (news_emb @ chart_emb.T)
        loss_nc = (F.cross_entropy(sim_nc, labels) + F.cross_entropy(sim_nc.T, labels)) / 2

        total_loss = loss_nc
        losses = {'news_chart': loss_nc}

        # 如果有因子数据，加入更多对比对
        if factors is not None:
            factor_emb = self.encode_factors(factors)

            # 新闻 ↔ 因子
            sim_nf = scale * (news_emb @ factor_emb.T)
            loss_nf = (F.cross_entropy(sim_nf, labels) + F.cross_entropy(sim_nf.T, labels)) / 2

            # K线 ↔ 因子
            sim_cf = scale * (chart_emb @ factor_emb.T)
            loss_cf = (F.cross_entropy(sim_cf, labels) + F.cross_entropy(sim_cf.T, labels)) / 2

            total_loss = total_loss + loss_nf + loss_cf
            losses['news_factor'] = loss_nf
            losses['chart_factor'] = loss_cf

        losses['total'] = total_loss
        return losses

    def zero_shot_event_classify(
        self,
        ohlcv: torch.Tensor,
        event_descriptions: List[str],
        tokenizer_fn=None,
    ) -> torch.Tensor:
        """
        零样本事件分类 / Zero-Shot Event Classification

        用自然语言描述金融事件，无需重新训练即可对新事件做分类。

        示例事件描述:
        - "stock price surges on earnings beat"  (业绩超预期上涨)
        - "stock crashes on regulatory action"   (监管利空暴跌)
        - "sideways consolidation before breakout" (突破前横盘整理)
        - "V-shaped reversal after panic selling" (恐慌抛售后的V型反转)

        Args:
            ohlcv: [batch, lookback, 5] K线数据
            event_descriptions: 事件描述文本列表
            tokenizer_fn: 文本→token IDs的函数

        Returns:
            probs: [batch, num_events] 各事件的概率
        """
        chart_emb = self.encode_chart(ohlcv)  # [B, D]

        # 编码事件描述 (简化: 随机ID，实际中应使用tokenizer)
        if tokenizer_fn is None:
            # 模拟: 随机token IDs
            text_ids = torch.randint(0, 30000, (len(event_descriptions), 50),
                                     device=ohlcv.device)
        else:
            text_ids = tokenizer_fn(event_descriptions)

        text_emb = self.encode_news(text_ids)  # [C, D]

        similarity = chart_emb @ text_emb.T
        probs = similarity.softmax(dim=-1)
        return probs

    def multimodal_alpha_signal(
        self,
        text_ids: torch.Tensor,
        ohlcv: torch.Tensor,
        factors: torch.Tensor,
        weights: Optional[Tuple[float, float, float]] = None,
    ) -> torch.Tensor:
        """
        多模态Alpha信号 / Multimodal Alpha Signal

        融合三种模态的表示作为综合Alpha信号。
        可配置各模态权重。

        Args:
            weights: (news_weight, chart_weight, factor_weight) 默认等权

        Returns:
            alpha_signal: [batch, projection_dim] 融合信号
        """
        if weights is None:
            weights = (1/3, 1/3, 1/3)

        news_emb = self.encode_news(text_ids)
        chart_emb = self.encode_chart(ohlcv)
        factor_emb = self.encode_factors(factors)

        alpha = (
            weights[0] * news_emb +
            weights[1] * chart_emb +
            weights[2] * factor_emb
        )
        return alpha


# ============================================================
# 2. 新闻情绪量化引擎 / News Sentiment Quantification Engine
# ============================================================

class NewsSentimentEngine:
    """
    基于FinanceCLIP的新闻情绪量化引擎

    传统方法: NLP情感分析 → 正/负/中 三分类 (粗粒度)
    本方法:  新闻嵌入 → 与"原型情绪"的余弦距离 → 连续情绪分数

    原型情绪:
    - "bullish": 强烈的看多信号
    - "bearish": 强烈的看空信号
    - "uncertain": 不确定性/模糊性
    - "neutral": 中性信息
    """

    def __init__(self, model: FinanceCLIP):
        self.model = model
        self.model.eval()

        # 预计算原型情绪向量 (简化版)
        with torch.no_grad():
            # 实际中应使用精心设计的prompt文本
            self.prototypes = {
                'bullish': torch.randn(128),    # 占位: 实际应由模型编码
                'bearish': torch.randn(128),
                'uncertain': torch.randn(128),
                'neutral': torch.randn(128),
            }
            # 归一化
            for k in self.prototypes:
                self.prototypes[k] = F.normalize(self.prototypes[k], dim=-1)

    def score_news(self, text_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        对新闻进行多维度情绪评分

        Returns:
            dict with sentiment scores for each prototype
        """
        with torch.no_grad():
            news_emb = self.model.encode_news(text_ids)  # [B, D]

        scores = {}
        for sentiment, prototype in self.prototypes.items():
            # 余弦相似度 → 情绪分数
            sim = (news_emb * prototype.unsqueeze(0)).sum(dim=-1)
            scores[sentiment] = sim

        # 综合情绪指标: bullish - bearish (范围约 [-2, 2])
        scores['composite'] = scores['bullish'] - scores['bearish']

        # 不确定性指标: 高不确定性 → 减小仓位
        scores['uncertainty'] = scores['uncertain']

        return scores


# ============================================================
# 训练与评估示例 / Training & Evaluation Example
# ============================================================

def train_finance_clip():
    """金融CLIP训练与评估流程"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FinanceCLIP(
        vocab_size=10000,
        embed_dim=128,
        projection_dim=64,
        lookback=20,
        num_factors=30,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 模拟金融数据
    batch_size = 32
    text_ids = torch.randint(0, 10000, (batch_size, 50), device=device)
    ohlcv = torch.randn(batch_size, 20, 5, device=device)  # 20天OHLCV
    factors = torch.randn(batch_size, 30, device=device)    # 30个因子

    # 训练
    model.train()
    losses = model.contrastive_loss(text_ids, ohlcv, factors)

    losses['total'].backward()
    optimizer.step()

    print("=" * 60)
    print("Finance CLIP Training")
    print("=" * 60)
    for name, loss in losses.items():
        print(f"  {name}: {loss.item():.4f}")

    # 零样本事件分类
    model.eval()
    test_charts = torch.randn(4, 20, 5, device=device)
    events = [
        "earnings beat, stock surges",
        "regulatory crackdown, stock crashes",
        "consolidation before breakout",
        "panic selling then V-reversal",
    ]

    with torch.no_grad():
        probs = model.zero_shot_event_classify(test_charts, events)

    print(f"\n{'='*60}")
    print("Zero-Shot Event Classification")
    print("=" * 60)
    for i in range(4):
        pred = probs[i].argmax().item()
        print(f"  Chart {i}: {events[pred]} (prob: {probs[i][pred]:.3f})")

    # 多模态Alpha信号
    with torch.no_grad():
        alpha = model.multimodal_alpha_signal(
            text_ids[:4], ohlcv[:4], factors[:4],
            weights=(0.3, 0.3, 0.4),  # 因子权重最高
        )

    print(f"\n{'='*60}")
    print("Multimodal Alpha Signal")
    print("=" * 60)
    print(f"  Signal shape: {alpha.shape}")
    print(f"  Signal L2 norm: {alpha.norm(dim=-1).numpy().round(3)}")

    # 新闻情绪引擎
    engine = NewsSentimentEngine(model)
    sentiments = engine.score_news(text_ids[:4])

    print(f"\n{'='*60}")
    print("News Sentiment Scoring")
    print("=" * 60)
    for key, val in sentiments.items():
        print(f"  {key}: {val[:4].numpy().round(3)}")

    # 模型统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Model Statistics")
    print(f"{'='*60}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  News encoder: {sum(p.numel() for p in model.news_encoder.parameters()):,}")
    print(f"  Chart encoder: {sum(p.numel() for p in model.chart_encoder.parameters()):,}")
    print(f"  Factor encoder: {sum(p.numel() for p in model.factor_encoder.parameters()):,}")


if __name__ == "__main__":
    print("=" * 60)
    print("CLIP × Quantitative Finance")
    print("金融多模态Alpha信号融合")
    print("=" * 60)

    train_finance_clip()
