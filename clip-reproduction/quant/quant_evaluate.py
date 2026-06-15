"""
Finance CLIP 评估脚本 / Finance CLIP Evaluation Script
=======================================================

用法 / Usage:
    python quant/quant_evaluate.py
    python quant/quant_evaluate.py --checkpoint checkpoints/finance_clip_final.pt

评估三个维度 / Evaluates three dimensions:
  1. 零样本事件分类准确率 / Zero-shot event classification accuracy
  2. 新闻情绪分数与收益的相关性 / News sentiment score correlation with returns
  3. 多模态Alpha信号的信息系数 (IC) / Multimodal alpha signal Information Coefficient (IC)

依赖 / Dependencies: pip install torch numpy
"""

import os
import sys
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# 添加父目录到路径以便导入 / Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant.quant_train import (
    FinanceCLIP,
    NewsEncoder,
    PriceChartEncoder,
    FactorEncoder,
    generate_mock_financial_data,
)


# ============================================================
# 金融事件描述模板 / Financial Event Description Templates
# ============================================================

# 用于零样本事件分类的金融事件描述
# Financial event descriptions for zero-shot classification
EVENT_DESCRIPTIONS = [
    "stock price surges on earnings beat and raised guidance",
    "stock crashes on regulatory action and compliance failure",
    "sideways consolidation before breakout above resistance",
    "V-shaped reversal after panic selling and capitulation",
    "gradual uptrend with increasing volume and momentum",
    "sharp selloff on macroeconomic data disappointment",
    "gap up on positive analyst upgrade and price target raise",
    "breakdown below support level with high volume",
    "mean reversion after overextended rally in overbought conditions",
    "range-bound trading with decreasing volatility before event",
]

# 用于情绪量化的原型文本 / Prototype texts for sentiment scoring
SENTIMENT_PROTOTYPES = {
    "bullish": "strong earnings growth, record revenue, raised outlook, buyback program, analyst upgrade",
    "bearish": "revenue miss, profit warning, regulatory fine, executive departure, debt downgrade",
    "uncertain": "mixed results, unclear guidance, pending litigation, market volatility, restructuring",
    "neutral": "routine announcement, scheduled maintenance, regular dividend, standard filing",
}


# ============================================================
# 1. 零样本事件分类 / Zero-Shot Event Classification
# ============================================================

@torch.no_grad()
def evaluate_zero_shot_events(
    model: FinanceCLIP,
    test_ohlcv: torch.Tensor,
    event_text_ids: torch.Tensor,
    true_event_labels: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    评估零样本事件分类的准确率。
    Evaluate zero-shot event classification accuracy.

    原理 / Rationale:
      给定K线数据，使用自然语言事件描述作为候选类别，
      无需在特定任务上微调即可分类。这展示了CLIP的核心优势:
      将视觉模式与自然语言语义对齐。

      Given chart data, use natural language event descriptions as
      candidate classes. Classifies without task-specific fine-tuning.
      This demonstrates CLIP's core strength: aligning visual patterns
      with natural language semantics.

    Args:
        model:             Finance CLIP模型 / Finance CLIP model.
        test_ohlcv:        [N, lookback, 5] K线数据 / Chart data.
        event_text_ids:    [C, T] 事件描述的token IDs / Event description token IDs.
        true_event_labels: [N] 真实事件标签 / True event labels.
        device:            设备 / Device.

    Returns:
        dict: {"accuracy": float, "num_classes": int, "num_samples": int}
    """
    model.eval()

    # 编码K线和事件描述 / Encode charts and event descriptions
    chart_emb = model.encode_chart(test_ohlcv.to(device))        # [N, D]
    event_emb = model.encode_news(event_text_ids.to(device))     # [C, D]

    # 相似度 → 预测 / Similarity -> prediction
    similarity = chart_emb @ event_emb.T                          # [N, C]
    predictions = similarity.argmax(dim=1)                        # [N]

    # 计算准确率 / Compute accuracy
    correct = (predictions == true_event_labels.to(device)).float().sum().item()
    accuracy = correct / len(true_event_labels)

    num_classes = event_text_ids.shape[0]

    # Top-3 准确率 / Top-3 accuracy
    k = min(3, num_classes)
    _, topk_indices = similarity.topk(k, dim=1)
    top3_correct = 0
    for i in range(len(true_event_labels)):
        if true_event_labels[i].item() in topk_indices[i].tolist():
            top3_correct += 1
    top3_accuracy = top3_correct / len(true_event_labels)

    return {
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "num_classes": num_classes,
        "num_samples": len(true_event_labels),
        "random_baseline": 1.0 / num_classes,
    }


# ============================================================
# 2. 情绪评分与收益相关性 / Sentiment Scoring & Return Correlation
# ============================================================

@torch.no_grad()
def evaluate_sentiment_correlation(
    model: FinanceCLIP,
    test_text_ids: torch.Tensor,
    test_returns: torch.Tensor,
    prototype_text_ids: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    """
    评估新闻情绪分数与未来收益的相关性。
    Evaluate correlation between news sentiment scores and future returns.

    原理 / Rationale:
      1. 将新闻编码为向量 / Encode news into vectors
      2. 计算与原型情绪 (bullish/bearish) 的余弦相似度
         Compute cosine similarity with prototype sentiments (bullish/bearish)
      3. 综合情绪 = bullish_sim - bearish_sim
         Composite sentiment = bullish_sim - bearish_sim
      4. 计算情绪与未来收益的相关系数 (Pearson)
         Compute Pearson correlation between sentiment and future returns

    量化意义 / Quant Significance:
      如果情绪分数与未来收益有显著相关性，说明模型学到了有意义的
      金融语义。这是将CLIP嵌入用于Alpha信号的前提条件。

      If sentiment scores significantly correlate with future returns,
      the model has learned meaningful financial semantics. This is a
      prerequisite for using CLIP embeddings as alpha signals.

    Args:
        model:              Finance CLIP模型 / Finance CLIP model.
        test_text_ids:      [N, T] 新闻token IDs / News token IDs.
        test_returns:       [N] 未来收益 / Future returns.
        prototype_text_ids: dict of [1, T] 原型token IDs / Prototype token IDs.
        device:             设备 / Device.

    Returns:
        dict: 相关性指标 / Correlation metrics.
    """
    model.eval()

    # 编码新闻 / Encode news
    news_emb = model.encode_news(test_text_ids.to(device))  # [N, D]

    # 编码原型情绪 / Encode prototype sentiments
    prototypes = {}
    for name, p_ids in prototype_text_ids.items():
        p_emb = model.encode_news(p_ids.to(device))  # [1, D]
        prototypes[name] = p_emb.squeeze(0)           # [D]

    # 计算各维度的情绪分数 / Compute sentiment scores for each dimension
    sentiment_scores = {}
    for name, proto in prototypes.items():
        # 余弦相似度 (特征已归一化，点积 = 余弦)
        # Cosine similarity (features are normalized, dot = cosine)
        sim = (news_emb * proto.unsqueeze(0)).sum(dim=-1)  # [N]
        sentiment_scores[name] = sim

    # 综合情绪指标 / Composite sentiment metric
    composite = sentiment_scores["bullish"] - sentiment_scores["bearish"]

    # 计算与收益的Pearson相关系数 / Compute Pearson correlation with returns
    returns = test_returns.to(device)

    # Pearson相关系数 = Cov(X,Y) / (Std(X) * Std(Y))
    # Pearson correlation = Cov(X,Y) / (Std(X) * Std(Y))
    def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        cov = (x_centered * y_centered).mean()
        std_x = x.std()
        std_y = y.std()
        if std_x < 1e-8 or std_y < 1e-8:
            return 0.0
        return (cov / (std_x * std_y)).item()

    results = {
        "composite_corr": pearson_corr(composite, returns),
        "bullish_corr": pearson_corr(sentiment_scores["bullish"], returns),
        "bearish_corr": pearson_corr(sentiment_scores["bearish"], returns),
        "uncertain_corr": pearson_corr(sentiment_scores["uncertain"], returns),
        "neutral_corr": pearson_corr(sentiment_scores["neutral"], returns),
        "mean_composite_sentiment": composite.mean().item(),
        "std_composite_sentiment": composite.std().item(),
    }

    # Rank IC (Spearman相关系数) / Rank IC (Spearman correlation)
    # 量化中常用Rank IC，对异常值更鲁棒
    # Rank IC is common in quant, more robust to outliers
    def rank_ic(x: torch.Tensor, y: torch.Tensor) -> float:
        """
        计算Rank IC (Spearman秩相关系数)。
        Compute Rank IC (Spearman rank correlation).
        """
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()

        # 计算秩 / Compute ranks
        from numpy import argsort, empty_like
        x_rank = empty_like(x_np)
        y_rank = empty_like(y_np)
        x_rank[argsort(x_np)] = torch.arange(len(x_np)).float().numpy()
        y_rank[argsort(y_np)] = torch.arange(len(y_np)).float().numpy()

        x_rank_t = torch.tensor(x_rank, device=device)
        y_rank_t = torch.tensor(y_rank, device=device)
        return pearson_corr(x_rank_t, y_rank_t)

    results["composite_rank_ic"] = rank_ic(composite, returns)

    return results


# ============================================================
# 3. 多模态Alpha信号IC / Multimodal Alpha Signal IC
# ============================================================

@torch.no_grad()
def evaluate_alpha_signal_ic(
    model: FinanceCLIP,
    text_ids: torch.Tensor,
    ohlcv: torch.Tensor,
    factors: torch.Tensor,
    returns: torch.Tensor,
    weights: Tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3),
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    评估多模态Alpha信号的信息系数 (IC)。
    Evaluate Information Coefficient (IC) of multimodal alpha signal.

    原理 / Rationale:
      信息系数 (IC) 是量化投资中评估因子预测能力的核心指标。
      IC = Corr(因子值, 未来收益)。|IC| > 0.03 通常被认为有意义，
      |IC| > 0.05 被认为有实际应用价值。

      Information Coefficient (IC) is the core metric for evaluating
      factor predictive power in quantitative investing.
      IC = Corr(factor value, future return). |IC| > 0.03 is usually
      considered meaningful, |IC| > 0.05 is practically useful.

    评估方式 / Evaluation:
      1. 单模态IC: 分别评估新闻、K线、因子各模态的IC
         Single-modality IC: evaluate news, chart, factor ICs separately
      2. 多模态融合IC: 加权融合后的IC
         Multimodal fusion IC: IC after weighted fusion
      3. 融合增益: 多模态 vs 最优单模态的IC提升
         Fusion gain: multimodal vs best single-modality IC improvement

    Args:
        model:    Finance CLIP模型 / Finance CLIP model.
        text_ids: [N, T] 新闻 / News.
        ohlcv:    [N, lookback, 5] K线 / Chart.
        factors:  [N, num_factors] 因子 / Factors.
        returns:  [N] 未来收益 / Future returns.
        weights:  (w_news, w_chart, w_factor) 融合权重 / Fusion weights.
        device:   设备 / Device.

    Returns:
        dict: IC指标 / IC metrics.
    """
    model.eval()

    # 编码各模态 / Encode each modality
    news_emb = model.encode_news(text_ids.to(device))      # [N, D]
    chart_emb = model.encode_chart(ohlcv.to(device))       # [N, D]
    factor_emb = model.encode_factors(factors.to(device))  # [N, D]

    ret = returns.to(device)

    def compute_ic(emb: torch.Tensor, returns: torch.Tensor) -> float:
        """
        计算嵌入向量与收益的IC。
        Compute IC between embedding vectors and returns.

        对于多维嵌入，使用嵌入的L2范数作为因子值。
        For multi-dimensional embeddings, use L2 norm as factor value.
        """
        factor_values = emb.norm(dim=-1)  # [N]
        f_centered = factor_values - factor_values.mean()
        r_centered = returns - returns.mean()
        cov = (f_centered * r_centered).mean()
        std_f = factor_values.std()
        std_r = returns.std()
        if std_f < 1e-8 or std_r < 1e-8:
            return 0.0
        return (cov / (std_f * std_r)).item()

    # 单模态IC / Single-modality ICs
    news_ic = compute_ic(news_emb, ret)
    chart_ic = compute_ic(chart_emb, ret)
    factor_ic = compute_ic(factor_emb, ret)

    # 多模态融合信号 / Multimodal fused signal
    alpha = (
        weights[0] * news_emb +
        weights[1] * chart_emb +
        weights[2] * factor_emb
    )
    alpha_ic = compute_ic(alpha, ret)

    # 最佳单模态IC / Best single-modality IC
    best_single_ic = max(abs(news_ic), abs(chart_ic), abs(factor_ic))

    results = {
        "news_ic": news_ic,
        "chart_ic": chart_ic,
        "factor_ic": factor_ic,
        "alpha_ic": alpha_ic,
        "abs_alpha_ic": abs(alpha_ic),
        "best_single_ic": best_single_ic,
        "fusion_gain": abs(alpha_ic) - best_single_ic if best_single_ic > 0 else 0.0,
        "weights": weights,
    }

    return results


# ============================================================
# 主评估函数 / Main Evaluation Function
# ============================================================

def parse_args():
    """解析命令行参数 / Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Finance CLIP 评估 / Finance CLIP Evaluation"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--num_factors", type=int, default=30)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--num_test_samples", type=int, default=200)
    return parser.parse_args()


def build_prototype_token_ids(
    proto_texts: Dict[str, str],
    vocab_size: int,
    seq_len: int,
) -> Dict[str, torch.Tensor]:
    """
    将原型文本转为token IDs (简化版)。
    Convert prototype texts to token IDs (simplified).

    注意: 完整实现应使用训练好的BPE分词器。
    Note: Full implementation should use a trained BPE tokenizer.
    这里使用hash-based模拟来确保可复现性。
    Here we use hash-based simulation for reproducibility.
    """
    result = {}
    for name, text in proto_texts.items():
        # 基于文本内容生成确定性的token IDs
        # Generate deterministic token IDs based on text content
        text_bytes = text.encode("utf-8")
        ids = [0]  # SOS
        for b in text_bytes[:seq_len - 2]:
            ids.append((b * 7 + 13) % (vocab_size - 4) + 4)  # 避免特殊token / Avoid special tokens
        ids.append(1)  # EOS
        while len(ids) < seq_len:
            ids.append(2)  # PAD
        ids = ids[:seq_len]
        result[name] = torch.tensor([ids], dtype=torch.long)  # [1, seq_len]
    return result


def main():
    """主评估函数 / Main evaluation function."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Finance CLIP 评估 / Finance CLIP Evaluation")
    print("=" * 60)
    print(f"设备 / Device: {device}")

    # ---- 构建模型 / Build model ----
    model = FinanceCLIP(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        projection_dim=args.projection_dim,
        lookback=args.lookback,
        num_factors=args.num_factors,
    ).to(device)

    # 加载检查点 (如果有) / Load checkpoint if available
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"已加载检查点 / Loaded checkpoint: {args.checkpoint}")
    else:
        print("未提供检查点，使用随机权重进行演示")
        print("No checkpoint provided, demo with random weights")

    model.eval()

    # ---- 生成测试数据 / Generate test data ----
    N = args.num_test_samples
    num_events = len(EVENT_DESCRIPTIONS)

    test_data = generate_mock_financial_data(
        batch_size=N,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        lookback=args.lookback,
        num_factors=args.num_factors,
        device=device,
    )

    # 随机事件标签 (用于分类评估)
    # Random event labels (for classification evaluation)
    true_event_labels = torch.randint(0, num_events, (N,), device=device)

    # 构建事件描述的token IDs / Build token IDs for event descriptions
    event_text_ids_list = []
    for desc in EVENT_DESCRIPTIONS:
        text_bytes = desc.encode("utf-8")
        ids = [0]  # SOS
        for b in text_bytes[:args.seq_len - 2]:
            ids.append((b * 7 + 13) % (args.vocab_size - 4) + 4)
        ids.append(1)  # EOS
        while len(ids) < args.seq_len:
            ids.append(2)  # PAD
        event_text_ids_list.append(ids[:args.seq_len])
    event_text_ids = torch.tensor(event_text_ids_list, dtype=torch.long, device=device)

    # ================================================================
    # 评估1: 零样本事件分类 / Eval 1: Zero-Shot Event Classification
    # ================================================================
    print("\n" + "-" * 60)
    print("1. 零样本事件分类 / Zero-Shot Event Classification")
    print("-" * 60)

    zsl_results = evaluate_zero_shot_events(
        model=model,
        test_ohlcv=test_data["ohlcv"],
        event_text_ids=event_text_ids,
        true_event_labels=true_event_labels,
        device=device,
    )

    print(f"  事件类别数 / Number of event classes: {zsl_results['num_classes']}")
    print(f"  测试样本数 / Number of test samples: {zsl_results['num_samples']}")
    print(f"  Top-1 准确率 / Top-1 Accuracy: {zsl_results['accuracy']:.4f}")
    print(f"  Top-3 准确率 / Top-3 Accuracy: {zsl_results['top3_accuracy']:.4f}")
    print(f"  随机基线 / Random baseline: {zsl_results['random_baseline']:.4f}")

    # 列出事件描述 / List event descriptions
    print("\n  事件描述 / Event descriptions:")
    for i, desc in enumerate(EVENT_DESCRIPTIONS):
        print(f"    [{i}] {desc}")

    # ================================================================
    # 评估2: 情绪评分与收益相关性 / Eval 2: Sentiment-Return Correlation
    # ================================================================
    print("\n" + "-" * 60)
    print("2. 情绪评分与收益相关性 / Sentiment-Return Correlation")
    print("-" * 60)

    prototype_ids = build_prototype_token_ids(
        SENTIMENT_PROTOTYPES, args.vocab_size, args.seq_len
    )

    sentiment_results = evaluate_sentiment_correlation(
        model=model,
        test_text_ids=test_data["text_ids"],
        test_returns=test_data["returns"],
        prototype_text_ids=prototype_ids,
        device=device,
    )

    print(f"  综合情绪与收益相关性 / Composite-Return Correlation: {sentiment_results['composite_corr']:.4f}")
    print(f"  看多情绪与收益相关性 / Bullish-Return Correlation:   {sentiment_results['bullish_corr']:.4f}")
    print(f"  看空情绪与收益相关性 / Bearish-Return Correlation:   {sentiment_results['bearish_corr']:.4f}")
    print(f"  不确定性情绪相关性 / Uncertainty-Return Correlation: {sentiment_results['uncertain_corr']:.4f}")
    print(f"  中性情绪相关性 / Neutral-Return Correlation:         {sentiment_results['neutral_corr']:.4f}")
    print(f"  综合Rank IC / Composite Rank IC:                     {sentiment_results['composite_rank_ic']:.4f}")
    print(f"  综合情绪均值 / Mean Composite Sentiment:             {sentiment_results['mean_composite_sentiment']:.4f}")
    print(f"  综合情绪标准差 / Std Composite Sentiment:            {sentiment_results['std_composite_sentiment']:.4f}")

    # 解读 / Interpretation
    print("\n  相关性解读 / Correlation Interpretation:")
    abs_corr = abs(sentiment_results["composite_corr"])
    if abs_corr > 0.1:
        print("    强相关 — 情绪信号可能有预测能力 / Strong — sentiment may have predictive power")
    elif abs_corr > 0.03:
        print("    弱相关 — 在大量数据中可能有统计显著性 / Weak — may be statistically significant with more data")
    else:
        print("    几乎无相关 — 随机权重下预期结果 / Near-zero — expected with random weights")

    # ================================================================
    # 评估3: 多模态Alpha信号IC / Eval 3: Multimodal Alpha Signal IC
    # ================================================================
    print("\n" + "-" * 60)
    print("3. 多模态Alpha信号IC / Multimodal Alpha Signal IC")
    print("-" * 60)

    # 等权融合 / Equal-weight fusion
    ic_results = evaluate_alpha_signal_ic(
        model=model,
        text_ids=test_data["text_ids"],
        ohlcv=test_data["ohlcv"],
        factors=test_data["factors"],
        returns=test_data["returns"],
        weights=(1.0 / 3, 1.0 / 3, 1.0 / 3),
        device=device,
    )

    print(f"  单模态IC / Single-Modality ICs:")
    print(f"    新闻IC / News IC:     {ic_results['news_ic']:.4f}")
    print(f"    K线IC / Chart IC:    {ic_results['chart_ic']:.4f}")
    print(f"    因子IC / Factor IC:   {ic_results['factor_ic']:.4f}")
    print(f"\n  多模态融合IC / Multimodal Fusion IC:  {ic_results['alpha_ic']:.4f}")
    print(f"  |IC| / Absolute IC:                    {ic_results['abs_alpha_ic']:.4f}")
    print(f"  最佳单模态|IC| / Best Single |IC|:     {ic_results['best_single_ic']:.4f}")
    print(f"  融合增益 / Fusion Gain:                {ic_results['fusion_gain']:.4f}")

    # IC解读 / IC Interpretation
    print("\n  IC解读 / IC Interpretation:")
    abs_ic = ic_results["abs_alpha_ic"]
    if abs_ic > 0.05:
        print("    |IC| > 0.05 — 有实际应用价值 / Practically useful")
    elif abs_ic > 0.03:
        print("    |IC| > 0.03 — 可能有意义 / Potentially meaningful")
    else:
        print("    |IC| < 0.03 — 随机权重下预期结果 / Expected with random weights")

    # 不同权重的IC对比 / IC comparison with different weights
    print("\n  不同融合权重的IC对比 / IC Comparison with Different Weights:")
    weight_configs = [
        (1.0, 0.0, 0.0, "纯新闻 / Pure News"),
        (0.0, 1.0, 0.0, "纯K线 / Pure Chart"),
        (0.0, 0.0, 1.0, "纯因子 / Pure Factor"),
        (0.5, 0.5, 0.0, "新闻+K线 / News+Chart"),
        (0.5, 0.0, 0.5, "新闻+因子 / News+Factor"),
        (0.0, 0.5, 0.5, "K线+因子 / Chart+Factor"),
        (1/3, 1/3, 1/3, "等权三模态 / Equal Tri-modal"),
        (0.2, 0.3, 0.5, "因子偏重 / Factor-heavy"),
    ]

    for w_news, w_chart, w_factor, label in weight_configs:
        r = evaluate_alpha_signal_ic(
            model=model,
            text_ids=test_data["text_ids"],
            ohlcv=test_data["ohlcv"],
            factors=test_data["factors"],
            returns=test_data["returns"],
            weights=(w_news, w_chart, w_factor),
            device=device,
        )
        print(f"    {label:30s}: IC = {r['alpha_ic']:+.4f}")

    # ---- 模型统计 / Model Statistics ----
    print("\n" + "-" * 60)
    print("模型统计 / Model Statistics")
    print("-" * 60)
    total = sum(p.numel() for p in model.parameters())
    print(f"  总参数量 / Total parameters: {total:,}")
    print(f"  NewsEncoder:  {sum(p.numel() for p in model.news_encoder.parameters()):,}")
    print(f"  ChartEncoder: {sum(p.numel() for p in model.chart_encoder.parameters()):,}")
    print(f"  FactorEncoder: {sum(p.numel() for p in model.factor_encoder.parameters()):,}")
    print(f"  温度 / Temperature: {1.0 / model.logit_scale.exp().item():.4f}")

    print("\n" + "=" * 60)
    print("评估完成 / Evaluation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
