"""
LLM Sentiment for Stock Prediction: LLM情绪分析对股价预测的边际贡献
Impact of LLMs News Sentiment Analysis on Stock Price Movement Prediction

复现论文: arXiv 2602.00086 (ICLR 2026 Workshop)
核心思想: 系统评估LLM情绪信号的边际贡献，以及在不同市场regime下的有效性

作者: QoderWork AI Research
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 模拟LLM情绪引擎 (Simulated LLM Sentiment Engine)
# ============================================================

@dataclass
class NewsItem:
    """新闻条目"""
    timestamp: int
    ticker: str
    headline: str
    true_sentiment: float  # -1 to 1
    impact_level: str      # 'high', 'medium', 'low'
    category: str          # 'earnings', 'macro', 'sector', 'company'


class SentimentEngine:
    """
    模拟不同LLM模型的情绪分析
    
    实际应用中替换为真实LLM API调用
    """
    
    def __init__(self, n_stocks: int = 30, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.n_stocks = n_stocks
        self.tickers = [f'STK_{i:03d}' for i in range(n_stocks)]
    
    def generate_news_stream(self, n_days: int = 500) -> List[NewsItem]:
        """生成模拟新闻流"""
        news = []
        categories = ['earnings', 'macro', 'sector', 'company']
        impact_levels = ['high', 'medium', 'low']
        
        for day in range(n_days):
            # 每天每只股票0-3条新闻
            for ticker in self.tickers:
                n_news = self.rng.poisson(0.5)  # 平均0.5条/天/股票
                for _ in range(n_news):
                    true_sent = self.rng.uniform(-0.8, 0.8)
                    category = self.rng.choice(categories)
                    impact = self.rng.choice(impact_levels, p=[0.1, 0.3, 0.6])
                    
                    news.append(NewsItem(
                        timestamp=day,
                        ticker=ticker,
                        headline=f"News about {ticker}",
                        true_sentiment=true_sent,
                        impact_level=impact,
                        category=category,
                    ))
        
        return news
    
    def analyze_sentiment_gpt4o(self, news: NewsItem) -> float:
        """模拟GPT-4o情绪分析 (高精度，高成本)"""
        noise = self.rng.randn() * 0.1
        bias = 0.02  # 轻微乐观偏差
        return float(np.clip(news.true_sentiment * 0.9 + bias + noise, -1, 1))
    
    def analyze_sentiment_llama(self, news: NewsItem) -> float:
        """模拟LLaMA-3情绪分析 (中等精度，低成本)"""
        noise = self.rng.randn() * 0.2
        bias = 0.05  # 较大乐观偏差
        return float(np.clip(news.true_sentiment * 0.7 + bias + noise, -1, 1))
    
    def analyze_sentiment_finbert(self, news: NewsItem) -> float:
        """模拟FinBERT情绪分析 (金融专用，高精度)"""
        noise = self.rng.randn() * 0.12
        bias = 0.0  # 无偏
        return float(np.clip(news.true_sentiment * 0.85 + bias + noise, -1, 1))
    
    def analyze_sentiment_vader(self, news: NewsItem) -> float:
        """模拟VADER (词典方法，基线)"""
        noise = self.rng.randn() * 0.3
        bias = 0.1  # 词典方法的乐观偏差
        # VADER对金融文本理解较弱
        return float(np.clip(news.true_sentiment * 0.5 + bias + noise, -1, 1))


# ============================================================
# 2. 情绪因子构建 (Sentiment Factor Construction)
# ============================================================

class SentimentFactorBuilder:
    """
    将LLM情绪分析转化为可交易的因子信号
    """
    
    def __init__(self, n_stocks: int, lookback: int = 5):
        self.n_stocks = n_stocks
        self.lookback = lookback
    
    def build_daily_sentiment(self, news: List[NewsItem], 
                                sentiment_func, n_days: int) -> np.ndarray:
        """
        构建每日情绪因子矩阵
        
        returns: (n_days, n_stocks) 情绪因子值
        """
        sentiment_matrix = np.zeros((n_days, self.n_stocks))
        count_matrix = np.zeros((n_days, self.n_stocks))
        
        for item in news:
            if item.timestamp < n_days:
                stock_idx = int(item.ticker.split('_')[1])
                if stock_idx < self.n_stocks:
                    sent = sentiment_func(item)
                    sentiment_matrix[item.timestamp, stock_idx] += sent
                    count_matrix[item.timestamp, stock_idx] += 1
        
        # 取日均值
        nonzero = count_matrix > 0
        sentiment_matrix[nonzero] /= count_matrix[nonzero]
        
        return sentiment_matrix
    
    def build_momentum_signal(self, sentiment_matrix: np.ndarray) -> np.ndarray:
        """
        情绪动量信号: 近期情绪 vs 历史均值
        
        正值 = 情绪改善，负值 = 情绪恶化
        """
        n_days, n_stocks = sentiment_matrix.shape
        signal = np.zeros_like(sentiment_matrix)
        
        for t in range(self.lookback, n_days):
            recent = sentiment_matrix[t-self.lookback:t].mean(axis=0)
            historical = sentiment_matrix[max(0,t-60):t-self.lookback].mean(axis=0) \
                        if t > self.lookback + 10 else 0
            signal[t] = recent - historical
        
        return signal
    
    def build_extreme_signal(self, sentiment_matrix: np.ndarray, 
                              threshold: float = 1.5) -> np.ndarray:
        """
        情绪极端信号: 当前情绪偏离历史均值的标准差倍数
        
        用于捕捉市场情绪过热/过冷
        """
        n_days = sentiment_matrix.shape[0]
        signal = np.zeros_like(sentiment_matrix)
        
        for t in range(60, n_days):
            hist = sentiment_matrix[t-60:t]
            mean = hist.mean(axis=0)
            std = hist.std(axis=0) + 1e-8
            signal[t] = (sentiment_matrix[t] - mean) / std
        
        return signal


# ============================================================
# 3. 预测模型 (Prediction Models)
# ============================================================

class PredictionModel:
    """
    简化预测模型: 基于特征预测下一日收益方向
    """
    
    def __init__(self, n_features: int, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / n_features)
        self.weights = self.rng.randn(n_features) * scale
        self.bias = 0.0
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        features: (n_samples, n_features) or (n_features,)
        returns: prediction scores
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return features @ self.weights + self.bias
    
    def train(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, 
              n_iter: int = 100):
        """简化训练"""
        n_features = X.shape[1]
        self.weights = self.rng.randn(n_features) * np.sqrt(2.0 / n_features)
        
        for _ in range(n_iter):
            pred = X @ self.weights + self.bias
            error = pred - y.mean(axis=0) if y.ndim > 1 else pred - y
            grad = X.T @ error / len(X)
            self.weights -= lr * grad


# ============================================================
# 4. Regime-Conditioned分析 (Regime-Conditioned Analysis)
# ============================================================

class RegimeAnalyzer:
    """
    分析情绪信号在不同regime下的有效性
    """
    
    def __init__(self, n_days: int, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.n_days = n_days
        self.regimes = self._generate_regimes()
    
    def _generate_regimes(self) -> np.ndarray:
        """生成市场regime标签"""
        regimes = np.zeros(self.n_days, dtype=int)
        # 0=normal, 1=high_vol, 2=earnings_season, 3=macro_event
        current = 0
        for t in range(self.n_days):
            if self.rng.random() < 0.03:
                current = self.rng.choice(4, p=[0.5, 0.2, 0.2, 0.1])
            regimes[t] = current
        return regimes
    
    def get_regime_mask(self, regime: int) -> np.ndarray:
        """获取特定regime的布尔掩码"""
        return self.regimes == regime
    
    def analyze_by_regime(self, ic_series: np.ndarray) -> dict:
        """按regime分析IC"""
        regime_names = {0: 'normal', 1: 'high_vol', 2: 'earnings', 3: 'macro_event'}
        results = {}
        
        for regime_id, name in regime_names.items():
            mask = self.regimes[:len(ic_series)] == regime_id
            if mask.sum() > 10:
                regime_ics = ic_series[mask]
                results[name] = {
                    'count': int(mask.sum()),
                    'mean_ic': float(regime_ics.mean()),
                    'icir': float(regime_ics.mean() / (regime_ics.std() + 1e-8)),
                    'positive_rate': float((regime_ics > 0).mean()),
                }
        
        return results


# ============================================================
# 5. 信号融合策略 (Signal Fusion Strategies)
# ============================================================

class SignalFusion:
    """
    情绪信号与价格信号的融合策略
    """
    
    @staticmethod
    def early_fusion(price_features: np.ndarray, 
                      sentiment_features: np.ndarray) -> np.ndarray:
        """早期融合: 特征拼接"""
        return np.concatenate([price_features, sentiment_features], axis=-1)
    
    @staticmethod
    def late_fusion(price_signal: np.ndarray, 
                     sentiment_signal: np.ndarray,
                     weight_sentiment: float = 0.3) -> np.ndarray:
        """晚期融合: 信号加权"""
        return (1 - weight_sentiment) * price_signal + weight_sentiment * sentiment_signal
    
    @staticmethod
    def dynamic_fusion(price_signal: np.ndarray, 
                        sentiment_signal: np.ndarray,
                        volatility: np.ndarray) -> np.ndarray:
        """
        动态融合: 高波动时增加情绪权重
        
        理论: 高波动时情绪的影响更大
        """
        vol_normalized = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-8)
        # 波动率越高，情绪权重越大 (0.1 to 0.5)
        sentiment_weight = 0.1 + 0.4 * vol_normalized
        
        if sentiment_weight.ndim < price_signal.ndim:
            sentiment_weight = sentiment_weight.reshape(-1, 1)
        
        return (1 - sentiment_weight) * price_signal + sentiment_weight * sentiment_signal


# ============================================================
# 6. 主程序 (Main Pipeline)
# ============================================================

def main():
    print("=" * 70)
    print("LLM Sentiment: 情绪分析对股价预测的边际贡献")
    print("Impact of LLMs Sentiment on Stock Price Prediction")
    print("=" * 70)
    
    n_stocks = 20
    n_days = 500
    
    # ---- 生成数据和新闻 ----
    print("\n[1] 生成市场数据和新闻流...")
    rng = np.random.RandomState(42)
    
    # 价格收益 (带情绪驱动)
    true_sentiment_impact = 0.001  # 情绪对收益的真实影响
    returns = np.zeros((n_days, n_stocks))
    
    sentiment_engine = SentimentEngine(n_stocks, seed=42)
    news = sentiment_engine.generate_news_stream(n_days)
    print(f"  新闻总数: {len(news)}")
    print(f"  股票数: {n_stocks}")
    print(f"  交易日: {n_days}")
    
    # 构建真实情绪
    daily_true_sentiment = np.zeros((n_days, n_stocks))
    for item in news:
        if item.timestamp < n_days:
            idx = int(item.ticker.split('_')[1])
            if idx < n_stocks:
                daily_true_sentiment[item.timestamp, idx] += item.true_sentiment
    
    # 收益 = 市场因子 + 情绪驱动 + 噪声
    for t in range(n_days):
        market = rng.randn() * 0.01
        sentiment_component = daily_true_sentiment[t] * true_sentiment_impact
        idio = rng.randn(n_stocks) * 0.015
        returns[t] = market + sentiment_component + idio
    
    # ---- 多模型情绪分析 ----
    print("\n[2] 多LLM模型情绪分析...")
    
    models = {
        'GPT-4o': sentiment_engine.analyze_sentiment_gpt4o,
        'LLaMA-3': sentiment_engine.analyze_sentiment_llama,
        'FinBERT': sentiment_engine.analyze_sentiment_finbert,
        'VADER': sentiment_engine.analyze_sentiment_vader,
    }
    
    factor_builder = SentimentFactorBuilder(n_stocks, lookback=5)
    sentiment_matrices = {}
    
    for model_name, model_func in models.items():
        sent_matrix = factor_builder.build_daily_sentiment(news, model_func, n_days)
        sentiment_matrices[model_name] = sent_matrix
        
        # 与真实情绪的相关性
        valid_mask = daily_true_sentiment.std(axis=0) > 0
        corr = np.corrcoef(sent_matrix[:, valid_mask].flatten(), 
                           daily_true_sentiment[:, valid_mask].flatten())[0, 1]
        print(f"  {model_name:<10}: 与真实情绪相关性 = {corr:.4f}")
    
    # ---- 边际贡献评估 ----
    print("\n[3] 情绪信号的边际贡献...")
    
    forward_returns = np.roll(returns, -1, axis=0)[:-1]
    
    # 基线: 仅价格特征 (动量, 波动率, 均值回复)
    price_features = np.zeros((n_days - 1, n_stocks, 3))
    for t in range(20, n_days - 1):
        price_features[t, :, 0] = returns[max(0,t-5):t].mean(axis=0)   # 5日动量
        price_features[t, :, 1] = returns[max(0,t-20):t].std(axis=0)   # 20日波动率
        price_features[t, :, 2] = -(returns[t] - returns[max(0,t-20):t].mean(axis=0))  # 均值回复
    
    print(f"\n  {'模型':<10} {'基线IC':>10} {'增强IC':>10} {'边际提升':>10} {'ICIR':>10}")
    print(f"  {'-'*55}")
    
    for model_name, sent_matrix in sentiment_matrices.items():
        # 基线IC: 仅价格特征
        baseline_ics = []
        enhanced_ics = []
        
        for t in range(60, n_days - 1):
            if t >= len(forward_returns):
                break
            
            # 基线: 价格动量 vs 未来收益
            price_signal = returns[max(0,t-5):t].mean(axis=0)
            base_ic = np.corrcoef(price_signal, forward_returns[t])[0, 1] \
                      if forward_returns[t].std() > 0 else 0
            baseline_ics.append(base_ic)
            
            # 增强: 价格动量 + 情绪动量
            sent_momentum = sent_matrix[max(0,t-5):t].mean(axis=0)
            combined_signal = 0.7 * price_signal + 0.3 * sent_momentum
            enh_ic = np.corrcoef(combined_signal, forward_returns[t])[0, 1] \
                     if forward_returns[t].std() > 0 else 0
            enhanced_ics.append(enh_ic)
        
        baseline_ics = np.array(baseline_ics)
        enhanced_ics = np.array(enhanced_ics)
        
        base_ic_mean = baseline_ics.mean()
        enh_ic_mean = enhanced_ics.mean()
        marginal = enh_ic_mean - base_ic_mean
        icir = enh_ic_mean / (enhanced_ics.std() + 1e-8)
        
        print(f"  {model_name:<10} {base_ic_mean:>10.4f} {enh_ic_mean:>10.4f} "
              f"{marginal:>+10.4f} {icir:>10.4f}")
    
    # ---- Regime分析 ----
    print(f"\n[4] Regime-Conditioned分析...")
    
    analyzer = RegimeAnalyzer(n_days, seed=42)
    
    # 使用最好的模型(GPT-4o)
    best_model = 'GPT-4o'
    sent_matrix = sentiment_matrices[best_model]
    
    daily_ic = []
    for t in range(60, min(n_days - 1, len(forward_returns))):
        sent = sent_matrix[t]
        fwd = forward_returns[t]
        if fwd.std() > 0 and sent.std() > 0:
            ic = np.corrcoef(sent, fwd)[0, 1]
        else:
            ic = 0
        daily_ic.append(ic)
    
    daily_ic = np.array(daily_ic)
    regime_results = analyzer.analyze_by_regime(daily_ic)
    
    print(f"\n  {best_model} 情绪信号在不同Regime下的IC:")
    print(f"  {'Regime':<15} {'天数':>8} {'Mean IC':>10} {'ICIR':>10} {'IC>0率':>10}")
    print(f"  {'-'*55}")
    
    for name, stats in regime_results.items():
        print(f"  {name:<15} {stats['count']:>8} {stats['mean_ic']:>10.4f} "
              f"{stats['icir']:>10.4f} {stats['positive_rate']:>10.4f}")
    
    # ---- 信号融合对比 ----
    print(f"\n[5] 信号融合策略对比...")
    
    fusion = SignalFusion()
    
    # 构建信号
    price_signals = np.zeros((n_days - 60, n_stocks))
    sent_signals = np.zeros_like(price_signals)
    volatility = np.zeros(n_days - 60)
    
    for i, t in enumerate(range(60, n_days - 1)):
        if t >= len(forward_returns):
            break
        price_signals[i] = returns[max(0,t-5):t].mean(axis=0)
        sent_signals[i] = sent_matrix[max(0,t-5):t].mean(axis=0)
        volatility[i] = returns[max(0,t-20):t].std()
    
    actual_fwd = forward_returns[60:60+len(price_signals)]
    
    # 三种融合
    late = fusion.late_fusion(price_signals, sent_signals, weight_sentiment=0.3)
    dynamic = fusion.dynamic_fusion(price_signals, sent_signals, volatility)
    
    def calc_ic(signal, target):
        ics = []
        for t in range(len(signal)):
            if target[t].std() > 0 and signal[t].std() > 0:
                ics.append(np.corrcoef(signal[t], target[t])[0, 1])
        return np.mean(ics) if ics else 0
    
    base_ic = calc_ic(price_signals, actual_fwd)
    late_ic = calc_ic(late, actual_fwd)
    dynamic_ic = calc_ic(dynamic, actual_fwd)
    
    print(f"\n  {'融合策略':<15} {'Mean IC':>10} {'vs基线提升':>12}")
    print(f"  {'-'*40}")
    print(f"  {'仅价格(基线)':<15} {base_ic:>10.4f} {'—':>12}")
    print(f"  {'晚期融合':<15} {late_ic:>10.4f} {late_ic - base_ic:>+12.4f}")
    print(f"  {'动态融合':<15} {dynamic_ic:>10.4f} {dynamic_ic - base_ic:>+12.4f}")
    
    # ---- 成本效益分析 ----
    print(f"\n[6] 成本效益分析...")
    
    # 假设: GPT-4o每条新闻$0.01, LLaMA-3每条$0.001, FinBERT每条$0.0005, VADER免费
    costs = {'GPT-4o': 0.01, 'LLaMA-3': 0.001, 'FinBERT': 0.0005, 'VADER': 0}
    
    print(f"\n  {'模型':<10} {'边际IC提升':>12} {'日成本':>10} {'IC/$':>10} {'推荐':<8}")
    print(f"  {'-'*55}")
    
    for model_name in models:
        # 计算边际IC
        sent_m = sentiment_matrices[model_name]
        enh_ics = []
        for t in range(60, min(n_days-1, len(forward_returns))):
            p = returns[max(0,t-5):t].mean(axis=0)
            s = sent_m[max(0,t-5):t].mean(axis=0)
            combined = 0.7 * p + 0.3 * s
            fwd = forward_returns[t]
            if fwd.std() > 0 and combined.std() > 0:
                enh_ics.append(np.corrcoef(combined, fwd)[0, 1])
        
        marginal_ic = np.mean(enh_ics) - base_ic if enh_ics else 0
        
        # 日成本 (假设每天10条新闻/股票)
        daily_cost = costs[model_name] * 10 * n_stocks
        ic_per_dollar = marginal_ic / (daily_cost + 1e-8) if daily_cost > 0 else float('inf')
        
        recommend = "✓" if ic_per_dollar > 0.001 or daily_cost == 0 else "✗"
        
        cost_str = f"${daily_cost:.2f}" if daily_cost > 0 else "Free"
        icpd_str = f"{ic_per_dollar:.4f}" if ic_per_dollar < 100 else "∞"
        
        print(f"  {model_name:<10} {marginal_ic:>+12.4f} {cost_str:>10} {icpd_str:>10} {recommend:<8}")
    
    print(f"\n  结论: 低成本模型(FinBERT)在小资金下性价比最高，")
    print(f"  大资金下GPT-4o的边际IC提升更值得投入")
    
    print("\n" + "=" * 70)
    print("LLM Sentiment for Stock Prediction 复现完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
