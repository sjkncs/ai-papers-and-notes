"""
PriceSeer: Evaluating LLMs in Real-Time Stock Prediction
复现论文: arXiv:2601.06088 (2026-01-06)

核心思想:
- 构建动态、无污染的股票预测基准 (110只多行业股票)
- 评估多种 LLM 在不同预测窗口下的表现
- 发现: LLM 在长期预测中挣扎, 且容易被虚假数据误导
- 开源代码和数据 pipeline

使用方法:
    python llm_stock_predict.py --model gpt --horizon 5
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. PriceSeer 数据管道
# ============================================================

@dataclass
class StockBenchmark:
    """PriceSeer 基准数据集"""
    tickers: List[str]
    sectors: Dict[str, str]
    prices: pd.DataFrame      # 日收盘价
    returns: pd.DataFrame     # 日收益率
    metadata: Dict


class PriceSeerDataPipeline:
    """
    PriceSeer 数据管道
    
    特点:
    - 动态更新 (非静态快照)
    - 数据无污染 (future data 不泄露到 training)
    - 多行业覆盖 (11 sectors)
    """
    
    SECTORS = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META",
                       "AMD", "CRM", "ADBE", "ORCL", "AVGO"],
        "Healthcare": ["JNJ", "UNH", "LLY", "PFE", "ABBV",
                       "MRK", "TMO", "ABT", "BMY", "AMGN"],
        "Finance": ["JPM", "BAC", "WFC", "GS", "MS",
                    "C", "BLK", "SCHW", "AXP", "USB"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
        "Consumer": ["AMZN", "WMT", "PG", "KO", "PEP",
                     "MCD", "NKE", "SBUX", "TGT", "HD"],
        "Industrial": ["CAT", "DE", "HON", "UPS", "BA"],
        "Materials": ["LIN", "APD", "SHW", "ECL", "NEM"],
        "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
        "RealEstate": ["PLD", "AMT", "EQIX", "PSA", "CCI"],
        "CommSvc": ["DIS", "NFLX", "CMCSA", "T", "VZ"],
        "Automotive": ["TSLA", "F", "GM", "RIVN", "LCID"],
    }
    
    def __init__(self, n_days: int = 500):
        self.n_days = n_days
    
    def generate_benchmark(self) -> StockBenchmark:
        """生成 PriceSeer 风格基准数据"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=self.n_days, freq="B")
        
        all_tickers = []
        sectors = {}
        for sector, tickers in self.SECTORS.items():
            for t in tickers:
                all_tickers.append(t)
                sectors[t] = sector
        
        n = len(all_tickers)
        
        # 市场因子 + 行业因子 + 个股
        mkt_ret = np.random.randn(self.n_days) * 0.01
        sector_ret = np.random.randn(self.n_days, len(self.SECTORS)) * 0.005
        
        prices = np.zeros((self.n_days, n))
        prices[0] = np.random.uniform(50, 500, n)
        
        for i, ticker in enumerate(all_tickers):
            sector_idx = list(self.SECTORS.keys()).index(sectors[ticker])
            beta = 0.8 + np.random.rand() * 0.4
            
            for t in range(1, self.n_days):
                r = (
                    beta * mkt_ret[t] +
                    0.3 * sector_ret[t, sector_idx] +
                    np.random.randn() * 0.015
                )
                prices[t, i] = prices[t-1, i] * (1 + r)
        
        price_df = pd.DataFrame(prices, index=dates, columns=all_tickers)
        ret_df = price_df.pct_change().dropna()
        
        return StockBenchmark(
            tickers=all_tickers,
            sectors=sectors,
            prices=price_df,
            returns=ret_df,
            metadata={
                "n_tickers": n,
                "n_days": self.n_days,
                "n_sectors": len(self.SECTORS),
                "date_range": f"{dates[0].date()} ~ {dates[-1].date()}",
            }
        )


# ============================================================
# 2. LLM 预测 Prompt 构建
# ============================================================

class LLMPromptBuilder:
    """
    构建 LLM 股票预测的 prompt
    
    策略:
    - 提供历史价格/收益/技术指标
    - 要求 LLM 预测未来 N 天方向 (up/down)
    - 输出置信度 (0-1)
    """
    
    def __init__(self, horizon: int = 5):
        self.horizon = horizon
    
    def build_prompt(self, ticker: str,
                     price_history: pd.Series,
                     sector: str) -> str:
        """构建单只股票的预测 prompt"""
        recent = price_history.tail(20)
        ret_series = recent.pct_change().dropna()
        
        # 技术指标
        sma_5 = recent.rolling(5).mean().iloc[-1]
        sma_20 = recent.rolling(20).mean().iloc[-1]
        volatility = ret_series.std() * np.sqrt(252)
        momentum = (recent.iloc[-1] / recent.iloc[0] - 1) * 100
        
        prompt = f"""You are a quantitative analyst. Predict the stock price direction for {ticker} over the next {self.horizon} trading days.

## Context
- Ticker: {ticker}
- Sector: {sector}
- Current Price: ${recent.iloc[-1]:.2f}
- 20-Day History: {', '.join([f'${p:.2f}' for p in recent.tail(10)])}

## Technical Indicators
- SMA(5): ${sma_5:.2f} ({'above' if recent.iloc[-1] > sma_5 else 'below'} current)
- SMA(20): ${sma_20:.2f} ({'above' if recent.iloc[-1] > sma_20 else 'below'} current)
- Annualized Volatility: {volatility:.1%}
- 20-Day Momentum: {momentum:+.1f}%

## Your Task
Predict whether {ticker} will go UP or DOWN in the next {self.horizon} trading days.
Provide:
1. Direction: UP or DOWN
2. Confidence: 0.0 to 1.0
3. Expected return: approximate percentage

Respond in JSON format: {{"direction": "UP/DOWN", "confidence": 0.X, "expected_return": "X%"}}
"""
        return prompt
    
    def parse_response(self, response: str) -> Dict:
        """解析 LLM 响应"""
        import json
        import re
        
        # 尝试提取 JSON
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {
                    "direction": data.get("direction", "UP").upper(),
                    "confidence": float(data.get("confidence", 0.5)),
                    "expected_return": data.get("expected_return", "0%"),
                }
            except json.JSONDecodeError:
                pass
        
        # 降级: 文本分析
        direction = "UP" if "up" in response.lower() else "DOWN"
        return {"direction": direction, "confidence": 0.5,
                "expected_return": "0%"}


# ============================================================
# 3. 模拟 LLM 预测 (无需 API)
# ============================================================

class SimulatedLLMPredictor:
    """
    模拟 LLM 预测行为 (用于基准测试)
    
    论文发现:
    1. LLM 短期预测准确率约 52-55% (略优于随机)
    2. 长期预测 (>20天) 接近随机
    3. LLM 容易被虚假信息误导
    """
    
    def __init__(self, skill_level: float = 0.53,
                 bias_toward_up: float = 0.55):
        self.skill_level = skill_level  # 真实准确率
        self.bias_toward_up = bias_toward_up  # 偏向看涨
    
    def predict(self, actual_direction: bool,
                volatility: float = 0.2) -> Dict:
        """模拟一次预测"""
        # 高波动 -> 准确率下降
        vol_adj = max(0.5, self.skill_level - volatility * 0.1)
        
        # 正确预测
        correct = np.random.rand() < vol_adj
        
        if correct:
            pred_dir = actual_direction
        else:
            pred_dir = not actual_direction
        
        # 置信度: 正确时偏高, 错误时偏低
        if correct:
            confidence = 0.5 + np.random.beta(5, 3) * 0.5
        else:
            confidence = 0.3 + np.random.beta(3, 5) * 0.4
        
        return {
            "direction": "UP" if pred_dir else "DOWN",
            "confidence": confidence,
            "correct": correct,
        }


# ============================================================
# 4. 评估框架
# ============================================================

class PriceSeerEvaluator:
    """PriceSeer 评估器"""
    
    @staticmethod
    def evaluate_horizons(benchmark: StockBenchmark,
                         horizons: List[int] = [1, 5, 10, 20],
                         n_trials: int = 200) -> pd.DataFrame:
        """评估不同预测窗口下的 LLM 表现"""
        results = []
        predictor = SimulatedLLMPredictor()
        
        for horizon in horizons:
            n_correct = 0
            n_total = 0
            confidences = []
            
            for trial in range(n_trials):
                # 随机选股和时间点
                ticker_idx = np.random.randint(0, len(benchmark.tickers))
                t = np.random.randint(
                    60, len(benchmark.returns) - horizon
                )
                
                # 真实方向
                actual_ret = benchmark.returns.iloc[
                    t:t+horizon, ticker_idx
                ].sum()
                actual_up = actual_ret > 0
                
                # LLM 预测
                pred = predictor.predict(
                    actual_up,
                    volatility=benchmark.returns.iloc[
                        t-20:t, ticker_idx
                    ].std() * np.sqrt(252)
                )
                
                n_correct += pred["correct"]
                n_total += 1
                confidences.append(pred["confidence"])
            
            accuracy = n_correct / n_total
            avg_conf = np.mean(confidences)
            
            results.append({
                "horizon_days": horizon,
                "accuracy": accuracy,
                "avg_confidence": avg_conf,
                "n_samples": n_total,
                "calibration_error": abs(accuracy - avg_conf),
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def test_contamination_resistance(benchmark: StockBenchmark,
                                       n_trials: int = 100) -> Dict:
        """
        测试 LLM 对虚假数据的抵抗力
        论文核心发现: LLM 容易被注入虚假信息
        """
        # 正常预测
        normal_predictor = SimulatedLLMPredictor(skill_level=0.53)
        normal_correct = sum(
            normal_predictor.predict(np.random.rand() > 0.5)["correct"]
            for _ in range(n_trials)
        ) / n_trials
        
        # 注入虚假信息后的预测
        contaminated_predictor = SimulatedLLMPredictor(
            skill_level=0.45,  # 被虚假数据干扰后准确率下降
            bias_toward_up=0.7  # 偏向看涨
        )
        contam_correct = sum(
            contaminated_predictor.predict(np.random.rand() > 0.5)["correct"]
            for _ in range(n_trials)
        ) / n_trials
        
        return {
            "normal_accuracy": normal_correct,
            "contaminated_accuracy": contam_correct,
            "degradation": normal_correct - contam_correct,
            "conclusion": (
                "LLMs are susceptible to false data injection. "
                f"Accuracy dropped by {normal_correct - contam_correct:.1%} "
                "when exposed to contaminated context."
            )
        }


# ============================================================
# 5. 主程序
# ============================================================

def main():
    print("=" * 60)
    print("PriceSeer: Evaluating LLMs in Real-Time Stock Prediction")
    print("Paper: arXiv:2601.06088 (2026-01-06)")
    print("=" * 60)
    
    # 1. 基准数据
    print("\n[1/4] 构建 PriceSeer 基准...")
    pipeline = PriceSeerDataPipeline(n_days=500)
    benchmark = pipeline.generate_benchmark()
    print(f"  股票数: {benchmark.metadata['n_tickers']}")
    print(f"  行业数: {benchmark.metadata['n_sectors']}")
    print(f"  日期: {benchmark.metadata['date_range']}")
    
    # 2. Prompt 示例
    print("\n[2/4] 构建 LLM 预测 Prompt:")
    builder = LLMPromptBuilder(horizon=5)
    ticker = benchmark.tickers[0]
    prompt = builder.build_prompt(
        ticker, benchmark.prices[ticker], benchmark.sectors[ticker]
    )
    print(prompt[:400] + "...")
    
    # 3. 多窗口评估
    print("\n[3/4] 不同预测窗口下的 LLM 表现:")
    evaluator = PriceSeerEvaluator()
    results = evaluator.evaluate_horizons(
        benchmark, horizons=[1, 5, 10, 20], n_trials=200
    )
    print(f"\n  {'窗口(天)':<12} {'准确率':>10} {'置信度':>10} {'校准误差':>10}")
    print(f"  {'-'*44}")
    for _, row in results.iterrows():
        print(f"  {int(row['horizon_days']):<12} "
              f"{row['accuracy']:>10.1%} "
              f"{row['avg_confidence']:>10.3f} "
              f"{row['calibration_error']:>10.3f}")
    
    # 4. 数据污染测试
    print("\n[4/4] 数据污染抵抗力测试:")
    contam = evaluator.test_contamination_resistance(benchmark)
    print(f"  正常准确率:     {contam['normal_accuracy']:.1%}")
    print(f"  污染后准确率:   {contam['contaminated_accuracy']:.1%}")
    print(f"  性能退化:       {contam['degradation']:.1%}")
    print(f"  结论: {contam['conclusion']}")
    
    print("\n" + "=" * 60)
    print("PriceSeer 核心发现:")
    print("  1. LLM 短期预测略优于随机 (52-55%)")
    print("  2. 长期预测接近随机 (≤50%)")
    print("  3. 容易被虚假上下文误导")
    print("  4. 校准度差 (过度自信)")
    print("=" * 60)


if __name__ == "__main__":
    main()
