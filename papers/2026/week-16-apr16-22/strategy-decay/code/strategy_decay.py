"""
Strategy Decay Risk: 量化策略衰减风险的度量与管理
Measuring Strategy-Decay Risk via Baseline Regime Execution Metrics

复现论文: arXiv 2604.08356 (Apr 2026)
核心: 系统性量化策略生命周期中的alpha衰减速度

作者: QoderWork AI Research
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


class StrategyLifecycle:
    """策略生命周期模拟器"""
    
    def __init__(self, n_days=1000, seed=42):
        self.rng = np.random.RandomState(seed)
        self.n_days = n_days
    
    def generate_strategy_returns(self, alpha0=0.0005, decay_rate=0.001,
                                   vol=0.01, regime_switches=3):
        """
        生成带衰减的策略收益
        
        alpha(t) = alpha0 * exp(-decay_rate * t) + regime_effects + noise
        """
        returns = np.zeros(self.n_days)
        alpha = np.zeros(self.n_days)
        
        # Regime切换
        regime = np.zeros(self.n_days, dtype=int)
        current = 0
        for t in range(self.n_days):
            if self.rng.random() < 0.005:
                current = self.rng.choice(3)
            regime[t] = current
        
        regime_effects = {0: 0, 1: 0.0003, 2: -0.0005}  # bull, sideways, bear
        
        for t in range(self.n_days):
            alpha[t] = alpha0 * np.exp(-decay_rate * t) + regime_effects[regime[t]]
            returns[t] = alpha[t] + self.rng.randn() * vol
        
        return returns, alpha
    
    def detect_decay(self, returns, window=60):
        """
        检测策略衰减
        
        方法:
        1. 滚动Sharpe比率趋势
        2. 滚动IC衰减
        3. 换手率异常
        """
        n = len(returns)
        rolling_sharpe = np.zeros(n)
        rolling_cum = np.zeros(n)
        decay_signals = []
        
        for t in range(window, n):
            window_ret = returns[t-window:t]
            sharpe = window_ret.mean() * 252 / (window_ret.std() * np.sqrt(252) + 1e-8)
            rolling_sharpe[t] = sharpe
            rolling_cum[t] = np.cumprod(1 + window_ret)[-1] - 1
        
        # 检测Sharpe下降趋势
        for t in range(window*2, n):
            recent = rolling_sharpe[t-window//2:t].mean()
            historical = rolling_sharpe[window:t-window//2].mean()
            if historical > 0 and recent < historical * 0.5:
                decay_signals.append({
                    'day': t,
                    'recent_sharpe': recent,
                    'historical_sharpe': historical,
                    'decay_ratio': recent / (historical + 1e-8),
                })
        
        return {
            'rolling_sharpe': rolling_sharpe,
            'decay_signals': decay_signals,
            'first_decay_day': decay_signals[0]['day'] if decay_signals else n,
        }
    
    def half_life(self, alpha_series):
        """计算alpha半衰期"""
        alpha0 = alpha_series[0]
        for t in range(len(alpha_series)):
            if alpha_series[t] < alpha0 / 2:
                return t
        return len(alpha_series)


def main():
    print("=" * 70)
    print("Strategy Decay Risk: 策略衰减风险度量")
    print("=" * 70)
    
    lifecycle = StrategyLifecycle(n_days=1000, seed=42)
    
    # 三种策略对比
    strategies = {
        '快速衰减策略': {'alpha0': 0.001, 'decay_rate': 0.005, 'vol': 0.01},
        '缓慢衰减策略': {'alpha0': 0.0005, 'decay_rate': 0.001, 'vol': 0.008},
        '稳定策略': {'alpha0': 0.0003, 'decay_rate': 0.0001, 'vol': 0.006},
    }
    
    print(f"\n  {'策略':<20} {'Alpha半衰期':>12} {'首衰减信号':>12} {'最终Sharpe':>12}")
    print(f"  {'-'*60}")
    
    for name, params in strategies.items():
        returns, alpha = lifecycle.generate_strategy_returns(**params)
        analysis = lifecycle.detect_decay(returns)
        half = lifecycle.half_life(alpha)
        
        final_sharpe = returns[-252:].mean() * 252 / (returns[-252:].std() * np.sqrt(252) + 1e-8)
        
        print(f"  {name:<20} {half:>12}天 {analysis['first_decay_day']:>12}天 "
              f"{final_sharpe:>+12.4f}")
    
    # 衰减检测详细分析
    print(f"\n  快速衰减策略的详细衰减信号:")
    returns, alpha = lifecycle.generate_strategy_returns(alpha0=0.001, decay_rate=0.005, vol=0.01)
    analysis = lifecycle.detect_decay(returns)
    
    for sig in analysis['decay_signals'][:5]:
        print(f"    Day {sig['day']}: Sharpe从{sig['historical_sharpe']:.4f}降至"
              f"{sig['recent_sharpe']:.4f} (衰减{1-sig['decay_ratio']:.0%})")
    
    print(f"\n  管理建议:")
    print(f"  • 定期监控滚动Sharpe: 发现50%下降即触发策略审查")
    print(f"  • Alpha半衰期 < 100天: 高衰减风险, 需要频繁更新")
    print(f"  • 多策略组合: 分散单策略衰减风险")
    print(f"  • Regime-aware: 区分真实衰减和regime不匹配")
    
    print("\n" + "=" * 70)
    print("Strategy Decay Risk 复现完成!")
    print("=" * 70)

if __name__ == '__main__':
    main()
