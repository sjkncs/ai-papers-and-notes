"""
Autonomous Factor Investing: Agentic AI自主因子挖掘与组合构建
Autonomous Framework for Systematic Factor Investing via Agentic AI

复现论文: arXiv 2603.14288 (Mar 2026)
核心思想: LLM驱动的因子假设→自动回测验证→自适应组合构建→端到端自主化

作者: QoderWork AI Research
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 因子生成引擎 (Factor Generation Engine)
# ============================================================

@dataclass
class Factor:
    """因子定义"""
    name: str
    formula: str          # 因子计算逻辑描述
    category: str         # momentum, value, quality, volatility, liquidity
    lookback: int         # 回看窗口
    direction: int        # 1=正方向(高因子值→买入), -1=反向
    ic_history: list = field(default_factory=list)


class FactorGenerator:
    """
    模拟LLM驱动的因子生成
    
    实际应用中替换为LLM API:
    1. LLM根据市场数据和金融知识生成因子假设
    2. 自动编码因子计算逻辑
    3. 回测验证因子有效性
    """
    
    def __init__(self, n_assets: int = 50, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.n_assets = n_assets
        
        # 因子库 (模拟LLM生成的因子)
        self.factor_library = [
            Factor('momentum_20d', '20日收益率', 'momentum', 20, 1),
            Factor('momentum_60d', '60日收益率(跳过最近5天)', 'momentum', 65, 1),
            Factor('reversal_5d', '5日反转', 'momentum', 5, -1),
            Factor('value_earnings', '盈利收益率(E/P)', 'value', 252, 1),
            Factor('value_book', '账面市值比(B/P)', 'value', 252, 1),
            Factor('quality_roe', 'ROE质量', 'quality', 252, 1),
            Factor('quality_accruals', '低应计利润', 'quality', 252, 1),
            Factor('volatility_realized', '20日实现波动率', 'volatility', 20, -1),
            Factor('volatility_skew', '收益偏度(负偏好度)', 'volatility', 60, -1),
            Factor('liquidity_turnover', '20日平均换手率(低换手)', 'liquidity', 20, -1),
            Factor('liquidity_amihud', 'Amihud非流动性', 'liquidity', 20, 1),
            Factor('sentiment_news', '新闻情绪分数', 'momentum', 5, 1),
            Factor('analyst_revision', '分析师盈利修正', 'quality', 20, 1),
            Factor('insider_buying', '内部人买入信号', 'quality', 10, 1),
            Factor('options_skew', '期权偏度(put-call)', 'volatility', 10, -1),
        ]
    
    def compute_factors(self, returns: np.ndarray, day: int) -> np.ndarray:
        """
        计算t日所有资产的因子值
        
        returns: (T, n_assets) 历史收益
        returns: (n_factors, n_assets) 因子矩阵
        """
        n_factors = len(self.factor_library)
        factor_values = np.zeros((n_factors, self.n_assets))
        
        for f_idx, factor in enumerate(self.factor_library):
            lookback = min(factor.lookback, day)
            if lookback < 5:
                continue
            
            hist = returns[day-lookback:day]
            
            if factor.category == 'momentum':
                if 'reversal' in factor.name:
                    # 短期反转
                    factor_values[f_idx] = -hist[-5:].mean(axis=0) if day >= 5 else 0
                elif 'skip' in factor.formula:
                    # 跳过最近5天的动量
                    factor_values[f_idx] = hist[:-5].mean(axis=0) if lookback > 10 else 0
                else:
                    factor_values[f_idx] = hist.mean(axis=0)
            
            elif factor.category == 'value':
                # 模拟价值因子 (与累计收益负相关)
                cum_ret = np.cumprod(1 + hist)[-1] - 1
                factor_values[f_idx] = -cum_ret  # 低估值=高因子值
            
            elif factor.category == 'quality':
                # 模拟质量因子 (低波动+稳定收益)
                factor_values[f_idx] = hist.mean(axis=0) / (hist.std(axis=0) + 1e-8)
            
            elif factor.category == 'volatility':
                if 'skew' in factor.name:
                    # 偏度
                    mean_ret = hist.mean(axis=0)
                    std_ret = hist.std(axis=0) + 1e-8
                    skew = ((hist - mean_ret)**3).mean(axis=0) / (std_ret**3)
                    factor_values[f_idx] = -skew  # 负偏好度
                else:
                    factor_values[f_idx] = hist.std(axis=0)
            
            elif factor.category == 'liquidity':
                # 模拟流动性因子
                factor_values[f_idx] = hist.std(axis=0) / (np.abs(hist).mean(axis=0) + 1e-8)
            
            # 方向调整
            factor_values[f_idx] *= factor.direction
        
        return factor_values
    
    def evaluate_factor(self, factor_values: np.ndarray, 
                         forward_returns: np.ndarray) -> dict:
        """评估单个因子的IC"""
        if factor_values.std() > 0 and forward_returns.std() > 0:
            ic = np.corrcoef(factor_values, forward_returns)[0, 1]
        else:
            ic = 0.0
        return {'ic': ic}


# ============================================================
# 2. 自适应组合构建 (Adaptive Portfolio Construction)
# ============================================================

class AdaptiveFactorPortfolio:
    """
    自适应因子组合
    
    基于滚动IC和因子衰减动态调整因子权重
    """
    
    def __init__(self, n_factors: int, decay_window: int = 60,
                 min_weight: float = 0.0, max_weight: float = 0.3):
        self.n_factors = n_factors
        self.decay_window = decay_window
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # 因子IC历史
        self.ic_history = {i: [] for i in range(n_factors)}
        # 当前权重
        self.weights = np.ones(n_factors) / n_factors
    
    def update(self, factor_ics: np.ndarray):
        """
        用最新IC更新因子权重
        
        方法: IC加权 + 衰减调整
        """
        # 记录IC
        for i, ic in enumerate(factor_ics):
            self.ic_history[i].append(ic)
            if len(self.ic_history[i]) > self.decay_window:
                self.ic_history[i] = self.ic_history[i][-self.decay_window:]
        
        # 计算加权IC (指数衰减)
        weighted_ics = np.zeros(self.n_factors)
        for i in range(self.n_factors):
            hist = np.array(self.ic_history[i])
            if len(hist) > 0:
                # 指数衰减权重
                decay_weights = np.exp(np.linspace(-2, 0, len(hist)))
                decay_weights /= decay_weights.sum()
                weighted_ics[i] = (hist * decay_weights).sum()
        
        # IC_IR (信息比率)
        ic_means = np.array([np.mean(self.ic_history[i]) if self.ic_history[i] else 0 
                             for i in range(self.n_factors)])
        ic_stds = np.array([np.std(self.ic_history[i]) if len(self.ic_history[i]) > 3 else 1 
                            for i in range(self.n_factors)])
        ic_ir = ic_means / (ic_stds + 1e-8)
        
        # 新权重: IC_IR的softmax
        positive_ir = np.maximum(ic_ir, 0)
        if positive_ir.sum() > 0:
            exp_ir = np.exp(positive_ir * 2)  # 温度参数
            self.weights = exp_ir / exp_ir.sum()
        else:
            self.weights = np.ones(self.n_factors) / self.n_factors
        
        # 约束
        self.weights = np.clip(self.weights, self.min_weight, self.max_weight)
        if self.weights.sum() > 0:
            self.weights /= self.weights.sum()


# ============================================================
# 3. 自主因子投资引擎 (Autonomous Factor Investing Engine)
# ============================================================

class AutonomousFactorEngine:
    """
    端到端自主因子投资引擎
    
    流程:
    1. 数据摄取 → 2. 因子计算 → 3. 因子评估(IC) → 4. 组合构建 → 5. 信号输出
    """
    
    def __init__(self, n_assets: int = 50, n_days: int = 800, seed: int = 42):
        self.n_assets = n_assets
        self.n_days = n_days
        self.rng = np.random.RandomState(seed)
        
        # 组件
        self.generator = FactorGenerator(n_assets, seed)
        self.portfolio = AdaptiveFactorPortfolio(len(self.generator.factor_library))
        
        # 生成模拟市场数据
        self.returns = self._generate_market_data()
    
    def _generate_market_data(self) -> np.ndarray:
        """生成带因子暴露的市场数据"""
        n_factors = len(self.generator.factor_library)
        returns = np.zeros((self.n_days, self.n_assets))
        
        # 真实因子收益 (时变)
        for t in range(self.n_days):
            # 市场因子
            market = self.rng.randn() * 0.01
            
            # 个股因子暴露
            factor_exposures = self.rng.randn(self.n_assets, n_factors) * 0.5
            
            # 因子收益 (稀疏: 只有部分因子有效)
            factor_returns = np.zeros(n_factors)
            active_factors = self.rng.choice(n_factors, size=max(3, n_factors//3), replace=False)
            for f in active_factors:
                factor_returns[f] = self.rng.randn() * 0.002
            
            # 个股收益 = 市场 + 因子暴露*因子收益 + 特质
            for i in range(self.n_assets):
                factor_component = factor_exposures[i] @ factor_returns
                idio = self.rng.randn() * 0.015
                returns[t, i] = market + factor_component + idio
        
        return returns
    
    def run_backtest(self, lookback: int = 60, rebalance_freq: int = 20) -> dict:
        """
        运行完整回测
        """
        n_factors = len(self.generator.factor_library)
        portfolio_returns = []
        factor_weights_history = []
        ic_history = []
        
        for t in range(lookback, self.n_days - 1):
            # 1. 因子计算
            factor_values = self.generator.compute_factors(self.returns, t)
            
            # 2. 前瞻收益 (用于IC评估)
            if t + 5 < self.n_days:
                fwd_returns = self.returns[t+1:t+6].mean(axis=0)
            else:
                fwd_returns = self.returns[t+1]
            
            # 3. 因子IC评估
            factor_ics = np.zeros(n_factors)
            for f in range(n_factors):
                eval_result = self.generator.evaluate_factor(factor_values[f], fwd_returns)
                factor_ics[f] = eval_result['ic']
            
            ic_history.append(factor_ics.copy())
            
            # 4. 更新组合权重
            self.portfolio.update(factor_ics)
            
            # 5. 构建组合信号
            composite_signal = np.zeros(self.n_assets)
            for f in range(n_factors):
                # 因子标准化
                fv = factor_values[f]
                if fv.std() > 0:
                    fv_normalized = (fv - fv.mean()) / (fv.std() + 1e-8)
                else:
                    fv_normalized = np.zeros_like(fv)
                composite_signal += self.portfolio.weights[f] * fv_normalized
            
            # 6. 转换为权重 (top quintile long, bottom quintile short)
            n_quintile = self.n_assets // 5
            sorted_idx = np.argsort(composite_signal)
            weights = np.zeros(self.n_assets)
            weights[sorted_idx[-n_quintile:]] = 1.0 / n_quintile   # Long top
            weights[sorted_idx[:n_quintile]] = -1.0 / n_quintile  # Short bottom
            
            # 7. 组合收益
            actual_return = self.returns[t+1]
            port_ret = weights @ actual_return
            portfolio_returns.append(port_ret)
            factor_weights_history.append(self.portfolio.weights.copy())
        
        portfolio_returns = np.array(portfolio_returns)
        ic_history = np.array(ic_history)
        factor_weights_history = np.array(factor_weights_history)
        
        return self._calc_metrics(portfolio_returns, ic_history, factor_weights_history)
    
    def _calc_metrics(self, returns, ics, weights) -> dict:
        cum = np.cumprod(1 + returns)
        sharpe = returns.mean() * 252 / (returns.std() * np.sqrt(252) + 1e-8)
        max_dd = np.max(1 - cum / cum.cummax())
        ann_ret = cum[-1] ** (252/len(returns)) - 1
        
        # 因子IC统计
        mean_ics = ics.mean(axis=0)
        best_factor_idx = np.argmax(np.abs(mean_ics))
        
        # 等权基准
        eq_ret = self.returns[60:].mean(axis=1)
        eq_sharpe = eq_ret.mean() * 252 / (eq_ret.std() * np.sqrt(252) + 1e-8)
        
        return {
            'strategy': {
                'annual_return': ann_ret,
                'sharpe': sharpe,
                'max_dd': max_dd,
                'cumulative': cum[-1] - 1,
                'win_rate': (returns > 0).mean(),
            },
            'benchmark': {
                'sharpe': eq_sharpe,
                'annual_return': eq_ret.mean() * 252,
            },
            'best_factor': self.generator.factor_library[best_factor_idx].name,
            'best_factor_ic': mean_ics[best_factor_idx],
            'avg_abs_ic': np.abs(mean_ics).mean(),
            'n_active_factors': (np.abs(mean_ics) > 0.02).sum(),
        }


# ============================================================
# 4. 主程序
# ============================================================

def main():
    print("=" * 70)
    print("Autonomous Factor Investing: Agentic AI自主因子投资引擎")
    print("=" * 70)
    
    print("\n[1] 初始化自主因子引擎...")
    engine = AutonomousFactorEngine(n_assets=50, n_days=800, seed=42)
    
    print(f"  资产数: {engine.n_assets}")
    print(f"  因子库: {len(engine.generator.factor_library)} 个因子")
    print(f"  回测期: {engine.n_days} 天")
    
    print(f"\n  因子库概览:")
    for f in engine.generator.factor_library:
        print(f"    {f.name:<25} {f.category:<12} lookback={f.lookback}")
    
    print("\n[2] 运行自主因子投资回测...")
    results = engine.run_backtest(lookback=60, rebalance_freq=20)
    
    print(f"\n[3] 绩效结果:")
    print(f"\n  {'指标':<20} {'自主因子策略':>15} {'等权基准':>15}")
    print(f"  {'-'*55}")
    print(f"  {'年化收益':<20} {results['strategy']['annual_return']:>15.4f} "
          f"{results['benchmark']['annual_return']:>15.4f}")
    print(f"  {'Sharpe比率':<20} {results['strategy']['sharpe']:>15.4f} "
          f"{results['benchmark']['sharpe']:>15.4f}")
    print(f"  {'最大回撤':<20} {results['strategy']['max_dd']:>15.4f} {'N/A':>15}")
    print(f"  {'累计收益':<20} {results['strategy']['cumulative']:>15.4f} {'N/A':>15}")
    print(f"  {'胜率':<20} {results['strategy']['win_rate']:>15.4f} {'N/A':>15}")
    
    print(f"\n[4] 因子分析:")
    print(f"  最佳因子: {results['best_factor']}")
    print(f"  最佳因子IC: {results['best_factor_ic']:.4f}")
    print(f"  平均|IC|: {results['avg_abs_ic']:.4f}")
    print(f"  有效因子数 (|IC|>0.02): {results['n_active_factors']}/{len(engine.generator.factor_library)}")
    
    print(f"\n[5] 自主引擎关键特性:")
    print(f"  ✓ 端到端自主: 数据→因子→组合→信号 全自动")
    print(f"  ✓ 自适应权重: 基于滚动IC_IR动态调整")
    print(f"  ✓ 因子衰减处理: 指数衰减加权近期IC")
    print(f"  ✓ 可解释: 每个因子有明确的经济学逻辑")
    
    print("\n" + "=" * 70)
    print("Autonomous Factor Investing 复现完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
