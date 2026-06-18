"""
量化回测框架
轻量级向量化回测 + 绩效分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings("ignore")


@dataclass
class BacktestResult:
    """回测结果"""
    strategy_name: str
    dates: pd.DatetimeIndex
    returns: pd.Series
    positions: pd.DataFrame
    metrics: Dict = field(default_factory=dict)
    
    def summary(self) -> str:
        """生成回测摘要"""
        lines = [
            f"策略: {self.strategy_name}",
            f"区间: {self.dates[0].date()} ~ {self.dates[-1].date()}",
            f"年化收益: {self.metrics.get('annual_return', 0):.2%}",
            f"年化波动: {self.metrics.get('annual_volatility', 0):.2%}",
            f"Sharpe:   {self.metrics.get('sharpe_ratio', 0):.3f}",
            f"最大回撤: {self.metrics.get('max_drawdown', 0):.2%}",
            f"Calmar:   {self.metrics.get('calmar_ratio', 0):.3f}",
            f"胜率:     {self.metrics.get('win_rate', 0):.1%}",
            f"盈亏比:   {self.metrics.get('profit_loss_ratio', 0):.2f}",
        ]
        return "\n".join(lines)


class VectorizedBacktester:
    """
    向量化回测引擎
    
    特点:
    - 快速: 全向量化计算, 无逐日循环
    - 灵活: 支持自定义信号函数
    - 完整: 含交易成本、换手率、绩效归因
    """
    
    def __init__(self, transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        self.tc = transaction_cost
        self.slippage = slippage
    
    def run(self, prices: pd.DataFrame,
            signal_func: Callable,
            signal_params: Dict = None,
            strategy_name: str = "Strategy") -> BacktestResult:
        """
        运行回测
        
        Args:
            prices: 收盘价 DataFrame (index=date, cols=assets)
            signal_func: 信号生成函数 f(prices, **params) -> weights DataFrame
            signal_params: 信号函数参数
            strategy_name: 策略名称
        """
        if signal_params is None:
            signal_params = {}
        
        # 生成仓位信号
        weights = signal_func(prices, **signal_params)
        
        # 对齐
        common_idx = prices.index.intersection(weights.index)
        prices = prices.loc[common_idx]
        weights = weights.loc[common_idx]
        
        # 日收益
        returns = prices.pct_change().fillna(0)
        
        # 策略收益 (含交易成本)
        port_returns = (returns * weights.shift(1)).sum(axis=1)
        
        # 交易成本
        turnover = weights.diff().abs().sum(axis=1)
        cost = turnover * (self.tc + self.slippage)
        port_returns -= cost
        
        # 绩效指标
        metrics = self._calc_metrics(port_returns)
        metrics["avg_daily_turnover"] = turnover.mean()
        metrics["total_cost"] = cost.sum()
        
        return BacktestResult(
            strategy_name=strategy_name,
            dates=port_returns.index,
            returns=port_returns,
            positions=weights,
            metrics=metrics
        )
    
    def _calc_metrics(self, returns: pd.Series) -> Dict:
        """计算绩效指标"""
        ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / max(ann_vol, 1e-8)
        
        # 最大回撤
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = dd.min()
        
        # Calmar
        calmar = ann_ret / max(abs(max_dd), 1e-8)
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        # 盈亏比
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1e-8
        pl_ratio = avg_win / avg_loss
        
        # Sortino
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = ann_ret / max(downside, 1e-8)
        
        return {
            "annual_return": ann_ret,
            "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_loss_ratio": pl_ratio,
        }


# ============================================================
# 内置策略信号函数
# ============================================================

def momentum_signal(prices: pd.DataFrame,
                    lookback: int = 20,
                    top_k: int = 5) -> pd.DataFrame:
    """
    动量策略: 买入过去 N 天涨幅最大的 K 只
    """
    momentum = prices.pct_change(lookback)
    weights = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    
    for date in prices.index:
        if date in momentum.index:
            mom = momentum.loc[date].dropna()
            if len(mom) >= top_k:
                top = mom.nlargest(top_k).index
                weights.loc[date, top] = 1.0 / top_k
    
    return weights


def mean_reversion_signal(prices: pd.DataFrame,
                          lookback: int = 20,
                          z_threshold: float = 1.5) -> pd.DataFrame:
    """
    均值回归策略: 买入 z-score 低于阈值的资产
    """
    sma = prices.rolling(lookback).mean()
    std = prices.rolling(lookback).std()
    z_score = (prices - sma) / std
    
    weights = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    
    for date in prices.index:
        if date in z_score.index:
            z = z_score.loc[date].dropna()
            oversold = z[z < -z_threshold].index
            if len(oversold) > 0:
                weights.loc[date, oversold] = 1.0 / len(oversold)
    
    return weights


def risk_parity_signal(prices: pd.DataFrame,
                       lookback: int = 60) -> pd.DataFrame:
    """
    风险平价: 按波动率倒数分配权重
    """
    returns = prices.pct_change()
    vol = returns.rolling(lookback).std() * np.sqrt(252)
    
    weights = pd.DataFrame(index=prices.index, columns=prices.columns,
                           dtype=float)
    
    for date in prices.index:
        if date in vol.index:
            v = vol.loc[date].dropna()
            if len(v) > 0 and (v > 0).all():
                inv_vol = 1.0 / v
                weights.loc[date] = inv_vol / inv_vol.sum()
            else:
                weights.loc[date] = 1.0 / len(prices.columns)
    
    return weights.fillna(1.0 / len(prices.columns))
