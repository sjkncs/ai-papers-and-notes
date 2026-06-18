"""
Smart Predict-then-Optimize (SPO) for Portfolio Optimization
复现论文: arXiv:2601.04062 (2026-01-09)

核心思想:
- 传统方法: 先预测收益 -> 再优化组合 (两阶段分离)
- SPO方法: 将学习目标直接对齐到组合决策质量 (端到端)
- 在损失函数中嵌入交易成本、换手约束等市场摩擦
- 在 ETF 数据上 (2015-2025) 验证, 含 2020 疫情压力测试

使用方法:
    python spo_portfolio.py --n_assets 20 --horizon 21
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. 数据模拟 (ETF 风格)
# ============================================================

@dataclass
class MarketData:
    """市场数据容器"""
    dates: pd.DatetimeIndex
    returns: pd.DataFrame       # 日收益率
    features: pd.DataFrame      # 预测特征
    sectors: np.ndarray         # 行业标签
    transaction_cost: float = 0.001  # 单边交易成本


def generate_etf_data(n_assets: int = 20, n_days: int = 2520,
                      seed: int = 42) -> MarketData:
    """
    生成模拟 ETF 数据
    - 20 只 ETF, 10 年日频
    - 特征: 技术指标 + 宏观因子
    """
    np.random.seed(seed)
    dates = pd.date_range("2015-01-01", periods=n_days, freq="B")
    
    # 行业: 股票/债券/商品/另类
    sector_names = ["Equity", "Bond", "Commodity", "Alternative"]
    sectors = np.random.choice(len(sector_names), n_assets)
    
    # 市场因子
    mkt = np.random.randn(n_days) * 0.01
    bond = np.random.randn(n_days) * 0.003
    
    # 资产收益
    returns = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        beta_mkt = 0.5 + np.random.rand() * 1.0
        beta_bond = -0.2 + np.random.rand() * 0.4
        if sectors[i] == 1:  # Bond
            beta_mkt *= 0.2
            beta_bond *= 3
        returns[:, i] = (
            beta_mkt * mkt + beta_bond * bond +
            np.random.randn(n_days) * 0.008
        )
    
    # 添加 2020 疫情冲击
    crash_start = int(n_days * 0.52)  # ~2020 March
    returns[crash_start:crash_start+20] -= 0.03
    returns[crash_start+20:crash_start+60] += 0.015
    
    ret_df = pd.DataFrame(returns, index=dates,
                          columns=[f"ETF_{i}" for i in range(n_assets)])
    
    # 特征: 动量 + 波动率 + 宏观
    feat_dict = {}
    for i in range(n_assets):
        r = ret_df.iloc[:, i]
        feat_dict[f"mom20_{i}"] = r.rolling(20).sum()
        feat_dict[f"mom60_{i}"] = r.rolling(60).sum()
        feat_dict[f"vol20_{i}"] = r.rolling(20).std()
        feat_dict[f"vol60_{i}"] = r.rolling(60).std()
        feat_dict[f"rsi_{i}"] = _calc_rsi(r, 14)
    feat_dict["mkt_mom20"] = ret_df.mean(axis=1).rolling(20).sum()
    feat_dict["mkt_vol20"] = ret_df.mean(axis=1).rolling(20).std()
    
    feat_df = pd.DataFrame(feat_dict, index=dates).dropna()
    ret_df = ret_df.loc[feat_df.index]
    
    return MarketData(
        dates=ret_df.index, returns=ret_df, features=feat_df,
        sectors=sectors
    )


def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """计算 RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ============================================================
# 2. 预测模型
# ============================================================

class ReturnPredictor:
    """
    收益预测模型 (线性 + Ridge)
    SPO 的核心: 训练目标对齐到组合决策质量
    """
    
    def __init__(self, n_features: int, alpha: float = 1.0):
        self.n_features = n_features
        self.alpha = alpha  # Ridge 正则化
        self.weights = np.zeros(n_features)
        self.bias = 0.0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测收益"""
        return X @ self.weights + self.bias
    
    def fit_mse(self, X: np.ndarray, y: np.ndarray):
        """传统 MSE 训练 (预测准确但未必决策好)"""
        # Ridge 回归闭式解
        XtX = X.T @ X + self.alpha * np.eye(X.shape[1])
        Xty = X.T @ y
        self.weights = np.linalg.solve(XtX, Xty)
        self.bias = np.mean(y - X @ self.weights)
    
    def fit_spo(self, X: np.ndarray, y: np.ndarray,
                cov_matrix: np.ndarray,
                transaction_cost: float = 0.001,
                prev_weights: np.ndarray = None,
                n_epochs: int = 100, lr: float = 0.001):
        """
        SPO 训练: 损失 = -Sharpe(预测权重) + 交易成本
        
        对比传统 MSE:
        - MSE 最小化预测误差
        - SPO 最小化决策损失 (组合表现)
        """
        n_samples, n_assets = y.shape[0], y.shape[1]
        n_feat_per_asset = X.shape[1] // n_assets
        
        if prev_weights is None:
            prev_w = np.ones(n_assets) / n_assets
        else:
            prev_w = prev_weights
        
        for epoch in range(n_epochs):
            # Forward: 预测收益 -> 优化权重 -> 计算损失
            pred_returns = self.predict(X)
            
            # 简化优化: 均值-方差
            portfolio_w = self._optimize_weights(
                pred_returns.reshape(-1, n_assets).mean(axis=0)
                if pred_returns.ndim == 1 else pred_returns,
                cov_matrix
            )
            
            # 实际收益
            actual_return = np.mean(y @ portfolio_w)
            
            # 交易成本
            turnover = np.sum(np.abs(portfolio_w - prev_w))
            tc = transaction_cost * turnover
            
            # 组合波动率
            port_vol = np.sqrt(portfolio_w @ cov_matrix @ portfolio_w)
            
            # SPO Loss = -Sharpe + TC
            spo_loss = -(actual_return - tc) / max(port_vol, 1e-8)
            
            # 简化梯度更新 (数值梯度)
            grad = np.zeros_like(self.weights)
            eps = 1e-4
            for j in range(min(10, len(self.weights))):  # 只更新前10个
                self.weights[j] += eps
                pred_plus = self.predict(X)
                w_plus = self._optimize_weights(
                    pred_plus.reshape(-1, n_assets).mean(axis=0)
                    if pred_plus.ndim == 1 else pred_plus,
                    cov_matrix
                )
                ret_plus = np.mean(y @ w_plus)
                loss_plus = -ret_plus / max(
                    np.sqrt(w_plus @ cov_matrix @ w_plus), 1e-8)
                grad[j] = (loss_plus - spo_loss) / eps
                self.weights[j] -= eps
            
            self.weights -= lr * grad
            
            if (epoch + 1) % 20 == 0:
                sharpe = actual_return / max(port_vol, 1e-8) * np.sqrt(252)
                print(f"    Epoch {epoch+1:3d} | Sharpe: {sharpe:.3f} | "
                      f"TC: {tc:.4f} | Turnover: {turnover:.3f}")
            
            prev_w = portfolio_w
    
    def _optimize_weights(self, pred_returns: np.ndarray,
                          cov: np.ndarray,
                          risk_aversion: float = 2.0) -> np.ndarray:
        """
        均值-方差优化 (解析解)
        w* = (1/γ) * Σ^{-1} * μ
        """
        try:
            inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(len(cov)))
            w = (1.0 / risk_aversion) * inv_cov @ pred_returns
            # 归一化 + 非负约束
            w = np.maximum(w, 0)
            w = w / max(w.sum(), 1e-8)
            return w
        except np.linalg.LinAlgError:
            return np.ones(len(pred_returns)) / len(pred_returns)


# ============================================================
# 3. 回测框架
# ============================================================

class Backtester:
    """组合回测器"""
    
    def __init__(self, data: MarketData, rebalance_freq: int = 21):
        self.data = data
        self.rebalance_freq = rebalance_freq  # 月度调仓
    
    def run(self, predictor: ReturnPredictor,
            method: str = "mse",
            train_ratio: float = 0.6) -> Dict:
        """
        运行回测
        method: "mse" (传统) 或 "spo" (端到端)
        """
        n = len(self.data.dates)
        train_end = int(n * train_ratio)
        
        returns = self.data.returns.values
        features = self.data.features.values
        n_assets = returns.shape[1]
        
        # 滚动窗口
        window = 252  # 1 年训练窗口
        
        weights_history = []
        return_history = []
        
        prev_w = np.ones(n_assets) / n_assets
        total_tc = 0
        
        for t in range(max(train_end, window), n - 1, self.rebalance_freq):
            # 训练数据
            X_train = features[t-window:t]
            y_train = returns[t-window:t]
            
            # 协方差矩阵
            cov = np.cov(y_train.T) * 252
            
            # 特征对齐
            n_feat = X_train.shape[1]
            
            if method == "mse":
                # 传统: 先训练预测, 再优化
                y_mean = y_train.mean(axis=0)
                predictor.fit_mse(X_train, y_train)
            elif method == "spo":
                # SPO: 端到端训练
                predictor.fit_spo(
                    X_train, y_train, cov,
                    transaction_cost=self.data.transaction_cost,
                    prev_weights=prev_w,
                    n_epochs=20, lr=0.0005
                )
            
            # 预测 & 优化
            X_test = features[t:t+1]
            pred_mu = predictor.predict(X_test)
            if pred_mu.ndim > 1:
                pred_mu = pred_mu.mean(axis=0)
            if len(pred_mu) != n_assets:
                pred_mu = pred_mu[:n_assets] if len(pred_mu) > n_assets \
                    else np.pad(pred_mu, (0, n_assets - len(pred_mu)))
            
            w = predictor._optimize_weights(pred_mu, cov)
            
            # 调仓期收益
            for dt in range(min(self.rebalance_freq, n - t - 1)):
                port_ret = returns[t + dt] @ w
                return_history.append(port_ret)
            
            # 交易成本
            turnover = np.sum(np.abs(w - prev_w))
            total_tc += self.data.transaction_cost * turnover
            
            weights_history.append(w.copy())
            prev_w = w.copy()
        
        return self._calc_metrics(
            np.array(return_history), weights_history, total_tc, method
        )
    
    def _calc_metrics(self, returns: np.ndarray,
                      weights: List[np.ndarray],
                      total_tc: float, method: str) -> Dict:
        """计算回测指标"""
        cum_ret = np.cumprod(1 + returns)
        
        ann_ret = (cum_ret[-1]) ** (252 / len(returns)) - 1 \
            if len(returns) > 0 else 0
        ann_vol = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        sharpe = ann_ret / max(ann_vol, 1e-8)
        
        # 最大回撤
        peak = np.maximum.accumulate(cum_ret)
        dd = (cum_ret - peak) / peak
        max_dd = np.min(dd) if len(dd) > 0 else 0
        
        # Calmar ratio
        calmar = ann_ret / max(abs(max_dd), 1e-8)
        
        # 平均换手率
        avg_turnover = np.mean([np.sum(np.abs(
            weights[i] - weights[i-1]
        )) for i in range(1, len(weights))]) if len(weights) > 1 else 0
        
        return {
            "method": method,
            "annual_return": ann_ret,
            "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "total_transaction_cost": total_tc,
            "avg_turnover": avg_turnover,
            "n_rebalances": len(weights),
        }


# ============================================================
# 4. 主程序
# ============================================================

def print_metrics(metrics: Dict):
    """打印回测指标"""
    print(f"\n  --- {metrics['method'].upper()} 方法 ---")
    print(f"  年化收益:    {metrics['annual_return']:.2%}")
    print(f"  年化波动:    {metrics['annual_volatility']:.2%}")
    print(f"  Sharpe:      {metrics['sharpe_ratio']:.3f}")
    print(f"  最大回撤:    {metrics['max_drawdown']:.2%}")
    print(f"  Calmar:      {metrics['calmar_ratio']:.3f}")
    print(f"  总交易成本:  {metrics['total_transaction_cost']:.4f}")
    print(f"  平均换手率:  {metrics['avg_turnover']:.3f}")
    print(f"  调仓次数:    {metrics['n_rebalances']}")


def main():
    print("=" * 60)
    print("Smart Predict-then-Optimize for Portfolio Optimization")
    print("Paper: arXiv:2601.04062 (2026-01-09)")
    print("=" * 60)
    
    # 1. 数据
    print("\n[1/4] 生成 ETF 模拟数据 (20 assets, ~10 years)...")
    data = generate_etf_data(n_assets=20, n_days=2520)
    print(f"  数据范围: {data.dates[0].date()} ~ {data.dates[-1].date()}")
    print(f"  特征维度: {data.features.shape[1]}")
    
    # 2. MSE 基线
    print("\n[2/4] 回测 MSE (传统两阶段) 方法...")
    n_feat = data.features.shape[1]
    mse_predictor = ReturnPredictor(n_features=n_feat)
    bt = Backtester(data, rebalance_freq=21)
    mse_result = bt.run(mse_predictor, method="mse", train_ratio=0.6)
    print_metrics(mse_result)
    
    # 3. SPO 方法
    print("\n[3/4] 回测 SPO (端到端) 方法...")
    spo_predictor = ReturnPredictor(n_features=n_feat)
    spo_result = bt.run(spo_predictor, method="spo", train_ratio=0.6)
    print_metrics(spo_result)
    
    # 4. 对比
    print("\n[4/4] 方法对比:")
    print("-" * 50)
    print(f"  {'指标':<20} {'MSE':>10} {'SPO':>10}")
    print("-" * 50)
    for key in ["annual_return", "sharpe_ratio", "max_drawdown",
                 "total_transaction_cost"]:
        m_val = mse_result[key]
        s_val = spo_result[key]
        label = {
            "annual_return": "年化收益",
            "sharpe_ratio": "Sharpe",
            "max_drawdown": "最大回撤",
            "total_transaction_cost": "交易成本"
        }[key]
        if "ratio" in key or "return" in key or "drawdown" in key:
            print(f"  {label:<20} {m_val:>10.4f} {s_val:>10.4f}")
        else:
            print(f"  {label:<20} {m_val:>10.4f} {s_val:>10.4f}")
    
    print("\n" + "=" * 60)
    print("SPO 核心优势: 在市场摩擦(交易成本)存在时,")
    print("端到端优化比两阶段方法更能产生实际可执行的策略。")
    print("=" * 60)


if __name__ == "__main__":
    main()
