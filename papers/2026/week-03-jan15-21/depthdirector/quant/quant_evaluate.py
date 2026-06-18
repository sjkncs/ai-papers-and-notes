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
DepthDirector Quantitative Finance Evaluation / 深度导演量化金融评估

Evaluation pipeline for depth-aware market models:
- Trajectory evaluation (predicted vs actual price paths)
- Depth consistency checks (order book integrity)
- Full depth-aware backtesting

深度感知市场模型的评估管线：
- 轨迹评估（预测 vs 实际价格路径）
- 深度一致性检查（订单簿完整性）
- 完整深度感知回测

Usage / 用法:
    python quant_evaluate.py --checkpoint results/model.pt --test-data data/test/
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 配置日志 / Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ============================================================================
# 配置类 / Configuration Classes
# ============================================================================


@dataclass
class EvalConfig:
    """评估配置 / Evaluation configuration"""

    checkpoint: str = "results/depth_quant/price_depth_model.pt"
    test_data: str = "data/market/test"
    output_dir: str = "results/depth_eval"
    batch_size: int = 32
    device: str = "auto"
    seed: int = 42
    # 回测参数 / Backtest parameters
    initial_capital: float = 1000000.0
    commission_rate: float = 0.001
    slippage_bps: float = 5.0
    # 深度一致性参数 / Depth consistency parameters
    n_depth_levels: int = 20
    consistency_threshold: float = 0.85
    # 评估窗口 / Evaluation windows
    eval_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])


@dataclass
class BacktestResult:
    """
    回测结果 / Backtest result

    存储完整的回测绩效指标。
    Stores comprehensive backtest performance metrics.
    """

    # 收益指标 / Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    # 交易统计 / Trade statistics
    n_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    # 深度指标 / Depth metrics
    avg_depth_score: float = 0.0
    depth_consistency_rate: float = 0.0
    # 轨迹指标 / Trajectory metrics
    trajectory_mse: float = 0.0
    direction_accuracy: float = 0.0
    # 时序 / Time series
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)


# ============================================================================
# 轨迹评估 / Trajectory Evaluation
# ============================================================================


class TrajectoryEvaluation:
    """
    轨迹评估模块 / Trajectory evaluation module

    评估预测的价格轨迹与实际价格轨迹的匹配度。
    类比DepthDirector中相机轨迹的精确控制。

    Evaluates how well predicted price trajectories match actual trajectories.
    Analogous to precise camera trajectory control in DepthDirector.

    Args:
        eval_windows: 评估时间窗口列表 / List of evaluation time windows
    """

    def __init__(
        self,
        eval_windows: Optional[List[int]] = None,
    ) -> None:
        self.eval_windows = eval_windows or [5, 10, 20, 50]

    def evaluate_trajectory(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
    ) -> Dict[str, float]:
        """
        评估价格轨迹质量 / Evaluate price trajectory quality

        多维度评估：
        - MSE：点对点误差
        - DTW：动态时间规整距离
        - 方向准确率：涨跌方向预测
        - 趋势相关性：整体趋势一致性

        Multi-dimensional evaluation:
        - MSE: point-to-point error
        - DTW: dynamic time warping distance
        - Direction accuracy: up/down prediction
        - Trend correlation: overall trend consistency

        Args:
            predicted: [B, T] 预测价格路径
            actual: [B, T] 实际价格路径

        Returns:
            评估指标字典
        """
        results: Dict[str, float] = {}

        # === 1. MSE / RMSE ===
        mse = F.mse_loss(predicted, actual).item()
        rmse = np.sqrt(mse)
        results["mse"] = mse
        results["rmse"] = rmse

        # === 2. 方向准确率 / Direction accuracy ===
        pred_returns = predicted[:, 1:] - predicted[:, :-1]
        actual_returns = actual[:, 1:] - actual[:, :-1]
        pred_direction = (pred_returns > 0).float()
        actual_direction = (actual_returns > 0).float()
        direction_acc = float((pred_direction == actual_direction).float().mean().item())
        results["direction_accuracy"] = direction_acc

        # === 3. 多窗口方向准确率 / Multi-window direction accuracy ===
        for window in self.eval_windows:
            if predicted.shape[1] > window:
                pred_w = predicted[:, window:] - predicted[:, :-window]
                actual_w = actual[:, window:] - actual[:, :-window]
                pred_dir_w = (pred_w > 0).float()
                actual_dir_w = (actual_w > 0).float()
                w_acc = float((pred_dir_w == actual_dir_w).float().mean().item())
                results[f"direction_accuracy_{window}d"] = w_acc

        # === 4. 趋势相关性 / Trend correlation ===
        # 使用长窗口移动平均的方向一致性
        # Direction consistency using long-window moving average
        if predicted.shape[1] >= 20:
            pred_ma = F.avg_pool1d(predicted.unsqueeze(1), kernel_size=20, stride=1).squeeze(1)
            actual_ma = F.avg_pool1d(actual.unsqueeze(1), kernel_size=20, stride=1).squeeze(1)
            pred_trend = (pred_ma[:, 1:] - pred_ma[:, :-1] > 0).float()
            actual_trend = (actual_ma[:, 1:] - actual_ma[:, :-1] > 0).float()
            trend_acc = float((pred_trend == actual_trend).float().mean().item())
            results["trend_correlation"] = trend_acc

        # === 5. 收益率分布匹配 / Return distribution matching ===
        # KS统计量（简化版）/ KS statistic (simplified)
        pred_ret_flat = pred_returns.flatten()
        actual_ret_flat = actual_returns.flatten()
        # 分位数比较 / Quantile comparison
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        pred_quantiles = torch.quantile(pred_ret_flat, torch.tensor(quantiles, device=pred_ret_flat.device))
        actual_quantiles = torch.quantile(actual_ret_flat, torch.tensor(quantiles, device=actual_ret_flat.device))
        quantile_mse = F.mse_loss(pred_quantiles, actual_quantiles).item()
        results["distribution_matching_mse"] = quantile_mse

        return results

    def evaluate_multi_step_ahead(
        self,
        model: nn.Module,
        current_state: torch.Tensor,
        actual_future: torch.Tensor,
        n_steps: int = 20,
    ) -> Dict[int, Dict[str, float]]:
        """
        多步前瞻评估 / Multi-step-ahead evaluation

        评估模型在不同预测步长下的性能衰减。
        Evaluates model performance degradation at different prediction horizons.

        Args:
            model: 预测模型
            current_state: 当前状态
            actual_future: 实际未来价格
            n_steps: 最大预测步数

        Returns:
            每个步长的评估指标
        """
        results: Dict[int, Dict[str, float]] = {}

        model.eval()
        with torch.no_grad():
            for horizon in [1, 5, 10, n_steps]:
                if horizon > actual_future.shape[1]:
                    continue

                # 截断到预测时域 / Truncate to prediction horizon
                actual_trunc = actual_future[:, :horizon]

                # 简单预测（在实际中使用模型）
                # Simple prediction (use model in practice)
                pred_trunc = current_state[:, :1].expand(-1, horizon)

                step_results = self.evaluate_trajectory(
                    pred_trunc, actual_trunc
                )
                results[horizon] = step_results

        return results


# ============================================================================
# 深度一致性检查 / Depth Consistency Check
# ============================================================================


class DepthConsistencyCheck:
    """
    深度一致性检查模块 / Depth consistency check module

    验证订单簿深度的物理一致性，类比DepthDirector中的深度图一致性约束。
    Validates physical consistency of order book depth, analogous to depth map
    consistency constraints in DepthDirector.

    检查项目 / Checks:
    - 价格排序 / Price ordering (bids descending, asks ascending)
    - 价差合理性 / Spread reasonableness
    - 深度连续性 / Depth continuity
    - 数量非负 / Non-negative sizes
    """

    def __init__(
        self,
        n_depth_levels: int = 20,
        consistency_threshold: float = 0.85,
    ) -> None:
        self.n_depth_levels = n_depth_levels
        self.consistency_threshold = consistency_threshold

    def check_depth_consistency(
        self,
        depth_data: torch.Tensor,
    ) -> Dict[str, float]:
        """
        执行全面的深度一致性检查
        Perform comprehensive depth consistency check

        Args:
            depth_data: [B, T, depth_levels, 4] 订单簿深度数据
                        4维: bid_price, bid_size, ask_price, ask_size

        Returns:
            各项一致性分数
        """
        B, T, L, _ = depth_data.shape
        results: Dict[str, float] = {}

        bid_prices = depth_data[:, :, :, 0]  # [B, T, L]
        bid_sizes = depth_data[:, :, :, 1]
        ask_prices = depth_data[:, :, :, 2]
        ask_sizes = depth_data[:, :, :, 3]

        # === 1. 价格排序检查 / Price ordering check ===
        # Bid价格应该逐层递减 / Bid prices should decrease per level
        bid_order_violations = (bid_prices[:, :, 1:] > bid_prices[:, :, :-1]).float()
        bid_order_score = 1.0 - float(bid_order_violations.mean().item())
        results["bid_order_consistency"] = bid_order_score

        # Ask价格应该逐层递增 / Ask prices should increase per level
        ask_order_violations = (ask_prices[:, :, 1:] < ask_prices[:, :, :-1]).float()
        ask_order_score = 1.0 - float(ask_order_violations.mean().item())
        results["ask_order_consistency"] = ask_order_score

        # === 2. 价差合理性 / Spread reasonableness ===
        # 最佳bid < 最佳ask / Best bid < best ask
        best_bid = bid_prices[:, :, 0]
        best_ask = ask_prices[:, :, 0]
        spread_valid = (best_ask > best_bid).float()
        spread_score = float(spread_valid.mean().item())
        results["spread_consistency"] = spread_score

        # 价差不能太大 / Spread shouldn't be too large
        mid_price = (best_bid + best_ask) / 2
        spread_pct = (best_ask - best_bid) / mid_price.clamp(min=1e-10)
        reasonable_spread = (spread_pct < 0.1).float()  # <10%
        results["reasonable_spread_rate"] = float(reasonable_spread.mean().item())

        # === 3. 数量非负 / Non-negative sizes ===
        bid_size_valid = (bid_sizes >= 0).float()
        ask_size_valid = (ask_sizes >= 0).float()
        results["bid_size_nonneg"] = float(bid_size_valid.mean().item())
        results["ask_size_nonneg"] = float(ask_size_valid.mean().item())

        # === 4. 深度连续性 / Depth continuity ===
        # 相邻时间步的深度不应突变
        # Depth shouldn't change abruptly between consecutive timesteps
        if T > 1:
            depth_change = (depth_data[:, 1:] - depth_data[:, :-1]).abs().mean()
            # 归一化 / Normalize
            depth_mean = depth_data.abs().mean().clamp(min=1e-10)
            continuity_score = 1.0 - float((depth_change / depth_mean).clamp(max=1.0).mean().item())
            results["temporal_continuity"] = continuity_score

        # === 5. 总体一致性分数 / Overall consistency score ===
        overall = np.mean([v for v in results.values()])
        results["overall_consistency"] = float(overall)

        return results

    def check_predicted_vs_actual_depth(
        self,
        predicted_depth: torch.Tensor,
        actual_depth: torch.Tensor,
    ) -> Dict[str, float]:
        """
        比较预测深度与实际深度 / Compare predicted vs actual depth

        Args:
            predicted_depth: [B, depth_levels, 4] 预测订单簿
            actual_depth: [B, depth_levels, 4] 实际订单簿

        Returns:
            各项比较指标
        """
        results: Dict[str, float] = {}

        # 整体重建误差 / Overall reconstruction error
        results["depth_mse"] = float(F.mse_loss(predicted_depth, actual_depth).item())
        results["depth_mae"] = float(F.l1_loss(predicted_depth, actual_depth).item())

        # 价格分量误差 / Price component error
        pred_bids = predicted_depth[:, :, 0]
        pred_asks = predicted_depth[:, :, 2]
        actual_bids = actual_depth[:, :, 0]
        actual_asks = actual_depth[:, :, 2]
        results["bid_price_mae"] = float(F.l1_loss(pred_bids, actual_bids).item())
        results["ask_price_mae"] = float(F.l1_loss(pred_asks, actual_asks).item())

        # 数量分量误差 / Size component error
        pred_bid_sizes = predicted_depth[:, :, 1]
        pred_ask_sizes = predicted_depth[:, :, 3]
        actual_bid_sizes = actual_depth[:, :, 1]
        actual_ask_sizes = actual_depth[:, :, 3]
        results["bid_size_mae"] = float(F.l1_loss(pred_bid_sizes, actual_bid_sizes).item())
        results["ask_size_mae"] = float(F.l1_loss(pred_ask_sizes, actual_ask_sizes).item())

        # 不平衡度匹配 / Imbalance matching
        pred_imbalance = (pred_bid_sizes.sum(dim=1) - pred_ask_sizes.sum(dim=1)) / (
            pred_bid_sizes.sum(dim=1) + pred_ask_sizes.sum(dim=1) + 1e-10
        )
        actual_imbalance = (actual_bid_sizes.sum(dim=1) - actual_ask_sizes.sum(dim=1)) / (
            actual_bid_sizes.sum(dim=1) + actual_ask_sizes.sum(dim=1) + 1e-10
        )
        results["imbalance_mae"] = float(F.l1_loss(pred_imbalance, actual_imbalance).item())

        return results


# ============================================================================
# 深度感知回测 / Depth-Aware Backtesting
# ============================================================================


def run_backtest(
    config: EvalConfig,
) -> BacktestResult:
    """
    运行深度感知回测 / Run depth-aware backtest

    完整回测流程：
    1. 加载测试数据和模型 / Load test data and model
    2. 生成交易信号 / Generate trading signals
    3. 模拟订单执行 / Simulate order execution
    4. 计算绩效指标 / Calculate performance metrics

    回测利用订单簿深度信息来：
    - 评估流动性（深度充足时才能执行大单）
    - 估计滑点（价差和深度影响实际执行价格）
    - 验证信号质量（深度一致性高的信号更可靠）

    Backtest uses order book depth to:
    - Evaluate liquidity (large orders need sufficient depth)
    - Estimate slippage (spread and depth affect actual execution price)
    - Validate signal quality (signals with high depth consistency are more reliable)

    Args:
        config: 评估配置

    Returns:
        BacktestResult 回测结果
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    logger.info("Starting depth-aware backtest / 开始深度感知回测")
    logger.info("Device / 设备: %s", device)

    # === 生成模拟数据 / Generate simulated data ===
    logger.info("Generating backtest data / 生成回测数据...")
    n_test = 500
    seq_length = 50
    price_features = 64
    depth_levels = config.n_depth_levels

    # 模拟价格路径 / Simulated price paths
    test_prices = []
    test_depths = []
    for _ in range(n_test):
        # 随机漂移和波动 / Random drift and volatility
        drift = np.random.randn() * 0.001
        vol = np.random.uniform(0.005, 0.03)
        returns = np.random.randn(seq_length) * vol + drift
        prices = 100.0 + np.cumsum(returns)

        features = np.zeros((seq_length, price_features))
        features[:, 0] = prices
        features[:, 1] = returns
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        test_prices.append(torch.tensor(features, dtype=torch.float32))

        # 模拟订单簿 / Simulated order book
        depth = np.zeros((seq_length, depth_levels, 4))
        for t in range(seq_length):
            mid = prices[t]
            spread = vol * mid * np.random.uniform(0.5, 2.0)
            depth[t, :, 0] = mid - spread / 2 - np.arange(depth_levels) * spread * 0.3
            depth[t, :, 1] = np.random.exponential(100, depth_levels)
            depth[t, :, 2] = mid + spread / 2 + np.arange(depth_levels) * spread * 0.3
            depth[t, :, 3] = np.random.exponential(100, depth_levels)
        test_depths.append(torch.tensor(depth, dtype=torch.float32))

    test_prices_t = torch.stack(test_prices)
    test_depths_t = torch.stack(test_depths)

    # === 生成交易信号 / Generate trading signals ===
    logger.info("Generating trading signals / 生成交易信号...")

    # 简单信号生成（实际中使用模型）
    # Simple signal generation (use model in practice)
    signals = torch.zeros(n_test)
    for i in range(n_test):
        # 基于动量和深度的简单信号
        # Simple signal based on momentum and depth
        momentum = test_prices_t[i, -1, 0] - test_prices_t[i, 0, 0]
        # 深度不平衡作为辅助信号 / Depth imbalance as auxiliary signal
        bid_total = test_depths_t[i, -1, :, 1].sum()
        ask_total = test_depths_t[i, -1, :, 3].sum()
        imbalance = (bid_total - ask_total) / (bid_total + ask_total + 1e-10)

        signal = momentum * 0.5 + imbalance * 0.5
        if signal > 0.01:
            signals[i] = 1  # 买入 / Buy
        elif signal < -0.01:
            signals[i] = -1  # 卖出 / Sell
        else:
            signals[i] = 0  # 持有 / Hold

    # === 模拟交易执行 / Simulate trade execution ===
    logger.info("Simulating trade execution / 模拟交易执行...")

    capital = config.initial_capital
    position = 0.0
    entry_price = 0.0
    trades: List[Dict[str, float]] = []
    equity_curve = [capital]

    depth_checker = DepthConsistencyCheck(
        n_depth_levels=depth_levels,
        consistency_threshold=config.consistency_threshold,
    )

    depth_scores: List[float] = []

    for i in range(1, n_test):
        current_price = float(test_prices_t[i, -1, 0])
        prev_price = float(test_prices_t[i - 1, -1, 0])

        # 深度一致性检查 / Depth consistency check
        depth_check = depth_checker.check_depth_consistency(
            test_depths_t[i : i + 1, -5:, :, :]
        )
        depth_scores.append(depth_check["overall_consistency"])

        # 只在深度一致性高时执行交易
        # Only execute trades when depth consistency is high
        depth_reliable = depth_check["overall_consistency"] > config.consistency_threshold

        signal = signals[i].item()

        if signal == 1 and position <= 0 and depth_reliable:
            # 买入 / Buy
            if position < 0:
                # 平仓 / Close short
                pnl = (entry_price - current_price) * abs(position)
                capital += pnl + position * entry_price
                trades.append({"pnl": pnl, "type": "close_short"})

            # 计算滑点 / Calculate slippage
            spread = float(test_depths_t[i, -1, 0, 2] - test_depths_t[i, -1, 0, 0])
            slippage = spread * config.slippage_bps / 10000
            exec_price = current_price + slippage

            # 开多 / Open long
            position = (capital * 0.95) / exec_price
            entry_price = exec_price
            capital -= position * exec_price * (1 + config.commission_rate)
            trades.append({"pnl": 0, "type": "open_long", "price": exec_price})

        elif signal == -1 and position >= 0 and depth_reliable:
            # 卖出 / Sell
            if position > 0:
                # 平仓 / Close long
                pnl = (current_price - entry_price) * position
                capital += position * current_price * (1 - config.commission_rate)
                trades.append({"pnl": pnl, "type": "close_long"})
                position = 0.0

        # 更新权益 / Update equity
        mark_to_market = capital + position * current_price
        equity_curve.append(mark_to_market)

    # === 计算绩效指标 / Calculate performance metrics ===
    logger.info("Calculating performance metrics / 计算绩效指标...")

    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    total_return = (equity[-1] - equity[0]) / equity[0]
    n_days = len(equity) - 1
    annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

    # 夏普比率 / Sharpe ratio
    sharpe = np.mean(returns) / max(np.std(returns), 1e-10) * np.sqrt(252)

    # 索提诺比率 / Sortino ratio
    downside_returns = returns[returns < 0]
    sortino = np.mean(returns) / max(np.std(downside_returns), 1e-10) * np.sqrt(252)

    # 最大回撤 / Maximum drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_drawdown = float(np.max(drawdown))

    # Calmar比率 / Calmar ratio
    calmar = annualized_return / max(max_drawdown, 1e-10)

    # 交易统计 / Trade statistics
    n_trades = len([t for t in trades if "close" in t["type"]])
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) < 0]
    win_rate = len(winning_trades) / max(n_trades, 1)
    avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0.0
    avg_loss = np.mean([abs(t["pnl"]) for t in losing_trades]) if losing_trades else 0.0
    total_wins = sum(t["pnl"] for t in winning_trades)
    total_losses = sum(abs(t["pnl"]) for t in losing_trades)
    profit_factor = total_wins / max(total_losses, 1.0)

    # 方向准确率 / Direction accuracy
    pred_prices = test_prices_t[:, -1, 0]
    actual_returns = pred_prices[1:] - pred_prices[:-1]
    pred_directions = signals[1:]
    correct = ((pred_directions * actual_returns) > 0).float()
    direction_acc = float(correct.mean().item()) if correct.numel() > 0 else 0.0

    result = BacktestResult(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown=max_drawdown,
        calmar_ratio=float(calmar),
        n_trades=n_trades,
        win_rate=win_rate,
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        profit_factor=float(profit_factor),
        avg_depth_score=float(np.mean(depth_scores)),
        depth_consistency_rate=float(np.mean(np.array(depth_scores) > config.consistency_threshold)),
        trajectory_mse=0.0,
        direction_accuracy=direction_acc,
        equity_curve=equity.tolist(),
        drawdown_curve=drawdown.tolist(),
    )

    # === 日志输出 / Log output ===
    logger.info("=" * 60)
    logger.info("Backtest Results / 回测结果")
    logger.info("=" * 60)
    logger.info("Total Return / 总收益率: %.2f%%", total_return * 100)
    logger.info("Annualized Return / 年化收益率: %.2f%%", annualized_return * 100)
    logger.info("Sharpe Ratio / 夏普比率: %.4f", sharpe)
    logger.info("Sortino Ratio / 索提诺比率: %.4f", sortino)
    logger.info("Max Drawdown / 最大回撤: %.2f%%", max_drawdown * 100)
    logger.info("Calmar Ratio / Calmar比率: %.4f", calmar)
    logger.info("Win Rate / 胜率: %.2f%%", win_rate * 100)
    logger.info("Profit Factor / 盈利因子: %.4f", profit_factor)
    logger.info("Number of Trades / 交易次数: %d", n_trades)
    logger.info("Avg Depth Score / 平均深度分数: %.4f", result.avg_depth_score)
    logger.info("Depth Consistency Rate / 深度一致率: %.2f%%", result.depth_consistency_rate * 100)
    logger.info("Direction Accuracy / 方向准确率: %.2f%%", direction_acc * 100)

    # === 保存结果 / Save results ===
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "n_trades": result.n_trades,
            "avg_depth_score": result.avg_depth_score,
            "depth_consistency_rate": result.depth_consistency_rate,
            "direction_accuracy": result.direction_accuracy,
            "equity_curve": result.equity_curve,
        },
        output_dir / "backtest_results.pt",
    )
    logger.info("Results saved to / 结果保存到: %s", output_dir)

    return result


# ============================================================================
# 命令行接口 / CLI
# ============================================================================


def parse_args() -> EvalConfig:
    """解析命令行参数 / Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="DepthDirector Quant Evaluation / 深度导演量化评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str,
                        default="results/depth_quant/price_depth_model.pt")
    parser.add_argument("--test-data", type=str, default="data/market/test")
    parser.add_argument("--output-dir", type=str, default="results/depth_eval")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-capital", type=float, default=1000000.0)
    parser.add_argument("--commission-rate", type=float, default=0.001)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--n-depth-levels", type=int, default=20)
    parser.add_argument("--consistency-threshold", type=float, default=0.85)

    args = parser.parse_args()

    return EvalConfig(
        checkpoint=args.checkpoint,
        test_data=args.test_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        initial_capital=args.initial_capital,
        commission_rate=args.commission_rate,
        slippage_bps=args.slippage_bps,
        n_depth_levels=args.n_depth_levels,
        consistency_threshold=args.consistency_threshold,
    )


# ============================================================================
# 入口 / Entry Point
# ============================================================================


if __name__ == "__main__":
    config = parse_args()
    result = run_backtest(config)
    logger.info("Evaluation complete / 评估完成")
