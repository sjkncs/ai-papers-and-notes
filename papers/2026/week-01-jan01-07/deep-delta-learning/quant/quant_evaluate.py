#!/usr/bin/env python3
"""
Deep Delta Learning x 量化金融 — 回测评估脚本
DDL for Quantitative Finance — Backtest Evaluation Script
===========================================================

基于论文 "Deep Delta Learning" (arXiv:2601.00417) 的量化金融回测评估。
Backtest evaluation for DDL quantitative finance applications.

评估指标 / Evaluation Metrics:
  1. Sharpe Ratio (年化 / Annualized)
  2. Maximum Drawdown (最大回撤)
  3. Regime Detection Accuracy (状态检测准确率)
  4. Win Rate (胜率)
  5. Calmar Ratio (年化收益/最大回撤)
  6. DDL Forecaster vs Standard Transformer Forecaster 对比

用法 / Usage:
    # 完整评估 (含训练) / Full evaluation (includes training)
    python quant/quant_evaluate.py

    # 快速演示 / Quick demo
    python quant/quant_evaluate.py --quick-demo

    # 从已保存的模型加载 / Load from saved models
    python quant/quant_evaluate.py --model-path quant_results/forecaster_model.pt

Author: Auto-generated from paper analysis
License: MIT
"""

import os
import sys
import math
import json
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 从quant_train导入模型和数据生成 / Import models and data generation from quant_train
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_train import (
    DDLTimeSeriesForecaster,
    MoEFactorRouter,
    MarketRegimeGate,
    simulate_market_data,
    train_forecaster,
)


# ============================================================
# 1. 标准Transformer预测器 (对比基线)
#    Standard Transformer Forecaster (Comparison Baseline)
# ============================================================

class StandardTransformerForecaster(nn.Module):
    """
    标准Transformer时序预测器 (使用ResNet残差连接)
    Standard Transformer Time-Series Forecaster (with ResNet residual connections)

    作为DDL Forecaster的对比基线。
    Serves as comparison baseline for DDL Forecaster.

    与DDL Forecaster的区别 / Difference from DDL Forecaster:
      - 无MarketRegimeGate (没有regime-aware重写)
      - 使用标准 x + sublayer(x) 残差连接
      - 无法主动"遗忘"过时的regime信号
    """

    def __init__(
        self,
        num_features: int = 50,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        forecast_horizon: int = 5,
        max_seq_len: int = 252,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 因子嵌入 / Factor embedding
        self.factor_proj = nn.Linear(num_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # 标准Transformer Blocks (无DDL) / Standard Transformer blocks (no DDL)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                ),
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                ),
            }))

        # 预测头 / Forecast head
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, forecast_horizon),
        )

        self.forecast_horizon = forecast_horizon
        self.max_seq_len = max_seq_len

    def forward(
        self,
        factor_data: torch.Tensor,
        regime_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            factor_data: [batch, seq_len, num_features]
            regime_labels: 未使用 (标准模型无regime感知) / Unused (no regime awareness)

        Returns:
            dict with 'returns' and 'all_hidden'
        """
        B, T, F = factor_data.shape
        x = self.factor_proj(factor_data)
        x = x + self.pos_embed[:, :T, :]

        for block in self.blocks:
            # 标准残差: x = x + sublayer(norm(x))
            # Standard residual: x = x + sublayer(norm(x))
            normed = block['norm1'](x)
            attn_out, _ = block['attn'](normed, normed, normed)
            x = x + attn_out
            x = x + block['ffn'](block['norm2'](x))

        last_hidden = x[:, -1, :]
        returns = self.forecast_head(last_hidden)

        return {
            'returns': returns,
            'all_hidden': x,
        }


def train_standard_forecaster(
    model: StandardTransformerForecaster,
    data: Dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    print_interval: int = 10,
) -> List[Dict]:
    """
    训练标准Transformer预测器 / Train standard Transformer forecaster

    与DDL forecaster使用相同的训练参数以确保公平对比。
    Uses identical training parameters as DDL forecaster for fair comparison.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num_samples = data['factors'].shape[0]
    num_batches = num_samples // batch_size

    history = []

    print(f"\n{'='*60}")
    print(f"  Training Standard Transformer Forecaster (Baseline)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_steps = 0

        perm = torch.randperm(num_samples)
        shuffled_factors = data['factors'][perm]
        shuffled_returns = data['returns'][perm]

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            factor_batch = shuffled_factors[start:end, :, 0, :].to(device)
            return_target = shuffled_returns[start:end, -5:, 0].mean(dim=-1).unsqueeze(-1).to(device)

            result = model(factor_batch)
            loss = F.mse_loss(result['returns'], return_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_steps += 1

        avg_loss = epoch_loss / max(num_steps, 1)
        history.append({"epoch": epoch + 1, "loss": avg_loss})

        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3d}/{epochs} | Loss: {avg_loss:.4f}")

    return history


# ============================================================
# 2. 回测指标计算 / Backtest Metrics Computation
# ============================================================

def compute_backtest_metrics(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    device: torch.device,
    asset_idx: int = 0,
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
) -> Dict[str, float]:
    """
    计算完整的回测指标 / Compute comprehensive backtest metrics

    策略: 预测收益 > 0 做多，否则空仓
    Strategy: Long when predicted return > 0, flat otherwise

    Args:
        model: 预测模型 (DDL或Standard) / Forecasting model (DDL or Standard)
        data: 测试数据 / Test data
        device: 设备 / Device
        asset_idx: 资产索引 / Asset index
        risk_free_rate: 无风险利率 (年化) / Risk-free rate (annualized)
        trading_days: 年交易日数 / Trading days per year

    Returns:
        dict: 完整回测指标 / Comprehensive backtest metrics
    """
    model.eval()
    num_samples = data['factors'].shape[0]

    all_predictions = []
    all_actual_returns = []
    all_regime_preds = []

    with torch.no_grad():
        batch_size = 64
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            factor_batch = data['factors'][start:end, :, asset_idx, :].to(device)
            result = model(factor_batch)

            pred = result['returns'][:, 0].cpu()
            all_predictions.append(pred)

            actual = data['returns'][start:end, -1, asset_idx].cpu()
            all_actual_returns.append(actual)

            # Regime预测 (仅DDL模型有) / Regime predictions (only DDL model has this)
            if hasattr(model, 'blocks') and 'regime_gate' in model.blocks[0]:
                regime_probs = F.softmax(result.get('regime_logits', torch.zeros(1))[:, -1, :], dim=-1)
                regime_pred = regime_probs.argmax(dim=-1).cpu()
                all_regime_preds.append(regime_pred)

    predictions = torch.cat(all_predictions)
    actuals = torch.cat(all_actual_returns)

    # 交易信号 / Trading signals
    signals = (predictions > 0).float()
    strategy_returns = signals * actuals

    # --- 基础统计 / Basic statistics ---
    total_return = strategy_returns.sum().item()
    avg_daily_return = strategy_returns.mean().item()
    std_daily_return = strategy_returns.std().item()
    num_trades = int(signals.sum().item())

    # --- Sharpe Ratio (年化 / Annualized) ---
    daily_rf = risk_free_rate / trading_days
    excess_returns = strategy_returns - daily_rf
    sharpe = (excess_returns.mean() / max(excess_returns.std(), 1e-8)) * math.sqrt(trading_days)
    sharpe = sharpe.item() if isinstance(sharpe, torch.Tensor) else sharpe

    # --- Maximum Drawdown ---
    cumulative = strategy_returns.cumsum(dim=0)
    running_max = cumulative.cummax(dim=0).values
    drawdowns = running_max - cumulative
    max_drawdown = drawdowns.max().item()

    # --- Calmar Ratio (年化收益 / 最大回撤) ---
    annual_return = avg_daily_return * trading_days
    calmar = annual_return / max(max_drawdown, 1e-8)

    # --- Win Rate ---
    trades = strategy_returns[strategy_returns != 0]
    win_rate = (trades > 0).float().mean().item() if len(trades) > 0 else 0.0

    # --- Average Win / Average Loss ---
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    avg_win = wins.mean().item() if len(wins) > 0 else 0.0
    avg_loss = losses.mean().item() if len(losses) > 0 else 0.0

    # --- Profit Factor ---
    gross_profit = wins.sum().item() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum().item()) if len(losses) > 0 else 1e-8
    profit_factor = gross_profit / gross_loss

    # --- Regime Detection Accuracy ---
    regime_accuracy = -1.0  # -1表示不适用 / -1 means not applicable
    if all_regime_preds:
        regime_preds = torch.cat(all_regime_preds)
        actual_regimes = data['regimes'][:len(regime_preds), -1]
        regime_accuracy = (regime_preds == actual_regimes).float().mean().item()

    return {
        # 收益指标 / Return metrics
        "total_return": total_return,
        "avg_daily_return": avg_daily_return,
        "annualized_return": annual_return,
        # 风险指标 / Risk metrics
        "std_daily_return": std_daily_return,
        "max_drawdown": max_drawdown,
        # 风险调整指标 / Risk-adjusted metrics
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        # 交易统计 / Trade statistics
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "num_trades": num_trades,
        # Regime指标 / Regime metrics
        "regime_accuracy": regime_accuracy,
        # 预测统计 / Prediction statistics
        "avg_prediction": predictions.mean().item(),
        "prediction_std": predictions.std().item(),
    }


# ============================================================
# 3. 对比评估 / Comparison Evaluation
# ============================================================

def run_full_comparison(
    data: Dict[str, torch.Tensor],
    device: torch.device,
    num_features: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    forecast_horizon: int,
    max_seq_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    num_assets: int = 5,
) -> Dict[str, Dict]:
    """
    运行完整的DDL vs Standard对比评估
    Run full DDL vs Standard comparison evaluation

    两个模型在相同数据上训练，对比回测指标。
    Both models train on same data, compared on backtest metrics.

    Args:
        data: 模拟数据 / Simulated data
        device: 设备 / Device
        其他模型和训练参数 / Other model and training parameters
        num_assets: 回测的资产数量 / Number of assets for backtesting

    Returns:
        dict: 对比结果 / Comparison results
    """
    results = {"ddl": {}, "standard": {}}

    # --- 训练DDL Forecaster ---
    ddl_model = DDLTimeSeriesForecaster(
        num_features=num_features, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, forecast_horizon=forecast_horizon,
        max_seq_len=max_seq_len,
    ).to(device)

    print("\n" + "=" * 70)
    print("  Phase 1: Training DDL Forecaster / 训练DDL预测器")
    print("=" * 70)
    ddl_history = train_forecaster(
        ddl_model, data, device,
        epochs=epochs, batch_size=batch_size, lr=lr,
    )

    # --- 训练Standard Forecaster ---
    std_model = StandardTransformerForecaster(
        num_features=num_features, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, forecast_horizon=forecast_horizon,
        max_seq_len=max_seq_len,
    ).to(device)

    print("\n" + "=" * 70)
    print("  Phase 2: Training Standard Forecaster / 训练标准预测器")
    print("=" * 70)
    std_history = train_standard_forecaster(
        std_model, data, device,
        epochs=epochs, batch_size=batch_size, lr=lr,
    )

    # --- 回测对比 / Backtest comparison ---
    print("\n" + "=" * 70)
    print("  Phase 3: Backtest Evaluation / 回测评估")
    print("=" * 70)

    # 对多个资产进行回测 / Backtest across multiple assets
    ddl_metrics_all = []
    std_metrics_all = []

    for asset_idx in range(min(num_assets, data['factors'].shape[2])):
        ddl_metrics = compute_backtest_metrics(ddl_model, data, device, asset_idx=asset_idx)
        std_metrics = compute_backtest_metrics(std_model, data, device, asset_idx=asset_idx)
        ddl_metrics_all.append(ddl_metrics)
        std_metrics_all.append(std_metrics)

    # 聚合多资产结果 / Aggregate multi-asset results
    results["ddl"]["per_asset"] = ddl_metrics_all
    results["standard"]["per_asset"] = std_metrics_all

    # 平均指标 / Average metrics
    ddl_avg = {}
    std_avg = {}
    for key in ddl_metrics_all[0]:
        ddl_vals = [m[key] for m in ddl_metrics_all if m[key] != -1.0]
        std_vals = [m[key] for m in std_metrics_all if m[key] != -1.0]
        ddl_avg[key] = sum(ddl_vals) / len(ddl_vals) if ddl_vals else -1.0
        std_avg[key] = sum(std_vals) / len(std_vals) if std_vals else -1.0

    results["ddl"]["average"] = ddl_avg
    results["standard"]["average"] = std_avg
    results["ddl"]["training_history"] = ddl_history
    results["standard"]["training_history"] = std_history

    # 参数对比 / Parameter comparison
    results["ddl"]["params"] = sum(p.numel() for p in ddl_model.parameters())
    results["standard"]["params"] = sum(p.numel() for p in std_model.parameters())

    return results


def print_comparison_report(results: Dict[str, Dict]):
    """
    打印完整的对比报告 / Print comprehensive comparison report

    包含训练收敛性、回测指标、regime检测的完整对比表格。
    Includes training convergence, backtest metrics, and regime detection tables.
    """
    ddl_avg = results["ddl"]["average"]
    std_avg = results["standard"]["average"]
    ddl_params = results["ddl"]["params"]
    std_params = results["standard"]["params"]

    print("\n")
    print("=" * 80)
    print("  DDL vs Standard Transformer — 完整对比报告")
    print("  DDL vs Standard Transformer — Full Comparison Report")
    print("=" * 80)

    # --- 模型参数对比 / Model parameter comparison ---
    print(f"\n  --- 模型参数 / Model Parameters ---")
    print(f"  {'Model':<20s} {'Parameters':>12s} {'Overhead':>10s}")
    print(f"  {'-'*45}")
    print(f"  {'DDL Forecaster':<20s} {ddl_params:>12,d} {(ddl_params-std_params)/std_params:>9.2%}")
    print(f"  {'Standard Forecaster':<20s} {std_params:>12,d} {'baseline':>10s}")

    # --- 训练收敛性 / Training convergence ---
    print(f"\n  --- 训练收敛性 / Training Convergence ---")
    ddl_hist = results["ddl"]["training_history"]
    std_hist = results["standard"]["training_history"]

    ddl_final_loss = ddl_hist[-1]["prediction_loss"] if "prediction_loss" in ddl_hist[-1] else ddl_hist[-1].get("total_loss", float("inf"))
    std_final_loss = std_hist[-1]["loss"]

    print(f"  {'Model':<20s} {'Final Loss':>12s} {'Epochs':>8s}")
    print(f"  {'-'*45}")
    print(f"  {'DDL Forecaster':<20s} {ddl_final_loss:>12.4f} {len(ddl_hist):>8d}")
    print(f"  {'Standard Forecaster':<20s} {std_final_loss:>12.4f} {len(std_hist):>8d}")

    # --- 回测指标对比 / Backtest metrics comparison ---
    print(f"\n  --- 回测指标对比 / Backtest Metrics Comparison ---")
    print(f"  {'Metric':<25s} {'DDL':>12s} {'Standard':>12s} {'Winner':>10s}")
    print(f"  {'-'*62}")

    # 关键指标 / Key metrics
    key_metrics = [
        ("Sharpe Ratio", "sharpe_ratio", True),
        ("Annualized Return", "annualized_return", True),
        ("Max Drawdown", "max_drawdown", False),  # 越低越好 / Lower is better
        ("Calmar Ratio", "calmar_ratio", True),
        ("Win Rate", "win_rate", True),
        ("Profit Factor", "profit_factor", True),
        ("Avg Daily Return", "avg_daily_return", True),
        ("Std Daily Return", "std_daily_return", False),
    ]

    for name, key, higher_better in key_metrics:
        ddl_val = ddl_avg.get(key, -1.0)
        std_val = std_avg.get(key, -1.0)

        if ddl_val == -1.0 and std_val == -1.0:
            continue

        if higher_better:
            winner = "DDL" if ddl_val > std_val else "Std"
        else:
            winner = "DDL" if ddl_val < std_val else "Std"

        # 格式化输出 / Format output
        if "rate" in key or "return" in key:
            ddl_str = f"{ddl_val:>11.4%}"
            std_str = f"{std_val:>11.4%}"
        else:
            ddl_str = f"{ddl_val:>12.4f}"
            std_str = f"{std_val:>12.4f}"

        print(f"  {name:<25s} {ddl_str:>12s} {std_str:>12s} {winner:>10s}")

    print(f"  {'-'*62}")

    # --- Regime检测 / Regime detection ---
    regime_acc = ddl_avg.get("regime_accuracy", -1.0)
    if regime_acc >= 0:
        print(f"\n  --- Regime Detection / 状态检测 ---")
        print(f"  DDL Regime Accuracy: {regime_acc:.2%}")
        print(f"  Standard: N/A (no regime awareness)")

    # --- 每资产详细结果 / Per-asset detailed results ---
    print(f"\n  --- Per-Asset Sharpe Ratios / 每资产Sharpe比率 ---")
    ddl_per_asset = results["ddl"]["per_asset"]
    std_per_asset = results["standard"]["per_asset"]

    print(f"  {'Asset':>6s} | {'DDL Sharpe':>12s} | {'Std Sharpe':>12s} | "
          f"{'DDL Return':>12s} | {'Std Return':>12s} | {'DDL MaxDD':>10s}")
    print(f"  {'-'*75}")

    for i, (ddl_m, std_m) in enumerate(zip(ddl_per_asset, std_per_asset)):
        print(f"  {i:>6d} | {ddl_m['sharpe_ratio']:>12.4f} | {std_m['sharpe_ratio']:>12.4f} | "
              f"{ddl_m['total_return']:>12.4f} | {std_m['total_return']:>12.4f} | "
              f"{ddl_m['max_drawdown']:>10.4f}")

    print(f"  {'-'*75}")

    # --- 总结 / Summary ---
    print(f"\n  --- Summary / 总结 ---")
    ddl_sharpe = ddl_avg.get("sharpe_ratio", 0)
    std_sharpe = std_avg.get("sharpe_ratio", 0)
    sharpe_improvement = (ddl_sharpe - std_sharpe) / max(abs(std_sharpe), 1e-8)

    ddl_dd = ddl_avg.get("max_drawdown", 0)
    std_dd = std_avg.get("max_drawdown", 0)
    dd_improvement = (std_dd - ddl_dd) / max(abs(std_dd), 1e-8)

    print(f"  Sharpe improvement: {sharpe_improvement:+.2%}")
    print(f"  Max drawdown improvement: {dd_improvement:+.2%}")
    if regime_acc >= 0:
        print(f"  Regime detection accuracy: {regime_acc:.2%}")
    print(f"  Parameter overhead: {(ddl_params - std_params) / std_params:.2%}")

    print("\n" + "=" * 80)


# ============================================================
# 4. 门控值分析 / Gate Value Analysis
# ============================================================

def analyze_gate_values(
    model: DDLTimeSeriesForecaster,
    data: Dict[str, torch.Tensor],
    device: torch.device,
    num_samples: int = 50,
) -> Dict:
    """
    分析DDL预测器的门控值分布 / Analyze gate value distribution of DDL forecaster

    分析内容 / Analysis:
      1. 每层门控均值 / Per-layer gate mean
      2. 不同regime下的门控差异 / Gate differences across regimes
      3. 门控与regime切换的相关性 / Correlation between gates and regime switches

    Args:
        model: DDL预测模型 / DDL forecasting model
        data: 测试数据 / Test data
        device: 设备 / Device
        num_samples: 分析样本数 / Number of samples to analyze

    Returns:
        dict: 门控分析结果 / Gate analysis results
    """
    model.eval()
    num_samples = min(num_samples, data['factors'].shape[0])

    # 收集门控值和regime信息 / Collect gate values and regime info
    layer_gate_values = [[] for _ in range(len(model.blocks))]
    regime_gate_values = {0: [], 1: [], 2: []}  # per regime

    with torch.no_grad():
        factor_batch = data['factors'][:num_samples, :, 0, :].to(device)
        regimes = data['regimes'][:num_samples]

        # 手动前向传播以获取中间门控值
        # Manual forward pass to get intermediate gate values
        x = model.factor_proj(factor_batch)
        x = x + model.pos_embed[:, :factor_batch.shape[1], :]

        for layer_idx, block in enumerate(model.blocks):
            delta, regime_logits = block['regime_gate'](x)

            # 门控值统计 / Gate value statistics
            gate_vals = block['regime_gate'].gate(x)[0] if hasattr(block['regime_gate'], 'gate') else torch.zeros_like(delta)
            # 简化: 使用delta的幅度作为门控代理
            # Simplified: use delta magnitude as gate proxy
            delta_magnitude = delta.abs().mean(dim=-1)  # [B, T]

            for sample_idx in range(min(num_samples, delta_magnitude.shape[0])):
                last_step_mag = delta_magnitude[sample_idx, -1].item()
                layer_gate_values[layer_idx].append(last_step_mag)

                # 按regime分组 / Group by regime
                last_regime = regimes[sample_idx, -1].item()
                regime_gate_values[last_regime].append(last_step_mag)

            # 继续前向传播 / Continue forward pass
            normed = block['norm1'](x)
            attn_out, _ = block['attn'](normed, normed, normed)
            x = x + delta + attn_out
            x = x + block['ffn'](block['norm2'](x))

    # 汇总分析 / Aggregate analysis
    analysis = {
        "per_layer": [],
        "per_regime": {},
    }

    print(f"\n{'='*60}")
    print(f"  DDL Gate Value Analysis / 门控值分析")
    print(f"{'='*60}")

    # 每层分析 / Per-layer analysis
    print(f"\n  {'Layer':>6s} | {'Mean':>10s} | {'Std':>10s} | {'Min':>10s} | {'Max':>10s}")
    print(f"  {'-'*55}")
    for layer_idx, vals in enumerate(layer_gate_values):
        if vals:
            arr = np.array(vals)
            stats = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
            analysis["per_layer"].append(stats)
            print(f"  {layer_idx:>6d} | {stats['mean']:>10.4f} | {stats['std']:>10.4f} | "
                  f"{stats['min']:>10.4f} | {stats['max']:>10.4f}")
    print(f"  {'-'*55}")

    # 每regime分析 / Per-regime analysis
    regime_names = ["Momentum/动量", "Mean-Reversion/均值回归", "High-Vol/高波动"]
    print(f"\n  Gate values by regime / 按regime分组的门控值:")
    for r, name in enumerate(regime_names):
        vals = regime_gate_values[r]
        if vals:
            arr = np.array(vals)
            analysis["per_regime"][name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "count": len(vals),
            }
            print(f"  {name:<25s}: mean={arr.mean():.4f}, std={arr.std():.4f}, n={len(vals)}")

    # 文本可视化 / Text visualization
    print(f"\n  Gate magnitude by layer (text bar chart):")
    for layer_idx, stats in enumerate(analysis["per_layer"]):
        bar = "#" * int(stats["mean"] * 100)
        print(f"  L{layer_idx}: [{bar:<30s}] {stats['mean']:.4f}")

    return analysis


# ============================================================
# 5. 主函数 / Main Function
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DDL Quant Backtest Evaluation / DDL量化回测评估"
    )
    parser.add_argument("--quick-demo", action="store_true",
                        help="运行快速演示 / Run quick demo")
    parser.add_argument("--model-path", type=str, default=None,
                        help="加载已保存的DDL模型 / Load saved DDL model path")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--num-assets", type=int, default=5)
    parser.add_argument("--num-factors", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="quant_eval_results")
    return parser.parse_args()


def main():
    args = parse_args()

    # 快速演示 / Quick demo mode
    if args.quick_demo:
        args.epochs = 5
        args.num_samples = 200
        args.seq_len = 30
        args.num_assets = 3
        args.num_factors = 10
        args.d_model = 32
        args.n_layers = 1
        args.batch_size = 16

    # 设置设备 / Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 设置随机种子 / Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 生成测试数据 / Generate test data
    print(f"\nGenerating simulated market data for evaluation...")
    data = simulate_market_data(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        num_assets=args.num_assets,
        num_factors=args.num_factors,
        seed=args.seed + 1000,  # 不同于训练数据的种子 / Different seed from training
    )

    # 运行完整对比 / Run full comparison
    results = run_full_comparison(
        data=data,
        device=device,
        num_features=args.num_factors,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        forecast_horizon=5,
        max_seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_assets=args.num_assets,
    )

    # 打印报告 / Print report
    print_comparison_report(results)

    # 门控值分析 / Gate value analysis
    ddl_model = DDLTimeSeriesForecaster(
        num_features=args.num_factors, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        forecast_horizon=5, max_seq_len=args.seq_len,
    ).to(device)

    gate_analysis = analyze_gate_values(ddl_model, data, device, num_samples=50)
    results["gate_analysis"] = gate_analysis

    # 保存结果 / Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # 清理不可序列化的数据 / Clean non-serializable data
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)
            return obj
        return obj

    results_clean = clean_for_json(results)

    results_path = os.path.join(args.output_dir, "backtest_results.json")
    with open(results_path, "w") as f:
        json.dump(results_clean, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print(f"\n{'='*80}")
    print(f"  Evaluation Complete / 评估完成")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
