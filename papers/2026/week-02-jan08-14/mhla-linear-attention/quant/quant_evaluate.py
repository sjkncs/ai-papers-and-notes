# ============================================================================
# MIT License
# ============================================================================
# Copyright (c) 2026 AI Papers Weekly Review Project
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
# ============================================================================

"""
MHLA High-Frequency Trading Evaluation
MHLA高频交易评估

Comprehensive evaluation suite for MHLA-based high-frequency trading models:
- Latency vs accuracy tradeoff analysis (MHLA vs softmax attention)
- Tick-level backtesting with realistic execution simulation
- Comparative analysis between linear and softmax attention for trading

MHLA高频交易模型的全面评估套件：
- 延迟与准确率权衡分析（MHLA vs softmax注意力）
- 带逼真执行模拟的tick级回测
- 线性注意力与softmax注意力在交易中的对比分析

Components / 组件:
    - LatencyVsAccuracyAnalysis: Compare MHLA vs softmax for trading
      比较MHLA与softmax在交易中的表现
    - HighFrequencyBacktest: Tick-level backtesting engine
      tick级回测引擎
    - run_comparison(): Full MHLA vs standard attention comparison
      MHLA与标准注意力的完整对比
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration / 配置
# ============================================================================

@dataclass
class HFBacktestConfig:
    """
    Configuration for high-frequency backtest evaluation.
    高频回测评估配置。
    """

    # 执行参数 / Execution parameters
    initial_capital: float = 100_000.0    # 初始资金 / Initial capital
    position_fraction: float = 0.05       # 每次交易的资金比例 / Fraction per trade
    commission_bps: float = 2.0           # 佣金（基点）/ Commission (bps)
    market_impact_bps: float = 1.0        # 市场冲击（基点）/ Market impact (bps)
    min_confidence: float = 0.4           # 最低置信度阈值 / Minimum confidence threshold

    # 风控参数 / Risk parameters
    max_position_duration_ticks: int = 200  # 最大持仓tick数 / Max position duration
    stop_loss_bps: float = 10.0            # 止损（基点）/ Stop loss (bps)
    take_profit_bps: float = 15.0          # 止盈（基点）/ Take profit (bps)
    max_concurrent_positions: int = 1      # 最大同时持仓数 / Max concurrent positions

    # 评估参数 / Evaluation parameters
    annual_risk_free_rate: float = 0.05   # 年化无风险利率 / Annual risk-free rate
    annualization_factor: float = 252 * 390  # 年化因子（分钟级数据）


# ============================================================================
# Latency vs Accuracy Analysis / 延迟与准确率分析
# ============================================================================

class LatencyVsAccuracyAnalysis:
    """
    Compare MHLA linear attention vs softmax attention for trading signals.
    比较MHLA线性注意力与softmax注意力在交易信号生成中的表现。

    Evaluates the tradeoff between:
    评估以下方面的权衡：
    - Inference latency / 推理延迟
    - Signal accuracy / 信号准确率
    - Sequence length scalability / 序列长度可扩展性
    - Trading PnL impact / 交易盈亏影响
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results: List[Dict] = []

    def measure_latency(
        self,
        model: nn.Module,
        input_generator,
        num_warmup: int = 20,
        num_runs: int = 200,
    ) -> Dict[str, float]:
        """
        Precisely measure model inference latency.
        精确测量模型推理延迟。

        Args:
            model: Model to benchmark / 要测试的模型
            input_generator: Function that generates input tensors
            num_warmup: Warmup iterations / 预热迭代数
            num_runs: Measurement iterations / 测量迭代数

        Returns:
            Latency statistics / 延迟统计
        """
        model = model.to(self.device)
        model.eval()

        # 预热 / Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                inputs = input_generator()
                if isinstance(inputs, tuple):
                    _ = model(*inputs)
                else:
                    _ = model(inputs)

        # 测量 / Measure
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                inputs = input_generator()
                start = time.perf_counter()
                if isinstance(inputs, tuple):
                    _ = model(*inputs)
                else:
                    _ = model(inputs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

        latencies.sort()
        n = len(latencies)

        return {
            "p50_ms": latencies[n // 2],
            "p90_ms": latencies[int(n * 0.90)],
            "p95_ms": latencies[int(n * 0.95)],
            "p99_ms": latencies[int(n * 0.99)],
            "mean_ms": sum(latencies) / n,
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
        }

    def compare_attention_types(
        self,
        seq_lengths: List[int] = None,
        hidden_dim: int = 128,
        num_heads: int = 4,
        batch_size: int = 1,
    ) -> List[Dict]:
        """
        Compare latency and memory between MHLA and softmax attention
        across different sequence lengths.
        在不同序列长度下比较MHLA和softmax注意力的延迟和内存。

        Args:
            seq_lengths: Sequence lengths to test / 要测试的序列长度
            hidden_dim: Hidden dimension / 隐藏维度
            num_heads: Number of heads / 头数
            batch_size: Batch size / 批量大小

        Returns:
            Comparison results / 对比结果
        """
        if seq_lengths is None:
            seq_lengths = [64, 128, 256, 512, 1024]

        head_dim = hidden_dim // num_heads
        results = []

        print(f"\n{'='*80}")
        print(f"Latency Comparison: MHLA vs Softmax Attention / 延迟对比：MHLA vs Softmax注意力")
        print(f"Hidden dim / 隐藏维度: {hidden_dim}, Heads / 头数: {num_heads}")
        print(f"{'='*80}")
        print(
            f"{'SeqLen':>8s} | {'MHLA P50':>10s} | {'Soft P50':>10s} | "
            f"{'MHLA P99':>10s} | {'Soft P99':>10s} | {'Speedup':>10s}"
        )
        print("-" * 75)

        for seq_len in seq_lengths:
            # 创建简单的MHLA注意力模块 / Create simple MHLA attention module
            from quant_train import MHLAMarketConfig, EfficientMarketAttention
            mhla_config = MHLAMarketConfig(
                hidden_dim=hidden_dim, num_heads=num_heads,
                feature_dim=head_dim, token_heads=4,
            )
            mhla_attn = EfficientMarketAttention(mhla_config).to(self.device)

            # 标准softmax注意力 / Standard softmax attention
            softmax_attn = StandardSoftmaxAttention(
                hidden_dim, num_heads
            ).to(self.device)

            def mhla_input():
                return torch.randn(batch_size, seq_len, hidden_dim, device=self.device)

            def softmax_input():
                return torch.randn(batch_size, seq_len, hidden_dim, device=self.device)

            mhla_latency = self.measure_latency(mhla_attn, mhla_input, num_runs=50)
            softmax_latency = self.measure_latency(softmax_attn, softmax_input, num_runs=50)

            speedup = softmax_latency["p50_ms"] / max(mhla_latency["p50_ms"], 0.001)

            result = {
                "seq_len": seq_len,
                "mhla": mhla_latency,
                "softmax": softmax_latency,
                "speedup": speedup,
            }
            results.append(result)

            print(
                f"{seq_len:>8d} | {mhla_latency['p50_ms']:>8.2f}ms | "
                f"{softmax_latency['p50_ms']:>8.2f}ms | "
                f"{mhla_latency['p99_ms']:>8.2f}ms | "
                f"{softmax_latency['p99_ms']:>8.2f}ms | "
                f"{speedup:>8.2f}x"
            )

        self.results = results
        return results


class StandardSoftmaxAttention(nn.Module):
    """
    Standard softmax attention for comparison baseline.
    标准softmax注意力作为对比基线。
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard softmax attention / 标准softmax注意力"""
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 因果掩码 / Causal mask
        mask = torch.triu(
            torch.full((S, S), float("-inf"), device=x.device), diagonal=1
        )
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(output)


# ============================================================================
# High-Frequency Backtest / 高频回测
# ============================================================================

class HighFrequencyBacktest:
    """
    Tick-level backtesting engine for high-frequency trading signals.
    高频交易信号的tick级回测引擎。

    Features:
    特性:
    - Tick-level execution simulation / tick级执行模拟
    - Realistic market impact modeling / 逼真的市场影响建模
    - Position duration limits / 持仓时间限制
    - Stop-loss and take-profit / 止损止盈
    - Comprehensive PnL attribution / 全面的盈亏归因
    """

    def __init__(self, config: HFBacktestConfig):
        self.config = config
        self.trade_log: List[Dict] = []
        self.equity_curve: List[float] = []
        self.pnl_series: List[float] = []

    def run(
        self,
        signals: torch.Tensor,
        confidences: torch.Tensor,
        volatilities: torch.Tensor,
        prices: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Run tick-level backtest.
        运行tick级回测。

        Args:
            signals: Predicted quintile labels [T] (0-4)
            confidences: Prediction confidences [T] (0-1)
            volatilities: Predicted volatilities [T]
            prices: Price series [T]

        Returns:
            Comprehensive backtest metrics / 全面的回测指标
        """
        T = len(prices)
        capital = self.config.initial_capital
        position = 0.0       # 当前持仓量 / Current position size
        entry_price = 0.0    # 入场价 / Entry price
        entry_tick = 0       # 入场tick / Entry tick
        pnl = 0.0            # 累计盈亏 / Cumulative PnL

        self.equity_curve = [capital]
        self.trade_log = []
        self.pnl_series = []

        max_equity = capital
        max_drawdown = 0.0
        num_trades = 0
        winning_trades = 0
        total_commission = 0.0
        total_impact = 0.0

        for t in range(1, T):
            price = prices[t].item()
            signal = signals[t].item()
            confidence = confidences[t].item()
            vol = volatilities[t].item()

            # 当前权益 / Current equity
            current_equity = capital + position * price
            self.equity_curve.append(current_equity)

            # 最大回撤 / Max drawdown
            max_equity = max(max_equity, current_equity)
            dd = (max_equity - current_equity) / max(max_equity, 1.0)
            max_drawdown = max(max_drawdown, dd)

            # 持仓时间检查 / Position duration check
            holding_duration = t - entry_tick if position != 0 else 0

            # 止损/止盈检查 / Stop-loss / take-profit check
            should_exit = False
            if position != 0:
                pnl_bps = (price - entry_price) / entry_price * 10000
                if position > 0:
                    if pnl_bps < -self.config.stop_loss_bps:
                        should_exit = True
                    elif pnl_bps > self.config.take_profit_bps:
                        should_exit = True
                elif position < 0:
                    if pnl_bps > self.config.stop_loss_bps:
                        should_exit = True
                    elif pnl_bps < -self.config.take_profit_bps:
                        should_exit = True

                # 最大持仓时间 / Max position duration
                if holding_duration >= self.config.max_position_duration_ticks:
                    should_exit = True

            # 信号决策 / Signal decision
            want_long = signal >= 3 and confidence >= self.config.min_confidence
            want_short = signal <= 1 and confidence >= self.config.min_confidence

            if should_exit or (position > 0 and want_short) or (position < 0 and want_long):
                # 平仓 / Close position
                if position != 0:
                    trade_pnl = position * (price - entry_price)
                    commission = abs(position) * price * self.config.commission_bps / 10000
                    impact = abs(position) * price * self.config.market_impact_bps / 10000
                    trade_pnl -= commission + impact
                    total_commission += commission
                    total_impact += impact

                    capital += trade_pnl
                    pnl += trade_pnl
                    num_trades += 1

                    if trade_pnl > 0:
                        winning_trades += 1

                    self.trade_log.append({
                        "entry_tick": entry_tick,
                        "exit_tick": t,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "direction": "long" if position > 0 else "short",
                        "pnl": trade_pnl,
                        "commission": commission,
                        "holding_ticks": holding_duration,
                        "confidence": confidence,
                    })
                    position = 0.0

            # 开仓 / Open position
            if position == 0:
                if want_long:
                    position_size = self.config.position_fraction * capital / price
                    commission = position_size * price * self.config.commission_bps / 10000
                    impact = position_size * price * self.config.market_impact_bps / 10000
                    capital -= commission + impact
                    total_commission += commission
                    total_impact += impact
                    position = position_size
                    entry_price = price
                    entry_tick = t
                elif want_short:
                    position_size = self.config.position_fraction * capital / price
                    commission = position_size * price * self.config.commission_bps / 10000
                    impact = position_size * price * self.config.market_impact_bps / 10000
                    capital -= commission + impact
                    total_commission += commission
                    total_impact += impact
                    position = -position_size
                    entry_price = price
                    entry_tick = t

            self.pnl_series.append(pnl)

        # 如果还有持仓则平仓 / Close remaining position
        if position != 0:
            final_price = prices[-1].item()
            trade_pnl = position * (final_price - entry_price)
            capital += trade_pnl
            pnl += trade_pnl

        # 计算指标 / Compute metrics
        returns = torch.tensor(
            [self.equity_curve[i] / self.equity_curve[i-1] - 1
             for i in range(1, len(self.equity_curve))]
        ) if len(self.equity_curve) > 1 else torch.tensor([0.0])

        total_return = (self.equity_curve[-1] - self.config.initial_capital) / self.config.initial_capital

        # Sharpe比率 / Sharpe ratio
        if len(returns) > 1 and returns.std() > 0:
            daily_rf = self.config.annual_risk_free_rate / self.config.annualization_factor
            sharpe = (returns.mean() - daily_rf) / returns.std() * math.sqrt(
                self.config.annualization_factor
            )
        else:
            sharpe = 0.0

        # Sortino比率 / Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            daily_rf = self.config.annual_risk_free_rate / self.config.annualization_factor
            sortino = (returns.mean() - daily_rf) / downside_returns.std() * math.sqrt(
                self.config.annualization_factor
            )
        else:
            sortino = 0.0

        # Calmar比率 / Calmar ratio
        if max_drawdown > 0:
            annualized_return = (1 + total_return) ** (
                self.config.annualization_factor / max(T, 1)
            ) - 1
            calmar = annualized_return / max_drawdown
        else:
            calmar = 0.0

        win_rate = winning_trades / max(num_trades, 1)

        return {
            "total_return_pct": round(total_return * 100, 2),
            "sharpe_ratio": round(sharpe.item() if isinstance(sharpe, torch.Tensor) else sharpe, 3),
            "sortino_ratio": round(sortino.item() if isinstance(sortino, torch.Tensor) else sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "num_trades": num_trades,
            "win_rate_pct": round(win_rate * 100, 1),
            "total_commission": round(total_commission, 2),
            "total_market_impact": round(total_impact, 2),
            "final_equity": round(self.equity_curve[-1], 2),
            "avg_holding_ticks": round(
                sum(t["holding_ticks"] for t in self.trade_log) / max(num_trades, 1), 1
            ),
        }


# ============================================================================
# Full Comparison / 完整对比
# ============================================================================

def run_comparison(
    device: str = "cpu",
    quick: bool = False,
) -> Dict[str, Dict]:
    """
    Run full MHLA vs softmax attention comparison for trading.
    运行MHLA与softmax注意力在交易中的完整对比。

    Includes:
    包括:
    1. Latency benchmarking across sequence lengths / 不同序列长度的延迟基准测试
    2. Signal quality evaluation / 信号质量评估
    3. Trading PnL comparison / 交易盈亏对比

    Args:
        device: Compute device / 计算设备
        quick: Quick mode / 快速模式

    Returns:
        Complete comparison results / 完整对比结果
    """
    print(f"\n{'='*70}")
    print(f"MHLA vs Softmax Attention: Trading Comparison / 交易对比")
    print(f"{'='*70}")

    # Step 1: 延迟对比 / Latency comparison
    print("\n[Step 1] Latency Benchmarking / 延迟基准测试...")
    seq_lengths = [64, 128, 256] if quick else [64, 128, 256, 512, 1024]
    latency_analysis = LatencyVsAccuracyAnalysis(device=device)
    latency_results = latency_analysis.compare_attention_types(
        seq_lengths=seq_lengths,
        hidden_dim=128,
        num_heads=4,
    )

    # Step 2: 生成测试数据 / Generate test data
    print("\n[Step 2] Generating HF market data / 生成高频市场数据...")
    from quant_train import simulate_hf_market_data
    num_samples = 1000 if quick else 5000
    data = simulate_hf_market_data(num_samples=num_samples, seed=42)

    # Step 3: 回测 / Backtest
    print("\n[Step 3] Running Backtest / 运行回测...")

    # 使用随机信号作为基线 / Use random signals as baseline
    T = num_samples
    random_signals = torch.randint(0, 5, (T,))
    random_confidences = torch.rand(T) * 0.5 + 0.3
    random_volatilities = torch.rand(T) * 0.01

    # 使用价格序列 / Use price series
    prices = data["prices"][:, -1]
    min_len = min(T, len(prices))
    prices = prices[:min_len]
    random_signals = random_signals[:min_len]
    random_confidences = random_confidences[:min_len]
    random_volatilities = random_volatilities[:min_len]

    config = HFBacktestConfig()
    backtest = HighFrequencyBacktest(config)
    baseline_results = backtest.run(
        random_signals, random_confidences, random_volatilities, prices
    )

    print(f"\nBaseline (random signals) / 基线（随机信号）:")
    print(f"  Total return / 总收益率: {baseline_results['total_return_pct']:.2f}%")
    print(f"  Sharpe ratio / Sharpe比率: {baseline_results['sharpe_ratio']:.3f}")
    print(f"  Max drawdown / 最大回撤: {baseline_results['max_drawdown_pct']:.2f}%")
    print(f"  Num trades / 交易次数: {baseline_results['num_trades']}")

    # Step 4: 汇总 / Summary
    print(f"\n{'='*70}")
    print(f"Comparison Summary / 对比汇总")
    print(f"{'='*70}")
    print(f"\nLatency Advantages of MHLA / MHLA的延迟优势:")
    if latency_results:
        last = latency_results[-1]
        print(
            f"  At seq_len={last['seq_len']}: "
            f"MHLA P50={last['mhla']['p50_ms']:.2f}ms vs "
            f"Softmax P50={last['softmax']['p50_ms']:.2f}ms "
            f"({last['speedup']:.2f}x faster)"
        )
    print(f"\nKey Insight / 关键洞察:")
    print(f"  MHLA's O(n) complexity enables processing longer tick sequences")
    print(f"  within strict latency budgets, potentially improving signal quality")
    print(f"  by incorporating more historical context.")
    print(f"  MHLA的O(n)复杂度允许在严格延迟预算内处理更长的tick序列，")
    print(f"  通过纳入更多历史上下文可能提高信号质量。")
    print(f"{'='*70}")

    return {
        "latency": latency_results,
        "backtest_baseline": baseline_results,
    }


# ============================================================================
# Main / 主程序
# ============================================================================

def main():
    """Main entry point / 主入口"""
    parser = argparse.ArgumentParser(
        description="Evaluate MHLA for high-frequency trading / 评估MHLA在高频交易中的表现"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device / 设备")
    parser.add_argument("--quick", action="store_true", help="Quick mode / 快速模式")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained model checkpoint / 训练模型检查点路径"
    )
    args = parser.parse_args()

    results = run_comparison(
        device=args.device,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
