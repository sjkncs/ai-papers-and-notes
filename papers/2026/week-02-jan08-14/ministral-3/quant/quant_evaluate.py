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
Backtest Evaluation for Edge-Deployed Quant Trading Model
边缘部署量化交易模型的回测评估

Provides comprehensive evaluation of the compact trading signal model including:
- Latency benchmarking vs signal quality tradeoff analysis
- Edge device deployment simulation
- Full backtesting with Sharpe ratio and drawdown metrics
- Comparison between compact (Ministral) and full-size model performance

提供紧凑交易信号模型的全面评估，包括：
- 延迟基准测试与信号质量权衡分析
- 边缘设备部署模拟
- 包含Sharpe比率和最大回撤指标的完整回测
- 紧凑（Ministral）模型与全尺寸模型性能对比

Components / 组件:
    - LatencyBenchmark: Measure inference latency vs signal quality
      测量推理延迟与信号质量
    - EdgeDeploymentSimulator: Simulate edge device constraints
      模拟边缘设备约束
    - run_full_evaluation(): Comprehensive evaluation pipeline
      全面评估流水线
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 导入训练模块的组件 / Import components from training module
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# Configuration / 配置
# ============================================================================

@dataclass
class EvaluationConfig:
    """
    Configuration for backtest evaluation.
    回测评估配置。
    """

    # 回测参数 / Backtest parameters
    initial_capital: float = 1_000_000.0  # 初始资金 / Initial capital (USD)
    position_size: float = 0.1            # 仓位比例 / Position size fraction
    transaction_cost_bps: float = 5.0     # 交易成本（基点）/ Transaction cost (basis points)
    slippage_bps: float = 2.0             # 滑点（基点）/ Slippage (basis points)

    # 风险参数 / Risk parameters
    max_drawdown_limit: float = 0.15      # 最大回撤限制 / Max drawdown limit
    stop_loss_pct: float = 0.02           # 止损百分比 / Stop loss percentage

    # 延迟约束 / Latency constraints
    max_latency_ms: float = 10.0          # 最大延迟 / Maximum latency (ms)
    latency_budget_ms: float = 5.0        # 延迟预算 / Latency budget (ms)

    # 评估周期 / Evaluation periods
    warmup_days: int = 5                  # 预热天数 / Warmup days
    risk_free_rate: float = 0.05          # 无风险利率 / Risk-free rate (annual)


# ============================================================================
# Latency Benchmark / 延迟基准测试
# ============================================================================

class LatencyBenchmark:
    """
    Benchmark inference latency vs signal quality tradeoff.
    推理延迟与信号质量权衡的基准测试。

    Measures how inference latency affects signal accuracy and trading
    performance across different model sizes and hardware configurations.
    测量推理延迟如何在不同模型规模和硬件配置下影响信号准确率和交易表现。
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results: List[Dict] = []

    def measure_inference_latency(
        self,
        model: nn.Module,
        input_shapes: Dict[str, Tuple],
        num_warmup: int = 20,
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """
        Measure inference latency statistics.
        测量推理延迟统计信息。

        Args:
            model: PyTorch model / PyTorch模型
            input_shapes: Dict of tensor name -> shape / 张量名称->形状字典
            num_warmup: Number of warmup iterations / 预热迭代数
            num_runs: Number of measurement runs / 测量运行数

        Returns:
            Latency statistics dictionary / 延迟统计字典
        """
        model = model.to(self.device)
        model.eval()

        # 创建虚拟输入 / Create dummy inputs
        dummy_inputs = {
            name: torch.randn(*shape, device=self.device)
            for name, shape in input_shapes.items()
        }

        # 预热 / Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                if isinstance(dummy_inputs, dict):
                    _ = model(**dummy_inputs) if hasattr(model, '__call__') else model(
                        dummy_inputs.get("orderbook", torch.randn(1, 64, 20, device=self.device)),
                        dummy_inputs.get("trades", torch.randn(1, 64, 4, device=self.device)),
                        dummy_inputs.get("stats", torch.randn(1, 64, 5, device=self.device)),
                    )

        # 测量 / Measure
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                try:
                    _ = model(
                        dummy_inputs.get("orderbook", torch.randn(1, 64, 20, device=self.device)),
                        dummy_inputs.get("trades", torch.randn(1, 64, 4, device=self.device)),
                        dummy_inputs.get("stats", torch.randn(1, 64, 5, device=self.device)),
                    )
                except TypeError:
                    _ = model(**dummy_inputs)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                latencies.append(elapsed)

        latencies.sort()

        stats = {
            "p50_ms": latencies[len(latencies) // 2],
            "p90_ms": latencies[int(len(latencies) * 0.90)],
            "p95_ms": latencies[int(len(latencies) * 0.95)],
            "p99_ms": latencies[int(len(latencies) * 0.99)],
            "mean_ms": sum(latencies) / len(latencies),
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
            "std_ms": (
                sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies)
                / len(latencies)
            ) ** 0.5,
        }

        return stats

    def benchmark_model_sizes(
        self,
        configs: List[Dict],
        lookback: int = 64,
        orderbook_levels: int = 5,
    ) -> List[Dict]:
        """
        Compare latency across different model sizes.
        比较不同模型规模的延迟。

        Args:
            configs: List of model config dicts / 模型配置字典列表
            lookback: Lookback window / 回看窗口
            orderbook_levels: Order book depth / 订单簿深度

        Returns:
            List of benchmark results / 基准测试结果列表
        """
        results = []

        for cfg in configs:
            from quant_train import CompactTradingSignalModel, QuantModelConfig

            config = QuantModelConfig(**cfg)
            model = CompactTradingSignalModel(config)

            total_params = sum(p.numel() for p in model.parameters())

            input_shapes = {
                "orderbook": (1, lookback, orderbook_levels * 4),
                "trades": (1, lookback, 4),
                "stats": (1, lookback, 5),
            }

            latency_stats = self.measure_inference_latency(model, input_shapes)

            result = {
                "model_config": cfg,
                "total_params": total_params,
                "total_params_M": total_params / 1e6,
                **latency_stats,
            }
            results.append(result)

        self.results = results
        return results

    def print_results(self):
        """Print formatted benchmark results / 打印格式化的基准测试结果"""
        print(f"\n{'='*80}")
        print(f"Latency Benchmark Results / 延迟基准测试结果")
        print(f"{'='*80}")
        print(
            f"{'Model':>12s} | {'Params(M)':>10s} | {'P50(ms)':>10s} | "
            f"{'P95(ms)':>10s} | {'P99(ms)':>10s} | {'Mean(ms)':>10s}"
        )
        print("-" * 80)

        for r in self.results:
            name = f"h{r['model_config'].get('hidden_dim', 0)}_l{r['model_config'].get('num_layers', 0)}"
            print(
                f"{name:>12s} | {r['total_params_M']:>10.1f} | {r['p50_ms']:>10.2f} | "
                f"{r['p95_ms']:>10.2f} | {r['p99_ms']:>10.2f} | {r['mean_ms']:>10.2f}"
            )


# ============================================================================
# Edge Deployment Simulator / 边缘设备部署模拟器
# ============================================================================

class EdgeDeploymentSimulator:
    """
    Simulate edge device constraints for deployment evaluation.
    模拟边缘设备约束以进行部署评估。

    Models hardware constraints including:
    建模硬件约束，包括：
    - Memory limits / 内存限制
    - Compute throughput / 计算吞吐量
    - Thermal throttling / 热节流
    - Network bandwidth (for cloud-edge hybrid) / 网络带宽（云边缘混合）
    """

    # 预定义的设备配置 / Predefined device profiles
    DEVICE_PROFILES = {
        "jetson-orin": {
            "name": "NVIDIA Jetson Orin / 英伟达Jetson Orin",
            "gpu_memory_gb": 64,
            "cpu_cores": 12,
            "compute_tflops": 275,       # INT8 TOPS
            "memory_bandwidth_gbps": 204,
            "power_watts": 60,
            "thermal_throttle_factor": 0.85,  # 节流后性能比例
        },
        "raspberry-pi5": {
            "name": "Raspberry Pi 5 / 树莓派5",
            "gpu_memory_gb": 8,
            "cpu_cores": 4,
            "compute_tflops": 2,
            "memory_bandwidth_gbps": 17,
            "power_watts": 12,
            "thermal_throttle_factor": 0.70,
        },
        "edge-server-a10": {
            "name": "Edge Server (A10) / 边缘服务器 (A10)",
            "gpu_memory_gb": 24,
            "cpu_cores": 32,
            "compute_tflops": 250,
            "memory_bandwidth_gbps": 600,
            "power_watts": 150,
            "thermal_throttle_factor": 0.95,
        },
    }

    def __init__(self, device_profile: str = "jetson-orin"):
        if device_profile not in self.DEVICE_PROFILES:
            raise ValueError(
                f"Unknown device profile / 未知设备配置: {device_profile}. "
                f"Available / 可选: {list(self.DEVICE_PROFILES.keys())}"
            )
        self.profile = self.DEVICE_PROFILES[device_profile]
        self.device_key = device_profile

    def check_memory_fit(self, model_params: int, precision: str = "fp16") -> Dict:
        """
        Check if model fits in device memory.
        检查模型是否适合设备内存。

        Args:
            model_params: Total number of parameters / 总参数数量
            precision: "fp16", "bf16", or "int8" / 精度

        Returns:
            Memory fit analysis / 内存适配分析
        """
        bytes_per_param = {"fp16": 2, "bf16": 2, "int8": 1, "fp32": 4}.get(precision, 2)
        model_memory_gb = model_params * bytes_per_param / 1e9

        # 推理时的额外开销（激活值、KV缓存等）
        # Additional overhead during inference (activations, KV cache, etc.)
        overhead_factor = 1.5
        total_memory_gb = model_memory_gb * overhead_factor

        available_memory = self.profile["gpu_memory_gb"]
        fits = total_memory_gb <= available_memory

        return {
            "model_memory_gb": round(model_memory_gb, 3),
            "total_memory_gb": round(total_memory_gb, 3),
            "available_memory_gb": available_memory,
            "fits": fits,
            "memory_utilization_pct": round(total_memory_gb / available_memory * 100, 1),
            "headroom_gb": round(available_memory - total_memory_gb, 3),
        }

    def estimate_throughput(
        self,
        model_params: int,
        seq_len: int = 64,
        batch_size: int = 1,
    ) -> Dict:
        """
        Estimate inference throughput on edge device.
        估算边缘设备上的推理吞吐量。

        Args:
            model_params: Total parameters / 总参数量
            seq_len: Input sequence length / 输入序列长度
            batch_size: Batch size / 批量大小

        Returns:
            Throughput estimates / 吞吐量估算
        """
        # 粗略估计：每个参数约需2次FLOP（乘法+加法）
        # Rough estimate: ~2 FLOPs per parameter (multiply + add)
        flops_per_sample = model_params * 2 * seq_len
        total_flops = flops_per_sample * batch_size

        # 设备计算能力（考虑热节流）
        # Device compute capability (with thermal throttling)
        effective_tflops = (
            self.profile["compute_tflops"]
            * self.profile["thermal_throttle_factor"]
        )

        # 估算每秒样本数 / Estimated samples per second
        estimated_sps = (effective_tflops * 1e12) / total_flops
        estimated_latency_ms = 1000.0 / max(estimated_sps, 1)

        return {
            "flops_per_sample": flops_per_sample,
            "estimated_samples_per_sec": round(estimated_sps, 1),
            "estimated_latency_ms": round(estimated_latency_ms, 2),
            "effective_tflops": effective_tflops,
            "thermal_limited": self.profile["thermal_throttle_factor"] < 0.9,
        }

    def simulate_deployment(
        self,
        model: nn.Module,
        precision: str = "fp16",
        batch_size: int = 1,
        seq_len: int = 64,
    ) -> Dict:
        """
        Full deployment simulation.
        完整的部署模拟。

        Args:
            model: PyTorch model / PyTorch模型
            precision: Inference precision / 推理精度
            batch_size: Expected batch size / 预期批量大小
            seq_len: Expected sequence length / 预期序列长度

        Returns:
            Comprehensive deployment analysis / 全面部署分析
        """
        total_params = sum(p.numel() for p in model.parameters())

        memory_analysis = self.check_memory_fit(total_params, precision)
        throughput_analysis = self.estimate_throughput(total_params, seq_len, batch_size)

        # 综合评估 / Comprehensive assessment
        deployment_ready = (
            memory_analysis["fits"]
            and throughput_analysis["estimated_latency_ms"] < 50  # 50ms hard limit
        )

        return {
            "device": self.profile["name"],
            "total_params": total_params,
            "precision": precision,
            "memory": memory_analysis,
            "throughput": throughput_analysis,
            "deployment_ready": deployment_ready,
            "power_watts": self.profile["power_watts"],
        }


# ============================================================================
# Backtest Engine / 回测引擎
# ============================================================================

class BacktestEngine:
    """
    Simple backtesting engine for trading signal evaluation.
    交易信号评估的简单回测引擎。

    Supports:
    - Long/short/flat positions / 多/空/平仓位
    - Transaction costs and slippage / 交易成本和滑点
    - Drawdown monitoring / 回撤监控
    - Risk-adjusted metrics / 风险调整指标
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.positions: List[float] = []
        self.returns: List[float] = []
        self.equity_curve: List[float] = []
        self.trades: List[Dict] = []

    def run_backtest(
        self,
        signals: torch.Tensor,
        confidences: torch.Tensor,
        prices: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Run backtest with given signals and price series.
        使用给定信号和价格序列运行回测。

        Args:
            signals: Trading signals (0=sell, 1=hold, 2=buy) [T]
            confidences: Signal confidences [T]
            prices: Price series [T]

        Returns:
            Backtest metrics / 回测指标
        """
        T = len(prices)
        capital = self.config.initial_capital
        position = 0.0  # 当前仓位 / Current position (shares)
        entry_price = 0.0

        self.equity_curve = [capital]
        self.returns = []
        self.trades = []
        max_equity = capital
        max_drawdown = 0.0
        num_trades = 0
        winning_trades = 0

        for t in range(1, T):
            price = prices[t].item()
            prev_price = prices[t - 1].item()
            signal = signals[t].item()
            confidence = confidences[t].item()

            # 计算当前权益 / Compute current equity
            current_equity = capital + position * price
            self.equity_curve.append(current_equity)

            # 计算收益率 / Compute return
            if self.equity_curve[-2] > 0:
                ret = (current_equity - self.equity_curve[-2]) / self.equity_curve[-2]
            else:
                ret = 0.0
            self.returns.append(ret)

            # 更新最大回撤 / Update max drawdown
            max_equity = max(max_equity, current_equity)
            drawdown = (max_equity - current_equity) / max_equity
            max_drawdown = max(max_drawdown, drawdown)

            # 交易成本 / Transaction cost
            cost_fraction = (
                self.config.transaction_cost_bps + self.config.slippage_bps
            ) / 10000.0

            # 信号执行 / Signal execution
            target_position = 0.0
            if signal == 2 and confidence > 0.5:  # 买入信号且置信度高 / Buy with high confidence
                target_position = self.config.position_size * capital / price
            elif signal == 0 and confidence > 0.5:  # 卖出信号 / Sell signal
                target_position = -self.config.position_size * capital / price

            # 止损检查 / Stop loss check
            if position > 0 and price < entry_price * (1 - self.config.stop_loss_pct):
                target_position = 0.0

            # 执行交易 / Execute trade
            if abs(target_position - position) > 1e-6:
                trade_value = abs(target_position - position) * price
                cost = trade_value * cost_fraction
                capital -= cost
                num_trades += 1

                if position != 0:
                    # 记录平仓盈亏 / Record closed P&L
                    pnl = position * (price - entry_price)
                    if pnl > 0:
                        winning_trades += 1
                    self.trades.append({
                        "entry_price": entry_price,
                        "exit_price": price,
                        "pnl": pnl,
                        "confidence": confidence,
                    })

                position = target_position
                if target_position != 0:
                    entry_price = price

        # 计算指标 / Compute metrics
        returns_tensor = torch.tensor(self.returns)
        total_return = (self.equity_curve[-1] - self.config.initial_capital) / self.config.initial_capital

        # 年化指标 / Annualized metrics
        trading_days = T / 390  # 假设分钟数据 / Assume minute data
        annual_factor = 252.0

        if len(self.returns) > 1 and returns_tensor.std() > 0:
            daily_returns = returns_tensor.view(-1, 390).mean(dim=1) if T >= 390 else returns_tensor
            sharpe = (
                (daily_returns.mean() - self.config.risk_free_rate / 252)
                / (daily_returns.std() + 1e-8)
                * math.sqrt(252)
            )
        else:
            sharpe = 0.0

        win_rate = winning_trades / max(num_trades, 1)

        return {
            "total_return": round(total_return * 100, 2),
            "sharpe_ratio": round(sharpe.item() if isinstance(sharpe, torch.Tensor) else sharpe, 3),
            "max_drawdown": round(max_drawdown * 100, 2),
            "num_trades": num_trades,
            "win_rate": round(win_rate * 100, 1),
            "final_equity": round(self.equity_curve[-1], 2),
            "avg_confidence": round(confidences.mean().item(), 3),
        }


# ============================================================================
# Full Evaluation Pipeline / 完整评估流水线
# ============================================================================

def run_full_evaluation(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    edge_device: str = "jetson-orin",
    quick: bool = False,
) -> Dict[str, Dict]:
    """
    Run complete evaluation pipeline.
    运行完整的评估流水线。

    Steps / 步骤:
    1. Load or create model / 加载或创建模型
    2. Run latency benchmarks / 运行延迟基准测试
    3. Simulate edge deployment / 模拟边缘部署
    4. Run backtest / 运行回测
    5. Generate comparison report / 生成对比报告

    Args:
        checkpoint_path: Path to model checkpoint / 模型检查点路径
        device: Compute device / 计算设备
        edge_device: Target edge device / 目标边缘设备
        quick: Quick evaluation mode / 快速评估模式

    Returns:
        Complete evaluation results / 完整评估结果
    """
    from quant_train import (
        CompactTradingSignalModel,
        QuantModelConfig,
        simulate_market_data,
    )

    print(f"\n{'='*70}")
    print(f"Full Evaluation Pipeline / 完整评估流水线")
    print(f"{'='*70}")

    # Step 1: 模型准备 / Model preparation
    print("\n[Step 1] Model Preparation / 模型准备...")
    config = QuantModelConfig(
        hidden_dim=256, num_layers=4, num_heads=8, num_kv_heads=2,
    )
    model = CompactTradingSignalModel(config)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded checkpoint / 加载检查点: {checkpoint_path}")
    else:
        print("  Using randomly initialized model (demo) / 使用随机初始化模型（演示）")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters / 总参数: {total_params:,} ({total_params/1e6:.1f}M)")

    # Step 2: 延迟基准测试 / Latency benchmarking
    print("\n[Step 2] Latency Benchmarking / 延迟基准测试...")
    benchmark = LatencyBenchmark(device=device)

    model_configs = [
        {"hidden_dim": 64, "num_layers": 2, "num_heads": 4, "num_kv_heads": 1},
        {"hidden_dim": 128, "num_layers": 3, "num_heads": 4, "num_kv_heads": 2},
        {"hidden_dim": 256, "num_layers": 4, "num_heads": 8, "num_kv_heads": 2},
        {"hidden_dim": 512, "num_layers": 6, "num_heads": 8, "num_kv_heads": 2},
    ]
    benchmark.benchmark_model_sizes(model_configs)
    benchmark.print_results()

    # Step 3: 边缘部署模拟 / Edge deployment simulation
    print(f"\n[Step 3] Edge Deployment Simulation / 边缘部署模拟 ({edge_device})...")
    simulator = EdgeDeploymentSimulator(edge_device)
    deployment = simulator.simulate_deployment(model, precision="fp16")

    print(f"  Device / 设备: {deployment['device']}")
    print(f"  Memory fit / 内存适配: {deployment['memory']['fits']}")
    print(f"  Memory usage / 内存使用: {deployment['memory']['total_memory_gb']:.3f} GB")
    print(f"  Memory utilization / 内存利用率: {deployment['memory']['memory_utilization_pct']:.1f}%")
    print(f"  Estimated latency / 估计延迟: {deployment['throughput']['estimated_latency_ms']:.2f} ms")
    print(f"  Deployment ready / 部署就绪: {deployment['deployment_ready']}")

    # Step 4: 回测 / Backtesting
    print("\n[Step 4] Backtesting / 回测...")
    num_samples = 2000 if quick else 10000
    data = simulate_market_data(num_samples=num_samples, seed=42)

    model.eval()
    all_signals = []
    all_confidences = []

    batch_size = 32
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end = min(i + batch_size, num_samples)
            ob = data["orderbook"][i:end].to(device)
            trades = data["trades"][i:end].to(device)
            stats = data["stats"][i:end].to(device)

            outputs = model(ob, trades, stats)
            signals = outputs["signal_logits"].argmax(dim=-1)
            confidences = outputs["confidence"].squeeze(-1)

            all_signals.extend(signals.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())

    # 运行回测引擎 / Run backtest engine
    eval_config = EvaluationConfig()
    backtest = BacktestEngine(eval_config)

    signals_tensor = torch.tensor(all_signals)
    confidences_tensor = torch.tensor(all_confidences)
    prices_tensor = data["prices"][:, -1]  # 使用最后时间步价格 / Use last timestep prices

    # 确保长度一致 / Ensure matching lengths
    min_len = min(len(signals_tensor), len(prices_tensor))
    signals_tensor = signals_tensor[:min_len]
    confidences_tensor = confidences_tensor[:min_len]
    prices_tensor = prices_tensor[:min_len]

    backtest_results = backtest.run_backtest(
        signals_tensor, confidences_tensor, prices_tensor
    )

    print(f"  Total return / 总收益率: {backtest_results['total_return']:.2f}%")
    print(f"  Sharpe ratio / Sharpe比率: {backtest_results['sharpe_ratio']:.3f}")
    print(f"  Max drawdown / 最大回撤: {backtest_results['max_drawdown']:.2f}%")
    print(f"  Number of trades / 交易次数: {backtest_results['num_trades']}")
    print(f"  Win rate / 胜率: {backtest_results['win_rate']:.1f}%")

    # Step 5: 汇总报告 / Summary report
    print(f"\n{'='*70}")
    print(f"Evaluation Summary / 评估汇总")
    print(f"{'='*70}")
    print(f"Model / 模型: CompactTradingSignalModel ({total_params/1e6:.1f}M params)")
    print(f"Edge device / 边缘设备: {deployment['device']}")
    print(f"Deployment ready / 部署就绪: {'Yes / 是' if deployment['deployment_ready'] else 'No / 否'}")
    print(f"P99 Latency / P99延迟: {benchmark.results[2]['p99_ms']:.2f}ms" if len(benchmark.results) > 2 else "")
    print(f"Sharpe Ratio / Sharpe比率: {backtest_results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown / 最大回撤: {backtest_results['max_drawdown']:.2f}%")
    print(f"{'='*70}")

    return {
        "latency": benchmark.results,
        "deployment": deployment,
        "backtest": backtest_results,
    }


# ============================================================================
# Main / 主程序
# ============================================================================

def main():
    """Main entry point / 主入口"""
    parser = argparse.ArgumentParser(
        description="Evaluate edge-deployed quant model / 评估边缘部署量化模型"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint / 模型检查点路径"
    )
    parser.add_argument(
        "--benchmark", type=str, default="full", choices=["full", "latency", "deployment"],
        help="Evaluation type / 评估类型"
    )
    parser.add_argument(
        "--edge-device", type=str, default="jetson-orin",
        choices=list(EdgeDeploymentSimulator.DEVICE_PROFILES.keys()),
        help="Target edge device / 目标边缘设备"
    )
    parser.add_argument("--quick", action="store_true", help="Quick evaluation / 快速评估")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device / 计算设备")
    args = parser.parse_args()

    results = run_full_evaluation(
        checkpoint_path=args.checkpoint,
        device=args.device,
        edge_device=args.edge_device,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
