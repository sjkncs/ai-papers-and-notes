#!/usr/bin/env python3
"""
Deep Delta Learning (DDL) — 评估脚本
=====================================
Evaluation Script

基于论文 "Deep Delta Learning" (arXiv:2601.00417) 的完整评估流程。
Complete evaluation pipeline based on "Deep Delta Learning" (arXiv:2601.00417).

功能 / Features:
  1. 困惑度计算 / Perplexity calculation
  2. DDL vs 标准ResNet残差对比 / DDL vs standard ResNet residual comparison
  3. 门控值训练过程可视化 / Gate value training dynamics visualization
  4. 各检查点性能对比表 / Checkpoint performance comparison table

用法 / Usage:
    # 对比DDL和标准模型 / Compare DDL and standard model
    python evaluate.py --ddl-checkpoint checkpoints/ddl/checkpoint_best.pt \
                        --std-checkpoint checkpoints/standard/checkpoint_best.pt

    # 仅评估DDL (无需检查点，使用随机初始化快速演示)
    # Evaluate DDL only (no checkpoint needed, quick demo with random init)
    python evaluate.py --quick-demo

    # 门控值可视化 / Gate value visualization
    python evaluate.py --visualize-gates --ddl-checkpoint checkpoints/ddl/checkpoint_best.pt

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

# 添加项目根目录到路径 / Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DDLModel, StandardTransformer, count_params
from train import (
    create_data_loader,
    evaluate,
    load_checkpoint,
    SyntheticTokenDataset,
)


# ============================================================
# 困惑度评估 / Perplexity Evaluation
# ============================================================

@torch.no_grad()
def compute_perplexity_detailed(
    model: nn.Module,
    data_loader,
    device: torch.device,
    max_steps: int = 200,
) -> Dict[str, float]:
    """
    详细困惑度计算 / Detailed perplexity calculation

    计算每个batch的困惑度并返回统计信息。
    Computes per-batch perplexity and returns statistics.

    Args:
        model: 待评估模型 / Model to evaluate
        data_loader: 数据加载器 / Data loader
        device: 设备 / Device
        max_steps: 最大评估步数 / Max eval steps

    Returns:
        dict: 包含平均损失、困惑度、各batch统计
              Contains avg loss, perplexity, per-batch stats
    """
    model.eval()
    all_losses = []
    total_loss = 0.0
    total_tokens = 0

    for step, (input_ids, target_ids) in enumerate(data_loader):
        if step >= max_steps:
            break

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            reduction="sum",
        )

        batch_avg_loss = loss.item() / target_ids.numel()
        all_losses.append(batch_avg_loss)
        total_loss += loss.item()
        total_tokens += target_ids.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))

    # 统计 / Statistics
    if all_losses:
        import statistics
        return {
            "avg_loss": avg_loss,
            "perplexity": perplexity,
            "min_batch_loss": min(all_losses),
            "max_batch_loss": max(all_losses),
            "std_batch_loss": statistics.stdev(all_losses) if len(all_losses) > 1 else 0.0,
            "num_batches": len(all_losses),
        }
    return {"avg_loss": float("inf"), "perplexity": float("inf"),
            "min_batch_loss": 0, "max_batch_loss": 0,
            "std_batch_loss": 0, "num_batches": 0}


# ============================================================
# DDL vs 标准ResNet 对比 / DDL vs Standard ResNet Comparison
# ============================================================

def run_comparison(
    vocab_size: int = 10000,
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 4,
    d_ff: Optional[int] = None,
    max_seq_len: int = 512,
    batch_size: int = 32,
    train_steps: int = 2000,
    eval_interval: int = 200,
    device: torch.device = None,
    ddl_checkpoint: Optional[str] = None,
    std_checkpoint: Optional[str] = None,
) -> Dict[str, List]:
    """
    运行DDL vs 标准ResNet对比实验
    Run DDL vs Standard ResNet comparison experiment

    两个模型使用相同的训练数据和超参数，对比语言建模性能。
    Both models train on the same data with identical hyperparameters.

    Args:
        模型和训练参数 / Model and training parameters
        ddl_checkpoint: DDL检查点路径 (可选) / DDL checkpoint path
        std_checkpoint: 标准模型检查点路径 (可选) / Standard checkpoint path

    Returns:
        dict: 包含每个检查点的损失和困惑度对比
              Contains loss and perplexity comparison at each checkpoint
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  DDL vs Standard ResNet 对比训练 / Comparison Training")
    print("=" * 70)

    # 创建两个模型 / Create both models
    ddl_model = DDLModel(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_ff=d_ff, max_seq_len=max_seq_len,
    ).to(device)

    std_model = StandardTransformer(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_ff=d_ff, max_seq_len=max_seq_len,
    ).to(device)

    # 加载检查点 (如果提供) / Load checkpoints if provided
    if ddl_checkpoint and os.path.exists(ddl_checkpoint):
        load_checkpoint(ddl_checkpoint, ddl_model, device=device)
        print(f"  Loaded DDL checkpoint: {ddl_checkpoint}")
    if std_checkpoint and os.path.exists(std_checkpoint):
        load_checkpoint(std_checkpoint, std_model, device=device)
        print(f"  Loaded Standard checkpoint: {std_checkpoint}")

    ddl_params = count_params(ddl_model)
    std_params = count_params(std_model)
    print(f"  DDL parameters:      {ddl_params:>12,}")
    print(f"  Standard parameters: {std_params:>12,}")
    print(f"  DDL overhead:        {(ddl_params - std_params):>12,} "
          f"({(ddl_params - std_params) / std_params:.2%})")
    print("-" * 70)

    # 创建共享数据 / Create shared data
    train_loader = create_data_loader(
        "synthetic", vocab_size, max_seq_len, batch_size, split="train", seed=42,
    )
    val_loader = create_data_loader(
        "synthetic", vocab_size, max_seq_len, batch_size, split="val", seed=42,
    )

    # 优化器 (相同配置) / Optimizers (same config)
    ddl_optimizer = torch.optim.AdamW(ddl_model.parameters(), lr=3e-4, weight_decay=0.01)
    std_optimizer = torch.optim.AdamW(std_model.parameters(), lr=3e-4, weight_decay=0.01)

    # 对比记录 / Comparison log
    comparison_log = {
        "steps": [],
        "ddl_loss": [],
        "ddl_ppl": [],
        "std_loss": [],
        "std_ppl": [],
        "ddl_gate_means": [],
    }

    # 对比训练循环 / Comparison training loop
    train_iter = iter(train_loader)

    for step in range(train_steps):
        try:
            input_ids, target_ids = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, target_ids = next(train_iter)

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # --- DDL训练步 / DDL training step ---
        ddl_model.train()
        ddl_logits = ddl_model(input_ids)
        ddl_loss = F.cross_entropy(
            ddl_logits.view(-1, ddl_logits.size(-1)), target_ids.view(-1)
        )
        ddl_optimizer.zero_grad()
        ddl_loss.backward()
        torch.nn.utils.clip_grad_norm_(ddl_model.parameters(), 1.0)
        ddl_optimizer.step()

        # --- 标准模型训练步 / Standard model training step ---
        std_model.train()
        std_logits = std_model(input_ids)
        std_loss = F.cross_entropy(
            std_logits.view(-1, std_logits.size(-1)), target_ids.view(-1)
        )
        std_optimizer.zero_grad()
        std_loss.backward()
        torch.nn.utils.clip_grad_norm_(std_model.parameters(), 1.0)
        std_optimizer.step()

        # 评估 / Evaluation at intervals
        if (step + 1) % eval_interval == 0:
            ddl_metrics = compute_perplexity_detailed(ddl_model, val_loader, device, max_steps=50)
            std_metrics = compute_perplexity_detailed(std_model, val_loader, device, max_steps=50)

            # 获取DDL门控统计 / Get DDL gate stats
            gate_means = []
            if isinstance(ddl_model, DDLModel):
                with torch.no_grad():
                    for block in ddl_model.blocks:
                        attn_mag = block.ddl_attn.delta_gate.gate_proj.weight.abs().mean().item()
                        ffn_mag = block.ddl_ffn.delta_gate.gate_proj.weight.abs().mean().item()
                        gate_means.append({"attn": attn_mag, "ffn": ffn_mag})

            comparison_log["steps"].append(step + 1)
            comparison_log["ddl_loss"].append(ddl_metrics["avg_loss"])
            comparison_log["ddl_ppl"].append(ddl_metrics["perplexity"])
            comparison_log["std_loss"].append(std_metrics["avg_loss"])
            comparison_log["std_ppl"].append(std_metrics["perplexity"])
            comparison_log["ddl_gate_means"].append(gate_means)

            # 打印对比 / Print comparison
            delta_loss = std_metrics["avg_loss"] - ddl_metrics["avg_loss"]
            winner = "DDL" if delta_loss > 0 else "Standard"
            print(
                f"  Step {step + 1:>5d} | "
                f"DDL: loss={ddl_metrics['avg_loss']:.4f} ppl={ddl_metrics['perplexity']:.2f} | "
                f"Std: loss={std_metrics['avg_loss']:.4f} ppl={std_metrics['perplexity']:.2f} | "
                f"Winner: {winner} (delta={delta_loss:+.4f})"
            )

    return comparison_log


def print_comparison_table(comparison_log: Dict[str, List]):
    """
    打印对比结果表格 / Print comparison results table

    生成清晰的DDL vs 标准模型对比表格。
    Generates a clear DDL vs standard model comparison table.
    """
    print("\n")
    print("=" * 85)
    print("  DDL vs Standard ResNet 对比结果表 / Comparison Results Table")
    print("=" * 85)

    # 表头 / Header
    print(f"{'Step':>8s} | {'DDL Loss':>10s} | {'DDL PPL':>10s} | "
          f"{'Std Loss':>10s} | {'Std PPL':>10s} | "
          f"{'Delta Loss':>10s} | {'Winner':>8s}")
    print("-" * 85)

    for i, step in enumerate(comparison_log["steps"]):
        ddl_loss = comparison_log["ddl_loss"][i]
        ddl_ppl = comparison_log["ddl_ppl"][i]
        std_loss = comparison_log["std_loss"][i]
        std_ppl = comparison_log["std_ppl"][i]
        delta = std_loss - ddl_loss
        winner = "DDL" if delta > 0 else "Std"

        print(f"{step:>8d} | {ddl_loss:>10.4f} | {ddl_ppl:>10.2f} | "
              f"{std_loss:>10.4f} | {std_ppl:>10.2f} | "
              f"{delta:>+10.4f} | {winner:>8s}")

    print("=" * 85)

    # 最终门控统计 / Final gate statistics
    if comparison_log["ddl_gate_means"]:
        final_gates = comparison_log["ddl_gate_means"][-1]
        print("\n  DDL Final Gate Statistics / 最终门控统计:")
        print(f"  {'Layer':>6s} | {'Attn Gate':>10s} | {'FFN Gate':>10s}")
        print("  " + "-" * 35)
        for i, g in enumerate(final_gates):
            print(f"  {i:>6d} | {g['attn']:>10.4f} | {g['ffn']:>10.4f}")
        print("  " + "-" * 35)


# ============================================================
# 门控值可视化 / Gate Value Visualization
# ============================================================

def visualize_gate_values(
    model: DDLModel,
    data_loader,
    device: torch.device,
    num_samples: int = 10,
    output_path: str = "gate_analysis.json",
):
    """
    分析和可视化DDL门控值 / Analyze and visualize DDL gate values

    对多个输入样本计算门控值统计，生成分析报告。
    Computes gate value statistics across multiple input samples.

    分析内容 / Analysis contents:
      1. 每层门控均值和方差 / Per-layer gate mean and variance
      2. 门控值分布直方图 (JSON) / Gate distribution histogram
      3. 不同token位置的门控变化 / Gate variation across token positions

    Args:
        model: DDL模型 / DDL model
        data_loader: 数据加载器 / Data loader
        device: 设备 / Device
        num_samples: 分析的样本数 / Number of samples to analyze
        output_path: 输出文件路径 / Output file path
    """
    model.eval()
    all_layer_stats = []

    print("\n" + "=" * 70)
    print("  DDL Gate Value Analysis / DDL门控值分析")
    print("=" * 70)

    sample_count = 0
    for input_ids, _ in data_loader:
        if sample_count >= num_samples:
            break

        input_ids = input_ids.to(device)

        with torch.no_grad():
            gate_values = model.get_gate_values(input_ids)

        # 对每个batch中的样本求平均 / Average across batch
        for layer_idx, gv in enumerate(gate_values):
            if sample_count == 0:
                all_layer_stats.append({
                    "attn_gate_means": [],
                    "attn_gate_stds": [],
                    "ffn_gate_means": [],
                    "ffn_gate_stds": [],
                })
            all_layer_stats[layer_idx]["attn_gate_means"].append(gv["attn_gate_mean"])
            all_layer_stats[layer_idx]["attn_gate_stds"].append(gv["attn_gate_std"])
            all_layer_stats[layer_idx]["ffn_gate_means"].append(gv["ffn_gate_mean"])
            all_layer_stats[layer_idx]["ffn_gate_stds"].append(gv["ffn_gate_std"])

        sample_count += 1

    # 聚合统计 / Aggregate statistics
    print(f"\n  Analyzed {sample_count} samples")
    print(f"\n  {'Layer':>6s} | {'Attn Mean':>10s} | {'Attn Std':>10s} | "
          f"{'FFN Mean':>10s} | {'FFN Std':>10s}")
    print("  " + "-" * 55)

    summary = []
    for i, stats in enumerate(all_layer_stats):
        attn_mean = sum(stats["attn_gate_means"]) / len(stats["attn_gate_means"])
        attn_std = sum(stats["attn_gate_stds"]) / len(stats["attn_gate_stds"])
        ffn_mean = sum(stats["ffn_gate_means"]) / len(stats["ffn_gate_means"])
        ffn_std = sum(stats["ffn_gate_stds"]) / len(stats["ffn_gate_stds"])

        print(f"  {i:>6d} | {attn_mean:>10.4f} | {attn_std:>10.4f} | "
              f"{ffn_mean:>10.4f} | {ffn_std:>10.4f}")

        summary.append({
            "layer": i,
            "attn_gate_mean": round(attn_mean, 6),
            "attn_gate_std": round(attn_std, 6),
            "ffn_gate_mean": round(ffn_mean, 6),
            "ffn_gate_std": round(ffn_std, 6),
        })

    print("  " + "-" * 55)

    # 保存JSON / Save JSON
    output = {
        "num_samples_analyzed": sample_count,
        "num_layers": len(all_layer_stats),
        "layer_stats": summary,
        "interpretation": {
            "gate_near_0": "Layer preserves original content (DDL ≈ standard residual)",
            "gate_near_0.5": "Layer moderately rewrites content",
            "gate_near_1": "Layer aggressively replaces content with target",
        },
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Gate analysis saved to: {output_path}")

    # 文本可视化门控分布 / Text visualization of gate distribution
    print("\n  Gate Distribution (text bar chart):")
    print("  " + "-" * 50)
    for s in summary:
        attn_bar = "#" * int(s["attn_gate_mean"] * 30)
        ffn_bar = "#" * int(s["ffn_gate_mean"] * 30)
        print(f"  L{s['layer']:02d} Attn: [{attn_bar:<30s}] {s['attn_gate_mean']:.3f}")
        print(f"  L{s['layer']:02d} FFN:  [{ffn_bar:<30s}] {s['ffn_gate_mean']:.3f}")
    print("  " + "-" * 50)

    return output


def visualize_gate_training_dynamics(
    comparison_log: Dict[str, List],
    output_path: str = "gate_dynamics.json",
):
    """
    可视化门控值在训练过程中的变化
    Visualize how gate values change during training

    帮助理解DDL的学习动态:
    Helps understand DDL learning dynamics:
      - 门控是否随训练分化？/ Do gates differentiate during training?
      - 浅层 vs 深层门控行为差异？/ Shallow vs deep layer gate behavior?
      - 门控是否收敛？/ Do gates converge?

    Args:
        comparison_log: 对比训练日志 / Comparison training log
        output_path: 输出文件路径 / Output file path
    """
    if not comparison_log.get("ddl_gate_means"):
        print("No gate dynamics data available.")
        return

    print("\n" + "=" * 70)
    print("  Gate Training Dynamics / 门控值训练动态")
    print("=" * 70)

    steps = comparison_log["steps"]
    gate_data = comparison_log["ddl_gate_means"]

    # 打印关键时间点的门控值 / Print gate values at key timepoints
    num_layers = len(gate_data[0]) if gate_data and gate_data[0] else 0

    print(f"\n  Steps tracked: {len(steps)}")
    print(f"  Layers: {num_layers}")

    # 早期 vs 晚期对比 / Early vs late comparison
    if len(gate_data) >= 2:
        early = gate_data[0]
        late = gate_data[-1]

        print(f"\n  Early (step {steps[0]}) vs Late (step {steps[-1]}) gate values:")
        print(f"  {'Layer':>6s} | {'Early Attn':>11s} | {'Late Attn':>11s} | "
              f"{'Change':>8s} | {'Early FFN':>11s} | {'Late FFN':>11s} | {'Change':>8s}")
        print("  " + "-" * 75)

        for i in range(num_layers):
            early_a = early[i]["attn"]
            late_a = late[i]["attn"]
            early_f = early[i]["ffn"]
            late_f = late[i]["ffn"]
            print(f"  {i:>6d} | {early_a:>11.4f} | {late_a:>11.4f} | "
                  f"{late_a - early_a:>+8.4f} | {early_f:>11.4f} | {late_f:>11.4f} | "
                  f"{late_f - early_f:>+8.4f}")
        print("  " + "-" * 75)

    # 保存动态数据 / Save dynamics data
    dynamics = {
        "steps": steps,
        "gate_means_per_step": gate_data,
    }
    with open(output_path, "w") as f:
        json.dump(dynamics, f, indent=2)
    print(f"\n  Gate dynamics saved to: {output_path}")


# ============================================================
# 快速演示 / Quick Demo
# ============================================================

def quick_demo(device: torch.device):
    """
    快速演示: 无需检查点的完整评估流程
    Quick demo: full evaluation pipeline without checkpoints

    使用随机初始化的模型运行对比实验。
    Runs comparison experiment with randomly initialized models.
    """
    print("=" * 70)
    print("  DDL Quick Demo / 快速演示")
    print("=" * 70)

    # 小规模快速演示 / Small-scale quick demo
    vocab_size = 5000
    d_model = 128
    n_layers = 4
    n_heads = 4
    max_seq_len = 128

    # 运行对比 / Run comparison
    comparison_log = run_comparison(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        batch_size=16,
        train_steps=1000,
        eval_interval=100,
        device=device,
    )

    # 打印对比表 / Print comparison table
    print_comparison_table(comparison_log)

    # 门控值分析 / Gate value analysis
    ddl_model = DDLModel(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_seq_len=max_seq_len,
    ).to(device)

    val_loader = create_data_loader(
        "synthetic", vocab_size, max_seq_len, 16, split="val", seed=42,
    )
    visualize_gate_values(ddl_model, val_loader, device, num_samples=5)

    # 门控动态 / Gate dynamics
    visualize_gate_training_dynamics(comparison_log)


# ============================================================
# 参数解析 / Argument Parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DDL Evaluation / DDL评估脚本"
    )

    parser.add_argument("--quick-demo", action="store_true",
                        help="运行快速演示 / Run quick demo")
    parser.add_argument("--ddl-checkpoint", type=str, default=None,
                        help="DDL模型检查点路径 / DDL checkpoint path")
    parser.add_argument("--std-checkpoint", type=str, default=None,
                        help="标准模型检查点路径 / Standard model checkpoint path")
    parser.add_argument("--visualize-gates", action="store_true",
                        help="可视化门控值 / Visualize gate values")

    # 模型参数 / Model parameters
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=512)

    # 训练/评估参数 / Training/eval parameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=200)

    # 输出 / Output
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="输出目录 / Output directory")

    return parser.parse_args()


# ============================================================
# 主函数 / Main Function
# ============================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.quick_demo:
        quick_demo(device)
        return

    # 运行对比实验 / Run comparison experiment
    comparison_log = run_comparison(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        device=device,
        ddl_checkpoint=args.ddl_checkpoint,
        std_checkpoint=args.std_checkpoint,
    )

    # 打印对比表 / Print comparison table
    print_comparison_table(comparison_log)

    # 保存对比结果 / Save comparison results
    comparison_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison_log, f, indent=2)
    print(f"\nComparison results saved to: {comparison_path}")

    # 门控值可视化 / Gate value visualization
    if args.visualize_gates and args.ddl_checkpoint:
        ddl_model = DDLModel(
            vocab_size=args.vocab_size, d_model=args.d_model,
            n_layers=args.n_layers, n_heads=args.n_heads,
            d_ff=args.d_ff, max_seq_len=args.max_seq_len,
        ).to(device)
        load_checkpoint(args.ddl_checkpoint, ddl_model, device=device)

        val_loader = create_data_loader(
            "synthetic", args.vocab_size, args.max_seq_len,
            args.batch_size, split="val", seed=42,
        )
        gate_path = os.path.join(args.output_dir, "gate_analysis.json")
        visualize_gate_values(ddl_model, val_loader, device, output_path=gate_path)

    # 门控训练动态 / Gate training dynamics
    dynamics_path = os.path.join(args.output_dir, "gate_dynamics.json")
    visualize_gate_training_dynamics(comparison_log, output_path=dynamics_path)

    print("\n" + "=" * 70)
    print("  Evaluation Complete / 评估完成")
    print(f"  Results saved to: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
