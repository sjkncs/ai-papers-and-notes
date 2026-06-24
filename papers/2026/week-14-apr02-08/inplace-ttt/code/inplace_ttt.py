"""
In-Place Test-Time Training (TTT) 复现
Reproduction of "In-Place Test-Time Training" (arXiv:2604.06169)

作者: Guhao Feng, Shengjie Luo, Kai Hua, Ge Zhang, Di He, Wenhao Huang, Tianle Cai
本代码为教学简化版，核心思路：
- 将 MLP 的最终投影矩阵 W_out 作为 fast weights
- 在推理时用 NTP-aligned 目标更新 W_out
- 与标准推理（无 TTT）和 Reconstruction 目标 TTT 对比

运行方式:
    pip install torch numpy
    python inplace_ttt.py
"""

import sys
import argparse
from dataclasses import dataclass

# Windows 终端 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -----------------------------------------------------------------------------
# 超参数 / Hyperparameters
# -----------------------------------------------------------------------------
@dataclass
class Config:
    vocab_size: int = 256
    hidden_dim: int = 64
    num_layers: int = 2
    seq_len: int = 32
    ttt_lr: float = 0.01          # TTT 学习率（fast weight 更新步长）
    ttt_steps: int = 3            # 每个 chunk 的 TTT 更新步数
    chunk_size: int = 8           # chunk-wise 更新的分块大小
    seed: int = 42
    device: str = "cpu"


# -----------------------------------------------------------------------------
# 简化的 Transformer MLP 块 / Simplified Transformer MLP Block
# -----------------------------------------------------------------------------
class MLPBlock(nn.Module):
    """
    标准 MLP 块：x -> W_up -> GELU -> W_out -> output
    In-Place TTT 将 W_out 作为 fast weights，在推理时更新。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_up = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.W_out = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (batch, seq, hidden_dim)
        h = self.act(self.W_up(x))   # (batch, seq, hidden_dim * 4)
        out = self.W_out(h)          # (batch, seq, hidden_dim)
        return out, h                # 同时返回中间激活 h，用于 TTT


class MiniTransformer(nn.Module):
    """
    简化 Transformer：Embedding + N 层 (Attention + MLP) + LM Head
    为简洁，注意力层用简单线性替代。
    """
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.Linear(hidden_dim, hidden_dim, bias=False),
                'mlp': MLPBlock(hidden_dim),
                'ln1': nn.LayerNorm(hidden_dim),
                'ln2': nn.LayerNorm(hidden_dim),
            }))
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            # 简化注意力（线性投影）
            x = layer['ln1'](x + layer['attn'](x))
            # MLP 块
            mlp_out, h = layer['mlp'](layer['ln2'](x))
            x = x + mlp_out
        logits = self.lm_head(x)
        return logits


# -----------------------------------------------------------------------------
# In-Place TTT 核心算法 / Core In-Place TTT Algorithm
# -----------------------------------------------------------------------------
class InPlaceTTT:
    """
    In-Place TTT 包装器：
    1. 保存 MLP W_out 的原始权重（slow weights）
    2. 在推理时用 NTP-aligned 目标更新 W_out（fast weights）
    3. 支持 chunk-wise 更新
    """
    def __init__(self, model: MiniTransformer, cfg: Config):
        self.model = model
        self.cfg = cfg
        # 保存每个 MLP 层的 W_out 原始权重（slow weights）
        self.slow_weights = []
        for layer in model.layers:
            W = layer['mlp'].W_out.weight.data.clone()
            self.slow_weights.append(W)

    def reset_fast_weights(self):
        """重置 fast weights 到 slow weights。"""
        for i, layer in enumerate(self.model.layers):
            layer['mlp'].W_out.weight.data.copy_(self.slow_weights[i])

    def ntp_aligned_loss(self, layer, x_chunk, target_chunk):
        """
        NTP-aligned TTT 目标：
        与标准 TTT 的重建目标（reconstruction）不同，这里使用
        下一 token 预测（NTP）对齐的损失。

        标准 TTT: L = ||W_out(h) - x||²（重建输入）
        NTP-aligned: L = CrossEntropy(W_out(h), next_token)（预测下一 token）

        理论上，NTP 目标与自回归语言建模目标一致，避免了目标不对齐问题。
        """
        ln_x = layer['ln2'](x_chunk)
        h = layer['mlp'].act(layer['mlp'].W_up(ln_x))
        # 使用当前的 W_out（fast weight）计算输出
        out = layer['mlp'].W_out(h)
        # NTP 目标：out 应帮助预测 next token
        # 简化：直接用 out 投影到 vocab 空间计算 CE loss
        logits = self.model.lm_head(out)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_chunk.reshape(-1),
        )
        return loss

    def reconstruction_loss(self, layer, x_chunk):
        """
        标准 TTT 重建目标（作为对比基线）：
        L = ||W_out(h) - x||²
        """
        ln_x = layer['ln2'](x_chunk)
        h = layer['mlp'].act(layer['mlp'].W_up(ln_x))
        out = layer['mlp'].W_out(h)
        loss = F.mse_loss(out, x_chunk)
        return loss

    def ttt_update_chunk(self, input_ids, mode='ntp'):
        """
        Chunk-wise TTT 更新：
        将序列分成 chunk，对每个 chunk 执行若干梯度步更新 W_out。
        mode: 'ntp' (NTP-aligned) 或 'recon' (reconstruction)
        """
        seq_len = input_ids.size(1)
        n_chunks = seq_len // self.cfg.chunk_size

        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.cfg.chunk_size
            end = start + self.cfg.chunk_size

            # 当前 chunk 的输入和目标（next token prediction）
            x_chunk = self.model.embed(input_ids[:, start:end])
            # target: 下一个 chunk 的 token ids（简化为当前 chunk 右移一位）
            if end < seq_len:
                target_chunk = input_ids[:, start + 1:end + 1]
            else:
                target_chunk = input_ids[:, start + 1:end]
                x_chunk = x_chunk[:, :-1]

            # 对每个 MLP 层执行 TTT 更新
            for layer_idx, layer in enumerate(self.model.layers):
                W_out = layer['mlp'].W_out.weight
                for step in range(self.cfg.ttt_steps):
                    if mode == 'ntp':
                        loss = self.ntp_aligned_loss(layer, x_chunk, target_chunk)
                    else:
                        loss = self.reconstruction_loss(layer, x_chunk)

                    # 计算 W_out 的梯度
                    grad = torch.autograd.grad(loss, W_out, create_graph=False)[0]
                    # 更新 fast weight（梯度下降）
                    with torch.no_grad():
                        W_out.data -= self.cfg.ttt_lr * grad

    @torch.no_grad()
    def evaluate(self, input_ids, target_ids):
        """评估模型在当前 fast weights 下的性能。"""
        logits = self.model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )
        # 计算准确率
        preds = logits.argmax(dim=-1)
        acc = (preds == target_ids).float().mean().item()
        return loss.item(), acc


# -----------------------------------------------------------------------------
# 主实验 / Main Experiment
# -----------------------------------------------------------------------------
def generate_synthetic_data(vocab_size: int, seq_len: int, num_samples: int):
    """生成合成数据：简单重复模式，便于验证 TTT 效果。"""
    data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))
    # 注入简单模式：每隔 4 个位置重复
    for i in range(4, seq_len + 1):
        data[:, i] = data[:, i - 4]
    return data[:, :-1], data[:, 1:]


def main(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 70)
    print("In-Place Test-Time Training 复现")
    print("=" * 70)

    # 创建模型
    model = MiniTransformer(cfg.vocab_size, cfg.hidden_dim, cfg.num_layers)
    model.eval()

    # 生成合成数据
    train_input, train_target = generate_synthetic_data(cfg.vocab_size, cfg.seq_len, 4)
    test_input, test_target = generate_synthetic_data(cfg.vocab_size, cfg.seq_len, 2)

    # 基线：标准推理（无 TTT）
    print("\n--- 基线：标准推理（无 TTT）---")
    with torch.no_grad():
        baseline_loss, baseline_acc = F.cross_entropy(
            model(test_input).reshape(-1, cfg.vocab_size),
            test_target.reshape(-1),
        ).item(), (model(test_input).argmax(-1) == test_target).float().mean().item()
    print(f"Loss: {baseline_loss:.4f} | Acc: {baseline_acc:.4f}")

    # 方法1: Reconstruction TTT（标准 TTT 基线）
    print("\n--- 方法1: Reconstruction TTT ---")
    ttt_recon = InPlaceTTT(model, cfg)
    ttt_recon.reset_fast_weights()
    ttt_recon.ttt_update_chunk(train_input, mode='recon')
    recon_loss, recon_acc = ttt_recon.evaluate(test_input, test_target)
    print(f"Loss: {recon_loss:.4f} | Acc: {recon_acc:.4f}")

    # 方法2: NTP-aligned TTT（本文方法）
    print("\n--- 方法2: NTP-aligned In-Place TTT ---")
    ttt_ntp = InPlaceTTT(model, cfg)
    ttt_ntp.reset_fast_weights()
    ttt_ntp.ttt_update_chunk(train_input, mode='ntp')
    ntp_loss, ntp_acc = ttt_ntp.evaluate(test_input, test_target)
    print(f"Loss: {ntp_loss:.4f} | Acc: {ntp_acc:.4f}")

    # 总结
    print(f"\n{'='*70}")
    print("总结 / Summary")
    print(f"{'='*70}")
    print(f"{'方法':<25} {'Loss':>10} {'Acc':>10}")
    print(f"{'─'*45}")
    print(f"{'Baseline (no TTT)':<25} {baseline_loss:>10.4f} {baseline_acc:>10.4f}")
    print(f"{'Reconstruction TTT':<25} {recon_loss:>10.4f} {recon_acc:>10.4f}")
    print(f"{'NTP-aligned TTT':<25} {ntp_loss:>10.4f} {ntp_acc:>10.4f}")
    print()
    if ntp_acc > baseline_acc:
        print(f"NTP-aligned TTT 准确率提升: +{ntp_acc - baseline_acc:.4f}")
    if ntp_acc > recon_acc:
        print(f"NTP-aligned TTT 优于 Reconstruction TTT: +{ntp_acc - recon_acc:.4f}")


# -----------------------------------------------------------------------------
# 命令行入口 / CLI Entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="In-Place TTT reproduction")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--ttt-lr", type=float, default=0.01)
    parser.add_argument("--ttt-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config()
    cfg.hidden_dim = args.hidden_dim
    cfg.ttt_lr = args.ttt_lr
    cfg.ttt_steps = args.ttt_steps
    cfg.seed = args.seed
    main(cfg)
