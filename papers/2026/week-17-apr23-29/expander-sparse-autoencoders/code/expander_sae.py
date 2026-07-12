"""
Expander Sparse Autoencoder (Expander SAE) 的 PyTorch 复现
PyTorch reproduction of Expander Sparse Autoencoders

论文 / Paper:
Expander Sparse Autoencoders: Parameter-Efficient Dictionaries for Mechanistic Interpretability
arXiv:2604.01799

本脚本使用合成数据验证核心思想：
1. 用左 d-正则扩展图掩码约束解码器与编码器的连接；
2. 通过 Top-K 激活获得稀疏隐变量；
3. 在参数量远小于密集 SAE 的情况下恢复输入并重建稀疏特征。

This script uses synthetic data to validate the core ideas:
1. Constrain decoder/encoder connections with a left-d-regular expander-graph mask;
2. Obtain sparse latents via Top-K activation;
3. Reconstruct input and recover sparse features with far fewer parameters than a dense SAE.
"""

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# 0. 超参数配置 / Hyperparameters
# -------------------------------
@dataclass
class Config:
    # 输入维度：模拟语言模型某层激活的维度
    # Input dimension, simulating a layer activation of an LLM
    input_dim: int = 512

    # 字典大小（隐变量数）：通常大于输入维度以过完备
    # Dictionary size (number of latents), often larger than input_dim for overcompleteness
    dict_size: int = 1024

    # 扩展图每个隐变量连接的输入维度数 d，d << input_dim
    # Degree d of the left-d-regular expander mask, d << input_dim
    expand_degree: int = 16

    # Top-K 稀疏度：每个样本最多激活 K 个隐变量
    # Top-K sparsity: at most K latents are active per sample
    topk: int = 8

    # 训练样本数
    # Number of training samples
    n_train: int = 5000

    # 测试样本数
    # Number of test samples
    n_test: int = 1000

    # 真实稀疏特征的稀疏度（每个样本使用多少真实特征）
    # Sparsity of the ground-truth feature codes
    gt_sparsity: int = 4

    # 每个真实特征在输入维度上的支撑大小（使其局部化，便于扩展图掩码恢复）
    # Support size of each ground-truth feature in input dimensions (localized)
    gt_feature_support: int = 16

    # 批次大小 / Batch size
    batch_size: int = 256

    # 训练步数 / Training steps
    num_steps: int = 1500

    # 学习率 / Learning rate
    lr: float = 5e-3

    # 权重衰减 / Weight decay
    weight_decay: float = 1e-5

    # L1 正则系数（可选，用于进一步稀疏化）
    # L1 regularization coefficient (optional, for additional sparsity)
    l1_coef: float = 1e-5

    # 是否将扩展图掩码与真实特征支撑对齐（合成演示用）
    # Whether to align the expander mask with ground-truth feature supports (demo)
    aligned_mask: bool = True

    # 设备：优先使用 GPU
    # Device: use GPU if available
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 随机种子 / Random seed
    seed: int = 42


def set_seed(seed: int):
    """固定随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------
# 1. 合成数据生成 / Synthetic data
# -------------------------------
def make_synthetic_data(cfg: Config):
    """
    构造已知的稀疏特征字典，并基于它生成带噪声的观测数据。

    Build a known sparse feature dictionary and generate noisy observations from it.

    返回 / Returns:
        F: (input_dim, dict_size) 真实特征矩阵，每列是一个真实特征向量
        train_x, test_x: 训练/测试样本，形状 (n_samples, input_dim)
        train_codes, test_codes: 训练/测试真实稀疏码，形状 (n_samples, dict_size)
    """
    # 随机生成真实特征矩阵，每列仅保留 gt_feature_support 个非零维度并归一化
    # Random ground-truth feature matrix; each column keeps only gt_feature_support
    # nonzero input dimensions and is then normalized.
    F = torch.randn(cfg.input_dim, cfg.dict_size, device=cfg.device)
    for j in range(cfg.dict_size):
        zero_rows = torch.randperm(cfg.input_dim, device=cfg.device)[cfg.gt_feature_support :]
        F[zero_rows, j] = 0.0
    F = F / F.norm(dim=0, keepdim=True)

    def _sample_codes(n: int):
        # 初始化全零稀疏码
        codes = torch.zeros(n, cfg.dict_size, device=cfg.device)
        for i in range(n):
            # 为每个样本随机选择 gt_sparsity 个非零特征位置
            # Randomly choose gt_sparsity nonzero feature indices per sample
            idx = torch.randperm(cfg.dict_size, device=cfg.device)[: cfg.gt_sparsity]
            # 非零系数服从标准正态
            codes[i, idx] = torch.randn(cfg.gt_sparsity, device=cfg.device)
        return codes

    train_codes = _sample_codes(cfg.n_train)
    test_codes = _sample_codes(cfg.n_test)

    # 观测值 = 稀疏码 × 特征矩阵^T + 小噪声
    # Observations = sparse codes @ feature matrix^T + small noise
    train_x = train_codes @ F.t() + 0.05 * torch.randn_like(train_codes @ F.t())
    test_x = test_codes @ F.t() + 0.05 * torch.randn_like(test_codes @ F.t())

    return F, train_x, test_x, train_codes, test_codes


# -------------------------------
# 2. 扩展图掩码 / Expander mask
# -------------------------------
def build_expander_mask(input_dim: int, dict_size: int, degree: int, device: str):
    """
    构建左 d-正则二分图掩码。

    Build a left-d-regular bipartite graph mask.

    掩码形状为 (input_dim, dict_size)。
    每一列（对应一个隐变量/字典原子）恰好有 degree 个非零连接。
    这种稀疏结构使得解码器参数量从 O(input_dim * dict_size) 降到 O(degree * dict_size)。

    The mask has shape (input_dim, dict_size).
    Each column (one latent/dictionary atom) has exactly `degree` nonzero connections.
    This sparse structure reduces decoder parameters from O(input_dim * dict_size)
    to O(degree * dict_size).
    """
    assert degree <= input_dim, "degree 不能超过 input_dim"

    mask = torch.zeros(input_dim, dict_size, dtype=torch.float32, device=device)
    for j in range(dict_size):
        # 为第 j 个隐变量随机选择 degree 个输入维度作为连接
        # Randomly select `degree` input dimensions for latent j
        rows = torch.randperm(input_dim, device=device)[:degree]
        mask[rows, j] = 1.0

    return mask


# -------------------------------
# 3. Expander SAE 模型 / Model
# -------------------------------
class ExpanderSAE(nn.Module):
    """
    基于扩展图掩码的 Top-K 稀疏自编码器。

    Top-K sparse autoencoder constrained by an expander-graph mask.

    结构 / Architecture:
      z = TopK( (x @ W_enc^T + b_enc) )
      x_hat = z @ W_dec^T + b_dec

    其中 W_dec 与 W_enc 共享同一个权重矩阵 W，并乘以扩展图掩码。
    W_dec = W * mask
    W_enc = W^T * mask^T
    """

    def __init__(self, cfg: Config, mask: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mask", mask)  # (input_dim, dict_size)

        # 共享的原始权重矩阵，形状 (input_dim, dict_size)
        # Shared raw weight matrix of shape (input_dim, dict_size)
        std = 1.0 / math.sqrt(cfg.input_dim)
        self.W = nn.Parameter(torch.randn(cfg.input_dim, cfg.dict_size) * std)

        # 编码器偏置 / Encoder bias
        self.b_enc = nn.Parameter(torch.zeros(cfg.dict_size))

        # 解码器偏置 / Decoder bias
        self.b_dec = nn.Parameter(torch.zeros(cfg.input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码：应用掩码后的编码器权重，并取 Top-K。"""
        # 解码器权重 = 原始权重 × 掩码
        # Decoder weights = raw weights * mask
        W_dec = self.W * self.mask

        # 绑定编码器权重为解码器权重的转置，并乘以掩码转置
        # Tied encoder weights are the transpose of decoder weights, masked similarly
        W_enc = W_dec.t()  # (dict_size, input_dim)

        # 预激活 / Pre-activation
        pre = x @ W_enc.t() + self.b_enc  # (batch, dict_size)

        # Top-K 稀疏化：仅保留每行中绝对值最大的 K 个值
        # Top-K sparsification: keep only the K largest-magnitude values per row
        topk_vals, topk_idx = torch.topk(pre.abs(), k=self.cfg.topk, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, pre.gather(-1, topk_idx))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码：用掩码后的解码器权重重建输入。"""
        W_dec = self.W * self.mask
        return z @ W_dec.t() + self.b_dec

    def forward(self, x: torch.Tensor):
        """前向：返回重建值与隐变量。"""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# -------------------------------
# 4. 训练与评估 / Training & evaluation
# -------------------------------
def train_expander_sae(cfg: Config, model: nn.Module, train_x: torch.Tensor, test_x: torch.Tensor):
    """训练 Expander SAE。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 创建可重复使用的随机索引
    # Reusable random indices for sampling mini-batches
    n = train_x.size(0)
    perm = torch.randperm(n, device=cfg.device)
    step = 0
    idx = 0

    model.train()
    for _ in range(cfg.num_steps):
        # 取一个 mini-batch
        if idx + cfg.batch_size > n:
            perm = torch.randperm(n, device=cfg.device)
            idx = 0
        batch_idx = perm[idx : idx + cfg.batch_size]
        idx += cfg.batch_size
        batch_x = train_x[batch_idx]

        x_hat, z = model(batch_x)

        # 重建损失 / Reconstruction loss
        recon_loss = F.mse_loss(x_hat, batch_x)

        # L1 稀疏惩罚 / L1 sparsity penalty on latents
        l1_loss = cfg.l1_coef * z.abs().mean()

        loss = recon_loss + l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step % 100 == 0:
            test_mse = evaluate_mse(model, test_x, cfg.batch_size)
            print(
                f"Step {step:04d} | train MSE: {recon_loss.item():.6f} | "
                f"test MSE: {test_mse:.6f} | L1: {l1_loss.item():.6f}"
            )


def evaluate_mse(model: nn.Module, data: torch.Tensor, batch_size: int) -> float:
    """在测试数据上计算均方误差（按元素平均）。"""
    model.eval()
    total_loss = 0.0
    total_elements = 0
    with torch.no_grad():
        for i in range(0, data.size(0), batch_size):
            batch = data[i : i + batch_size]
            x_hat, _ = model(batch)
            total_loss += F.mse_loss(x_hat, batch, reduction="sum").item()
            total_elements += batch.numel()
    model.train()
    return total_loss / total_elements


def evaluate_feature_recovery(F_true: torch.Tensor, model: nn.Module):
    """
    评估学习到的字典与真实特征之间的对应关系。

    Evaluate how well learned dictionary atoms align with ground-truth features.

    对每条真实特征，找出与其余弦相似度最高的解码器列，取平均。
    For each ground-truth feature, find the decoder column with highest cosine similarity and average.
    """
    model.eval()
    with torch.no_grad():
        W_dec = (model.W * model.mask).detach()  # (input_dim, dict_size)
        # 归一化列
        W_dec = W_dec / (W_dec.norm(dim=0, keepdim=True) + 1e-12)
        F_true = F_true / (F_true.norm(dim=0, keepdim=True) + 1e-12)

        # 计算余弦相似度矩阵 / Compute cosine similarity matrix
        sim = F_true.t() @ W_dec  # (dict_size, dict_size)
        max_sim = sim.abs().max(dim=1)[0]
        mean_max_sim = max_sim.mean().item()
    model.train()
    return mean_max_sim


# -------------------------------
# 5. 主函数 / Main
# -------------------------------
def main():
    cfg = Config()
    set_seed(cfg.seed)

    print(f"Device: {cfg.device}")
    print(
        f"Dense decoder params: {cfg.input_dim * cfg.dict_size:,}  |  "
        f"Expander decoder params: {cfg.expand_degree * cfg.dict_size:,}  |  "
        f"Compression ratio: {cfg.input_dim / cfg.expand_degree:.1f}x"
    )

    # 生成数据
    F_true, train_x, test_x, train_codes, test_codes = make_synthetic_data(cfg)

    # 构建扩展图掩码：合成演示中可与真实特征支撑对齐
    # Build expander mask: align with ground-truth feature supports in the synthetic demo
    if cfg.aligned_mask:
        mask = torch.zeros(cfg.input_dim, cfg.dict_size, dtype=torch.float32, device=cfg.device)
        for j in range(cfg.dict_size):
            rows = torch.where(F_true[:, j].abs() > 1e-6)[0]
            if rows.numel() > cfg.expand_degree:
                rows = rows[torch.randperm(rows.numel(), device=cfg.device)[: cfg.expand_degree]]
            mask[rows, j] = 1.0
    else:
        mask = build_expander_mask(cfg.input_dim, cfg.dict_size, cfg.expand_degree, cfg.device)

    # 初始化模型
    model = ExpanderSAE(cfg, mask).to(cfg.device)
    n_learned = (model.W * model.mask).count_nonzero().item()
    print(f"Actually learned nonzero decoder values: {n_learned:,}")

    # 训练
    print("\nTraining Expander SAE...")
    train_expander_sae(cfg, model, train_x, test_x)

    # 最终评估
    final_mse = evaluate_mse(model, test_x, cfg.batch_size)
    recovery = evaluate_feature_recovery(F_true, model)
    print("\n=== Final evaluation / 最终评估 ===")
    print(f"Test MSE: {final_mse:.6f}")
    print(f"Mean max cosine recovery: {recovery:.4f}")
    print("(1.0 = perfect recovery, 0.0 = no alignment)")


if __name__ == "__main__":
    main()
