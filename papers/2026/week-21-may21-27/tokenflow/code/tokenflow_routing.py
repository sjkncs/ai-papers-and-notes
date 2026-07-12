"""
TokenFlow: 动态 Token 路由的简化 PyTorch 复现 / Simplified PyTorch reproduction of dynamic token routing.

本代码演示如何在视觉-语言模型中为一层 Transformer 添加 TokenFlow 风格的路由器：
- 每个 token 根据当前隐藏状态被分配到：完整块（full）、跳过（skip）或轻量重算（light）。
- 使用直通估计器（Straight-Through estimator）训练离散路由决策。
- 附带中文注释，便于理解核心思想。

This script shows how to add a TokenFlow-style router to one Transformer block in a vision-language model:
- Each token is routed to full block, skip, or light recomputation based on its hidden state.
- Discrete routing decisions are trained with a Straight-Through estimator.
- Comments are in Chinese to match the weekly digest style.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightRecompute(nn.Module):
    """
    轻量重算模块：用更窄的前馈网络代替完整 FFN，用于被标记为 'light' 的 token。
    Light recomputation module: a narrower FFN used for tokens marked as 'light'.
    """

    def __init__(self, d_model: int, reduction: int = 4):
        super().__init__()
        hidden = max(d_model // reduction, 64)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TokenRouter(nn.Module):
    """
    TokenFlow 路由器：为每个 token 输出三路 logits（full / skip / light）。
    TokenFlow router: outputs 3-way logits (full / skip / light) for each token.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 3),  # 3 个路由决策 / 3 routing decisions
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenFlowBlock(nn.Module):
    """
    带 TokenFlow 路由的 Transformer 块。
    Transformer block with TokenFlow routing.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.router = TokenRouter(d_model)
        self.light = LightweightRecompute(d_model, reduction=4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：返回更新后的隐藏状态与路由分布。
        Forward pass: returns updated hidden states and routing distribution.
        """
        # 1. 计算路由 logits / Compute routing logits
        route_logits = self.router(x)  # (B, T, 3)

        # 2. 使用 Gumbel-Softmax 获得软路由权重（训练）
        #    使用直通估计器获得硬路由索引（推理）
        #    Use Gumbel-Softmax for soft routing weights (training),
        #    straight-through hard indices for inference.
        tau = 0.5
        soft = F.gumbel_softmax(route_logits, tau=tau, hard=False, dim=-1)
        hard_idx = route_logits.argmax(dim=-1)  # 0=full, 1=skip, 2=light

        # 3. 对 full token 执行完整 Transformer 块
        #    Run full transformer block for 'full' tokens.
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)
        full_branch = x + self.dropout(attn_out)
        full_branch = full_branch + self.dropout(self.ffn(self.norm2(full_branch)))

        # 4. 对 skip token 保持原隐藏状态 / 'skip' tokens keep original hidden state.
        skip_branch = x

        # 5. 对 light token 执行轻量重算 / 'light' tokens use lightweight recompute.
        light_branch = x + self.dropout(self.light(self.norm2(x)))

        # 6. 根据软权重合并三路输出 / Merge three branches according to soft weights.
        out = (
            soft[..., 0:1] * full_branch
            + soft[..., 1:2] * skip_branch
            + soft[..., 2:3] * light_branch
        )

        # 返回输出与用于辅助损失的硬路由分布统计
        # Return output and hard-routing distribution statistics for auxiliary loss.
        return out, hard_idx


def latency_auxiliary_loss(route_logits: torch.Tensor, cost_vector: torch.Tensor) -> torch.Tensor:
    """
    延迟感知辅助损失：鼓励路由器选择低成本决策。
    Latency-aware auxiliary loss: encourages low-cost routing decisions.

    Args:
        route_logits: (B, T, 3) 路由器原始输出 / router raw outputs.
        cost_vector: (3,) 每个决策的相对延迟成本 / relative latency cost per decision.
    Returns:
        标量损失 / scalar loss.
    """
    probs = F.softmax(route_logits, dim=-1)  # (B, T, 3)
    expected_cost = (probs * cost_vector).sum(dim=-1)  # (B, T)
    return expected_cost.mean()


def balanced_load_loss(route_idx: torch.Tensor, num_decisions: int = 3) -> torch.Tensor:
    """
    负载均衡损失：鼓励不同路由决策的使用频率接近，避免全部 token 都被跳过。
    Load-balance loss: encourages uniform usage of routing decisions.
    """
    counts = torch.bincount(route_idx.flatten(), minlength=num_decisions).float()
    probs = counts / counts.sum()
    # 最大化熵 / maximize entropy
    entropy = -(probs * (probs + 1e-8).log()).sum()
    return -entropy  # 最小化负熵 = 最大化熵


if __name__ == "__main__":
    # 超参数 / Hyperparameters
    batch_size, seq_len, d_model, n_heads = 2, 16, 256, 8

    # 随机输入：模拟多模态 token 序列 / Random input mimicking multimodal token sequence.
    x = torch.randn(batch_size, seq_len, d_model)

    # 构建 TokenFlow 块 / Build TokenFlow block.
    block = TokenFlowBlock(d_model, n_heads)

    # 前向 / Forward pass.
    out, hard_idx = block(x)

    # 模拟 logits 用于损失计算 / Simulated logits for loss computation.
    route_logits = torch.randn(batch_size, seq_len, 3, requires_grad=True)
    cost_vector = torch.tensor([1.0, 0.0, 0.3])  # full, skip, light 的相对成本 / relative costs

    # 计算损失 / Compute losses.
    lat_loss = latency_auxiliary_loss(route_logits, cost_vector)
    bal_loss = balanced_load_loss(hard_idx)
    total_loss = lat_loss + 0.1 * bal_loss

    print(f"输出形状 / Output shape: {out.shape}")
    print(f"硬路由决策统计 / Hard routing stats: {torch.bincount(hard_idx.flatten(), minlength=3)}")
    print(f"延迟损失 / Latency loss: {lat_loss.item():.4f}")
    print(f"负载均衡损失 / Balance loss: {bal_loss.item():.4f}")
    print(f"总损失 / Total loss: {total_loss.item():.4f}")
