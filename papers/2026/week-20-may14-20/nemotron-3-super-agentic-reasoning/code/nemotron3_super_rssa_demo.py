"""
Nemotron 3 Super - 简化版 RSSA（Routed State-Space Attention）复现
Simplified reproduction of the Routed State-Space Attention (RSSA) block from
"Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer
Model for Agentic Reasoning" (arXiv:2605.04217).

Requirements / 环境要求:
    - PyTorch >= 2.1 (for nn.RMSNorm / 因为使用 nn.RMSNorm)
    - Python >= 3.10 (for `torch.Tensor | None` union syntax / 因为使用联合类型语法)

本脚本是一个可独立运行的 PyTorch 演示，展示论文第 4.1 节中提出的核心思想：
对序列中的每个 token，轻量路由器决定将其送往 (a) 局部的状态空间递归模块
(Mamba 风格) 还是 (b) 稀疏注意力专家；同时通过负载均衡辅助损失避免专家崩塌。

This script is a self-contained runnable PyTorch demo of the core idea from
Section 4.1 of the paper: a lightweight router decides, per token, whether to
route it to (a) a local state-space recurrence (Mamba-style) or (b) a sparse
attention expert, with a load-balancing auxiliary loss to prevent expert collapse.

运行方式 / How to run:
    python nemotron3_super_rssa_demo.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1. 工具函数 / Utility functions
# -----------------------------------------------------------------------------

def clones(module, n):
    """生成 n 个共享结构但不共享参数的模块副本。"""
    return nn.ModuleList([module for _ in range(n)])


# -----------------------------------------------------------------------------
# 2. 简化状态空间模块 (Mamba-风格离散化 SSM)
#    Simplified State-Space Module (Mamba-style discretized SSM)
# -----------------------------------------------------------------------------

class SimpleStateSpace(nn.Module):
    """
    一个极度简化的结构化状态空间序列模型（S4/S6 风格）。
    为了演示可运行性，这里使用标量对角化 A 与输入相关的步长 Δ，
    不实现完整 Mamba 的选择性扫描，但保留了“线性时间 + 循环状态”的核心特性。

    An extremely simplified structured state-space sequence model (S4/S6-style).
    For demo purposes it uses scalar diagonal A and input-dependent step sizes Δ,
    without the full Mamba selective scan, but keeps the "linear-time + recurrent
    state" core property.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # 输入到 x_t、B_t、C_t 的线性投影
        self.in_proj = nn.Linear(d_model, d_state * 3, bias=False)

        # 对角化状态矩阵 A（可学习，负值保证稳定）
        self.A = nn.Parameter(torch.randn(d_state) * 0.01 - 0.5)

        # 输出投影
        self.out_proj = nn.Linear(d_state, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        """
        参数 / Args:
            x: [batch, seq_len, d_model]
        返回 / Returns:
            y: [batch, seq_len, d_model]
        """
        bsz, seq_len, _ = x.shape

        # 将输入投影为 (x, B, C) 三个分量
        u = self.in_proj(x)                       # [b, l, 3 * d_state]
        u = u.view(bsz, seq_len, 3, self.d_state)  # [b, l, 3, d_state]
        x_ssm = u[:, :, 0]                        # [b, l, d_state]
        B = u[:, :, 1]                            # [b, l, d_state]
        C = u[:, :, 2]                            # [b, l, d_state]

        # 离散化：Δ 简化为固定小常数，便于演示
        delta = torch.sigmoid(self.A) * 0.1 + 1e-4

        # 离散化后的 A_bar, B_bar
        A_bar = torch.exp(delta * self.A)         # [d_state]
        B_bar = delta.unsqueeze(0) * B             # [b, l, d_state]

        # 线性递推：h_t = A_bar * h_{t-1} + B_bar_t * x_t
        h = torch.zeros(bsz, self.d_state, device=x.device, dtype=x.dtype)
        h_list = []
        for t in range(seq_len):
            h = A_bar * h + B_bar[:, t] * x_ssm[:, t]
            h_list.append(h)
        h_seq = torch.stack(h_list, dim=1)        # [b, l, d_state]

        # 输出 y_t = C_t * h_t
        y = (C * h_seq).sum(dim=-1, keepdim=True)  # [b, l, 1]
        # 广播到 d_state 后再投影回 d_model
        y = y.expand(-1, -1, self.d_state)
        y = self.out_proj(y)                       # [b, l, d_model]
        return y


# -----------------------------------------------------------------------------
# 3. 稀疏注意力专家 (Top-k 稀疏自注意力)
#    Sparse Attention Expert (Top-k sparse self-attention)
# -----------------------------------------------------------------------------

class SparseAttentionExpert(nn.Module):
    """
    每个 token 只关注与自身相似度最高的 top-k 个 token 的稀疏自注意力。
    复杂度从 O(L^2) 降到 O(L * k)，适合长序列中的全局依赖。

    Each token attends only to its top-k most similar tokens. Complexity drops
    from O(L^2) to O(L*k), suitable for global dependencies in long sequences.
    """

    def __init__(self, d_model: int, n_heads: int = 4, top_k: int = 8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.top_k = top_k

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        bsz, seq_len, _ = x.shape

        # 计算 Q, K, V 并拆分为多注意力头
        Q = self.W_q(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # -> [b, h, l, d_k]

        # 缩放点积相似度
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # [b, h, l, l]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 为每个查询只保留 top-k 个最大相似度的键，其余置为 -inf
        topk_vals, topk_idx = torch.topk(scores, k=min(self.top_k, seq_len), dim=-1)
        sparse_scores = torch.full_like(scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_idx, topk_vals)

        attn = F.softmax(sparse_scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, V)              # [b, h, l, d_k]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.W_o(out)
        return out


# -----------------------------------------------------------------------------
# 4. RSSA 块：路由 + 专家融合 + 负载均衡损失
#    RSSA Block: routing + expert fusion + load-balancing loss
# -----------------------------------------------------------------------------

class RSSABlock(nn.Module):
    """
    Routed State-Space Attention (RSSA) 块的简化实现。
    对输入序列中的每个 token，路由器输出两个 logits，分别对应：
        - 路径 0：SimpleStateSpace（局部递归，线性复杂度）
        - 路径 1：SparseAttentionExpert（稀疏全局注意力）
    最终输出是两条路径输出的软加权和，并在训练时返回负载均衡辅助损失。

    Simplified implementation of the RSSA block. For each token the router emits
    two logits corresponding to:
        - path 0: SimpleStateSpace (local recurrence, linear complexity)
        - path 1: SparseAttentionExpert (sparse global attention)
    The final output is a soft weighted sum of both paths, and a load-balancing
    auxiliary loss is returned during training.
    """

    def __init__(self, d_model: int, d_state: int = 16, n_heads: int = 4,
                 top_k: int = 8, lb_alpha: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.lb_alpha = lb_alpha

        self.router = nn.Linear(d_model, 2, bias=False)
        self.ssm = SimpleStateSpace(d_model, d_state=d_state)
        self.sparse_attn = SparseAttentionExpert(d_model, n_heads=n_heads, top_k=top_k)

        # 输出层归一化 + 前馈
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        参数 / Args:
            x:    [batch, seq_len, d_model]
            mask: [batch, 1, 1, seq_len] 或兼容的注意力掩码
        返回 / Returns:
            out:      [batch, seq_len, d_model]
            lb_loss:  标量辅助损失（训练时），推理时可忽略
        """
        bsz, seq_len, _ = x.shape

        # 4.1 路由打分 / Routing scores
        route_logits = self.router(x)             # [b, l, 2]
        route_probs = F.softmax(route_logits, dim=-1)  # [b, l, 2]

        # 4.2 两条专家路径 / Two expert paths
        ssm_out = self.ssm(x)                     # [b, l, d]
        attn_out = self.sparse_attn(x, mask=mask) # [b, l, d]

        # 4.3 软融合：每个 token 按 router 概率加权两条路径的输出
        #     Soft fusion: weight the two path outputs per token by router prob
        fused = route_probs[:, :, 0:1] * ssm_out + route_probs[:, :, 1:2] * attn_out

        # 残差连接 + 层归一化
        x = self.norm1(x + fused)

        # 前馈网络
        x = self.norm2(x + self.ffn(x))

        # 4.4 负载均衡辅助损失（参考 Switch Transformer / Mixtral 风格）
        #     Load-balancing auxiliary loss
        #     我们希望每个专家在所有 token 上的平均使用比例接近 0.5
        f = route_probs.mean(dim=(0, 1))          # [2], 每个专家在所有 token 上的平均使用率
        P = route_logits.softmax(dim=-1).mean(dim=(0, 1))  # [2], 平均路由概率
        # 损失 = alpha * num_experts * sum(f_i * P_i)
        lb_loss = self.lb_alpha * 2.0 * (f * P).sum()

        return x, lb_loss


# -----------------------------------------------------------------------------
# 5. 微型 Nemotron 3 Super 风格模型（多层 RSSA 堆叠）
#    Tiny Nemotron 3 Super-style model (stack of RSSA blocks)
# -----------------------------------------------------------------------------

class TinyNemotron3Super(nn.Module):
    """
    一个仅用于演示的非常小的 Nemotron 3 Super 风格语言模型。
    由嵌入层、若干 RSSA 块和输出头组成。

    A very small Nemotron 3 Super-style language model for demonstration only,
    consisting of an embedding layer, several RSSA blocks, and an output head.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 3,
                 d_state: int = 16, n_heads: int = 4, top_k: int = 8,
                 max_len: int = 256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            RSSABlock(d_model, d_state=d_state, n_heads=n_heads, top_k=top_k)
            for _ in range(n_layers)
        ])

        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # 构造因果下三角掩码，用于稀疏注意力
        # Build causal lower-triangular mask for sparse attention
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, l, l]

        total_lb_loss = 0.0
        for block in self.blocks:
            x, lb_loss = block(x, mask=mask)
            total_lb_loss = total_lb_loss + lb_loss

        x = self.norm(x)
        logits = self.lm_head(x)                # [b, l, vocab_size]
        return logits, total_lb_loss


# -----------------------------------------------------------------------------
# 6. 主函数：构造随机数据并执行一次前向传播
#    Main: build random data and run one forward pass
# -----------------------------------------------------------------------------

def main():
    # 超参数 / Hyperparameters
    VOCAB_SIZE = 1000
    BATCH_SIZE = 2
    SEQ_LEN = 32
    D_MODEL = 128
    N_LAYERS = 3

    torch.manual_seed(42)

    # 构造模型 / Build model
    model = TinyNemotron3Super(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_state=16,
        n_heads=4,
        top_k=8,
    )

    # 随机输入 token / Random input tokens
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    # 前向传播 / Forward pass
    logits, lb_loss = model(input_ids)

    print("=" * 60)
    print("Nemotron 3 Super - 简化 RSSA 复现 / Simplified RSSA reproduction")
    print("=" * 60)
    print(f"输入形状 / Input shape:        {input_ids.shape}")
    print(f"输出 logits 形状 / Logits shape: {logits.shape}")
    print(f"总参数量 / Total parameters:     {sum(p.numel() for p in model.parameters()):,}")
    print(f"负载均衡损失 / Load-balancing loss: {lb_loss.item():.6f}")
    print("=" * 60)
    print("前向传播成功完成 / Forward pass completed successfully.")

    # 演示：使用 logits 计算一个简单的 next-token 交叉熵损失
    # Demo: compute a simple next-token cross-entropy loss from logits
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    ce_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
    total_loss = ce_loss + lb_loss
    print(f"交叉熵损失 / Cross-entropy loss: {ce_loss.item():.4f}")
    print(f"总损失 (CE + LB) / Total loss:    {total_loss.item():.4f}")


if __name__ == "__main__":
    main()
