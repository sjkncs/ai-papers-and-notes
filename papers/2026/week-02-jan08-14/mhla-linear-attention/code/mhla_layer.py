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
MHLA: Multi-Head Linear Attention Implementation
MHLA: 多头线性注意力实现

Self-contained PyTorch implementation of the Token-Level Multi-Head (MHLA)
mechanism for restoring expressivity of linear attention. Features linear
attention with token-level multi-head projection, achieving near-softmax
quality at O(n) complexity.

自包含的PyTorch实现，包含Token级多头（MHLA）机制用于恢复线性注意力
的表达能力。以O(n)复杂度实现接近softmax质量的注意力。

Reference / 参考文献:
    Zhang, K. et al. "MHLA: Restoring Expressivity of Linear Attention
    via Token-Level Multi-Head." arXiv:2601.07832 (2026).
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration / 配置
# ============================================================================

@dataclass
class MHLAConfig:
    """
    Configuration for MHLA model.
    MHLA模型配置。
    """

    # 词表大小 / Vocabulary size
    vocab_size: int = 32000

    # 隐藏层维度 / Hidden dimension
    hidden_dim: int = 768

    # Transformer层数 / Number of transformer layers
    num_layers: int = 12

    # 注意力头数 / Number of attention heads
    num_heads: int = 12

    # 每个token的投影头数（MHLA核心参数）
    # Number of projection heads per token (core MHLA parameter)
    token_heads: int = 4

    # 线性注意力核函数类型 / Kernel function type for linear attention
    # 可选 / Options: "elu_plus", "relu", "rff"
    kernel_type: str = "elu_plus"

    # 特征维度（线性注意力投影维度）/ Feature dim (linear attention projection dim)
    feature_dim: int = 64

    # FFN中间层缩放因子 / FFN intermediate scaling factor
    ffn_scale: float = 4.0

    # 最大序列长度 / Maximum sequence length
    max_seq_len: int = 2048

    # Dropout率 / Dropout rate
    dropout: float = 0.1

    # 层归一化epsilon / Layer norm epsilon
    layer_norm_eps: float = 1e-5


# ============================================================================
# Linear Attention Kernels / 线性注意力核函数
# ============================================================================

class LinearAttentionKernel(nn.Module):
    """
    Kernel function for linear attention approximation.
    线性注意力近似的核函数。

    Supports multiple kernel types:
    支持多种核函数类型：
    - elu_plus: ELU+1 kernel / ELU+1核
    - relu: ReLU kernel / ReLU核
    - rff: Random Fourier Features / 随机傅里叶特征
    """

    def __init__(self, head_dim: int, feature_dim: int, kernel_type: str = "elu_plus"):
        super().__init__()
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.kernel_type = kernel_type

        if kernel_type == "rff":
            # 随机傅里叶特征需要可学习的投影矩阵
            # Random Fourier features require learnable projection matrix
            self.projection = nn.Linear(head_dim, feature_dim, bias=False)
            nn.init.normal_(self.projection.weight, std=1.0 / math.sqrt(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply kernel function to input.
        对输入应用核函数。

        Args:
            x: Input tensor [..., head_dim]

        Returns:
            Feature-mapped tensor [..., feature_dim]
        """
        if self.kernel_type == "elu_plus":
            # ELU+1核：φ(x) = elu(x) + 1
            return F.elu(x) + 1.0

        elif self.kernel_type == "relu":
            # ReLU核：φ(x) = max(0, x) + epsilon
            return F.relu(x) + 1e-6

        elif self.kernel_type == "rff":
            # 随机傅里叶特征：φ(x) = cos(Wx + b)
            # Random Fourier features
            projected = self.projection(x)
            return torch.cos(projected) / math.sqrt(self.feature_dim)

        else:
            raise ValueError(
                f"Unknown kernel type / 未知核函数类型: {self.kernel_type}. "
                f"Choose from: elu_plus, relu, rff"
            )


# ============================================================================
# Token-Level Multi-Head / Token级多头
# ============================================================================

class TokenLevelMultiHead(nn.Module):
    """
    Core MHLA mechanism: Token-level multi-head projection.
    核心MHLA机制：Token级多头投影。

    Unlike standard multi-head attention which applies different heads at the
    sequence level, MHLA applies multiple projection heads PER TOKEN before
    the linear attention recurrence. This restores token-specific attention
    distributions that standard linear attention lacks.

    与标准的多头注意力（在序列级别应用不同头）不同，MHLA在线性注意力递推
    之前为每个TOKEN应用多个投影头。这恢复了标准线性注意力所缺乏的
    token级特定注意力分布。
    """

    def __init__(self, config: MHLAConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.token_heads = config.token_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.feature_dim = config.feature_dim

        # Token级投影：每个token有token_heads个投影子空间
        # Token-level projection: each token has token_heads projection subspaces
        self.token_proj = nn.Linear(
            self.head_dim,
            self.token_heads * self.feature_dim,
            bias=False,
        )

        # 核函数 / Kernel function for each token head
        self.kernels = nn.ModuleList([
            LinearAttentionKernel(self.feature_dim, self.feature_dim, config.kernel_type)
            for _ in range(self.token_heads)
        ])

        # 输出投影 / Output projection
        self.out_proj = nn.Linear(
            self.token_heads * self.feature_dim,
            self.head_dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply token-level multi-head projection.
        应用token级多头投影。

        Args:
            x: Input per-head features [B, num_heads, seq_len, head_dim]

        Returns:
            Token-level multi-head output [B, num_heads, seq_len, head_dim]
        """
        B, H, S, D = x.shape

        # 投影到token级多头空间 / Project to token-level multi-head space
        # [B, H, S, D] -> [B, H, S, token_heads * feature_dim]
        projected = self.token_proj(x)
        # 重塑为 [B, H, S, token_heads, feature_dim]
        projected = projected.view(B, H, S, self.token_heads, self.feature_dim)

        # 对每个token头应用核函数 / Apply kernel function to each token head
        token_features = []
        for th in range(self.token_heads):
            # [B, H, S, feature_dim]
            feat = self.kernels[th](projected[..., th, :])
            token_features.append(feat)

        # 拼接所有token头的特征 / Concatenate features from all token heads
        # [B, H, S, token_heads * feature_dim]
        combined = torch.cat(token_features, dim=-1)

        # 输出投影回原始维度 / Project output back to original dimension
        output = self.out_proj(combined)
        return output


# ============================================================================
# MHLA Linear Attention / MHLA线性注意力
# ============================================================================

class MHLALinearAttention(nn.Module):
    """
    Linear attention with token-level multi-head mechanism.
    带token级多头机制的线性注意力。

    Achieves O(n) complexity while recovering near-softmax expressivity
    through the MHLA mechanism.
    通过MHLA机制以O(n)复杂度恢复接近softmax的表达能力。
    """

    def __init__(self, config: MHLAConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.feature_dim = config.feature_dim

        # 查询、键、值投影 / Query, Key, Value projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # 核函数（用于标准线性注意力）/ Kernel for standard linear attention
        self.kernel = LinearAttentionKernel(
            self.head_dim, self.feature_dim, config.kernel_type
        )

        # Token级多头机制 / Token-level multi-head mechanism
        self.token_mh = TokenLevelMultiHead(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for MHLA linear attention.
        MHLA线性注意力的前向传播。

        Uses the token-level multi-head mechanism to enhance expressivity
        while maintaining O(n) complexity via linear attention recurrence.

        使用token级多头机制增强表达能力，同时通过线性注意力递推保持O(n)复杂度。

        Args:
            hidden_states: Input [B, seq_len, hidden_dim]
            causal: Whether to apply causal masking / 是否应用因果掩码

        Returns:
            Output [B, seq_len, hidden_dim]
        """
        B, S, _ = hidden_states.shape

        # 线性投影 / Linear projections
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)

        # 转置为 [B, heads, seq, dim] / Transpose to [B, heads, seq, dim]
        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 应用token级多头机制 / Apply token-level multi-head mechanism
        q_enhanced = self.token_mh(q)
        k_enhanced = self.token_mh(k)

        # 应用核函数映射到特征空间 / Apply kernel to map to feature space
        q_feat = self.kernel(q_enhanced)  # [B, H, S, feature_dim]
        k_feat = self.kernel(k_enhanced)

        # 线性注意力计算（O(n)复杂度）/ Linear attention computation (O(n))
        if causal:
            output = self._causal_linear_attention(q_feat, k_feat, v)
        else:
            output = self._full_linear_attention(q_feat, k_feat, v)

        # 重塑并投影输出 / Reshape and project output
        output = output.transpose(1, 2).contiguous()  # [B, S, H*D]
        output = output.view(B, S, -1)
        return self.o_proj(output)

    def _causal_linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Causal linear attention via cumulative sum (prefix sum).
        通过累积和（前缀和）实现因果线性注意力。

        Complexity: O(n * d * d_v) where n=seq_len, d=feature_dim, d_v=head_dim
        复杂度：O(n * d * d_v)

        Args:
            q: Query features [B, H, S, feature_dim]
            k: Key features [B, H, S, feature_dim]
            v: Values [B, H, S, head_dim]

        Returns:
            Attention output [B, H, S, head_dim]
        """
        # kv = outer product of k and v at each position
        # kv = 每个位置上k和v的外积
        # [B, H, S, feature_dim, head_dim]
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)

        # 前缀和实现因果性 / Prefix sum for causality
        # cumulative_kv[t] = sum_{s=0}^{t} k_s * v_s^T
        cumulative_kv = torch.cumsum(kv, dim=2)

        # 累计归一化因子 / Cumulative normalization factor
        cumulative_k = torch.cumsum(k, dim=2)  # [B, H, S, feature_dim]

        # 注意力输出 / Attention output
        # output[t] = (q[t] @ cumulative_kv[t]) / (q[t] @ cumulative_k[t])
        numerator = torch.einsum("bhse,bhsev->bhsv", q, cumulative_kv)
        denominator = torch.einsum("bhse,bhse->bhs", q, cumulative_k)
        denominator = denominator.unsqueeze(-1).clamp(min=1e-6)

        return numerator / denominator

    def _full_linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Non-causal (bidirectional) linear attention.
        非因果（双向）线性注意力。

        Args:
            q: Query features [B, H, S, feature_dim]
            k: Key features [B, H, S, feature_dim]
            v: Values [B, H, S, head_dim]

        Returns:
            Attention output [B, H, S, head_dim]
        """
        # KV上下文矩阵 / KV context matrix
        # [B, H, feature_dim, head_dim]
        kv_context = torch.einsum("bhse,bhsd->bhed", k, v)
        # 归一化 / Normalization
        k_sum = k.sum(dim=2, keepdim=True)  # [B, H, 1, feature_dim]

        numerator = torch.einsum("bhse,bhed->bhsd", q, kv_context)
        denominator = torch.einsum("bhse,bhse->bhs", q, k_sum.expand_as(k))
        denominator = denominator.unsqueeze(-1).clamp(min=1e-6)

        return numerator / denominator


# ============================================================================
# Transformer Block / Transformer块
# ============================================================================

class MHLATransformerBlock(nn.Module):
    """
    Transformer block with MHLA linear attention.
    使用MHLA线性注意力的Transformer块。

    Architecture: LayerNorm -> MHLA Attention -> Residual ->
                  LayerNorm -> FFN -> Residual
    """

    def __init__(self, config: MHLAConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.attention = MHLALinearAttention(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

        # 标准SwiGLU FFN / Standard SwiGLU FFN
        intermediate_dim = int(config.hidden_dim * config.ffn_scale)
        self.ffn_gate = nn.Linear(config.hidden_dim, intermediate_dim, bias=False)
        self.ffn_up = nn.Linear(config.hidden_dim, intermediate_dim, bias=False)
        self.ffn_down = nn.Linear(intermediate_dim, config.hidden_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Forward pass through one MHLA transformer block."""
        # 注意力 + 残差 / Attention + residual
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attention(hidden_states, causal=causal)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # FFN + 残差 / FFN + residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        gate = F.silu(self.ffn_gate(hidden_states))
        up = self.ffn_up(hidden_states)
        hidden_states = self.ffn_down(gate * up)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================================
# Full MHLA Language Model / 完整MHLA语言模型
# ============================================================================

class MHLAModel(nn.Module):
    """
    Complete language model using MHLA linear attention.
    使用MHLA线性注意力的完整语言模型。

    Replaces all softmax attention layers with MHLA linear attention,
    achieving O(n) per-layer complexity with near-softmax quality.
    将所有softmax注意力层替换为MHLA线性注意力，
    以O(n)每层复杂度实现接近softmax的质量。
    """

    def __init__(self, config: Optional[MHLAConfig] = None):
        super().__init__()
        if config is None:
            config = MHLAConfig()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([
            MHLATransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # 权重绑定 / Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights / 初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the MHLA language model.
        通过MHLA语言模型的前向传播。

        Args:
            input_ids: Token IDs [B, seq_len]
            labels: Optional labels for loss [B, seq_len]

        Returns:
            Tuple of (logits, optional loss)
        """
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, causal=True)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss


# ============================================================================
# Comparison Utility / 比较工具
# ============================================================================

def compare_with_softmax(
    seq_lengths: Optional[list] = None,
    hidden_dim: int = 256,
    num_heads: int = 4,
    device: str = "cpu",
):
    """
    Compare MHLA linear attention vs standard softmax attention
    in terms of latency and memory usage.
    比较MHLA线性注意力与标准softmax注意力的延迟和内存使用。

    Args:
        seq_lengths: List of sequence lengths to test / 要测试的序列长度列表
        hidden_dim: Hidden dimension / 隐藏维度
        num_heads: Number of heads / 头数
        device: Device to run on / 运行设备
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048]

    print(f"\n{'='*70}")
    print(f"MHLA vs Softmax Attention Comparison / MHLA与Softmax注意力对比")
    print(f"Hidden dim / 隐藏维度: {hidden_dim}, Heads / 头数: {num_heads}")
    print(f"{'='*70}")

    config = MHLAConfig(
        hidden_dim=hidden_dim, num_heads=num_heads,
        num_layers=1, feature_dim=hidden_dim // num_heads,
        token_heads=4, vocab_size=1000,
    )

    # 创建MHLA注意力 / Create MHLA attention
    mhla_attn = MHLALinearAttention(config).to(device)

    print(f"\n{'Seq Len':>10} | {'MHLA (ms)':>12} | {'Softmax (ms)':>14} | {'Speedup':>10}")
    print(f"{'序列长度':>10} | {'MHLA延迟':>12} | {'Softmax延迟':>14} | {'加速比':>10}")
    print("-" * 55)

    batch_size = 4
    warmup_iters = 5
    measure_iters = 20

    for seq_len in seq_lengths:
        hidden = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        # 预热 / Warmup
        for _ in range(warmup_iters):
            _ = mhla_attn(hidden, causal=True)
            # Softmax attention baseline / Softmax注意力基线
            q = hidden.view(batch_size, seq_len, num_heads, -1).transpose(1, 2)
            scores = torch.matmul(q, q.transpose(-2, -1)) / math.sqrt(hidden_dim // num_heads)
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), 1)
            scores = F.softmax(scores + mask, dim=-1)
            _ = torch.matmul(scores, q)

        torch.cuda.synchronize() if device == "cuda" else None

        # 测量MHLA / Measure MHLA
        start = time.time()
        for _ in range(measure_iters):
            _ = mhla_attn(hidden, causal=True)
        torch.cuda.synchronize() if device == "cuda" else None
        mhla_time = (time.time() - start) / measure_iters * 1000

        # 测量Softmax / Measure Softmax
        start = time.time()
        for _ in range(measure_iters):
            q = hidden.view(batch_size, seq_len, num_heads, -1).transpose(1, 2)
            scores = torch.matmul(q, q.transpose(-2, -1)) / math.sqrt(hidden_dim // num_heads)
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), 1)
            scores = F.softmax(scores + mask, dim=-1)
            _ = torch.matmul(scores, q)
        torch.cuda.synchronize() if device == "cuda" else None
        softmax_time = (time.time() - start) / measure_iters * 1000

        speedup = softmax_time / max(mhla_time, 0.001)
        print(f"{seq_len:>10} | {mhla_time:>10.2f}  | {softmax_time:>12.2f}  | {speedup:>8.2f}x")

    print(f"\nNote / 注意: MHLA advantage grows with sequence length (O(n) vs O(n²))")
    print(f"MHLA的优势随序列长度增长（O(n) vs O(n²)）\n")


# ============================================================================
# Main / 主程序
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MHLA Linear Attention Demo / MHLA线性注意力演示"
    )
    parser.add_argument(
        "--action", type=str, default="compare",
        choices=["compare", "info", "train-smoke"],
        help="Action to perform / 执行操作"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device / 设备")
    args = parser.parse_args()

    if args.action == "info":
        config = MHLAConfig()
        model = MHLAModel(config)
        total = sum(p.numel() for p in model.parameters())
        print(f"MHLA Model (base config) / MHLA模型（基础配置）:")
        print(f"  Total params / 总参数: {total:,} ({total/1e6:.1f}M)")
        print(f"  Token heads / Token头数: {config.token_heads}")
        print(f"  Kernel / 核函数: {config.kernel_type}")
        print(f"  Feature dim / 特征维度: {config.feature_dim}")

    elif args.action == "compare":
        compare_with_softmax(device=args.device)

    elif args.action == "train-smoke":
        config = MHLAConfig(
            hidden_dim=128, num_layers=2, num_heads=4,
            feature_dim=32, token_heads=4, vocab_size=500,
        )
        model = MHLAModel(config).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        total = sum(p.numel() for p in model.parameters())
        print(f"MHLA Smoke Test / 冒烟测试 ({total:,} params)")

        for step in range(10):
            ids = torch.randint(0, 500, (2, 64), device=args.device)
            logits, loss = model(ids, labels=ids)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"  Step {step+1:2d} | Loss: {loss.item():.4f}")

        print("Done / 完成!")
