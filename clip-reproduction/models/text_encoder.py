"""
CLIP 文本编码器 — 基于因果 Transformer
CLIP Text Encoder — Causal Transformer Based
=============================================

论文参考 / Paper Reference:
  Radford et al., "Learning Transferable Visual Models From Natural
  Language Supervision", ICML 2021.

本模块实现 / This module implements:
  1. TokenEmbedding      — 词表嵌入 / Vocabulary embedding
  2. PositionalEncoding  — 可学习位置编码 / Learnable positional encoding
  3. CausalTransformerBlock — 带因果掩码的Transformer块 / Transformer block with causal mask
  4. TextEncoder          — 完整文本编码器，EOS token池化 / Full text encoder with EOS token pooling

与图像编码器的关键区别 / Key differences from Image Encoder:
  - 使用因果注意力 (上三角掩码)，每个token只能看到之前的token
    Uses causal attention (upper-triangular mask), each token only sees previous tokens
  - 使用EOS (End-Of-Sequence) token位置的输出作为文本表示
    Uses EOS token position output as text representation
"""

import torch
import torch.nn as nn
import math
from typing import Optional


# ============================================================
# 多头自注意力 (支持因果掩码)
# Multi-Head Self-Attention (with causal mask support)
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    多头自注意力，支持可选因果掩码。
    Multi-head self-attention with optional causal masking.

    Args:
        embed_dim (int):  输入/输出维度 / Input/output dimension.
        num_heads (int):  注意力头数 / Number of attention heads.
        attn_dropout (float): 注意力dropout / Attention dropout.
        proj_dropout (float): 输出投影dropout / Output projection dropout.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) 必须被 num_heads ({num_heads}) 整除 / "
            f"embed_dim must be divisible by num_heads"
        )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """
        前向传播 / Forward pass.

        Args:
            x:      [B, N, C] 输入序列 / Input sequence.
            causal: 是否使用因果掩码 / Whether to use causal mask.

        Returns:
            [B, N, C] 输出 / Output.
        """
        B, N, C = x.shape

        # Q, K, V 联合投影 / Joint Q, K, V projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 缩放点积注意力 / Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 因果掩码: 上三角矩阵为True的位置被屏蔽 (设为-inf)
        # Causal mask: positions in the upper triangle are masked (set to -inf)
        # 这确保位置i只能关注位置 ≤ i
        # This ensures position i can only attend to positions <= i
        if causal:
            causal_mask = torch.triu(
                torch.ones(N, N, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 加权求和 / Weighted sum
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


# ============================================================
# 因果 Transformer 块 / Causal Transformer Block
# ============================================================

class CausalTransformerBlock(nn.Module):
    """
    带因果注意力的 Transformer 编码器块 (Pre-LN)。
    Transformer encoder block with causal attention (Pre-LN).

    与图像编码器的TransformerBlock类似，但自注意力层使用因果掩码。
    Similar to the image encoder's TransformerBlock, but the self-attention
    layer uses a causal mask.

    Args:
        embed_dim (int):   隐藏维度 / Hidden dimension.
        num_heads (int):   注意力头数 / Number of attention heads.
        mlp_ratio (float): MLP扩展倍率 / MLP expansion ratio.
        dropout (float):   Dropout概率 / Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim, num_heads,
            attn_dropout=dropout,
            proj_dropout=dropout,
        )

        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        # MLP: 两层线性 + GELU 激活 / Two linear layers + GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (因果注意力 + 残差) / Forward pass (causal attention + residual).

        Args:
            x: [B, N, C]

        Returns:
            [B, N, C]
        """
        # 因果自注意力 + 残差 / Causal self-attention + residual
        x = x + self.attn(self.norm1(x), causal=True)

        # 前馈网络 + 残差 / Feed-forward network + residual
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# 完整文本编码器 / Complete Text Encoder
# ============================================================

class TextEncoder(nn.Module):
    """
    CLIP 文本编码器 / CLIP Text Encoder.

    工作流程 / Workflow:
      1. Token Embedding: 将token IDs映射为稠密向量 / Map token IDs to dense vectors
      2. Positional Encoding: 加入可学习的位置信息 / Add learnable positional information
      3. Causal Transformer: 多层因果Transformer编码 / Multi-layer causal Transformer encoding
      4. EOS Pooling: 取EOS token位置的输出作为文本表示 / Take EOS token position output as text representation

    原始CLIP使用 / Original CLIP uses:
      - vocab_size=49408 (BPE tokenizer)
      - max_seq_len=77 (SOS + 75 tokens + EOS)
      - embed_dim=512, 12 layers, 8 heads (for ViT-B/32)

    特殊token / Special tokens:
      - SOS (Start-Of-Sequence): token_id = 0, 序列起始标记
      - EOS (End-Of-Sequence):   token_id = 49407, 序列结束标记
      本实现中EOS位置通过查找最后一个非零token来确定。
      In this implementation, EOS position is determined by finding the last
      non-zero token.

    Args:
        vocab_size (int):   词表大小 / Vocabulary size.
        embed_dim (int):    嵌入维度 / Embedding dimension.
        num_heads (int):    注意力头数 / Number of attention heads.
        depth (int):        Transformer层数 / Number of Transformer layers.
        max_seq_len (int):  最大序列长度 / Maximum sequence length.
        mlp_ratio (float):  MLP扩展倍率 / MLP expansion ratio.
        dropout (float):    Dropout概率 / Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 512,
        num_heads: int = 8,
        depth: int = 6,
        max_seq_len: int = 77,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token嵌入层 / Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # 可学习的位置编码 / Learnable positional encoding
        # 与图像编码器不同，文本编码器的位置编码覆盖整个max_seq_len
        # Unlike image encoder, text encoder's positional encoding covers full max_seq_len
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # 因果Transformer块堆栈 / Stack of causal Transformer blocks
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # 最终归一化 / Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # 权重初始化 / Weight initialization
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        编码文本token序列为固定维度向量。
        Encode text token sequence into a fixed-dimension vector.

        Args:
            text_ids: [B, seq_len] token ID序列。
                      期望格式: [SOS, token1, token2, ..., EOS, PAD, PAD, ...]
                      Token ID sequence.
                      Expected format: [SOS, token1, token2, ..., EOS, PAD, PAD, ...]

        Returns:
            [B, embed_dim] 文本特征 (EOS token位置的输出)
            Text features (output at EOS token position).
        """
        B, seq_len = text_ids.shape

        # 1. Token嵌入: [B, seq_len] -> [B, seq_len, embed_dim]
        #    Token embedding
        x = self.token_embedding(text_ids)

        # 2. 加入位置编码 / Add positional encoding
        #    注意: 如果seq_len < max_seq_len，只取前seq_len个位置
        #    Note: if seq_len < max_seq_len, only take first seq_len positions
        x = self.pos_dropout(x + self.pos_embed[:, :seq_len, :])

        # 3. 通过因果Transformer块 / Pass through causal Transformer blocks
        #    因果掩码确保每个位置只能看到它自己和之前的位置
        #    Causal mask ensures each position only sees itself and prior positions
        for block in self.blocks:
            x = block(x)

        # 4. 最终归一化 / Final normalization
        x = self.norm(x)

        # 5. EOS池化: 找到每个样本中最后一个非零token (EOS) 的位置
        #    EOS Pooling: find the last non-zero token (EOS) position for each sample
        #    这比简单取最后一个位置更准确，因为序列可能有padding
        #    This is more accurate than simply taking the last position,
        #    as the sequence may have padding
        eos_indices = self._find_eos_positions(text_ids)  # [B]

        # 收集每个样本EOS位置的输出 / Gather output at EOS position for each sample
        # x[batch_idx, eos_indices[batch_idx]] for each batch_idx
        batch_indices = torch.arange(B, device=x.device)
        text_features = x[batch_indices, eos_indices]  # [B, embed_dim]

        return text_features

    @staticmethod
    def _find_eos_positions(text_ids: torch.Tensor) -> torch.Tensor:
        """
        找到每个样本中EOS token的位置。
        Find EOS token position for each sample.

        策略 / Strategy:
          - 从右向左查找第一个非零token的位置 (假设0是padding)
            Scan right-to-left for the first non-zero token (assuming 0 is padding)
          - 如果找不到非零token，返回最后一个位置
            If no non-zero token found, return the last position

        Args:
            text_ids: [B, seq_len] token IDs.

        Returns:
            [B] 每个样本的EOS位置索引 / EOS position index for each sample.
        """
        # 非零位置掩码 / Non-zero position mask
        nonzero_mask = text_ids != 0  # [B, seq_len]

        # 将True/False转为int，然后找每行最后一个True的位置
        # Convert True/False to int, then find last True position per row
        # 技巧: 乘以位置索引，取最大值
        # Trick: multiply by position indices, take max
        seq_len = text_ids.shape[1]
        positions = torch.arange(seq_len, device=text_ids.device).unsqueeze(0)  # [1, seq_len]
        masked_positions = positions * nonzero_mask.long()  # [B, seq_len]

        # 每行最大值即为最后一个非零元素的位置
        # Max per row gives the position of the last non-zero element
        eos_positions = masked_positions.max(dim=1).values  # [B]

        return eos_positions
