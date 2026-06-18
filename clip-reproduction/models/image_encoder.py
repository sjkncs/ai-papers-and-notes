"""
CLIP 图像编码器 — 基于 Vision Transformer (ViT)
CLIP Image Encoder — Vision Transformer (ViT) Based
====================================================

论文参考 / Paper Reference:
  Radford et al., "Learning Transferable Visual Models From Natural
  Language Supervision", ICML 2021.

本模块实现 / This module implements:
  1. PatchEmbedding  — 将图像切分为patches并线性嵌入 / Split image into patches and linearly embed
  2. MultiHeadAttention — 多头自注意力 / Multi-head self-attention
  3. TransformerBlock   — 含残差连接的Transformer编码器块 / Transformer encoder block with residuals
  4. ImageEncoder       — 完整ViT编码器，使用CLS token聚合 / Full ViT encoder with CLS token pooling
"""

import torch
import torch.nn as nn
import math
from typing import Optional


# ============================================================
# Patch 嵌入 / Patch Embedding
# ============================================================

class PatchEmbedding(nn.Module):
    """
    将图像切分为固定大小的patches并通过卷积线性嵌入。
    Split an image into fixed-size patches and linearly embed via convolution.

    原理 / Rationale:
      原始CLIP使用 image_size=224, patch_size=16 → 196 patches。
      Original CLIP uses image_size=224, patch_size=16 -> 196 patches.

      Conv2d(kernel=patch_size, stride=patch_size) 等效于先切patch再做线性投影。
      Conv2d(kernel=patch_size, stride=patch_size) is equivalent to
      non-overlapping patch extraction followed by a linear projection.

    Args:
        image_size (int): 输入图像的高宽 / Input image height/width.
        patch_size (int): 每个patch的高宽 / Each patch's height/width.
        in_channels (int): 输入通道数 (RGB=3) / Input channels (RGB=3).
        embed_dim (int):  嵌入维度 / Embedding dimension.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 使用步长卷积实现非重叠patch投影
        # Use strided convolution for non-overlapping patch projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward pass.

        Args:
            x: [B, C, H, W] 输入图像 / Input images.

        Returns:
            [B, num_patches, embed_dim] patch嵌入序列 / Patch embedding sequence.
        """
        # 卷积投影: [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        # Conv projection
        x = self.proj(x)

        # 展平空间维度: [B, embed_dim, H/P, W/P] -> [B, embed_dim, num_patches]
        # Flatten spatial dims
        x = x.flatten(2)

        # 转置为序列格式: [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        # Transpose to sequence format
        x = x.transpose(1, 2)
        return x


# ============================================================
# 多头自注意力 / Multi-Head Self-Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制。
    Multi-head self-attention mechanism.

    原理 / Rationale:
      将输入拆分为多个"头"，每个头独立计算缩放点积注意力，
      最后拼接输出。这允许模型同时关注不同位置的不同表示子空间。

      Split input into multiple "heads"; each head independently computes
      scaled dot-product attention, then outputs are concatenated. This
      allows the model to jointly attend to information from different
      representation subspaces at different positions.

    Args:
        embed_dim (int):  输入/输出维度 / Input/output dimension.
        num_heads (int):  注意力头数 / Number of attention heads.
        attn_dropout (float): 注意力权重dropout / Attention weight dropout.
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
        self.scale = self.head_dim ** -0.5  # 缩放因子 / Scaling factor: 1/sqrt(d_k)

        # Q/K/V 合并投影 (效率更高) / Combined Q/K/V projection (more efficient)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 / Forward pass.

        Args:
            x:    [B, N, C] 输入序列 / Input sequence.
            mask: [N, N] 可选注意力掩码 / Optional attention mask (True = masked out).

        Returns:
            [B, N, C] 注意力输出 / Attention output.
        """
        B, N, C = x.shape

        # 生成 Q, K, V: [B, N, 3*C] -> [3, B, heads, N, head_dim]
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 缩放点积注意力 / Scaled dot-product attention
        # attn: [B, heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 如果提供了掩码，将masked位置设为负无穷
        # If mask provided, set masked positions to -inf
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 加权求和并拼接多头: [B, heads, N, head_dim] -> [B, N, C]
        # Weighted sum and concatenate heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


# ============================================================
# Transformer 编码器块 / Transformer Encoder Block
# ============================================================

class TransformerBlock(nn.Module):
    """
    预归一化 Transformer 编码器块 (Pre-LN)。
    Pre-normalization Transformer encoder block.

    原理 / Rationale:
      使用 Pre-LN 架构 (先 LayerNorm 再 Attention/MLP)，相比 Post-LN
      训练更稳定，梯度流动更顺畅。这是ViT和原始CLIP采用的方式。

      Uses Pre-LN architecture (LayerNorm before Attention/MLP). Compared
      to Post-LN, it trains more stably with better gradient flow. This
      is the approach used by ViT and the original CLIP.

    Args:
        embed_dim (int):   隐藏维度 / Hidden dimension.
        num_heads (int):   注意力头数 / Number of attention heads.
        mlp_ratio (float): MLP中间层扩展倍率 / MLP intermediate expansion ratio.
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

        # 子层1: 多头自注意力 / Sub-layer 1: Multi-head self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads,
                                       attn_dropout=dropout,
                                       proj_dropout=dropout)

        # 子层2: 前馈网络 (MLP) / Sub-layer 2: Feed-forward network (MLP)
        # GELU 激活函数比 ReLU 更平滑，有利于Transformer训练
        # GELU activation is smoother than ReLU, beneficial for Transformer training
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播，含残差连接 / Forward pass with residual connections.

        Args:
            x:    [B, N, C] 输入 / Input.
            mask: 可选注意力掩码 / Optional attention mask.

        Returns:
            [B, N, C] 输出 / Output.
        """
        # 残差 + 注意力 / Residual + Attention
        x = x + self.attn(self.norm1(x), mask=mask)

        # 残差 + MLP / Residual + MLP
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# 完整图像编码器 / Complete Image Encoder (ViT)
# ============================================================

class ImageEncoder(nn.Module):
    """
    基于 Vision Transformer 的 CLIP 图像编码器。
    ViT-based CLIP Image Encoder.

    工作流程 / Workflow:
      1. PatchEmbedding 将图像切分为patches / Split image into patches
      2. 添加可学习的 CLS token / Prepend learnable CLS token
      3. 加入可学习的位置编码 / Add learnable positional encoding
      4. 经过N层 TransformerBlock / Pass through N TransformerBlocks
      5. 取 CLS token 的输出作为整张图像的表示 / Use CLS token output as image representation

    论文细节 / Paper Detail:
      原始CLIP的ViT-B/32使用 embed_dim=768, depth=12, num_heads=12。
      本实现为简化版，参数可通过配置调整。
      Original CLIP ViT-B/32 uses embed_dim=768, depth=12, num_heads=12.
      This is a simplified implementation with configurable parameters.

    Args:
        image_size (int):  输入图像尺寸 / Input image size.
        patch_size (int):  Patch大小 / Patch size.
        embed_dim (int):   隐藏维度 / Hidden dimension.
        num_heads (int):   注意力头数 / Number of attention heads.
        depth (int):       Transformer层数 / Number of Transformer layers.
        mlp_ratio (float): MLP扩展倍率 / MLP expansion ratio.
        dropout (float):   Dropout概率 / Dropout probability.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 512,
        num_heads: int = 8,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch嵌入层 / Patch embedding layer
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token: 可学习的分类token，拼接在patch序列前面
        # Learnable classification token, prepended to the patch sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 可学习的位置编码: 1 (CLS) + num_patches 个位置向量
        # Learnable positional encoding: 1 (CLS) + num_patches position vectors
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer编码器块堆栈 / Stack of Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # 最终归一化层 / Final normalization layer
        self.norm = nn.LayerNorm(embed_dim)

        # 权重初始化 / Weight initialization
        # 使用正态分布初始化，标准差0.02 (BERT/ViT常用)
        # Normal distribution init with std=0.02 (common in BERT/ViT)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码图像为固定维度向量 / Encode image into fixed-dimension vector.

        Args:
            x: [B, 3, H, W] 输入图像批次 / Batch of input images.

        Returns:
            [B, embed_dim] 图像特征 (CLS token输出) / Image features (CLS token output).
        """
        B = x.shape[0]

        # 1. Patch嵌入: [B, 3, H, W] -> [B, num_patches, embed_dim]
        #    Patch embedding
        x = self.patch_embed(x)

        # 2. 在序列头部添加CLS token / Prepend CLS token to the sequence
        #    [B, num_patches, D] -> [B, num_patches+1, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 3. 加入位置编码并dropout / Add positional encoding and dropout
        x = self.pos_dropout(x + self.pos_embed)

        # 4. 依次通过所有Transformer块 / Pass through all Transformer blocks
        for block in self.blocks:
            x = block(x)

        # 5. 最终归一化 / Final normalization
        x = self.norm(x)

        # 6. 返回CLS token位置的输出作为图像表示
        #    Return CLS token position output as image representation
        return x[:, 0]

    def get_num_patches(self) -> int:
        """返回patch数量 / Return number of patches."""
        return self.patch_embed.num_patches
