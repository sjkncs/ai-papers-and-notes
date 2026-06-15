"""
CLIP (Contrastive Language-Image Pre-training) — 代码复现
========================================================

基于论文 "Learning Transferable Visual Models From Natural Language Supervision"
(Radford et al., ICML 2021) 的简化复现。

核心思想: 通过对比学习对齐图像编码器和文本编码器的表示空间。

架构:
  Image Encoder: 简化版ViT (Vision Transformer)
  Text Encoder: 简化版Transformer
  Contrastive Loss: InfoNCE (对称交叉熵)

Usage:
    model = SimpleCLIP(image_size=224, patch_size=16)
    loss = model(image_batch, text_batch)

Author: Auto-generated from paper analysis
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# ============================================================
# Vision Transformer (简化版ViT) / Simplified ViT
# ============================================================

class PatchEmbedding(nn.Module):
    """将图像分割为patches并嵌入 / Split image into patches and embed"""

    def __init__(self, image_size: int = 224, patch_size: int = 16, embed_dim: int = 512):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            3, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 3, H, W] images

        Returns:
            [batch, num_patches, embed_dim] patch embeddings
        """
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力 / Multi-head self-attention"""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块 / Transformer encoder block"""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# Image Encoder / 图像编码器
# ============================================================

class ImageEncoder(nn.Module):
    """
    简化版ViT图像编码器 / Simplified ViT Image Encoder

    将图像编码为固定维度的向量表示。
    使用CLS token的输出作为图像表示。
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 512,
        num_heads: int = 8,
        depth: int = 6,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token + 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 初始化
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 3, H, W] images

        Returns:
            [batch, embed_dim] image features
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, D]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, D]

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 返回CLS token的输出 / Return CLS token output
        return x[:, 0]


# ============================================================
# Text Encoder / 文本编码器
# ============================================================

class TextEncoder(nn.Module):
    """
    简化版文本编码器 / Simplified Text Encoder

    使用Transformer编码文本token序列。
    使用EOS token的输出作为文本表示。
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 512,
        num_heads: int = 8,
        depth: int = 6,
        max_seq_len: int = 77,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_ids: [batch, seq_len] token IDs (含SOS和EOS)

        Returns:
            [batch, embed_dim] text features
        """
        B, seq_len = text_ids.shape

        # Token + positional embedding
        x = self.token_embedding(text_ids)
        x = x + self.pos_embed[:, :seq_len]

        # 因果掩码 / Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=text_ids.device),
            diagonal=1,
        ).bool()

        # Transformer (with causal attention)
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 使用EOS token位置 (最后一个非padding token)
        # 简化: 使用序列最后一个位置
        return x[:, -1]


# ============================================================
# CLIP Model / CLIP模型
# ============================================================

class SimpleCLIP(nn.Module):
    """
    简化版CLIP / Simplified CLIP

    核心组件:
    1. Image Encoder (ViT) → image features
    2. Text Encoder (Transformer) → text features
    3. Projection heads → 共享嵌入空间
    4. InfoNCE contrastive loss → 对比学习
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 512,
        projection_dim: int = 256,
        num_heads: int = 8,
        image_depth: int = 6,
        text_depth: int = 6,
        vocab_size: int = 49408,
        max_seq_len: int = 77,
    ):
        super().__init__()

        # 编码器 / Encoders
        self.image_encoder = ImageEncoder(
            image_size, patch_size, embed_dim, num_heads, image_depth
        )
        self.text_encoder = TextEncoder(
            vocab_size, embed_dim, num_heads, text_depth, max_seq_len
        )

        # 投影头 / Projection heads → shared embedding space
        self.image_projection = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # 可学习温度参数 / Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像 → 归一化特征"""
        features = self.image_encoder(images)
        features = self.image_projection(features)
        return F.normalize(features, dim=-1)

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        """编码文本 → 归一化特征"""
        features = self.text_encoder(text_ids)
        features = self.text_projection(features)
        return F.normalize(features, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CLIP对比学习前向传播 / CLIP Contrastive Forward Pass

        Args:
            images: [batch, 3, H, W]
            text_ids: [batch, seq_len]

        Returns:
            loss: InfoNCE对比损失
            logits_per_image: [batch, batch] image→text相似度
            logits_per_text: [batch, batch] text→image相似度
        """
        # 编码 / Encode
        image_features = self.encode_image(images)   # [B, proj_dim]
        text_features = self.encode_text(text_ids)   # [B, proj_dim]

        # 计算余弦相似度 / Cosine similarity
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = logit_scale * (image_features @ text_features.T)
        logits_per_text = logits_per_image.T

        # 对称InfoNCE损失 / Symmetric InfoNCE loss
        batch_size = images.shape[0]
        labels = torch.arange(batch_size, device=images.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2.0

        return loss, logits_per_image, logits_per_text

    @torch.no_grad()
    def zero_shot_classify(
        self,
        images: torch.Tensor,
        class_text_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        零样本分类 / Zero-shot Classification

        Args:
            images: [batch, 3, H, W]
            class_text_ids: [num_classes, seq_len] 每个类别的文本描述

        Returns:
            probs: [batch, num_classes] 分类概率
        """
        image_features = self.encode_image(images)       # [B, D]
        text_features = self.encode_text(class_text_ids) # [C, D]

        similarity = image_features @ text_features.T    # [B, C]
        probs = similarity.softmax(dim=-1)
        return probs


# ============================================================
# 训练示例 / Training Example
# ============================================================

def train_clip_example():
    """CLIP训练示例 / CLIP training example"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCLIP(
        image_size=64,      # 小尺寸用于演示
        patch_size=8,
        embed_dim=128,
        projection_dim=64,
        num_heads=4,
        image_depth=2,
        text_depth=2,
        vocab_size=1000,
        max_seq_len=32,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 模拟数据 / Simulated data
    batch_size = 16
    images = torch.randn(batch_size, 3, 64, 64, device=device)
    text_ids = torch.randint(0, 1000, (batch_size, 32), device=device)

    # 训练一步 / Single training step
    model.train()
    loss, logits_i, logits_t = model(images, text_ids)

    loss.backward()
    optimizer.step()

    print(f"Training loss: {loss.item():.4f}")
    print(f"Logit scale: {model.logit_scale.exp().item():.2f}")
    print(f"Image-text similarity range: [{logits_i.min().item():.2f}, {logits_i.max().item():.2f}]")

    # 零样本分类演示 / Zero-shot classification demo
    model.eval()
    test_images = torch.randn(4, 3, 64, 64, device=device)
    class_prompts = torch.randint(0, 1000, (5, 32), device=device)  # 5 classes

    probs = model.zero_shot_classify(test_images, class_prompts)
    print(f"\nZero-shot classification probabilities:")
    print(f"Shape: {probs.shape}")  # [4, 5]
    for i in range(4):
        pred = probs[i].argmax().item()
        print(f"  Image {i}: predicted class {pred} (prob: {probs[i][pred]:.3f})")

    # 模型参数量 / Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    img_params = sum(p.numel() for p in model.image_encoder.parameters())
    txt_params = sum(p.numel() for p in model.text_encoder.parameters())

    print(f"\nTotal parameters: {total_params:,}")
    print(f"  Image encoder: {img_params:,}")
    print(f"  Text encoder:  {txt_params:,}")
    print(f"  Projections:   {total_params - img_params - txt_params:,}")


if __name__ == "__main__":
    print("=" * 60)
    print("CLIP — Contrastive Language-Image Pre-training")
    print("Code Reproduction")
    print("=" * 60)

    train_clip_example()
