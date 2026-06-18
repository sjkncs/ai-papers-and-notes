"""
CLIP 完整模型 — 图像编码器 + 文本编码器 + 投影头 + 对比学习
CLIP Full Model — Image Encoder + Text Encoder + Projection Heads + Contrastive Learning
========================================================================================

论文参考 / Paper Reference:
  Radford et al., "Learning Transferable Visual Models From Natural
  Language Supervision", ICML 2021.

本模块组装完整CLIP模型 / This module assembles the full CLIP model:
  1. ImageEncoder    → 图像特征 / Image features
  2. TextEncoder     → 文本特征 / Text features
  3. Projection Heads → 映射到共享嵌入空间 / Map to shared embedding space
  4. InfoNCE Loss    → 对称对比学习 / Symmetric contrastive learning
  5. zero_shot_classify → 零样本分类推理 / Zero-shot classification inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class SimpleCLIP(nn.Module):
    """
    简化版 CLIP 模型 / Simplified CLIP Model.

    架构 / Architecture:
      Image ──→ ImageEncoder(ViT) ──→ Image Projection ──→ ┐
                                                             ├─ Cosine Similarity × temperature → InfoNCE Loss
      Text  ──→ TextEncoder(Transform) ──→ Text Projection ──→ ┘

    核心思想 / Core Idea:
      通过对比学习，使匹配的 (图像, 文本) 对在共享嵌入空间中靠近，
      不匹配的对远离。使用对称交叉熵 (InfoNCE) 作为损失函数。

      Through contrastive learning, matched (image, text) pairs are pulled
      close in a shared embedding space, while unmatched pairs are pushed
      apart. Symmetric cross-entropy (InfoNCE) is used as the loss function.

    可学习温度 / Learnable Temperature:
      logit_scale = log(1 / tau)，其中 tau 初始化为 0.07。
      训练过程中 tau 自适应调整。
      logit_scale = log(1 / tau), where tau is initialized to 0.07.
      tau adapts during training.

    Args:
        image_size (int):    输入图像尺寸 / Input image size.
        patch_size (int):    Patch大小 / Patch size.
        embed_dim (int):     编码器隐藏维度 / Encoder hidden dimension.
        projection_dim (int): 投影输出维度 / Projection output dimension.
        num_heads (int):     注意力头数 / Number of attention heads.
        image_depth (int):   图像编码器层数 / Image encoder depth.
        text_depth (int):    文本编码器层数 / Text encoder depth.
        vocab_size (int):    词表大小 / Vocabulary size.
        max_seq_len (int):   最大文本长度 / Maximum text length.
        mlp_ratio (float):   MLP扩展倍率 / MLP expansion ratio.
        dropout (float):     Dropout概率 / Dropout probability.
        temperature_init (float): 温度初始值 / Initial temperature value.
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
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        temperature_init: float = 0.07,
    ):
        super().__init__()
        self.projection_dim = projection_dim

        # ---- 编码器 / Encoders ----
        self.image_encoder = ImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=image_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=text_depth,
            max_seq_len=max_seq_len,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # ---- 投影头 / Projection Heads ----
        # 将编码器输出映射到共享的低维嵌入空间
        # Map encoder outputs to a shared lower-dimensional embedding space
        # 使用两层MLP，中间ReLU激活 / Two-layer MLP with ReLU in between
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

        # ---- 可学习温度参数 / Learnable Temperature Parameter ----
        # 论文中 tau 初始化为 0.07, 我们用 logit_scale = log(1/tau) 参数化
        # Paper uses tau initialized to 0.07; we parameterize as logit_scale = log(1/tau)
        # 这样 exp(logit_scale) = 1/tau, 保证 logit_scale > 0
        # So exp(logit_scale) = 1/tau, ensuring logit_scale > 0
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1.0 / temperature_init))
        )
        self.temperature_max = 100.0  # 温度上限 / Temperature upper bound

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像并归一化 / Encode images and L2-normalize.

        Args:
            images: [B, 3, H, W] 图像批次 / Batch of images.

        Returns:
            [B, projection_dim] 归一化的图像特征 / L2-normalized image features.
        """
        features = self.image_encoder(images)       # [B, embed_dim]
        features = self.image_projection(features)   # [B, projection_dim]
        return F.normalize(features, dim=-1)         # L2归一化 / L2 normalization

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        编码文本并归一化 / Encode text and L2-normalize.

        Args:
            text_ids: [B, seq_len] token ID序列 / Token ID sequences.

        Returns:
            [B, projection_dim] 归一化的文本特征 / L2-normalized text features.
        """
        features = self.text_encoder(text_ids)       # [B, embed_dim]
        features = self.text_projection(features)     # [B, projection_dim]
        return F.normalize(features, dim=-1)          # L2归一化 / L2 normalization

    def forward(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CLIP 对比学习前向传播 / CLIP Contrastive Learning Forward Pass.

        计算过程 / Computation:
          1. 分别编码图像和文本 / Encode images and text separately
          2. 计算余弦相似度矩阵 / Compute cosine similarity matrix
          3. 乘以温度缩放因子 / Multiply by temperature scaling factor
          4. 计算对称交叉熵损失 / Compute symmetric cross-entropy loss

        损失公式 / Loss Formula:
          L_i2t = -sum_j [ j==i ] * log( softmax(sim(I_i, T_j) / tau) )
          L_t2i = -sum_j [ j==i ] * log( softmax(sim(T_i, I_j) / tau) )
          L = (L_i2t + L_t2i) / 2

        Args:
            images:   [B, 3, H, W] 图像批次 / Batch of images.
            text_ids: [B, seq_len] 文本token序列 / Text token sequences.

        Returns:
            loss:             标量InfoNCE损失 / Scalar InfoNCE loss.
            logits_per_image: [B, B] 图像到文本的相似度 / Image-to-text similarities.
            logits_per_text:  [B, B] 文本到图像的相似度 / Text-to-image similarities.
        """
        # 编码并归一化 / Encode and normalize
        image_features = self.encode_image(images)    # [B, D]
        text_features = self.encode_text(text_ids)    # [B, D]

        # 温度缩放: exp(logit_scale) = 1/tau
        # Temperature scaling
        logit_scale = self.logit_scale.exp().clamp(max=self.temperature_max)

        # 余弦相似度矩阵 (特征已归一化，点积=余弦相似度)
        # Cosine similarity matrix (features are normalized, dot product = cosine similarity)
        logits_per_image = logit_scale * (image_features @ text_features.T)  # [B, B]
        logits_per_text = logits_per_image.T                                 # [B, B]

        # 对称InfoNCE损失 / Symmetric InfoNCE loss
        # 对角线元素是正样本，其他是负样本
        # Diagonal elements are positive pairs, others are negatives
        batch_size = images.shape[0]
        labels = torch.arange(batch_size, device=images.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)  # 图像→文本 / Image→Text
        loss_t2i = F.cross_entropy(logits_per_text, labels)   # 文本→图像 / Text→Image
        loss = (loss_i2t + loss_t2i) / 2.0

        return loss, logits_per_image, logits_per_text

    @torch.no_grad()
    def zero_shot_classify(
        self,
        images: torch.Tensor,
        class_text_ids: torch.Tensor,
        prompt_prefix: str = "a photo of ",
    ) -> torch.Tensor:
        """
        零样本分类 / Zero-Shot Classification.

        原理 / Rationale:
          无需在目标任务上微调，通过将图像与候选类别的自然语言描述比较
          来进行分类。使用 "a photo of {class}" 这样的prompt模板。

          Without fine-tuning on the target task, classify by comparing
          images against natural language descriptions of candidate classes.
          Uses prompt templates like "a photo of {class}".

        计算过程 / Computation:
          1. 编码图像 / Encode images
          2. 编码各类别文本描述 / Encode text descriptions for each class
          3. 计算图像与各类别的余弦相似度 / Compute cosine similarity between images and classes
          4. Softmax归一化为概率分布 / Softmax normalize to probability distribution

        Args:
            images:         [B, 3, H, W] 待分类图像 / Images to classify.
            class_text_ids: [C, seq_len] 各类别的token ID / Token IDs for each class.
            prompt_prefix:  prompt模板前缀 / Prompt template prefix (unused here;
                            assumes text_ids already include prompt).

        Returns:
            probs: [B, C] 每个图像属于各类别的概率 / Probability of each image belonging to each class.
        """
        # 编码图像 / Encode images
        image_features = self.encode_image(images)             # [B, D]

        # 编码所有类别的文本描述 / Encode text descriptions for all classes
        text_features = self.encode_text(class_text_ids)       # [C, D]

        # 计算相似度 / Compute similarity
        similarity = image_features @ text_features.T          # [B, C]

        # 使用温度缩放后的softmax / Use temperature-scaled softmax
        logit_scale = self.logit_scale.exp().clamp(max=self.temperature_max)
        probs = (logit_scale * similarity).softmax(dim=-1)

        return probs

    @torch.no_grad()
    def get_similarity_matrix(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        获取图像-文本相似度矩阵 (用于检索评估)。
        Get image-text similarity matrix (for retrieval evaluation).

        Args:
            images:   [N, 3, H, W] 图像 / Images.
            text_ids: [M, seq_len] 文本 / Texts.

        Returns:
            [N, M] 相似度矩阵 / Similarity matrix.
        """
        image_features = self.encode_image(images)   # [N, D]
        text_features = self.encode_text(text_ids)   # [M, D]
        return image_features @ text_features.T      # [N, M]

    def get_temperature(self) -> float:
        """
        获取当前温度值 / Get current temperature value.

        Returns:
            当前温度 tau / Current temperature tau.
        """
        return 1.0 / self.logit_scale.exp().item()
