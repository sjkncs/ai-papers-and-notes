"""
CLIP 模型组件包 / CLIP Model Components Package
================================================

提供 CLIP 模型的各个核心组件:
  - ImageEncoder: 基于ViT的图像编码器
  - TextEncoder:  基于Transformer的文本编码器
  - SimpleCLIP:   完整CLIP模型（含投影头与对比学习）

Provides core CLIP model components:
  - ImageEncoder: ViT-based image encoder
  - TextEncoder:  Transformer-based text encoder
  - SimpleCLIP:   Full CLIP model with projection heads and contrastive learning
"""

from .image_encoder import PatchEmbedding, ImageEncoder
from .text_encoder import TextEncoder
from .clip_model import SimpleCLIP

__all__ = [
    "PatchEmbedding",
    "ImageEncoder",
    "TextEncoder",
    "SimpleCLIP",
]
