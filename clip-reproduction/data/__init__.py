"""
CLIP 数据模块 / CLIP Data Module
=================================

提供图像-文本对数据集加载器和简化的BPE分词器。
Provides image-text pair dataset loaders and a simplified BPE tokenizer.
"""

from .dataset import ImageTextDataset, SimpleBPETokenizer, build_transforms

__all__ = [
    "ImageTextDataset",
    "SimpleBPETokenizer",
    "build_transforms",
]
