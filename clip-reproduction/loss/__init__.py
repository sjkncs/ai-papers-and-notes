"""
CLIP 损失函数模块 / CLIP Loss Functions Module
===============================================

提供 InfoNCE 对称对比损失及其梯度分析工具。
Provides InfoNCE symmetric contrastive loss and gradient analysis tools.
"""

from .clip_loss import ClipLoss, compute_retrieval_metrics

__all__ = [
    "ClipLoss",
    "compute_retrieval_metrics",
]
