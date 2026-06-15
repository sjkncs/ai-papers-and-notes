"""
InfoNCE 对称对比损失 + 梯度分析
InfoNCE Symmetric Contrastive Loss + Gradient Analysis
======================================================

论文参考 / Paper Reference:
  Radford et al., "Learning Transferable Visual Models From Natural
  Language Supervision", ICML 2021.
  以及 / And: Oord et al., "Representation Learning with Contrastive
  Predictive Coding", 2018.

核心损失函数 / Core Loss:
  L = (L_image_to_text + L_text_to_image) / 2

  L_i2t = -log[ exp(sim(I_i, T_i) / tau) / sum_j exp(sim(I_i, T_j) / tau) ]
  L_t2i = -log[ exp(sim(T_i, I_i) / tau) / sum_j exp(sim(T_i, I_j) / tau) ]

  其中 / Where:
    - sim(a, b) = a . b (L2归一化后的点积 = 余弦相似度)
      sim(a, b) = a . b (dot product after L2 normalization = cosine similarity)
    - tau 是可学习的温度参数 / tau is a learnable temperature parameter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ClipLoss(nn.Module):
    """
    CLIP 的 InfoNCE 对称对比损失。
    InfoNCE Symmetric Contrastive Loss for CLIP.

    原理 / Rationale:
      给定一个batch中的N个(图像,文本)对，构造 N×N 的相似度矩阵。
      正样本在对角线上，其余都是负样本。对两个方向 (图像→文本 和
      文本→图像) 各计算一次交叉熵损失，取平均。

      Given N (image, text) pairs in a batch, construct an NxN similarity
      matrix. Positive pairs are on the diagonal, rest are negatives.
      Compute cross-entropy loss for both directions (image->text and
      text->image), then average.

    可学习温度 / Learnable Temperature:
      损失函数接受 logit_scale 参数 (= log(1/tau))。
      该参数由模型维护，在训练中可被优化。
      The loss accepts logit_scale parameter (= log(1/tau)).
      This parameter is maintained by the model and optimized during training.

    梯度分析 / Gradient Analysis:
      提供 get_gradient_stats() 方法，用于监控训练过程中梯度的健康程度。
      Provides get_gradient_stats() to monitor gradient health during training.

    Args:
        temperature_max (float): 温度缩放因子上限 / Temperature scaling factor upper bound.
    """

    def __init__(self, temperature_max: float = 100.0):
        super().__init__()
        self.temperature_max = temperature_max

        # 用于存储最近一次的梯度统计 / Store latest gradient statistics
        self._last_grad_stats: Optional[Dict[str, float]] = None

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 InfoNCE 对称对比损失 / Compute InfoNCE symmetric contrastive loss.

        Args:
            image_features: [B, D] L2归一化的图像特征 / L2-normalized image features.
            text_features:  [B, D] L2归一化的文本特征 / L2-normalized text features.
            logit_scale:    标量, exp(logit_scale) = 1/tau / Scalar, exp(logit_scale) = 1/tau.

        Returns:
            loss:    标量损失值 / Scalar loss value.
            details: 包含中间值的字典 / Dictionary with intermediate values.
        """
        # 温度缩放 (限制上限防止数值爆炸)
        # Temperature scaling (clamp upper bound to prevent numerical explosion)
        scale = logit_scale.exp().clamp(max=self.temperature_max)

        # 相似度矩阵 / Similarity matrix
        # 特征已经L2归一化，所以点积 = 余弦相似度
        # Features are L2-normalized, so dot product = cosine similarity
        logits_per_image = scale * (image_features @ text_features.T)  # [B, B]
        logits_per_text = logits_per_image.T                           # [B, B]

        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        # 图像→文本损失: 每行做softmax交叉熵
        # Image->Text loss: softmax cross-entropy per row
        loss_i2t = F.cross_entropy(logits_per_image, labels)

        # 文本→图像损失: 每行做softmax交叉熵 (等价于对列做)
        # Text->Image loss: softmax cross-entropy per row (equivalent to per column)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        # 对称损失 = 两个方向的平均 / Symmetric loss = average of both directions
        loss = (loss_i2t + loss_t2i) / 2.0

        # 记录详细信息 / Record detailed information
        details = {
            "loss": loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "logit_scale": scale.detach(),
            "temperature": (1.0 / scale).detach(),
            "similarity_mean": (image_features @ text_features.T).mean().detach(),
            "similarity_diag": (image_features * text_features).sum(dim=-1).mean().detach(),
        }

        return loss, details

    def compute_accuracy(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> Dict[str, float]:
        """
        计算批次内的检索准确率 (无需梯度)。
        Compute in-batch retrieval accuracy (no gradients needed).

        对于每个图像，找到最相似的文本 (反之亦然)，检查是否匹配。
        For each image, find the most similar text (and vice versa), check if it matches.

        Args:
            image_features: [B, D] 图像特征 / Image features.
            text_features:  [B, D] 文本特征 / Text features.
            logit_scale:    温度缩放 / Temperature scaling.

        Returns:
            dict: {"i2t_acc": float, "t2i_acc": float, "mean_acc": float}
        """
        with torch.no_grad():
            scale = logit_scale.exp().clamp(max=self.temperature_max)
            logits = scale * (image_features @ text_features.T)
            batch_size = image_features.shape[0]
            labels = torch.arange(batch_size, device=image_features.device)

            # 图像→文本准确率 / Image->Text accuracy
            i2t_pred = logits.argmax(dim=1)
            i2t_acc = (i2t_pred == labels).float().mean().item()

            # 文本→图像准确率 / Text->Image accuracy
            t2i_pred = logits.T.argmax(dim=1)
            t2i_acc = (t2i_pred == labels).float().mean().item()

            return {
                "i2t_acc": i2t_acc,
                "t2i_acc": t2i_acc,
                "mean_acc": (i2t_acc + t2i_acc) / 2.0,
            }


# ============================================================
# 检索评估指标 / Retrieval Evaluation Metrics
# ============================================================

def compute_retrieval_metrics(
    similarity_matrix: torch.Tensor,
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    计算图文检索的 Recall@K 指标。
    Compute Recall@K metrics for image-text retrieval.

    假设 / Assumption:
      第 i 行对应第 i 个查询 (图像或文本)，
      第 i 列对应正确的匹配。
      Row i corresponds to the i-th query (image or text),
      Column i corresponds to the correct match.

    Args:
        similarity_matrix: [N, M] 相似度矩阵 / Similarity matrix.
        k_values:          要计算的K值元组 / Tuple of K values to compute.

    Returns:
        dict: {"R@1": float, "R@5": float, "R@10": float, ...}
    """
    with torch.no_grad():
        N = similarity_matrix.shape[0]
        labels = torch.arange(N, device=similarity_matrix.device)

        # 按相似度从高到低排序 / Sort by similarity, descending
        _, indices = similarity_matrix.sort(dim=1, descending=True)

        metrics = {}
        for k in k_values:
            # 取前K个检索结果 / Take top-K retrieval results
            topk = indices[:, :k]

            # 检查正确答案是否在前K个中 / Check if correct answer is in top-K
            correct = (topk == labels.unsqueeze(1)).any(dim=1)
            recall = correct.float().mean().item()

            metrics[f"R@{k}"] = recall

        return metrics


# ============================================================
# 梯度分析工具 / Gradient Analysis Tools
# ============================================================

def analyze_gradients(model: nn.Module) -> Dict[str, float]:
    """
    分析模型梯度的健康状态。
    Analyze model gradient health.

    监控指标 / Monitored metrics:
      1. 梯度范数 / Gradient norm: 过大可能表示训练不稳定
         Too large may indicate training instability
      2. 梯度中零的比例 / Fraction of zero gradients: 过高可能表示梯度消失
         Too high may indicate vanishing gradients
      3. 温度参数的梯度 / Temperature parameter gradient: 监控温度学习是否正常
         Monitor whether temperature learning is normal
      4. 各模块梯度范数 / Per-module gradient norms: 检查是否有模块梯度异常
         Check for anomalous per-module gradients

    Args:
        model: CLIP模型 / CLIP model.

    Returns:
        梯度统计字典 / Dictionary of gradient statistics.
    """
    stats = {
        "total_grad_norm": 0.0,
        "num_params": 0,
        "num_zero_grad": 0,
        "max_grad": float("-inf"),
        "min_grad": float("inf"),
    }

    module_norms: Dict[str, float] = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            stats["total_grad_norm"] += grad_norm ** 2
            stats["num_params"] += param.numel()
            stats["num_zero_grad"] += (param.grad == 0).sum().item()
            stats["max_grad"] = max(stats["max_grad"], param.grad.abs().max().item())
            stats["min_grad"] = min(stats["min_grad"], param.grad.abs().min().item())

            # 按顶层模块分组 / Group by top-level module
            top_module = name.split(".")[0]
            if top_module not in module_norms:
                module_norms[top_module] = 0.0
            module_norms[top_module] += grad_norm ** 2

    # 计算总梯度范数 / Compute total gradient norm
    stats["total_grad_norm"] = stats["total_grad_norm"] ** 0.5

    # 零梯度比例 / Fraction of zero gradients
    total_elements = sum(p.numel() for p in model.parameters() if p.grad is not None)
    stats["zero_grad_fraction"] = (
        stats["num_zero_grad"] / max(total_elements, 1)
    )

    # 各模块梯度范数 / Per-module gradient norms
    for module_name in module_norms:
        module_norms[module_name] = module_norms[module_name] ** 0.5
        stats[f"grad_norm_{module_name}"] = module_norms[module_name]

    # 清理非有限值 / Clean up non-finite values
    if stats["max_grad"] == float("-inf"):
        stats["max_grad"] = 0.0
    if stats["min_grad"] == float("inf"):
        stats["min_grad"] = 0.0

    return stats
