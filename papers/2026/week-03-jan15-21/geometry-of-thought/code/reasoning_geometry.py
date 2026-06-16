# ==============================================================================
# MIT License
#
# Copyright (c) 2026 AI Papers and Notes
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
# ==============================================================================

"""
Reasoning Geometry Analyzer / 推理几何分析器

Analyzes the geometric structure of reasoning paths in large language models.
Based on "The Geometry of Thought" (2601.13358).

分析大语言模型中推理路径的几何结构。
基于《The Geometry of Thought》(2601.13358)。

Core components / 核心组件:
- ThoughtPathEmbedding: Embed reasoning steps into geometric space
- ReasoningGeometryAnalyzer: Analyze reasoning path geometry
- ScaleComparisonFramework: Compare geometry across model scales
- DomainSpecificAnalysis: Per-domain restructuring detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 配置日志 / Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ============================================================================
# 枚举和数据类 / Enums and Data Classes
# ============================================================================


class ReasoningDomain(Enum):
    """
    推理领域枚举 / Reasoning domain enumeration

    每个领域在嵌入空间中可能形成不同的几何结构。
    Each domain may form distinct geometric structures in embedding space.
    """

    MATHEMATICS = "mathematics"
    LEGAL = "legal"
    COMMON_SENSE = "common_sense"
    SCIENCE = "science"
    CODE = "code"


class DimensionReductionMethod(Enum):
    """
    降维方法枚举 / Dimensionality reduction method enumeration
    """

    PCA = "pca"
    UMAP = "umap"
    TSNE = "tsne"
    ISOMAP = "isomap"


@dataclass
class GeometryMetrics:
    """
    几何度量结果 / Geometry metrics result container

    存储推理路径的几何属性测量值。
    Stores geometric property measurements of reasoning paths.
    """

    # 内在维度估计 / Intrinsic dimensionality estimate
    intrinsic_dim: float = 0.0
    # 平均曲率 / Mean curvature
    mean_curvature: float = 0.0
    # 曲率标准差 / Curvature standard deviation
    curvature_std: float = 0.0
    # 路径长度统计 / Path length statistics
    mean_path_length: float = 0.0
    path_length_std: float = 0.0
    # 流形紧致度 / Manifold compactness
    compactness: float = 0.0
    # 分形维度 / Fractal dimension
    fractal_dim: float = 0.0
    # 额外度量 / Additional metrics
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScaleComparisonResult:
    """
    跨尺度比较结果 / Cross-scale comparison result

    存储不同模型尺度间的几何差异。
    Stores geometric differences between model scales.
    """

    # 尺度标识 / Scale identifiers
    scale_a: str = ""
    scale_b: str = ""
    # Procrustes距离 / Procrustes distance
    procrustes_distance: float = 0.0
    # 维度变化 / Dimensionality change
    dim_change: float = 0.0
    # 曲率变化 / Curvature change
    curvature_change: float = 0.0
    # 拓扑相似度 / Topological similarity
    topological_similarity: float = 0.0
    # 是否存在相变 / Whether phase transition detected
    phase_transition_detected: bool = False
    # 额外信息 / Additional info
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 数据集 / Dataset
# ============================================================================


class ReasoningPathDataset(Dataset):
    """
    推理路径数据集 / Reasoning path dataset

    加载和管理多步推理过程的激活数据。
    Loads and manages activation data from multi-step reasoning processes.

    Args:
        data_dir: 数据目录路径 / Path to data directory
        domain: 推理领域 / Reasoning domain
        max_steps: 最大推理步数 / Maximum reasoning steps
    """

    def __init__(
        self,
        data_dir: str | Path,
        domain: ReasoningDomain,
        max_steps: int = 20,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.domain = domain
        self.max_steps = max_steps
        self.samples: List[Dict[str, torch.Tensor]] = []
        self._load_data()

    def _load_data(self) -> None:
        """
        从磁盘加载推理路径数据 / Load reasoning path data from disk

        每个样本包含：多步推理的中间激活、最终输出、以及领域标签。
        Each sample contains: intermediate activations of multi-step reasoning,
        final output, and domain label.
        """
        domain_path = self.data_dir / self.domain.value
        if not domain_path.exists():
            logger.warning(
                "Domain data not found at %s, generating synthetic data / "
                "领域数据未找到，生成合成数据: %s",
                domain_path,
                self.domain.value,
            )
            self._generate_synthetic_data()
            return

        for sample_file in sorted(domain_path.glob("*.pt")):
            sample = torch.load(sample_file, map_location="cpu", weights_only=True)
            # 截断或填充到最大步数 / Truncate or pad to max steps
            activations = sample["activations"]
            if activations.shape[0] > self.max_steps:
                activations = activations[: self.max_steps]
            self.samples.append(
                {
                    "activations": activations,
                    "label": sample.get("label", torch.tensor(0)),
                    "domain": torch.tensor(
                        list(ReasoningDomain).index(self.domain)
                    ),
                }
            )

    def _generate_synthetic_data(self, n_samples: int = 200) -> None:
        """
        生成合成推理路径数据用于测试 / Generate synthetic reasoning path data for testing

        合成数据模拟不同领域的推理路径几何特征：
        - 数学：螺旋形路径
        - 法律：低维收敛路径
        - 常识：高维分散路径
        Synthetic data simulates reasoning path geometry per domain:
        - Math: spiral paths
        - Legal: low-dimensional convergent paths
        - Common sense: high-dimensional scattered paths
        """
        embed_dim = 128
        for i in range(n_samples):
            n_steps = torch.randint(3, self.max_steps + 1, (1,)).item()
            if self.domain == ReasoningDomain.MATHEMATICS:
                # 数学推理：螺旋形路径 / Math reasoning: spiral paths
                t = torch.linspace(0, 2 * np.pi, n_steps)
                base = torch.stack(
                    [t * torch.cos(t), t * torch.sin(t), t * 0.5], dim=1
                )
                activations = F.pad(
                    base, (0, embed_dim - 3)
                ) + torch.randn(n_steps, embed_dim) * 0.1
            elif self.domain == ReasoningDomain.LEGAL:
                # 法律推理：快速收敛到低维子空间 / Legal: rapid convergence to low-dim subspace
                activations = torch.randn(n_steps, embed_dim) * 0.5
                decay = torch.exp(-torch.arange(n_steps, dtype=torch.float32) * 0.3)
                activations = activations * decay.unsqueeze(1)
                # 在低维方向添加结构 / Add structure in low dimensions
                activations[:, :4] += torch.linspace(0, 1, n_steps).unsqueeze(1)
            else:
                # 常识推理：高维分散 / Common sense: high-dimensional scattered
                activations = torch.randn(n_steps, embed_dim) * 1.5

            self.samples.append(
                {
                    "activations": activations,
                    "label": torch.tensor(i % 3),
                    "domain": torch.tensor(
                        list(ReasoningDomain).index(self.domain)
                    ),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


# ============================================================================
# 思维路径嵌入 / Thought Path Embedding
# ============================================================================


class ThoughtPathEmbedding(nn.Module):
    """
    思维路径嵌入模块 / Thought path embedding module

    将多步推理过程的激活序列嵌入到统一的几何空间中。
    使用Transformer编码器捕获步骤间的关系，然后通过池化获得路径级表示。

    Embeds multi-step reasoning activation sequences into a unified geometric space.
    Uses a Transformer encoder to capture inter-step relationships, then pools
    to obtain path-level representations.

    Args:
        input_dim: 输入激活维度 / Input activation dimension
        embed_dim: 嵌入空间维度 / Embedding space dimension
        n_heads: 注意力头数 / Number of attention heads
        n_layers: Transformer层数 / Number of Transformer layers
        max_steps: 最大推理步数 / Maximum reasoning steps
    """

    def __init__(
        self,
        input_dim: int = 4096,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        max_steps: int = 20,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # 输入投影：将高维激活投影到可处理的维度
        # Input projection: project high-dim activations to manageable dimension
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # 步骤位置编码 / Step positional encoding
        self.step_pos_enc = nn.Parameter(
            torch.randn(max_steps, embed_dim) * 0.02
        )

        # Transformer编码器层 / Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 路径级池化投影 / Path-level pooling projection
        self.path_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, activations: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播：将推理步骤序列映射到几何嵌入空间
        Forward pass: map reasoning step sequences to geometric embedding space

        Args:
            activations: [batch, steps, input_dim] 推理步骤激活序列
            mask: [batch, steps] 有效步掩码（1=有效，0=填充）

        Returns:
            [batch, embed_dim] 路径级几何嵌入
        """
        batch_size, n_steps, _ = activations.shape

        # 投影输入 / Project input
        x = self.input_proj(activations)

        # 添加步骤位置编码 / Add step positional encoding
        x = x + self.step_pos_enc[:n_steps].unsqueeze(0)

        # Transformer编码（捕获步骤间关系）
        # Transformer encoding (captures inter-step relationships)
        x = self.transformer(x, src_key_padding_mask=mask)

        # 注意力池化：聚合所有步骤的信息
        # Attention pooling: aggregate information across all steps
        if mask is not None:
            # 掩码平均池化 / Masked average pooling
            weights = (~mask).float().unsqueeze(-1)
            x = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # 最终投影和归一化 / Final projection and normalization
        x = self.path_proj(x)
        x = self.layer_norm(x)

        return x


# ============================================================================
# 推理几何分析器 / Reasoning Geometry Analyzer
# ============================================================================


class ReasoningGeometryAnalyzer(nn.Module):
    """
    推理几何分析器 / Reasoning geometry analyzer

    分析LLM推理路径的几何属性，包括内在维度、曲率、紧致度等。
    支持多领域比较和跨尺度分析。

    Analyzes geometric properties of LLM reasoning paths, including intrinsic
    dimensionality, curvature, compactness, etc. Supports multi-domain comparison
    and cross-scale analysis.

    Args:
        embed_dim: 嵌入空间维度 / Embedding space dimension
        n_domains: 推理领域数量 / Number of reasoning domains
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_domains: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_domains = n_domains

        # 领域分类头：辅助验证嵌入质量
        # Domain classification head: auxiliary verification of embedding quality
        self.domain_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, n_domains),
        )

        # 曲率估计网络 / Curvature estimation network
        self.curvature_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        # 相变检测器 / Phase transition detector
        self.phase_detector = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def compute_intrinsic_dimension(
        self, embeddings: torch.Tensor, k: int = 10
    ) -> float:
        """
        使用最大似然估计计算内在维度
        Estimate intrinsic dimensionality using maximum likelihood estimation

        基于Levina & Bickel (2004)的方法，利用最近邻距离估计流形的内在维度。

        Args:
            embeddings: [N, embed_dim] 点集嵌入
            k: 最近邻数量 / Number of nearest neighbors

        Returns:
            内在维度估计值 / Intrinsic dimensionality estimate
        """
        n_points = embeddings.shape[0]
        if n_points <= k:
            logger.warning(
                "Sample size (%d) <= k (%d), returning embed_dim / "
                "样本量小于k，返回嵌入维度",
                n_points,
                k,
            )
            return float(self.embed_dim)

        # 计算成对距离 / Compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        # 排除自身距离 / Exclude self-distance
        dist_matrix.fill_diagonal_(float("inf"))

        # 获取k最近邻距离 / Get k-nearest neighbor distances
        knn_dists, _ = dist_matrix.topk(k, dim=1, largest=False)

        # MLE内在维度估计 / MLE intrinsic dimension estimation
        # d_hat = (1/n) * sum_i [ 1/(k-1) * sum_{j=1}^{k-1} log(T_k / T_j) ]^{-1}
        log_ratios = torch.log(
            knn_dists[:, -1:].expand_as(knn_dists[:, :-1])
            / knn_dists[:, :-1].clamp(min=1e-10)
        )
        m_k = log_ratios.mean(dim=1)
        intrinsic_dim = 1.0 / m_k.clamp(min=1e-10)
        return float(intrinsic_dim.mean().item())

    def compute_curvature(
        self, path_embeddings: torch.Tensor
    ) -> Tuple[float, float]:
        """
        计算推理路径的曲率统计 / Compute curvature statistics of reasoning paths

        使用离散曲率估计：对于路径中的三个连续点，计算Menger曲率。
        Uses discrete curvature estimation: for three consecutive points in a path,
        compute Menger curvature.

        Args:
            path_embeddings: [N, steps, embed_dim] 多步路径嵌入

        Returns:
            (mean_curvature, curvature_std) 平均曲率和标准差
        """
        n_paths, n_steps, _ = path_embeddings.shape
        if n_steps < 3:
            return 0.0, 0.0

        all_curvatures = []
        for i in range(n_paths):
            path = path_embeddings[i]
            # 计算连续三点组的Menger曲率
            # Compute Menger curvature for consecutive triplets
            for j in range(n_steps - 2):
                p1, p2, p3 = path[j], path[j + 1], path[j + 2]
                # 三角形面积的4倍 / 4 * area of triangle
                area_4x = torch.norm(
                    torch.cross(
                        (p2 - p1)[:3], (p3 - p1)[:3]
                    )
                ) * 2.0
                # 三边长度乘积 / Product of three side lengths
                d12 = torch.norm(p2 - p1)
                d23 = torch.norm(p3 - p2)
                d13 = torch.norm(p3 - p1)
                denom = d12 * d23 * d13
                if denom > 1e-10:
                    curvature = area_4x / denom
                    all_curvatures.append(curvature.item())

        if not all_curvatures:
            return 0.0, 0.0

        curvatures = np.array(all_curvatures)
        return float(np.mean(curvatures)), float(np.std(curvatures))

    def compute_compactness(self, embeddings: torch.Tensor) -> float:
        """
        计算流形紧致度 / Compute manifold compactness

        紧致度 = 1 - (平均成对距离 / 最大可能距离)
        值越接近1表示越紧凑。

        Compactness = 1 - (mean pairwise distance / max possible distance)
        Values closer to 1 indicate tighter clustering.

        Args:
            embeddings: [N, embed_dim] 点集嵌入

        Returns:
            紧致度值 / Compactness value in [0, 1]
        """
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        mean_dist = dist_matrix.mean().item()
        max_dist = dist_matrix.max().item()
        if max_dist < 1e-10:
            return 1.0
        return max(0.0, 1.0 - mean_dist / max_dist)

    def detect_phase_transition(
        self,
        embeddings_scale_a: torch.Tensor,
        embeddings_scale_b: torch.Tensor,
    ) -> float:
        """
        检测两个尺度之间的相变 / Detect phase transition between two scales

        比较两个不同模型尺度下的推理嵌入分布，判断是否发生了质的重构。
        Compares reasoning embedding distributions at two different model scales
        to determine if qualitative restructuring occurred.

        Args:
            embeddings_scale_a: [N, embed_dim] 尺度A的嵌入
            embeddings_scale_b: [N, embed_dim] 尺度B的嵌入

        Returns:
            相变概率 [0, 1] / Phase transition probability
        """
        # 计算各尺度的质心 / Compute centroids per scale
        centroid_a = embeddings_scale_a.mean(dim=0)
        centroid_b = embeddings_scale_b.mean(dim=0)

        # 拼接质心对 / Concatenate centroid pair
        pair = torch.cat([centroid_a, centroid_b])

        # 通过相变检测器 / Pass through phase transition detector
        prob = self.phase_detector(pair.unsqueeze(0))
        return float(prob.item())

    def analyze_domain(
        self,
        embeddings: torch.Tensor,
        path_embeddings: Optional[torch.Tensor] = None,
    ) -> GeometryMetrics:
        """
        对单个领域的推理嵌入进行全面几何分析
        Perform comprehensive geometric analysis for a single domain's reasoning embeddings

        Args:
            embeddings: [N, embed_dim] 路径级嵌入
            path_embeddings: [N, steps, embed_dim] 多步路径嵌入（用于曲率计算）

        Returns:
            GeometryMetrics 包含所有几何度量
        """
        intrinsic_dim = self.compute_intrinsic_dimension(embeddings)
        compactness = self.compute_compactness(embeddings)

        # 路径长度统计 / Path length statistics
        norms = torch.norm(embeddings, dim=1)
        mean_path_length = float(norms.mean().item())
        path_length_std = float(norms.std().item())

        # 曲率计算 / Curvature computation
        mean_curv = 0.0
        curv_std = 0.0
        if path_embeddings is not None:
            mean_curv, curv_std = self.compute_curvature(path_embeddings)

        # 分形维度（简化版关联维度估计）
        # Fractal dimension (simplified correlation dimension estimate)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        dist_matrix.fill_diagonal_(float("inf"))
        min_dists = dist_matrix.min(dim=1)[0]
        mean_min = float(min_dists.mean().item())
        fractal_dim = intrinsic_dim * (1.0 + mean_min / (mean_path_length + 1e-10))

        return GeometryMetrics(
            intrinsic_dim=intrinsic_dim,
            mean_curvature=mean_curv,
            curvature_std=curv_std,
            mean_path_length=mean_path_length,
            path_length_std=path_length_std,
            compactness=compactness,
            fractal_dim=fractal_dim,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        path_embeddings: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        完整的几何分析前向传播 / Full geometric analysis forward pass

        Args:
            embeddings: [N, embed_dim] 路径级嵌入
            path_embeddings: [N, steps, embed_dim] 多步路径嵌入
            domain_labels: [N] 领域标签（用于分类损失）

        Returns:
            包含几何度量和分类logits的字典
        """
        metrics = self.analyze_domain(embeddings, path_embeddings)

        # 领域分类（辅助训练信号）
        # Domain classification (auxiliary training signal)
        domain_logits = self.domain_classifier(embeddings)

        result: Dict[str, Any] = {
            "metrics": metrics,
            "domain_logits": domain_logits,
        }

        # 如果有领域标签，计算分类损失 / Compute classification loss if labels available
        if domain_labels is not None:
            loss = F.cross_entropy(domain_logits, domain_labels)
            result["domain_loss"] = loss

        return result


# ============================================================================
# 跨尺度比较框架 / Cross-Scale Comparison Framework
# ============================================================================


class ScaleComparisonFramework:
    """
    跨尺度比较框架 / Cross-scale comparison framework

    系统性比较不同模型尺度下的推理几何差异，检测相变现象。
    Systematically compares reasoning geometry across model scales and detects
    phase transition phenomena.

    Args:
        analyzer: 几何分析器实例 / Geometry analyzer instance
    """

    def __init__(self, analyzer: ReasoningGeometryAnalyzer) -> None:
        self.analyzer = analyzer
        self.scale_results: Dict[str, GeometryMetrics] = {}

    def procrustes_distance(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> float:
        """
        计算Procrustes距离 / Compute Procrustes distance

        通过最优刚性变换对齐两个点集，测量残余距离。
        Aligns two point sets via optimal rigid transformation and measures residual distance.

        Args:
            X: [N, d] 第一个点集
            Y: [N, d] 第二个点集

        Returns:
            Procrustes距离 / Procrustes distance
        """
        # 中心化 / Center
        X_c = X - X.mean(dim=0, keepdim=True)
        Y_c = Y - Y.mean(dim=0, keepdim=True)

        # 缩放归一化 / Scale normalization
        X_c = X_c / torch.norm(X_c, dim=1, keepdim=True).mean()
        Y_c = Y_c / torch.norm(Y_c, dim=1, keepdim=True).mean()

        # SVD求解最优旋转 / SVD to solve for optimal rotation
        M = X_c.T @ Y_c
        U, _, Vh = torch.linalg.svd(M)
        R = Vh.T @ U.T

        # 对齐后计算距离 / Compute distance after alignment
        Y_aligned = Y_c @ R.T
        distance = torch.norm(X_c - Y_aligned, p="fro") / X_c.shape[0]
        return float(distance.item())

    def compare_scales(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        scale_a_name: str,
        scale_b_name: str,
    ) -> ScaleComparisonResult:
        """
        比较两个尺度的推理几何 / Compare reasoning geometry between two scales

        Args:
            embeddings_a: [N, d] 尺度A的嵌入
            embeddings_b: [N, d] 尺度B的嵌入
            scale_a_name: 尺度A名称
            scale_b_name: 尺度B名称

        Returns:
            ScaleComparisonResult 比较结果
        """
        # 计算各尺度的几何度量 / Compute geometry metrics per scale
        metrics_a = self.analyzer.analyze_domain(embeddings_a)
        metrics_b = self.analyzer.analyze_domain(embeddings_b)

        # Procrustes距离 / Procrustes distance
        proc_dist = self.procrustes_distance(embeddings_a, embeddings_b)

        # 相变检测 / Phase transition detection
        phase_prob = self.analyzer.detect_phase_transition(embeddings_a, embeddings_b)

        return ScaleComparisonResult(
            scale_a=scale_a_name,
            scale_b=scale_b_name,
            procrustes_distance=proc_dist,
            dim_change=metrics_b.intrinsic_dim - metrics_a.intrinsic_dim,
            curvature_change=metrics_b.mean_curvature - metrics_a.mean_curvature,
            topological_similarity=max(0.0, 1.0 - proc_dist),
            phase_transition_detected=phase_prob > 0.5,
            details={
                "metrics_a": metrics_a,
                "metrics_b": metrics_b,
                "phase_probability": phase_prob,
            },
        )


# ============================================================================
# 领域特异性分析 / Domain-Specific Analysis
# ============================================================================


class DomainSpecificAnalysis:
    """
    领域特异性分析 / Domain-specific analysis

    检测和分析不同推理领域的特异性重构模式。
    Detects and analyzes domain-specific restructuring patterns.

    Args:
        analyzer: 几何分析器实例
        domains: 要分析的领域列表
    """

    def __init__(
        self,
        analyzer: ReasoningGeometryAnalyzer,
        domains: Optional[List[ReasoningDomain]] = None,
    ) -> None:
        self.analyzer = analyzer
        self.domains = domains or list(ReasoningDomain)
        self.domain_metrics: Dict[ReasoningDomain, GeometryMetrics] = {}

    def analyze_all_domains(
        self, domain_embeddings: Dict[ReasoningDomain, torch.Tensor]
    ) -> Dict[ReasoningDomain, GeometryMetrics]:
        """
        分析所有领域的几何属性 / Analyze geometry across all domains

        Args:
            domain_embeddings: 每个领域的嵌入字典

        Returns:
            每个领域的几何度量字典
        """
        for domain, embeddings in domain_embeddings.items():
            if domain in self.domains:
                metrics = self.analyzer.analyze_domain(embeddings)
                self.domain_metrics[domain] = metrics
                logger.info(
                    "Domain %s: intrinsic_dim=%.2f, curvature=%.4f, compactness=%.4f / "
                    "领域 %s: 内在维度=%.2f, 曲率=%.4f, 紧致度=%.4f",
                    domain.value,
                    metrics.intrinsic_dim,
                    metrics.mean_curvature,
                    metrics.compactness,
                    domain.value,
                    metrics.intrinsic_dim,
                    metrics.mean_curvature,
                    metrics.compactness,
                )
        return self.domain_metrics

    def find_restructuring_outliers(
        self, threshold: float = 2.0
    ) -> List[Tuple[ReasoningDomain, str, float]]:
        """
        发现重构异常值 / Find restructuring outliers

        识别在某些几何度量上显著偏离平均水平的领域。
        Identifies domains that significantly deviate from the average on certain metrics.

        Args:
            threshold: Z-score阈值 / Z-score threshold

        Returns:
            异常值列表: (领域, 度量名, z-score)
        """
        if not self.domain_metrics:
            return []

        outliers: List[Tuple[ReasoningDomain, str, float]] = []
        metric_names = ["intrinsic_dim", "mean_curvature", "compactness", "fractal_dim"]

        for metric_name in metric_names:
            values = [
                getattr(m, metric_name) for m in self.domain_metrics.values()
            ]
            if len(values) < 2:
                continue
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val < 1e-10:
                continue

            for domain, metrics in self.domain_metrics.items():
                z_score = (getattr(metrics, metric_name) - mean_val) / std_val
                if abs(z_score) > threshold:
                    outliers.append((domain, metric_name, z_score))
                    logger.info(
                        "Outlier detected: %s on %s (z=%.2f) / "
                        "检测到异常值: %s 在 %s 上 (z=%.2f)",
                        domain.value,
                        metric_name,
                        z_score,
                        domain.value,
                        metric_name,
                        z_score,
                    )

        return outliers


# ============================================================================
# 可视化 / Visualization
# ============================================================================


def visualize_geometry(
    embeddings: torch.Tensor,
    domain_labels: torch.Tensor,
    save_path: str = "geometry_plot.png",
    method: str = "pca",
    title: str = "Reasoning Path Geometry / 推理路径几何",
) -> None:
    """
    可视化推理路径的几何结构 / Visualize reasoning path geometry

    将高维嵌入投影到2D/3D空间，按领域着色。
    Projects high-dimensional embeddings to 2D/3D space, colored by domain.

    Args:
        embeddings: [N, embed_dim] 嵌入
        domain_labels: [N] 领域标签
        save_path: 保存路径 / Save path
        method: 降维方法 / Dimensionality reduction method
        title: 图表标题 / Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib required for visualization / 可视化需要matplotlib"
        )
        return

    emb_np = embeddings.detach().cpu().numpy()
    labels_np = domain_labels.detach().cpu().numpy()

    # 降维到2D / Reduce to 2D
    if method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
            emb_2d = reducer.fit_transform(emb_np)
        except ImportError:
            logger.warning(
                "umap-learn not available, falling back to PCA / "
                "umap-learn不可用，回退到PCA"
            )
            method = "pca"

    if method == "pca":
        from numpy.linalg import svd
        emb_centered = emb_np - emb_np.mean(axis=0)
        U, S, Vt = svd(emb_centered, full_matrices=False)
        emb_2d = emb_centered @ Vt[:2].T

    # 绘图 / Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    unique_labels = np.unique(labels_np)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

    for i, label in enumerate(unique_labels):
        mask = labels_np == label
        domain_name = (
            list(ReasoningDomain)[int(label)].value if int(label) < len(list(ReasoningDomain)) else f"domain_{label}"
        )
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=[colors[i]],
            label=domain_name,
            alpha=0.6,
            s=20,
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1 / 成分1")
    ax.set_ylabel("Component 2 / 成分2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Geometry visualization saved to / 几何可视化已保存到: %s", save_path)


# ============================================================================
# 主函数 / Main
# ============================================================================


def main() -> None:
    """
    演示推理几何分析的完整流程
    Demonstrates the full reasoning geometry analysis pipeline
    """
    logger.info(
        "Starting reasoning geometry analysis demo / "
        "开始推理几何分析演示"
    )

    # 初始化模型 / Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 128
    input_dim = 128  # 与合成数据匹配 / Match synthetic data

    path_embedder = ThoughtPathEmbedding(
        input_dim=input_dim, embed_dim=embed_dim
    ).to(device)
    analyzer = ReasoningGeometryAnalyzer(embed_dim=embed_dim).to(device)
    scale_framework = ScaleComparisonFramework(analyzer)
    domain_analysis = DomainSpecificAnalysis(analyzer)

    # 加载各领域数据 / Load data for each domain
    domains = [
        ReasoningDomain.MATHEMATICS,
        ReasoningDomain.LEGAL,
        ReasoningDomain.COMMON_SENSE,
        ReasoningDomain.SCIENCE,
    ]

    domain_embeddings: Dict[ReasoningDomain, torch.Tensor] = {}
    all_embeddings = []
    all_labels = []

    for domain in domains:
        dataset = ReasoningPathDataset(
            data_dir="data/reasoning_paths",
            domain=domain,
        )
        logger.info(
            "Loaded %d samples for domain %s / "
            "领域 %s 加载了 %d 个样本",
            len(dataset),
            domain.value,
            domain.value,
            len(dataset),
        )

        # 生成嵌入 / Generate embeddings
        with torch.no_grad():
            batch_embs = []
            for i in range(len(dataset)):
                sample = dataset[i]
                acts = sample["activations"].unsqueeze(0).to(device)
                emb = path_embedder(acts)
                batch_embs.append(emb.cpu())
            domain_emb = torch.cat(batch_embs, dim=0)
            domain_embeddings[domain] = domain_emb

        all_embeddings.append(domain_emb)
        all_labels.append(
            torch.full((domain_emb.shape[0],), domains.index(domain))
        )

    # 合并所有嵌入 / Concatenate all embeddings
    combined_embeddings = torch.cat(all_embeddings, dim=0)
    combined_labels = torch.cat(all_labels, dim=0)

    # 领域特异性分析 / Domain-specific analysis
    logger.info("=" * 60)
    logger.info("Domain-Specific Analysis / 领域特异性分析")
    logger.info("=" * 60)
    metrics_dict = domain_analysis.analyze_all_domains(domain_embeddings)

    # 查找重构异常值 / Find restructuring outliers
    outliers = domain_analysis.find_restructuring_outliers(threshold=1.5)
    if outliers:
        logger.info(
            "Restructuring outliers found / 发现重构异常值: %d entries",
            len(outliers),
        )

    # 跨尺度比较（模拟两个"尺度"）
    # Cross-scale comparison (simulating two "scales")
    logger.info("=" * 60)
    logger.info("Cross-Scale Comparison / 跨尺度比较")
    logger.info("=" * 60)
    # 模拟小模型：添加噪声扰动 / Simulate small model: add noise perturbation
    small_scale_emb = combined_embeddings + torch.randn_like(combined_embeddings) * 0.5
    large_scale_emb = combined_embeddings

    comparison = scale_framework.compare_scales(
        small_scale_emb, large_scale_emb, "small-8B", "large-70B"
    )
    logger.info(
        "Procrustes distance: %.4f | Phase transition: %s | "
        "Procrustes距离: %.4f | 相变: %s",
        comparison.procrustes_distance,
        comparison.phase_transition_detected,
        comparison.procrustes_distance,
        comparison.phase_transition_detected,
    )

    # 可视化 / Visualization
    logger.info("Generating visualization / 生成可视化")
    visualize_geometry(
        combined_embeddings,
        combined_labels,
        save_path="reasoning_geometry_demo.png",
        method="pca",
    )

    logger.info("Analysis complete / 分析完成")


if __name__ == "__main__":
    main()
