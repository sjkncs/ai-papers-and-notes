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
DepthDirector: Depth-Guided Video Generation / 深度导演：深度引导视频生成

Implementation of the core architecture from "Beyond Inpainting: Unleash 3D
Understanding for Precise Camera-Controlled Video Generation" (2601.10214).

实现《Beyond Inpainting: Unleash 3D Understanding for Precise Camera-Controlled
Video Generation》(2601.10214) 的核心架构。

Core components / 核心组件:
- DepthGuidedVideoGenerator: Core depth-guided generation model
- DualStreamMotionPerceiver: Dual-stream motion perception
- CameraTrajectoryEncoder: Camera path encoding with depth awareness
- LoRAAdapter: Efficient LoRA fine-tuning module
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 配置日志 / Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ============================================================================
# 数据类 / Data Classes
# ============================================================================


@dataclass
class CameraParameters:
    """
    相机参数 / Camera parameters

    描述单帧的相机状态，包含内参和外参。
    Describes camera state for a single frame, including intrinsics and extrinsics.
    """

    # 内参 / Intrinsics
    fov: float = 60.0  # 视野角度 / Field of view (degrees)
    aspect_ratio: float = 16.0 / 9.0
    # 外参（位置和旋转） / Extrinsics (position and rotation)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 欧拉角（度）/ Euler angles (degrees)

    def to_tensor(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """将参数转换为张量 / Convert parameters to tensor"""
        return torch.tensor(
            [
                self.fov,
                self.aspect_ratio,
                *self.position,
                *self.rotation,
            ],
            dtype=torch.float32,
            device=device,
        )


@dataclass
class CameraTrajectory:
    """
    相机轨迹 / Camera trajectory

    描述跨多帧的相机运动路径。
    Describes camera motion path across multiple frames.
    """

    # 逐帧参数 / Per-frame parameters
    frames: List[CameraParameters] = field(default_factory=list)
    # 帧率 / Frame rate
    fps: int = 30
    # 运动类型标签 / Motion type label
    motion_type: str = "free_trajectory"


# ============================================================================
# LoRA适配器 / LoRA Adapter
# ============================================================================


class LoRAAdapter(nn.Module):
    """
    LoRA低秩适配器 / LoRA low-rank adapter

    实现参数高效的微调，通过在原始权重旁添加低秩分解矩阵。
    Implements parameter-efficient fine-tuning by adding low-rank decomposition
    matrices alongside original weights.

    Args:
        in_features: 输入特征维度 / Input feature dimension
        out_features: 输出特征维度 / Output feature dimension
        rank: 低秩分解的秩 / Rank of low-rank decomposition
        alpha: 缩放因子 / Scaling factor
        dropout: 丢弃率 / Dropout rate
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # 低秩分解矩阵 / Low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算LoRA增量 / Compute LoRA increment

        输出 = x @ (A @ B) * scaling
        这个增量会被加到原始线性层的输出上。

        Args:
            x: 输入张量

        Returns:
            LoRA增量输出
        """
        x = self.dropout(x)
        # 低秩变换 / Low-rank transformation
        lora_out = x @ self.lora_A @ self.lora_B
        return lora_out * self.scaling


# ============================================================================
# 相机轨迹编码器 / Camera Trajectory Encoder
# ============================================================================


class CameraTrajectoryEncoder(nn.Module):
    """
    相机轨迹编码器 / Camera trajectory encoder

    将相机参数序列编码为条件向量，同时融合深度感知信息。
    Encodes camera parameter sequences into conditioning vectors,
    fused with depth-aware information.

    Args:
        param_dim: 相机参数维度 / Camera parameter dimension
        depth_channels: 深度图通道数 / Depth map channels
        embed_dim: 嵌入维度 / Embedding dimension
        max_frames: 最大帧数 / Maximum number of frames
    """

    def __init__(
        self,
        param_dim: int = 8,
        depth_channels: int = 64,
        embed_dim: int = 1024,
        max_frames: int = 25,
    ) -> None:
        super().__init__()
        self.param_dim = param_dim
        self.embed_dim = embed_dim

        # 相机参数MLP / Camera parameter MLP
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

        # 帧位置编码 / Frame positional encoding
        self.frame_pos_enc = nn.Parameter(
            torch.randn(max_frames, embed_dim) * 0.02
        )

        # 深度特征提取器 / Depth feature extractor
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, depth_channels, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.depth_proj = nn.Linear(depth_channels * 16, embed_dim)

        # 融合层 / Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # 时序聚合 / Temporal aggregation
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )

    def forward(
        self,
        camera_params: torch.Tensor,
        depth_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        编码相机轨迹 / Encode camera trajectory

        Args:
            camera_params: [B, T, param_dim] 逐帧相机参数
            depth_maps: [B, T, 1, H, W] 逐帧深度图（可选）

        Returns:
            [B, T, embed_dim] 编码后的相机条件
        """
        batch_size, n_frames, _ = camera_params.shape

        # 编码相机参数 / Encode camera parameters
        cam_emb = self.param_encoder(camera_params)

        # 添加帧位置编码 / Add frame positional encoding
        cam_emb = cam_emb + self.frame_pos_enc[:n_frames].unsqueeze(0)

        # 深度编码（如果可用）/ Depth encoding (if available)
        if depth_maps is not None:
            B, T, C, H, W = depth_maps.shape
            depth_flat = depth_maps.reshape(B * T, C, H, W)
            depth_feat = self.depth_encoder(depth_flat)
            depth_feat = depth_feat.reshape(B * T, -1)
            depth_emb = self.depth_proj(depth_feat)
            depth_emb = depth_emb.reshape(B, T, -1)
            # 融合相机和深度嵌入 / Fuse camera and depth embeddings
            combined = torch.cat([cam_emb, depth_emb], dim=-1)
        else:
            combined = torch.cat(
                [cam_emb, torch.zeros_like(cam_emb)], dim=-1
            )

        fused = self.fusion(combined)

        # 时序注意力 / Temporal attention
        fused, _ = self.temporal_attn(fused, fused, fused)

        return fused


# ============================================================================
# 双流运动感知器 / Dual-Stream Motion Perceiver
# ============================================================================


class DualStreamMotionPerceiver(nn.Module):
    """
    双流运动感知模块 / Dual-stream motion perception module

    分离建模相机自身运动和物体动态，通过交叉注意力实现信息交换。
    Separately models camera ego-motion and object dynamics, with cross-attention
    for information exchange when needed.

    Args:
        hidden_dim: 隐藏层维度 / Hidden dimension
        n_heads: 注意力头数 / Number of attention heads
        n_layers: 每流的层数 / Layers per stream
        n_cross_layers: 交叉注意力层数 / Cross-attention layers
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        n_heads: int = 8,
        n_layers: int = 4,
        n_cross_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # === 相机运动流 / Camera motion stream ===
        camera_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.camera_stream = nn.TransformerEncoder(
            camera_layer, num_layers=n_layers
        )

        # === 物体动态流 / Object dynamics stream ===
        object_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.object_stream = nn.TransformerEncoder(
            object_layer, num_layers=n_layers
        )

        # === 交叉注意力融合 / Cross-attention fusion ===
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(n_cross_layers):
            # 相机→物体交叉注意力 / Camera-to-object cross-attention
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    hidden_dim, n_heads, batch_first=True, dropout=0.1
                )
            )
            # 物体→相机交叉注意力 / Object-to-camera cross-attention
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    hidden_dim, n_heads, batch_first=True, dropout=0.1
                )
            )

        self.cross_layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_cross_layers * 2)]
        )

        # 最终融合 / Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        camera_features: torch.Tensor,
        object_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        双流前向传播 / Dual-stream forward pass

        Args:
            camera_features: [B, T, hidden_dim] 相机特征
            object_features: [B, T, hidden_dim] 物体特征

        Returns:
            (camera_out, object_out, fused) 三个输出张量
        """
        # 独立流处理 / Independent stream processing
        cam_out = self.camera_stream(camera_features)
        obj_out = self.object_stream(object_features)

        # 交叉注意力融合 / Cross-attention fusion
        for i in range(0, len(self.cross_attn_layers), 2):
            # 相机→物体 / Camera-to-object
            cross1, _ = self.cross_attn_layers[i](
                query=obj_out, key=cam_out, value=cam_out
            )
            obj_out = self.cross_layer_norms[i](obj_out + cross1)

            # 物体→相机 / Object-to-camera
            cross2, _ = self.cross_attn_layers[i + 1](
                query=cam_out, key=obj_out, value=obj_out
            )
            cam_out = self.cross_layer_norms[i + 1](cam_out + cross2)

        # 最终融合 / Final fusion
        fused = self.final_fusion(torch.cat([cam_out, obj_out], dim=-1))

        return cam_out, obj_out, fused


# ============================================================================
# 深度引导视频生成器 / Depth-Guided Video Generator
# ============================================================================


class DepthGuidedVideoGenerator(nn.Module):
    """
    深度引导视频生成器 / Depth-guided video generator

    整合所有组件：深度编码、相机控制、双流运动感知和LoRA适配。
    Integrates all components: depth encoding, camera control, dual-stream
    motion perception, and LoRA adaptation.

    Args:
        hidden_dim: 隐藏层维度 / Hidden dimension
        n_frames: 输出帧数 / Number of output frames
        image_size: 输出图像尺寸 / Output image size (H, W)
        lora_rank: LoRA秩 / LoRA rank
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        n_frames: int = 25,
        image_size: Tuple[int, int] = (576, 1024),
        lora_rank: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_frames = n_frames
        self.image_size = image_size

        # 视觉编码器（冻结预训练权重）
        # Visual encoder (frozen pretrained weights)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(8),
        )
        self.visual_proj = nn.Linear(256 * 64, hidden_dim)

        # 相机轨迹编码器 / Camera trajectory encoder
        self.camera_encoder = CameraTrajectoryEncoder(
            embed_dim=hidden_dim, max_frames=n_frames
        )

        # 双流运动感知器 / Dual-stream motion perceiver
        self.motion_perceiver = DualStreamMotionPerceiver(
            hidden_dim=hidden_dim
        )

        # LoRA适配器（用于高效微调）
        # LoRA adapters (for efficient fine-tuning)
        self.lora_q = LoRAAdapter(hidden_dim, hidden_dim, rank=lora_rank)
        self.lora_v = LoRAAdapter(hidden_dim, hidden_dim, rank=lora_rank)

        # 视频解码器 / Video decoder
        self.video_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256 * 64),
            nn.GELU(),
        )
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        # 深度一致性约束网络 / Depth consistency constraint network
        self.depth_consistency_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def _extract_visual_features(
        self, frames: torch.Tensor
    ) -> torch.Tensor:
        """
        提取视觉特征 / Extract visual features

        Args:
            frames: [B, T, C, H, W] 视频帧

        Returns:
            [B, T, hidden_dim] 视觉特征
        """
        B, T, C, H, W = frames.shape
        flat_frames = frames.reshape(B * T, C, H, W)
        feat = self.visual_encoder(flat_frames)
        feat = feat.reshape(B * T, -1)
        feat = self.visual_proj(feat)
        return feat.reshape(B, T, -1)

    def forward(
        self,
        reference_frame: torch.Tensor,
        camera_trajectory: CameraTrajectory,
        depth_maps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        生成深度引导的视频 / Generate depth-guided video

        Args:
            reference_frame: [B, 3, H, W] 参考帧
            camera_trajectory: 相机轨迹参数
            depth_maps: [B, T, 1, H, W] 深度图序列
            noise: [B, T, 3, H, W] 初始噪声（可选）

        Returns:
            [B, T, 3, H, W] 生成的视频帧
        """
        batch_size = reference_frame.shape[0]
        device = reference_frame.device
        n_frames = len(camera_trajectory.frames)

        # 编码参考帧 / Encode reference frame
        ref_feat = self._extract_visual_features(
            reference_frame.unsqueeze(1)
        )  # [B, 1, hidden_dim]

        # 编码相机轨迹 / Encode camera trajectory
        cam_params = torch.stack(
            [f.to_tensor(device) for f in camera_trajectory.frames], dim=0
        )  # [T, param_dim]
        cam_params = cam_params.unsqueeze(0).expand(batch_size, -1, -1)

        cam_emb = self.camera_encoder(
            cam_params, depth_maps
        )  # [B, T, hidden_dim]

        # 扩展参考特征到所有帧 / Expand reference features to all frames
        obj_feat = ref_feat.expand(-1, n_frames, -1)

        # 添加LoRA增量到物体特征 / Add LoRA increment to object features
        obj_feat = obj_feat + self.lora_q(obj_feat)

        # 双流运动感知 / Dual-stream motion perception
        cam_out, obj_out, fused = self.motion_perceiver(cam_emb, obj_feat)

        # 添加LoRA增量到融合特征 / Add LoRA increment to fused features
        fused = fused + self.lora_v(fused)

        # 解码视频帧 / Decode video frames
        B, T, D = fused.shape
        decoded = self.video_decoder(fused)  # [B, T, 256*64]
        decoded = decoded.reshape(B * T, 256, 8, 8)
        video_frames = self.upsampler(decoded)  # [B*T, 3, H', W']

        # 调整到目标尺寸 / Resize to target size
        video_frames = F.interpolate(
            video_frames,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        video_frames = video_frames.reshape(
            B, T, 3, self.image_size[0], self.image_size[1]
        )

        return video_frames

    def compute_depth_consistency_loss(
        self,
        generated_frames: torch.Tensor,
        target_depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算深度一致性损失 / Compute depth consistency loss

        确保生成视频的深度结构与目标深度图一致。
        Ensures depth structure of generated video matches target depth maps.

        Args:
            generated_frames: [B, T, 3, H, W] 生成帧
            target_depth: [B, T, 1, H, W] 目标深度图

        Returns:
            深度一致性损失标量
        """
        B, T, _, H, W = generated_frames.shape

        # 从生成帧估计深度 / Estimate depth from generated frames
        gray_frames = generated_frames.mean(dim=2, keepdim=True)  # [B, T, 1, H, W]
        gray_flat = gray_frames.reshape(B * T, 1, H, W)
        estimated_depth = self.depth_consistency_net(gray_flat)

        # 调整目标深度尺寸 / Resize target depth
        target_flat = target_depth.reshape(B * T, 1, H, W)
        target_resized = F.interpolate(
            target_flat,
            size=estimated_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # L1 + 梯度一致性损失 / L1 + gradient consistency loss
        l1_loss = F.l1_loss(estimated_depth, target_resized)

        # 梯度一致性（确保深度边缘匹配）
        # Gradient consistency (ensure depth edges match)
        est_grad_x = estimated_depth[:, :, :, 1:] - estimated_depth[:, :, :, :-1]
        est_grad_y = estimated_depth[:, :, 1:, :] - estimated_depth[:, :, :-1, :]
        tgt_grad_x = target_resized[:, :, :, 1:] - target_resized[:, :, :, :-1]
        tgt_grad_y = target_resized[:, :, 1:, :] - target_resized[:, :, :-1, :]

        grad_loss = F.l1_loss(est_grad_x, tgt_grad_x) + F.l1_loss(
            est_grad_y, tgt_grad_y
        )

        return l1_loss + 0.5 * grad_loss


# ============================================================================
# 视频生成函数 / Video Generation Function
# ============================================================================


def generate_video(
    model: DepthGuidedVideoGenerator,
    reference_image: torch.Tensor,
    camera_motion: str = "pan_left_right",
    duration_sec: float = 2.0,
    fps: int = 25,
    depth_maps: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    生成相机控制视频 / Generate camera-controlled video

    高级API：输入参考图像和相机运动描述，生成视频。
    High-level API: input reference image and camera motion description,
    generate video.

    Args:
        model: 视频生成模型
        reference_image: [1, 3, H, W] 参考图像
        camera_motion: 运动类型（pan_left_right, tilt_up_down, orbit等）
        duration_sec: 视频持续时间（秒）
        fps: 帧率
        depth_maps: 可选的深度图序列
        device: 计算设备

    Returns:
        [1, T, 3, H, W] 生成的视频
    """
    n_frames = int(duration_sec * fps)

    # 构建相机轨迹 / Build camera trajectory
    trajectory = CameraTrajectory(fps=fps, motion_type=camera_motion)

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)  # 归一化时间 / Normalized time

        if camera_motion == "pan_left_right":
            # 水平平移 / Horizontal pan
            angle = -30.0 + 60.0 * t
            params = CameraParameters(
                position=(0.0, 0.0, 0.0),
                rotation=(0.0, angle, 0.0),
            )
        elif camera_motion == "tilt_up_down":
            # 垂直俯仰 / Vertical tilt
            angle = -15.0 + 30.0 * t
            params = CameraParameters(
                position=(0.0, 0.0, 0.0),
                rotation=(angle, 0.0, 0.0),
            )
        elif camera_motion == "dolly_in_out":
            # 前后推进 / Dolly in/out
            z = 5.0 - 4.0 * t
            params = CameraParameters(
                position=(0.0, 0.0, z),
                rotation=(0.0, 0.0, 0.0),
            )
        elif camera_motion == "orbit":
            # 环绕运动 / Orbital motion
            angle = 360.0 * t
            radius = 3.0
            x = radius * math.cos(math.radians(angle))
            z = radius * math.sin(math.radians(angle))
            params = CameraParameters(
                position=(x, 0.0, z),
                rotation=(0.0, -angle, 0.0),
            )
        else:
            # 自由轨迹 / Free trajectory
            params = CameraParameters()

        trajectory.frames.append(params)

    # 生成视频 / Generate video
    model.eval()
    with torch.no_grad():
        video = model(
            reference_frame=reference_image.to(device),
            camera_trajectory=trajectory,
            depth_maps=depth_maps.to(device) if depth_maps is not None else None,
        )

    logger.info(
        "Generated video: shape=%s, motion=%s, duration=%.1fs / "
        "生成视频: 形状=%s, 运动=%s, 时长=%.1fs",
        tuple(video.shape),
        camera_motion,
        duration_sec,
        tuple(video.shape),
        camera_motion,
        duration_sec,
    )

    return video


# ============================================================================
# 主函数 / Main
# ============================================================================


def main() -> None:
    """
    演示深度引导视频生成的完整流程
    Demonstrates the full depth-guided video generation pipeline
    """
    logger.info(
        "Starting DepthDirector demo / 开始DepthDirector演示"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型 / Initialize model
    model = DepthGuidedVideoGenerator(
        hidden_dim=256,
        n_frames=10,
        image_size=(128, 128),
        lora_rank=8,
    ).to(device)

    # 创建参考图像 / Create reference image
    ref_image = torch.randn(1, 3, 128, 128).to(device)

    # 创建模拟深度图 / Create mock depth maps
    depth_maps = torch.rand(1, 10, 1, 128, 128).to(device)

    # 生成视频 / Generate video
    video = generate_video(
        model=model,
        reference_image=ref_image,
        camera_motion="pan_left_right",
        duration_sec=0.4,
        fps=25,
        depth_maps=depth_maps,
        device=device,
    )

    logger.info(
        "Output video shape / 输出视频形状: %s",
        tuple(video.shape),
    )

    # 计算深度一致性损失 / Compute depth consistency loss
    depth_loss = model.compute_depth_consistency_loss(video, depth_maps)
    logger.info(
        "Depth consistency loss / 深度一致性损失: %.4f",
        depth_loss.item(),
    )

    # 统计参数量 / Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(
        p.numel() for n, p in model.named_parameters() if "lora" in n
    )
    logger.info(
        "Total params: %d | LoRA params: %d (%.2f%%) / "
        "总参数: %d | LoRA参数: %d (%.2f%%)",
        total_params,
        lora_params,
        lora_params / total_params * 100,
        total_params,
        lora_params,
        lora_params / total_params * 100,
    )

    logger.info("Demo complete / 演示完成")


if __name__ == "__main__":
    main()
