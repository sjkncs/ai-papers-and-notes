"""
Markovian Scale Prediction (MSP) 简化复现 / Simplified Reproduction
=========================================================
论文: "Markovian Scale Prediction: A New Era of Visual Autoregressive Generation"
会议: CVPR 2026
作者: Yu Zhang et al.

本代码演示 Markovian Scale Prediction 的核心思想:
1. 将图像编码为多尺度 token map (模拟 VAR 的离散化表示)
2. 对比 Full-Context VAR (依赖所有先前尺度) 与 MSP (仅依赖最近 w 个尺度)
3. 展示 MSP 如何在保持生成质量的同时大幅降低内存开销

运行方式: python markovian_var.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import List, Tuple


# ------------------------------
# 1. 多尺度图像 Tokenizer / Multi-Scale Image Tokenizer
# ------------------------------
class MultiScaleTokenizer(nn.Module):
    """
    将输入图像编码为 S 个尺度的 token map。
    模拟 VAR 中从低分辨率到高分辨率的离散化过程。
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 256, num_scales: int = 5):
        super().__init__()
        self.num_scales = num_scales
        self.embed_dim = embed_dim
        # 每个尺度使用独立的卷积编码器
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 2, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ) for _ in range(num_scales)
        ])
        # 位置编码 (二维正弦余弦)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, embed_dim))  # 预分配最大长度

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        输入: x [B, C, H, W]
        输出: 长度为 num_scales 的列表，每个元素为 [B, embed_dim, h_s, w_s]
        """
        tokens = []
        for s in range(self.num_scales):
            # 对原图进行 2^s 倍下采样，模拟 VAR 中从 1x1 到 HxW 的尺度序列
            scale_factor = 1.0 / (2 ** s)
            x_scaled = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            t = self.encoders[s](x_scaled)  # [B, embed_dim, h, w]
            tokens.append(t)
        return tokens


# ------------------------------
# 2. Full-Context VAR 注意力 / Full-Context VAR Attention
# ------------------------------
class FullContextVARBlock(nn.Module):
    """
    原始 VAR: 预测第 s 个尺度时，使用所有先前尺度的完整上下文。
    将之前所有尺度的 token flatten 后拼接，作为 Key/Value。
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, current_scale_tokens: torch.Tensor, previous_scales: List[torch.Tensor]) -> torch.Tensor:
        """
        current_scale_tokens: [B, embed_dim, h, w] — 当前待预测的尺度
        previous_scales: 列表，包含所有之前尺度的 token map
        返回: [B, embed_dim, h, w]
        """
        B, C, H, W = current_scale_tokens.shape
        # 将当前尺度 flatten 为 query
        q = current_scale_tokens.flatten(2).transpose(1, 2)  # [B, H*W, C]
        # 将所有先前尺度的 token flatten 并拼接，作为 key/value 的上下文
        context_tokens = []
        for ps in previous_scales:
            context_tokens.append(ps.flatten(2).transpose(1, 2))  # [B, h_i*w_i, C]
        if len(context_tokens) > 0:
            context = torch.cat(context_tokens, dim=1)  # [B, total_prev_tokens, C]
            full_context = torch.cat([context, q], dim=1)  # [B, total_prev + H*W, C]
        else:
            full_context = q

        # 自注意力计算
        qkv = self.qkv(full_context).reshape(B, full_context.size(1), 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, seq_len, head_dim]
        q_all, k_all, v_all = qkv[0], qkv[1], qkv[2]

        attn = (q_all @ k_all.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v_all).transpose(1, 2).reshape(B, full_context.size(1), C)
        out = self.proj(out)
        out = self.norm(out)

        # 只取出当前尺度对应的部分
        current_out = out[:, -H*W:, :].transpose(1, 2).reshape(B, C, H, W)
        return current_out


# ------------------------------
# 3. Markovian Scale Prediction (MSP) / 马尔可夫尺度预测
# ------------------------------
class MarkovianScaleBlock(nn.Module):
    """
    MSP 核心: 预测第 s 个尺度时，只使用最近 window_size 个尺度的上下文。
    通过滑动窗口截断远距离尺度，显著降低内存峰值。
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, window_size: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, current_scale_tokens: torch.Tensor, previous_scales: List[torch.Tensor]) -> torch.Tensor:
        """
        current_scale_tokens: [B, embed_dim, h, w]
        previous_scales: 所有之前尺度的列表，但只取最近 window_size 个
        """
        B, C, H, W = current_scale_tokens.shape
        q = current_scale_tokens.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # --- MSP 核心操作: 只保留最近 window_size 个尺度的上下文 ---
        context_tokens = []
        # 取 previous_scales 的最后 window_size 个元素
        recent_scales = previous_scales[-self.window_size:] if len(previous_scales) > 0 else []
        for ps in recent_scales:
            context_tokens.append(ps.flatten(2).transpose(1, 2))
        # -------------------------------

        if len(context_tokens) > 0:
            context = torch.cat(context_tokens, dim=1)  # [B, recent_tokens, C]
            full_context = torch.cat([context, q], dim=1)
        else:
            full_context = q

        qkv = self.qkv(full_context).reshape(B, full_context.size(1), 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q_all, k_all, v_all = qkv[0], qkv[1], qkv[2]

        attn = (q_all @ k_all.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v_all).transpose(1, 2).reshape(B, full_context.size(1), C)
        out = self.proj(out)
        out = self.norm(out)

        current_out = out[:, -H*W:, :].transpose(1, 2).reshape(B, C, H, W)
        return current_out


# ------------------------------
# 4. 生成器 / Generator
# ------------------------------
class ScalePredictionGenerator(nn.Module):
    """
    自回归地逐个尺度预测 token map。
    从最低分辨率 (scale 0) 开始，逐步生成到最高分辨率。
    """
    def __init__(self, embed_dim: int = 256, num_scales: int = 5, use_markovian: bool = True, window_size: int = 2):
        super().__init__()
        self.num_scales = num_scales
        self.use_markovian = use_markovian
        self.scale_blocks = nn.ModuleList([
            MarkovianScaleBlock(embed_dim, window_size=window_size) if use_markovian
            else FullContextVARBlock(embed_dim)
            for _ in range(num_scales)
        ])
        # 每个尺度的输出 head: 预测下一个尺度的特征
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            ) for _ in range(num_scales)
        ])
        # 上采样模块，将当前尺度映射到下一个尺度的分辨率
        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
            for _ in range(num_scales - 1)
        ])

    def forward(self, tokens: List[torch.Tensor]) -> Tuple[List[torch.Tensor], dict]:
        """
        tokens: 真实的多尺度 token (用于 teacher forcing 训练)
        返回: (predicted_tokens, stats)
        """
        B = tokens[0].size(0)
        predicted = []
        previous = []
        peak_memory = 0.0
        total_tokens_seen = 0

        for s in range(self.num_scales):
            # 当前尺度的输入: 如果是第一个尺度，用随机初始化; 否则用上一步预测+上采样
            if s == 0:
                current = torch.randn_like(tokens[0]) * 0.01
            else:
                current = self.upsamplers[s-1](predicted[-1])
                # 裁剪到目标尺寸
                _, _, th, tw = tokens[s].shape
                current = F.interpolate(current, size=(th, tw), mode='bilinear', align_corners=False)

            # 通过注意力块预测
            out = self.scale_blocks[s](current, previous)
            out = self.heads[s](out)
            predicted.append(out)
            previous.append(out)

            # 记录内存统计 (仅当前 GPU 显存峰值)
            if torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                peak_memory = max(peak_memory, mem_mb)
            total_tokens_seen += sum(p.numel() for p in previous)

        stats = {
            'peak_memory_mb': peak_memory,
            'total_tokens_seen': total_tokens_seen,
            'num_scales': self.num_scales,
            'mode': 'Markovian' if self.use_markovian else 'FullContext',
        }
        return predicted, stats


# ------------------------------
# 5. 完整 VAR-MSP 模型 / Full VAR-MSP Model
# ------------------------------
class MarkovianVAR(nn.Module):
    """
    端到端模型: Tokenizer + Scale-by-Scale Generator
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 256, num_scales: int = 5,
                 use_markovian: bool = True, window_size: int = 2):
        super().__init__()
        self.tokenizer = MultiScaleTokenizer(in_channels, embed_dim, num_scales)
        self.generator = ScalePredictionGenerator(embed_dim, num_scales, use_markovian, window_size)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], dict]:
        target_tokens = self.tokenizer(images)
        predicted_tokens, stats = self.generator(target_tokens)
        return predicted_tokens, stats


# ------------------------------
# 6. 训练与对比实验 / Training & Comparison
# ------------------------------
def compare_memory_and_quality():
    """
    对比 Full-Context VAR 与 Markovian Scale Prediction 的内存开销。
    使用随机数据模拟训练过程。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"运行设备 / Device: {device}")
    print("=" * 60)

    # 超参数
    batch_size = 2
    num_scales = 5
    embed_dim = 256
    img_size = 128
    window_size = 2

    # 模拟输入图像
    dummy_images = torch.randn(batch_size, 3, img_size, img_size).to(device)

    results = {}

    for mode_name, use_markov in [("Full-Context VAR", False), ("Markovian VAR (w=2)", True)]:
        model = MarkovianVAR(
            in_channels=3,
            embed_dim=embed_dim,
            num_scales=num_scales,
            use_markovian=use_markov,
            window_size=window_size
        ).to(device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        start = time.time()
        predicted, stats = model(dummy_images)
        elapsed = time.time() - start

        # 计算与目标 token 的 MSE loss (模拟训练信号)
        target_tokens = model.tokenizer(dummy_images)
        loss = 0.0
        for pred, tgt in zip(predicted, target_tokens):
            loss += F.mse_loss(pred, tgt).item()

        peak_mem = stats['peak_memory_mb']
        results[mode_name] = {
            'peak_memory_mb': peak_mem,
            'loss': loss,
            'time_sec': elapsed,
            'total_tokens': stats['total_tokens_seen'],
        }

        print(f"\n【{mode_name}】")
        print(f"  峰值内存 / Peak Memory: {peak_mem:.2f} MB")
        print(f"  重构损失 / Reconstruction Loss: {loss:.6f}")
        print(f"  前向时间 / Forward Time: {elapsed:.4f} sec")
        print(f"  累计处理token数 / Total Tokens: {stats['total_tokens_seen']:,}")

    # 计算对比指标
    print("\n" + "=" * 60)
    print("对比分析 / Comparison Analysis")
    print("=" * 60)
    full_mem = results["Full-Context VAR"]['peak_memory_mb']
    markov_mem = results["Markovian VAR (w=2)"]['peak_memory_mb']
    if full_mem > 0 and markov_mem > 0:
        reduction = (1 - markov_mem / full_mem) * 100
        print(f"内存降低 / Memory Reduction: {reduction:.1f}%")
    else:
        # CPU 模式下显存为0，用理论token数估算
        full_tokens = results["Full-Context VAR"]['total_tokens']
        markov_tokens = results["Markovian VAR (w=2)"]['total_tokens']
        reduction = (1 - markov_tokens / full_tokens) * 100
        print(f"累计token降低 / Token Reduction: {reduction:.1f}% (理论内存节省 / theoretical memory saving)")

    full_loss = results["Full-Context VAR"]['loss']
    markov_loss = results["Markovian VAR (w=2)"]['loss']
    loss_diff = ((markov_loss - full_loss) / full_loss) * 100 if full_loss > 0 else 0
    print(f"损失变化 / Loss Change: {loss_diff:+.2f}% (负值表示MSP更优 / negative means MSP is better)")
    print("=" * 60)


def demo_sliding_window_attention():
    """
    可视化演示: 滑动窗口如何截断远距离尺度的上下文。
    """
    print("\n【滑动窗口机制可视化 / Sliding Window Visualization】")
    num_scales = 5
    window_size = 2
    print(f"总尺度数 / Total scales: {num_scales}, 窗口大小 / Window size: {window_size}")
    for s in range(num_scales):
        if s == 0:
            used = "None (第一个尺度无先前上下文 / first scale has no prior context)"
        else:
            start = max(0, s - window_size)
            used = list(range(start, s))
        print(f"  预测尺度 {s}: 使用先前尺度 / Using previous scales: {used}")
    print()


def demo_markovian_vs_full_context():
    """
    在合成数据上做一个完整的训练循环演示。
    """
    print("\n【合成数据训练演示 / Synthetic Data Training Demo】")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MarkovianVAR(in_channels=3, embed_dim=128, num_scales=4,
                         use_markovian=True, window_size=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 生成简单的合成目标 (带结构的彩色方块)
    def make_synthetic_batch(batch_size=4, size=64):
        imgs = torch.zeros(batch_size, 3, size, size)
        for b in range(batch_size):
            cx, cy = size // 2, size // 2
            color = torch.rand(3)
            radius = size // 4 + int(torch.randint(-5, 6, (1,)).item())
            y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
            mask = ((x - cx)**2 + (y - cy)**2) < radius**2
            imgs[b, :, mask] = color.view(3, 1)
        return imgs.to(device)

    print("训练 20 个 epoch / Training 20 epochs...")
    for epoch in range(20):
        imgs = make_synthetic_batch()
        pred, _ = model(imgs)
        target = model.tokenizer(imgs)
        loss = sum(F.mse_loss(p, t) for p, t in zip(pred, target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d}, Loss: {loss.item():.6f}")
    print("训练完成 / Training complete.\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Markovian Scale Prediction (MSP) 简化复现")
    print("Simplified Reproduction of Markovian Scale Prediction")
    print("=" * 60)

    # 演示1: 滑动窗口机制可视化
    demo_sliding_window_attention()

    # 演示2: 内存与质量对比
    compare_memory_and_quality()

    # 演示3: 合成数据训练
    demo_markovian_vs_full_context()

    print("全部演示完成 / All demos completed.")
