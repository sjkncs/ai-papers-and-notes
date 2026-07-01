"""
GEAR: Guided End-to-End AutoRegression for Image Synthesis —— 核心思想复现
GEAR: Guided End-to-End AutoRegression for Image Synthesis — Core Idea Reproduction

作者 / Authors: Bin Lin et al. (Peking University, Tencent Hunyuan)
论文 / Paper: arXiv:2606.32039

本代码为教学级简化实现，展示 GEAR 的核心创新：
This is a pedagogical simplified implementation demonstrating GEAR's core innovation:
    1. VQ tokenizer 与 AR generator 联合端到端训练 / Joint end-to-end training of VQ tokenizer + AR generator
    2. Dual read-out: hard one-hot branch + soft differentiable branch / Dual read-out mechanism
    3. Representation alignment (REPA-style) 引导 tokenizer / REPA-style alignment guiding the tokenizer

注意：使用合成数据（随机图像）运行一个训练步，无需真实数据集。
Note: Uses synthetic random images for one training step; no real dataset required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------ 超参数 / Hyperparameters ------------------------------
BATCH_SIZE = 4
IMG_SIZE = 32          # 输入图像尺寸 / input image size
IN_CH = 3              # 输入通道 / input channels
LATENT_H = 8           # 下采样后空间尺寸 / spatial size after downsample
LATENT_DIM = 32        # 每个 latent 向量维度 / latent vector dimension
CODEBOOK_SIZE = 64     # codebook 大小 / codebook size
N_HEADS = 4
N_LAYERS = 3
AR_DIM = 128           # AR transformer 维度 / AR transformer dimension
TEMPERATURE = 0.5      # soft assignment 温度 / temperature for soft assignment
ALIGN_WEIGHT = 0.1     # REPA alignment loss 权重 / REPA alignment loss weight


def pretrain_dinov2_stub():
    """
    论文使用预训练 DINOv2 作为 alignment target。
    这里用一个小型 CNN 模拟，仅用于演示 loss 计算。
    In the paper a pretrained DINOv2 is used as alignment target.
    Here we use a tiny CNN stub just to demonstrate the loss computation.

    设计为两次 stride=2 下采样，使 32x32 输入得到 8x8 的 patch 特征，
    与 tokenizer latent 的空间分辨率一致。
    Two stride-2 convolutions so a 32x32 input yields 8x8 patch features,
    matching the tokenizer latent spatial resolution.
    """
    return nn.Sequential(
        nn.Conv2d(IN_CH, 32, 4, 2, 1),  # 32x32 -> 16x16
        nn.ReLU(),
        nn.Conv2d(32, LATENT_DIM, 4, 2, 1),  # 16x16 -> 8x8
    )


# ------------------------------ VQ Tokenizer ------------------------------
class VQTokenizer(nn.Module):
    """
    简化版 VQ-VAE tokenizer：Encoder -> Codebook -> Decoder
    Simplified VQ-VAE tokenizer: Encoder -> Codebook -> Decoder
    """
    def __init__(self):
        super().__init__()
        # 下采样编码器 / downsampling encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(IN_CH, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, LATENT_DIM, 4, 2, 1),  # 32x32 -> 8x8
        )
        # 可学习 codebook / learnable codebook
        self.codebook = nn.Embedding(CODEBOOK_SIZE, LATENT_DIM)
        # 上采样解码器 / upsampling decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, IN_CH, 4, 2, 1),
        )

    def forward(self, x):
        """
        返回：重建图像、硬索引、硬 latent、软 latent、到 codebook 的距离
        Returns: reconstructed image, hard indices, hard latents, soft latents, distances to codebook
        """
        z = self.encoder(x)                       # (B, LATENT_DIM, H, W)
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

        # 计算到每个 codebook entry 的 L2 距离 / L2 distance to each codebook entry
        # distances[b,k] = ||z_flat[b] - codebook[k]||^2
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1, keepdim=True).t()
        )

        # Hard branch: argmin -> one-hot / 硬分支：最近邻索引
        indices = distances.argmin(dim=1)
        hard_latents = self.codebook(indices)     # (B*H*W, C)
        hard_latents = hard_latents.view(B, H, W, C).permute(0, 3, 1, 2)

        # Soft branch: temperature-scaled softmax / 软分支：温度缩放 softmax
        soft_assign = F.softmax(-distances / TEMPERATURE, dim=1)  # (B*H*W, K)
        soft_latents = soft_assign @ self.codebook.weight         # (B*H*W, C)
        soft_latents = soft_latents.view(B, H, W, C).permute(0, 3, 1, 2)

        # Straight-through estimator for reconstruction / 重建用 STE
        z_q = z + (hard_latents - z).detach()
        x_recon = self.decoder(z_q)

        return x_recon, indices.view(B, H, W), hard_latents, soft_latents, z, distances.view(B, H, W, CODEBOOK_SIZE)


# ------------------------------ AR Generator ------------------------------
class ARGenerator(nn.Module):
    """
    简化版自回归生成器：将 VQ 索引序列按 raster order 预测下一个 token
    Simplified autoregressive generator predicts next token in raster order.
    """
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(CODEBOOK_SIZE, AR_DIM)
        self.pos_emb = nn.Parameter(torch.zeros(1, LATENT_H * LATENT_H, AR_DIM))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=AR_DIM, nhead=N_HEADS, dim_feedforward=AR_DIM * 4, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=N_LAYERS)
        self.head = nn.Linear(AR_DIM, CODEBOOK_SIZE)

    def forward(self, indices):
        """
        indices: (B, H*W)
        返回 logits 和最后一层隐藏状态（用于 REPA） / returns logits and last-layer hidden states for REPA
        """
        B, L = indices.shape
        x = self.token_emb(indices) + self.pos_emb[:, :L, :]
        # causal mask / 因果掩码
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
        h = self.transformer(x, x, tgt_mask=causal_mask)  # (B, L, AR_DIM)
        logits = self.head(h)  # (B, L, CODEBOOK_SIZE)
        return logits, h


# ------------------------------ REPA Alignment Loss ------------------------------
class REPAligner(nn.Module):
    """
    将 AR 的隐藏状态投影到 target 表示空间，计算 MSE 对齐损失。
    Projects AR hidden states into target representation space and computes MSE alignment loss.
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(AR_DIM, LATENT_DIM)
        self.target_encoder = pretrain_dinov2_stub()
        # 冻结 target encoder / freeze target encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(self, ar_hidden, images):
        """
        ar_hidden: (B, L, AR_DIM)
        images: (B, 3, H, W)
        """
        with torch.no_grad():
            target = self.target_encoder(images)  # (B, LATENT_DIM, H, W)
        target = target.flatten(2).transpose(1, 2)  # (B, H*W, LATENT_DIM)
        # AR 输入前 L-1 个 token，隐藏状态对应预测第 2~L 个位置，
        # 因此与 target 的第 1~L 个 patch 特征对齐。
        # The AR receives the first L-1 tokens, so its hidden states predict positions 2..L;
        # align them with target patch features at positions 1..L.
        target = target[:, 1:1 + ar_hidden.size(1), :]
        pred = self.proj(ar_hidden)  # (B, L, LATENT_DIM)
        return F.mse_loss(pred, target)

    def forward_tokenizer(self, soft_latents, images):
        """
        论文中 soft branch 的 REPA loss：绕过上层 AR blocks，直接通过可微 soft assignment
        回传给 tokenizer。这里我们将 soft_latents 与 target encoder 的 patch features 对齐。
        In the paper, the soft branch REPA loss bypasses upper AR blocks and flows back to the tokenizer
        through differentiable soft assignment. Here we align soft_latents with target encoder patch features.

        soft_latents: (B, LATENT_DIM, H, W)
        images: (B, 3, H, W)
        """
        with torch.no_grad():
            target = self.target_encoder(images)  # (B, LATENT_DIM, H, W)
        return F.mse_loss(soft_latents, target)


# ------------------------------ GEAR 训练流程 / GEAR Training Step ------------------------------
def gear_training_step(tokenizer, generator, aligner, images, tokenizer_opt, ar_opt):
    """
    一个 GEAR 训练步：
    A single GEAR training step:
      1. Tokenizer 产生 hard/soft latents / Tokenizer produces hard and soft latents
      2. Hard branch: AR 用 hard indices 做 next-token prediction / AR next-token prediction on hard indices
      3. Soft branch: 可微 alignment loss 只回传给 tokenizer / Differentiable alignment loss only updates tokenizer
      4. Tokenizer 由 reconstruction + alignment 更新 / Tokenizer updated by reconstruction + alignment
    """
    # 1) Tokenizer 前向 / Tokenizer forward
    x_recon, indices, hard_latents, soft_latents, z_encoder, _ = tokenizer(images)

    # 2) Hard branch: AR next-token prediction / 硬分支：AR 下一个 token 预测
    # 将 2D 索引展平为 raster order / flatten 2D indices to raster order
    B = images.size(0)
    flat_indices = indices.view(B, -1)  # (B, L)
    logits, ar_hidden = generator(flat_indices[:, :-1])  # 输入前 L-1 个 / input first L-1 tokens
    ntp_loss = F.cross_entropy(
        logits.reshape(-1, CODEBOOK_SIZE),
        flat_indices[:, 1:].reshape(-1)
    )

    # 3) Soft branch: alignment loss 只更新 tokenizer / 软分支：对齐损失只更新 tokenizer
    # 论文中 soft branch 承载 REPA loss 回传 tokenizer：
    # 将可微的 soft_latents 与冻结 target encoder（DINOv2 stub）提取的 patch 特征对齐。
    # In the paper, the soft branch carries the REPA loss back to the tokenizer:
    # align the differentiable soft_latents with patch features from a frozen target encoder (DINOv2 stub).
    alignment_loss = aligner.forward_tokenizer(soft_latents, images)

    # 4) Reconstruction loss / 重建损失
    recon_loss = F.mse_loss(x_recon, images)

    # 5) 分别优化 / Separate optimization
    # AR 只由 NTP loss 更新 / AR updated only by NTP loss
    ar_opt.zero_grad()
    ntp_loss.backward(retain_graph=True)
    ar_opt.step()

    # Tokenizer 由 recon + alignment 更新，不收 NTP gradient / Tokenizer updated by recon + alignment, no NTP gradient
    tokenizer_opt.zero_grad()
    total_tokenizer_loss = recon_loss + ALIGN_WEIGHT * alignment_loss
    total_tokenizer_loss.backward()
    tokenizer_opt.step()

    return {
        "recon_loss": recon_loss.item(),
        "ntp_loss": ntp_loss.item(),
        "alignment_loss": alignment_loss.item(),
    }


# ------------------------------ 主函数 / Main ------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型 / Models
    tokenizer = VQTokenizer().to(device)
    generator = ARGenerator().to(device)
    aligner = REPAligner().to(device)

    # 优化器：AR 和 Tokenizer 分别优化 / Optimizers: separate AR and tokenizer
    tokenizer_opt = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)
    ar_opt = torch.optim.Adam(generator.parameters(), lr=1e-3)

    # 合成数据 / Synthetic data
    synthetic_images = torch.randn(64, IN_CH, IMG_SIZE, IMG_SIZE)
    dataset = TensorDataset(synthetic_images)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("\n开始训练 / Starting training...")
    for epoch in range(2):
        for batch_idx, (imgs,) in enumerate(loader):
            imgs = imgs.to(device)
            metrics = gear_training_step(tokenizer, generator, aligner, imgs, tokenizer_opt, ar_opt)
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}: {metrics}")
            if batch_idx >= 9:  # 仅演示前 10 个 batch / demo first 10 batches
                break

    print("\n训练完成 / Training finished.")
    print("\n说明 / Note:")
    print("  - 本 demo 使用合成数据和简化模型，仅用于展示 GEAR 的 dual-readout 训练流程。")
    print("  - 真实复现需要完整 ImageNet 数据、大规模 AR transformer、预训练 DINOv2 等。")
    print("  - This demo uses synthetic data and a tiny model to illustrate the dual-readout training flow.")
    print("  - Full reproduction needs ImageNet, a large AR transformer, pretrained DINOv2, etc.")


if __name__ == "__main__":
    main()
