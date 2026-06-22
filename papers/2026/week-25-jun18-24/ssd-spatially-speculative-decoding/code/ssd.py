#!/usr/bin/env python3
"""
SSD: Spatially Speculative Decoding Accelerates Autoregressive Image Generation
PyTorch 简化复现（可在 CPU 直接运行）

论文: https://arxiv.org/abs/2606.20543

本脚本提供一个最小可运行的 Spatially Speculative Decoding (SSD) 演示：
1. 用一个小型 Transformer 作为图像 token 的 backbone；
2. 在 backbone 最后一层隐藏状态上挂载轻量 SSD Head；
3. SSD Head 根据当前 (y, x) 位置的隐藏状态和 token 嵌入，
   同时预测水平方向（同行右侧）和垂直方向（下方多行）token 的隐藏状态；
4. 通过 SmoothL1 损失监督这些预测隐藏状态，使其逼近 backbone 真实前向结果；
5. 推理阶段演示 "draft -> verify -> auto-correct" 的循环。

注意：这是教学级简化实现，未使用真实图像数据集，而是用随机合成的
      离散 token 序列来验证算法流程。
"""

import math
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------- 超参数 ----------------------------
VOCAB_SIZE = 256          # 离散视觉词表大小
D_MODEL = 128             # Transformer / hidden 维度
N_HEAD = 4                # 多头注意力头数
N_LAYER = 4               # Transformer 层数
IMG_H = 8                 # 合成图像高度（token 行数）
IMG_W = 8                 # 合成图像宽度（token 列数）
BATCH_SIZE = 4
LR = 1e-3
TRAIN_STEPS = 200
SEED = 42

# 空间 draft 偏移：水平向右 1~3 格，垂直向下 1~2 格
HORIZONTAL_OFFSETS = [1, 2, 3]
VERTICAL_OFFSETS = [1, 2]


def set_seed(seed: int = SEED):
    """固定随机种子，保证可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)


def build_2d_positions(h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    为 h x w 的图像网格构造二维位置编码。
    返回形状 (h*w, d_model) 的位置嵌入。
    """
    pe = torch.zeros(h * w, D_MODEL, device=device)
    positions = torch.arange(h * w, device=device).float()
    div_term = torch.exp(
        torch.arange(0, D_MODEL, 2, device=device).float()
        * (-math.log(10000.0) / D_MODEL)
    )
    pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
    pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term)
    return pe


# ---------------------------- 模型定义 ----------------------------
class ImageTokenBackbone(nn.Module):
    """
    小型自回归 Transformer，输入为按光栅扫描展平的图像 token 序列。
    输出最后一层隐藏状态，用于 SSD Head 的蒸馏目标。
    """

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=D_MODEL * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYER)

    def forward(self, tokens: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, L)
        pos_emb: (L, D)
        返回 hidden: (B, L, D)
        """
        x = self.token_emb(tokens) + pos_emb.unsqueeze(0)  # (B, L, D)
        # 构造因果 mask，保证自回归性质
        L = tokens.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(tokens.device)
        hidden = self.transformer(x, mask=causal_mask, is_causal=True)
        return hidden


class SSDHead(nn.Module):
    """
    轻量 SSD 预测头。
    输入：当前位置 (y, x) 的隐藏状态 h 与 token 嵌入 e 的拼接。
    输出：每个空间偏移 δ 对应的预测隐藏状态 h_hat。
    """

    def __init__(self, offsets: List[int], direction: str):
        super().__init__()
        self.offsets = offsets
        self.direction = direction  # 'horizontal' 或 'vertical'
        # 轻量 MLP：RMSNorm + SwiGLU 风格
        self.norm = nn.RMSNorm(2 * D_MODEL)
        self.fc1 = nn.Linear(2 * D_MODEL, D_MODEL * 4)
        self.fc2 = nn.Linear(D_MODEL * 4, D_MODEL * 4)
        # 每个偏移一个独立的输出头
        self.heads = nn.ModuleList(
            [nn.Linear(D_MODEL * 4, D_MODEL) for _ in offsets]
        )

    def forward(
        self, h: torch.Tensor, e: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        h: (B, L, D)  当前位置隐藏状态
        e: (B, L, D)  当前位置 token 嵌入
        返回 {offset: (B, L, D)}
        """
        x = torch.cat([h, e], dim=-1)          # (B, L, 2D)
        x = self.norm(x)
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        x = F.silu(x)
        out = {}
        for off, head in zip(self.offsets, self.heads):
            out[off] = head(x)
        return out


class SSDModel(nn.Module):
    """
    将 Backbone 与水平和垂直 SSD Head 组合。
    """

    def __init__(self):
        super().__init__()
        self.backbone = ImageTokenBackbone()
        self.h_head = SSDHead(HORIZONTAL_OFFSETS, "horizontal")
        self.v_head = SSDHead(VERTICAL_OFFSETS, "vertical")

    def forward(
        self, tokens: torch.Tensor, pos_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        hidden = self.backbone(tokens, pos_emb)  # (B, L, D)
        token_emb = self.backbone.token_emb(tokens)  # (B, L, D)
        h_pred = self.h_head(hidden, token_emb)
        v_pred = self.v_head(hidden, token_emb)
        return hidden, h_pred, v_pred


# ---------------------------- 训练辅助函数 ----------------------------
def ssd_loss(
    hidden: torch.Tensor,
    h_pred: Dict[int, torch.Tensor],
    v_pred: Dict[int, torch.Tensor],
    coords: List[Tuple[int, int]],
    h: int,
    w: int,
) -> torch.Tensor:
    """
    计算 SSD 的 SmoothL1 蒸馏损失。

    hidden: (B, L, D) backbone 真实隐藏状态
    h_pred/v_pred: SSD Head 预测结果
    coords: 当前 batch 中每个样本的 (y, x) 坐标列表，长度 B
    """
    total_loss = 0.0
    count = 0
    B = hidden.size(0)

    for b in range(B):
        y, x = coords[b]
        idx = y * w + x
        h_true = hidden[b, idx]  # (D,)

        # 水平偏移：同一行，列 +offset
        for off, pred in h_pred.items():
            nx = x + off
            if nx < w:
                target_idx = y * w + nx
                target_h = hidden[b, target_idx]
                total_loss += F.smooth_l1_loss(pred[b, idx], target_h)
                count += 1

        # 垂直偏移：行 +off，同一列
        for off, pred in v_pred.items():
            ny = y + off
            if ny < h:
                target_idx = ny * w + x
                target_h = hidden[b, target_idx]
                total_loss += F.smooth_l1_loss(pred[b, idx], target_h)
                count += 1

    return total_loss / max(count, 1)


def make_batch(
    batch_size: int, h: int, w: int, device: torch.device
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    生成随机图像 token 序列，并随机选择一个 "当前位置" (y, x)。
    返回 tokens (B, h*w) 和 coords 列表。
    """
    tokens = torch.randint(0, VOCAB_SIZE, (batch_size, h * w), device=device)
    coords = [(random.randint(0, h - 1), random.randint(0, w - 1)) for _ in range(batch_size)]
    return tokens, coords


# ---------------------------- 推理：Draft + Verify ----------------------------
@torch.no_grad()
def spatial_speculative_decode_step(
    model: SSDModel,
    partial_tokens: torch.Tensor,
    pos_emb: torch.Tensor,
    h: int,
    w: int,
    n_rounds: int = 2,
) -> torch.Tensor:
    """
    演示一次空间投机解码步骤：
    1. 用 backbone 得到当前已生成 token 的隐藏状态；
    2. 用 SSD Head draft 出若干空间邻域 token 的隐藏状态；
    3. 将这些预测隐藏状态映射为 logits，采样得到 draft token；
    4. 用 backbone 做一次完整前向得到目标位置真实 logits；
    5. 比较并保留一致的 token（简化版 verification）。

    注意：这里为了演示，只接受第一个不匹配位置之前的 token。
    """
    model.eval()
    device = partial_tokens.device
    L = partial_tokens.size(1)

    # 1. backbone 前向（只取与 partial_tokens 等长的位置编码）
    hidden = model.backbone(partial_tokens, pos_emb[:L])  # (1, L, D)
    last_idx = L - 1
    h_state = hidden[:, last_idx:last_idx + 1]        # (1, 1, D)
    token_emb = model.backbone.token_emb(partial_tokens[:, last_idx:last_idx + 1])

    # 2. draft：先水平后垂直
    drafts: List[Tuple[int, torch.Tensor]] = []
    h_pred = model.h_head(h_state, token_emb)
    v_pred = model.v_head(h_state, token_emb)

    # 从最后一个已知 token 的位置推断
    # 简化：假设最后一个 token 位于 (y, x)
    y, x = divmod(last_idx, w)

    for off in sorted(h_pred.keys()):
        nx = x + off
        if nx < w:
            pred_h = h_pred[off][:, 0]  # (1, D)
            logits = pred_h @ model.backbone.token_emb.weight.t()  # (1, V)
            drafts.append((y * w + nx, logits))

    for off in sorted(v_pred.keys()):
        ny = y + off
        if ny < h:
            pred_h = v_pred[off][:, 0]
            logits = pred_h @ model.backbone.token_emb.weight.t()
            drafts.append((ny * w + x, logits))

    if not drafts:
        return partial_tokens

    # 3. verification：构造包含 draft token 的候选序列，用 backbone 验证
    # 为简化，我们只验证第一个 draft 位置
    target_idx, draft_logits = drafts[0]
    draft_token = draft_logits.argmax(dim=-1, keepdim=True)  # (1, 1)

    # 构造完整序列：在 target_idx 处填入 draft_token
    candidate = partial_tokens.clone()
    if target_idx >= candidate.size(1):
        candidate = torch.cat([candidate, draft_token], dim=1)
    else:
        candidate[:, target_idx] = draft_token.squeeze(1)

    # 用 backbone 在扩展后的序列上重新计算目标位置 logits
    cand_hidden = model.backbone(candidate, pos_emb[: candidate.size(1)])
    target_hidden = cand_hidden[:, target_idx]
    true_logits = target_hidden @ model.backbone.token_emb.weight.t()
    true_token = true_logits.argmax(dim=-1)

    # 接受或拒绝（简化：直接接受 true_token 作为校正结果）
    accepted = (draft_token.squeeze() == true_token).item()
    if target_idx >= partial_tokens.size(1):
        new_tokens = torch.cat([partial_tokens, true_token.unsqueeze(0)], dim=1)
    else:
        partial_tokens[:, target_idx] = true_token
        new_tokens = partial_tokens

    return new_tokens, accepted


# ---------------------------- 主流程 ----------------------------
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SSDModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    pos_emb = build_2d_positions(IMG_H, IMG_W, device)

    print("\n--- Stage 1: 训练 SSD Head（蒸馏 backbone 隐藏状态） ---")
    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        tokens, coords = make_batch(BATCH_SIZE, IMG_H, IMG_W, device)
        hidden, h_pred, v_pred = model(tokens, pos_emb)
        loss = ssd_loss(hidden, h_pred, v_pred, coords, IMG_H, IMG_W)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:03d}/{TRAIN_STEPS} | SSD distill loss: {loss.item():.4f}")

    print("\n--- Stage 2: 演示一次空间投机解码 ---")
    # 初始化部分序列：只给前 10 个 token，让模型 draft 后续 token
    partial_len = 10
    partial_tokens = torch.randint(0, VOCAB_SIZE, (1, partial_len), device=device)
    new_tokens, accepted = spatial_speculative_decode_step(
        model, partial_tokens, pos_emb, IMG_H, IMG_W, n_rounds=2
    )
    print(f"Partial length: {partial_len} -> New length: {new_tokens.size(1)}")
    print(f"First draft accepted: {accepted}")
    print("SSD demo finished.")


if __name__ == "__main__":
    main()
