"""
Lift3D-VLA: Lifting VLA Models to 3D Geometry and Dynamics-Aware Manipulation
PyTorch 概念复现（带中文注释）

本代码演示 Lift3D-VLA 的三个核心机制：
1. 2D→3D 提升：将 3D 点云坐标投影到 2D 图像网格，使用预训练的 2D 位置嵌入
   初始化点云 token，减少几何信息损失。
2. 几何中心掩码自编码（GC-MAE）：随机掩码部分点云 token，编码器重建被掩码
   点的当前 3D 坐标，并预测其下一时刻位置，从而内隐 3D 结构与物理动态。
3. 分层时序动作建模（Layer-wise Temporal Action Modeling）：把动作块视作
   一组可学习的动作 token，依次经过多层 Transformer 逐步精化，输出时序连贯
   的机械臂动作轨迹。

运行方式：
    pip install torch
    python lift3d_vla_demo.py
"""

import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- 超参数 / Hyperparameters ----------------------------
NUM_POINTS = 64           # 每帧点云数量
IMG_SIZE = 16             # 模拟 2D 图像网格大小（16x16）
EMBED_DIM = 64            # token 嵌入维度
NUM_ENCODER_LAYERS = 2    # 点云编码器层数
NUM_ACTION_LAYERS = 4     # 分层动作建模层数
ACTION_DIM = 7            # 每个动作步的维度（如 3 位置 + 3 旋转 + 1 夹爪）
ACTION_CHUNK = 8          # 一次预测的动作步数
MASK_RATIO = 0.4          # GC-MAE 掩码比例
DYNAMICS_HORIZON = 1      # 未来几何预测步长
LEARNING_RATE = 2e-3
NUM_STEPS = 120
DEVICE = torch.device("cpu")


# ---------------------------- 2D→3D 提升模块 / 2D-to-3D Lifting ----------------------------
class Lifting3D(nn.Module):
    """
    将 3D 点云通过投影与 2D 位置嵌入几何对齐。
    为简单起见，这里把 3D 点正交投到 16x16 网格，取出对应 2D 位置嵌入，
    再与 3D 坐标拼接后投影成点云 token。
    """
    def __init__(self, img_size: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.pos_embed_2d = nn.Parameter(torch.randn(img_size * img_size, embed_dim // 2) * 0.02)
        # 输入 = 3D 坐标(3) + 2D 位置嵌入(embed_dim/2)
        self.proj = nn.Linear(3 + embed_dim // 2, embed_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) 点云坐标，已归一化到 [-1, 1]
        Returns:
            tokens: (B, N, embed_dim) 点云 token
        """
        B, N, _ = points.shape
        # 正交投影到 2D 网格索引
        grid_xy = ((points[:, :, :2] + 1.0) / 2.0 * (self.img_size - 1)).long().clamp(0, self.img_size - 1)
        grid_idx = grid_xy[:, :, 0] * self.img_size + grid_xy[:, :, 1]  # (B, N)
        pos_2d = self.pos_embed_2d[grid_idx]  # (B, N, embed_dim/2)
        lifted = torch.cat([points, pos_2d], dim=-1)
        return self.proj(lifted)


# ---------------------------- GC-MAE 编码器 / Geometry-Centric MAE ----------------------------
class PointMAE(nn.Module):
    """
    对点云 token 做掩码自编码：重建当前 3D 坐标 + 预测未来坐标。
    """
    def __init__(self, embed_dim: int, num_layers: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.recon_head = nn.Linear(embed_dim, 3)
        self.future_head = nn.Linear(embed_dim, 3)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, N, embed_dim)
            mask: (B, N) bool, True 表示被掩码的 token
        Returns:
            recon_points: (B, N, 3)
            future_points: (B, N, 3)
        """
        # 用可学习的掩码 token 替换被掩码位置
        masked_tokens = tokens.clone()
        mask_token = nn.Parameter(torch.zeros(tokens.size(-1)))
        masked_tokens[mask] = mask_token.to(tokens.device)
        h = self.encoder(masked_tokens)
        return self.recon_head(h), self.future_head(h)


# ---------------------------- 动作模型 / Layer-wise Action Model ----------------------------
class LayerwiseActionModel(nn.Module):
    """
    将动作块视为动作 token，逐层精化，实现时间一致性动作生成。
    """
    def __init__(self, embed_dim: int, action_dim: int, action_chunk: int, num_layers: int):
        super().__init__()
        self.action_chunk = action_chunk
        self.action_embed = nn.Parameter(torch.randn(action_chunk, embed_dim) * 0.02)
        layers = []
        for _ in range(num_layers):
            layers.append(nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 4,
                batch_first=True,
            ))
        self.layers = nn.ModuleList(layers)
        self.instruction_proj = nn.Linear(embed_dim, embed_dim)
        self.head = nn.Linear(embed_dim, action_dim)

    def forward(self, instruction: torch.Tensor) -> torch.Tensor:
        """
        Args:
            instruction: (B, embed_dim) 语言指令嵌入
        Returns:
            actions: (B, action_chunk, action_dim)
        """
        B = instruction.size(0)
        x = self.action_embed.unsqueeze(0).expand(B, -1, -1)  # (B, T, D)
        instr = self.instruction_proj(instruction).unsqueeze(1)  # (B, 1, D)
        for layer in self.layers:
            x = layer(x + instr)
        return torch.tanh(self.head(x))


# ---------------------------- 完整 VLA 模型 / Full VLA Model ----------------------------
class Lift3DVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.lifting = Lifting3D(IMG_SIZE, EMBED_DIM)
        self.point_mae = PointMAE(EMBED_DIM, NUM_ENCODER_LAYERS)
        self.instruction_encoder = nn.Embedding(20, EMBED_DIM)  # 模拟 20 条指令
        self.action_model = LayerwiseActionModel(EMBED_DIM, ACTION_DIM, ACTION_CHUNK, NUM_ACTION_LAYERS)

    def forward(
        self,
        points: torch.Tensor,
        points_future: torch.Tensor,
        instruction_id: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回重建点云、未来点云预测、动作块。"""
        tokens = self.lifting(points)
        recon, future_pred = self.point_mae(tokens, mask)
        instr_emb = self.instruction_encoder(instruction_id)
        actions = self.action_model(instr_emb)
        return recon, future_pred, actions


# ---------------------------- 工具函数 / Utilities ----------------------------
def make_batch(batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造模拟数据：当前点云、未来点云、指令 id、掩码。"""
    points = torch.randn(batch_size, NUM_POINTS, 3) * 0.3  # 当前帧
    # 未来帧：当前帧 + 小位移，模拟动态
    points_future = points + torch.randn_like(points) * 0.05
    instruction_id = torch.randint(0, 20, (batch_size,))
    # 随机掩码
    mask = torch.rand(batch_size, NUM_POINTS) < MASK_RATIO
    return points, points_future, instruction_id, mask


def total_loss(recon, future_pred, actions, points, points_future, actions_target):
    """GC-MAE 重建损失 + 未来预测损失 + 动作回归损失。"""
    loss_recon = F.mse_loss(recon, points)
    loss_dyn = F.mse_loss(future_pred, points_future)
    loss_act = F.mse_loss(actions, actions_target)
    return loss_recon + loss_dyn + loss_act


# ---------------------------- 主程序 / Main ----------------------------
def main():
    random.seed(0)
    torch.manual_seed(0)

    model = Lift3DVLA().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("开始训练 Lift3D-VLA 概念模型 / Training Lift3D-VLA toy model...")
    for step in range(1, NUM_STEPS + 1):
        points, points_future, instr_id, mask = make_batch(batch_size=8)
        points = points.to(DEVICE)
        points_future = points_future.to(DEVICE)
        instr_id = instr_id.to(DEVICE)
        mask = mask.to(DEVICE)
        # 模拟目标动作：随指令变化的简单正弦轨迹
        t = torch.linspace(0, 1, ACTION_CHUNK).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        actions_target = torch.sin(2 * torch.pi * (instr_id.unsqueeze(1).unsqueeze(2).float() / 20.0 + t)) \
            .expand(-1, -1, ACTION_DIM)

        optimizer.zero_grad()
        recon, future_pred, actions = model(points, points_future, instr_id, mask)
        loss = total_loss(recon, future_pred, actions, points, points_future, actions_target)
        loss.backward()
        optimizer.step()

        if step % 30 == 0:
            print(f"  Step {step}/{NUM_STEPS}, Loss={loss.item():.4f}")

    # 推理示例
    model.eval()
    with torch.no_grad():
        points, points_future, instr_id, mask = make_batch(batch_size=1)
        recon, future_pred, actions = model(
            points.to(DEVICE), points_future.to(DEVICE), instr_id.to(DEVICE), mask.to(DEVICE)
        )
        print("\n=== Lift3D-VLA 推理示例 / Inference Example ===")
        print(f"指令 ID / Instruction ID: {instr_id.item()}")
        print(f"点云重建 MSE / Point recon MSE: {F.mse_loss(recon, points).item():.4f}")
        print(f"未来几何预测 MSE / Future geometry MSE: {F.mse_loss(future_pred, points_future).item():.4f}")
        print(f"预测动作块形状 / Predicted action chunk shape: {actions.shape}")
        print(f"前 3 步动作 / First 3 action steps:\n{actions[0, :3].numpy()}")


if __name__ == "__main__":
    main()
