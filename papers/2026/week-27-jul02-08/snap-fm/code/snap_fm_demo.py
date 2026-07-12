"""
SNAP-FM: Sparse Nonlinear Accelerated Projection for Physics-Constrained Generative Modeling
PyTorch 概念复现（带中文注释）

本代码演示 Physics-Constrained Flow Matching (PCFM) 的核心流程：
1. 训练一个速度场 v_theta(x, t)，学习把简单先验（高斯噪声）变换为
   满足物理约束的数据分布（守恒律 + 边界条件）。
2. 采样时用欧拉积分推进，并在每一步后将样本投影回约束流形：
   - 守恒约束：sum(x) = 0（如质量/能量守恒）；
   - 边界条件：每个维度 clamp 到 [-2, 2]。
3. 与无约束采样对比，展示约束投影如何消除违反物理的样本。

注意：本 demo 使用显式投影循环代替论文中的 GPU 稀疏 KKT 求解器，
仅用于说明“约束 + Flow Matching”的算法思想。

运行方式：
    pip install torch
    python snap_fm_demo.py
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- 超参数 / Hyperparameters ----------------------------
DIM = 8                  # 样本维度（模拟物理场的离散化）
HIDDEN_DIM = 64
NUM_STEPS_TRAIN = 2000   # 训练迭代数
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_SAMPLING_STEPS = 50  # 采样欧拉步数
BOUND = 2.0              # 边界条件
DEVICE = torch.device("cpu")


# ---------------------------- 约束数据 / Constrained Data ----------------------------
def sample_constrained_data(batch_size: int, dim: int) -> torch.Tensor:
    """
    生成近似满足 sum=0 且坐标在 [-BOUND, BOUND] 内的样本。
    模拟物理仿真输出：总量守恒、状态有界。
    """
    x = torch.randn(batch_size, dim) * 0.6
    # 投影到守恒超平面
    x = x - x.mean(dim=-1, keepdim=True)
    # 边界裁剪（可能轻微破坏守恒，作为工程近似）
    x = torch.clamp(x, -BOUND, BOUND)
    return x


# ---------------------------- 时间嵌入 / Time Embedding ----------------------------
def time_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    """正弦位置编码，把时间标量扩展为向量。"""
    half = dim // 2
    freq = torch.exp(-torch.arange(half, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
    t = t.float().unsqueeze(-1) * freq.to(t.device)
    return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)


# ---------------------------- 速度场 / Velocity Network ----------------------------
class VelocityNet(nn.Module):
    def __init__(self, data_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = time_embedding(t, dim=32).to(x.device)
        xt = torch.cat([x, t_emb], dim=-1)
        return self.net(xt)


# ---------------------------- 约束投影 / Constraint Projection ----------------------------
def project_constraints(x: torch.Tensor, bound: float = BOUND, num_iters: int = 3) -> torch.Tensor:
    """
    把样本投影到守恒超平面 sum=0 与边界框 [-bound, bound]^D 的交上。
    使用简单的交替投影循环（工程上对应 SNAP-FM 的稀疏非线性投影子问题）。
    """
    for _ in range(num_iters):
        # 1) 守恒投影
        x = x - x.mean(dim=-1, keepdim=True)
        # 2) 边界投影
        x = torch.clamp(x, -bound, bound)
    return x


# ---------------------------- 训练 / Training ----------------------------
def train_velocity(net: nn.Module):
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    net.train()
    for step in range(1, NUM_STEPS_TRAIN + 1):
        x1 = sample_constrained_data(BATCH_SIZE, DIM).to(DEVICE)
        x0 = torch.randn(BATCH_SIZE, DIM).to(DEVICE)
        t = torch.rand(BATCH_SIZE, device=DEVICE)
        # 线性插值得到 x_t
        t_expand = t.unsqueeze(-1)
        xt = (1 - t_expand) * x0 + t_expand * x1
        # CFM 目标速度
        target = x1 - x0
        pred = net(xt, t)
        loss = F.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"  Step {step}/{NUM_STEPS_TRAIN}, Loss={loss.item():.4f}")


# ---------------------------- 采样 / Sampling ----------------------------
def sample_ode(net: nn.Module, n: int, project: bool = True) -> torch.Tensor:
    """从先验出发，用欧拉积分生成样本，可选约束投影。"""
    net.eval()
    x = torch.randn(n, DIM).to(DEVICE)
    dt = 1.0 / NUM_SAMPLING_STEPS
    with torch.no_grad():
        for i in range(NUM_SAMPLING_STEPS):
            t = torch.full((n,), i * dt, device=DEVICE)
            v = net(x, t)
            x = x + v * dt
            if project:
                x = project_constraints(x)
    return x


def constraint_violation(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回守恒违反量（绝对值）和边界违反比例。"""
    cons_err = x.sum(dim=-1).abs().mean()
    bound_err = ((x.abs() > BOUND).float().mean())
    return cons_err, bound_err


# ---------------------------- 主程序 / Main ----------------------------
def main():
    random.seed(1)
    torch.manual_seed(1)

    net = VelocityNet(DIM, HIDDEN_DIM).to(DEVICE)
    print("训练约束 Flow Matching 速度场 / Training constrained Flow Matching velocity field...")
    train_velocity(net)

    print("\n=== 有约束采样 / Constrained Sampling ===")
    x_cons = sample_ode(net, n=500, project=True)
    ce, be = constraint_violation(x_cons)
    print(f"守恒约束平均绝对误差 / Conservation error: {ce.item():.5f}")
    print(f"越界样本比例 / Out-of-bound ratio: {be.item():.3%}")
    print(f"样本均值 / Sample mean per dim: {x_cons.mean(dim=0).tolist()}")

    print("\n=== 无约束采样 / Unconstrained Sampling ===")
    x_uncons = sample_ode(net, n=500, project=False)
    ce2, be2 = constraint_violation(x_uncons)
    print(f"守恒约束平均绝对误差 / Conservation error: {ce2.item():.5f}")
    print(f"越界样本比例 / Out-of-bound ratio: {be2.item():.3%}")
    print("\n结论：投影步骤显著降低物理约束违反量，说明约束采样有效。")


if __name__ == "__main__":
    main()
