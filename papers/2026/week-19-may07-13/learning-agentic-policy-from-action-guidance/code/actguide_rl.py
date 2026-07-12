"""
ActGuide-RL 极简复现 (Minimal reproduction)
基于论文: Learning Agentic Policy from Action Guidance (arXiv:2605.12004)

本脚本用一个 5x5 网格世界演示"动作指导(action guidance)"如何被蒸馏进策略梯度目标，
从而在稀疏奖励下加速收敛。这里用合成的方向性指导代替真实 LLM 提供的动作先验。

运行方式:
    python actguide_rl.py

依赖:
    torch, numpy
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# --------------------------- 超参数 ---------------------------
GRID_SIZE = 5                 # 网格世界的边长
N_ACTIONS = 4                 # 动作数: 0=上, 1=右, 2=下, 3=左
HIDDEN_DIM = 64               # 策略网络隐藏层维度
LR = 3e-3                     # 学习率
GAMMA = 0.95                  # 折扣因子
LAMBDA_GUIDE = 0.5            # 动作指导损失的权重
ENTROPY_COEF = 0.05           # 熵正则系数
VALUE_COEF = 0.5              # 价值函数损失系数
EPISODES_PER_EPOCH = 20       # 每个 epoch 采样的轨迹数
N_EPOCHS = 200                # 训练 epoch 数
MAX_STEPS = 50                # 每条轨迹最大步数
SEED = 42                     # 随机种子

# 固定随机种子以保证结果可复现
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 动作对应的 (dx, dy)
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# --------------------------- 环境定义 ---------------------------
class GridWorld:
    """
    简单的网格世界环境。
    智能体从左上角 (0,0) 出发，目标是右下角 (GRID_SIZE-1, GRID_SIZE-1)。
    地图中有若干障碍物，撞到障碍物或边界会受惩罚。
    """

    def __init__(self, size: int = GRID_SIZE):
        self.size = size
        self.goal = (size - 1, size - 1)
        # 预定义障碍物坐标
        self.obstacles = {(1, 1), (2, 2), (1, 3), (3, 1)}
        self.pos = (0, 0)

    def reset(self):
        """重置智能体位置并返回初始状态。"""
        self.pos = (0, 0)
        return self._get_state()

    def step(self, action: int):
        """
        执行一个动作并返回 (next_state, reward, done)。
        action: 0=上, 1=右, 2=下, 3=左
        """
        dx, dy = ACTIONS[action]
        x, y = self.pos
        nx, ny = x + dx, y + dy

        # 检查边界与障碍物
        if not (0 <= nx < self.size and 0 <= ny < self.size):
            # 撞墙：位置不变，给予惩罚
            reward = -0.2
            done = False
            return self._get_state(), reward, done

        if (nx, ny) in self.obstacles:
            # 撞到障碍物：位置不变，给予惩罚
            reward = -0.2
            done = False
            return self._get_state(), reward, done

        # 正常移动
        self.pos = (nx, ny)
        reward = -0.01  # 每步小额惩罚，鼓励更快到达目标

        # 到达目标获得正奖励并结束
        if self.pos == self.goal:
            reward = 1.0
            done = True
        else:
            done = False

        return self._get_state(), reward, done

    def _get_state(self):
        """
        将当前位置编码为 one-hot 向量，形状 (size*size,)。
        """
        vec = np.zeros(self.size * self.size, dtype=np.float32)
        x, y = self.pos
        vec[x * self.size + y] = 1.0
        return vec


# --------------------------- 策略网络 ---------------------------
class PolicyNet(nn.Module):
    """
    同时输出动作 logits 与状态价值的简单全连接网络。
    在真实 LLM-agent 场景中，这里可以是一个带 LoRA 适配器的大语言模型；
    本示例使用小型 MLP 以便 CPU 上快速运行。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, output_dim)   # 动作 logits
        self.critic = nn.Linear(hidden_dim, 1)           # 状态价值

    def forward(self, x: torch.Tensor):
        """返回动作 logits 与状态价值。"""
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


# --------------------------- 动作指导生成器 ---------------------------
def get_action_guidance(state_vec: np.ndarray, goal: tuple, eps: float = 0.15):
    """
    根据当前位置与目标位置生成一个软动作指导分布。
    指导会偏向朝向目标的方向，同时保留 eps 的均匀噪声以模拟不完美的外部建议。

    返回：numpy 数组，形状 (N_ACTIONS,)，元素和为 1。
    """
    size = GRID_SIZE
    idx = np.argmax(state_vec)
    x, y = divmod(idx, size)
    gx, gy = goal

    # 计算指向目标的单位方向向量
    dx = gx - x
    dy = gy - y
    if dx == 0 and dy == 0:
        return np.ones(N_ACTIONS) / N_ACTIONS

    # 每个动作与目标方向的点积
    scores = np.array([a[0] * dx + a[1] * dy for a in ACTIONS], dtype=np.float32)
    # 点积为负的动作给予较小概率
    scores = np.maximum(scores, 0.0)

    if scores.sum() == 0:
        guide = np.ones(N_ACTIONS) / N_ACTIONS
    else:
        guide = scores / scores.sum()

    # 加入均匀噪声，模拟有噪声的外部指导
    guide = (1 - eps) * guide + eps * (np.ones(N_ACTIONS) / N_ACTIONS)
    guide = guide / guide.sum()
    return guide


# --------------------------- 训练辅助函数 ---------------------------
def collect_episodes(env: GridWorld, policy: PolicyNet, n_episodes: int, max_steps: int):
    """
    收集 n_episodes 条轨迹，返回每个 epoch 的原始数据。
    每条轨迹包含：状态、动作、奖励、动作指导分布、logits、价值估计。
    """
    trajectories = []
    for _ in range(n_episodes):
        state = env.reset()
        states, actions, rewards, guides, log_probs, values = [], [], [], [], [], []
        for _ in range(max_steps):
            s_t = torch.from_numpy(state).unsqueeze(0)  # (1, input_dim)
            logits, value = policy(s_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # 生成当前状态的动作指导
            guide = get_action_guidance(state, env.goal)

            next_state, reward, done = env.step(action.item())

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            guides.append(guide)
            log_probs.append(log_prob)
            values.append(value.squeeze(0))

            state = next_state
            if done:
                break
        trajectories.append({
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "guides": guides,
            "log_probs": log_probs,
            "values": values,
        })
    return trajectories


def compute_returns(rewards, gamma=GAMMA):
    """
    计算折扣回报 (Monte-Carlo returns)。
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def update_policy(policy: PolicyNet, optimizer: optim.Optimizer, trajectories):
    """
    执行一次策略更新，包含：
      1. 带基线的策略梯度损失；
      2. 价值函数损失；
      3. 动作指导 KL 散度损失；
      4. 熵正则损失。
    """
    policy_loss_total = 0.0
    value_loss_total = 0.0
    guide_loss_total = 0.0
    entropy_total = 0.0
    n_samples = 0

    for traj in trajectories:
        states = np.stack(traj["states"])
        actions = torch.tensor(traj["actions"], dtype=torch.long)
        returns = torch.tensor(compute_returns(traj["rewards"]), dtype=torch.float32)
        guides = torch.tensor(np.stack(traj["guides"]), dtype=torch.float32)

        # 前向传播
        s_t = torch.from_numpy(states)
        logits, values = policy(s_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # 优势 = 回报 - 价值估计
        advantages = returns - values.detach()

        # 1. 策略梯度损失（带优势函数）
        policy_loss = -(log_probs * advantages).mean()

        # 2. 价值函数损失
        value_loss = F.mse_loss(values, returns)

        # 3. 动作指导损失：将策略与外部指导分布拉近
        #    使用 KL(policy || guide)，对应论文中把指导作为软先验的思想
        log_policy = F.log_softmax(logits, dim=-1)
        guide_loss = F.kl_div(log_policy, guides, reduction="batchmean")

        # 4. 熵正则，鼓励探索
        entropy_loss = -entropy.mean()

        policy_loss_total += policy_loss.item()
        value_loss_total += value_loss.item()
        guide_loss_total += guide_loss.item()
        entropy_total += entropy.mean().item()
        n_samples += 1

        loss = (
            policy_loss
            + VALUE_COEF * value_loss
            + LAMBDA_GUIDE * guide_loss
            + ENTROPY_COEF * entropy_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        "policy_loss": policy_loss_total / n_samples,
        "value_loss": value_loss_total / n_samples,
        "guide_loss": guide_loss_total / n_samples,
        "entropy": entropy_total / n_samples,
    }


# --------------------------- 主训练循环 ---------------------------
def main():
    env = GridWorld(GRID_SIZE)
    input_dim = GRID_SIZE * GRID_SIZE
    policy = PolicyNet(input_dim, HIDDEN_DIM, N_ACTIONS)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # 滑动平均回报，用于观察训练趋势
    reward_window = deque(maxlen=20)

    print("开始训练 ActGuide-RL ...")
    print(f"网格大小: {GRID_SIZE}x{GRID_SIZE}, 指导权重 lambda_guide={LAMBDA_GUIDE}")

    for epoch in range(1, N_EPOCHS + 1):
        trajectories = collect_episodes(env, policy, EPISODES_PER_EPOCH, MAX_STEPS)
        metrics = update_policy(policy, optimizer, trajectories)

        # 统计本 epoch 的平均回报与成功率
        ep_returns = [sum(t["rewards"]) for t in trajectories]
        success = sum(1 for t in trajectories if t["rewards"][-1] > 0.5)
        for r in ep_returns:
            reward_window.append(r)
        avg_return = np.mean(list(reward_window))

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | 平均回报(平滑): {avg_return:6.3f} | "
                f"成功率: {success}/{EPISODES_PER_EPOCH} | "
                f"PG loss: {metrics['policy_loss']:.3f} | "
                f"Guide loss: {metrics['guide_loss']:.3f} | "
                f"Entropy: {metrics['entropy']:.3f}"
            )

    # 测试最终策略（按照学习到的策略分布采样动作，比贪心 argmax 更稳定）
    print("\n最终策略测试:")
    test_returns = []
    test_success = 0
    for _ in range(10):
        state = env.reset()
        total_reward = 0.0
        reached = False
        for _ in range(MAX_STEPS):
            with torch.no_grad():
                s_t = torch.from_numpy(state).unsqueeze(0)
                logits, _ = policy(s_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                reached = True
                break
        test_returns.append(total_reward)
        if reached:
            test_success += 1
    print(
        f"测试平均回报: {np.mean(test_returns):.3f}, "
        f"最高: {np.max(test_returns):.3f}, "
        f"成功到达目标: {test_success}/10"
    )


if __name__ == "__main__":
    main()
