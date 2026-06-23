"""
Success Visitation Matching (SVM) 过程奖励 + PPO 复现
Reproduction of "Learning Process Rewards via Success Visitation Matching for Efficient RL"

作者: Raymond Tsao, Andrew Wagenmaker, Sergey Levine (arXiv:2606.23640)
本代码为教学简化版，使用 MountainCar-v0 的稀疏奖励变体：
- 仅在到达目标位置时给出 +1 的终端奖励，其余所有步骤奖励为 0
- 目标：验证 SVM 奖励能将稀疏结果奖励转化为密集过程奖励，并加速学习

运行方式:
    pip install torch numpy gymnasium matplotlib
    python svm_reward_ppo.py
"""

import argparse
import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

# 在 Windows 终端中强制使用 UTF-8 输出，避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# -----------------------------------------------------------------------------
# 超参数 / Hyperparameters
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # 默认使用 CartPole-v1 的稀疏奖励版本：只有 episode 成功结束时才获得 +1 奖励
    # 用户可切换为 MountainCar-v0 以测试更难的连续控制稀疏奖励任务
    env_name: str = "CartPole-v1"
    seed: int = 42
    max_episode_steps: int = 500  # CartPole-v1 默认 500 步
    total_timesteps: int = 50_000

    # 策略网络 / Policy network
    gamma: float = 0.99           # 折扣因子
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    hidden_dim: int = 128

    # PPO
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    batch_size: int = 64

    # SVM 判别器 / SVM discriminator
    use_svm: bool = True
    svm_update_freq: int = 2_000  # 每 2k 步更新一次判别器
    svm_buffer_size: int = 500    # 保存最近 500 条轨迹
    svm_hidden_dim: int = 128
    svm_lr: float = 3e-4
    svm_epochs: int = 10
    svm_batch_size: int = 256
    svm_reward_scale: float = 5.0
    svm_success_threshold: float = -110.0  # 保留配置；本次复现以环境 terminated 作为成功判断

    device: str = "cpu"


# -----------------------------------------------------------------------------
# 工具函数 / Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    """设置随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(cfg: Config):
    """创建稀疏奖励版本的 MountainCar 环境。"""
    env = gym.make(cfg.env_name)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
    return env


# -----------------------------------------------------------------------------
# 策略与价值网络 / Policy and Value Networks
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    """通用多层感知机。"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, output_activation=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_activation = output_activation

    def forward(self, x):
        x = self.net(x)
        return self.output_activation(x) if self.output_activation else x


class ActorCritic(nn.Module):
    """Actor-Critic 架构：共享特征提取层，分别输出策略 logits 和状态价值。"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        feat = self.feature(state)
        logits = self.actor(feat)
        value = self.critic(feat)
        return logits, value

    def get_action_and_value(self, state):
        """采样动作并返回 log_prob、价值。"""
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)


# -----------------------------------------------------------------------------
# SVM 判别器 / Success Visitation Matching Discriminator
# -----------------------------------------------------------------------------
class SVMDiscriminator(nn.Module):
    """
    判别器 D(s, a)：预测当前 (s, a) 来自成功轨迹还是失败轨迹。
    输出概率 p(success | s, a)。
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        # 动作离散，使用 one-hot 编码；输入维度 = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)


def train_svm_discriminator(
    disc: SVMDiscriminator,
    optimizer: optim.Optimizer,
    success_buffer: List[List[Tuple[np.ndarray, int]]],
    failure_buffer: List[List[Tuple[np.ndarray, int]]],
    cfg: Config,
):
    """
    使用成功/失败轨迹缓冲区训练 SVM 判别器。
    正样本：成功轨迹中的所有 (s, a) 对
    负样本：失败轨迹中的所有 (s, a) 对
    """
    if len(success_buffer) < 2 or len(failure_buffer) < 2:
        return 0.0

    # 构建训练数据集
    states, action_indices, labels = [], [], []
    for ep in success_buffer:
        for s, a in ep:
            states.append(s)
            action_indices.append(a)
            labels.append(1.0)
    for ep in failure_buffer:
        for s, a in ep:
            states.append(s)
            action_indices.append(a)
            labels.append(0.0)

    states = torch.tensor(np.array(states), dtype=torch.float32, device=cfg.device)
    # 动作 one-hot 编码
    action_dim = cfg.env_action_dim
    actions = torch.zeros(len(action_indices), action_dim, device=cfg.device)
    for i, a in enumerate(action_indices):
        actions[i, a] = 1.0
    labels = torch.tensor(labels, dtype=torch.float32, device=cfg.device)

    # 计算类别权重，处理成功样本通常远少于失败样本的不平衡问题
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    pos_weight = (n_neg / max(n_pos, 1.0)) * 0.5  # 缩放避免正样本权重过大

    dataset = torch.utils.data.TensorDataset(states, actions, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.svm_batch_size, shuffle=True)

    disc.train()
    total_loss = 0.0
    for _ in range(cfg.svm_epochs):
        epoch_loss = 0.0
        for s_batch, a_batch, y_batch in loader:
            pred = disc(s_batch, a_batch)
            # BCE with positive weight: 正样本加权
            loss = nn.functional.binary_cross_entropy(pred, y_batch, weight=y_batch * pos_weight + (1 - y_batch) * 1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(s_batch)
        total_loss += epoch_loss / len(dataset)

    return total_loss / cfg.svm_epochs


def compute_svm_rewards(
    disc: SVMDiscriminator,
    states: torch.Tensor,
    actions: torch.Tensor,
    cfg: Config,
):
    """
    根据判别器输出计算 SVM 过程奖励。
    r_svm(s, a) = logit(p_success) * scale
                = log(p / (1 - p)) * scale
    使用 logit 而不是原始概率，奖励范围从 -inf 到 +inf，更适合 RL。
    """
    with torch.no_grad():
        disc.eval()
        p_success = disc(states, actions).clamp(1e-6, 1 - 1e-6)
        # logit 转换
        reward = torch.log(p_success / (1 - p_success)) * cfg.svm_reward_scale
    return reward


# -----------------------------------------------------------------------------
# PPO 收集与训练 / PPO Collection and Training
# -----------------------------------------------------------------------------
@dataclass
class RolloutBuffer:
    """用于存储 PPO 一条 rollout 的数据。"""
    states: List[np.ndarray]
    actions: List[int]
    log_probs: List[float]
    rewards: List[float]
    values: List[float]
    dones: List[bool]


def collect_rollout(env, ac: ActorCritic, disc: SVMDiscriminator, cfg: Config):
    """收集一条 rollout，并在启用 SVM 时添加过程奖励。"""
    state, _ = env.reset(seed=cfg.seed)
    buffer = RolloutBuffer([], [], [], [], [], [])
    ep_reward = 0.0
    ep_len = 0
    done = False

    while not done and ep_len < cfg.max_episode_steps:
        state_t = torch.tensor(state, dtype=torch.float32, device=cfg.device).unsqueeze(0)
        action, log_prob, _, value = ac.get_action_and_value(state_t)
        action = action.item()

        next_state, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_len += 1

        buffer.states.append(state)
        buffer.actions.append(action)
        buffer.log_probs.append(log_prob.item())
        # 使用稀疏结果奖励：只有最终成功时才给 +1，中间步骤奖励为 0
        # 这样模拟论文中描述的二元稀疏奖励场景
        buffer.rewards.append(0.0)
        buffer.values.append(value.item())
        buffer.dones.append(done)

        state = next_state

    # 判断 episode 是否成功：
    # - CartPole-v1：未 terminated（杆子未倒）即视为成功
    # - MountainCar-v0：terminated 表示到达目标，视为成功
    success = (cfg.env_name == "CartPole-v1" and not terminated) or \
              (cfg.env_name == "MountainCar-v0" and terminated)
    # 若成功，将最后一个时刻的奖励改为 +1
    if success and buffer.rewards:
        buffer.rewards[-1] = 1.0

    # 如果启用 SVM，叠加过程奖励
    if cfg.use_svm and disc is not None:
        states_t = torch.tensor(np.array(buffer.states), dtype=torch.float32, device=cfg.device)
        action_dim = cfg.env_action_dim
        actions_oh = torch.zeros(len(buffer.actions), action_dim, device=cfg.device)
        for i, a in enumerate(buffer.actions):
            actions_oh[i, a] = 1.0
        svm_r = compute_svm_rewards(disc, states_t, actions_oh, cfg)
        svm_r = svm_r.cpu().numpy()
        # 叠加：总奖励 = 原始奖励 + SVM 过程奖励
        buffer.rewards = [r + sr for r, sr in zip(buffer.rewards, svm_r)]

    # 稀疏奖励下的 episode 回报
    ep_reward = 1.0 if success else 0.0
    return buffer, ep_reward, ep_len, success, [
        (s, a) for s, a in zip(buffer.states, buffer.actions)
    ]


def compute_gae(rewards, values, dones, gamma, lam=0.95):
    """计算 Generalized Advantage Estimation (GAE)。"""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_advantage = 0.0
    # 这里简化为单条 rollout，没有 bootstrap 的下一个价值
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - float(dones[t])
            next_v = next_value
        else:
            next_non_terminal = 1.0 - float(dones[t])
            next_v = values[t + 1]
        delta = rewards[t] + gamma * next_v * next_non_terminal - values[t]
        advantages[t] = delta + gamma * lam * next_non_terminal * last_advantage
        last_advantage = advantages[t]
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def train_ppo(ac: ActorCritic, optimizer: optim.Optimizer, buffers: List[RolloutBuffer], cfg: Config):
    """使用收集到的 rollout 训练 PPO。"""
    # 拼接数据
    all_states, all_actions = [], []
    all_log_probs, all_advantages, all_returns = [], [], []
    for buf in buffers:
        adv, ret = compute_gae(buf.rewards, buf.values, buf.dones, cfg.gamma)
        all_states.extend(buf.states)
        all_actions.extend(buf.actions)
        all_log_probs.extend(buf.log_probs)
        all_advantages.extend(adv.tolist())
        all_returns.extend(ret.tolist())

    states_t = torch.tensor(np.array(all_states), dtype=torch.float32, device=cfg.device)
    actions_t = torch.tensor(all_actions, dtype=torch.long, device=cfg.device)
    old_log_probs_t = torch.tensor(all_log_probs, dtype=torch.float32, device=cfg.device)
    advantages_t = torch.tensor(all_advantages, dtype=torch.float32, device=cfg.device)
    returns_t = torch.tensor(all_returns, dtype=torch.float32, device=cfg.device)

    # 标准化优势
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    dataset = torch.utils.data.TensorDataset(
        states_t, actions_t, old_log_probs_t, advantages_t, returns_t
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    for _ in range(cfg.ppo_epochs):
        for s, a, old_lp, adv, ret in loader:
            logits, value = ac(s)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(a)
            entropy = dist.entropy()

            # PPO clipped surrogate loss
            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(value.squeeze(-1), ret)

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

    n_batches = max(1, len(loader) * cfg.ppo_epochs)
    return {
        "policy_loss": total_policy_loss / n_batches,
        "value_loss": total_value_loss / n_batches,
        "entropy": total_entropy / n_batches,
    }


# -----------------------------------------------------------------------------
# 主训练循环 / Main Training Loop
# -----------------------------------------------------------------------------
def main(cfg: Config):
    set_seed(cfg.seed)
    env = make_env(cfg)
    cfg.env_action_dim = int(env.action_space.n)
    cfg.env_state_dim = int(env.observation_space.shape[0])

    ac = ActorCritic(
        state_dim=cfg.env_state_dim,
        action_dim=cfg.env_action_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(cfg.device)

    ac_optimizer = optim.Adam([
        {"params": ac.actor.parameters(), "lr": cfg.lr_policy},
        {"params": list(ac.feature.parameters()) + list(ac.critic.parameters()), "lr": cfg.lr_value},
    ])

    disc = None
    disc_optimizer = None
    if cfg.use_svm:
        disc = SVMDiscriminator(
            state_dim=cfg.env_state_dim,
            action_dim=cfg.env_action_dim,
            hidden_dim=cfg.svm_hidden_dim,
        ).to(cfg.device)
        disc_optimizer = optim.Adam(disc.parameters(), lr=cfg.svm_lr)

    success_buffer = deque(maxlen=cfg.svm_buffer_size)
    failure_buffer = deque(maxlen=cfg.svm_buffer_size)

    timestep = 0
    episode = 0
    best_reward = -float("inf")
    recent_rewards = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)

    print("=" * 70)
    print("开始训练 / Start training")
    print(f"SVM 启用 / SVM enabled: {cfg.use_svm}")
    print("=" * 70)

    while timestep < cfg.total_timesteps:
        buffers = []
        batch_ep_rewards = []
        batch_ep_lengths = []

        # 收集一个 mini-batch 的 rollout
        for _ in range(8):
            buf, ep_reward, ep_len, success, ep_sa = collect_rollout(env, ac, disc, cfg)
            buffers.append(buf)
            batch_ep_rewards.append(ep_reward)
            batch_ep_lengths.append(ep_len)
            timestep += ep_len
            episode += 1

            if cfg.use_svm:
                if success:
                    success_buffer.append(ep_sa)
                else:
                    failure_buffer.append(ep_sa)

        recent_rewards.extend(batch_ep_rewards)
        recent_lengths.extend(batch_ep_lengths)
        best_reward = max(best_reward, max(batch_ep_rewards))

        # 训练 PPO
        train_info = train_ppo(ac, ac_optimizer, buffers, cfg)

        # 定期更新 SVM 判别器
        svm_loss = None
        if cfg.use_svm and timestep > 0 and timestep % cfg.svm_update_freq < max(
            1, sum(batch_ep_lengths)
        ):
            svm_loss = train_svm_discriminator(
                disc, disc_optimizer, list(success_buffer), list(failure_buffer), cfg
            )

        if episode % 10 == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else -float("inf")
            avg_length = np.mean(recent_lengths) if recent_lengths else 0
            log = (
                f"Episode {episode:5d} | "
                f"Timestep {timestep:7d} | "
                f"AvgReward {avg_reward:8.2f} | "
                f"AvgLength {avg_length:6.1f} | "
                f"Best {best_reward:8.2f} | "
                f"PolicyLoss {train_info['policy_loss']:.3f} | "
                f"ValueLoss {train_info['value_loss']:.3f}"
            )
            if svm_loss is not None:
                log += f" | SVMLoss {svm_loss:.3f}"
            print(log)

    env.close()
    print("=" * 70)
    print(f"训练结束 / Training finished. Best reward: {best_reward:.2f}")
    print("=" * 70)


# -----------------------------------------------------------------------------
# 命令行入口 / CLI Entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Reward + PPO reproduction")
    parser.add_argument("--no-svm", action="store_true", help="禁用 SVM 奖励 / Disable SVM reward")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 / Random seed")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="总训练步数 / Total timesteps")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    cfg.use_svm = not args.no_svm
    cfg.total_timesteps = args.total_timesteps
    main(cfg)
