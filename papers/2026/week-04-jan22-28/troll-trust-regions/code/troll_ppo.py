"""
TROLL: Trust Regions improve RL for LLMs — 简化复现 / Simplified Reproduction
=========================================================
论文: "TROLL: Trust Regions improve Reinforcement Learning for Large Language Models"
会议: ICLR 2026
作者: Philipp Becker, Niklas Freymuth, Serge Thilges, Fabian Otto, Gerhard Neumann

本代码演示 TROLL 的核心思想:
1. 标准 PPO 使用 clipping (clip(ratio, 1-ε, 1+ε)) 来限制策略更新幅度
2. TROLL 用可微分的 KL 信任区域投影替代 clipping，为每个 token 提供精确的 KL 约束
3. 投影仅作用于最重要的 token logits 稀疏子集，控制计算开销
4. 对比标准 PPO clipping 与 TROLL 在训练稳定性和策略更新质量上的差异

运行方式: python troll_ppo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# ------------------------------
# 1. 标准 PPO Clipping / Standard PPO Clipping
# ------------------------------
def ppo_clipped_loss(new_logprobs: torch.Tensor,
                     old_logprobs: torch.Tensor,
                     advantages: torch.Tensor,
                     epsilon: float = 0.2) -> Tuple[torch.Tensor, dict]:
    """
    标准 PPO clipped surrogate loss。
    new_logprobs, old_logprobs: [batch, seq_len]
    advantages: [batch, seq_len]

    返回: (loss, stats)
    """
    # 计算概率比率
    ratio = torch.exp(new_logprobs - old_logprobs)  # [batch, seq_len]

    # 未裁剪的目标
    surr1 = ratio * advantages
    # 裁剪后的目标
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages

    # 取最小值（防止策略过度优化优势估计）
    loss = -torch.min(surr1, surr2).mean()

    # 计算 KL 散度作为参考
    kl = (old_logprobs - new_logprobs).mean()

    stats = {
        'ratio_mean': ratio.mean().item(),
        'ratio_max': ratio.max().item(),
        'kl': kl.item(),
        'clipped_fraction': (ratio != torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)).float().mean().item(),
    }
    return loss, stats


# ------------------------------
# 2. TROLL: KL 信任区域投影 / TROLL: KL Trust Region Projection
# ------------------------------
def troll_trust_region_projection(new_logits: torch.Tensor,
                                  old_logits: torch.Tensor,
                                  kl_delta: float = 0.01,
                                  top_k_ratio: float = 0.3,
                                  num_steps: int = 5,
                                  step_size: float = 0.5) -> Tuple[torch.Tensor, dict]:
    """
    TROLL 核心: 将新策略的 logits 投影到以旧策略为中心、KL 半径为 kl_delta 的信任区域内。

    new_logits, old_logits: [batch, seq_len, vocab_size]
    kl_delta: 最大允许 KL 散度
    top_k_ratio: 稀疏子集比例 (只投影最重要的 top-k token)
    num_steps: 投影迭代步数
    step_size: 投影步长

    返回: (projected_logits, stats)
    """
    batch_size, seq_len, vocab_size = new_logits.shape
    device = new_logits.device

    # 计算旧策略概率分布
    old_probs = F.softmax(old_logits, dim=-1)  # [batch, seq_len, vocab_size]
    old_logprobs = F.log_softmax(old_logits, dim=-1)

    # 初始化投影后的 logits
    proj_logits = new_logits.clone()

    for step in range(num_steps):
        proj_logprobs = F.log_softmax(proj_logits, dim=-1)
        proj_probs = F.softmax(proj_logits, dim=-1)

        # 计算逐 token 的 KL 散度: KL(old || proj)
        # KL = sum(old_probs * (old_logprobs - proj_logprobs))
        kl_per_token = (old_probs * (old_logprobs - proj_logprobs)).sum(dim=-1)  # [batch, seq_len]

        # 对于 KL 已经小于阈值的 token，跳过投影
        mask = (kl_per_token > kl_delta).float()  # [batch, seq_len]

        if mask.sum() == 0:
            break  # 所有 token 都已满足 KL 约束

        # --- 稀疏子集选择: 只选择概率变化最大的 top-k tokens ---
        # 计算每个 token 位置的概率变化幅度
        prob_change = torch.abs(proj_probs - old_probs).sum(dim=-1)  # [batch, seq_len]

        # 选择 top-k 最重要的位置进行投影
        k = max(1, int(seq_len * top_k_ratio))
        _, top_indices = torch.topk(prob_change * mask, k=k, dim=-1)  # [batch, k]

        # 创建稀疏掩码
        sparse_mask = torch.zeros_like(mask)
        sparse_mask.scatter_(1, top_indices, 1.0)
        sparse_mask = sparse_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        # --------------------------------------------------------

        # 计算 KL 关于 proj_logits 的梯度
        # d(KL)/d(logits) = proj_probs - old_probs
        kl_grad = (proj_probs - old_probs) * sparse_mask  # [batch, seq_len, vocab_size]

        # 沿梯度方向更新 logits，使 KL 减小
        proj_logits = proj_logits - step_size * kl_grad

    final_proj_logprobs = F.log_softmax(proj_logits, dim=-1)
    final_kl = (old_probs * (old_logprobs - final_proj_logprobs)).sum(dim=-1).mean()

    stats = {
        'final_kl': final_kl.item(),
        'projection_steps': step + 1,
        'active_tokens': mask.sum().item(),
    }
    return proj_logits, stats


def troll_loss(new_logits: torch.Tensor,
               old_logits: torch.Tensor,
               old_logprobs: torch.Tensor,
               actions: torch.Tensor,
               advantages: torch.Tensor,
               kl_delta: float = 0.01,
               top_k_ratio: float = 0.3) -> Tuple[torch.Tensor, dict]:
    """
    TROLL 的完整损失函数。
    先用信任区域投影约束新策略的 logits，再计算策略梯度损失。

    new_logits: [batch, seq_len, vocab_size] — 当前策略的 logits
    old_logits: [batch, seq_len, vocab_size] — 旧策略的 logits
    old_logprobs: [batch, seq_len] — 旧策略在 actions 上的 log 概率
    actions: [batch, seq_len] — 采样的动作 (token ids)
    advantages: [batch, seq_len] — 优势估计
    """
    # --- TROLL 核心: 信任区域投影 ---
    proj_logits, proj_stats = troll_trust_region_projection(
        new_logits, old_logits, kl_delta=kl_delta, top_k_ratio=top_k_ratio
    )
    # --------------------------------

    # 计算投影后策略在 actions 上的 log 概率
    proj_logprobs = F.log_softmax(proj_logits, dim=-1)
    # 收集 actions 对应的 log 概率
    batch_size, seq_len = actions.shape
    proj_action_logprobs = proj_logprobs.gather(
        -1, actions.unsqueeze(-1)
    ).squeeze(-1)  # [batch, seq_len]

    # 计算比率
    ratio = torch.exp(proj_action_logprobs - old_logprobs)

    # TROLL 不需要 clipping，因为投影已经保证了信任区域
    loss = -(ratio * advantages).mean()

    stats = {
        'ratio_mean': ratio.mean().item(),
        'ratio_max': ratio.max().item(),
        'ratio_std': ratio.std().item(),
        **proj_stats,
    }
    return loss, stats


# ------------------------------
# 3. 简化的策略网络 / Simple Policy Network
# ------------------------------
class SimplePolicy(nn.Module):
    """
    简化的 Transformer 策略网络，用于演示 TROLL。
    """
    def __init__(self, vocab_size: int = 100, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(50, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, batch_first=True),
            num_layers=num_layers
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        返回 logits: [batch, seq_len, vocab_size]
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.transformer(x)
        return self.lm_head(x)


# ------------------------------
# 4. 对比实验 / Comparison Experiment
# ------------------------------
def compare_ppo_vs_troll():
    """
    在一个简化的合成任务上对比 PPO clipping 与 TROLL。
    任务: 从随机初始化的策略出发，通过 RL 优化使其趋向目标分布。
    """
    print("=" * 60)
    print("TROLL: Trust Regions improve RL for LLMs — 简化复现")
    print("Simplified Reproduction of TROLL")
    print("=" * 60)

    device = torch.device('cpu')
    vocab_size = 50
    seq_len = 10
    batch_size = 32
    num_epochs = 30

    # 创建策略网络
    policy = SimplePolicy(vocab_size=vocab_size, hidden_dim=128, num_layers=2).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # 创建目标分布（用于计算奖励）
    torch.manual_seed(42)
    target_logits = torch.randn(vocab_size).to(device) * 0.5

    def compute_reward(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        奖励函数: 当 action 对应的目标分布概率高时，奖励高。
        logits: [batch, seq_len, vocab_size]
        actions: [batch, seq_len]
        返回: [batch, seq_len]
        """
        target_probs = F.softmax(target_logits, dim=-1)
        rewards = target_probs[actions]  # [batch, seq_len]
        return rewards

    results = {}

    for method_name, use_troll in [("PPO-Clipping", False), ("TROLL", True)]:
        print(f"\n【训练方法 / Method: {method_name}】")

        # 重新初始化策略
        policy = SimplePolicy(vocab_size=vocab_size, hidden_dim=128, num_layers=2).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        kl_history = []
        reward_history = []
        ratio_history = []

        for epoch in range(num_epochs):
            # 生成随机输入序列
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

            # 旧策略的前向传播
            with torch.no_grad():
                old_logits = policy(input_ids)  # [batch, seq_len, vocab_size]
                old_logprobs_full = F.log_softmax(old_logits, dim=-1)

            # 采样动作
            actions = torch.multinomial(
                F.softmax(old_logits.reshape(-1, vocab_size), dim=-1),
                num_samples=1
            ).reshape(batch_size, seq_len)

            old_action_logprobs = old_logprobs_full.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

            # 计算奖励和优势
            rewards = compute_reward(old_logits, actions)
            advantages = rewards - rewards.mean(dim=1, keepdim=True)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 新策略的前向传播
            new_logits = policy(input_ids)

            if use_troll:
                loss, stats = troll_loss(
                    new_logits, old_logits, old_action_logprobs,
                    actions, advantages, kl_delta=0.02, top_k_ratio=0.3
                )
            else:
                new_logprobs_full = F.log_softmax(new_logits, dim=-1)
                new_action_logprobs = new_logprobs_full.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
                loss, stats = ppo_clipped_loss(
                    new_action_logprobs, old_action_logprobs, advantages, epsilon=0.2
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            kl_history.append(stats['kl'] if 'kl' in stats else stats['final_kl'])
            reward_history.append(rewards.mean().item())
            ratio_history.append(stats['ratio_mean'])

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:02d}: AvgReward={reward_history[-1]:.4f}, "
                      f"KL={kl_history[-1]:.4f}, RatioMean={ratio_history[-1]:.4f}")

        results[method_name] = {
            'kl_history': kl_history,
            'reward_history': reward_history,
            'ratio_history': ratio_history,
        }

    # --- 对比分析 ---
    print("\n" + "=" * 60)
    print("对比分析 / Comparison Analysis")
    print("=" * 60)

    ppo_rewards = results["PPO-Clipping"]['reward_history']
    troll_rewards = results["TROLL"]['reward_history']
    ppo_kls = results["PPO-Clipping"]['kl_history']
    troll_kls = results["TROLL"]['kl_history']

    print(f"  PPO 最终平均奖励 / PPO Final Avg Reward: {ppo_rewards[-1]:.4f}")
    print(f"  TROLL 最终平均奖励 / TROLL Final Avg Reward: {troll_rewards[-1]:.4f}")

    reward_improve = (troll_rewards[-1] - ppo_rewards[-1]) / (abs(ppo_rewards[-1]) + 1e-6) * 100
    print(f"  奖励相对变化 / Reward Change: {reward_improve:+.1f}%")

    print(f"  PPO 平均 KL / PPO Avg KL: {sum(ppo_kls)/len(ppo_kls):.4f}")
    print(f"  TROLL 平均 KL / TROLL Avg KL: {sum(troll_kls)/len(troll_kls):.4f}")

    # KL 稳定性分析
    ppo_kl_std = torch.tensor(ppo_kls).std().item()
    troll_kl_std = torch.tensor(troll_kls).std().item()
    print(f"  PPO KL 标准差 / PPO KL Std: {ppo_kl_std:.4f}")
    print(f"  TROLL KL 标准差 / TROLL KL Std: {troll_kl_std:.4f}")
    if troll_kl_std < ppo_kl_std:
        print(f"  TROLL KL 稳定性提升 / TROLL KL Stability: {((ppo_kl_std - troll_kl_std) / ppo_kl_std * 100):.1f}%")
    print()


def demo_projection_mechanism():
    """
    可视化演示 TROLL 的信任区域投影机制。
    """
    print("=" * 60)
    print("【信任区域投影机制演示 / Trust Region Projection Demo】")
    print("=" * 60)

    vocab_size = 20
    batch_size = 1
    seq_len = 1

    # 模拟旧策略 ( peaked 分布 )
    old_logits = torch.zeros(batch_size, seq_len, vocab_size)
    old_logits[0, 0, 5] = 2.0
    old_logits[0, 0, 8] = 1.5
    old_logits[0, 0, 12] = 1.0

    # 模拟新策略 ( 更分散的分布，可能偏离旧策略 )
    new_logits = torch.randn(batch_size, seq_len, vocab_size) * 1.5

    print(f"  投影前 KL / KL before projection: "
          f"{F.kl_div(F.log_softmax(old_logits, -1), F.softmax(new_logits, -1), reduction='batchmean').item():.4f}")

    # 应用 TROLL 投影
    proj_logits, stats = troll_trust_region_projection(
        new_logits, old_logits, kl_delta=0.1, top_k_ratio=0.5, num_steps=10
    )

    print(f"  投影后 KL / KL after projection: "
          f"{F.kl_div(F.log_softmax(old_logits, -1), F.softmax(proj_logits, -1), reduction='batchmean').item():.4f}")
    print(f"  投影步数 / Projection steps: {stats['projection_steps']}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("TROLL: Trust Regions improve RL for LLMs — 简化复现")
    print("Simplified Reproduction of TROLL")
    print("=" * 60)

    # 演示1: 信任区域投影机制
    demo_projection_mechanism()

    # 演示2: PPO vs. TROLL 对比训练
    compare_ppo_vs_troll()

    print("全部演示完成 / All demos completed.")
