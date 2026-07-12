"""
reasoning_allocator.py
A minimal PyTorch reproduction of the adaptive test-time compute allocator
from "Reasoning on Demand: Adaptive Test-Time Compute Allocation for LLM Agents"
arXiv: 2606.04123.

本脚本是一个最小化的 PyTorch 复现，演示如何训练一个轻量级价值网络，
让它在每一步决定：继续推理（think）还是立即作答（answer）。
因为完整 LLM 推理开销过大，这里用一个可运行的合成数学任务来展示核心思想：
每个问题由若干子步骤组成，模型需要在“准确率”与“token/步数开销”之间做权衡。
"""

import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# 1. 合成任务环境 / Synthetic Task Environment
# ---------------------------------------------------------------------------
# 我们构造一类简单的多步算术问题：
#   x0 -> [op0] -> x1 -> [op1] -> x2 -> ... -> answer
# 每一步 op 是 +k, -k, *k, /k 之一（k 为小整数）。
# 智能体必须在最多 MAX_STEPS 步内给出最终答案；答对得正奖励，答错得负奖励。
# 这样即可模拟真实推理中“子问题难度不同、计算预算有限”的场景。

MAX_STEPS = 8          # 每道题允许的最大推理步数
MAX_VALUE = 1000.0     # 答案绝对值上限，防止数值爆炸
OPS = ["+", "-", "*", "/"]


def generate_problem(num_steps: int = None) -> Tuple[List[Tuple[str, float]], float, float]:
    """
    随机生成一道多步算术题。
    返回：操作序列 ops、初始值 initial_value、以及正确答案 true_answer。
    同时返回初始值是为了保证 rollout 与真实答案的计算起点一致。
    """
    if num_steps is None:
        num_steps = random.randint(3, MAX_STEPS)
    initial_value = float(random.randint(1, 20))
    x = initial_value
    ops = []
    for _ in range(num_steps):
        op = random.choice(OPS)
        # 除法时选择能整除的小整数，避免浮点误差干扰奖励判断
        if op == "/":
            k = float(random.choice([2, 4, 5, 10]))
        else:
            k = float(random.randint(1, 9))
        ops.append((op, k))
        # 预计算真实中间结果
        if op == "+":
            x = x + k
        elif op == "-":
            x = x - k
        elif op == "*":
            x = x * k
        elif op == "/":
            x = x / k
        x = max(min(x, MAX_VALUE), -MAX_VALUE)
    return ops, initial_value, x


def apply_op(value: float, op: Tuple[str, float]) -> float:
    """将单个操作 op 应用于当前值 value。"""
    symbol, k = op
    if symbol == "+":
        return value + k
    if symbol == "-":
        return value - k
    if symbol == "*":
        return value * k
    if symbol == "/":
        return value / k if k != 0 else value
    return value


# ---------------------------------------------------------------------------
# 2. 状态编码 / State Encoding
# ---------------------------------------------------------------------------
# 在真实 LLM 中，状态是隐藏层表示；这里我们用一个固定维度的向量来近似：
# 包含当前已用步数、当前中间值、以及已执行操作的 one-hot 累计统计。

STATE_DIM = 4 + len(OPS)  # [steps_ratio, value_norm, has_answered, budget_left, op_counts]


def encode_state(step: int, current_value: float, op_history: List[Tuple[str, float]]) -> torch.Tensor:
    """
    将当前推理状态编码为向量。
    - step / MAX_STEPS：已用预算比例
    - current_value / MAX_VALUE：当前中间值归一化
    - 是否已作答标志（训练时恒为 0，直到 agent 选择 answer）
    - 剩余预算比例
    - 每种操作的历史出现次数归一化
    """
    vec = torch.zeros(STATE_DIM)
    vec[0] = step / MAX_STEPS
    vec[1] = current_value / MAX_VALUE
    vec[2] = 0.0
    vec[3] = 1.0 - step / MAX_STEPS
    op_counts = {op: 0 for op in OPS}
    for symbol, _ in op_history:
        op_counts[symbol] += 1
    for i, op in enumerate(OPS):
        vec[4 + i] = op_counts[op] / max(len(op_history), 1)
    return vec


# ---------------------------------------------------------------------------
# 3. 计算分配器（价值网络）/ Compute Allocator (Value Network)
# ---------------------------------------------------------------------------
# 这是论文核心：一个轻量级网络，根据当前状态预测“继续推理”与“立即作答”
# 各自的期望效用，从而动态分配测试时计算。

class ComputeAllocator(nn.Module):
    """
    价值网络：输入当前状态，输出两个动作的 logit。
    动作 0：继续推理（think）
    动作 1：立即作答（answer）
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 两个动作：think / answer
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        根据当前状态选择动作。
        返回：动作索引，以及对应的对数概率（用于策略梯度）。
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        if deterministic:
            action = int(torch.argmax(logits).item())
        else:
            action = int(dist.sample().item())
        log_prob = dist.log_prob(torch.tensor(action))
        return action, log_prob


# ---------------------------------------------------------------------------
# 4. 交互式推理 episode / Rollout Function
# ---------------------------------------------------------------------------
# 模拟一次完整的推理过程：分配器决定 think/answer，基础模型执行 think。
# 当选择 answer 或达到最大步数时，episode 结束，返回奖励和轨迹。

ACTION_THINK = 0
ACTION_ANSWER = 1


def rollout(
    allocator: ComputeAllocator,
    ops: List[Tuple[str, float]],
    initial_value: float,
    true_answer: float,
    cost_per_step: float = 0.05,
    deterministic: bool = False,
) -> Tuple[float, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    运行一个推理 episode。

    参数：
        allocator: 计算分配器。
        ops: 题目操作序列。
        initial_value: 与 generate_problem 一致的初始值。
        true_answer: 正确答案。
        cost_per_step: 每多一步推理的惩罚系数，模拟 token/FLOPs 开销。
        deterministic: 是否使用确定性策略（评估时 True，训练时 False）。

    返回：
        reward: 最终奖励（准确率奖励 - 计算开销）。
        states: 状态列表。
        actions: 动作列表。
        log_probs: 对应动作的对数概率列表。
    """
    states, actions, log_probs = [], [], []
    current_value = initial_value
    op_index = 0

    for step in range(MAX_STEPS):
        state_vec = encode_state(step, current_value, ops[:op_index])
        states.append(state_vec)

        action, log_prob = allocator.select_action(state_vec, deterministic=deterministic)
        actions.append(torch.tensor(action))
        log_probs.append(log_prob)

        if action == ACTION_ANSWER:
            # 立即作答：使用当前中间值作为最终答案
            break

        # 继续推理：执行下一个操作
        if op_index < len(ops):
            op = ops[op_index]
            # 基础模型预测下一中间值。在完整论文中，这一步由 LLM 生成；
            # 为了单独演示分配器的学习，这里使用真实下一值（perfect reasoner），
            # 这样奖励信号只反映“何时停止推理”，而非基础模型的推理准确性。
            current_value = apply_op(current_value, op)
            op_index += 1
        else:
            # 没有更多操作了，强制作答
            break

    # 计算奖励
    accuracy_reward = 1.0 if abs(current_value - true_answer) < 1e-3 else 0.0
    num_steps = len(actions)
    compute_penalty = cost_per_step * num_steps
    # 未完成的操作视作错误惩罚（模拟过早作答）
    completion_penalty = 0.2 * max(0, len(ops) - op_index)
    reward = accuracy_reward - compute_penalty - completion_penalty

    return reward, states, actions, log_probs


# ---------------------------------------------------------------------------
# 5. 训练循环 / Training Loop
# ---------------------------------------------------------------------------
# 使用 REINFORCE（策略梯度）训练计算分配器。每一 batch 生成若干题目，
# 根据最终奖励更新分配器的策略，使其学会“该停则停、该想则想”。


def train_allocator(
    allocator: ComputeAllocator,
    num_iterations: int = 500,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    cost_per_step: float = 0.05,
) -> List[float]:
    """训练计算分配器，返回每轮平均奖励历史。"""
    optimizer = optim.Adam(allocator.parameters(), lr=learning_rate)
    reward_history = []

    for iteration in range(num_iterations):
        episodes = []
        for _ in range(batch_size):
            ops, initial_value, true_answer = generate_problem()
            reward, states, actions, log_probs = rollout(
                allocator, ops, initial_value, true_answer,
                cost_per_step=cost_per_step, deterministic=False
            )
            episodes.append((reward, log_probs))

        # REINFORCE 损失：期望回报 = - E[R * log_pi(a|s)]
        # 这里用平均奖励作为基线（baseline），降低方差
        rewards = [r for r, _ in episodes]
        mean_reward = sum(rewards) / len(rewards)
        reward_history.append(mean_reward)

        loss = 0.0
        for reward, log_probs in episodes:
            advantage = reward - mean_reward
            for lp in log_probs:
                loss -= advantage * lp
        loss = loss / batch_size

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止训练初期方差过大导致参数爆炸
        torch.nn.utils.clip_grad_norm_(allocator.parameters(), max_norm=1.0)
        optimizer.step()

        if (iteration + 1) % 50 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, "
                  f"mean reward: {mean_reward:.3f}, "
                  f"loss: {loss.item():.3f}")

    return reward_history


# ---------------------------------------------------------------------------
# 6. 评估 / Evaluation
# ---------------------------------------------------------------------------

def evaluate_detailed(
    allocator: ComputeAllocator,
    num_episodes: int = 100,
    cost_per_step: float = 0.05,
) -> Tuple[float, float, float]:
    """在确定性策略下评估分配器，返回平均奖励、准确率、平均使用步数。"""
    total_reward = 0.0
    correct = 0
    total_steps = 0

    for _ in range(num_episodes):
        ops, initial_value, true_answer = generate_problem()
        current_value = initial_value
        op_index = 0
        reward = 0.0

        for step in range(MAX_STEPS):
            state_vec = encode_state(step, current_value, ops[:op_index])
            action, _ = allocator.select_action(state_vec, deterministic=True)
            total_steps += 1

            if action == ACTION_ANSWER:
                break

            if op_index < len(ops):
                op = ops[op_index]
                # 与 rollout 一致，使用真实下一值进行演示
                current_value = apply_op(current_value, op)
                op_index += 1
            else:
                break

        accuracy_reward = 1.0 if abs(current_value - true_answer) < 1e-3 else 0.0
        compute_penalty = cost_per_step * (step + 1)
        completion_penalty = 0.2 * max(0, len(ops) - op_index)
        reward = accuracy_reward - compute_penalty - completion_penalty

        total_reward += reward
        correct += int(accuracy_reward)

    return total_reward / num_episodes, correct / num_episodes, total_steps / num_episodes


# ---------------------------------------------------------------------------
# 7. 主函数 / Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)
    torch.manual_seed(42)

    # 初始化计算分配器
    allocator = ComputeAllocator(state_dim=STATE_DIM, hidden_dim=64)

    cost_per_step = 0.05
    print("开始训练计算分配器... / Starting allocator training...")
    reward_history = train_allocator(
        allocator,
        num_iterations=500, batch_size=32,
        learning_rate=3e-4, cost_per_step=cost_per_step
    )

    print("\n训练完成。/ Training complete.")
    print(f"最终平均奖励: {reward_history[-1]:.3f} / Final mean reward: {reward_history[-1]:.3f}")

    avg_reward, accuracy, avg_steps = evaluate_detailed(
        allocator, num_episodes=100, cost_per_step=cost_per_step
    )
    print("\n===== 评估结果 / Evaluation Results =====")
    print(f"平均奖励 / Avg reward: {avg_reward:.3f}")
    print(f"准确率 / Accuracy: {accuracy:.2%}")
    print(f"平均步数 / Avg steps: {avg_steps:.2f}")

    # 演示一个具体例子 / Demo one example
    print("\n===== 单题演示 / Single Problem Demo =====")
    ops, initial_value, true_answer = generate_problem(num_steps=5)
    print(f"操作序列 / Operations: {ops}")
    print(f"正确答案 / True answer: {true_answer:.3f}")

    current_value = initial_value
    print(f"初始值 / Initial value: {current_value:.3f}")
    op_index = 0
    for step in range(MAX_STEPS):
        state_vec = encode_state(step, current_value, ops[:op_index])
        action, _ = allocator.select_action(state_vec, deterministic=True)
        action_name = "THINK" if action == ACTION_THINK else "ANSWER"
        print(f"Step {step}: action={action_name}, current_value={current_value:.3f}")
        if action == ACTION_ANSWER:
            break
        if op_index < len(ops):
            op = ops[op_index]
            # 演示中仍使用真实下一值，突出分配器的决策行为
            current_value = apply_op(current_value, op)
            op_index += 1
        else:
            # 没有更多操作，强制作答
            break
    print(f"模型最终答案 / Model final answer: {current_value:.3f}")
    print(f"是否正确 / Correct: {abs(current_value - true_answer) < 1e-3}")


if __name__ == "__main__":
    main()
