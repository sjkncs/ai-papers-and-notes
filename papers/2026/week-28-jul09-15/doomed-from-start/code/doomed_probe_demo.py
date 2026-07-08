"""
Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade
PyTorch 概念复现（带中文注释）

本代码演示论文核心思想：
1. LLM Agent 的 episode 失败可以从其隐藏层激活中早期预测。
2. 训练一个轻量级探针，为每一轮输出“该 episode 注定要失败”的概率。
3. 按用户指定的全局 recall 目标（如 90% 的成功 episode 必须存活），
   为每一轮校准一个门限，形成级联早停策略。
4. 统计级联策略在实际测试中节省的推理轮数与真实 recall。

运行方式：
    pip install torch
    python doomed_probe_demo.py
"""

import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- 超参数 / Hyperparameters ----------------------------
HIDDEN_DIM = 32          # 模拟的 LLM 隐藏状态维度
MAX_ROUNDS = 10          # 每个 episode 的最大轮数
NUM_EPISODES = 8000      # 模拟 episode 总数
SUCCESS_RATE = 0.35      # 成功 episode 比例
LEARNING_RATE = 1e-2
NUM_EPOCHS = 5
BATCH_SIZE = 64
GLOBAL_RECALL_TARGET = 0.90  # 全局 recall 目标：90% 的成功 episode 不被误杀
DEVICE = torch.device("cpu")


# ---------------------------- 数据生成 / Episode Simulation ----------------------------
def generate_episodes(n: int, hidden_dim: int, max_rounds: int, success_rate: float):
    """
    模拟 Agent episode：
    - 成功 episode 的隐藏状态有一个稳定的“成功方向”，失败分数较低；
    - 失败 episode 的隐藏状态随着时间推移越来越混乱，失败分数较高。
    返回 [(states, label), ...]，label=1 表示成功，0 表示失败。
    """
    episodes = []
    direction = torch.randn(hidden_dim)
    direction = direction / direction.norm()

    for _ in range(n):
        label = 1 if random.random() < success_rate else 0
        states = []
        for r in range(max_rounds):
            if label == 1:
                # 成功：失败分数低，随轮次略有波动
                fail_score = 0.2 + 0.05 * r + torch.randn(1).item() * 0.3
            else:
                # 失败：失败分数随轮次明显上升
                fail_score = -0.1 + 0.25 * r + torch.randn(1).item() * 0.4
            state = direction * fail_score + torch.randn(hidden_dim) * 0.2
            states.append(state)
        episodes.append((states, label))
    return episodes


# ---------------------------- 探针模型 / Failure Probe ----------------------------
class RoundAwareProbe(nn.Module):
    """输入隐藏状态与轮次索引，输出 episode 失败的 logit。"""
    def __init__(self, hidden_dim: int, max_rounds: int):
        super().__init__()
        self.round_embed = nn.Embedding(max_rounds, 8)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, hidden: torch.Tensor, round_idx: torch.Tensor) -> torch.Tensor:
        r = self.round_embed(round_idx)
        x = torch.cat([hidden, r], dim=-1)
        return self.mlp(x).squeeze(-1)


def make_batches(episodes: List[Tuple[List[torch.Tensor], int]], batch_size: int):
    """将 episode 列表按轮次展开成 (hidden, round, label) 批次。"""
    all_hidden, all_round, all_label = [], [], []
    for states, label in episodes:
        for r, h in enumerate(states):
            all_hidden.append(h)
            all_round.append(r)
            all_label.append(1 - label)  # label=失败概率：成功=0，失败=1
    data = list(zip(all_hidden, all_round, all_label))
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        h = torch.stack([b[0] for b in batch])
        r = torch.tensor([b[1] for b in batch], dtype=torch.long)
        y = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        yield h, r, y


# ---------------------------- 校准与级联 / Calibration & Cascade ----------------------------
def calibrate_gates(probe: nn.Module, val_episodes: List[Tuple[List[torch.Tensor], int]], target: float):
    """
    为每一轮计算门限，使得全局成功 recall >= target。
    假设每轮 recall 预算相等：r_per_round = target^(1/max_rounds)。
    对每一轮的成功 episode，取失败分数的 target_quantile 作为门限。
    """
    probe.eval()
    max_rounds = MAX_ROUNDS
    per_round_recall = target ** (1.0 / max_rounds)
    thresholds = []
    with torch.no_grad():
        for r in range(max_rounds):
            success_scores = []
            for states, label in val_episodes:
                if label == 1:
                    h = states[r].unsqueeze(0).to(DEVICE)
                    ri = torch.tensor([r], dtype=torch.long, device=DEVICE)
                    logit = probe(h, ri).item()
                    success_scores.append(logit)
            if len(success_scores) == 0:
                thresholds.append(float("inf"))
                continue
            q = per_round_recall
            # torch.quantile 需要 torch 张量
            scores_t = torch.tensor(success_scores)
            thr = torch.quantile(scores_t, q).item()
            thresholds.append(thr)
    return thresholds


def evaluate_cascade(probe: nn.Module, test_episodes: List[Tuple[List[torch.Tensor], int]], thresholds: List[float]):
    """
    应用级联早停：每一轮若失败分数超过该轮门限则中止 episode。
    返回全局 recall、成功 episode 终止率、节省的推理轮数比例。
    """
    probe.eval()
    max_rounds = MAX_ROUNDS
    total_possible_rounds = len(test_episodes) * max_rounds
    executed_rounds = 0
    success_total = 0
    success_survived = 0

    with torch.no_grad():
        for states, label in test_episodes:
            aborted = False
            for r in range(max_rounds):
                executed_rounds += 1
                h = states[r].unsqueeze(0).to(DEVICE)
                ri = torch.tensor([r], dtype=torch.long, device=DEVICE)
                score = probe(h, ri).item()
                if score > thresholds[r]:
                    aborted = True
                    break
            if label == 1:
                success_total += 1
                if not aborted:
                    success_survived += 1

    recall = success_survived / success_total if success_total > 0 else 0.0
    savings = 1.0 - executed_rounds / total_possible_rounds
    return recall, savings


# ---------------------------- 主程序 / Main ----------------------------
def main():
    random.seed(42)
    torch.manual_seed(42)

    print("生成模拟 episode 数据 / Generating synthetic episodes...")
    all_eps = generate_episodes(NUM_EPISODES, HIDDEN_DIM, MAX_ROUNDS, SUCCESS_RATE)
    random.shuffle(all_eps)
    n_train = int(0.6 * NUM_EPISODES)
    n_val = int(0.2 * NUM_EPISODES)
    train_eps = all_eps[:n_train]
    val_eps = all_eps[n_train:n_train + n_val]
    test_eps = all_eps[n_train + n_val:]

    probe = RoundAwareProbe(HIDDEN_DIM, MAX_ROUNDS).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LEARNING_RATE)

    print("\n训练失败探针 / Training failure probe...")
    for epoch in range(1, NUM_EPOCHS + 1):
        probe.train()
        epoch_loss = 0.0
        batches = 0
        for h, r, y in make_batches(train_eps, BATCH_SIZE):
            h = h.to(DEVICE)
            r = r.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            logits = probe(h, r)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        print(f"  Epoch {epoch}/{NUM_EPOCHS}, Avg BCE Loss={epoch_loss / max(batches, 1):.4f}")

    print(f"\n按全局 recall 目标 {GLOBAL_RECALL_TARGET} 校准级联门限 / Calibrating cascade gates...")
    thresholds = calibrate_gates(probe, val_eps, GLOBAL_RECALL_TARGET)
    print(f"  每轮门限 / Thresholds per round: {[f'{t:.3f}' for t in thresholds]}")

    recall, savings = evaluate_cascade(probe, test_eps, thresholds)
    print("\n=== 级联早停测试结果 / Cascade Test Results ===")
    print(f"全局 Recall / Global recall: {recall:.3%} (目标 / target: {GLOBAL_RECALL_TARGET:.0%})")
    print(f"推理轮数节省 / Inference-round savings: {savings:.2%}")
    print("（级联在尽量不误杀成功 episode 的前提下，提前中止了大量注定失败的轨迹。）")


if __name__ == "__main__":
    main()
