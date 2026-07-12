"""
DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop RAG
PyTorch 概念复现（带中文注释）

本代码演示 DynaKRAG 的核心机制：
1. 将多跳证据获取建模为状态条件控制：每一步根据当前证据状态选择
   检索(RETRIEVE)、查询重构(REFORMULATE)、证据批判(CRITIQUE)、
   充分性判断(JUDGE) 或 生成答案(GENERATE)。
2. 有效性层（Validity Layer）：在每一步构造合法动作集合，防止控制器
   选择当前不可执行的操作（如未检索就判断充分性）。
3. 学习一个共享策略网络，在带掩码的交叉熵损失下模仿专家轨迹。

运行方式：
    pip install torch
    python dynakrag_demo.py
"""

import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- 超参数 / Hyperparameters ----------------------------
STATE_DIM = 8            # 证据状态向量维度
NUM_ACTIONS = 5          # 原子动作数量
HIDDEN_DIM = 64
MAX_DOCS = 4             # 最多检索文档数
MIN_EVIDENCE = 2         # 判断充分性所需最少证据数
MAX_STEPS = 10           # 单条轨迹最大步数
LEARNING_RATE = 1e-2
NUM_EPOCHS = 6
BATCH_SIZE = 32
NUM_TRAIN_TRAJ = 2000
NUM_TEST_TRAJ = 300
DEVICE = torch.device("cpu")


# 动作语义 / Action semantics
RETRIEVE, REFORMULATE, CRITIQUE, JUDGE, GENERATE = 0, 1, 2, 3, 4
ACTION_NAMES = ["RETRIEVE", "REFORMULATE", "CRITIQUE", "JUDGE", "GENERATE"]


# ---------------------------- 证据状态 / Evidence State ----------------------------
class EvidenceState:
    """模拟多跳 RAG 的证据状态，包含可解释的特征。"""
    def __init__(self):
        self.num_retrieved = 0
        self.query_clarity = random.uniform(0.2, 0.8)
        self.evidence_confidence = 0.0
        self.has_contradiction = 0
        self.judged = 0
        self.sufficiency = 0.0
        self.reformulated = 0

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.num_retrieved / MAX_DOCS,
            self.query_clarity,
            self.evidence_confidence,
            self.has_contradiction,
            self.judged,
            self.sufficiency,
            self.reformulated,
            1.0 if self.judged == 1 else 0.0,
        ], dtype=torch.float32)

    def copy(self):
        s = EvidenceState.__new__(EvidenceState)
        s.__dict__.update(self.__dict__)
        return s


# ---------------------------- 专家策略 / Expert Policy ----------------------------
def expert_action(state: EvidenceState) -> int:
    """
    确定性专家：
    - 证据不足时先检索，若查询不清晰则重构；
    - 证据足够但置信度低/有矛盾时进行批判；
    - 证据充分且清晰时进行充分性判断；
    - 判断完成后生成答案。
    """
    if state.num_retrieved < MIN_EVIDENCE:
        if state.query_clarity < 0.5 and state.reformulated == 0:
            return REFORMULATE
        return RETRIEVE
    if state.judged == 0:
        if state.evidence_confidence < 0.6 or state.has_contradiction:
            return CRITIQUE
        return JUDGE
    return GENERATE


def apply_action(state: EvidenceState, action: int):
    """执行动作并更新状态（模拟环境）。"""
    if action == RETRIEVE and state.num_retrieved < MAX_DOCS:
        state.num_retrieved += 1
        # 检索带来新证据，可能引入矛盾
        state.evidence_confidence = random.uniform(0.4, 0.9)
        state.has_contradiction = 1 if random.random() < 0.25 else 0
    elif action == REFORMULATE and state.reformulated == 0:
        state.query_clarity = min(1.0, state.query_clarity + random.uniform(0.2, 0.4))
        state.reformulated = 1
    elif action == CRITIQUE and state.num_retrieved > 0:
        state.evidence_confidence = min(1.0, state.evidence_confidence + random.uniform(0.1, 0.3))
        state.has_contradiction = 0
    elif action == JUDGE and state.num_retrieved >= MIN_EVIDENCE:
        state.judged = 1
        state.sufficiency = state.evidence_confidence * state.query_clarity
    elif action == GENERATE:
        pass  # 终止


def validity_mask(state: EvidenceState) -> torch.Tensor:
    """
    有效性层：根据当前状态返回合法动作掩码。
    """
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
    mask[RETRIEVE] = state.num_retrieved < MAX_DOCS
    mask[REFORMULATE] = (state.reformulated == 0) and (state.num_retrieved < MAX_DOCS)
    mask[CRITIQUE] = state.num_retrieved > 0 and state.judged == 0
    mask[JUDGE] = state.num_retrieved >= MIN_EVIDENCE and state.judged == 0
    mask[GENERATE] = state.judged == 1
    return mask


# ---------------------------- 数据生成 / Trajectory Generation ----------------------------
def generate_trajectory() -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    """生成一条专家轨迹，返回 (state, validity_mask, action) 列表。"""
    state = EvidenceState()
    traj = []
    for _ in range(MAX_STEPS):
        action = expert_action(state)
        mask = validity_mask(state)
        traj.append((state.to_tensor(), mask, action))
        apply_action(state, action)
        if action == GENERATE:
            break
    return traj


def generate_dataset(n: int):
    """生成 n 条轨迹并展开为 (state, mask, action) 样本。"""
    samples = []
    for _ in range(n):
        samples.extend(generate_trajectory())
    return samples


# ---------------------------- 控制器网络 / Learned Controller ----------------------------
class Controller(nn.Module):
    """共享策略网络：根据证据状态输出各原子动作的 logits。"""
    def __init__(self, state_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim)
            mask:  (B, num_actions) bool, True 表示动作合法
        Returns:
            logits: (B, num_actions)，非法动作被设为 -inf
        """
        logits = self.net(state)
        # 将非法动作的概率设为极小
        logits = logits.masked_fill(~mask, float("-1e9"))
        return logits


# ---------------------------- 训练 / Training ----------------------------
def train_controller(controller: nn.Module, samples: List[Tuple]):
    optimizer = torch.optim.Adam(controller.parameters(), lr=LEARNING_RATE)
    dataset = samples[:]
    random.shuffle(dataset)
    n = len(dataset)
    for epoch in range(1, NUM_EPOCHS + 1):
        random.shuffle(dataset)
        total_loss = 0.0
        correct = 0
        total = 0
        for i in range(0, n, BATCH_SIZE):
            batch = dataset[i:i + BATCH_SIZE]
            states = torch.stack([s[0] for s in batch])
            masks = torch.stack([s[1] for s in batch])
            actions = torch.tensor([s[2] for s in batch], dtype=torch.long)
            states, masks, actions = states.to(DEVICE), masks.to(DEVICE), actions.to(DEVICE)

            optimizer.zero_grad()
            logits = controller(states, masks)
            loss = F.cross_entropy(logits, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * states.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == actions).sum().item()
            total += states.size(0)
        print(f"  Epoch {epoch}/{NUM_EPOCHS}, Loss={total_loss / total:.4f}, Acc={correct / total:.3%}")


# ---------------------------- 评估 / Evaluation ----------------------------
def rollout(controller: nn.Module) -> Tuple[List[int], bool]:
    """用学习到的控制器执行一条轨迹，返回动作序列和是否合法。"""
    state = EvidenceState()
    actions = []
    legal = True
    with torch.no_grad():
        for _ in range(MAX_STEPS):
            mask = validity_mask(state)
            logits = controller(state.to_tensor().unsqueeze(0).to(DEVICE), mask.unsqueeze(0).to(DEVICE))
            action = logits.argmax(dim=-1).item()
            if not mask[action].item():
                legal = False
            actions.append(action)
            apply_action(state, action)
            if action == GENERATE:
                break
    return actions, legal


def evaluate(controller: nn.Module, n: int):
    """与专家轨迹对比，计算动作一致率与合法性。"""
    match_count = 0
    total_count = 0
    legal_count = 0
    for _ in range(n):
        expert_traj = generate_trajectory()
        pred_actions, legal = rollout(controller)
        if legal:
            legal_count += 1
        # 按步对齐比较
        min_len = min(len(expert_traj), len(pred_actions))
        for i in range(min_len):
            if pred_actions[i] == expert_traj[i][2]:
                match_count += 1
            total_count += 1
    print(f"  专家动作一致率 / Expert action agreement: {match_count / total_count:.2%}")
    print(f"  合法轨迹比例 / Legal trajectory ratio: {legal_count / n:.2%}")


# ---------------------------- 主程序 / Main ----------------------------
def main():
    random.seed(7)
    torch.manual_seed(7)

    print("生成专家轨迹训练数据 / Generating expert trajectories...")
    train_samples = generate_dataset(NUM_TRAIN_TRAJ)
    print(f"  训练样本数 / Training samples: {len(train_samples)}")

    controller = Controller(STATE_DIM, HIDDEN_DIM, NUM_ACTIONS).to(DEVICE)
    print("\n训练 DynaKRAG 控制器 / Training DynaKRAG controller...")
    train_controller(controller, train_samples)

    print("\n=== 测试集评估 / Test Evaluation ===")
    evaluate(controller, NUM_TEST_TRAJ)

    # 打印一条示例轨迹
    print("\n=== 示例轨迹 / Sample Trajectory ===")
    state = EvidenceState()
    controller.eval()
    with torch.no_grad():
        for step in range(MAX_STEPS):
            mask = validity_mask(state)
            logits = controller(state.to_tensor().unsqueeze(0).to(DEVICE), mask.unsqueeze(0).to(DEVICE))
            action = logits.argmax(dim=-1).item()
            print(f"Step {step}: {ACTION_NAMES[action]}  (valid={mask[action].item()})")
            apply_action(state, action)
            if action == GENERATE:
                break


if __name__ == "__main__":
    main()
