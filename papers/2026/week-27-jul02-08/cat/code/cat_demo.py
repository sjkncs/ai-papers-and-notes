"""
CAT: Confidence-Adaptive Thinking for Efficient Reasoning of Large Reasoning Models
PyTorch 概念复现（带中文注释）

本代码演示 CAT 的核心机制，不依赖真实大模型，而是使用一个随机 Transformer
作为“推理模型”来生成思维链（chain-of-thought）token，并在每一步估计自身的
“自信度”。当自信度高于阈值时提前终止推理并输出答案；当自信度较低时继续生成
更多推理步骤。通过简单的偏好优化风格损失，鼓励模型在简单问题上给出更短的正确
回答、在难题上保留足够推理长度。

运行方式：
    pip install torch
    python cat_demo.py
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- 超参数 / Hyperparameters ----------------------------
VOCAB_SIZE = 128          # 词汇表大小（随机模型较小，便于演示）
EMBED_DIM = 128           # 词嵌入维度
NUM_HEADS = 4             # Transformer 注意力头数
NUM_LAYERS = 4            # Transformer 层数
MAX_REASON_LEN = 24       # 最大推理步数
CONF_STOP = 0.75          # 自信度阈值：高于此值可提前输出答案
CONF_UNCERTAIN = 0.35     # 低自信度阈值：低于此值强制继续推理
TEMPERATURE = 0.8         # 温度：控制生成分布的锐度
LOGIT_SCALE = 8.0         # 输出 logit 放大倍数：模拟真实模型的峰值分布
DEVICE = torch.device("cpu")
LEARNING_RATE = 1e-3
NUM_TRAIN_STEPS = 200     # 演示训练步数


# ---------------------------- 工具函数 / Utilities ----------------------------
def confidence_from_logits(logits: torch.Tensor, temperature: float = TEMPERATURE) -> float:
    """
    简单自信度估计：温度缩放后最大 softmax 概率。
    在真实 LRM 中可替换为 PRM/ORM/内部状态估计器。
    """
    probs = F.softmax(logits / temperature, dim=-1)
    return probs.max().item()


def stop_token(token_id: int) -> bool:
    """终止符：遇到特殊 token 则停止当前路径。"""
    return token_id in {0, 1}  # 0: <eos>, 1: <answer>


# ---------------------------- 随机推理模型 / Toy Reasoner ----------------------------
class ToyReasoner(nn.Module):
    """
    一个极简 Transformer，充当“大推理模型”。
    输入 token 序列，输出下一个 token 的 logits 分布。
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """返回每个位置的 logits（已放大以模拟真实模型的峰值分布）。"""
        x = self.embedding(tokens)
        mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1)).to(DEVICE)
        h = self.transformer(x, mask=mask, is_causal=True)
        return self.head(h) * LOGIT_SCALE


# ---------------------------- 推理结果结构 / Inference Result ----------------------------
@dataclass
class ReasonResult:
    """记录一次推理的完整信息。"""
    prompt: List[int]
    reasoning_tokens: List[int]
    answer_token: int
    confidences: List[float]
    stopped_early: bool


# ---------------------------- CAT 推理器 / CAT Reasoner ----------------------------
class CATReasoner:
    """
    自信度自适应推理器。

    核心逻辑：
    1. 对当前序列调用模型，得到下一个 token 的 logits；
    2. 估计该步骤的自信度；
    3. 若自信度高于 CONF_STOP 且已生成一定推理长度，则输出答案 token 并停止；
    4. 若自信度低于 CONF_UNCERTAIN，则强制继续生成更多推理；
    5. 重复直到遇到终止符或达到最大长度。
    """
    def __init__(self, model: ToyReasoner):
        self.model = model
        self.model.eval()

    def generate(self, prompt: List[int], min_reason_len: int = 3) -> ReasonResult:
        """根据自信度自适应地生成推理链与答案。"""
        tokens = list(prompt)
        reasoning_tokens: List[int] = []
        confidences: List[float] = []
        stopped_early = False

        for step in range(MAX_REASON_LEN):
            input_ids = torch.tensor([tokens[-MAX_REASON_LEN:]], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                logits = self.model(input_ids)[:, -1, :]  # 只取最后一个位置的 logits

            conf = confidence_from_logits(logits)
            confidences.append(conf)

            # 若已有足够推理且自信度足够高，提前输出答案
            if step >= min_reason_len and conf >= CONF_STOP:
                answer_token = logits.argmax(dim=-1).item()
                stopped_early = True
                return ReasonResult(
                    prompt=prompt,
                    reasoning_tokens=reasoning_tokens,
                    answer_token=answer_token,
                    confidences=confidences,
                    stopped_early=stopped_early,
                )

            # 低自信度时强制继续推理；否则采样下一个推理 token
            if conf < CONF_UNCERTAIN:
                # 强制继续：仍从分布中采样，但标记为不确定性步骤
                pass

            probs = F.softmax(logits / TEMPERATURE, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if stop_token(next_token):
                # 模型自己决定终止，此时用 greedy 选答案
                answer_token = logits.argmax(dim=-1).item()
                return ReasonResult(
                    prompt=prompt,
                    reasoning_tokens=reasoning_tokens,
                    answer_token=answer_token,
                    confidences=confidences,
                    stopped_early=stopped_early,
                )

            reasoning_tokens.append(next_token)
            tokens.append(next_token)

        # 达到最大长度仍未停止，直接 greedy 选答案
        input_ids = torch.tensor([tokens[-MAX_REASON_LEN:]], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            logits = self.model(input_ids)[:, -1, :]
        answer_token = logits.argmax(dim=-1).item()
        return ReasonResult(
            prompt=prompt,
            reasoning_tokens=reasoning_tokens,
            answer_token=answer_token,
            confidences=confidences,
            stopped_early=stopped_early,
        )


# ---------------------------- 偏好优化风格训练 / Preference-Style Training ----------------------------
def preference_loss(
    model: ToyReasoner,
    optimizer: torch.optim.Optimizer,
    prompt: List[int],
    short_correct: List[int],
    long_correct: List[int],
    short_wrong: List[int],
) -> float:
    """
    简单的偏好优化风格损失：
    - 鼓励“短且正确”的回答获得高分；
    - 鼓励“长且正确”的回答获得高于“短但错误”回答的分数。
    这里使用简化的 Bradley-Terry 风格损失。
    """
    model.train()
    optimizer.zero_grad()

    def seq_score(seq: List[int]) -> torch.Tensor:
        """计算序列中每个 token 的 log-softmax 得分之和。"""
        if len(seq) < 2:
            return torch.tensor(0.0, device=DEVICE, requires_grad=True)
        input_ids = torch.tensor([seq[:-1]], dtype=torch.long, device=DEVICE)
        targets = torch.tensor(seq[1:], dtype=torch.long, device=DEVICE)
        logits = model(input_ids)  # (1, seq_len-1, vocab)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.squeeze(0).gather(1, targets.unsqueeze(1)).squeeze(1)
        return token_log_probs.sum()

    short_correct_seq = prompt + short_correct
    long_correct_seq = prompt + long_correct
    short_wrong_seq = prompt + short_wrong

    s_chosen = seq_score(short_correct_seq)
    s_long = seq_score(long_correct_seq)
    s_rejected = seq_score(short_wrong_seq)

    # 损失 1：短正确 > 短错误
    loss1 = -F.logsigmoid(s_chosen - s_rejected)
    # 损失 2：长正确 > 短错误（鼓励难题保留推理）
    loss2 = -F.logsigmoid(s_long - s_rejected)
    loss = (loss1 + loss2) / 2.0

    loss.backward()
    optimizer.step()
    return loss.item()


# ---------------------------- 主程序 / Main ----------------------------
def main():
    # 固定随机种子便于复现
    random.seed(42)
    torch.manual_seed(42)

    # 初始化玩具模型与优化器
    model = ToyReasoner(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 构造一些虚拟的偏好对（短正确、长正确、短错误）
    prompt = [10, 20, 30]
    short_correct = [40, 50, 1]           # 1 为答案终止符
    long_correct = [41, 42, 43, 44, 50, 1]
    short_wrong = [60, 61, 1]

    print("开始偏好优化风格训练 / Starting preference-style training...")
    for step in range(NUM_TRAIN_STEPS):
        loss = preference_loss(model, optimizer, prompt, short_correct, long_correct, short_wrong)
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{NUM_TRAIN_STEPS}, Loss={loss:.4f}")

    # 使用训练后的模型进行 CAT 自适应推理
    cat = CATReasoner(model)

    print("\n=== CAT 自适应推理示例 / CAT Adaptive Reasoning Example ===")
    test_prompts = [
        [10, 20, 30],
        [11, 21, 31],
        [12, 22, 32],
    ]
    for p in test_prompts:
        result = cat.generate(p, min_reason_len=2)
        print(f"Prompt: {p}")
        print(f"  推理长度 / Reasoning length: {len(result.reasoning_tokens)}")
        print(f"  是否提前停止 / Stopped early: {result.stopped_early}")
        print(f"  自信度序列 / Confidence trace: {[f'{c:.3f}' for c in result.confidences]}")
        print(f"  答案 token / Answer token: {result.answer_token}")
        print()


if __name__ == "__main__":
    main()
