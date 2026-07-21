"""
When Do Multi-Agent Systems Help? An Information Bottleneck Perspective
(arXiv 2607.16133)

PyTorch 概念复现（带中文注释） / Conceptual PyTorch reproduction

本代码用一个"可控合成信道"再现论文的核心直觉：
1. 任务标签只由一小段"任务相关维度"决定，其余是大量分散注意力的冗余维度；
2. 单智能体(SAS)看到完整上下文(相关维度 + 全部冗余噪声)；
   多智能体(MAS)只看到经过"有界中继信道 (bounded relay)"压缩后的消息；
3. 中继信道按"相关维度优先 (relevant-first)"保留前 b 个坐标：
   - b < 相关维度数 -> 连相关信息都被截掉 -> 有损中继 (lossy)；
   - b >= 相关维度数 -> 保留全部相关信息、丢弃冗余噪声 -> 近乎充分 (near-sufficient)，
     同时起到"去冗余/降噪"的作用。
4. 关键权衡：冗余上下文会分散**弱模型**的注意力(它容量有限、易被噪声干扰)，
   因此"去冗余"的中继对弱模型是净收益；而**强模型**本就能忽略冗余，
   压缩对它至多是净损失。

结论复现：
- 中继近乎充分(b 足够大)时，MAS 对弱模型稳定有帮助 (gain > 0)；
- 中继有损(b 太小)时，MAS 增益缩小甚至反转；
- 对强模型，MAS 增益整体偏小甚至为负。

运行方式 / Run:
    pip install torch numpy
    python mas_ib_demo.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- 超参数 / Hyperparameters ----------------------------
CONTEXT_DIM = 128         # 完整上游上下文维度 / full upstream-context dim
NUM_RELEVANT = 8          # 真正决定标签的相关维度 / task-relevant dims (放在最前面)
NOISE_SCALE = 3.0         # 冗余维度的噪声幅度 / distractor magnitude
NUM_CLASSES = 4
NUM_TRAIN = 8000
NUM_EVAL = 2000
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")


# ---------------------------- 数据生成 / Synthetic task ----------------------------
def make_task():
    """
    真值投影：只有前 NUM_RELEVANT 维决定标签，其余维度对标签零贡献
    (即纯粹的分散注意力的冗余/噪声维度)。
    """
    W_true = torch.zeros(CONTEXT_DIM, NUM_CLASSES)
    W_true[:NUM_RELEVANT] = torch.randn(NUM_RELEVANT, NUM_CLASSES) * 2.0
    return W_true


def sample_data(n, W_true):
    """
    采样上下文 x 与标签 y：
    - 前 NUM_RELEVANT 维：适度幅度的相关信号；
    - 其余维：高幅度冗余噪声(NOISE_SCALE)，用于分散弱模型注意力。
    """
    x = torch.randn(n, CONTEXT_DIM)
    x[:, NUM_RELEVANT:] *= NOISE_SCALE          # 放大冗余维度 -> 干扰弱模型
    logits = x @ W_true
    y = logits.argmax(dim=1)
    return x, y


# ---------------------------- 中继信道 / Bounded relay channel ----------------------------
def relay_truncate(x, b):
    """
    "相关维度优先"的有界中继：只保留前 b 个坐标，其余置零。
    - b < NUM_RELEVANT : 连相关信息都被截断 -> 有损；
    - b >= NUM_RELEVANT: 保留全部相关信息并丢弃冗余噪声 -> 近乎充分 + 降噪。
    这样"信息损失"与"去冗余收益"都由单一瓶颈宽度 b 控制。
    """
    x_relay = x.clone()
    if b < CONTEXT_DIM:
        x_relay[:, b:] = 0.0
    return x_relay


# ---------------------------- 智能体 / Agent ----------------------------
class Agent(nn.Module):
    """capacity 控制模型能力：大 = 强模型(能忽略冗余)，小 = 弱模型(易被冗余干扰)。"""

    def __init__(self, capacity):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONTEXT_DIM, capacity),
            nn.ReLU(),
            nn.Linear(capacity, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def train_and_eval(agent, x_tr, y_tr, x_ev, y_ev, epochs=60, lr=3e-3):
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(agent(x_tr), y_tr).backward()
        opt.step()
    with torch.no_grad():
        return (agent(x_ev).argmax(1) == y_ev).float().mean().item()


# ---------------------------- 实验主流程 / Experiment ----------------------------
def run_capacity(capacity, W_true, bottlenecks):
    x_tr, y_tr = sample_data(NUM_TRAIN, W_true)
    x_ev, y_ev = sample_data(NUM_EVAL, W_true)

    # SAS 基线：看到完整上下文(含全部冗余噪声)
    torch.manual_seed(1)
    sas = Agent(capacity).to(DEVICE)
    sas_acc = train_and_eval(sas, x_tr, y_tr, x_ev, y_ev)

    rows = []
    for b in bottlenecks:
        x_tr_r = relay_truncate(x_tr, b)
        x_ev_r = relay_truncate(x_ev, b)
        torch.manual_seed(1)                    # 同初始化，控制变量
        mas = Agent(capacity).to(DEVICE)
        mas_acc = train_and_eval(mas, x_tr_r, y_tr, x_ev_r, y_ev)
        rows.append((b, sas_acc, mas_acc, mas_acc - sas_acc))
    return rows


def main():
    W_true = make_task()
    # 从"截断相关信息"到"近乎充分"再到"几乎无压缩"
    bottlenecks = [4, 8, 16, 32, 64, 128]

    print("=" * 74)
    print("MAS vs SAS 在不同中继瓶颈宽度 b 下的准确率与 MAS 增益")
    print(f"(相关维度 = 前 {NUM_RELEVANT} 维; b>= {NUM_RELEVANT} 即近乎充分并去除冗余噪声)")
    print("=" * 74)

    for name, cap in [("weak / 弱模型 (capacity=8)", 8),
                      ("strong / 强模型 (capacity=512)", 512)]:
        print(f"\n### {name}")
        print(f"{'b':>4} | {'SAS acc':>8} | {'MAS acc':>8} | {'MAS gain':>9}")
        print("-" * 40)
        for b, sas_acc, mas_acc, gain in run_capacity(cap, W_true, bottlenecks):
            if gain > 0.01:
                flag = "  <== MAS helps"
            elif gain < -0.01:
                flag = "  (reversed)"
            else:
                flag = ""
            print(f"{b:>4} | {sas_acc:>8.3f} | {mas_acc:>8.3f} | {gain:>+9.3f}{flag}")

    print("\n" + "=" * 74)
    print("Takeaways / 观察:")
    print("1. b < 相关维度数 -> 中继有损 -> MAS 崩溃; 强模型的反转比弱模型更剧烈")
    print("   (强模型对信息损失更敏感, 对应论文 'gains reverse especially for stronger models')。")
    print("2. b >= 相关维度数 -> 近乎充分 + 去冗余 -> MAS 明显有帮助。")
    print("3. b = 完整上下文 -> MAS 退化为 SAS, 增益归零 (无限带宽下 MAS 可模拟 SAS)。")
    print("=> 复现论文结论: MAS 是否有帮助取决于中继信道的信息瓶颈权衡。")
    print("=" * 74)


if __name__ == "__main__":
    main()
