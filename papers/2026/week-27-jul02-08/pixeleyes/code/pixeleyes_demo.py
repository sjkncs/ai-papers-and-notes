"""
PixelEyes: Decoupling Perception and Reasoning for Pinpoint Visual Evidence Seeking
PyTorch 概念复现（带中文注释）

本代码演示 PixelEyes 的核心思想：
1. 把多轮视觉推理中的“推理器”与“感知器”解耦：
   - 推理器（Reasoner）根据问题生成“目标语义向量”；
   - 感知器（Perception Tool）用该向量在图像区域上做相似度排序，
     并通过 Mask 引导视觉搜索 + 语义区域 BFS 定位目标。
2. 用一个小型分类任务验证：给定问题，模型能否在 4x4 语义区域网格中
   找到正确区域，并生成对应 Mask。

运行方式：
    pip install torch
    python pixeleyes_demo.py
"""

import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- 超参数 / Hyperparameters ----------------------------
GRID_H = 4               # 图像区域网格高度
GRID_W = 4               # 图像区域网格宽度
NUM_REGIONS = GRID_H * GRID_W
REGION_DIM = 16          # 每个区域的视觉特征维度
QUESTION_DIM = 8         # 问题嵌入维度
NUM_QUESTIONS = 20       # 模拟问题数量
HIDDEN_DIM = 32
LEARNING_RATE = 2e-3
NUM_EPOCHS = 8
BATCH_SIZE = 32
DEVICE = torch.device("cpu")


# ---------------------------- 数据集 / Synthetic Dataset ----------------------------
class VisualSearchDataset:
    """
    模拟视觉搜索数据：
    - 每个问题对应一个固定的目标区域（由问题 id 决定）；
    - 图像区域特征是随机的，但目标区域特征会与问题嵌入更相关。
    """
    def __init__(self, num_samples: int = 2000):
        self.num_samples = num_samples
        self.question_embed = nn.Embedding(NUM_QUESTIONS, QUESTION_DIM)
        self.region_embed = nn.Embedding(NUM_REGIONS, REGION_DIM)
        # 每个问题对应一个区域偏好向量，用于把目标区域从随机特征中区分开
        self.question_region_bias = nn.Parameter(torch.randn(NUM_QUESTIONS, REGION_DIM) * 0.5)
        # 问题→目标区域的确定性映射
        self.q2target = {q: q % NUM_REGIONS for q in range(NUM_QUESTIONS)}

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_id = random.randrange(NUM_QUESTIONS)
        target = self.q2target[q_id]
        # 图像特征：目标区域叠加问题相关的区域偏好，其他区域为随机噪声
        regions = self.region_embed(torch.arange(NUM_REGIONS))
        regions = regions + 0.1 * torch.randn_like(regions)
        regions[target] = regions[target] + self.question_region_bias[q_id]
        q_emb = self.question_embed.weight[q_id]
        return q_emb, regions, torch.tensor(target, dtype=torch.long)

    def __len__(self):
        return self.num_samples


# ---------------------------- 推理器 / Reasoner ----------------------------
class Reasoner(nn.Module):
    """根据问题嵌入生成目标语义向量（决定“找什么”）。"""
    def __init__(self, question_dim: int, hidden_dim: int, target_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(question_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, question: torch.Tensor) -> torch.Tensor:
        return self.net(question)


# ---------------------------- 感知器 / Perception Tool ----------------------------
class PerceptionTool(nn.Module):
    """
    用目标语义向量对图像区域打分，并执行语义区域 BFS 扩展。
    输出：每个区域的 logits（用于分类）和一个软 Mask。
    """
    def __init__(self, region_dim: int, target_dim: int, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(region_dim + target_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def region_logits(self, regions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            regions: (B, N, region_dim)
            target:  (B, target_dim)
        Returns:
            logits: (B, N)
        """
        B, N, _ = regions.shape
        target = target.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([regions, target], dim=-1)
        return self.scorer(x).squeeze(-1)

    def bfs_mask(self, scores: torch.Tensor, start_k: int = 3) -> torch.Tensor:
        """
        简化的语义区域 BFS：从得分最高的 start_k 个区域出发，
        把相邻且得分高于均值的区域加入 Mask（模拟逐步扩展搜索范围）。
        Args:
            scores: (B, N) 每个区域的分数
        Returns:
            mask: (B, N) 软 Mask（0/1）
        """
        B, N = scores.shape
        device = scores.device
        mask = torch.zeros(B, N, device=device)
        # 选择 top-k 起始区域
        _, topk_idx = torch.topk(scores, k=start_k, dim=-1)
        mask.scatter_(1, topk_idx, 1.0)

        # 按网格邻接关系扩展
        mean_score = scores.mean(dim=-1, keepdim=True)
        h, w = GRID_H, GRID_W
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                # 如果当前区域已被选中，检查四邻域
                if mask[:, idx].sum() > 0:
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            nidx = ni * w + nj
                            neighbor_high = scores[:, nidx] > mean_score.squeeze(-1)
                            mask[:, nidx] = torch.where(neighbor_high, torch.ones_like(mask[:, nidx]), mask[:, nidx])
        return mask


# ---------------------------- 完整 PixelEyes 模型 / Full Model ----------------------------
class PixelEyes(nn.Module):
    def __init__(self):
        super().__init__()
        self.reasoner = Reasoner(QUESTION_DIM, HIDDEN_DIM, REGION_DIM)
        self.perception = PerceptionTool(REGION_DIM, REGION_DIM, HIDDEN_DIM)

    def forward(self, question: torch.Tensor, regions: torch.Tensor):
        target_vec = self.reasoner(question)
        logits = self.perception.region_logits(regions, target_vec)
        mask = self.perception.bfs_mask(logits)
        return logits, mask


# ---------------------------- 训练 / Training ----------------------------
def train(model: nn.Module, dataset: VisualSearchDataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for _ in range(len(dataset) // BATCH_SIZE):
            batch = [dataset[i] for i in range(BATCH_SIZE)]
            questions = torch.stack([b[0] for b in batch])
            regions = torch.stack([b[1] for b in batch])
            targets = torch.stack([b[2] for b in batch])

            questions, regions, targets = questions.to(DEVICE), regions.to(DEVICE), targets.to(DEVICE)
            logits, mask = model(questions, regions)

            # 主损失：目标区域分类
            cls_loss = F.cross_entropy(logits, targets)
            # 辅助损失：Mask 应覆盖目标区域
            mask_target = F.one_hot(targets, num_classes=NUM_REGIONS).float()
            mask_loss = F.binary_cross_entropy(mask, mask_target)
            loss = cls_loss + 0.2 * mask_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * BATCH_SIZE
            pred = logits.argmax(dim=-1)
            correct += (pred == targets).sum().item()
            total += BATCH_SIZE
        print(f"  Epoch {epoch}/{NUM_EPOCHS}, Loss={total_loss / total:.4f}, Acc={correct / total:.3%}")


# ---------------------------- 主程序 / Main ----------------------------
def main():
    random.seed(0)
    torch.manual_seed(0)

    dataset = VisualSearchDataset(num_samples=2000)
    model = PixelEyes().to(DEVICE)

    print("训练 PixelEyes 概念模型 / Training PixelEyes toy model...")
    train(model, dataset)

    model.eval()
    print("\n=== 推理示例 / Inference Example ===")
    q_emb, regions, target = dataset[0]
    with torch.no_grad():
        logits, mask = model(q_emb.unsqueeze(0).to(DEVICE), regions.unsqueeze(0).to(DEVICE))
        pred = logits.argmax(dim=-1).item()
        print(f"真实目标区域 / Ground-truth region: {target.item()}")
        print(f"预测目标区域 / Predicted region: {pred}")
        print(f"BFS Mask 覆盖目标 / Mask covers target: {mask[0, target].item():.2f}")


if __name__ == "__main__":
    main()
