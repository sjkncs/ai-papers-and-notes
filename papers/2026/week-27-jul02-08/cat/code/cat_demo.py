"""
CAT: Confidence-Adaptive Tree Search for Efficient Large Reasoning Models
PyTorch 概念复现（带中文注释）

本代码演示 CAT 的核心机制，不依赖真实大模型，而是使用一个随机 Transformer
作为“推理模型”来生成 token，并通过 softmax 熵/最大概率估计步骤置信度，
进而实现置信度自适应的树搜索、 early-stop 与回退。

运行方式：
    pip install torch
    python cat_demo.py
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- 超参数 / Hyperparameters ----------------------------
VOCAB_SIZE = 64           # 词汇表大小（随机模型较小，便于演示）
EMBED_DIM = 64            # 词嵌入维度
NUM_HEADS = 4             # Transformer 注意力头数
NUM_LAYERS = 3            # Transformer 层数
MAX_LEN = 24              # 最大序列长度
BRANCH_HIGH = 2           # 高置信度节点的分支数（少）
BRANCH_LOW = 5            # 低置信度节点的分支数（多）
CONF_HIGH = 0.35          # 高置信度阈值（针对随机模型的低峰值概率调整）
CONF_LOW = 0.20           # 低置信度阈值（低于则回退）
TOP_K = 8                 # 采样候选数量
TEMPERATURE = 0.6         # 温度：小于 1 可锐化分布
LOGIT_SCALE = 5.0         # 输出 logit 放大倍数：随机模型默认分布太平，放大后更似真实模型
DEVICE = torch.device("cpu")


# ---------------------------- 工具函数 / Utilities ----------------------------
def entropy_from_logits(logits: torch.Tensor) -> float:
    """从 logits 计算香农熵；熵越低通常越确定。"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs).item()


def confidence_from_logits(logits: torch.Tensor, temperature: float = TEMPERATURE) -> float:
    """
    简单置信度估计：经过温度缩放后的最大 softmax 概率。
    在真实 LRM 中可替换为 PRM/ORM/内部状态估计器。
    """
    probs = F.softmax(logits / temperature, dim=-1)
    return probs.max().item()


def stop_token(token_id: int) -> bool:
    """终止符：遇到特殊 token 则停止当前路径。"""
    return token_id in {0, 1}  # 0: <eos>, 1: <eop>


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
        """返回最后一个位置的 logits（已放大以模拟真实模型的峰值分布）。"""
        x = self.embedding(tokens)
        mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1)).to(DEVICE)
        h = self.transformer(x, mask=mask, is_causal=True)
        return self.head(h[:, -1, :]) * LOGIT_SCALE


# ---------------------------- 树搜索节点 / Tree Node ----------------------------
@dataclass
class ReasonNode:
    """推理树上的一个节点，记录当前 token 序列、父节点、置信度与访问状态。"""
    tokens: List[int]
    parent: Optional["ReasonNode"] = None
    children: List["ReasonNode"] = field(default_factory=list)
    confidence: float = 0.0          # 到达该节点最后一步的置信度
    is_terminal: bool = False
    depth: int = 0

    def path(self) -> List[int]:
        """从根节点到当前节点的完整 token 序列。"""
        if self.parent is None:
            return list(self.tokens)
        return self.parent.path() + self.tokens


# ---------------------------- CAT 搜索器 / CAT Searcher ----------------------------
class CATSearcher:
    """
    置信度自适应树搜索器。

    核心逻辑：
    1. 对当前节点调用模型，得到下一个 token 的 logits；
    2. 估计该步骤的置信度；
    3. 根据置信度决定分支数、是否 early-stop、是否回退；
    4. 重复直到找到满足终止条件的路径或预算耗尽。
    """
    def __init__(self, model: ToyReasoner):
        self.model = model
        self.model.eval()

    def expand(self, node: ReasonNode, budget: int) -> Tuple[List[ReasonNode], int]:
        """
        扩展一个节点，返回子节点列表与剩余预算。
        """
        if budget <= 0 or node.is_terminal:
            return [], budget

        input_ids = torch.tensor([node.path()[-MAX_LEN:]], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            logits = self.model(input_ids)

        conf = confidence_from_logits(logits)
        node.confidence = conf

        # 置信度极低：触发回退，向父节点返回空扩展
        if conf < CONF_LOW and node.parent is not None:
            print(f"  [回退] depth={node.depth}, conf={conf:.3f} < {CONF_LOW}")
            return [], budget

        # 决定分支数：高置信度少分支，低置信度多分支
        if conf >= CONF_HIGH:
            n_branch = BRANCH_HIGH
        else:
            n_branch = BRANCH_LOW

        # 从 top-k 中采样/选择下一个 token，并计算每个候选的概率作为子节点置信度
        scaled_logits = logits / TEMPERATURE
        probs = F.softmax(scaled_logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, k=TOP_K, dim=-1)
        selected = torch.topk(topk_probs, k=min(n_branch, TOP_K)).indices.squeeze(0)

        children = []
        for idx in selected.tolist():
            next_token = topk_ids[0, idx].item()
            token_conf = topk_probs[0, idx].item()
            child = ReasonNode(
                tokens=[next_token],
                parent=node,
                confidence=token_conf,
                depth=node.depth + 1,
                is_terminal=stop_token(next_token) or node.depth + 1 >= MAX_LEN,
            )
            children.append(child)

        node.children.extend(children)
        budget -= 1  # 每次扩展消耗一个模型调用预算
        return children, budget

    def search(self, prompt: List[int], max_calls: int = 50) -> ReasonNode:
        """
        从 prompt 开始执行置信度自适应树搜索。
        返回最优终节点（这里简化为置信度最高且为终节点的路径）。
        """
        root = ReasonNode(tokens=prompt, depth=0)
        budget = max_calls
        frontier: List[ReasonNode] = [root]
        completed: List[ReasonNode] = []
        expand_count: dict[int, int] = {id(root): 0}

        print(f"开始 CAT 搜索，初始 prompt={prompt}，预算={max_calls}")

        while frontier and budget > 0:
            # 优先扩展最新节点（深度优先），也可改为按置信度排序
            node = frontier.pop()

            # 限制同一节点重复扩展次数，避免根节点被无限回退后反复展开
            node_id = id(node)
            expand_count[node_id] = expand_count.get(node_id, 0) + 1
            if expand_count[node_id] > 3:
                continue

            children, budget = self.expand(node, budget)

            if not children and node.parent is not None and node.confidence < CONF_LOW:
                # 回退：将父节点重新加入前沿，但避免无限循环
                if node.parent not in frontier:
                    frontier.append(node.parent)
                continue

            for child in children:
                if child.is_terminal:
                    completed.append(child)
                else:
                    frontier.append(child)

            print(
                f"  扩展 depth={node.depth}, conf={node.confidence:.3f}, "
                f"生成分支数={len(children)}, 剩余预算={budget}, 已完成路径={len(completed)}"
            )

        if not completed:
            # 若无完成路径，返回当前最深节点
            return max(frontier, key=lambda n: n.depth, default=root)

        # 选择平均置信度最高的完成路径
        def path_avg_conf(n: ReasonNode) -> float:
            confs = []
            cur: Optional[ReasonNode] = n
            while cur is not None:
                confs.append(cur.confidence)
                cur = cur.parent
            return sum(confs) / max(len(confs), 1)

        best = max(completed, key=path_avg_conf)
        return best


# ---------------------------- 主程序 / Main ----------------------------
def main():
    # 固定随机种子便于复现
    random.seed(42)
    torch.manual_seed(42)

    # 初始化玩具模型
    model = ToyReasoner(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS).to(DEVICE)

    # prompt：用 token [10, 20, 30] 作为问题开始
    prompt = [10, 20, 30]

    searcher = CATSearcher(model)
    best_node = searcher.search(prompt, max_calls=30)

    full_path = best_node.path()
    print("\n=== 最优路径 / Best Path ===")
    print(full_path)
    print(f"路径长度: {len(full_path)}, 最终置信度: {best_node.confidence:.3f}")

    # 可视化：打印搜索过程中产生的树结构（仅展示前两层）
    print("\n=== 树结构预览 / Tree Preview (depth <= 2) ===")

    def print_tree(node: ReasonNode, indent: int = 0, max_depth: int = 2):
        if node.depth > max_depth:
            return
        token_str = str(node.tokens[-1]) if node.tokens else "ROOT"
        marker = "[T]" if node.is_terminal else ""
        print("  " * indent + f"-{token_str} d={node.depth} conf={node.confidence:.3f} {marker}")
        for child in node.children:
            print_tree(child, indent + 1, max_depth)

    # 回溯到根节点再打印整棵树
    root_node = best_node
    while root_node.parent is not None:
        root_node = root_node.parent
    print_tree(root_node)


if __name__ == "__main__":
    main()
