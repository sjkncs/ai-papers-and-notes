"""
Exploratory Sampling (ESamp) 简化复现 / Simplified Reproduction
=========================================================
论文: "Large Language Models Explore by Latent Distilling"
会议: ICML 2026
作者: Yuanhao Zeng, Ao Lu, Lufei Li, Zheng Zhang, Yexin Li, Kan Ren

本代码演示 ESamp 的核心思想:
1. 在 LLM 生成过程中，同时提取浅层和深层的隐藏表示
2. 训练一个轻量级 Latent Distiller（2层MLP）来预测深层表示
3. 用预测误差作为新颖性信号，重新加权候选 token 的 logits
4. 引导采样向语义上未探索的区域倾斜

运行方式: python esamp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


# ------------------------------
# 1. 轻量级 Latent Distiller
# ------------------------------
class LatentDistiller(nn.Module):
    """
    轻量级 MLP，将浅层隐藏表示映射到深层隐藏表示的预测。
    论文使用 2层 MLP，hidden_dim=384。
    """
    def __init__(self, hidden_dim: int = 512, bottleneck_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, hidden_dim),
        )

    def forward(self, shallow_hidden: torch.Tensor) -> torch.Tensor:
        """
        shallow_hidden: [batch, seq_len, hidden_dim]
        返回: predicted_deep: [batch, seq_len, hidden_dim]
        """
        return self.net(shallow_hidden)


# ------------------------------
# 2. 模拟 LLM 的简化 Transformer
# ------------------------------
class SimpleLLM(nn.Module):
    """
    简化的 Transformer 解码器，用于演示 ESamp。
    包含多个 Transformer 层，可以提取中间层的隐藏表示。
    """
    def __init__(self, vocab_size: int = 50000, hidden_dim: int = 512,
                 num_layers: int = 6, num_heads: int = 8, max_len: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)

        # Transformer 层
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, return_all_hidden: bool = False):
        """
        input_ids: [batch, seq_len]
        返回:
            logits: [batch, seq_len, vocab_size]
            hidden_states: 如果 return_all_hidden=True，返回所有层的隐藏状态列表
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        hidden_states = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if return_all_hidden:
                hidden_states.append(x.clone())

        x = self.ln_final(x)
        logits = self.lm_head(x)

        if return_all_hidden:
            return logits, hidden_states
        return logits


# ------------------------------
# 3. ESamp 解码器 / ESamp Decoder
# ------------------------------
class ESampDecoder:
    """
    Exploratory Sampling 解码器。
    在生成每个 token 时，利用 Distiller 的预测误差来重加权 logits。
    """
    def __init__(self, llm: SimpleLLM, distiller: LatentDistiller,
                 shallow_layer_idx: int = 2, deep_layer_idx: int = 5,
                 beta: float = 0.25, top_k: int = 50):
        self.llm = llm
        self.distiller = distiller
        self.shallow_layer_idx = shallow_layer_idx
        self.deep_layer_idx = deep_layer_idx
        self.beta = beta  # novelty weight，论文最优值 0.25
        self.top_k = top_k
        self.distiller_optimizer = torch.optim.Adam(distiller.parameters(), lr=1e-4)

    def compute_novelty_signal(self, shallow_hidden: torch.Tensor,
                                deep_hidden: torch.Tensor) -> torch.Tensor:
        """
        计算新颖性信号 = Distiller 预测误差的大小。
        shallow_hidden: [batch, seq_len, hidden_dim]
        deep_hidden: [batch, seq_len, hidden_dim]
        返回: novelty: [batch, seq_len, 1]
        """
        with torch.no_grad():
            predicted_deep = self.distiller(shallow_hidden)
        # 预测误差作为新颖性指标
        error = torch.norm(deep_hidden - predicted_deep, dim=-1, keepdim=True)
        return error

    def update_distiller(self, shallow_hidden: torch.Tensor,
                         deep_hidden: torch.Tensor):
        """
        在线更新 Distiller: 最小化预测误差 (MSE)。
        """
        self.distiller_optimizer.zero_grad()
        predicted = self.distiller(shallow_hidden.detach())
        loss = F.mse_loss(predicted, deep_hidden.detach())
        loss.backward()
        self.distiller_optimizer.step()
        return loss.item()

    def sample_next_token(self, input_ids: torch.Tensor,
                          temperature: float = 1.0) -> Tuple[int, dict]:
        """
        生成下一个 token，使用 ESamp 重加权。
        返回: (next_token_id, stats)
        """
        self.llm.eval()
        self.distiller.eval()

        with torch.no_grad():
            # 前向传播，获取所有层的隐藏状态
            logits, hidden_states = self.llm(input_ids, return_all_hidden=True)
            # 只取最后一个位置的 logits 和 hidden states
            next_logits = logits[:, -1, :]  # [batch, vocab_size]

            # 提取浅层和深层表示
            shallow_hidden = hidden_states[self.shallow_layer_idx][:, -1:, :]  # [batch, 1, hidden_dim]
            deep_hidden = hidden_states[self.deep_layer_idx][:, -1:, :]        # [batch, 1, hidden_dim]

            # 计算新颖性信号
            novelty = self.compute_novelty_signal(shallow_hidden, deep_hidden)  # [batch, 1, 1]
            novelty = novelty.squeeze(-1).squeeze(-1)  # [batch]

            # --- ESamp 核心: 用新颖性重加权 logits ---
            # 高新颖性 -> 增加对应 token 的 logit
            # 由于 novelty 是标量，我们将其作为全局温度调节因子
            # 更精细的实现可以对每个候选 token 计算独立的新颖性，但这里简化为全局调节
            adjusted_logits = next_logits + self.beta * novelty.unsqueeze(-1) * torch.randn_like(next_logits) * 0.1
            # 注意: 上述简化版本用噪声注入模拟探索
            # 实际论文中对每个候选 extension 计算独立 novelty
            # ------------------------------------------

            # Top-K 采样
            probs = F.softmax(adjusted_logits / temperature, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
            next_token = torch.multinomial(top_k_probs, num_samples=1)
            next_token_id = top_k_indices.gather(-1, next_token).item()

        stats = {
            'novelty': novelty.item(),
            'top_k_entropy': -(top_k_probs * torch.log(top_k_probs + 1e-10)).sum().item(),
        }
        return next_token_id, stats

    def generate(self, prompt_ids: List[int], max_new_tokens: int = 20,
                 temperature: float = 1.0, update_distiller_every: int = 4) -> Tuple[List[int], List[dict]]:
        """
        自回归生成序列。
        """
        generated = prompt_ids.copy()
        stats_history = []
        input_tensor = torch.tensor([generated], dtype=torch.long)

        for step in range(max_new_tokens):
            next_token_id, stats = self.sample_next_token(input_tensor, temperature)
            generated.append(next_token_id)
            stats_history.append(stats)
            input_tensor = torch.tensor([generated], dtype=torch.long)

            # 定期更新 Distiller（在线学习）
            if (step + 1) % update_distiller_every == 0:
                with torch.no_grad():
                    _, hidden_states = self.llm(input_tensor, return_all_hidden=True)
                    shallow = hidden_states[self.shallow_layer_idx][:, :-1, :]
                    deep = hidden_states[self.deep_layer_idx][:, :-1, :]
                loss_val = self.update_distiller(shallow, deep)
                stats_history[-1]['distiller_loss'] = loss_val

            # 遇到 EOS 停止 (假设 EOS token id = 2)
            if next_token_id == 2:
                break

        return generated, stats_history


# ------------------------------
# 4. 对比实验: Vanilla vs. ESamp
# ------------------------------
def compare_vanilla_vs_esamp():
    """
    对比标准采样与 ESamp 的生成行为。
    使用简化的合成任务：从一个有限的状态空间中采样，
    观察 ESamp 是否能更均匀地覆盖整个空间。
    """
    print("=" * 60)
    print("Exploratory Sampling (ESamp) 简化复现")
    print("Simplified Reproduction of Exploratory Sampling")
    print("=" * 60)

    device = torch.device('cpu')
    vocab_size = 100  # 小规模词汇表便于分析覆盖度
    hidden_dim = 256
    num_layers = 6

    # 初始化模型
    llm = SimpleLLM(vocab_size=vocab_size, hidden_dim=hidden_dim,
                    num_layers=num_layers, num_heads=8).to(device)
    distiller = LatentDistiller(hidden_dim=hidden_dim, bottleneck_dim=192).to(device)
    esamp_decoder = ESampDecoder(llm, distiller, shallow_layer_idx=2,
                                  deep_layer_idx=5, beta=0.25, top_k=20)

    prompt = [1, 5, 10]  # 模拟 prompt token ids
    num_samples = 100
    max_len = 15

    # --- Vanilla 采样 ---
    print("\n【Vanilla Sampling】生成 100 条序列...")
    vanilla_sequences = []
    for _ in range(num_samples):
        seq = prompt.copy()
        input_t = torch.tensor([seq], dtype=torch.long)
        for _ in range(max_len):
            with torch.no_grad():
                logits = llm(input_t)
                next_logit = logits[:, -1, :] / 1.0
                probs = F.softmax(next_logit, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            seq.append(next_id)
            input_t = torch.tensor([seq], dtype=torch.long)
            if next_id == 2:
                break
        vanilla_sequences.append(seq)

    # --- ESamp 采样 ---
    print("【ESamp Sampling】生成 100 条序列...")
    esamp_sequences = []
    for _ in range(num_samples):
        seq, _ = esamp_decoder.generate(prompt, max_new_tokens=max_len, temperature=1.0)
        esamp_sequences.append(seq)

    # --- 分析覆盖度 / Coverage Analysis ---
    def compute_coverage(sequences, vocab_size):
        """计算生成序列中 unique token 的覆盖比例。"""
        all_tokens = set()
        for seq in sequences:
            all_tokens.update(seq)
        return len(all_tokens) / vocab_size

    def compute_avg_unique_per_seq(sequences):
        """计算每条序列的平均 unique token 数。"""
        return sum(len(set(seq)) for seq in sequences) / len(sequences)

    vanilla_coverage = compute_coverage(vanilla_sequences, vocab_size)
    esamp_coverage = compute_coverage(esamp_sequences, vocab_size)
    vanilla_unique = compute_avg_unique_per_seq(vanilla_sequences)
    esamp_unique = compute_avg_unique_per_seq(esamp_sequences)

    print("\n【覆盖度分析 / Coverage Analysis】")
    print(f"  Vanilla Coverage: {vanilla_coverage:.2%}")
    print(f"  ESamp Coverage:   {esamp_coverage:.2%}")
    print(f"  Vanilla Avg Unique/Seq: {vanilla_unique:.2f}")
    print(f"  ESamp Avg Unique/Seq:   {esamp_unique:.2f}")

    if esamp_coverage > vanilla_coverage:
        improvement = (esamp_coverage - vanilla_coverage) / vanilla_coverage * 100
        print(f"  覆盖度提升 / Coverage Improvement: +{improvement:.1f}%")
    print()


def demo_distiller_training():
    """
    演示 Distiller 的在线训练过程。
    """
    print("=" * 60)
    print("【Distiller 在线训练演示 / Distiller Online Training Demo】")
    print("=" * 60)

    hidden_dim = 256
    distiller = LatentDistiller(hidden_dim=hidden_dim, bottleneck_dim=192)
    optimizer = torch.optim.Adam(distiller.parameters(), lr=1e-3)

    # 模拟数据: 浅层表示 -> 深层表示
    # 设定一个非线性映射关系
    torch.manual_seed(42)
    for epoch in range(10):
        shallow = torch.randn(32, 10, hidden_dim)
        # 模拟深层表示 = shallow 的某种非线性变换
        true_deep = torch.sin(shallow) + 0.5 * shallow ** 2

        optimizer.zero_grad()
        pred_deep = distiller(shallow)
        loss = F.mse_loss(pred_deep, true_deep)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1:02d}, MSE Loss: {loss.item():.6f}")
    print("Distiller 训练完成 / Distiller training complete.\n")


def demo_novelty_signal():
    """
    可视化演示: 预测误差如何作为新颖性信号。
    """
    print("=" * 60)
    print("【新颖性信号演示 / Novelty Signal Demo】")
    print("=" * 60)

    hidden_dim = 128
    distiller = LatentDistiller(hidden_dim=hidden_dim, bottleneck_dim=96)

    # 区域 A: "已探索"区域（与训练数据相似）
    # 区域 B: "未探索"区域（远离训练数据分布）
    torch.manual_seed(0)
    with torch.no_grad():
        # 模拟已探索区域（浅层表示来自标准高斯）
        explored_shallow = torch.randn(5, 1, hidden_dim) * 0.5
        explored_deep = torch.sin(explored_shallow)  # 假设真实映射
        explored_pred = distiller(explored_shallow)
        explored_error = torch.norm(explored_deep - explored_pred, dim=-1).squeeze()

        # 模拟未探索区域（浅层表示来自不同分布）
        unexplored_shallow = torch.randn(5, 1, hidden_dim) * 2.0 + 5.0
        unexplored_deep = torch.sin(unexplored_shallow)
        unexplored_pred = distiller(unexplored_shallow)
        unexplored_error = torch.norm(unexplored_deep - unexplored_pred, dim=-1).squeeze()

    print(f"  已探索区域平均误差 / Explored region avg error: {explored_error.mean().item():.4f}")
    print(f"  未探索区域平均误差 / Unexplored region avg error: {unexplored_error.mean().item():.4f}")
    if unexplored_error.mean() > explored_error.mean():
        print("  结果: 未探索区域的预测误差显著更高 -> 可作为新颖性信号")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Exploratory Sampling (ESamp) 简化复现")
    print("Simplified Reproduction of Exploratory Sampling")
    print("论文: Large Language Models Explore by Latent Distilling (ICML 2026)")
    print("=" * 60)

    # 演示1: Distiller 在线训练
    demo_distiller_training()

    # 演示2: 新颖性信号机制
    demo_novelty_signal()

    # 演示3: Vanilla vs. ESamp 覆盖度对比
    compare_vanilla_vs_esamp()

    print("全部演示完成 / All demos completed.")
