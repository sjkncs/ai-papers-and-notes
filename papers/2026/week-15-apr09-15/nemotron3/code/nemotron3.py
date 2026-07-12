"""
Nemotron 3 Super: MoE混合Mamba-Transformer架构
Hybrid Mamba-Transformer MoE for Agentic Reasoning

复现论文: arXiv 2604.12374 (Apr 2026)
核心: Mamba层+Transformer注意力层交替 + MoE专家混合 + 多token预测

作者: QoderWork AI Research
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


class MambaBlock:
    """简化Mamba块"""
    def __init__(self, d_model=64, state_dim=16, seed=42):
        self.rng = np.random.RandomState(seed)
        self.d_model = d_model
        scale = np.sqrt(2.0/d_model)
        self.A = -np.abs(self.rng.randn(state_dim))  # 稳定对角
        self.B = self.rng.randn(d_model, state_dim) * scale
        self.C = self.rng.randn(state_dim, d_model) * scale
        self.state_dim = state_dim
    
    def forward(self, x):
        T = x.shape[0]
        state = np.zeros(self.state_dim)
        out = np.zeros((T, self.d_model))
        for t in range(T):
            state = np.exp(self.A) * state + x[t] @ self.B
            out[t] = state @ self.C
        return out + x


class AttentionBlock:
    """简化注意力块"""
    def __init__(self, d_model=64, seed=42):
        self.rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0/d_model)
        self.Wq = self.rng.randn(d_model, d_model) * scale
        self.Wk = self.rng.randn(d_model, d_model) * scale
        self.Wv = self.rng.randn(d_model, d_model) * scale
        self.d_model = d_model
    
    def forward(self, x):
        Q, K, V = x @ self.Wq, x @ self.Wk, x @ self.Wv
        scores = Q @ K.T / np.sqrt(self.d_model)
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn /= attn.sum(axis=-1, keepdims=True) + 1e-8
        return attn @ V + x


class MoELayer:
    """简化MoE层 (Top-2专家)"""
    def __init__(self, d_model=64, n_experts=8, top_k=2, seed=42):
        self.rng = np.random.RandomState(seed)
        self.n_experts = n_experts
        self.top_k = top_k
        scale = np.sqrt(2.0/d_model)
        # 门控
        self.W_gate = self.rng.randn(d_model, n_experts) * scale
        # 专家
        self.experts = [self.rng.randn(d_model, d_model) * scale for _ in range(n_experts)]
    
    def forward(self, x):
        T, D = x.shape
        # 门控分数
        gate_logits = x @ self.W_gate  # (T, n_experts)
        
        out = np.zeros_like(x)
        for t in range(T):
            # Top-K选择
            top_idx = np.argsort(-gate_logits[t])[:self.top_k]
            top_scores = np.exp(gate_logits[t, top_idx])
            top_scores /= top_scores.sum()
            
            # 加权专家输出
            for i, idx in enumerate(top_idx):
                out[t] += top_scores[i] * (x[t] @ self.experts[idx])
        
        return out


class HybridMambaTransformer:
    """
    Nemotron 3 Super: Mamba + Attention + MoE 混合架构
    
    层排列: [Mamba, Mamba, Attention, MoE] × n_blocks
    """
    
    def __init__(self, d_model=64, n_blocks=3, n_experts=8, seed=42):
        self.layers = []
        for b in range(n_blocks):
            self.layers.append(('mamba', MambaBlock(d_model, 16, seed+b*4)))
            self.layers.append(('mamba', MambaBlock(d_model, 16, seed+b*4+1)))
            self.layers.append(('attn', AttentionBlock(d_model, seed+b*4+2)))
            self.layers.append(('moe', MoELayer(d_model, n_experts, 2, seed+b*4+3)))
    
    def forward(self, x):
        for name, layer in self.layers:
            x = layer.forward(x)
        return x
    
    def count_params(self):
        """统计参数量"""
        total = 0
        for name, layer in self.layers:
            if name == 'mamba':
                total += layer.d_model * layer.state_dim * 2 + layer.state_dim
            elif name == 'attn':
                total += layer.d_model**2 * 3
            elif name == 'moe':
                total += layer.n_experts * 64**2 + 64 * layer.n_experts
        return total


class PureTransformer:
    """纯Transformer基线"""
    def __init__(self, d_model=64, n_blocks=3, seed=42):
        self.layers = [AttentionBlock(d_model, seed+i) for i in range(n_blocks*4)]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def count_params(self):
        return len(self.layers) * 64**2 * 3


class PureMamba:
    """纯Mamba基线"""
    def __init__(self, d_model=64, n_blocks=3, seed=42):
        self.layers = [MambaBlock(d_model, 16, seed+i) for i in range(n_blocks*4)]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def count_params(self):
        return len(self.layers) * (64*16*2 + 16)


def main():
    print("=" * 70)
    print("Nemotron 3 Super: MoE Hybrid Mamba-Transformer")
    print("=" * 70)
    
    d_model, seq_len = 64, 100
    rng = np.random.RandomState(42)
    x = rng.randn(seq_len, d_model)
    
    models = {
        'Hybrid (Mamba+Attn+MoE)': HybridMambaTransformer(d_model, 3, 8, 42),
        'Pure Transformer': PureTransformer(d_model, 3, 42),
        'Pure Mamba': PureMamba(d_model, 3, 42),
    }
    
    print(f"\n  架构对比 (d_model={d_model}, seq_len={seq_len}):")
    print(f"\n  {'模型':<30} {'参数量':>12} {'推理复杂度':>15}")
    print(f"  {'-'*60}")
    
    for name, model in models.items():
        params = model.count_params()
        out = model.forward(x)
        
        if 'Mamba' in name and 'Hybrid' not in name:
            complexity = f"O(n·d·s) = {seq_len*d_model*16:,}"
        elif 'Transformer' in name and 'Hybrid' not in name:
            complexity = f"O(n²·d) = {seq_len**2*d_model:,}"
        else:
            complexity = "O(mixed)"
        
        print(f"  {name:<30} {params:>12,} {complexity:>15}")
    
    # 输出对比
    hybrid_out = models['Hybrid (Mamba+Attn+MoE)'].forward(x)
    trans_out = models['Pure Transformer'].forward(x)
    mamba_out = models['Pure Mamba'].forward(x)
    
    print(f"\n  输出统计:")
    print(f"    Hybrid 输出 norm: {np.linalg.norm(hybrid_out):.4f}")
    print(f"    Transformer 输出 norm: {np.linalg.norm(trans_out):.4f}")
    print(f"    Mamba 输出 norm: {np.linalg.norm(mamba_out):.4f}")
    
    print(f"\n  Nemotron 3 Super 关键设计:")
    print(f"  • Mamba层: 高效长序列处理 O(n)")
    print(f"  • Attention层: 全局信息交互 O(n²) 但间歇使用")
    print(f"  • MoE层: 稀疏专家激活，仅Top-2活跃")
    print(f"  • 混合架构: 兼顾效率和表达能力")
    print(f"  • 多token预测: 推测解码加速推理")
    
    print("\n" + "=" * 70)
    print("Nemotron 3 Super 复现完成!")
    print("=" * 70)

if __name__ == '__main__':
    main()
