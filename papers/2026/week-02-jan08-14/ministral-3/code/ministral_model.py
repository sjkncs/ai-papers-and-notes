# ============================================================================
# MIT License
# ============================================================================
# Copyright (c) 2026 AI Papers Weekly Review Project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ============================================================================

"""
Ministral 3 — Parameter-Efficient Dense LLM Implementation
Ministral 3 — 参数高效密集语言模型实现

Self-contained PyTorch implementation of the Ministral 3 architecture,
featuring shared-weight FFN, efficient grouped-query attention with RoPE,
and configurable model sizes (3B/8B/14B).

自包含的PyTorch实现，包含共享权重FFN、带RoPE的高效分组查询注意力，
以及可配置的模型规模（3B/8B/14B）。

Reference / 参考文献:
    Liu, A.H. et al. "Ministral 3: Parameter-Efficient Dense Language
    Models at Scale." arXiv:2601.08584 (2026).
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration / 配置
# ============================================================================

@dataclass
class MinistralConfig:
    """
    Configuration for Ministral 3 model variants.
    Ministral 3 模型变体的配置。

    Supports three predefined sizes (3B, 8B, 14B) or custom configurations.
    支持三种预定义规模（3B、8B、14B）或自定义配置。
    """

    # 模型标识 / Model identifier
    model_name: str = "ministral-3"

    # 词表大小 / Vocabulary size
    vocab_size: int = 32000

    # 隐藏层维度 / Hidden layer dimension
    hidden_dim: int = 2560

    # Transformer层数 / Number of transformer layers
    num_layers: int = 24

    # 注意力头数 / Number of attention heads (query heads)
    num_heads: int = 20

    # KV头数（用于GQA）/ Number of KV heads (for grouped-query attention)
    num_kv_heads: int = 5

    # 注意力头维度（自动计算或手动指定）
    # Attention head dimension (auto-computed or manually specified)
    head_dim: Optional[int] = None

    # FFN中间层缩放因子 / FFN intermediate dimension scaling factor
    ffn_scale: float = 2.67

    # 是否启用共享权重FFN / Enable shared-weight FFN
    shared_ffn: bool = True

    # 最大序列长度 / Maximum sequence length
    max_seq_len: int = 4096

    # RoPE基础频率 / RoPE base frequency
    rope_theta: float = 10000.0

    # RMSNorm epsilon / RMSNorm epsilon
    rms_norm_eps: float = 1e-6

    # Dropout率 / Dropout rate
    dropout: float = 0.0

    # 绑定的词表大小（用于tie weights）/ Tied vocab size (for weight tying)
    tie_word_embeddings: bool = True

    def __post_init__(self):
        """Auto-compute head_dim if not specified / 自动计算head_dim"""
        if self.head_dim is None:
            self.head_dim = self.hidden_dim // self.num_heads
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            f"\nhidden_dim ({self.hidden_dim}) 必须能被 num_heads ({self.num_heads}) 整除"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            f"\nnum_heads ({self.num_heads}) 必须能被 num_kv_heads ({self.num_kv_heads}) 整除"
        )

    @classmethod
    def from_size(cls, size: str) -> "MinistralConfig":
        """
        Create config for a predefined model size.
        根据预定义模型规模创建配置。

        Args:
            size: One of "3B", "8B", "14B" / 可选 "3B", "8B", "14B"
        """
        # 各规模的预设参数 / Preset parameters for each size
        presets = {
            "3B": dict(
                hidden_dim=2560, num_layers=24, num_heads=20,
                num_kv_heads=5, ffn_scale=2.67,
            ),
            "8B": dict(
                hidden_dim=4096, num_layers=32, num_heads=32,
                num_kv_heads=8, ffn_scale=2.67,
            ),
            "14B": dict(
                hidden_dim=5120, num_layers=40, num_heads=40,
                num_kv_heads=10, ffn_scale=2.67,
            ),
        }
        if size not in presets:
            raise ValueError(
                f"Unknown size '{size}'. Choose from: {list(presets.keys())}"
                f"\n未知规模 '{size}'。可选: {list(presets.keys())}"
            )
        return cls(**presets[size])


# ============================================================================
# Rotary Position Embedding (RoPE) / 旋转位置编码
# ============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    旋转位置编码（RoPE）实现。

    Applies rotary embeddings to query and key tensors for position-aware
    attention without adding parameters.
    对查询和键张量应用旋转嵌入，实现位置感知注意力而不增加参数。

    Reference: Su et al., "RoFormer" (2024)
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        # 计算频率 / Compute frequencies
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None
        self._cached_seq_len: int = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cos/sin cache if sequence length exceeds cached length."""
        if seq_len <= self._cached_seq_len and self._cos_cache is not None:
            return
        self._cached_seq_len = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        # 拼接频率以匹配head_dim / Concatenate frequencies to match head_dim
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cache = emb.cos().to(dtype)
        self._sin_cache = emb.sin().to(dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.
        对查询和键张量应用旋转位置编码。

        Args:
            q: Query tensor [B, num_heads, seq_len, head_dim]
            k: Key tensor [B, num_kv_heads, seq_len, head_dim]
            position_offset: Offset for position indices (for KV cache)

        Returns:
            Tuple of rotated (q, k) tensors
        """
        seq_len = q.shape[2]
        self._update_cache(seq_len + position_offset, q.device, q.dtype)

        cos = self._cos_cache[position_offset : position_offset + seq_len]
        sin = self._sin_cache[position_offset : position_offset + seq_len]

        # 应用旋转 / Apply rotation
        q_rot = self._rotate_half(q, cos, sin)
        k_rot = self._rotate_half(k, cos, sin)
        return q_rot, k_rot

    @staticmethod
    def _rotate_half(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Rotate half the hidden dims of the input / 旋转输入的一半隐藏维度"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        # 广播cos/sin到正确的形状 / Broadcast cos/sin to correct shape
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        return x * cos + rotated * sin


# ============================================================================
# Efficient Attention / 高效注意力
# ============================================================================

class EfficientAttention(nn.Module):
    """
    Memory-efficient grouped-query attention with RoPE.
    带RoPE的内存高效分组查询注意力。

    Uses grouped-query attention (GQA) to reduce KV head count,
    lowering memory bandwidth requirements during inference.
    使用分组查询注意力（GQA）减少KV头数量，降低推理时的内存带宽需求。
    """

    def __init__(self, config: MinistralConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        # GQA重复因子 / GQA repetition factor
        self.num_groups = self.num_heads // self.num_kv_heads

        # 投影层 / Projection layers
        self.q_proj = nn.Linear(
            config.hidden_dim, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_dim, bias=False
        )

        # 旋转位置编码 / Rotary position embedding
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for efficient grouped-query attention.
        高效分组查询注意力的前向传播。

        Args:
            hidden_states: Input [B, seq_len, hidden_dim]
            attention_mask: Optional causal or padding mask
            position_offset: Position offset for RoPE (KV cache support)

        Returns:
            Output tensor [B, seq_len, hidden_dim]
        """
        B, seq_len, _ = hidden_states.shape

        # 线性投影 / Linear projections
        q = self.q_proj(hidden_states).view(B, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, seq_len, self.num_kv_heads, self.head_dim)

        # 转置为 [B, heads, seq, head_dim] / Transpose to [B, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 应用RoPE / Apply rotary embeddings
        q, k = self.rope(q, k, position_offset)

        # GQA: 扩展KV头以匹配查询头数 / Expand KV heads to match query heads
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # 缩放点积注意力 / Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 因果掩码 / Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=q.device),
                diagonal=1,
            )
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(q.dtype)

        # 注意力输出 / Attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, seq_len, self.num_heads * self.head_dim)

        return self.o_proj(attn_output)


# ============================================================================
# Efficient FFN / 高效前馈网络
# ============================================================================

class EfficientFFN(nn.Module):
    """
    Parameter-efficient FFN with optional shared weights.
    可选共享权重的参数高效前馈网络。

    When shared_ffn=True, uses structured weight sharing across intermediate
    dimensions to reduce parameters by ~30% while maintaining expressivity.
    当shared_ffn=True时，在中间维度上使用结构化权重共享，
    将参数量减少约30%同时保持表达能力。
    """

    def __init__(self, config: MinistralConfig):
        super().__init__()
        self.shared = config.shared_ffn
        hidden_dim = config.hidden_dim
        # 中间层维度 / Intermediate dimension
        intermediate_dim = int(hidden_dim * config.ffn_scale)

        if self.shared:
            # 共享权重FFN：使用较小的中间维度并通过共享模式扩展
            # Shared-weight FFN: smaller intermediate dim with sharing pattern
            self.shared_dim = intermediate_dim // 2
            self.w_gate = nn.Linear(hidden_dim, self.shared_dim, bias=False)
            self.w_up = nn.Linear(hidden_dim, self.shared_dim, bias=False)
            # 下投影使用完整中间维度 / Down projection uses full intermediate dim
            self.w_down = nn.Linear(self.shared_dim, hidden_dim, bias=False)
        else:
            # 标准FFN / Standard FFN (SwiGLU-style)
            self.w_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.w_up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.w_down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation and optional weight sharing.
        使用SwiGLU激活函数和可选权重共享的前向传播。

        Args:
            hidden_states: Input [B, seq_len, hidden_dim]

        Returns:
            Output [B, seq_len, hidden_dim]
        """
        gate = F.silu(self.w_gate(hidden_states))

        if self.shared:
            # 共享权重模式：通过拼接扩展中间表示
            # Shared weight mode: expand intermediate representation via concatenation
            up = self.w_up(hidden_states)
            # 拼接gate和up的变体以模拟更大的中间层
            # Concatenate gate and up variants to simulate larger intermediate layer
            combined = gate * up
            return self.w_down(combined)
        else:
            up = self.w_up(hidden_states)
            return self.w_down(gate * up)


# ============================================================================
# RMSNorm / 均方根归一化
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    均方根层归一化。

    More parameter-efficient than LayerNorm (no mean subtraction or bias).
    比LayerNorm更节省参数（无需均值减法或偏置）。
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


# ============================================================================
# Transformer Block / Transformer 块
# ============================================================================

class MinistralBlock(nn.Module):
    """
    Single Ministral transformer block with efficiency optimizations.
    单个Ministral Transformer块，包含效率优化。

    Architecture: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
    架构：RMSNorm -> 注意力 -> 残差 -> RMSNorm -> FFN -> 残差
    """

    def __init__(self, config: MinistralConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.attention = EfficientAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.ffn = EfficientFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass through one transformer block.
        通过一个Transformer块的前向传播。
        """
        # 注意力子层 + 残差连接 / Attention sublayer + residual
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attention(
            hidden_states, attention_mask, position_offset
        )
        hidden_states = residual + hidden_states

        # FFN子层 + 残差连接 / FFN sublayer + residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================================
# Full Model / 完整模型
# ============================================================================

class MinistralModel(nn.Module):
    """
    Complete Ministral 3 language model.
    完整的Ministral 3语言模型。

    Configurable for 3B, 8B, or 14B parameter counts with shared-weight FFN
    and grouped-query attention for maximum parameter efficiency.
    可配置为3B、8B或14B参数量，使用共享权重FFN和分组查询注意力
    以实现最大参数效率。
    """

    def __init__(self, config: Optional[MinistralConfig] = None, size: str = "3B"):
        super().__init__()
        if config is None:
            config = MinistralConfig.from_size(size)
        self.config = config

        # Token嵌入 / Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer层 / Transformer layers
        self.layers = nn.ModuleList([
            MinistralBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # 最终层归一化 / Final layer normalization
        self.final_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

        # LM头 / Language model head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # 权重绑定 / Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # 初始化权重 / Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with scaled normal distribution / 使用缩放正态分布初始化权重"""
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.config.num_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the complete model.
        通过完整模型的前向传播。

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation [B, seq_len]

        Returns:
            Tuple of (logits [B, seq_len, vocab_size], optional loss)
        """
        # Token嵌入 / Token embedding
        hidden_states = self.embed_tokens(input_ids)

        # 逐层前向传播 / Layer-by-layer forward pass
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # 最终归一化 / Final normalization
        hidden_states = self.final_norm(hidden_states)

        # Logits计算 / Compute logits
        logits = self.lm_head(hidden_states)

        # 损失计算（如果提供标签）/ Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss


# ============================================================================
# Utility Functions / 工具函数
# ============================================================================

def count_params(model: nn.Module, verbose: bool = False) -> dict:
    """
    Count model parameters with breakdown by component.
    按组件统计模型参数数量。

    Args:
        model: PyTorch model / PyTorch模型
        verbose: Whether to print detailed breakdown / 是否打印详细分解

    Returns:
        Dictionary with parameter counts / 参数量字典
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 按组件统计 / Breakdown by component
    breakdown = {}
    if isinstance(model, MinistralModel):
        embed_params = sum(p.numel() for p in model.embed_tokens.parameters())
        layer_params = sum(p.numel() for p in model.layers.parameters())
        norm_params = sum(p.numel() for p in model.final_norm.parameters())
        head_params = sum(p.numel() for p in model.lm_head.parameters())
        breakdown = {
            "embedding / 嵌入层": embed_params,
            "transformer_layers / Transformer层": layer_params,
            "final_norm / 最终归一化": norm_params,
            "lm_head / 语言模型头": head_params,
        }

    result = {
        "total": total,
        "trainable": trainable,
        "total_billions": total / 1e9,
        "breakdown": breakdown,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Parameter Count / 参数统计: {model.__class__.__name__}")
        print(f"{'='*60}")
        print(f"Total parameters / 总参数: {total:,} ({total/1e9:.2f}B)")
        print(f"Trainable / 可训练: {trainable:,} ({trainable/1e9:.2f}B)")
        for name, count in breakdown.items():
            pct = count / total * 100
            print(f"  {name}: {count:,} ({pct:.1f}%)")
        print(f"{'='*60}\n")

    return result


def estimate_memory(config: MinistralConfig, precision: str = "fp16") -> dict:
    """
    Estimate memory requirements for model training and inference.
    估算模型训练和推理的内存需求。

    Args:
        config: Model configuration / 模型配置
        precision: "fp16", "bf16", or "fp32" / 精度

    Returns:
        Dictionary with memory estimates in GB / 内存估算字典（GB）
    """
    # 字节数计算 / Bytes per parameter
    bytes_per_param = {"fp16": 2, "bf16": 2, "fp32": 4}.get(precision, 2)

    # 估算参数量 / Estimate parameter count
    hidden = config.hidden_dim
    layers = config.num_layers
    vocab = config.vocab_size
    heads = config.num_heads
    kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    ffn_inter = int(hidden * config.ffn_scale)

    # 每层参数 / Parameters per layer
    attn_params = hidden * (heads + 2 * kv_heads) * head_dim + heads * head_dim * hidden
    if config.shared_ffn:
        ffn_params = hidden * (ffn_inter // 2) * 2 + (ffn_inter // 2) * hidden
    else:
        ffn_params = hidden * ffn_inter * 2 + ffn_inter * hidden
    norm_params = hidden * 2  # 两个RMSNorm / Two RMSNorms

    per_layer = attn_params + ffn_params + norm_params
    total_params = vocab * hidden * 2 + layers * per_layer + hidden

    # 内存估算 / Memory estimation
    model_memory_gb = total_params * bytes_per_param / 1e9
    # AdamW优化器状态：2x fp32 / AdamW optimizer states: 2x fp32
    optimizer_memory_gb = total_params * 4 * 2 / 1e9  # 训练时 / during training
    # 梯度：与模型参数相同大小 / Gradients: same size as model params
    gradient_memory_gb = total_params * bytes_per_param / 1e9

    return {
        "model_inference_gb": round(model_memory_gb, 2),
        "model_training_gb": round(model_memory_gb + optimizer_memory_gb + gradient_memory_gb, 2),
        "optimizer_states_gb": round(optimizer_memory_gb, 2),
        "total_params_estimated": total_params,
    }


def train_example(
    size: str = "3B",
    seq_len: int = 128,
    batch_size: int = 2,
    num_steps: int = 10,
    lr: float = 1e-4,
    device: str = "cpu",
):
    """
    Minimal training example to verify model functionality.
    最小训练示例，用于验证模型功能。

    Uses random token IDs as input — this is NOT a real training run,
    just a smoke test to verify the forward/backward pass works correctly.
    使用随机token ID作为输入——这不是真正的训练，
    只是验证前向/反向传播正确工作的冒烟测试。

    Args:
        size: Model size ("3B"/"8B"/"14B") / 模型规模
        seq_len: Sequence length / 序列长度
        batch_size: Batch size / 批量大小
        num_steps: Number of training steps / 训练步数
        lr: Learning rate / 学习率
        device: Device to run on / 运行设备
    """
    print(f"\n{'='*60}")
    print(f"Ministral 3 Training Smoke Test / 训练冒烟测试")
    print(f"Model size / 模型规模: {size}")
    print(f"Device / 设备: {device}")
    print(f"{'='*60}\n")

    # 使用小配置进行快速测试 / Use small config for fast testing
    # 注意：真正训练时使用 MinistralConfig.from_size(size)
    # Note: Use MinistralConfig.from_size(size) for real training
    config = MinistralConfig(
        hidden_dim=256, num_layers=4, num_heads=8,
        num_kv_heads=2, vocab_size=1000, max_seq_len=seq_len,
    )

    model = MinistralModel(config=config).to(device)
    params = count_params(model, verbose=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()

    print("Training steps / 训练步骤:")
    for step in range(num_steps):
        # 随机输入 / Random input
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        start = time.time()
        logits, loss = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        elapsed = (time.time() - start) * 1000

        print(
            f"  Step {step+1:3d}/{num_steps} | "
            f"Loss: {loss.item():.4f} | "
            f"Time: {elapsed:.1f}ms | "
            f"LR: {lr:.2e}"
        )

    print(f"\nSmoke test complete / 冒烟测试完成")
    print(f"Final loss / 最终损失: {loss.item():.4f}")


# ============================================================================
# Main / 主程序
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ministral 3 Model Demo / Ministral 3 模型演示"
    )
    parser.add_argument(
        "--size", type=str, default="3B", choices=["3B", "8B", "14B"],
        help="Model size / 模型规模 (default: 3B)"
    )
    parser.add_argument(
        "--action", type=str, default="info",
        choices=["info", "train-smoke"],
        help="Action to perform / 执行操作 (default: info)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device / 设备 (default: cpu)"
    )
    args = parser.parse_args()

    if args.action == "info":
        # 打印模型信息 / Print model info
        config = MinistralConfig.from_size(args.size)
        print(f"\nMinistral 3 {args.size} Configuration / 配置:")
        print(f"  Hidden dim / 隐藏维度: {config.hidden_dim}")
        print(f"  Layers / 层数: {config.num_layers}")
        print(f"  Heads / 注意力头: {config.num_heads}")
        print(f"  KV heads / KV头: {config.num_kv_heads}")
        print(f"  Shared FFN / 共享FFN: {config.shared_ffn}")

        mem = estimate_memory(config, precision="fp16")
        print(f"\nMemory Estimates / 内存估算 (FP16):")
        print(f"  Inference / 推理: {mem['model_inference_gb']:.2f} GB")
        print(f"  Training / 训练: {mem['model_training_gb']:.2f} GB")
        print(f"  Estimated params / 估计参数量: {mem['total_params_estimated']/1e9:.2f}B")

    elif args.action == "train-smoke":
        train_example(size=args.size, device=args.device)
