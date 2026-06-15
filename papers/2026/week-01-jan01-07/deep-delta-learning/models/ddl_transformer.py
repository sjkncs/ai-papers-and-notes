"""
Deep Delta Learning (DDL) Transformer — 完整独立实现
=====================================================
Complete Standalone Implementation

基于论文 "Deep Delta Learning" (arXiv:2601.00417) 的完整复现代码。
本文件完全自包含，不依赖 ddl_layer.py 或其他外部模块。

Based on the paper "Deep Delta Learning" (arXiv:2601.00417).
This file is fully self-contained and does not import from ddl_layer.py.

核心思想 / Core Idea:
  标准ResNet残差:  x_{l+1} = x_l + f_l(x_l)           (只能累加 / append-only)
  DDL残差:        x_{l+1} = x_l + g_l(x_l) * (target_l(x_l) - x_l) + f_l(x_l)
                                                        (可选择性重写 / selective rewrite)

  其中 / Where:
    g_l(x)     — 学习到的门控函数 (per-dimension scalar) / Learned gate function
    target_l(x) — 学习到的目标值投影 / Learned target projection
    f_l(x)     — 标准子层输出 (attention / FFN) / Standard sublayer output

包含组件 / Components:
  1. GatedDeltaGate       — 门控Delta修正模块 / Gated delta correction module
  2. DDLLayer             — DDL残差层 / DDL residual layer
  3. DDLTransformerBlock  — DDL Transformer Block
  4. DDLModel             — 完整DDL语言模型 / Full DDL language model
  5. StandardTransformer  — 标准ResNet Transformer (对比基线) / Standard baseline
  6. count_params         — 参数计数 / Parameter counting
  7. estimate_flops       — FLOPs估计 / FLOPs estimation

Usage:
    model = DDLModel(vocab_size=10000, d_model=256, n_layers=6)
    logits = model(input_ids)

Author: Auto-generated from paper analysis
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Dict, Tuple


# ============================================================
# 1. 门控Delta修正模块 / Gated Delta Correction Module
# ============================================================

class GatedDeltaGate(nn.Module):
    """
    门控Delta修正模块 / Gated Delta Correction Module

    计算门控值 g(x) 和目标值 target(x)，用于选择性重写残差内容。
    Computes gate values g(x) and target values target(x) for selective
    rewriting of residual content.

    数学形式 / Mathematical Form:
        g(x) = sigmoid(W_g * LayerNorm(x))  in [0, 1]^d
        target(x) = W_t * LayerNorm(x)      (learned target projection)

    量化意义 / Significance:
        - g(x) 接近 0: 保留原始残差内容 / Keep original residual content
        - g(x) 接近 1: 用目标值替换当前内容 / Replace with target value
        - 这使得网络能够主动"遗忘"过时信息 / Allows active "forgetting"
    """

    def __init__(self, d_model: int, gate_init: float = 0.0):
        """
        Args:
            d_model: 隐藏维度 / Hidden dimension size
            gate_init: 门控初始偏置 / Gate initial bias
                       0.0 → sigmoid(0) = 0.5 (50%开度 / 50% open)
                       负值 → 初始更保守 / Negative → initially conservative
        """
        super().__init__()
        self.d_model = d_model

        # 层归一化 (用于门控和目标投影的输入)
        # Layer normalization (input for gate and target projections)
        self.norm = nn.LayerNorm(d_model)

        # 门控投影 / Gate projection: outputs values in [0, 1]
        # sigmoid(W_g * norm(x)) 控制每个维度的"重写强度"
        self.gate_proj = nn.Linear(d_model, d_model)
        nn.init.constant_(self.gate_proj.bias, gate_init)
        # 权重小初始化，使初始门控接近 gate_init 决定的值
        # Small weight init so initial gate is close to bias-determined value
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.01)

        # 目标值投影 / Target value projection
        # 初始化为近恒等映射，使delta修正初始接近零
        # Initialized near identity so delta correction starts near zero
        self.target_proj = nn.Linear(d_model, d_model)
        nn.init.eye_(self.target_proj.weight)
        nn.init.zeros_(self.target_proj.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 / Forward pass

        Args:
            x: 当前隐状态 / Current hidden state [batch, seq_len, d_model]

        Returns:
            gate: 门控值 / Gate values in [0, 1], shape [batch, seq_len, d_model]
            target: 目标值 / Target values, shape [batch, seq_len, d_model]
        """
        normed = self.norm(x)
        gate = torch.sigmoid(self.gate_proj(normed))
        target = self.target_proj(normed)
        return gate, target


# ============================================================
# 2. DDL残差层 / DDL Residual Layer
# ============================================================

class DDLLayer(nn.Module):
    """
    Deep Delta Learning 残差层 / DDL Residual Layer

    将标准残差连接 x + f(x) 替换为 DDL 更新规则:
    Replaces standard residual x + f(x) with DDL update rule:

        x_{l+1} = x_l + g_l(x_l) * (target_l(x_l) - x_l) + f_l(x_l)

    三项含义 / Three terms:
        1. x_l             — 恒等路径 / Identity path (保留梯度流)
        2. g*(target - x)  — Delta修正 / Delta correction (选择性重写)
        3. f_l(x_l)        — 子层输出 / Sublayer output (attention/FFN)

    与标准残差的对比 / Comparison with standard residual:
        标准 / Standard: x + f(x)         → 只能累加特征 / Can only accumulate
        DDL:            x + delta + f(x)  → 可重写+累加 / Can rewrite + accumulate
    """

    def __init__(
        self,
        d_model: int,
        sublayer_type: str = "generic",
        gate_init: float = 0.0,
        use_pre_norm: bool = True,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model: 隐藏维度 / Hidden dimension
            sublayer_type: 子层类型标记 (仅用于日志) / Sublayer type tag (logging only)
            gate_init: 门控初始偏置 / Gate initial bias
            use_pre_norm: 是否使用Pre-LN / Whether to use Pre-LayerNorm
            dropout: Dropout率 / Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.sublayer_type = sublayer_type
        self.use_pre_norm = use_pre_norm

        # DDL门控修正模块 / Gated delta correction module
        self.delta_gate = GatedDeltaGate(d_model, gate_init=gate_init)

        # Pre-layer normalization (在子层之前应用 / Applied before sublayer)
        self.pre_norm = nn.LayerNorm(d_model) if use_pre_norm else nn.Identity()

        # 子层输出的Dropout / Dropout for sublayer output
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        sublayer_fn: Callable[[torch.Tensor], torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """
        DDL前向传播 / DDL Forward Pass

        计算流程 / Computation flow:
            1. 计算门控和目标值 / Compute gate and target
            2. Delta修正: g(x) * (target(x) - x)
            3. 子层计算: f(norm(x))
            4. 组合: x + delta + f(x)

        Args:
            x: 输入隐状态 / Input hidden state [batch, seq_len, d_model]
            sublayer_fn: 子层函数 (attention 或 FFN) / Sublayer function
            **kwargs: 传递给sublayer_fn的额外参数 / Extra args for sublayer_fn

        Returns:
            更新后的隐状态 / Updated hidden state [batch, seq_len, d_model]
        """
        # Step 1: 计算门控和目标值 / Compute gate and target
        gate, target = self.delta_gate(x)

        # Step 2: Delta修正 — 选择性重写 / Selective rewrite
        # delta = g(x) * (target(x) - x)
        # 当g≈1时，x被拉向target；当g≈0时，x保持不变
        # When g≈1, x is pulled toward target; when g≈0, x stays unchanged
        delta_correction = gate * (target - x)

        # Step 3: 标准子层计算 / Standard sublayer computation
        if self.use_pre_norm:
            sublayer_input = self.pre_norm(x)
        else:
            sublayer_input = x

        sublayer_output = sublayer_fn(sublayer_input, **kwargs)
        sublayer_output = self.dropout(sublayer_output)

        # Step 4: DDL组合 / DDL combination
        # x_{l+1} = x_l + delta_correction + f_l(x_l)
        output = x + delta_correction + sublayer_output

        return output


# ============================================================
# 3. DDL Transformer Block / DDL Transformer Block
# ============================================================

class DDLTransformerBlock(nn.Module):
    """
    使用DDL残差连接的完整Transformer Block
    Full Transformer Block with DDL Residual Connections

    结构 / Structure:
        DDL(Attention) -> DDL(FFN)

    每个残差连接都使用DDL门控修正，替代标准的 x + sublayer(x)。
    Every residual connection uses DDL gated correction instead of
    the standard x + sublayer(x).

    与标准Transformer Block的对比 / Comparison with standard block:
        标准 / Standard:
            x = x + Attention(LayerNorm(x))
            x = x + FFN(LayerNorm(x))

        DDL:
            x = x + Gate_Attn*(Target_Attn - x) + Attention(LayerNorm(x))
            x = x + Gate_FFN*(Target_FFN - x)   + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        gate_init: float = 0.0,
        use_pre_norm: bool = True,
    ):
        """
        Args:
            d_model: 隐藏维度 / Hidden dimension
            n_heads: 注意力头数 / Number of attention heads
            d_ff: FFN中间维度 / FFN intermediate dim (default: 4 * d_model)
            dropout: Dropout率 / Dropout rate
            gate_init: DDL门控初始偏置 / DDL gate initial bias
            use_pre_norm: 是否使用Pre-LN / Whether to use Pre-LayerNorm
        """
        super().__init__()
        d_ff = d_ff or 4 * d_model

        # --- Attention 子层 / Attention sublayer ---
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # --- FFN 子层 / FFN sublayer ---
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # --- DDL残差层 (替代标准残差) / DDL residual layers ---
        self.ddl_attn = DDLLayer(
            d_model, sublayer_type="attention",
            gate_init=gate_init, dropout=dropout,
            use_pre_norm=use_pre_norm,
        )
        self.ddl_ffn = DDLLayer(
            d_model, sublayer_type="ffn",
            gate_init=gate_init, dropout=dropout,
            use_pre_norm=use_pre_norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DDL Transformer Block 前向传播 / Forward pass

        Args:
            x: 输入 / Input [batch, seq_len, d_model]
            attn_mask: 注意力掩码 / Attention mask
            key_padding_mask: Padding掩码 / Padding mask

        Returns:
            输出 / Output [batch, seq_len, d_model]
        """
        # DDL Attention: x = x + gate*(target - x) + Attention(norm(x))
        def attn_fn(x_normed, **kw):
            attn_out, _ = self.attn(
                x_normed, x_normed, x_normed,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            return attn_out

        x = self.ddl_attn(x, attn_fn)

        # DDL FFN: x = x + gate*(target - x) + FFN(norm(x))
        x = self.ddl_ffn(x, self.ffn)

        return x

    def get_gate_stats(self) -> Dict[str, torch.Tensor]:
        """
        获取门控统计信息 / Get gate statistics

        Returns:
            dict: 包含attention和FFN的门控均值和方差
                  Contains mean and variance of attention and FFN gates
        """
        with torch.no_grad():
            attn_gate_weight = self.ddl_attn.delta_gate.gate_proj.weight.abs().mean()
            ffn_gate_weight = self.ddl_ffn.delta_gate.gate_proj.weight.abs().mean()
        return {
            "attn_gate_magnitude": attn_gate_weight,
            "ffn_gate_magnitude": ffn_gate_weight,
        }


# ============================================================
# 4. 完整DDL语言模型 / Full DDL Language Model
# ============================================================

class DDLModel(nn.Module):
    """
    使用DDL的完整Decoder-Only语言模型
    Full Decoder-Only Language Model with DDL

    架构 / Architecture:
        Token Embedding + Position Embedding
        -> N x DDLTransformerBlock
        -> LayerNorm
        -> LM Head (tied weights with embedding)

    对比实验 / Comparison experiment:
        用DDL替换所有标准残差连接，观察语言建模质量变化。
        Replace all standard residual connections with DDL and
        observe changes in language modeling quality.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        gate_init: float = 0.0,
        use_pre_norm: bool = True,
        tie_weights: bool = True,
    ):
        """
        Args:
            vocab_size: 词汇表大小 / Vocabulary size
            d_model: 隐藏维度 / Hidden dimension
            n_layers: Transformer层数 / Number of Transformer layers
            n_heads: 注意力头数 / Number of attention heads
            d_ff: FFN中间维度 / FFN intermediate dimension
            max_seq_len: 最大序列长度 / Maximum sequence length
            dropout: Dropout率 / Dropout rate
            gate_init: DDL门控初始偏置 / DDL gate initial bias
            use_pre_norm: 是否使用Pre-LN / Whether to use Pre-LayerNorm
            tie_weights: 是否绑定输入输出权重 / Whether to tie embedding and LM head weights
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # Token和位置嵌入 / Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # DDL Transformer Blocks
        self.blocks = nn.ModuleList([
            DDLTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                gate_init=gate_init,
                use_pre_norm=use_pre_norm,
            )
            for _ in range(n_layers)
        ])

        # 最终层归一化和语言模型头 / Final LayerNorm and LM head
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定 / Weight tying
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # 初始化权重 / Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        权重初始化策略 / Weight initialization strategy
        遵循GPT-2风格初始化 / Follows GPT-2 style initialization
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward pass

        Args:
            input_ids: Token ID序列 / Token ID sequences [batch, seq_len]

        Returns:
            logits: 语言模型logits / Language model logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
        )

        # 位置索引 / Position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # 嵌入 / Embedding
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.embed_dropout(x)

        # 因果掩码 / Causal mask (上三角为True表示需要mask的位置)
        # Upper triangle True means positions to mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1,
        ).bool()

        # 通过所有DDL Transformer Block / Pass through all DDL blocks
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        # 最终归一化和logits / Final normalization and logits
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits

    def get_all_gate_stats(self) -> list:
        """
        获取所有层的门控统计信息 / Get gate statistics for all layers

        Returns:
            list of dicts, one per layer
        """
        return [block.get_gate_stats() for block in self.blocks]

    def get_gate_values(self, input_ids: torch.Tensor) -> list:
        """
        获取给定输入的实际门控值 / Get actual gate values for given input

        用于可视化训练过程中门控值的变化。
        For visualizing how gate values change during training.

        Args:
            input_ids: [batch, seq_len]

        Returns:
            list of (gate_attn, gate_ffn) tuples per layer
        """
        gate_values = []
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1,
        ).bool()

        with torch.no_grad():
            for block in self.blocks:
                # 获取当前层的门控值 / Get this layer's gate values
                gate_attn, _ = block.ddl_attn.delta_gate(x)
                gate_ffn, _ = block.ddl_ffn.delta_gate(x)

                gate_values.append({
                    "attn_gate_mean": gate_attn.mean().item(),
                    "attn_gate_std": gate_attn.std().item(),
                    "ffn_gate_mean": gate_ffn.mean().item(),
                    "ffn_gate_std": gate_ffn.std().item(),
                })

                # 继续前向传播 / Continue forward pass
                x = block(x, attn_mask=causal_mask)

        return gate_values


# ============================================================
# 5. 标准Transformer (对比基线) / Standard Transformer (Baseline)
# ============================================================

class StandardTransformerBlock(nn.Module):
    """
    标准Pre-LN Transformer Block (使用ResNet残差连接)
    Standard Pre-LN Transformer Block (with ResNet residual connections)

    用于与DDL进行对比实验的基线模型。
    Baseline model for comparison experiments with DDL.

    结构 / Structure:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_pre_norm: bool = True,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.use_pre_norm = use_pre_norm

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 标准残差: x + sublayer(norm(x)) / Standard residual
        if self.use_pre_norm:
            attn_out, _ = self.attn(
                self.norm1(x), self.norm1(x), self.norm1(x),
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            x = x + self.dropout(attn_out)
            x = x + self.ffn(self.norm2(x))
        else:
            attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask)
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.ffn(x))
        return x


class StandardTransformer(nn.Module):
    """
    标准Decoder-Only语言模型 (ResNet残差连接基线)
    Standard Decoder-Only LM (ResNet residual connection baseline)

    与DDLModel结构完全一致，但使用标准 x + f(x) 残差连接。
    Identical architecture to DDLModel but uses standard x + f(x) residuals.

    用于公平的A/B对比实验 / For fair A/B comparison experiments.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_pre_norm: bool = True,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # 标准Transformer Blocks (无DDL) / Standard blocks (no DDL)
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_pre_norm=use_pre_norm,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.embed_dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1,
        ).bool()

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


# ============================================================
# 6. 工具函数 / Utility Functions
# ============================================================

def count_params(model: nn.Module, verbose: bool = False) -> int:
    """
    统计模型参数数量 / Count model parameters

    Args:
        model: PyTorch模型 / PyTorch model
        verbose: 是否打印详细信息 / Whether to print details

    Returns:
        总参数数量 / Total parameter count
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"  总参数 / Total params:     {total:>12,}")
        print(f"  可训练 / Trainable params: {trainable:>12,}")
        print(f"  非训练 / Non-trainable:    {total - trainable:>12,}")
        # 按模块分组统计 / Per-module breakdown
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            print(f"    {name:30s}: {module_params:>12,}")

    return total


def estimate_flops(
    d_model: int,
    n_layers: int,
    n_heads: int,
    seq_len: int,
    vocab_size: int,
    d_ff: Optional[int] = None,
    use_ddl: bool = True,
) -> Dict[str, float]:
    """
    估计前向传播的FLOPs / Estimate forward-pass FLOPs

    使用近似公式估计，不考虑激活函数和LayerNorm的FLOPs。
    Uses approximate formulas; ignores activation and LayerNorm FLOPs.

    Args:
        d_model: 隐藏维度 / Hidden dimension
        n_layers: 层数 / Number of layers
        n_heads: 头数 / Number of heads
        seq_len: 序列长度 / Sequence length
        vocab_size: 词汇表大小 / Vocabulary size
        d_ff: FFN维度 / FFN dimension
        use_ddl: 是否使用DDL / Whether using DDL

    Returns:
        dict: 各组件的FLOPs估计 / FLOPs estimates per component
    """
    d_ff = d_ff or 4 * d_model

    # 注意力FLOPs (QKV投影 + 注意力矩阵 + 值加权)
    # Attention FLOPs (QKV projection + attention matrix + value weighting)
    # QKV: 3 * 2 * B * T * D * D (矩阵乘法 ~ 2*M*N*K FLOPs)
    attn_qkv_flops = 3 * 2 * seq_len * d_model * d_model
    # 注意力分数: 2 * T * T * D (Q*K^T)
    attn_score_flops = 2 * seq_len * seq_len * d_model
    # 值加权: 2 * T * D * T (attn * V)
    attn_value_flops = 2 * seq_len * d_model * seq_len
    attn_total = (attn_qkv_flops + attn_score_flops + attn_value_flops) * n_layers

    # FFN FLOPs (两层线性)
    # FFN FLOPs (two linear layers)
    ffn_flops = 2 * (2 * seq_len * d_model * d_ff + 2 * seq_len * d_ff * d_model) * n_layers

    # DDL额外FLOPs / DDL extra FLOPs
    ddl_flops = 0
    if use_ddl:
        # 每层: gate_proj + target_proj = 2 * 2 * T * D * D
        # Per layer: gate_proj + target_proj
        ddl_per_layer = 2 * 2 * seq_len * d_model * d_model
        ddl_flops = ddl_per_layer * n_layers * 2  # attention + FFN 各一次

    # LM Head FLOPs
    lm_head_flops = 2 * seq_len * d_model * vocab_size

    total = attn_total + ffn_flops + ddl_flops + lm_head_flops

    return {
        "attention_flops": attn_total,
        "ffn_flops": ffn_flops,
        "ddl_extra_flops": ddl_flops,
        "lm_head_flops": lm_head_flops,
        "total_flops": total,
        "ddl_overhead_ratio": ddl_flops / (total - ddl_flops) if (total - ddl_flops) > 0 else 0.0,
    }


def compare_architectures(
    vocab_size: int = 10000,
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 4,
    d_ff: Optional[int] = None,
    max_seq_len: int = 512,
    seq_len: int = 128,
) -> Dict[str, Dict]:
    """
    对比DDL与标准Transformer的参数和FLOPs
    Compare DDL vs Standard Transformer parameters and FLOPs

    Args:
        vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len: 模型参数
        seq_len: 用于FLOPs估计的序列长度

    Returns:
        dict: 对比结果 / Comparison results
    """
    ddl_model = DDLModel(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_ff=d_ff, max_seq_len=max_seq_len,
    )
    std_model = StandardTransformer(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_ff=d_ff, max_seq_len=max_seq_len,
    )

    ddl_params = count_params(ddl_model)
    std_params = count_params(std_model)

    ddl_flops = estimate_flops(d_model, n_layers, n_heads, seq_len, vocab_size, d_ff, use_ddl=True)
    std_flops = estimate_flops(d_model, n_layers, n_heads, seq_len, vocab_size, d_ff, use_ddl=False)

    print("=" * 65)
    print("  DDL vs Standard Transformer 对比 / Comparison")
    print("=" * 65)
    print(f"  {'指标 / Metric':<30s} {'DDL':>14s} {'Standard':>14s} {'Overhead':>10s}")
    print("-" * 65)
    print(f"  {'参数量 / Parameters':<30s} {ddl_params:>14,d} {std_params:>14,d} "
          f"{(ddl_params - std_params) / std_params:>9.2%}")
    print(f"  {'总FLOPs / Total FLOPs':<30s} {ddl_flops['total_flops']:>14,.0f} "
          f"{std_flops['total_flops']:>14,.0f} "
          f"{ddl_flops['ddl_overhead_ratio']:>9.2%}")
    print(f"  {'DDL额外FLOPs / DDL extra':<30s} {ddl_flops['ddl_extra_flops']:>14,.0f} "
          f"{'—':>14s} {'—':>10s}")
    print("=" * 65)

    # DDL每层额外参数 / Extra params per DDL layer
    d_ff_val = d_ff or 4 * d_model
    extra_per_layer = (
        d_model * d_model + d_model   # gate_proj (weight + bias)
        + d_model * d_model + d_model  # target_proj (weight + bias)
        + 2 * d_model                  # LayerNorm (weight + bias)
    )
    print(f"  DDL每层额外参数 / Extra params per layer: {extra_per_layer:,}")
    print(f"  DDL总额外参数 ({n_layers}层) / Total extra ({n_layers}L): "
          f"{extra_per_layer * n_layers * 2:,}")  # *2 for attn + ffn
    print("=" * 65)

    return {
        "ddl": {"params": ddl_params, "flops": ddl_flops},
        "standard": {"params": std_params, "flops": std_flops},
    }


# ============================================================
# 演示入口 / Demo Entry Point
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Deep Delta Learning (DDL) Transformer")
    print("完整独立实现 / Complete Standalone Implementation")
    print("=" * 60)

    # --- 架构对比 / Architecture comparison ---
    print("\n--- 架构对比 / Architecture Comparison ---")
    compare_architectures()

    # --- 前向传播演示 / Forward pass demo ---
    print("\n--- 前向传播演示 / Forward Pass Demo ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DDLModel(
        vocab_size=10000, d_model=128, n_layers=3,
        n_heads=4, d_ff=512, max_seq_len=256,
    ).to(device)

    input_ids = torch.randint(0, 10000, (2, 64), device=device)
    logits = model(input_ids)
    print(f"Input shape:  {list(input_ids.shape)}")
    print(f"Output shape: {list(logits.shape)}")
    print(f"Parameters:   {count_params(model):,}")

    # --- 门控值统计 / Gate value statistics ---
    print("\n--- 门控值统计 / Gate Statistics ---")
    gate_values = model.get_gate_values(input_ids)
    for i, gv in enumerate(gate_values):
        print(f"  Layer {i}: attn_gate_mean={gv['attn_gate_mean']:.4f}, "
              f"ffn_gate_mean={gv['ffn_gate_mean']:.4f}")

    # --- FLOPs估计 / FLOPs estimation ---
    print("\n--- FLOPs估计 / FLOPs Estimation ---")
    flops = estimate_flops(
        d_model=128, n_layers=3, n_heads=4,
        seq_len=64, vocab_size=10000, d_ff=512,
    )
    print(f"  Total FLOPs:     {flops['total_flops']:,.0f}")
    print(f"  DDL extra FLOPs: {flops['ddl_extra_flops']:,.0f}")
    print(f"  DDL overhead:    {flops['ddl_overhead_ratio']:.2%}")
