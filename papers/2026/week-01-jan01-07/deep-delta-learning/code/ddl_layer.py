"""
Deep Delta Learning (DDL) Layer — PyTorch Implementation
=========================================================

基于论文 "Deep Delta Learning" (arXiv:2601.00417) 的复现代码。
DDL残差层：将标准残差连接从"只能累加"升级为"可选择性重写"。

Standard ResNet:  x_{l+1} = x_l + f_l(x_l)
DDL:             x_{l+1} = x_l + g_l(x_l) * (target_l(x_l) - x_l) + f_l(x_l)

其中:
  g_l(x)    — 学习到的门控函数 (per-dimension scalar)
  target_l(x) — 学习到的目标值投影
  f_l(x)    — 标准子层输出 (attention / FFN)

Usage:
    # 替换Transformer中的标准残差连接
    layer = DDLLayer(d_model=512, sublayer_type="attention")
    output = layer(x, sublayer_fn=attention_fn)

Author: Auto-generated from paper analysis
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional


class GatedDeltaGate(nn.Module):
    """
    门控Delta修正模块 / Gated Delta Correction Module

    计算门控值 g(x) 和目标值 target(x)，用于选择性重写残差内容。

    g(x) = sigmoid(W_g * LayerNorm(x))  ∈ [0, 1]^d
    target(x) = W_t * LayerNorm(x)      (learned target projection)
    """

    def __init__(self, d_model: int, gate_init: float = 0.0):
        """
        Args:
            d_model: 隐藏维度 / Hidden dimension size
            gate_init: 门控初始偏置 / Gate initial bias (0 = ~50% open)
        """
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # 门控投影 / Gate projection: outputs values in [0, 1]
        self.gate_proj = nn.Linear(d_model, d_model)
        nn.init.constant_(self.gate_proj.bias, gate_init)

        # 目标值投影 / Target value projection
        self.target_proj = nn.Linear(d_model, d_model)
        # Initialize target to identity so delta starts near zero
        nn.init.eye_(self.target_proj.weight)
        nn.init.zeros_(self.target_proj.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 当前隐状态 [batch, seq_len, d_model]

        Returns:
            gate: 门控值 g(x) ∈ [0, 1]
            target: 目标值 target(x)
        """
        normed = self.norm(x)
        gate = torch.sigmoid(self.gate_proj(normed))
        target = self.target_proj(normed)
        return gate, target


class DDLLayer(nn.Module):
    """
    Deep Delta Learning 残差层 / DDL Residual Layer

    将标准残差 x + f(x) 替换为:
      x + g(x) * (target(x) - x) + f(x)

    第一项 g(x) * (target(x) - x) 是"delta修正"——选择性地将x向目标值拉近
    第二项 f(x) 是标准的子层输出（如attention或FFN）
    """

    def __init__(
        self,
        d_model: int,
        sublayer_type: str = "attention",
        gate_init: float = 0.0,
        use_pre_norm: bool = True,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model: 隐藏维度
            sublayer_type: 子层类型标记 (仅用于日志)
            gate_init: 门控初始偏置
            use_pre_norm: 是否使用Pre-LN
            dropout: dropout率
        """
        super().__init__()
        self.d_model = d_model
        self.sublayer_type = sublayer_type
        self.use_pre_norm = use_pre_norm

        # DDL门控修正模块 / Gated delta correction module
        self.delta_gate = GatedDeltaGate(d_model, gate_init=gate_init)

        # Pre-layer normalization (applied before sublayer)
        self.pre_norm = nn.LayerNorm(d_model) if use_pre_norm else nn.Identity()

        # Dropout for sublayer output
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        sublayer_fn: Callable[[torch.Tensor], torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """
        DDL前向传播 / DDL Forward Pass

        Args:
            x: 输入隐状态 [batch, seq_len, d_model]
            sublayer_fn: 子层函数 (如 attention 或 FFN)
            **kwargs: 传递给sublayer_fn的额外参数 (如 attention mask)

        Returns:
            更新后的隐状态 [batch, seq_len, d_model]
        """
        # Step 1: 计算门控和目标值 / Compute gate and target
        gate, target = self.delta_gate(x)

        # Step 2: Delta修正 — 选择性重写 / Selective rewrite
        # delta = g(x) * (target(x) - x)
        delta_correction = gate * (target - x)

        # Step 3: 标准子层计算 / Standard sublayer computation
        if self.use_pre_norm:
            sublayer_input = self.pre_norm(x)
        else:
            sublayer_input = x

        sublayer_output = sublayer_fn(sublayer_input, **kwargs)
        sublayer_output = self.dropout(sublayer_output)

        # Step 4: DDL组合 — 残差 + delta修正 + 子层输出
        # x_{l+1} = x_l + delta_correction + f_l(x_l)
        output = x + delta_correction + sublayer_output

        return output


class DDLTransformerBlock(nn.Module):
    """
    使用DDL残差连接的完整Transformer Block / Full Transformer Block with DDL

    结构: DDL(Attention) → DDL(FFN)
    每个残差连接都使用DDL门控修正。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        gate_init: float = 0.0,
    ):
        """
        Args:
            d_model: 隐藏维度
            n_heads: 注意力头数
            d_ff: FFN中间维度 (默认 4 * d_model)
            dropout: dropout率
            gate_init: DDL门控初始偏置
        """
        super().__init__()
        d_ff = d_ff or 4 * d_model

        # Attention子层
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # FFN子层
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # DDL残差层 (替代标准残差)
        self.ddl_attn = DDLLayer(
            d_model, sublayer_type="attention",
            gate_init=gate_init, dropout=dropout
        )
        self.ddl_ffn = DDLLayer(
            d_model, sublayer_type="ffn",
            gate_init=gate_init, dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            attn_mask: 注意力掩码
            key_padding_mask: padding掩码

        Returns:
            [batch, seq_len, d_model]
        """
        # DDL Attention
        def attn_fn(x_normed, **kw):
            attn_out, _ = self.attn(
                x_normed, x_normed, x_normed,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            return attn_out

        x = self.ddl_attn(x, attn_fn)

        # DDL FFN
        x = self.ddl_ffn(x, self.ffn)

        return x


class DDLModel(nn.Module):
    """
    使用DDL的完整Decoder-Only语言模型 / Full Decoder-Only LM with DDL

    对比实验: 用DDL替换所有标准残差连接，观察语言建模质量变化。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        gate_init: float = 0.0,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            DDLTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                gate_init=gate_init,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] token IDs

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        # 因果掩码 / Causal mask
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
# 辅助工具: 对比标准Transformer和DDL的参数量
# Helper: Compare parameter counts between standard and DDL
# ============================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compare_architectures(vocab_size=32000, d_model=512, n_layers=6, n_heads=8):
    """对比标准Transformer与DDL的参数量差异"""
    ddl_model = DDLModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
    )

    ddl_params = count_params(ddl_model)

    # DDL额外参数 = 每层的 gate_proj + target_proj + gate_norm
    extra_per_layer = (
        d_model * d_model + d_model   # gate_proj (weight + bias)
        + d_model * d_model + d_model  # target_proj (weight + bias)
        + 2 * d_model                  # LayerNorm (weight + bias)
    )
    total_extra = extra_per_layer * n_layers

    print(f"DDL Model Parameters: {ddl_params:,}")
    print(f"Extra params per DDL layer: {extra_per_layer:,}")
    print(f"Total extra params ({n_layers} layers): {total_extra:,}")
    print(f"Overhead ratio: {total_extra / (ddl_params - total_extra):.2%}")

    return ddl_model


# ============================================================
# 训练示例 / Training Example
# ============================================================

def train_example():
    """
    简单的训练循环示例，展示如何用DDL替换标准Transformer。
    Simple training loop demonstrating DDL usage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DDLModel(
        vocab_size=10000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1,
        gate_init=0.0,  # 门控初始50%开度
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 模拟训练数据 / Simulated training data
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)

    model.train()
    logits = model(input_ids)

    # 标准语言建模损失 / Standard LM loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1),
    )

    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print(f"Model parameters: {count_params(model):,}")

    # 可视化门控值分布 / Visualize gate value distribution
    with torch.no_grad():
        for i, block in enumerate(model.blocks):
            gate_val = block.ddl_attn.delta_gate.gate_proj.weight.abs().mean()
            print(f"Block {i} attention gate weight magnitude: {gate_val:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Deep Delta Learning (DDL) — Code Reproduction")
    print("=" * 60)

    print("\n--- Architecture Comparison ---")
    compare_architectures()

    print("\n--- Training Example ---")
    train_example()
