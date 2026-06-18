# CLIP 论文复现 / CLIP Paper Reproduction

基于 Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021) 的简化复现。

## 文件 / Files

- `clip_reproduction.py` — 完整实现，包含:
  - `PatchEmbedding` — 图像patch嵌入
  - `MultiHeadAttention` — 多头注意力
  - `TransformerBlock` — Transformer编码器块
  - `ImageEncoder` — 简化版ViT图像编码器
  - `TextEncoder` — 简化版Transformer文本编码器
  - `SimpleCLIP` — CLIP模型（对比学习 + 投影头 + InfoNCE损失）
  - `zero_shot_classify` — 零样本分类

## 运行 / Run

```bash
pip install torch
python clip_reproduction.py
```

## 核心架构 / Core Architecture

```
Image ──→ ViT Encoder ──→ Projection ──→ ┐
                                          ├─ Cosine Similarity → InfoNCE Loss
Text  ──→ Transformer ──→ Projection ──→ ┘
```

## 关键公式 / Key Formulas

- **InfoNCE Loss:** L = -log[exp(sim(I,T)/τ) / Σexp(sim(I,T_j)/τ)]
- **Temperature:** τ is a learnable parameter (initialized at 0.07)
- **Similarity:** cos(I_proj, T_proj) with L2 normalization
