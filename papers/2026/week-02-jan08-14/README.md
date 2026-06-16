# Week 2: 2026-01-08 — 2026-01-14 / 第2周

## 本周学术动态 / Weekly Highlights

**中文：** 2026年第二周，轻量化语言模型和多模态理解成为焦点。Ministral 3系列（3B/8B/14B）展示了参数高效密集模型在资源受限场景下的竞争力；MHLA通过token级多头机制恢复了线性注意力的表达力；BabyVision揭示了当前多模态大模型在视觉推理上的"倒置能力"问题；STEP3-VL-10B以100亿参数挑战了"紧凑模型推理能力有限"的传统认知。

**English:** Week 2 of 2026 spotlighted lightweight LLMs and multimodal understanding. The Ministral 3 series (3B/8B/14B) demonstrated parameter-efficient dense models' competitiveness under resource constraints; MHLA restored linear attention's expressivity via token-level multi-head; BabyVision exposed the "inverted competence profile" of current MLLMs; STEP3-VL-10B challenged the belief that compact models are inherently limited in reasoning.

## 重点论文 / Featured Papers

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [Ministral 3](https://arxiv.org/abs/2601.08584) | LLM Architecture | [review.md](ministral-3/review.md) | [code/](ministral-3/code/) |
| 2 | [MHLA: Restoring Expressivity of Linear Attention](https://arxiv.org/abs/2601.07832) | Efficient Attention | [review.md](mhla-linear-attention/review.md) | [code/](mhla-linear-attention/code/) |
| 3 | [BabyVision: Visual Reasoning Beyond Language](https://arxiv.org/abs/2601.06521) | VLM Evaluation | [review.md](babyvision/review.md) | — |
| 4 | [STEP3-VL-10B](https://arxiv.org/abs/2601.09668) | Multimodal VLM | [review.md](step3-vl-10b/review.md) | — |

## 其他论文 / Other Papers

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| Video Generation Models in Robotics | Survey / Robotics | 视频生成模型在机器人世界建模中的应用综述 |
| Inference-Time Alignment for Diffusion Models | Diffusion / Alignment | Doob's Matching实现推理时扩散模型对齐 |
| Constitutional Classifiers++ | Safety / Alignment | 生产就绪的激活探针用于越狱检测 |

## 领域趋势 / Trends

1. 小模型的逆袭 / Small models fight back — 3B-14B dense models matching larger MoE
2. 线性注意力复兴 / Linear attention renaissance — new architectures challenge softmax dominance
3. 多模态评估的冷水 / Multimodal evaluation reality check — benchmarks expose fundamental gaps
4. 推理能力与参数量的脱钩 / Reasoning decoupled from scale — compact VLMs challenging assumptions
5. 扩散模型推理时对齐 / Inference-time alignment for diffusion — adapting models without retraining

## 重要时间节点 / Key Dates

| 事件 / Event | 日期 / Date |
|---|---|
| ICLR 2026 Notification | Jan 25, 2026 |
| AAAI 2026 Conference | Feb–Mar 2026 |
| NeurIPS 2026 Submissions Open | Apr 2026 |
