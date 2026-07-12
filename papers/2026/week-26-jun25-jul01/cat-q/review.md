# CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs

**Authors / 作者：** Shigeng Wang, Chao Li, Yangyuxuan Kang, Jiawei Fan, Anbang Yao  
**arXiv / 链接：** [https://arxiv.org/abs/2606.26650](https://arxiv.org/abs/2606.26650)  
**Published / 发表时间：** 2026-06-25  
**Venue / 会议：** ICML 2026 (Oral)

---

## 摘要 / Abstract（原文 verbatim）

> In this paper, we present CAT-Q, Cost-efficient and Accurate Ternary Quantization, for compressing and accelerating LLMs. Unlike existing state-of-the-art ternary quantization methods that rely on data-intensive and costly quantization-aware training to mitigate severe performance degradation, CAT-Q is a simple yet effective post-training quantization scheme that is readily applicable to LLMs with diverse architectures and model sizes. It has two key components, learnable modulation (LM) and softened ternarization (ST), which are coupled from an optimization perspective. LM leverages a composition of learnable factors to modulate the distribution of pre-trained high-precision weights and the ternary threshold, making them less sensitive to ternarization. ST further introduces a differentiable transition function to guide the ternarization process toward stable convergence. We show that, for pre-trained LLMs with 1.7B to 8B parameters, CAT-Q can efficiently quantize them into ternary models using only 512 calibration samples, while achieving superior performance than the seminal BitNet 1.58-bit v1 and v2 families (with 1.3B to 7B parameters) trained with 100B tokens, yielding about a 100,000X reduction in training tokens. Moreover, we show for the first time that CAT-Q can quantize much larger pre-trained LLMs having 14B to 235B parameters into leading ternary models within just 8 to 60 hours on 8 A100-80GB GPUs. Code is available at https://github.com/IntelChina-AI/BitTern.

---

## 为什么选它做代码复现？ / Why reproduce this paper in code?

**中文：** CAT-Q 是一篇 ICML 2026 Oral，核心算法清晰（可学习调制 + 软三值化），且可以用纯 PyTorch 在小型 LLM 上演示其关键思想。相比需要真实机器人数据或闭源 API 的论文，CAT-Q 更适合作为“端到端可运行”的复现示例。本目录下的 `code/cat_q_demo.py` 提供了一个最小可运行实现，展示如何在 GPT-2 上将部分线性层权重量化为 {-t, 0, +t}。

**English:** CAT-Q is an ICML 2026 Oral paper with a clear core algorithm (learnable modulation + softened ternarization) and can be demonstrated in pure PyTorch on a small LLM. Compared to papers requiring real robot data or closed APIs, CAT-Q is better suited as an end-to-end runnable reproduction example. The script `code/cat_q_demo.py` in this directory provides a minimal runnable implementation showing how to quantize parts of a GPT-2 model's linear weights to {-t, 0, +t}.

---

## 复现脚本说明 / Reproduction script notes

- **依赖 / Dependencies**: `torch`, `transformers`
- **运行 / Run**: `python code/cat_q_demo.py`
- **内容 / Contents**:
  - `softened_ternarize`: 用 sigmoid 近似硬三值化阈值的可微函数。
  - `TernaryLinear`: 将 `nn.Linear` 替换为带可学习 scale/threshold 的三值线性层。
  - `replace_linear_with_ternary`: 自动替换模型中的目标线性层。
  - `calibrate_ternary_layers`: 用少量校准数据优化 scale 和 threshold。

- **局限 / Limitations**:
  - 演示仅使用 logits/权重 MSE 作为损失；论文原实现可能使用 perplexity、层输出 MSE 或端到端交叉熵。
  - 未实现分组量化、异常值处理、KV Cache 量化等工程细节。
  - 使用 GPT-2 仅为演示；论文实验覆盖 1.7B–235B 参数模型。

---

## 关键收获 / Key takeaways

1. **后训练量化（PTQ）即可达到 QAT 效果**：CAT-Q 用 512 条校准样本和几小时训练，就能媲美用 100B token 训练的 BitNet 1.58 系列，这对端侧部署极具吸引力。
2. **LM + ST 的耦合设计**：不是单独调 scale 或单独做软化，而是把“调制权重分布”和“软化三值化”放在同一个优化目标里联合求解。
3. **工程意义重大**：首次把三值 PTQ 推到 235B 参数规模，意味着未来超大模型也可能以极低比特在消费级硬件上推理。

---

## 代码 / Code

- [code/cat_q_demo.py](code/cat_q_demo.py)
