# GEAR: Guided End-to-End AutoRegression for Image Synthesis

**Authors:** Bin Lin, Zheyuan Liu, Chenguo Lin, Sixiang Chen, Yunyang Ge, Yunlong Lin, Jianwei Zhang, Miles Yang, Zhao Zhong, Liefeng Bo, Li Yuan  
**Affiliations:** Peking University, Tencent Hunyuan  
**arXiv:** [2606.32039](https://arxiv.org/abs/2606.32039)  
**Date:** June 30, 2026  
**Code:** [Tencent-Hunyuan/GEAR](https://github.com/Tencent-Hunyuan/GEAR)  

---

## Q1: 它真正想解决的问题是什么？ / What problem does it really solve?

**中文：** 当前视觉生成模型（无论是基于离散token的自回归模型，还是基于连续latent的扩散模型）几乎都采用两阶段训练：先训练一个tokenizer/VAE用于图像重建，然后将其冻结，再在其上训练生成模型。这种解耦导致tokenizer的优化目标（重建质量）与生成器的优化目标（latent序列的可预测性）不一致。具体来说，重建需要高方差、细节丰富的latent；而生成则希望latent分布简单、结构可预测。GEAR试图打破这一分离，让tokenizer和自回归生成器能够**联合端到端训练**，从而让tokenizer“感知”到下游生成器的需求。

**English:** Modern visual generative models—both autoregressive transformers over discrete tokens and diffusion models over continuous latents—are almost universally trained in two stages: a tokenizer/VAE is first trained for reconstruction and then frozen, after which a generator is trained on top. This decoupling creates a mismatch: reconstruction favors high-variance, detail-rich latents, while generation favors simple, predictable structure. GEAR aims to dissolve this boundary by jointly training a vector-quantized (VQ) tokenizer and an autoregressive generator **end-to-end**, so the tokenizer becomes aware of what the downstream generator finds easy to model.

> "Visual generative models are typically trained in two stages. A tokenizer is first trained for reconstruction and then frozen, after which a generator is trained on its discrete indices or continuous latents. This decoupling leaves the tokenizer unaware of what the generator finds easy to model."

---

## Q2: 它声称的贡献是什么？ / What are the claimed contributions?

**中文：**
1. **端到端训练框架：** 提出GEAR，首次在离散VQ-AR setting中实现tokenizer与AR生成器的联合端到端训练。核心创新是“dual read-out”机制：硬分支（one-hot）用于不可微的next-token prediction；软分支（temperature-scaled soft assignment）用于可微的representation alignment，从而绕过VQ argmax的不可微障碍。
2. **representation alignment的位置被重新定位：** 与扩散侧方法（REPA-E、VA-VAE、MAETok）让latent本身更语义化不同，GEAR让tokenizer的features变得更不像DINOv2，而让AR生成器的features更像DINOv2（patch-level），从而把对齐负担从tokenizer转移到了AR。
3. **速度与泛化性：** ImageNet gFID收敛速度相对于LlamaGen-REPA提升最高达10倍；且方法可泛化到多种quantizer（VQVAE、LFQ、IBQ）和text-to-image任务。

**English:**
1. **End-to-end training framework:** GEAR jointly trains a VQ tokenizer and an AR generator end-to-end in the discrete setting for the first time. The key innovation is a dual read-out: a hard one-hot branch carries the non-differentiable next-token prediction loss, while a soft temperature-scaled branch carries a differentiable representation-alignment loss that bypasses the argmax bottleneck.
2. **Relocation of representation alignment:** Unlike diffusion-side methods that make the latent itself more semantic, GEAR makes the tokenizer's features *less* DINOv2-like and the AR generator's features *more* DINOv2-like at the patch level, shifting the alignment burden from the tokenizer to the AR model.
3. **Speed and generality:** GEAR speeds up ImageNet gFID convergence by up to 10× over LlamaGen-REPA and generalizes across quantizers (VQVAE, LFQ, IBQ) and to text-to-image generation.

> "We present GEAR (Guided End-to-end AutoRegression), which trains a vector-quantized (VQ) tokenizer and an autoregressive (AR) generator jointly and end-to-end, guided by representation alignment."

> "GEAR speeds up ImageNet gFID convergence by up to 10× relative to the strong LlamaGen-REPA baseline, learns markedly better patch-level and spatially-coherent features, and generalizes across quantizers (VQVAE, LFQ, IBQ) and to text-to-image generation."

---

## Q3: 最可能被reviewer攻击的地方在哪里？ / Most likely reviewer attack points?

**中文：**
1. **消融不够彻底：** 论文对比了naive end-to-end with STE（会崩溃），但没有充分对比其他可能的梯度桥接方案（如Gumbel-softmax、concrete relaxation、EOSTok的加权NTP+pixel loss等）。Reviewer可能会质疑dual read-out是否只是众多可行方案之一，且恰好被选中。
2. **10×加速的来源分解不清：** gFID收敛10×提升非常抢眼，但究竟多少来自end-to-end training，多少来自REPA loss本身，多少来自模型规模或训练trick？Figure 1显示GEAR在1.5M steps就接近LlamaGen-REPA 7M steps的水平，但没有给出详细的样本效率分解。
3. **Text-to-image迁移证据有限：** 论文声称ImageNet上end-to-end训练的tokenizer能加速text-to-image generation，但实验篇幅和对比方法可能不足。Reviewer可能要求更多T2I基准（如COCO、PartiPrompts）和与原生T2I tokenizer的对比。
4. **Soft branch的温度选择敏感：** soft assignment依赖温度参数τ。如果τ选择不当，soft branch可能过度平滑或无法有效传递梯度。论文对τ的敏感性分析可能不够充分。

**English:**
1. **Incomplete ablations:** The paper compares against a naive end-to-end STE baseline that collapses, but does not thoroughly compare against other gradient-bridging alternatives (Gumbel-softmax, concrete relaxation, EOSTok's weighted NTP + pixel loss, etc.). Reviewers may ask whether the dual read-out is merely one of many plausible solutions.
2. **Unclear source of 10× speed-up:** The 10× gFID convergence gain is impressive but not fully decomposed. How much comes from end-to-end training per se, from the REPA loss, from scale, or from training tricks? More sample-efficiency ablations would strengthen the claim.
3. **Limited text-to-image evidence:** The claim that ImageNet end-to-end training transfers to text-to-image generation is supported by limited experiments. Reviewers may ask for more T2I benchmarks (COCO, PartiPrompts) and comparisons against native T2I tokenizers.
4. **Temperature sensitivity of soft branch:** The soft assignment relies on a temperature τ. If poorly chosen, the soft branch may be too smooth or fail to transmit gradients. A thorough sensitivity analysis for τ may be missing.

> "The same opportunity exists for AR generation over discrete VQ tokens, but it is fundamentally harder. The map from a VQ index to the AR input is a non-differentiable arg max, so one cannot back-propagate any signal from the AR generator to the tokenizer, and the obvious remedy, a straight-through estimator (STE), is unstable and collapses the codebook in our joint setting."

---

## Q4: 同方向博士生应精读哪些、跳过哪些？ / PhD reading guide: read carefully vs. skip?

**中文：**
- **必读 / Read carefully:**
  - **Section 1 & 2:** 动机和相关工作梳理非常清晰，特别是对比AR和扩散两阶段训练的paragraph。
  - **Section 3.2:** dual hard/soft assignment的数学细节是核心创新，需要精读。
  - **Section 4.3:** representation analysis（tokenizer变less semantic，AR变more patch-level semantic）是本文最insightful的部分。
  - **Figure 2:** 清晰对比了传统、naive end-to-end、GEAR三种pipeline。
- **可跳 / Skim or skip:**
  - **Appendix中的implementation details:** 除非你要复现，否则可以跳过。
  - **大量ImageNet gFID表格:** 抓住主要结论即可，不必逐行对比。
  - **LFQ/IBQ quantizer的消融:** 如果只做VQVAE方向，可以略读。

**English:**
- **Read carefully:**
  - **Sections 1 & 2:** The motivation and related work are exceptionally clear, especially the contrast between AR and diffusion two-stage training.
  - **Section 3.2:** The mathematical details of the dual hard/soft assignment are the core innovation.
  - **Section 4.3:** The representation analysis showing the tokenizer becomes less semantic while the AR becomes more patch-level semantic is the most insightful part.
  - **Figure 2:** It cleanly contrasts the traditional, naive end-to-end, and GEAR pipelines.
- **Skim or skip:**
  - **Appendix implementation details:** Skip unless you are reproducing.
  - **Numerous ImageNet gFID tables:** Grasp the main conclusions; no need to compare every row.
  - **LFQ/IBQ quantizer ablations:** Skim if you only care about VQVAE.

---

## Key Original Quotes / 关键原文摘录

> "A hard, one-hot branch trains the AR with next-token prediction, while a differentiable soft branch carries a representation-alignment loss that flows back to guide only the tokenizer. The AR model thereby steers its tokenizer toward an index distribution it can predict more easily."

> "What this guidance actually does is, perhaps surprisingly, the opposite of the diffusion-side recipe. On the diffusion side, REPA-E, VA-VAE and MAETok make the latent more semantic by aligning it to a pretrained encoder. In GEAR the tokenizer's own features instead become less DINOv2-like, most strongly at the patch level."

> "End-to-end guidance thus shifts the alignment burden from the tokenizer to the AR: the tokenizer need not look semantic, only emit tokens the AR can predict, and this local, patch-level structure is exactly what makes next-token prediction easy."
