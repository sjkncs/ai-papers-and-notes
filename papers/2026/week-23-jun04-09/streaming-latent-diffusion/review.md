# Streaming Latent Diffusion for Minute-Scale Video Generation

**arXiv:** 2606.04567  
**Date:** 2026-06-07  
**Authors:** Hao Chen, Sarah Müller, Jun-Yan Zhu, Katerina Fragkiadaki, Antonio Torralba  
**Institution:** Carnegie Mellon University, MPI for Informatics, MIT

---

## Key Excerpts / 核心摘录

> "Generating minute-scale video with diffusion models requires more than scaling temporal context; it demands a recurrent memory that propagates motion and appearance across arbitrarily long horizons." / "用扩散模型生成分钟级视频不仅需要扩大时序上下文，更需要一种递归记忆，在任意长的时间跨度上传播运动与外观。"

> "Streaming Latent Diffusion (SLD) processes video as a sequence of short clips, conditioned on a compact latent state that summarizes past dynamics." / "流式潜空间扩散（SLD）将视频处理为一系列短视频片段，并以一个紧凑的潜状态为条件，该状态总结了过去的动态。"

> "At 30 fps, our largest model generates coherent 90-second videos without explicit storyboarding, with a subject-consistency score of 0.87 on the Long-VideoBench." / "在 30 fps 下，我们最大的模型无需显式故事板即可生成连贯的 90 秒视频，在 Long-VideoBench 上的主体一致性得分达 0.87。"

> "Training stability is achieved through a temporal consistency discriminator that operates directly in latent space, avoiding the cost of full video decoding." / "通过在潜空间直接操作的时序一致性判别器实现训练稳定性，避免了解码完整视频的开销。"

> "We release SLD-S, SLD-B, and SLD-L checkpoints together with a streaming inference engine that runs on consumer GPUs." / "我们发布 SLD-S、SLD-B 和 SLD-L 检查点，以及可在消费级 GPU 上运行的流式推理引擎。"

---

## Q1 Core Problem / 核心问题

**English:**  
Video diffusion models have advanced to tens of seconds, but minute-scale generation remains unstable: autoregressive extension leads to motion drift, identity drift, and scene discontinuities; meanwhile, naively increasing temporal context length is computationally prohibitive. The paper asks how to design a diffusion architecture that can generate arbitrarily long videos in a streaming fashion, preserving long-term visual consistency without requiring the entire video to be processed in parallel.

**中文：**  
视频扩散模型已能生成数十秒视频，但分钟级生成仍不稳定：自回归扩展会导致运动漂移、身份漂移和场景不连续；而简单地增加时序上下文长度在计算上又过于昂贵。本文研究如何设计一种流式扩散架构，使其能够以流式方式生成任意长视频，并在无需并行处理整个视频的前提下保持长期视觉一致性。

---

## Q2 Claimed Contributions / 声称贡献

1. **Streaming latent diffusion architecture / 流式潜空间扩散架构**: A diffusion model that consumes and produces short latent clips while maintaining a recurrent state vector. The state is updated by a lightweight temporal consistency module.
2. **Latent-space temporal consistency discriminator / 潜空间时序一致性判别器**: A discriminator trained adversarially to enforce frame-to-frame coherence without decoding latents to pixels, reducing training cost.
3. **Arbitrary-length inference / 任意长度推理**: The model can stream generation indefinitely at inference time, conditioned on an initial frame and text prompt, with bounded memory growth.
4. **Strong long-video metrics / 优秀的长视频指标**: On Long-VideoBench and InternVid-V2, SLD-L outperforms prior open models in subject consistency, motion smoothness, and text–video alignment.
5. **Consumer-GPU inference / 消费级 GPU 推理**: The authors release an engine that runs SLD-S at 512x320 resolution on a single RTX 4090, lowering the barrier to long-form video research.

---

## Q3 Reviewer Attack Points / 审稿人攻击点

1. **Quantitative evaluation on synthetic benchmarks / 在合成基准上的定量评估**: Long-VideoBench and InternVid-V2 are useful but may not capture real-world distribution shifts. The paper lacks human evaluation on open-ended prompts, which is the true test for long-form generation.
2. **Recurrent state collapse / 递归状态崩溃**: Although memory is bounded, the recurrent state could collapse to a generic distribution after many steps, causing repetitive motion. The authors show 90-second examples but do not characterize failure modes at 3–5 minutes.
3. **Comparison with stronger baselines / 与更强基线的比较**: The main comparisons are against earlier open models. Proprietary systems such as Sora-like models are not included for legal reasons, which is understandable but limits the reader's sense of absolute progress.
4. **Training data scale / 训练数据规模**: The largest model is trained on a curated internal dataset. The exact size and filtering criteria are only partially disclosed, making it hard to assess reproducibility.
5. **Temporal discriminator artifacts / 时序判别器伪影**: The latent-space discriminator may introduce high-frequency temporal flicker not captured by standard metrics. A visual inspection of released samples will be the real judge.

---

## Q4 PhD Reading Guide / 博士阅读指南

**English:**  
Begin with the architecture figure in Section 3 to understand how the recurrent state is updated between clip windows. Then read Section 4.3 carefully: the latent-space discriminator is the key technical novelty that makes long-horizon training feasible. Try to re-implement the streaming inference loop on a small latent autoencoder; the main insight is to treat temporal attention as a recurrent operation rather than a full attention block. If you work on video generation, ask whether your current model can be converted to a streaming form by replacing full temporal attention with a recurrent bottleneck. For evaluation, supplement automated metrics with carefully designed human studies on long-form consistency.

**中文：**  
从第 3 节的架构图开始，理解递归状态如何在视频片段窗口之间更新。然后仔细阅读 4.3 节：潜空间判别器是使长程训练可行的关键技术创新。尝试在小型潜自编码器上复现流式推理循环；核心洞见是将时序注意力视为递归操作而非完整注意力块。若从事视频生成研究，请思考当前模型是否可通过将完整时序注意力替换为递归瓶颈而转换为流式形式。在评估方面，应用针对长程一致性精心设计的人工研究来补充自动指标。
