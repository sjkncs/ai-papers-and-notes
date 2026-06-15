# 2026 顶会综合报告 — CVPR / ICLR / ICML / NeurIPS

# 2026 Top Conference Summary Report

---

## CVPR 2026

**时间 / Date:** June 1–5, 2026 | **地点 / Location:** Nashville, TN, USA

### 获奖论文 / Award Winners

| 奖项 / Award | 论文 / Paper | 机构 / Affiliation |
|---|---|---|
| **Best Paper** | D4RT: Efficiently Reconstructing Dynamic Scenes | Google DeepMind, UCL, Oxford |
| **Best Student Paper** | TRELLIS.2: Native and Compact Structured Latents for 3D | Microsoft Research, Tsinghua |

### 重点Oral论文 / Key Oral Papers

| 论文 / Paper | 方向 / Area |
|---|---|
| SAM 3D: 3Dfy Anything in Images | 3D Vision |
| NitroGen: Open Foundation Model for Gaming Agents | Agents |
| SAM 3D Body: Robust Full-Body Human Mesh Recovery | Pose Estimation |
| MAMMA: Markerless Multi-person Motion Acquisition | Pose Estimation |
| INSID3: Training-Free Segmentation with DINOv3 | Segmentation |
| MARCO: Navigating Semantic Correspondence | Segmentation |
| VGGT-Segmentor: Geometry-Enhanced Cross-View Segmentation | Segmentation |
| Molmo2: Open VLM with Video Understanding | VLM |

### 重点Highlight论文 / Key Highlights

| 论文 / Paper | 方向 / Area |
|---|---|
| UltraFlux: Native 4K Text-to-Image | Generation |
| HoloCine: Multi-Shot Long Video Narratives | Video Generation |
| FastGS: Training 3DGS in 100 Seconds | 3DGS |
| Transition Models: Rethinking Generative Learning | Diffusion |
| GlyphPrinter: Region-Grouped DPO for Text Rendering | Text Generation |
| OmniVGGT: Omni-Modality Visual Geometry Transformer | Foundation Models |

### 审稿人视角要点 / Reviewer Perspective Notes

- **3D/4D方向大爆发:** Best Paper + Best Student Paper + 多个Oral均在3D方向
- **Gaussian Splatting主流化:** 多篇论文使用3DGS，从研究工具变为生产工具
- **开源成为标配:** TRELLIS.2, Molmo2, NitroGen均强调开源

---

## ICLR 2026

**时间 / Date:** April 23–27, 2026 | **地点 / Location:** Singapore

### Outstanding Papers

| 论文 / Paper | 方向 / Area | 核心贡献 / Core Contribution |
|---|---|---|
| **Transformers are Inherently Succinct** | Theory | 证明Transformer比RNN指数级更简洁 |
| **LLMs Get Lost In Multi-Turn Conversation** | LLM Evaluation | 多轮对话中LLM平均性能下降39% |

### 审稿人视角要点 / Reviewer Perspective Notes

- **理论深度获认可:** Outstanding Paper之一为纯理论工作（形式语言/自动机理论）
- **实证评估的价值:** 另一篇Outstanding Paper为大规模实证研究
- **社区偏好:** 理论和实证并重，"理解为什么"成为评审加分项

### 关键发现摘录 / Key Findings

> **Transformers are Inherently Succinct:**
> "Fixed-precision transformers are exponentially more succinct than both LTL and RNNs. Doubly exponentially more succinct than finite automata. Verification problems are EXPSPACE-complete."

> **LLMs Get Lost In Multi-Turn:**
> "Average drop of 39% across six generation tasks. A minor loss in aptitude and a significant increase in unreliability. When LLMs take a wrong turn, they get lost and do not recover."

---

## ICML 2026

**时间 / Date:** July 2026 (upcoming) | **地点 / Location:** Vancouver, Canada
**投稿量 / Submissions:** 6,500+

### 重点论文 / Key Papers

| 论文 / Paper | 方向 / Area | 亮点 / Highlight |
|---|---|---|
| Self-Supervised Flow Matching (Self-Flow) | Generation | 可扩展多模态合成 |
| Weight-sparse Transformers | Architecture | 可解释稀疏电路 (Leo Gao) |
| Set Diffusion | Generation | AR与Diffusion统一框架 |
| Reasoning Cache | LLM | 推理长度外推 |
| Spurious Rewards | RL | 弱相关reward也能激发推理 |
| Unified Multimodal Pretraining | Pretraining | 从头统一多模态预训练 |
| Any-Diffusion | Generation | 掩码离散扩散统一多模态 |
| Why Are Linear RNNs More Parallelizable? | Theory | RNN类型与复杂度类 |
| Scaling Agentic Verifier | Agents | 执行验证agent for competitive coding |
| Maximum Likelihood RL | RL | Compute-indexed sampling objectives |

---

## NeurIPS 2026

**投稿系统开放 / Submissions Open:** April 2026
**会议时间 / Conference:** December 2026

*待更新 / To be updated*

---

## 跨会议趋势 / Cross-Conference Trends

**中文：**

1. **3D/4D重建:** CVPR最佳论文（D4RT）+ 最佳学生论文（TRELLIS.2）+ ICML Set Diffusion → 3D成为2026年最大热点
2. **LLM可靠性:** ICLR Outstanding Paper（多轮退化）+ Week 1论文（LLMs Aren't Scientists）→ 社区开始系统性审视LLM的局限
3. **Diffusion Everywhere:** ICML的Set Diffusion, Any-Diffusion + DLM路线图 → 扩散模型从视觉扩展到语言和多模态
4. **Agent范式扩展:** NitroGen (游戏) + Scaling Agentic Verifier (代码) → Agent从特定领域走向通用
5. **理论复兴:** ICLR（Transformer succinctness）+ ICML（Linear RNN complexity）→ 形式化理论工作获得高度认可

**English:**

1. **3D/4D reconstruction:** CVPR Best + Best Student + ICML Set Diffusion → 3D is the biggest hotspot of 2026
2. **LLM reliability:** ICLR Outstanding (multi-turn degradation) + DDL week 1 (LLMs as Scientists) → systematic scrutiny of LLM limitations
3. **Diffusion everywhere:** ICML Set Diffusion, Any-Diffusion + DLM roadmap → diffusion extending beyond vision
4. **Agent paradigm expansion:** NitroGen (gaming) + Agentic Verifier (coding) → agents going general
5. **Theory renaissance:** ICLR (Transformer succinctness) + ICML (Linear RNN complexity) → formal theory highly recognized

---

*Last updated: 2026-06-15*
