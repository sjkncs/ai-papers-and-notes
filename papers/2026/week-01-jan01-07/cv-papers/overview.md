# CV论文概览 — Week 1 / Computer Vision Papers Overview — Week 1

## 2026-01-01 — 2026-01-07 | arXiv cs.CV

---

## 4D世界模型与重建 / 4D World Models & Reconstruction

### TeleWorld: Towards Dynamic Multimodal Synthesis with a 4D World Model
- **方向:** 4D世界模型，多模态合成
- **亮点:** 首个支持动态多模态合成的4D世界模型
- **关注理由:** 4D世界模型是2026年CV的核心热点之一

### Spatial4D-Bench: A Versatile 4D Spatial Intelligence Benchmark
- **方向:** 4D空间智能评测
- **亮点:** 首个系统性4D空间智能benchmark
- **关注理由:** 为4D理解提供了标准化评测，后续工作会广泛引用

### NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos
- **方向:** 4D世界模型 + 野外视频
- **亮点:** 利用野外单目视频增强4D世界模型的泛化能力

### SV-GS: Sparse View 4D Reconstruction with Skeleton-Driven Gaussian Splatting
- **方向:** 稀疏视角4D重建
- **亮点:** 骨架驱动的3DGS方法处理稀疏视角问题

---

## 3D生成与处理 / 3D Generation & Processing

### MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing
- **方向:** 3D形变
- **亮点:** 利用结构化latent实现任意3D形变，与TRELLIS.2等工作方向一致

### Mask-Conditioned Voxel Diffusion for Joint Geometry and Color Inpainting
- **方向:** 3D修复
- **亮点:** 体素扩散同时处理几何和颜色修复

### Compressed Map Priors for 3D Perception
- **方向:** 3D感知
- **亮点:** 压缩地图先验提升3D感知效率

---

## 生成模型 / Generative Models

### It's Never Too Late: Noise Optimization for Collapse Recovery in Trained Diffusion Models
- **方向:** Diffusion模型修复
- **亮点:** 对已训练好的扩散模型的collapse问题进行后训练修复

### FreeText: Training-Free Text Rendering in Diffusion Transformers
- **方向:** 文本渲染
- **亮点:** 免训练的DiT文本渲染，利用attention定位和频谱字形注入

### DynaDrag: Dynamic Drag-Style Image Editing by Motion Prediction
- **方向:** 图像编辑
- **亮点:** 通过运动预测实现动态拖拽式编辑

---

## 视觉理解 / Visual Understanding

### From Sight to Insight: Improving Visual Reasoning via RL
- **方向:** VLM推理
- **亮点:** 通过强化学习提升多模态模型的视觉推理能力

### AEGIS: Exploring the Limit of World Knowledge for Unified Multimodal Models
- **方向:** VLM评测
- **亮点:** 系统评测统一多模态模型的世界知识能力边界

### FaithSCAN: Model-Driven Single-Pass Hallucination Detection for VQA
- **方向:** VQA幻觉检测
- **亮点:** 单次推理检测VQA幻觉，模型驱动方法

---

## 检测与分割 / Detection & Segmentation

### Attention to Detail: Global-Local Attention for AI-Generated Image Detection
- **方向:** AI生成图检测
- **亮点:** 全局-局部注意力结合的检测方法

### Boosting Segment Anything Model for Visually Non-Salient Scenarios
- **方向:** SAM增强
- **亮点:** 提升SAM在非显著场景下的分割能力

---

## Agent与RL / Agents & RL

### CPPO: Contrastive Perception Policy Optimization for VLM Agents
- **方向:** VLM Agent
- **亮点:** 对比感知策略优化，提升VLM agent的环境理解

---

## 领域观察 / Observations

1. **4D成为新焦点:** 至少4篇论文直接涉及4D世界模型/重建/评测
2. **3DGS持续火热:** Gaussian Splatting在4D重建中的应用
3. **Diffusion后训练优化:** 开始关注训练后修复和优化
4. **VLM能力评测:** 多篇论文聚焦VLM的能力边界和幻觉问题
