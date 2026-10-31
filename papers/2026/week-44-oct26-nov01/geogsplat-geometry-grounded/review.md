# GeoGSplat: Geometry-Grounded Non-Pixel-Aligned Gaussian Splatting

**arXiv:** [2610.36760](https://arxiv.org/abs/2610.36760)
**Authors:** Jiale Wang, Chen Zhao, Haoyang Zhang, Yanhao Wu, Fei He, Yanhu Shan, Rui Wu, Wei Ke, Tong Zhang
**Categories:** cs.CV

---

## Q1: 它真正想解决的问题是什么？ / What problem does it really solve?

**核心问题：** 前馈高斯泼溅中，像素对齐预测导致原语数量随分辨率和视角数线性增长，如何在非像素对齐的前提下保证高斯中心的几何准确性？

**深层动机：** 前馈 GS 方法无需逐场景优化，推理速度快，但面临两难：像素对齐方法精确但冗余（原语数 ∝ 分辨率 × 视角数），非像素对齐方法紧凑但高斯中心位置不准，导致大视角变化下的渲染伪影。GeoGSplat 试图通过引入显式几何先验来打破这一两难。

---

## Q2: 它声称的贡献是什么？ / What are the claimed contributions?

1. **双分支编码器：** Anchors 分支捕获粗略 3D 结构 + Triplane 分支捕获精细场景特征
2. **层次化解码器：** 逐步解码密集 3D 点和高斯原语
3. **全面的 SOTA：** 在 DTU 和 NRGBD 上同时超越像素对齐和非像素对齐的 GS 方法
4. **大视角泛化：** 目标视角离输入相机越远，几何先验的优势越明显

---

## Q3: 最可能被 reviewer 攻击的地方在哪里？ / Most likely reviewer attack points?

1. **几何先验的来源和质量：** 双分支中的几何信息来自哪里？如果依赖预训练的深度估计模型，那么几何先验的质量上限受限于该模型
2. **计算效率：** 双分支 + 层次化解码是否引入了显著的额外计算开销？与前馈方法的"快速"初衷是否矛盾？
3. **与 pixel-aligned 方法的公平对比：** 像素对齐方法在固定原语数量下是否也能达到类似质量？
4. **泛化场景覆盖：** DTU 和 NRGBD 主要是室内场景，室外和大规模场景的泛化性如何？

---

## Q4: 同方向博士生应精读哪些、跳过哪些？ / PhD reading guide

**精读：**
- 双分支编码器的设计，特别是 Anchors 和 Triplane 如何协作
- 几何先验的注入方式和层次化解码的渐进策略
- 大视角变化下的泛化实验——这是核心优势的体现

**跳读：**
- 标准 NVS 指标的详细数值表（关注趋势和大视角实验）

---

## 与本研究方向的关联

直接相关：3D 视觉与 Gaussian Splatting 方向。GeoGSplat 的几何先验注入方法对 3D 场景重建的质量和效率有直接帮助，尤其是在前馈推理场景下。
