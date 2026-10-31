# GraftRay: Physically Based Mesh Insertion into Gaussian Splatting Scenes via Path Tracing

**arXiv:** [2610.37036](https://arxiv.org/abs/2610.37036)
**Authors:** Jinxin Ai, Matthias Niessner
**Categories:** cs.CV, cs.GR

---

## Q1: 它真正想解决的问题是什么？ / What problem does it really solve?

**核心问题：** 如何将虚拟网格物体物理一致地插入到 2D/3D 高斯泼溅重建的场景中——使插入物体被场景光照照亮，同时向场景投射真实阴影？

**深层动机：** AR 内容创建、合成数据生成等应用需要将虚拟物体无缝融入真实场景。现有方法要么在图像空间做后处理（牺牲多视角一致性），要么用环境贴图近似场景光照（无法表示窗户、灯具等近场面光源）。GraftRay 选择正面解决物理光照问题。

---

## Q2: 它声称的贡献是什么？ / What are the claimed contributions?

1. **扩散先验恢复场景材质和法线：** 从高斯场景中提取物理属性
2. **显式建模空间扩展发射体：** 闭式求解扩展光源辐射
3. **混合渲染器：** 联合路径追踪场景高斯和插入网格，产生视角一致的着色、互反射和软阴影
4. **FIPT 上的 SOTA：** 在渲染保真度和阴影精度上超越 ComGS 和 EAG-PT

---

## Q3: 最可能被 reviewer 攻击的地方在哪里？ / Most likely reviewer attack points?

1. **渲染效率：** 路径追踪的计算开销 vs 实时渲染需求之间的矛盾
2. **材质恢复精度：** 扩散先验恢复的材质属性在复杂材质（金属、玻璃）上的准确性
3. **场景复杂度限制：** 仅在有限室内场景上验证，大规模或室外场景的泛化性
4. **与动态场景的结合：** 如果场景高斯是动态的（如 4D GS），插入物体如何跟随？

---

## Q4: 同方向博士生应精读哪些、跳过哪些？ / PhD reading guide

**精读：**
- 物理光照模型的完整推导——从场景高斯中提取材质和光源的方法
- 混合渲染器的设计——如何联合处理高斯和网格
- FIPT 上的定量对比，特别是阴影精度指标

**跳读：**
- 标准渲染指标的详细对比（重点关注阴影和光照一致性）

---

## 与本研究方向的关联

直接相关：3D 视觉与 Gaussian Splatting。GraftRay 解决的是 3DGS 应用落地的关键问题——如何将虚拟内容融入真实场景，对 AR/VR 和数字孪生应用有直接价值。
