# Scaling Vision Models Does Not Consistently Improve Localisation-Based Explanation Quality

**arXiv:** [2605.10142](https://arxiv.org/abs/2605.10142)  
**Date:** 2026-05-11  
**Authors:** Mateusz Cedro, Marcin Chlebus

---

## Key Excerpts / 核心摘录

> "We investigate whether increasing the scale of vision models improves the quality of localization-based explanations."
>
> 我们研究增大视觉模型规模是否能提升基于定位的解释质量。

> "Across eleven vision models of varying depth and width, larger scale rarely leads to statistically better explanation maps."
>
> 在 11 个不同深度与宽度的视觉模型中，更大的规模很少带来统计上更优的解释图。

> "Our findings warn against assuming that bigger models are inherently safer or more interpretable."
>
> 我们的发现警告不要假设更大的模型本质上更安全或可解释。

> "Explanation quality should be validated independently of accuracy gains when models are scaled for high-stakes applications."
>
> 当模型因高风险应用而被放大时，解释质量应独立于精度提升进行验证。

---

## Q1 Core Problem / 核心问题

As vision models grow larger, practitioners and regulators often assume that improvements in accuracy are accompanied by better interpretability and safety. However, explanation quality is usually measured independently from scale. The paper tests whether scaling actually improves localization-based explanations such as Grad-CAM and LIME.

随着视觉模型规模增大，从业者和监管者常假设精度提升会伴随更好的可解释性与安全性。然而，解释质量通常独立于规模进行衡量。本文测试缩放是否确实能改善 Grad-CAM 与 LIME 等基于定位的解释。

---

## Q2 Claimed Contributions / 声称贡献

1. **Systematic scaling study.** The authors evaluate eleven vision models with varying depths and widths using multiple localization-based explanation methods.
   **系统的 scaling 研究。** 作者使用多种基于定位的解释方法评估了 11 个不同深度与宽度的视觉模型。

2. **Weak scale-explanation correlation.** They find no consistent or statistically significant improvement in explanation quality as model capacity increases.
   **规模与解释质量弱相关。** 他们发现随着模型容量增加，解释质量没有一致或统计显著的提升。

3. **Safety-oriented warning.** The paper argues that scaling models for high-stakes domains without measuring explanations can create a false sense of reliability.
   **面向安全的警示。** 论文认为，在高风险领域放大模型而不度量解释质量会产生虚假的可靠感。

4. **Metric recommendations.** The authors recommend evaluating localization faithfulness and localization accuracy separately from top-1 accuracy when comparing scaled models.
   **指标建议。** 作者建议在比较放大模型时，单独评估定位忠实度与定位准确率，而非只看 top-1 精度。

---

## Q3 Reviewer Attack Points / 审稿人攻击点

1. **Small model set.** Eleven models may not be enough to draw broad conclusions about scaling laws; the study lacks very large models and different architectural families.
   **模型集合较小。** 11 个模型可能不足以得出关于 scaling law 的广泛结论；研究缺少超大模型与不同架构族。

2. **Limited explanation methods.** Only localization-based methods are tested. Other explanation families, such as concept-based or counterfactual explanations, might scale differently.
   **解释方法有限。** 仅测试基于定位的方法。基于概念或反事实的解释可能呈现不同的 scaling 行为。

3. **Proxy metrics for explanation quality.** Localization accuracy relies on ground-truth bounding boxes, which may not capture all relevant visual evidence; faithfulness metrics can be noisy.
   **解释质量的代理指标。** 定位准确率依赖真实边界框，可能无法捕捉所有相关视觉证据；忠实度指标也可能有噪声。

4. **Correlation vs. causation.** The paper shows association between scale and explanation quality but does not isolate which architectural or training choices drive the observed variation.
   **相关 vs 因果。** 论文展示了规模与解释质量之间的关联，但未分离出哪些架构或训练选择导致了观察到的差异。

5. **No mitigation path.** While the warning is valuable, the paper does not propose concrete methods for improving explanations at scale.
   **缺少改进路径。** 尽管警示有价值，论文未提出在更大规模上改进解释的具体方法。

---

## Q4 PhD Reading Guide / 博士阅读指南

**Time budget / 时间预算:** 1 hour.

1. **Session 1: Setup and metrics.**
   Read Sections 1-2 carefully. Understand which explanation methods are used and how localization quality is quantified.
   **第一节：设定与指标。** 仔细阅读第 1-2 节。理解使用了哪些解释方法以及如何量化定位质量。

2. **Session 2: Results and implications.**
   Focus on the correlation plots and statistical tests. Ask: are the observed differences meaningful in practice, or just noise?
   **第二节：结果与启示。** 关注相关图与统计检验。思考：观察到的差异在实践中是否有意义，还是只是噪声？

3. **Hands-on exercise / 动手练习:**
   Reproduce a small version of the study using `timm` models of different scales on ImageNet validation images. Compute Grad-CAM maps and compare localization accuracy across scales.
   使用不同规模的 `timm` 模型在 ImageNet 验证图像上复现一个小型研究。计算 Grad-CAM 图并比较不同规模的定位准确率。

4. **Extension idea / 拓展方向:**
   Add calibration and robustness metrics to the analysis. Investigate whether better-calibrated models also produce better localization explanations, independent of scale.
   在分析中加入校准与鲁棒性指标。研究校准更好的模型是否也会生成更好的定位解释，而不受规模影响。

---
