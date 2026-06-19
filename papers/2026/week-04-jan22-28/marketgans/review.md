# MarketGANs: Multivariate Financial Time-Series Data Augmentation Using Generative Adversarial Networks

> **arXiv:** [2601.17773](https://arxiv.org/abs/2601.17773) | **日期 / Date:** 2026-01-23 | **作者 / Authors:** Jeonggyu Huh, Seungwon Jeong, Hyun-Gyoon Kim, Hyeng Keun Koo, Byung Hwa Lim

---

## 关键摘录 / Key Excerpts

> 1. "We generate asset returns as a single joint vector to accurately maintain extreme co-movements and cross-sectional correlations that are destroyed by independent asset-level generation."
>    / "我们将资产收益率作为单一联合向量生成，以准确保持被独立资产级生成所破坏的极端联动和截面相关性。"

> 2. "Our factor-based generative framework embeds financial inductive biases — stochastic time-varying factor loadings and volatilities — to capture variance clustering and leverage effects."
>    / "我们的因子生成框架嵌入了金融归纳偏置——随机时变因子载荷和波动率——以捕获方差聚集和杠杆效应。"

> 3. "Experiments with US stocks demonstrate superior replication of market characteristics compared to standard bootstrap techniques, yielding enhanced covariance approximations for portfolio optimization."
>    / "美股实验表明，相比标准bootstrap技术，我们的方法更优地复制了市场特征，为组合优化提供了增强的协方差近似。"

---

## Q1: 核心问题 / Core Problem

**中文：**
金融时序数据面临严重的数据稀缺问题——历史数据有限，极端事件罕见但影响巨大。传统数据增强方法（如bootstrap、block bootstrap）存在根本缺陷：

1. 破坏时序依赖结构（如波动率聚集）
2. 无法生成新的市场状态（regime）
3. 难以保持资产间的截面相关性
4. 对尾部事件的复制能力不足

MarketGANs提出将金融因子模型（如Fama-French三因子）的经济学先验嵌入GAN架构，实现多资产联合时序生成，同时保持：
- 截面相关性结构
- 时序波动率动态
- 尾部依赖（极端联动）
- 因子载荷的时变特性

**English:**
Financial time series suffers from severe data scarcity — limited history, rare but impactful extreme events. Traditional augmentation methods (bootstrap, block bootstrap) have fundamental limitations:
1. Destroy temporal dependence structures (e.g., volatility clustering)
2. Cannot generate new market regimes
3. Fail to preserve cross-sectional correlations
4. Poor replication of tail events

MarketGANs embeds financial factor model priors (e.g., Fama-French 3-factor) into GAN architecture for joint multi-asset time series generation while preserving:
- Cross-sectional correlation structure
- Temporal volatility dynamics
- Tail dependence (extreme co-movements)
- Time-varying factor loadings

---

## Q2: 核心贡献 / Claimed Contributions

**中文：**

1. **因子嵌入GAN架构 (Factor-Embedded GAN Architecture)**：
   - 生成器输出因子载荷β(t)和因子收益f(t)，而非直接生成资产收益
   - 通过 r(t) = β(t)·f(t) + ε(t) 构建资产收益
   - 保证生成数据遵循资产定价理论

2. **时序卷积网络骨干 (TCN Backbone)**：
   - 因果卷积确保不泄露未来信息
   - 膨胀卷积捕获长期依赖
   - 残差连接稳定训练

3. **多尺度判别器 (Multi-Scale Discriminator)**：
   - 时序判别器：检验单资产时序特性
   - 截面判别器：检验同时刻资产间相关性
   - 联合判别器：检验整体数据分布

4. **金融特异性损失 (Finance-Specific Losses)**：
   - 协方差匹配损失
   - 尾部依赖损失（极值相关性）
   - 波动率聚集损失（GARCH效应）

**English:**

1. **Factor-Embedded GAN Architecture**: Generator outputs factor loadings β(t) and factor returns f(t) rather than direct asset returns. Constructs returns via r(t) = β(t)·f(t) + ε(t), ensuring adherence to asset pricing theory.

2. **TCN Backbone**: Causal convolutions prevent future leakage; dilated convolutions capture long-term dependencies; residual connections stabilize training.

3. **Multi-Scale Discriminator**: Temporal discriminator (single asset time series), cross-sectional discriminator (same-timepoint correlations), joint discriminator (overall distribution).

4. **Finance-Specific Losses**: Covariance matching loss, tail dependence loss (extreme correlations), volatility clustering loss (GARCH effects).

---

## Q3: 审稿攻击点 / Reviewer Attack Points

**中文：**

1. **生成数据的"过拟合"风险 (Overfitting Risk)**：
   - GAN是否在记忆训练数据而非学习分布？
   - 如何检测生成数据与训练数据的近似重复？
   - 隐私泄露风险（如生成数据暴露了特定交易日的信息）

2. **因子模型的局限性 (Factor Model Limitations)**：
   - 预设因子数量是否足够？是否需要动态因子模型？
   - 非线性因子关系的表达能力
   - 因子模型本身可能遗漏的信息（如idionsyncratic risk的结构）

3. **下游任务评估 (Downstream Task Evaluation)**：
   - 仅在协方差匹配上评估不够——需要组合优化、风险管理等端到端评估
   - 增强数据后训练的模型是否过拟合于生成数据？
   - 实盘验证的缺失

4. **GAN训练稳定性 (GAN Training Stability)**：
   - 模式崩溃问题在金融数据中的表现
   - 训练收敛的判定标准
   - 不同随机种子下生成质量的方差

**English:**

1. **Overfitting Risk**: Is GAN memorizing training data rather than learning distribution? Detecting near-duplicates? Privacy leakage risk?
2. **Factor Model Limitations**: Preset factor count sufficient? Need dynamic factor models? Non-linear factor relationships? Information missed by factor model (idiosyncratic risk structure)?
3. **Downstream Evaluation**: Covariance matching insufficient — need end-to-end evaluation in portfolio optimization, risk management. Does augmented data cause overfitting? Missing live trading validation.
4. **GAN Training Stability**: Mode collapse in financial data? Convergence criteria? Variance across random seeds?

---

## Q4: PhD阅读指南 / PhD Reading Guide

**中文：**

**推荐阅读顺序 / Recommended Reading Order:**
1. 第3节（模型架构）——理解因子嵌入GAN的设计
2. 第4节（实验）——重点关注与bootstrap的对比
3. 第5节（下游应用）——组合优化的实际效果
4. 附录（金融因子模型回顾）——补充经济学背景

**关键方法论需要掌握 / Key Methodology to Master:**
- 条件GAN在时序生成中的应用
- 时序卷积网络（TCN）的因果卷积设计
- 多尺度判别器架构
- 金融因子模型（Fama-French, Carhart等）

**潜在研究方向 / Potential Research Directions:**
- 将MarketGANs扩展为条件生成：给定市场状态生成对应regime数据
- 结合diffusion model替代GAN以获得更稳定的训练
- 生成反事实数据用于压力测试
- 与VAE结合实现可控的隐空间金融数据生成

**English:**

**Recommended Reading Order:**
1. Section 3 (Model Architecture) — factor-embedded GAN design
2. Section 4 (Experiments) — focus on bootstrap comparisons
3. Section 5 (Downstream Applications) — portfolio optimization results
4. Appendix (Factor Model Review) — economics background

**Key Methodology to Master:**
- Conditional GANs for time series generation
- Temporal Convolutional Network (TCN) causal convolution design
- Multi-scale discriminator architecture
- Financial factor models (Fama-French, Carhart, etc.)

**Potential Research Directions:**
- Extend MarketGANs to conditional generation: generate regime-specific data
- Replace GAN with diffusion models for more stable training
- Generate counterfactual data for stress testing
- Combine with VAE for controllable latent space financial data generation
