# Week 12: 2026-03-19 — 2026-03-25 / 第12周

---

## 本周学术动态 / Weekly Highlights

**中文：** 2026 年第 12 周（3 月 19–25 日），arXiv 五个核心类别（cs.CV / cs.LG / cs.AI / cs.CL / stat.ML）共新增约 **2,490** 篇论文。本周焦点集中在“AI 驱动的量化投资”“多智能体金融工作流”与“大模型推理效率”：

1. **市场逻辑驱动的多智能体因子挖掘**：*AlphaLogics* 用多代理协作与市场逻辑约束，实现可扩展且可解释的 alpha 因子生成；
2. **增强 LSTM 的股票趋势预测**：*E-TRENDS* 通过趋势门控与多尺度输入改进传统时序模型，用于股票趋势预测；
3. **收益-风险联合建模**：*Joint Return and Risk Modeling* 用双头深度网络同时输出预期收益与协方差矩阵，端到端生成组合权重；
4. **LLM 选股中的人机协作**：*Large Language Models and Stock Investing* 比较纯 LLM、纯人类与混合模式，探讨人类因素在选股中的不可替代性；
5. **推理效率的软硬协同优化**：从 *ROM* 的实时过度思考干预到 *PRISM* 的光子长上下文选择，研究者在算法与硬件层面压缩推理成本；
6. **多智能体系统的实证化**：金融文档处理、投资组合筛选等场景出现多 Agent 架构对比与成本-精度权衡研究。

**English:** Week 12 of 2026 (March 19–25) saw approximately **2,490** new papers across the five core arXiv categories (cs.CV / cs.LG / cs.AI / cs.CL / stat.ML). The week centered on AI-driven quantitative investing, multi-agent financial workflows, and LLM inference efficiency:

1. **Market-logic-driven multi-agent alpha mining**: *AlphaLogics* uses multi-agent collaboration and market-logic constraints for scalable and interpretable alpha factor generation;
2. **Enhanced LSTM for equity trend forecasting**: *E-TRENDS* improves traditional sequence models with trend gating and multi-scale inputs for equity trend prediction;
3. **Joint return-risk modeling**: *Joint Return and Risk Modeling* uses a dual-headed deep network to simultaneously output expected returns and covariance matrices, producing portfolio weights end-to-end;
4. **Human-AI collaboration in stock selection**: *Large Language Models and Stock Investing* compares pure LLM, pure human, and hybrid modes to examine when human judgment remains irreplaceable;
5. **Algorithm-hardware co-optimization for reasoning efficiency**: from *ROM*'s real-time overthinking mitigation to *PRISM*'s photonic long-context selection, researchers cut inference costs at both algorithm and hardware levels;
6. **Empiricization of multi-agent systems**: multi-agent architecture comparisons and cost-accuracy tradeoff studies emerged in financial document processing, portfolio screening, and related scenarios.

---

## 重点论文深度分析 / Featured Papers with Deep Reviews

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [AlphaLogics: A Market Logic-Driven Multi-Agent System for Scalable and Interpretable Alpha Factor Generation](https://arxiv.org/abs/2603.20247) | Factor Mining | [review.md](alphalogics/review.md) | [code/](alphalogics/code/) |
| 2 | [E-TRENDS: Enhanced LSTM Trend Forecasting for Equities](https://arxiv.org/abs/2603.14453) | Time Series | [review.md](e-trends-lstm/review.md) | [code/](e-trends-lstm/code/) |
| 3 | [Joint Return and Risk Modeling with Deep Neural Networks for Portfolio Construction](https://arxiv.org/abs/2603.19288) | Portfolio Construction | [review.md](joint-risk-return/review.md) | [code/](joint-risk-return/code/) |
| 4 | [Large Language Models and Stock Investing: Is the Human Factor Required?](https://arxiv.org/abs/2603.19944) | LLM Finance | [review.md](llm-human-stock/review.md) | [code/](llm-human-stock/code/) |

---

## 本周论文列表（按领域分类）/ Weekly Papers by Area

> 本周 arXiv 五个主要类别新增约 **2,490** 篇论文（submitted dates: 2026-03-19 to 2026-03-25）。以下按领域精选代表性工作，并附简短双语要点。

> This week added approximately **2,490** papers across five main arXiv categories (submitted dates: 2026-03-19 to 2026-03-25). Representative works by area are listed below with brief bilingual key points.

### 计算机视觉 / Computer Vision

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model](https://arxiv.org/abs/2603.22281v2) | CV / World Model | 将大规模视觉-语言推理模型与联合嵌入预测架构结合，赋予潜在世界模型跨模态推理与长程预测能力。It integrates large vision-language reasoning models with joint-embedding predictive architectures to empower latent world models with cross-modal reasoning and long-horizon prediction. |
| [SAVeS: Steering Safety Judgments in Vision-Language Models via Semantic Cues](https://arxiv.org/abs/2603.19092v1) | CV / VLM Safety | 通过语义线索引导视觉-语言模型的安全判断，缓解有害或不当内容生成的风险。It steers safety judgments in vision-language models via semantic cues to mitigate the risk of generating harmful or inappropriate content. |

### 自然语言处理与大模型 / NLP & LLM

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [When Names Change Verdicts: Intervention Consistency Reveals Systematic Bias in LLM Decision-Making](https://arxiv.org/abs/2603.18530v1) | NLP / LLM Bias | 揭示大语言模型决策中因名称或身份改变而产生的系统性偏见，并提出干预一致性评测框架。It reveals systematic bias in LLM decision-making triggered by changes in names or identities, and proposes an intervention-consistency evaluation framework. |
| [ROM: Real-time Overthinking Mitigation via Streaming Detection and Intervention](https://arxiv.org/abs/2603.22016v2) | NLP / Inference Efficiency | 通过流式检测与实时干预减轻大语言模型过度思考，降低推理延迟与计算开销。It mitigates LLM overthinking via streaming detection and real-time intervention, reducing inference latency and computational cost. |

### 智能体与强化学习 / Agents & RL

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [Memento-Skills: Let Agents Design Agents](https://arxiv.org/abs/2603.18743v1) | Agents / RL | 让智能体自动设计并组合新技能，实现自我进化的代理工作流与持续能力扩展。It lets agents autonomously design and compose new skills, enabling self-evolving agent workflows and continuous capability expansion. |
| [Benchmarking Multi-Agent LLM Architectures for Financial Document Processing: A Comparative Study of Orchestration Patterns, Cost-Accuracy Tradeoffs and Production Scaling Strategies](https://arxiv.org/abs/2603.22651v1) | Agents / Finance | 系统比较金融文档处理中多 Agent 架构的编排模式、成本-精度权衡与生产扩展策略。It systematically compares orchestration patterns, cost-accuracy tradeoffs, and production scaling strategies of multi-agent LLM architectures for financial document processing. |

### 机器学习理论 / ML Theory

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [A Theory of LLM Information Susceptibility](https://arxiv.org/abs/2603.23626v1) | ML Theory | 从统计力学视角建立大语言模型信息易感性的理论框架，解释模型对输入扰动的响应规律。It establishes a theoretical framework for LLM information susceptibility from a statistical-mechanics perspective, explaining how models respond to input perturbations. |

### 金融与量化 / Finance & Quant

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [Designing Agentic AI-Based Screening for Portfolio Investment](https://arxiv.org/abs/2603.23300v1) | Quant / Portfolio | 构建基于 Agentic AI 的投资筛选系统，将大模型推理与组合约束及风险预算相结合。It builds an Agentic AI-based screening system for portfolio investment that combines large-model reasoning with portfolio constraints and risk budgets. |
| [FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading](https://arxiv.org/abs/2603.21330v1) | Quant / Trading | 提出面向量化交易的 AI 原生模块化基础设施，支持策略快速迭代、回测与实盘部署。It proposes an AI-native modular infrastructure for quantitative trading that supports rapid strategy iteration, backtesting, and live deployment. |
| [Mislearning of Factor Risk Premia under Structural Breaks: A Misspecified Bayesian Learning Framework](https://arxiv.org/abs/2603.21672v3) | Finance / Risk | 在结构断点下建立错误设定贝叶斯学习框架，解释因子风险溢价的误学习机制及其对资产配置的影响。It builds a misspecified Bayesian learning framework under structural breaks to explain the mislearning of factor risk premia and its impact on asset allocation. |
| [Implementation Risk in Portfolio Backtesting: A Previously Unquantified Source of Error](https://arxiv.org/abs/2603.20319v1) | Quant / Backtesting | 量化回测中此前未被度量的实施风险来源，揭示过度拟合与执行缺口对业绩的侵蚀。It quantifies a previously unmeasured source of implementation risk in portfolio backtesting, highlighting how overfitting and execution gaps erode reported performance. |

### 系统与高效推理 / Systems & Efficient Inference

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| [PRISM: Breaking the O(n) Memory Wall in Long-Context LLM Inference via O(1) Photonic Block Selection](https://arxiv.org/abs/2603.21576v2) | Systems / Hardware | 利用光子块选择打破长上下文 LLM 推理的 O(n) 内存墙，实现 O(1) 复杂度的关键块检索。It breaks the O(n) memory wall in long-context LLM inference via O(1) photonic block selection for retrieving relevant context blocks. |

---

## 领域趋势 / Trends

1. **金融领域的 Agentic AI / Agentic AI in finance** — 从 AlphaLogics 的多智能体因子挖掘、投资组合筛选到金融文档处理基准，Agentic AI 正在将量化投资从孤立预测任务推向端到端工作流。
2. **推理效率与过度思考控制 / Reasoning efficiency and overthinking control** — ROM、PRISM 等工作从算法干预与硬件加速两个方向压缩大模型推理成本，强调“想多少”与“选哪块”同样重要。
3. **视觉-语言模型的安全与物理推理 / VLM safety and physical-world reasoning** — SAVeS、ThinkJEPA 等研究关注 VLM 的安全判断机制以及跨模态世界模型对物理一致性的建模。
4. **多智能体编排进入实证阶段 / Multi-agent orchestration becomes empirical** — 金融文档处理等场景的架构对比与成本-精度权衡研究，推动多 Agent 系统从概念设计走向可量化选型。
5. **风险感知的量化金融 / Risk-aware quant finance** — 联合收益-风险建模、回测实施风险与 AI 系统性风险模型共同强调：预测精度之外，风险度量的完整性才是组合构建的关键。
6. **高风险决策中的人机协作 / Human-AI collaboration in high-stakes decisions** — LLM 选股、LLM 决策偏见等研究持续追问：在哪些环节人类判断仍然不可替代，以及如何动态配置人机分工。

---

## PhD 阅读路线图 / PhD Reading Roadmap

### 阶段一：建立本周全景（30 分钟）
- 通读本 README 的 Weekly Highlights 与 Trends，明确 3 个核心叙事：Agentic AI 与金融、推理效率控制、VLM/世界模型安全。

### 阶段二：精读重点论文（2–3 小时）
1. **AlphaLogics**：聚焦“市场逻辑约束 + 多代理协作”的因子生成流程，思考可解释性与可扩展性的 trade-off。
2. **E-TRENDS**：对比趋势门控 LSTM 与标准 LSTM/GRU/Transformer，分析趋势定义对预测稳定性的影响。
3. **Joint Return and Risk Modeling**：结合 [review](joint-risk-return/review.md) 理解双头 DNN 如何同时建模收益与协方差，并嵌入 CVaR/MaxDD 约束。
4. **Large Language Models and Stock Investing**：从三种模式对比出发，设计自己研究领域中的人机协作框架。

### 阶段三：按兴趣深入子领域（各 1–2 小时）
- **金融 Agent 工作流**：AlphaLogics → Designing Agentic AI-Based Screening → Benchmarking Multi-Agent LLM Architectures for Financial Document Processing。
- **推理效率**：ROM → PRISM（从算法过度思考干预到光子硬件加速）。
- **VLM 安全与世界模型**：SAVeS → ThinkJEPA（从安全判断到跨模态潜在世界模型）。
- **风险与量化**：Joint Return and Risk Modeling → Implementation Risk in Portfolio Backtesting → Artificial Intelligence and Systemic Risk。

### 阶段四：动手实验（2–4 小时）
- 浏览并运行 [alphalogics/code/](alphalogics/code/) 与 [joint-risk-return/code/](joint-risk-return/code/) 中的示例，尝试修改市场逻辑约束或风险惩罚项，观察因子质量或组合权重的变化。
- 用 FinRL-X 或 Implementation Risk in Portfolio Backtesting 中的思想，在自有回测框架里加入实施成本或滑点扰动，重新评估 Sharpe 与最大回撤。

---

## 数据来源 / Sources

- arXiv API (`export.arxiv.org`)，查询类别 cs.CV / cs.LG / cs.AI / cs.CL / stat.ML，提交日期范围 `2026-03-19` 至 `2026-03-25`。
- 本仓库 Week 12 原始 featured reviews 由先前会话整理，本次补充完整分类列表、趋势、路线图与代码链接。
