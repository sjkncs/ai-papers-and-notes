# Week 3: 2026-01-15 — 2026-01-21 / 第3周

## 本周学术动态 / Weekly Highlights

**中文：** 2026年第三周，LLM推理机制和视觉控制生成成为研究热点。《The Geometry of Thought》揭示了规模如何重构而非均匀提升LLM的推理能力；DepthDirector引入显式深度信息实现精准相机控制视频生成；城市社会语义分割将视觉-语言推理引入城市空间分析；《The Assistant Axis》发现语言模型存在低维"助手轴"，在情绪化对话中易漂移。ICLR 2026通知日临近（1月25日），社区关注度持续升温。

**English:** Week 3 of 2026 saw LLM reasoning mechanisms and visually-controlled generation as hot topics. "The Geometry of Thought" revealed that scale restructures rather than uniformly improves reasoning; DepthDirector introduced explicit depth for precise camera-controlled video generation; Urban Socio-Semantic Segmentation brought VL reasoning to urban spatial analysis; "The Assistant Axis" discovered a low-dimensional persona axis in LLMs that drifts during emotional conversations. ICLR 2026 notification approaching (Jan 25).

## 重点论文 / Featured Papers

| # | 论文 / Paper | 方向 / Area | 审稿分析 / Review | 代码复现 / Code |
|---|---|---|---|---|
| 1 | [The Geometry of Thought](https://arxiv.org/abs/2601.13358) | LLM Reasoning | [review.md](geometry-of-thought/review.md) | [code/](geometry-of-thought/code/) |
| 2 | [DepthDirector: 3D Understanding for Video Generation](https://arxiv.org/abs/2601.10214) | Video Generation | [review.md](depthdirector/review.md) | [code/](depthdirector/code/) |
| 3 | [Urban Socio-Semantic Segmentation](https://arxiv.org/abs/2601.10477) | VL Reasoning | [review.md](urban-socioseg/review.md) | — |
| 4 | [The Assistant Axis](https://arxiv.org/abs/2601.10387) | LLM Safety | [review.md](assistant-axis/review.md) | — |

## 其他论文 / Other Papers

| 论文 / Paper | 方向 / Area | 要点 / Key Point |
|---|---|---|
| Eliciting Harmful Capabilities by Fine-Tuning | Safety | 微调安全防护输出可恢复49%有害能力 |
| Alignment Pretraining | Safety / Alignment | 合成对齐文档将misaligned行为从45%降至9% |
| Production-Ready Probes (GDM) | Safety | 短上下文训练的探针在长上下文中灾难性失效 |

## 领域趋势 / Trends

1. 规模重构推理 / Scale restructures reasoning — 非均匀提升，领域间差异显著
2. 深度感知视频生成 / Depth-aware video generation — 显式3D结构取代隐式inpainting
3. 视觉-语言空间分析 / VL spatial reasoning — 从像素分割到社会语义理解
4. LLM人格稳定性 / LLM persona stability — 低维轴在对抗场景中漂移
5. 安全攻防升级 / Safety arms race escalates — 微调攻击与探针防御的博弈

## 重要时间节点 / Key Dates

| 事件 / Event | 日期 / Date |
|---|---|
| ICLR 2026 Notification | Jan 25, 2026 |
| AAAI 2026 Conference | Feb–Mar 2026 |
| NeurIPS 2026 Submissions Open | Apr 2026 |
