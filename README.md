# AI Paper Reproductions / AI论文复现计划

> 系统性复现2026年计算机顶会论文 + 量化金融应用，每篇论文独立仓库，统一模板，中英双语。
>
> Systematic reproduction of 2026 top CS conference papers + quantitative finance applications. Each paper in its own repo, unified template, bilingual CN/EN.

## 📋 仓库总览 / Repository Index

| # | 仓库 / Repo | 论文 / Paper | 会议 / Venue | 文件 / Files | 代码行 / Lines |
|---|---|---|---|---|---|
| 1 | [DINOv2-reproduction](https://github.com/sjkncs/DINOv2-reproduction) | DINOv2 自监督视觉表示学习 | Meta 2023 | 18 | 4,116 |
| 2 | [DDL-reproduction](https://github.com/sjkncs/DDL-reproduction) | Deep Delta Learning | arXiv Jan 2026 | 20 | 4,187 |
| 3 | [CLIP-reproduction](https://github.com/sjkncs/CLIP-reproduction) | CLIP 对比语言-图像预训练 | ICML 2021 | 22 | 3,943 |
| 4 | [D4RT-reproduction](https://github.com/sjkncs/D4RT-reproduction) | D4RT 4D动态场景重建 | CVPR 2026 Best Paper | 20 | 3,872 |
| 5 | [TRELLIS2-reproduction](https://github.com/sjkncs/TRELLIS2-reproduction) | TRELLIS.2 3D生成 | CVPR 2026 Best Student | 20 | 3,917 |
| 6 | [transformer-succinctness-reproduction](https://github.com/sjkncs/transformer-succinctness-reproduction) | Transformer简洁性理论证明 | ICLR 2026 Outstanding | 21 | 5,345 |
| 7 | [llms-lost-multiturn-reproduction](https://github.com/sjkncs/llms-lost-multiturn-reproduction) | LLM多轮对话性能退化 | ICLR 2026 Outstanding | 21 | 4,968 |
| 📚 | [ai-papers-and-notes](https://github.com/sjkncs/ai-papers-and-notes) | 周报汇总 + 审稿分析 + 考研笔记 | Weekly reports | 35+ | 10,000+ |

## 🏗️ 统一复现模板 / Unified Reproduction Template

每个仓库遵循统一结构（参考 DINOv2-reproduction）：

```
<paper>-reproduction/
├── README.md                    # 双语项目说明 + 论文引用
├── requirements.txt             # 依赖
├── .gitignore
├── paper_analysis/
│   └── review.md                # 审稿人级四问深度分析
├── configs/
│   └── default.yaml             # 全部训练超参数
├── models/                      # 模型实现 (PyTorch)
├── data/                        # 数据加载 + 增强
├── loss/                        # 损失函数
├── utils/                       # 工具函数
├── train.py                     # 完整训练循环
├── evaluate.py                  # 评估脚本
└── quant/                       # 量化金融应用 (独立复现)
    ├── README.md
    ├── quant_models.py          # 量化模型
    ├── quant_train.py           # 量化训练
    └── quant_evaluate.py        # 回测评估
```

## 🔬 审稿分析四问 / The Four Reviewer Questions

每篇论文的 `paper_analysis/review.md` 均回答：

1. **它真正想解决的问题是什么？** / What problem does it really solve?
2. **它声称的贡献是什么？** / What are the claimed contributions?
3. **最可能被reviewer攻击的地方在哪里？** / Most likely reviewer attack points?
4. **同方向博士生应精读哪些、跳过哪些？** / PhD reading guide: read carefully vs. skip?

## 💰 量化金融映射 / Quantitative Finance Mapping

每篇论文均包含 `quant/` 目录，独立实现量化金融应用：

| 论文 | 量化应用 | 核心映射 |
|---|---|---|
| DINOv2 | K线图自监督regime检测 | 视觉特征 → 市场状态分类 |
| DDL | Regime切换时序预测 + MoE因子路由 | 残差重写 → 市场regime适应 |
| CLIP | 多模态Alpha信号融合 | 图文对齐 → 新闻+K线+因子对齐 |
| D4RT | 价格轨迹3D跟踪 | 4D点查询 → 价格-量-时间3D追踪 |
| TRELLIS.2 | 风险收益3D曲面 | O-Voxel → 投资组合风险面建模 |
| Transformer简洁性 | 模型表达力对比 | 理论证明 → 金融模式建模能力 |
| LLM多轮退化 | 策略生成退化检测 | 多轮评估 → 策略迭代质量监控 |

## 📅 每周推送 / Weekly Push Schedule

- **时间 / Time:** 每日 10:00 CST
- **逻辑 / Logic:** 从2026年1月1日起逐周推进（Day N = Week N）
- **覆盖 / Coverage:** NeurIPS, CVPR, ICML, ICLR, ECCV, AAAI, ACL, EMNLP
- **内容 / Content:** 周报 + 审稿分析 + 代码复现 + 量化映射
- **目标仓库 / Target:** [ai-papers-and-notes](https://github.com/sjkncs/ai-papers-and-notes)

## 🎯 职业目标 / Career Goal

**量化AI研究员** — 将前沿AI论文研究与量化金融实战结合。

JD核心要求：985/海外硕博（数/理/统/CS/EE）、深度学习/强化学习/组合优化、PyTorch/TensorFlow、顶会论文、大规模ML应用经验。

详见 [quant-career-roadmap.md](https://github.com/sjkncs/ai-papers-and-notes/blob/main/quant-career-roadmap.md)

## 📖 各仓库详情 / Repo Details

### 1. DINOv2-reproduction
> Self-supervised visual representation learning (Oquab et al., 2023)

- **核心实现:** ViT backbone (S/B/L/G) + DINO投影头 + EMA学生-教师框架
- **训练:** Multi-crop增强 + 中心化锐化损失 + 余弦LR
- **评估:** kNN分类 + 线性探测
- **量化:** K线图自监督市场状态检测

### 2. DDL-reproduction
> Deep Delta Learning (Zhang et al., arXiv 2601.00417, 2026)

- **核心实现:** GatedDeltaGate + DDLLayer (残差从"累加"升级为"可替换")
- **对比实验:** DDLModel vs StandardTransformer A/B测试
- **评估:** 困惑度 + 门控值分布分析
- **量化:** DDL时序预测器 + MoE因子路由 + 端到端组合优化

### 3. CLIP-reproduction
> Learning Transferable Visual Models (Radford et al., ICML 2021)

- **核心实现:** ViT图像编码器 + Transformer文本编码器 + InfoNCE损失
- **评估:** 零样本分类 + 图文检索 (R@1/5/10)
- **量化:** FinanceCLIP三模态对比学习 (新闻+K线+因子)

### 4. D4RT-reproduction
> Efficiently Reconstructing Dynamic Scenes (Zhang et al., CVPR 2026 **Best Paper**)

- **核心实现:** ViT backbone + SRT-style点查询解码器 + 深度/相机头
- **训练:** 合成动态场景数据 + 深度/对应/相机联合损失
- **评估:** 3D跟踪AUC + 深度RMSE + 相机ATE
- **量化:** 金融D4RT — 价格+量+时间的3D轨迹跟踪

### 5. TRELLIS2-reproduction
> Native and Compact Structured Latents for 3D (Microsoft/Tsinghua, CVPR 2026 **Best Student**)

- **核心实现:** O-Voxel稀疏体素 + 3D-VAE (16×下采样) + 自回归生成
- **训练:** 两阶段 (VAE预训练 + AR微调)
- **评估:** Chamfer距离 + F-Score + 材质准确率
- **量化:** O-Voxel投资组合风险收益3D曲面

### 6. transformer-succinctness-reproduction
> Transformers are Inherently Succinct (Bergsträßer et al., ICLR 2026 **Outstanding**)

- **核心实现:** 固定精度Transformer + RNN + FSA + LTL形式化
- **实验:** 计算性验证指数/双指数简洁性差距
- **量化:** Transformer vs RNN金融模式建模能力对比

### 7. llms-lost-multiturn-reproduction
> LLMs Get Lost In Multi-Turn Conversation (Laban et al., ICLR 2026 **Outstanding**)

- **核心实现:** 6任务评估框架 + 多轮对话管理器 + 能力/可靠性分解评分
- **评估:** 性能退化曲线 + 失败模式分类
- **量化:** 策略生成多轮退化检测 + "过度乐观"回测检测器

## 🛠️ 技术栈 / Tech Stack

- **框架:** PyTorch (all repos)
- **语言:** Python 3.10+, bilingual CN/EN comments
- **数据:** 全部支持mock/synthetic data独立运行 (无需外部数据集)
- **配置:** YAML configs + argparse CLI
- **部署:** pip install -r requirements.txt → python train.py

## 📊 统计 / Statistics

- **总仓库数 / Total Repos:** 8
- **总文件数 / Total Files:** ~197
- **总代码行 / Total Lines:** ~40,000+
- **覆盖会议 / Conferences:** CVPR, ICLR, ICML, NeurIPS, AAAI
- **量化应用 / Quant Apps:** 7 independent implementations

---

*最后更新 / Last Updated: 2026-06-15*
*自动更新中 / Auto-updating via daily cron at 10:00 CST*
