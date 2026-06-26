# AlphaLogics: Market Logic-Driven Multi-Agent System for Alpha Factor Generation

> **arXiv:** [2603.20247](https://arxiv.org/abs/2603.20247) | **日期 / Date:** 2026-03-10 | **作者 / Authors:** Weng et al.

---

## Q1: 核心问题

**中文：** 现有alpha因子挖掘方法要么依赖专家经验（不可扩展），要么纯数据驱动（不可解释）。AlphaLogics提出用市场逻辑驱动的多代理系统，让不同角色的AI代理协作生成既可扩展又可解释的alpha因子。

**English:** Existing alpha factor mining is either expert-dependent (not scalable) or purely data-driven (not interpretable). AlphaLogics uses market-logic-driven multi-agent system for scalable and interpretable alpha generation.

## Q2: 核心贡献

1. **多代理架构**: 研究员代理(假设生成)、验证员代理(回测验证)、逻辑员代理(经济学解释)
2. **市场逻辑约束**: 因子必须通过经济学合理性检查才接受
3. **可扩展性**: 自动化流程支持大规模因子挖掘
4. **可解释性**: 每个因子都有清晰的市场逻辑说明

## Q3: 审稿攻击点

1. 代理间通信开销？ 2) 市场逻辑的形式化定义？ 3) 与Autonomous Factor Investing (Week 10)的对比？

## Q4: 量化映射

- 直接构建多代理因子挖掘系统
- 市场逻辑约束用于因子质量筛选
- 可解释因子库用于策略报告和客户沟通
