# DesignX: Human-Competitive Algorithm Designer for Black-Box Optimization (NeurIPS 2025)

> 深度解读：用 **双 Agent RL** 自动化算法设计 — 工作流生成 + 超参控制
> In-depth: Dual-Agent Reinforcement Learning for automated algorithm design

---

## 一、为什么需要自动化算法设计？

### 1.1 现实痛点

设计一个 black-box optimizer 通常需要：

- **数月**手动尝试
- **深度领域知识**（optimization 算法、benchmark、领域经验）
- **反复调参**（learning rate、population size、search strategy）
- **大量计算资源**验证

对于非专业用户（生物学家、物流工程师、量化研究员）：
- 找不到合适的 optimizer
- 即使找到，也要花大量时间调参

### 1.2 现有方法的局限

| 方法 | 问题 |
|---|---|
| 手工设计 | 慢、贵、依赖专家 |
| Hyperparameter Optimization (HPO) | 只调参，不选算法 |
| Neural Architecture Search (NAS) | 只针对 NN，不针对 optimizer |
| MetaBBO | 大多只做其中一项 |

### 1.3 DesignX 的目标

> **几秒钟内**为给定 black-box 问题生成一个 **定制化 optimizer**，无需人工干预。

---

## 二、核心思想 / Core Idea

### 2.1 问题分解

自动化算法设计 = 两个子任务：

```text
┌─────────────────────────────┐    ┌─────────────────────────────┐
│  Sub-task 1                 │    │  Sub-task 2                 │
│  Algorithm Workflow         │    │  Hyperparameter Control     │
│  Generation                 │    │                             │
├─────────────────────────────┤    ├─────────────────────────────┤
│ 从 116 个模块中              │    │ 根据实时反馈                 │
│ 组合出 optimizer workflow    │    │ 动态调整超参                 │
│                             │    │                             │
│ e.g. CMA-ES + Restart +     │    │ pop_size, lr, sigma, ...    │
│       Local Search          │    │                             │
└─────────────────────────────┘    └─────────────────────────────┘
```

### 2.2 Modular-EC: 116 模块库

收集 **数十年**算法研究的成果：

```text
Modular-EC
├── Initialization (10+ 种)
├── Selection (5+ 种: Tournament, Roulette, ...)
├── Mutation (20+ 种: Polynomial, Gaussian, ...)
├── Crossover (15+ 种: SBX, DE/rand/1, ...)
├── Restart Strategies (5+ 种)
├── Local Search (5+ 种: Powell, Nelder-Mead, ...)
└── Constraint Handling (5+ 种)
```

---

## 三、双 Agent 架构 / Dual-Agent Architecture

### 3.1 总览

```text
                    ┌──────────────────────┐
   Problem p   →    │  Agent-1 (π_φ)       │
                    │  Workflow Generator   │
                    └──────────┬───────────┘
                               │ A_p (workflow)
                               ↓
                    ┌──────────────────────┐
                    │  Black-Box           │
                    │  Optimization Loop   │
                    └──────────┬───────────┘
                               │ O_t (progress)
                               ↓
                    ┌──────────────────────┐
                    │  Agent-2 (π_θ)       │
                    │  Hyperparameter      │
                    │  Controller          │
                    └──────────────────────┘
```

### 3.2 Agent-1: Workflow Generator

**输入**:
- 问题实例特征向量: `p` (维度、搜索范围、ELA 统计特征...)

**输出**:
- 完整的 optimizer workflow `A_p`
- 由模块 ID 序列组成

**机制** (Transformer):
```text
Input:  [CLS] feat(p) mod_1 mod_2 ... mod_n
            ↓ (Transformer)
Output: prob distribution over next module
            ↓ (autoregressive sampling)
Workflow:  CMA-ES → Polynomial Mutation → Tournament Selection → ...
```

### 3.3 Agent-2: Hyperparameter Controller

**输入**:
- 优化进度观察 `O_t` (9 维 progress feature vector)
- 模块 ID (来自 Agent-1)

**输出**:
- 所有可控模块的超参值 `C_t^m`

**机制** (Transformer + Gaussian):
```python
# 输入编码
H = Transformer([module_ids; O_t])

# 输出参数
μ = W_μ^T · H_dec
Σ = W_Σ^T · H_dec

# 采样超参
C_t^m ~ N(μ^(m), Σ^(m))
```

### 3.4 协同训练 / Cooperative Training

**协同 reward**:
\[
J_\text{joint} = J_\phi + J_\theta
\]

其中:
- \(J_\phi\): Agent-1 生成的 workflow 质量（端到端优化结果）
- \(J_\theta\): Agent-2 的超参控制质量

**训练规模**:
- 10,000 合成问题
- 数天自主学习
- Transformer 协同优化

---

## 四、实验结果 / Experimental Results

### 4.1 合成测试集

| Method | Best Found | Budget |
|---|---|---|
| 手工设计 CMA-ES | 0.42 | 1000 evals |
| AutoML-Zero | 0.38 | 1000 evals |
| **DesignX** | **0.12** | **1000 evals** |

> DesignX 超出人类手工优化器 **数个数量级**

### 4.2 真实场景

| 场景 | 任务 | 提升 |
|---|---|---|
| Protein Docking | 蛋白质对接 | +25% |
| AutoML | 模型选择 + 超参 | +18% |
| UAV Path Planning | 无人机路径规划 | +30% |

### 4.3 算法发现能力

> DesignX 能发现 **非平凡**、**超出专家直觉** 的算法模式。

示例:
- 传统方法: CMA-ES → restart
- DesignX 发现: CMA-ES → **specific crossover** → **partial restart with elitism** → local refinement

这种结构在传统文献中 **没有被显式提出过**！

---

## 五、对食品安全的启示

### 5.1 冷链调度优化

- 冷链路径规划 = BBO 问题
- DesignX 自动设计调度算法 + 动态调整温度阈值
- 节省冷链能耗

### 5.2 食品库存优化

- 多 SKU、多约束（保质期、温度、HACCP）
- DesignX 自动发现库存策略 + 动态补货参数

### 5.3 食品检测模型选择

- 多种食品检测模型（YOLOv8 / Faster R-CNN / DETR）
- DesignX 选模型 + 调超参

---

## 六、复现路线 / Reproduction

```bash
git clone https://github.com/MetaEvo/DesignX
cd DesignX
pip install -r requirements.txt

# 训练双 Agent
python train.py \
  --agent1 transformer \
  --agent2 transformer \
  --modular_library Modular-EC \
  --n_problems 10000

# 测试
python evaluate.py --problem protein_docking
```

### 核心 API
```python
from designx import DesignX

dx = DesignX()
optimizer = dx.generate(problem_features)
# optimizer 已包含完整 workflow + 初始超参

result = optimizer.optimize(
    problem_instance,
    update_fn=dx.agent2.adjust  # 动态超参调整
)
```

---

## 七、与其他方法对比 / Comparison

| 方法 | 工作流生成 | 超参控制 | 协同 |
|---|---|---|---|
| Manual Design | ✗ | ✓ | - |
| HPO (Optuna) | ✗ | ✓ | - |
| AutoML-Zero | ✓ | ✗ | - |
| MetaBBO + HPO | ✓ | ✓ | ✗ (分开优化) |
| **DesignX** | **✓** | **✓** | **✓ (联合)** |

---

## 八、限制与未来方向 / Limitations & Future Work

### 限制
1. **冷启动**: Modular-EC 需手工预定义
2. **迁移性**: 在合成问题训练，真实问题可能域漂移
3. **可解释性**: 黑盒 Transformer 决策

### 未来方向
- 自动发现新模块（不再手工预定义）
- 在真实问题分布上训练
- 可解释 RL（解释为什么选这个 workflow）

---

## 九、一句话总结

> **DesignX 是首个端到端联合学习 "算法结构 + 超参" 的 MetaBBO 框架，用双 Transformer Agent 协同训练，在合成和真实问题上都超越了人类设计。**

---

### 参考文献 / References

1. Zhang et al. (2025). "DesignX: Human-Competitive Algorithm Designer for Black-Box Optimization." NeurIPS 2025
2. arXiv: https://arxiv.org/pdf/2505.17866
3. OpenReview: https://openreview.net/forum?id=FAiIRMvIwy
4. Code: https://github.com/MetaEvo/DesignX
