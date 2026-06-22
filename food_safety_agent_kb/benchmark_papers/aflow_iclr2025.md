# AFlow: Automating Agentic Workflow Generation (ICLR 2025 Oral)

> 深度解读：用 **MCTS** 在 code-represented workflow 空间自动搜索最优 Agent 工作流
> In-depth: MCTS over code-represented workflows to automate agent design

---

## 一、为什么需要自动化的 Agent 工作流？

### 1.1 当前 Agent 构建的痛点

要构建一个高质量 Agent workflow，通常需要：

- **大量人工设计**：分析任务、选择模块、定义流程
- **反复试错**：调整 prompt、修改流程、添加 review 步骤
- **领域专家**：需要既懂任务又懂 LLM 行为

例如：编程任务的典型 workflow

```text
Generate → Test → Debug → Reflect → Revise → Final
```

但每一步的实现细节（用什么 prompt、是否 ensemble、是否 review）需要大量调优。

### 1.2 现有自动化的局限

| 方法 | 问题 |
|---|---|
| 手工 prompt + 反复试错 | 慢、贵、不可扩展 |
| DSPy / TextGrad | 只优化 prompt，不优化结构 |
| 模块库 + 启发式搜索 | 仍是有限枚举 |

### 1.3 AFlow 的野心

> **用 LLM + MCTS 自动发现 workflow 结构 + prompt**，在多个 benchmark 上超越手工设计。

---

## 二、核心思想 / Core Idea

### 2.1 Workflow = Code

```python
# AFlow 把 workflow 表示为代码
class Workflow:
    def __init__(self):
        self.nodes = [...]      # LLM 调用单元
        self.edges = [...]      # 依赖关系

    def run(self, input):
        # 执行流程
        ...
```

**优势**:
- 灵活（图、神经网络、代码都可表达）
- 可序列化（容易持久化、迁移）
- 易扩展（加新节点类型容易）

### 2.2 搜索空间 = Code Space

```text
搜索维度
├── Node 类型选择 (Generate / Review / Revise / Ensemble / ...)
├── Node 顺序 (线性 / 分支 / 循环)
├── Edge 拓扑 (DAG / 树 / 链)
├── Prompt 模板
├── 温度 / Top-p / Model 选择
└── Control flow (loop until convergence)
```

---

## 三、架构组件 / Architecture

### 3.1 五大组件

```text
┌──────────────────────────────────────────────────┐
│  AFlow Framework                                │
├──────────────────────────────────────────────────┤
│                                                  │
│  1. Node       — LLM 调用单元                   │
│                  (prompt / T / format / model)   │
│                                                  │
│  2. Operator   — 预定义节点组合                   │
│                  (Generate / Review / Revise /   │
│                   Ensemble / Test / Programmer)  │
│                                                  │
│  3. Workflow   — Node + Edge = 图 / 序列 / 代码  │
│                                                  │
│  4. Optimizer  — MCTS 探索 Workflow 变体         │
│                                                  │
│  5. Evaluator  — 在任务上评估 Workflow            │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 3.2 Node 详解

```python
# metagpt_core/action_nodes/action_node.py
class ActionNode:
    def __init__(self, llm, prompt, temperature=0.0, format=None):
        self.llm = llm
        self.prompt = prompt
        self.temperature = temperature
        self.format = format

    async def run(self, context):
        return await self.llm(self.prompt.format(**context))
```

### 3.3 Operator 详解

预定义的高效组合：

| Operator | 描述 |
|---|---|
| **Generate** | 单 LLM 生成 |
| **Review** | 一次 LLM 调用检查 |
| **Revise** | 基于 review 修改 |
| **Ensemble** | 多 LLM 投票 |
| **Test** | 程序化测试 |
| **Programmer** | 代码生成 + 执行 |

---

## 四、MCTS 优化流程 / MCTS Optimization

### 4.1 树结构

每个 **节点 = 一个完整 workflow**：

```text
                     ┌─ Workflow A (F1=0.85)
                     │
Root Workflow ───────┼─ Workflow B (F1=0.72)
                     │
                     └─ Workflow C (F1=0.91)
                              │
                              ├─ Workflow C1 (F1=0.88)
                              └─ Workflow C2 (F1=0.93)
```

### 4.2 四步 MCTS

```text
Step 1: Selection
   从根开始，用 UCB 选择最值得探索的子节点

Step 2: Expansion
   添加一个新子节点（修改 workflow 的一部分）

Step 3: Simulation (Evaluation)
   在任务上评估新 workflow，记录 F1 / accuracy

Step 4: Backpropagation
   沿路径回传 reward 更新节点值
```

### 4.3 Workflow 修改操作

```python
def modify_workflow(workflow, llm_suggester):
    """
    可能的修改：
    1. 替换一个 node (e.g., change prompt)
    2. 插入一个新 node (e.g., add Review step)
    3. 删除一个 node
    4. 改变 edge 拓扑
    5. 改变 operator 组合
    """
    suggestions = llm_suggester.propose(workflow)
    new_workflow = apply_random_modification(workflow, suggestions)
    return new_workflow
```

---

## 五、实验结果 / Experimental Results

### 5.1 Benchmark 数据集

| 数据集 | 任务 | 规模 |
|---|---|---|
| HumanEval | 代码生成 | 164 题 |
| MBPP | Python 编程 | 974 题 |
| GSM8K | 数学推理 | 1319 题 |
| MATH | 高级数学 | 5000 题 |
| HotpotQA | 多跳 QA | 7405 题 |
| DROP | 离散推理 | 7740 题 |

### 5.2 主要结果

| 方法 | 平均得分 | 备注 |
|---|---|---|
| GPT-4o (zero-shot) | 71.3% | 基线 |
| GPT-4o (manual workflow) | 78.4% | 手工设计 |
| AutoAgent | 80.1% | 自动化 |
| **AFlow (GPT-4o)** | **84.1%** | **+5.7% vs SOTA** |

### 5.3 关键发现：小模型 + AFlow > GPT-4o

> AFlow enables **smaller models** to **outperform GPT-4o** on specific tasks at **4.55% of its inference cost in dollars**.

| 任务 | GPT-4o | AFlow + 小模型 | 节省成本 |
|---|---|---|---|
| HumanEval | 92% | 95% | 95% |
| GSM8K | 95% | 97% | 95% |

含义：
- **自动化 workflow 设计**让小模型性价比反超大模型
- 对成本敏感的食品安全项目尤其重要

---

## 六、对食品安全的启示

### 6.1 食品安全检测 Workflow

```text
传统 (手工):
  食品图像 → YOLOv8 → 输出结果

AFlow 自动发现:
  食品图像 → [YOLOv8 + CLIP] Ensemble →
  → Review: 是否误检? →
  → Revise: 重新检测 (if low confidence) →
  → Test: 是否与历史一致? →
  → 输出
```

### 6.2 食品召回决策

```text
AFlow 自动发现:
  召回事件 →
  ├─ Hazard Classifier (BERT)
  ├─ Severity Estimator (LLM)
  ├─ Affected Product Extractor (NER)
  └─ Regulatory Mapping (RAG) →
  → Review (LLM 评估严重性) →
  → Decision (召回 vs 警告)
```

### 6.3 食品安全报告生成

```text
AFlow 自动发现:
  事件数据 →
  ├─ Summary Generation (LLM)
  ├─ Compliance Check (RAG)
  ├─ Risk Score (Classifier)
  └─ Format (Markdown / PDF) →
  → Review (LLM 审核)
  → Final Report
```

---

## 七、复现 / Reproduction

```bash
git clone https://github.com/FoundationAgents/AFLOW
cd AFLOW
pip install -r requirements.txt

# 运行 MCTS 优化
python -m AFlow.scripts.optimize \
  --dataset humaneval \
  --model gpt-4o-mini \
  --n_trials 50 \
  --output_dir results/

# 评估最优 workflow
python -m AFlow.scripts.evaluate \
  --workflow results/best_workflow.py \
  --dataset humaneval
```

### 自定义 Operator
```python
# operator.py
class FoodSafetyOperator:
    def __init__(self, llm, rag):
        self.llm = llm
        self.rag = rag

    async def check_compliance(self, text):
        # 1. RAG 检索法规
        context = await self.rag.search(text)
        # 2. LLM 评估
        result = await self.llm(f"Context: {context}\nText: {text}\n是否合规？")
        return result
```

---

## 八、与 DSPy 对比 / Comparison with DSPy

| 维度 | AFlow | DSPy |
|---|---|---|
| **优化对象** | Workflow 结构 + prompt | 主要优化 prompt |
| **搜索算法** | MCTS | BootstrapFewShot, MIPRO |
| **适用场景** | 复杂多步骤 | 模块化 LLM pipeline |
| **学习曲线** | 较陡 | 平缓 |
| **可组合性** | ✅ AFlow + DSPy | ✅ DSPy + LangGraph |

**最佳实践**: LangGraph 管 workflow，DSPy 优化 prompt，AFlow 自动发现结构。

---

## 九、限制与未来 / Limitations

### 限制
1. **初始 workflow 仍需手工设计**
2. **MCTS 收敛慢**（需数十次 trial）
3. **复杂任务评估成本高**

### 未来方向
- **冷启动**: 自动生成初始 workflow
- **迁移**: 跨任务共享 learned workflow
- **多目标**: 同时优化 cost + quality

---

## 十、一句话总结

> **AFlow 用 MCTS 在 code-represented workflow 空间自动搜索，用机器劳动替代人工设计 Agent 工作流，在多个 benchmark 上超越 SOTA 5.7%，并让小模型以 4.55% 的 GPT-4o 成本超越 GPT-4o。**

---

### 参考文献 / References

1. Zhang et al. (2025). "AFLOW: Automating Agentic Workflow Generation." ICLR 2025 Oral
2. arXiv: https://arxiv.org/abs/2410.10762
3. ICLR: https://proceedings.iclr.cc/paper_files/paper/2025/file/5492ecbce4439401798dcd2c90be94cd-Paper-Conference.pdf
4. Code: https://github.com/FoundationAgents/AFLOW
