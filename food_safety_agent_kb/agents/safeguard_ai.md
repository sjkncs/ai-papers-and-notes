# SafeGuard AI: Multi-Agent Food Compliance System

> CrewAI + Groq 多 Agent HACCP 合规系统

## 📦 仓库信息

- **Repo**: https://github.com/Sharma-Aditya7/multiagent-food-compliance
- **Stars**: 5 | **Forks**: 1 | **License**: MIT
- **创建**: 2024-11-07

## 🎯 核心目标

通过 **多 Agent 协作** 自动化 HACCP 食品合规审计：
- 实时风险识别
- 自动文档生成
- 减少人工审计错误

## 🏗️ 架构

```text
SafeGuard AI
├── Risk Identification Agent   # 识别生物/化学/物理危害
├── HACCP Compliance Agent      # 检查 HACCP 七大原则
├── Documentation Agent         # 生成审计跟踪
└── Gradio UI                   # 交互界面
```

## 🛠️ 技术栈

| 组件 | 说明 |
|---|---|
| **Python** | 核心语言 |
| **CrewAI** | 多 Agent 编排框架 |
| **Groq Mixtral-8x7b** | 推理 LLM |
| **Gradio** | UI 框架 |

## 🚀 快速开始

```bash
git clone https://github.com/Sharma-Aditya7/multiagent-food-compliance.git
cd multiagent-food-compliance
pip install -r requirements.txt

# 设置 Groq API Key
export GROQ_API_KEY=<your_key>

# 启动
python app.py
# → 访问 http://localhost:7860
```

## 💡 关键代码片段

```python
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq

llm = ChatGroq(model="mixtral-8x7b-32768")

risk_agent = Agent(
    role="Risk Identification Specialist",
    goal="识别食品加工中的潜在危害",
    backstory="你是食品安全专家，擅长识别生物、化学、物理危害",
    llm=llm
)

haccp_agent = Agent(
    role="HACCP Compliance Officer",
    goal="确保 HACCP 七大原则的合规性",
    backstory="你是 HACCP 合规审计专家",
    llm=llm
)

doc_agent = Agent(
    role="Documentation Specialist",
    goal="生成符合 FSMA 的审计跟踪文档",
    backstory="你是食品安全文档专家",
    llm=llm
)

# 任务链
crew = Crew(agents=[risk_agent, haccp_agent, doc_agent], tasks=[...])
result = crew.kickoff(inputs={"operation": "..."})
```

## 📊 食品安全场景应用

| 场景 | 应用 |
|---|---|
| 肉禽加工 | CCP 监控、致病菌风险 |
| 海产品 | 组胺、贝类毒素 |
| 乳制品 | 巴氏杀菌监控 |
| 烘焙 | 过敏原控制 |
| 餐饮 | 温度-时间监控 |

## ⚠️ 限制

- ⭐ 仅 5 stars，相对早期
- 依赖 Groq API（成本 + 数据出境）
- 文档英文为主

## 🔗 关联项目

- [afrexai-food-safety](https://level8.bg/tools/skills/afrexai-food-safety) — 1933★ HACCP Skill
- [Sharma-Aditya7/multiagent-food-compliance](https://github.com/Sharma-Aditya7/multiagent-food-compliance)
