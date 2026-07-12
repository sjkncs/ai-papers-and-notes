# LangGraph: Industrial-Grade Stateful Agent Orchestration (35k★)

> 低阶编排框架，用于构建、管理、部署长运行 stateful agents
> Low-level orchestration framework for building, managing, and deploying long-running, stateful agents

## 📦 仓库信息

- **Repo**: https://github.com/langchain-ai/langgraph
- **Stars**: 35,346 | **Forks**: 5,930 | **License**: MIT
- **Homepage**: https://docs.langchain.com/oss/python/langgraph/
- **创建**: 2023-08-09
- **生产用户**: Klarna, Replit, Elastic 等

## 🎯 为什么 LangGraph 是首选？

### 五大核心能力

| 能力 | 说明 |
|---|---|
| **Durable Execution** | 失败持久化，长任务自动恢复 |
| **Comprehensive Memory** | 短期 / 长期 / 跨线程持久化 |
| **Human-in-the-Loop** | 内置 interrupt / resume 机制 |
| **Time Travel** | 状态可回放、调试 |
| **Production Deployment** | LangSmith Deployment 平台 |

## 🚀 快速开始

### 安装

```bash
pip install langgraph
# 或
pip install langchain langgraph langsmith
```

### 最小示例

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()

# 运行
result = app.invoke({"messages": [("user", "HACCP 是什么？")]})
```

## 🏗️ 核心概念

### 1. StateGraph

```text
         ┌──────┐
START ──→│ Node │──→ END
         └──────┘

多 Node:
         ┌────────┐     ┌────────┐
START ──→│ Node A │──→ │ Node B │──→ END
         └────────┘     └────────┘
```

### 2. Conditional Edge

```python
def should_continue(state):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

graph.add_conditional_edges("agent", should_continue, {
    "tools": "tool_node",
    END: END
})
```

### 3. 循环图

```python
def should_retry(state):
    if state["error_count"] < 3:
        return "retry"
    return "fail"

graph.add_edge("retry", "process")  # 循环
```

### 4. Human-in-the-Loop

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory, interrupt_before=["human_review"])

# 暂停在 human_review
for event in app.stream({"query": "..."}, config):
    if "human_review" in event:
        # 等人工审核
        user_approval = input("Approve? (y/n): ")
        # resume
        app.invoke(None, config)  # 继续
```

## 🧩 工具集成

### LangChain 集成

```python
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults

llm = ChatOpenAI(model="gpt-4o")
tools = [TavilySearchResults()]

# 自动绑定工具
llm_with_tools = llm.bind_tools(tools)
```

### MCP 集成

```python
from langchain_mcp_adapters import MultiServerMCPClient

mcp_client = MultiServerMCPClient({
    "food-safety-kb": {
        "url": "http://localhost:8765/mcp",
        "transport": "streamable_http",
    }
})
tools = await mcp_client.get_tools()
```

## 🧠 内存管理

### 短期内存

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-123"}}
app.invoke({"messages": [...]}, config)  # 同一 thread 跨调用记忆
```

### 长期内存 (Store)

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
app = graph.compile(store=store)

# 跨 thread 存储用户偏好
app.update_store(
    config,
    namespace=("user-prefs", user_id),
    key="allergies",
    value=["peanut", "shellfish"]
)
```

## 🚀 部署

### LangSmith Deployment

```bash
# 部署到 LangSmith
langgraph deploy --config langgraph.json

# 自动获得：
# - 水平扩展
# - 队列管理
# - 监控告警
# - Cron jobs
```

### langgraph.json

```json
{
  "dependencies": ["./"],
  "graphs": {
    "food_safety_agent": "./agent.py:app"
  },
  "env": "./.env"
}
```

## 🍔 食品安全应用案例

### 1. 食品危害事件分析 Agent

```python
class State(TypedDict):
    report: str
    hazard: str
    product: str
    severity: str
    regulatory_refs: list

graph = StateGraph(State)
graph.add_node("classify_hazard", hazard_classifier_node)
graph.add_node("extract_product", product_extractor_node)
graph.add_node("assess_severity", severity_assessor_node)
graph.add_node("fetch_regulations", regulation_fetcher_node)
graph.add_node("generate_report", report_generator_node)

graph.add_edge(START, "classify_hazard")
graph.add_edge("classify_hazard", "extract_product")
graph.add_edge("extract_product", "assess_severity")
graph.add_edge("assess_severity", "fetch_regulations")
graph.add_edge("fetch_regulations", "generate_report")
graph.add_edge("generate_report", END)
```

### 2. HACCP 多步审计

```python
# 状态化 HACCP 审计
graph.add_node("hazard_analysis", ha_node)
graph.add_node("determine_ccp", ccp_node)
graph.add_node("set_critical_limits", limits_node)
graph.add_node("monitoring_procedures", monitor_node)
graph.add_node("corrective_actions", corrective_node)
graph.add_node("verification", verify_node)
graph.add_node("record_keeping", record_node)
```

## 🔗 生态

### Deep Agents

```python
from langchain.agents import create_deep_agent

deep_agent = create_deep_agent(
    llm=llm,
    tools=[...],
    system_prompt="You are a food safety auditor..."
)
```

### LangSmith Studio

可视化原型设计 + 调试 + 部署。

## 📚 学习资源

- [Docs](https://docs.langchain.com/oss/python/langgraph/)
- [awesome-LangGraph](https://github.com/vonzosten/awesome-LangGraph) — 资源索引
- [ChatFood](https://github.com/mohammadi-milad-mim/ChatFood) — 食品 RAG 示例
- [Annapurna](https://github.com/shashanksrajak/chatbot-agent-food-ordering) — 食品订购 Agent
- [WhatsEat](https://github.com/NUS-AIS-Practice-Modules/WhatsEat-backend-LangGraph-supervisor-py) — 多 Agent supervisor

## 🌟 为什么是 35k★？

1. **生产级可靠**（Klarna, Replit, Elastic 在用）
2. **完整生态**（LangChain + LangSmith + Deep Agents）
3. **状态管理清晰**（StateGraph + Checkpointer）
4. **Human-in-the-Loop 内置**
5. **部署友好**（LangSmith Deployment）

## 🔧 与其他框架对比

| 框架 | 特点 | 适合 |
|---|---|---|
| **LangGraph** | 图编排、状态化 | 复杂生产 |
| **CrewAI** | 角色化快速原型 | 4+ Agent 团队 |
| **AutoGen** | 对话驱动 | 多 Agent 对话 |
| **AFlow** | MCTS 自动发现 | 工作流自动化 |
| **DSPy** | Prompt 自动优化 | 模块化 LLM 流水线 |

最佳实践：**LangGraph 管 workflow，DSPy 优化 prompt，AFlow 自动发现结构**。
