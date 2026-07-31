# 第九章：LangGraph 基础篇

本章介绍 LangGraph 的核心概念与基础用法。LangGraph 是 LangChain 团队推出的图编排框架，用于构建有状态的、多角色的 AI 应用。掌握 StateGraph、State、Node、Edge 等概念是构建复杂 AI 工作流的前提。

> 下一章：[10_LangGraph高级篇](10_LangGraph高级篇.md) | 上一章：[08_RAG篇](08_RAG篇.md)

---

## 9.1 LangGraph 概述

### 什么是 LangGraph

LangGraph 是一个基于有向图的 AI 工作流编排框架。它将 Agent 工作流建模为有向图，支持循环、分支和人工介入，弥补了 LangChain LCEL 在复杂流程编排上的不足。

### 为什么需要 LangGraph

| 痛点 | LangChain LCEL 的局限 | LangGraph 的解决方案 |
|------|----------------------|---------------------|
| **循环推理** | LCEL 链是线性的，无法循环 | 条件边可实现节点循环（如 Agent 反复推理） |
| **分支路由** | 需要手动编写 if-else | 条件边 + 路由函数，声明式定义分支 |
| **状态管理** | 无内置状态管理 | State + Reducer，自动管理状态传递与更新 |
| **人工介入** | 不支持暂停/恢复 | interrupt() + Command(resume) 实现断点 |
| **并行处理** | 需要额外编排 | Send 动态扇出，自动并行执行 |

### LangGraph 与 LangChain 的关系

```
LangChain 生态
├── LangChain Core    → LCEL、Prompt、OutputParser 等基础组件
├── LangGraph         → 图编排框架，构建有状态的工作流
├── LangServe         → 将 Chain/Graph 部署为 REST API
└── LangSmith         → 追踪、调试、评估
```

LangGraph 基于 LangChain Core 构建，复用了 LCEL 的 Runnable 接口、消息类型等基础组件，同时提供了图编排能力。

---

## 9.2 核心概念总览

| 概念 | 说明 | 对应 API |
|------|------|----------|
| **StateGraph** | 状态图，工作流的核心载体 | `StateGraph(StateType)` |
| **State** | 状态，存储工作流运行过程中的数据 | `TypedDict` 定义 |
| **Node** | 节点，执行具体逻辑的单元 | `graph.add_node(name, func)` |
| **Edge** | 边，连接节点、定义流程走向 | `graph.add_edge(from, to)` |
| **条件边** | 根据状态动态选择下一个节点 | `graph.add_conditional_edges()` |
| **START / END** | 特殊节点，标记流程的起点和终点 | `from langgraph.graph import START, END` |

### StateGraph 工作流程

```
START → NodeA → NodeB → [条件边] → NodeC/NodeD → END
```

1. **定义状态**：用 `TypedDict` 定义状态结构
2. **构建图**：创建 `StateGraph`，添加节点和边
3. **编译图**：调用 `graph.compile()` 生成可执行的应用
4. **运行图**：调用 `app.invoke(input)` 执行工作流

---

## 9.3 状态图与节点（langgraph_basics.py）

### 运行方式

```bash
python src/chains/langgraph_basics.py
```

程序启动后进入交互式菜单，可选择运行 4 个示例。

### 知识点

| 概念 | 说明 |
|------|------|
| **StateGraph** | 状态图，通过定义状态类型和节点函数来构建有向图 |
| **add_node()** | 添加节点，参数为节点名称和节点函数 |
| **add_edge()** | 添加边，参数为源节点和目标节点 |
| **compile()** | 编译图，生成可执行的 CompiledGraph |
| **invoke()** | 运行图，传入初始状态，返回最终状态 |

### 示例1：简单状态图 — 订单处理流程

**场景**：用户下单后，自动流转：下单 → 确认 → 发货 → 完成

**核心代码**：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 定义状态
class OrderState(TypedDict):
    order_info: str
    current_step: str
    confirm_result: str
    ship_result: str
    final_result: str

# 定义节点函数
def order_place(state: OrderState) -> dict:
    # 处理订单，返回状态更新
    return {"current_step": "confirmed", "confirm_result": response.content}

# 构建状态图
graph = StateGraph(OrderState)
graph.add_node("place", order_place)
graph.add_node("confirm", order_confirm)
graph.add_node("ship", order_ship)
graph.add_node("complete", order_complete)

# 添加边：定义流程走向
graph.add_edge(START, "place")
graph.add_edge("place", "confirm")
graph.add_edge("confirm", "ship")
graph.add_edge("ship", "complete")
graph.add_edge("complete", END)

# 编译并运行
app = graph.compile()
result = app.invoke({"order_info": order_info, "current_step": ""})
```

**学习要点**：

- `StateGraph` 通过 `TypedDict` 定义状态结构，字段有明确类型
- 节点函数接收完整状态，返回部分更新（只需返回变化的字段）
- `add_edge(START, "place")` 定义入口，`add_edge("complete", END)` 定义出口
- `compile()` 将图定义编译为可执行对象

### 示例2：节点与边 — 学习助手

**场景**：用户输入问题 → 自动分类 → 路由到不同专家回答

**核心代码**：

```python
# 添加条件边：分类后路由到不同专家
graph.add_conditional_edges(
    "classify",           # 源节点
    route_by_category,    # 路由函数：接收状态，返回目标节点名
    {                     # 映射表：返回值与节点的对应关系
        "math_expert": "math_expert",
        "programming_expert": "programming_expert",
        "language_expert": "language_expert",
        "general_expert": "general_expert",
    }
)
```

**学习要点**：

- `add_conditional_edges()` 实现动态路由，根据状态选择下一个节点
- 路由函数接收状态，返回目标节点名称字符串
- 映射表定义了路由函数返回值与节点的对应关系
- 一个节点可以通过条件边连接到多个后续节点

### 示例3：条件边 — 智能客服

**场景**：用户问题 → 意图检测 → 路由到不同客服处理（退货/产品/物流/投诉）

**核心代码**：

```python
def route_intent(state: CustomerServiceState) -> str:
    """根据意图路由到不同处理节点"""
    intent = state.get("intent", "产品咨询")
    route_map = {
        "退货退款": "refund",
        "产品咨询": "product",
        "物流查询": "logistics",
        "投诉建议": "complaint",
    }
    return route_map.get(intent, "product")

graph.add_conditional_edges("detect", route_intent, {
    "refund": "refund", "product": "product",
    "logistics": "logistics", "complaint": "complaint",
})
```

**学习要点**：

- 条件边是 LangGraph 最强大的特性之一
- 实现了"意图识别 → 分发处理"的典型客服架构
- 每个处理节点独立实现，互不影响，便于维护

### 示例4：完整工作流 — 旅行规划助手

**场景**：目的地 → 景点推荐 → 美食攻略 → 行程规划 → 预算建议 → 最终方案

**核心代码**：

```python
# 多节点串联构建复杂工作流
graph.add_edge(START, "attractions")
graph.add_edge("attractions", "food")
graph.add_edge("food", "itinerary")
graph.add_edge("itinerary", "budget")
graph.add_edge("budget", "final")
graph.add_edge("final", END)
```

**学习要点**：

- 多节点串联构建复杂工作流，每个节点职责单一
- 前序节点的输出作为后序节点的输入，State 自动传递
- START/END 标记起止，图结构清晰可读

---

## 9.4 状态管理（langgraph_state.py）

### 运行方式

```bash
python src/chains/langgraph_state.py
```

### 知识点

| 概念 | 说明 |
|------|------|
| **TypedDict** | 定义状态类型，确保字段有明确类型 |
| **Annotated** | 绑定更新策略，指定字段如何被更新 |
| **operator.add** | 追加更新（列表拼接），适合累积增长场景 |
| **add_messages** | 消息专用更新策略，智能处理消息ID去重 |
| **MemorySaver** | 内存检查点，实现状态持久化 |

### 状态更新策略对比

| 更新策略 | 代码写法 | 行为 | 适用场景 |
|---------|---------|------|---------|
| **覆盖更新**（默认） | `field: str` | 新值替换旧值 | 每次重新生成的数据 |
| **追加更新** | `field: Annotated[list, operator.add]` | 旧值 + 新值 | 累积增长的列表 |
| **消息追加** | `field: Annotated[list, add_messages]` | 智能追加+ID去重 | 聊天消息列表 |

### 示例1：基础状态 — 计数器

**核心代码**：

```python
class CounterState(TypedDict):
    count: int       # 覆盖更新
    history: list    # 覆盖更新（需手动拼接）
    user_input: str

def add_number(state: CounterState) -> dict:
    old_count = state["count"]
    new_count = old_count + num           # 手动读取旧值
    new_history = state["history"] + [...]  # 手动拼接
    return {"count": new_count, "history": new_history}
```

**学习要点**：基础类型字段默认覆盖更新，需要手动读取旧值来计算新值。

### 示例2：Reducer 追加更新 — 聊天记录

**核心代码**：

```python
class ChatState(TypedDict):
    messages: Annotated[list, operator.add]  # 追加更新
    user_input: str

def chat_reply(state: ChatState) -> dict:
    new_messages = [HumanMessage(content=user_input), AIMessage(content=response.content)]
    return {"messages": new_messages}  # operator.add 自动追加到旧列表
```

**学习要点**：`Annotated[list, operator.add]` 让列表字段自动追加，节点只需返回新增部分。

### 示例3：Annotated 状态 — 任务管理

**核心代码**：

```python
class TaskState(TypedDict):
    tasks: Annotated[list, operator.add]   # 追加更新
    completed: Annotated[list, operator.add]  # 追加更新
    summary: str                            # 覆盖更新
```

**学习要点**：同一状态中不同字段可使用不同更新策略，覆盖适合"重新生成"，追加适合"累积增长"。

### 示例4：状态持久化 — 对话记忆

**核心代码**：

```python
from langgraph.checkpoint.memory import MemorySaver

class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "session1"}}
result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
state = app.get_state(config)  # 获取当前状态
```

**学习要点**：

- `add_messages` 是 LangGraph 内置的消息专用更新策略，智能处理ID去重
- `MemorySaver` 提供内存检查点，支持中断恢复
- `thread_id` 区分不同会话，实现多会话隔离

---

## 9.5 控制流（langgraph_control_flow.py）

### 运行方式

```bash
python src/chains/langgraph_control_flow.py
```

### 知识点

| 概念 | 说明 |
|------|------|
| **add_conditional_edges** | 条件边，根据状态动态选择下一个节点 |
| **循环逻辑** | 条件边指向源节点自身，形成循环（必须有退出条件） |
| **Send** | 动态扇出，一个节点向多个节点发送数据（Map-Reduce） |
| **子图(Subgraph)** | 图嵌套，将复杂工作流拆分为多个子图 |

### 示例1：条件边 — 智能路由

**场景**：问题 → 分类 → 路由到医学/法律/技术/教育专家

**学习要点**：`add_conditional_edges` 实现动态路由，比硬编码 if-else 更灵活。

### 示例2：循环逻辑 — 代码审查

**核心代码**：

```python
# 条件边实现循环
graph.add_conditional_edges(
    "review",
    should_continue,    # 路由函数
    {
        "review": "review",       # 循环：回到自身
        "approved": "approved",   # 退出：通过
        "max_rounds": "max_rounds",  # 退出：最大轮次
    }
)
```

**学习要点**：

- 条件边可以指向源节点自身，形成循环
- 必须有退出条件（审查通过/最大轮次），防止无限循环
- 每次循环将上一次的反馈作为输入，实现迭代优化

### 示例3：Send 动态扇出 — 多维度分析

**核心代码**：

```python
from langgraph.types import Send

def continue_to_dimensions(state: AnalysisState) -> list:
    """动态扇出函数：根据维度列表发送到多个分析节点"""
    return [
        Send("analyze", {"topic": state["topic"], "dimension": dim})
        for dim in state["dimensions"]
    ]

graph.add_conditional_edges("determine", continue_to_dimensions, ["analyze"])
```

**学习要点**：

- `Send` 实现动态扇出，维度数量在运行时动态决定
- 类似 Map-Reduce 模式：并行分析 → 汇总报告
- `analysis_results` 使用 `operator.add` 自动追加各维度结果

### 示例4：子图嵌套 — 复杂工作流

**核心代码**：

```python
# 构建子图
requirement_subgraph = build_requirement_subgraph().compile()
design_subgraph = build_design_subgraph().compile()

# 在父图中使用子图（像普通节点一样）
graph.add_node("requirement", requirement_subgraph)
graph.add_node("design", design_subgraph)
```

**学习要点**：

- 子图有独立的状态和逻辑，降低复杂度
- 子图编译后像普通节点一样被添加到父图
- 适合将复杂工作流拆分为可复用的子模块

---

## 9.6 学习路径与建议

### 推荐学习顺序

```
langgraph_basics.py（示例1→2→3→4）
    ↓ 掌握 StateGraph/Node/Edge 基础
langgraph_state.py（示例1→2→3→4）
    ↓ 掌握状态管理与持久化
langgraph_control_flow.py（示例1→2→3→4）
    ↓ 掌握条件边/循环/Send/子图
```

### 核心概念掌握检查

| 阶段 | 必须掌握 | 进阶理解 |
|------|---------|---------|
| 基础 | StateGraph、add_node、add_edge、compile、invoke | 节点函数的设计模式 |
| 状态 | TypedDict、Annotated、operator.add | add_messages、MemorySaver |
| 控制流 | add_conditional_edges、循环逻辑 | Send 动态扇出、子图嵌套 |

### 实战建议

1. **先理解状态**：LangGraph 的核心是状态驱动的图，先搞清楚"状态是什么、怎么更新"
2. **画图再写代码**：先在纸上画出节点和边的流程图，再翻译为代码
3. **从简单到复杂**：先写线性流程（示例1），再加条件边（示例2/3），最后加循环和并行
4. **注意退出条件**：循环必须有退出条件，否则会无限循环

> 下一章：[10_LangGraph高级篇](10_LangGraph高级篇.md) → 学习 Agent、Human-in-the-Loop、多 Agent 协作
