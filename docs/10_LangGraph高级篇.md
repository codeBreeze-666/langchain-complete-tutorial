# 第十章：LangGraph 高级篇

本章介绍 LangGraph 的高级应用：Agent 模式、Human-in-the-Loop（人工介入）和多 Agent 协作。这些模式基于基础篇的 StateGraph/Node/Edge 构建，是实现复杂 AI 系统的关键能力。

> 下一章：[11_LangServe部署篇](11_LangServe部署篇.md) | 上一章：[09_LangGraph基础篇](09_LangGraph基础篇.md)

---

## 10.1 Agent 模式（langgraph_agent.py）

### 运行方式

```bash
python src/chains/langgraph_agent.py
```

程序启动后进入交互式菜单，可选择运行 4 个示例。

### 知识点

| 概念 | 说明 |
|------|------|
| **create_agent** | LangChain v1 内置函数，一行代码创建 ReAct Agent（底层运行在 LangGraph 上） |
| **ToolNode** | 专门的图节点，自动处理 LLM 的工具调用请求 |
| **should_continue** | 路由函数，决定 Agent 是继续调用工具还是结束循环 |
| **工具调用循环** | Agent推理 → 调用工具 → 获取结果 → 继续推理 → 直到完成 |

### Agent 架构对比

| Agent 模式 | 特点 | 适用场景 |
|-----------|------|---------|
| **ReAct Agent** | 推理→行动→观察循环，自动选择工具 | 知识问答、信息查询 |
| **工具调用 Agent** | 手动构建 StateGraph，理解内部原理 | 自定义 Agent 逻辑 |
| **循环推理 Agent** | 多轮迭代逐步逼近目标 | 代码调试、问题修复 |
| **自我纠错 Agent** | 生成→评估→修正循环，质量门槛退出 | 文本优化、方案完善 |

### 示例1：ReAct Agent — 知识问答

**场景**：用户输入问题，Agent 自动推理和调用知识库工具

**核心代码**：

```python
from langchain.agents import create_agent

llm = get_default_llm()
tools = [search_knowledge]

# 一行代码创建 ReAct Agent
agent = create_agent(llm, tools)

# 调用 Agent
result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
```

**学习要点**：

- `create_agent(llm, tools)` 一行创建 ReAct Agent
- 内部自动构建：Agent节点 → ToolNode → 路由判断 → 循环或结束
- LLM 根据工具的 docstring 自动决定何时调用哪个工具
- ReAct 循环：Reason(推理) → Act(调用工具) → Observe(观察结果) → 继续推理或结束

### 示例2：工具调用 Agent — 生活助手

**场景**：用户需求自动调用天气/计算/搜索等工具

**核心代码**：

```python
from langgraph.prebuilt import ToolNode

# 绑定工具到 LLM
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# 定义路由函数
def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# 手动构建 StateGraph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
```

**学习要点**：

- `ToolNode(tools)` 自动处理 LLM 的工具调用请求
- `should_continue` 路由函数通过检查 `tool_calls` 决定流程走向
- 手动构建揭示了 `create_agent` 的内部原理
- 工具调用循环：Agent → tools → Agent → ... → END

### 示例3：循环推理 Agent — 代码调试

**核心代码**：

```python
MAX_ITERATIONS = 5

def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_messages = [m for m in state["messages"] if m.type == "tool"]
        if len(tool_messages) >= MAX_ITERATIONS:
            return END  # 达到最大迭代次数，退出
        return "tools"
    return END
```

**学习要点**：

- 循环推理让 Agent 可以多轮迭代，逐步逼近目标
- `MAX_ITERATIONS` 防止无限循环，是生产环境必备的安全阀
- 每轮推理的中间结果保存在状态中，Agent 可以回顾历史
- 适用于调试、修复、优化等需要迭代改进的场景

### 示例4：自我纠错 Agent — 文本优化

**核心代码**：

```python
# 自我纠错 = 生成节点 + 评估节点 + 路由函数
workflow.add_node("generate", generate_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("final", final_node)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "evaluate")
workflow.add_conditional_edges("evaluate", should_continue_correction, {
    "generate": "generate",  # 不满意，继续修正
    "final": "final"          # 满意，输出结果
})
```

**学习要点**：

- 自我纠错 = 生成节点 + 评估节点 + 路由函数
- 评估节点对输出打分，达标则退出，不达标则继续修正
- 质量门槛（评分阈值）和迭代上限是两道安全阀
- 适用于文本优化、代码修复、方案完善等需要迭代的场景

---

## 10.2 Human-in-the-Loop（langgraph_human_in_loop.py）

### 运行方式

```bash
python src/chains/langgraph_human_in_loop.py
```

### 知识点

| 概念 | 说明 |
|------|------|
| **interrupt()** | 在工作流中插入断点，暂停执行并等待人工介入 |
| **Command(resume=...)** | 恢复被中断的工作流，传入人工审批结果 |
| **checkpointer** | 检查点机制，自动保存工作流状态，支持断点恢复 |
| **断点恢复** | 从中断的位置继续执行，无需从头开始 |

### Human-in-the-Loop 模式对比

| 模式 | 流程 | 适用场景 |
|------|------|---------|
| **内容审批** | 生成 → 审批 → 发布/修订 | 敏感内容发布 |
| **邮件审批** | 撰写 → 预览 → 确认发送 | 邮件等不可撤回操作 |
| **交互式决策** | 分析 → 提供选项 → 用户选择 → 执行 | 数据处理方式选择 |
| **断点恢复** | 分步执行 → 每步暂停 → 可恢复 | 长任务管理 |

### 示例1：interrupt() 函数 — 内容审批

**场景**：AI 生成内容后暂停，等待人工审批通过才发布

**核心代码**：

```python
from langgraph.types import interrupt, Command

def review_node(state: ApprovalState) -> dict:
    """审批节点 - 使用 interrupt() 暂停等待人工审批"""
    decision = interrupt({
        "type": "content_review",
        "content": state["content"],
        "message": "AI 已生成内容，请审批："
    })
    # 当工作流恢复时，decision 就是 Command(resume=...) 传入的值
    return {
        "approved": decision.get("approved", False),
        "feedback": decision.get("feedback", ""),
    }

# 恢复工作流，传入审批结果
graph.invoke(Command(resume={"approved": True, "feedback": "通过"}), config=config)
```

**学习要点**：

- `interrupt()` 在节点中调用，暂停工作流并返回信息给调用者
- `Command(resume=...)` 恢复工作流，传入人工审批结果
- 审批不通过可路由到修订节点，修订后再次审批
- 必须配合 `checkpointer` 使用，否则中断后无法恢复

### 示例2：人工审批流程 — 邮件发送

**场景**：AI 撰写邮件后暂停，等待人工确认才发送

**学习要点**：

- 多步审批：生成 → 审核 → 发送，每步可插入 interrupt
- 条件恢复：根据用户选择决定恢复后的执行路径
- 确保敏感操作（如邮件发送）不会自动执行

### 示例3：交互式决策 — 数据处理

**场景**：AI 分析数据后暂停，让用户选择处理方式（A/B/C）

**核心代码**：

```python
def decision_node(state: DataState) -> dict:
    """决策节点 - 让用户选择处理方式"""
    decision = interrupt({
        "type": "data_processing_decision",
        "analysis": state["analysis"],
        "options": ["A", "B", "C"],
    })
    return {"chosen_method": decision.get("method", "A")}

# 路由函数根据用户选择分发到不同处理节点
def should_process(state: DataState) -> str:
    method = state.get("chosen_method", "A")
    if method == "A": return "process_a"
    elif method == "B": return "process_b"
    else: return "process_c"
```

**学习要点**：

- `interrupt()` 可以传入多个选项，让用户做决策
- 路由函数根据用户选择分发到不同处理节点
- 所有处理路径最终汇聚到汇总节点，统一输出

### 示例4：断点恢复 — 长任务管理

**场景**：AI 分步执行复杂任务，每步暂停等待继续指令

**学习要点**：

- 分步执行 + interrupt()：长任务拆分为多步，每步暂停等待指令
- checkpointer 自动保存状态，中断后可恢复
- `thread_id` 标识不同对话线程，每个线程有独立检查点
- 断点恢复：用相同的 `thread_id` 调用 `graph.invoke()` 即可从断点继续

---

## 10.3 多 Agent 协作（langgraph_multi_agent.py）

### 运行方式

```bash
python src/chains/langgraph_multi_agent.py
```

### 知识点

| 概念 | 说明 |
|------|------|
| **多Agent协作** | 多个智能体在同一个图中协同工作，各自承担不同职责 |
| **主管(Supervisor)** | 负责任务分解和委派的中心 Agent |
| **流水线(Pipeline)** | 多个 Agent 按顺序执行，前一步输出是后一步输入 |
| **状态共享** | 所有 Agent 通过共享的 State 对象交换信息 |

### 多 Agent 协作模式对比

| 模式 | 结构 | 特点 | 适用场景 |
|------|------|------|---------|
| **角色委派** | START → 多角色(并行) → 汇总 | 各角色独立分析，结果汇聚 | 需求评审、方案评审 |
| **流水线** | Agent1 → Agent2 → Agent3 → ... | 顺序执行，逐步精炼 | 文章创作、翻译润色 |
| **辩论模式** | 正方 → 反方 → 总结 → 评委 | 对抗性分析，结论客观 | 决策分析、风险评估 |
| **主管模式** | 主管 → 专业Agent → 主管 → ... | 中心调度，反馈迭代 | 复杂任务、混合类型 |

### 示例1：角色委派 — 项目管理

**场景**：产品经理、技术负责人、测试负责人并行分析需求后汇总

**核心代码**：

```python
# 从 START 分发到三个角色（LangGraph 自动并行执行）
workflow.add_edge(START, "pm_agent")
workflow.add_edge(START, "tech_agent")
workflow.add_edge(START, "qa_agent")

# 三个角色完成后汇聚到汇总节点
workflow.add_edge("pm_agent", "summary_agent")
workflow.add_edge("tech_agent", "summary_agent")
workflow.add_edge("qa_agent", "summary_agent")
```

**学习要点**：

- 从 START 到多个角色的边实现并行执行
- 多个角色到汇总节点的边实现结果汇聚
- 所有 Agent 通过共享的 State 对象交换信息

### 示例2：流水线模式 — 文章创作

**场景**：大纲 → 初稿 → 润色 → 终稿，四个 Agent 依次处理

**核心代码**：

```python
# 顺序边：确保严格按流水线执行
workflow.add_edge(START, "outline")
workflow.add_edge("outline", "draft")
workflow.add_edge("draft", "polish")
workflow.add_edge("polish", "final")
workflow.add_edge("final", END)
```

**学习要点**：

- 流水线 = 顺序边连接的多个 Agent 节点
- 上一步的输出写入 State，下一步从 State 读取
- `add_edge(A, B)` 保证 A 完成后 B 才执行

### 示例3：辩论模式 — 决策分析

**场景**：正方发言 → 反方反驳 → 正方总结 → 评委评判

**学习要点**：

- 辩论模式 = 顺序边连接的对抗性 Agent 节点
- 反方通过 State 读取正方论点，形成针对性反驳
- 评委节点综合所有信息，输出比单方分析更客观的结论

### 示例4：主管 Agent — 复杂任务

**场景**：主管分析任务 → 委派给专业 Agent → 评估结果 → 决定继续或完成

**核心代码**：

```python
# 主管根据任务类型分发到专业 Agent
workflow.add_conditional_edges("supervisor", route_to_worker, {
    "writing": "writing",
    "research": "research",
    "coding": "coding",
})

# 专业 Agent 完成后回到主管评估
workflow.add_edge("writing", "supervisor")
workflow.add_edge("research", "supervisor")
workflow.add_edge("coding", "supervisor")

# 主管评估后决定继续还是完成
workflow.add_conditional_edges("supervisor", should_continue_supervisor, {
    "writing": "writing", "research": "research",
    "coding": "coding", "final": "final",
})
```

**学习要点**：

- 主管节点是图的中心，负责任务分析和分配
- 条件路由根据任务类型分发到不同 Agent
- 专业 Agent 完成后回到主管，形成反馈循环
- 主管可以多轮迭代，直到输出质量达标

---

## 10.4 学习路径与建议

### 推荐学习顺序

```
langgraph_agent.py（示例1→2→3→4）
    ↓ 掌握 Agent 模式（ReAct/工具调用/循环推理/自我纠错）
langgraph_human_in_loop.py（示例1→2→3→4）
    ↓ 掌握人工介入（interrupt/Command/checkpointer）
langgraph_multi_agent.py（示例1→2→3→4）
    ↓ 掌握多 Agent 协作（角色委派/流水线/辩论/主管）
```

### 核心概念掌握检查

| 阶段 | 必须掌握 | 进阶理解 |
|------|---------|---------|
| Agent | create_agent、ToolNode、should_continue | 循环推理、自我纠错 |
| 人工介入 | interrupt()、Command(resume) | 断点恢复、多步审批 |
| 多Agent | 角色委派、流水线 | 辩论模式、主管模式 |

### 实战建议

1. **从 create_agent 开始**：先用快捷函数创建 Agent，理解 ReAct 循环后再手动构建
2. **理解 interrupt 机制**：Human-in-the-Loop 是生产环境的关键能力，确保敏感操作有人审批
3. **多 Agent 不是越多越好**：根据任务复杂度选择协作模式，简单任务用单 Agent 即可
4. **设置迭代上限**：所有循环都必须有退出条件，防止无限循环
5. **利用 LangSmith 追踪**：复杂 Agent 系统的调试非常困难，善用追踪工具

> 下一章：[11_LangServe部署篇](11_LangServe部署篇.md) → 学习将 Chain/Graph 部署为 REST API
