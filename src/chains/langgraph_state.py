"""
LangGraph 状态管理案例 - 基础状态、Reducer追加、Annotated状态、状态持久化
============================================================================

本示例演示 LangGraph 中状态管理的核心概念和高级用法，包含四个交互式案例。

核心概念：
- TypedDict: 定义状态类型，Python 标准库的类型提示工具，确保状态字段有明确类型
- Annotated: 绑定更新策略，指定状态字段如何被更新（覆盖 vs 追加）
- operator.add: 追加更新（列表拼接），新值与旧值用 + 合并，适合列表类状态
- add_messages: 消息追加更新，LangGraph 内置的消息专用更新策略，智能处理消息ID去重
- MemorySaver: 内存检查点（状态持久化），将状态保存到内存，支持中断恢复

应用场景：
- 计数器：简单的状态累加场景
- 聊天记录：消息追加与历史管理
- 任务管理：复杂状态的多字段管理
- 对话记忆：多轮对话的状态持久化与恢复
"""

import os
import sys
import operator
from typing import TypedDict, Annotated

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.utils.llm_loader import get_default_llm


# ============================================================
# 示例1: 基础状态 - 计数器
# ============================================================

class CounterState(TypedDict):
    """计数器状态 - 使用基础类型字段

    默认行为：每次节点返回的字段值会覆盖旧值
    例如：count=5 → 节点返回 count=3 → count 变为 3
    """
    count: int             # 当前计数值
    history: list          # 历史记录
    user_input: str        # 用户输入的数字


def add_number(state: CounterState) -> dict:
    """累加节点：将用户输入的数字加到计数器上

    注意：基础类型字段是覆盖更新，不是累加
    所以需要手动读取旧值并计算新值
    """
    try:
        num = int(state["user_input"])
    except ValueError:
        num = 0
        print(f"  [提示] 输入无效，使用默认值 0")

    old_count = state["count"]
    new_count = old_count + num
    new_history = state["history"] + [f"{old_count} + {num} = {new_count}"]

    print(f"  [累加] {old_count} + {num} = {new_count}")
    return {"count": new_count, "history": new_history}


def show_summary(state: CounterState) -> dict:
    """汇总节点：显示当前状态"""
    print(f"  [汇总] 当前值: {state['count']}")
    print(f"  [汇总] 历史记录: {state['history']}")
    return {}  # 不更新状态


def demo_counter():
    """示例1：基础状态 - 计数器

    实战要点：
    - TypedDict 定义状态结构，字段有明确类型
    - 基础类型字段（int/str）默认是覆盖更新
    - 需要手动读取旧值来计算新值（如累加）
    - 列表字段也需要手动拼接（新列表 = 旧列表 + 新元素）
    """
    print("\n" + "=" * 60)
    print("示例1：基础状态 - 计数器")
    print("=" * 60)
    print("""
核心概念：
  TypedDict: 定义状态类型，确保字段有明确类型
  默认更新策略: 覆盖（新值替换旧值）

注意：基础类型字段（int/str）默认是覆盖更新
  - count=5 → 节点返回 count=3 → count 变为 3
  - 需要手动读取旧值来计算新值
    """)

    # 构建状态图
    graph = StateGraph(CounterState)

    # 添加节点
    graph.add_node("add", add_number)
    graph.add_node("summary", show_summary)

    # 添加边
    graph.add_edge(START, "add")
    graph.add_edge("add", "summary")
    graph.add_edge("summary", END)

    # 编译图
    app = graph.compile()

    print("【交互式计数器】")
    print("输入数字，状态自动累加")
    print("\n输入 '退出' 结束，输入 '重置' 清零\n")

    while True:
        user_input = input("请输入一个数字：").strip()
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if user_input.lower() in ["重置", "reset"]:
            print("计数器已重置为 0")
            print("-" * 40)
            continue

        try:
            # 每次都从初始状态开始（基础状态没有持久化）
            result = app.invoke({
                "count": 0,
                "history": [],
                "user_input": user_input,
            })
            print(f"  结果: count={result['count']}, history={result['history']}")
        except Exception as e:
            print(f"错误：{e}")

        print("-" * 40)

    print("\n实战要点总结：")
    print("   1. TypedDict 定义状态，字段有明确类型")
    print("   2. 基础类型字段默认覆盖更新，需手动读取旧值")
    print("   3. 列表字段也需手动拼接，不能直接 append")


# ============================================================
# 示例2: Reducer追加更新 - 聊天记录
# ============================================================

class ChatState(TypedDict):
    """聊天状态 - 使用 Annotated + operator.add 实现列表追加

    operator.add 的作用：
    - 默认行为：messages = 新值（覆盖）
    - 使用 operator.add：messages = 旧值 + 新值（追加/拼接）
    - 适合列表类状态字段，无需手动拼接
    """
    messages: Annotated[list, operator.add]   # 消息列表（追加更新）
    user_input: str                            # 用户输入


def chat_reply(state: ChatState) -> dict:
    """聊天回复节点：用户输入消息，AI自动回复"""
    llm = get_default_llm()
    user_input = state["user_input"]

    # 构建消息历史
    messages = state["messages"]
    # 添加用户消息
    new_messages = [HumanMessage(content=user_input)]

    # 调用 LLM
    all_messages = messages + new_messages
    response = llm.invoke(all_messages)

    # 添加 AI 回复
    new_messages.append(AIMessage(content=response.content))

    print(f"  [用户] {user_input}")
    print(f"  [AI] {response.content[:80]}...")

    # 返回新消息，operator.add 会自动追加到旧列表
    return {"messages": new_messages}


def demo_chat_history():
    """示例2：Reducer追加更新 - 聊天记录

    实战要点：
    - Annotated[list, operator.add] 让列表字段自动追加更新
    - 节点只需返回新增的消息，框架自动拼接
    - 对比示例1：不再需要手动 messages = old + new
    - operator.add 本质是列表的 + 操作符（拼接）
    """
    print("\n" + "=" * 60)
    print("示例2：Reducer追加更新 - 聊天记录")
    print("=" * 60)
    print("""
核心概念：
  Annotated: 绑定更新策略，指定字段如何被更新
  operator.add: 追加更新（列表拼接）

对比基础状态：
  - 基础：messages = 新值（覆盖）
  - operator.add：messages = 旧值 + 新值（追加）

代码示例：
  messages: Annotated[list, operator.add]
  节点返回 {"messages": [new_msg]}
  → 自动变为 旧messages + [new_msg]
    """)

    # 构建状态图
    graph = StateGraph(ChatState)

    # 添加节点
    graph.add_node("chat", chat_reply)

    # 添加边
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    # 编译图
    app = graph.compile()

    print("【交互式聊天记录】")
    print("输入消息，自动追加到历史记录")
    print("\n输入 '退出' 结束，输入 '清空' 重置\n")

    # 用一个变量手动维护多轮对话状态
    messages = []

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if user_input.lower() in ["清空", "clear"]:
            messages = []
            print("聊天记录已清空")
            continue
        if not user_input:
            continue

        try:
            result = app.invoke({
                "messages": messages,
                "user_input": user_input,
            })
            # 更新消息列表，保持多轮对话
            messages = result["messages"]
            print(f"  [状态] 消息数: {len(messages)}")
        except Exception as e:
            print(f"错误：{e}")

        print("-" * 40)

    print("\n实战要点总结：")
    print("   1. Annotated[list, operator.add] 实现列表自动追加")
    print("   2. 节点只返回新增部分，框架自动拼接")
    print("   3. operator.add 本质是列表的 + 操作符（拼接）")


# ============================================================
# 示例3: Annotated状态 - 任务管理
# ============================================================

class TaskState(TypedDict):
    """任务管理状态 - 使用多种 Annotated 策略

    展示同一状态中不同字段使用不同更新策略：
    - tasks: 列表追加（新任务追加到列表）
    - completed: 列表追加（已完成任务追加）
    - summary: 覆盖更新（每次重新生成摘要）
    """
    tasks: Annotated[list, operator.add]       # 待办任务（追加更新）
    completed: Annotated[list, operator.add]   # 已完成任务（追加更新）
    summary: str                                # 任务摘要（覆盖更新）
    user_input: str                             # 用户输入
    action: str                                 # 操作类型


def add_task(state: TaskState) -> dict:
    """添加任务节点"""
    llm = get_default_llm()
    user_input = state["user_input"]

    # 使用 LLM 生成任务描述
    response = llm.invoke(
        f"用户要添加一个任务：{user_input}。请将这个任务整理为简洁的任务描述，"
        f"包含优先级（高/中/低）和预估时间。只输出任务描述。"
    )
    new_task = f"[待办] {response.content}"
    print(f"  [添加任务] {new_task[:60]}...")

    return {"tasks": [new_task]}


def complete_task(state: TaskState) -> dict:
    """完成任务节点"""
    tasks = state["tasks"]
    if not tasks:
        print("  [完成] 没有待办任务")
        return {"summary": "当前没有待办任务"}

    # 完成第一个任务
    completed_task = tasks[0].replace("[待办]", "[完成]")
    remaining_tasks = tasks[1:]

    print(f"  [完成] {completed_task[:60]}...")

    return {
        "tasks": remaining_tasks,  # 覆盖更新：移除已完成的任务
        "completed": [completed_task],  # 追加更新：添加到已完成列表
    }


def generate_summary(state: TaskState) -> dict:
    """生成摘要节点"""
    llm = get_default_llm()
    tasks = state["tasks"]
    completed = state["completed"]

    response = llm.invoke(
        f"当前任务状态：\n"
        f"待办任务：{tasks}\n"
        f"已完成：{completed}\n\n"
        f"请生成一份简洁的任务进度摘要。"
    )
    print(f"  [摘要] {response.content[:80]}...")

    # summary 是覆盖更新，每次重新生成
    return {"summary": response.content}


def route_task_action(state: TaskState) -> str:
    """根据操作类型路由"""
    action = state.get("action", "add")
    if action == "add":
        return "add_task"
    elif action == "complete":
        return "complete_task"
    else:
        return "add_task"


def demo_task_manager():
    """示例3：Annotated状态 - 任务管理

    实战要点：
    - 同一状态中不同字段可以使用不同更新策略
    - tasks 列表追加、completed 列表追加、summary 覆盖
    - 覆盖更新适合"每次重新生成"的场景（如摘要）
    - 追加更新适合"累积增长"的场景（如任务列表）
    """
    print("\n" + "=" * 60)
    print("示例3：Annotated状态 - 任务管理")
    print("=" * 60)
    print("""
核心概念：
  Annotated: 绑定更新策略，同一状态中不同字段可用不同策略

字段更新策略：
  - tasks: Annotated[list, operator.add] → 追加更新
  - completed: Annotated[list, operator.add] → 追加更新
  - summary: str → 覆盖更新（默认）

选择策略的原则：
  - 累积增长（列表追加）→ operator.add
  - 每次重新生成（覆盖）→ 默认
    """)

    # 构建状态图
    graph = StateGraph(TaskState)

    # 添加节点
    graph.add_node("add_task", add_task)
    graph.add_node("complete_task", complete_task)
    graph.add_node("summary", generate_summary)

    # 添加边
    graph.add_edge(START, "add_task")  # 默认从添加任务开始
    graph.add_conditional_edges(
        START,
        route_task_action,
        {
            "add_task": "add_task",
            "complete_task": "complete_task",
        }
    )
    graph.add_edge("add_task", "summary")
    graph.add_edge("complete_task", "summary")
    graph.add_edge("summary", END)

    # 编译图
    app = graph.compile()

    print("【交互式任务管理】")
    print("输入任务描述添加任务，或输入 '完成' 来完成第一个任务")
    print("\n输入 '退出' 结束，输入 '清空' 重置\n")

    # 手动维护多轮状态
    tasks = []
    completed = []

    while True:
        user_input = input("请输入操作：").strip()
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if user_input.lower() in ["清空", "clear"]:
            tasks = []
            completed = []
            print("任务已清空")
            continue
        if not user_input:
            continue

        # 判断操作类型
        action = "complete" if user_input.lower() in ["完成", "done"] else "add"

        try:
            result = app.invoke({
                "tasks": tasks,
                "completed": completed,
                "summary": "",
                "user_input": user_input,
                "action": action,
            })
            # 更新状态
            tasks = result["tasks"]
            completed = result["completed"]

            print(f"\n  [当前状态]")
            print(f"  待办: {len(tasks)} 个, 已完成: {len(completed)} 个")
            print(f"  摘要: {result['summary'][:100]}...")
        except Exception as e:
            print(f"错误：{e}")

        print("-" * 40)

    print("\n实战要点总结：")
    print("   1. 同一状态中不同字段可使用不同更新策略")
    print("   2. operator.add 适合累积增长场景，默认覆盖适合重新生成场景")
    print("   3. 选择策略的关键：数据是累积还是替换")


# ============================================================
# 示例4: 状态持久化 - 对话记忆
# ============================================================

class ConversationState(TypedDict):
    """对话状态 - 使用 add_messages 管理消息列表

    add_messages 是 LangGraph 内置的消息专用更新策略：
    - 自动追加新消息到列表
    - 智能处理消息ID去重（相同ID的消息会被更新而非追加）
    - 适合聊天场景，是 LangGraph 推荐的消息管理方式
    """
    messages: Annotated[list, add_messages]    # 消息列表（add_messages更新策略）


def chat_with_memory(state: ConversationState) -> dict:
    """对话节点：带有记忆的对话"""
    llm = get_default_llm()
    messages = state["messages"]

    # 调用 LLM
    response = llm.invoke(messages)

    # 获取用户最后一条消息
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    print(f"  [用户] {last_user_msg}")
    print(f"  [AI] {response.content[:80]}...")

    # 返回 AI 消息，add_messages 会自动追加
    return {"messages": [AIMessage(content=response.content)]}


def demo_conversation_memory():
    """示例4：状态持久化 - 对话记忆

    实战要点：
    - add_messages 是 LangGraph 内置的消息专用更新策略
    - 智能处理消息ID去重（相同ID的消息会被更新）
    - MemorySaver 提供内存检查点，支持中断恢复
    - thread_id 区分不同会话，实现多会话隔离
    - 持久化后即使程序中断，也能从上次的状态继续
    """
    print("\n" + "=" * 60)
    print("示例4：状态持久化 - 对话记忆")
    print("=" * 60)
    print("""
核心概念：
  add_messages: 消息追加更新策略（LangGraph 内置）
  - 自动追加新消息到列表
  - 智能处理消息ID去重

  MemorySaver: 内存检查点（状态持久化）
  - 每次节点执行后自动保存状态
  - 支持 thread_id 区分不同会话
  - 程序中断后可从上次状态恢复

代码示例：
  checkpointer = MemorySaver()
  app = graph.compile(checkpointer=checkpointer)
  config = {"configurable": {"thread_id": "session1"}}
  result = app.invoke(input, config)
    """)

    # 构建状态图
    graph = StateGraph(ConversationState)

    # 添加节点
    graph.add_node("chat", chat_with_memory)

    # 添加边
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    # 使用 MemorySaver 作为检查点
    checkpointer = MemorySaver()

    # 编译图（带检查点）
    app = graph.compile(checkpointer=checkpointer)

    print("【交互式对话记忆】")
    print("输入消息进行对话，状态自动保存和恢复")
    print("\n支持多会话：输入 '新会话' 创建新会话，输入 '切换' 切换会话")
    print("\n输入 '退出' 结束\n")

    # 会话管理
    sessions = {}
    current_session = "default"

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if user_input.lower() in ["新会话", "new"]:
            session_name = input("请输入会话名称：").strip()
            if session_name:
                current_session = session_name
                sessions[current_session] = True
                print(f"已创建新会话: {current_session}")
            continue
        if user_input.lower() in ["切换", "switch"]:
            print(f"当前会话: {current_session}")
            print(f"已有会话: {list(sessions.keys()) if sessions else ['default']}")
            new_session = input("请输入要切换的会话名称：").strip()
            if new_session:
                current_session = new_session
                print(f"已切换到会话: {current_session}")
            continue
        if user_input.lower() in ["清空", "clear"]:
            current_session = f"reset_{current_session}"
            print("会话已重置")
            continue
        if not user_input:
            continue

        try:
            # 使用 thread_id 区分不同会话
            config = {"configurable": {"thread_id": current_session}}

            # 添加系统提示（首次对话时）
            messages = [HumanMessage(content=user_input)]
            result = app.invoke({"messages": messages}, config)

            # 显示当前状态
            state = app.get_state(config)
            msg_count = len(state.values.get("messages", []))
            print(f"  [状态] 会话: {current_session}, 消息数: {msg_count}")
        except Exception as e:
            print(f"错误：{e}")

        print("-" * 40)

    print("\n实战要点总结：")
    print("   1. add_messages 是 LangGraph 推荐的消息管理方式")
    print("   2. MemorySaver 实现状态持久化，支持中断恢复")
    print("   3. thread_id 区分不同会话，实现多会话隔离")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "=" * 60)
    print("  LangGraph 状态管理案例 - 基础状态、Reducer、Annotated、持久化")
    print("=" * 60)
    print("\n核心概念：")
    print("  • TypedDict: 定义状态类型，确保字段有明确类型")
    print("  • Annotated: 绑定更新策略，指定字段如何被更新")
    print("  • operator.add: 追加更新（列表拼接），适合累积增长场景")
    print("  • add_messages: 消息追加更新，LangGraph 内置的消息管理策略")
    print("  • MemorySaver: 内存检查点，实现状态持久化")

    while True:
        print("\n" + "=" * 60)
        print("请选择要运行的示例：")
        print("=" * 60)
        print("  1. 基础状态 - 计数器（用户输入数字，状态自动累加）")
        print("  2. Reducer追加更新 - 聊天记录（消息自动追加到历史）")
        print("  3. Annotated状态 - 任务管理（多字段不同更新策略）")
        print("  4. 状态持久化 - 对话记忆（多轮对话，状态自动保存恢复）")
        print("\n  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_counter()
        elif choice == "2":
            demo_chat_history()
        elif choice == "3":
            demo_task_manager()
        elif choice == "4":
            demo_conversation_memory()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
