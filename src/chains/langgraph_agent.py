"""
LangGraph Agent 高级案例 - 实战交互式示例
==========================================

本示例演示 LangGraph 中四种 Agent 模式的构建方式，
每种模式都使用 LangGraph 的核心原语（StateGraph、ToolNode、路由函数等）

核心概念：
- create_react_agent: LangGraph 提供的快捷函数，一行代码创建 ReAct Agent
- ToolNode: 专门的图节点，自动处理 LLM 的工具调用请求并返回结果
- should_continue: 路由函数（条件边），决定 Agent 是继续调用工具还是结束循环
- 工具调用循环: Agent推理 → 调用工具 → 获取结果 → 继续推理 → 直到完成

应用场景：
- 知识问答：Agent 自动选择知识库工具回答问题
- 生活助手：根据用户需求自动调用天气/计算/搜索等工具
- 代码调试：自动分析代码问题并迭代修复
- 文本优化：多轮自我评估和修正，逐步提升输出质量
"""

import os
import sys
import json
import math

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, create_react_agent
from src.utils.llm_loader import get_default_llm


# ============================================================
# 工具定义
# ============================================================

# --- 知识库模拟数据 ---
_KNOWLEDGE_BASE = {
    "Python": "Python 是一种高级编程语言，以简洁优雅的语法著称。支持面向对象、函数式和过程式编程范式，"
              "广泛应用于 Web 开发、数据科学、人工智能和自动化脚本等领域。Python 由 Guido van Rossum 于 1991 年发布。",
    "LangChain": "LangChain 是一个用于构建大语言模型应用的开源框架。它提供了链式调用、Agent、"
                 "记忆管理和工具集成等核心组件，让开发者可以快速搭建基于 LLM 的智能应用。",
    "LangGraph": "LangGraph 是 LangChain 团队推出的图编排框架，用于构建有状态的、多角色的 AI 应用。"
                 "核心思想是将 Agent 工作流建模为有向图，支持循环、分支和人工介入。",
    "机器学习": "机器学习是人工智能的核心子领域，通过算法让计算机从数据中自动学习规律和模式。"
               "主要分为监督学习、无监督学习和强化学习三大类。",
    "RAG": "RAG（检索增强生成）是一种将外部知识检索与 LLM 生成能力结合的技术。"
           "通过先检索相关文档，再将检索结果作为上下文输入 LLM，显著提升回答的准确性和时效性。",
    "Transformer": "Transformer 是一种基于自注意力机制的深度学习架构，由 Google 在 2017 年提出。"
                   "它是 GPT、BERT 等大语言模型的基础架构，彻底改变了自然语言处理领域。",
}

# --- 天气模拟数据 ---
_WEATHER_DATA = {
    "北京": {"temperature": "28°C", "weather": "晴", "humidity": "45%", "wind": "北风3级"},
    "上海": {"temperature": "32°C", "weather": "多云", "humidity": "72%", "wind": "东南风2级"},
    "广州": {"temperature": "35°C", "weather": "雷阵雨", "humidity": "85%", "wind": "南风4级"},
    "深圳": {"temperature": "33°C", "weather": "阵雨", "humidity": "80%", "wind": "南风3级"},
    "成都": {"temperature": "26°C", "weather": "阴", "humidity": "68%", "wind": "微风"},
    "杭州": {"temperature": "30°C", "weather": "多云转晴", "humidity": "65%", "wind": "东风2级"},
}


@tool
def search_knowledge(query: str) -> str:
    """搜索知识库，查询技术概念的解释和说明

    适用于用户询问技术概念、框架介绍、原理说明等知识类问题

    Args:
        query: 要查询的关键词或概念名称

    Returns:
        知识库中的相关解释
    """
    # 精确匹配
    if query in _KNOWLEDGE_BASE:
        return f"📚 {query}：{_KNOWLEDGE_BASE[query]}"

    # 模糊匹配
    for key in _KNOWLEDGE_BASE:
        if key.lower() in query.lower() or query.lower() in key.lower():
            return f"📚 {key}：{_KNOWLEDGE_BASE[key]}"

    available = "、".join(_KNOWLEDGE_BASE.keys())
    return f"未找到「{query}」的相关知识。当前知识库包含：{available}"


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气信息

    适用于用户想了解某个城市的当前天气状况

    Args:
        city: 城市名称，如"北京"、"上海"

    Returns:
        该城市的天气详情
    """
    if city in _WEATHER_DATA:
        info = _WEATHER_DATA[city]
        return (f"🌤️ {city}天气：{info['weather']}，温度 {info['temperature']}，"
                f"湿度 {info['humidity']}，{info['wind']}")
    available = "、".join(_WEATHER_DATA.keys())
    return f"未找到「{city}」的天气信息。支持的城市：{available}"


@tool
def calculate(expression: str) -> str:
    """执行数学计算，支持基本运算和常用数学函数

    适用于用户需要进行数学计算的场景，如加减乘除、幂运算、对数等

    Args:
        expression: 数学表达式，如 "2+3*4"、"sqrt(16)"、"log(100)"

    Returns:
        计算结果
    """
    try:
        # 安全的数学计算环境
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "pi": math.pi, "e": math.e, "pow": pow,
        }
        # 清理表达式中的中文符号
        expression = expression.replace("×", "*").replace("÷", "/").replace("＋", "+").replace("－", "-")
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"🧮 计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{expression}，原因：{e}"


@tool
def web_search(query: str) -> str:
    """模拟网络搜索，根据关键词返回相关信息

    适用于用户想搜索某个话题或查找特定信息

    Args:
        query: 搜索关键词

    Returns:
        搜索结果摘要
    """
    # 模拟搜索结果
    results = {
        "AI": "人工智能（AI）正在改变各个行业，从医疗到金融...",
        "编程": "2024年最热门的编程语言：Python、JavaScript、TypeScript...",
        "科技": "最新科技趋势：大语言模型、AI Agent、量子计算...",
    }
    for key in results:
        if key in query or query in key:
            return f"🔍 搜索「{query}」结果：{results[key]}"
    return f"🔍 搜索「{query}」：暂未找到相关结果，建议更换关键词"


@tool
def analyze_code(code: str) -> str:
    """分析代码片段，找出潜在的问题和错误

    适用于用户需要检查代码是否有语法错误、逻辑问题或不良实践

    Args:
        code: 要分析的代码片段

    Returns:
        代码分析结果，包括发现的问题和建议
    """
    issues = []

    # 检查常见问题
    if "print " in code and "(" not in code.split("print ")[1][:5]:
        issues.append("⚠️ Python3 中 print 是函数，需要使用 print() 而非 print ")
    if "= =" in code:
        issues.append("⚠️ 比较运算符应为 '==' 而非 '= ='")
    if "except:" in code and "except Exception" not in code:
        issues.append("💡 建议使用 'except Exception as e:' 而非裸 except，避免捕获意外异常")
    if len(code.strip()) > 0 and not code.strip().endswith((":", "\\", ")", "]", "}", ",", ".")):
        if "def " in code or "class " in code or "if " in code or "for " in code:
            pass
        else:
            issues.append("💡 代码看起来不完整，请确认是否遗漏了部分内容")
    if "import *" in code:
        issues.append("⚠️ 不建议使用 'import *'，应明确导入需要的名称")

    if issues:
        return "代码分析结果：\n" + "\n".join(issues)
    else:
        return "代码分析结果：暂未发现明显问题，代码结构看起来合理。"


@tool
def fix_code(code: str, error_info: str) -> str:
    """根据错误信息尝试修复代码

    适用于用户提供了代码和错误描述，需要给出修复建议

    Args:
        code: 原始代码
        error_info: 错误描述信息

    Returns:
        修复后的代码和说明
    """
    # 简单的修复逻辑
    fixed = code
    fixes = []

    if "print " in fixed and "print(" not in fixed:
        import re
        fixed = re.sub(r'print\s+', 'print(', fixed)
        if not fixed.rstrip().endswith(")"):
            fixed += ")"
        fixes.append("将 print 语句改为 print() 函数调用")

    if "= =" in fixed:
        fixed = fixed.replace("= =", "==")
        fixes.append("修正比较运算符 '= =' 为 '=='")

    if "except:" in fixed and "except Exception" not in fixed:
        fixed = fixed.replace("except:", "except Exception as e:")
        fixes.append("将裸 except 改为 except Exception as e:")

    if "import *" in fixed:
        fixed = fixed.replace("import *", "import needed_module")
        fixes.append("避免使用 import *，改为明确导入")

    if fixes:
        return f"修复说明：{'；'.join(fixes)}\n\n修复后的代码：\n{fixed}"
    else:
        return f"无法自动修复该错误（{error_info}），建议手动检查代码逻辑。"


# ============================================================
# 示例1: ReAct Agent - 知识问答
# ============================================================

def demo_react_agent():
    """ReAct Agent - 用户输入问题，Agent 自动推理和调用知识库工具

    核心概念：
    - create_react_agent: LangGraph 内置函数，自动构建 ReAct 循环图
    - ReAct 循环: Reason(推理) → Act(调用工具) → Observe(观察结果) → 继续推理或结束
    - 工具选择由 LLM 自动决定，无需手动编写路由逻辑
    """
    print("\n" + "=" * 60)
    print("示例1：ReAct Agent - 知识问答")
    print("=" * 60)
    print("\n💡 核心概念：")
    print("   - create_react_agent: 一行代码创建 ReAct Agent")
    print("   - ReAct 循环: 推理→调用工具→观察结果→继续推理或结束")
    print("   - 工具选择由 LLM 自动决定，无需手动路由")

    llm = get_default_llm()
    tools = [search_knowledge]

    # 使用 create_react_agent 快速创建 Agent
    # 内部自动构建: Agent节点 → ToolNode → 路由判断 → 循环或结束
    agent = create_react_agent(llm, tools)

    print("\n【交互式知识问答】")
    print("可用工具：search_knowledge（搜索知识库）")
    print("\n试试问：")
    print("  • '什么是 LangGraph？'")
    print("  • '解释一下 RAG 技术'")
    print("  • 'Python 和机器学习有什么关系？'")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("你的问题：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效问题")
            continue

        try:
            print("\n⏳ Agent 正在推理...")
            # 调用 Agent，传入消息列表
            result = agent.invoke({"messages": [HumanMessage(content=user_input)]})

            # 提取最终回答（最后一条 AI 消息）
            messages = result["messages"]
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\n🤖 回答：{msg.content}")
                    break

            # 显示工具调用过程
            tool_calls = [m for m in messages if hasattr(m, 'tool_calls') and m.tool_calls]
            if tool_calls:
                print(f"\n📋 工具调用过程（共 {len(tool_calls)} 次）：")
                for i, tc_msg in enumerate(tool_calls, 1):
                    for tc in tc_msg.tool_calls:
                        print(f"  第{i}次：调用 {tc['name']}({tc['args']})")

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 核心概念总结：")
    print("   1. create_react_agent(llm, tools) 一行创建 ReAct Agent")
    print("   2. Agent 内部自动构建 StateGraph + ToolNode + should_continue 路由")
    print("   3. LLM 根据工具的 docstring 自动决定何时调用哪个工具")
    print("   4. 整个 ReAct 循环（推理→调用→观察→继续）完全自动")


# ============================================================
# 示例2: 工具调用Agent - 生活助手
# ============================================================

def demo_tool_calling_agent():
    """工具调用 Agent - 用户输入需求，Agent 自动调用天气/计算/搜索工具

    核心概念：
    - ToolNode: 专门处理工具调用的图节点，自动执行 LLM 请求的工具
    - should_continue: 路由函数，检查 LLM 输出是否包含工具调用
    - 手动构建 StateGraph，理解 create_react_agent 的内部原理
    """
    print("\n" + "=" * 60)
    print("示例2：工具调用 Agent - 生活助手")
    print("=" * 60)
    print("\n💡 核心概念：")
    print("   - ToolNode: 自动执行 LLM 请求的工具调用")
    print("   - should_continue: 路由函数，决定继续循环还是结束")
    print("   - 手动构建 StateGraph，理解 Agent 内部工作原理")

    llm = get_default_llm()
    tools = [get_weather, calculate, web_search]

    # 绑定工具到 LLM
    llm_with_tools = llm.bind_tools(tools)

    # 创建 ToolNode（自动处理工具调用）
    tool_node = ToolNode(tools)

    # 定义 Agent 节点函数：调用 LLM 推理
    def agent_node(state: MessagesState):
        """Agent 节点：将消息发给 LLM，获取推理结果"""
        system_msg = SystemMessage(content=(
            "你是一个生活助手，可以查询天气、进行数学计算和搜索信息。"
            "根据用户的需求，选择合适的工具来帮助他们。"
            "如果不需要工具就能回答，直接回答即可。"
        ))
        response = llm_with_tools.invoke([system_msg] + state["messages"])
        return {"messages": [response]}

    # 定义路由函数：决定是继续调用工具还是结束
    def should_continue(state: MessagesState) -> str:
        """路由函数：检查最后一条消息是否包含工具调用"""
        last_message = state["messages"][-1]
        # 如果 LLM 请求调用工具，路由到工具节点
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        # 否则结束
        return END

    # 手动构建 StateGraph
    workflow = StateGraph(MessagesState)

    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # 设置入口
    workflow.add_edge(START, "agent")

    # 添加条件边：Agent 输出后，根据路由函数决定下一步
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})

    # 工具执行后回到 Agent 继续推理
    workflow.add_edge("tools", "agent")

    # 编译图
    graph = workflow.compile()

    print("\n【交互式生活助手】")
    print("可用工具：")
    print("  • get_weather - 查询天气")
    print("  • calculate - 数学计算")
    print("  • web_search - 搜索信息")
    print("\n试试问：")
    print("  • '北京今天天气怎么样？'")
    print("  • '帮我算一下 (3.14 * 10) + sqrt(144)'")
    print("  • '搜索一下最近的AI趋势'")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("你的需求：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效内容")
            continue

        try:
            print("\n⏳ Agent 正在处理...")
            result = graph.invoke({"messages": [HumanMessage(content=user_input)]})

            # 提取最终回答
            messages = result["messages"]
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\n🤖 回答：{msg.content}")
                    break

            # 显示工具调用过程
            tool_calls = [m for m in messages if hasattr(m, 'tool_calls') and m.tool_calls]
            if tool_calls:
                print(f"\n📋 工具调用过程（共 {len(tool_calls)} 次）：")
                for i, tc_msg in enumerate(tool_calls, 1):
                    for tc in tc_msg.tool_calls:
                        args_str = json.dumps(tc['args'], ensure_ascii=False)
                        print(f"  第{i}次：调用 {tc['name']}({args_str})")

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 核心概念总结：")
    print("   1. ToolNode(tools) 自动处理 LLM 的工具调用请求")
    print("   2. should_continue 路由函数通过检查 tool_calls 决定流程走向")
    print("   3. StateGraph 手动构建揭示了 create_react_agent 的内部原理")
    print("   4. 工具调用循环：Agent → tools → Agent → ... → END")


# ============================================================
# 示例3: 循环推理Agent - 代码调试
# ============================================================

def demo_loop_reasoning_agent():
    """循环推理 Agent - 用户输入代码，Agent 自动分析问题并迭代修复

    核心概念：
    - 循环推理: Agent 不是一步到位，而是多轮推理逐步逼近目标
    - 最大迭代次数: 防止 Agent 陷入无限循环
    - 状态累积: 每轮推理的结果都累积在 messages 中，Agent 可以回顾之前的步骤
    """
    print("\n" + "=" * 60)
    print("示例3：循环推理 Agent - 代码调试")
    print("=" * 60)
    print("\n💡 核心概念：")
    print("   - 循环推理: 多轮推理逐步逼近目标，而非一步到位")
    print("   - 最大迭代次数: 防止 Agent 陷入无限循环")
    print("   - 状态累积: 每轮结果保存在 messages 中，Agent 可回顾历史")

    llm = get_default_llm()
    tools = [analyze_code, fix_code]

    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    # 最大推理轮数，防止无限循环
    MAX_ITERATIONS = 5

    def agent_node(state: MessagesState):
        """Agent 节点：推理下一步操作"""
        system_msg = SystemMessage(content=(
            "你是一个代码调试助手。用户会给你一段代码，你需要：\n"
            "1. 先用 analyze_code 工具分析代码问题\n"
            "2. 根据分析结果，用 fix_code 工具尝试修复\n"
            "3. 如果修复后仍有问题，继续分析和修复\n"
            "4. 修复完成后，给出最终的正确代码和解释\n"
            "注意：每次只调用一个工具，逐步推进。"
        ))
        response = llm_with_tools.invoke([system_msg] + state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        """路由函数：决定是否继续调用工具"""
        last_message = state["messages"][-1]
        # 检查是否还有工具调用
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # 检查迭代次数，防止无限循环
            tool_messages = [m for m in state["messages"] if m.type == "tool"]
            if len(tool_messages) >= MAX_ITERATIONS:
                return END
            return "tools"
        return END

    # 构建图
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    graph = workflow.compile()

    print("\n【交互式代码调试】")
    print("输入代码，Agent 会自动分析和修复问题")
    print("\n试试输入：")
    print("  • 'for i in range(10) print i'  (Python3 语法错误)")
    print("  • 'x = 10\\nif x = 5:\\n    print(x)'  (赋值与比较混淆)")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("你的代码：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效代码")
            continue

        try:
            print("\n⏳ Agent 正在分析和调试...")
            result = graph.invoke({"messages": [HumanMessage(content=user_input)]})

            # 显示完整推理过程
            messages = result["messages"]
            print(f"\n📋 推理过程（共 {len(messages)} 条消息）：")

            step = 0
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    continue
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        step += 1
                        args_str = json.dumps(tc['args'], ensure_ascii=False)
                        print(f"\n  步骤{step}：调用 {tc['name']}({args_str})")
                elif msg.type == "tool":
                    # 截取工具返回的前200字符，避免输出过长
                    content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    print(f"  工具返回：{content}")
                elif isinstance(msg, AIMessage) and msg.content:
                    print(f"\n🤖 最终结论：{msg.content}")

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 核心概念总结：")
    print("   1. 循环推理让 Agent 可以多轮迭代，逐步逼近目标")
    print("   2. MAX_ITERATIONS 防止无限循环，是生产环境必备的安全阀")
    print("   3. 每轮推理的中间结果都保存在状态中，Agent 可以回顾历史")
    print("   4. 循环推理特别适合调试、修复、优化等需要迭代改进的场景")


# ============================================================
# 示例4: 自我纠错Agent - 文本优化
# ============================================================

def demo_self_correction_agent():
    """自我纠错 Agent - 用户输入文本，Agent 自动多轮优化直到满意

    核心概念：
    - 自我评估: Agent 对自己的输出进行评估，判断是否达标
    - 自我纠错: 根据评估结果自动修正，形成「生成→评估→修正」循环
    - 质量门槛: 设定明确的退出条件，达标后才结束循环
    """
    print("\n" + "=" * 60)
    print("示例4：自我纠错 Agent - 文本优化")
    print("=" * 60)
    print("\n💡 核心概念：")
    print("   - 自我评估: Agent 对自己的输出进行质量评估")
    print("   - 自我纠错: 根据评估结果自动修正，形成优化循环")
    print("   - 质量门槛: 设定退出条件，达标后才结束")

    llm = get_default_llm()

    # 定义状态结构（使用 TypedDict）
    from typing import TypedDict, Annotated
    from langgraph.graph.message import add_messages

    class CorrectionState(TypedDict):
        """自我纠错 Agent 的状态"""
        messages: Annotated[list, add_messages]  # 消息历史
        original_text: str       # 原始文本
        current_text: str        # 当前优化版本
        iteration: int           # 当前迭代次数
        max_iterations: int      # 最大迭代次数
        satisfied: bool          # 是否满意

    MAX_ITERATIONS = 3

    def generate_node(state: CorrectionState) -> dict:
        """生成/优化节点：对文本进行优化"""
        if state["iteration"] == 0:
            # 第一轮：初始优化
            prompt = [
                SystemMessage(content=(
                    "你是一个文本优化专家。请优化用户提供的文本，使其：\n"
                    "1. 表达更加清晰精炼\n"
                    "2. 逻辑更加连贯\n"
                    "3. 用词更加准确\n"
                    "4. 保持原意不变\n"
                    "请直接输出优化后的文本，不要添加额外说明。"
                )),
                HumanMessage(content=f"请优化以下文本：\n\n{state['original_text']}")
            ]
        else:
            # 后续轮次：根据评估意见修正
            prompt = [
                SystemMessage(content=(
                    "你是一个文本优化专家。请根据评估意见修正文本，解决指出的问题。"
                    "请直接输出修正后的文本，不要添加额外说明。"
                )),
                HumanMessage(content=(
                    f"当前文本：\n{state['current_text']}\n\n"
                    f"评估意见：\n{state['messages'][-1].content}\n\n"
                    f"请根据评估意见修正文本。"
                ))
            ]

        response = llm.invoke(prompt)
        return {
            "messages": [response],
            "current_text": response.content,
            "iteration": state["iteration"] + 1,
        }

    def evaluate_node(state: CorrectionState) -> dict:
        """评估节点：评估当前文本质量"""
        prompt = [
            SystemMessage(content=(
                "你是一个文本质量评估专家。请评估以下文本的质量，要求：\n"
                "1. 列出优点（做得好的地方）\n"
                "2. 列出不足（需要改进的地方）\n"
                "3. 给出总体评分（1-10分）\n"
                "4. 如果评分 >= 8，输出「满意」；否则输出「不满意」\n\n"
                "输出格式：\n"
                "优点：...\n"
                "不足：...\n"
                "评分：X/10\n"
                "结论：满意/不满意"
            )),
            HumanMessage(content=f"请评估以下文本：\n\n{state['current_text']}")
        ]

        response = llm.invoke(prompt)
        content = response.content

        # 判断是否满意
        satisfied = "满意" in content and "不满意" not in content
        # 同时检查评分
        if "评分" in content:
            import re
            score_match = re.search(r'评分[：:]\s*(\d+)', content)
            if score_match:
                score = int(score_match.group(1))
                if score >= 8:
                    satisfied = True

        return {
            "messages": [response],
            "satisfied": satisfied,
        }

    def should_continue_correction(state: CorrectionState) -> str:
        """路由函数：判断是否继续纠错"""
        # 满意或达到最大迭代次数，则结束
        if state["satisfied"] or state["iteration"] >= state["max_iterations"]:
            return "final"
        return "generate"

    def final_node(state: CorrectionState) -> dict:
        """最终节点：输出最终结果"""
        if state["satisfied"]:
            summary = f"经过 {state['iteration']} 轮优化，文本质量达标！"
        else:
            summary = f"已达到最大迭代次数 {state['max_iterations']}，当前为最佳优化版本。"
        return {"messages": [AIMessage(content=summary)]}

    # 构建图
    workflow = StateGraph(CorrectionState)
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("final", final_node)

    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        should_continue_correction,
        {"generate": "generate", "final": "final"}
    )
    workflow.add_edge("final", END)

    graph = workflow.compile()

    print("\n【交互式文本优化】")
    print("输入文本，Agent 会自动多轮优化（生成→评估→修正→...→满意）")
    print("\n试试输入：")
    print("  • '这个东西很好用我觉得大家可以试试看'")
    print("  • '今天天气不错所以我想出去玩但是又不想走太远'")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("你的文本：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效文本")
            continue

        try:
            print("\n⏳ Agent 正在优化文本...")

            initial_state = {
                "messages": [],
                "original_text": user_input,
                "current_text": "",
                "iteration": 0,
                "max_iterations": MAX_ITERATIONS,
                "satisfied": False,
            }
            result = graph.invoke(initial_state)

            # 显示优化过程
            messages = result["messages"]
            print(f"\n{'='*60}")
            print(f"📝 原始文本：{user_input}")
            print(f"{'='*60}")

            # 提取各轮优化结果
            generate_count = 0
            for msg in messages:
                if isinstance(msg, AIMessage):
                    content = msg.content
                    if content.startswith("优点") or content.startswith("不足"):
                        # 评估结果
                        print(f"\n📊 评估结果：\n{content[:300]}")
                    elif "轮优化" in content or "最大迭代" in content:
                        # 最终总结
                        print(f"\n🏁 {content}")
                    else:
                        # 优化后的文本
                        generate_count += 1
                        print(f"\n✨ 第{generate_count}轮优化：{content[:300]}")

            print(f"\n{'='*60}")
            print(f"📌 最终优化版本：\n{result['current_text'][:500]}")
            print(f"{'='*60}")

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 核心概念总结：")
    print("   1. 自我纠错 = 生成节点 + 评估节点 + 路由函数")
    print("   2. 评估节点对输出打分，达标则退出，不达标则继续修正")
    print("   3. 质量门槛（评分阈值）和迭代上限是两道安全阀")
    print("   4. 自我纠错适用于文本优化、代码修复、方案完善等需要迭代的场景")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "=" * 60)
    print("  LangGraph Agent 高级案例 - 实战交互式示例")
    print("=" * 60)
    print("\n本示例演示 LangGraph 中四种 Agent 模式的构建方式")
    print("\n核心概念：")
    print("  • create_react_agent: 快速创建 ReAct Agent")
    print("  • ToolNode: 工具节点，自动处理工具调用")
    print("  • should_continue: 路由函数，决定是否继续循环")
    print("  • 工具调用循环: Agent推理→调用工具→获取结果→继续推理")
    print("\n应用场景：")
    print("  • 知识问答、生活助手、代码调试、文本优化")

    while True:
        print("\n" + "=" * 60)
        print("请选择要运行的示例：")
        print("=" * 60)
        print("  1. ReAct Agent - 知识问答（Agent 自动推理和调用工具）")
        print("  2. 工具调用 Agent - 生活助手（天气/计算/搜索工具）")
        print("  3. 循环推理 Agent - 代码调试（自动分析问题并迭代修复）")
        print("  4. 自我纠错 Agent - 文本优化（多轮优化直到满意）")
        print("\n  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_react_agent()
        elif choice == "2":
            demo_tool_calling_agent()
        elif choice == "3":
            demo_loop_reasoning_agent()
        elif choice == "4":
            demo_self_correction_agent()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
