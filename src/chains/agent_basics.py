"""
LangChain Agent 基础 - 实战交互式案例
=====================================

本示例演示 Agent 的核心概念和使用方式(使用现代 LangGraph 框架)

核心概念：
- Agent：能够自主选择工具、规划步骤的智能体
- Tool Calling：Agent 通过调用工具完成实际任务
- LangGraph：新一代 Agent 框架(推荐)，使用 create_agent

历史演进：
- 传统方式(已废弃): create_tool_calling_agent + AgentExecutor
- 旧版 LangGraph(已弃用): create_react_agent
- 现代方式(推荐): LangChain v1 的 create_agent

应用场景：
- 智能问答：根据问题自动选择合适的信息源
- 多步推理：拆解复杂任务并逐步执行
- 工具编排：协调多个工具完成综合性任务
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from src.utils.llm_loader import get_default_llm


# ============================================================
# 工具定义（原创语言工具集）
# ============================================================

# 模拟字典数据库
_DICTIONARY = {
    "人工智能": "人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，致力于创建能够模拟人类智能行为的系统，包括学习、推理、感知和决策等能力。",
    "区块链": "区块链（Blockchain）是一种分布式数据库技术，通过密码学方法将数据块按时间顺序链接成链式结构，具有去中心化、不可篡改、可追溯等特性。",
    "量子计算": "量子计算（Quantum Computing）利用量子力学原理（如叠加态和纠缠）进行信息处理，能够在特定问题上实现远超经典计算机的运算速度。",
    "元宇宙": "元宇宙（Metaverse）是一个由虚拟现实、增强现实和互联网融合而成的沉浸式数字世界，用户可以在其中进行社交、工作和娱乐。",
    "深度学习": "深度学习（Deep Learning）是机器学习的一个子领域，使用多层神经网络从数据中自动学习层次化的特征表示，广泛应用于图像识别、自然语言处理等领域。",
    "边缘计算": "边缘计算（Edge Computing）是一种将数据处理从中心节点迁移到网络边缘的分布式计算范式，能够降低延迟、节省带宽并提高响应速度。",
}

# 同义词数据库
_SYNONYMS = {
    "美丽": ["漂亮", "秀丽", "动人", "优雅", "精致"],
    "快乐": ["开心", "愉快", "欢乐", "喜悦", "欣喜"],
    "聪明": ["智慧", "机智", "聪慧", "敏锐", "灵巧"],
    "勇敢": ["英勇", "无畏", "刚毅", "果敢", "坚韧"],
    "困难": ["艰难", "艰巨", "棘手", "复杂", "繁难"],
    "重要": ["关键", "核心", "首要", "根本", "要紧"],
    "迅速": ["快速", "敏捷", "迅捷", "飞快", "疾速"],
    "强大": ["强盛", "雄厚", "有力", "强劲", "彪悍"],
}

# 缩写数据库
_ABBREVIATIONS = {
    "AI": {"full": "Artificial Intelligence", "cn": "人工智能", "desc": "模拟人类智能的计算机系统"},
    "API": {"full": "Application Programming Interface", "cn": "应用程序编程接口", "desc": "软件系统之间交互的约定"},
    "LLM": {"full": "Large Language Model", "cn": "大语言模型", "desc": "基于海量文本训练的大规模语言生成模型"},
    "RAG": {"full": "Retrieval-Augmented Generation", "cn": "检索增强生成", "desc": "结合外部知识检索的文本生成技术"},
    "MCP": {"full": "Model Context Protocol", "cn": "模型上下文协议", "desc": "让 AI 模型与外部工具和数据源交互的开放协议"},
    "SaaS": {"full": "Software as a Service", "cn": "软件即服务", "desc": "通过互联网提供软件应用的云端交付模式"},
    "NLP": {"full": "Natural Language Processing", "cn": "自然语言处理", "desc": "让计算机理解和生成人类语言的技术"},
    "GPU": {"full": "Graphics Processing Unit", "cn": "图形处理器", "desc": "擅长并行计算的处理器，广泛用于 AI 训练"},
}


@tool
def get_word_meaning(word: str) -> str:
    """查询词语的含义，适用于用户想了解某个词语定义的场景

    Args:
        word: 要查询的词语

    Returns:
        词语的含义解释
    """
    # 精确匹配
    if word in _DICTIONARY:
        return f"「{word}」的含义：{_DICTIONARY[word]}"

    # 模糊匹配：检查是否包含关键词
    for key in _DICTIONARY:
        if key in word or word in key:
            return f"「{key}」的含义：{_DICTIONARY[key]}"

    # 无匹配时给出提示
    available = "、".join(_DICTIONARY.keys())
    return f"未找到「{word}」的含义。当前词库包含：{available}"


@tool
def get_synonym(word: str) -> str:
    """获取词语的同义词，适用于用户想寻找替换词或丰富表达的场景

    Args:
        word: 要查找同义词的词语

    Returns:
        同义词列表
    """
    if word in _SYNONYMS:
        synonyms = "、".join(_SYNONYMS[word])
        return f"「{word}」的同义词：{synonyms}"

    # 模糊匹配
    for key in _SYNONYMS:
        if key in word or word in key:
            synonyms = "、".join(_SYNONYMS[key])
            return f"「{key}」的同义词：{synonyms}"

    available = "、".join(_SYNONYMS.keys())
    return f"未找到「{word}」的同义词。当前词库包含：{available}"


@tool
def get_abbreviation(abbr: str) -> str:
    """获取缩写的全称和解释，适用于用户遇到不认识的缩写想了解含义的场景

    Args:
        abbr: 缩写（如 AI、API、LLM）

    Returns:
        缩写的全称和详细解释
    """
    # 统一转大写匹配
    abbr_upper = abbr.upper()

    if abbr_upper in _ABBREVIATIONS:
        info = _ABBREVIATIONS[abbr_upper]
        return (
            f"「{abbr_upper}」\n"
            f"  全称：{info['full']}\n"
            f"  中文：{info['cn']}\n"
            f"  解释：{info['desc']}"
        )

    available = "、".join(_ABBREVIATIONS.keys())
    return f"未找到缩写「{abbr}」。当前词库包含：{available}"


# ============================================================
# 1. 基础 Agent
# ============================================================

def demo_basic_agent():
    """示例1：基础 Agent - 用户输入问题，Agent 自动选择工具回答"""
    print("\n" + "=" * 60)
    print("示例1：基础 Agent - 自动选择工具")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - Agent 根据用户问题自动判断需要哪个工具")
    print("   - 工具的 docstring 是 Agent 决策的关键依据")
    print("   - Agent 会将工具返回的结果整合为自然语言回答")

    model = get_default_llm()
    tools = [get_word_meaning, get_synonym, get_abbreviation]

    agent = create_agent(model, tools, system_prompt="你是一个语言助手，可以帮助用户查询词语含义、同义词和缩写。请根据用户的问题选择合适的工具来回答。")

    print("\n【交互式问答】")
    print("可用工具：")
    print("  • get_word_meaning - 查询词语含义")
    print("  • get_synonym - 获取同义词")
    print("  • get_abbreviation - 获取缩写全称")
    print("\n试试问：")
    print("  • '什么是人工智能？'")
    print("  • '快乐有哪些同义词？'")
    print("  • 'RAG 是什么缩写？'")
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
            result = agent.invoke({"messages": [("user", user_input)]})
            final_message = result["messages"][-1]
            print(f"\n回答：{final_message.content}\n")
        except Exception as e:
            print(f"❌ 错误：{e}\n")

        print("-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. Agent 通过 docstring 理解工具用途")
    print("   2. 根据用户意图自动路由到对应工具")
    print("   3. 将工具结果包装为自然语言输出")


# ============================================================
# 2. 带记忆的 Agent
# ============================================================

def demo_agent_with_memory():
    """示例2：带记忆的 Agent - Agent 可以记住之前的对话"""
    print("\n" + "=" * 60)
    print("示例2：带记忆的 Agent - 对话上下文感知")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - 通过在 messages 中传入历史消息实现对话记忆")
    print("   - 通过 HumanMessage/AIMessage 构建对话记忆")
    print("   - Agent 能理解上下文中的指代和追问")

    model = get_default_llm()
    tools = [get_word_meaning, get_synonym, get_abbreviation]

    agent = create_agent(model, tools, system_prompt="你是一个语言助手，可以帮助用户查询词语含义、同义词和缩写。你可以记住之前的对话内容，理解用户的追问。")

    # 对话记忆列表
    chat_history = []

    print("\n【交互式对话】")
    print("提示：Agent 会记住之前的对话，你可以追问和指代")
    print("试试这样的对话流程：")
    print("  • 第一轮：'什么是深度学习？'")
    print("  • 第二轮：'它有哪些同义词？'（Agent 能理解'它'指深度学习）")
    print("\n输入 '退出' 结束 | 输入 '清空' 重置记忆\n")

    while True:
        user_input = input("你：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if user_input.lower() in ["清空", "clear"]:
            chat_history.clear()
            print("对话记忆已清空\n")
            continue

        if not user_input:
            print("请输入有效内容")
            continue

        try:
            result = agent.invoke({"messages": chat_history + [("user", user_input)]})

            # 将当前对话加入记忆
            chat_history.append(("user", user_input))
            final_message = result["messages"][-1]
            chat_history.append(("assistant", final_message.content))

            print(f"\n助手：{final_message.content}\n")
        except Exception as e:
            print(f"❌ 错误：{e}\n")

        print("-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. 通过 messages 列表传递对话上下文")
    print("   2. 用户消息 + 助手消息 构建记忆链")
    print("   3. Agent 能理解追问中的指代关系（如'它'、'这个'）")


# ============================================================
# 3. 自定义工具 Agent
# ============================================================

def demo_agent_with_custom_tools():
    """示例3：自定义工具 Agent - 体验不同类型的工具"""
    print("\n" + "=" * 60)
    print("示例3：自定义工具 Agent - 体验不同工具组合")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - 同一个 Agent 可以搭配不同的工具集")
    print("   - 工具数量和类型影响 Agent 的决策范围")
    print("   - system prompt 引导 Agent 的行为风格")

    model = get_default_llm()

    # --- 工具集 A：词典模式 ---
    dict_tools = [get_word_meaning]

    dict_agent = create_agent(model, dict_tools, system_prompt="你是一个专业词典助手。你的职责是精确解释词语的含义，回答要严谨、详尽。")

    # --- 工具集 B：全功能模式 ---
    full_tools = [get_word_meaning, get_synonym, get_abbreviation]

    full_agent = create_agent(model, full_tools, system_prompt="你是一个全方位语言助手，可以查询含义、同义词和缩写。尽可能综合利用多种工具，给出丰富的回答。")

    print("\n【工具模式选择】")
    print("  1. 词典模式 - 仅使用「查询含义」工具，适合精确释义")
    print("  2. 全功能模式 - 使用含义、同义词、缩写三个工具，适合综合查询")
    print("\n输入 '退出' 结束\n")

    current_agent = None

    while True:
        if current_agent is None:
            mode = input("选择模式 (1/2): ").strip()
            if mode == "1":
                current_agent = dict_agent
                print("\n已切换到【词典模式】")
            elif mode == "2":
                current_agent = full_agent
                print("\n已切换到【全功能模式】")
            elif mode.lower() in ["退出", "exit", "quit"]:
                print("结束演示")
                return
            else:
                print("无效选择，请输入 1 或 2")
                continue

        user_input = input("你的问题：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if user_input.lower() in ["切换", "switch"]:
            current_agent = None
            print("\n已重置，请重新选择模式\n")
            continue

        if not user_input:
            print("请输入有效问题")
            continue

        try:
            result = current_agent.invoke({"messages": [("user", user_input)]})
            final_message = result["messages"][-1]
            print(f"\n回答：{final_message.content}\n")
        except Exception as e:
            print(f"❌ 错误：{e}\n")

        print("-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. 不同工具集 = 不同能力边界")
    print("   2. system prompt 决定 Agent 的行为风格")
    print("   3. 工具越多 Agent 能力越强，但决策复杂度也越高")


# ============================================================
# 4. Agent 调试
# ============================================================

def demo_agent_debug():
    """示例4：Agent 调试 - 展示 Agent 的思考过程"""
    print("\n" + "=" * 60)
    print("示例4：Agent 调试 - 观察思考过程")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - LangGraph 的 agent.invoke 返回值包含完整的执行轨迹")
    print("   - messages 字段记录了每一步的对话和工具调用")
    print("   - 调试是优化 Agent 提示词和工具设计的关键手段")

    model = get_default_llm()
    tools = [get_word_meaning, get_synonym, get_abbreviation]

    agent = create_agent(model, tools, system_prompt="你是一个语言助手，可以帮助用户查询词语含义、同义词和缩写。")

    print("\n【调试模式】")
    print("提示：每次问答后会展示 Agent 的完整思考过程")
    print("包括：工具选择 → 参数解析 → 工具返回 → 最终回答")
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
            result = agent.invoke({"messages": [("user", user_input)]})

            # 展示最终回答
            final_message = result["messages"][-1]
            print(f"\n📝 最终回答：{final_message.content}")

            # 展示中间步骤（从 messages 中提取工具调用信息）
            tool_calls = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(tc)
                if msg.type == "tool":
                    tool_calls.append(msg)

            if tool_calls:
                print(f"\n🔍 调试信息（共 {len([m for m in result['messages'] if m.type == 'tool'])} 步工具调用）：")
                step = 0
                for msg in result["messages"]:
                    if msg.type == "tool":
                        step += 1
                        print(f"\n  ── 第 {step} 步 ──")
                        print(f"  工具返回：{msg.content[:200]}")
            else:
                print("\n🔍 调试信息：Agent 未调用任何工具，直接回答")

        except Exception as e:
            print(f"❌ 错误：{e}\n")

        print("\n" + "-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. result['messages'] 包含完整的执行轨迹")
    print("   2. 通过消息类型可以提取工具调用和返回值")
    print("   3. 通过调试可以诊断 Agent 选错工具、参数错误等问题")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "=" * 60)
    print("  LangChain Agent 基础 - 实战案例")
    print("=" * 60)
    print("\n本示例演示 Agent 的核心概念和使用方式(使用 LangGraph)")
    print("\n核心概念：")
    print("  • Agent：自主选择工具、规划步骤的智能体")
    print("  • Tool Calling：通过调用工具完成实际任务")
    print("  • LangGraph：新一代 Agent 框架(推荐)")
    print("\n历史演进：")
    print("  • 传统方式(已废弃): create_tool_calling_agent + AgentExecutor")
    print("  • 现代方式(推荐): LangChain v1 的 create_agent")
    print("\n应用场景：")
    print("  • 智能问答、多步推理、工具编排")

    while True:
        print("\n" + "=" * 60)
        print("请选择要运行的示例：")
        print("=" * 60)
        print("  1. 基础 Agent - 自动选择工具回答")
        print("  2. 带记忆的 Agent - 对话上下文感知")
        print("  3. 自定义工具 Agent - 体验不同工具组合")
        print("  4. Agent 调试 - 观察思考过程")
        print("\n  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_basic_agent()
        elif choice == "2":
            demo_agent_with_memory()
        elif choice == "3":
            demo_agent_with_custom_tools()
        elif choice == "4":
            demo_agent_debug()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
