"""
LangChain 四种核心消息类型完整示例
====================================

本示例演示 LangChain 中的四种核心消息类型：
1. SystemMessage - 系统消息（设定 AI 行为）
2. HumanMessage - 人类消息（用户输入）
3. AIMessage - AI 消息（AI 回复）
4. ToolMessage - 工具消息（工具执行结果）

应用场景：
- 多轮对话管理
- Agent 工具调用
- 对话历史记录
- 角色扮演系统
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_loader import get_default_llm


# ============================================================
# 1. SystemMessage - 系统消息
# ============================================================

def demo_system_message():
    """示例 1：SystemMessage - 设定 AI 行为（实战：交互式）"""
    print("\n" + "="*60)
    print("示例 1：SystemMessage（系统消息）- 实战交互")
    print("="*60)
    print("\n💡 实战要点：SystemMessage 必须放在第一条，设定 AI 的行为规则")

    model = get_default_llm()

    # 实战：创建系统消息，定义角色和行为规则
    messages = [
        SystemMessage(content="你是一个专业的 Python 编程助手。你的回答必须：\n"
                             "1. 简洁明了\n"
                             "2. 包含代码示例\n"
                             "3. 遵循 PEP8 规范")
    ]

    print(f"\n系统设定：{messages[0].content}")

    # 实战：用户自己输入问题
    print("\n请输入你的 Python 问题（输入 '退出' 结束）：")
    while True:
        user_input = input("\n你的问题：").strip()

        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not user_input:
            print("请输入有效问题")
            continue

        # 用户发送消息
        messages.append(HumanMessage(content=user_input))

        # 调用模型（会遵循系统设定的行为）
        response = model.invoke(messages)

        print(f"\nAI 回复：\n{response.content}")

        # 实战：保存 AI 回复
        messages.append(AIMessage(content=response.content))

    print("\n✅ 实战要点：")
    print("   1. SystemMessage 放在 messages 列表的第一位")
    print("   2. 定义 AI 的角色、行为规则、输出格式")
    print("   3. 后续所有对话都会遵循这个设定")

    return messages


# ============================================================
# 2. HumanMessage - 人类消息
# ============================================================

def demo_human_message():
    """示例 2：HumanMessage - 用户输入（实战：交互式多轮对话）"""
    print("\n" + "="*60)
    print("示例 2：HumanMessage（人类消息）- 实战交互")
    print("="*60)
    print("\n💡 实战要点：每个 HumanMessage 代表一次用户输入")

    model = get_default_llm()

    # 初始化对话
    messages = [SystemMessage(content="你是一个友好的助手，能记住用户信息")]

    print("\n" + "="*60)
    print("【交互式多轮对话】")
    print("="*60)
    print("提示：你可以多次输入，AI 会记住之前的内容")
    print("输入 '退出' 结束对话\n")

    round_count = 0

    while True:
        round_count += 1
        print(f"\n【第 {round_count} 轮】")
        print("-"*60)

        # 用户输入
        user_input = input("你的输入：").strip()

        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not user_input:
            print("请输入有效内容")
            round_count -= 1
            continue

        # 用户发送消息
        messages.append(HumanMessage(content=user_input))
        print(f"用户：{user_input}")

        # 模型回复
        response = model.invoke(messages)
        print(f"AI：{response.content}")

        # 自动保存 AI 回复
        messages.append(AIMessage(content=response.content))

    print("\n" + "="*60)
    print("【完整对话历史】")
    print("="*60)

    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            print(f"{i}. [系统] {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"{i}. [用户] {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"{i}. [AI] {msg.content[:40]}...")

    print("\n✅ 实战要点：")
    print("   1. 每次用户输入都创建一个 HumanMessage")
    print("   2. messages.append(HumanMessage(content=user_input))")
    print("   3. 模型会记住所有历史，实现上下文对话")

    return messages


# ============================================================
# 3. AIMessage - AI 消息
# ============================================================

def demo_ai_message():
    """示例 3：AIMessage - AI 回复（实战：交互式自动保存）"""
    print("\n" + "="*60)
    print("示例 3：AIMessage（AI 消息）- 实战自动保存")
    print("="*60)
    print("\n💡 核心要点：AIMessage 是模型返回后自动保存的")

    model = get_default_llm()

    # 初始化对话列表（只有系统消息）
    messages = [SystemMessage(content="你是一个数学老师")]

    print("\n" + "="*60)
    print("【交互式对话 - 自动保存 AI 回复】")
    print("="*60)
    print("提示：多轮对话，AI 会记住历史")
    print("输入 '退出' 结束\n")

    round_count = 0

    while True:
        round_count += 1
        print(f"\n【第 {round_count} 轮对话】")
        print("-"*60)

        # 用户提问
        user_input = input("你的问题：").strip()

        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not user_input:
            print("请输入有效问题")
            round_count -= 1
            continue

        # 用户发送消息
        messages.append(HumanMessage(content=user_input))
        print(f"用户：{user_input}")

        # 调用模型
        response = model.invoke(messages)
        print(f"AI：{response.content}")

        # 👇 实战关键：自动保存 AI 回复
        messages.append(AIMessage(content=response.content))

    print("\n" + "="*60)
    print("【完整对话历史】（自动保存的结果）")
    print("="*60)

    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            print(f"{i}. [系统] {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"{i}. [用户] {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"{i}. [AI] {msg.content[:50]}...")

    print("\n✅ 实战要点：")
    print("   1. 初始化 messages = [SystemMessage(...)]")
    print("   2. 用户消息：messages.append(HumanMessage(content=user_input))")
    print("   3. 调用模型：response = model.invoke(messages)")
    print("   4. 保存回复：messages.append(AIMessage(content=response.content))")
    print("   5. 循环往复，自动构建对话历史")

    return messages


# ============================================================
# 4. ToolMessage - 工具消息
# ============================================================

def demo_tool_message():
    """示例 4：ToolMessage - 工具执行结果（实战：手动构建工具调用流程）"""
    print("\n" + "="*60)
    print("示例 4：ToolMessage（工具消息）- 实战手动流程")
    print("="*60)
    print("\n💡 实战要点：ToolMessage 是工具执行后的结果，需要手动构建")

    # 定义工具函数
    @tool
    def get_weather(city: str) -> str:
        """获取天气信息"""
        weather_data = {
            "北京": "晴天，15°C",
            "上海": "多云，18°C",
        }
        return weather_data.get(city, "未找到城市")

    model = get_default_llm()

    # 绑定工具到模型
    model_with_tools = model.bind_tools([get_weather])

    print("\n" + "="*60)
    print("【第一步：用户提问】")
    print("="*60)

    messages = [
        SystemMessage(content="你是一个助手，可以使用 get_weather 工具查询天气"),
        HumanMessage(content="北京今天天气怎么样？")
    ]

    for msg in messages:
        if isinstance(msg, SystemMessage):
            print(f"[系统] {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"[用户] {msg.content}")

    print("\n" + "="*60)
    print("【第二步：模型决定调用工具】")
    print("="*60)

    # 调用模型（模型会返回工具调用请求）
    response = model_with_tools.invoke(messages)
    print(f"\n模型响应：{response}")

    # 检查是否有工具调用
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"\n模型请求调用工具：")
        for tool_call in response.tool_calls:
            print(f"  - 工具名：{tool_call['name']}")
            print(f"  - 参数：{tool_call['args']}")

        # 保存 AI 的工具调用请求
        messages.append(response)

        print("\n" + "="*60)
        print("【第三步：执行工具并创建 ToolMessage】")
        print("="*60)

        # 执行工具
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'get_weather':
                city = tool_call['args']['city']
                tool_result = get_weather.invoke(tool_call['args'])

                print(f"\n工具执行结果：{tool_result}")

                # 👇 实战关键：创建 ToolMessage
                tool_message = ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call['id']  # 必须匹配工具调用 ID
                )

                # 添加到消息历史
                messages.append(tool_message)
                print(f"\n已创建 ToolMessage：{tool_message}")

        print("\n" + "="*60)
        print("【第四步：模型基于工具结果生成最终回复】")
        print("="*60)

        # 模型基于工具结果生成回复
        final_response = model_with_tools.invoke(messages)
        print(f"\n最终回复：{final_response.content}")

        # 保存最终回复
        messages.append(AIMessage(content=final_response.content))

    print("\n" + "="*60)
    print("【完整消息流程】")
    print("="*60)

    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            print(f"{i}. [系统] {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"{i}. [用户] {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"{i}. [AI] {msg.content[:40]}...")
        elif isinstance(msg, ToolMessage):
            print(f"{i}. [工具] {msg.content}")

    print("\n✅ 实战要点：")
    print("   1. 模型返回工具调用请求（tool_calls）")
    print("   2. 执行工具获取结果")
    print("   3. 创建 ToolMessage(content=结果, tool_call_id=ID)")
    print("   4. 添加到 messages 列表")
    print("   5. 再次调用模型生成最终回复")

    return messages


# ============================================================
# 5. 综合示例：完整的消息流程
# ============================================================

def demo_complete_conversation():
    """示例 5：完整实战流程（交互式多轮对话系统）"""
    print("\n" + "="*60)
    print("示例 5：完整实战流程（交互式多轮对话）")
    print("="*60)
    print("\n💡 实战要点：真实的多轮对话系统")

    model = get_default_llm()

    # 初始化：只有系统消息
    messages = [SystemMessage(content="你是一个智能助手，可以帮助用户解决问题。")]

    print("\n" + "="*60)
    print("【交互式多轮对话系统】")
    print("="*60)
    print("提示：自由对话，AI 会记住所有历史")
    print("输入 '退出' 结束，输入 '历史' 查看对话记录\n")

    round_count = 0

    while True:
        round_count += 1
        print(f"\n【第 {round_count} 轮对话】")
        print("-"*60)

        # 用户输入
        user_input = input("你的输入：").strip()

        # 特殊命令
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if user_input.lower() == '历史':
            print("\n【对话历史】")
            for i, msg in enumerate(messages):
                if isinstance(msg, SystemMessage):
                    print(f"{i}. [系统] {msg.content}")
                elif isinstance(msg, HumanMessage):
                    print(f"{i}. [用户] {msg.content}")
                elif isinstance(msg, AIMessage):
                    print(f"{i}. [AI] {msg.content[:40]}...")
            round_count -= 1
            continue

        if not user_input:
            print("请输入有效内容")
            round_count -= 1
            continue

        # 1. 用户发送消息
        messages.append(HumanMessage(content=user_input))
        print(f"用户：{user_input}")

        # 2. 模型回复（基于完整历史）
        response = model.invoke(messages)
        print(f"AI：{response.content}")

        # 3. 自动保存 AI 回复（关键！）
        messages.append(AIMessage(content=response.content))

    print("\n" + "="*60)
    print("【完整对话历史】")
    print("="*60)

    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            print(f"{i}. [系统] {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"{i}. [用户] {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"{i}. [AI] {msg.content[:50]}...")

    print("\n✅ 实战要点：")
    print("   1. 每次对话都基于完整历史（messages 列表）")
    print("   2. 用户消息和 AI 回复都要添加到 messages")
    print("   3. 模型能记住所有历史，实现连续对话")

    return messages


# ============================================================
# 6. 消息类型识别
# ============================================================

def print_message_types():
    """演示如何识别不同消息类型"""
    print("\n" + "="*60)
    print("示例 6：消息类型识别")
    print("="*60)
    
    messages = [
        SystemMessage(content="你是助手"),
        HumanMessage(content="你好"),
        AIMessage(content="你好！有什么可以帮助你？"),
        HumanMessage(content="谢谢"),
    ]
    
    print("\n消息类型识别：")
    for i, msg in enumerate(messages, 1):
        msg_type = type(msg).__name__
        print(f"{i}. 类型：{msg_type:15s} | 内容：{msg.content[:30]}")


# ============================================================
# 7. 消息转换
# ============================================================

def demo_message_conversion():
    """示例 7：消息格式转换"""
    print("\n" + "="*60)
    print("示例 7：消息格式转换")
    print("="*60)
    
    # 字典格式
    dict_messages = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    
    # 转换为 LangChain 消息对象
    from langchain_core.messages import messages_from_dict
    
    lc_messages = messages_from_dict(dict_messages)
    
    print("\n原始格式（字典）：")
    for msg in dict_messages:
        print(f"  {msg}")
    
    print("\n转换后（LangChain 消息对象）：")
    for msg in lc_messages:
        print(f"  {type(msg).__name__}: {msg.content}")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain 四种核心消息类型示例")
    print("="*60)
    print("\n四种核心消息类型：")
    print("  1. SystemMessage  - 系统消息（设定行为）")
    print("  2. HumanMessage   - 人类消息（用户输入）")
    print("  3. AIMessage      - AI 消息（AI 回复）")
    print("  4. ToolMessage    - 工具消息（工具结果）")
    
    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. SystemMessage 示例")
        print("  2. HumanMessage 示例")
        print("  3. AIMessage 示例")
        print("  4. ToolMessage 示例")
        print("  5. 完整消息流程示例")
        print("  6. 消息类型识别")
        print("  7. 消息格式转换")
        print("\n  0. 退出")
        print("="*60)
        
        choice = input("\n请输入选项 (0-7): ").strip()
        
        if choice == "1":
            demo_system_message()
        elif choice == "2":
            demo_human_message()
        elif choice == "3":
            demo_ai_message()
        elif choice == "4":
            demo_tool_message()
        elif choice == "5":
            demo_complete_conversation()
        elif choice == "6":
            print_message_types()
        elif choice == "7":
            demo_message_conversion()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")
        
        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()