"""
LangChain 工具基础 - 实战交互式案例
====================================

本示例演示如何使用 @tool 装饰器创建自定义工具

核心概念：
- @tool 装饰器：将 Python 函数转换为 LangChain 工具
- 工具定义：定义工具的功能和参数
- 工具调用：让 Agent 能够使用工具

应用场景：
- 搜索工具：搜索网络信息
- 计算工具：执行数学计算
- API工具：调用外部API
- 自定义工具：根据需求定制功能
"""

import os
import sys
import datetime
import random

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_loader import get_default_llm


# ============================================================
# 工具定义
# ============================================================

@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间
    
    Args:
        timezone: 时区（默认：Asia/Shanghai）
    
    Returns:
        当前时间的字符串
    """
    from datetime import datetime
    import pytz
    
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return f"当前时间（{timezone}）：{current_time.strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"获取时间失败：{str(e)}"


@tool
def calculate(expression: str) -> str:
    """执行数学计算
    
    Args:
        expression: 数学表达式（如：'2 + 3 * 4'）
    
    Returns:
        计算结果
    """
    try:
        # 安全计算（仅支持基本运算）
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含非法字符"
        
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算失败：{str(e)}"


@tool
def generate_random_number(min_val: int = 1, max_val: int = 100) -> str:
    """生成随机数
    
    Args:
        min_val: 最小值（默认：1）
        max_val: 最大值（默认：100）
    
    Returns:
        随机数
    """
    try:
        result = random.randint(min_val, max_val)
        return f"随机数（{min_val}-{max_val}）：{result}"
    except Exception as e:
        return f"生成失败：{str(e)}"


@tool
def word_count(text: str) -> str:
    """统计文本字数
    
    Args:
        text: 要统计的文本
    
    Returns:
        字数统计结果
    """
    try:
        # 统计中文字符
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        # 统计英文单词
        english_words = len([w for w in text.split() if w.isalpha()])
        # 总字符数
        total_chars = len(text)
        
        return f"字数统计：\n中文字符：{chinese_chars}\n英文单词：{english_words}\n总字符：{total_chars}"
    except Exception as e:
        return f"统计失败：{str(e)}"


@tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """温度单位转换
    
    Args:
        value: 温度值
        from_unit: 原始单位（C/F/K）
        to_unit: 目标单位（C/F/K）
    
    Returns:
        转换结果
    """
    try:
        # 转换为摄氏度
        if from_unit.upper() == 'C':
            celsius = value
        elif from_unit.upper() == 'F':
            celsius = (value - 32) * 5/9
        elif from_unit.upper() == 'K':
            celsius = value - 273.15
        else:
            return f"不支持的原始单位：{from_unit}"
        
        # 从摄氏度转换为目标单位
        if to_unit.upper() == 'C':
            result = celsius
        elif to_unit.upper() == 'F':
            result = celsius * 9/5 + 32
        elif to_unit.upper() == 'K':
            result = celsius + 273.15
        else:
            return f"不支持的目标单位：{to_unit}"
        
        return f"温度转换：{value}°{from_unit.upper()} = {result:.2f}°{to_unit.upper()}"
    except Exception as e:
        return f"转换失败：{str(e)}"


# ============================================================
# 1. 单个工具演示
# ============================================================

def demo_single_tool():
    """示例1：单个工具使用"""
    print("\n" + "="*60)
    print("示例1：单个工具使用")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - @tool 装饰器将函数转换为工具")
    print("   - Docstring 定义工具的描述和参数")
    print("   - Agent 可以自动选择和调用工具")

    # 测试单个工具
    print("\n【工具测试】")
    print("可用工具：")
    print("  1. get_current_time - 获取当前时间")
    print("  2. calculate - 数学计算")
    print("  3. generate_random_number - 生成随机数")
    print("  4. word_count - 统计字数")
    print("  5. convert_temperature - 温度转换")
    print("\n输入 '退出' 结束")

    tools = {
        "1": ("时间", get_current_time),
        "2": ("计算", calculate),
        "3": ("随机数", generate_random_number),
        "4": ("字数", word_count),
        "5": ("温度", convert_temperature)
    }

    while True:
        print("\n" + "-"*60)
        choice = input("选择工具 (1-5): ").strip()

        if choice.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if choice not in tools:
            print("无效选择")
            continue

        tool_name, tool_func = tools[choice]
        print(f"\n【使用工具：{tool_name}】")

        # 根据工具类型获取输入
        if choice == "1":
            timezone = input("时区（默认 Asia/Shanghai）：").strip() or "Asia/Shanghai"
            result = tool_func.invoke({"timezone": timezone})
        elif choice == "2":
            expression = input("数学表达式：").strip()
            result = tool_func.invoke({"expression": expression})
        elif choice == "3":
            min_val = input("最小值（默认 1）：").strip()
            max_val = input("最大值（默认 100）：").strip()
            args = {}
            if min_val:
                args["min_val"] = int(min_val)
            if max_val:
                args["max_val"] = int(max_val)
            result = tool_func.invoke(args)
        elif choice == "4":
            text = input("输入文本：").strip()
            result = tool_func.invoke({"text": text})
        elif choice == "5":
            value = float(input("温度值：").strip())
            from_unit = input("原始单位（C/F/K）：").strip()
            to_unit = input("目标单位（C/F/K）：").strip()
            result = tool_func.invoke({
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit
            })

        print(f"\n结果：{result}")

    print("\n✅ 实战要点总结：")
    print("   1. @tool 装饰器定义工具")
    print("   2. Docstring 描述工具功能")
    print("   3. 可以直接 invoke() 调用工具")


# ============================================================
# 2. Agent 使用工具
# ============================================================

def demo_agent_with_tools():
    """示例2：Agent 使用工具"""
    print("\n" + "="*60)
    print("示例2：Agent 使用工具")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - Agent 可以自动选择合适的工具")
    print("   - 根据用户问题决定调用哪个工具")
    print("   - 可以组合使用多个工具")

    model = get_default_llm()

    # 定义工具列表
    tools = [
        get_current_time,
        calculate,
        generate_random_number,
        word_count,
        convert_temperature
    ]

    # 创建提示词
    agent = create_react_agent(model, tools, state_modifier="你是一个智能助手，可以使用工具帮助用户。")

    print("\n【交互式 Agent】")
    print("提示：用自然语言提问，Agent 会自动选择工具")
    print("例如：")
    print("  • '现在几点了？'")
    print("  • '计算 123 * 456'")
    print("  • '统计这句话的字数'")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("你的问题：").strip()

        if user_input.lower() in ['退出', 'exit', 'quit']:
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

        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. Agent 自动理解用户意图")
    print("   2. 自动选择合适的工具")
    print("   3. 可以处理自然语言输入")


# ============================================================
# 3. 多工具协作
# ============================================================

def demo_multi_tool_collaboration():
    """示例3：多工具协作"""
    print("\n" + "="*60)
    print("示例3：多工具协作")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - Agent 可以组合使用多个工具")
    print("   - 处理复杂的多步骤任务")
    print("   - 工具之间可以协同工作")

    model = get_default_llm()

    tools = [
        get_current_time,
        calculate,
        generate_random_number,
        word_count,
        convert_temperature
    ]

    agent = create_react_agent(model, tools, state_modifier="你是一个智能助手，可以组合使用多个工具完成复杂任务。")

    print("\n【复杂任务演示】")
    print("提示：输入复杂任务，Agent 会组合使用工具")
    print("例如：")
    print("  • '现在几点了？再帮我算一下 100 * 50'")
    print("  • '生成一个随机数，然后计算它的平方'")
    print("  • '统计这段话的字数：你好世界'")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("你的任务：").strip()

        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效任务")
            continue

        try:
            result = agent.invoke({"messages": [("user", user_input)]})
            final_message = result["messages"][-1]
            print(f"\n完成：{final_message.content}\n")
        except Exception as e:
            print(f"❌ 错误：{e}\n")

        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. Agent 可以组合使用多个工具")
    print("   2. 处理复杂的多步骤任务")
    print("   3. 工具之间可以协同工作")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain 工具基础 - 实战案例")
    print("="*60)
    print("\n本示例演示如何使用 @tool 装饰器创建工具")
    print("\n核心概念：")
    print("  • @tool 装饰器：将函数转换为工具")
    print("  • 工具定义：定义功能和参数")
    print("  • 工具调用：Agent 自动选择使用")
    print("\n应用场景：")
    print("  • 搜索、计算、API调用、自定义功能")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 单个工具使用")
        print("  2. Agent 使用工具")
        print("  3. 多工具协作")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-3): ").strip()

        if choice == "1":
            demo_single_tool()
        elif choice == "2":
            demo_agent_with_tools()
        elif choice == "3":
            demo_multi_tool_collaboration()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()