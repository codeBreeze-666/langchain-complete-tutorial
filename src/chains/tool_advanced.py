"""
LangChain 工具高级特性 - 实战交互式案例
========================================

本示例演示 LangChain 工具的高级特性

核心概念：
- StructuredTool：结构化工具定义
- 输入验证：使用 Pydantic 模型验证参数
- 错误处理：工具的异常处理机制
- 工具链：多个工具组合使用

应用场景：
- 数据验证：确保输入数据合法
- 复杂工具：需要多参数的工具
- 工具编排：多个工具协同工作
"""

import os
import sys
import re

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.tools import StructuredTool, tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from src.utils.llm_loader import get_default_llm


# ============================================================
# 1. StructuredTool - 结构化工具
# ============================================================

class EmailInput(BaseModel):
    """邮件输入验证模型"""
    recipient: str = Field(description="收件人邮箱")
    subject: str = Field(description="邮件主题")
    content: str = Field(description="邮件内容")

    @field_validator('recipient')
    @classmethod
    def validate_email(cls, v):
        """验证邮箱格式"""
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', v):
            raise ValueError('邮箱格式不正确')
        return v


def send_email(recipient: str, subject: str, content: str) -> str:
    """发送邮件（模拟）

    Args:
        recipient: 收件人邮箱
        subject: 邮件主题
        content: 邮件内容

    Returns:
        发送结果
    """
    # 模拟发送邮件
    return f"✅ 邮件已发送\n收件人：{recipient}\n主题：{subject}\n内容：{content[:50]}..."


# 使用 StructuredTool 创建工具
email_tool = StructuredTool(
    name="send_email",
    description="发送邮件给指定收件人",
    func=send_email,
    args_schema=EmailInput
)


def demo_structured_tool():
    """示例1：StructuredTool 结构化工具"""
    print("\n" + "="*60)
    print("示例1：StructuredTool - 结构化工具")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - StructuredTool 支持复杂的参数验证")
    print("   - 使用 Pydantic 模型定义参数结构")
    print("   - 自动验证输入数据格式")

    print("\n【交互式邮件发送】")
    print("提示：输入邮件信息，工具会自动验证格式")
    print("输入 '退出' 结束\n")

    while True:
        recipient = input("收件人邮箱：").strip()

        if recipient.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not recipient:
            print("请输入邮箱")
            continue

        subject = input("邮件主题：").strip()
        if not subject:
            print("请输入主题")
            continue

        content = input("邮件内容：").strip()
        if not content:
            print("请输入内容")
            continue

        # 调用工具（会自动验证）
        try:
            result = email_tool.invoke({
                "recipient": recipient,
                "subject": subject,
                "content": content
            })
            print(f"\n{result}\n")
        except Exception as e:
            print(f"\n❌ 验证失败：{e}\n")

        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. StructuredTool 支持复杂参数验证")
    print("   2. Pydantic 模型定义参数结构")
    print("   3. field_validator 自定义验证规则")


# ============================================================
# 2. 输入验证 - 多种验证规则
# ============================================================

class UserProfile(BaseModel):
    """用户资料模型"""
    username: str = Field(description="用户名", min_length=3, max_length=20)
    age: int = Field(description="年龄", ge=0, le=150)
    email: str = Field(description="邮箱")
    phone: str = Field(description="手机号", pattern=r'^1[3-9]\d{9}$')

    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        """验证用户名"""
        if not v.isalnum():
            raise ValueError('用户名只能包含字母和数字')
        return v

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """验证邮箱"""
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', v):
            raise ValueError('邮箱格式不正确')
        return v


@tool(args_schema=UserProfile)
def register_user(username: str, age: int, email: str, phone: str) -> str:
    """注册用户

    Args:
        username: 用户名（3-20个字符）
        age: 年龄（0-150）
        email: 邮箱地址
        phone: 手机号

    Returns:
        注册结果
    """
    return f"✅ 用户注册成功\n用户名：{username}\n年龄：{age}\n邮箱：{email}\n手机：{phone}"


def demo_input_validation():
    """示例2：输入验证"""
    print("\n" + "="*60)
    print("示例2：输入验证 - 多种验证规则")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - Pydantic 提供多种内置验证器")
    print("   - 可以自定义 field_validator")
    print("   - 支持正则表达式验证")

    print("\n【交互式用户注册】")
    print("提示：输入用户信息，工具会自动验证")
    print("输入 '退出' 结束\n")

    while True:
        username = input("用户名（3-20个字符，字母数字）：").strip()

        if username.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not username:
            print("请输入用户名")
            continue

        age_str = input("年龄（0-150）：").strip()
        email = input("邮箱：").strip()
        phone = input("手机号（如：13812345678）：").strip()

        # 调用工具（会自动验证）
        try:
            age = int(age_str) if age_str else 0
            result = register_user.invoke({
                "username": username,
                "age": age,
                "email": email,
                "phone": phone
            })
            print(f"\n{result}\n")
        except Exception as e:
            print(f"\n❌ 验证失败：{e}\n")

        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. min_length/max_length 限制字符串长度")
    print("   2. ge/le 限制数值范围")
    print("   3. pattern 使用正则表达式验证")


# ============================================================
# 3. 错误处理 - 友好的错误提示
# ============================================================

@tool
def divide_numbers(a: float, b: float) -> str:
    """除法计算

    Args:
        a: 被除数
        b: 除数

    Returns:
        计算结果
    """
    try:
        if b == 0:
            return "❌ 错误：除数不能为零"
        result = a / b
        return f"计算结果：{a} ÷ {b} = {result}"
    except Exception as e:
        return f"❌ 计算失败：{str(e)}"


@tool
def read_file(file_path: str) -> str:
    """读取文件内容

    Args:
        file_path: 文件路径

    Returns:
        文件内容
    """
    try:
        if not os.path.exists(file_path):
            return f"❌ 错误：文件不存在 - {file_path}"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return f"文件内容（前100字符）：\n{content[:100]}..."

    except PermissionError:
        return f"❌ 错误：没有权限读取文件 - {file_path}"
    except Exception as e:
        return f"❌ 读取失败：{str(e)}"


def demo_error_handling():
    """示例3：错误处理"""
    print("\n" + "="*60)
    print("示例3：错误处理 - 友好的错误提示")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 工具内部应该处理可能的异常")
    print("   - 返回用户友好的错误提示")
    print("   - 避免工具崩溃影响 Agent")

    print("\n【交互式错误处理演示】")
    print("可用工具：")
    print("  1. divide_numbers - 除法计算")
    print("  2. read_file - 读取文件")
    print("输入 '退出' 结束\n")

    tools_map = {
        "1": ("除法", divide_numbers),
        "2": ("读取文件", read_file)
    }

    while True:
        print("\n选择工具 (1-2): ")
        choice = input().strip()

        if choice.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if choice not in tools_map:
            print("无效选择")
            continue

        tool_name, tool_func = tools_map[choice]
        print(f"\n【使用工具：{tool_name}】")

        if choice == "1":
            a = float(input("被除数：").strip() or "0")
            b = float(input("除数：").strip() or "0")
            result = tool_func.invoke({"a": a, "b": b})
        elif choice == "2":
            file_path = input("文件路径：").strip()
            result = tool_func.invoke({"file_path": file_path})

        print(f"\n{result}\n")
        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. try-except 捕获异常")
    print("   2. 返回友好的错误提示")
    print("   3. 检查文件是否存在、权限等")


# ============================================================
# 4. 工具链 - 多个工具组合
# ============================================================

@tool
def get_weather(city: str) -> str:
    """获取天气（模拟）

    Args:
        city: 城市名称

    Returns:
        天气信息
    """
    # 模拟天气数据
    weather_data = {
        "北京": "晴天 18°C",
        "上海": "多云 22°C",
        "广州": "小雨 25°C"
    }
    return weather_data.get(city, f"未找到{city}的天气信息")


@tool
def get_activity_recommendation(weather: str) -> str:
    """根据天气推荐活动

    Args:
        weather: 天气信息

    Returns:
        活动推荐
    """
    if "晴" in weather:
        return "推荐活动：户外跑步、野餐、骑行"
    elif "云" in weather:
        return "推荐活动：散步、逛商场、看电影"
    elif "雨" in weather:
        return "推荐活动：室内运动、看书、烹饪"
    else:
        return "推荐活动：根据实际情况安排"


def demo_tool_chain():
    """示例4：工具链组合"""
    print("\n" + "="*60)
    print("示例4：工具链 - 多个工具组合")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - Agent 可以自动组合使用多个工具")
    print("   - 前一个工具的输出可以作为后一个工具的输入")
    print("   - 适合复杂的多步骤任务")

    model = get_default_llm()

    # 定义工具
    tools = [get_weather, get_activity_recommendation]

    # 创建 Agent (使用新的 create_agent API)
    agent = create_agent(model, tools, system_prompt="你是一个智能助手，可以使用工具帮助用户规划活动。")

    print("\n【交互式活动规划】")
    print("提示：输入城市，Agent 会查询天气并推荐活动")
    print("例如：'北京今天适合做什么？'")
    print("输入 '退出' 结束\n")

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
    print("   1. Agent 自动决定工具调用顺序")
    print("   2. 工具之间可以传递数据")
    print("   3. 适合复杂的多步骤任务")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain 工具高级特性 - 实战案例")
    print("="*60)
    print("\n本示例演示工具的高级特性")
    print("\n核心概念：")
    print("  • StructuredTool：结构化工具定义")
    print("  • 输入验证：Pydantic 模型验证")
    print("  • 错误处理：异常处理机制")
    print("  • 工具链：多个工具组合")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. StructuredTool - 结构化工具")
        print("  2. 输入验证 - 多种验证规则")
        print("  3. 错误处理 - 友好的错误提示")
        print("  4. 工具链 - 多个工具组合")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_structured_tool()
        elif choice == "2":
            demo_input_validation()
        elif choice == "3":
            demo_error_handling()
        elif choice == "4":
            demo_tool_chain()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()