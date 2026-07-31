"""
LangChain 提示词模板 - 实战交互式案例
=======================================

本示例演示 LangChain 的提示词管理核心概念

核心概念：
- System Prompt：设定 AI 的角色和行为规则
- Dynamic Prompt：动态生成提示词内容
- Prompt Template：可复用的提示词模板

应用场景：
- 角色扮演：设定特定的助手角色
- 任务定制：为不同任务定制提示词
- 上下文管理：动态注入信息到提示词
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from src.utils.llm_loader import get_default_llm


# ============================================================
# 1. System Prompt - 设定角色
# ============================================================

def demo_system_prompt():
    """示例1：System Prompt（设定角色）"""
    print("\n" + "="*60)
    print("示例1：System Prompt - 设定 AI 角色")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - System Prompt 定义 AI 的行为规则")
    print("   - 放在对话的第一条，设定全局行为")
    print("   - 可以定义角色、风格、输出格式等")

    model = get_default_llm()

    print("\n【交互式角色定制】")
    print("提示：选择不同的角色，看 AI 如何改变行为")
    print("输入 '退出' 结束\n")

    # 预定义的角色模板
    roles = {
        "1": {
            "name": "Python 编程专家",
            "system": "你是一位资深的 Python 编程专家。你的回答必须：\n"
                     "1. 包含可运行的代码示例\n"
                     "2. 遵循 PEP8 规范\n"
                     "3. 解释核心概念"
        },
        "2": {
            "name": "数学老师",
            "system": "你是一位耐心的数学老师。你的回答必须：\n"
                     "1. 循序渐进地讲解\n"
                     "2. 提供具体例子\n"
                     "3. 鼓励学生思考"
        },
        "3": {
            "name": "产品经理",
            "system": "你是一位经验丰富的产品经理。你的回答必须：\n"
                     "1. 从用户需求出发\n"
                     "2. 考虑商业价值\n"
                     "3. 提供可落地的建议"
        }
    }

    # 选择角色
    print("可选角色：")
    for key, role in roles.items():
        print(f"  {key}. {role['name']}")

    while True:
        role_choice = input("\n选择角色 (1-3): ").strip()

        if role_choice.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if role_choice not in roles:
            print("无效选择，请输入 1-3")
            continue

        selected_role = roles[role_choice]
        print(f"\n已选择：{selected_role['name']}")

        # 创建带 System Prompt 的提示词
        prompt = ChatPromptTemplate.from_messages([
            ("system", selected_role['system']),
            ("human", "{question}")
        ])

        # 创建链
        chain = prompt | model | StrOutputParser()

        # 交互式问答
        print(f"\n现在你是{selected_role['name']}，请提问（输入 '返回' 切换角色）:\n")

        while True:
            question = input("你的问题：").strip()

            if question.lower() in ['返回', 'back']:
                break

            if question.lower() in ['退出', 'exit', 'quit']:
                print("结束演示")
                return

            if not question:
                print("请输入有效问题")
                continue

            # 调用链
            response = chain.invoke({"question": question})
            print(f"\n回答：{response}\n")

    print("\n✅ 实战要点总结：")
    print("   1. System Prompt 设定全局行为")
    print("   2. 不同角色有不同的回答风格")
    print("   3. 可以动态切换角色")


# ============================================================
# 2. Dynamic Prompt - 动态生成
# ============================================================

def demo_dynamic_prompt():
    """示例2：Dynamic Prompt（动态内容）"""
    print("\n" + "="*60)
    print("示例2：Dynamic Prompt - 动态生成内容")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - Dynamic Prompt 根据上下文动态生成")
    print("   - 可以注入用户信息、时间、环境等")
    print("   - 让提示词更加个性化和上下文相关")

    model = get_default_llm()

    print("\n【交互式个性化问候】")
    print("提示：输入你的信息，AI 会生成个性化问候")
    print("输入 '退出' 结束\n")

    while True:
        name = input("你的名字：").strip()
        if name.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not name:
            print("请输入有效名字")
            continue

        mood = input("今天的心情（如：开心、疲惫、兴奋）：").strip()
        if not mood:
            mood = "平静"

        # 动态生成 System Prompt
        current_hour = __import__('datetime').datetime.now().hour
        time_greeting = "早上好" if 5 <= current_hour < 12 else "下午好" if 12 <= current_hour < 18 else "晚上好"

        dynamic_system = f"""
        你是一个贴心的助手。
        
        当前信息：
        - 时间：{time_greeting}
        - 用户名字：{name}
        - 用户心情：{mood}
        
        请根据以上信息，生成一个温暖、个性化的问候语。
        """

        # 创建动态提示词
        prompt = ChatPromptTemplate.from_messages([
            ("system", dynamic_system),
            ("human", "请和我打个招呼")
        ])

        # 调用
        chain = prompt | model | StrOutputParser()
        response = chain.invoke({})

        print(f"\n个性化问候：{response}\n")
        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. Dynamic Prompt 可以注入动态内容")
    print("   2. 可以使用当前时间、用户信息等")
    print("   3. 让 AI 的回答更加个性化")


# ============================================================
# 3. Prompt Template - 模板复用
# ============================================================

def demo_prompt_template():
    """示例3：Prompt Template（模板复用）"""
    print("\n" + "="*60)
    print("示例3：Prompt Template - 可复用模板")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - Prompt Template 提供可复用的提示词模板")
    print("   - 使用 {变量} 定义模板变量")
    print("   - 支持部分填充（partial）和格式化")

    model = get_default_llm()

    # 创建可复用的模板
    templates = {
        "翻译": ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的翻译助手"),
            ("human", "将以下文本从{source_lang}翻译成{target_lang}：\n\n{text}")
        ]),
        "总结": ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的总结助手"),
            ("human", "用{length}句话总结以下文本：\n\n{text}")
        ]),
        "改写": ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的改写助手"),
            ("human", "以{style}的风格改写以下文本：\n\n{text}")
        ])
    }

    print("\n【交互式模板应用】")
    print("提示：选择任务类型，输入文本，应用模板")
    print("输入 '退出' 结束\n")

    while True:
        print("\n可选任务：")
        print("  1. 翻译")
        print("  2. 总结")
        print("  3. 改写")

        task_choice = input("\n选择任务 (1-3): ").strip()

        if task_choice.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if task_choice not in ["1", "2", "3"]:
            print("无效选择")
            continue

        text = input("输入文本：").strip()
        if not text:
            print("请输入有效文本")
            continue

        # 根据任务类型应用模板
        if task_choice == "1":
            source_lang = input("源语言（如：中文）：").strip() or "中文"
            target_lang = input("目标语言（如：英文）：").strip() or "英文"

            chain = templates["翻译"] | model | StrOutputParser()
            result = chain.invoke({
                "source_lang": source_lang,
                "target_lang": target_lang,
                "text": text
            })

        elif task_choice == "2":
            length = input("总结句数（如：3）：").strip() or "3"

            chain = templates["总结"] | model | StrOutputParser()
            result = chain.invoke({
                "length": length,
                "text": text
            })

        elif task_choice == "3":
            style = input("改写风格（如：正式、幽默）：").strip() or "正式"

            chain = templates["改写"] | model | StrOutputParser()
            result = chain.invoke({
                "style": style,
                "text": text
            })

        print(f"\n结果：{result}\n")
        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. Prompt Template 可以复用")
    print("   2. 使用 {变量} 定义模板变量")
    print("   3. 支持多变量和动态填充")


# ============================================================
# 4. Messages Placeholder - 消息占位符
# ============================================================

def demo_messages_placeholder():
    """示例4：Messages Placeholder（消息占位符）"""
    print("\n" + "="*60)
    print("示例4：Messages Placeholder - 消息占位符")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - MessagesPlaceholder 用于插入消息列表")
    print("   - 常用于插入对话历史")
    print("   - 让 Agent 能够访问之前的对话")

    model = get_default_llm()

    print("\n【交互式对话记忆】")
    print("提示：输入问题，AI 会记住历史对话")
    print("输入 '历史' 查看对话记录，'退出' 结束\n")

    # 存储对话历史
    conversation_history = []

    # 创建带消息占位符的模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手，会记住之前的对话"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # 创建链
    chain = prompt | model | StrOutputParser()

    while True:
        user_input = input("你的输入：").strip()

        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if user_input.lower() == '历史':
            print("\n【对话历史】")
            for i, msg in enumerate(conversation_history, 1):
                print(f"{i}. {msg}")
            print()
            continue

        if not user_input:
            print("请输入有效内容")
            continue

        # 调用链（传入历史）
        response = chain.invoke({
            "history": conversation_history,
            "input": user_input
        })

        print(f"\nAI：{response}\n")

        # 更新历史
        conversation_history.append(f"用户：{user_input}")
        conversation_history.append(f"AI：{response}")

    print("\n✅ 实战要点总结：")
    print("   1. MessagesPlaceholder 用于插入消息列表")
    print("   2. 常用于管理对话历史")
    print("   3. 让 Agent 能够记住之前的对话")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain 提示词模板 - 实战案例")
    print("="*60)
    print("\n本示例演示 LangChain 的提示词管理")
    print("\n核心概念：")
    print("  • System Prompt：设定 AI 的角色和行为")
    print("  • Dynamic Prompt：动态生成提示词内容")
    print("  • Prompt Template：可复用的提示词模板")
    print("  • Messages Placeholder：消息占位符")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. System Prompt - 设定角色")
        print("  2. Dynamic Prompt - 动态生成")
        print("  3. Prompt Template - 模板复用")
        print("  4. Messages Placeholder - 消息占位符")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_system_prompt()
        elif choice == "2":
            demo_dynamic_prompt()
        elif choice == "3":
            demo_prompt_template()
        elif choice == "4":
            demo_messages_placeholder()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()