"""
LangChain 基础链式调用 - 实战交互式案例
==========================================

本示例演示 LangChain 的核心概念：LCEL（LangChain Expression Language）

核心概念：
- 链式调用（Chain）：将多个组件串联起来
- 管道操作符（|）：连接不同的处理步骤
- Runnable 接口：统一的调用接口

应用场景：
- 简单对话：单次问答
- 多步处理：先生成大纲，再写内容
- 并行处理：同时执行多个任务
- 流式输出：实时显示生成内容
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from src.utils.llm_loader import get_default_llm


# ============================================================
# 1. 简单链式调用 - 基础交互
# ============================================================

def demo_simple_chain():
    """示例1：简单链式调用（交互式对话）"""
    print("\n" + "="*60)
    print("示例1：简单链式调用（基础交互）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 使用 | 连接 Prompt、LLM、OutputParser")
    print("   - 链式调用：prompt | llm | parser")
    print("   - 一行代码完成整个流程")

    model = get_default_llm()

    # 创建提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手"),
        ("human", "{user_input}")
    ])

    # 创建链
    chain = prompt | model | StrOutputParser()

    print("\n【交互式对话】")
    print("提示：输入任何问题，AI 会回答")
    print("输入 '退出' 结束\n")

    while True:
        user_input = input("你的问题：").strip()

        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not user_input:
            print("请输入有效内容")
            continue

        # 调用链
        response = chain.invoke({"user_input": user_input})
        print(f"\nAI：{response}\n")

    print("\n✅ 实战要点总结：")
    print("   1. prompt | model | parser 是最基础的链")
    print("   2. 用户输入通过 {变量} 传递")
    print("   3. invoke() 执行整个链")


# ============================================================
# 2. 多步骤链 - 复杂任务分解
# ============================================================

def demo_multi_step_chain():
    """示例2：多步骤链（先生成大纲，再写内容）"""
    print("\n" + "="*60)
    print("示例2：多步骤链（复杂任务分解）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 将复杂任务拆解为多个步骤")
    print("   - 前一步骤的输出作为后一步骤的输入")
    print("   - 实现流水线式的处理流程")

    model = get_default_llm()

    print("\n【应用场景：文章创作助手】")
    print("提示：输入主题，AI 先生成大纲，再根据大纲写文章")
    print("输入 '退出' 结束\n")

    while True:
        topic = input("文章主题：").strip()

        if topic.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not topic:
            print("请输入有效主题")
            continue

        # 第一步：生成大纲
        outline_prompt = ChatPromptTemplate.from_template(
            "请为以下主题生成一个详细的文章大纲：\n\n主题：{topic}"
        )
        outline_chain = outline_prompt | model | StrOutputParser()

        print("\n【第一步：生成大纲】")
        outline = outline_chain.invoke({"topic": topic})
        print(outline)

        # 第二步：根据大纲写文章
        article_prompt = ChatPromptTemplate.from_template(
            "根据以下大纲，写一篇完整的文章：\n\n大纲：\n{outline}"
        )
        article_chain = article_prompt | model | StrOutputParser()

        print("\n【第二步：根据大纲写文章】")
        article = article_chain.invoke({"outline": outline})
        print(article)

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 多步骤链适合复杂任务（如写作、分析）")
    print("   2. 每个步骤可以独立调用和调试")
    print("   3. 前一步骤的输出传递给后一步骤")


# ============================================================
# 3. 并行链 - 同时执行多个任务
# ============================================================

def demo_parallel_chain():
    """示例3：并行链（同时执行多个任务）"""
    print("\n" + "="*60)
    print("示例3：并行链（同时执行多个任务）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - RunnableParallel 同时执行多个链")
    print("   - 提高效率，减少等待时间")
    print("   - 适合需要多个结果的任务")

    model = get_default_llm()

    print("\n【应用场景：多维度分析助手】")
    print("提示：输入一段文本，AI 同时进行翻译、总结、情感分析")
    print("输入 '退出' 结束\n")

    while True:
        text = input("输入文本：").strip()

        if text.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not text:
            print("请输入有效内容")
            continue

        # 创建多个并行任务
        parallel_chain = RunnableParallel(
            # 任务1：翻译
            translation=ChatPromptTemplate.from_template(
                "将以下文本翻译成英文：\n{text}"
            ) | model | StrOutputParser(),

            # 任务2：总结
            summary=ChatPromptTemplate.from_template(
                "用一句话总结以下文本：\n{text}"
            ) | model | StrOutputParser(),

            # 任务3：情感分析
            sentiment=ChatPromptTemplate.from_template(
                "分析以下文本的情感（正面/负面/中性）：\n{text}"
            ) | model | StrOutputParser()
        )

        # 并行执行
        print("\n【并行执行结果】")
        results = parallel_chain.invoke({"text": text})

        print("\n【翻译】")
        print(results["translation"])

        print("\n【总结】")
        print(results["summary"])

        print("\n【情感分析】")
        print(results["sentiment"])

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. RunnableParallel 同时执行多个任务")
    print("   2. 每个任务独立定义，结果通过字典返回")
    print("   3. 适合需要多维度分析的场景")


# ============================================================
# 4. 流式输出 - 实时显示
# ============================================================

def demo_streaming_chain():
    """示例4：流式输出（实时显示生成过程）"""
    print("\n" + "="*60)
    print("示例4：流式输出（实时显示）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 使用 stream() 方法而非 invoke()")
    print("   - 实时显示生成内容，提升用户体验")
    print("   - 适合长文本生成场景")

    model = get_default_llm()

    # 创建提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个创意写作助手"),
        ("human", "{user_request}")
    ])

    # 创建链
    chain = prompt | model | StrOutputParser()

    print("\n【应用场景：创意写作助手】")
    print("提示：输入创作需求，AI 实时流式输出内容")
    print("输入 '退出' 结束\n")

    while True:
        user_request = input("你的需求：").strip()

        if user_request.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not user_request:
            print("请输入有效需求")
            continue

        print("\n【AI 实时创作】")
        # 流式输出
        for chunk in chain.stream({"user_request": user_request}):
            print(chunk, end="", flush=True)
        print("\n\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. stream() 方法逐块返回内容")
    print("   2. 实时显示，减少用户等待时间")
    print("   3. 适合长文本生成、聊天对话等场景")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain 基础链式调用 - 实战案例")
    print("="*60)
    print("\n本示例演示 LangChain 的核心概念：LCEL")
    print("\n核心概念：")
    print("  • 链式调用（Chain）：串联多个组件")
    print("  • 管道操作符（|）：连接处理步骤")
    print("  • Runnable 接口：统一调用方式")
    print("\n应用场景：")
    print("  • 简单对话、多步处理、并行任务、流式输出")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 简单链式调用（基础交互）")
        print("  2. 多步骤链（复杂任务分解）")
        print("  3. 并行链（同时执行多个任务）")
        print("  4. 流式输出（实时显示）")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_simple_chain()
        elif choice == "2":
            demo_multi_step_chain()
        elif choice == "3":
            demo_parallel_chain()
        elif choice == "4":
            demo_streaming_chain()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()