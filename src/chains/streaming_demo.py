"""
LangChain 流式输出 - 实战交互式案例
====================================

本示例演示 LangChain 的流式输出（Streaming）功能

核心概念：
- stream() 方法：逐块返回生成内容
- 实时显示：减少用户等待时间
- 事件处理：监控生成过程
- Token 统计：实时统计使用量

应用场景：
- 长文本生成：文章、报告、故事
- 实时对话：聊天机器人的实时响应
- 进度反馈：让用户看到生成过程
"""

import os
import sys
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_loader import get_default_llm


# ============================================================
# 1. 基础流式输出
# ============================================================

def demo_basic_streaming():
    """示例1：基础流式输出"""
    print("\n" + "="*60)
    print("示例1：基础流式输出")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 使用 stream() 方法替代 invoke()")
    print("   - 逐块返回内容，实时显示")
    print("   - 减少用户等待时间")

    model = get_default_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个创意写作助手"),
        ("human", "{topic}")
    ])

    chain = prompt | model | StrOutputParser()

    print("\n【交互式创意写作】")
    print("提示：输入主题，AI 实时流式生成内容")
    print("输入 '退出' 结束\n")

    while True:
        topic = input("写作主题：").strip()

        if topic.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not topic:
            print("请输入有效主题")
            continue

        print("\n【开始流式生成】")
        start_time = time.time()

        # 流式输出
        for chunk in chain.stream({"topic": topic}):
            print(chunk, end="", flush=True)

        end_time = time.time()
        print(f"\n\n生成完成！耗时：{end_time - start_time:.2f}秒")
        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. stream() 方法逐块返回内容")
    print("   2. 使用 end='', flush=True 实现实时显示")
    print("   3. 可以统计生成时间")


# ============================================================
# 2. 对比流式 vs 普通输出
# ============================================================

def demo_streaming_vs_invoke():
    """示例2：流式输出 vs 普通输出对比"""
    print("\n" + "="*60)
    print("示例2：流式输出 vs 普通输出对比")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - invoke()：等待完整响应，延迟高")
    print("   - stream()：立即开始输出，延迟低")
    print("   - 流式输出更适合实时交互场景")

    model = get_default_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个写作助手"),
        ("human", "写一个关于{topic}的短故事（200字左右）")
    ])

    chain = prompt | model | StrOutputParser()

    print("\n【交互式对比】")
    print("提示：输入主题，对比两种输出方式")
    print("输入 '退出' 结束\n")

    while True:
        topic = input("故事主题：").strip()

        if topic.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not topic:
            print("请输入有效主题")
            continue

        # 方式1：普通输出
        print("\n【方式1：invoke() - 等待完整响应】")
        start = time.time()
        result = chain.invoke({"topic": topic})
        invoke_time = time.time() - start
        print(result)
        print(f"耗时：{invoke_time:.2f}秒")

        # 方式2：流式输出
        print("\n【方式2：stream() - 实时流式输出】")
        start = time.time()
        for chunk in chain.stream({"topic": topic}):
            print(chunk, end="", flush=True)
        stream_time = time.time() - start
        print(f"\n耗时：{stream_time:.2f}秒")

        print(f"\n时间对比：invoke={invoke_time:.2f}s, stream={stream_time:.2f}s")
        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. invoke()：适合短文本、批处理")
    print("   2. stream()：适合长文本、实时交互")
    print("   3. 流式输出用户体验更好")


# ============================================================
# 3. 带进度显示的流式输出
# ============================================================

def demo_streaming_with_progress():
    """示例3：带进度显示的流式输出"""
    print("\n" + "="*60)
    print("示例3：带进度显示的流式输出")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 可以添加进度指示器")
    print("   - 统计 Token 数量")
    print("   - 显示生成状态")

    model = get_default_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个写作助手"),
        ("human", "{request}")
    ])

    chain = prompt | model | StrOutputParser()

    print("\n【交互式进度显示】")
    print("提示：输入写作需求，AI 实时生成并显示进度")
    print("输入 '退出' 结束\n")

    while True:
        request = input("写作需求：").strip()

        if request.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not request:
            print("请输入有效需求")
            continue

        print("\n【生成中...】")
        print("-"*60)

        # 带进度的流式输出
        chunk_count = 0
        start_time = time.time()

        print("进度：", end="", flush=True)

        for chunk in chain.stream({"request": request}):
            chunk_count += 1

            # 每10个chunk显示一个点
            if chunk_count % 10 == 0:
                print("●", end="", flush=True)

            # 实际内容输出
            print(chunk, end="", flush=True)

        elapsed_time = time.time() - start_time

        print(f"\n\n{'='*60}")
        print(f"✅ 生成完成")
        print(f"   - 耗时：{elapsed_time:.2f}秒")
        print(f"   - 数据块：{chunk_count} 个")
        print(f"{'='*60}\n")

    print("\n✅ 实战要点总结：")
    print("   1. 可以添加自定义进度指示")
    print("   2. 统计生成统计数据（时间、chunk数）")
    print("   3. 提供更好的用户体验")


# ============================================================
# 4. 流式输出的实际应用
# ============================================================

def demo_streaming_applications():
    """示例4：流式输出的实际应用"""
    print("\n" + "="*60)
    print("示例4：流式输出的实际应用")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 实时翻译：边翻译边显示")
    print("   - 文章生成：逐步展示内容")
    print("   - 代码生成：实时显示代码")

    model = get_default_llm()

    # 创建不同的应用场景
    applications = {
        "1": {
            "name": "实时翻译",
            "system": "你是一个专业的翻译助手",
            "template": "将以下文本翻译成英文，并实时输出：\n\n{text}"
        },
        "2": {
            "name": "文章生成",
            "system": "你是一个写作助手",
            "template": "根据以下主题写一篇文章（300字），实时输出：\n\n主题：{text}"
        },
        "3": {
            "name": "代码生成",
            "system": "你是一个编程助手",
            "template": "用 Python 实现以下功能，实时输出代码：\n\n{text}"
        }
    }

    print("\n【交互式应用演示】")
    print("提示：选择应用场景，体验流式输出")
    print("输入 '退出' 结束\n")

    while True:
        print("\n可选应用：")
        for key, app in applications.items():
            print(f"  {key}. {app['name']}")

        app_choice = input("\n选择应用 (1-3): ").strip()

        if app_choice.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if app_choice not in applications:
            print("无效选择")
            continue

        selected_app = applications[app_choice]
        print(f"\n已选择：{selected_app['name']}")

        text = input("输入内容：").strip()
        if not text:
            print("请输入有效内容")
            continue

        # 创建提示词
        prompt = ChatPromptTemplate.from_messages([
            ("system", selected_app['system']),
            ("human", selected_app['template'])
        ])

        chain = prompt | model | StrOutputParser()

        print(f"\n【{selected_app['name']} - 实时输出】")
        print("-"*60)

        # 流式输出
        for chunk in chain.stream({"text": text}):
            print(chunk, end="", flush=True)

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 流式输出适合多种应用场景")
    print("   2. 实时翻译、文章生成、代码生成都受益")
    print("   3. 提供即时反馈，提升用户体验")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain 流式输出 - 实战案例")
    print("="*60)
    print("\n本示例演示 LangChain 的流式输出功能")
    print("\n核心概念：")
    print("  • stream() 方法：逐块返回生成内容")
    print("  • 实时显示：减少用户等待时间")
    print("  • 进度反馈：提供生成过程信息")
    print("\n应用场景：")
    print("  • 长文本生成、实时对话、进度反馈")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 基础流式输出")
        print("  2. 流式 vs 普通输出对比")
        print("  3. 带进度显示的流式输出")
        print("  4. 流式输出的实际应用")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_basic_streaming()
        elif choice == "2":
            demo_streaming_vs_invoke()
        elif choice == "3":
            demo_streaming_with_progress()
        elif choice == "4":
            demo_streaming_applications()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()