"""
LangChain 学习项目主程序
演示项目的核心功能
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.llm_loader import get_default_llm
from src.chains.basic_chain import (
    demo_simple_chain,
    demo_multi_step_chain,
    demo_parallel_chain,
    demo_streaming_chain
)
from src.chains.agent_basics import (
    demo_basic_agent,
    demo_agent_with_memory,
    demo_agent_with_custom_tools,
    demo_agent_debug
)
from src.chains.tool_basics import (
    demo_single_tool,
    demo_agent_with_tools,
    demo_multi_tool_collaboration
)
from src.chains.tool_advanced import (
    demo_structured_tool,
    demo_input_validation,
    demo_error_handling,
    demo_tool_chain
)
from src.chains.langgraph_agent import (
    demo_react_agent,
    demo_tool_calling_agent,
    demo_loop_reasoning_agent,
    demo_self_correction_agent
)
from src.chains.rag_basics import (
    demo_simple_rag,
    demo_document_qa,
    demo_similarity_search,
    demo_rag_with_source
)
from src.chains.rag_agent import (
    demo_simple_rag_agent,
    demo_multi_source_rag,
    demo_conversational_rag,
    demo_rag_with_tools
)
from src.chains.model_parameters_chain import (
    demo_temperature,
    demo_max_tokens,
    demo_top_p,
    demo_frequency_penalty,
    demo_presence_penalty
)


def show_banner():
    """显示欢迎横幅"""
    print("\n" + "=" * 60)
    print("  LangChain 学习项目 - 功能演示")
    print("=" * 60)


def show_menu():
    """显示主菜单"""
    print("\n请选择要运行的演示:")
    print("\n【基础功能】")
    print("  1. 基础 LLM 调用")
    print("  2. Chain 链式调用")

    print("\n【Agent 智能体】")
    print("  3. Agent 基础")
    print("  4. 工具使用")
    print("  5. 工具进阶")
    print("  6. LangGraph Agent")

    print("\n【RAG 检索增强】")
    print("  7. RAG 基础")
    print("  8. RAG Agent")

    print("\n【其他】")
    print("  9. 模型参数详解")
    print("  10. 交互式聊天")
    print("  0. 退出")


def demo_basic_llm():
    """演示基础 LLM 调用"""
    print("\n" + "=" * 50)
    print("[1] 基础 LLM 调用")
    print("=" * 50)

    llm = get_default_llm()
    response = llm.invoke("你好,请用一句话介绍 LangChain")
    print(f"\n回答: {response.content}")


def demo_all_chains():
    """演示所有链功能"""
    print("\n" + "=" * 50)
    print("[2] Chain 链式调用")
    print("=" * 50)

    demo_simple_chain()
    demo_multi_step_chain()
    demo_parallel_chain()
    demo_streaming_chain()


def demo_agent_basics_menu():
    """Agent 基础子菜单"""
    print("\n" + "=" * 50)
    print("[3] Agent 基础")
    print("=" * 50)

    print("\n选择 Agent 演示:")
    print("  1. 基础 Agent")
    print("  2. Agent + 记忆")
    print("  3. Agent + 自定义工具")
    print("  4. Agent 调试")
    print("  5. 运行所有")

    choice = input("\n请选择 (1-5): ").strip()

    if choice == "1":
        demo_basic_agent()
    elif choice == "2":
        demo_agent_with_memory()
    elif choice == "3":
        demo_agent_with_custom_tools()
    elif choice == "4":
        demo_agent_debug()
    elif choice == "5":
        demo_basic_agent()
        demo_agent_with_memory()
        demo_agent_with_custom_tools()
        demo_agent_debug()
    else:
        print("无效选择")


def demo_tool_basics_menu():
    """工具使用子菜单"""
    print("\n" + "=" * 50)
    print("[4] 工具使用")
    print("=" * 50)

    print("\n选择工具演示:")
    print("  1. 单工具演示")
    print("  2. Agent + 工具")
    print("  3. 多工具协作")
    print("  4. 运行所有")

    choice = input("\n请选择 (1-4): ").strip()

    if choice == "1":
        demo_single_tool()
    elif choice == "2":
        demo_agent_with_tools()
    elif choice == "3":
        demo_multi_tool_collaboration()
    elif choice == "4":
        demo_single_tool()
        demo_agent_with_tools()
        demo_multi_tool_collaboration()
    else:
        print("无效选择")


def demo_tool_advanced_menu():
    """工具进阶子菜单"""
    print("\n" + "=" * 50)
    print("[5] 工具进阶")
    print("=" * 50)

    print("\n选择工具进阶演示:")
    print("  1. 结构化工具")
    print("  2. 输入验证")
    print("  3. 错误处理")
    print("  4. 工具链")
    print("  5. 运行所有")

    choice = input("\n请选择 (1-5): ").strip()

    if choice == "1":
        demo_structured_tool()
    elif choice == "2":
        demo_input_validation()
    elif choice == "3":
        demo_error_handling()
    elif choice == "4":
        demo_tool_chain()
    elif choice == "5":
        demo_structured_tool()
        demo_input_validation()
        demo_error_handling()
        demo_tool_chain()
    else:
        print("无效选择")


def demo_langgraph_agent_menu():
    """LangGraph Agent 子菜单"""
    print("\n" + "=" * 50)
    print("[6] LangGraph Agent")
    print("=" * 50)

    print("\n选择 LangGraph Agent 演示:")
    print("  1. ReAct Agent")
    print("  2. 工具调用 Agent")
    print("  3. 循环推理 Agent")
    print("  4. 自我纠正 Agent")
    print("  5. 运行所有")

    choice = input("\n请选择 (1-5): ").strip()

    if choice == "1":
        demo_react_agent()
    elif choice == "2":
        demo_tool_calling_agent()
    elif choice == "3":
        demo_loop_reasoning_agent()
    elif choice == "4":
        demo_self_correction_agent()
    elif choice == "5":
        demo_react_agent()
        demo_tool_calling_agent()
        demo_loop_reasoning_agent()
        demo_self_correction_agent()
    else:
        print("无效选择")


def demo_rag_basics_menu():
    """RAG 基础子菜单"""
    print("\n" + "=" * 50)
    print("[7] RAG 基础")
    print("=" * 50)

    print("\n选择 RAG 演示:")
    print("  1. 基础 RAG")
    print("  2. 文档问答")
    print("  3. 相似度搜索")
    print("  4. RAG + 来源追踪")
    print("  5. 运行所有")

    choice = input("\n请选择 (1-5): ").strip()

    if choice == "1":
        demo_simple_rag()
    elif choice == "2":
        demo_document_qa()
    elif choice == "3":
        demo_similarity_search()
    elif choice == "4":
        demo_rag_with_source()
    elif choice == "5":
        demo_simple_rag()
        demo_document_qa()
        demo_similarity_search()
        demo_rag_with_source()
    else:
        print("无效选择")


def demo_rag_agent_menu():
    """RAG Agent 子菜单"""
    print("\n" + "=" * 50)
    print("[8] RAG Agent")
    print("=" * 50)

    print("\n选择 RAG Agent 演示:")
    print("  1. 基础 RAG Agent")
    print("  2. 多源 RAG")
    print("  3. 对话式 RAG")
    print("  4. RAG + 工具")
    print("  5. 运行所有")

    choice = input("\n请选择 (1-5): ").strip()

    if choice == "1":
        demo_simple_rag_agent()
    elif choice == "2":
        demo_multi_source_rag()
    elif choice == "3":
        demo_conversational_rag()
    elif choice == "4":
        demo_rag_with_tools()
    elif choice == "5":
        demo_simple_rag_agent()
        demo_multi_source_rag()
        demo_conversational_rag()
        demo_rag_with_tools()
    else:
        print("无效选择")


def demo_model_parameters():
    """演示模型参数"""
    print("\n" + "=" * 50)
    print("[9] 模型参数详解")
    print("=" * 50)

    print("\n选择参数演示:")
    print("  1. Temperature - 创造性与确定性")
    print("  2. Max Tokens - 输出长度控制")
    print("  3. Top P - 核采样控制")
    print("  4. Frequency Penalty - 频率惩罚")
    print("  5. Presence Penalty - 存在惩罚")
    print("  6. 运行所有")

    choice = input("\n请选择 (1-6): ").strip()

    if choice == "1":
        demo_temperature()
    elif choice == "2":
        demo_max_tokens()
    elif choice == "3":
        demo_top_p()
    elif choice == "4":
        demo_frequency_penalty()
    elif choice == "5":
        demo_presence_penalty()
    elif choice == "6":
        demo_temperature()
        demo_max_tokens()
        demo_top_p()
        demo_frequency_penalty()
        demo_presence_penalty()
    else:
        print("无效选择")


def demo_interactive():
    """交互式演示"""
    print("\n" + "=" * 50)
    print("[10] 交互式聊天")
    print("=" * 50)
    print("输入 'quit' 或 'exit' 退出\n")

    llm = get_default_llm()

    while True:
        try:
            user_input = input("你: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break

            if not user_input:
                continue

            response = llm.invoke(user_input)
            print(f"\n助手: {response.content}\n")

        except KeyboardInterrupt:
            print("\n\n已退出")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")


def run_all_demos():
    """运行所有演示"""
    print("\n" + "=" * 60)
    print("  运行所有演示")
    print("=" * 60)

    # 基础功能
    demo_basic_llm()
    demo_all_chains()

    # Agent
    demo_basic_agent()
    demo_agent_with_memory()

    # 工具
    demo_single_tool()
    demo_agent_with_tools()

    # RAG
    demo_simple_rag()

    # 模型参数
    demo_temperature()
    demo_max_tokens()

    print("\n" + "=" * 60)
    print("  所有演示完成!")
    print("=" * 60)


def main():
    """主函数"""
    show_banner()

    while True:
        show_menu()

        try:
            choice = input("\n请输入选项 (0-10): ").strip()

            if choice == "0":
                print("\n退出程序")
                break
            elif choice == "1":
                demo_basic_llm()
            elif choice == "2":
                demo_all_chains()
            elif choice == "3":
                demo_agent_basics_menu()
            elif choice == "4":
                demo_tool_basics_menu()
            elif choice == "5":
                demo_tool_advanced_menu()
            elif choice == "6":
                demo_langgraph_agent_menu()
            elif choice == "7":
                demo_rag_basics_menu()
            elif choice == "8":
                demo_rag_agent_menu()
            elif choice == "9":
                demo_model_parameters()
            elif choice == "10":
                demo_interactive()
            else:
                print("无效选项，请重新输入")

            input("\n按 Enter 继续...")

        except KeyboardInterrupt:
            print("\n\n程序已退出")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")


if __name__ == "__main__":
    main()
