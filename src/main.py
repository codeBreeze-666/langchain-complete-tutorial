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
from src.chains.agent_chain import (
    demo_basic_agent,
    demo_calculator_agent,
    demo_search_agent,
    demo_multi_tool_agent
)
from src.chains.memory_chain import (
    demo_memory
)
from src.chains.rag_chain import (
    demo_basic_rag,
    demo_similarity_search,
    demo_multi_query_rag
)
from src.chains.model_parameters_chain import (
    demo_temperature,
    demo_max_tokens,
    demo_timeout_retries,
    demo_base_url,
    demo_top_p,
    demo_stop,
    demo_seed,
    show_parameter_table,
    demo_comprehensive
)


def show_banner():
    """显示欢迎横幅"""
    print("\n" + "="*60)
    print("  🎓 LangChain 学习项目 - 功能演示")
    print("="*60)


def show_menu():
    """显示主菜单"""
    print("\n📋 请选择要运行的演示:")
    print("\n【基础功能】")
    print("  1. 基础 LLM 调用")
    print("  2. Chain 链式调用")
    
    print("\n【高级功能】")
    print("  3. Agent 智能体")
    print("  4. Memory 记忆功能")
    print("  5. RAG 检索增强生成")
    print("  6. 模型参数详解")
    
    print("\n【其他】")
    print("  7. 交互式聊天")
    print("  8. 运行所有演示")
    print("  0. 退出")


def demo_basic_llm():
    """演示基础 LLM 调用"""
    print("\n" + "="*50)
    print("1️⃣  基础 LLM 调用")
    print("="*50)
    
    llm = get_default_llm()
    
    # 直接调用
    response = llm.invoke("你好,请用一句话介绍 LangChain")
    print(f"\n回答: {response.content}")


def demo_all_chains():
    """演示所有链功能"""
    print("\n" + "="*50)
    print("2️⃣  Chain 链式调用")
    print("="*50)
    
    # 简单链
    demo_simple_chain()
    
    # 多步骤链
    demo_multi_step_chain()
    
    # 并行链
    demo_parallel_chain()
    
    # 流式输出
    demo_streaming_chain()


def demo_agent():
    """演示 Agent"""
    print("\n" + "="*50)
    print("3️⃣  Agent 智能体")
    print("="*50)
    
    print("\n选择 Agent 演示:")
    print("  1. 基础智能体")
    print("  2. 计算器 Agent")
    print("  3. 搜索 Agent")
    print("  4. 多工具 Agent")
    print("  5. 运行所有")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice == "1":
        demo_basic_agent()
    elif choice == "2":
        demo_calculator_agent()
    elif choice == "3":
        demo_search_agent()
    elif choice == "4":
        demo_multi_tool_agent()
    elif choice == "5":
        demo_basic_agent()
        demo_calculator_agent()
        demo_search_agent()
        demo_multi_tool_agent()
    else:
        print("无效选择")


def demo_memory_menu():
    """演示 Memory"""
    print("\n" + "="*50)
    print("4️⃣  Memory 记忆功能")
    print("="*50)
    
    demo_memory()


def demo_rag():
    """演示 RAG"""
    print("\n" + "="*50)
    print("5️⃣  RAG 检索增强生成")
    print("="*50)
    
    print("\n选择 RAG 演示:")
    print("  1. 基础 RAG")
    print("  2. 相似度搜索")
    print("  3. 多查询 RAG")
    print("  4. 运行所有")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    if choice == "1":
        demo_basic_rag()
    elif choice == "2":
        demo_similarity_search()
    elif choice == "3":
        demo_multi_query_rag()
    elif choice == "4":
        demo_basic_rag()
        demo_similarity_search()
        demo_multi_query_rag()
    else:
        print("无效选择")


def demo_model_parameters():
    """演示模型参数"""
    print("\n" + "="*50)
    print("6️⃣  模型参数详解")
    print("="*50)
    
    print("\n选择参数演示:")
    print("  1. Temperature - 创造性与确定性")
    print("  2. Max Tokens - 输出长度控制")
    print("  3. Timeout & Retries - 网络可靠性")
    print("  4. Base URL - 自定义 API 地址")
    print("  5. Top P - 核采样控制")
    print("  6. Stop - 停止序列")
    print("  7. Seed - 可重复性")
    print("  8. 参数速查表")
    print("  9. 综合示例")
    print("  10. 运行所有")
    
    choice = input("\n请选择 (1-10): ").strip()
    
    if choice == "1":
        demo_temperature()
    elif choice == "2":
        demo_max_tokens()
    elif choice == "3":
        demo_timeout_retries()
    elif choice == "4":
        demo_base_url()
    elif choice == "5":
        demo_top_p()
    elif choice == "6":
        demo_stop()
    elif choice == "7":
        demo_seed()
    elif choice == "8":
        show_parameter_table()
    elif choice == "9":
        demo_comprehensive()
    elif choice == "10":
        demo_temperature()
        demo_max_tokens()
        demo_timeout_retries()
        demo_base_url()
        demo_top_p()
        demo_stop()
        demo_seed()
        show_parameter_table()
        demo_comprehensive()
    else:
        print("无效选择")


def demo_interactive():
    """交互式演示"""
    print("\n" + "="*50)
    print("7️⃣  交互式聊天")
    print("="*50)
    print("输入 'quit' 或 'exit' 退出\n")
    
    llm = get_default_llm()
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见! 👋")
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
    print("\n" + "="*60)
    print("  🚀 运行所有演示")
    print("="*60)
    
    # 基础功能
    demo_basic_llm()
    demo_all_chains()
    
    # 高级功能
    demo_basic_agent()
    demo_calculator_agent()
    demo_search_agent()
    demo_multi_tool_agent()
    
    demo_buffer_memory()
    demo_window_memory()
    
    demo_basic_rag()
    
    # 模型参数
    demo_temperature()
    demo_max_tokens()
    
    print("\n" + "="*60)
    print("  ✅ 所有演示完成!")
    print("="*60)


def main():
    """主函数"""
    show_banner()
    
    while True:
        show_menu()
        
        try:
            choice = input("\n请输入选项 (0-8): ").strip()
            
            if choice == "0":
                print("\n👋 退出程序")
                break
            elif choice == "1":
                demo_basic_llm()
            elif choice == "2":
                demo_all_chains()
            elif choice == "3":
                demo_agent()
            elif choice == "4":
                demo_memory_menu()
            elif choice == "5":
                demo_rag()
            elif choice == "6":
                demo_model_parameters()
            elif choice == "7":
                demo_interactive()
            elif choice == "8":
                run_all_demos()
            else:
                print("❌ 无效选项，请重新输入")
                
            input("\n按 Enter 继续...")
            
        except KeyboardInterrupt:
            print("\n\n👋 程序已退出")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main()