"""
LangChain 模型参数详解 - 实战交互式案例
==========================================

本示例演示 LangChain Chat Model 的关键参数及其最佳实践

核心参数：
- temperature：控制创造性与确定性
- max_tokens：控制输出长度与成本
- top_p：控制采样范围
- frequency_penalty：减少重复内容
- presence_penalty：鼓励话题多样性

应用场景：
- 事实查询：低温度（0.0-0.3）
- 日常对话：中温度（0.4-0.7）
- 创意写作：高温度（0.8-1.2）
- 成本控制：限制 max_tokens
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain.chat_models import init_chat_model
from src.utils.llm_loader import ModelConfig


# ============================================================
# 1. Temperature - 控制创造性与确定性
# ============================================================

def demo_temperature():
    """示例1：Temperature 参数（交互式对比）"""
    print("\n" + "="*60)
    print("示例1：Temperature - 控制创造性与确定性")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - temperature 范围：0.0 - 2.0")
    print("   - 低温度（0.0-0.3）：确定性高，适合事实查询")
    print("   - 高温度（0.8-1.2）：创造性高，适合创意写作")

    config = ModelConfig.get_provider_config()

    print("\n【交互式对比】")
    print("提示：输入同一问题，对比不同 temperature 的输出效果")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()

        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束对比")
            break

        if not question:
            print("请输入有效问题")
            continue

        # 低温度：确定性输出
        print("\n【temperature=0.0 - 确定性】")
        model_low = init_chat_model(
            model=f"openai:{config['model']}",
            temperature=0.0,
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )
        response_low = model_low.invoke(question)
        print(f"结果：{response_low.content}")

        # 高温度：创造性输出
        print("\n【temperature=1.0 - 创造性】")
        model_high = init_chat_model(
            model=f"openai:{config['model']}",
            temperature=1.0,
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )
        response_high = model_high.invoke(question)
        print(f"结果：{response_high.content}")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. temperature=0.0：每次结果几乎相同（事实查询、代码生成）")
    print("   2. temperature=0.5：平衡创造与稳定（日常对话）")
    print("   3. temperature=1.0：输出多样化（创意写作、头脑风暴）")


# ============================================================
# 2. Max Tokens - 控制输出长度与成本
# ============================================================

def demo_max_tokens():
    """示例2：Max Tokens 参数（交互式成本控制）"""
    print("\n" + "="*60)
    print("示例2：Max Tokens - 控制输出长度与成本")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - max_tokens 限制输出的最大长度")
    print("   - 值越小，输出越短，成本越低")
    print("   - 适合需要控制成本的场景")

    config = ModelConfig.get_provider_config()

    print("\n【交互式成本控制】")
    print("提示：输入问题，对比不同 max_tokens 的输出")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()

        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束对比")
            break

        if not question:
            print("请输入有效问题")
            continue

        # 不限制长度
        print("\n【无限制 - 完整回答】")
        model_unlimited = init_chat_model(
            model=f"openai:{config['model']}",
            temperature=0.0,
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )
        response_full = model_unlimited.invoke(question)
        print(f"结果：{response_full.content}")

        # 限制长度
        print("\n【max_tokens=50 - 简短回答】")
        model_limited = init_chat_model(
            model=f"openai:{config['model']}",
            max_tokens=50,
            temperature=0.0,
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )
        response_short = model_limited.invoke(question)
        print(f"结果：{response_short.content}")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. max_tokens 不设置：模型自由决定长度")
    print("   2. max_tokens=50：适合简短回答（摘要、标题）")
    print("   3. max_tokens=500：适合中等长度回答（文章段落）")


# ============================================================
# 3. Top P - 核采样
# ============================================================

def demo_top_p():
    """示例3：Top P 参数（交互式采样控制）"""
    print("\n" + "="*60)
    print("示例3：Top P - 核采样控制")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - top_p 控制采样的概率范围")
    print("   - top_p=0.1：只从概率最高的 10% 词汇中采样")
    print("   - top_p=0.9：从概率最高的 90% 词汇中采样")

    config = ModelConfig.get_provider_config()

    print("\n【交互式采样控制】")
    print("提示：输入问题，对比不同 top_p 的输出")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()

        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束对比")
            break

        if not question:
            print("请输入有效问题")
            continue

        # 低 top_p：保守采样
        print("\n【top_p=0.1 - 保守采样】")
        model_conservative = init_chat_model(
            model=f"openai:{config['model']}",
            top_p=0.1,
            temperature=1.0,
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )
        response_conservative = model_conservative.invoke(question)
        print(f"结果：{response_conservative.content}")

        # 高 top_p：自由采样
        print("\n【top_p=0.9 - 自由采样】")
        model_creative = init_chat_model(
            model=f"openai:{config['model']}",
            top_p=0.9,
            temperature=1.0,
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )
        response_creative = model_creative.invoke(question)
        print(f"结果：{response_creative.content}")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. top_p 与 temperature 功能类似，但更精细")
    print("   2. top_p=0.1：输出最安全、最保守")
    print("   3. top_p=0.9：输出更自由、更有创意")


# ============================================================
# 4. Frequency Penalty - 减少重复
# ============================================================

def demo_frequency_penalty():
    """示例4：Frequency Penalty - 减少重复内容"""
    print("\n" + "="*60)
    print("示例4：Frequency Penalty - 减少重复")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - frequency_penalty 范围：-2.0 到 2.0")
    print("   - 正值：减少重复词汇的出现")
    print("   - 负值：增加重复词汇的出现")

    config = ModelConfig.get_provider_config()

    print("\n【交互式重复控制】")
    print("提示：输入问题，对比有无重复惩罚的输出")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()

        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束对比")
            break

        if not question:
            print("请输入有效问题")
            continue

        # 无惩罚：可能重复
        print("\n【frequency_penalty=0.0 - 允许重复】")
        model_no_penalty = init_chat_model(
            model=f"openai:{config['model']}",
            frequency_penalty=0.0,
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )
        response_no_penalty = model_no_penalty.invoke(question)
        print(f"结果：{response_no_penalty.content}")

        # 高惩罚：减少重复
        print("\n【frequency_penalty=1.5 - 减少重复】")
        model_penalty = init_chat_model(
            model=f"openai:{config['model']}",
            frequency_penalty=1.5,
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )
        response_penalty = model_penalty.invoke(question)
        print(f"结果：{response_penalty.content}")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. frequency_penalty=0.0：无惩罚，允许重复")
    print("   2. frequency_penalty=1.0：轻微惩罚，减少重复")
    print("   3. frequency_penalty=2.0：强惩罚，几乎无重复")


# ============================================================
# 5. Presence Penalty - 话题多样性
# ============================================================

def demo_presence_penalty():
    """示例5：Presence Penalty - 鼓励话题多样性"""
    print("\n" + "="*60)
    print("示例5：Presence Penalty - 话题多样性")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - presence_penalty 范围：-2.0 到 2.0")
    print("   - 正值：鼓励讨论新话题")
    print("   - 负值：倾向讨论已提及话题")

    config = ModelConfig.get_provider_config()

    print("\n【交互式话题控制】")
    print("提示：输入问题，对比不同话题惩罚的输出")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()

        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束对比")
            break

        if not question:
            print("请输入有效问题")
            continue

        # 无惩罚：可能重复话题
        print("\n【presence_penalty=0.0 - 允许重复话题】")
        model_no_penalty = init_chat_model(
            model=f"openai:{config['model']}",
            presence_penalty=0.0,
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )
        response_no_penalty = model_no_penalty.invoke(question)
        print(f"结果：{response_no_penalty.content}")

        # 高惩罚：鼓励新话题
        print("\n【presence_penalty=1.5 - 鼓励新话题】")
        model_penalty = init_chat_model(
            model=f"openai:{config['model']}",
            presence_penalty=1.5,
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )
        response_penalty = model_penalty.invoke(question)
        print(f"结果：{response_penalty.content}")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. presence_penalty=0.0：无惩罚，允许重复话题")
    print("   2. presence_penalty=1.0：轻微惩罚，鼓励新话题")
    print("   3. presence_penalty=2.0：强惩罚，强制新话题")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain 模型参数详解 - 实战案例")
    print("="*60)
    print("\n本示例演示 LangChain Chat Model 的关键参数")
    print("\n核心参数：")
    print("  • temperature：控制创造性与确定性")
    print("  • max_tokens：控制输出长度与成本")
    print("  • top_p：控制采样范围")
    print("  • frequency_penalty：减少重复内容")
    print("  • presence_penalty：鼓励话题多样性")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. Temperature - 控制创造性与确定性")
        print("  2. Max Tokens - 控制输出长度与成本")
        print("  3. Top P - 核采样控制")
        print("  4. Frequency Penalty - 减少重复")
        print("  5. Presence Penalty - 话题多样性")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-5): ").strip()

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
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()