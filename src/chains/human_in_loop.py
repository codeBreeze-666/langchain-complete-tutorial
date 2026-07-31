"""
LangChain 人工介入（Human-in-the-Loop）- 实战交互式案例
=====================================================

本示例演示如何在 AI 工作流中添加人工审核、确认和协作环节

核心概念：
- 审批流程：AI 生成内容后，用户确认是否执行
- 内容审核：AI 生成内容后，用户可以修改再确认
- 决策检查：AI 做出重要决策前，需要用户确认
- 协作编辑：AI 和用户共同编辑内容

应用场景：
- 金融交易审批：AI 生成交易方案，人工确认执行
- 内容发布审核：AI 生成文案，人工修改后发布
- 关键决策确认：AI 给出建议，人工拍板决定
- 文档协作编写：AI 草拟内容，人工精修打磨
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_loader import get_default_llm


# ============================================================
# 1. 审批流程 - AI 生成内容后，用户确认是否执行
# ============================================================

def demo_approval_workflow():
    """示例1：审批流程 - AI 生成内容后，用户确认是否执行"""
    print("\n" + "="*60)
    print("示例1：审批流程（AI 生成内容后，用户确认是否执行）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - AI 生成方案后不能自动执行，需人工审批")
    print("   - 用户可以批准、拒绝或要求重新生成")
    print("   - 适用于邮件发送、数据操作等不可逆场景")

    model = get_default_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的商务助手，根据用户需求生成方案。方案要具体、可执行。"),
        ("human", "{task}")
    ])
    chain = prompt | model | StrOutputParser()

    print("\n【交互式审批流程】")
    print("提示：输入任务描述，AI 生成方案后需要你审批")
    print("输入 '退出' 结束\n")

    while True:
        task = input("任务描述：").strip()

        if task.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not task:
            print("请输入有效内容")
            continue

        # AI 生成方案
        print("\n⏳ AI 正在生成方案...")
        proposal = chain.invoke({"task": task})

        print(f"\n{'='*60}")
        print("📋 AI 生成的方案：")
        print(f"{'='*60}")
        print(proposal)
        print(f"{'='*60}")

        # 审批环节
        while True:
            print("\n请选择操作：")
            print("  1. ✅ 批准执行")
            print("  2. ❌ 拒绝执行")
            print("  3. 🔄 重新生成")
            print("  0. 🚪 取消")

            choice = input("你的选择 (0-3)：").strip()

            if choice == "1":
                print("\n✅ 方案已批准，正在执行...")
                print("📌 执行结果：方案已按审批内容落实")
                break
            elif choice == "2":
                reason = input("请输入拒绝原因：").strip()
                print(f"\n❌ 方案已拒绝，原因：{reason if reason else '未说明'}")
                print("📌 执行结果：操作已取消")
                break
            elif choice == "3":
                print("\n🔄 重新生成方案...")
                proposal = chain.invoke({"task": task})
                print(f"\n{'='*60}")
                print("📋 重新生成的方案：")
                print(f"{'='*60}")
                print(proposal)
                print(f"{'='*60}")
                continue
            elif choice == "0":
                print("\n🚪 已取消审批流程")
                break
            else:
                print("❌ 无效选项，请重新选择")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. AI 生成不可逆操作的方案时，必须加入审批环节")
    print("   2. 提供「重新生成」选项，避免一次不满意就全部推倒")
    print("   3. 拒绝时要求填写原因，便于后续分析优化")
    print("   4. 审批日志应持久化存储，满足合规审计需求")


# ============================================================
# 2. 内容审核 - AI 生成内容后，用户可以修改再确认
# ============================================================

def demo_content_review():
    """示例2：内容审核 - AI 生成内容后，用户可以修改再确认"""
    print("\n" + "="*60)
    print("示例2：内容审核（AI 生成内容后，用户可以修改再确认）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - AI 生成的内容可能不完全符合预期")
    print("   - 允许用户在 AI 输出基础上修改，而非从零开始")
    print("   - 适用于文案撰写、报告生成等需要人工精修的场景")

    model = get_default_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个资深文案策划，擅长撰写吸引人的营销文案。"),
        ("human", "请为以下产品撰写一段营销文案：\n{product_info}")
    ])
    chain = prompt | model | StrOutputParser()

    print("\n【交互式内容审核】")
    print("提示：输入产品信息，AI 生成文案后你可以修改再确认")
    print("输入 '退出' 结束\n")

    while True:
        product_info = input("产品信息：").strip()

        if product_info.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not product_info:
            print("请输入有效内容")
            continue

        # AI 生成初稿
        print("\n⏳ AI 正在生成文案...")
        draft = chain.invoke({"product_info": product_info})

        # 当前内容（可能被修改）
        current_content = draft

        # 审核循环
        while True:
            print(f"\n{'='*60}")
            print("📝 当前文案内容：")
            print(f"{'='*60}")
            print(current_content)
            print(f"{'='*60}")

            print("\n请选择操作：")
            print("  1. ✅ 确认发布")
            print("  2. ✏️  修改内容")
            print("  3. 🔄 重新生成")
            print("  0. 🚫 放弃")

            choice = input("你的选择 (0-3)：").strip()

            if choice == "1":
                print("\n✅ 文案已确认，正式发布！")
                print(f"📌 发布内容：\n{current_content}")
                break
            elif choice == "2":
                print("\n请输入修改后的内容（输入完成后按 Enter）：")
                print("提示：可直接复制上方内容进行修改")
                modified = input("修改内容：").strip()
                if modified:
                    current_content = modified
                    print("✏️ 内容已更新")
                else:
                    print("⚠️ 修改内容为空，保持原内容不变")
                continue
            elif choice == "3":
                print("\n🔄 重新生成文案...")
                current_content = chain.invoke({"product_info": product_info})
                print("✅ 已重新生成")
                continue
            elif choice == "0":
                print("\n🚫 已放弃当前文案")
                break
            else:
                print("❌ 无效选项，请重新选择")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 内容审核的核心是「在 AI 输出基础上修改」，效率远高于从零写")
    print("   2. 修改后应展示完整内容供用户再次确认，避免误操作")
    print("   3. 保留重新生成选项，当修改量过大时不如重写")
    print("   4. 生产中可记录修改轨迹，用于分析 AI 输出的薄弱环节")


# ============================================================
# 3. 决策检查 - AI 做出重要决策前，需要用户确认
# ============================================================

def demo_decision_check():
    """示例3：决策检查 - AI 做出重要决策前，需要用户确认"""
    print("\n" + "="*60)
    print("示例3：决策检查（AI 做出重要决策前，需要用户确认）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - AI 在关键分支上应暂停，请求人类决策")
    print("   - 展示决策依据和影响范围，辅助人类判断")
    print("   - 适用于策略选择、方案取舍等高风险决策场景")

    model = get_default_llm()

    # 分析链：生成决策建议和影响分析
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个决策分析助手。针对用户的问题，请：\n"
         "1. 分析当前情况\n"
         "2. 列出 2-3 个可选方案\n"
         "3. 对每个方案说明利弊和风险\n"
         "4. 给出你的推荐方案及理由"),
        ("human", "{situation}")
    ])
    analysis_chain = analysis_prompt | model | StrOutputParser()

    # 执行链：根据确认的方案生成执行计划
    execution_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个执行规划专家。根据选定的方案，生成详细的执行步骤。"),
        ("human", "情况：{situation}\n\n选定的方案：{chosen_plan}")
    ])
    execution_chain = execution_prompt | model | StrOutputParser()

    print("\n【交互式决策检查】")
    print("提示：描述你面临的决策场景，AI 会分析并请你选择")
    print("输入 '退出' 结束\n")

    while True:
        situation = input("决策场景：").strip()

        if situation.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not situation:
            print("请输入有效内容")
            continue

        # AI 分析决策
        print("\n⏳ AI 正在分析决策方案...")
        analysis = analysis_chain.invoke({"situation": situation})

        print(f"\n{'='*60}")
        print("🔍 决策分析报告：")
        print(f"{'='*60}")
        print(analysis)
        print(f"{'='*60}")

        # 人工决策环节
        print("\n⚠️ 这是一个重要决策，需要你确认！")
        print("请选择：")
        print("  1. ✅ 采纳 AI 推荐方案")
        print("  2. 🔄 选择其他方案（请输入你的方案）")
        print("  3. 🛑 暂不决策，终止流程")

        while True:
            choice = input("你的选择 (1-3)：").strip()

            if choice == "1":
                print("\n✅ 已确认采纳推荐方案")
                print("⏳ 正在生成执行计划...")
                execution = execution_chain.invoke({
                    "situation": situation,
                    "chosen_plan": "采纳 AI 推荐方案"
                })
                print(f"\n📋 执行计划：\n{execution}")
                break
            elif choice == "2":
                custom_plan = input("请输入你选择的方案：").strip()
                if not custom_plan:
                    print("⚠️ 方案不能为空，请重新选择")
                    continue
                print(f"\n✅ 已确认自定义方案：{custom_plan}")
                print("⏳ 正在生成执行计划...")
                execution = execution_chain.invoke({
                    "situation": situation,
                    "chosen_plan": custom_plan
                })
                print(f"\n📋 执行计划：\n{execution}")
                break
            elif choice == "3":
                print("\n🛑 决策已终止，流程未执行")
                break
            else:
                print("❌ 无效选项，请重新选择")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. AI 只做分析建议，最终决策权在人类手中")
    print("   2. 展示多方案对比，避免 AI 只给一个答案就执行")
    print("   3. 决策确认后再生执行计划，避免无效计算")
    print("   4. 保留「暂不决策」选项，不强迫用户在信息不足时做决定")


# ============================================================
# 4. 协作编辑 - AI 和用户共同编辑内容
# ============================================================

def demo_collaborative_editing():
    """示例4：协作编辑 - AI 和用户共同编辑内容"""
    print("\n" + "="*60)
    print("示例4：协作编辑（AI 和用户共同编辑内容）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - AI 和人类交替编辑，逐步完善内容")
    print("   - 每轮编辑后双方都能看到完整内容")
    print("   - 适用于长文写作、方案策划等需要反复打磨的场景")

    model = get_default_llm()

    # 起草链：根据主题生成初稿
    draft_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业写手，根据主题撰写内容草稿。"),
        ("human", "请就以下主题撰写一篇内容：{topic}")
    ])
    draft_chain = draft_prompt | model | StrOutputParser()

    # 修订链：根据用户意见修改内容
    revise_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个内容编辑助手。根据用户的修改意见，对现有内容进行修订。"),
        ("human", "当前内容：\n{content}\n\n修改意见：{feedback}")
    ])
    revise_chain = revise_prompt | model | StrOutputParser()

    print("\n【交互式协作编辑】")
    print("提示：输入主题，AI 起草初稿后你和 AI 交替编辑")
    print("输入 '退出' 结束\n")

    while True:
        topic = input("编辑主题：").strip()

        if topic.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not topic:
            print("请输入有效内容")
            continue

        # AI 起草初稿
        print("\n⏳ AI 正在起草初稿...")
        content = draft_chain.invoke({"topic": topic})

        round_num = 0

        # 协作编辑循环
        while True:
            round_num += 1
            print(f"\n{'='*60}")
            print(f"📝 第 {round_num} 轮 - 当前内容：")
            print(f"{'='*60}")
            print(content)
            print(f"{'='*60}")

            print(f"\n第 {round_num} 轮编辑（轮到你）：")
            print("  1. ✏️  提出修改意见（AI 根据意见修订）")
            print("  2. 📝 直接修改内容（你手动编辑）")
            print("  3. ✅ 满意，结束编辑")
            print("  0. 🗑️ 放弃本次编辑")

            choice = input("你的选择 (0-3)：").strip()

            if choice == "1":
                # 用户提供修改意见，AI 修订
                feedback = input("请输入修改意见：").strip()
                if not feedback:
                    print("⚠️ 修改意见不能为空")
                    round_num -= 1
                    continue
                print("\n⏳ AI 正在根据你的意见修订...")
                content = revise_chain.invoke({
                    "content": content,
                    "feedback": feedback
                })
                print("✅ AI 已完成修订")
                continue

            elif choice == "2":
                # 用户直接修改内容
                print("请输入修改后的完整内容（输入完成后按 Enter）：")
                print("提示：可先复制上方内容，在此基础上修改")
                user_edit = input("修改内容：").strip()
                if user_edit:
                    content = user_edit
                    print("✅ 内容已更新")

                    # 用户编辑后，AI 可以做进一步润色
                    polish_choice = input("\n是否让 AI 对你的修改做润色？(y/n)：").strip().lower()
                    if polish_choice == 'y':
                        print("⏳ AI 正在润色...")
                        content = revise_chain.invoke({
                            "content": content,
                            "feedback": "请对内容进行润色优化，保持核心意思不变，提升表达质量"
                        })
                        print("✅ AI 已完成润色")
                    continue
                else:
                    print("⚠️ 修改内容为空，保持原内容不变")
                    round_num -= 1
                    continue

            elif choice == "3":
                print(f"\n✅ 编辑完成！共经过 {round_num} 轮协作")
                print(f"\n📌 最终内容：\n{'='*60}")
                print(content)
                print(f"{'='*60}")
                break

            elif choice == "0":
                print("\n🗑️ 已放弃本次编辑")
                break

            else:
                print("❌ 无效选项，请重新选择")
                round_num -= 1

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 协作编辑的核心是「AI 起草 + 人类精修」的循环")
    print("   2. 提供两种修改方式：给意见让 AI 改、自己直接改")
    print("   3. 用户手动编辑后可选 AI 润色，兼顾效率和品质")
    print("   4. 记录编辑轮数，方便评估协作效率")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain 人工介入（Human-in-the-Loop）- 实战案例")
    print("="*60)
    print("\n本示例演示如何在 AI 工作流中添加人工审核和决策环节")
    print("\n核心概念：")
    print("  • 审批流程：AI 生成内容后，用户确认是否执行")
    print("  • 内容审核：AI 生成内容后，用户可以修改再确认")
    print("  • 决策检查：AI 做出重要决策前，需要用户确认")
    print("  • 协作编辑：AI 和用户共同编辑内容")
    print("\n应用场景：")
    print("  • 金融交易审批")
    print("  • 内容发布审核")
    print("  • 关键决策确认")
    print("  • 文档协作编写")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 审批流程（AI 生成内容后，用户确认是否执行）")
        print("  2. 内容审核（AI 生成内容后，用户可以修改再确认）")
        print("  3. 决策检查（AI 做出重要决策前，需要用户确认）")
        print("  4. 协作编辑（AI 和用户共同编辑内容）")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_approval_workflow()
        elif choice == "2":
            demo_content_review()
        elif choice == "3":
            demo_decision_check()
        elif choice == "4":
            demo_collaborative_editing()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
