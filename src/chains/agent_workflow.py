"""
LangChain Agent 工作流 - 实战交互式案例
=====================================

本示例演示 LangChain 中四种常见的工作流编排模式

核心概念：
- 顺序工作流 (Sequential)：步骤1 → 步骤2 → 步骤3，前一步输出作为后一步输入
- 条件工作流 (Conditional)：根据中间结果选择不同执行路径
- 循环工作流 (Loop)：反复执行直到满足退出条件
- 并行工作流 (Parallel)：多个任务同时执行，最后汇总结果

应用场景：
- 旅行规划：多步骤编排行程、根据偏好选择路线、迭代优化方案
- 投资分析：分步收集数据、条件判断策略、循环验证假设
- 学习计划：按阶段推进、根据水平选择内容、反复练习达标
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch
from src.utils.llm_loader import get_default_llm


# ============================================================
# 1. 顺序工作流
# ============================================================

def demo_sequential_workflow():
    """顺序工作流：步骤1完成后再执行步骤2

    场景：旅行规划助手
    - 步骤1：根据目的地生成景点推荐
    - 步骤2：根据景点推荐生成美食攻略
    - 步骤3：根据美食攻略和景点生成完整行程
    """
    print("\n" + "=" * 60)
    print("示例1：顺序工作流 - 旅行规划助手")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - 使用 LCEL 的 pipe (|) 操作符串联多个步骤")
    print("   - 前一步的输出自动成为后一步的输入")
    print("   - 每个步骤用独立的 prompt 模板聚焦单一职责")

    llm = get_default_llm()
    parser = StrOutputParser()

    # 步骤1：景点推荐
    spot_prompt = ChatPromptTemplate.from_template(
        "你是一个旅行顾问。用户想去{destination}旅行，旅行天数{days}天，"
        "偏好风格是{style}。\n"
        "请推荐5个最值得去的景点，每个景点用一句话说明推荐理由。\n"
        "只输出景点推荐，不要其他内容。"
    )

    # 步骤2：美食攻略
    food_prompt = ChatPromptTemplate.from_template(
        "根据以下景点推荐，为旅行者制定美食攻略：\n{spots}\n\n"
        "请推荐每个景点附近的地道美食，包括菜品名称和简短介绍。\n"
        "只输出美食攻略，不要其他内容。"
    )

    # 步骤3：完整行程
    itinerary_prompt = ChatPromptTemplate.from_template(
        "请根据以下信息，编排一份完整的旅行行程：\n\n"
        "【景点推荐】\n{spots}\n\n"
        "【美食攻略】\n{food}\n\n"
        "要求：\n"
        "1. 按天安排，每天有上午、下午、晚上的行程\n"
        "2. 合理安排景点和美食的时间\n"
        "3. 标注每段行程的预估时长"
    )

    # 构建顺序链：景点 → 美食 → 行程
    # 关键：用字典键名传递中间结果
    spots_chain = spot_prompt | llm | parser

    # 美食链：需要景点结果作为输入
    food_chain = food_prompt | llm | parser

    # 行程链：需要景点 + 美食结果
    itinerary_chain = itinerary_prompt | llm | parser

    print("\n【交互式旅行规划】")
    print("我将按 景点推荐 → 美食攻略 → 完整行程 的顺序为你规划旅行")
    print("\n输入 '退出' 结束\n")

    while True:
        destination = input("你想去哪里旅行？：").strip()
        if destination.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if not destination:
            print("请输入目的地")
            continue

        days = input("旅行几天？：").strip()
        if not days:
            days = "3"

        style = input("偏好什么风格？(如：文化/自然/休闲/冒险)：").strip()
        if not style:
            style = "休闲"

        try:
            # 步骤1：生成景点推荐
            print("\n" + "─" * 40)
            print("📍 步骤1/3：生成景点推荐...")
            spots_result = spots_chain.invoke({
                "destination": destination,
                "days": days,
                "style": style,
            })
            print(spots_result)

            # 步骤2：根据景点生成美食攻略
            print("\n" + "─" * 40)
            print("🍜 步骤2/3：生成美食攻略...")
            food_result = food_chain.invoke({"spots": spots_result})
            print(food_result)

            # 步骤3：编排完整行程
            print("\n" + "─" * 40)
            print("🗓️ 步骤3/3：编排完整行程...")
            itinerary_result = itinerary_chain.invoke({
                "spots": spots_result,
                "food": food_result,
            })
            print(itinerary_result)

            print("\n✅ 旅行规划完成！")

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. 顺序工作流用 | 操作符串联，数据自动流向下一步")
    print("   2. 每个步骤的 prompt 只关注自己的职责，降低复杂度")
    print("   3. 中间结果可以用变量保存，供后续步骤或最终汇总使用")


# ============================================================
# 2. 条件工作流
# ============================================================

def demo_conditional_workflow():
    """条件工作流：根据条件选择不同的执行路径

    场景：投资分析助手
    - 根据用户的风险偏好（保守/稳健/激进）选择不同的分析策略
    - 每种策略关注点不同：保本策略、均衡配置、高收益机会
    """
    print("\n" + "=" * 60)
    print("示例2：条件工作流 - 投资分析助手")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - RunnableBranch 根据条件值路由到不同的处理链")
    print("   - 每个分支是独立的 chain，可以有不同的 prompt 和逻辑")
    print("   - 最后一个参数是默认分支，处理未匹配的情况")

    llm = get_default_llm()
    parser = StrOutputParser()

    # 保守型策略链
    conservative_prompt = ChatPromptTemplate.from_template(
        "你是一位保守型投资顾问。用户有{budget}元资金，投资期限{horizon}，"
        "关注点是{concern}。\n\n"
        "请制定保守型投资方案：\n"
        "1. 70%以上配置低风险产品（国债、货币基金、银行理财）\n"
        "2. 严格设定止损线\n"
        "3. 预期年化收益率不超过5%\n"
        "4. 强调本金安全，列出具体产品建议和配置比例"
    )

    # 稳健型策略链
    balanced_prompt = ChatPromptTemplate.from_template(
        "你是一位稳健型投资顾问。用户有{budget}元资金，投资期限{horizon}，"
        "关注点是{concern}。\n\n"
        "请制定稳健型投资方案：\n"
        "1. 股债均衡配置，债券类占50-60%\n"
        "2. 适度参与指数基金定投\n"
        "3. 预期年化收益率5-10%\n"
        "4. 兼顾收益与安全，列出具体产品建议和配置比例"
    )

    # 激进型策略链
    aggressive_prompt = ChatPromptTemplate.from_template(
        "你是一位激进型投资顾问。用户有{budget}元资金，投资期限{horizon}，"
        "关注点是{concern}。\n\n"
        "请制定激进型投资方案：\n"
        "1. 高比例配置股票型基金和行业主题基金\n"
        "2. 可考虑新兴赛道（AI、新能源、生物医药）\n"
        "3. 预期年化收益率15%以上\n"
        "4. 说明高风险产品的潜在收益和最大回撤风险，列出具体产品建议和配置比例"
    )

    # 默认策略链
    default_prompt = ChatPromptTemplate.from_template(
        "你是一位投资顾问。用户有{budget}元资金，投资期限{horizon}，"
        "关注点是{concern}。\n\n"
        "请给出通用投资建议，包括资产配置的基本原则和常见误区。"
    )

    # 构建各分支链
    conservative_chain = conservative_prompt | llm | parser
    balanced_chain = balanced_prompt | llm | parser
    aggressive_chain = aggressive_prompt | llm | parser
    default_chain = default_prompt | llm | parser

    # 使用 RunnableBranch 构建条件路由
    # 签名：RunnableBranch(条件1, 链1, 条件2, 链2, ..., 默认链)
    # 条件函数接收输入字典，返回布尔值
    def is_conservative(x):
        return x.get("risk_type") == "conservative"

    def is_balanced(x):
        return x.get("risk_type") == "balanced"

    def is_aggressive(x):
        return x.get("risk_type") == "aggressive"

    branch_chain = RunnableBranch(
        (is_conservative, conservative_chain),
        (is_balanced, balanced_chain),
        (is_aggressive, aggressive_chain),
        default_chain,
    )

    print("\n【交互式投资分析】")
    print("根据你的风险偏好，选择不同的分析策略")
    print("\n风险类型：")
    print("  • 保守型 - 追求本金安全，低风险产品为主")
    print("  • 稳健型 - 均衡配置，兼顾收益与安全")
    print("  • 激进型 - 追求高收益，承受较大波动")
    print("\n输入 '退出' 结束\n")

    risk_map = {
        "1": ("conservative", "保守型"),
        "2": ("balanced", "稳健型"),
        "3": ("aggressive", "激进型"),
    }

    while True:
        print("─" * 40)
        print("选择你的风险偏好：")
        print("  1. 保守型")
        print("  2. 稳健型")
        print("  3. 激进型")

        choice = input("请选择 (1/2/3)：").strip()
        if choice.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if choice not in risk_map:
            print("无效选择，请输入 1、2 或 3")
            continue

        risk_type, risk_label = risk_map[choice]

        budget = input("投资金额（元）：").strip()
        if not budget:
            budget = "10万"

        horizon = input("投资期限（如：1年/3年/5年）：").strip()
        if not horizon:
            horizon = "3年"

        concern = input("最关注什么？（如：保本/收益/流动性/通胀）：").strip()
        if not concern:
            concern = "综合平衡"

        try:
            print(f"\n📊 正在生成【{risk_label}】投资方案...\n")
            result = branch_chain.invoke({
                "risk_type": risk_type,
                "budget": budget,
                "horizon": horizon,
                "concern": concern,
            })
            print(result)

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. RunnableBranch 实现条件路由，替代传统的 if-else 硬编码")
    print("   2. 每个分支是独立 chain，可以自由组合不同的 prompt/model/parser")
    print("   3. 最后一个参数是无条件默认分支，确保所有输入都有处理")


# ============================================================
# 3. 循环工作流
# ============================================================

def demo_loop_workflow():
    """循环工作流：反复执行直到满足条件

    场景：学习计划优化
    - 初始生成学习计划
    - 用户评价计划质量（1-5分）
    - 低于4分则根据反馈修改，直到满意或达到最大迭代次数
    """
    print("\n" + "=" * 60)
    print("示例3：循环工作流 - 学习计划优化")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - LangChain 没有内置循环原语，需要用 Python 循环实现")
    print("   - 每次迭代将上一次的结果和反馈作为新输入")
    print("   - 设置最大迭代次数防止无限循环")

    llm = get_default_llm()
    parser = StrOutputParser()

    # 生成学习计划的 prompt
    plan_prompt = ChatPromptTemplate.from_template(
        "你是一位学习规划师。用户要学习「{subject}」，"
        "当前水平：{level}，可用时间：每天{hours}小时，目标：{goal}。\n\n"
        "请制定一份详细的学习计划，包括：\n"
        "1. 分阶段目标（每周目标）\n"
        "2. 推荐学习资源（书籍/课程/项目）\n"
        "3. 每日时间分配建议\n"
        "4. 检验学习效果的里程碑\n\n"
        "{feedback_section}"
    )

    plan_chain = plan_prompt | llm | parser

    print("\n【交互式学习计划优化】")
    print("我会生成学习计划，你可以打分并提出修改意见")
    print("低于4分我会根据你的反馈优化，直到满意为止")
    print("\n输入 '退出' 结束\n")

    while True:
        subject = input("你想学什么？：").strip()
        if subject.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if not subject:
            print("请输入学习内容")
            continue

        level = input("你当前的水平？(零基础/入门/中级/高级)：").strip()
        if not level:
            level = "入门"

        hours = input("每天能学几个小时？：").strip()
        if not hours:
            hours = "2"

        goal = input("学习目标是什么？(如：能做项目/通过考试/转行就业)：").strip()
        if not goal:
            goal = "掌握基本技能"

        max_iterations = 5  # 最大迭代次数
        current_plan = ""
        feedback_section = ""

        for iteration in range(1, max_iterations + 1):
            try:
                print(f"\n{'─' * 40}")
                print(f"🔄 第 {iteration} 轮生成学习计划...")

                result = plan_chain.invoke({
                    "subject": subject,
                    "level": level,
                    "hours": hours,
                    "goal": goal,
                    "feedback_section": feedback_section,
                })

                current_plan = result
                print(f"\n📋 学习计划（第 {iteration} 版）：\n")
                print(result)

            except Exception as e:
                print(f"❌ 生成错误：{e}")
                break

            # 评分环节
            print(f"\n{'─' * 40}")
            score = input("请给这个计划打分（1-5分，5分最满意）：").strip()

            try:
                score_num = int(score)
            except ValueError:
                print("已退出评分，采用当前计划")
                break

            if score_num >= 4:
                print(f"\n🎉 计划满意！经过 {iteration} 轮优化完成")
                break
            elif iteration >= max_iterations:
                print(f"\n⚠️ 已达到最大迭代次数 {max_iterations}，采用当前计划")
                break
            else:
                user_feedback = input("请说说哪里需要改进：").strip()
                if not user_feedback:
                    user_feedback = "希望更具体、更可执行"

                # 将反馈注入下一轮的 prompt
                feedback_section = (
                    f"【之前的学习计划】\n{current_plan}\n\n"
                    f"【用户反馈】\n{user_feedback}\n\n"
                    f"请根据以上反馈修改学习计划，重点解决用户提出的问题。\n\n"
                )
                print(f"\n📝 收到反馈，正在优化...")

        print("\n" + "-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. 循环工作流用 Python while/for 循环 + chain.invoke 实现")
    print("   2. 每次迭代把历史结果和反馈拼接到 prompt 中，让 LLM 优化")
    print("   3. 必须设置最大迭代次数，防止无限循环消耗 token")


# ============================================================
# 4. 并行工作流
# ============================================================

def demo_parallel_workflow():
    """并行工作流：多个任务同时执行

    场景：面试准备助手
    - 同时生成：技术题、项目经验梳理、行为面试准备
    - 最后汇总成一份完整的面试备战手册
    """
    print("\n" + "=" * 60)
    print("示例4：并行工作流 - 面试准备助手")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - RunnableParallel 让多个 chain 同时执行，互不阻塞")
    print("   - 所有并行任务的结果以字典形式返回")
    print("   - 适合多维度分析、多视角评估等可并行的场景")

    llm = get_default_llm()
    parser = StrOutputParser()

    # 并行任务1：技术面试题
    tech_prompt = ChatPromptTemplate.from_template(
        "你是一位技术面试官。候选人应聘「{position}」岗位，"
        "技术栈：{tech_stack}，工作年限：{experience}年。\n\n"
        "请准备10道技术面试题，覆盖：\n"
        "1. 基础知识（3道）\n"
        "2. 场景设计（3道）\n"
        "3. 深度追问（2道）\n"
        "4. 代码手写（2道）\n\n"
        "每道题附带参考答案要点。"
    )

    # 并行任务2：项目经验梳理
    project_prompt = ChatPromptTemplate.from_template(
        "你是一位职业顾问。候选人应聘「{position}」岗位，"
        "技术栈：{tech_stack}，工作年限：{experience}年。\n\n"
        "请帮助梳理项目经验的表达框架：\n"
        "1. 如何用 STAR 法则描述项目经历\n"
        "2. 该岗位最看重哪些项目经验\n"
        "3. 如何突出技术深度和业务价值\n"
        "4. 给出2个 STAR 法则的示例回答模板"
    )

    # 并行任务3：行为面试准备
    behavior_prompt = ChatPromptTemplate.from_template(
        "你是一位 HR 顾问。候选人应聘「{position}」岗位，"
        "技术栈：{tech_stack}，工作年限：{experience}年。\n\n"
        "请准备行为面试指南：\n"
        "1. 5个常见行为面试问题及参考回答\n"
        "2. 如何展示团队协作能力\n"
        "3. 如何回答'最大的困难是什么'\n"
        "4. 如何应对薪资谈判"
    )

    # 汇总 prompt
    summary_prompt = ChatPromptTemplate.from_template(
        "请将以下三部分内容整合为一份结构清晰的面试备战手册：\n\n"
        "=== 技术面试题 ===\n{tech}\n\n"
        "=== 项目经验梳理 ===\n{project}\n\n"
        "=== 行为面试准备 ===\n{behavior}\n\n"
        "要求：\n"
        "1. 添加目录和页码标记\n"
        "2. 每部分添加核心要点摘要\n"
        "3. 最后添加'面试前一天检查清单'"
    )

    # 构建并行链
    tech_chain = tech_prompt | llm | parser
    project_chain = project_prompt | llm | parser
    behavior_chain = behavior_prompt | llm | parser

    # 使用 RunnableParallel 并行执行
    parallel_chain = RunnableParallel(
        tech=tech_chain,
        project=project_chain,
        behavior=behavior_chain,
    )

    # 汇总链
    summary_chain = summary_prompt | llm | parser

    print("\n【交互式面试准备】")
    print("我将同时从技术、项目、行为三个维度为你准备面试")
    print("\n输入 '退出' 结束\n")

    while True:
        position = input("你应聘什么岗位？：").strip()
        if position.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if not position:
            print("请输入岗位名称")
            continue

        tech_stack = input("你的技术栈？（如：Python/React/Java）：").strip()
        if not tech_stack:
            tech_stack = "通用技术栈"

        experience = input("工作年限？：").strip()
        if not experience:
            experience = "3"

        common_input = {
            "position": position,
            "tech_stack": tech_stack,
            "experience": experience,
        }

        try:
            # 并行执行三个维度
            print(f"\n{'─' * 40}")
            print("⚡ 正在并行生成三个维度的面试准备...")

            parallel_results = parallel_chain.invoke(common_input)

            # 展示各维度结果
            print("\n" + "═" * 40)
            print("📐 维度一：技术面试题")
            print("═" * 40)
            print(parallel_results["tech"])

            print("\n" + "═" * 40)
            print("📁 维度二：项目经验梳理")
            print("═" * 40)
            print(parallel_results["project"])

            print("\n" + "═" * 40)
            print("🤝 维度三：行为面试准备")
            print("═" * 40)
            print(parallel_results["behavior"])

            # 汇总
            print(f"\n{'─' * 40}")
            print("📋 正在汇总为面试备战手册...")
            summary_result = summary_chain.invoke(parallel_results)
            print(summary_result)

            print("\n✅ 面试准备完成！")

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. RunnableParallel 让多个 chain 同时执行，大幅缩短总耗时")
    print("   2. 结果以字典形式返回，键名对应 RunnableParallel 中的键名")
    print("   3. 并行结果可以直接传递给后续 chain 做汇总处理")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "=" * 60)
    print("  LangChain Agent 工作流 - 实战案例")
    print("=" * 60)
    print("\n本示例演示 LangChain 中四种常见的工作流编排模式")
    print("\n核心概念：")
    print("  • 顺序工作流：步骤1 → 步骤2 → 步骤3")
    print("  • 条件工作流：根据条件选择不同执行路径")
    print("  • 循环工作流：反复执行直到满足退出条件")
    print("  • 并行工作流：多个任务同时执行并汇总")
    print("\n应用场景：")
    print("  • 旅行规划、投资分析、学习计划、面试准备")

    while True:
        print("\n" + "=" * 60)
        print("请选择要运行的示例：")
        print("=" * 60)
        print("  1. 顺序工作流 - 旅行规划助手（景点→美食→行程）")
        print("  2. 条件工作流 - 投资分析助手（保守/稳健/激进）")
        print("  3. 循环工作流 - 学习计划优化（迭代优化直到满意）")
        print("  4. 并行工作流 - 面试准备助手（技术/项目/行为并行）")
        print("\n  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_sequential_workflow()
        elif choice == "2":
            demo_conditional_workflow()
        elif choice == "3":
            demo_loop_workflow()
        elif choice == "4":
            demo_parallel_workflow()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
