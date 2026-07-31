"""
LangChain 多 Agent 协作 - 实战交互式案例
=========================================

本示例演示四种多 Agent 协作模式，每种都是可交互的实战案例

核心概念：
- 角色委派：不同角色的 Agent 各司其职，协同完成复杂任务
- 流水线 Agent：多个 Agent 按顺序处理，前一步输出是后一步输入
- 辩论 Agent：两个 Agent 对同一问题从不同立场出发，碰撞出更深见解
- 主管 Agent：一个主管负责分析任务并智能分配给专业 Agent

应用场景：
- 项目团队协作（角色委派）
- 内容生产流水线（流水线 Agent）
- 方案评审与决策（辩论 Agent）
- 智能任务调度（主管 Agent）
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_loader import get_default_llm


# ============================================================
# 1. 角色委派：不同角色的 Agent 负责不同任务
# ============================================================

def demo_role_delegation():
    """角色委派：产品经理、技术负责人、测试负责人各司其职

    场景：需求评审会
    - 产品经理：分析需求价值和用户场景
    - 技术负责人：评估技术可行性和实现方案
    - 测试负责人：指出潜在风险和测试要点
    - 最后汇总三方意见，形成评审结论
    """
    print("\n" + "=" * 60)
    print("示例1：角色委派 - 需求评审会")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - 每个 Agent 用独立的 system prompt 定义角色和职责")
    print("   - 同一个 LLM 通过不同 prompt 扮演不同角色")
    print("   - 并行调用各角色 Agent，最后汇总输出")

    llm = get_default_llm()
    parser = StrOutputParser()

    # 产品经理 Agent
    pm_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位资深产品经理。你擅长分析用户需求、定义产品价值、"
                   "规划功能优先级。请始终从用户价值和商业价值角度分析问题。"),
        ("human", "请从产品经理的角度分析以下需求：\n\n{requirement}\n\n"
                  "请包含：1)用户场景分析 2)核心价值点 3)优先级建议 4)可能的用户反馈"),
    ])

    # 技术负责人 Agent
    tech_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位经验丰富的技术负责人。你擅长评估技术可行性、"
                   "设计实现方案、预估工作量。请始终从技术实现角度分析问题。"),
        ("human", "请从技术负责人的角度分析以下需求：\n\n{requirement}\n\n"
                  "请包含：1)技术可行性评估 2)推荐实现方案 3)预估工作量 4)技术风险点"),
    ])

    # 测试负责人 Agent
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位严谨的测试负责人。你擅长发现需求漏洞、"
                   "设计测试策略、评估质量风险。请始终从质量保障角度分析问题。"),
        ("human", "请从测试负责人的角度分析以下需求：\n\n{requirement}\n\n"
                  "请包含：1)需求模糊点 2)边界场景 3)测试策略建议 4)质量风险提醒"),
    ])

    # 汇总 Agent
    summary_prompt = ChatPromptTemplate.from_template(
        "以下是三方对同一需求的分析意见：\n\n"
        "【产品经理意见】\n{pm_opinion}\n\n"
        "【技术负责人意见】\n{tech_opinion}\n\n"
        "【测试负责人意见】\n{qa_opinion}\n\n"
        "请综合三方意见，输出一份结构化的评审结论：\n"
        "1. 需求概述（一句话）\n"
        "2. 三方共识点\n"
        "3. 三方分歧点\n"
        "4. 综合建议（是否推进、注意事项）"
    )

    # 构建各角色链
    pm_chain = pm_prompt | llm | parser
    tech_chain = tech_prompt | llm | parser
    qa_chain = qa_prompt | llm | parser
    summary_chain = summary_prompt | llm | parser

    print("\n【交互式需求评审】")
    print("产品经理、技术负责人、测试负责人将分别分析你的需求")
    print("\n输入 '退出' 结束\n")

    while True:
        requirement = input("请输入要评审的需求：").strip()
        if requirement.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if not requirement:
            print("请输入需求内容")
            continue

        try:
            # 产品经理分析
            print("\n" + "─" * 40)
            print("📋 产品经理正在分析...")
            pm_result = pm_chain.invoke({"requirement": requirement})
            print(f"\n【产品经理意见】\n{pm_result}")

            # 技术负责人分析
            print("\n" + "─" * 40)
            print("🔧 技术负责人正在分析...")
            tech_result = tech_chain.invoke({"requirement": requirement})
            print(f"\n【技术负责人意见】\n{tech_result}")

            # 测试负责人分析
            print("\n" + "─" * 40)
            print("🧪 测试负责人正在分析...")
            qa_result = qa_chain.invoke({"requirement": requirement})
            print(f"\n【测试负责人意见】\n{qa_result}")

            # 汇总
            print("\n" + "─" * 40)
            print("📝 正在汇总评审结论...")
            summary_result = summary_chain.invoke({
                "pm_opinion": pm_result,
                "tech_opinion": tech_result,
                "qa_opinion": qa_result,
            })
            print(f"\n【评审结论】\n{summary_result}")

            print("\n✅ 需求评审完成！")

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. 角色委派的核心是为每个 Agent 设计清晰的角色 prompt")
    print("   2. 同一个 LLM 实例可以复用，prompt 决定了 Agent 的'性格'")
    print("   3. 各角色输出独立，最后通过汇总 Agent 整合意见")


# ============================================================
# 2. 流水线 Agent：多个 Agent 按顺序处理
# ============================================================

def demo_pipeline_agents():
    """流水线 Agent：选题→大纲→初稿→润色，逐步推进

    场景：文章创作流水线
    - 选题 Agent：根据主题生成选题方向和切入点
    - 大纲 Agent：根据选题生成文章大纲
    - 初稿 Agent：根据大纲撰写初稿
    - 润色 Agent：对初稿进行润色优化
    """
    print("\n" + "=" * 60)
    print("示例2：流水线 Agent - 文章创作流水线")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - 流水线中每一步的输出是下一步的输入")
    print("   - 每个步骤的 prompt 只聚焦当前环节，降低复杂度")
    print("   - 中间结果可以用变量保存，供最终步骤汇总使用")

    llm = get_default_llm()
    parser = StrOutputParser()

    # 步骤1：选题 Agent
    topic_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位资深选题编辑。你擅长挖掘话题的新颖切入点，"
                   "让文章有吸引力。输出简洁明确，只输出选题建议。"),
        ("human", "主题：{theme}\n目标读者：{audience}\n\n"
                  "请给出3个选题方向，每个方向包含：切入角度、标题建议、一句话亮点。"),
    ])

    # 步骤2：大纲 Agent
    outline_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业内容架构师。你擅长设计逻辑清晰的内容结构。"
                   "只输出大纲，不要写正文。"),
        ("human", "根据以下选题方向，生成文章大纲：\n\n{topic}\n\n"
                  "要求：\n"
                  "1. 包含引言、3-4个核心章节、结尾\n"
                  "2. 每个章节标注2-3个要点\n"
                  "3. 标注每个章节的预估字数"),
    ])

    # 步骤3：初稿 Agent
    draft_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位高效的撰稿人。你擅长根据大纲快速撰写内容丰富的初稿。"
                   "文字流畅，逻辑清晰，但不追求完美措辞。"),
        ("human", "请根据以下大纲撰写文章初稿：\n\n{outline}\n\n"
                  "要求：\n"
                  "1. 严格按照大纲结构展开\n"
                  "2. 每个要点用1-2段详细阐述\n"
                  "3. 总字数 800-1200 字"),
    ])

    # 步骤4：润色 Agent
    polish_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位文字匠人。你擅长打磨文字：精简冗余、强化表达、"
                   "优化节奏。保持原意不变，让文字更有力量。"),
        ("human", "请润色以下文章初稿：\n\n{draft}\n\n"
                  "润色要点：\n"
                  "1. 删减冗余表达，让每句话都有信息量\n"
                  "2. 优化段落过渡，让行文更流畅\n"
                  "3. 强化开头和结尾的感染力\n"
                  "4. 修正语病和逻辑不清的地方"),
    ])

    # 构建各步骤链
    topic_chain = topic_prompt | llm | parser
    outline_chain = outline_prompt | llm | parser
    draft_chain = draft_prompt | llm | parser
    polish_chain = polish_prompt | llm | parser

    print("\n【交互式文章创作流水线】")
    print("选题 → 大纲 → 初稿 → 润色，四个 Agent 逐步推进")
    print("\n输入 '退出' 结束\n")

    while True:
        theme = input("你想写什么主题的文章？：").strip()
        if theme.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if not theme:
            print("请输入主题")
            continue

        audience = input("目标读者是谁？(如：技术新人/管理层/大众)：").strip()
        if not audience:
            audience = "一般读者"

        try:
            # 步骤1：选题
            print("\n" + "─" * 40)
            print("💡 步骤1/4：选题 Agent 工作中...")
            topic_result = topic_chain.invoke({
                "theme": theme,
                "audience": audience,
            })
            print(f"\n【选题方向】\n{topic_result}")

            # 让用户选择方向
            chosen_topic = input("\n请选择一个方向（直接输入或按回车使用第一个方向）：").strip()
            if not chosen_topic:
                chosen_topic = topic_result

            # 步骤2：大纲
            print("\n" + "─" * 40)
            print("📑 步骤2/4：大纲 Agent 工作中...")
            outline_result = outline_chain.invoke({"topic": chosen_topic})
            print(f"\n【文章大纲】\n{outline_result}")

            # 步骤3：初稿
            print("\n" + "─" * 40)
            print("✍️ 步骤3/4：初稿 Agent 工作中...")
            draft_result = draft_chain.invoke({"outline": outline_result})
            print(f"\n【文章初稿】\n{draft_result}")

            # 步骤4：润色
            print("\n" + "─" * 40)
            print("🎨 步骤4/4：润色 Agent 工作中...")
            polish_result = polish_chain.invoke({"draft": draft_result})
            print(f"\n【润色完成】\n{polish_result}")

            print("\n✅ 文章创作流水线完成！")

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. 流水线模式的核心是'上一步输出 = 下一步输入'")
    print("   2. 每个步骤的 prompt 职责单一，避免一个 prompt 做太多事")
    print("   3. 可以在步骤之间插入用户交互（如选择选题方向）")


# ============================================================
# 3. 辩论 Agent：两个 Agent 对同一问题给出不同观点
# ============================================================

def demo_debate_agents():
    """辩论 Agent：正方与反方对同一议题展开辩论

    场景：方案辩论赛
    - 正方 Agent：支持议题，阐述优势和可行性
    - 反方 Agent：反对议题，指出风险和问题
    - 评委 Agent：综合双方观点，给出客观评判
    """
    print("\n" + "=" * 60)
    print("示例3：辩论 Agent - 方案辩论赛")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - 两个 Agent 持相反立场，用对立 prompt 实现")
    print("   - 正反方依次发言，反方可以看到正方论点并反驳")
    print("   - 评委 Agent 不站队，综合双方观点做评判")

    llm = get_default_llm()
    parser = StrOutputParser()

    # 正方 Agent
    pro_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位正方辩手。你的任务是全力支持给定的议题，"
                   "从优势、可行性、成功案例等角度阐述理由。"
                   "论证要有力，但不能无理取闹。保持专业和理性。"),
        ("human", "辩论议题：{motion}\n\n"
                  "请作为正方，提出3-5个核心论点支持该议题。"
                  "每个论点需要：论点陈述 + 论据支撑 + 现实案例。"),
    ])

    # 反方 Agent
    con_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位反方辩手。你的任务是全力反对给定的议题，"
                   "从风险、弊端、失败案例等角度阐述理由。"
                   "论证要有力，但不能无理取闹。保持专业和理性。"),
        ("human", "辩论议题：{motion}\n\n"
                  "正方论点：\n{pro_arguments}\n\n"
                  "请作为反方：\n"
                  "1. 针对正方的每个论点进行反驳\n"
                  "2. 提出反方的3-5个核心反对论点\n"
                  "3. 每个论点需要：反驳理由 + 独立论据 + 风险警示"),
    ])

    # 正方总结陈词
    pro_closing_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位正方辩手，正在进行总结陈词。"
                   "回应反方质疑，巩固己方论点，保持理性和说服力。"),
        ("human", "辩论议题：{motion}\n\n"
                  "正方论点：\n{pro_arguments}\n\n"
                  "反方反驳：\n{con_arguments}\n\n"
                  "请进行正方总结陈词：\n"
                  "1. 回应反方的核心质疑\n"
                  "2. 重申正方最有力的2-3个论点\n"
                  "3. 用一句话总结正方立场"),
    ])

    # 评委 Agent
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位客观公正的评委。你不站队，综合正反双方观点给出评判。"
                   "指出双方的优劣势，给出平衡的建议。"),
        ("human", "辩论议题：{motion}\n\n"
                  "【正方论点】\n{pro_arguments}\n\n"
                  "【反方反驳】\n{con_arguments}\n\n"
                  "【正方总结】\n{pro_closing}\n\n"
                  "请作为评委给出评判：\n"
                  "1. 正方论点评分（1-10）及理由\n"
                  "2. 反方论点评分（1-10）及理由\n"
                  "3. 双方最有说服力的论点各一个\n"
                  "4. 综合建议：如何在实践中兼顾双方关切"),
    ])

    # 构建各链
    pro_chain = pro_prompt | llm | parser
    con_chain = con_prompt | llm | parser
    pro_closing_chain = pro_closing_prompt | llm | parser
    judge_chain = judge_prompt | llm | parser

    print("\n【交互式方案辩论赛】")
    print("正方支持、反方反对、评委评判，三方交锋")
    print("\n输入 '退出' 结束\n")

    while True:
        motion = input("请输入辩论议题（如：公司应该全面推行远程办公）：").strip()
        if motion.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if not motion:
            print("请输入议题")
            continue

        try:
            # 正方发言
            print("\n" + "═" * 40)
            print("🟢 正方发言")
            print("═" * 40)
            pro_result = pro_chain.invoke({"motion": motion})
            print(pro_result)

            # 反方反驳
            print("\n" + "═" * 40)
            print("🔴 反方反驳")
            print("═" * 40)
            con_result = con_chain.invoke({
                "motion": motion,
                "pro_arguments": pro_result,
            })
            print(con_result)

            # 正方总结陈词
            print("\n" + "═" * 40)
            print("🟢 正方总结陈词")
            print("═" * 40)
            pro_closing_result = pro_closing_chain.invoke({
                "motion": motion,
                "pro_arguments": pro_result,
                "con_arguments": con_result,
            })
            print(pro_closing_result)

            # 评委评判
            print("\n" + "═" * 40)
            print("⚖️ 评委评判")
            print("═" * 40)
            judge_result = judge_chain.invoke({
                "motion": motion,
                "pro_arguments": pro_result,
                "con_arguments": con_result,
                "pro_closing": pro_closing_result,
            })
            print(judge_result)

            print("\n✅ 辩论结束！")

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. 辩论 Agent 的核心是用对立的 prompt 让 LLM 产生不同立场")
    print("   2. 反方可以看到正方论点再反驳，形成真正的交锋而非各说各话")
    print("   3. 评委 Agent 综合双方观点，输出比单方分析更全面、更客观")


# ============================================================
# 4. 主管 Agent：一个 Agent 负责分配任务给其他 Agent
# ============================================================

def demo_supervisor_agent():
    """主管 Agent：分析任务并智能分配给专业 Agent

    场景：智能客服主管
    - 主管 Agent：分析用户问题，判断需要哪些专业 Agent 参与
    - 技术支持 Agent：处理技术问题
    - 售后服务 Agent：处理售后问题
    - 产品建议 Agent：收集产品改进建议
    """
    print("\n" + "=" * 60)
    print("示例4：主管 Agent - 智能客服主管")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - 主管 Agent 的职责是'理解任务 → 分配任务 → 汇总结果'")
    print("   - 主管先分析问题类型，决定派给哪些专业 Agent")
    print("   - 专业 Agent 只处理自己领域的问题，输出更精准")

    llm = get_default_llm()
    parser = StrOutputParser()

    # 主管 Agent：分析问题并分配任务
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位智能客服主管。你的职责是分析用户的问题，"
                   "判断需要哪些专业团队处理，并为每个团队明确任务要求。\n\n"
                   "可用的专业团队：\n"
                   "- 技术支持：处理产品使用、故障排查、技术配置等问题\n"
                   "- 售后服务：处理退换货、物流、质量投诉等问题\n"
                   "- 产品建议：收集功能需求、体验反馈、改进建议\n\n"
                   "请输出JSON格式（不要用markdown代码块）：\n"
                   '{{\"tech_support\": \"技术支持任务描述，无则为空字符串\",'
                   ' \"after_sales\": \"售后服务任务描述，无则为空字符串\",'
                   ' \"product_feedback\": \"产品建议任务描述，无则为空字符串\"}}'),
        ("human", "用户问题：{question}"),
    ])

    # 技术支持 Agent
    tech_support_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的技术支持工程师。你擅长排查问题、"
                   "提供解决方案。回答要具体、可操作，包含步骤说明。"),
        ("human", "{task}"),
    ])

    # 售后服务 Agent
    after_sales_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位耐心的售后服务专员。你擅长处理退换货、"
                   "物流查询、质量投诉等问题。态度友善，流程清晰。"),
        ("human", "{task}"),
    ])

    # 产品建议 Agent
    product_feedback_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位产品经理。你擅长理解用户需求，"
                   "将反馈转化为可执行的产品改进方案。"),
        ("human", "{task}"),
    ])

    # 主管汇总 Agent
    supervisor_summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位智能客服主管。请将各专业团队的回复整合为一份"
                   "连贯、完整的回复给用户。语言友好专业，逻辑清晰。"),
        ("human", "用户原始问题：{question}\n\n各团队回复：\n{team_replies}\n\n"
                  "请整合为一份给用户的完整回复。"),
    ])

    # 构建各链
    supervisor_chain = supervisor_prompt | llm | parser
    tech_chain = tech_support_prompt | llm | parser
    sales_chain = after_sales_prompt | llm | parser
    feedback_chain = product_feedback_prompt | llm | parser
    summary_chain = supervisor_summary_prompt | llm | parser

    print("\n【交互式智能客服】")
    print("主管 Agent 会分析你的问题并分配给专业团队处理")
    print("\n输入 '退出' 结束\n")

    while True:
        question = input("请描述你遇到的问题：").strip()
        if question.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if not question:
            print("请输入问题")
            continue

        try:
            # 主管分析并分配任务
            print("\n" + "─" * 40)
            print("👔 主管 Agent 正在分析问题...")
            allocation_result = supervisor_chain.invoke({"question": question})
            print(f"\n【任务分配结果】\n{allocation_result}")

            # 解析分配结果
            import json
            try:
                # 尝试从结果中提取 JSON
                result_text = allocation_result.strip()
                # 去除可能的 markdown 代码块标记
                if result_text.startswith("```"):
                    lines = result_text.split("\n")
                    result_text = "\n".join(lines[1:-1])
                tasks = json.loads(result_text)
            except json.JSONDecodeError:
                print("⚠️ 主管分配结果解析失败，将所有团队都派上用场")
                tasks = {
                    "tech_support": question,
                    "after_sales": question,
                    "product_feedback": question,
                }

            # 各专业 Agent 处理任务
            team_replies = ""

            if tasks.get("tech_support"):
                print("\n" + "─" * 40)
                print("🔧 技术支持 Agent 工作中...")
                tech_result = tech_chain.invoke({"task": tasks["tech_support"]})
                print(f"\n【技术支持回复】\n{tech_result}")
                team_replies += f"【技术支持】\n{tech_result}\n\n"

            if tasks.get("after_sales"):
                print("\n" + "─" * 40)
                print("📦 售后服务 Agent 工作中...")
                sales_result = sales_chain.invoke({"task": tasks["after_sales"]})
                print(f"\n【售后服务回复】\n{sales_result}")
                team_replies += f"【售后服务】\n{sales_result}\n\n"

            if tasks.get("product_feedback"):
                print("\n" + "─" * 40)
                print("💡 产品建议 Agent 工作中...")
                feedback_result = feedback_chain.invoke({"task": tasks["product_feedback"]})
                print(f"\n【产品建议回复】\n{feedback_result}")
                team_replies += f"【产品建议】\n{feedback_result}\n\n"

            if not team_replies:
                team_replies = "所有团队均无相关任务。"

            # 主管汇总
            print("\n" + "─" * 40)
            print("👔 主管 Agent 正在汇总回复...")
            final_result = summary_chain.invoke({
                "question": question,
                "team_replies": team_replies,
            })
            print(f"\n【主管汇总回复】\n{final_result}")

            print("\n✅ 问题处理完成！")

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. 主管 Agent 负责理解需求、分配任务、汇总结果，是调度核心")
    print("   2. 专业 Agent 只处理自己领域的任务，输出更精准")
    print("   3. 主管可以让 LLM 输出结构化数据（如 JSON）来控制任务路由")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "=" * 60)
    print("  LangChain 多 Agent 协作 - 实战交互式案例")
    print("=" * 60)
    print("\n本示例演示四种多 Agent 协作模式")
    print("\n核心概念：")
    print("  • 角色委派：不同角色各司其职，协同完成复杂任务")
    print("  • 流水线 Agent：按顺序处理，上一步输出 = 下一步输入")
    print("  • 辩论 Agent：正反方交锋，评委客观评判")
    print("  • 主管 Agent：智能分析任务并分配给专业 Agent")
    print("\n应用场景：")
    print("  • 需求评审会、文章创作、方案辩论、智能客服")

    while True:
        print("\n" + "=" * 60)
        print("请选择要运行的示例：")
        print("=" * 60)
        print("  1. 角色委派 - 需求评审会（产品/技术/测试三方协作）")
        print("  2. 流水线 Agent - 文章创作流水线（选题→大纲→初稿→润色）")
        print("  3. 辩论 Agent - 方案辩论赛（正方/反方/评委三方交锋）")
        print("  4. 主管 Agent - 智能客服主管（主管分配→专业处理→汇总回复）")
        print("\n  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_role_delegation()
        elif choice == "2":
            demo_pipeline_agents()
        elif choice == "3":
            demo_debate_agents()
        elif choice == "4":
            demo_supervisor_agent()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
