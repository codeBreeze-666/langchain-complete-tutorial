"""
LangGraph 多 Agent 协作 - 实战交互式案例
==========================================

本示例演示 LangGraph 中四种多 Agent 协作模式，
使用 StateGraph 构建多 Agent 图，展示 Agent 间如何通过状态共享协同工作

核心概念：
- 多Agent协作: 多个智能体在同一个图中协同工作，各自承担不同职责
- 主管(Supervisor): 负责任务分解和委派的中心 Agent，协调其他 Agent
- 流水线(Pipeline): 多个 Agent 按顺序执行，前一步输出是后一步输入
- 状态共享: 所有 Agent 通过共享的 State 对象交换信息，避免消息丢失

应用场景：
- 项目管理：主管委派任务给不同角色（产品、技术、测试）
- 文章创作：大纲→初稿→润色→终稿的流水线加工
- 决策分析：正反方 Agent 辩论后给出客观结论
- 复杂任务：主管 Agent 分解需求并智能分配
"""

import os
import sys
import json

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from src.utils.llm_loader import get_default_llm


# ============================================================
# 示例1: 角色委派 - 项目管理
# ============================================================

def demo_role_delegation():
    """角色委派 - 用户输入需求，自动委派给不同角色 Agent 处理

    核心概念：
    - 角色节点: 每个 Agent 是图中的一个节点，通过 system prompt 定义角色
    - 状态共享: 所有角色从同一个 State 中读取任务，将结果写回 State
    - 汇总节点: 收集各角色输出，整合为最终结果
    """
    print("\n" + "=" * 60)
    print("示例1：角色委派 - 项目管理")
    print("=" * 60)
    print("\n💡 核心概念：")
    print("   - 角色节点: 每个 Agent 是图中的一个节点")
    print("   - 状态共享: 所有角色从同一个 State 读取和写入")
    print("   - 汇总节点: 收集各角色输出，整合为最终结果")

    llm = get_default_llm()

    # 定义多 Agent 共享状态
    class ProjectState(TypedDict):
        messages: Annotated[list, add_messages]
        requirement: str        # 原始需求
        pm_analysis: str        # 产品经理分析
        tech_analysis: str      # 技术负责人分析
        qa_analysis: str        # 测试负责人分析
        final_report: str       # 最终汇总报告

    # 产品经理 Agent 节点
    def pm_agent(state: ProjectState) -> dict:
        """产品经理：分析需求价值和用户场景"""
        prompt = [
            SystemMessage(content=(
                "你是一位资深产品经理。你擅长分析用户需求、定义产品价值、"
                "规划功能优先级。请从用户价值和商业价值角度分析需求。"
            )),
            HumanMessage(content=(
                f"请从产品经理的角度分析以下需求：\n\n{state['requirement']}\n\n"
                "请包含：1)用户场景分析 2)核心价值点 3)优先级建议 4)可能的用户反馈"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "pm_analysis": response.content}

    # 技术负责人 Agent 节点
    def tech_agent(state: ProjectState) -> dict:
        """技术负责人：评估技术可行性和实现方案"""
        prompt = [
            SystemMessage(content=(
                "你是一位经验丰富的技术负责人。你擅长评估技术可行性、"
                "设计实现方案、预估工作量。请从技术实现角度分析需求。"
            )),
            HumanMessage(content=(
                f"请从技术负责人的角度分析以下需求：\n\n{state['requirement']}\n\n"
                "请包含：1)技术可行性评估 2)推荐实现方案 3)预估工作量 4)技术风险点"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "tech_analysis": response.content}

    # 测试负责人 Agent 节点
    def qa_agent(state: ProjectState) -> dict:
        """测试负责人：指出潜在风险和测试要点"""
        prompt = [
            SystemMessage(content=(
                "你是一位严谨的测试负责人。你擅长发现需求漏洞、"
                "设计测试策略、评估质量风险。请从质量保障角度分析需求。"
            )),
            HumanMessage(content=(
                f"请从测试负责人的角度分析以下需求：\n\n{state['requirement']}\n\n"
                "请包含：1)需求模糊点 2)边界场景 3)测试策略建议 4)质量风险提醒"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "qa_analysis": response.content}

    # 汇总 Agent 节点
    def summary_agent(state: ProjectState) -> dict:
        """汇总 Agent：综合三方意见，输出评审报告"""
        prompt = [
            SystemMessage(content="你是一位项目管理专家，请综合三方意见输出评审报告。"),
            HumanMessage(content=(
                f"原始需求：{state['requirement']}\n\n"
                f"【产品经理意见】\n{state['pm_analysis']}\n\n"
                f"【技术负责人意见】\n{state['tech_analysis']}\n\n"
                f"【测试负责人意见】\n{state['qa_analysis']}\n\n"
                "请综合三方意见，输出评审报告：\n"
                "1. 需求概述\n2. 三方共识点\n3. 三方分歧点\n4. 综合建议"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "final_report": response.content}

    # 构建图：需求 → 产品/技术/测试（并行）→ 汇总
    workflow = StateGraph(ProjectState)

    # 添加各角色节点
    workflow.add_node("pm_agent", pm_agent)
    workflow.add_node("tech_agent", tech_agent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summary_agent", summary_agent)

    # 从 START 分发到三个角色（LangGraph 自动并行执行）
    workflow.add_edge(START, "pm_agent")
    workflow.add_edge(START, "tech_agent")
    workflow.add_edge(START, "qa_agent")

    # 三个角色完成后汇聚到汇总节点
    workflow.add_edge("pm_agent", "summary_agent")
    workflow.add_edge("tech_agent", "summary_agent")
    workflow.add_edge("qa_agent", "summary_agent")

    workflow.add_edge("summary_agent", END)

    graph = workflow.compile()

    print("\n【交互式需求评审】")
    print("产品经理、技术负责人、测试负责人将并行分析你的需求")
    print("\n试试说：")
    print("  • '开发一个在线教育平台，支持视频课程和实时互动'")
    print("  • '做一个智能客服系统，能自动回答常见问题'")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("需求描述：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效内容")
            continue

        try:
            print("\n⏳ 三个角色正在并行分析...")

            result = graph.invoke({
                "messages": [HumanMessage(content=user_input)],
                "requirement": user_input,
                "pm_analysis": "", "tech_analysis": "", "qa_analysis": "", "final_report": "",
            })

            # 展示各角色分析
            print(f"\n{'='*60}")
            print("📋 产品经理分析：")
            print(f"{'='*60}")
            print(result["pm_analysis"][:400])
            print("...")

            print(f"\n{'='*60}")
            print("🔧 技术负责人分析：")
            print(f"{'='*60}")
            print(result["tech_analysis"][:400])
            print("...")

            print(f"\n{'='*60}")
            print("🧪 测试负责人分析：")
            print(f"{'='*60}")
            print(result["qa_analysis"][:400])
            print("...")

            print(f"\n{'='*60}")
            print("📝 评审报告：")
            print(f"{'='*60}")
            print(result["final_report"])

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 核心概念总结：")
    print("   1. 每个 Agent 是 StateGraph 中的一个节点")
    print("   2. 从 START 到多个角色的边实现并行执行")
    print("   3. 多个角色到汇总节点的边实现结果汇聚")
    print("   4. 所有 Agent 通过共享的 State 对象交换信息")


# ============================================================
# 示例2: 流水线模式 - 文章创作
# ============================================================

def demo_pipeline_mode():
    """流水线模式 - 用户输入主题，多个 Agent 依次处理：大纲→初稿→润色→终稿

    核心概念：
    - 流水线(Pipeline): 多个 Agent 按顺序执行，上一步输出是下一步输入
    - 顺序边: add_edge(A, B) 确保 A 完成后 B 才开始
    - 状态传递: 每个节点将结果写入 State，下一个节点从 State 读取
    """
    print("\n" + "=" * 60)
    print("示例2：流水线模式 - 文章创作")
    print("=" * 60)
    print("\n💡 核心概念：")
    print("   - 流水线(Pipeline): Agent 按顺序执行，上一步输出=下一步输入")
    print("   - 顺序边: add_edge(A, B) 确保 A 完成后 B 才开始")
    print("   - 状态传递: 每个节点写结果到 State，下一节点从 State 读取")

    llm = get_default_llm()

    class ArticleState(TypedDict):
        messages: Annotated[list, add_messages]
        topic: str             # 文章主题
        outline: str           # 大纲
        draft: str             # 初稿
        polished: str          # 润色稿
        final_article: str     # 终稿

    def outline_agent(state: ArticleState) -> dict:
        """大纲 Agent：根据主题生成文章大纲"""
        prompt = [
            SystemMessage(content="你是一位专业内容架构师，擅长设计逻辑清晰的文章大纲。只输出大纲。"),
            HumanMessage(content=(
                f"请为以下主题生成文章大纲：\n\n主题：{state['topic']}\n\n"
                "要求：包含引言、3-4个核心章节、结尾，每个章节标注要点。"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "outline": response.content}

    def draft_agent(state: ArticleState) -> dict:
        """初稿 Agent：根据大纲撰写初稿"""
        prompt = [
            SystemMessage(content="你是一位高效的撰稿人，擅长根据大纲快速撰写内容丰富的初稿。"),
            HumanMessage(content=(
                f"请根据以下大纲撰写文章初稿：\n\n{state['outline']}\n\n"
                "要求：严格按照大纲结构，每点用1-2段阐述，800-1200字。"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "draft": response.content}

    def polish_agent(state: ArticleState) -> dict:
        """润色 Agent：对初稿进行润色"""
        prompt = [
            SystemMessage(content="你是一位文字匠人，擅长打磨文字：精简冗余、强化表达、优化节奏。"),
            HumanMessage(content=(
                f"请润色以下文章初稿：\n\n{state['draft']}\n\n"
                "润色要点：删减冗余、优化过渡、强化首尾、修正语病。"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "polished": response.content}

    def final_agent(state: ArticleState) -> dict:
        """终稿 Agent：最终审核和定稿"""
        prompt = [
            SystemMessage(content="你是主编，做最终审核。确认文章质量达标，添加标题，输出终稿。"),
            HumanMessage(content=(
                f"主题：{state['topic']}\n\n润色稿：\n{state['polished']}\n\n"
                "请审核并定稿：添加合适标题，做必要微调，输出完整文章。"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "final_article": response.content}

    # 构建流水线图：大纲 → 初稿 → 润色 → 终稿
    workflow = StateGraph(ArticleState)

    workflow.add_node("outline", outline_agent)
    workflow.add_node("draft", draft_agent)
    workflow.add_node("polish", polish_agent)
    workflow.add_node("final", final_agent)

    # 顺序边：确保严格按流水线执行
    workflow.add_edge(START, "outline")
    workflow.add_edge("outline", "draft")
    workflow.add_edge("draft", "polish")
    workflow.add_edge("polish", "final")
    workflow.add_edge("final", END)

    graph = workflow.compile()

    print("\n【交互式文章创作流水线】")
    print("大纲 → 初稿 → 润色 → 终稿，四个 Agent 依次处理")
    print("\n试试说：")
    print("  • '如何培养良好的编程习惯'")
    print("  • '人工智能对教育的影响'")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("文章主题：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效主题")
            continue

        try:
            print("\n⏳ 流水线正在运行...")

            result = graph.invoke({
                "messages": [],
                "topic": user_input,
                "outline": "", "draft": "", "polished": "", "final_article": "",
            })

            # 展示各阶段输出
            print(f"\n{'═'*60}")
            print("📑 第1步 - 文章大纲：")
            print(f"{'═'*60}")
            print(result["outline"][:300])
            print("...")

            print(f"\n{'═'*60}")
            print("✍️ 第2步 - 文章初稿：")
            print(f"{'═'*60}")
            print(result["draft"][:300])
            print("...")

            print(f"\n{'═'*60}")
            print("🎨 第3步 - 润色稿：")
            print(f"{'═'*60}")
            print(result["polished"][:300])
            print("...")

            print(f"\n{'═'*60}")
            print("📖 第4步 - 最终定稿：")
            print(f"{'═'*60}")
            print(result["final_article"])

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 核心概念总结：")
    print("   1. 流水线模式 = 顺序边连接的多个 Agent 节点")
    print("   2. 上一步的输出写入 State，下一步从 State 读取")
    print("   3. add_edge(A, B) 保证 A 完成后 B 才执行")
    print("   4. 流水线适用于需要逐步精炼的场景（写作、翻译、代码审查）")


# ============================================================
# 示例3: 辩论模式 - 决策分析
# ============================================================

def demo_debate_mode():
    """辩论模式 - 用户输入议题，正反方 Agent 辩论后给出结论

    核心概念：
    - 辩论结构: 正方发言 → 反方反驳 → 正方总结 → 评委评判
    - 信息传递: 反方可以看到正方论点，形成真正的交锋
    - 评委节点: 综合双方观点，输出客观结论
    """
    print("\n" + "=" * 60)
    print("示例3：辩论模式 - 决策分析")
    print("=" * 60)
    print("\n💡 核心概念：")
    print("   - 辩论结构: 正方→反方→总结→评判，形成交锋")
    print("   - 信息传递: 反方可以看到正方论点并针对性反驳")
    print("   - 评委节点: 综合双方观点，输出客观结论")

    llm = get_default_llm()

    class DebateState(TypedDict):
        messages: Annotated[list, add_messages]
        motion: str            # 辩论议题
        pro_arguments: str     # 正方论点
        con_arguments: str     # 反方论点
        pro_closing: str       # 正方总结陈词
        judge_verdict: str     # 评委裁决

    def pro_agent(state: DebateState) -> dict:
        """正方 Agent：全力支持议题"""
        prompt = [
            SystemMessage(content=(
                "你是一位正方辩手，全力支持给定议题。"
                "从优势、可行性、成功案例等角度阐述，保持专业和理性。"
            )),
            HumanMessage(content=(
                f"辩论议题：{state['motion']}\n\n"
                "请提出3-5个核心论点支持该议题，每个论点需有论据和案例。"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "pro_arguments": response.content}

    def con_agent(state: DebateState) -> dict:
        """反方 Agent：全力反对议题，针对正方论点反驳"""
        prompt = [
            SystemMessage(content=(
                "你是一位反方辩手，全力反对给定议题。"
                "从风险、弊端、失败案例等角度阐述，保持专业和理性。"
            )),
            HumanMessage(content=(
                f"辩论议题：{state['motion']}\n\n"
                f"正方论点：\n{state['pro_arguments']}\n\n"
                "请：1) 针对正方论点反驳 2) 提出3-5个反对论点 3) 每个论点需有论据"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "con_arguments": response.content}

    def pro_closing_agent(state: DebateState) -> dict:
        """正方总结陈词 Agent"""
        prompt = [
            SystemMessage(content="你是正方辩手，进行总结陈词。回应质疑，巩固论点。"),
            HumanMessage(content=(
                f"议题：{state['motion']}\n\n"
                f"正方论点：\n{state['pro_arguments']}\n\n"
                f"反方反驳：\n{state['con_arguments']}\n\n"
                "请进行总结陈词：回应质疑、重申核心论点、总结立场。"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "pro_closing": response.content}

    def judge_agent(state: DebateState) -> dict:
        """评委 Agent：客观评判"""
        prompt = [
            SystemMessage(content="你是客观公正的评委，不站队，综合双方观点给出评判。"),
            HumanMessage(content=(
                f"辩论议题：{state['motion']}\n\n"
                f"【正方论点】\n{state['pro_arguments']}\n\n"
                f"【反方反驳】\n{state['con_arguments']}\n\n"
                f"【正方总结】\n{state['pro_closing']}\n\n"
                "请给出评判：\n"
                "1. 正方评分(1-10)及理由\n2. 反方评分(1-10)及理由\n"
                "3. 双方最有说服力的论点\n4. 综合建议"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "judge_verdict": response.content}

    # 构建辩论图：正方 → 反方 → 正方总结 → 评委
    workflow = StateGraph(DebateState)

    workflow.add_node("pro", pro_agent)
    workflow.add_node("con", con_agent)
    workflow.add_node("pro_closing", pro_closing_agent)
    workflow.add_node("judge", judge_agent)

    workflow.add_edge(START, "pro")
    workflow.add_edge("pro", "con")
    workflow.add_edge("con", "pro_closing")
    workflow.add_edge("pro_closing", "judge")
    workflow.add_edge("judge", END)

    graph = workflow.compile()

    print("\n【交互式决策辩论】")
    print("正方支持、反方反对、评委评判，三方交锋")
    print("\n试试说：")
    print("  • '公司应该全面推行远程办公'")
    print("  • 'AI 会取代大部分人类工作'")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("辩论议题：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效议题")
            continue

        try:
            print("\n⏳ 辩论正在进行...")

            result = graph.invoke({
                "messages": [],
                "motion": user_input,
                "pro_arguments": "", "con_arguments": "",
                "pro_closing": "", "judge_verdict": "",
            })

            # 展示辩论过程
            print(f"\n{'═'*60}")
            print("🟢 正方论点：")
            print(f"{'═'*60}")
            print(result["pro_arguments"][:300])
            print("...")

            print(f"\n{'═'*60}")
            print("🔴 反方反驳：")
            print(f"{'═'*60}")
            print(result["con_arguments"][:300])
            print("...")

            print(f"\n{'═'*60}")
            print("🟢 正方总结陈词：")
            print(f"{'═'*60}")
            print(result["pro_closing"][:200])
            print("...")

            print(f"\n{'═'*60}")
            print("⚖️ 评委裁决：")
            print(f"{'═'*60}")
            print(result["judge_verdict"])

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 核心概念总结：")
    print("   1. 辩论模式 = 顺序边连接的对抗性 Agent 节点")
    print("   2. 反方通过 State 读取正方论点，形成针对性反驳")
    print("   3. 评委节点综合所有信息，输出比单方分析更客观的结论")
    print("   4. 辩论模式适用于需要多角度审视的决策场景")


# ============================================================
# 示例4: 主管Agent - 复杂任务
# ============================================================

def demo_supervisor_mode():
    """主管 Agent - 用户输入复杂需求，主管分解任务并委派

    核心概念：
    - 主管(Supervisor): 中心节点，负责理解任务并分配给专业 Agent
    - 条件路由: 主管根据任务类型决定委派给哪个 Agent
    - 反馈循环: 专业 Agent 完成后回到主管，主管决定是否需要进一步处理
    """
    print("\n" + "=" * 60)
    print("示例4：主管 Agent - 复杂任务")
    print("=" * 60)
    print("\n💡 核心概念：")
    print("   - 主管(Supervisor): 中心节点，负责理解任务并分配")
    print("   - 条件路由: 主管根据任务类型决定委派给哪个 Agent")
    print("   - 反馈循环: 专业 Agent 完成后回到主管，决定下一步")

    llm = get_default_llm()

    class SupervisorState(TypedDict):
        messages: Annotated[list, add_messages]
        task: str              # 原始任务
        task_type: str         # 任务类型（writing/research/coding）
        worker_result: str     # 专业 Agent 的输出
        supervisor_notes: str  # 主管备注
        final_output: str      # 最终输出
        iteration: int         # 迭代次数

    MAX_ITERATIONS = 3

    def supervisor_node(state: SupervisorState) -> dict:
        """主管节点：分析任务，决定委派给谁"""
        if state["iteration"] == 0:
            # 首次：分析任务类型
            prompt = [
                SystemMessage(content=(
                    "你是一位任务主管。分析用户的任务，判断属于哪类：\n"
                    "- writing: 文案写作、内容创作、翻译\n"
                    "- research: 信息调研、知识问答、数据分析\n"
                    "- coding: 编程开发、代码审查、技术方案\n\n"
                    "只输出一个英文单词（writing/research/coding），不要其他内容。"
                )),
                HumanMessage(content=f"任务：{state['task']}")
            ]
            response = llm.invoke(prompt)
            task_type = response.content.strip().lower()

            # 规范化任务类型
            if "writ" in task_type:
                task_type = "writing"
            elif "research" in task_type or "resear" in task_type:
                task_type = "research"
            elif "cod" in task_type:
                task_type = "coding"
            else:
                task_type = "writing"  # 默认

            return {
                "messages": [response],
                "task_type": task_type,
                "iteration": 1,
            }
        else:
            # 后续：评估专业 Agent 的输出，决定是否需要继续
            prompt = [
                SystemMessage(content=(
                    "你是任务主管。专业团队已完成了工作，请评估结果。\n"
                    "如果结果满足需求，输出「完成」。\n"
                    "如果需要改进，输出具体的改进要求。"
                )),
                HumanMessage(content=(
                    f"原始任务：{state['task']}\n\n"
                    f"专业团队输出：\n{state['worker_result']}\n\n"
                    f"这是第 {state['iteration']} 次迭代。"
                ))
            ]
            response = llm.invoke(prompt)
            return {
                "messages": [response],
                "supervisor_notes": response.content,
            }

    def writing_agent(state: SupervisorState) -> dict:
        """写作 Agent：处理文案写作类任务"""
        prompt = [
            SystemMessage(content="你是一位资深文案写手，擅长各类文案和内容创作。"),
            HumanMessage(content=(
                f"任务：{state['task']}\n\n"
                f"主管备注：{state.get('supervisor_notes', '请完成写作任务')}\n\n"
                "请完成写作任务，输出高质量的内容。"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "worker_result": response.content}

    def research_agent(state: SupervisorState) -> dict:
        """调研 Agent：处理信息调研类任务"""
        prompt = [
            SystemMessage(content="你是一位专业调研分析师，擅长信息收集和深度分析。"),
            HumanMessage(content=(
                f"任务：{state['task']}\n\n"
                f"主管备注：{state.get('supervisor_notes', '请完成调研任务')}\n\n"
                "请完成调研任务，输出详细的分析报告。"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "worker_result": response.content}

    def coding_agent(state: SupervisorState) -> dict:
        """编程 Agent：处理编程开发类任务"""
        prompt = [
            SystemMessage(content="你是一位资深软件工程师，擅长编程开发和技术方案设计。"),
            HumanMessage(content=(
                f"任务：{state['task']}\n\n"
                f"主管备注：{state.get('supervisor_notes', '请完成编程任务')}\n\n"
                "请完成编程任务，输出详细的方案和代码。"
            ))
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "worker_result": response.content}

    def final_node(state: SupervisorState) -> dict:
        """最终节点：汇总输出"""
        return {
            "messages": [AIMessage(content=(
                f"✅ 任务完成！\n\n"
                f"任务类型：{state['task_type']}\n"
                f"迭代次数：{state['iteration']}\n\n"
                f"结果：\n{state['worker_result']}"
            ))],
            "final_output": state["worker_result"],
        }

    def route_to_worker(state: SupervisorState) -> str:
        """路由函数：根据任务类型分发到专业 Agent"""
        return state.get("task_type", "writing")

    def should_continue_supervisor(state: SupervisorState) -> str:
        """路由函数：主管评估后决定继续还是完成"""
        # 达到最大迭代次数，直接完成
        if state["iteration"] >= MAX_ITERATIONS:
            return "final"
        # 检查主管评估
        notes = state.get("supervisor_notes", "")
        if "完成" in notes:
            return "final"
        # 需要改进，回到对应专业 Agent
        return state.get("task_type", "writing")

    # 构建主管模式图
    workflow = StateGraph(SupervisorState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("writing", writing_agent)
    workflow.add_node("research", research_agent)
    workflow.add_node("coding", coding_agent)
    workflow.add_node("final", final_node)

    # 入口 → 主管
    workflow.add_edge(START, "supervisor")

    # 主管根据任务类型分发到专业 Agent
    workflow.add_conditional_edges("supervisor", route_to_worker, {
        "writing": "writing",
        "research": "research",
        "coding": "coding",
    })

    # 专业 Agent 完成后回到主管评估
    workflow.add_edge("writing", "supervisor")
    workflow.add_edge("research", "supervisor")
    workflow.add_edge("coding", "supervisor")

    # 主管评估后决定继续还是完成
    workflow.add_conditional_edges("supervisor", should_continue_supervisor, {
        "writing": "writing",
        "research": "research",
        "coding": "coding",
        "final": "final",
    })

    workflow.add_edge("final", END)

    graph = workflow.compile()

    print("\n【交互式主管 Agent】")
    print("主管 Agent 会分析你的任务，委派给专业 Agent 处理")
    print("\n试试说：")
    print("  • '帮我写一封商务合作邀请函'（写作任务）")
    print("  • '分析2024年AI行业发展趋势'（调研任务）")
    print("  • '设计一个用户登录注册系统的技术方案'（编程任务）")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("任务描述：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效内容")
            continue

        try:
            print("\n⏳ 主管 Agent 正在分析任务...")

            result = graph.invoke({
                "messages": [HumanMessage(content=user_input)],
                "task": user_input,
                "task_type": "",
                "worker_result": "",
                "supervisor_notes": "",
                "final_output": "",
                "iteration": 0,
            })

            # 展示结果
            task_type_names = {"writing": "写作", "research": "调研", "coding": "编程"}
            print(f"\n{'═'*60}")
            print(f"👔 任务类型：{task_type_names.get(result['task_type'], result['task_type'])}")
            print(f"📊 迭代次数：{result['iteration']}")
            print(f"{'═'*60}")

            if result.get("worker_result"):
                print(f"\n📋 专业 Agent 输出：")
                print(result["worker_result"])

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 核心概念总结：")
    print("   1. 主管节点是图的中心，负责任务分析和分配")
    print("   2. 条件路由（add_conditional_edges）根据任务类型分发到不同 Agent")
    print("   3. 专业 Agent 完成后回到主管，形成反馈循环")
    print("   4. 主管可以多轮迭代，直到输出质量达标")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "=" * 60)
    print("  LangGraph 多 Agent 协作 - 实战交互式案例")
    print("=" * 60)
    print("\n本示例演示 LangGraph 中四种多 Agent 协作模式")
    print("\n核心概念：")
    print("  • 多Agent协作: 多个智能体协同工作")
    print("  • 主管(Supervisor): 负责任务分解和委派")
    print("  • 流水线(Pipeline): 顺序执行多个 Agent")
    print("  • 状态共享: Agent 之间通过 State 共享信息")
    print("\n应用场景：")
    print("  • 项目管理、文章创作、决策分析、复杂任务")

    while True:
        print("\n" + "=" * 60)
        print("请选择要运行的示例：")
        print("=" * 60)
        print("  1. 角色委派 - 项目管理（产品/技术/测试并行分析→汇总）")
        print("  2. 流水线模式 - 文章创作（大纲→初稿→润色→终稿）")
        print("  3. 辩论模式 - 决策分析（正方→反方→总结→评判）")
        print("  4. 主管 Agent - 复杂任务（主管分析→委派→评估→迭代）")
        print("\n  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_role_delegation()
        elif choice == "2":
            demo_pipeline_mode()
        elif choice == "3":
            demo_debate_mode()
        elif choice == "4":
            demo_supervisor_mode()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
