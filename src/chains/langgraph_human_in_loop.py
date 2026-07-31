"""
LangGraph 人工介入（Human-in-the-Loop）- 实战交互式案例
========================================================

本示例演示 LangGraph 中四种人工介入模式的实现方式，
使用 LangGraph 的 interrupt()、Command(resume=...) 和 checkpointer 等核心原语

核心概念：
- interrupt(): 在工作流中插入断点，暂停执行并等待人工介入
- Command(resume=...): 恢复被中断的工作流，传入人工审批结果
- checkpointer: 检查点机制，自动保存工作流状态，支持断点恢复
- 断点恢复: 从中断的位置继续执行，无需从头开始

应用场景：
- 内容审批：AI 生成内容后暂停，等待人工审批通过
- 邮件发送：AI 撰写邮件后暂停，等待人工确认才发送
- 数据处理：AI 分析数据后暂停，等待人工选择处理方式
- 长任务管理：AI 分步执行复杂任务，每步暂停等待指令
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from src.utils.llm_loader import get_default_llm


# ============================================================
# 示例1: interrupt()函数 - 内容审批
# ============================================================

def demo_content_approval():
    """内容审批 - AI 生成内容后暂停，等待人工审批

    核心概念：
    - interrupt(): 在节点中调用，暂停工作流并返回信息给调用者
    - Command(resume=...): 恢复工作流，传入人工审批结果
    - MemorySaver: 内存检查点，保存工作流状态
    """
    print("\n" + "=" * 60)
    print("示例1：interrupt() 函数 - 内容审批")
    print("=" * 60)
    print("\n💡 核心概念：")
    print("   - interrupt(): 暂停工作流，将信息返回给调用者")
    print("   - Command(resume=...): 恢复工作流，传入审批结果")
    print("   - MemorySaver: 内存检查点，保存状态以支持恢复")

    llm = get_default_llm()

    # 定义状态
    class ApprovalState(TypedDict):
        messages: Annotated[list, add_messages]
        content: str        # AI 生成的内容
        approved: bool      # 是否已审批通过
        feedback: str       # 审批反馈

    def generate_node(state: ApprovalState) -> dict:
        """生成内容节点"""
        user_request = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_request = msg.content
                break

        prompt = [
            SystemMessage(content="你是一个专业的内容创作者，擅长撰写高质量的文案和方案。"),
            HumanMessage(content=f"请根据以下需求生成内容：\n{user_request}")
        ]
        response = llm.invoke(prompt)
        return {
            "messages": [response],
            "content": response.content,
            "approved": False,
            "feedback": "",
        }

    def review_node(state: ApprovalState) -> dict:
        """审批节点 - 使用 interrupt() 暂停等待人工审批"""
        # 🎯 核心：interrupt() 暂停工作流，将内容展示给用户
        decision = interrupt({
            "type": "content_review",
            "content": state["content"],
            "message": "AI 已生成内容，请审批："
        })
        # 当工作流恢复时，decision 就是 Command(resume=...) 传入的值
        return {
            "approved": decision.get("approved", False),
            "feedback": decision.get("feedback", ""),
        }

    def should_continue_approval(state: ApprovalState) -> str:
        """路由函数：根据审批结果决定走向"""
        if state["approved"]:
            return "publish"
        return "revise"

    def publish_node(state: ApprovalState) -> dict:
        """发布节点：审批通过，正式发布"""
        return {
            "messages": [AIMessage(content=f"✅ 内容已审批通过并发布！\n\n{state['content']}")]
        }

    def revise_node(state: ApprovalState) -> dict:
        """修订节点：根据反馈重新生成"""
        prompt = [
            SystemMessage(content="你是一个专业的内容创作者。请根据反馈意见修订内容。"),
            HumanMessage(content=(
                f"原始内容：\n{state['content']}\n\n"
                f"审批反馈：\n{state['feedback']}\n\n"
                f"请根据反馈修订内容。"
            ))
        ]
        response = llm.invoke(prompt)
        return {
            "messages": [response],
            "content": response.content,
            "approved": False,
            "feedback": "",
        }

    # 构建图
    workflow = StateGraph(ApprovalState)
    workflow.add_node("generate", generate_node)
    workflow.add_node("review", review_node)
    workflow.add_node("publish", publish_node)
    workflow.add_node("revise", revise_node)

    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "review")
    workflow.add_conditional_edges("review", should_continue_approval, {
        "publish": "publish",
        "revise": "revise",
    })
    workflow.add_edge("revise", "review")  # 修订后再审批
    workflow.add_edge("publish", END)

    # 使用 MemorySaver 作为检查点
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    print("\n【交互式内容审批】")
    print("输入内容需求，AI 生成后需要你审批才能发布")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("内容需求：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效内容")
            continue

        try:
            # 每次对话使用唯一的 thread_id
            thread_id = f"approval_{id(user_input)}"
            config = {"configurable": {"thread_id": thread_id}}

            # 第一轮：生成内容并暂停等待审批
            print("\n⏳ AI 正在生成内容...")
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)], "content": "", "approved": False, "feedback": ""},
                config=config,
            )

            # 检查是否被 interrupt 暂停
            state = graph.get_state(config)
            if state.next:  # 还有下一步未执行，说明被 interrupt 暂停了
                # 获取 interrupt 传入的信息
                interrupt_info = state.tasks[0].interrupts[0].value if state.tasks else {}
                print(f"\n{'='*60}")
                print("📝 AI 生成的内容：")
                print(f"{'='*60}")
                print(interrupt_info.get("content", result.get("content", "")))
                print(f"{'='*60}")

                # 人工审批
                print("\n请选择：")
                print("  1. ✅ 审批通过")
                print("  2. ❌ 审批驳回（请输入反馈意见）")
                print("  3. 🚫 取消")

                choice = input("你的选择 (1-3)：").strip()

                if choice == "1":
                    # 恢复工作流，传入审批通过
                    graph.invoke(
                        Command(resume={"approved": True, "feedback": "通过"}),
                        config=config,
                    )
                    print("\n✅ 内容已审批通过并发布！")
                elif choice == "2":
                    feedback = input("请输入反馈意见：").strip()
                    if not feedback:
                        feedback = "内容需要改进"
                    # 恢复工作流，传入审批驳回和反馈
                    result2 = graph.invoke(
                        Command(resume={"approved": False, "feedback": feedback}),
                        config=config,
                    )
                    # 修订后再次审批
                    state2 = graph.get_state(config)
                    if state2.next:
                        interrupt_info2 = state2.tasks[0].interrupts[0].value if state2.tasks else {}
                        print(f"\n{'='*60}")
                        print("📝 修订后的内容：")
                        print(f"{'='*60}")
                        revised_content = interrupt_info2.get("content", "")
                        print(revised_content if revised_content else "（内容正在修订中...）")
                        print(f"{'='*60}")

                        final_choice = input("\n是否批准修订版？(y/n)：").strip().lower()
                        if final_choice == "y":
                            graph.invoke(
                                Command(resume={"approved": True, "feedback": "修订版通过"}),
                                config=config,
                            )
                            print("\n✅ 修订版已审批通过并发布！")
                        else:
                            print("\n❌ 修订版未通过")
                else:
                    print("\n🚫 已取消审批")

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 核心概念总结：")
    print("   1. interrupt() 在节点中调用，暂停工作流并返回信息")
    print("   2. Command(resume=...) 恢复工作流，传入人工审批结果")
    print("   3. MemorySaver 检查点保存状态，支持中断恢复")
    print("   4. 审批不通过可路由到修订节点，修订后再次审批")


# ============================================================
# 示例2: 人工审批流程 - 邮件发送
# ============================================================

def demo_email_approval():
    """邮件审批 - AI 撰写邮件后暂停，等待人工确认后发送

    核心概念：
    - 多步审批: 生成 → 审核 → 发送，每步都可以插入 interrupt
    - 条件恢复: 根据用户的选择决定恢复后的执行路径
    - 状态持久化: checkpointer 确保中断后状态不丢失
    """
    print("\n" + "=" * 60)
    print("示例2：人工审批流程 - 邮件发送")
    print("=" * 60)
    print("\n💡 核心概念：")
    print("   - 多步审批: 生成 → 审核 → 发送，每步可插入 interrupt")
    print("   - 条件恢复: 根据用户选择决定恢复后的执行路径")
    print("   - 状态持久化: checkpointer 确保中断后状态不丢失")

    llm = get_default_llm()

    class EmailState(TypedDict):
        messages: Annotated[list, add_messages]
        recipient: str       # 收件人
        subject: str         # 邮件主题
        body: str            # 邮件正文
        approved: bool       # 是否审批通过
        sent: bool           # 是否已发送

    def compose_node(state: EmailState) -> dict:
        """撰写邮件节点"""
        user_request = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_request = msg.content
                break

        prompt = [
            SystemMessage(content=(
                "你是一个专业的邮件撰写助手。请根据用户需求撰写一封正式的商务邮件。"
                "邮件需要包含主题和正文。"
                "输出格式：\n主题：...\n正文：..."
            )),
            HumanMessage(content=f"请撰写邮件：\n{user_request}")
        ]
        response = llm.invoke(prompt)
        content = response.content

        # 解析主题和正文
        subject = ""
        body = content
        if "主题：" in content or "主题:" in content:
            parts = content.split("\n", 1)
            for line in content.split("\n"):
                if line.startswith("主题：") or line.startswith("主题:"):
                    subject = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                    body_start = content.index(line) + len(line)
                    body = content[body_start:].strip()
                    break

        return {
            "messages": [response],
            "subject": subject,
            "body": body,
            "approved": False,
            "sent": False,
        }

    def confirm_node(state: EmailState) -> dict:
        """确认发送节点 - 使用 interrupt 暂停等待确认"""
        decision = interrupt({
            "type": "email_confirm",
            "subject": state["subject"],
            "body": state["body"],
            "recipient": state.get("recipient", "收件人"),
            "message": "邮件已撰写完成，请确认是否发送："
        })
        return {
            "approved": decision.get("approved", False),
        }

    def should_send(state: EmailState) -> str:
        """路由函数：是否发送邮件"""
        if state["approved"]:
            return "send"
        return "reject"

    def send_node(state: EmailState) -> dict:
        """发送邮件节点"""
        return {
            "messages": [AIMessage(content=(
                f"📧 邮件已发送！\n\n"
                f"主题：{state['subject']}\n"
                f"正文：\n{state['body']}"
            ))],
            "sent": True,
        }

    def reject_node(state: EmailState) -> dict:
        """拒绝发送节点"""
        return {
            "messages": [AIMessage(content="🚫 邮件发送已取消。")]
        }

    # 构建图
    workflow = StateGraph(EmailState)
    workflow.add_node("compose", compose_node)
    workflow.add_node("confirm", confirm_node)
    workflow.add_node("send", send_node)
    workflow.add_node("reject", reject_node)

    workflow.add_edge(START, "compose")
    workflow.add_edge("compose", "confirm")
    workflow.add_conditional_edges("confirm", should_send, {
        "send": "send",
        "reject": "reject",
    })
    workflow.add_edge("send", END)
    workflow.add_edge("reject", END)

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    print("\n【交互式邮件审批】")
    print("描述你要发送的邮件，AI 撰写后需要你确认才能发送")
    print("\n试试说：")
    print("  • '给老板写一封请假邮件，下周三到周五请假'")
    print("  • '给客户写一封项目进度汇报邮件'")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("邮件需求：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效内容")
            continue

        try:
            thread_id = f"email_{id(user_input)}"
            config = {"configurable": {"thread_id": thread_id}}

            # 生成邮件并暂停
            print("\n⏳ AI 正在撰写邮件...")
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "recipient": "", "subject": "", "body": "",
                    "approved": False, "sent": False,
                },
                config=config,
            )

            # 获取中断信息
            state = graph.get_state(config)
            if state.next:
                interrupt_info = state.tasks[0].interrupts[0].value if state.tasks else {}
                print(f"\n{'='*60}")
                print("📧 邮件预览：")
                print(f"{'='*60}")
                print(f"主题：{interrupt_info.get('subject', '')}")
                print(f"\n正文：\n{interrupt_info.get('body', '')}")
                print(f"{'='*60}")

                print("\n请确认：")
                print("  1. ✅ 确认发送")
                print("  2. ❌ 取消发送")
                print("  3. ✏️  修改后发送")

                choice = input("你的选择 (1-3)：").strip()

                if choice == "1":
                    graph.invoke(Command(resume={"approved": True}), config=config)
                    print("\n✅ 邮件已发送！")
                elif choice == "3":
                    # 修改邮件
                    new_body = input("请输入修改后的正文：").strip()
                    if new_body:
                        # 重新生成并确认
                        print(f"\n📧 修改后的邮件：")
                        print(f"主题：{interrupt_info.get('subject', '')}")
                        print(f"正文：{new_body}")
                        send_choice = input("\n确认发送修改后的邮件？(y/n)：").strip().lower()
                        if send_choice == "y":
                            graph.invoke(Command(resume={"approved": True}), config=config)
                            print("\n✅ 修改后的邮件已发送！")
                        else:
                            graph.invoke(Command(resume={"approved": False}), config=config)
                            print("\n🚫 已取消发送")
                    else:
                        graph.invoke(Command(resume={"approved": False}), config=config)
                        print("\n🚫 已取消发送")
                else:
                    graph.invoke(Command(resume={"approved": False}), config=config)
                    print("\n🚫 邮件发送已取消")

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 核心概念总结：")
    print("   1. interrupt() 在确认节点暂停，展示邮件预览等待人工确认")
    print("   2. Command(resume={approved: True/False}) 传入审批结果")
    print("   3. 路由函数根据审批结果决定发送还是取消")
    print("   4. 多步审批确保敏感操作（如邮件发送）不会自动执行")


# ============================================================
# 示例3: 交互式决策 - 数据处理
# ============================================================

def demo_interactive_decision():
    """交互式决策 - AI 分析数据后暂停，等待用户选择处理方式

    核心概念：
    - interrupt() 传入多个选项：让用户在多个处理方式中选择
    - Command(resume=...) 传入用户选择：恢复工作流时携带用户的决策
    - 根据用户选择路由到不同处理节点
    """
    print("\n" + "=" * 60)
    print("示例3：交互式决策 - 数据处理")
    print("=" * 60)
    print("\n💡 核心概念：")
    print("   - interrupt() 传入多个选项，让用户选择处理方式")
    print("   - Command(resume=...) 传入用户选择的处理方式")
    print("   - 根据用户选择路由到不同的处理节点")

    llm = get_default_llm()

    class DataState(TypedDict):
        messages: Annotated[list, add_messages]
        data_description: str   # 数据描述
        analysis: str           # 分析结果
        options: list           # 可选处理方式
        chosen_method: str      # 用户选择的方式
        result: str             # 处理结果

    def analyze_node(state: DataState) -> dict:
        """分析数据节点"""
        user_data = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_data = msg.content
                break

        prompt = [
            SystemMessage(content=(
                "你是一个数据分析专家。请分析用户描述的数据，并给出：\n"
                "1. 数据概要：数据类型、规模、特征\n"
                "2. 数据问题：可能存在的质量问题\n"
                "3. 建议的3种处理方式，每种方式简要说明\n\n"
                "输出格式：\n"
                "数据概要：...\n"
                "数据问题：...\n"
                "处理方式A：...\n"
                "处理方式B：...\n"
                "处理方式C：..."
            )),
            HumanMessage(content=f"请分析以下数据：\n{user_data}")
        ]
        response = llm.invoke(prompt)
        return {
            "messages": [response],
            "data_description": user_data,
            "analysis": response.content,
            "chosen_method": "",
            "result": "",
        }

    def decision_node(state: DataState) -> dict:
        """决策节点 - 使用 interrupt 暂停，让用户选择处理方式"""
        decision = interrupt({
            "type": "data_processing_decision",
            "analysis": state["analysis"],
            "message": "数据分析完成，请选择处理方式：",
            "options": ["A", "B", "C"],
        })
        return {
            "chosen_method": decision.get("method", "A"),
        }

    def should_process(state: DataState) -> str:
        """路由函数：根据用户选择路由到不同处理节点"""
        method = state.get("chosen_method", "A")
        if method == "A":
            return "process_a"
        elif method == "B":
            return "process_b"
        else:
            return "process_c"

    def process_a_node(state: DataState) -> dict:
        """处理方式A：基础清洗"""
        prompt = [
            SystemMessage(content="你是数据清洗专家。请为基础数据清洗方案生成详细的执行步骤。"),
            HumanMessage(content=f"数据描述：{state['data_description']}\n分析结果：{state['analysis']}\n\n请生成基础清洗方案的详细步骤。")
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "result": response.content}

    def process_b_node(state: DataState) -> dict:
        """处理方式B：深度分析"""
        prompt = [
            SystemMessage(content="你是深度数据分析专家。请为深度分析方案生成详细的执行步骤。"),
            HumanMessage(content=f"数据描述：{state['data_description']}\n分析结果：{state['analysis']}\n\n请生成深度分析方案的详细步骤。")
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "result": response.content}

    def process_c_node(state: DataState) -> dict:
        """处理方式C：自动化处理"""
        prompt = [
            SystemMessage(content="你是数据自动化处理专家。请为自动化处理方案生成详细的执行步骤。"),
            HumanMessage(content=f"数据描述：{state['data_description']}\n分析结果：{state['analysis']}\n\n请生成自动化处理方案的详细步骤。")
        ]
        response = llm.invoke(prompt)
        return {"messages": [response], "result": response.content}

    def summarize_node(state: DataState) -> dict:
        """汇总节点"""
        return {
            "messages": [AIMessage(content=(
                f"✅ 数据处理方案已生成！\n\n"
                f"选择的方式：{state['chosen_method']}\n\n"
                f"处理方案：\n{state['result']}"
            ))]
        }

    # 构建图
    workflow = StateGraph(DataState)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("process_a", process_a_node)
    workflow.add_node("process_b", process_b_node)
    workflow.add_node("process_c", process_c_node)
    workflow.add_node("summarize", summarize_node)

    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "decision")
    workflow.add_conditional_edges("decision", should_process, {
        "process_a": "process_a",
        "process_b": "process_b",
        "process_c": "process_c",
    })
    workflow.add_edge("process_a", "summarize")
    workflow.add_edge("process_b", "summarize")
    workflow.add_edge("process_c", "summarize")
    workflow.add_edge("summarize", END)

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    print("\n【交互式数据处理】")
    print("描述你的数据，AI 分析后让你选择处理方式")
    print("\n试试说：")
    print("  • '我有一份销售数据CSV，包含日期、产品、金额，有些缺失值'")
    print("  • '一批用户行为日志数据，格式不统一，需要整理'")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("数据描述：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效内容")
            continue

        try:
            thread_id = f"data_{id(user_input)}"
            config = {"configurable": {"thread_id": thread_id}}

            print("\n⏳ AI 正在分析数据...")
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "data_description": "", "analysis": "",
                    "options": [], "chosen_method": "", "result": "",
                },
                config=config,
            )

            # 获取中断信息
            state = graph.get_state(config)
            if state.next:
                interrupt_info = state.tasks[0].interrupts[0].value if state.tasks else {}
                print(f"\n{'='*60}")
                print("📊 数据分析结果：")
                print(f"{'='*60}")
                print(interrupt_info.get("analysis", ""))
                print(f"{'='*60}")

                print("\n请选择处理方式：")
                print("  A. 基础清洗（快速处理缺失值和异常值）")
                print("  B. 深度分析（全面分析并挖掘数据洞察）")
                print("  C. 自动化处理（构建自动化数据处理管道）")

                choice = input("你的选择 (A/B/C)：").strip().upper()

                if choice in ["A", "B", "C"]:
                    print(f"\n⏳ 正在生成{choice}方案的详细步骤...")
                    graph.invoke(
                        Command(resume={"method": choice}),
                        config=config,
                    )
                    # 获取最终结果
                    final_state = graph.get_state(config)
                    for msg in reversed(final_state.values.get("messages", [])):
                        if isinstance(msg, AIMessage) and msg.content:
                            print(f"\n{msg.content}")
                            break
                else:
                    print("❌ 无效选择")
                    graph.invoke(Command(resume={"method": "A"}), config=config)

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 核心概念总结：")
    print("   1. interrupt() 可以传入多个选项，让用户做决策")
    print("   2. Command(resume={method: 'A/B/C'}) 传入用户选择")
    print("   3. 路由函数根据选择分发到不同的处理节点")
    print("   4. 所有处理路径最终汇聚到汇总节点，统一输出")


# ============================================================
# 示例4: 断点恢复 - 长任务管理
# ============================================================

def demo_checkpoint_recovery():
    """断点恢复 - AI 分步执行复杂任务，每步暂停等待继续指令

    核心概念：
    - 分步执行: 将长任务拆分为多个步骤，每步完成后暂停
    - checkpointer: 自动保存每步状态，支持从任意步骤恢复
    - 断点恢复: 工作流被中断后可以从上次的位置继续，不必从头开始
    - thread_id: 每个对话线程有独立的检查点，互不干扰
    """
    print("\n" + "=" * 60)
    print("示例4：断点恢复 - 长任务管理")
    print("=" * 60)
    print("\n💡 核心概念：")
    print("   - 分步执行: 将长任务拆分为多步，每步完成后暂停")
    print("   - checkpointer: 自动保存状态，支持从任意步骤恢复")
    print("   - 断点恢复: 中断后从上次位置继续，不必从头开始")
    print("   - thread_id: 每个对话线程有独立的检查点")

    llm = get_default_llm()

    class TaskState(TypedDict):
        messages: Annotated[list, add_messages]
        task: str            # 任务描述
        step: int            # 当前步骤
        total_steps: int     # 总步骤数
        step_results: list   # 各步骤结果
        completed: bool      # 是否完成

    TOTAL_STEPS = 3

    def plan_node(state: TaskState) -> dict:
        """规划节点：拆分任务为步骤"""
        user_task = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_task = msg.content
                break

        prompt = [
            SystemMessage(content=(
                f"你是一个任务规划专家。请将用户的任务拆分为 {TOTAL_STEPS} 个步骤，"
                "每个步骤需要简明扼要地描述要做什么。"
                f"输出 {TOTAL_STEPS} 行，每行一个步骤。"
            )),
            HumanMessage(content=f"请将以下任务拆分为{TOTAL_STEPS}个步骤：\n{user_task}")
        ]
        response = llm.invoke(prompt)
        return {
            "messages": [response],
            "task": user_task,
            "step": 0,
            "total_steps": TOTAL_STEPS,
            "step_results": [],
            "completed": False,
        }

    def execute_step_node(state: TaskState) -> dict:
        """执行单步节点：执行当前步骤的任务"""
        current_step = state["step"] + 1
        prompt = [
            SystemMessage(content=(
                "你是一个任务执行专家。请执行指定的任务步骤，给出详细的结果。"
            )),
            HumanMessage(content=(
                f"原始任务：{state['task']}\n\n"
                f"之前步骤的结果：\n{chr(10).join(state['step_results']) if state['step_results'] else '无'}\n\n"
                f"请执行第 {current_step}/{state['total_steps']} 步。"
            ))
        ]
        response = llm.invoke(prompt)
        new_results = state["step_results"] + [f"步骤{current_step}：{response.content}"]
        return {
            "messages": [response],
            "step": current_step,
            "step_results": new_results,
        }

    def checkpoint_node(state: TaskState) -> dict:
        """检查点节点：每步完成后暂停，等待用户确认继续"""
        # 判断是否还有后续步骤
        has_more = state["step"] < state["total_steps"]

        if has_more:
            decision = interrupt({
                "type": "step_checkpoint",
                "current_step": state["step"],
                "total_steps": state["total_steps"],
                "step_result": state["step_results"][-1] if state["step_results"] else "",
                "message": f"第 {state['step']}/{state['total_steps']} 步已完成，是否继续？",
            })
            return {"completed": False}
        else:
            return {"completed": True}

    def should_continue_task(state: TaskState) -> str:
        """路由函数：判断是否继续执行"""
        if state["completed"] or state["step"] >= state["total_steps"]:
            return "finish"
        return "execute_step"

    def finish_node(state: TaskState) -> dict:
        """完成节点：汇总所有步骤结果"""
        results_text = "\n\n".join(state["step_results"])
        return {
            "messages": [AIMessage(content=(
                f"🎉 任务完成！共执行 {state['total_steps']} 步\n\n"
                f"{'='*60}\n"
                f"{results_text}\n"
                f"{'='*60}"
            ))]
        }

    # 构建图
    workflow = StateGraph(TaskState)
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute_step", execute_step_node)
    workflow.add_node("checkpoint", checkpoint_node)
    workflow.add_node("finish", finish_node)

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "execute_step")
    workflow.add_edge("execute_step", "checkpoint")
    workflow.add_conditional_edges("checkpoint", should_continue_task, {
        "execute_step": "execute_step",
        "finish": "finish",
    })
    workflow.add_edge("finish", END)

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    print("\n【交互式长任务管理】")
    print("输入复杂任务，AI 会分步执行，每步完成后暂停等待你的指令")
    print("\n试试说：")
    print("  • '帮我策划一场产品发布会'")
    print("  • '制定一个学习Python的3个月计划'")
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
            thread_id = f"task_{id(user_input)}"
            config = {"configurable": {"thread_id": thread_id}}

            # 第一轮：规划和执行第一步
            print("\n⏳ AI 正在规划任务...")
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "task": "", "step": 0, "total_steps": TOTAL_STEPS,
                    "step_results": [], "completed": False,
                },
                config=config,
            )

            # 循环处理检查点
            while True:
                state = graph.get_state(config)
                if not state.next:
                    # 工作流已完成
                    for msg in reversed(state.values.get("messages", [])):
                        if isinstance(msg, AIMessage) and "任务完成" in msg.content:
                            print(f"\n{msg.content}")
                            break
                    break

                # 获取中断信息
                interrupt_info = state.tasks[0].interrupts[0].value if state.tasks else {}
                current_step = interrupt_info.get("current_step", 0)
                total_steps = interrupt_info.get("total_steps", 0)
                step_result = interrupt_info.get("step_result", "")

                print(f"\n{'='*60}")
                print(f"📍 第 {current_step}/{total_steps} 步完成")
                print(f"{'='*60}")
                # 截取步骤结果
                display_result = step_result[:300] + "..." if len(step_result) > 300 else step_result
                print(display_result)
                print(f"{'='*60}")

                print("\n请选择：")
                print("  1. ▶️  继续执行下一步")
                print("  2. ⏸️  暂停（状态已保存，下次可恢复）")
                print("  3. 🛑  终止任务")

                choice = input("你的选择 (1-3)：").strip()

                if choice == "1":
                    print(f"\n⏳ 正在执行第 {current_step + 1} 步...")
                    graph.invoke(Command(resume={"continue": True}), config=config)
                elif choice == "2":
                    print("\n⏸️ 任务已暂停，状态已保存。")
                    print(f"📌 Thread ID: {thread_id}")
                    print("💡 提示：在真实应用中，你可以稍后用相同的 thread_id 恢复任务")
                    break
                elif choice == "3":
                    print("\n🛑 任务已终止")
                    break
                else:
                    print("❌ 无效选择")

        except Exception as e:
            print(f"❌ 错误：{e}")

        print("\n" + "-" * 60)

    print("\n✅ 核心概念总结：")
    print("   1. 分步执行 + interrupt()：长任务拆分为多步，每步暂停等待指令")
    print("   2. checkpointer 自动保存状态，中断后可恢复")
    print("   3. thread_id 标识不同的对话线程，每个线程有独立的检查点")
    print("   4. 断点恢复：用相同的 thread_id 调用 graph.invoke() 即可从断点继续")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "=" * 60)
    print("  LangGraph 人工介入（Human-in-the-Loop）- 实战案例")
    print("=" * 60)
    print("\n本示例演示 LangGraph 中四种人工介入模式")
    print("\n核心概念：")
    print("  • interrupt(): 暂停工作流，等待人工介入")
    print("  • Command(resume=...): 恢复工作流")
    print("  • checkpointer: 检查点，保存工作流状态")
    print("  • 断点恢复: 从中断的地方继续执行")
    print("\n应用场景：")
    print("  • 内容审批、邮件审批、数据处理、长任务管理")

    while True:
        print("\n" + "=" * 60)
        print("请选择要运行的示例：")
        print("=" * 60)
        print("  1. interrupt() 函数 - 内容审批（AI生成→人工审批→发布/修订）")
        print("  2. 人工审批流程 - 邮件发送（撰写→预览→确认发送）")
        print("  3. 交互式决策 - 数据处理（分析→选择方式→处理→汇总）")
        print("  4. 断点恢复 - 长任务管理（分步执行→每步暂停→可恢复）")
        print("\n  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_content_approval()
        elif choice == "2":
            demo_email_approval()
        elif choice == "3":
            demo_interactive_decision()
        elif choice == "4":
            demo_checkpoint_recovery()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
