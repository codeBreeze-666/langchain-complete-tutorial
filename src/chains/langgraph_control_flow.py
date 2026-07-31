"""
LangGraph 控制流案例 - 条件边、循环逻辑、Send动态扇出、子图嵌套
================================================================

本示例演示 LangGraph 中高级控制流模式，包含四个交互式案例。

核心概念：
- add_conditional_edges: 条件边，根据状态动态选择下一个节点
  是 LangGraph 中实现分支逻辑的核心 API，让工作流可以根据运行时状态
  动态选择不同的执行路径，而非固定顺序

- Send: 动态扇出，一个节点向多个节点发送数据
  实现类似 Map-Reduce 的并行处理模式：一个节点根据状态动态决定
  要向哪些节点发送数据，每个接收节点独立处理自己的数据

- 子图(Subgraph): 图嵌套，将复杂工作流拆分为多个子图
  将大图拆分为多个小图，每个子图有独立的状态和逻辑，
  子图可以像普通节点一样被添加到父图中

- Command: 新版API，用于在节点中返回指令
  可以在节点返回值中同时指定状态更新和下一个要执行的节点，
  是一种更灵活的控制流写法

应用场景：
- 智能路由：根据问题类型路由到不同专家
- 代码审查：多轮迭代审查直到通过
- 多维度分析：并行分析多个维度后汇总
- 复杂工作流：子流程自动处理嵌套任务
"""

import os
import sys
import operator
from typing import TypedDict, Annotated

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.utils.llm_loader import get_default_llm


# ============================================================
# 示例1: 条件边 - 智能路由
# ============================================================

class RoutingState(TypedDict):
    """智能路由状态"""
    question: str              # 用户问题
    expert_type: str           # 专家类型
    answer: str                # 回答
    expert_name: str           # 专家名称


def classify_expert(state: RoutingState) -> dict:
    """分类节点：判断问题应该由哪位专家回答"""
    llm = get_default_llm()
    question = state["question"]
    response = llm.invoke(
        f"请判断以下问题应该由哪位专家回答，只输出专家类型：\n"
        f"- 医学专家：涉及健康、疾病、药物、养生\n"
        f"- 法律专家：涉及法律、法规、合同、权益\n"
        f"- 技术专家：涉及编程、软件、硬件、网络\n"
        f"- 教育专家：涉及学习、考试、升学、培训\n\n"
        f"问题：{question}\n\n只输出专家类型。"
    )
    expert_type = response.content.strip()
    valid_types = ["医学专家", "法律专家", "技术专家", "教育专家"]
    for vt in valid_types:
        if vt in expert_type:
            expert_type = vt
            break
    else:
        expert_type = "技术专家"
    print(f"  [分类] 路由到：{expert_type}")
    return {"expert_type": expert_type}


def medical_expert(state: RoutingState) -> dict:
    """医学专家节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"你是一位资深医学专家。请用专业但易懂的方式回答：\n{state['question']}\n"
        f"注意：请提醒用户此建议仅供参考，具体诊疗请咨询医生。"
    )
    print(f"  [医学专家] 已生成回答")
    return {"answer": response.content, "expert_name": "医学专家"}


def legal_expert(state: RoutingState) -> dict:
    """法律专家节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"你是一位资深法律顾问。请用专业但易懂的方式回答：\n{state['question']}\n"
        f"注意：请提醒用户此建议仅供参考，具体法律问题请咨询律师。"
    )
    print(f"  [法律专家] 已生成回答")
    return {"answer": response.content, "expert_name": "法律专家"}


def tech_expert(state: RoutingState) -> dict:
    """技术专家节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"你是一位资深技术专家。请用专业但易懂的方式回答：\n{state['question']}\n"
        f"请提供代码示例或具体操作步骤。"
    )
    print(f"  [技术专家] 已生成回答")
    return {"answer": response.content, "expert_name": "技术专家"}


def education_expert(state: RoutingState) -> dict:
    """教育专家节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"你是一位资深教育顾问。请用专业但易懂的方式回答：\n{state['question']}\n"
        f"请提供具体的学习建议和资源推荐。"
    )
    print(f"  [教育专家] 已生成回答")
    return {"answer": response.content, "expert_name": "教育专家"}


def route_to_expert(state: RoutingState) -> str:
    """路由函数：根据专家类型路由到对应节点"""
    route_map = {
        "医学专家": "medical",
        "法律专家": "legal",
        "技术专家": "tech",
        "教育专家": "education",
    }
    return route_map.get(state.get("expert_type", "技术专家"), "tech")


def demo_smart_routing():
    """示例1：条件边 - 智能路由

    实战要点：
    - add_conditional_edges 实现动态路由
    - 路由函数根据状态字段返回目标节点名称
    - 映射表定义返回值与节点的对应关系
    - 比硬编码 if-else 更灵活，图结构更清晰
    """
    print("\n" + "=" * 60)
    print("示例1：条件边 - 智能路由")
    print("=" * 60)
    print("""
核心概念：
  add_conditional_edges: 条件边
  - 根据状态动态选择下一个节点
  - 路由函数：接收状态，返回目标节点名
  - 映射表：定义返回值与节点的对应关系

代码示例：
  graph.add_conditional_edges(
      "classify",           # 源节点
      route_to_expert,      # 路由函数
      {                     # 映射表
          "medical": "medical",
          "legal": "legal",
      }
  )
    """)

    # 构建状态图
    graph = StateGraph(RoutingState)

    # 添加节点
    graph.add_node("classify", classify_expert)
    graph.add_node("medical", medical_expert)
    graph.add_node("legal", legal_expert)
    graph.add_node("tech", tech_expert)
    graph.add_node("education", education_expert)

    # 添加边
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_to_expert,
        {
            "medical": "medical",
            "legal": "legal",
            "tech": "tech",
            "education": "education",
        }
    )
    graph.add_edge("medical", END)
    graph.add_edge("legal", END)
    graph.add_edge("tech", END)
    graph.add_edge("education", END)

    # 编译图
    app = graph.compile()

    print("【交互式智能路由】")
    print("输入问题，自动路由到对应专家回答")
    print("\n支持专家：医学、法律、技术、教育")
    print("\n输入 '退出' 结束\n")

    while True:
        question = input("请输入你的问题：").strip()
        if question.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if not question:
            print("请输入问题")
            continue

        try:
            print("\n" + "─" * 40)
            result = app.invoke({"question": question, "expert_type": "", "answer": "", "expert_name": ""})
            print("─" * 40)
            print(f"\n路由到：{result['expert_name']}")
            print(f"\n回答：\n{result['answer']}")
        except Exception as e:
            print(f"错误：{e}")

        print("\n" + "-" * 60)

    print("\n实战要点总结：")
    print("   1. add_conditional_edges 实现动态路由，比硬编码 if-else 更灵活")
    print("   2. 路由函数根据状态字段返回目标节点名称")
    print("   3. 映射表定义返回值与节点的对应关系，图结构清晰")


# ============================================================
# 示例2: 循环逻辑 - 代码审查
# ============================================================

class CodeReviewState(TypedDict):
    """代码审查状态"""
    code: str                           # 用户输入的代码
    review_round: int                   # 当前审查轮次
    max_rounds: int                     # 最大审查轮次
    review_result: str                  # 审查结果
    review_passed: bool                 # 是否通过审查
    feedback: str                       # 审查反馈
    messages: Annotated[list, operator.add]  # 消息历史（追加更新）


def review_code(state: CodeReviewState) -> dict:
    """代码审查节点：审查代码并给出反馈"""
    llm = get_default_llm()
    code = state["code"]
    review_round = state.get("review_round", 0) + 1
    previous_feedback = state.get("feedback", "")

    # 构建审查提示
    prompt = f"你是一位严格的代码审查专家。请审查以下代码：\n\n{code}\n\n"
    if previous_feedback:
        prompt += f"【上一轮审查反馈】\n{previous_feedback}\n\n"
    prompt += (
        "请从以下维度审查：\n"
        "1. 代码正确性：是否有逻辑错误\n"
        "2. 代码风格：命名规范、缩进、注释\n"
        "3. 安全性：是否有潜在安全风险\n"
        "4. 性能：是否有明显性能问题\n\n"
        "最后给出结论：通过 或 不通过，并说明原因。"
    )

    response = llm.invoke(prompt)
    result = response.content

    # 判断是否通过
    passed = "通过" in result and "不通过" not in result

    print(f"  [第{review_round}轮审查] {'通过' if passed else '不通过'}")

    return {
        "review_round": review_round,
        "review_result": result,
        "review_passed": passed,
        "feedback": result,
        "messages": [f"[第{review_round}轮] {'通过' if passed else '不通过'}"],
    }


def should_continue(state: CodeReviewState) -> str:
    """判断是否继续审查"""
    if state.get("review_passed", False):
        return "approved"
    if state.get("review_round", 0) >= state.get("max_rounds", 3):
        return "max_rounds"
    return "review"


def approved(state: CodeReviewState) -> dict:
    """审查通过节点"""
    print(f"  [审查通过] 代码已通过审查！")
    return {"messages": ["[最终] 代码审查通过"]}


def max_rounds_reached(state: CodeReviewState) -> dict:
    """达到最大轮次节点"""
    print(f"  [最大轮次] 已达到最大审查轮次，需要人工检查")
    return {"messages": ["[最终] 达到最大审查轮次"]}


def demo_code_review():
    """示例2：循环逻辑 - 代码审查

    实战要点：
    - 条件边可以实现循环：节点输出 → 条件判断 → 回到同一节点
    - 循环需要退出条件：审查通过 或 达到最大轮次
    - 每次循环将上一次的反馈作为输入，实现迭代优化
    - messages 用 operator.add 追加，记录审查历史
    """
    print("\n" + "=" * 60)
    print("示例2：循环逻辑 - 代码审查")
    print("=" * 60)
    print("""
核心概念：
  循环逻辑：通过条件边实现节点循环
  - 条件边可以指向源节点自身，形成循环
  - 必须有退出条件，防止无限循环
  - 每次循环将上一次的反馈作为输入

代码示例：
  graph.add_conditional_edges(
      "review",             # 源节点
      should_continue,      # 路由函数
      {
          "review": "review",   # 循环：回到自身
          "approved": "approved",  # 退出：通过
          "max_rounds": "max_rounds",  # 退出：最大轮次
      }
  )
    """)

    # 构建状态图
    graph = StateGraph(CodeReviewState)

    # 添加节点
    graph.add_node("review", review_code)
    graph.add_node("approved", approved)
    graph.add_node("max_rounds", max_rounds_reached)

    # 添加边
    graph.add_edge(START, "review")

    # 条件边：实现循环
    graph.add_conditional_edges(
        "review",
        should_continue,
        {
            "review": "review",       # 继续审查（循环）
            "approved": "approved",   # 审查通过
            "max_rounds": "max_rounds",  # 达到最大轮次
        }
    )

    graph.add_edge("approved", END)
    graph.add_edge("max_rounds", END)

    # 编译图
    app = graph.compile()

    print("【交互式代码审查】")
    print("输入代码，自动多轮审查直到通过")
    print("\n输入 '退出' 结束\n")

    while True:
        code = input("请输入要审查的代码：").strip()
        if code.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if not code:
            print("请输入代码")
            continue

        max_rounds_input = input("最大审查轮次（默认3）：").strip()
        try:
            max_rounds = int(max_rounds_input) if max_rounds_input else 3
        except ValueError:
            max_rounds = 3

        try:
            print("\n" + "─" * 40)
            print("开始代码审查...")
            result = app.invoke({
                "code": code,
                "review_round": 0,
                "max_rounds": max_rounds,
                "review_result": "",
                "review_passed": False,
                "feedback": "",
                "messages": [],
            })
            print("─" * 40)
            print(f"\n审查轮次: {result['review_round']}")
            print(f"最终结果: {'通过' if result['review_passed'] else '未通过'}")
            print(f"\n审查详情：\n{result['review_result']}")
        except Exception as e:
            print(f"错误：{e}")

        print("\n" + "-" * 60)

    print("\n实战要点总结：")
    print("   1. 条件边可以指向源节点自身，形成循环")
    print("   2. 必须有退出条件（通过/最大轮次），防止无限循环")
    print("   3. 每次循环将上一次的反馈作为输入，实现迭代优化")


# ============================================================
# 示例3: Send动态扇出 - 多维度分析
# ============================================================

class AnalysisState(TypedDict):
    """多维度分析状态（主图）"""
    topic: str                    # 分析主题
    dimensions: list              # 分析维度列表
    analysis_results: Annotated[list, operator.add]  # 分析结果（追加更新）
    final_report: str             # 最终报告


class DimensionState(TypedDict):
    """单维度分析状态（子节点）"""
    topic: str                    # 分析主题
    dimension: str                # 分析维度
    analysis: str                 # 分析结果


def determine_dimensions(state: AnalysisState) -> dict:
    """确定分析维度节点：根据主题决定分析维度"""
    llm = get_default_llm()
    topic = state["topic"]
    response = llm.invoke(
        f"用户想要分析主题：{topic}。\n"
        f"请确定3个分析维度，每个维度用2-4个字描述，用逗号分隔。\n"
        f"例如：技术前景,市场机会,风险评估\n\n"
        f"只输出3个维度，用逗号分隔。"
    )
    dimensions = [d.strip() for d in response.content.split(",") if d.strip()]
    # 确保至少有3个维度
    while len(dimensions) < 3:
        dimensions.append(f"维度{len(dimensions) + 1}")

    print(f"  [维度确定] 分析维度: {dimensions}")
    return {"dimensions": dimensions[:3]}


def analyze_dimension(state: DimensionState) -> dict:
    """单维度分析节点：分析指定维度"""
    llm = get_default_llm()
    topic = state["topic"]
    dimension = state["dimension"]

    response = llm.invoke(
        f"你是一位专业分析师。请从「{dimension}」维度分析以下主题：\n"
        f"主题：{topic}\n\n"
        f"请提供：\n"
        f"1. 该维度下的关键发现\n"
        f"2. 数据支撑或事实依据\n"
        f"3. 该维度的建议\n"
        f"请用简洁的段落回答。"
    )
    print(f"  [{dimension}] 分析完成")
    return {"analysis": f"【{dimension}】\n{response.content}"}


def continue_to_dimensions(state: AnalysisState) -> list:
    """动态扇出函数：根据维度列表发送到多个分析节点

    Send 的作用：
    - 根据运行时状态动态决定要向哪些节点发送数据
    - 每个接收节点独立处理自己的数据
    - 类似 Map-Reduce 中的 Map 阶段
    """
    dimensions = state.get("dimensions", [])
    return [
        Send("analyze", {"topic": state["topic"], "dimension": dim, "analysis": ""})
        for dim in dimensions
    ]


def generate_report(state: AnalysisState) -> dict:
    """生成报告节点：汇总所有维度分析结果"""
    llm = get_default_llm()
    results = state.get("analysis_results", [])
    topic = state["topic"]

    # 合并所有维度分析结果
    combined = "\n\n".join(results)

    response = llm.invoke(
        f"请将以下多维度分析结果汇总为一份完整的分析报告：\n\n"
        f"主题：{topic}\n\n"
        f"{combined}\n\n"
        f"请添加：\n1. 总体评价\n2. 核心发现\n3. 行动建议"
    )
    print(f"  [报告生成] 已生成最终报告")
    return {"final_report": response.content}


def demo_multi_dimension_analysis():
    """示例3：Send动态扇出 - 多维度分析

    实战要点：
    - Send 实现动态扇出，一个节点向多个节点发送数据
    - 类似 Map-Reduce 模式：Map(并行分析) → Reduce(汇总报告)
    - 维度数量在运行时动态决定，不是固定的
    - 每个维度的分析独立进行，最后汇总
    - analysis_results 使用 operator.add 自动追加各维度结果
    """
    print("\n" + "=" * 60)
    print("示例3：Send动态扇出 - 多维度分析")
    print("=" * 60)
    print("""
核心概念：
  Send: 动态扇出，一个节点向多个节点发送数据
  - 实现类似 Map-Reduce 的并行处理模式
  - 维度数量在运行时动态决定
  - 每个接收节点独立处理自己的数据

代码示例：
  def continue_to_dimensions(state):
      return [
          Send("analyze", {"topic": state["topic"], "dimension": dim})
          for dim in state["dimensions"]
      ]

  graph.add_conditional_edges("determine", continue_to_dimensions)

流程：
  确定维度 → 并行分析(多个Send) → 汇总报告
    """)

    # 构建状态图
    graph = StateGraph(AnalysisState)

    # 添加节点
    graph.add_node("determine", determine_dimensions)
    graph.add_node("analyze", analyze_dimension)
    graph.add_node("report", generate_report)

    # 添加边
    graph.add_edge(START, "determine")

    # 使用 Send 实现动态扇出
    graph.add_conditional_edges("determine", continue_to_dimensions, ["analyze"])

    # 所有分析完成后汇总
    graph.add_edge("analyze", "report")
    graph.add_edge("report", END)

    # 编译图
    app = graph.compile()

    print("【交互式多维度分析】")
    print("输入主题，自动确定分析维度并并行分析")
    print("\n输入 '退出' 结束\n")

    while True:
        topic = input("请输入要分析的主题：").strip()
        if topic.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if not topic:
            print("请输入主题")
            continue

        try:
            print("\n" + "─" * 40)
            print("正在分析...")
            result = app.invoke({
                "topic": topic,
                "dimensions": [],
                "analysis_results": [],
                "final_report": "",
            })
            print("─" * 40)
            print(f"\n分析维度: {result['dimensions']}")
            print(f"\n最终报告：\n{result['final_report']}")
        except Exception as e:
            print(f"错误：{e}")

        print("\n" + "-" * 60)

    print("\n实战要点总结：")
    print("   1. Send 实现动态扇出，维度数量运行时决定")
    print("   2. 类似 Map-Reduce：并行分析 → 汇总报告")
    print("   3. operator.add 自动追加各维度结果")


# ============================================================
# 示例4: 子图嵌套 - 复杂工作流
# ============================================================

# --- 子图1: 需求分析子图 ---

class RequirementState(TypedDict):
    """需求分析状态"""
    requirement: str              # 用户需求
    analysis: str                 # 分析结果
    priority: str                 # 优先级
    feasibility: str              # 可行性评估


def analyze_requirement(state: RequirementState) -> dict:
    """分析需求节点"""
    llm = get_default_llm()
    requirement = state["requirement"]
    response = llm.invoke(
        f"你是一位需求分析师。请分析以下需求：\n{requirement}\n\n"
        f"请提供：\n1. 需求拆解（拆分为3-5个子需求）\n2. 优先级评估\n3. 依赖关系分析"
    )
    print(f"  [需求分析] 已完成")
    return {"analysis": response.content}


def evaluate_feasibility(state: RequirementState) -> dict:
    """评估可行性节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"根据以下需求分析，评估可行性：\n{state['analysis']}\n\n"
        f"请评估：\n1. 技术可行性（高/中/低）\n2. 资源需求\n3. 风险点\n4. 建议的实现路径"
    )
    print(f"  [可行性评估] 已完成")

    # 判断优先级
    priority = "高" if "紧急" in state.get("requirement", "") or "重要" in state.get("requirement", "") else "中"

    return {"feasibility": response.content, "priority": priority}


def build_requirement_subgraph() -> StateGraph:
    """构建需求分析子图"""
    subgraph = StateGraph(RequirementState)

    subgraph.add_node("analyze", analyze_requirement)
    subgraph.add_node("evaluate", evaluate_feasibility)

    subgraph.add_edge(START, "analyze")
    subgraph.add_edge("analyze", "evaluate")
    subgraph.add_edge("evaluate", END)

    return subgraph


# --- 子图2: 方案设计子图 ---

class DesignState(TypedDict):
    """方案设计状态"""
    requirement: str              # 用户需求
    analysis: str                 # 需求分析结果
    design: str                   # 设计方案
    tech_stack: str               # 技术栈建议


def design_solution(state: DesignState) -> dict:
    """设计方案节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"你是一位架构师。根据以下需求分析，设计技术方案：\n"
        f"需求：{state['requirement']}\n"
        f"分析：{state.get('analysis', '无')}\n\n"
        f"请提供：\n1. 系统架构设计\n2. 模块划分\n3. 接口设计\n4. 数据流设计"
    )
    print(f"  [方案设计] 已完成")
    return {"design": response.content}


def suggest_tech_stack(state: DesignState) -> dict:
    """推荐技术栈节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"根据以下设计方案，推荐技术栈：\n{state['design']}\n\n"
        f"请推荐：\n1. 前端技术栈\n2. 后端技术栈\n3. 数据库\n4. 部署方案\n5. 推荐理由"
    )
    print(f"  [技术栈推荐] 已完成")
    return {"tech_stack": response.content}


def build_design_subgraph() -> StateGraph:
    """构建方案设计子图"""
    subgraph = StateGraph(DesignState)

    subgraph.add_node("design", design_solution)
    subgraph.add_node("tech_stack", suggest_tech_stack)

    subgraph.add_edge(START, "design")
    subgraph.add_edge("design", "tech_stack")
    subgraph.add_edge("tech_stack", END)

    return subgraph


# --- 主图: 整合子图 ---

class ProjectState(TypedDict):
    """项目工作流状态"""
    requirement: str              # 用户需求
    analysis: str                 # 需求分析结果
    feasibility: str              # 可行性评估
    priority: str                 # 优先级
    design: str                   # 设计方案
    tech_stack: str               # 技术栈建议
    final_report: str             # 最终报告


def generate_project_report(state: ProjectState) -> dict:
    """生成项目报告节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"请将以下内容整合为一份项目规划报告：\n\n"
        f"【需求分析】\n{state.get('analysis', '无')}\n\n"
        f"【可行性评估】\n{state.get('feasibility', '无')}\n\n"
        f"【设计方案】\n{state.get('design', '无')}\n\n"
        f"【技术栈建议】\n{state.get('tech_stack', '无')}\n\n"
        f"请添加项目概述、时间线规划和下一步行动。"
    )
    print(f"  [报告生成] 已完成")
    return {"final_report": response.content}


def demo_subgraph_workflow():
    """示例4：子图嵌套 - 复杂工作流

    实战要点：
    - 子图可以像普通节点一样被添加到父图
    - 子图有独立的状态和逻辑，降低复杂度
    - 子图编译后通过 add_node 添加到父图
    - 适合将复杂工作流拆分为多个可复用的子模块
    """
    print("\n" + "=" * 60)
    print("示例4：子图嵌套 - 复杂工作流")
    print("=" * 60)
    print("""
核心概念：
  子图(Subgraph): 图嵌套，将复杂工作流拆分为多个子图
  - 子图有独立的状态和逻辑
  - 子图编译后像普通节点一样被添加到父图
  - 适合将复杂工作流拆分为可复用的子模块

代码示例：
  # 构建子图
  subgraph = StateGraph(SubState)
  subgraph.add_node(...)
  subgraph.compile()

  # 在父图中使用子图
  parent_graph.add_node("requirement", compiled_subgraph)

流程：
  需求分析(子图) → 方案设计(子图) → 生成报告
    """)

    # 构建子图
    requirement_subgraph = build_requirement_subgraph().compile()
    design_subgraph = build_design_subgraph().compile()

    # 构建主图
    graph = StateGraph(ProjectState)

    # 添加子图作为节点
    graph.add_node("requirement", requirement_subgraph)
    graph.add_node("design", design_subgraph)
    graph.add_node("report", generate_project_report)

    # 添加边
    graph.add_edge(START, "requirement")
    graph.add_edge("requirement", "design")
    graph.add_edge("design", "report")
    graph.add_edge("report", END)

    # 编译主图
    app = graph.compile()

    print("【交互式项目工作流】")
    print("输入需求，自动经过需求分析、方案设计，生成项目报告")
    print("\n输入 '退出' 结束\n")

    while True:
        requirement = input("请输入你的项目需求：").strip()
        if requirement.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if not requirement:
            print("请输入需求")
            continue

        try:
            print("\n" + "─" * 40)
            print("开始项目工作流...")
            result = app.invoke({
                "requirement": requirement,
                "analysis": "",
                "feasibility": "",
                "priority": "",
                "design": "",
                "tech_stack": "",
                "final_report": "",
            })
            print("─" * 40)
            print(f"\n项目优先级: {result.get('priority', '中')}")
            print(f"\n最终报告：\n{result['final_report']}")
        except Exception as e:
            print(f"错误：{e}")

        print("\n" + "-" * 60)

    print("\n实战要点总结：")
    print("   1. 子图有独立的状态和逻辑，降低复杂度")
    print("   2. 子图编译后像普通节点一样被添加到父图")
    print("   3. 适合将复杂工作流拆分为可复用的子模块")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "=" * 60)
    print("  LangGraph 控制流案例 - 条件边、循环、Send扇出、子图嵌套")
    print("=" * 60)
    print("\n核心概念：")
    print("  • add_conditional_edges: 条件边，根据状态动态选择下一个节点")
    print("  • Send: 动态扇出，一个节点向多个节点发送数据")
    print("  • 子图(Subgraph): 图嵌套，将复杂工作流拆分为多个子图")
    print("  • Command: 新版API，用于在节点中返回指令")

    while True:
        print("\n" + "=" * 60)
        print("请选择要运行的示例：")
        print("=" * 60)
        print("  1. 条件边 - 智能路由（问题分类→路由到不同专家）")
        print("  2. 循环逻辑 - 代码审查（多轮审查直到通过）")
        print("  3. Send动态扇出 - 多维度分析（并行分析多个维度）")
        print("  4. 子图嵌套 - 复杂工作流（子流程自动处理）")
        print("\n  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_smart_routing()
        elif choice == "2":
            demo_code_review()
        elif choice == "3":
            demo_multi_dimension_analysis()
        elif choice == "4":
            demo_subgraph_workflow()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
