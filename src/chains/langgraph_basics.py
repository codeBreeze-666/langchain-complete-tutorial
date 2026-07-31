"""
LangGraph 基础案例 - 状态图、节点与边、条件边、完整工作流
========================================================

本示例演示 LangGraph 的核心概念和基础用法，包含四个交互式案例。

核心概念：
- StateGraph: 状态图，工作流的核心载体。通过定义状态类型和节点函数来构建有向图
- State: 状态，存储工作流运行过程中的数据，使用 TypedDict 定义类型
- Node: 节点，执行具体逻辑的单元，接收状态、返回状态更新
- Edge: 边，连接节点、定义流程走向，包括普通边和条件边
- START/END: 特殊节点，标记流程的起点和终点

应用场景：
- 订单处理：多步骤业务流程自动化
- 学习助手：根据问题类型智能路由
- 智能客服：根据用户意图分发到不同处理模块
- 旅行规划：多阶段编排复杂任务
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from src.utils.llm_loader import get_default_llm


# ============================================================
# 示例1: 简单状态图 - 订单处理流程
# ============================================================

class OrderState(TypedDict):
    """订单状态定义"""
    order_info: str          # 用户输入的订单信息
    current_step: str        # 当前步骤
    confirm_result: str      # 确认结果
    ship_result: str         # 发货结果
    final_result: str        # 最终结果


def order_place(state: OrderState) -> dict:
    """下单节点：处理用户订单信息"""
    llm = get_default_llm()
    order_info = state["order_info"]
    response = llm.invoke(
        f"用户下单了：{order_info}。请用一句话确认订单内容，包括商品名称和数量。"
    )
    print(f"\n  [下单] {response.content}")
    return {"current_step": "confirmed", "confirm_result": response.content}


def order_confirm(state: OrderState) -> dict:
    """确认节点：确认订单信息"""
    llm = get_default_llm()
    confirm_result = state["confirm_result"]
    response = llm.invoke(
        f"订单已确认：{confirm_result}。请生成一条订单确认通知，包含预计发货时间。"
    )
    print(f"  [确认] {response.content}")
    return {"current_step": "shipping", "ship_result": response.content}


def order_ship(state: OrderState) -> dict:
    """发货节点：处理发货"""
    llm = get_default_llm()
    ship_result = state["ship_result"]
    response = llm.invoke(
        f"基于订单信息：{ship_result}。请生成一条发货通知，包含快递公司和预计到达时间。"
    )
    print(f"  [发货] {response.content}")
    return {"current_step": "completed", "final_result": response.content}


def order_complete(state: OrderState) -> dict:
    """完成节点：生成订单完成总结"""
    llm = get_default_llm()
    final_result = state["final_result"]
    response = llm.invoke(
        f"订单已发货：{final_result}。请生成一条订单完成确认，感谢用户购买。"
    )
    print(f"  [完成] {response.content}")
    return {"current_step": "done", "final_result": state["final_result"] + "\n" + response.content}


def demo_order_workflow():
    """示例1：简单状态图 - 订单处理流程

    实战要点：
    - StateGraph 是最基础的图结构，定义状态后添加节点和边
    - add_node() 添加节点，add_edge() 添加边
    - add_edge(START, "node") 定义入口，add_edge("node", END) 定义出口
    - 节点函数接收状态，返回状态更新（部分更新即可）
    """
    print("\n" + "=" * 60)
    print("示例1：简单状态图 - 订单处理流程")
    print("=" * 60)
    print("""
核心概念：
  StateGraph: 状态图，工作流的核心载体
  - 定义状态类型（TypedDict）
  - 添加节点（add_node）
  - 添加边（add_edge）
  - 编译并运行（compile + invoke）

流程：下单 → 确认 → 发货 → 完成
    """)

    # 构建状态图
    graph = StateGraph(OrderState)

    # 添加节点
    graph.add_node("place", order_place)       # 下单
    graph.add_node("confirm", order_confirm)   # 确认
    graph.add_node("ship", order_ship)         # 发货
    graph.add_node("complete", order_complete)  # 完成

    # 添加边：定义流程走向
    graph.add_edge(START, "place")             # 起点 → 下单
    graph.add_edge("place", "confirm")         # 下单 → 确认
    graph.add_edge("confirm", "ship")          # 确认 → 发货
    graph.add_edge("ship", "complete")         # 发货 → 完成
    graph.add_edge("complete", END)            # 完成 → 终点

    # 编译图
    app = graph.compile()

    print("【交互式订单处理】")
    print("输入订单信息，自动流转：下单 → 确认 → 发货 → 完成")
    print("\n输入 '退出' 结束\n")

    while True:
        order_info = input("请输入订单信息（如：2本Python编程书）：").strip()
        if order_info.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if not order_info:
            print("请输入订单信息")
            continue

        try:
            # 运行状态图
            print("\n" + "─" * 40)
            print("订单处理流程开始...")
            result = app.invoke({"order_info": order_info, "current_step": ""})
            print("─" * 40)
            print(f"\n订单处理完成！最终状态: {result['current_step']}")
        except Exception as e:
            print(f"错误：{e}")

        print("\n" + "-" * 60)

    print("\n实战要点总结：")
    print("   1. StateGraph 通过 TypedDict 定义状态结构")
    print("   2. 节点函数接收完整状态，返回部分更新")
    print("   3. add_edge() 定义固定流转路径，START/END 标记起止")


# ============================================================
# 示例2: 节点与边 - 学习助手
# ============================================================

class LearningState(TypedDict):
    """学习助手状态"""
    question: str              # 用户问题
    category: str              # 问题分类
    answer: str                # 回答
    explanation: str           # 详细解释


def classify_question(state: LearningState) -> dict:
    """分类节点：将用户问题分类"""
    llm = get_default_llm()
    question = state["question"]
    response = llm.invoke(
        f"请将以下问题分类为以下类别之一（只输出类别名）：\n"
        f"- 数学：涉及计算、公式、数字\n"
        f"- 编程：涉及代码、算法、软件\n"
        f"- 语言：涉及语法、写作、翻译\n"
        f"- 常识：其他一般性问题\n\n"
        f"问题：{question}\n\n只输出类别名。"
    )
    category = response.content.strip()
    # 确保分类在有效范围内
    valid_categories = ["数学", "编程", "语言", "常识"]
    for vc in valid_categories:
        if vc in category:
            category = vc
            break
    else:
        category = "常识"
    print(f"  [分类] 问题类型：{category}")
    return {"category": category}


def answer_math(state: LearningState) -> dict:
    """数学回答节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"你是一位数学老师。请用清晰的步骤解答以下数学问题：\n{state['question']}"
    )
    print(f"  [数学老师] {response.content[:100]}...")
    return {"answer": response.content, "explanation": "数学问题由数学老师解答"}


def answer_programming(state: LearningState) -> dict:
    """编程回答节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"你是一位编程导师。请用代码示例和详细解释回答以下编程问题：\n{state['question']}"
    )
    print(f"  [编程导师] {response.content[:100]}...")
    return {"answer": response.content, "explanation": "编程问题由编程导师解答"}


def answer_language(state: LearningState) -> dict:
    """语言回答节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"你是一位语言学专家。请详细解答以下语言相关问题：\n{state['question']}"
    )
    print(f"  [语言学专家] {response.content[:100]}...")
    return {"answer": response.content, "explanation": "语言问题由语言学专家解答"}


def answer_general(state: LearningState) -> dict:
    """常识回答节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"你是一位博学的助手。请用通俗易懂的方式回答以下问题：\n{state['question']}"
    )
    print(f"  [博学助手] {response.content[:100]}...")
    return {"answer": response.content, "explanation": "常识问题由博学助手解答"}


def route_by_category(state: LearningState) -> str:
    """路由函数：根据问题分类路由到不同处理节点"""
    category = state.get("category", "常识")
    route_map = {
        "数学": "math_expert",
        "编程": "programming_expert",
        "语言": "language_expert",
    }
    return route_map.get(category, "general_expert")


def demo_learning_assistant():
    """示例2：节点与边 - 学习助手

    实战要点：
    - add_conditional_edges() 添加条件边，根据状态动态选择下一个节点
    - 路由函数接收状态，返回目标节点名称
    - 一个节点可以通过条件边连接到多个后续节点
    - 条件边实现了"分发"模式：根据输入类型路由到不同处理逻辑
    """
    print("\n" + "=" * 60)
    print("示例2：节点与边 - 学习助手")
    print("=" * 60)
    print("""
核心概念：
  Node: 节点，执行具体逻辑的单元
  Edge: 边，连接节点、定义流程走向
  条件边: 根据状态动态选择下一个节点

流程：用户问题 → 分类 → 路由到不同专家 → 回答
    """)

    # 构建状态图
    graph = StateGraph(LearningState)

    # 添加节点
    graph.add_node("classify", classify_question)
    graph.add_node("math_expert", answer_math)
    graph.add_node("programming_expert", answer_programming)
    graph.add_node("language_expert", answer_language)
    graph.add_node("general_expert", answer_general)

    # 添加边
    graph.add_edge(START, "classify")

    # 添加条件边：分类后路由到不同专家
    graph.add_conditional_edges(
        "classify",
        route_by_category,
        {
            "math_expert": "math_expert",
            "programming_expert": "programming_expert",
            "language_expert": "language_expert",
            "general_expert": "general_expert",
        }
    )

    # 各专家节点结束后都到 END
    graph.add_edge("math_expert", END)
    graph.add_edge("programming_expert", END)
    graph.add_edge("language_expert", END)
    graph.add_edge("general_expert", END)

    # 编译图
    app = graph.compile()

    print("【交互式学习助手】")
    print("输入问题，自动分类并路由到对应专家回答")
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
            result = app.invoke({"question": question, "category": "", "answer": "", "explanation": ""})
            print("─" * 40)
            print(f"\n分类结果：{result['category']}")
            print(f"处理方式：{result['explanation']}")
            print(f"\n完整回答：\n{result['answer']}")
        except Exception as e:
            print(f"错误：{e}")

        print("\n" + "-" * 60)

    print("\n实战要点总结：")
    print("   1. add_conditional_edges() 实现动态路由")
    print("   2. 路由函数根据状态返回目标节点名称")
    print("   3. 条件边映射表定义了路由函数返回值与节点的对应关系")


# ============================================================
# 示例3: 条件边 - 智能客服
# ============================================================

class CustomerServiceState(TypedDict):
    """智能客服状态"""
    user_input: str        # 用户输入
    intent: str            # 意图分类
    response: str          # 回复内容


def detect_intent(state: CustomerServiceState) -> dict:
    """意图检测节点：分析用户问题的意图"""
    llm = get_default_llm()
    user_input = state["user_input"]
    response = llm.invoke(
        f"请判断以下用户问题的意图，只输出以下类别之一：\n"
        f"- 退货退款：涉及退换货、退款、售后\n"
        f"- 产品咨询：涉及产品功能、规格、价格\n"
        f"- 物流查询：涉及快递、配送、物流状态\n"
        f"- 投诉建议：涉及投诉、建议、不满\n\n"
        f"用户问题：{user_input}\n\n只输出类别名。"
    )
    intent = response.content.strip()
    valid_intents = ["退货退款", "产品咨询", "物流查询", "投诉建议"]
    for vi in valid_intents:
        if vi in intent:
            intent = vi
            break
    else:
        intent = "产品咨询"
    print(f"  [意图检测] 识别意图：{intent}")
    return {"intent": intent}


def handle_refund(state: CustomerServiceState) -> dict:
    """退货退款处理节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"你是一位专业的售后客服。用户问题：{state['user_input']}\n"
        f"请提供退货退款相关的解决方案，包括：\n"
        f"1. 退货流程说明\n2. 退款时效\n3. 注意事项"
    )
    return {"response": response.content}


def handle_product(state: CustomerServiceState) -> dict:
    """产品咨询处理节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"你是一位产品顾问。用户问题：{state['user_input']}\n"
        f"请提供详细的产品咨询回复，包括：\n"
        f"1. 产品功能介绍\n2. 适用场景\n3. 价格信息（如了解）"
    )
    return {"response": response.content}


def handle_logistics(state: CustomerServiceState) -> dict:
    """物流查询处理节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"你是一位物流客服。用户问题：{state['user_input']}\n"
        f"请提供物流相关的帮助，包括：\n"
        f"1. 物流查询方式\n2. 配送时效说明\n3. 异常件处理流程"
    )
    return {"response": response.content}


def handle_complaint(state: CustomerServiceState) -> dict:
    """投诉建议处理节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"你是一位客户关系经理。用户问题：{state['user_input']}\n"
        f"请认真对待用户的投诉/建议，提供：\n"
        f"1. 诚恳的道歉和共情\n2. 问题解决方案\n3. 后续跟进承诺"
    )
    return {"response": response.content}


def route_intent(state: CustomerServiceState) -> str:
    """根据意图路由到不同处理节点"""
    intent = state.get("intent", "产品咨询")
    route_map = {
        "退货退款": "refund",
        "产品咨询": "product",
        "物流查询": "logistics",
        "投诉建议": "complaint",
    }
    return route_map.get(intent, "product")


def demo_customer_service():
    """示例3：条件边 - 智能客服

    实战要点：
    - 条件边是 LangGraph 最强大的特性之一
    - 可以根据状态中的任意字段动态选择下一个节点
    - 路由函数可以包含复杂的判断逻辑
    - 实现了"意图识别 → 分发处理"的典型客服架构
    """
    print("\n" + "=" * 60)
    print("示例3：条件边 - 智能客服")
    print("=" * 60)
    print("""
核心概念：
  条件边（add_conditional_edges）：根据状态动态选择下一个节点
  - 路由函数：接收状态，返回目标节点名
  - 映射表：定义返回值与节点的对应关系

流程：用户问题 → 意图检测 → 路由到不同处理节点 → 回复
    """)

    # 构建状态图
    graph = StateGraph(CustomerServiceState)

    # 添加节点
    graph.add_node("detect", detect_intent)
    graph.add_node("refund", handle_refund)
    graph.add_node("product", handle_product)
    graph.add_node("logistics", handle_logistics)
    graph.add_node("complaint", handle_complaint)

    # 添加边
    graph.add_edge(START, "detect")
    graph.add_conditional_edges(
        "detect",
        route_intent,
        {
            "refund": "refund",
            "product": "product",
            "logistics": "logistics",
            "complaint": "complaint",
        }
    )
    graph.add_edge("refund", END)
    graph.add_edge("product", END)
    graph.add_edge("logistics", END)
    graph.add_edge("complaint", END)

    # 编译图
    app = graph.compile()

    print("【交互式智能客服】")
    print("输入你的问题，自动识别意图并路由到对应客服")
    print("\n支持类型：退货退款、产品咨询、物流查询、投诉建议")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("请描述你的问题：").strip()
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if not user_input:
            print("请输入问题")
            continue

        try:
            print("\n" + "─" * 40)
            result = app.invoke({"user_input": user_input, "intent": "", "response": ""})
            print("─" * 40)
            print(f"\n意图识别：{result['intent']}")
            print(f"\n客服回复：\n{result['response']}")
        except Exception as e:
            print(f"错误：{e}")

        print("\n" + "-" * 60)

    print("\n实战要点总结：")
    print("   1. 条件边实现意图识别后的动态分发")
    print("   2. 路由函数根据状态中的意图字段返回目标节点")
    print("   3. 每个处理节点独立实现，互不影响，便于维护")


# ============================================================
# 示例4: 完整工作流 - 旅行规划助手
# ============================================================

class TravelState(TypedDict):
    """旅行规划状态"""
    destination: str        # 目的地
    preferences: str        # 偏好
    attractions: str        # 景点推荐
    food_guide: str         # 美食攻略
    itinerary: str          # 行程规划
    budget_advice: str      # 预算建议
    final_plan: str         # 最终方案


def recommend_attractions(state: TravelState) -> dict:
    """景点推荐节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"你是一位旅行顾问。用户想去{state['destination']}旅行，"
        f"偏好{state['preferences']}。\n"
        f"请推荐5个最值得去的景点，每个景点用一句话说明推荐理由。"
    )
    print(f"  [景点推荐] 已生成推荐")
    return {"attractions": response.content}


def recommend_food(state: TravelState) -> dict:
    """美食攻略节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"根据以下景点推荐，为{state['destination']}的旅行者制定美食攻略：\n"
        f"{state['attractions']}\n\n"
        f"请推荐每个景点附近的地道美食，包括菜品名称和简短介绍。"
    )
    print(f"  [美食攻略] 已生成攻略")
    return {"food_guide": response.content}


def plan_itinerary(state: TravelState) -> dict:
    """行程规划节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"请根据以下信息编排一份完整的旅行行程：\n\n"
        f"目的地：{state['destination']}\n"
        f"偏好：{state['preferences']}\n"
        f"【景点推荐】\n{state['attractions']}\n\n"
        f"【美食攻略】\n{state['food_guide']}\n\n"
        f"要求：按天安排，每天有上午、下午、晚上的行程，标注预估时长。"
    )
    print(f"  [行程规划] 已生成行程")
    return {"itinerary": response.content}


def budget_advice(state: TravelState) -> dict:
    """预算建议节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"根据以下旅行行程，给出预算建议：\n"
        f"目的地：{state['destination']}\n"
        f"行程：{state['itinerary']}\n\n"
        f"请提供：\n1. 交通费用估算\n2. 住宿费用估算\n3. 餐饮费用估算\n4. 门票费用估算\n5. 总预算范围"
    )
    print(f"  [预算建议] 已生成建议")
    return {"budget_advice": response.content}


def generate_final_plan(state: TravelState) -> dict:
    """生成最终方案节点"""
    llm = get_default_llm()
    response = llm.invoke(
        f"请将以下内容整合为一份完整的旅行规划方案：\n\n"
        f"【景点推荐】\n{state['attractions']}\n\n"
        f"【美食攻略】\n{state['food_guide']}\n\n"
        f"【行程安排】\n{state['itinerary']}\n\n"
        f"【预算建议】\n{state['budget_advice']}\n\n"
        f"请添加标题、小贴士和注意事项，使方案更加完整。"
    )
    return {"final_plan": response.content}


def demo_travel_planner():
    """示例4：完整工作流 - 旅行规划助手

    实战要点：
    - 多节点串联构建复杂工作流
    - 前序节点的输出作为后序节点的输入
    - StateGraph 自动管理状态传递
    - 每个节点只关注自己的职责，降低复杂度
    """
    print("\n" + "=" * 60)
    print("示例4：完整工作流 - 旅行规划助手")
    print("=" * 60)
    print("""
核心概念：
  完整工作流：多个节点串联，前序输出作为后序输入
  - START/END 标记流程的起点和终点
  - State 自动在节点间传递和更新
  - 每个节点只关注自己的职责

流程：目的地 → 景点推荐 → 美食攻略 → 行程规划 → 预算建议 → 最终方案
    """)

    # 构建状态图
    graph = StateGraph(TravelState)

    # 添加节点
    graph.add_node("attractions", recommend_attractions)
    graph.add_node("food", recommend_food)
    graph.add_node("itinerary", plan_itinerary)
    graph.add_node("budget", budget_advice)
    graph.add_node("final", generate_final_plan)

    # 添加边：定义完整流程
    graph.add_edge(START, "attractions")
    graph.add_edge("attractions", "food")
    graph.add_edge("food", "itinerary")
    graph.add_edge("itinerary", "budget")
    graph.add_edge("budget", "final")
    graph.add_edge("final", END)

    # 编译图
    app = graph.compile()

    print("【交互式旅行规划助手】")
    print("输入目的地，自动生成完整旅行规划方案")
    print("\n输入 '退出' 结束\n")

    while True:
        destination = input("你想去哪里旅行？：").strip()
        if destination.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break
        if not destination:
            print("请输入目的地")
            continue

        preferences = input("你的旅行偏好？(如：文化/自然/休闲/冒险)：").strip()
        if not preferences:
            preferences = "休闲"

        try:
            print("\n" + "─" * 40)
            print("正在规划旅行方案...")
            result = app.invoke({
                "destination": destination,
                "preferences": preferences,
                "attractions": "",
                "food_guide": "",
                "itinerary": "",
                "budget_advice": "",
                "final_plan": "",
            })
            print("─" * 40)
            print(f"\n{result['final_plan']}")
            print("\n旅行规划完成！")
        except Exception as e:
            print(f"错误：{e}")

        print("\n" + "-" * 60)

    print("\n实战要点总结：")
    print("   1. 多节点串联构建复杂工作流，每个节点职责单一")
    print("   2. State 自动在节点间传递，前序输出作为后序输入")
    print("   3. START/END 标记起止，图结构清晰可读")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "=" * 60)
    print("  LangGraph 基础案例 - 状态图、节点与边、条件边、工作流")
    print("=" * 60)
    print("\n核心概念：")
    print("  • StateGraph: 状态图，工作流的核心载体")
    print("  • State: 状态，存储工作流运行过程中的数据")
    print("  • Node: 节点，执行具体逻辑的单元")
    print("  • Edge: 边，连接节点、定义流程走向")
    print("  • START/END: 特殊节点，标记流程的起点和终点")

    while True:
        print("\n" + "=" * 60)
        print("请选择要运行的示例：")
        print("=" * 60)
        print("  1. 简单状态图 - 订单处理流程（下单→确认→发货→完成）")
        print("  2. 节点与边 - 学习助手（问题分类→路由到不同专家）")
        print("  3. 条件边 - 智能客服（意图识别→路由到不同处理）")
        print("  4. 完整工作流 - 旅行规划助手（景点→美食→行程→预算）")
        print("\n  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_order_workflow()
        elif choice == "2":
            demo_learning_assistant()
        elif choice == "3":
            demo_customer_service()
        elif choice == "4":
            demo_travel_planner()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
