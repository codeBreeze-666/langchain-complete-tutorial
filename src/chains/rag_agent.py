"""
LangChain RAG Agent - 实战交互式案例
====================================

本示例演示 RAG 与 Agent 的深度结合，使用关键词匹配模拟检索功能，
不依赖外部向量数据库。

核心概念：
- RAG Agent：将检索能力封装为工具，让 Agent 自主决定何时检索
- 多源检索：从不同知识源检索并交叉验证
- 对话式 RAG：在多轮对话中持续利用检索结果
- RAG + 工具：Agent 同时使用检索工具和功能工具

应用场景：
- 智能知识问答：Agent 自主判断是否需要检索再回答
- 多源知识融合：从多个领域知识库中综合信息
- 多轮对话检索：对话中保持上下文连续性
- 检索+计算混合：既需要知识查找又需要工具操作
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from src.utils.llm_loader import get_default_llm


# ============================================================
# 文本相似度工具（基于关键词匹配，替代向量数据库）
# ============================================================

def keyword_similarity(query: str, text: str) -> float:
    """
    基于关键词匹配计算文本相似度

    Args:
        query: 查询文本
        text: 待比较的文本

    Returns:
        相似度分数（0-1 之间）
    """
    q_lower = query.lower()
    t_lower = text.lower()

    # 停用词过滤
    stopwords = {"的", "了", "是", "在", "有", "和", "与", "及", "等", "个",
                 "一", "这", "那", "不", "也", "都", "就", "要", "会", "能",
                 "什么", "怎么", "如何", "哪些", "为什么", "吗", "呢", "吧",
                 "the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did",
                 "will", "would", "could", "should", "may", "might"}

    # 按分隔符切分并生成关键词
    keywords = set()
    for word in q_lower.replace("，", " ").replace("。", " ").replace("？", " ") \
                       .replace(",", " ").replace(".", " ").replace("?", " ") \
                       .replace("、", " ").replace("：", " ").replace(":", " ") \
                       .split():
        word = word.strip()
        if word and word not in stopwords:
            keywords.add(word)
            # 中文 2-gram 和 3-gram
            if len(word) >= 2:
                for i in range(len(word) - 1):
                    keywords.add(word[i:i+2])
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    keywords.add(word[i:i+3])

    if not keywords:
        return 0.0

    hits = sum(1 for kw in keywords if kw in t_lower)
    return hits / len(keywords)


def retrieve_texts(query: str, documents: list[str], top_k: int = 3) -> list[tuple[int, float, str]]:
    """
    从文档列表中检索与查询最相关的文本片段

    Args:
        query: 查询文本
        documents: 文档片段列表
        top_k: 返回前 k 个最相关的结果

    Returns:
        列表，每项为 (文档索引, 相似度分数, 文档内容)
    """
    scored = []
    for i, doc in enumerate(documents):
        score = keyword_similarity(query, doc)
        scored.append((i, score, doc))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ============================================================
# 预设知识库
# ============================================================

# 技术知识库
TECH_KNOWLEDGE_BASE = [
    "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。"
    "Python 以简洁优雅的语法著称，支持面向对象、函数式和过程式编程范式。"
    "Python 广泛应用于 Web 开发、数据分析、人工智能和自动化运维等领域。",

    "LangChain 是一个用于开发大语言模型应用的开源框架。"
    "它的核心组件包括：LLM 接口、Prompt 模板、Chain 链式调用、Agent 智能体、Memory 记忆模块。"
    "LangChain 支持与多种大模型提供商集成，包括 OpenAI、智谱、DeepSeek 等。",

    "RAG（检索增强生成）是一种结合检索和生成的技术方案。"
    "它的工作流程是：先从知识库中检索相关文档，再将检索结果作为上下文输入给大模型。"
    "RAG 能有效减少模型的幻觉问题，让回答有据可依。",

    "Agent（智能体）是 LangChain 中的核心概念，指能够自主决策和调用工具的 LLM 应用。"
    "Agent 通过 ReAct（Reasoning + Acting）范式交替进行思考和行动。"
    "LangGraph 是新一代 Agent 框架，通过 create_agent 简化 Agent 构建。",

    "向量数据库是专门用于存储和检索向量嵌入的数据库系统。"
    "常见的向量数据库包括 Chroma、Pinecone、Milvus 和 Weaviate。"
    "它们支持高效的相似度搜索，是 RAG 系统的关键基础设施。",
]

# 业务知识库
BUSINESS_KNOWLEDGE_BASE = [
    "公司年假政策：入职满1年享有5天年假，满3年享有10天年假，满5年享有15天年假。"
    "年假不可跨年累积，需提前3个工作日申请，经直属主管审批后生效。",

    "报销流程：员工需在费用发生后30天内提交报销申请，附上发票原件和部门主管签字。"
    "单笔金额超过5000元需额外获得财务总监审批，报销周期约为5-10个工作日。",

    "远程办公规定：每周最多2天远程办公，需提前1天在OA系统申请并获主管审批。"
    "远程办公期间需保持工作软件在线，参加所有已安排的线上会议。",

    "培训制度：公司每年提供不少于40小时的培训时间，包括内部分享和外部课程。"
    "员工可申请外部培训费用报销，年度上限为5000元，需提交培训总结。",

    "绩效考核：每季度进行一次绩效评估，年度综合评估结果影响年终奖金和职级调整。"
    "绩效等级分为 S（卓越）、A（优秀）、B（良好）、C（待改进）、D（不合格）五档。",
]

# 产品知识库
PRODUCT_KNOWLEDGE_BASE = [
    "SmartAssist Pro 是公司旗舰AI助手产品，支持多轮对话、文档问答和代码生成。"
    "定价：基础版 99元/月，专业版 299元/月，企业版按需定制。"
    "专业版支持自定义知识库和API集成。",

    "DataFlow Engine 是企业级数据处理平台，支持ETL、实时流处理和数据质量监控。"
    "定价：标准版 5000元/月，高级版 12000元/月。"
    "高级版支持自定义数据源接入和分布式部署。",

    "CloudGuard 是云安全监控平台，提供漏洞扫描、合规审计和威胁检测功能。"
    "定价：入门版 199元/月，商业版 799元/月。"
    "商业版支持多云环境管理和自定义安全策略。",

    "DocuMind 是智能文档管理平台，支持OCR识别、自动分类和全文检索。"
    "定价：个人版 49元/月，团队版 199元/月，企业版 999元/月。"
    "团队版及以上支持多人协作和版本管理。",

    "CodePilot 是AI辅助编程工具，支持代码补全、重构建议和Bug检测。"
    "定价：免费版（基础补全），专业版 79元/月。"
    "专业版支持私有仓库接入和团队代码风格统一。",
]


# ============================================================
# 1. 简单 RAG Agent - 基于检索结果回答问题
# ============================================================

def demo_simple_rag_agent():
    """示例1：简单 RAG Agent - 检索封装为工具，Agent 自主决定何时检索"""
    print("\n" + "=" * 60)
    print("示例1：简单 RAG Agent - 检索增强的智能体")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - 将检索功能封装为 @tool，Agent 自主决定是否调用")
    print("   - 与普通 RAG 链不同：Agent 可以先思考，再决定是否检索")
    print("   - Agent 能判断问题是否需要外部知识，避免无意义检索")

    model = get_default_llm()
    knowledge_base = TECH_KNOWLEDGE_BASE

    # 将检索封装为 Agent 工具
    @tool
    def search_tech_knowledge(query: str) -> str:
        """搜索技术知识库，当用户询问Python、LangChain、RAG、Agent、向量数据库等技术问题时使用此工具

        Args:
            query: 搜索关键词

        Returns:
            检索到的相关知识内容
        """
        results = retrieve_texts(query, knowledge_base, top_k=2)
        if not results or results[0][1] == 0.0:
            return "未在知识库中找到相关内容"
        return "\n\n".join([f"[知识片段{i+1}] {doc}" for i, (_, _, doc) in enumerate(results)])

    tools = [search_tech_knowledge]

    agent = create_agent(model, tools, system_prompt="你是一个技术知识助手。当用户的问题需要参考知识库时，使用搜索工具获取信息；当问题属于常识或你已有把握回答时，可以直接回答。如果使用了知识库信息，请在回答中说明参考来源。")

    # 显示知识库概览
    print(f"\n📚 技术知识库包含 {len(knowledge_base)} 条知识：")
    for i, doc in enumerate(knowledge_base):
        preview = doc[:40] + "..." if len(doc) > 40 else doc
        print(f"   [{i+1}] {preview}")

    print("\n【交互式问答】")
    print("提示：Agent 会自动判断是否需要检索知识库来回答你的问题")
    print("试试问：")
    print("  • '什么是 RAG？'（需要检索）")
    print("  • '1+1等于几？'（无需检索，直接回答）")
    print("  • 'LangChain 有哪些核心组件？'（需要检索）")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("你的问题：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效问题")
            continue

        try:
            result = agent.invoke({"messages": [("user", user_input)]})
            final_message = result["messages"][-1]
            print(f"\n🤖 Agent 回答：{final_message.content}\n")
        except Exception as e:
            print(f"❌ 错误：{e}\n")

        print("-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. RAG Agent = 检索工具 + Agent 推理，比纯 RAG 链更灵活")
    print("   2. Agent 能判断问题是否需要检索，避免冗余调用")
    print("   3. 工具的 docstring 决定 Agent 何时使用检索")


# ============================================================
# 2. 多源 RAG - 从多个知识源检索
# ============================================================

def demo_multi_source_rag():
    """示例2：多源 RAG - Agent 从多个知识源中选择性检索"""
    print("\n" + "=" * 60)
    print("示例2：多源 RAG - 多知识源智能检索")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - 不同知识源封装为不同工具，Agent 根据问题类型选择")
    print("   - Agent 可以同时调用多个知识源进行交叉验证")
    print("   - 多源检索是构建企业级 RAG 系统的基础架构")

    model = get_default_llm()

    # 为每个知识源创建独立的检索工具
    @tool
    def search_tech_docs(query: str) -> str:
        """搜索技术文档知识库，当用户询问编程语言、框架、AI技术等技术开发问题时使用

        Args:
            query: 搜索关键词

        Returns:
            检索到的技术文档内容
        """
        results = retrieve_texts(query, TECH_KNOWLEDGE_BASE, top_k=2)
        if not results or results[0][1] == 0.0:
            return "技术文档中未找到相关内容"
        return "\n\n".join([f"[技术文档{i+1}] {doc}" for i, (_, _, doc) in enumerate(results)])

    @tool
    def search_business_policy(query: str) -> str:
        """搜索公司制度知识库，当用户询问年假、报销、远程办公、培训、绩效等公司制度问题时使用

        Args:
            query: 搜索关键词

        Returns:
            检索到的公司制度内容
        """
        results = retrieve_texts(query, BUSINESS_KNOWLEDGE_BASE, top_k=2)
        if not results or results[0][1] == 0.0:
            return "公司制度中未找到相关内容"
        return "\n\n".join([f"[公司制度{i+1}] {doc}" for i, (_, _, doc) in enumerate(results)])

    @tool
    def search_product_info(query: str) -> str:
        """搜索产品信息知识库，当用户询问产品功能、定价、版本等公司产品问题时使用

        Args:
            query: 搜索关键词

        Returns:
            检索到的产品信息内容
        """
        results = retrieve_texts(query, PRODUCT_KNOWLEDGE_BASE, top_k=2)
        if not results or results[0][1] == 0.0:
            return "产品信息中未找到相关内容"
        return "\n\n".join([f"[产品信息{i+1}] {doc}" for i, (_, _, doc) in enumerate(results)])

    tools = [search_tech_docs, search_business_policy, search_product_info]

    agent = create_agent(model, tools, system_prompt="你是一个全能知识助手，可以搜索技术文档、公司制度和产品信息三个知识源。请根据用户的问题，选择合适的知识源进行搜索。如果问题涉及多个领域，可以同时搜索多个知识源。回答时请标注信息来源（技术文档/公司制度/产品信息）。")

    # 显示知识源概览
    print(f"\n📚 可用知识源：")
    print(f"   🔧 技术文档库 - {len(TECH_KNOWLEDGE_BASE)} 条（Python/LangChain/RAG/Agent/向量数据库）")
    print(f"   🏢 公司制度库 - {len(BUSINESS_KNOWLEDGE_BASE)} 条（年假/报销/远程办公/培训/绩效）")
    print(f"   📦 产品信息库 - {len(PRODUCT_KNOWLEDGE_BASE)} 条（SmartAssist/DataFlow/CloudGuard/DocuMind/CodePilot）")

    print("\n【交互式问答】")
    print("提示：Agent 会自动选择合适的知识源检索")
    print("试试问：")
    print("  • '什么是向量数据库？'（技术文档）")
    print("  • '公司的年假政策是什么？'（公司制度）")
    print("  • 'SmartAssist Pro 多少钱？'（产品信息）")
    print("  • '我想了解RAG和公司的培训制度'（跨源检索）")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("你的问题：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效问题")
            continue

        try:
            result = agent.invoke({"messages": [("user", user_input)]})
            final_message = result["messages"][-1]
            print(f"\n🤖 Agent 回答：{final_message.content}\n")
        except Exception as e:
            print(f"❌ 错误：{e}\n")

        print("-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. 每个知识源封装为独立工具，Agent 按需选择")
    print("   2. 工具描述清晰区分适用场景，减少错误路由")
    print("   3. Agent 可以并行调用多源检索，实现跨领域问答")


# ============================================================
# 3. 对话式 RAG - 支持多轮对话的 RAG
# ============================================================

def demo_conversational_rag():
    """示例3：对话式 RAG - 多轮对话中持续利用检索结果"""
    print("\n" + "=" * 60)
    print("示例3：对话式 RAG - 多轮对话中的检索增强")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - 对话式 RAG 通过 chat_history 传递上下文")
    print("   - Agent 能理解追问中的指代（如'它'、'这个'）")
    print("   - 历史检索结果可以复用，避免重复检索相同内容")

    model = get_default_llm()
    knowledge_base = TECH_KNOWLEDGE_BASE + BUSINESS_KNOWLEDGE_BASE

    @tool
    def search_knowledge(query: str) -> str:
        """搜索综合知识库，当需要查找具体的技术或公司制度信息时使用此工具

        Args:
            query: 搜索关键词

        Returns:
            检索到的知识内容
        """
        results = retrieve_texts(query, knowledge_base, top_k=3)
        if not results or results[0][1] == 0.0:
            return "知识库中未找到相关内容"
        return "\n\n".join([f"[知识{i+1}] {doc}" for i, (_, _, doc) in enumerate(results)])

    tools = [search_knowledge]

    agent = create_agent(model, tools, system_prompt="你是一个对话式知识助手，可以在对话中回答技术问题和公司制度问题。当需要具体信息时使用搜索工具。请注意理解用户追问中的指代，结合之前的对话上下文来回答。如果之前已经搜索过相关信息，且问题可以在已有信息中回答，可以不重复搜索。")

    # 对话历史
    chat_history = []

    print(f"\n📚 知识库包含 {len(knowledge_base)} 条知识（技术+公司制度）")

    print("\n【多轮对话问答】")
    print("提示：Agent 会记住之前的对话，你可以追问和指代")
    print("试试这样的对话流程：")
    print("  • 第一轮：'什么是 RAG？'")
    print("  • 第二轮：'它能解决什么问题？'（Agent 理解'它'指 RAG）")
    print("  • 第三轮：'公司有哪些相关培训？'（切换话题到公司制度）")
    print("\n输入 '退出' 结束 | 输入 '清空' 重置记忆\n")

    while True:
        user_input = input("你：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if user_input.lower() in ["清空", "clear"]:
            chat_history.clear()
            print("对话记忆已清空\n")
            continue

        if not user_input:
            print("请输入有效内容")
            continue

        try:
            result = agent.invoke({"messages": chat_history + [("user", user_input)]})

            # 将对话加入历史
            final_message = result["messages"][-1]
            chat_history.append(("user", user_input))
            chat_history.append(("assistant", final_message.content))

            print(f"\n🤖 助手：{final_message.content}\n")
        except Exception as e:
            print(f"❌ 错误：{e}\n")

        print("-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. chat_history 传递对话上下文，Agent 理解追问和指代")
    print("   2. 对话记忆 + 检索工具 = 持续性知识问答")
    print("   3. 长对话时注意记忆窗口管理，避免上下文过长")


# ============================================================
# 4. RAG + 工具 - Agent 同时使用检索和工具
# ============================================================

def demo_rag_with_tools():
    """示例4：RAG + 工具 - Agent 同时使用检索工具和功能工具"""
    print("\n" + "=" * 60)
    print("示例4：RAG + 工具 - 检索与功能工具协同")
    print("=" * 60)
    print("\n💡 实战要点：")
    print("   - RAG 检索只是 Agent 工具集的一部分")
    print("   - Agent 可以先用检索获取知识，再用功能工具处理数据")
    print("   - 实际业务中常需要「查信息 + 做计算」的组合能力")

    model = get_default_llm()
    knowledge_base = PRODUCT_KNOWLEDGE_BASE + BUSINESS_KNOWLEDGE_BASE

    # 检索工具
    @tool
    def search_product_and_policy(query: str) -> str:
        """搜索产品信息和公司制度知识库，当需要了解产品定价、功能或公司政策时使用

        Args:
            query: 搜索关键词

        Returns:
            检索到的产品或制度信息
        """
        results = retrieve_texts(query, knowledge_base, top_k=3)
        if not results or results[0][1] == 0.0:
            return "知识库中未找到相关内容"
        return "\n\n".join([f"[信息{i+1}] {doc}" for i, (_, _, doc) in enumerate(results)])

    # 功能工具：计算器
    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式，当需要进行价格计算、数量统计等数学运算时使用

        Args:
            expression: 数学表达式，如 '299 * 12' 或 '5000 + 3000'

        Returns:
            计算结果
        """
        try:
            # 安全计算：只允许数字和基本运算符
            allowed = set("0123456789+-*/().% ")
            if not all(c in allowed for c in expression):
                return "表达式包含不支持的字符，只支持数字和基本运算符"
            result = eval(expression)
            return f"计算结果：{expression} = {result}"
        except Exception as e:
            return f"计算错误：{e}"

    # 功能工具：日期计算
    @tool
    def date_calculator(operation: str) -> str:
        """计算日期相关的问题，如计算工作日数、判断是否在期限内等

        Args:
            operation: 日期计算描述，如 '30天后是哪天' 或 '从2024-01-01到2024-03-15多少天'

        Returns:
            日期计算结果
        """
        from datetime import datetime, timedelta

        today = datetime.now()

        # 尝试解析 "X天后" 格式
        if "天后" in operation:
            try:
                days = int("".join(filter(str.isdigit, operation)))
                future = today + timedelta(days=days)
                return f"今天是 {today.strftime('%Y-%m-%d')}，{days} 天后是 {future.strftime('%Y-%m-%d')}"
            except ValueError:
                pass

        # 尝试计算两个日期之间的天数
        import re
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', operation)
        if len(dates) >= 2:
            try:
                d1 = datetime.strptime(dates[0], "%Y-%m-%d")
                d2 = datetime.strptime(dates[1], "%Y-%m-%d")
                diff = abs((d2 - d1).days)
                return f"从 {dates[0]} 到 {dates[1]} 共 {diff} 天"
            except ValueError:
                pass

        return f"今天是 {today.strftime('%Y-%m-%d')}，无法解析日期计算请求，请使用 'X天后' 或 'YYYY-MM-DD到YYYY-MM-DD' 格式"

    tools = [search_product_and_policy, calculator, date_calculator]

    agent = create_agent(model, tools, system_prompt="你是一个企业助手，可以查询产品信息和公司制度，还能进行数学计算和日期计算。当用户的问题涉及具体数据时，请先用搜索工具获取信息，再用计算工具处理数据。例如：查价格后算总价、查制度后算期限。请给出详细的分析过程。")

    print(f"\n📚 知识库：产品信息({len(PRODUCT_KNOWLEDGE_BASE)}条) + 公司制度({len(BUSINESS_KNOWLEDGE_BASE)}条)")
    print("🔧 工具集：知识搜索 + 计算器 + 日期计算")

    print("\n【交互式问答】")
    print("提示：Agent 会结合检索和工具来回答复合问题")
    print("试试问：")
    print("  • 'SmartAssist Pro 专业版一年多少钱？'（检索+计算）")
    print("  • 'DataFlow Engine 标准版和高级版价格差多少？'（检索+计算）")
    print("  • '报销的30天期限，如果费用发生在今天，最晚哪天提交？'（检索+日期计算）")
    print("  • '3个SmartAssist Pro基础版和2个CodePilot专业版一个月共多少？'（检索+计算）")
    print("\n输入 '退出' 结束\n")

    while True:
        user_input = input("你的问题：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效问题")
            continue

        try:
            result = agent.invoke({"messages": [("user", user_input)]})
            final_message = result["messages"][-1]
            print(f"\n🤖 Agent 回答：{final_message.content}\n")
        except Exception as e:
            print(f"❌ 错误：{e}\n")

        print("-" * 60)

    print("\n✅ 实战要点总结：")
    print("   1. RAG + 工具 = 知识获取 + 数据处理，覆盖更复杂的业务场景")
    print("   2. Agent 能编排多步操作：先检索信息，再基于信息做计算")
    print("   3. 工具描述越清晰，Agent 的工具选择和参数传递越准确")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "=" * 60)
    print("  LangChain RAG Agent - 实战案例")
    print("=" * 60)
    print("\n本示例演示 RAG 与 Agent 的深度结合")
    print("使用关键词匹配模拟检索，无需外部向量数据库")
    print("\n核心概念：")
    print("  • RAG Agent：检索能力封装为工具，Agent 自主决策")
    print("  • 多源检索：不同知识源封装为不同工具")
    print("  • 对话式 RAG：多轮对话中持续利用检索结果")
    print("  • RAG + 工具：检索与功能工具协同工作")
    print("\n应用场景：")
    print("  • 智能知识问答、多源知识融合、多轮对话检索、检索+计算混合")

    while True:
        print("\n" + "=" * 60)
        print("请选择要运行的示例：")
        print("=" * 60)
        print("  1. 简单 RAG Agent - 检索增强的智能体")
        print("  2. 多源 RAG - 多知识源智能检索")
        print("  3. 对话式 RAG - 多轮对话中的检索增强")
        print("  4. RAG + 工具 - 检索与功能工具协同")
        print("\n  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_simple_rag_agent()
        elif choice == "2":
            demo_multi_source_rag()
        elif choice == "3":
            demo_conversational_rag()
        elif choice == "4":
            demo_rag_with_tools()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
