"""
LangChain RAG（检索增强生成）基础 - 实战交互式案例
====================================================

本示例演示 RAG 的核心概念与实战用法，不依赖外部向量数据库，
使用基于关键词匹配的简单文本相似度来模拟检索功能。

核心概念：
- RAG（Retrieval-Augmented Generation）：检索增强生成
- 文档切分：将长文本拆分为可检索的片段
- 相似度检索：找到与问题最相关的文本片段
- 上下文注入：将检索结果作为上下文提供给 LLM

应用场景：
- 简单 RAG：从预设文本中检索并回答
- 文档问答：基于用户输入的文档内容回答问题
- 相似度搜索：找到与问题最匹配的文本片段
- 带来源的 RAG：回答时标注信息来源
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_loader import get_default_llm


# ============================================================
# 文本相似度工具（基于关键词匹配，替代向量数据库）
# ============================================================

def keyword_similarity(query: str, text: str) -> float:
    """
    基于关键词匹配计算文本相似度

    原理：
    1. 对查询和文本进行分词（按字符级和词级双重匹配）
    2. 统计查询关键词在文本中出现的比例
    3. 考虑词序和词长权重

    Args:
        query: 查询文本
        text: 待比较的文本

    Returns:
        相似度分数（0-1 之间）
    """
    # 统一转为小写
    q_lower = query.lower()
    t_lower = text.lower()

    # 提取查询中的关键词（过滤掉常见停用词）
    stopwords = {"的", "了", "是", "在", "有", "和", "与", "及", "等", "个",
                 "一", "这", "那", "不", "也", "都", "就", "要", "会", "能",
                 "什么", "怎么", "如何", "哪些", "为什么", "吗", "呢", "吧"}
    # 按空格和标点切分，同时保留连续中文字符子串
    keywords = set()
    # 按常见分隔符切分
    for word in q_lower.replace("，", " ").replace("。", " ").replace("？", " ") \
                       .replace(",", " ").replace(".", " ").replace("?", " ") \
                       .split():
        word = word.strip()
        if word and word not in stopwords:
            keywords.add(word)
            # 对中文文本额外生成 2-gram 和 3-gram
            if len(word) >= 2:
                for i in range(len(word) - 1):
                    keywords.add(word[i:i+2])
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    keywords.add(word[i:i+3])

    if not keywords:
        return 0.0

    # 计算命中数
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

    # 按相似度降序排序，返回 top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ============================================================
# 预设知识库（用于示例1和示例4）
# ============================================================

DEFAULT_KNOWLEDGE_BASE = [
    "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。"
    "Python 以简洁优雅的语法著称，支持面向对象、函数式和过程式编程范式。"
    'Python 的设计哲学强调代码可读性，其核心格言是"优雅"和"明确"。',

    "LangChain 是一个用于开发大语言模型应用的开源框架。"
    "它的核心组件包括：LLM 接口、Prompt 模板、Chain 链式调用、Agent 智能体、Memory 记忆模块。"
    "LangChain 支持与多种大模型提供商集成，包括 OpenAI、智谱、DeepSeek 等。",

    "RAG（检索增强生成）是一种结合检索和生成的技术方案。"
    "它的工作流程是：先从知识库中检索相关文档，再将检索结果作为上下文输入给大模型，"
    "最后由大模型基于上下文生成答案。RAG 能有效减少模型的幻觉问题。",

    "机器学习是人工智能的重要分支，主要包括监督学习、无监督学习和强化学习三大类型。"
    "监督学习使用有标签的数据进行训练，常见算法包括线性回归、决策树和支持向量机。"
    "无监督学习用于发现数据中的隐藏模式，如聚类和降维。",

    "向量数据库是专门用于存储和检索向量嵌入的数据库系统。"
    "常见的向量数据库包括 Chroma、Pinecone、Milvus 和 Weaviate。"
    "它们支持高效的相似度搜索，是 RAG 系统的关键基础设施。",

    "Prompt Engineering（提示词工程）是优化大模型输出的关键技术。"
    "核心技巧包括：明确指令、提供示例（Few-shot）、分步思考（Chain of Thought）、"
    "角色设定和输出格式约束。好的提示词能显著提升模型输出的质量和稳定性。",
]


# ============================================================
# 1. 简单 RAG - 从预设文本中检索相关内容并回答
# ============================================================

def demo_simple_rag():
    """示例1：简单 RAG（从预设知识库中检索并回答）"""
    print("\n" + "="*60)
    print("示例1：简单 RAG（从预设知识库中检索并回答）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - RAG = 检索（Retrieve）+ 增强（Augmented）+ 生成（Generate）")
    print("   - 先检索相关文档，再让 LLM 基于上下文生成答案")
    print('   - 避免模型"凭空编造"，答案有据可依')

    model = get_default_llm()
    knowledge_base = DEFAULT_KNOWLEDGE_BASE

    # 显示知识库内容
    print(f"\n📚 当前知识库包含 {len(knowledge_base)} 条知识：")
    for i, doc in enumerate(knowledge_base):
        preview = doc[:40] + "..." if len(doc) > 40 else doc
        print(f"   [{i+1}] {preview}")

    # RAG 提示词模板
    prompt = ChatPromptTemplate.from_template(
        "请根据以下参考信息回答用户的问题。\n\n"
        "参考信息：\n{context}\n\n"
        "问题：{question}\n\n"
        "要求：\n"
        "1. 只根据参考信息回答，如果参考信息中没有相关内容，请说明\n"
        "2. 回答要简洁准确"
    )

    chain = prompt | model | StrOutputParser()

    print("\n【交互式问答】")
    print("提示：输入问题，AI 从知识库中检索相关内容并回答")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()

        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not question:
            print("请输入有效内容")
            continue

        # 第一步：检索相关文档
        results = retrieve_texts(question, knowledge_base, top_k=3)
        context = "\n\n".join([f"[片段{i+1}] {doc}" for i, (_, _, doc) in enumerate(results)])

        print(f"\n🔍 检索到 {len(results)} 条相关内容：")
        for i, (idx, score, doc) in enumerate(results):
            preview = doc[:50] + "..." if len(doc) > 50 else doc
            print(f"   [{i+1}] 相似度={score:.2f} | {preview}")

        # 第二步：基于检索结果生成答案
        print("\n🤖 AI 回答：")
        response = chain.invoke({"context": context, "question": question})
        print(response)
        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. RAG 的核心是「先检索，后生成」")
    print("   2. 检索质量直接决定回答质量")
    print("   3. 提示词中明确要求「只根据参考信息回答」可减少幻觉")


# ============================================================
# 2. 文档问答 - 用户输入文档，AI 基于文档回答
# ============================================================

def demo_document_qa():
    """示例2：文档问答（用户输入文档内容，AI 基于文档回答）"""
    print("\n" + "="*60)
    print("示例2：文档问答（用户输入文档，AI 基于文档回答）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 用户自定义知识源，灵活应对不同场景")
    print("   - 文档切分：将长文档拆分为合理大小的片段")
    print("   - 上下文窗口有限，需要控制注入的文档量")

    model = get_default_llm()

    print("\n【第一步：输入文档内容】")
    print("提示：输入一段文本作为知识库（可多行，输入空行结束）")
    print("或输入 '默认' 使用预设示例文档\n")

    lines = []
    while True:
        line = input("文档内容：")
        if line.strip() == '':
            if lines:
                break
            print("请输入至少一行内容，或输入 '默认'")
            continue
        if line.strip() == '默认':
            lines = [
                "公司年假政策：入职满1年享有5天年假，满3年享有10天年假，满5年享有15天年假。",
                "报销流程：员工需在费用发生后30天内提交报销申请，附上发票原件和部门主管签字。",
                "远程办公规定：每周最多2天远程办公，需提前1天在OA系统申请并获主管审批。",
                "培训制度：公司每年提供不少于40小时的培训时间，包括内部分享和外部课程。",
                "绩效考核：每季度进行一次绩效评估，年度综合评估结果影响年终奖金和职级调整。",
            ]
            break
        lines.append(line.strip())

    document_text = "\n".join(lines)

    # 将文档按句号切分为片段
    chunks = []
    for paragraph in document_text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # 按句号切分，每个片段不超过200字
        sentences = [s.strip() for s in paragraph.replace("。", "。\n").split("\n") if s.strip()]
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) > 200 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sent
            else:
                current_chunk += sent
        if current_chunk:
            chunks.append(current_chunk.strip())

    print(f"\n📄 文档已切分为 {len(chunks)} 个片段：")
    for i, chunk in enumerate(chunks):
        preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
        print(f"   [{i+1}] {preview}")

    # 文档问答提示词
    prompt = ChatPromptTemplate.from_template(
        "你是一个文档问答助手。请严格根据以下文档内容回答问题。\n\n"
        "文档内容：\n{context}\n\n"
        "问题：{question}\n\n"
        '注意：如果文档中没有相关信息，请明确回答"文档中未提及此内容"。'
    )

    chain = prompt | model | StrOutputParser()

    print("\n【第二步：基于文档提问】")
    print("提示：输入关于文档的问题，AI 会从文档中找到答案")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()

        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not question:
            print("请输入有效问题")
            continue

        # 检索相关片段
        results = retrieve_texts(question, chunks, top_k=3)
        context = "\n\n".join([f"[片段{i+1}] {doc}" for i, (_, _, doc) in enumerate(results)])

        print(f"\n🔍 检索到 {len(results)} 条相关片段")
        response = chain.invoke({"context": context, "question": question})
        print(f"\n🤖 AI 回答：{response}")
        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 文档切分是 RAG 的关键预处理步骤")
    print("   2. 切分粒度要适中：太粗会引入噪声，太细会丢失上下文")
    print('   3. 提示词要求"严格基于文档回答"可约束模型行为')


# ============================================================
# 3. 相似度搜索 - 找到与问题最相关的文本片段
# ============================================================

def demo_similarity_search():
    """示例3：相似度搜索（找到与问题最相关的文本片段）"""
    print("\n" + "="*60)
    print("示例3：相似度搜索（找到与问题最相关的文本片段）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 相似度搜索是 RAG 的「检索」环节")
    print("   - 关键词匹配是最基础的相似度计算方式")
    print("   - 生产环境建议使用向量嵌入 + 余弦相似度")

    knowledge_base = DEFAULT_KNOWLEDGE_BASE

    print(f"\n📚 知识库共 {len(knowledge_base)} 条内容")
    print("\n【交互式搜索】")
    print("提示：输入搜索内容，查看最相关的文本片段")
    print("输入 '退出' 结束\n")

    while True:
        query = input("搜索内容：").strip()

        if query.lower() in ['退出', 'exit', 'quit']:
            print("结束搜索")
            break

        if not query:
            print("请输入有效搜索内容")
            continue

        # 设置 top_k
        k_input = input("返回结果数（默认3，直接回车跳过）：").strip()
        top_k = int(k_input) if k_input.isdigit() and int(k_input) > 0 else 3

        # 执行搜索
        results = retrieve_texts(query, knowledge_base, top_k=top_k)

        print(f"\n🔍 搜索结果（共 {len(results)} 条）：")
        print("="*50)
        for i, (idx, score, doc) in enumerate(results):
            print(f"\n📄 结果 {i+1}（相似度：{score:.2f}，原文索引：{idx}）")
            print(f"   {doc}")

        # 额外显示：所有文档的相似度排名
        all_scored = retrieve_texts(query, knowledge_base, top_k=len(knowledge_base))
        print(f"\n📊 全部文档相似度排名：")
        for i, (idx, score, doc) in enumerate(all_scored):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            preview = doc[:30] + "..." if len(doc) > 30 else doc
            print(f"   {i+1}. [{bar}] {score:.2f} | {preview}")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 关键词匹配简单有效，但无法理解语义相似性")
    print("   2. 生产环境推荐使用 Embedding + 余弦相似度")
    print("   3. top_k 参数需要根据场景调优")


# ============================================================
# 4. 带来源的 RAG - 回答问题时标注信息来源
# ============================================================

def demo_rag_with_source():
    """示例4：带来源的 RAG（回答时标注信息来源）"""
    print("\n" + "="*60)
    print("示例4：带来源的 RAG（回答时标注信息来源）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 来源标注提升答案的可信度和可追溯性")
    print("   - 用户可以验证答案是否来自可靠的上下文")
    print("   - 企业级 RAG 系统的必备功能")

    model = get_default_llm()

    # 使用带来源标记的知识库
    knowledge_sources = [
        {"source": "《Python编程：从入门到实践》", "content": DEFAULT_KNOWLEDGE_BASE[0]},
        {"source": "LangChain 官方文档", "content": DEFAULT_KNOWLEDGE_BASE[1]},
        {"source": "《自然语言处理实战》", "content": DEFAULT_KNOWLEDGE_BASE[2]},
        {"source": "《机器学习》周志华", "content": DEFAULT_KNOWLEDGE_BASE[3]},
        {"source": "Milvus 技术白皮书", "content": DEFAULT_KNOWLEDGE_BASE[4]},
        {"source": "OpenAI Prompt 工程指南", "content": DEFAULT_KNOWLEDGE_BASE[5]},
    ]

    print(f"\n📚 知识库来源：")
    for i, item in enumerate(knowledge_sources):
        preview = item["content"][:35] + "..." if len(item["content"]) > 35 else item["content"]
        print(f"   [{i+1}] 来源：{item['source']}")
        print(f"       内容：{preview}")

    # 带来源标注的提示词
    prompt = ChatPromptTemplate.from_template(
        "你是一个严谨的知识问答助手。请根据以下参考信息回答问题，"
        "并在回答中标注信息来源。\n\n"
        "参考信息：\n{context_with_source}\n\n"
        "问题：{question}\n\n"
        "要求：\n"
        "1. 回答时必须标注来源，格式为：[来源：xxx]\n"
        "2. 如果多个来源都相关，都需要标注\n"
        "3. 如果参考信息不足以回答问题，请明确说明\n"
        "4. 回答格式示例：xxx[来源：xxx]"
    )

    chain = prompt | model | StrOutputParser()

    print("\n【交互式问答（带来源）】")
    print("提示：输入问题，AI 回答时会标注信息来源")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()

        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not question:
            print("请输入有效内容")
            continue

        # 检索相关文档（带来源信息）
        doc_texts = [item["content"] for item in knowledge_sources]
        results = retrieve_texts(question, doc_texts, top_k=3)

        # 构建带来源标注的上下文
        context_parts = []
        for i, (idx, score, doc) in enumerate(results):
            source = knowledge_sources[idx]["source"]
            context_parts.append(f"[来源：{source}]\n{doc}")
        context_with_source = "\n\n".join(context_parts)

        print(f"\n🔍 检索结果：")
        for i, (idx, score, doc) in enumerate(results):
            source = knowledge_sources[idx]["source"]
            preview = doc[:40] + "..." if len(doc) > 40 else doc
            print(f"   [{i+1}] 来源：{source} | 相似度：{score:.2f}")
            print(f"       {preview}")

        # 生成带来源的答案
        print("\n🤖 AI 回答：")
        response = chain.invoke({
            "context_with_source": context_with_source,
            "question": question
        })
        print(response)
        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 来源标注让答案可追溯、可验证")
    print("   2. 多来源交叉验证可提升答案可靠性")
    print("   3. 提示词中明确来源格式要求，确保输出规范")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain RAG 基础 - 实战案例")
    print("="*60)
    print("\n本示例演示 RAG（检索增强生成）的核心概念与实战用法")
    print("使用关键词匹配模拟检索，无需外部向量数据库")
    print("\n核心概念：")
    print("  • RAG：检索 + 增强 + 生成")
    print("  • 文档切分：将长文本拆分为可检索片段")
    print("  • 相似度检索：找到与问题最相关的内容")
    print("  • 来源标注：让答案可追溯、可验证")
    print("\n应用场景：")
    print("  • 知识库问答、文档问答、相似度搜索、可信问答")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 简单 RAG（从预设知识库中检索并回答）")
        print("  2. 文档问答（用户输入文档，AI 基于文档回答）")
        print("  3. 相似度搜索（找到与问题最相关的文本片段）")
        print("  4. 带来源的 RAG（回答时标注信息来源）")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_simple_rag()
        elif choice == "2":
            demo_document_qa()
        elif choice == "3":
            demo_similarity_search()
        elif choice == "4":
            demo_rag_with_source()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
