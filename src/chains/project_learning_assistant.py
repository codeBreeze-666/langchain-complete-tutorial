"""
AI 学习助手 / AI Learning Assistant
====================================

完整的端到端AI学习助手项目，整合多种LangChain技术

技术整合 / Tech Integration:
- Chain: 知识点讲解（LLMChain + PromptTemplate）
- Agent: 学习路径推荐（Tool Calling Agent + 自定义工具）
- Tool: 搜索、计算等辅助工具（@tool 装饰器）
- Memory: 学习进度追踪（ConversationBufferMemory + 消息历史）
- Structured Output: 错题分析（PydanticOutputParser + 结构化模型）
- RAG: 智能答疑（关键词检索 + 上下文注入）

功能模块 / Features:
1. 智能答疑 - 搜索知识库回答问题（RAG）
2. 知识点讲解 - 生成详细讲解（Chain）
3. 学习路径推荐 - 个性化学习路线（Agent + Tool）
4. 错题分析 - 分析错误原因（Structured Output）
5. 学习进度追踪 - 历史记录和推荐（Memory）

应用场景 / Use Cases:
- 学生自主学习：随时提问、获取讲解、追踪进度
- 在线教育平台：智能答疑与个性化推荐
- 企业培训系统：员工学习路径规划与效果评估

Core Concepts:
- RAG (Retrieval-Augmented Generation): Retrieve relevant docs then generate answers
- Chain: Compose LLM calls with prompt templates and output parsers
- Agent: Autonomous tool selection for multi-step reasoning
- Memory: Persist conversation history for context-aware interactions
- Structured Output: Extract typed, validated data from LLM responses
"""

import os
import sys
import json
import sqlite3
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from src.utils.llm_loader import get_default_llm


# ============================================================
# SQLite 数据库配置
# ============================================================

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "learning_assistant.db")


def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库，创建表和示例数据"""
    conn = get_db()
    cursor = conn.cursor()

    # 创建表
    cursor.execute("""CREATE TABLE IF NOT EXISTS knowledge_base (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL,
        content TEXT NOT NULL,
        keywords TEXT
    )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS learning_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        question TEXT,
        answer TEXT,
        topic TEXT
    )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS study_progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL,
        score REAL DEFAULT 0,
        mastered INTEGER DEFAULT 0,
        last_review TEXT
    )""")

    # 插入示例数据（如果表为空）
    cursor.execute("SELECT COUNT(*) FROM knowledge_base")
    if cursor.fetchone()[0] == 0:
        sample_knowledge = [
            ("Python基础", "Python 基础语法：变量、数据类型（int, float, str, list, dict）、控制流（if/for/while）、"
             "函数定义（def）、模块导入（import）。Python 使用缩进表示代码块，不需要大括号。", "Python,基础,语法,变量,数据类型"),
            ("Python面向对象", "Python 面向对象：类定义（class）、继承（class Child(Parent)）、多态、封装（_protected, __private）、"
             "类方法（@classmethod）、静态方法（@staticmethod）、属性装饰器（@property）。", "Python,OOP,类,继承,多态,封装"),
            ("Python数据结构", "Python 数据结构：列表（list）有序可变、元组（tuple）有序不可变、字典（dict）键值对、"
             "集合（set）无序不重复。推导式（comprehension）是 Python 的特色语法。", "Python,数据结构,列表,元组,字典,集合"),
            ("机器学习", "机器学习基础：监督学习（分类、回归）、无监督学习（聚类、降维）、强化学习。"
             "常用算法：线性回归、逻辑回归、决策树、随机森林、SVM、K-means。"
             "评估指标：准确率、精确率、召回率、F1-score、AUC-ROC。", "机器学习,ML,监督学习,分类,回归,聚类"),
            ("深度学习", "深度学习基础：神经网络由输入层、隐藏层、输出层组成。激活函数（ReLU、Sigmoid、Tanh）、"
             "损失函数（MSE、CrossEntropy）、优化器（SGD、Adam）。反向传播算法是训练的核心。", "深度学习,DL,神经网络,激活函数,反向传播"),
            ("NLP", "自然语言处理（NLP）：分词、词性标注、命名实体识别、情感分析、文本分类、"
             "机器翻译、文本生成。预训练模型：BERT、GPT、T5。常用库：NLTK、spaCy、Hugging Face。", "NLP,自然语言处理,分词,BERT,GPT"),
            ("LangChain", "LangChain 框架：用于构建 LLM 应用的开源框架。核心组件包括 Model（模型接口）、"
             "Prompt（提示模板）、Chain（链式调用）、Agent（智能体）、Tool（工具）、Memory（记忆）、"
             "RAG（检索增强生成）。支持与 OpenAI、智谱、DeepSeek 等多种模型集成。", "LangChain,LLM,Agent,Chain,RAG,Memory"),
            ("数据科学", "数据科学工具链：NumPy（数值计算）、Pandas（数据分析）、Matplotlib/Seaborn（数据可视化）、"
             "Scikit-learn（机器学习）、Jupyter Notebook（交互式开发）。数据清洗和特征工程占项目 80% 时间。", "数据科学,NumPy,Pandas,Scikit-learn,可视化"),
            ("Git", "Git 版本控制：git init（初始化）、git add（暂存）、git commit（提交）、git push（推送）、"
             "git pull（拉取）、git branch（分支）、git merge（合并）、git rebase（变基）。"
             "分支策略：Git Flow、GitHub Flow、Trunk-Based Development。", "Git,版本控制,分支,合并,提交"),
            ("数据库", "数据库基础：SQL（Structured Query Language）用于关系型数据库操作。"
             "CRUD 操作：CREATE（创建）、READ（查询）、UPDATE（更新）、DELETE（删除）。"
             "常见数据库：MySQL、PostgreSQL、MongoDB、Redis。索引可以加速查询但增加写入开销。", "数据库,SQL,MySQL,CRUD,索引"),
        ]
        cursor.executemany(
            "INSERT INTO knowledge_base (topic, content, keywords) VALUES (?, ?, ?)",
            sample_knowledge
        )

    conn.commit()
    conn.close()


# ============================================================
# RAG 检索辅助函数
# ============================================================

def keyword_similarity(query: str, text: str) -> float:
    """基于关键词匹配计算文本相似度"""
    q_lower = query.lower()
    t_lower = text.lower()

    stopwords = {"的", "了", "是", "在", "有", "和", "与", "及", "等", "个",
                 "一", "这", "那", "不", "也", "都", "就", "要", "会", "能",
                 "什么", "怎么", "如何", "哪些", "为什么", "吗", "呢", "吧",
                 "我", "你", "他", "她", "它", "请", "想", "能"}

    keywords = set()
    for word in q_lower.replace("，", " ").replace("。", " ").replace("？", " ") \
                       .replace(",", " ").replace(".", " ").replace("?", " ") \
                       .split():
        word = word.strip()
        if word and word not in stopwords:
            keywords.add(word)
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


def retrieve_knowledge(query: str, top_k: int = 3) -> list[tuple[int, float, str]]:
    """从SQLite知识库中检索与查询最相关的内容"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM knowledge_base")
    rows = cursor.fetchall()
    conn.close()

    scored = [(row["id"], keyword_similarity(query, row["content"]), row["content"]) for row in rows]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ============================================================
# Structured Output 模型（错题分析）
# ============================================================

class MistakeAnalysis(BaseModel):
    """错题分析结果模型"""
    question_summary: str = Field(description="题目概要：简述题目考查的知识点")
    mistake_type: str = Field(description="错误类型：如概念混淆、计算失误、逻辑错误、审题不清、知识遗忘等")
    root_cause: str = Field(description="根本原因：分析导致错误的深层原因")
    correct_approach: str = Field(description="正确思路：给出正确的解题思路和步骤")
    related_knowledge: list[str] = Field(description="相关知识：需要复习或巩固的知识点列表")
    practice_suggestion: str = Field(description="练习建议：推荐的练习方向和方法")


# ============================================================
# Agent 工具定义（学习路径推荐）
# ============================================================

_SUBJECT_REGISTRY = {
    "python": {
        "name": "Python 编程",
        "levels": ["基础语法", "数据结构", "面向对象", "高级特性", "项目实战"],
        "estimated_hours": [20, 15, 20, 25, 40],
    },
    "machine_learning": {
        "name": "机器学习",
        "levels": ["数学基础", "经典算法", "模型评估", "特征工程", "项目实战"],
        "estimated_hours": [30, 25, 15, 20, 35],
    },
    "deep_learning": {
        "name": "深度学习",
        "levels": ["神经网络基础", "CNN", "RNN/Transformer", "框架实战", "前沿论文"],
        "estimated_hours": [25, 20, 25, 30, 40],
    },
    "nlp": {
        "name": "自然语言处理",
        "levels": ["文本预处理", "词向量", "序列模型", "预训练模型", "应用开发"],
        "estimated_hours": [15, 20, 25, 30, 35],
    },
    "data_science": {
        "name": "数据科学",
        "levels": ["Python 数据分析", "统计分析", "数据可视化", "机器学习入门", "项目实战"],
        "estimated_hours": [20, 25, 15, 25, 30],
    },
    "langchain": {
        "name": "LangChain 开发",
        "levels": ["LLM 基础", "Prompt 工程", "Chain 和 Agent", "RAG 和 Memory", "项目部署"],
        "estimated_hours": [10, 15, 20, 20, 25],
    },
}


@tool
def search_subject(subject: str) -> str:
    """搜索学习科目信息，包括学习阶段和预估时长

    Args:
        subject: 科目名称或关键词（如 python、机器学习、深度学习）

    Returns:
        科目的详细学习阶段和预估时长
    """
    subject_lower = subject.lower().replace(" ", "_")

    # 精确匹配
    if subject_lower in _SUBJECT_REGISTRY:
        info = _SUBJECT_REGISTRY[subject_lower]
        result = f"科目：{info['name']}\n学习阶段：\n"
        total = 0
        for i, (level, hours) in enumerate(zip(info["levels"], info["estimated_hours"])):
            result += f"  第{i+1}阶段：{level}（约 {hours} 小时）\n"
            total += hours
        result += f"总计预估：约 {total} 小时"
        return result

    # 模糊匹配
    matches = []
    for key, info in _SUBJECT_REGISTRY.items():
        if subject_lower in key or key in subject_lower or subject_lower in info["name"].lower():
            matches.append((key, info))

    if matches:
        result = f"找到 {len(matches)} 个相关科目：\n\n"
        for key, info in matches:
            result += f"【{info['name']}】\n"
            for i, (level, hours) in enumerate(zip(info["levels"], info["estimated_hours"])):
                result += f"  第{i+1}阶段：{level}（约 {hours} 小时）\n"
            result += "\n"
        return result

    available = "、".join(info["name"] for info in _SUBJECT_REGISTRY.values())
    return f"未找到科目「{subject}」。当前可用科目：{available}"


@tool
def estimate_study_time(current_level: str, target_level: str, daily_hours: float) -> str:
    """根据当前水平和目标水平估算学习时间

    Args:
        current_level: 当前水平描述（如 零基础、入门、进阶、高级）
        target_level: 目标水平描述（如 入门、进阶、高级、专家）
        daily_hours: 每天可投入的学习时间（小时）

    Returns:
        学习时间估算和计划建议
    """
    level_map = {"零基础": 0, "入门": 1, "进阶": 2, "高级": 3, "专家": 4}

    curr = level_map.get(current_level, 1)
    tgt = level_map.get(target_level, 2)

    if curr >= tgt:
        return f"当前水平「{current_level}」已达到或超过目标水平「{target_level}」，建议选择更高的目标。"

    gap = tgt - curr
    # 每个阶段平均 100 小时
    total_hours = gap * 100
    days = total_hours / max(daily_hours, 0.5)
    weeks = days / 7

    result = (
        f"学习时间估算：\n"
        f"  当前水平：{current_level}\n"
        f"  目标水平：{target_level}\n"
        f"  阶段差距：{gap} 个阶段\n"
        f"  预估总时长：约 {total_hours} 小时\n"
        f"  每日投入：{daily_hours} 小时\n"
        f"  预估天数：约 {days:.0f} 天（约 {weeks:.1f} 周）\n\n"
        f"建议：\n"
        f"  - 每周至少学习 5 天，保持连续性\n"
        f"  - 每个阶段结束做项目巩固\n"
        f"  - 定期回顾前面学过的内容"
    )
    return result


@tool
def get_learning_resources(topic: str) -> str:
    """获取学习资源推荐，包括书籍、课程和在线资源

    Args:
        topic: 学习主题（如 Python、机器学习、深度学习）

    Returns:
        推荐的学习资源列表
    """
    resources_db = {
        "python": "Python 学习资源：\n"
                  "📚 书籍：《Python编程：从入门到实践》《流畅的Python》\n"
                  "🎥 课程：廖雪峰Python教程、MIT 6.0001\n"
                  "🌐 在线：LeetCode（刷题）、Real Python、Python官方文档",

        "machine_learning": "机器学习学习资源：\n"
                           "📚 书籍：《机器学习》周志华、《统计学习方法》李航\n"
                           "🎥 课程：吴恩达Machine Learning、李宏毅ML课程\n"
                           "🌐 在线：Kaggle（竞赛）、Scikit-learn文档",

        "deep_learning": "深度学习学习资源：\n"
                        "📚 书籍：《深度学习》花书、《动手学深度学习》李沐\n"
                        "🎥 课程：吴恩达Deep Learning Specialization、李沐动手学\n"
                        "🌐 在线：Papers With Code、PyTorch官方教程",

        "nlp": "NLP 学习资源：\n"
               "📚 书籍：《Speech and Language Processing》《自然语言处理入门》\n"
               "🎥 课程：斯坦福CS224N、Hugging Face课程\n"
               "🌐 在线：Hugging Face Hub、ACL Anthology",

        "langchain": "LangChain 学习资源：\n"
                     "📚 书籍：《Building LLM Apps》、LangChain官方文档\n"
                     "🎥 课程：DeepLearning.AI LangChain课程\n"
                     "🌐 在线：LangChain Docs、LangSmith、LangGraph Docs",
    }

    topic_lower = topic.lower().replace(" ", "_")

    # 精确或模糊匹配
    for key, val in resources_db.items():
        if key in topic_lower or topic_lower in key:
            return val

    # 默认通用推荐
    return (
        f"通用学习资源推荐：\n"
        f"📚 书籍：根据主题选择经典教材\n"
        f"🎥 课程：Coursera、B站、YouTube 教育频道\n"
        f"🌐 在线：GitHub、Stack Overflow、相关技术社区\n"
        f"💡 建议：理论 + 实践结合，多动手做项目"
    )


# ============================================================
# Memory - 学习进度追踪器（SQLite持久化）
# ============================================================

class LearningTracker:
    """学习进度追踪器，记录用户的学习历史和偏好，使用SQLite持久化"""

    def __init__(self):
        self.chat_history: list = []            # 对话历史（内存中保留，供Agent使用）
        self.topics_explored: set = set()       # 已探索的主题
        self.mistakes_analyzed: int = 0         # 错题分析次数
        self.questions_asked: int = 0           # 提问次数

        # 从SQLite加载统计信息
        self._load_stats()

    def _load_stats(self):
        """从SQLite加载统计信息"""
        conn = get_db()
        cursor = conn.cursor()

        # 统计提问次数
        cursor.execute("SELECT COUNT(*) FROM learning_history WHERE question IS NOT NULL")
        self.questions_asked = cursor.fetchone()[0]

        # 统计错题分析次数（通过study_progress中mastered=0的记录估算）
        cursor.execute("SELECT COUNT(*) FROM study_progress WHERE mastered = 0 AND score > 0")
        self.mistakes_analyzed = cursor.fetchone()[0]

        # 加载已探索的主题
        cursor.execute("SELECT DISTINCT topic FROM study_progress WHERE topic IS NOT NULL")
        self.topics_explored = {row["topic"] for row in cursor.fetchall() if row["topic"]}

        conn.close()

    def record(self, action: str, detail: str):
        """记录一次学习活动到SQLite"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO learning_history (timestamp, question, answer, topic) VALUES (?, ?, ?, ?)",
            (timestamp, action, detail, detail[:10])
        )
        conn.commit()
        conn.close()

    def add_chat(self, human_msg: str, ai_msg: str):
        """添加对话记录到记忆"""
        self.chat_history.append(HumanMessage(content=human_msg))
        self.chat_history.append(AIMessage(content=ai_msg))
        # 保留最近 20 条对话
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

    def save_qa_record(self, question: str, answer: str, topic: str = ""):
        """保存问答记录到SQLite"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO learning_history (timestamp, question, answer, topic) VALUES (?, ?, ?, ?)",
            (timestamp, question, answer[:500], topic)
        )
        conn.commit()
        conn.close()

    def save_mistake(self, question_summary: str, mistake_type: str, score: float = 0):
        """保存错题分析记录到SQLite"""
        conn = get_db()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO study_progress (topic, score, mastered, last_review) VALUES (?, ?, 0, ?)",
            (mistake_type, score, now)
        )
        conn.commit()
        conn.close()

    def update_progress(self, topic: str, score: float, mastered: int = 0):
        """更新学习进度到SQLite"""
        conn = get_db()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 检查是否已有该主题的进度记录
        cursor.execute("SELECT id FROM study_progress WHERE topic = ?", (topic,))
        row = cursor.fetchone()
        if row:
            cursor.execute(
                "UPDATE study_progress SET score = ?, mastered = ?, last_review = ? WHERE topic = ?",
                (score, mastered, now, topic)
            )
        else:
            cursor.execute(
                "INSERT INTO study_progress (topic, score, mastered, last_review) VALUES (?, ?, ?, ?)",
                (topic, score, mastered, now)
            )
        conn.commit()
        conn.close()

    def get_summary(self) -> str:
        """获取学习进度摘要"""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM learning_history")
        total_records = cursor.fetchone()[0]
        conn.close()

        if total_records == 0:
            return "暂无学习记录，开始你的学习之旅吧！"

        lines = [f"📊 学习进度报告（共 {total_records} 条记录）"]
        lines.append(f"  - 提问次数：{self.questions_asked}")
        lines.append(f"  - 错题分析：{self.mistakes_analyzed}")
        lines.append(f"  - 探索主题：{', '.join(self.topics_explored) if self.topics_explored else '无'}")

        # 最近 5 条记录
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp, question, answer FROM learning_history ORDER BY id DESC LIMIT 5"
        )
        recent = cursor.fetchall()
        conn.close()

        lines.append("\n  最近学习活动：")
        for entry in reversed(recent):
            action = entry["question"] if entry["question"] else ""
            detail = entry["answer"] if entry["answer"] else ""
            lines.append(f"    [{entry['timestamp']}] {action}：{detail[:30]}")

        return "\n".join(lines)

    def get_recommendation(self) -> str:
        """基于学习历史推荐下一步"""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM learning_history")
        total_records = cursor.fetchone()[0]
        conn.close()

        if total_records == 0:
            return "建议从「智能答疑」开始，输入你感兴趣的编程或AI问题。"

        # 统计活动类型
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT question, COUNT(*) as cnt FROM learning_history GROUP BY question")
        rows = cursor.fetchall()
        conn.close()

        action_counts = {row["question"]: row["cnt"] for row in rows}

        lines = ["🎯 个性化推荐："]

        if action_counts.get("智能答疑", 0) >= 3 and "知识点讲解" not in action_counts:
            lines.append("  你已经问了不少问题，试试「知识点讲解」获取系统化的知识梳理。")

        if action_counts.get("错题分析", 0) == 0 and self.questions_asked >= 2:
            lines.append("  建议使用「错题分析」功能，帮助你发现和纠正知识盲点。")

        if action_counts.get("学习路径推荐", 0) == 0:
            lines.append("  还没有规划学习路径？试试「学习路径推荐」获取个性化路线。")

        if len(self.topics_explored) >= 3:
            lines.append(f"  你已探索 {len(self.topics_explored)} 个主题，继续保持！尝试跨领域关联学习。")

        if len(lines) == 1:
            lines.append("  继续使用各项功能，系统会根据你的学习模式给出更精准的推荐。")

        return "\n".join(lines)


# ============================================================
# 1. 智能答疑 - RAG 检索知识库并回答
# ============================================================

def feature_smart_qa(tracker: LearningTracker):
    """功能1：智能答疑 - 搜索知识库回答问题（整合 RAG）"""
    print("\n" + "=" * 60)
    print("  智能答疑 - 搜索知识库回答问题")
    print("=" * 60)
    print("\n💡 技术整合：RAG（检索增强生成）")
    print("   先从知识库检索相关内容，再让 LLM 基于上下文生成答案")
    print("   有效减少 AI 幻觉，确保回答有据可依")

    # 从SQLite读取知识库数量
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM knowledge_base")
    kb_count = cursor.fetchone()[0]
    conn.close()
    print(f"\n📚 知识库包含 {kb_count} 条知识（Python、ML、DL、NLP、LangChain 等）")

    model = get_default_llm()

    prompt = ChatPromptTemplate.from_template(
        "你是一个专业的学习助手。请根据以下参考信息回答学生的问题。\n\n"
        "参考信息：\n{context}\n\n"
        "学生问题：{question}\n\n"
        "要求：\n"
        "1. 优先根据参考信息回答，如果参考信息不足可以补充自己的知识\n"
        "2. 回答要清晰易懂，适合学习者理解\n"
        "3. 如果涉及代码，给出简短示例\n"
        "4. 在回答末尾标注信息来源"
    )

    chain = prompt | model | StrOutputParser()

    print("\n【交互式答疑】")
    print("输入你的学习问题，AI 从知识库检索并回答")
    print("输入 '退出' 返回主菜单\n")

    while True:
        question = input("你的问题：").strip()

        if question.lower() in ["退出", "exit", "quit", "0"]:
            break

        if not question:
            print("请输入有效问题")
            continue

        try:
            # RAG 第一步：从SQLite知识库检索
            results = retrieve_knowledge(question, top_k=3)
            context = "\n\n".join(
                f"[知识片段{i+1}] {doc}" for i, (_, _, doc) in enumerate(results)
            )

            # 显示检索结果
            print(f"\n🔍 检索到 {len(results)} 条相关知识：")
            for i, (idx, score, doc) in enumerate(results):
                preview = doc[:45] + "..." if len(doc) > 45 else doc
                print(f"   [{i+1}] 相似度={score:.2f} | {preview}")

            # RAG 第二步：生成
            print("\n🤖 AI 回答：")
            response = chain.invoke({"context": context, "question": question})
            print(response)

            # 记录学习活动到SQLite
            tracker.record("智能答疑", question)
            tracker.save_qa_record(question, response[:500], question[:10])
            tracker.questions_asked += 1
            tracker.topics_explored.add(question[:10])
            tracker.add_chat(question, response[:100])

        except Exception as e:
            print(f"❌ 回答失败：{e}")

        print("\n" + "-" * 60)


# ============================================================
# 2. 知识点讲解 - Chain 生成详细讲解
# ============================================================

def feature_topic_explanation(tracker: LearningTracker):
    """功能2：知识点讲解 - 输入主题，AI 生成详细讲解（整合 Chain）"""
    print("\n" + "=" * 60)
    print("  知识点讲解 - 生成详细讲解")
    print("=" * 60)
    print("\n💡 技术整合：Chain（LLMChain + PromptTemplate + StrOutputParser）")
    print("   使用精心设计的提示词模板，让 AI 生成结构化的知识讲解")
    print("   Chain = Prompt | Model | OutputParser，LangChain 的核心编排模式")

    model = get_default_llm()

    prompt = ChatPromptTemplate.from_template(
        "你是一位经验丰富的技术讲师，擅长将复杂概念讲解得深入浅出。\n\n"
        "请为以下主题生成详细的知识点讲解：\n\n"
        "主题：{topic}\n"
        "学习者水平：{level}\n\n"
        "讲解要求：\n"
        "1. 【核心概念】用简洁的语言概括这个主题的核心要点\n"
        "2. 【详细讲解】分层次、循序渐进地展开讲解\n"
        "3. 【代码示例】给出一个简短但完整的示例代码（如果适用）\n"
        "4. 【常见误区】列出学习者容易犯的错误和混淆点\n"
        "5. 【延伸学习】推荐相关的进阶主题\n\n"
        "注意：讲解要适合{level}水平的学习者，语言通俗易懂。"
    )

    chain = prompt | model | StrOutputParser()

    print("\n【交互式知识点讲解】")
    print("输入你想学习的主题，AI 生成详细讲解")
    print("输入 '退出' 返回主菜单\n")

    while True:
        topic = input("学习主题：").strip()

        if topic.lower() in ["退出", "exit", "quit", "0"]:
            break

        if not topic:
            print("请输入有效主题")
            continue

        # 选择水平
        print("\n选择你的水平：")
        print("  1. 零基础")
        print("  2. 入门")
        print("  3. 进阶")
        level_choice = input("请选择 (1-3，默认2)：").strip()
        level_map = {"1": "零基础", "2": "入门", "3": "进阶"}
        level = level_map.get(level_choice, "入门")

        try:
            print(f"\n📖 正在生成「{topic}」的知识讲解（{level}水平）...\n")
            response = chain.invoke({"topic": topic, "level": level})
            print(response)

            # 记录到SQLite
            tracker.record("知识点讲解", f"{topic}（{level}）")
            tracker.topics_explored.add(topic)
            tracker.update_progress(topic, 0.5, mastered=0)

        except Exception as e:
            print(f"❌ 讲解生成失败：{e}")

        print("\n" + "-" * 60)


# ============================================================
# 3. 学习路径推荐 - Agent + Tool
# ============================================================

def feature_learning_path(tracker: LearningTracker):
    """功能3：学习路径推荐 - Agent 自动选择工具推荐学习路线"""
    print("\n" + "=" * 60)
    print("  学习路径推荐 - 个性化学习路线")
    print("=" * 60)
    print("\n💡 技术整合：Agent（Tool Calling Agent）+ Tool（自定义工具）")
    print("   Agent 根据用户目标自动选择合适的工具：")
    print("   - search_subject：搜索科目信息和学习阶段")
    print("   - estimate_study_time：估算学习时间")
    print("   - get_learning_resources：获取学习资源推荐")

    model = get_default_llm()
    tools = [search_subject, estimate_study_time, get_learning_resources]

    agent = create_react_agent(model, tools, state_modifier="你是一个专业的学习规划师。根据用户的学习目标，综合利用可用工具为其制定个性化的学习路径。你需要：1）搜索相关科目信息 2）估算学习时间 3）推荐学习资源。最终给出一份完整的学习路径规划，包括学习阶段、时间安排和资源推荐。")

    print("\n【交互式学习路径推荐】")
    print("输入你的学习目标，AI 自动搜索科目、估算时间、推荐资源")
    print("示例：'我想从零开始学习机器学习，每天2小时'")
    print("输入 '退出' 返回主菜单\n")

    while True:
        goal = input("你的学习目标：").strip()

        if goal.lower() in ["退出", "exit", "quit", "0"]:
            break

        if not goal:
            print("请输入有效目标")
            continue

        try:
            result = agent.invoke({"messages": tracker.chat_history + [("user", goal)]})
            final_message = result["messages"][-1]

            print(f"\n🗺️ 学习路径规划：\n{final_message.content}")

            # 记录到SQLite
            tracker.record("学习路径推荐", goal)
            tracker.add_chat(goal, final_message.content[:100])

        except Exception as e:
            print(f"❌ 推荐失败：{e}")

        print("\n" + "-" * 60)


# ============================================================
# 4. 错题分析 - Structured Output
# ============================================================

def feature_mistake_analysis(tracker: LearningTracker):
    """功能4：错题分析 - 输入错题，AI 分析原因（整合 Structured Output）"""
    print("\n" + "=" * 60)
    print("  错题分析 - 分析错误原因")
    print("=" * 60)
    print("\n💡 技术整合：Structured Output（PydanticOutputParser）")
    print("   使用 Pydantic 模型定义分析结果的结构")
    print("   AI 输出经过解析和验证，确保数据格式正确、字段完整")
    print("   字段包括：错误类型、根本原因、正确思路、相关知识、练习建议")

    model = get_default_llm()
    parser = PydanticOutputParser(pydantic_object=MistakeAnalysis)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一位经验丰富的学习诊断专家，擅长分析学生的错误并给出改进建议。\n\n"
         "{format_instructions}\n\n"
         "请仔细分析学生的错题，给出结构化的诊断报告。"),
        ("human", "{mistake_info}")
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | model | parser

    print("\n【交互式错题分析】")
    print("输入你的错题信息（题目 + 你的答案 + 正确答案），AI 分析错误原因")
    print("示例：'题目：Python中list和tuple的区别？我的答案：都是有序集合，没区别。正确答案：list可变，tuple不可变'")
    print("输入 '退出' 返回主菜单\n")

    while True:
        mistake_info = input("错题信息：").strip()

        if mistake_info.lower() in ["退出", "exit", "quit", "0"]:
            break

        if not mistake_info:
            print("请输入有效内容")
            continue

        try:
            result = chain.invoke({"mistake_info": mistake_info})

            print("\n📋 错题分析报告：")
            print("=" * 40)
            print(f"📌 题目概要：{result.question_summary}")
            print(f"❌ 错误类型：{result.mistake_type}")
            print(f"🔍 根本原因：{result.root_cause}")
            print(f"✅ 正确思路：{result.correct_approach}")
            print(f"📚 相关知识：{', '.join(result.related_knowledge)}")
            print(f"💡 练习建议：{result.practice_suggestion}")
            print("=" * 40)

            # 保存错题记录到SQLite
            tracker.record("错题分析", f"{result.mistake_type}：{result.question_summary[:20]}")
            tracker.save_mistake(result.question_summary, result.mistake_type, score=0)
            tracker.mistakes_analyzed += 1

        except Exception as e:
            print(f"❌ 分析失败：{e}")
            print("提示：请确保输入包含题目、你的答案和正确答案")

        print("\n" + "-" * 60)


# ============================================================
# 5. 学习进度 - Memory 追踪
# ============================================================

def feature_learning_progress(tracker: LearningTracker):
    """功能5：学习进度 - 查看历史记录和推荐（整合 Memory）"""
    print("\n" + "=" * 60)
    print("  学习进度 - 历史记录和推荐")
    print("=" * 60)
    print("\n💡 技术整合：Memory（ConversationBufferMemory + 学习追踪器）")
    print("   自动记录每次学习活动，包括提问、讲解、路径推荐、错题分析")
    print("   基于历史数据生成学习摘要和个性化推荐")

    # 显示学习摘要（从SQLite读取）
    print("\n" + tracker.get_summary())

    # 显示个性化推荐
    print("\n" + tracker.get_recommendation())

    # 对话记忆分析（如果有足够历史）
    if len(tracker.chat_history) >= 4:
        print("\n💬 最近对话摘要：")
        recent = tracker.chat_history[-6:]  # 最近 3 轮对话
        for i in range(0, len(recent), 2):
            if i + 1 < len(recent):
                human = recent[i].content[:40]
                ai = recent[i+1].content[:40]
                print(f"  你：{human}...")
                print(f"  AI：{ai}...")
    else:
        print("\n💬 对话记忆较少，继续使用功能后这里会显示对话摘要。")

    # 从SQLite读取学习进度
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM learning_history")
    total_records = cursor.fetchone()[0]
    conn.close()

    # 进阶：用 LLM 基于历史做深度分析
    if total_records >= 3:
        print("\n🤖 AI 深度分析中...")
        try:
            model = get_default_llm()

            # 从SQLite读取历史记录
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT timestamp, question, answer FROM learning_history ORDER BY id DESC LIMIT 20"
            )
            rows = cursor.fetchall()
            conn.close()

            history_text = "\n".join(
                f"[{row['timestamp']}] {row['question']}：{row['answer']}" for row in rows
            )

            prompt = ChatPromptTemplate.from_template(
                "基于以下学习历史记录，给出简短的学习评价和建议（不超过 200 字）：\n\n"
                "{history}\n\n"
                "评价要点：学习频率、知识覆盖面、薄弱环节、下一步建议。"
            )
            chain = prompt | model | StrOutputParser()
            analysis = chain.invoke({"history": history_text})
            print(f"\n📝 AI 评价：\n{analysis}")

        except Exception as e:
            print(f"❌ 深度分析失败：{e}")

    print("\n" + "-" * 60)

    # 子菜单
    while True:
        print("\n选项：")
        print("  1. 清空学习记录")
        print("  2. 导出学习记录")
        print("  0. 返回主菜单")

        sub_choice = input("\n请选择：").strip()
        if sub_choice == "1":
            confirm = input("确认清空所有学习记录？(y/n)：").strip().lower()
            if confirm == "y":
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM learning_history")
                cursor.execute("DELETE FROM study_progress")
                conn.commit()
                conn.close()
                tracker.chat_history.clear()
                tracker.topics_explored.clear()
                tracker.mistakes_analyzed = 0
                tracker.questions_asked = 0
                print("✅ 学习记录已清空")
            else:
                print("已取消")
        elif sub_choice == "2":
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM learning_history")
            count = cursor.fetchone()[0]
            conn.close()

            if count == 0:
                print("暂无记录可导出")
            else:
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute("SELECT id, timestamp, question, answer, topic FROM learning_history ORDER BY id")
                rows = cursor.fetchall()
                conn.close()

                export_data = {
                    "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_records": len(rows),
                    "questions_asked": tracker.questions_asked,
                    "mistakes_analyzed": tracker.mistakes_analyzed,
                    "topics_explored": list(tracker.topics_explored),
                    "records": [
                        {
                            "id": row["id"],
                            "time": row["timestamp"],
                            "action": row["question"],
                            "detail": row["answer"],
                        }
                        for row in rows
                    ],
                }
                filename = f"learning_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                print(f"✅ 学习记录已导出到 {filename}")
        elif sub_choice == "0":
            break


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    # 初始化数据库
    init_db()

    print("\n" + "=" * 60)
    print("  AI 学习助手 - 端到端实战项目")
    print("=" * 60)
    print("\n完整的 AI 学习助手，整合 RAG / Chain / Agent / Tool / Memory / Structured Output")

    print("\n功能模块：")
    print("  1. 智能答疑（输入问题，AI 搜索知识库回答）    [RAG]")
    print("  2. 知识点讲解（输入主题，AI 生成详细讲解）     [Chain]")
    print("  3. 学习路径推荐（输入目标，AI 推荐学习路线）   [Agent+Tool]")
    print("  4. 错题分析（输入错题，AI 分析原因）           [Structured Output]")
    print("  5. 学习进度（查看历史和推荐）                  [Memory]")

    print("\n应用场景：学生自主学习 / 在线教育 / 企业培训")

    # 创建学习进度追踪器（整个会话共享）
    tracker = LearningTracker()

    while True:
        print("\n" + "=" * 60)
        print("  AI 学习助手")
        print("=" * 60)
        print("  1. 智能答疑（输入问题，AI 搜索知识库回答）")
        print("  2. 知识点讲解（输入主题，AI 生成详细讲解）")
        print("  3. 学习路径推荐（输入目标，AI 推荐学习路线）")
        print("  4. 错题分析（输入错题，AI 分析原因）")
        print("  5. 学习进度（查看历史和推荐）")
        print("\n  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-5): ").strip()

        if choice == "1":
            feature_smart_qa(tracker)
        elif choice == "2":
            feature_topic_explanation(tracker)
        elif choice == "3":
            feature_learning_path(tracker)
        elif choice == "4":
            feature_mistake_analysis(tracker)
        elif choice == "5":
            feature_learning_progress(tracker)
        elif choice == "0":
            # 退出前显示学习总结
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM learning_history")
            has_records = cursor.fetchone()[0] > 0
            conn.close()
            if has_records:
                print("\n" + tracker.get_summary())
            print("\n感谢使用 AI 学习助手！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
