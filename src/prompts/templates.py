"""
提示词模板定义
集中管理所有提示词模板
"""
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


class PromptTemplates:
    """提示词模板集合"""
    
    # ==================== 基础对话模板 ====================
    
    @staticmethod
    def basic_chat() -> ChatPromptTemplate:
        """基础对话模板"""
        return ChatPromptTemplate.from_template("请介绍一下{topic}")
    
    @staticmethod
    def role_play() -> ChatPromptTemplate:
        """角色扮演模板"""
        return ChatPromptTemplate.from_messages([
            ("system", "你是一个{role}，请用专业的态度回答问题。"),
            ("human", "{question}")
        ])
    
    # ==================== 链式调用模板 ====================
    
    @staticmethod
    def simple_chain() -> ChatPromptTemplate:
        """简单链模板"""
        return ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
    
    @staticmethod
    def story_outline() -> ChatPromptTemplate:
        """故事大纲模板"""
        return ChatPromptTemplate.from_template(
            "为关于{topic}的故事生成一个简短的大纲"
        )
    
    @staticmethod
    def story_writer() -> ChatPromptTemplate:
        """故事写作模板"""
        return ChatPromptTemplate.from_template(
            "基于以下大纲写一个100字左右的小故事:\n{outline}"
        )
    
    # ==================== Agent 模板 ====================
    
    @staticmethod
    def agent_prompt() -> ChatPromptTemplate:
        """Agent 提示词模板"""
        return ChatPromptTemplate.from_messages([
            ("system", "你是一个有用的助手,可以使用工具来帮助用户。"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
    
    # ==================== RAG 模板 ====================
    
    @staticmethod
    def rag_qa() -> ChatPromptTemplate:
        """RAG 问答模板"""
        return ChatPromptTemplate.from_template(
            """基于以下上下文回答问题。如果上下文中没有相关信息,请说"我没有找到相关信息"。

上下文:
{context}

问题: {question}

答案:"""
        )
    
    @staticmethod
    def multi_query_generator() -> ChatPromptTemplate:
        """多查询生成模板"""
        return ChatPromptTemplate.from_template(
            """你是一个AI语言模型助手。你的任务是生成用户问题的3个不同版本,
以帮助用户从向量数据库中检索相关文档。

原始问题: {question}

请生成3个不同的问题版本(用换行分隔):"""
        )
    
    # ==================== 分析类模板 ====================
    
    @staticmethod
    def sentiment_analysis() -> ChatPromptTemplate:
        """情感分析模板"""
        return ChatPromptTemplate.from_template(
            """分析以下文本的情感:
{text}

请用一句话总结情感倾向。"""
        )
    
    @staticmethod
    def code_explanation() -> ChatPromptTemplate:
        """代码解释模板"""
        return ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的 Python 编程助手,擅长解释代码和技术概念"),
            ("human", "{code}")
        ])
    
    # ==================== 工具类模板 ====================
    
    @staticmethod
    def email_extractor() -> ChatPromptTemplate:
        """邮箱提取模板"""
        return ChatPromptTemplate.from_template(
            """从以下文本中提取邮箱地址:
{text}
只返回邮箱地址,不要其他内容。"""
        )
    
    @staticmethod
    def summarization() -> ChatPromptTemplate:
        """文本摘要模板"""
        return ChatPromptTemplate.from_template(
            """请总结以下文本:

{text}

摘要:"""
        )