# 第八章：RAG篇

## 8.1 RAG基础（rag_basics.py）

### 知识点：简单RAG、文档问答、相似度搜索、带来源RAG

**运行方式：**

```bash
python src/chains/rag_basics.py
```

**核心概念：**

- **RAG（Retrieval-Augmented Generation）**：检索增强生成，先检索相关文档，再让 LLM 基于上下文生成答案
- **文档切分**：将长文本拆分为可检索的片段
- **相似度检索**：找到与问题最相关的文本片段（本示例使用关键词匹配模拟）
- **上下文注入**：将检索结果作为上下文提供给 LLM

**RAG 工作流程：**

```
用户提问 → 检索相关文档 → 注入上下文 → LLM 基于上下文生成答案
```

> 本示例使用关键词匹配模拟检索功能，不依赖外部向量数据库，便于理解 RAG 核心原理。

---

### 示例1：简单RAG — 从预设知识库中检索并回答

**功能说明：** 从预设的6条技术知识库中检索相关内容，基于检索结果生成答案。

**关键代码：**

```python
# 预设知识库
DEFAULT_KNOWLEDGE_BASE = [
    "Python 是一种高级编程语言...",
    "LangChain 是一个用于开发大语言模型应用的开源框架...",
    "RAG（检索增强生成）是一种结合检索和生成的技术方案...",
    ...
]

# 第一步：检索相关文档
results = retrieve_texts(question, knowledge_base, top_k=3)
context = "\n\n".join([f"[片段{i+1}] {doc}" for i, (_, _, doc) in enumerate(results)])

# 第二步：基于检索结果生成答案
prompt = ChatPromptTemplate.from_template(
    "请根据以下参考信息回答用户的问题。\n\n"
    "参考信息：\n{context}\n\n"
    "问题：{question}\n\n"
    "要求：只根据参考信息回答，如果参考信息中没有相关内容，请说明"
)
chain = prompt | model | StrOutputParser()
response = chain.invoke({"context": context, "question": question})
```

**实战要点：**

1. RAG 的核心是**"先检索，后生成"**
2. 检索质量直接决定回答质量
3. 提示词中明确要求"只根据参考信息回答"可减少幻觉
4. 本示例使用关键词匹配，生产环境建议使用向量嵌入 + 余弦相似度

---

### 示例2：文档问答 — 用户输入文档后基于文档回答

**功能说明：** 用户输入文档内容（或使用预设示例），AI 自动切分后基于文档回答问题。

**关键代码：**

```python
# 文档切分：按句号切分，每个片段不超过200字
chunks = []
for paragraph in document_text.split("\n"):
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

# 基于切分后的片段检索并回答
prompt = ChatPromptTemplate.from_template(
    "你是一个文档问答助手。请严格根据以下文档内容回答问题。\n\n"
    "文档内容：\n{context}\n\n"
    "问题：{question}\n\n"
    '注意：如果文档中没有相关信息，请明确回答"文档中未提及此内容"。'
)
```

**实战要点：**

1. 文档切分是 RAG 的关键预处理步骤
2. 切分粒度要适中：太粗会引入噪声，太细会丢失上下文
3. 提示词要求"严格基于文档回答"可约束模型行为
4. 本示例使用简单的句号切分，生产环境推荐 `RecursiveCharacterTextSplitter`

---

### 示例3：相似度搜索 — 找到与问题最相关的文本片段

**功能说明：** 纯检索演示，展示关键词匹配的相似度计算方式和搜索结果排名。

**关键代码：**

```python
def keyword_similarity(query, text):
    """基于关键词匹配计算文本相似度"""
    # 停用词过滤 + 2-gram/3-gram 生成
    keywords = set()
    for word in query.split():
        if word not in stopwords:
            keywords.add(word)
            # 中文 n-gram
            for i in range(len(word) - 1):
                keywords.add(word[i:i+2])

    hits = sum(1 for kw in keywords if kw in text.lower())
    return hits / len(keywords)

# 搜索并排序
results = retrieve_texts(query, knowledge_base, top_k=3)
for idx, score, doc in results:
    print(f"相似度={score:.2f} | {doc[:50]}")
```

**实战要点：**

1. 关键词匹配简单有效，但无法理解语义相似性
2. 中文文本需要额外生成 2-gram 和 3-gram 提升匹配率
3. 生产环境推荐使用 **Embedding + 余弦相似度**
4. `top_k` 参数需要根据场景调优

---

### 示例4：带来源的RAG — 回答时标注信息来源

**功能说明：** 检索结果附带来源信息（如"LangChain 官方文档"），回答时标注来源，提升可信度。

**关键代码：**

```python
# 使用带来源标记的知识库
knowledge_sources = [
    {"source": "《Python编程：从入门到实践》", "content": DEFAULT_KNOWLEDGE_BASE[0]},
    {"source": "LangChain 官方文档", "content": DEFAULT_KNOWLEDGE_BASE[1]},
    ...
]

# 构建带来源标注的上下文
context_parts = []
for i, (idx, score, doc) in enumerate(results):
    source = knowledge_sources[idx]["source"]
    context_parts.append(f"[来源：{source}]\n{doc}")

# 提示词要求标注来源
prompt = ChatPromptTemplate.from_template(
    "请根据以下参考信息回答问题，并在回答中标注信息来源。\n\n"
    "要求：回答时必须标注来源，格式为：[来源：xxx]"
)
```

**实战要点：**

1. 来源标注让答案**可追溯、可验证**
2. 多来源交叉验证可提升答案可靠性
3. 提示词中明确来源格式要求，确保输出规范
4. 企业级 RAG 系统的必备功能

---

## 8.2 文档加载与切分（document_loader.py）

### 知识点：文本切分(RecursiveCharacterTextSplitter)、文档加载、切片管理、自定义切分器

**运行方式：**

```bash
python src/chains/document_loader.py
```

**核心概念：**

- **RecursiveCharacterTextSplitter**：递归字符切分器，按分隔符优先级逐级拆分（最常用）
- **CharacterTextSplitter**：单字符切分器，按单一分隔符切分
- **Document 对象**：LangChain 文档的统一表示（`page_content` + `metadata`）
- **自定义切分器**：继承 `TextSplitter` 实现业务特定的切分规则

---

### 示例1：文本切分 — 递归字符切分 vs 单字符切分

**功能说明：** 演示 `RecursiveCharacterTextSplitter` 和 `CharacterTextSplitter` 的区别，支持自定义参数和分隔符优先级。

**关键代码：**

```python
# 递归字符切分（推荐）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,        # 每个切片最大长度
    chunk_overlap=50,      # 相邻切片重叠字符数
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 分隔符优先级
    length_function=len,
)

# 单字符切分（简单直接）
splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separator="\n",
)
```

**两种切分器对比：**

| 特性 | RecursiveCharacterTextSplitter | CharacterTextSplitter |
|------|-------------------------------|----------------------|
| 分隔符 | 多级优先级逐级尝试 | 单一分隔符 |
| 切片均匀度 | 更均匀 | 可能过短或过长 |
| 适用场景 | 通用推荐 | 简单场景 |
| 语义完整性 | 更好（优先在段落/句子边界切分） | 较差 |

**实战要点：**

1. `RecursiveCharacterTextSplitter` 是最常用的切分器，按优先级逐级拆分
2. `chunk_overlap` 保证上下文连续性，一般设为 chunk_size 的 **10%~25%**
3. 选择合适的分隔符优先级对中文文本切分效果至关重要
4. `chunk_overlap` 必须 **小于 chunk_size**

---

### 示例2：文档加载 — 从不同来源加载文档

**功能说明：** 演示从纯文本、模拟文件、JSON 数据等多种来源创建 `Document` 对象。

**关键代码：**

```python
from langchain_core.documents import Document

# 从纯文本创建
doc = Document(
    page_content=SAMPLE_LONG_TEXT,
    metadata={"source": "内置示例文本", "type": "纯文本", "char_count": len(SAMPLE_LONG_TEXT)}
)

# 从 JSON 数据创建
for item in json_data:
    doc = Document(
        page_content=item["content"],
        metadata={"source": "json_data", "title": item["title"], "author": item["author"]}
    )

# 加载后切分（保留元数据）
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_documents([doc])  # split_documents 比 split_text 更好
```

**实战要点：**

1. `Document` 对象是 LangChain 文档处理的核心数据结构
2. `metadata` 存储来源信息，切分后会**自动继承**到每个切片
3. `split_documents` 方法比 `split_text` 更好，因为会保留元数据
4. 实际项目中使用 `PyPDFLoader`、`TextLoader` 等加载真实文件

---

### 示例3：切片管理 — 管理文档切片的元数据

**功能说明：** 对切分后的文档切片添加自定义元数据，支持按分类、重要性、作者等维度过滤，并提供切片质量评估。

**关键代码：**

```python
# 切分后添加自定义元数据
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["chunk_size"] = len(chunk.page_content)

# 按分类过滤
filtered = [c for c in chunks if c.metadata.get("category") == "编程语言"]

# 切片质量评估
sizes = [c.metadata["chunk_size"] for c in chunks]
avg_size = sum(sizes) / len(sizes)
short_chunks = [c for c in chunks if c.metadata["chunk_size"] < 30]  # 检测过短切片
```

**实战要点：**

1. 元数据是切片管理的基础，支持溯源、过滤和分类
2. 切分后自动继承原文档元数据，也可手动添加自定义字段
3. 切片质量评估帮助发现过短切片、长度不均等问题
4. 元数据过滤在向量数据库中可大幅提升检索精度

---

### 示例4：自定义切分器 — 按特定规则切分文本

**功能说明：** 实现三种自定义切分器（按标题切分 Markdown、按代码块切分 Python、按分隔线切分），以及组合切分策略。

**关键代码：**

```python
from langchain_text_splitters import TextSplitter

class HeadingSplitter(TextSplitter):
    """按标题（# 开头）切分 Markdown 文本"""
    def split_text(self, text):
        chunks = []
        current_chunk = ""
        for line in text.split("\n"):
            if line.startswith("#") and current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

class CodeBlockSplitter(TextSplitter):
    """按函数/类定义切分 Python 代码"""
    def split_text(self, text):
        ...

# 组合切分：先按标题切分，再按字符限制二次切分
heading_splitter = HeadingSplitter()
first_pass = heading_splitter.split_text(sample_markdown)

final_chunks = []
for chunk in first_pass:
    if len(chunk) <= max_size:
        final_chunks.append(chunk)
    else:
        sub_splitter = RecursiveCharacterTextSplitter(chunk_size=max_size, chunk_overlap=30)
        final_chunks.extend(sub_splitter.split_text(chunk))
```

**实战要点：**

1. 继承 `TextSplitter` 并实现 `split_text` 方法即可自定义切分器
2. **组合切分**是实战常用策略：先按结构切分，再按长度限制二次切分
3. 代码切分要按函数/类边界切分，避免切断逻辑单元
4. Markdown 切分优先按标题层级切分，保持语义完整性

---

## 8.3 RAG Agent（rag_agent.py）

### 知识点：RAG Agent、多源RAG、对话式RAG、RAG+工具

**运行方式：**

```bash
python src/chains/rag_agent.py
```

**核心概念：**

| 模式 | 核心机制 | 与纯 RAG 链的区别 |
|------|---------|------------------|
| RAG Agent | 检索封装为 @tool，Agent 自主决策 | Agent 可判断是否需要检索 |
| 多源 RAG | 不同知识源封装为不同工具 | Agent 按需选择知识源 |
| 对话式 RAG | chat_history + 检索工具 | 多轮对话中持续利用检索结果 |
| RAG + 工具 | 检索工具 + 功能工具 | 既查信息又做计算 |

---

### 示例1：RAG Agent — 检索封装为工具

**功能说明：** 将检索功能封装为 `@tool`，Agent 自主决定何时检索。与纯 RAG 链不同，Agent 可以先思考再决定是否需要检索。

**关键代码：**

```python
@tool
def search_tech_knowledge(query: str) -> str:
    """搜索技术知识库，当用户询问Python、LangChain、RAG等技术问题时使用此工具"""
    results = retrieve_texts(query, knowledge_base, top_k=2)
    if not results or results[0][1] == 0.0:
        return "未在知识库中找到相关内容"
    return "\n\n".join([f"[知识片段{i+1}] {doc}" for i, (_, _, doc) in enumerate(results)])

# Agent 可以判断问题是否需要检索
prompt = ChatPromptTemplate.from_messages([
    ("system", "当用户的问题需要参考知识库时，使用搜索工具获取信息；"
               "当问题属于常识或你已有把握回答时，可以直接回答。"),
    ...
])
```

**实战要点：**

1. RAG Agent = 检索工具 + Agent 推理，比纯 RAG 链更灵活
2. Agent 能判断问题是否需要检索，避免无意义检索（如"1+1等于几"）
3. 工具的 docstring 决定 Agent 何时使用检索
4. Agent 可以选择直接回答或先检索再回答

---

### 示例2：多源RAG — Agent从多个知识源中选择性检索

**功能说明：** 为技术文档、公司制度、产品信息三个知识源创建独立的检索工具，Agent 根据问题类型自动选择。

**关键代码：**

```python
@tool
def search_tech_docs(query: str) -> str:
    """搜索技术文档知识库，当用户询问编程语言、框架、AI技术等技术开发问题时使用"""

@tool
def search_business_policy(query: str) -> str:
    """搜索公司制度知识库，当用户询问年假、报销、远程办公等公司制度问题时使用"""

@tool
def search_product_info(query: str) -> str:
    """搜索产品信息知识库，当用户询问产品功能、定价等公司产品问题时使用"""

tools = [search_tech_docs, search_business_policy, search_product_info]
```

**实战要点：**

1. 每个知识源封装为独立工具，Agent 按需选择
2. 工具描述清晰区分适用场景，减少错误路由
3. Agent 可以并行调用多源检索，实现跨领域问答
4. 多源检索是构建企业级 RAG 系统的基础架构

---

### 示例3：对话式RAG — 多轮对话中的检索增强

**功能说明：** 在多轮对话中持续利用检索结果，Agent 能理解追问中的指代，历史检索结果可以复用。

**关键代码：**

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "请注意理解用户追问中的指代，结合之前的对话上下文来回答。"
               "如果之前已经搜索过相关信息，且问题可以在已有信息中回答，可以不重复搜索。"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 调用时传入对话历史
result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
```

**实战要点：**

1. `chat_history` 传递对话上下文，Agent 理解追问和指代
2. 对话记忆 + 检索工具 = 持续性知识问答
3. Agent 可以判断是否需要重复检索，避免冗余调用
4. 长对话时注意记忆窗口管理，避免上下文过长

---

### 示例4：RAG+工具 — 检索与功能工具协同

**功能说明：** Agent 同时使用检索工具（搜索知识库）和功能工具（计算器、日期计算），实现"查信息 + 做计算"的组合能力。

**关键代码：**

```python
@tool
def search_product_and_policy(query: str) -> str:
    """搜索产品信息和公司制度知识库"""

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，如 '299 * 12'"""

@tool
def date_calculator(operation: str) -> str:
    """计算日期相关的问题，如 '30天后是哪天'"""

tools = [search_product_and_policy, calculator, date_calculator]
```

**典型应用场景：**

- "SmartAssist Pro 专业版一年多少钱？" → 检索定价 + 计算年费
- "报销的30天期限，最晚哪天提交？" → 检索制度 + 日期计算
- "3个基础版+2个专业版一个月共多少？" → 检索定价 + 多项计算

**实战要点：**

1. RAG + 工具 = 知识获取 + 数据处理，覆盖更复杂的业务场景
2. Agent 能编排多步操作：先检索信息，再基于信息做计算
3. 工具描述越清晰，Agent 的工具选择和参数传递越准确
4. 计算器工具需做安全校验，只允许数字和基本运算符

---

## 8.4 可观测性（langsmith_demo.py）

### 知识点：追踪基础、评估系统、数据集管理、自定义评估器

**运行方式：**

```bash
python src/chains/langsmith_demo.py
```

> 本示例使用模拟实现演示 LangSmith 的核心功能，无需真实 LangSmith 账号。

**核心概念：**

| 功能 | 核心机制 | 应用场景 |
|------|---------|---------|
| 追踪 | 记录每次调用的输入/输出/耗时 | 调试排错、性能分析 |
| 评估 | 衡量模型输出的质量 | 质量保障、迭代优化 |
| 数据集 | 管理测试用例（输入+期望输出） | 回归测试、对比评估 |
| 自定义评估器 | 按业务规则定制评估逻辑 | 业务定制评估 |

---

### 示例1：追踪基础 — 记录每次调用的追踪信息

**功能说明：** 每次调用自动记录输入、输出、耗时和状态，模拟 LangSmith 的追踪功能。

**关键代码：**

```python
class TraceStore:
    """模拟 LangSmith 追踪数据存储"""
    _traces = []

    @classmethod
    def record(cls, trace_data):
        trace_data["id"] = f"trace-{len(cls._traces) + 1:04d}"
        trace_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cls._traces.append(trace_data)
        return trace_data["id"]

# 执行调用并记录追踪
start_time = time.time()
answer = chain.invoke({"question": question})
elapsed = time.time() - start_time

trace_id = TraceStore.record({
    "name": "qa_chain", "input": {"question": question},
    "output": answer, "duration_ms": round(elapsed * 1000, 2),
    "status": "success",
    "metadata": {"model": "default_llm", "prompt_tokens": len(question) * 2}
})
```

**实战要点：**

1. 追踪记录包含输入、输出、耗时、状态
2. 可用于调试、性能分析和成本监控
3. 真实 LangSmith 提供可视化追踪界面
4. 追踪 ID 用于关联和检索特定调用

---

### 示例2：评估系统 — 评估模型输出的质量

**功能说明：** 输入问题和期望答案（关键词），系统评估模型输出的准确性、相关性和完整性。

**关键代码：**

```python
# 模拟评估
expected_keywords = [k.strip() for k in expected.split(",")]
matched = [k for k in expected_keywords if k in answer]
accuracy = len(matched) / len(expected_keywords) * 100

# 相关性评估（基于回答长度比例，模拟）
relevance = min(100, max(20, len(answer) / max(len(question), 1) * 30))

# 完整性评估（关键词覆盖率）
completeness = accuracy

print(f"准确性：{accuracy:.0f}% （{len(matched)}/{len(expected_keywords)} 关键词命中）")
print(f"相关性：{relevance:.0f}%")
print(f"完整性：{completeness:.0f}%")
```

**实战要点：**

1. 评估需要明确的衡量维度和标准
2. 关键词匹配是最基础的评估方法
3. 真实 LangSmith 支持 **LLM 作为评判者**进行评估
4. 评估维度可根据业务需求扩展

---

### 示例3：数据集管理 — 创建和管理测试数据集

**功能说明：** 通过 `DatasetStore` 创建、添加样本、查看、运行测试的数据集管理系统。

**关键代码：**

```python
class DatasetStore:
    """模拟 LangSmith 数据集存储"""
    _datasets = {}

    @classmethod
    def create(cls, name, description=""):
        cls._datasets[name] = {"name": name, "description": description, "examples": []}

    @classmethod
    def add_example(cls, name, example):
        example["id"] = f"ex-{len(cls._datasets[name]['examples']) + 1:03d}"
        cls._datasets[name]["examples"].append(example)

# 运行数据集测试
for ex in ds["examples"]:
    result = chain.invoke({"input": ex["input"]})
    is_match = ex["expected_output"].lower() in result.lower()
    if is_match: passed += 1
```

**实战要点：**

1. 数据集是结构化的测试用例集合（输入 + 期望输出）
2. 每个样本包含输入和期望输出，可附加标签
3. 数据集可用于回归测试和质量监控
4. 真实 LangSmith 提供在线数据集管理界面

---

### 示例4：自定义评估器 — 创建自定义的评估规则

**功能说明：** 提供4种自定义评估器——长度评估器、关键词评估器、情感评估器、格式评估器，展示如何按业务规则定制评估逻辑。

**4种评估器对比：**

| 评估器 | 评估维度 | 适用场景 |
|--------|---------|---------|
| 长度评估器 | 回答长度是否在合理范围 | 控制回答篇幅 |
| 关键词评估器 | 是否包含关键信息 | 验证信息完整性 |
| 情感评估器 | 回答的情感倾向 | 客服质检 |
| 格式评估器 | 是否包含列表/代码等结构 | 验证输出格式 |

**关键代码：**

```python
evaluators = {
    "1": {
        "name": "长度评估器",
        "evaluate": lambda answer, expected: {
            "score": 100 if 20 <= len(answer) <= 500 else 70,
            "detail": f"长度 {len(answer)} 字，{'适中' if 20 <= len(answer) <= 500 else '偏短或偏长'}"
        }
    },
    "2": {
        "name": "关键词评估器",
        "evaluate": lambda answer, expected: {
            "score": len([k for k in expected if k in answer]) / len(expected) * 100,
            "detail": f"命中 {len([k for k in expected if k in answer])}/{len(expected)} 个关键词"
        }
    },
    ...
}
```

**实战要点：**

1. 自定义评估器可以匹配特定业务需求
2. 评估器接收回答和参考，返回结构化评分
3. 可组合多个评估器进行综合评估
4. 真实 LangSmith 支持继承 `RunEvaluator` 类

---

## 本章小结

本章深入探讨了 RAG（检索增强生成）的四大核心主题：

1. **RAG基础**：掌握了 RAG 的核心流程——检索 → 注入上下文 → 生成答案。学习了简单 RAG、文档问答、相似度搜索和带来源 RAG 四种模式，理解了检索质量决定回答质量的核心原则。

2. **文档加载与切分**：学习了 `RecursiveCharacterTextSplitter`（递归字符切分，最常用）、`CharacterTextSplitter`（单字符切分）、`Document` 对象的创建与元数据管理，以及自定义切分器（按标题/代码/分隔线切分和组合切分策略）。

3. **RAG Agent**：将检索能力封装为 `@tool`，让 Agent 自主决定何时检索。学习了 RAG Agent（自主决策检索）、多源 RAG（多知识源选择）、对话式 RAG（多轮对话 + 检索）和 RAG + 工具（检索 + 计算）四种模式。

4. **可观测性**：学习了追踪（记录调用信息）、评估（衡量输出质量）、数据集管理（管理测试用例）和自定义评估器（按业务规则评估）。这些是 RAG 系统质量保障和持续优化的基础设施。

**关键原则：**
- RAG 的核心是"先检索，后生成"，检索质量决定回答质量
- 文档切分要在"保留上下文完整性"和"避免冗余信息"之间取得平衡
- RAG Agent 比纯 RAG 链更灵活，能自主判断是否需要检索
- 可观测性是 RAG 系统从实验走向生产的必备能力
