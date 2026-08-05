"""
AI 代码助手 / AI Code Assistant
====================================

完整的端到端AI代码助手项目，整合多种LangChain技术

技术整合 / Tech Integration:
- Chain: 代码生成、代码解释（LLMChain + PromptTemplate + StrOutputParser）
- Agent: Bug调试（Tool Calling Agent + 代码分析/搜索工具）
- Tool: 代码分析、语法检查等辅助工具（@tool 装饰器）
- Structured Output: 代码审查（PydanticOutputParser + 审查报告模型）
- RAG: 代码模式知识库检索（关键词检索 + 上下文注入）
- Memory: 调试会话追踪（对话记忆 + 代码上下文）

功能模块 / Features:
1. 代码生成 - 输入需求描述，AI生成代码（Chain）
2. Bug调试 - 输入代码和报错，AI分析修复（Agent + Tool）
3. 代码审查 - 输入代码，AI审查改进（Structured Output）
4. 代码解释 - 输入代码片段，AI逐行解释（Chain）
5. 代码转换 - 输入代码和目标语言，AI转换代码（Tool）

应用场景 / Use Cases:
- 日常编程辅助：快速生成代码、理解代码、修复Bug
- 代码质量提升：自动化代码审查、获取改进建议
- 多语言开发：代码语言转换、跨语言学习
- 技术面试准备：代码审查和优化实践

Core Concepts:
- Chain: Compose LLM calls for code generation and explanation workflows
- Agent: Autonomous tool selection for multi-step debugging
- Tool: Custom functions for code analysis and syntax checking
- Structured Output: Typed review report with severity ratings
- RAG: Knowledge retrieval for coding patterns and best practices
- Memory: Session context for iterative debugging conversations
"""

import os
import sys
import json
import re
import sqlite3
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from src.utils.llm_loader import get_default_llm


# ============================================================
# SQLite 数据库配置
# ============================================================

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "code_assistant.db")


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
    cursor.execute("""CREATE TABLE IF NOT EXISTS code_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        language TEXT NOT NULL,
        pattern_name TEXT NOT NULL,
        description TEXT NOT NULL,
        code_example TEXT,
        tags TEXT
    )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS debug_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT,
        error_msg TEXT,
        fix_suggestion TEXT,
        language TEXT,
        created_at TEXT NOT NULL
    )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS code_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT,
        quality_score INTEGER,
        issues_json TEXT,
        suggestions_json TEXT,
        created_at TEXT NOT NULL
    )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS conversion_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_lang TEXT NOT NULL,
        target_lang TEXT NOT NULL,
        rule_description TEXT,
        example TEXT
    )""")

    # 插入示例数据（如果表为空）
    cursor.execute("SELECT COUNT(*) FROM code_patterns")
    if cursor.fetchone()[0] == 0:
        sample_patterns = [
            ("python", "设计模式", "Python 常见设计模式：单例模式（使用模块或装饰器实现）、工厂模式（根据条件创建对象）、"
             "观察者模式（事件驱动编程）、策略模式（替换算法实现）。"
             "Pythonic 写法：列表推导式、上下文管理器（with语句）、生成器（yield）。",
             "class Singleton:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance",
             "Python,设计模式,单例,工厂,观察者"),

            ("python", "错误处理", "Python 错误处理最佳实践：使用具体的异常类型而非裸 except、"
             "finally 块确保资源释放、自定义异常类继承 Exception、"
             "使用 logging 而非 print 记录错误、避免过深的异常嵌套。"
             "常见运行时错误：TypeError、KeyError、IndexError、AttributeError。",
             "try:\n    result = risky_operation()\nexcept ValueError as e:\n    logger.error(f'Value error: {e}')\nfinally:\n    cleanup()",
             "Python,错误处理,异常,try,except"),

            ("python", "性能优化", "Python 性能优化：使用内置函数和标准库（通常比手写循环快）、"
             "列表推导式优于 for+append、生成器处理大数据集、"
             "collections 模块（Counter、defaultdict、deque）、"
             "functools.lru_cache 缓存计算结果、避免在循环内频繁创建对象。",
             "from functools import lru_cache\n\n@lru_cache(maxsize=128)\ndef fibonacci(n):\n    if n < 2: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
             "Python,性能,优化,缓存,生成器"),

            ("javascript", "基础语法", "JavaScript/TypeScript 基础：var/let/const 区别、箭头函数、"
             "Promise 和 async/await 异步编程、解构赋值、模板字符串。"
             "Node.js 事件循环机制：宏任务（setTimeout、I/O）和微任务（Promise.then）。"
             "TypeScript 类型系统：interface、type、泛型、枚举。",
             "const fetchData = async (url) => {\n  const response = await fetch(url);\n  return response.json();\n};",
             "JavaScript,TypeScript,异步,Promise,async"),

            ("java", "基础语法", "Java 基础：强类型语言、JVM 运行机制、垃圾回收（GC）。"
             "核心概念：类和接口、继承和多态、集合框架（List/Map/Set）、"
             "异常体系（Checked vs Unchecked Exception）、泛型。"
             "Spring 框架：IoC（控制反转）、AOP（面向切面编程）、自动配置。",
             "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}",
             "Java,Spring,面向对象,集合,泛型"),

            ("sql", "优化原则", "SQL 优化原则：避免 SELECT *、合理使用索引、"
             "JOIN 优于子查询（多数场景）、EXPLAIN 分析执行计划、"
             "分页查询用 LIMIT OFFSET（大数据量考虑游标分页）、"
             "避免在 WHERE 子句中对列使用函数、 UNION ALL 优于 UNION（不需要去重时）。",
             "SELECT u.name, o.total\nFROM users u\nJOIN orders o ON u.id = o.user_id\nWHERE o.created_at > '2024-01-01'\nORDER BY o.total DESC\nLIMIT 10;",
             "SQL,优化,索引,JOIN,查询"),

            ("git", "问题修复", "Git 常见问题修复：git stash 暂存未提交修改、"
             "git reset --soft 撤销提交但保留修改、git cherry-pick 选择性合并提交、"
             "git reflog 找回丢失的提交、git revert 生成反向提交。"
             "合并冲突解决流程：拉取最新 → 手动解决冲突 → 标记已解决 → 提交。",
             "git stash\ngit pull origin main\ngit stash pop\n# 解决冲突\ngit add .\ngit commit -m 'resolve conflicts'",
             "Git,冲突,stash,reset,revert"),

            ("python", "API设计", "RESTful API 设计原则：资源用名词不用动词、"
             "HTTP 方法语义（GET 查询/POST 创建/PUT 更新/DELETE 删除）、"
             "状态码规范（200 成功/201 创建/400 请求错误/401 未认证/404 不存在/500 服务器错误）、"
             "版本管理（URL路径或Header）、分页和过滤参数设计。",
             "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/users/{user_id}')\nasync def get_user(user_id: int):\n    return {'id': user_id}",
             "API,REST,HTTP,设计,FastAPI"),

            ("python", "代码审查", "代码审查要点：命名规范（变量名表意清晰、函数名动词开头）、"
             "单一职责原则（函数只做一件事）、DRY原则（不重复代码）、"
             "错误处理完整性、边界条件覆盖、安全漏洞检查（SQL注入、XSS）。"
             "代码复杂度控制：圈复杂度<10、函数长度<50行、嵌套深度<4层。",
             "# Good: 单一职责\ndef calculate_total(items):\n    return sum(item.price * item.quantity for item in items)",
             "代码审查,命名,单一职责,DRY,安全"),

            ("python", "测试实践", "测试最佳实践：单元测试覆盖核心逻辑、"
             "测试金字塔（单元测试 > 集成测试 > 端到端测试）、"
             "AAA模式（Arrange-Act-Assert）、Mock 隔离外部依赖、"
             "参数化测试覆盖边界值、测试命名规范（test_功能_场景_预期结果）。"
             "Python 测试工具：pytest、unittest.mock、coverage。",
             "import pytest\n\ndef test_addition():\n    # Arrange\n    a, b = 2, 3\n    # Act\n    result = add(a, b)\n    # Assert\n    assert result == 5",
             "测试,pytest,单元测试,Mock,覆盖率"),
        ]
        cursor.executemany(
            "INSERT INTO code_patterns (language, pattern_name, description, code_example, tags) VALUES (?, ?, ?, ?, ?)",
            sample_patterns
        )

    # 插入转换规则示例数据
    cursor.execute("SELECT COUNT(*) FROM conversion_rules")
    if cursor.fetchone()[0] == 0:
        sample_rules = [
            ("python", "javascript", "print() → console.log()；def → function/const=()=>；True/False → true/false；None → null/undefined", "print('hello') → console.log('hello')"),
            ("python", "javascript", "and/or/not → &&/||/!；len() → .length；list.append() → .push()", "len(arr) → arr.length"),
            ("python", "javascript", "字典dict → Object/Map；列表推导式 → .map()/.filter()；f-string → 模板字符串", "[x*2 for x in arr] → arr.map(x => x*2)"),
            ("javascript", "python", "console.log() → print()；function/=> → def；true/false → True/False", "console.log('hi') → print('hi')"),
            ("javascript", "python", "&&/||/! → and/or/not；.length → len()；.push() → .append()", "arr.length → len(arr)"),
            ("python", "java", "print() → System.out.println()；def → 访问修饰符 返回类型 方法名；class → class（需要显式类型声明）", "print('hi') → System.out.println('hi')"),
            ("python", "java", "True/False → true/false；None → null；dict → HashMap/Map；list → ArrayList/List", "None → null"),
            ("python", "go", "print() → fmt.Println()；def → func；class → struct + method；None → nil", "print('hi') → fmt.Println('hi')"),
            ("python", "go", "dict → map[type]type；list → slice []type；try/except → defer + error 返回值", "try/except → defer + error"),
        ]
        cursor.executemany(
            "INSERT INTO conversion_rules (source_lang, target_lang, rule_description, example) VALUES (?, ?, ?, ?)",
            sample_rules
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
                 "我", "你", "他", "她", "它", "请", "想", "能", "做"}

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


def retrieve_code_knowledge(query: str, top_k: int = 3) -> list[tuple[int, float, str]]:
    """从SQLite代码模式知识库中检索相关内容"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, description, tags FROM code_patterns")
    rows = cursor.fetchall()
    conn.close()

    # 同时搜索description和tags
    scored = []
    for row in rows:
        # 组合description和tags进行相似度计算
        combined_text = row["description"]
        if row["tags"]:
            combined_text += " " + row["tags"]
        score = keyword_similarity(query, combined_text)
        scored.append((row["id"], score, row["description"]))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ============================================================
# Structured Output 模型（代码审查）
# ============================================================

class CodeReviewReport(BaseModel):
    """代码审查报告模型"""
    overall_rating: str = Field(description="整体评分：优秀/良好/一般/需改进")
    code_quality_score: int = Field(description="代码质量评分：1-10分")
    issues: list[dict] = Field(description="问题列表：每个dict包含 severity(严重程度)、location(位置)、description(描述)、suggestion(建议)")
    strengths: list[str] = Field(description="代码优点列表")
    refactoring_suggestions: list[str] = Field(description="重构建议列表")
    security_concerns: list[str] = Field(description="安全隐患列表，没有则为空列表")
    summary: str = Field(description="审查总结")


# ============================================================
# Agent 工具定义（Bug调试 + 代码转换）
# ============================================================

# 错误类型数据库（保留内存中，供Agent工具快速使用）
_ERROR_PATTERNS = {
    "python": {
        "TypeError": "类型错误：操作或函数被应用于类型不合适的对象。常见原因：参数类型不匹配、字符串和数字直接相加、对None值调用方法。",
        "KeyError": "键错误：字典中不存在指定的键。解决方法：使用 dict.get(key, default) 或先检查 key in dict。",
        "IndexError": "索引错误：序列索引超出范围。常见原因：访问空列表、索引从0开始但用了1开始的逻辑。",
        "AttributeError": "属性错误：对象没有该属性或方法。常见原因：变量为None、拼写错误、导入的模块不包含该方法。",
        "ImportError": "导入错误：无法导入模块或名称。常见原因：模块未安装、路径错误、名称拼写错误。",
        "SyntaxError": "语法错误：代码不符合Python语法规则。检查：括号匹配、冒号、缩进。",
        "NameError": "名称错误：使用了未定义的变量或函数。检查：拼写、作用域、是否先定义后使用。",
        "ValueError": "值错误：操作收到了正确类型但不合适的值。如 int('abc')。",
        "IndentationError": "缩进错误：Python使用缩进表示代码块，检查缩进是否一致（空格与Tab混用）。",
        "RecursionError": "递归错误：递归层数超过限制。检查递归终止条件。",
    },
    "javascript": {
        "TypeError": "类型错误：undefined is not a function、Cannot read property of undefined。常见原因：访问未定义对象的属性、调用非函数类型。",
        "ReferenceError": "引用错误：使用了未声明的变量。检查：变量是否在作用域内、是否有拼写错误。",
        "SyntaxError": "语法错误：代码不符合JS语法。常见原因：缺少括号、缺少逗号、模板字符串语法。",
        "RangeError": "范围错误：值超出有效范围。如无效的数组长度、递归过深。",
        "URIError": "URI错误：encodeURI/decodeURI参数无效。",
    },
}


@tool
def analyze_error(error_message: str, language: str = "python") -> str:
    """分析错误信息，识别错误类型和常见原因

    Args:
        error_message: 错误信息文本
        language: 编程语言，python 或 javascript

    Returns:
        错误分析和解决建议
    """
    lang_key = language.lower()
    if lang_key not in _ERROR_PATTERNS:
        lang_key = "python"

    patterns = _ERROR_PATTERNS[lang_key]

    # 匹配错误类型
    matched_errors = []
    for error_type, description in patterns.items():
        if error_type.lower() in error_message.lower():
            matched_errors.append((error_type, description))

    if not matched_errors:
        result = (
            f"🔍 错误分析结果：\n\n"
            f"错误信息：{error_message}\n"
            f"编程语言：{language}\n\n"
            f"未能自动匹配到已知错误类型。常见排查步骤：\n"
            f"  1. 仔细阅读错误信息中的文件名和行号\n"
            f"  2. 检查出错行的上下文代码\n"
            f"  3. 检查变量类型和值\n"
            f"  4. 搜索错误信息的核心关键词"
        )
        return result

    result = f"🔍 错误分析结果：\n\n"
    result += f"编程语言：{language}\n"
    result += f"匹配到 {len(matched_errors)} 个已知错误类型：\n\n"

    for error_type, description in matched_errors:
        result += f"❌ {error_type}\n"
        result += f"   {description}\n\n"

    result += "💡 通用调试建议：\n"
    result += "  1. 在出错行添加 print/logging 语句查看变量状态\n"
    result += "  2. 使用调试器（pdb/IDE断点）单步执行\n"
    result += "  3. 缩小问题范围：注释掉部分代码逐步定位\n"
    result += "  4. 检查最近的代码修改是否引入了问题"

    return result


@tool
def check_syntax(code: str, language: str = "python") -> str:
    """检查代码的基本语法问题

    Args:
        code: 要检查的代码
        language: 编程语言

    Returns:
        语法检查结果
    """
    issues = []

    if language.lower() == "python":
        # 检查括号匹配
        stack = []
        pairs = {"(": ")", "[": "]", "{": "}"}
        for i, char in enumerate(code):
            if char in pairs:
                stack.append((char, i))
            elif char in pairs.values():
                if not stack:
                    issues.append(f"位置 {i+1}：多余的闭合括号 '{char}'")
                elif pairs[stack[-1][0]] != char:
                    issues.append(f"位置 {i+1}：括号不匹配，期望 '{pairs[stack[-1][0]]}' 但得到 '{char}'")
                else:
                    stack.pop()
        for open_char, pos in stack:
            issues.append(f"位置 {pos+1}：未闭合的 '{open_char}'")

        # 检查常见缩进问题
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if line and not line.strip().startswith("#"):
                # 检查行首Tab和空格混用
                leading = ""
                for ch in line:
                    if ch in " \t":
                        leading += ch
                    else:
                        break
                if "\t" in leading and " " in leading:
                    issues.append(f"第 {i+1} 行：Tab和空格混用")

        # 检查常见遗漏
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("if ") and stripped.endswith(":") and ":\n" not in stripped:
                # 检查 if/for/while/def/class 后的冒号和缩进
                pass
            if stripped in ["if", "for", "while", "def", "class", "try", "except", "finally", "with", "elif", "else"]:
                issues.append(f"第 {i+1} 行：可能缺少冒号或后续代码")

    elif language.lower() in ["javascript", "js", "typescript", "ts"]:
        # JS/TS 基本检查
        stack = []
        pairs = {"(": ")", "[": "]", "{": "}"}
        for i, char in enumerate(code):
            if char in pairs:
                stack.append((char, i))
            elif char in pairs.values():
                if not stack:
                    issues.append(f"位置 {i+1}：多余的闭合括号 '{char}'")
                elif pairs[stack[-1][0]] != char:
                    issues.append(f"位置 {i+1}：括号不匹配")
                else:
                    stack.pop()
        for open_char, pos in stack:
            issues.append(f"位置 {pos+1}：未闭合的 '{open_char}'")

        # 检查分号（可选但值得提示）
        lines = code.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith(("//", "/*", "*", "function", "if", "for", "while", "class")):
                if not stripped.endswith((";", "{", "}", ")", ",", "//")) and "=" in stripped:
                    pass  # 分号可选，不强制

    if not issues:
        return f"✅ 基本语法检查通过（{language}），未发现明显问题。\n注意：这是简单的模式检查，建议配合专业 linter 使用。"

    result = f"🔍 语法检查结果（{language}）：\n发现 {len(issues)} 个潜在问题：\n\n"
    for i, issue in enumerate(issues, 1):
        result += f"  {i}. {issue}\n"
    result += "\n建议修复以上问题后重新检查。"
    return result


@tool
def convert_code_snippet(code: str, source_lang: str, target_lang: str) -> str:
    """代码片段的语言转换辅助工具，提供语言特性映射参考

    Args:
        code: 源代码
        source_lang: 源语言
        target_lang: 目标语言

    Returns:
        语言转换参考信息和映射
    """
    # 从SQLite读取转换规则
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT rule_description, example FROM conversion_rules WHERE source_lang = ? AND target_lang = ?",
        (source_lang.lower(), target_lang.lower())
    )
    rules = cursor.fetchall()
    conn.close()

    if not rules:
        # 如果数据库中没有，使用内置语言映射表
        lang_map = {
            ("python", "javascript"): {
                "print()": "console.log()",
                "def": "function / const = () =>",
                "elif": "else if",
                "True/False": "true/false",
                "None": "null/undefined",
                "and/or/not": "&&/||/!",
                "len()": ".length",
                "list.append()": ".push()",
                "dict": "Object / Map",
                "list comprehension": ".map() / .filter()",
                "f-string": "模板字符串 ``",
                "self": "this",
                "__init__": "constructor",
                "try/except": "try/catch",
                "import": "import / require",
            },
            ("javascript", "python"): {
                "console.log()": "print()",
                "function/=>": "def",
                "else if": "elif",
                "true/false": "True/False",
                "null/undefined": "None",
                "&&/||/!": "and/or/not",
                ".length": "len()",
                ".push()": ".append()",
                "Object/Map": "dict",
                ".map()/.filter()": "列表推导式",
                "模板字符串": "f-string",
                "this": "self",
                "constructor": "__init__",
                "try/catch": "try/except",
                "import/require": "import",
            },
            ("python", "java"): {
                "print()": "System.out.println()",
                "def": "访问修饰符 返回类型 方法名",
                "class": "class（需要显式类型声明）",
                "True/False": "true/false",
                "None": "null",
                "and/or/not": "&&/||/!",
                "len()": ".length / .size()",
                "dict": "HashMap / Map",
                "list": "ArrayList / List",
                "self": "this",
                "__init__": "构造方法（与类同名）",
                "try/except": "try/catch",
                "import": "import",
            },
            ("python", "go"): {
                "print()": "fmt.Println()",
                "def": "func",
                "class": "struct + method",
                "True/False": "true/false",
                "None": "nil",
                "and/or/not": "&&/||/!",
                "dict": "map[type]type",
                "list": "slice []type",
                "self": "receiver（显式声明）",
                "__init__": "func New*() 构造函数",
                "try/except": "defer + error 返回值",
            },
        }

        key = (source_lang.lower(), target_lang.lower())
        mapping = lang_map.get(key)

        if not mapping:
            available = ", ".join(f"{k[0]}→{k[1]}" for k in lang_map.keys())
            return f"暂不支持 {source_lang}→{target_lang} 的映射参考。当前支持：{available}"

        result = (
            f"🔄 语言转换参考（{source_lang} → {target_lang}）：\n\n"
            f"语法映射表：\n"
        )
        for src, tgt in mapping.items():
            result += f"  {src:30s} → {tgt}\n"

        result += (
            f"\n💡 转换注意事项：\n"
            f"  - 不同语言的惯用法不同，直接翻译可能不是最佳实践\n"
            f"  - 注意类型系统的差异（动态类型 vs 静态类型）\n"
            f"  - 标准库和生态差异需要考虑\n"
            f"  - AI 会基于此映射表和代码内容生成完整的转换代码"
        )

        return result

    # 从数据库规则生成映射参考
    result = (
        f"🔄 语言转换参考（{source_lang} → {target_lang}）：\n\n"
        f"转换规则（来自数据库）：\n"
    )
    for i, rule in enumerate(rules, 1):
        result += f"  {i}. {rule['rule_description']}\n"
        if rule["example"]:
            result += f"     示例：{rule['example']}\n"

    result += (
        f"\n💡 转换注意事项：\n"
        f"  - 不同语言的惯用法不同，直接翻译可能不是最佳实践\n"
        f"  - 注意类型系统的差异（动态类型 vs 静态类型）\n"
        f"  - 标准库和生态差异需要考虑\n"
        f"  - AI 会基于此映射表和代码内容生成完整的转换代码"
    )

    return result


@tool
def search_code_pattern(pattern_query: str) -> str:
    """搜索代码模式和最佳实践

    Args:
        pattern_query: 代码模式查询，如 设计模式、排序算法、异常处理

    Returns:
        相关代码模式和最佳实践参考
    """
    # 从SQLite检索代码模式
    results = retrieve_code_knowledge(pattern_query, top_k=3)

    if not results or results[0][1] < 0.05:
        return f"未找到与「{pattern_query}」相关的代码模式。请尝试其他关键词。"

    output = f"📚 代码模式检索结果：\n\n"
    for i, (idx, score, doc) in enumerate(results):
        preview = doc[:80] + "..." if len(doc) > 80 else doc
        output += f"[模式{i+1}] 相似度={score:.2f}\n{preview}\n\n"

    return output


# ============================================================
# Memory - 代码会话追踪器（SQLite持久化）
# ============================================================

class CodeSessionTracker:
    """代码会话追踪器，使用SQLite持久化"""

    def __init__(self):
        self.chat_history: list = []

    def record(self, action: str, detail: str):
        """记录一次代码活动到SQLite"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO debug_history (code, error_msg, fix_suggestion, language, created_at) VALUES (?, ?, ?, ?, ?)",
            (action, detail, "", "general")
        )
        conn.commit()
        conn.close()

    def add_code(self, language: str, code: str, purpose: str):
        """保存代码片段到SQLite"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO code_patterns (language, pattern_name, description, code_example, tags) VALUES (?, ?, ?, ?, ?)",
            (language, purpose, code[:200], code[:500], "")
        )
        conn.commit()
        conn.close()

    def add_chat(self, human_msg: str, ai_msg: str):
        """添加对话记录"""
        self.chat_history.append(HumanMessage(content=human_msg))
        self.chat_history.append(AIMessage(content=ai_msg))
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

    def get_summary(self) -> str:
        """获取会话摘要"""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM debug_history")
        total = cursor.fetchone()[0]
        conn.close()

        if total == 0:
            return "暂无代码活动记录，开始你的编程之旅吧！"

        lines = [f"💻 代码会话报告（共 {total} 条记录）"]

        # 统计活动类型
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT language, COUNT(*) as cnt FROM debug_history GROUP BY language")
        rows = cursor.fetchall()
        conn.close()

        lang_counts = {row["language"]: row["cnt"] for row in rows}
        for lang, count in lang_counts.items():
            lines.append(f"  - {lang}：{count} 次")

        # 最近记录
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT code, error_msg, created_at FROM debug_history ORDER BY id DESC LIMIT 5")
        recent = cursor.fetchall()
        conn.close()

        lines.append("\n  最近活动：")
        for entry in reversed(recent):
            action = entry["code"] if entry["code"] else ""
            detail = entry["error_msg"] if entry["error_msg"] else ""
            lines.append(f"    [{entry['created_at']}] {action}：{detail[:40]}")

        return "\n".join(lines)


# ============================================================
# 1. 代码生成 - Chain + RAG
# ============================================================

def feature_code_generation(tracker: CodeSessionTracker):
    """功能1：代码生成 - 输入需求描述，AI 生成代码（整合 Chain + RAG）"""
    print("\n" + "=" * 60)
    print("  代码生成 - AI 生成代码")
    print("=" * 60)
    print("\n💡 技术整合：Chain（LLMChain）+ RAG（知识库检索）")
    print("   先检索相关代码模式和最佳实践，再由 LLM 生成高质量代码")

    # 从SQLite读取代码模式数量
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM code_patterns")
    pattern_count = cursor.fetchone()[0]
    conn.close()
    print(f"\n📚 代码模式库包含 {pattern_count} 条模式")

    model = get_default_llm()

    prompt = ChatPromptTemplate.from_template(
        "你是一位资深程序员。请根据以下需求生成代码。\n\n"
        "代码模式参考：\n{pattern_reference}\n\n"
        "需求描述：{requirement}\n"
        "编程语言：{language}\n\n"
        "要求：\n"
        "1. 生成完整可运行的代码\n"
        "2. 包含必要的注释说明\n"
        "3. 遵循该语言的最佳实践和惯用写法\n"
        "4. 包含基本的错误处理\n"
        "5. 给出使用示例\n"
        "6. 如果适用，说明时间/空间复杂度"
    )

    chain = prompt | model | StrOutputParser()

    print("\n【交互式代码生成】")
    print("输入需求描述，AI 生成完整代码")
    print("示例：'写一个函数，判断字符串是否为回文' 或 '实现一个简单的LRU缓存'")
    print("输入 '退出' 返回主菜单\n")

    while True:
        requirement = input("需求描述：").strip()

        if requirement.lower() in ["退出", "exit", "quit", "0"]:
            break

        if not requirement:
            print("请输入有效需求")
            continue

        language = input("编程语言（默认python）：").strip() or "python"

        try:
            # RAG 检索代码模式（从SQLite）
            results = retrieve_code_knowledge(requirement + " " + language, top_k=3)
            pattern_reference = "\n\n".join(
                f"[参考{i+1}] {doc}" for i, (_, _, doc) in enumerate(results)
            )

            print(f"\n🔍 检索到 {len(results)} 条相关代码模式")
            print("🤖 AI 正在生成代码...\n")
            response = chain.invoke({
                "pattern_reference": pattern_reference,
                "requirement": requirement,
                "language": language,
            })
            print(response)

            # 保存到SQLite
            tracker.record("代码生成", f"{language}：{requirement[:30]}")
            tracker.add_code(language, response[:200], requirement)

        except Exception as e:
            print(f"❌ 代码生成失败：{e}")

        print("\n" + "-" * 60)


# ============================================================
# 2. Bug调试 - Agent + Tool
# ============================================================

def feature_bug_debugging(tracker: CodeSessionTracker):
    """功能2：Bug调试 - Agent 调用工具分析和修复Bug（整合 Agent + Tool）"""
    print("\n" + "=" * 60)
    print("  Bug调试 - AI 分析修复")
    print("=" * 60)
    print("\n💡 技术整合：Agent（Tool Calling Agent）+ Tool（自定义工具）")
    print("   Agent 可调用的工具：")
    print("   - analyze_error：分析错误信息，识别错误类型")
    print("   - check_syntax：检查代码基本语法问题")
    print("   - search_code_pattern：搜索代码模式和最佳实践")

    model = get_default_llm()
    tools = [analyze_error, check_syntax, search_code_pattern]

    agent = create_agent(model, tools, system_prompt="你是一位经验丰富的调试专家。根据用户提供的代码和错误信息，利用可用工具分析问题并给出修复方案。你需要：1）分析错误信息识别错误类型 2）检查代码语法 3）搜索相关代码模式。最终给出完整的Bug分析报告和修复代码。")

    print("\n【交互式Bug调试】")
    print("输入代码和报错信息，AI 分析原因并给出修复方案")
    print("格式：先输入代码，再输入报错信息")
    print("示例：先粘贴代码，再输入 'TypeError: unsupported operand type(s)'")
    print("输入 '退出' 返回主菜单\n")

    while True:
        print("请输入代码（输入完代码后输入 '---' 结束）：")
        code_lines = []
        while True:
            line = input()
            if line.strip() == "---":
                break
            code_lines.append(line)
            if line.strip() in ["退出", "exit", "quit", "0"]:
                break

        code = "\n".join(code_lines)

        if not code.strip() or code.strip().lower() in ["退出", "exit", "quit", "0"]:
            break

        error_msg = input("报错信息（直接回车如果没有）：").strip()
        language = input("编程语言（默认python）：").strip() or "python"

        # 构建Agent输入
        agent_input = f"代码如下：\n```\n{code}\n```\n"
        if error_msg:
            agent_input += f"\n报错信息：{error_msg}\n"
        else:
            agent_input += "\n没有具体报错信息，请帮我检查代码问题。\n"
        agent_input += f"编程语言：{language}"

        try:
            result = agent.invoke({"messages": tracker.chat_history + [("user", agent_input)]})
            final_message = result["messages"][-1]

            print(f"\n🔧 调试报告：\n{final_message.content}")

            # 保存调试记录到SQLite
            tracker.record("Bug调试", f"{language}：{error_msg[:30] if error_msg else '代码检查'}")
            tracker.add_chat(agent_input[:50], final_message.content[:100])

            # 保存到debug_history表
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO debug_history (code, error_msg, fix_suggestion, language, created_at) VALUES (?, ?, ?, ?, ?)",
                (code[:500], error_msg[:200], final_message.content[:500], language,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"❌ 调试失败：{e}")

        print("\n" + "-" * 60)


# ============================================================
# 3. 代码审查 - Structured Output
# ============================================================

def feature_code_review(tracker: CodeSessionTracker):
    """功能3：代码审查 - 输入代码，AI 审查并给出改进建议（整合 Structured Output）"""
    print("\n" + "=" * 60)
    print("  代码审查 - AI 审查改进")
    print("=" * 60)
    print("\n💡 技术整合：Structured Output（PydanticOutputParser）")
    print("   使用 Pydantic 模型定义审查报告结构")
    print("   字段包括：整体评分、质量分数、问题列表、优点、重构建议、安全隐患")

    model = get_default_llm()
    parser = PydanticOutputParser(pydantic_object=CodeReviewReport)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一位严谨的高级代码审查专家。请对提交的代码进行全面审查，"
         "从代码质量、可读性、安全性、性能等多个维度给出结构化的审查报告。\n\n"
         "{format_instructions}\n\n"
         "注意：issues中每个dict包含 severity(高/中/低)、location(位置描述)、description(问题描述)、suggestion(修改建议) 四个字段。"),
        ("human", "编程语言：{language}\n\n代码如下：\n```\n{code}\n```")
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | model | parser

    print("\n【交互式代码审查】")
    print("输入代码，AI 进行全面审查并给出改进建议")
    print("输入 '退出' 返回主菜单\n")

    while True:
        print("请输入代码（输入完代码后输入 '---' 结束）：")
        code_lines = []
        while True:
            line = input()
            if line.strip() == "---":
                break
            code_lines.append(line)
            if line.strip() in ["退出", "exit", "quit", "0"]:
                break

        code = "\n".join(code_lines)

        if not code.strip() or code.strip().lower() in ["退出", "exit", "quit", "0"]:
            break

        language = input("编程语言（默认python）：").strip() or "python"

        try:
            result = chain.invoke({"code": code, "language": language})

            print("\n📋 代码审查报告：")
            print("=" * 50)
            print(f"📊 整体评分：{result.overall_rating}（{result.code_quality_score}/10）")
            print("-" * 50)

            # 优点
            if result.strengths:
                print(f"\n✅ 代码优点：")
                for s in result.strengths:
                    print(f"   • {s}")

            # 问题
            if result.issues:
                print(f"\n⚠️ 发现问题（{len(result.issues)} 个）：")
                severity_icon = {"高": "🔴", "中": "🟡", "低": "🟢"}
                for i, issue in enumerate(result.issues, 1):
                    severity = issue.get("severity", "中")
                    location = issue.get("location", "未知")
                    description = issue.get("description", "")
                    suggestion = issue.get("suggestion", "")
                    icon = severity_icon.get(severity, "⚪")
                    print(f"   {i}. {icon} [{severity}] {location}")
                    print(f"      问题：{description}")
                    print(f"      建议：{suggestion}")

            # 安全隐患
            if result.security_concerns:
                print(f"\n🔒 安全隐患：")
                for concern in result.security_concerns:
                    print(f"   • {concern}")

            # 重构建议
            if result.refactoring_suggestions:
                print(f"\n🔄 重构建议：")
                for rec in result.refactoring_suggestions:
                    print(f"   • {rec}")

            # 总结
            print(f"\n📝 审查总结：{result.summary}")
            print("=" * 50)

            # 保存审查结果到SQLite
            tracker.record("代码审查", f"{language}：评分{result.code_quality_score}/10")

            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO code_reviews (code, quality_score, issues_json, suggestions_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (code[:500], result.code_quality_score,
                 json.dumps(result.issues, ensure_ascii=False),
                 json.dumps(result.refactoring_suggestions, ensure_ascii=False),
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"❌ 审查失败：{e}")
            print("提示：请确保输入的是有效代码")

        print("\n" + "-" * 60)


# ============================================================
# 4. 代码解释 - Chain
# ============================================================

def feature_code_explanation(tracker: CodeSessionTracker):
    """功能4：代码解释 - 输入代码片段，AI 逐行解释（整合 Chain）"""
    print("\n" + "=" * 60)
    print("  代码解释 - AI 逐行解释")
    print("=" * 60)
    print("\n💡 技术整合：Chain（LLMChain + PromptTemplate）")
    print("   使用精心设计的提示词让 AI 逐行解释代码逻辑")

    model = get_default_llm()

    prompt = ChatPromptTemplate.from_template(
        "你是一位耐心的编程导师，擅长解释代码的每一行。请逐行解释以下代码。\n\n"
        "编程语言：{language}\n\n"
        "代码：\n```\n{code}\n```\n\n"
        "请按以下格式解释：\n"
        "1. 【整体功能】用一两句话概括代码的整体功能\n"
        "2. 【逐行解释】对每一行（或逻辑块）进行详细解释\n"
        "3. 【关键概念】指出代码中涉及的重要编程概念\n"
        "4. 【执行流程】描述代码的执行顺序和逻辑\n"
        "5. 【潜在问题】指出可能存在的问题或改进空间"
    )

    chain = prompt | model | StrOutputParser()

    print("\n【交互式代码解释】")
    print("输入代码片段，AI 逐行解释其逻辑")
    print("输入 '退出' 返回主菜单\n")

    while True:
        print("请输入代码（输入完代码后输入 '---' 结束）：")
        code_lines = []
        while True:
            line = input()
            if line.strip() == "---":
                break
            code_lines.append(line)
            if line.strip() in ["退出", "exit", "quit", "0"]:
                break

        code = "\n".join(code_lines)

        if not code.strip() or code.strip().lower() in ["退出", "exit", "quit", "0"]:
            break

        language = input("编程语言（默认python）：").strip() or "python"

        try:
            print(f"\n📖 正在解释代码...\n")
            response = chain.invoke({"code": code, "language": language})
            print(response)

            # 记录到SQLite
            tracker.record("代码解释", f"{language}：{code[:30]}...")

        except Exception as e:
            print(f"❌ 解释失败：{e}")

        print("\n" + "-" * 60)


# ============================================================
# 5. 代码转换 - Tool + Chain
# ============================================================

def feature_code_conversion(tracker: CodeSessionTracker):
    """功能5：代码转换 - 输入代码和目标语言，AI 转换代码（整合 Tool + Chain）"""
    print("\n" + "=" * 60)
    print("  代码转换 - AI 转换代码语言")
    print("=" * 60)
    print("\n💡 技术整合：Tool（convert_code_snippet）+ Chain（LLMChain）")
    print("   先用工具获取语言映射参考，再由 LLM 结合映射表生成转换代码")
    print("   转换规则从SQLite读取")

    model = get_default_llm()

    print("\n【交互式代码转换】")
    print("输入代码和目标语言，AI 将代码转换为目标语言")
    print("支持的语言对：Python↔JavaScript、Python→Java、Python→Go")
    print("输入 '退出' 返回主菜单\n")

    while True:
        print("请输入代码（输入完代码后输入 '---' 结束）：")
        code_lines = []
        while True:
            line = input()
            if line.strip() == "---":
                break
            code_lines.append(line)
            if line.strip() in ["退出", "exit", "quit", "0"]:
                break

        code = "\n".join(code_lines)

        if not code.strip() or code.strip().lower() in ["退出", "exit", "quit", "0"]:
            break

        source_lang = input("源语言（默认python）：").strip() or "python"
        target_lang = input("目标语言：").strip()

        if not target_lang:
            print("请输入目标语言")
            continue

        try:
            # 使用工具获取语言映射参考（从SQLite读取转换规则）
            mapping_result = convert_code_snippet.invoke({
                "code": code,
                "source_lang": source_lang,
                "target_lang": target_lang,
            })
            print(f"\n📋 语言映射参考：\n{mapping_result}\n")

            # LLM 结合映射表生成转换代码
            prompt = ChatPromptTemplate.from_template(
                "你是一位精通多种编程语言的开发者。请将以下代码从 {source_lang} 转换为 {target_lang}。\n\n"
                "语言映射参考：\n{lang_mapping}\n\n"
                "源代码：\n```{source_lang}\n{code}\n```\n\n"
                "要求：\n"
                "1. 使用目标语言的惯用写法，不要逐字翻译\n"
                "2. 保持相同的逻辑功能和算法\n"
                "3. 使用目标语言的标准库替代源语言特有的库\n"
                "4. 包含必要的注释\n"
                "5. 说明转换中的关键差异和注意事项"
            )

            chain = prompt | model | StrOutputParser()
            response = chain.invoke({
                "source_lang": source_lang,
                "target_lang": target_lang,
                "lang_mapping": mapping_result,
                "code": code,
            })

            print(f"\n🔄 转换结果：\n{response}")

            # 保存到SQLite
            tracker.record("代码转换", f"{source_lang}→{target_lang}")
            tracker.add_code(target_lang, response[:200], f"从{source_lang}转换")

        except Exception as e:
            print(f"❌ 转换失败：{e}")

        print("\n" + "-" * 60)


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    # 初始化数据库
    init_db()

    print("\n" + "=" * 60)
    print("  AI 代码助手 - 端到端实战项目")
    print("=" * 60)
    print("\n完整的 AI 代码助手，整合 Chain / Agent / Tool / Structured Output / RAG / Memory")

    print("\n功能模块：")
    print("  1. 代码生成（输入需求，AI 生成代码）            [Chain+RAG]")
    print("  2. Bug调试（输入代码和报错，AI 修复）            [Agent+Tool]")
    print("  3. 代码审查（输入代码，AI 审查改进）             [Structured Output]")
    print("  4. 代码解释（输入代码，AI 逐行解释）             [Chain]")
    print("  5. 代码转换（输入代码和目标语言，AI 转换）       [Tool+Chain]")

    print("\n应用场景：日常编程辅助 / 代码质量提升 / 多语言开发 / 技术面试准备")

    # 创建会话追踪器（整个会话共享）
    tracker = CodeSessionTracker()

    while True:
        print("\n" + "=" * 60)
        print("  AI 代码助手")
        print("=" * 60)
        print("  1. 代码生成（输入需求，AI 生成代码）")
        print("  2. Bug调试（输入代码和报错，AI 修复）")
        print("  3. 代码审查（输入代码，AI 审查改进）")
        print("  4. 代码解释（输入代码，AI 逐行解释）")
        print("  5. 代码转换（输入代码和目标语言，AI 转换）")
        print("\n  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-5): ").strip()

        if choice == "1":
            feature_code_generation(tracker)
        elif choice == "2":
            feature_bug_debugging(tracker)
        elif choice == "3":
            feature_code_review(tracker)
        elif choice == "4":
            feature_code_explanation(tracker)
        elif choice == "5":
            feature_code_conversion(tracker)
        elif choice == "0":
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM debug_history")
            has_records = cursor.fetchone()[0] > 0
            conn.close()
            if has_records:
                print("\n" + tracker.get_summary())
            print("\n感谢使用 AI 代码助手！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
