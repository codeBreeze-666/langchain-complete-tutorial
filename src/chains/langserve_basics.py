"""
LangServe 基础部署 - 交互式实战案例
==========================================

本示例演示 LangServe 的基础用法，将 LangChain 应用部署为 REST API 服务。

核心概念：
- FastAPI: 高性能 Web 框架，LangServe 基于 FastAPI 构建
  LangServe 利用 FastAPI 的异步特性和自动文档功能，
  将 LangChain 的 Runnable 对象快速暴露为 HTTP API

- add_routes: LangServe 的核心装饰器/函数，将 Runnable 对象自动注册为 API
  调用 add_routes(app, runnable, path) 后，自动生成以下端点：
  - /path/invoke  — 单次调用
  - /path/batch   — 批量调用
  - /path/stream  — 流式输出
  - /path/input_schema  — 输入 Schema
  - /path/output_schema — 输出 Schema
  - /path/config_schema — 配置 Schema

- invoke 端点: 单次调用，发送一个输入，返回一个输出
- batch 端点: 批量调用，发送多个输入，返回多个输出
- stream 端点: 流式输出，服务器发送事件(SSE)，实时返回生成内容
- /docs: FastAPI 自动生成的交互式 API 文档（Swagger UI）

应用场景：
- 将 AI 能力封装为微服务 API
- 为前端应用提供 LLM 后端
- 快速构建 AI API 原型
- 多端点 AI 服务（翻译、总结、改写等）

依赖安装：
    pip install langserve fastapi uvicorn httpx sse-starlette
"""

import os
import sys
import json
import threading
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_loader import get_default_llm


# ============================================================
# 示例1: FastAPI集成 - 创建API服务
# ============================================================

def demo_fastapi_integration():
    """
    示例1：FastAPI集成 - 将LLM封装为REST API

    演示如何使用 FastAPI + LangServe 创建一个基本的 API 服务，
    用户可以通过 HTTP 请求调用 LLM。
    """
    print("\n" + "="*60)
    print("示例1：FastAPI集成 - 创建API服务")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - FastAPI: 高性能异步 Web 框架")
    print("   - LangServe 基于 FastAPI，自动生成 API 端点")
    print("   - 无需手写路由，add_routes 自动完成注册")
    print()

    # 展示服务端代码
    server_code = '''
# ============ 服务端代码 (server.py) ============
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from src.utils.llm_loader import get_default_llm

# 1. 创建 FastAPI 应用
app = FastAPI(
    title="LangServe 基础 API",
    version="1.0",
    description="基于 LangServe 的 LLM API 服务"
)

# 2. 加载模型
llm = get_default_llm()

# 3. 创建 Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的AI助手，请用中文回答。"),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

# 4. 添加路由 — 自动生成 /chat/invoke, /chat/batch, /chat/stream 等端点
add_routes(app, chain, path="/chat")

# 5. 启动服务: uvicorn server:app --host 0.0.0.0 --port 8000
'''
    print(server_code)

    # 展示客户端调用方式
    client_code = '''
# ============ 客户端调用 ============
import requests

# 单次调用 (invoke)
response = requests.post(
    "http://localhost:8000/chat/invoke",
    json={"input": {"question": "什么是量子计算？"}}
)
result = response.json()
print("调用结果:", result["output"])

# 批量调用 (batch)
response = requests.post(
    "http://localhost:8000/chat/batch",
    json={"inputs": [
        {"question": "什么是AI？"},
        {"question": "什么是机器学习？"}
    ]}
)
results = response.json()
for i, r in enumerate(results):
    print(f"结果{i+1}:", r["output"])
'''
    print(client_code)

    # 模拟演示
    print("\n📋 模拟演示：直接调用 Chain（跳过服务器启动）")
    print("-" * 40)

    try:
        llm = get_default_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有帮助的AI助手，请用中文回答。"),
            ("human", "{question}")
        ])
        chain = prompt | llm | StrOutputParser()

        user_input = input("\n请输入你的问题（回车使用默认问题）: ").strip()
        if not user_input:
            user_input = "用一句话解释什么是 LangServe"

        print(f"\n📤 模拟 API 调用: POST /chat/invoke")
        print(f"   请求体: {{\"input\": {{\"question\": \"{user_input}\"}}}}")

        result = chain.invoke({"question": user_input})
        print(f"\n📥 API 响应:")
        print(f"   {{\"output\": \"{result}\"}}")
    except Exception as e:
        print(f"❌ 演示失败: {e}")

    print("\n💡 要点总结：")
    print("   1. FastAPI 应用是 LangServe 的基础容器")
    print("   2. add_routes() 将 Chain 自动注册为 REST API")
    print("   3. 自动生成 /invoke, /batch, /stream 端点")
    print("   4. 访问 http://localhost:8000/docs 查看自动文档")


# ============================================================
# 示例2: add_routes - 添加路由
# ============================================================

def demo_add_routes():
    """
    示例2：add_routes - 将Chain封装为API

    深入演示 add_routes 的用法，包括 invoke/batch/stream 三种调用方式。
    """
    print("\n" + "="*60)
    print("示例2：add_routes - 将Chain封装为API")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - add_routes(app, runnable, path): 将 Runnable 注册为 API")
    print("   - 自动生成 invoke/batch/stream 三种端点")
    print("   - 每种端点对应不同的调用模式")
    print()

    # 详细展示三种端点
    endpoints_info = '''
┌─────────────────────────────────────────────────────────┐
│  add_routes 自动生成的 API 端点                          │
├──────────────┬──────────────────────────────────────────┤
│  端点         │  说明                                    │
├──────────────┼──────────────────────────────────────────┤
│  /path/invoke│  单次调用：一个输入 → 一个输出             │
│  /path/batch │  批量调用：多个输入 → 多个输出             │
│  /path/stream│  流式输出：实时逐字返回结果               │
│  /path/input │  输入 Schema (GET)                       │
│  /path/output│  输出 Schema (GET)                       │
│  /path/config│  配置 Schema (GET)                       │
└──────────────┴──────────────────────────────────────────┘
'''
    print(endpoints_info)

    # 服务端代码
    server_code = '''
# ============ 服务端代码 ============
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from src.utils.llm_loader import get_default_llm

app = FastAPI(title="add_routes 演示")
llm = get_default_llm()

# 创建翻译 Chain
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业翻译，将用户输入翻译成{target_language}。只输出翻译结果。"),
    ("human", "{text}")
])
translate_chain = translate_prompt | llm | StrOutputParser()

# 添加路由 — 会在 /translate 下生成 invoke/batch/stream 端点
add_routes(app, translate_chain, path="/translate")

# 启动: uvicorn server:app --port 8000
'''
    print(server_code)

    # 客户端调用代码
    client_code = '''
# ============ 客户端调用 ============
import requests
import json

BASE_URL = "http://localhost:8000"

# --- invoke: 单次调用 ---
print("=== invoke 单次调用 ===")
response = requests.post(f"{BASE_URL}/translate/invoke", json={
    "input": {"text": "你好世界", "target_language": "英语"}
})
print("结果:", response.json()["output"])

# --- batch: 批量调用 ---
print("\\n=== batch 批量调用 ===")
response = requests.post(f"{BASE_URL}/translate/batch", json={
    "inputs": [
        {"text": "人工智能", "target_language": "英语"},
        {"text": "机器学习", "target_language": "日语"},
        {"text": "深度学习", "target_language": "韩语"},
    ]
})
for i, r in enumerate(response.json()):
    print(f"  翻译{i+1}: {r['output']}")

# --- stream: 流式输出 ---
print("\\n=== stream 流式输出 ===")
import httpx
with httpx.stream("POST", f"{BASE_URL}/translate/stream", json={
    "input": {"text": "大语言模型正在改变世界", "target_language": "英语"}
}, timeout=30) as response:
    for line in response.iter_lines():
        if line.startswith("data:"):
            data = json.loads(line[5:].strip())
            if "output" in data:
                print(data["output"], end="", flush=True)
print()
'''
    print(client_code)

    # 交互式模拟
    print("\n📋 模拟演示：直接调用 Chain")
    print("-" * 40)

    try:
        llm = get_default_llm()
        translate_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业翻译，将用户输入翻译成{target_language}。只输出翻译结果。"),
            ("human", "{text}")
        ])
        chain = translate_prompt | llm | StrOutputParser()

        # 模拟 invoke
        text = input("\n请输入要翻译的中文（回车使用默认）: ").strip()
        if not text:
            text = "人工智能正在改变世界"
        target = input("翻译成什么语言（回车默认英语）: ").strip()
        if not target:
            target = "英语"

        print(f"\n📤 模拟 invoke 调用: POST /translate/invoke")
        result = chain.invoke({"text": text, "target_language": target})
        print(f"📥 翻译结果: {result}")

        # 模拟 batch
        print(f"\n📤 模拟 batch 调用: POST /translate/batch")
        print(f"   批量翻译 3 个词...")
        batch_inputs = [
            {"text": "人工智能", "target_language": target},
            {"text": "机器学习", "target_language": target},
            {"text": "深度学习", "target_language": target},
        ]
        batch_results = chain.batch(batch_inputs)
        for i, (inp, res) in enumerate(zip(batch_inputs, batch_results)):
            print(f"   {inp['text']} → {res}")

    except Exception as e:
        print(f"❌ 演示失败: {e}")

    print("\n💡 要点总结：")
    print("   1. invoke: 单次请求-响应，适合单条数据处理")
    print("   2. batch: 批量处理，适合多数据并行处理")
    print("   3. stream: 流式响应，适合长文本实时输出")


# ============================================================
# 示例3: 多端点服务 - 多个API端点
# ============================================================

def demo_multi_endpoint():
    """
    示例3：多端点服务 - 同时提供翻译、总结、改写等API

    演示如何在一个 FastAPI 应用中注册多个 Chain，
    提供多种 AI 能力的 API 端点。
    """
    print("\n" + "="*60)
    print("示例3：多端点服务 - 多个API端点")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 一个 FastAPI 应用可注册多个 Chain")
    print("   - 每个 Chain 对应不同的 AI 能力")
    print("   - 通过不同的 path 区分不同端点")
    print()

    server_code = '''
# ============ 服务端代码 ============
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from src.utils.llm_loader import get_default_llm

app = FastAPI(
    title="多端点 AI 服务",
    version="1.0",
    description="同时提供翻译、总结、改写等多种 AI 能力"
)

llm = get_default_llm()

# --- 翻译 Chain ---
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业翻译，将文本翻译成{language}。只输出翻译结果。"),
    ("human", "{text}")
])
translate_chain = translate_prompt | llm | StrOutputParser()
add_routes(app, translate_chain, path="/translate")

# --- 总结 Chain ---
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业总结助手。请用{length}字以内总结以下内容："),
    ("human", "{text}")
])
summarize_chain = summarize_prompt | llm | StrOutputParser()
add_routes(app, summarize_chain, path="/summarize")

# --- 改写 Chain ---
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业写作助手。请将以下文本改写为{style}风格："),
    ("human", "{text}")
])
rewrite_chain = rewrite_prompt | llm | StrOutputParser()
add_routes(app, rewrite_chain, path="/rewrite")

# --- 问答 Chain ---
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是知识渊博的助手。请详细回答以下问题："),
    ("human", "{question}")
])
qa_chain = qa_prompt | llm | StrOutputParser()
add_routes(app, qa_chain, path="/qa")

# 启动: uvicorn server:app --port 8000
# 访问 http://localhost:8000/docs 查看所有端点
'''
    print(server_code)

    endpoints_map = '''
┌──────────────────────────────────────────────────────────────┐
│  多端点服务 API 映射                                          │
├──────────────┬───────────────────────────────────────────────┤
│  路径         │  功能                                         │
├──────────────┼───────────────────────────────────────────────┤
│  /translate  │  翻译服务 (invoke/batch/stream)                │
│  /summarize  │  总结服务 (invoke/batch/stream)                │
│  /rewrite    │  改写服务 (invoke/batch/stream)                │
│  /qa         │  问答服务 (invoke/batch/stream)                │
│  /docs       │  Swagger UI 交互式文档                         │
│  /redoc      │  ReDoc 格式文档                                │
└──────────────┴───────────────────────────────────────────────┘
'''
    print(endpoints_map)

    client_code = '''
# ============ 客户端调用 ============
import requests

BASE = "http://localhost:8000"

# 翻译
r = requests.post(f"{BASE}/translate/invoke", json={
    "input": {"text": "今天天气真好", "language": "英语"}
})
print("翻译:", r.json()["output"])

# 总结
r = requests.post(f"{BASE}/summarize/invoke", json={
    "input": {"text": "人工智能是...", "length": "50"}
})
print("总结:", r.json()["output"])

# 改写
r = requests.post(f"{BASE}/rewrite/invoke", json={
    "input": {"text": "这个产品很好用", "style": "学术"}
})
print("改写:", r.json()["output"])

# 问答
r = requests.post(f"{BASE}/qa/invoke", json={
    "input": {"question": "什么是量子纠缠？"}
})
print("问答:", r.json()["output"])
'''
    print(client_code)

    # 交互式模拟
    print("\n📋 模拟演示：多端点调用")
    print("-" * 40)

    try:
        llm = get_default_llm()

        # 创建多个 Chain
        chains = {
            "1": {
                "name": "翻译",
                "chain": ChatPromptTemplate.from_messages([
                    ("system", "你是专业翻译，将文本翻译成{language}。只输出翻译结果。"),
                    ("human", "{text}")
                ]) | llm | StrOutputParser(),
                "inputs": lambda: {
                    "text": input("  输入要翻译的文本: ").strip() or "人工智能改变世界",
                    "language": input("  目标语言: ").strip() or "英语"
                }
            },
            "2": {
                "name": "总结",
                "chain": ChatPromptTemplate.from_messages([
                    ("system", "你是专业总结助手。请用{length}字以内总结以下内容："),
                    ("human", "{text}")
                ]) | llm | StrOutputParser(),
                "inputs": lambda: {
                    "text": input("  输入要总结的文本: ").strip() or "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。",
                    "length": input("  总结字数限制: ").strip() or "30"
                }
            },
            "3": {
                "name": "改写",
                "chain": ChatPromptTemplate.from_messages([
                    ("system", "你是专业写作助手。请将以下文本改写为{style}风格："),
                    ("human", "{text}")
                ]) | llm | StrOutputParser(),
                "inputs": lambda: {
                    "text": input("  输入要改写的文本: ").strip() or "这个产品很好用",
                    "style": input("  目标风格(学术/口语/文艺): ").strip() or "学术"
                }
            },
            "4": {
                "name": "问答",
                "chain": ChatPromptTemplate.from_messages([
                    ("system", "你是知识渊博的助手，请用中文回答。"),
                    ("human", "{question}")
                ]) | llm | StrOutputParser(),
                "inputs": lambda: {
                    "question": input("  输入你的问题: ").strip() or "什么是量子纠缠？"
                }
            }
        }

        print("\n可用端点:")
        for k, v in chains.items():
            print(f"  {k}. /{v['name']} - {v['name']}服务")

        choice = input("\n选择端点 (1-4): ").strip()
        if choice in chains:
            selected = chains[choice]
            inputs = selected["inputs"]()
            print(f"\n📤 调用 /{selected['name']}/invoke")
            result = selected["chain"].invoke(inputs)
            print(f"📥 结果: {result}")
        else:
            print("⚠️ 无效选择")

    except Exception as e:
        print(f"❌ 演示失败: {e}")

    print("\n💡 要点总结：")
    print("   1. 多次调用 add_routes 可注册多个端点")
    print("   2. 每个 Chain 有独立的输入/输出 Schema")
    print("   3. /docs 页面可同时查看所有端点的文档")


# ============================================================
# 示例4: 交互式测试 - API测试工具
# ============================================================

def demo_interactive_test():
    """
    示例4：交互式测试 - API测试工具

    提供一个交互式工具，用户可以：
    1. 输入内容
    2. 选择调用方式（invoke/batch/stream）
    3. 查看API请求和响应
    """
    print("\n" + "="*60)
    print("示例4：交互式测试 - API测试工具")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 交互式测试：模拟 API 调用流程")
    print("   - invoke/batch/stream：三种调用模式")
    print("   - 请求-响应格式：了解 API 的输入输出结构")
    print()

    # 展示完整的交互式服务
    server_code = '''
# ============ 服务端代码 ============
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from src.utils.llm_loader import get_default_llm

app = FastAPI(title="交互式测试 API")
llm = get_default_llm()

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，请用中文回答。"),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

add_routes(app, chain, path="/ask")

# 启动: uvicorn server:app --port 8000
'''
    print(server_code)

    # 完整的交互式客户端
    client_code = '''
# ============ 交互式客户端 ============
import requests
import json
import httpx

BASE_URL = "http://localhost:8000"

def test_invoke(question: str):
    """测试 invoke 端点"""
    print(f"\\n--- invoke 调用 ---")
    payload = {"input": {"question": question}}
    print(f"请求: POST {BASE_URL}/ask/invoke")
    print(f"请求体: {json.dumps(payload, ensure_ascii=False)}")
    response = requests.post(f"{BASE_URL}/ask/invoke", json=payload)
    print(f"响应: {response.json()}")
    return response.json()

def test_batch(questions: list):
    """测试 batch 端点"""
    print(f"\\n--- batch 调用 ---")
    payload = {"inputs": [{"question": q} for q in questions]}
    print(f"请求: POST {BASE_URL}/ask/batch")
    print(f"批量数量: {len(questions)}")
    response = requests.post(f"{BASE_URL}/ask/batch", json=payload)
    for i, r in enumerate(response.json()):
        print(f"  问题{i+1} 结果: {r['output']}")
    return response.json()

def test_stream(question: str):
    """测试 stream 端点"""
    print(f"\\n--- stream 调用 ---")
    payload = {"input": {"question": question}}
    print(f"请求: POST {BASE_URL}/ask/stream")
    print(f"流式输出: ", end="")
    with httpx.stream("POST", f"{BASE_URL}/ask/stream",
                       json=payload, timeout=60) as resp:
        for line in resp.iter_lines():
            if line.startswith("data:"):
                data = json.loads(line[5:].strip())
                if "output" in data:
                    print(data["output"], end="", flush=True)
    print()

# 交互式循环
while True:
    q = input("\\n输入问题 (q退出): ")
    if q.lower() == 'q':
        break
    mode = input("调用模式 (invoke/batch/stream): ").strip()
    if mode == "batch":
        test_batch([q, f"详细解释{q}", f"{q}的应用"])
    elif mode == "stream":
        test_stream(q)
    else:
        test_invoke(q)
'''
    print(client_code)

    # 本地交互式模拟
    print("\n📋 模拟演示：交互式 API 测试")
    print("-" * 40)

    try:
        llm = get_default_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手，请用中文回答。"),
            ("human", "{question}")
        ])
        chain = prompt | llm | StrOutputParser()

        while True:
            print("\n可用操作:")
            print("  1. invoke - 单次调用")
            print("  2. batch  - 批量调用")
            print("  3. stream - 流式输出（模拟）")
            print("  0. 返回")

            choice = input("\n选择操作: ").strip()

            if choice == "0":
                break
            elif choice == "1":
                question = input("输入问题: ").strip()
                if not question:
                    question = "什么是 LangServe？"

                print(f"\n📤 POST /ask/invoke")
                print(f"   请求体: {{\"input\": {{\"question\": \"{question}\"}}}}")

                result = chain.invoke({"question": question})
                print(f"📥 响应:")
                print(f"   {{\"output\": \"{result}\"}}")

            elif choice == "2":
                q1 = input("问题1 (回车默认): ").strip() or "什么是AI？"
                q2 = input("问题2 (回车默认): ").strip() or "什么是机器学习？"
                q3 = input("问题3 (回车默认): ").strip() or "什么是深度学习？"

                print(f"\n📤 POST /ask/batch")
                print(f"   批量数量: 3")

                results = chain.batch([
                    {"question": q1},
                    {"question": q2},
                    {"question": q3}
                ])
                print(f"📥 批量响应:")
                for i, (q, r) in enumerate(zip([q1, q2, q3], results)):
                    print(f"   [{i+1}] {q} → {r}")

            elif choice == "3":
                question = input("输入问题: ").strip()
                if not question:
                    question = "用三句话解释什么是 LangServe"

                print(f"\n📤 POST /ask/stream")
                print(f"   流式输出: ", end="", flush=True)

                # 使用流式输出模拟
                streaming_llm = get_default_llm()
                streaming_chain = prompt | streaming_llm | StrOutputParser()
                for chunk in streaming_chain.stream({"question": question}):
                    print(chunk, end="", flush=True)
                print()

            else:
                print("⚠️ 无效选择")

    except Exception as e:
        print(f"❌ 演示失败: {e}")

    print("\n💡 要点总结：")
    print("   1. invoke: 适合简单问答，一次请求一次响应")
    print("   2. batch: 适合批量处理，减少请求开销")
    print("   3. stream: 适合长文本生成，实时查看进度")


# ============================================================
# 交互式主菜单
# ============================================================

def main():
    """交互式主菜单"""
    print("\n" + "="*60)
    print("  LangServe 基础部署 - 交互式案例")
    print("="*60)
    print("\n📚 核心概念回顾：")
    print("   - FastAPI: Web 框架，LangServe 基于它构建")
    print("   - add_routes: 将 Runnable 自动注册为 API")
    print("   - invoke: 单次调用  |  batch: 批量调用")
    print("   - stream: 流式输出  |  /docs: 自动文档")

    demos = {
        "1": ("FastAPI集成 - 创建API服务", demo_fastapi_integration),
        "2": ("add_routes - 添加路由", demo_add_routes),
        "3": ("多端点服务 - 多个API端点", demo_multi_endpoint),
        "4": ("交互式测试 - API测试工具", demo_interactive_test),
    }

    while True:
        print("\n" + "-"*60)
        print("可用示例：")
        for key, (name, _) in demos.items():
            print(f"  {key}. {name}")
        print("  0. 退出")

        choice = input("\n请选择示例编号: ").strip()

        if choice == "0":
            print("\n👋 再见！")
            break
        elif choice in demos:
            demos[choice][1]()
        else:
            print("⚠️ 无效选择，请重新输入")


if __name__ == "__main__":
    main()
