"""
LangServe 高级部署 - 交互式实战案例
==========================================

本示例演示 LangServe 的高级用法，包括远程调用、流式传输、异步调用和自定义端点。

核心概念：
- RemoteRunnable: LangServe 提供的远程调用客户端
  通过 RemoteRunnable，客户端可以像调用本地 Runnable 一样调用远程的 LangServe 服务。
  无需手动构造 HTTP 请求，直接使用 .invoke()/.batch()/.stream() 等方法。
  示例: remote = RemoteRunnable("http://localhost:8000/chat")
        result = remote.invoke({"question": "你好"})

- 流式传输 (SSE): 服务器发送事件 (Server-Sent Events)
  LangServe 的 /stream 端点使用 SSE 协议实现流式输出。
  服务器逐步生成内容，客户端实时接收每个 chunk，
  适用于长文本生成、实时对话等场景，用户体验更好。

- 异步调用: 使用 async/await 模式调用 API
  FastAPI 原生支持异步，LangServe 也提供了异步端点：
  - /ainvoke: 异步单次调用
  - /abatch:  异步批量调用
  - /astream: 异步流式输出
  适合高并发场景，可以在等待 LLM 响应时处理其他请求。

- 自定义端点: 在 LangServe 服务上扩展自定义 API
  除了 add_routes 自动生成的端点，还可以使用 FastAPI 的
  @app.get/@app.post 等装饰器添加自定义端点，实现更复杂的业务逻辑。

应用场景：
- 微服务架构中远程调用 AI 服务
- 实时对话、长文本生成的流式输出
- 高并发场景的异步处理
- 需要额外业务逻辑的自定义 API

依赖安装：
    pip install langserve fastapi uvicorn httpx sse-starlette
"""

import os
import sys
import json
import asyncio

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_loader import get_default_llm


# ============================================================
# 示例1: RemoteRunnable - 远程调用
# ============================================================

def demo_remote_runnable():
    """
    示例1：RemoteRunnable - 从客户端调用远程LangServe服务

    演示如何使用 RemoteRunnable 在客户端调用远程的 LangServe API，
    像调用本地 Runnable 一样调用远程服务。
    """
    print("\n" + "="*60)
    print("示例1：RemoteRunnable - 远程调用")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - RemoteRunnable: LangServe 的远程调用客户端")
    print("   - 像调用本地 Runnable 一样调用远程服务")
    print("   - 支持 invoke/batch/stream 全部方法")
    print()

    # 服务端代码
    server_code = '''
# ============ 服务端代码 ============
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from src.utils.llm_loader import get_default_llm

app = FastAPI(title="RemoteRunnable 演示服务")
llm = get_default_llm()

# 问答 Chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，请用中文简洁回答。"),
    ("human", "{question}")
])
qa_chain = qa_prompt | llm | StrOutputParser()
add_routes(app, qa_chain, path="/qa")

# 翻译 Chain
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是翻译专家，将文本翻译成{language}。只输出翻译结果。"),
    ("human", "{text}")
])
translate_chain = translate_prompt | llm | StrOutputParser()
add_routes(app, translate_chain, path="/translate")

# 启动: uvicorn server:app --port 8000
'''
    print(server_code)

    # 客户端代码
    client_code = '''
# ============ 客户端代码（使用 RemoteRunnable）============
from langserve import RemoteRunnable

# 连接远程服务 — 像本地对象一样使用
qa_remote = RemoteRunnable("http://localhost:8000/qa")
translate_remote = RemoteRunnable("http://localhost:8000/translate")

# --- invoke: 远程单次调用 ---
result = qa_remote.invoke({"question": "什么是量子计算？"})
print("问答结果:", result)

# --- batch: 远程批量调用 ---
results = qa_remote.batch([
    {"question": "什么是AI？"},
    {"question": "什么是ML？"},
    {"question": "什么是DL？"},
])
for r in results:
    print("批量结果:", r)

# --- stream: 远程流式输出 ---
for chunk in qa_remote.stream({"question": "解释相对论"}):
    print(chunk, end="", flush=True)
print()

# --- 调用不同的远程端点 ---
result = translate_remote.invoke({"text": "你好世界", "language": "英语"})
print("翻译结果:", result)
'''
    print(client_code)

    # 模拟演示
    print("\n📋 模拟演示：本地调用 Chain（模拟 RemoteRunnable 行为）")
    print("-" * 40)

    try:
        llm = get_default_llm()
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手，请用中文简洁回答。"),
            ("human", "{question}")
        ])
        qa_chain = qa_prompt | llm | StrOutputParser()

        question = input("\n请输入问题（回车使用默认）: ").strip()
        if not question:
            question = "用一句话解释什么是量子计算"

        print(f"\n📤 模拟 RemoteRunnable.invoke()")
        print(f"   remote = RemoteRunnable('http://localhost:8000/qa')")
        print(f"   remote.invoke({{'question': '{question}'}})")

        result = qa_chain.invoke({"question": question})
        print(f"\n📥 远程返回结果: {result}")

    except Exception as e:
        print(f"❌ 演示失败: {e}")

    print("\n💡 要点总结：")
    print("   1. RemoteRunnable 让远程调用像本地调用一样简单")
    print("   2. 自动处理 HTTP 请求/响应的序列化")
    print("   3. 支持 invoke/batch/stream 全部 Runnable 方法")
    print("   4. 适合微服务架构，AI 服务独立部署")


# ============================================================
# 示例2: 流式传输 - 实时输出
# ============================================================

def demo_streaming():
    """
    示例2：流式传输 - 客户端实时接收流式输出

    演示如何使用 SSE (Server-Sent Events) 实现流式传输，
    让用户实时看到 LLM 的生成过程。
    """
    print("\n" + "="*60)
    print("示例2：流式传输 - 实时输出")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - SSE (Server-Sent Events): 服务器推送协议")
    print("   - LangServe /stream 端点自动使用 SSE")
    print("   - 客户端逐步接收生成的文本块")
    print("   - 用户体验：实时看到生成过程，无需等待完整响应")
    print()

    # 流式传输原理
    sse_diagram = '''
┌─────────────────────────────────────────────────────────┐
│  流式传输 (SSE) 工作原理                                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  客户端                          服务端                  │
│    │                               │                    │
│    │── POST /chat/stream ─────────▶│                    │
│    │                               │── 调用 LLM         │
│    │◀── data: {"output": "你"} ───│◀─ chunk 1          │
│    │◀── data: {"output": "好"} ───│◀─ chunk 2          │
│    │◀── data: {"output": "，"} ───│◀─ chunk 3          │
│    │◀── data: {"output": "世"} ───│◀─ chunk 4          │
│    │◀── data: {"output": "界"} ───│◀─ chunk 5          │
│    │◀── data: [DONE] ─────────────│◀─ 结束             │
│    │                               │                    │
└─────────────────────────────────────────────────────────┘
'''
    print(sse_diagram)

    server_code = '''
# ============ 服务端代码 ============
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from src.utils.llm_loader import get_default_llm, create_streaming_llm

app = FastAPI(title="流式传输演示")

# 使用流式 LLM
streaming_llm = create_streaming_llm()

story_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个创意写作助手。请根据用户的主题写一段短文。"),
    ("human", "主题：{topic}")
])
story_chain = story_prompt | streaming_llm | StrOutputParser()
add_routes(app, story_chain, path="/story")

# 启动: uvicorn server:app --port 8000
'''
    print(server_code)

    client_code = '''
# ============ 客户端代码（流式调用）============

# 方式1: 使用 RemoteRunnable（推荐）
from langserve import RemoteRunnable

remote = RemoteRunnable("http://localhost:8000/story")
print("流式输出: ", end="")
for chunk in remote.stream({"topic": "星空"}):
    print(chunk, end="", flush=True)
print()

# 方式2: 使用 httpx 直接调用 SSE
import httpx
import json

print("\\n流式输出: ", end="")
with httpx.stream(
    "POST",
    "http://localhost:8000/story/stream",
    json={"input": {"topic": "星空"}},
    timeout=60
) as response:
    for line in response.iter_lines():
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            data = json.loads(data_str)
            if "output" in data:
                print(data["output"], end="", flush=True)
print()
'''
    print(client_code)

    # 本地模拟流式输出
    print("\n📋 模拟演示：本地流式输出")
    print("-" * 40)

    try:
        from src.utils.llm_loader import create_streaming_llm

        streaming_llm = create_streaming_llm()
        story_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个创意写作助手。请根据用户的主题写一段短文（100字以内）。"),
            ("human", "主题：{topic}")
        ])
        chain = story_prompt | streaming_llm | StrOutputParser()

        topic = input("\n请输入写作主题（回车使用默认）: ").strip()
        if not topic:
            topic = "星空下的旅行"

        print(f"\n📤 模拟流式调用: POST /story/stream")
        print(f"   请求体: {{\"input\": {{\"topic\": \"{topic}\"}}}}")
        print(f"\n📥 流式输出: ", end="", flush=True)

        for chunk in chain.stream({"topic": topic}):
            print(chunk, end="", flush=True)
        print()

    except Exception as e:
        print(f"❌ 演示失败: {e}")

    print("\n💡 要点总结：")
    print("   1. SSE 协议实现服务器到客户端的实时推送")
    print("   2. 用户无需等待完整响应，体验更好")
    print("   3. RemoteRunnable.stream() 封装了 SSE 解析")
    print("   4. 长文本生成场景必用流式输出")


# ============================================================
# 示例3: 异步调用 - 高并发
# ============================================================

def demo_async_call():
    """
    示例3：异步调用 - 使用async方式调用API

    演示如何使用异步调用提高并发性能，适合高并发场景。
    """
    print("\n" + "="*60)
    print("示例3：异步调用 - 高并发")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 异步调用: 使用 async/await 提高并发性能")
    print("   - LangServe 自动提供 /ainvoke, /abatch, /astream")
    print("   - 在等待 LLM 响应时可处理其他请求")
    print("   - 适合高并发生产环境")
    print()

    # 同步 vs 异步对比
    sync_async_compare = '''
┌─────────────────────────────────────────────────────────┐
│  同步 vs 异步 调用对比                                    │
├─────────────────────────┬───────────────────────────────┤
│  同步 (sync)            │  异步 (async)                  │
├─────────────────────────┼───────────────────────────────┤
│  请求1 ──等待──▶响应1   │  请求1 ──┐                     │
│  请求2 ──等待──▶响应2   │  请求2 ──┼──并行──▶响应1,2,3  │
│  请求3 ──等待──▶响应3   │  请求3 ──┘                     │
│  总时间 = 3x            │  总时间 ≈ 1x                   │
└─────────────────────────┴───────────────────────────────┘
'''
    print(sync_async_compare)

    server_code = '''
# ============ 服务端代码 ============
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from src.utils.llm_loader import get_default_llm

app = FastAPI(title="异步调用演示")
llm = get_default_llm()

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手，请简洁回答。"),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()
add_routes(app, chain, path="/ask")

# LangServe 自动提供异步端点:
#   /ask/ainvoke — 异步单次调用
#   /ask/abatch  — 异步批量调用
#   /ask/astream — 异步流式输出

# 启动: uvicorn server:app --port 8000
'''
    print(server_code)

    client_code = '''
# ============ 客户端代码（异步调用）============
import asyncio
import httpx
import time

BASE = "http://localhost:8000"

async def async_invoke(client, question: str):
    """异步单次调用"""
    response = await client.post(
        f"{BASE}/ask/ainvoke",
        json={"input": {"question": question}}
    )
    return response.json()

async def async_batch(client, questions: list):
    """异步批量调用"""
    response = await client.post(
        f"{BASE}/ask/abatch",
        json={"inputs": [{"question": q} for q in questions]}
    )
    return response.json()

async def main():
    async with httpx.AsyncClient(timeout=60) as client:
        # --- 异步并发调用 ---
        print("=== 异步并发调用 ===")
        start = time.time()
        tasks = [
            async_invoke(client, "什么是AI？"),
            async_invoke(client, "什么是ML？"),
            async_invoke(client, "什么是DL？"),
        ]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        for i, r in enumerate(results):
            print(f"结果{i+1}: {r['output']}")
        print(f"耗时: {elapsed:.2f}s")

        # --- 异步批量调用 ---
        print("\\n=== 异步批量调用 ===")
        start = time.time()
        results = await async_batch(client, [
            "什么是Python？",
            "什么是FastAPI？",
        ])
        elapsed = time.time() - start
        for r in results:
            print(f"结果: {r['output']}")
        print(f"耗时: {elapsed:.2f}s")

asyncio.run(main())
'''
    print(client_code)

    # 本地异步模拟
    print("\n📋 模拟演示：本地异步调用")
    print("-" * 40)

    async def run_async_demo():
        try:
            llm = get_default_llm()
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个助手，请用一句话回答。"),
                ("human", "{question}")
            ])
            chain = prompt | llm | StrOutputParser()

            questions_input = input("\n输入多个问题，用逗号分隔（回车使用默认）: ").strip()
            if not questions_input:
                questions = ["什么是AI？", "什么是机器学习？", "什么是深度学习？"]
            else:
                questions = [q.strip() for q in questions_input.split(",") if q.strip()]

            print(f"\n📤 异步并发调用 {len(questions)} 个问题...")

            # 使用 ainvoke 异步调用
            import time
            start = time.time()
            tasks = [chain.ainvoke({"question": q}) for q in questions]
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start

            for i, (q, r) in enumerate(zip(questions, results)):
                print(f"   [{i+1}] {q} → {r}")
            print(f"\n⏱️ 异步并发耗时: {elapsed:.2f}s")

            # 对比同步调用
            print(f"\n📤 同步顺序调用 {len(questions)} 个问题...")
            start = time.time()
            sync_results = []
            for q in questions:
                r = chain.invoke({"question": q})
                sync_results.append(r)
            elapsed_sync = time.time() - start

            for i, (q, r) in enumerate(zip(questions, sync_results)):
                print(f"   [{i+1}] {q} → {r}")
            print(f"\n⏱️ 同步顺序耗时: {elapsed_sync:.2f}s")

        except Exception as e:
            print(f"❌ 演示失败: {e}")

    asyncio.run(run_async_demo())

    print("\n💡 要点总结：")
    print("   1. 异步调用可并行处理多个请求，显著提高吞吐量")
    print("   2. LangServe 自动提供 ainvoke/abatch/astream 端点")
    print("   3. 使用 httpx.AsyncClient 进行异步 HTTP 调用")
    print("   4. asyncio.gather() 实现并发等待")


# ============================================================
# 示例4: 自定义端点 - 扩展API
# ============================================================

def demo_custom_endpoint():
    """
    示例4：自定义端点 - 添加自定义的API端点

    演示如何在 LangServe 服务上添加自定义 API 端点，
    实现更复杂的业务逻辑，如带缓存的调用、带认证的接口等。
    """
    print("\n" + "="*60)
    print("示例4：自定义端点 - 扩展API")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 自定义端点: 在 LangServe 服务上添加额外 API")
    print("   - 使用 FastAPI 装饰器 @app.get/@app.post")
    print("   - 可以组合多个 Chain 的结果")
    print("   - 可以添加中间件、认证、缓存等逻辑")
    print()

    server_code = '''
# ============ 服务端代码（含自定义端点）============
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from src.utils.llm_loader import get_default_llm
from pydantic import BaseModel
from typing import Optional
import time
import hashlib
import json

app = FastAPI(title="自定义端点演示")
llm = get_default_llm()

# ---------- LangServe 标准端点 ----------
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手，请用中文回答。"),
    ("human", "{question}")
])
qa_chain = qa_prompt | llm | StrOutputParser()
add_routes(app, qa_chain, path="/qa")

# ---------- 自定义数据模型 ----------
class ChatRequest(BaseModel):
    """聊天请求"""
    question: str
    style: Optional[str] = "简洁"  # 回答风格
    language: Optional[str] = "中文"  # 回答语言

class ChatResponse(BaseModel):
    """聊天响应"""
    question: str
    answer: str
    style: str
    language: str
    latency_ms: float

# ---------- 简单缓存 ----------
cache = {}

def get_cache_key(question: str, style: str) -> str:
    """生成缓存键"""
    content = f"{question}:{style}"
    return hashlib.md5(content.encode()).hexdigest()

# ---------- 自定义端点1: 带缓存的聊天 ----------
@app.post("/chat/cached", response_model=ChatResponse)
async def cached_chat(request: ChatRequest):
    """带缓存的聊天接口 — 相同问题直接返回缓存"""
    start = time.time()
    cache_key = get_cache_key(request.question, request.style)

    # 检查缓存
    if cache_key in cache:
        cached = cache[cache_key]
        cached["latency_ms"] = (time.time() - start) * 1000
        print(f"  [缓存命中] {request.question[:20]}...")
        return cached

    # 缓存未命中，调用 LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"你是助手，请用{request.language}以{request.style}风格回答。"),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    answer = await chain.ainvoke({"question": request.question})

    result = {
        "question": request.question,
        "answer": answer,
        "style": request.style,
        "language": request.language,
        "latency_ms": (time.time() - start) * 1000
    }

    # 存入缓存
    cache[cache_key] = result
    print(f"  [缓存未命中] {request.question[:20]}...")

    return result

# ---------- 自定义端点2: 多步处理 ----------
@app.post("/analyze")
async def analyze_text(request: Request):
    """多步分析 — 先总结，再提取关键词，最后给出建议"""
    body = await request.json()
    text = body.get("text", "")

    # 步骤1: 总结
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", "请用一句话总结以下内容："),
        ("human", "{text}")
    ])
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    summary = await summarize_chain.ainvoke({"text": text})

    # 步骤2: 提取关键词
    keyword_prompt = ChatPromptTemplate.from_messages([
        ("system", "请提取以下文本的3个关键关键词，用逗号分隔："),
        ("human", "{text}")
    ])
    keyword_chain = keyword_prompt | llm | StrOutputParser()
    keywords = await keyword_chain.ainvoke({"text": text})

    # 步骤3: 给出建议
    advice_prompt = ChatPromptTemplate.from_messages([
        ("system", "基于以下总结，给出一条简短建议：\\n总结：{summary}"),
        ("human", "请给出建议")
    ])
    advice_chain = advice_prompt | llm | StrOutputParser()
    advice = await advice_chain.ainvoke({"summary": summary})

    return {
        "original_text": text,
        "summary": summary,
        "keywords": keywords,
        "advice": advice
    }

# ---------- 自自定义端点3: 健康检查 ----------
@app.get("/health")
async def health_check():
    """服务健康检查"""
    return {
        "status": "healthy",
        "service": "LangServe 自定义端点演示",
        "cache_size": len(cache),
        "version": "1.0.0"
    }

# ---------- 自定义端点4: 清除缓存 ----------
@app.delete("/cache")
async def clear_cache():
    """清除缓存"""
    size = len(cache)
    cache.clear()
    return {"message": f"已清除 {size} 条缓存"}

# 启动: uvicorn server:app --port 8000
'''
    print(server_code)

    client_code = '''
# ============ 客户端调用 ============
import requests

BASE = "http://localhost:8000"

# 1. 带缓存的聊天
print("=== 带缓存的聊天 ===")
r = requests.post(f"{BASE}/chat/cached", json={
    "question": "什么是量子计算？",
    "style": "简洁",
    "language": "中文"
})
data = r.json()
print(f"回答: {data['answer']}")
print(f"耗时: {data['latency_ms']:.0f}ms")

# 再次调用（缓存命中）
r = requests.post(f"{BASE}/chat/cached", json={
    "question": "什么是量子计算？",
    "style": "简洁",
    "language": "中文"
})
data = r.json()
print(f"回答（缓存）: {data['answer']}")
print(f"耗时: {data['latency_ms']:.0f}ms")

# 2. 多步分析
print("\\n=== 多步分析 ===")
r = requests.post(f"{BASE}/analyze", json={
    "text": "人工智能正在改变各个行业，从医疗到金融，从教育到交通。"
})
data = r.json()
print(f"总结: {data['summary']}")
print(f"关键词: {data['keywords']}")
print(f"建议: {data['advice']}")

# 3. 健康检查
print("\\n=== 健康检查 ===")
r = requests.get(f"{BASE}/health")
print(r.json())

# 4. 清除缓存
print("\\n=== 清除缓存 ===")
r = requests.delete(f"{BASE}/cache")
print(r.json())
'''
    print(client_code)

    # 本地交互式模拟
    print("\n📋 模拟演示：自定义端点功能")
    print("-" * 40)

    try:
        llm = get_default_llm()

        # 模拟缓存聊天
        print("\n--- 模拟1: 带缓存的聊天 ---")
        question = input("输入问题（回车默认）: ").strip()
        if not question:
            question = "什么是量子计算？"

        import time
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是助手，请用中文简洁回答。"),
            ("human", "{question}")
        ])
        chain = prompt | llm | StrOutputParser()

        start = time.time()
        result = chain.invoke({"question": question})
        latency = (time.time() - start) * 1000
        print(f"首次调用: {result}")
        print(f"耗时: {latency:.0f}ms")

        # 模拟缓存命中
        print(f"模拟缓存命中: {result}")
        print(f"耗时: 1ms (缓存)")

        # 模拟多步分析
        print("\n--- 模拟2: 多步分析 ---")
        text = input("输入要分析的文本（回车默认）: ").strip()
        if not text:
            text = "人工智能正在改变各个行业，从医疗到金融，从教育到交通。"

        # 步骤1: 总结
        summarize_chain = ChatPromptTemplate.from_messages([
            ("system", "请用一句话总结以下内容："),
            ("human", "{text}")
        ]) | llm | StrOutputParser()
        summary = summarize_chain.invoke({"text": text})
        print(f"总结: {summary}")

        # 步骤2: 关键词
        keyword_chain = ChatPromptTemplate.from_messages([
            ("system", "请提取以下文本的3个关键关键词，用逗号分隔："),
            ("human", "{text}")
        ]) | llm | StrOutputParser()
        keywords = keyword_chain.invoke({"text": text})
        print(f"关键词: {keywords}")

        # 步骤3: 建议
        advice_chain = ChatPromptTemplate.from_messages([
            ("system", "基于以下总结，给出一条简短建议：\n总结：{summary}"),
            ("human", "请给出建议")
        ]) | llm | StrOutputParser()
        advice = advice_chain.invoke({"summary": summary})
        print(f"建议: {advice}")

        # 模拟健康检查
        print("\n--- 模拟3: 健康检查 ---")
        print({"status": "healthy", "service": "LangServe", "cache_size": 1, "version": "1.0.0"})

    except Exception as e:
        print(f"❌ 演示失败: {e}")

    print("\n💡 要点总结：")
    print("   1. 自定义端点可实现缓存、多步处理等复杂逻辑")
    print("   2. Pydantic 模型确保请求/响应的数据校验")
    print("   3. 健康检查端点是生产环境必备")
    print("   4. 缓存可显著减少重复调用的延迟和成本")


# ============================================================
# 交互式主菜单
# ============================================================

def main():
    """交互式主菜单"""
    print("\n" + "="*60)
    print("  LangServe 高级部署 - 交互式案例")
    print("="*60)
    print("\n📚 核心概念回顾：")
    print("   - RemoteRunnable: 远程调用 LangServe 服务")
    print("   - 流式传输 (SSE): 实时推送生成内容")
    print("   - 异步调用: async/await 高并发处理")
    print("   - 自定义端点: 扩展 API 功能")

    demos = {
        "1": ("RemoteRunnable - 远程调用", demo_remote_runnable),
        "2": ("流式传输 - 实时输出", demo_streaming),
        "3": ("异步调用 - 高并发", demo_async_call),
        "4": ("自定义端点 - 扩展API", demo_custom_endpoint),
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
