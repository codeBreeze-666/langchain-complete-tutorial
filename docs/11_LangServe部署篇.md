# 第十一章：LangServe 部署篇

本章介绍 LangServe 的核心概念与实战用法。LangServe 基于 FastAPI 构建，能将 LangChain 的 Runnable 对象快速暴露为 REST API，是 AI 应用从开发到部署的关键桥梁。

> 下一章：[12_LangSmith调试篇](12_LangSmith调试篇.md) | 上一章：[10_LangGraph高级篇](10_LangGraph高级篇.md)

---

## 11.1 LangServe 概述

### 什么是 LangServe

LangServe 是 LangChain 团队提供的部署工具库，用于将 LangChain 的 Runnable 对象（Chain、Agent 等）快速部署为 REST API 服务。它基于 FastAPI 构建，自动生成 API 端点和交互式文档。

### 为什么需要 LangServe

| 痛点 | 手动部署的困难 | LangServe 的解决方案 |
|------|--------------|---------------------|
| **API 搭建** | 需要手写路由、序列化、文档 | `add_routes()` 自动生成所有端点 |
| **多模式调用** | 需要分别实现 invoke/batch/stream | 自动生成三种调用模式 |
| **类型校验** | 需要手写 Pydantic 模型 | 从 Chain 的 Schema 自动推断 |
| **API 文档** | 需要手写 Swagger/OpenAPI | FastAPI 自动生成交互式文档 |

### LangServe 架构

```
客户端 (httpx / RemoteRunnable / 前端)
    │
    ├── POST /path/invoke   → 单次调用
    ├── POST /path/batch    → 批量调用
    ├── POST /path/stream   → 流式输出 (SSE)
    ├── GET  /path/input_schema  → 输入 Schema
    └── GET  /path/output_schema → 输出 Schema
    │
LangServe (add_routes)
    │
FastAPI 应用
    │
LangChain Runnable (Chain / Agent / Graph)
```

---

## 11.2 核心概念总览

| 概念 | 说明 |
|------|------|
| **FastAPI** | 高性能异步 Web 框架，LangServe 的基础 |
| **add_routes** | 核心函数，将 Runnable 自动注册为 API |
| **invoke** | 单次调用端点，一个输入 → 一个输出 |
| **batch** | 批量调用端点，多个输入 → 多个输出 |
| **stream** | 流式输出端点，SSE 实时返回生成内容 |
| **RemoteRunnable** | 远程调用客户端，像本地调用一样调用远程服务 |

### add_routes 自动生成的端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/path/invoke` | POST | 单次调用：一个输入 → 一个输出 |
| `/path/batch` | POST | 批量调用：多个输入 → 多个输出 |
| `/path/stream` | POST | 流式输出：实时逐块返回结果 |
| `/path/input_schema` | GET | 输入 Schema |
| `/path/output_schema` | GET | 输出 Schema |
| `/docs` | GET | Swagger UI 交互式文档 |

---

## 11.3 基础部署（langserve_basics.py）

### 运行方式

```bash
python src/chains/langserve_basics.py
```

程序启动后进入交互式菜单，可选择运行 4 个示例。

### 示例1：FastAPI 集成 — 创建 API 服务

**场景**：将 LLM 封装为 REST API

**核心代码**：

```python
from fastapi import FastAPI
from langserve import add_routes

# 1. 创建 FastAPI 应用
app = FastAPI(title="LangServe 基础 API", version="1.0")

# 2. 创建 Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的AI助手，请用中文回答。"),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

# 3. 添加路由 — 自动生成 /chat/invoke, /chat/batch, /chat/stream 等端点
add_routes(app, chain, path="/chat")

# 4. 启动服务: uvicorn server:app --host 0.0.0.0 --port 8000
```

**客户端调用**：

```python
import requests

# 单次调用 (invoke)
response = requests.post(
    "http://localhost:8000/chat/invoke",
    json={"input": {"question": "什么是量子计算？"}}
)
result = response.json()
print("调用结果:", result["output"])
```

**学习要点**：

- FastAPI 应用是 LangServe 的基础容器
- `add_routes()` 将 Chain 自动注册为 REST API
- 自动生成 `/invoke`、`/batch`、`/stream` 端点
- 访问 `http://localhost:8000/docs` 查看自动文档

### 示例2：add_routes — 添加路由

**场景**：翻译服务，展示 invoke/batch/stream 三种调用方式

**核心代码**：

```python
# 翻译 Chain
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业翻译，将用户输入翻译成{target_language}。只输出翻译结果。"),
    ("human", "{text}")
])
translate_chain = translate_prompt | llm | StrOutputParser()

# 添加路由
add_routes(app, translate_chain, path="/translate")
```

**学习要点**：

| 调用方式 | 端点 | 适用场景 |
|---------|------|---------|
| invoke | `/translate/invoke` | 单条数据处理 |
| batch | `/translate/batch` | 多数据并行处理 |
| stream | `/translate/stream` | 长文本实时输出 |

### 示例3：多端点服务 — 多个 API 端点

**场景**：一个 FastAPI 应用同时提供翻译、总结、改写、问答等 API

**核心代码**：

```python
# 多次调用 add_routes 注册多个端点
add_routes(app, translate_chain, path="/translate")
add_routes(app, summarize_chain, path="/summarize")
add_routes(app, rewrite_chain, path="/rewrite")
add_routes(app, qa_chain, path="/qa")
```

**端点映射**：

| 路径 | 功能 |
|------|------|
| `/translate` | 翻译服务 (invoke/batch/stream) |
| `/summarize` | 总结服务 (invoke/batch/stream) |
| `/rewrite` | 改写服务 (invoke/batch/stream) |
| `/qa` | 问答服务 (invoke/batch/stream) |

**学习要点**：一个 FastAPI 应用可注册多个 Chain，通过不同 path 区分。

### 示例4：交互式测试 — API 测试工具

**场景**：交互式选择调用方式（invoke/batch/stream），查看请求和响应

**学习要点**：

- invoke：适合简单问答，一次请求一次响应
- batch：适合批量处理，减少请求开销
- stream：适合长文本生成，实时查看进度

---

## 11.4 高级部署（langserve_advanced.py）

### 运行方式

```bash
python src/chains/langserve_advanced.py
```

### 知识点

| 概念 | 说明 |
|------|------|
| **RemoteRunnable** | 远程调用客户端，像本地调用一样调用远程服务 |
| **SSE 流式传输** | Server-Sent Events，服务器推送协议 |
| **异步调用** | async/await 模式，适合高并发 |
| **自定义端点** | 在 LangServe 上扩展额外 API |

### 示例1：RemoteRunnable — 远程调用

**核心代码**：

```python
from langserve import RemoteRunnable

# 连接远程服务 — 像本地对象一样使用
qa_remote = RemoteRunnable("http://localhost:8000/qa")

# invoke: 远程单次调用
result = qa_remote.invoke({"question": "什么是量子计算？"})

# batch: 远程批量调用
results = qa_remote.batch([
    {"question": "什么是AI？"},
    {"question": "什么是ML？"},
])

# stream: 远程流式输出
for chunk in qa_remote.stream({"question": "解释相对论"}):
    print(chunk, end="", flush=True)
```

**学习要点**：

- RemoteRunnable 让远程调用像本地调用一样简单
- 自动处理 HTTP 请求/响应的序列化
- 支持 invoke/batch/stream 全部 Runnable 方法
- 适合微服务架构，AI 服务独立部署

### 示例2：流式传输 — 实时输出

**学习要点**：

- SSE 协议实现服务器到客户端的实时推送
- 用户无需等待完整响应，体验更好
- `RemoteRunnable.stream()` 封装了 SSE 解析
- 长文本生成场景必用流式输出

### 示例3：异步调用 — 高并发

**核心代码**：

```python
async def async_invoke(client, question: str):
    response = await client.post(
        f"{BASE}/ask/ainvoke",
        json={"input": {"question": question}}
    )
    return response.json()

# 异步并发调用
tasks = [async_invoke(client, q) for q in questions]
results = await asyncio.gather(*tasks)
```

**学习要点**：

- 异步调用可并行处理多个请求，显著提高吞吐量
- LangServe 自动提供 `/ainvoke`、`/abatch`、`/astream` 异步端点
- 使用 `httpx.AsyncClient` 进行异步 HTTP 调用

### 示例4：自定义端点 — 扩展 API

**场景**：添加带缓存的调用、多步处理、健康检查等自定义端点

**学习要点**：

- 自定义端点可实现缓存、多步处理等复杂逻辑
- Pydantic 模型确保请求/响应的数据校验
- 健康检查端点是生产环境必备
- 缓存可显著减少重复调用的延迟和成本

---

## 11.5 生产部署（langserve_production.py）

### 运行方式

```bash
python src/chains/langserve_production.py
```

### 知识点

| 概念 | 说明 |
|------|------|
| **Docker 部署** | 将 LangServe 服务容器化，环境一致 |
| **环境配置** | .env 文件管理多环境（开发/测试/生产） |
| **日志监控** | 中间件 + 日志记录，排查问题 |
| **错误处理** | 重试 + 降级 + 超时，保障可用性 |

### 示例1：Docker 部署 — 容器化

**核心代码（Dockerfile）**：

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

**学习要点**：Dockerfile 用 slim 镜像减小体积，HEALTHCHECK 确保自动重启，env_file 管理敏感配置。

### 示例2：环境配置 — 多环境管理

**多环境配置对比**：

| 配置项 | 开发环境 | 测试环境 | 生产环境 |
|-------|---------|-----------|---------|
| LOG_LEVEL | DEBUG | INFO | WARNING |
| TEMPERATURE | 0.9 | 0.7 | 0.5 |
| WORKERS | 1 | 2 | 4 |
| ENABLE_DOCS | true | true | false |

**学习要点**：`.env` 文件管理敏感配置，Pydantic Settings 提供类型安全的配置管理，环境变量优先级：系统环境 > .env 文件 > 默认值。

### 示例3：日志监控 — 生产监控

**学习要点**：

- 日志级别：DEBUG < INFO < WARNING < ERROR < CRITICAL
- 中间件自动记录所有请求的日志和性能指标
- `RotatingFileHandler` 防止日志文件无限增长
- `/metrics` 端点可集成 Prometheus 等监控

### 示例4：错误处理 — 生产级错误处理

**错误处理层次**：

| 层次 | 机制 | 说明 |
|------|------|------|
| 第1层 | 全局异常处理器 | 捕获所有未处理异常，返回友好错误 |
| 第2层 | 重试机制 | 临时性错误自动重试（指数退避） |
| 第3层 | 优雅降级 | 主模型不可用时切换备用或返回缓存 |
| 第4层 | 超时控制 | 设置合理超时，快速失败 |

**学习要点**：全局异常处理器是最后一道防线，重试使用指数退避避免雪崩，优雅降级确保用户始终能收到响应。

---

## 11.6 学习路径与建议

### 推荐学习顺序

```
langserve_basics.py（示例1→2→3→4）
    ↓ 掌握 FastAPI/add_routes/invoke/batch/stream
langserve_advanced.py（示例1→2→3→4）
    ↓ 掌握 RemoteRunnable/流式/异步/自定义端点
langserve_production.py（示例1→2→3→4）
    ↓ 掌握 Docker/环境配置/日志/错误处理
```

### 核心概念掌握检查

| 阶段 | 必须掌握 | 进阶理解 |
|------|---------|---------|
| 基础 | FastAPI、add_routes、invoke/batch/stream | 多端点服务、API 测试 |
| 高级 | RemoteRunnable、SSE 流式 | 异步调用、自定义端点 |
| 生产 | Docker 部署、环境配置 | 日志监控、错误处理 |

### 实战建议

1. **先本地调试**：用 `uvicorn` 本地启动服务，用 `/docs` 页面交互式测试
2. **理解三种调用模式**：invoke 适合简单场景，batch 适合批量，stream 适合长文本
3. **生产必加监控**：日志中间件 + 健康检查 + 错误处理是生产环境三件套
4. **善用 RemoteRunnable**：微服务架构中，用 RemoteRunnable 调用远程 AI 服务
5. **Docker 部署**：容器化确保开发/生产环境一致

> 下一章：[12_LangSmith调试篇](12_LangSmith调试篇.md) → 学习追踪、调试 AI 应用
