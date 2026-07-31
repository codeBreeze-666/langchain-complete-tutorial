"""
LangServe 生产部署 - 交互式实战案例
==========================================

本示例演示 LangServe 的生产级部署，包括 Docker 容器化、多环境配置、日志监控和错误处理。

核心概念：
- Docker 部署: 将 LangServe 服务容器化，实现一致的开发和生产环境
  Dockerfile 定义了应用的运行环境，包括 Python 版本、依赖包、代码等。
  通过容器化部署，可以确保应用在不同环境中行为一致，
  同时方便扩展（横向扩容）和运维（CI/CD 集成）。

- 环境配置: 使用 .env 文件管理不同环境的配置
  开发/测试/生产环境通常需要不同的配置（API Key、端口、日志级别等）。
  通过 .env 文件和 python-dotenv 管理环境变量，
  避免硬编码敏感信息，支持灵活的多环境切换。

- 日志监控: 生产环境必须具备完善的日志和监控
  日志记录帮助排查问题、审计请求、性能分析。
  生产级日志应包含：请求时间、响应时间、错误信息、调用链路等。
  可以集成 Prometheus/Grafana 等监控工具。

- 错误处理: 生产级错误处理包括优雅降级和错误重试
  优雅降级：当 LLM 不可用时，返回缓存结果或默认回答
  错误重试：临时性错误（网络超时）自动重试
  超时控制：设置合理的请求超时时间
  全局异常处理：捕获所有未处理异常，避免服务崩溃

应用场景：
- 企业级 AI 服务部署
- 需要 SLA 保障的生产环境
- 团队协作的多环境开发
- 需要监控和告警的在线服务

依赖安装：
    pip install langserve fastapi uvicorn httpx sse-starlette tenacity
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_loader import get_default_llm


# ============================================================
# 示例1: Docker部署 - 容器化
# ============================================================

def demo_docker_deployment():
    """
    示例1：Docker部署 - 创建Dockerfile，部署LangServe服务

    演示如何将 LangServe 服务容器化，包括 Dockerfile 编写、
    docker-compose 配置、以及容器化部署的最佳实践。
    """
    print("\n" + "="*60)
    print("示例1：Docker部署 - 容器化")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - Docker: 将应用和依赖打包为容器，确保环境一致")
    print("   - Dockerfile: 定义容器镜像的构建步骤")
    print("   - docker-compose: 编排多个容器（应用 + 数据库等）")
    print("   - 容器化优势：可移植、可扩展、可复现")
    print()

    # Dockerfile
    dockerfile = '''
# ============ Dockerfile ============
# 使用 Python 3.12 slim 镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1

# 安装系统依赖
RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# 启动命令（生产环境使用 gunicorn + uvicorn worker）
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
'''
    print(dockerfile)

    # docker-compose.yml
    docker_compose = '''
# ============ docker-compose.yml ============
version: "3.8"

services:
  # LangServe API 服务
  langserve-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: langserve-api
    ports:
      - "${PORT:-8000}:8000"
    environment:
      - MODEL_PROVIDER=${MODEL_PROVIDER:-zhipu}
      - ZHIPU_API_KEY=${ZHIPU_API_KEY}
      - ZHIPU_API_BASE=${ZHIPU_API_BASE}
      - ZHIPU_MODEL_NAME=${ZHIPU_MODEL_NAME:-glm-4.7-flash}
      - TEMPERATURE=${TEMPERATURE:-0.7}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - ./logs:/app/logs    # 日志持久化
    networks:
      - ai-network

  # Redis 缓存（可选）
  redis:
    image: redis:7-alpine
    container_name: langserve-redis
    ports:
      - "6379:6379"
    restart: unless-stopped
    networks:
      - ai-network

networks:
  ai-network:
    driver: bridge
'''
    print(docker_compose)

    # 服务端代码
    server_code = '''
# ============ server.py（Docker 内运行的入口文件）============
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from src.utils.llm_loader import get_default_llm
import os

app = FastAPI(
    title="LangServe 生产服务",
    version=os.getenv("APP_VERSION", "1.0.0"),
)

llm = get_default_llm()

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手，请用中文回答。"),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()
add_routes(app, chain, path="/chat")

@app.get("/health")
async def health():
    return {"status": "healthy", "version": os.getenv("APP_VERSION", "1.0.0")}
'''
    print(server_code)

    # 部署命令
    deploy_commands = '''
# ============ 部署命令 ============

# 构建镜像
docker build -t langserve-api:latest .

# 运行容器
docker run -d \\
    --name langserve-api \\
    -p 8000:8000 \\
    --env-file .env \\
    langserve-api:latest

# 使用 docker-compose 启动
docker-compose up -d

# 查看日志
docker-compose logs -f langserve-api

# 扩容（增加 worker 数量）
docker-compose up -d --scale langserve-api=3

# 停止服务
docker-compose down
'''
    print(deploy_commands)

    # 本地模拟
    print("\n📋 模拟演示：容器化环境检查")
    print("-" * 40)

    try:
        # 检查依赖
        print("\n检查生产环境依赖:")
        dependencies = {
            "fastapi": False,
            "uvicorn": False,
            "langchain": False,
        }
        for dep in dependencies:
            try:
                __import__(dep)
                dependencies[dep] = True
                print(f"   ✅ {dep}: 已安装")
            except ImportError:
                print(f"   ❌ {dep}: 未安装")

        # 检查环境变量
        print("\n检查环境变量:")
        env_vars = ["MODEL_PROVIDER", "ZHIPU_API_KEY", "ZHIPU_API_BASE", "ZHIPU_MODEL_NAME"]
        for var in env_vars:
            val = os.getenv(var, "")
            if val:
                if "KEY" in var:
                    print(f"   ✅ {var}: {'*' * 8}{val[-4:]}")
                else:
                    print(f"   ✅ {var}: {val}")
            else:
                print(f"   ⚠️ {var}: 未设置")

        # 模拟健康检查
        print("\n模拟健康检查:")
        health_status = {
            "status": "healthy",
            "version": os.getenv("APP_VERSION", "1.0.0"),
            "timestamp": datetime.now().isoformat(),
            "python": sys.version.split()[0],
        }
        print(f"   {json.dumps(health_status, ensure_ascii=False, indent=2)}")

    except Exception as e:
        print(f"❌ 检查失败: {e}")

    print("\n💡 要点总结：")
    print("   1. Dockerfile 用 slim 镜像，减小体积")
    print("   2. HEALTHCHECK 确保容器自动重启")
    print("   3. env_file + .env 管理敏感配置")
    print("   4. docker-compose 编排多容器，方便扩展")


# ============================================================
# 示例2: 环境配置 - 多环境管理
# ============================================================

def demo_environment_config():
    """
    示例2：环境配置 - 开发/测试/生产环境配置

    演示如何管理不同环境的配置，包括 .env 文件、
    环境变量覆盖、配置验证等。
    """
    print("\n" + "="*60)
    print("示例2：环境配置 - 多环境管理")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - .env 文件: 存储环境变量，不提交到版本控制")
    print("   - 多环境: .env.dev / .env.test / .env.prod")
    print("   - 配置优先级: 环境变量 > .env 文件 > 默认值")
    print("   - Pydantic Settings: 类型安全的配置管理")
    print()

    # 多环境配置文件
    env_files = '''
# ============ .env.dev（开发环境）============
MODEL_PROVIDER=zhipu
ZHIPU_API_KEY=your-dev-api-key
ZHIPU_API_BASE=https://open.bigmodel.cn/api/paas/v4
ZHIPU_MODEL_NAME=glm-4.7-flash
TEMPERATURE=0.9
LOG_LEVEL=DEBUG
PORT=8000
APP_VERSION=0.1.0-dev
WORKERS=1
ENABLE_DOCS=true

# ============ .env.test（测试环境）============
MODEL_PROVIDER=zhipu
ZHIPU_API_KEY=your-test-api-key
ZHIPU_API_BASE=https://open.bigmodel.cn/api/paas/v4
ZHIPU_MODEL_NAME=glm-4.7-flash
TEMPERATURE=0.7
LOG_LEVEL=INFO
PORT=8000
APP_VERSION=0.2.0-test
WORKERS=2
ENABLE_DOCS=true

# ============ .env.prod（生产环境）============
MODEL_PROVIDER=zhipu
ZHIPU_API_KEY=your-prod-api-key
ZHIPU_API_BASE=https://open.bigmodel.cn/api/paas/v4
ZHIPU_MODEL_NAME=glm-4.7-flash
TEMPERATURE=0.5
LOG_LEVEL=WARNING
PORT=8000
APP_VERSION=1.0.0
WORKERS=4
ENABLE_DOCS=false
REQUEST_TIMEOUT=30
MAX_RETRIES=3
'''
    print(env_files)

    # Pydantic Settings 配置类
    settings_code = '''
# ============ config.py（Pydantic Settings 配置管理）============
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """应用配置 — 类型安全的环境变量管理"""

    # 应用配置
    APP_NAME: str = "LangServe API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    ENABLE_DOCS: bool = True

    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # LLM 配置
    MODEL_PROVIDER: str = "zhipu"
    ZHIPU_API_KEY: str = ""
    ZHIPU_API_BASE: Optional[str] = None
    ZHIPU_MODEL_NAME: str = "glm-4.7-flash"
    TEMPERATURE: float = 0.7
    REQUEST_TIMEOUT: float = 60.0
    MAX_RETRIES: int = 3

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None

    # 缓存配置
    CACHE_ENABLED: bool = False
    CACHE_TTL: int = 3600  # 秒
    REDIS_URL: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# 全局配置实例
settings = Settings()

# 使用方式:
#   from config import settings
#   print(settings.ZHIPU_API_KEY)
#   print(settings.LOG_LEVEL)
'''
    print(settings_code)

    # 环境切换
    switch_code = '''
# ============ 环境切换 ============

# 方式1: 指定 .env 文件
#   cp .env.dev .env      # 切换到开发环境
#   cp .env.prod .env     # 切换到生产环境

# 方式2: 环境变量覆盖（优先级最高）
#   export MODEL_PROVIDER=zhipu
#   export LOG_LEVEL=DEBUG
#   uvicorn server:app --port 8000

# 方式3: 启动时指定
#   uvicorn server:app --port 8000 --env-file .env.prod

# 方式4: Docker 启动时指定
#   docker run --env-file .env.prod langserve-api
'''
    print(switch_code)

    # 本地模拟
    print("\n📋 模拟演示：当前环境配置")
    print("-" * 40)

    try:
        # 读取当前配置
        config_items = {
            "MODEL_PROVIDER": os.getenv("MODEL_PROVIDER", "zhipu (默认)"),
            "ZHIPU_MODEL_NAME": os.getenv("ZHIPU_MODEL_NAME", "glm-4.7-flash (默认)"),
            "TEMPERATURE": os.getenv("TEMPERATURE", "0.7 (默认)"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO (默认)"),
            "REQUEST_TIMEOUT": os.getenv("REQUEST_TIMEOUT", "60 (默认)"),
            "MAX_RETRIES": os.getenv("MAX_RETRIES", "3 (默认)"),
        }

        print("\n当前环境配置:")
        for key, value in config_items.items():
            if "KEY" in key and value and value != f"{key.split('_')[-1].lower()} (默认)":
                print(f"   {key}: {'*' * 8}{value[-4:] if len(value) > 4 else '****'}")
            else:
                print(f"   {key}: {value}")

        # 环境判断
        log_level = os.getenv("LOG_LEVEL", "INFO")
        if log_level == "DEBUG":
            env_type = "开发环境"
        elif log_level == "WARNING":
            env_type = "生产环境"
        else:
            env_type = "测试环境"

        print(f"\n🚦 当前环境推断: {env_type}")

        # 交互式模拟切换
        print("\n模拟环境切换:")
        print("  1. 开发环境 (DEBUG, TEMPERATURE=0.9)")
        print("  2. 测试环境 (INFO, TEMPERATURE=0.7)")
        print("  3. 生产环境 (WARNING, TEMPERATURE=0.5)")

        choice = input("\n选择环境查看配置差异: ").strip()
        if choice == "1":
            print("\n开发环境配置:")
            print("   LOG_LEVEL=DEBUG, TEMPERATURE=0.9, WORKERS=1, ENABLE_DOCS=true")
            print("   特点: 详细日志，高创造性，单 worker，API 文档可用")
        elif choice == "2":
            print("\n测试环境配置:")
            print("   LOG_LEVEL=INFO, TEMPERATURE=0.7, WORKERS=2, ENABLE_DOCS=true")
            print("   特点: 常规日志，标准创造性，2 worker，API 文档可用")
        elif choice == "3":
            print("\n生产环境配置:")
            print("   LOG_LEVEL=WARNING, TEMPERATURE=0.5, WORKERS=4, ENABLE_DOCS=false")
            print("   特点: 精简日志，低随机性，4 worker，关闭 API 文档")
        else:
            print("⚠️ 无效选择")

    except Exception as e:
        print(f"❌ 演示失败: {e}")

    print("\n💡 要点总结：")
    print("   1. .env 文件管理敏感配置，不提交到 Git")
    print("   2. Pydantic Settings 提供类型安全的配置管理")
    print("   3. 不同环境使用不同的 .env 文件")
    print("   4. 环境变量优先级: 系统环境 > .env 文件 > 默认值")


# ============================================================
# 示例3: 日志监控 - 生产监控
# ============================================================

def demo_logging_monitoring():
    """
    示例3：日志监控 - 添加日志和监控中间件

    演示如何为 LangServe 服务添加完善的日志记录和监控中间件，
    包括请求日志、性能指标、错误追踪等。
    """
    print("\n" + "="*60)
    print("示例3：日志监控 - 生产监控")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 日志级别: DEBUG < INFO < WARNING < ERROR < CRITICAL")
    print("   - 请求日志: 记录每次 API 调用的详情")
    print("   - 性能监控: 记录响应时间、成功率等指标")
    print("   - 中间件: 拦截请求/响应，自动添加日志")
    print()

    # 日志配置
    logging_code = '''
# ============ logging_config.py（日志配置）============
import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    配置生产级日志

    Args:
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_file: 日志文件路径（为 None 则只输出到控制台）
    """
    # 日志格式
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # 根日志配置
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt))
    root_logger.addHandler(console_handler)

    # 文件输出（可选）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(fmt, datefmt))
        root_logger.addHandler(file_handler)

    return root_logger

# 使用:
#   setup_logging(log_level="INFO", log_file="logs/app.log")
'''
    print(logging_code)

    # 监控中间件
    middleware_code = '''
# ============ middleware.py（监控中间件）============
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
import json

logger = logging.getLogger("langserve.monitor")

class MonitoringMiddleware(BaseHTTPMiddleware):
    """请求监控中间件 — 自动记录请求日志和性能指标"""

    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 请求信息
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        # 记录请求开始
        logger.info(f"→ {method} {path} | client={client_ip}")

        try:
            # 调用下一个中间件/路由
            response = await call_next(request)

            # 计算延迟
            latency_ms = (time.time() - start_time) * 1000
            self.request_count += 1
            self.total_latency += latency_ms

            # 记录响应
            logger.info(
                f"← {method} {path} | "
                f"status={response.status_code} | "
                f"latency={latency_ms:.1f}ms"
            )

            # 添加自定义响应头
            response.headers["X-Response-Time"] = f"{latency_ms:.1f}ms"
            response.headers["X-Request-Id"] = str(self.request_count)

            return response

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.error_count += 1
            self.request_count += 1

            logger.error(
                f"✗ {method} {path} | "
                f"error={str(e)} | "
                f"latency={latency_ms:.1f}ms"
            )
            raise

# 集成到 FastAPI
app = FastAPI(title="带监控的 LangServe 服务")
app.add_middleware(MonitoringMiddleware)

# 监控指标端点
@app.get("/metrics")
async def metrics():
    """返回监控指标"""
    middleware = None
    for m in app.user_middleware:
        if isinstance(m.cls, type) and issubclass(m.cls, MonitoringMiddleware):
            middleware = m
            break

    return {
        "request_count": "see logs",
        "error_count": "see logs",
        "timestamp": time.time(),
    }
'''
    print(middleware_code)

    # 完整服务端代码
    server_code = '''
# ============ server.py（带监控的生产服务）============
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from src.utils.llm_loader import get_default_llm
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("langserve")

app = FastAPI(title="带监控的 LangServe 服务")

# 添加监控中间件
app.add_middleware(MonitoringMiddleware)

llm = get_default_llm()
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手，请用中文回答。"),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()
add_routes(app, chain, path="/chat")

@app.get("/health")
async def health():
    logger.info("健康检查请求")
    return {"status": "healthy", "timestamp": time.time()}

# 启动: uvicorn server:app --port 8000
'''
    print(server_code)

    # 本地模拟
    print("\n📋 模拟演示：日志监控效果")
    print("-" * 40)

    try:
        # 配置本地日志
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True
        )
        logger = logging.getLogger("langserve.demo")

        # 模拟请求日志
        print("\n--- 模拟请求日志 ---")
        logger.info("→ POST /chat/invoke | client=127.0.0.1")
        logger.info("← POST /chat/invoke | status=200 | latency=1523.5ms")

        logger.info("→ POST /chat/batch | client=127.0.0.1")
        logger.info("← POST /chat/batch | status=200 | latency=3201.2ms")

        logger.warning("→ POST /chat/invoke | client=127.0.0.1 | slow_request")
        logger.info("← POST /chat/invoke | status=200 | latency=8500.1ms")

        logger.error("→ POST /chat/invoke | client=127.0.0.1 | timeout")
        logger.error("✗ POST /chat/invoke | error=RequestTimeout | latency=60000.0ms")

        # 实际调用演示
        print("\n--- 实际调用 + 日志 ---")
        llm = get_default_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是助手，请用中文简洁回答。"),
            ("human", "{question}")
        ])
        chain = prompt | llm | StrOutputParser()

        question = input("\n输入问题（回车默认）: ").strip()
        if not question:
            question = "什么是 LangServe？"

        logger.info(f"→ POST /chat/invoke | question={question[:30]}")
        start = time.time()
        result = chain.invoke({"question": question})
        latency = (time.time() - start) * 1000
        logger.info(f"← POST /chat/invoke | status=200 | latency={latency:.1f}ms")

        print(f"\n结果: {result}")
        print(f"延迟: {latency:.1f}ms")

    except Exception as e:
        print(f"❌ 演示失败: {e}")

    print("\n💡 要点总结：")
    print("   1. 日志是生产环境排查问题的首要工具")
    print("   2. 中间件自动记录所有请求的日志和性能指标")
    print("   3. RotatingFileHandler 防止日志文件无限增长")
    print("   4. /metrics 端点可集成 Prometheus 等监控")


# ============================================================
# 示例4: 错误处理 - 生产级错误处理
# ============================================================

def demo_error_handling():
    """
    示例4：错误处理 - 优雅降级、错误重试

    演示生产级的错误处理策略，包括：
    - 全局异常处理
    - 错误重试（指数退避）
    - 优雅降级
    - 超时控制
    """
    print("\n" + "="*60)
    print("示例4：错误处理 - 生产级错误处理")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 全局异常处理: 捕获所有未处理异常，避免服务崩溃")
    print("   - 错误重试: 临时性错误自动重试（指数退避）")
    print("   - 优雅降级: 主服务不可用时返回缓存/默认结果")
    print("   - 超时控制: 设置合理超时，避免请求无限等待")
    print()

    # 错误处理层次
    error_hierarchy = '''
┌─────────────────────────────────────────────────────────┐
│  错误处理层次                                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  第1层: 全局异常处理器                                    │
│    ↓ 捕获所有未处理异常，返回友好错误信息                   │
│                                                         │
│  第2层: 重试机制                                         │
│    ↓ 临时性错误（超时、限流）自动重试                       │
│                                                         │
│  第3层: 优雅降级                                         │
│    ↓ 主模型不可用时切换备用模型或返回缓存结果               │
│                                                         │
│  第4层: 超时控制                                         │
│    ↓ 设置合理超时，快速失败而非无限等待                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
'''
    print(error_hierarchy)

    # 完整的服务端错误处理代码
    server_code = '''
# ============ server.py（带错误处理的生产服务）============
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from src.utils.llm_loader import get_default_llm
import logging
import time
import hashlib

logger = logging.getLogger("langserve")

app = FastAPI(title="带错误处理的 LangServe 服务")

# ========== 全局异常处理器 ==========

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器 — 捕获所有未处理异常"""
    logger.error(f"未处理异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "服务暂时不可用，请稍后重试",
            "detail": str(exc) if os.getenv("DEBUG") == "true" else None,
        }
    )

@app.exception_handler(TimeoutError)
async def timeout_handler(request: Request, exc: TimeoutError):
    """超时异常处理器"""
    logger.warning(f"请求超时: {request.url}")
    return JSONResponse(
        status_code=504,
        content={
            "error": "timeout",
            "message": "请求超时，请稍后重试",
        }
    )

# ========== 缓存 + 优雅降级 ==========

cache = {}

def get_cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

async def call_with_fallback(question: str, max_retries: int = 3):
    """带重试和降级的 LLM 调用"""

    # 检查缓存
    cache_key = get_cache_key(question)
    if cache_key in cache:
        logger.info(f"缓存命中: {question[:20]}...")
        return cache[cache_key]

    # 重试逻辑
    for attempt in range(max_retries):
        try:
            llm = get_default_llm()
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是助手，请用中文回答。"),
                ("human", "{question}")
            ])
            chain = prompt | llm | StrOutputParser()

            result = await asyncio.wait_for(
                chain.ainvoke({"question": question}),
                timeout=30.0  # 30秒超时
            )

            # 缓存结果
            cache[cache_key] = result
            return result

        except TimeoutError:
            logger.warning(f"第{attempt+1}次尝试超时: {question[:20]}...")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                await asyncio.sleep(wait_time)
            continue

        except Exception as e:
            logger.error(f"第{attempt+1}次调用失败: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            continue

    # 所有重试失败 — 优雅降级
    logger.warning(f"所有重试失败，返回降级响应: {question[:20]}...")
    fallback = "抱歉，服务暂时繁忙，请稍后再试。"

    # 如果有缓存，优先返回缓存
    if cache_key in cache:
        fallback = f"[缓存结果] {cache[cache_key]}"

    return fallback

# ========== 带错误处理的自定义端点 ==========

@app.post("/ask")
async def safe_ask(request: Request):
    """带完整错误处理的问答接口"""
    try:
        body = await request.json()
        question = body.get("question", "")

        if not question:
            return JSONResponse(
                status_code=400,
                content={"error": "bad_request", "message": "question 不能为空"}
            )

        result = await call_with_fallback(question)
        return {"question": question, "answer": result}

    except Exception as e:
        logger.error(f"safe_ask 错误: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "服务异常"}
        )

# LangServe 标准端点
llm = get_default_llm()
chain = ChatPromptTemplate.from_messages([
    ("system", "你是助手，请用中文回答。"),
    ("human", "{question}")
]) | llm | StrOutputParser()
add_routes(app, chain, path="/chat")

# 启动: uvicorn server:app --port 8000
'''
    print(server_code)

    # 本地模拟
    print("\n📋 模拟演示：错误处理流程")
    print("-" * 40)

    try:
        llm = get_default_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是助手，请用中文简洁回答。"),
            ("human", "{question}")
        ])
        chain = prompt | llm | StrOutputParser()

        # 模拟正常调用
        print("\n--- 模拟1: 正常调用 ---")
        question = input("输入问题（回车默认）: ").strip()
        if not question:
            question = "什么是 LangServe？"

        try:
            start = time.time()
            result = chain.invoke({"question": question})
            latency = (time.time() - start) * 1000
            print(f"✅ 调用成功 | latency={latency:.1f}ms")
            print(f"   结果: {result}")
        except Exception as e:
            print(f"❌ 调用失败: {e}")

        # 模拟错误场景
        print("\n--- 模拟2: 错误处理流程 ---")
        scenarios = {
            "1": ("超时重试", "请求超过30秒 → 等待2^n秒 → 重试 → 成功/降级"),
            "2": ("API 限流", "429 Too Many Requests → 指数退避 → 重试"),
            "3": ("模型不可用", "连接失败 → 重试3次 → 降级返回缓存"),
            "4": ("输入错误", "空字符串 → 400 Bad Request → 友好提示"),
        }

        for key, (name, desc) in scenarios.items():
            print(f"  {key}. {name}: {desc}")

        choice = input("\n选择场景 (1-4): ").strip()
        if choice == "1":
            print("\n📋 超时重试流程:")
            print("   第1次尝试: 超时 → 等待1秒")
            print("   第2次尝试: 超时 → 等待2秒")
            print("   第3次尝试: 成功 → 返回结果")
            print("   ⚠️ 指数退避避免雪崩：1s → 2s → 4s")
        elif choice == "2":
            print("\n📋 API 限流流程:")
            print("   第1次调用: 429 Rate Limited → 等待1秒")
            print("   第2次调用: 429 Rate Limited → 等待2秒")
            print("   第3次调用: 200 OK → 返回结果")
        elif choice == "3":
            print("\n📋 模型不可用降级:")
            print("   第1次尝试: ConnectionError → 等待1秒")
            print("   第2次尝试: ConnectionError → 等待2秒")
            print("   第3次尝试: ConnectionError → 等待4秒")
            print("   降级: 返回缓存结果或默认提示")
        elif choice == "4":
            print("\n📋 输入验证:")
            print('   请求: {"question": ""}')
            print('   响应: 400 {"error": "bad_request", "message": "question 不能为空"}')

        # 模拟实际重试
        print("\n--- 模拟3: 实际调用（带计时）---")
        question2 = input("再输入一个问题（回车默认）: ").strip()
        if not question2:
            question2 = "Python 的优势是什么？"

        retry_count = 0
        max_retries = 2
        while retry_count <= max_retries:
            try:
                start = time.time()
                result = chain.invoke({"question": question2})
                latency = (time.time() - start) * 1000
                print(f"✅ 第{retry_count+1}次调用成功 | latency={latency:.1f}ms")
                print(f"   结果: {result}")
                break
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    wait = 2 ** (retry_count - 1)
                    print(f"❌ 第{retry_count}次调用失败: {e}")
                    print(f"   等待 {wait}s 后重试...")
                    time.sleep(wait)
                else:
                    print(f"❌ 所有重试失败，降级处理")
                    print(f'   降级响应: "抱歉，服务暂时繁忙，请稍后再试。"')

    except Exception as e:
        print(f"❌ 演示失败: {e}")

    print("\n💡 要点总结：")
    print("   1. 全局异常处理器是最后一道防线")
    print("   2. 重试使用指数退避，避免加剧服务压力")
    print("   3. 优雅降级确保用户始终能收到响应")
    print("   4. 超时控制防止请求无限等待")


# ============================================================
# 交互式主菜单
# ============================================================

def main():
    """交互式主菜单"""
    print("\n" + "="*60)
    print("  LangServe 生产部署 - 交互式案例")
    print("="*60)
    print("\n📚 核心概念回顾：")
    print("   - Docker 部署: 容器化，环境一致")
    print("   - 环境配置: .env 文件管理多环境")
    print("   - 日志监控: 中间件 + 日志记录")
    print("   - 错误处理: 重试 + 降级 + 超时")

    demos = {
        "1": ("Docker部署 - 容器化", demo_docker_deployment),
        "2": ("环境配置 - 多环境管理", demo_environment_config),
        "3": ("日志监控 - 生产监控", demo_logging_monitoring),
        "4": ("错误处理 - 生产级错误处理", demo_error_handling),
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
