# LangChain 实战学习项目

> **原创交互式 LangChain 学习项目** - 39个实战案例，158个交互式示例，100%原创

[![English](https://img.shields.io/badge/English-README-blue)](README_EN.md)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目简介

这是一个**原创、交互式、实战导向**的 LangChain 学习项目。所有案例都是可交互的，用户可以自由输入，而非被动观看固定输出。

## 学习目录

| 章节 | 文档 | 案例文件 | 示例数 |
|------|------|---------|-------|
| 第一章：基础篇 | [docs/01_基础篇.md](docs/01_基础篇.md) | `basic_chain.py`、`model_parameters_chain.py`、`message_types_demo.py` | 13 |
| 第二章：提示词篇 | [docs/02_提示词篇.md](docs/02_提示词篇.md) | `prompt_basics.py` | 4 |
| 第三章：输出篇 | [docs/03_输出篇.md](docs/03_输出篇.md) | `output_strategies.py`、`structured_output_fixed.py`、`streaming_demo.py` | 12 |
| 第四章：工具篇 | [docs/04_工具篇.md](docs/04_工具篇.md) | `tool_basics.py`、`tool_advanced.py`、`tool_injection.py` | 11 |
| 第五章：Agent篇 | [docs/05_Agent篇.md](docs/05_Agent篇.md) | `agent_basics.py`、`agent_workflow.py`、`multi_agent.py`、`human_in_loop.py` | 16 |
| 第六章：中间件篇 | [docs/06_中间件篇.md](docs/06_中间件篇.md) | `middleware_basics.py`、`error_handling.py` | 8 |
| 第七章：记忆与存储篇 | [docs/07_记忆与存储篇.md](docs/07_记忆与存储篇.md) | `checkpointer.py`、`store_basics.py` | 8 |
| 第八章：RAG篇 | [docs/08_RAG篇.md](docs/08_RAG篇.md) | `rag_basics.py`、`document_loader.py`、`rag_agent.py`、`langsmith_demo.py` | 16 |
| 第九章：LangGraph基础篇 | [docs/09_LangGraph基础篇.md](docs/09_LangGraph基础篇.md) | `langgraph_basics.py`、`langgraph_state.py`、`langgraph_control_flow.py` | 12 |
| 第十章：LangGraph高级篇 | [docs/10_LangGraph高级篇.md](docs/10_LangGraph高级篇.md) | `langgraph_agent.py`、`langgraph_human_in_loop.py`、`langgraph_multi_agent.py` | 12 |
| 第十一章：LangServe部署篇 | [docs/11_LangServe部署篇.md](docs/11_LangServe部署篇.md) | `langserve_basics.py`、`langserve_advanced.py`、`langserve_production.py` | 12 |
| 第十二章：LangSmith调试篇 | [docs/12_LangSmith调试篇.md](docs/12_LangSmith调试篇.md) | `langsmith_tracing.py`、`langsmith_debugging.py` | 8 |
| 第十三章：LangSmith评估篇 | [docs/13_LangSmith评估篇.md](docs/13_LangSmith评估篇.md) | `langsmith_evaluation.py`、`langsmith_prompt_management.py`、`langsmith_monitoring.py` | 12 |
| 第十四章：实战项目篇 | [docs/14_实战项目篇.md](docs/14_实战项目篇.md) | `project_learning_assistant.py`、`project_data_analyst.py`、`project_code_assistant.py` | 15 |

> 完整索引：[EXAMPLES_INDEX.md](EXAMPLES_INDEX.md)

## 项目结构

```
langchain/
├── .env.example                    # 环境变量模板
├── .env                            # 环境变量（不提交）
├── requirements.txt                # 依赖包
├── README.md                       # 项目说明
├── EXAMPLES_INDEX.md               # 案例索引
│
├── docs/                           # 章节文档
│   ├── 01_基础篇.md
│   ├── 02_提示词篇.md
│   ├── 03_输出篇.md
│   ├── 04_工具篇.md
│   ├── 05_Agent篇.md
│   ├── 06_中间件篇.md
│   ├── 07_记忆与存储篇.md
│   ├── 08_RAG篇.md
│   └── MODEL_CONFIG.md
│
└── src/
    ├── main.py                     # 程序入口
    ├── chains/                     # 39个实战案例
    │   ├── basic_chain.py          # 1.1 LCEL链式调用
    │   ├── model_parameters_chain.py # 1.2 模型参数详解
    │   ├── message_types_demo.py   # 1.3 消息类型
    │   ├── prompt_basics.py        # 2.1 提示词模板
    │   ├── output_strategies.py    # 3.1 输出策略
    │   ├── structured_output_fixed.py # 3.2 结构化输出
    │   ├── streaming_demo.py       # 3.3 流式输出
    │   ├── tool_basics.py          # 4.1 工具基础
    │   ├── tool_advanced.py        # 4.2 工具高级特性
    │   ├── tool_injection.py       # 4.3 工具注入
    │   ├── agent_basics.py         # 5.1 Agent基础
    │   ├── agent_workflow.py       # 5.2 Agent工作流
    │   ├── multi_agent.py          # 5.3 多Agent协作
    │   ├── human_in_loop.py        # 5.4 人工介入
    │   ├── middleware_basics.py    # 6.1 中间件基础
    │   ├── error_handling.py       # 6.2 错误处理
    │   ├── checkpointer.py         # 7.1 对话记忆
    │   ├── store_basics.py         # 7.2 跨会话存储
    │   ├── rag_basics.py           # 8.1 RAG基础
    │   ├── document_loader.py      # 8.2 文档加载与切分
    │   ├── rag_agent.py            # 8.3 RAG Agent
    │   ├── langsmith_demo.py       # 8.4 可观测性
    │   ├── langgraph_basics.py     # 9.1 LangGraph基础
    │   ├── langgraph_state.py      # 9.2 状态管理
    │   ├── langgraph_control_flow.py # 9.3 控制流
    │   ├── langgraph_agent.py      # 10.1 LangGraph Agent
    │   ├── langgraph_human_in_loop.py # 10.2 人工介入
    │   ├── langgraph_multi_agent.py # 10.3 多Agent协作
    │   ├── langserve_basics.py     # 11.1 LangServe基础
    │   ├── langserve_advanced.py   # 11.2 高级特性
    │   ├── langserve_production.py # 11.3 生产部署
    │   ├── langsmith_tracing.py    # 12.1 链路追踪
    │   ├── langsmith_debugging.py  # 12.2 调试
    │   ├── langsmith_evaluation.py # 13.1 评估
    │   ├── langsmith_prompt_management.py # 13.2 Prompt管理
    │   ├── langsmith_monitoring.py # 13.3 监控
    │   ├── project_learning_assistant.py # 14.1 AI学习助手
    │   ├── project_data_analyst.py # 14.2 AI数据分析师
    │   └── project_code_assistant.py # 14.3 AI代码助手
    ├── prompts/
    │   └── templates.py
    └── utils/
        └── llm_loader.py           # LLM加载器（支持5种模型）
```

## 快速开始

```bash
# 1. 安装依赖
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置环境（推荐智谱AI - 有免费模型可用）
cp .env.example .env
# 编辑 .env，填入 ZHIPU_API_KEY

# 3. 运行案例
python src/chains/basic_chain.py
```

## 学习路线

**入门（1-2周）：** 基础篇 → 提示词篇 → 输出篇

**进阶（3-4周）：** 工具篇 → Agent篇

**高级（5-6周）：** 中间件篇 → 记忆与存储篇 → RAG篇

**专家（7-10周）：** LangGraph基础 → LangGraph高级 → LangServe部署 → LangSmith

**实战（11-12周）：** AI学习助手 → AI数据分析师 → AI代码助手

## 📚 核心示例（按优先级排序）

### ⭐ LangGraph 工作流（第9-10章）
- langgraph_basics.py - StateGraph、State、Node、Edge
- langgraph_state.py - 状态管理、Reducer、Annotated
- langgraph_control_flow.py - 条件边、循环、Send、子图
- langgraph_agent.py - ReAct Agent、工具调用、循环推理
- langgraph_human_in_loop.py - interrupt()、人工审批、断点恢复
- langgraph_multi_agent.py - 多Agent协作、主管Agent、流水线

### ⭐ LangServe 部署（第11章）
- langserve_basics.py - FastAPI集成、add_routes、invoke/batch/stream
- langserve_advanced.py - RemoteRunnable、流式传输、异步调用
- langserve_production.py - Docker部署、环境配置、日志监控

### ⭐ LangSmith 可观测性（第12-13章）
- langsmith_tracing.py - @traceable、链路追踪、错误追踪
- langsmith_debugging.py - 运行回放、中间变量、对比实验
- langsmith_evaluation.py - 数据集管理、评估器、LLM自评
- langsmith_prompt_management.py - Prompt版本管理、A/B测试
- langsmith_monitoring.py - 生产监控、告警配置、成本追踪

### ⭐ 实战项目（第14章）
- project_learning_assistant.py - AI学习助手（RAG+Agent+Memory+结构化输出）
- project_data_analyst.py - AI数据分析师（Chain+Agent+Tool+结构化输出）
- project_code_assistant.py - AI代码助手（Agent+Tool+结构化输出+RAG）

## 项目统计

- **总计 39 个完整示例**
- **涵盖 158 个具体功能演示**
- **覆盖 LangChain + LangGraph + LangServe + LangSmith 全流程**
- **支持 5 种主流模型提供商**
- **100% 中文注释和文档**

## 模型配置

| 提供商 | 费用 | 配置方式 |
|--------|------|---------|
| 智谱AI | 有免费模型 | `MODEL_PROVIDER=zhipu` |
| OpenAI | 付费 | `MODEL_PROVIDER=openai` |
| DeepSeek | 部分免费 | `MODEL_PROVIDER=deepseek` |
| 通义千问 | 部分免费 | `MODEL_PROVIDER=qwen` |
| Ollama | 完全免费 | `MODEL_PROVIDER=ollama` |

详细配置：[docs/MODEL_CONFIG.md](docs/MODEL_CONFIG.md)

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件