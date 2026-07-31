# LangChain Hands-on Learning Project

> **Original Interactive LangChain Learning Project** - 36 hands-on cases, 143 interactive examples, 100% original

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)](https://langchain-ai.github.io/langgraph/)
[![LangServe](https://img.shields.io/badge/LangServe-0.3%2B-purple)](https://python.langchain.com/docs/langserve)
[![LangSmith](https://img.shields.io/badge/LangSmith-SDK-red)](https://docs.smith.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Project Overview

This is an **original, interactive, practice-oriented** learning project covering the full LangChain ecosystem: **LangChain + LangGraph + LangServe + LangSmith**. All examples are interactive — users can freely input their own data rather than passively watching fixed outputs.

## Features

- **36 Cases** — Comprehensive coverage from basics to production deployment
- **143 Examples** — Each case contains multiple interactive sub-examples
- **100% Chinese** — All comments, documentation, and prompts in Chinese
- **Interactive** — Every example supports user input and real-time feedback
- **5 Model Providers** — ZhipuAI (free models available), OpenAI, DeepSeek, Qwen, Ollama
- **14 Chapters** — Systematic learning path from beginner to expert

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd langchain

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (recommended: ZhipuAI - free models available)
cp .env.example .env
# Edit .env and fill in ZHIPU_API_KEY

# 5. Run your first example
python src/chains/basic_chain.py
```

> For detailed setup instructions, see [QUICK_START_EN.md](QUICK_START_EN.md)

## Project Structure

```
langchain/
├── .env.example                    # Environment variable template
├── .env                            # Environment variables (not committed)
├── requirements.txt                # Dependencies
├── README.md                       # Chinese README
├── README_EN.md                    # English README
├── QUICK_START_EN.md               # English Quick Start Guide
├── EXAMPLES_INDEX.md               # Chinese Examples Index
├── EXAMPLES_INDEX_EN.md            # English Examples Index
│
├── docs/                           # Chapter documentation (Chinese)
│   ├── 01_基础篇.md
│   ├── 02_提示词篇.md
│   ├── 03_输出篇.md
│   ├── 04_工具篇.md
│   ├── 05_Agent篇.md
│   ├── 06_中间件篇.md
│   ├── 07_记忆与存储篇.md
│   ├── 08_RAG篇.md
│   ├── 09_LangGraph基础篇.md
│   ├── 10_LangGraph高级篇.md
│   ├── 11_LangServe部署篇.md
│   ├── 12_LangSmith调试篇.md
│   ├── 13_LangSmith评估篇.md
│   └── MODEL_CONFIG.md
│
└── src/
    ├── main.py                     # Program entry point
    ├── chains/                     # 36 hands-on examples
    │   ├── basic_chain.py          # 1.1 LCEL Chain
    │   ├── model_parameters_chain.py # 1.2 Model Parameters
    │   ├── message_types_demo.py   # 1.3 Message Types
    │   ├── prompt_basics.py        # 2.1 Prompt Templates
    │   ├── output_strategies.py    # 3.1 Output Strategies
    │   ├── structured_output_fixed.py # 3.2 Structured Output
    │   ├── streaming_demo.py       # 3.3 Streaming Output
    │   ├── tool_basics.py          # 4.1 Tool Basics
    │   ├── tool_advanced.py        # 4.2 Advanced Tools
    │   ├── tool_injection.py       # 4.3 Tool Injection
    │   ├── agent_basics.py         # 5.1 Agent Basics
    │   ├── agent_workflow.py       # 5.2 Agent Workflow
    │   ├── multi_agent.py          # 5.3 Multi-Agent
    │   ├── human_in_loop.py        # 5.4 Human-in-the-Loop
    │   ├── middleware_basics.py    # 6.1 Middleware Basics
    │   ├── error_handling.py       # 6.2 Error Handling
    │   ├── checkpointer.py         # 7.1 Conversation Memory
    │   ├── store_basics.py         # 7.2 Cross-session Store
    │   ├── rag_basics.py           # 8.1 RAG Basics
    │   ├── document_loader.py      # 8.2 Document Loader
    │   ├── rag_agent.py            # 8.3 RAG Agent
    │   ├── langsmith_demo.py       # 8.4 Observability
    │   ├── langgraph_basics.py     # 9.1 StateGraph Basics
    │   ├── langgraph_state.py      # 9.2 State Management
    │   ├── langgraph_control_flow.py # 9.3 Control Flow
    │   ├── langgraph_agent.py      # 10.1 ReAct Agent
    │   ├── langgraph_human_in_loop.py # 10.2 Human-in-the-Loop
    │   ├── langgraph_multi_agent.py # 10.3 Multi-Agent
    │   ├── langserve_basics.py     # 11.1 LangServe Basics
    │   ├── langserve_advanced.py   # 11.2 Advanced LangServe
    │   ├── langserve_production.py # 11.3 Production Deployment
    │   ├── langsmith_tracing.py    # 12.1 Tracing
    │   ├── langsmith_debugging.py  # 12.2 Debugging
    │   ├── langsmith_evaluation.py # 12.3 Evaluation
    │   ├── langsmith_prompt_management.py # 13.1 Prompt Management
    │   ├── langsmith_monitoring.py # 13.2 Monitoring
    │   ├── project_learning_assistant.py # 14.1 Learning Assistant
    │   ├── project_data_analyst.py # 14.2 Data Analyst
    │   └── project_code_assistant.py # 14.3 Code Assistant
    ├── prompts/
    │   └── templates.py
    └── utils/
        └── llm_loader.py           # LLM Loader (supports 5 providers)
```

## Learning Path

| Stage | Chapters | Duration | Topics |
|-------|----------|----------|--------|
| **Beginner** | Ch.1-3 | 1-2 weeks | Basics, Prompts, Output |
| **Intermediate** | Ch.4-5 | 2-3 weeks | Tools, Agents |
| **Advanced** | Ch.6-8 | 3-4 weeks | Middleware, Memory, RAG |
| **Expert** | Ch.9-14 | 5-6 weeks | LangGraph, LangServe, LangSmith, Projects |

## Core Examples by Chapter

### Chapter 1: Basics
- `basic_chain.py` — LCEL chain, pipe operator, parallel chains, streaming
- `model_parameters_chain.py` — Temperature, MaxTokens, TopP, penalties
- `message_types_demo.py` — SystemMessage, HumanMessage, AIMessage, ToolMessage

### Chapter 2: Prompts
- `prompt_basics.py` — SystemPrompt, DynamicPrompt, PromptTemplate, MessagesPlaceholder

### Chapter 3: Output
- `output_strategies.py` — Str/Json/CSV output parsers, custom parser
- `structured_output_fixed.py` — Pydantic models, structured extraction, batch extraction
- `streaming_demo.py` — Streaming output, stream() vs invoke(), real-time display

### Chapter 4: Tools
- `tool_basics.py` — @tool decorator, agent tool usage, multi-tool collaboration
- `tool_advanced.py` — StructuredTool, input validation, field_validator, tool chains
- `tool_injection.py` — InjectedState, InjectedStore, context-aware tools

### Chapter 5: Agents
- `agent_basics.py` — create_tool_calling_agent, AgentExecutor, agent with memory
- `agent_workflow.py` — Sequential/conditional/loop/parallel workflows
- `multi_agent.py` — Role delegation, pipeline agents, debate agents, supervisor
- `human_in_loop.py` — Approval flows, content moderation, collaborative editing

### Chapter 6: Middleware
- `middleware_basics.py` — Callback handlers, logging, token tracking, performance monitoring
- `error_handling.py` — Retry with backoff, model fallbacks, output validation, graceful degradation

### Chapter 7: Memory & Storage
- `checkpointer.py` — Conversation buffer/summary/window, session management, persistence
- `store_basics.py` — Key-value store, user preferences, shared context, persistent storage

### Chapter 8: RAG
- `rag_basics.py` — Simple RAG, document Q&A, similarity search, sourced RAG
- `document_loader.py` — Text splitting, document loading, chunk management
- `rag_agent.py` — RAG Agent, multi-source RAG, conversational RAG, RAG + tools
- `langsmith_demo.py` — Tracing basics, evaluation system, dataset management

### Chapter 9: LangGraph Basics
- `langgraph_basics.py` — StateGraph, State, Node, Edge, graph construction
- `langgraph_state.py` — State management, Reducer, Annotated, state aggregation
- `langgraph_control_flow.py` — Conditional edges, loops, Send, subgraphs

### Chapter 10: LangGraph Advanced
- `langgraph_agent.py` — ReAct Agent, tool calling, cyclic reasoning
- `langgraph_human_in_loop.py` — interrupt(), human approval, breakpoint recovery
- `langgraph_multi_agent.py` — Multi-agent collaboration, supervisor, pipeline

### Chapter 11: LangServe
- `langserve_basics.py` — FastAPI integration, add_routes, invoke/batch/stream
- `langserve_advanced.py` — RemoteRunnable, streaming, async invocation
- `langserve_production.py` — Docker deployment, env config, logging & monitoring

### Chapter 12: LangSmith Debugging
- `langsmith_tracing.py` — @traceable, trace analysis, error tracking
- `langsmith_debugging.py` — Run replay, intermediate variables, comparison experiments
- `langsmith_evaluation.py` — Dataset management, evaluators, LLM self-evaluation

### Chapter 13: LangSmith Evaluation
- `langsmith_prompt_management.py` — Prompt versioning, A/B testing, template management
- `langsmith_monitoring.py` — Production monitoring, alerting, cost tracking

### Chapter 14: Real-world Projects
- `project_learning_assistant.py` — Interactive learning assistant with RAG and adaptive feedback
- `project_data_analyst.py` — Data analysis agent with visualization and reporting
- `project_code_assistant.py` — Code assistant with generation, review, and debugging

## Model Configuration

| Provider | Cost | Config |
|----------|------|--------|
| ZhipuAI | Free | `MODEL_PROVIDER=zhipu` |
| OpenAI | Paid | `MODEL_PROVIDER=openai` |
| DeepSeek | Partially Free | `MODEL_PROVIDER=deepseek` |
| Qwen (Tongyi) | Partially Free | `MODEL_PROVIDER=qwen` |
| Ollama (Local) | Free | `MODEL_PROVIDER=ollama` |

> Detailed configuration: [docs/MODEL_CONFIG.md](docs/MODEL_CONFIG.md)

## Project Statistics

- **36** complete examples
- **143** interactive feature demonstrations
- **5** model providers supported
- **100%** Chinese comments and documentation
- **14** chapters covering the full LangChain ecosystem

## License

MIT License — see [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or contributions, please open an issue or pull request.

---

> For the Chinese version of this document, see [README.md](README.md)
>
> For the complete examples index, see [EXAMPLES_INDEX_EN.md](EXAMPLES_INDEX_EN.md)
>
> For the quick start guide, see [QUICK_START_EN.md](QUICK_START_EN.md)
