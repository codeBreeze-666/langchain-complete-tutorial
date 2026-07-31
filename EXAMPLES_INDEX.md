# LangChain 实战案例索引

> 36个实战案例，143个交互式示例，100%原创

---

## 📚 学习目录

### [第一章：基础篇](docs/01_基础篇.md)
| 序号 | 案例文件 | 知识点 | 示例数 | 难度 |
|------|---------|--------|-------|------|
| 1.1 | `basic_chain.py` | LCEL链式调用、管道操作符、并行链、流式输出 | 4 | ⭐ |
| 1.2 | `model_parameters_chain.py` | Temperature、MaxTokens、TopP、FrequencyPenalty、PresencePenalty | 5 | ⭐ |
| 1.3 | `message_types_demo.py` | SystemMessage、HumanMessage、AIMessage、ToolMessage | 4 | ⭐⭐ |

### [第二章：提示词篇](docs/02_提示词篇.md)
| 序号 | 案例文件 | 知识点 | 示例数 | 难度 |
|------|---------|--------|-------|------|
| 2.1 | `prompt_basics.py` | SystemPrompt、DynamicPrompt、PromptTemplate、MessagesPlaceholder | 4 | ⭐⭐ |

### [第三章：输出篇](docs/03_输出篇.md)
| 序号 | 案例文件 | 知识点 | 示例数 | 难度 |
|------|---------|--------|-------|------|
| 3.1 | `output_strategies.py` | StrOutputParser、JsonOutputParser、CommaSeparatedListOutputParser、自定义解析器 | 4 | ⭐⭐ |
| 3.2 | `structured_output_fixed.py` | Pydantic模型、PydanticOutputParser、信息提取、批量提取 | 4 | ⭐⭐⭐ |
| 3.3 | `streaming_demo.py` | 流式输出、stream()vs invoke()、进度显示、实时应用 | 4 | ⭐⭐ |

### [第四章：工具篇](docs/04_工具篇.md)
| 序号 | 案例文件 | 知识点 | 示例数 | 难度 |
|------|---------|--------|-------|------|
| 4.1 | `tool_basics.py` | @tool装饰器、Agent使用工具、多工具协作 | 3 | ⭐⭐ |
| 4.2 | `tool_advanced.py` | StructuredTool、输入验证、field_validator、错误处理、工具链 | 4 | ⭐⭐⭐ |
| 4.3 | `tool_injection.py` | InjectedState、InjectedStore、配置注入、上下文感知工具 | 4 | ⭐⭐⭐⭐ |

### [第五章：Agent篇](docs/05_Agent篇.md)
| 序号 | 案例文件 | 知识点 | 示例数 | 难度 |
|------|---------|--------|-------|------|
| 5.1 | `agent_basics.py` | create_tool_calling_agent、AgentExecutor、带记忆Agent、Agent调试 | 4 | ⭐⭐⭐ |
| 5.2 | `agent_workflow.py` | 顺序工作流、条件工作流(RunnableBranch)、循环工作流、并行工作流 | 4 | ⭐⭐⭐ |
| 5.3 | `multi_agent.py` | 角色委派、流水线Agent、辩论Agent、主管Agent | 4 | ⭐⭐⭐⭐ |
| 5.4 | `human_in_loop.py` | 审批流程、内容审核、决策检查、协作编辑 | 4 | ⭐⭐⭐ |

### [第六章：中间件篇](docs/06_中间件篇.md)
| 序号 | 案例文件 | 知识点 | 示例数 | 难度 |
|------|---------|--------|-------|------|
| 6.1 | `middleware_basics.py` | BaseCallbackHandler、日志中间件、Token追踪、性能监控、自定义回调 | 4 | ⭐⭐⭐ |
| 6.2 | `error_handling.py` | 错误重试(指数退避)、模型降级(with_fallbacks)、输出验证、优雅降级 | 4 | ⭐⭐⭐ |

### [第七章：记忆与存储篇](docs/07_记忆与存储篇.md)
| 序号 | 案例文件 | 知识点 | 示例数 | 难度 |
|------|---------|--------|-------|------|
| 7.1 | `checkpointer.py` | 对话缓冲、对话摘要、对话窗口、会话管理、持久化 | 4 | ⭐⭐⭐ |
| 7.2 | `store_basics.py` | 键值存储、用户偏好、共享上下文、持久化存储 | 4 | ⭐⭐⭐ |

### [第八章：RAG篇](docs/08_RAG篇.md)
| 序号 | 案例文件 | 知识点 | 示例数 | 难度 |
|------|---------|--------|-------|------|
| 8.1 | `rag_basics.py` | 简单RAG、文档问答、相似度搜索、带来源RAG | 4 | ⭐⭐⭐ |
| 8.2 | `document_loader.py` | 文本切分、文档加载、切片管理、自定义切分器 | 4 | ⭐⭐⭐ |
| 8.3 | `rag_agent.py` | RAG Agent、多源RAG、对话式RAG、RAG+工具 | 4 | ⭐⭐⭐⭐ |
| 8.4 | `langsmith_demo.py` | 追踪基础、评估系统、数据集管理、自定义评估器 | 4 | ⭐⭐⭐ |

### [第九章：LangGraph 基础](docs/09_LangGraph基础篇.md)
| 序号 | 案例文件 | 知识点 | 示例数 | 难度 |
|------|---------|--------|-------|------|
| 9.1 | `langgraph_basics.py` | StateGraph、State、Node、Edge、图构建与执行 | 4 | ⭐⭐⭐ |
| 9.2 | `langgraph_state.py` | 状态管理、Reducer、Annotated、状态聚合与更新 | 4 | ⭐⭐⭐ |
| 9.3 | `langgraph_control_flow.py` | 条件边、循环、Send、子图、复杂控制流 | 4 | ⭐⭐⭐⭐ |

**运行方式：**
```bash
python src/chains/langgraph_basics.py
python src/chains/langgraph_state.py
python src/chains/langgraph_control_flow.py
```

**学习要点：**
- 理解 StateGraph 的核心概念：State、Node、Edge
- 掌握状态管理机制：Reducer、Annotated 类型注解
- 学会构建复杂控制流：条件边、循环、Send 操作、子图嵌套

**难度等级：** ⭐⭐⭐ - ⭐⭐⭐⭐

### [第十章：LangGraph Agent](docs/10_LangGraph_Agent篇.md)
| 序号 | 案例文件 | 知识点 | 示例数 | 难度 |
|------|---------|--------|-------|------|
| 10.1 | `langgraph_agent.py` | ReAct Agent、工具调用、循环推理、状态驱动决策 | 4 | ⭐⭐⭐⭐ |
| 10.2 | `langgraph_human_in_loop.py` | interrupt()、人工审批、断点恢复、交互式决策 | 4 | ⭐⭐⭐⭐ |
| 10.3 | `langgraph_multi_agent.py` | 多Agent协作、主管Agent、流水线、Agent间通信 | 4 | ⭐⭐⭐⭐⭐ |

**运行方式：**
```bash
python src/chains/langgraph_agent.py
python src/chains/langgraph_human_in_loop.py
python src/chains/langgraph_multi_agent.py
```

**学习要点：**
- 掌握 ReAct Agent 模式：推理-行动-观察循环
- 理解人工介入机制：interrupt()、断点恢复
- 学会多 Agent 协作架构：主管模式、流水线模式

**难度等级：** ⭐⭐⭐⭐ - ⭐⭐⭐⭐⭐

### [第十一章：LangServe 部署](docs/11_LangServe部署篇.md)
| 序号 | 案例文件 | 知识点 | 示例数 | 难度 |
|------|---------|--------|-------|------|
| 11.1 | `langserve_basics.py` | FastAPI集成、add_routes、invoke/batch/stream 接口 | 4 | ⭐⭐⭐ |
| 11.2 | `langserve_advanced.py` | RemoteRunnable、流式传输、异步调用、远程调用 | 4 | ⭐⭐⭐⭐ |
| 11.3 | `langserve_production.py` | Docker部署、环境配置、日志监控、生产最佳实践 | 4 | ⭐⭐⭐⭐ |

**运行方式：**
```bash
python src/chains/langserve_basics.py
python src/chains/langserve_advanced.py
python src/chains/langserve_production.py
```

**学习要点：**
- 掌握 LangServe 与 FastAPI 的集成方式
- 理解远程调用机制：RemoteRunnable、流式传输
- 学会生产部署：Docker化、环境配置、日志与监控

**难度等级：** ⭐⭐⭐ - ⭐⭐⭐⭐

### [第十二章：LangSmith 追踪与调试](docs/12_LangSmith追踪调试篇.md)
| 序号 | 案例文件 | 知识点 | 示例数 | 难度 |
|------|---------|--------|-------|------|
| 12.1 | `langsmith_tracing.py` | @traceable、链路追踪、错误追踪、Trace分析 | 4 | ⭐⭐⭐ |
| 12.2 | `langsmith_debugging.py` | 运行回放、中间变量、对比实验、调试技巧 | 4 | ⭐⭐⭐ |
| 12.3 | `langsmith_evaluation.py` | 数据集管理、评估器、LLM自评、评估流水线 | 4 | ⭐⭐⭐⭐ |

**运行方式：**
```bash
python src/chains/langsmith_tracing.py
python src/chains/langsmith_debugging.py
python src/chains/langsmith_evaluation.py
```

**学习要点：**
- 掌握链路追踪：@traceable 装饰器、错误追踪
- 学会调试技巧：运行回放、中间变量检查、对比实验
- 理解评估体系：数据集管理、自定义评估器、LLM自评

**难度等级：** ⭐⭐⭐ - ⭐⭐⭐⭐

### [第十三章：LangSmith Prompt管理与监控](docs/13_LangSmith_Prompt监控篇.md)
| 序号 | 案例文件 | 知识点 | 示例数 | 难度 |
|------|---------|--------|-------|------|
| 13.1 | `langsmith_prompt_management.py` | Prompt版本管理、A/B测试、Prompt模板管理 | 4 | ⭐⭐⭐⭐ |
| 13.2 | `langsmith_monitoring.py` | 生产监控、告警配置、成本追踪、性能分析 | 4 | ⭐⭐⭐⭐ |

**运行方式：**
```bash
python src/chains/langsmith_prompt_management.py
python src/chains/langsmith_monitoring.py
```

**学习要点：**
- 掌握 Prompt 版本管理：版本控制、A/B测试、模板管理
- 学会生产监控：告警配置、成本追踪、性能分析

**难度等级：** ⭐⭐⭐⭐

---

## 📊 知识点统计

| 章节 | 知识点数 | 示例数 | 难度范围 |
|------|---------|-------|---------|
| 基础篇 | 12 | 13 | ⭐ - ⭐⭐ |
| 提示词篇 | 4 | 4 | ⭐⭐ |
| 输出篇 | 11 | 12 | ⭐⭐ - ⭐⭐⭐ |
| 工具篇 | 11 | 11 | ⭐⭐ - ⭐⭐⭐⭐ |
| Agent篇 | 14 | 16 | ⭐⭐⭐ - ⭐⭐⭐⭐ |
| 中间件篇 | 8 | 8 | ⭐⭐⭐ |
| 记忆与存储篇 | 8 | 8 | ⭐⭐⭐ |
| RAG篇 | 14 | 16 | ⭐⭐⭐ - ⭐⭐⭐⭐ |
| LangGraph基础篇 | 12 | 12 | ⭐⭐⭐ - ⭐⭐⭐⭐ |
| LangGraph Agent篇 | 12 | 12 | ⭐⭐⭐⭐ - ⭐⭐⭐⭐⭐ |
| LangServe部署篇 | 12 | 12 | ⭐⭐⭐ - ⭐⭐⭐⭐ |
| LangSmith追踪调试篇 | 12 | 12 | ⭐⭐⭐ - ⭐⭐⭐⭐ |
| LangSmith Prompt监控篇 | 8 | 8 | ⭐⭐⭐⭐ |
| **总计** | **138** | **144** | ⭐ - ⭐⭐⭐⭐⭐ |

---

## 🚀 快速运行

```bash
# 激活环境
cd /Users/mr.liu/Desktop/langchain
source .venv/bin/activate

# 运行任意案例
python src/chains/basic_chain.py          # 基础链
python src/chains/agent_basics.py         # Agent基础
python src/chains/rag_basics.py           # RAG基础
```

---

## 📖 学习建议

**入门路线（1-2周）：**
1. 基础篇 → 2. 提示词篇 → 3. 输出篇

**进阶路线（3-4周）：**
4. 工具篇 → 5. Agent篇

**高级路线（5-6周）：**
6. 中间件篇 → 7. 记忆与存储篇 → 8. RAG篇

**专家路线（7-10周）：**
9. LangGraph基础篇 → 10. LangGraph Agent篇 → 11. LangServe部署篇 → 12. LangSmith追踪调试篇 → 13. LangSmith Prompt监控篇