# LangChain Examples Index

> 36 hands-on cases, 143 interactive examples, 100% original

---

## Project Statistics

| Metric | Count |
|--------|-------|
| Total Cases | 36 |
| Total Examples | 143 |
| Chapters | 14 |
| Model Providers | 5 |
| Difficulty Levels | 5 (⭐ to ⭐⭐⭐⭐⭐) |

---

## Examples by Chapter

### Chapter 1: Basics
| # | File | Key Topics | Examples | Difficulty |
|---|------|-----------|----------|-----------|
| 1.1 | `basic_chain.py` | LCEL chain, pipe operator, parallel chains, streaming output | 4 | ⭐ |
| 1.2 | `model_parameters_chain.py` | Temperature, MaxTokens, TopP, FrequencyPenalty, PresencePenalty | 5 | ⭐ |
| 1.3 | `message_types_demo.py` | SystemMessage, HumanMessage, AIMessage, ToolMessage | 4 | ⭐⭐ |

### Chapter 2: Prompts
| # | File | Key Topics | Examples | Difficulty |
|---|------|-----------|----------|-----------|
| 2.1 | `prompt_basics.py` | SystemPrompt, DynamicPrompt, PromptTemplate, MessagesPlaceholder | 4 | ⭐⭐ |

### Chapter 3: Output
| # | File | Key Topics | Examples | Difficulty |
|---|------|-----------|----------|-----------|
| 3.1 | `output_strategies.py` | StrOutputParser, JsonOutputParser, CommaSeparatedListOutputParser, custom parser | 4 | ⭐⭐ |
| 3.2 | `structured_output_fixed.py` | Pydantic model, PydanticOutputParser, information extraction, batch extraction | 4 | ⭐⭐⭐ |
| 3.3 | `streaming_demo.py` | Streaming output, stream() vs invoke(), progress display, real-time application | 4 | ⭐⭐ |

### Chapter 4: Tools
| # | File | Key Topics | Examples | Difficulty |
|---|------|-----------|----------|-----------|
| 4.1 | `tool_basics.py` | @tool decorator, agent tool usage, multi-tool collaboration | 3 | ⭐⭐ |
| 4.2 | `tool_advanced.py` | StructuredTool, input validation, field_validator, error handling, tool chains | 4 | ⭐⭐⭐ |
| 4.3 | `tool_injection.py` | InjectedState, InjectedStore, config injection, context-aware tools | 4 | ⭐⭐⭐⭐ |

### Chapter 5: Agents
| # | File | Key Topics | Examples | Difficulty |
|---|------|-----------|----------|-----------|
| 5.1 | `agent_basics.py` | create_tool_calling_agent, AgentExecutor, agent with memory, agent debugging | 4 | ⭐⭐⭐ |
| 5.2 | `agent_workflow.py` | Sequential workflow, conditional workflow (RunnableBranch), loop workflow, parallel workflow | 4 | ⭐⭐⭐ |
| 5.3 | `multi_agent.py` | Role delegation, pipeline agents, debate agents, supervisor agent | 4 | ⭐⭐⭐⭐ |
| 5.4 | `human_in_loop.py` | Approval flow, content moderation, decision checkpoint, collaborative editing | 4 | ⭐⭐⭐ |

### Chapter 6: Middleware
| # | File | Key Topics | Examples | Difficulty |
|---|------|-----------|----------|-----------|
| 6.1 | `middleware_basics.py` | BaseCallbackHandler, logging middleware, token tracking, performance monitoring, custom callbacks | 4 | ⭐⭐⭐ |
| 6.2 | `error_handling.py` | Retry with exponential backoff, model fallback (with_fallbacks), output validation, graceful degradation | 4 | ⭐⭐⭐ |

### Chapter 7: Memory & Storage
| # | File | Key Topics | Examples | Difficulty |
|---|------|-----------|----------|-----------|
| 7.1 | `checkpointer.py` | Conversation buffer, conversation summary, conversation window, session management, persistence | 4 | ⭐⭐⭐ |
| 7.2 | `store_basics.py` | Key-value store, user preferences, shared context, persistent storage | 4 | ⭐⭐⭐ |

### Chapter 8: RAG
| # | File | Key Topics | Examples | Difficulty |
|---|------|-----------|----------|-----------|
| 8.1 | `rag_basics.py` | Simple RAG, document Q&A, similarity search, sourced RAG | 4 | ⭐⭐⭐ |
| 8.2 | `document_loader.py` | Text splitting, document loading, chunk management, custom splitter | 4 | ⭐⭐⭐ |
| 8.3 | `rag_agent.py` | RAG Agent, multi-source RAG, conversational RAG, RAG + tools | 4 | ⭐⭐⭐⭐ |
| 8.4 | `langsmith_demo.py` | Tracing basics, evaluation system, dataset management, custom evaluator | 4 | ⭐⭐⭐ |

### Chapter 9: LangGraph Basics
| # | File | Key Topics | Examples | Difficulty |
|---|------|-----------|----------|-----------|
| 9.1 | `langgraph_basics.py` | StateGraph, State, Node, Edge, graph construction and execution | 4 | ⭐⭐⭐ |
| 9.2 | `langgraph_state.py` | State management, Reducer, Annotated, state aggregation and update | 4 | ⭐⭐⭐ |
| 9.3 | `langgraph_control_flow.py` | Conditional edges, loops, Send, subgraphs, complex control flow | 4 | ⭐⭐⭐⭐ |

**How to run:**
```bash
python src/chains/langgraph_basics.py
python src/chains/langgraph_state.py
python src/chains/langgraph_control_flow.py
```

**Key takeaways:**
- Understand StateGraph core concepts: State, Node, Edge
- Master state management: Reducer, Annotated type annotations
- Learn complex control flow: conditional edges, loops, Send operations, subgraph nesting

### Chapter 10: LangGraph Advanced
| # | File | Key Topics | Examples | Difficulty |
|---|------|-----------|----------|-----------|
| 10.1 | `langgraph_agent.py` | ReAct Agent, tool calling, cyclic reasoning, state-driven decisions | 4 | ⭐⭐⭐⭐ |
| 10.2 | `langgraph_human_in_loop.py` | interrupt(), human approval, breakpoint recovery, interactive decisions | 4 | ⭐⭐⭐⭐ |
| 10.3 | `langgraph_multi_agent.py` | Multi-agent collaboration, supervisor agent, pipeline, inter-agent communication | 4 | ⭐⭐⭐⭐⭐ |

**How to run:**
```bash
python src/chains/langgraph_agent.py
python src/chains/langgraph_human_in_loop.py
python src/chains/langgraph_multi_agent.py
```

**Key takeaways:**
- Master ReAct Agent pattern: reasoning-action-observation loop
- Understand human-in-the-loop: interrupt(), breakpoint recovery
- Learn multi-agent architectures: supervisor mode, pipeline mode

### Chapter 11: LangServe Deployment
| # | File | Key Topics | Examples | Difficulty |
|---|------|-----------|----------|-----------|
| 11.1 | `langserve_basics.py` | FastAPI integration, add_routes, invoke/batch/stream endpoints | 4 | ⭐⭐⭐ |
| 11.2 | `langserve_advanced.py` | RemoteRunnable, streaming, async invocation, remote calls | 4 | ⭐⭐⭐⭐ |
| 11.3 | `langserve_production.py` | Docker deployment, environment config, logging & monitoring, production best practices | 4 | ⭐⭐⭐⭐ |

**How to run:**
```bash
python src/chains/langserve_basics.py
python src/chains/langserve_advanced.py
python src/chains/langserve_production.py
```

**Key takeaways:**
- Master LangServe + FastAPI integration
- Understand remote invocation: RemoteRunnable, streaming
- Learn production deployment: Docker, env config, logging & monitoring

### Chapter 12: LangSmith Tracing & Debugging
| # | File | Key Topics | Examples | Difficulty |
|---|------|-----------|----------|-----------|
| 12.1 | `langsmith_tracing.py` | @traceable, trace analysis, error tracking, Trace inspection | 4 | ⭐⭐⭐ |
| 12.2 | `langsmith_debugging.py` | Run replay, intermediate variables, comparison experiments, debugging tips | 4 | ⭐⭐⭐ |
| 12.3 | `langsmith_evaluation.py` | Dataset management, evaluators, LLM self-evaluation, evaluation pipeline | 4 | ⭐⭐⭐⭐ |

**How to run:**
```bash
python src/chains/langsmith_tracing.py
python src/chains/langsmith_debugging.py
python src/chains/langsmith_evaluation.py
```

**Key takeaways:**
- Master tracing: @traceable decorator, error tracking
- Learn debugging: run replay, intermediate variable inspection, comparison experiments
- Understand evaluation: dataset management, custom evaluators, LLM self-evaluation

### Chapter 13: LangSmith Prompt Management & Monitoring
| # | File | Key Topics | Examples | Difficulty |
|---|------|-----------|----------|-----------|
| 13.1 | `langsmith_prompt_management.py` | Prompt versioning, A/B testing, prompt template management | 4 | ⭐⭐⭐⭐ |
| 13.2 | `langsmith_monitoring.py` | Production monitoring, alerting, cost tracking, performance analysis | 4 | ⭐⭐⭐⭐ |

**How to run:**
```bash
python src/chains/langsmith_prompt_management.py
python src/chains/langsmith_monitoring.py
```

**Key takeaways:**
- Master prompt version management: version control, A/B testing, template management
- Learn production monitoring: alert configuration, cost tracking, performance analysis

### Chapter 14: Real-world Projects
| # | File | Key Topics | Examples | Difficulty |
|---|------|-----------|----------|-----------|
| 14.1 | `project_learning_assistant.py` | Interactive learning assistant, RAG-based Q&A, adaptive feedback, knowledge graph | 4 | ⭐⭐⭐⭐⭐ |
| 14.2 | `project_data_analyst.py` | Data analysis agent, SQL query generation, visualization, report generation | 4 | ⭐⭐⭐⭐⭐ |
| 14.3 | `project_code_assistant.py` | Code generation, code review, debugging assistant, test generation | 4 | ⭐⭐⭐⭐⭐ |

**How to run:**
```bash
python src/chains/project_learning_assistant.py
python src/chains/project_data_analyst.py
python src/chains/project_code_assistant.py
```

**Key takeaways:**
- Integrate multiple LangChain components into real-world applications
- Combine RAG, agents, memory, and tools in production-like scenarios
- Learn end-to-end application architecture and best practices

---

## Quick Run

```bash
# Activate environment
source .venv/bin/activate

# Run any example
python src/chains/basic_chain.py          # Basics
python src/chains/agent_basics.py         # Agents
python src/chains/rag_basics.py           # RAG
python src/chains/langgraph_basics.py     # LangGraph
python src/chains/project_learning_assistant.py  # Real-world project
```

---

## Learning Suggestions

### Beginner (1-2 weeks)
1. Chapter 1: Basics → 2. Chapter 2: Prompts → 3. Chapter 3: Output

Build a solid foundation in LCEL, message types, prompt engineering, and output parsing.

### Intermediate (2-3 weeks)
4. Chapter 4: Tools → 5. Chapter 5: Agents

Learn to extend LLMs with custom tools and build autonomous agents.

### Advanced (3-4 weeks)
6. Chapter 6: Middleware → 7. Chapter 7: Memory & Storage → 8. Chapter 8: RAG

Master production concerns: monitoring, error handling, persistence, and retrieval-augmented generation.

### Expert (5-6 weeks)
9. Chapter 9: LangGraph Basics → 10. Chapter 10: LangGraph Advanced → 11. Chapter 11: LangServe → 12. Chapter 12: LangSmith Debugging → 13. Chapter 13: LangSmith Evaluation → 14. Chapter 14: Real-world Projects

Build production-grade applications with stateful workflows, deployment, observability, and evaluation.

---

> For the Chinese version of this index, see [EXAMPLES_INDEX.md](EXAMPLES_INDEX.md)
>
> For the project overview, see [README_EN.md](README_EN.md)
>
> For the quick start guide, see [QUICK_START_EN.md](QUICK_START_EN.md)
