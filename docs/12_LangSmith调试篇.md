# 第十二章：LangSmith 调试篇

本章介绍 LangSmith 的追踪与调试功能。LangSmith 是 LangChain 团队提供的云平台，用于追踪、调试、评估和监控 LLM 应用。掌握 Tracing、@traceable、Run/Span 等概念，是排查 AI 应用问题、优化性能的关键。

> 下一章：[13_LangSmith评估篇](13_LangSmith评估篇.md) | 上一章：[11_LangServe部署篇](11_LangServe部署篇.md)

---

## 12.1 LangSmith 概述

### 什么是 LangSmith

LangSmith 是一个专为 LLM 应用设计的开发者平台，提供追踪（Tracing）、调试（Debugging）、评估（Evaluation）和监控（Monitoring）四大核心功能。它可以帮助开发者理解 AI 应用的内部运行过程，快速定位和解决问题。

### 为什么需要 LangSmith

| 痛点 | 没有 LangSmith 的困难 | LangSmith 的解决方案 |
|------|---------------------|---------------------|
| **黑盒问题** | LLM 调用是黑盒，不知道内部发生了什么 | 追踪每次调用的完整链路 |
| **调试困难** | 出错时无法定位是哪个步骤出了问题 | Run/Span 记录每步的输入输出 |
| **性能瓶颈** | 不知道哪个步骤耗时最长 | 耗时统计 + 性能分析 |
| **错误追踪** | 生产环境出错后无法复现 | Run 回放 + 错误上下文记录 |

### 核心功能

| 功能 | 说明 | 对应章节 |
|------|------|---------|
| **Tracing（追踪）** | 记录每次调用的完整链路 | 本章 12.3 |
| **Debugging（调试）** | Run 回放、中间变量、对比实验 | 本章 12.4 |
| **Evaluation（评估）** | 数据集、评估器、LLM 自评 | [13_LangSmith评估篇](13_LangSmith评估篇.md) |
| **Monitoring（监控）** | 延迟、错误率、Token 消耗 | [13_LangSmith评估篇](13_LangSmith评估篇.md) |

### 环境配置

```bash
# 在 .env 文件中配置
LANGSMITH_API_KEY=your-api-key
LANGSMITH_PROJECT=my-project
LANGSMITH_TRACING=true
```

> 注意：本教程的示例使用模拟模式演示概念，配置 `LANGSMITH_API_KEY` 后可连接真实 LangSmith 服务。

---

## 12.2 核心概念总览

| 概念 | 说明 |
|------|------|
| **Tracing** | 链路追踪，记录每次调用的完整链路，包括嵌套调用 |
| **@traceable** | 装饰器，标记函数使其自动追踪 |
| **Run** | 一次完整追踪记录，从开始到结束的完整执行过程 |
| **Span** | 追踪中的一个步骤，Run 由多个 Span 组成 |
| **Debugging** | 调试，包括 Run 回放、中间变量查看、对比实验 |

### Run 与 Span 的关系

```
Run（智能问答链）
├── Span1（知识检索）[retriever]  120ms
├── Span2（答案生成）[llm]        1500ms
│   └── Child Span（文档评分）[tool]  50ms
└── Span3（结果校验）[tool]        30ms
```

- **Run** 是顶层追踪记录，代表一次完整的调用
- **Span** 是 Run 中的子步骤，记录输入、输出、耗时、状态
- Span 可以嵌套（子步骤），形成树形结构

---

## 12.3 追踪（langsmith_tracing.py）

### 运行方式

```bash
python src/chains/langsmith_tracing.py
```

程序启动后进入交互式菜单，可选择运行 4 个示例。

### 示例1：@traceable 装饰器 — 函数追踪

**场景**：追踪函数调用链，记录每步的输入、输出、耗时

**核心代码**：

```python
from langsmith import traceable

@traceable(name="知识检索", run_type="retriever")
def retrieve_knowledge(query: str) -> str:
    """模拟知识检索步骤"""
    return f"关于'{query}'的检索结果：[相关文档1, 相关文档2]"

@traceable(name="答案生成", run_type="chain")
def generate_answer(query: str, context: str) -> str:
    """使用 LLM 生成答案"""
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"query": query, "context": context})
```

**学习要点**：

- `@traceable` 装饰器自动追踪函数调用
- 每个步骤记录输入、输出、耗时和状态
- 嵌套调用形成完整的调用链
- 真实 LangSmith 中可在 Web 界面查看追踪树

### 示例2：自动追踪 — LangChain 追踪

**场景**：LangChain) Chain 组件自动支持追踪

**核心代码**：

```bash
# 设置环境变量即可开启自动追踪
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your-key
```

**学习要点**：

- LangChain 组件（Prompt/LLM/Parser）自动支持追踪
- 设置 `LANGSMITH_TRACING=true` 即可开启
- 每个组件的调用都会被记录为 Span
- 可查看每步的 Token 消耗和耗时

### 示例3：链路可视化 — 调用链展示

**场景**：以树形结构展示完整的调用链路

**输出示例**：

```
✅ RAG问答链 [chain] (3250ms)
   ↳ 输入 question: "什么是量子计算？"
   ✅ 问题理解 [chain] (800ms)
   ✅ 知识检索 [retriever] (150ms)
      ✅ 文档评分 [tool] (50ms)
   ✅ 答案生成 [llm] (2000ms)
   ✅ 答案优化 [chain] (300ms)
```

**学习要点**：

- 调用链路以树形结构展示，清晰直观
- 每个步骤记录输入输出和耗时
- 子步骤（如文档评分）嵌套在父步骤中
- 真实 LangSmith 提供 Web 界面交互式可视化

### 示例4：错误追踪 — 错误定位

**场景**：自动追踪错误发生的位置和上下文

**学习要点**：

- 错误追踪自动记录错误位置和上下文
- 可回溯错误链路到根因
- 降级处理也会被追踪记录
- 真实 LangSmith 中错误会标红显示

---

## 12.4 调试（langsmith_debugging.py）

### 运行方式

```bash
python src/chains/langsmith_debugging.py
```

### 知识点

| 概念 | 说明 |
|------|------|
| **Run 回放** | 复现之前的追踪记录，重新执行相同的调用 |
| **中间变量** | 查看每一步的输入输出，定位问题 |
| **对比实验** | 对比不同参数的输出差异，优化效果 |
| **性能分析** | 分析每一步的耗时和 Token 消耗 |

### 调试模式对比

| 调试方式 | 说明 | 适用场景 |
|---------|------|---------|
| **Run 回放** | 复现之前的追踪，重新执行 | 生产环境问题复现 |
| **中间变量** | 查看每步的输入输出 | 定位链中哪个步骤出错 |
| **对比实验** | A/B 测试不同参数 | 优化 Prompt/参数 |
| **性能分析** | 分析耗时和 Token 消耗 | 性能优化、成本控制 |

### 示例1：Run 回放 — 复现问题

**场景**：复现生产环境中出现的问题

**学习要点**：

- Run 回放可以重新执行之前的追踪记录
- 使用相同的输入，验证A查看输出是否一致
- 适合复现生产环境中的偶发问题

### 示例2：中间变量 — 查看每步输入输出

**场景**：调试 Chain 中某个步骤的输出不符合预期

**学习要点**：

- 每个步骤的输入输出都被记录
- 可以精确定位是哪一步出了问题
- 比在代码中加 print 更优雅

### 示例3：对比实验 — A/B 测试

**场景**：对比不同 Prompt 或参数的效果

**学习要点**：

- 对比实验可以同时运行多个版本
- 直观看到不同参数的输出差异
- 适合优化 Prompt 模板、温度等参数

### 示例4：性能分析 — 耗时与 Token

**场景**：分析性能瓶颈和 Token 消耗

**学习要点**：

- 每个步骤的耗时被精确记录
- Token 消耗按步骤统计
- 可以找出耗时最长的步骤进行优化

---

## 12.5 追踪与调试的协同工作流

### 典型调试流程

```
1. 开启追踪 → 运行应用 → LangSmith 自动记录
2. 发现问题 → 查看追踪链路 → 定位出错的 Span
3. 查看中间变量 → 理解每步的输入输出
4. Run 回放 → 复现问题
5. 对比实验 → 验证修复效果
6. 性能分析 → 优化瓶颈步骤
```

### 环境变量速查

| 环境变量 | 说明 | 示例值 |
|---------|------|-------|
| `LANGSMITH_API_KEY` | API 密钥 | `ls-xxxxx` |
| `LANGSMITH_PROJECT` | 项目名称 | `my-llm-app` |
| `LANGSMITH_TRACING` | 开启追踪 | `true` |
| `LANGSMITH_ENDPOINT` | 自定义端点 | `https://api.smith.langchain.com` |

---

## 12.6 学习路径与建议

### 推荐学习顺序

```
langsmith_tracing.py（示例1→2→3→4）
    ↓ 掌握 @traceable/Run/Span/链路可视化
langsmith_debugging.py（示例1→2→3→4）
    ↓ 掌握 Run回放/中间变量/对比实验/性能分析
```

### 核心概念掌握检查

| 阶段 | 必须掌握 | 进阶理解 |
|------|---------|---------|
| 追踪 | @traceable、Run、Span | 链路可视化、错误追踪 |
| 调试 | 中间变量查看 | Run 回放、对比实验、性能分析 |

### 实战建议

1. **开发阶段就开启追踪**：不要等到出问题才追踪，开发时就养成追踪习惯
2. **关注 Token 消耗**：Token 是 LLM 应用的主要成本，追踪可以帮助优化
3. **善用 Run 回放**：生产环境的问题用 Run 回放复现，比加日志更高效
4. **对比实验优化 Prompt**：用 A/B 测试对比不同 Prompt 的效果，数据驱动优化
5. **先模拟后真实**：本教程示例使用模拟模式，理解概念后配置 API Key 连接真实服务

> 下一章：[13_LangSmith评估篇](13_LangSmith评估篇.md) → 学习评估、Prompt 管理、监控
