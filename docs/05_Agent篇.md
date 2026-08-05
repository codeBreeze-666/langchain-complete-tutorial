# 第五章：Agent篇

## 5.1 Agent基础（agent_basics.py）

### 知识点：create_agent、带记忆Agent、Agent调试

> **历史演进**：旧版使用 `create_tool_calling_agent` + `AgentExecutor`（已废弃），现代方式使用 LangChain v1 的 `create_agent`。

**运行方式：**

```bash
python src/chains/agent_basics.py
```

**核心概念：**

- **Agent**：能够自主选择工具、规划步骤的智能体
- **Tool Calling**：Agent 通过调用工具完成实际任务
- **create_agent**：LangChain v1 的 Agent 创建函数（底层运行在 LangGraph 上）
- **system_prompt**：Agent 的系统提示词，定义 Agent 的行为风格

**原创工具集：**

本示例使用 3 个原创语言工具：

| 工具 | 功能 | 适用场景 |
|------|------|----------|
| `get_word_meaning` | 查询词语含义 | 用户想了解某个词语的定义 |
| `get_synonym` | 获取同义词 | 用户想寻找替换词或丰富表达 |
| `get_abbreviation` | 获取缩写全称 | 用户遇到不认识的缩写想了解含义 |

---

### 示例1：基础Agent — 自动选择工具

**功能说明：** 用户输入问题，Agent 根据问题自动判断需要哪个工具并调用。

**关键代码：**

```python
from langchain.agents import create_agent

# 定义工具
tools = [get_word_meaning, get_synonym, get_abbreviation]

# 创建 Agent
agent = create_agent(model, tools, system_prompt="你是一个语言助手...")

# 调用
result = agent.invoke({"messages": [("user", "什么是人工智能？")]})
```

**实战要点：**

1. Agent 通过工具的 **docstring** 理解工具用途，决定何时调用
2. `system_prompt` 参数定义 Agent 的行为风格，替代旧版的提示词模板
3. Agent 会将工具返回的结果整合为自然语言回答
4. 调用时使用 `{"messages": [...]}` 格式传入消息

---

### 示例2：带记忆的Agent — 对话上下文感知

**功能说明：** Agent 通过 `chat_history` 参数记住之前的对话，能理解追问中的指代。

**关键代码：**

```python
# 创建带记忆的 Agent
agent = create_agent(model, tools, system_prompt="你是一个语言助手...你可以记住之前的对话内容，理解用户的追问。")

# 调用时传入消息历史
result = agent.invoke({
    "messages": chat_history + [("user", user_input)],
})

# 调用后更新记忆
chat_history.append(HumanMessage(content=user_input))
chat_history.append(AIMessage(content=final_answer))
```

**实战要点：**

1. `chat_history` 参数传递对话上下文，使用 `HumanMessage` + `AIMessage` 构建
2. Agent 能理解追问中的指代关系（如"它"、"这个"）
3. 记忆列表需要手动维护——每次对话后都要追加到 `chat_history`
4. 输入"清空"可重置记忆，开始新的对话

---

### 示例3：自定义工具Agent — 体验不同工具组合

**功能说明：** 同一个 Agent 搭配不同的工具集，体验工具数量对 Agent 决策范围的影响。

**两种模式对比：**

| 模式 | 工具集 | 特点 |
|------|--------|------|
| 词典模式 | 仅 `get_word_meaning` | 精确释义，回答严谨 |
| 全功能模式 | 含义+同义词+缩写 | 综合查询，回答丰富 |

**关键代码：**

```python
# 词典模式
dict_tools = [get_word_meaning]
dict_agent = create_agent(model, dict_tools, system_prompt="你是一个专业词典助手...")

# 全功能模式
full_tools = [get_word_meaning, get_synonym, get_abbreviation]
full_agent = create_agent(model, full_tools, system_prompt="你是一个全方位语言助手...尽可能综合利用多种工具...")
```

**实战要点：**

1. 不同工具集 = 不同能力边界，工具越多 Agent 能力越强
2. `system prompt` 决定 Agent 的行为风格（严谨 vs 综合利用）
3. 工具越多决策复杂度也越高，需平衡工具数量与准确性
4. 可以在运行时动态切换工具集

---

### 示例4：Agent调试 — 观察思考过程

**功能说明：** 通过分析 Agent 返回的 messages 获取完整执行轨迹。

**关键代码：**

```python
# 创建 Agent
agent = create_agent(model, tools, system_prompt="你是一个语言助手...")

result = agent.invoke({"messages": [("user", user_input)]})

# 解析 Agent 的思考过程
messages = result.get("messages", [])
for msg in messages:
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"工具名称：{tc['name']}")
            print(f"调用参数：{tc['args']}")
```

**实战要点：**

1. `result["messages"]` 包含完整的对话和工具调用历史
2. 通过检查 `msg.tool_calls` 可以看到每一步的工具调用详情
3. 通过调试可以诊断：Agent 选错工具、参数错误、多步推理逻辑等问题
4. 如果没有 tool_calls，说明 Agent 未调用任何工具，直接回答

---

## 5.2 Agent工作流（agent_workflow.py）

### 知识点：顺序工作流、条件工作流(RunnableBranch)、循环工作流、并行工作流

**运行方式：**

```bash
python src/chains/agent_workflow.py
```

**核心概念：**

| 工作流类型 | 核心机制 | 典型场景 |
|-----------|---------|---------|
| 顺序工作流 | `\|` 管道符串联 | 旅行规划（景点→美食→行程） |
| 条件工作流 | `RunnableBranch` | 投资分析（保守/稳健/激进） |
| 循环工作流 | Python 循环 + chain | 学习计划迭代优化 |
| 并行工作流 | `RunnableParallel` | 面试准备（技术/项目/行为并行） |

---

### 示例1：顺序工作流 — 旅行规划助手

**功能说明：** 按景点推荐 → 美食攻略 → 完整行程的顺序，前一步输出作为后一步输入。

**关键代码：**

```python
from langchain_core.output_parsers import StrOutputParser

# 每个步骤独立的 prompt 和 chain
spots_chain = spot_prompt | llm | parser
food_chain = food_prompt | llm | parser
itinerary_chain = itinerary_prompt | llm | parser

# 手动串联执行
spots_result = spots_chain.invoke({"destination": "成都", "days": "3", "style": "休闲"})
food_result = food_chain.invoke({"spots": spots_result})
itinerary_result = itinerary_chain.invoke({"spots": spots_result, "food": food_result})
```

**实战要点：**

1. 使用 LCEL 的 `|` 操作符串联步骤，数据自动流向下一步
2. 每个 prompt 只关注自己的职责，降低复杂度
3. 中间结果用变量保存，供后续步骤或最终汇总使用
4. 顺序工作流的瓶颈在于最慢的那一步

---

### 示例2：条件工作流 — 投资分析助手

**功能说明：** 根据用户的风险偏好（保守/稳健/激进），通过 `RunnableBranch` 路由到不同的分析策略。

**关键代码：**

```python
from langchain_core.runnables import RunnableBranch

# 定义条件函数
def is_conservative(x): return x.get("risk_type") == "conservative"
def is_balanced(x): return x.get("risk_type") == "balanced"
def is_aggressive(x): return x.get("risk_type") == "aggressive"

# 构建条件路由（最后一个参数是默认分支）
branch_chain = RunnableBranch(
    (is_conservative, conservative_chain),
    (is_balanced, balanced_chain),
    (is_aggressive, aggressive_chain),
    default_chain,  # 无条件默认分支
)

# 调用
result = branch_chain.invoke({"risk_type": "aggressive", "budget": "10万", ...})
```

**实战要点：**

1. `RunnableBranch` 替代传统的 if-else 硬编码，更声明式
2. 每个分支是独立 chain，可自由组合不同 prompt/model/parser
3. 最后一个参数是无条件默认分支，确保所有输入都有处理
4. 条件函数接收输入字典，返回布尔值

---

### 示例3：循环工作流 — 学习计划优化

**功能说明：** 初始生成学习计划，用户评分，低于4分则根据反馈迭代修改，直到满意或达到最大迭代次数。

**关键代码：**

```python
max_iterations = 5

for iteration in range(1, max_iterations + 1):
    result = plan_chain.invoke({
        "subject": subject, "level": level,
        "hours": hours, "goal": goal,
        "feedback_section": feedback_section,  # 上一轮的反馈
    })

    score = int(input("请给这个计划打分（1-5分）："))
    if score >= 4:
        break  # 满意则退出循环

    # 将反馈注入下一轮 prompt
    feedback_section = f"【之前的学习计划】\n{result}\n\n【用户反馈】\n{feedback}\n\n"
```

**实战要点：**

1. LangChain 没有内置循环原语，用 Python while/for 循环 + chain.invoke 实现
2. 每次迭代把历史结果和反馈拼接到 prompt，让 LLM 优化
3. **必须设置最大迭代次数**，防止无限循环消耗 token
4. 反馈注入是循环工作流的核心——让 LLM 知道"哪里不好，怎么改"

---

### 示例4：并行工作流 — 面试准备助手

**功能说明：** 同时生成技术题、项目经验梳理、行为面试准备，最后汇总为面试备战手册。

**关键代码：**

```python
from langchain_core.runnables import RunnableParallel

# 并行执行三个维度
parallel_chain = RunnableParallel(
    tech=tech_chain,
    project=project_chain,
    behavior=behavior_chain,
)

# 调用——三个 chain 同时执行
parallel_results = parallel_chain.invoke(common_input)

# 结果以字典形式返回
print(parallel_results["tech"])
print(parallel_results["project"])
print(parallel_results["behavior"])

# 汇总
summary_result = summary_chain.invoke(parallel_results)
```

**实战要点：**

1. `RunnableParallel` 让多个 chain 同时执行，大幅缩短总耗时
2. 结果以字典形式返回，键名对应 `RunnableParallel` 中的键名
3. 并行结果可直接传递给后续 chain 做汇总处理
4. 适合多维度分析、多视角评估等可并行的场景

---

## 5.3 多Agent协作（multi_agent.py）

### 知识点：角色委派、流水线Agent、辩论Agent、主管Agent

**运行方式：**

```bash
python src/chains/multi_agent.py
```

**核心概念：**

| 协作模式 | 核心机制 | 典型场景 |
|---------|---------|---------|
| 角色委派 | 不同角色独立分析后汇总 | 需求评审会 |
| 流水线 Agent | 上一步输出 = 下一步输入 | 文章创作流水线 |
| 辩论 Agent | 对立立场交锋 + 评委评判 | 方案辩论赛 |
| 主管 Agent | 主管分配 → 专业处理 → 汇总回复 | 智能客服 |

---

### 示例1：角色委派 — 需求评审会

**功能说明：** 产品经理、技术负责人、测试负责人从各自角度分析需求，最后汇总三方意见。

**关键代码：**

```python
# 三个角色的独立 chain
pm_chain = pm_prompt | llm | parser        # 产品经理
tech_chain = tech_prompt | llm | parser    # 技术负责人
qa_chain = qa_prompt | llm | parser        # 测试负责人
summary_chain = summary_prompt | llm | parser  # 汇总 Agent

# 依次调用
pm_result = pm_chain.invoke({"requirement": requirement})
tech_result = tech_chain.invoke({"requirement": requirement})
qa_result = qa_chain.invoke({"requirement": requirement})

# 汇总三方意见
summary_result = summary_chain.invoke({
    "pm_opinion": pm_result,
    "tech_opinion": tech_result,
    "qa_opinion": qa_result,
})
```

**实战要点：**

1. 角色委派的核心是为每个 Agent 设计**清晰的角色 prompt**
2. 同一个 LLM 实例可以复用，**prompt 决定了 Agent 的"性格"**
3. 各角色输出独立，最后通过汇总 Agent 整合意见
4. 可以并行调用各角色 Agent 提升效率

---

### 示例2：流水线Agent — 文章创作流水线

**功能说明：** 选题 → 大纲 → 初稿 → 润色，四个 Agent 按顺序处理，每步聚焦单一职责。

**关键代码：**

```python
topic_chain = topic_prompt | llm | parser      # 选题 Agent
outline_chain = outline_prompt | llm | parser   # 大纲 Agent
draft_chain = draft_prompt | llm | parser       # 初稿 Agent
polish_chain = polish_prompt | llm | parser     # 润色 Agent

# 流水线执行
topic_result = topic_chain.invoke({"theme": theme, "audience": audience})
outline_result = outline_chain.invoke({"topic": chosen_topic})
draft_result = draft_chain.invoke({"outline": outline_result})
polish_result = polish_chain.invoke({"draft": draft_result})
```

**实战要点：**

1. 流水线模式的核心是"上一步输出 = 下一步输入"
2. 每个 prompt 职责单一，避免一个 prompt 做太多事
3. 可在步骤之间插入用户交互（如选择选题方向）
4. 每步的 system prompt 精确定义职责（如"只输出大纲，不要写正文"）

---

### 示例3：辩论Agent — 方案辩论赛

**功能说明：** 正方支持、反方反对、评委评判，三方交锋产生更全面的分析。

**关键代码：**

```python
pro_chain = pro_prompt | llm | parser           # 正方
con_chain = con_prompt | llm | parser           # 反方（可以看到正方论点）
pro_closing_chain = pro_closing_prompt | llm | parser  # 正方总结
judge_chain = judge_prompt | llm | parser       # 评委

# 辩论流程
pro_result = pro_chain.invoke({"motion": motion})
con_result = con_chain.invoke({"motion": motion, "pro_arguments": pro_result})
pro_closing = pro_closing_chain.invoke({"motion": motion, "pro_arguments": pro_result, "con_arguments": con_result})
judge_result = judge_chain.invoke({"motion": motion, "pro_arguments": pro_result, "con_arguments": con_result, "pro_closing": pro_closing})
```

**实战要点：**

1. 辩论 Agent 的核心是用**对立的 prompt** 让 LLM 产生不同立场
2. 反方可以看到正方论点再反驳，形成真正的交锋而非各说各话
3. 评委 Agent 综合双方观点，输出比单方分析更全面、更客观
4. 正方有总结陈词环节，回应反方质疑

---

### 示例4：主管Agent — 智能客服主管

**功能说明：** 主管 Agent 分析问题类型并分配任务给专业 Agent（技术支持/售后服务/产品建议），最后汇总回复。

**关键代码：**

```python
# 主管分析并分配任务（输出 JSON）
allocation_result = supervisor_chain.invoke({"question": question})
tasks = json.loads(allocation_result)  # 解析分配结果

# 专业 Agent 处理各自任务
if tasks.get("tech_support"):
    tech_result = tech_chain.invoke({"task": tasks["tech_support"]})
if tasks.get("after_sales"):
    sales_result = sales_chain.invoke({"task": tasks["after_sales"]})

# 主管汇总
final_result = summary_chain.invoke({"question": question, "team_replies": team_replies})
```

**实战要点：**

1. 主管 Agent 负责"理解需求 → 分配任务 → 汇总结果"，是调度核心
2. 主管可以让 LLM 输出结构化数据（如 JSON）来控制任务路由
3. 专业 Agent 只处理自己领域的任务，输出更精准
4. JSON 解析失败时需要降级策略（如所有团队都派上用场）

---

## 5.4 人工介入（human_in_loop.py）

### 知识点：审批流程、内容审核、决策检查、协作编辑

**运行方式：**

```bash
python src/chains/human_in_loop.py
```

**核心概念：**

| 模式 | 核心机制 | 适用场景 |
|------|---------|---------|
| 审批流程 | AI 生成 → 人工批准/拒绝/重新生成 | 邮件发送、数据操作 |
| 内容审核 | AI 生成 → 人工修改 → 确认发布 | 文案撰写、报告生成 |
| 决策检查 | AI 分析方案 → 人工选择 → 生成执行计划 | 策略选择、方案取舍 |
| 协作编辑 | AI 起草 ↔ 人工精修 循环 | 长文写作、方案策划 |

---

### 示例1：审批流程

**功能说明：** AI 生成方案后不能自动执行，需人工审批。用户可批准、拒绝或要求重新生成。

**交互流程：**

```
用户输入任务 → AI 生成方案 → 展示方案 → 用户审批
  ├─ 批准执行 → 执行方案
  ├─ 拒绝执行 → 填写拒绝原因，取消操作
  └─ 重新生成 → 再次调用 LLM 生成新方案
```

**实战要点：**

1. AI 生成**不可逆操作**的方案时，必须加入审批环节
2. 提供"重新生成"选项，避免一次不满意就全部推倒
3. 拒绝时要求填写原因，便于后续分析优化
4. 审批日志应持久化存储，满足合规审计需求

---

### 示例2：内容审核

**功能说明：** AI 生成文案后，用户可以在 AI 输出基础上修改再确认，而非从零开始。

**交互流程：**

```
用户输入产品信息 → AI 生成初稿 → 展示当前内容 → 用户选择
  ├─ 确认发布 → 正式发布
  ├─ 修改内容 → 用户手动修改 → 再次确认
  ├─ 重新生成 → AI 再次生成
  └─ 放弃 → 取消当前文案
```

**实战要点：**

1. 内容审核的核心是"在 AI 输出基础上修改"，效率远高于从零写
2. 修改后应展示完整内容供用户再次确认，避免误操作
3. 保留重新生成选项，当修改量过大时不如重写
4. 生产中可记录修改轨迹，用于分析 AI 输出的薄弱环节

---

### 示例3：决策检查

**功能说明：** AI 分析决策方案并给出建议，但最终决策权在人类手中。用户可选择采纳推荐方案、自定义方案或暂不决策。

**交互流程：**

```
用户描述决策场景 → AI 生成决策分析报告（含多方案对比）
  → 人工决策：
  ├─ 采纳 AI 推荐 → 生成执行计划
  ├─ 选择自定义方案 → 生成执行计划
  └─ 暂不决策 → 终止流程
```

**实战要点：**

1. AI 只做分析建议，**最终决策权在人类手中**
2. 展示多方案对比，避免 AI 只给一个答案就执行
3. 决策确认后再生成执行计划，避免无效计算
4. 保留"暂不决策"选项，不强迫用户在信息不足时做决定

---

### 示例4：协作编辑

**功能说明：** AI 和用户交替编辑内容，逐步完善。用户可提修改意见让 AI 改、也可直接手动修改。

**交互流程：**

```
用户输入主题 → AI 起草初稿 → 多轮协作编辑
  ├─ 提出修改意见 → AI 根据意见修订
  ├─ 直接修改内容 → 可选 AI 润色
  ├─ 满意结束 → 展示最终内容
  └─ 放弃 → 取消本次编辑
```

**实战要点：**

1. 协作编辑的核心是"AI 起草 + 人类精修"的循环
2. 提供两种修改方式：给意见让 AI 改、自己直接改
3. 用户手动编辑后可选 AI 润色，兼顾效率和品质
4. 记录编辑轮数，方便评估协作效率

---

## 本章小结

本章深入探讨了 LangChain 中 Agent 的四大核心主题：

1. **Agent基础**：掌握了 `create_agent` 的使用，理解了工具定义（docstring 是 Agent 决策的关键）、对话记忆（chat_history）、自定义工具组合和调试技巧（messages 分析）。

2. **Agent工作流**：学习了四种工作流编排模式——顺序（管道串联）、条件（RunnableBranch 路由）、循环（Python 循环 + 反馈注入）、并行（RunnableParallel 多维度同时执行）。

3. **多Agent协作**：理解了角色委派（多角色独立分析汇总）、流水线（上一步输出 = 下一步输入）、辩论（对立立场交锋）、主管（任务分配与汇总）四种协作模式。

4. **人工介入**：掌握了审批流程、内容审核、决策检查、协作编辑四种 Human-in-the-Loop 模式，确保 AI 系统的关键环节有人类把关。

**关键原则：**
- Agent 的能力边界由工具集和 prompt 决定
- 工作流编排要根据业务特点选择合适的模式
- 多 Agent 协作的关键是角色分工和信息传递
- 人工介入是不可逆操作和高风险决策的必备环节
