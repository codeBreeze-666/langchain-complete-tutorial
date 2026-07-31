"""
LangSmith 追踪（Tracing）- 实战交互式案例
============================================

本示例演示 LangSmith 追踪功能的核心概念和使用方法

核心概念：
- @traceable 装饰器：自动追踪函数调用，记录输入输出和耗时
- Tracing（链路追踪）：记录每次调用的完整链路，包括嵌套调用
- Run（一次完整追踪记录）：从开始到结束的一次完整执行过程
- Span（追踪中的一个步骤）：Run 中的每一个子步骤

应用场景：
- 函数追踪：追踪函数调用链，定位性能瓶颈
- LangChain追踪：自动追踪 LangChain 组件的调用过程
- 链路可视化：以树形结构展示完整的调用链路
- 错误追踪：自动追踪错误发生的位置和上下文
"""

import os
import sys
import json
import time
import uuid
import traceback
from datetime import datetime
from typing import Any, Callable, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_loader import get_default_llm


# ============================================================
# 检测 LangSmith API Key
# ============================================================

def has_langsmith_key() -> bool:
    """检测是否配置了 LANGSMITH_API_KEY"""
    return bool(os.getenv("LANGSMITH_API_KEY"))


# ============================================================
# 模拟追踪存储（模拟 LangSmith 后端）
# ============================================================

class Span:
    """模拟 LangSmith 的 Span（追踪步骤）"""

    def __init__(self, name: str, run_type: str, parent_id: Optional[str] = None):
        self.id = f"span-{uuid.uuid4().hex[:8]}"
        self.name = name
        self.run_type = run_type
        self.parent_id = parent_id
        self.inputs: dict = {}
        self.outputs: dict = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status: str = "pending"
        self.error: Optional[str] = None
        self.children: list = []

    def start(self, inputs: dict = None):
        """开始追踪"""
        self.start_time = time.time()
        self.inputs = inputs or {}
        self.status = "running"

    def end(self, outputs: dict = None, error: str = None):
        """结束追踪"""
        self.end_time = time.time()
        self.outputs = outputs or {}
        if error:
            self.status = "error"
            self.error = error
        else:
            self.status = "success"

    @property
    def duration_ms(self) -> float:
        """耗时（毫秒）"""
        if self.start_time and self.end_time:
            return round((self.end_time - self.start_time) * 1000, 2)
        return 0

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "run_type": self.run_type,
            "parent_id": self.parent_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "children": [c.to_dict() for c in self.children],
        }


class Run:
    """模拟 LangSmith 的 Run（一次完整追踪记录）"""

    def __init__(self, name: str, run_type: str = "chain"):
        self.id = f"run-{uuid.uuid4().hex[:8]}"
        self.name = name
        self.run_type = run_type
        self.root_span: Optional[Span] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status: str = "pending"
        self.error: Optional[str] = None
        self.metadata: dict = {}
        self.timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def start(self, inputs: dict = None):
        """开始追踪"""
        self.start_time = time.time()
        self.root_span = Span(self.name, self.run_type)
        self.root_span.start(inputs)
        self.status = "running"

    def end(self, outputs: dict = None, error: str = None):
        """结束追踪"""
        self.end_time = time.time()
        self.root_span.end(outputs, error)
        if error:
            self.status = "error"
            self.error = error
        else:
            self.status = "success"

    @property
    def duration_ms(self) -> float:
        """耗时（毫秒）"""
        if self.start_time and self.end_time:
            return round((self.end_time - self.start_time) * 1000, 2)
        return 0

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "run_type": self.run_type,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "root_span": self.root_span.to_dict() if self.root_span else None,
        }


class TraceStore:
    """追踪数据存储"""

    _runs: list = []

    @classmethod
    def record(cls, run: Run) -> str:
        """记录一次追踪"""
        cls._runs.append(run)
        return run.id

    @classmethod
    def get_all(cls) -> list:
        """获取所有追踪记录"""
        return cls._runs

    @classmethod
    def get_by_id(cls, run_id: str) -> Optional[Run]:
        """按 ID 获取追踪记录"""
        for run in cls._runs:
            if run.id == run_id:
                return run
        return None

    @classmethod
    def clear(cls):
        """清空追踪记录"""
        cls._runs = []


# ============================================================
# 模拟 @traceable 装饰器
# ============================================================

def traceable(name: str = None, run_type: str = "tool"):
    """
    模拟 LangSmith 的 @traceable 装饰器
    自动追踪函数的输入、输出、耗时和错误

    Args:
        name: 追踪名称，默认使用函数名
        run_type: 运行类型（chain/tool/retriever等）
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            trace_name = name or func.__name__
            span = Span(trace_name, run_type)
            span.start({"args": str(args)[:200], "kwargs": str(kwargs)[:200]})

            try:
                result = func(*args, **kwargs)
                span.end({"result": str(result)[:500]})
                return result
            except Exception as e:
                span.end(error=str(e))
                raise

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# ============================================================
# 辅助函数：打印追踪树
# ============================================================

def print_span_tree(span_dict: dict, indent: int = 0):
    """以树形结构打印 Span"""
    prefix = "  " * indent
    status_icon = "✅" if span_dict["status"] == "success" else "❌"
    duration = f"{span_dict['duration_ms']:.0f}ms" if span_dict["duration_ms"] else "..."
    print(f"{prefix}{status_icon} {span_dict['name']} [{span_dict['run_type']}] ({duration})")

    if span_dict.get("inputs"):
        for k, v in span_dict["inputs"].items():
            val_str = str(v)[:80]
            print(f"{prefix}   ↳ 输入 {k}: {val_str}")

    if span_dict.get("outputs"):
        for k, v in span_dict["outputs"].items():
            val_str = str(v)[:80]
            print(f"{prefix}   ↳ 输出 {k}: {val_str}")

    if span_dict.get("error"):
        print(f"{prefix}   ⚠️ 错误: {span_dict['error']}")

    for child in span_dict.get("children", []):
        print_span_tree(child, indent + 1)


# ============================================================
# 示例1: @traceable 装饰器 - 函数追踪
# ============================================================

def demo_traceable_decorator():
    """示例1：@traceable装饰器 - 函数追踪（用户输入问题，自动追踪函数调用链）"""
    print("\n" + "="*60)
    print("示例1：@traceable装饰器 - 函数追踪")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - @traceable: 装饰器，自动追踪函数调用")
    print("   - 装饰后的函数会自动记录输入、输出、耗时")
    print("   - 支持嵌套追踪，形成完整的调用链")
    print("\n📊 应用场景：")
    print("   - 追踪复杂函数调用链")
    print("   - 定位性能瓶颈")
    print("   - 记录每一步的输入输出")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
        print("   本示例仍使用模拟模式演示概念，真实模式请参考 LangSmith 文档")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    # 定义被追踪的函数
    @traceable(name="知识检索", run_type="retriever")
    def retrieve_knowledge(query: str) -> str:
        """模拟知识检索步骤"""
        time.sleep(0.1)  # 模拟检索延迟
        return f"关于'{query}'的检索结果：[相关文档1, 相关文档2, 相关文档3]"

    @traceable(name="答案生成", run_type="chain")
    def generate_answer(query: str, context: str) -> str:
        """使用 LLM 生成答案"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "根据上下文回答问题，简洁准确\n上下文：{context}"),
            ("human", "{query}")
        ])
        chain = prompt | model | StrOutputParser()
        return chain.invoke({"query": query, "context": context})

    @traceable(name="结果校验", run_type="tool")
    def validate_answer(answer: str) -> dict:
        """校验答案质量"""
        time.sleep(0.05)  # 模拟校验延迟
        return {
            "length": len(answer),
            "has_content": len(answer) > 10,
            "quality_score": min(100, len(answer) * 2)
        }

    print("\n【交互式函数追踪演示】")
    print("提示：输入问题，系统自动追踪函数调用链")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()
        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break
        if not question:
            print("请输入有效问题")
            continue

        # 创建追踪记录
        run = Run("智能问答链", "chain")
        run.start({"question": question})

        print(f"\n📋 追踪开始 [Run ID: {run.id}]")
        print("-"*60)

        # 步骤1：知识检索
        step1_span = Span("知识检索", "retriever", run.root_span.id)
        step1_span.start({"query": question})
        print("  🔍 步骤1：知识检索...")
        try:
            context = retrieve_knowledge(question)
            step1_span.end({"context": context})
            print(f"  ✅ 检索完成 ({step1_span.duration_ms:.0f}ms)")
        except Exception as e:
            step1_span.end(error=str(e))
            context = ""
            print(f"  ❌ 检索失败: {e}")

        # 步骤2：答案生成
        step2_span = Span("答案生成", "chain", run.root_span.id)
        step2_span.start({"query": question, "context": context})
        print("  🤖 步骤2：答案生成...")
        try:
            answer = generate_answer(question, context)
            step2_span.end({"answer": answer[:200]})
            print(f"  ✅ 生成完成 ({step2_span.duration_ms:.0f}ms)")
        except Exception as e:
            step2_span.end(error=str(e))
            answer = ""
            print(f"  ❌ 生成失败: {e}")

        # 步骤3：结果校验
        step3_span = Span("结果校验", "tool", run.root_span.id)
        step3_span.start({"answer": answer[:100]})
        print("  ✅ 步骤3：结果校验...")
        try:
            validation = validate_answer(answer)
            step3_span.end({"validation": validation})
            print(f"  ✅ 校验完成 ({step3_span.duration_ms:.0f}ms)")
        except Exception as e:
            step3_span.end(error=str(e))
            validation = {}
            print(f"  ❌ 校验失败: {e}")

        # 结束追踪
        run.root_span.children = [step1_span, step2_span, step3_span]
        run.end({"answer": answer, "validation": validation})
        TraceStore.record(run)

        # 显示结果
        print(f"\n🤖 回答：{answer}")
        if validation:
            print(f"📊 校验结果：质量分={validation.get('quality_score', 0)}, 长度={validation.get('length', 0)}")

        # 显示追踪信息
        print(f"\n📋 追踪详情：")
        print(f"   Run ID: {run.id}")
        print(f"   总耗时: {run.duration_ms:.0f}ms")
        print(f"   步骤数: {len(run.root_span.children)}")
        for i, span in enumerate(run.root_span.children, 1):
            icon = "✅" if span.status == "success" else "❌"
            print(f"   {icon} 步骤{i}: {span.name} ({span.duration_ms:.0f}ms)")
        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. @traceable 装饰器自动追踪函数调用")
    print("   2. 每个步骤记录输入、输出、耗时和状态")
    print("   3. 嵌套调用形成完整的调用链")
    print("   4. 真实 LangSmith 中可在 Web 界面查看追踪树")


# ============================================================
# 示例2: 自动追踪 - LangChain追踪
# ============================================================

def demo_langchain_tracing():
    """示例2：自动追踪 - LangChain追踪（用户输入问题，自动追踪LangChain调用）"""
    print("\n" + "="*60)
    print("示例2：自动追踪 - LangChain追踪")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - LangChain 组件（Prompt/LLM/Parser）自动支持追踪")
    print("   - 设置 LANGSMITH_TRACING=true 即可开启自动追踪")
    print("   - 每个组件的调用都会被记录为 Span")
    print("\n📊 应用场景：")
    print("   - 追踪 LangChain 链的完整执行过程")
    print("   - 分析每个组件的耗时和输出")
    print("   - 调试链中的问题")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    print("\n【交互式 LangChain 追踪演示】")
    print("提示：输入问题，系统自动追踪 LangChain 链的调用过程")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()
        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break
        if not question:
            print("请输入有效问题")
            continue

        # 创建追踪记录
        run = Run("LangChain问答链", "chain")
        run.start({"question": question})

        # 模拟 LangChain 自动追踪的三个步骤
        # 步骤1：Prompt 模板渲染
        prompt_span = Span("ChatPromptTemplate", "prompt", run.root_span.id)
        prompt_span.start({"template": "system: 你是助手\\nhuman: {question}", "variables": {"question": question}})
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个知识渊博的助手，用简洁准确的语言回答问题"),
            ("human", "{question}")
        ])
        messages = prompt.invoke({"question": question})
        prompt_span.end({"messages": str(messages)[:200]})
        time.sleep(0.01)  # 模拟渲染耗时

        # 步骤2：LLM 调用
        llm_span = Span("ChatOpenAI (GLM-4)", "llm", run.root_span.id)
        llm_span.start({"messages": str(messages)[:200]})
        llm_start = time.time()
        try:
            llm_response = model.invoke(messages)
            llm_elapsed = time.time() - llm_start
            llm_span.end({
                "response": str(llm_response.content)[:200],
                "response_metadata": {
                    "model_name": llm_response.response_metadata.get("model_name", "unknown"),
                    "token_usage": llm_response.response_metadata.get("token_usage", {}),
                }
            })
            print(f"  🤖 LLM 调用完成 ({llm_elapsed*1000:.0f}ms)")
        except Exception as e:
            llm_elapsed = time.time() - llm_start
            llm_span.end(error=str(e))
            llm_response = None
            print(f"  ❌ LLM 调用失败: {e}")

        # 步骤3：输出解析
        parser_span = Span("StrOutputParser", "parser", run.root_span.id)
        parser_span.start({"input": str(llm_response.content)[:100] if llm_response else "None"})
        parser = StrOutputParser()
        try:
            answer = parser.invoke(llm_response) if llm_response else ""
            parser_span.end({"output": answer[:200]})
        except Exception as e:
            parser_span.end(error=str(e))
            answer = ""

        # 结束追踪
        run.root_span.children = [prompt_span, llm_span, parser_span]
        run.end({"answer": answer})

        # 获取 token 使用信息
        token_info = {}
        if llm_response and hasattr(llm_response, 'response_metadata'):
            token_info = llm_response.response_metadata.get("token_usage", {})

        # 记录追踪
        TraceStore.record(run)

        # 显示结果
        print(f"\n🤖 回答：{answer}")

        # 显示追踪信息
        print(f"\n📋 LangChain 追踪详情：")
        print(f"   Run ID: {run.id}")
        print(f"   总耗时: {run.duration_ms:.0f}ms")
        print(f"   组件追踪：")
        for span in run.root_span.children:
            icon = "✅" if span.status == "success" else "❌"
            print(f"     {icon} {span.name} [{span.run_type}] ({span.duration_ms:.0f}ms)")

        # 显示 Token 信息
        if token_info:
            print(f"\n📊 Token 使用：")
            for k, v in token_info.items():
                print(f"     {k}: {v}")

        # 显示模式提示
        if not has_langsmith_key():
            print(f"\n💡 模拟模式：以上追踪信息在真实 LangSmith 中会自动记录")
            print(f"   配置 LANGSMITH_API_KEY 和 LANGSMITH_TRACING=true 可启用真实追踪")
        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. LangChain 组件自动支持追踪")
    print("   2. Prompt/LLM/Parser 每步都有详细记录")
    print("   3. 可查看每步的 Token 消耗和耗时")
    print("   4. 设置环境变量即可开启真实追踪")


# ============================================================
# 示例3: 链路可视化 - 调用链展示
# ============================================================

def demo_trace_visualization():
    """示例3：链路可视化 - 调用链展示（用户输入问题，展示完整的调用链路）"""
    print("\n" + "="*60)
    print("示例3：链路可视化 - 调用链展示")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - Run: 一次完整的追踪记录")
    print("   - Span: 追踪中的一个步骤")
    print("   - 调用链路：由多个 Span 组成的树形结构")
    print("   - 真实 LangSmith 提供 Web 界面可视化")
    print("\n📊 应用场景：")
    print("   - 查看完整调用链路")
    print("   - 定位耗时最长的步骤")
    print("   - 理解复杂系统的执行流程")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    print("\n【交互式链路可视化演示】")
    print("提示：输入问题，系统展示完整的调用链路树")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()
        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break
        if not question:
            print("请输入有效问题")
            continue

        # 创建追踪记录
        run = Run("RAG问答链", "chain")
        run.start({"question": question})

        # 步骤1：问题理解
        understand_span = Span("问题理解", "chain", run.root_span.id)
        understand_span.start({"question": question})
        understand_prompt = ChatPromptTemplate.from_messages([
            ("system", "分析用户问题，提取关键信息和意图。简洁输出。"),
            ("human", "{question}")
        ])
        understand_chain = understand_prompt | model | StrOutputParser()
        try:
            understanding = understand_chain.invoke({"question": question})
            understand_span.end({"understanding": understanding[:200]})
        except Exception as e:
            understand_span.end(error=str(e))
            understanding = question

        # 步骤2：知识检索（模拟）
        retrieve_span = Span("知识检索", "retriever", run.root_span.id)
        retrieve_span.start({"query": understanding[:100]})
        time.sleep(0.1)  # 模拟检索延迟
        context = f"[模拟检索结果] 关于'{question}'的相关文档内容..."
        retrieve_span.end({"context": context, "doc_count": 3})

        # 步骤2.1：文档评分（子步骤）
        score_span = Span("文档评分", "tool", retrieve_span.id)
        score_span.start({"documents": "3篇文档"})
        time.sleep(0.05)
        scores = {"doc1": 0.95, "doc2": 0.82, "doc3": 0.71}
        score_span.end({"scores": scores})
        retrieve_span.children = [score_span]

        # 步骤3：答案生成
        generate_span = Span("答案生成", "llm", run.root_span.id)
        generate_span.start({"question": question, "context": context})
        generate_prompt = ChatPromptTemplate.from_messages([
            ("system", "根据以下上下文回答问题，简洁准确\n上下文：{context}"),
            ("human", "{question}")
        ])
        generate_chain = generate_prompt | model | StrOutputParser()
        try:
            answer = generate_chain.invoke({"question": question, "context": context})
            generate_span.end({"answer": answer[:200]})
        except Exception as e:
            generate_span.end(error=str(e))
            answer = f"生成失败: {e}"

        # 步骤4：答案优化
        refine_span = Span("答案优化", "chain", run.root_span.id)
        refine_span.start({"answer": answer[:100]})
        refine_prompt = ChatPromptTemplate.from_messages([
            ("system", "优化以下答案，使其更加简洁流畅：\n{answer}"),
            ("human", "请优化")
        ])
        refine_chain = refine_prompt | model | StrOutputParser()
        try:
            refined = refine_chain.invoke({"answer": answer})
            refine_span.end({"refined_answer": refined[:200]})
            final_answer = refined
        except Exception as e:
            refine_span.end(error=str(e))
            final_answer = answer

        # 结束追踪
        run.root_span.children = [understand_span, retrieve_span, generate_span, refine_span]
        run.end({"final_answer": final_answer})
        TraceStore.record(run)

        # 显示结果
        print(f"\n🤖 最终回答：{final_answer}")

        # 显示调用链路树
        print(f"\n📊 调用链路可视化 [Run ID: {run.id}]：")
        print("="*60)
        root_dict = run.root_span.to_dict()
        print_span_tree(root_dict)
        print("="*60)

        # 显示耗时统计
        print(f"\n⏱️ 耗时统计：")
        total = run.duration_ms
        print(f"   总耗时: {total:.0f}ms")
        for span in run.root_span.children:
            pct = span.duration_ms / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"   {span.name:<12} {bar} {span.duration_ms:.0f}ms ({pct:.0f}%)")

        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 调用链路以树形结构展示，清晰直观")
    print("   2. 每个步骤记录输入输出和耗时")
    print("   3. 子步骤（如文档评分）嵌套在父步骤中")
    print("   4. 真实 LangSmith 提供 Web 界面交互式可视化")


# ============================================================
# 示例4: 错误追踪 - 错误定位
# ============================================================

def demo_error_tracing():
    """示例4：错误追踪 - 错误定位（用户输入问题，自动追踪错误位置）"""
    print("\n" + "="*60)
    print("示例4：错误追踪 - 错误定位")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 错误追踪：自动记录错误发生的位置和上下文")
    print("   - 错误链路：从错误位置回溯到根因")
    print("   - 真实 LangSmith 中错误会标红显示")
    print("\n📊 应用场景：")
    print("   - 定位生产环境中的错误")
    print("   - 分析错误原因和影响范围")
    print("   - 回溯错误链路到根因")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    # 模拟可能出错的函数
    def risky_analysis(text: str, mode: str = "safe") -> dict:
        """模拟可能出错的分析函数"""
        if mode == "error" and len(text) > 10:
            raise ValueError("输入文本过长，超出处理限制（最大10字符）")
        if mode == "timeout" and len(text) > 5:
            raise TimeoutError("处理超时，请缩短输入")
        return {"result": f"分析完成: {text[:20]}", "confidence": 0.95}

    print("\n【交互式错误追踪演示】")
    print("提示：输入问题，系统模拟可能出错的情况并追踪错误")
    print("输入 '退出' 结束\n")

    while True:
        print("\n选择测试模式：")
        print("  1. 正常模式（不会出错）")
        print("  2. 错误模式（模拟输入过长错误）")
        print("  3. 超时模式（模拟处理超时错误）")
        print("  4. LLM错误模式（使用无效提示触发错误）")
        print("\n  0. 退出")

        choice = input("\n请选择 (0-4): ").strip()
        if choice == "0":
            print("结束演示")
            break
        if choice not in ["1", "2", "3", "4"]:
            print("❌ 无效选项")
            continue

        question = input("输入问题：").strip()
        if not question:
            print("请输入有效问题")
            continue

        # 创建追踪记录
        run = Run("错误追踪链", "chain")
        run.start({"question": question, "mode": choice})

        # 步骤1：输入验证（总是成功）
        validate_span = Span("输入验证", "tool", run.root_span.id)
        validate_span.start({"input": question})
        validate_span.end({"valid": True, "length": len(question)})
        print(f"  ✅ 步骤1：输入验证通过")

        # 步骤2：分析处理（可能出错）
        analyze_span = Span("分析处理", "chain", run.root_span.id)
        analyze_span.start({"input": question, "mode": choice})

        if choice in ["2", "3"]:
            mode = "error" if choice == "2" else "timeout"
            try:
                result = risky_analysis(question, mode)
                analyze_span.end({"result": result})
                print(f"  ✅ 步骤2：分析处理完成")
            except Exception as e:
                analyze_span.end(error=str(e))
                print(f"  ❌ 步骤2：分析处理失败 - {e}")

                # 步骤3：错误处理
                error_span = Span("错误处理", "tool", run.root_span.id)
                error_span.start({"error": str(e)})

                # 尝试降级处理
                fallback_span = Span("降级处理", "chain", error_span.id)
                fallback_span.start({"original_error": str(e)})
                try:
                    # 使用 LLM 进行降级处理
                    fallback_prompt = ChatPromptTemplate.from_messages([
                        ("system", "之前的处理失败了，请用简单的方式回答问题"),
                        ("human", "{question}")
                    ])
                    fallback_chain = fallback_prompt | model | StrOutputParser()
                    fallback_result = fallback_chain.invoke({"question": question})
                    fallback_span.end({"result": fallback_result[:200]})
                    error_span.end({"fallback_result": "成功"})
                    print(f"  ✅ 步骤3：降级处理成功")
                except Exception as e2:
                    fallback_span.end(error=str(e2))
                    error_span.end(error=f"降级也失败: {e2}")
                    print(f"  ❌ 步骤3：降级处理也失败 - {e2}")

                error_span.children = [fallback_span]
                run.root_span.children = [validate_span, analyze_span, error_span]
                run.end(error=f"分析失败: {e}")
                TraceStore.record(run)

                # 显示错误追踪信息
                print(f"\n📋 错误追踪详情：")
                print("="*60)
                root_dict = run.root_span.to_dict()
                print_span_tree(root_dict)
                print("="*60)

                print(f"\n🔍 错误分析：")
                print(f"   错误位置：分析处理步骤")
                print(f"   错误类型：{type(e).__name__}")
                print(f"   错误信息：{e}")
                print(f"   降级处理：{'成功' if fallback_span.status == 'success' else '失败'}")
                print("-"*60)
                continue

        elif choice == "4":
            # LLM 错误模式
            try:
                # 使用无效的提示触发错误
                bad_prompt = ChatPromptTemplate.from_messages([
                    ("system", None),  # 故意设置无效值
                    ("human", "{question}")
                ])
                bad_chain = bad_prompt | model | StrOutputParser()
                result = bad_chain.invoke({"question": question})
                analyze_span.end({"result": result[:200]})
            except Exception as e:
                analyze_span.end(error=str(e))
                print(f"  ❌ 步骤2：LLM 调用失败 - {e}")

                run.root_span.children = [validate_span, analyze_span]
                run.end(error=f"LLM错误: {e}")
                TraceStore.record(run)

                print(f"\n📋 错误追踪详情：")
                print("="*60)
                root_dict = run.root_span.to_dict()
                print_span_tree(root_dict)
                print("="*60)

                print(f"\n🔍 错误分析：")
                print(f"   错误位置：LLM 调用步骤")
                print(f"   错误类型：{type(e).__name__}")
                print(f"   错误信息：{e}")
                print("-"*60)
                continue
        else:
            # 正常模式
            try:
                result = risky_analysis(question, "safe")
                analyze_span.end({"result": result})
                print(f"  ✅ 步骤2：分析处理完成")
            except Exception as e:
                analyze_span.end(error=str(e))
                print(f"  ❌ 步骤2：分析处理失败 - {e}")

        # 步骤3：结果格式化（正常流程）
        format_span = Span("结果格式化", "tool", run.root_span.id)
        format_span.start({"input": "分析结果"})
        try:
            formatted = f"✅ 分析完成：{question}"
            format_span.end({"formatted": formatted})
            print(f"  ✅ 步骤3：结果格式化完成")
        except Exception as e:
            format_span.end(error=str(e))

        run.root_span.children = [validate_span, analyze_span, format_span]
        run.end({"result": "成功"})
        TraceStore.record(run)

        print(f"\n🤖 处理结果：成功")
        print(f"\n📋 追踪详情：")
        print("="*60)
        root_dict = run.root_span.to_dict()
        print_span_tree(root_dict)
        print("="*60)
        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 错误追踪自动记录错误位置和上下文")
    print("   2. 可回溯错误链路到根因")
    print("   3. 降级处理也会被追踪记录")
    print("   4. 真实 LangSmith 中错误会标红显示")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangSmith 追踪（Tracing）- 实战案例")
    print("="*60)
    print("\n本示例演示 LangSmith 追踪功能的核心概念和使用方法")

    mode = "真实模式" if has_langsmith_key() else "模拟模式"
    print(f"\n当前模式：{mode}")
    if not has_langsmith_key():
        print("提示：配置 LANGSMITH_API_KEY 可连接真实 LangSmith 服务")

    print("\n核心概念：")
    print("  • @traceable: 装饰器，自动追踪函数调用")
    print("  • Tracing: 链路追踪，记录每次调用的完整链路")
    print("  • Run: 一次完整的追踪记录")
    print("  • Span: 追踪中的一个步骤")

    print("\n应用场景：")
    print("  • 函数追踪、LangChain追踪、链路可视化、错误追踪")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. @traceable装饰器 - 函数追踪")
        print("  2. 自动追踪 - LangChain追踪")
        print("  3. 链路可视化 - 调用链展示")
        print("  4. 错误追踪 - 错误定位")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_traceable_decorator()
        elif choice == "2":
            demo_langchain_tracing()
        elif choice == "3":
            demo_trace_visualization()
        elif choice == "4":
            demo_error_tracing()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
