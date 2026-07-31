"""
LangChain 中间件基础 - 实战交互式案例
========================================

本示例演示 LangChain 中间件（回调处理器）的基础用法

核心概念：
- BaseCallbackHandler：LangChain 的回调基类
- on_llm_start / on_llm_end：模型调用前后的钩子
- 中间件模式：在不修改业务逻辑的前提下添加横切功能

应用场景：
- 日志记录：追踪每次调用的输入输出
- Token 统计：监控 Token 使用量和成本
- 性能监控：分析调用耗时和瓶颈
- 自定义回调：实现灵活的事件处理逻辑
"""

import os
import sys
import time
from typing import Any, Dict, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_loader import get_default_llm


# ============================================================
# 1. 日志中间件 - 记录每次调用的输入输出
# ============================================================

class LoggingMiddleware(BaseCallbackHandler):
    """日志中间件：记录每次 LLM 调用的输入和输出"""

    def __init__(self):
        self.call_logs: List[Dict[str, Any]] = []
        self._current_prompts: List[str] = []

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """LLM 调用开始时记录输入"""
        self._current_prompts = prompts
        print(f"\n  [日志中间件] 开始调用 - 输入: {prompts[0][:80]}..." if len(prompts[0]) > 80 else f"\n  [日志中间件] 开始调用 - 输入: {prompts[0]}")

    def on_llm_end(self, response, **kwargs: Any) -> None:
        """LLM 调用结束时记录输出"""
        output_text = ""
        if response.generations and response.generations[0]:
            output_text = response.generations[0][0].text if hasattr(response.generations[0][0], 'text') else str(response.generations[0][0])

        log_entry = {
            "input": self._current_prompts[0] if self._current_prompts else "N/A",
            "output": output_text[:200] if output_text else "N/A",
        }
        self.call_logs.append(log_entry)

        preview = output_text[:80] if len(output_text) > 80 else output_text
        print(f"  [日志中间件] 调用完成 - 输出: {preview}...")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """LLM 调用出错时记录"""
        print(f"  [日志中间件] 调用出错 - 错误: {error}")

    def print_summary(self):
        """打印日志摘要"""
        print(f"\n  日志摘要：共记录 {len(self.call_logs)} 次调用")
        for i, log in enumerate(self.call_logs, 1):
            input_preview = log['input'][:50] if len(log['input']) > 50 else log['input']
            output_preview = log['output'][:50] if len(log['output']) > 50 else log['output']
            print(f"  #{i} 输入: {input_preview}... -> 输出: {output_preview}...")


def demo_logging_middleware():
    """示例1：日志中间件 - 记录每次调用的输入输出"""
    print("\n" + "="*60)
    print("示例1：日志中间件 - 记录每次调用的输入输出")
    print("="*60)
    print("\n 实战要点：")
    print("   - 继承 BaseCallbackHandler 实现自定义日志逻辑")
    print("   - on_llm_start 捕获输入，on_llm_end 捕获输出")
    print("   - 通过 callbacks 参数将中间件注入链中")

    logging_middleware = LoggingMiddleware()
    model = get_default_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个简洁的助手，用一句话回答问题"),
        ("human", "{question}")
    ])

    chain = prompt | model | StrOutputParser()

    print("\n【交互式日志记录】")
    print("提示：输入问题，日志中间件会自动记录输入输出")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()

        if question.lower() in ['退出', 'exit', 'quit']:
            break

        if not question:
            print("请输入有效内容")
            continue

        print("-" * 40)
        response = chain.invoke(
            {"question": question},
            config={"callbacks": [logging_middleware]}
        )
        print(f"\n  AI 回答：{response}")
        print("-" * 40)

    # 打印日志摘要
    logging_middleware.print_summary()

    print("\n 实战要点总结：")
    print("   1. BaseCallbackHandler 是所有中间件的基类")
    print("   2. on_llm_start / on_llm_end 实现调用前后的拦截")
    print("   3. 通过 config={'callbacks': [...]} 注入中间件，无需修改业务代码")


# ============================================================
# 2. Token 追踪中间件 - 统计每次调用的 Token 使用量
# ============================================================

class TokenTrackerMiddleware(BaseCallbackHandler):
    """Token 追踪中间件：统计每次调用的 Token 使用量"""

    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_records: List[Dict[str, Any]] = []

    def on_llm_end(self, response, **kwargs: Any) -> None:
        """LLM 调用结束时统计 Token"""
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            t = usage.get("total_tokens", 0)
            p = usage.get("prompt_tokens", 0)
            c = usage.get("completion_tokens", 0)

            self.total_tokens += t
            self.prompt_tokens += p
            self.completion_tokens += c

            self.call_records.append({
                "total": t,
                "prompt": p,
                "completion": c
            })

            print(f"  [Token追踪] 本次: 总{t} (输入{p} + 输出{c})")

    def print_summary(self):
        """打印 Token 使用汇总"""
        print(f"\n  Token 使用汇总：")
        print(f"  {'='*40}")
        print(f"  调用次数：{len(self.call_records)}")
        print(f"  总 Token：{self.total_tokens}")
        print(f"  输入 Token：{self.prompt_tokens}")
        print(f"  输出 Token：{self.completion_tokens}")
        if self.call_records:
            avg_total = self.total_tokens / len(self.call_records)
            print(f"  平均每次：{avg_total:.1f} Token")
        print(f"  {'='*40}")


def demo_token_tracker():
    """示例2：Token 追踪 - 统计每次调用的 Token 使用量"""
    print("\n" + "="*60)
    print("示例2：Token 追踪 - 统计每次调用的 Token 使用量")
    print("="*60)
    print("\n 实战要点：")
    print("   - on_llm_end 的 response.llm_output 包含 token_usage")
    print("   - 追踪 Token 有助于成本控制和预算管理")
    print("   - 可以按调用、按时间段汇总 Token 消耗")

    token_tracker = TokenTrackerMiddleware()
    model = get_default_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个助手，请简洁回答"),
        ("human", "{question}")
    ])

    chain = prompt | model | StrOutputParser()

    print("\n【交互式 Token 追踪】")
    print("提示：输入问题，追踪每次调用的 Token 消耗")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()

        if question.lower() in ['退出', 'exit', 'quit']:
            break

        if not question:
            print("请输入有效内容")
            continue

        print("-" * 40)
        response = chain.invoke(
            {"question": question},
            config={"callbacks": [token_tracker]}
        )
        print(f"  AI 回答：{response}")
        print("-" * 40)

    token_tracker.print_summary()

    print("\n 实战要点总结：")
    print("   1. token_usage 在 llm_output 中返回（部分模型可能不返回）")
    print("   2. 追踪 Token 消耗对成本管理至关重要")
    print("   3. 可按需求扩展为按用户、按功能维度的 Token 统计")


# ============================================================
# 3. 性能监控中间件 - 记录每次调用的耗时
# ============================================================

class PerformanceMonitorMiddleware(BaseCallbackHandler):
    """性能监控中间件：记录每次调用的耗时"""

    def __init__(self):
        self._start_times: Dict[str, float] = {}
        self.call_metrics: List[Dict[str, Any]] = []

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """记录调用开始时间"""
        run_id = str(kwargs.get("run_id", id(prompts)))
        self._start_times[run_id] = time.time()
        print(f"  [性能监控] 调用开始...")

    def on_llm_end(self, response, **kwargs: Any) -> None:
        """记录调用结束时间并计算耗时"""
        run_id = str(kwargs.get("run_id", "unknown"))
        start_time = self._start_times.pop(run_id, None)

        if start_time is not None:
            elapsed = time.time() - start_time
            metric = {"elapsed_seconds": round(elapsed, 3)}
            self.call_metrics.append(metric)
            print(f"  [性能监控] 调用完成 - 耗时: {elapsed:.3f}秒")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """调用出错时清理计时"""
        run_id = str(kwargs.get("run_id", "unknown"))
        self._start_times.pop(run_id, None)
        print(f"  [性能监控] 调用失败 - 错误: {error}")

    def print_summary(self):
        """打印性能监控汇总"""
        if not self.call_metrics:
            print("\n  暂无性能数据")
            return

        times = [m["elapsed_seconds"] for m in self.call_metrics]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\n  性能监控汇总：")
        print(f"  {'='*40}")
        print(f"  调用次数：{len(self.call_metrics)}")
        print(f"  平均耗时：{avg_time:.3f}秒")
        print(f"  最快耗时：{min_time:.3f}秒")
        print(f"  最慢耗时：{max_time:.3f}秒")
        print(f"  总耗时：{sum(times):.3f}秒")
        print(f"  {'='*40}")


def demo_performance_monitor():
    """示例3：性能监控 - 记录每次调用的耗时"""
    print("\n" + "="*60)
    print("示例3：性能监控 - 记录每次调用的耗时")
    print("="*60)
    print("\n 实战要点：")
    print("   - on_llm_start 记录开始时间，on_llm_end 记录结束时间")
    print("   - run_id 用于关联同一次调用的开始和结束事件")
    print("   - 性能数据可用于优化提示词和选择模型")

    perf_monitor = PerformanceMonitorMiddleware()
    model = get_default_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个助手，请简洁回答"),
        ("human", "{question}")
    ])

    chain = prompt | model | StrOutputParser()

    print("\n【交互式性能监控】")
    print("提示：输入问题，监控每次调用的耗时")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()

        if question.lower() in ['退出', 'exit', 'quit']:
            break

        if not question:
            print("请输入有效内容")
            continue

        print("-" * 40)
        response = chain.invoke(
            {"question": question},
            config={"callbacks": [perf_monitor]}
        )
        print(f"  AI 回答：{response}")
        print("-" * 40)

    perf_monitor.print_summary()

    print("\n 实战要点总结：")
    print("   1. on_llm_start / on_llm_end 配合 run_id 精确计时")
    print("   2. 性能数据帮助识别慢查询和优化方向")
    print("   3. 可扩展为设置耗时阈值、超时告警等功能")


# ============================================================
# 4. 自定义回调中间件 - 实现自定义的回调处理逻辑
# ============================================================

class CustomCallbackMiddleware(BaseCallbackHandler):
    """自定义回调中间件：实现灵活的事件处理逻辑"""

    def __init__(self, on_start=None, on_end=None, on_error=None):
        """
        Args:
            on_start: 调用开始时的自定义处理函数
            on_end: 调用结束时的自定义处理函数
            on_error: 调用出错时的自定义处理函数
        """
        self.on_start = on_start
        self.on_end = on_end
        self.on_error = on_error
        self._current_prompts: List[str] = []

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """调用开始时执行自定义逻辑"""
        self._current_prompts = prompts
        if self.on_start:
            self.on_start(prompts)

    def on_llm_end(self, response, **kwargs: Any) -> None:
        """调用结束时执行自定义逻辑"""
        output_text = ""
        if response.generations and response.generations[0]:
            output_text = response.generations[0][0].text if hasattr(response.generations[0][0], 'text') else str(response.generations[0][0])

        if self.on_end:
            self.on_end(self._current_prompts, output_text)

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """调用出错时执行自定义逻辑"""
        if self.on_error:
            self.on_error(error)


def demo_custom_callback():
    """示例4：自定义回调 - 实现自定义的回调处理逻辑"""
    print("\n" + "="*60)
    print("示例4：自定义回调 - 实现自定义的回调处理逻辑")
    print("="*60)
    print("\n 实战要点：")
    print("   - 通过构造函数注入自定义处理函数")
    print("   - 实现策略模式，灵活切换不同的回调行为")
    print("   - 适合需要动态改变监控逻辑的场景")

    # 定义不同的回调策略
    def on_start_echo(prompts: List[str]):
        """策略1：简单回显输入"""
        print(f"  [自定义回调] 收到输入: {prompts[0][:60]}...")

    def on_end_count(input_prompts: List[str], output: str):
        """策略2：统计输入输出字符数"""
        input_len = len(input_prompts[0]) if input_prompts else 0
        output_len = len(output) if output else 0
        print(f"  [自定义回调] 输入 {input_len} 字符 -> 输出 {output_len} 字符")

    def on_error_log(error: Exception):
        """策略3：错误记录"""
        print(f"  [自定义回调] 发生错误: {type(error).__name__}: {error}")

    # 创建自定义回调中间件
    custom_callback = CustomCallbackMiddleware(
        on_start=on_start_echo,
        on_end=on_end_count,
        on_error=on_error_log
    )

    model = get_default_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个助手，请简洁回答"),
        ("human", "{question}")
    ])

    chain = prompt | model | StrOutputParser()

    print("\n【交互式自定义回调】")
    print("提示：输入问题，自定义回调会执行注入的处理逻辑")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()

        if question.lower() in ['退出', 'exit', 'quit']:
            break

        if not question:
            print("请输入有效内容")
            continue

        print("-" * 40)
        response = chain.invoke(
            {"question": question},
            config={"callbacks": [custom_callback]}
        )
        print(f"  AI 回答：{response}")
        print("-" * 40)

    print("\n 实战要点总结：")
    print("   1. CustomCallbackMiddleware 通过构造函数注入处理函数")
    print("   2. 不同场景可注入不同的 on_start / on_end / on_error")
    print("   3. 策略模式让回调逻辑可动态切换，无需修改中间件代码")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain 中间件基础 - 实战案例")
    print("="*60)
    print("\n本示例演示 LangChain 中间件（回调处理器）的基础用法")
    print("\n核心概念：")
    print("  BaseCallbackHandler：回调基类")
    print("  on_llm_start / on_llm_end：调用前后钩子")
    print("  中间件模式：横切功能注入")
    print("\n应用场景：")
    print("  日志记录、Token 追踪、性能监控、自定义回调")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 日志中间件 - 记录每次调用的输入输出")
        print("  2. Token 追踪 - 统计每次调用的 Token 使用量")
        print("  3. 性能监控 - 记录每次调用的耗时")
        print("  4. 自定义回调 - 实现自定义的回调处理逻辑")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_logging_middleware()
        elif choice == "2":
            demo_token_tracker()
        elif choice == "3":
            demo_performance_monitor()
        elif choice == "4":
            demo_custom_callback()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print(" 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
