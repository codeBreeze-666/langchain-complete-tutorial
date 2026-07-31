"""
LangSmith 调试（Debugging）- 实战交互式案例
==============================================

本示例演示 LangSmith 调试功能的核心概念和使用方法

核心概念：
- Run回放（Run Replay）：复现之前的追踪记录，重新执行相同的调用
- 中间变量（Intermediate Variables）：查看每一步的输入输出，定位问题
- 对比实验（A/B Testing）：对比不同参数的输出差异，优化效果
- 性能分析（Performance Analysis）：分析每一步的耗时和Token消耗

应用场景：
- 运行回放：复现生产环境中的问题
- 中间变量查看：调试链中的每一步
- 对比实验：A/B测试不同参数的效果
- 性能分析：优化链的性能和成本
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from typing import Optional

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
# 调试数据存储（模拟 LangSmith 后端）
# ============================================================

class DebugRun:
    """模拟一次可调试的运行记录"""

    def __init__(self, name: str, inputs: dict):
        self.id = f"run-{uuid.uuid4().hex[:8]}"
        self.name = name
        self.inputs = inputs
        self.steps: list = []  # 每步的输入输出
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status: str = "pending"
        self.error: Optional[str] = None
        self.timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.token_usage: dict = {}  # Token 使用记录

    def add_step(self, step_name: str, step_type: str, inputs: dict, outputs: dict,
                 duration_ms: float, error: str = None, tokens: dict = None):
        """添加一个步骤"""
        self.steps.append({
            "name": step_name,
            "type": step_type,
            "inputs": inputs,
            "outputs": outputs,
            "duration_ms": round(duration_ms, 2),
            "error": error,
            "tokens": tokens or {},
        })

    @property
    def duration_ms(self) -> float:
        """总耗时"""
        if self.start_time and self.end_time:
            return round((self.end_time - self.start_time) * 1000, 2)
        return 0

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "inputs": self.inputs,
            "steps": self.steps,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "timestamp": self.timestamp,
            "token_usage": self.token_usage,
        }


class DebugStore:
    """调试数据存储"""

    _runs: list = []

    @classmethod
    def save(cls, run: DebugRun) -> str:
        """保存运行记录"""
        cls._runs.append(run)
        return run.id

    @classmethod
    def get_all(cls) -> list:
        """获取所有运行记录"""
        return cls._runs

    @classmethod
    def get_by_id(cls, run_id: str) -> Optional[DebugRun]:
        """按 ID 获取运行记录"""
        for run in cls._runs:
            if run.id == run_id:
                return run
        return None

    @classmethod
    def clear(cls):
        """清空记录"""
        cls._runs = []


# ============================================================
# 示例1: 运行回放 - 复现问题
# ============================================================

def demo_run_replay():
    """示例1：运行回放 - 复现问题（回放之前的追踪记录，定位问题）"""
    print("\n" + "="*60)
    print("示例1：运行回放 - 复现问题")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - Run回放：复现之前的追踪记录")
    print("   - 使用相同的输入重新执行链")
    print("   - 对比回放结果和原始结果")
    print("\n📊 应用场景：")
    print("   - 复现生产环境中的问题")
    print("   - 验证修复后是否正常")
    print("   - 回归测试")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    # 先执行一次运行，保存记录
    print("\n【步骤1：执行原始运行】")
    question = input("输入问题：").strip()
    if not question:
        print("❌ 问题不能为空")
        return

    # 创建并执行原始运行
    original_run = DebugRun("问答链", {"question": question})
    original_run.start_time = time.time()

    # 步骤1：问题理解
    step1_start = time.time()
    prompt1 = ChatPromptTemplate.from_messages([
        ("system", "分析用户问题，提取关键信息。简洁输出。"),
        ("human", "{question}")
    ])
    chain1 = prompt1 | model | StrOutputParser()
    understanding = chain1.invoke({"question": question})
    step1_ms = (time.time() - step1_start) * 1000
    original_run.add_step("问题理解", "chain", {"question": question}, {"understanding": understanding}, step1_ms)

    # 步骤2：答案生成
    step2_start = time.time()
    prompt2 = ChatPromptTemplate.from_messages([
        ("system", "根据理解的问题回答，简洁准确。"),
        ("human", "{understanding}")
    ])
    chain2 = prompt2 | model | StrOutputParser()
    answer = chain2.invoke({"understanding": understanding})
    step2_ms = (time.time() - step2_start) * 1000
    original_run.add_step("答案生成", "chain", {"understanding": understanding}, {"answer": answer}, step2_ms)

    original_run.end_time = time.time()
    original_run.status = "success"
    DebugStore.save(original_run)

    print(f"\n🤖 原始回答：{answer}")
    print(f"📋 原始运行记录已保存 [ID: {original_run.id}]")

    # 回放运行
    print("\n【步骤2：回放运行】")
    print("使用相同的输入重新执行链...\n")

    replay_run = DebugRun("问答链(回放)", {"question": question})
    replay_run.start_time = time.time()

    # 回放步骤1
    step1_start = time.time()
    understanding_replay = chain1.invoke({"question": question})
    step1_ms_replay = (time.time() - step1_start) * 1000
    replay_run.add_step("问题理解", "chain", {"question": question}, {"understanding": understanding_replay}, step1_ms_replay)

    # 回放步骤2
    step2_start = time.time()
    answer_replay = chain2.invoke({"understanding": understanding_replay})
    step2_ms_replay = (time.time() - step2_start) * 1000
    replay_run.add_step("答案生成", "chain", {"understanding": understanding_replay}, {"answer": answer_replay}, step2_ms_replay)

    replay_run.end_time = time.time()
    replay_run.status = "success"
    DebugStore.save(replay_run)

    print(f"🤖 回放回答：{answer_replay}")

    # 对比原始和回放
    print(f"\n📊 原始 vs 回放 对比：")
    print("="*60)
    print(f"{'指标':<15} {'原始运行':<20} {'回放运行':<20}")
    print("-"*60)
    print(f"{'Run ID':<15} {original_run.id:<20} {replay_run.id:<20}")
    print(f"{'总耗时':<15} {original_run.duration_ms:.0f}ms{'':<13} {replay_run.duration_ms:.0f}ms")

    for i, (orig_step, replay_step) in enumerate(zip(original_run.steps, replay_run.steps)):
        print(f"\n  步骤{i+1}: {orig_step['name']}")
        print(f"  {'耗时':<13} {orig_step['duration_ms']:.0f}ms{'':<15} {replay_step['duration_ms']:.0f}ms")
        print(f"  {'输出长度':<13} {len(str(orig_step['outputs']))}字符{'':<14} {len(str(replay_step['outputs']))}字符")

    print("="*60)

    # 查看历史记录
    all_runs = DebugStore.get_all()
    print(f"\n📚 历史运行记录（共 {len(all_runs)} 条）：")
    for run in all_runs:
        icon = "✅" if run.status == "success" else "❌"
        print(f"   {icon} [{run.id}] {run.name} - {run.duration_ms:.0f}ms ({run.timestamp})")

    print("\n✅ 实战要点总结：")
    print("   1. 运行回放使用相同的输入重新执行链")
    print("   2. 可对比原始和回放的结果差异")
    print("   3. 真实 LangSmith 中可一键回放历史运行")
    print("   4. 适用于复现和调试生产环境问题")


# ============================================================
# 示例2: 中间变量查看 - 调试
# ============================================================

def demo_intermediate_variables():
    """示例2：中间变量查看 - 调试（查看每一步的输入输出）"""
    print("\n" + "="*60)
    print("示例2：中间变量查看 - 调试")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 中间变量：链中每一步的输入和输出")
    print("   - 查看中间变量可以定位问题在哪一步")
    print("   - 真实 LangSmith 在 Web 界面展示每步详情")
    print("\n📊 应用场景：")
    print("   - 调试链中某一步的输出异常")
    print("   - 检查数据在步骤间的传递")
    print("   - 优化特定步骤的提示词")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    print("\n【交互式中间变量查看演示】")
    print("提示：输入问题，系统展示每一步的输入输出")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()
        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break
        if not question:
            print("请输入有效问题")
            continue

        # 创建运行记录
        run = DebugRun("多步问答链", {"question": question})
        run.start_time = time.time()

        # 步骤1：问题分类
        step1_start = time.time()
        prompt1 = ChatPromptTemplate.from_messages([
            ("system", "将问题分类为以下之一：技术/生活/学术/其他。只输出分类结果。"),
            ("human", "{question}")
        ])
        chain1 = prompt1 | model | StrOutputParser()
        category = chain1.invoke({"question": question})
        step1_ms = (time.time() - step1_start) * 1000
        run.add_step("问题分类", "classifier", {"question": question}, {"category": category}, step1_ms)

        # 步骤2：根据分类选择策略
        strategy_map = {
            "技术": "用专业的技术语言回答，包含技术细节",
            "生活": "用通俗易懂的语言回答，贴近生活",
            "学术": "用严谨的学术语言回答，引用相关理论",
        }
        strategy = strategy_map.get(category, "用简洁的语言回答")
        step2_start = time.time()
        run.add_step("策略选择", "router", {"category": category}, {"strategy": strategy}, 0.1)

        # 步骤3：生成答案
        step3_start = time.time()
        prompt3 = ChatPromptTemplate.from_messages([
            ("system", "{strategy}"),
            ("human", "{question}")
        ])
        chain3 = prompt3 | model | StrOutputParser()
        answer = chain3.invoke({"question": question, "strategy": strategy})
        step3_ms = (time.time() - step3_start) * 1000
        run.add_step("答案生成", "llm", {"question": question, "strategy": strategy}, {"answer": answer}, step3_ms)

        # 步骤4：答案校验
        step4_start = time.time()
        is_valid = len(answer) > 10 and category in answer or len(answer) > 20
        step4_ms = (time.time() - step4_start) * 1000
        run.add_step("答案校验", "validator", {"answer": answer[:100]}, {"is_valid": is_valid}, step4_ms)

        run.end_time = time.time()
        run.status = "success"
        DebugStore.save(run)

        # 显示最终结果
        print(f"\n🤖 最终回答：{answer}")

        # 显示每一步的中间变量
        print(f"\n📊 中间变量详情 [Run ID: {run.id}]：")
        print("="*60)
        for i, step in enumerate(run.steps, 1):
            print(f"\n  步骤{i}: {step['name']} [{step['type']}] ({step['duration_ms']:.0f}ms)")
            print(f"  {'─'*50}")
            print(f"  📥 输入：")
            for k, v in step["inputs"].items():
                val_str = str(v)[:100]
                print(f"     {k}: {val_str}")
            print(f"  📤 输出：")
            for k, v in step["outputs"].items():
                val_str = str(v)[:100]
                print(f"     {k}: {val_str}")
            if step.get("error"):
                print(f"  ⚠️ 错误: {step['error']}")
        print("="*60)

        # 数据流可视化
        print(f"\n🔄 数据流可视化：")
        flow = "问题"
        for step in run.steps:
            flow += f" → [{step['name']}]"
        print(f"   {flow}")
        print(f"   {'─'*50}")
        for i, step in enumerate(run.steps, 1):
            output_keys = list(step["outputs"].keys())
            print(f"   步骤{i}输出: {', '.join(output_keys)} → 传递给步骤{i+1}" if i < len(run.steps) else f"   步骤{i}输出: {', '.join(output_keys)} (最终结果)")

        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 中间变量显示每一步的输入输出")
    print("   2. 可定位问题出在哪一步")
    print("   3. 数据流可视化展示数据传递路径")
    print("   4. 真实 LangSmith 在 Web 界面交互式查看")


# ============================================================
# 示例3: 对比实验 - A/B测试
# ============================================================

def demo_ab_testing():
    """示例3：对比实验 - A/B测试（对比不同参数的输出差异）"""
    print("\n" + "="*60)
    print("示例3：对比实验 - A/B测试")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 对比实验：用不同参数运行同一链，对比结果")
    print("   - A/B测试：对比两个版本的输出差异")
    print("   - 真实 LangSmith 支持自动化对比实验")
    print("\n📊 应用场景：")
    print("   - 对比不同提示词的效果")
    print("   - 对比不同模型参数的输出")
    print("   - 对比不同模型的性能")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    print("\n【交互式 A/B 测试演示】")
    print("提示：输入问题，系统用两种不同的提示词策略对比输出")
    print("输入 '退出' 结束\n")

    # 定义两种策略
    strategy_a = {
        "name": "策略A：简洁模式",
        "system_prompt": "你是一个助手，用最简洁的语言回答问题，不超过50字。"
    }
    strategy_b = {
        "name": "策略B：详细模式",
        "system_prompt": "你是一个助手，用详细的语言回答问题，包含解释和示例。"
    }

    while True:
        question = input("你的问题：").strip()
        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break
        if not question:
            print("请输入有效问题")
            continue

        # 执行策略A
        run_a = DebugRun("A/B测试-策略A", {"question": question, "strategy": strategy_a["name"]})
        run_a.start_time = time.time()
        prompt_a = ChatPromptTemplate.from_messages([
            ("system", strategy_a["system_prompt"]),
            ("human", "{question}")
        ])
        chain_a = prompt_a | model | StrOutputParser()
        start_a = time.time()
        try:
            answer_a = chain_a.invoke({"question": question})
            duration_a = (time.time() - start_a) * 1000
            run_a.add_step("答案生成", "llm", {"question": question}, {"answer": answer_a}, duration_a)
            run_a.status = "success"
        except Exception as e:
            answer_a = f"[失败: {e}]"
            duration_a = (time.time() - start_a) * 1000
            run_a.add_step("答案生成", "llm", {"question": question}, {}, duration_a, error=str(e))
            run_a.status = "error"
        run_a.end_time = time.time()
        DebugStore.save(run_a)

        # 执行策略B
        run_b = DebugRun("A/B测试-策略B", {"question": question, "strategy": strategy_b["name"]})
        run_b.start_time = time.time()
        prompt_b = ChatPromptTemplate.from_messages([
            ("system", strategy_b["system_prompt"]),
            ("human", "{question}")
        ])
        chain_b = prompt_b | model | StrOutputParser()
        start_b = time.time()
        try:
            answer_b = chain_b.invoke({"question": question})
            duration_b = (time.time() - start_b) * 1000
            run_b.add_step("答案生成", "llm", {"question": question}, {"answer": answer_b}, duration_b)
            run_b.status = "success"
        except Exception as e:
            answer_b = f"[失败: {e}]"
            duration_b = (time.time() - start_b) * 1000
            run_b.add_step("答案生成", "llm", {"question": question}, {}, duration_b, error=str(e))
            run_b.status = "error"
        run_b.end_time = time.time()
        DebugStore.save(run_b)

        # 对比结果
        print(f"\n📊 A/B 测试对比：")
        print("="*60)
        print(f"  问题：{question}")
        print("="*60)

        print(f"\n  🅰️ {strategy_a['name']}")
        print(f"  {'─'*50}")
        print(f"  回答：{answer_a}")
        print(f"  耗时：{duration_a:.0f}ms")
        print(f"  字数：{len(answer_a)}")

        print(f"\n  🅱️ {strategy_b['name']}")
        print(f"  {'─'*50}")
        print(f"  回答：{answer_b}")
        print(f"  耗时：{duration_b:.0f}ms")
        print(f"  字数：{len(answer_b)}")

        # 差异分析
        print(f"\n  📈 差异分析：")
        print(f"  {'─'*50}")
        len_diff = len(answer_b) - len(answer_a)
        time_diff = duration_b - duration_a
        print(f"  字数差异：策略B比策略A多 {len_diff} 字 ({'+' if len_diff > 0 else ''}{len_diff})")
        print(f"  耗时差异：策略B比策略A多 {time_diff:.0f}ms ({'+' if time_diff > 0 else ''}{time_diff:.0f}ms)")
        print(f"  字数比：{len(answer_a)} : {len(answer_b)} = 1 : {len(answer_b)/max(len(answer_a),1):.1f}")

        # 质量评估（简单模拟）
        quality_a = min(100, len(answer_a) * 2) if len(answer_a) > 10 else 30
        quality_b = min(100, len(answer_b) * 1.5) if len(answer_b) > 20 else 30
        print(f"\n  🏆 质量评估（模拟）：")
        print(f"  策略A得分：{quality_a}/100")
        print(f"  策略B得分：{quality_b}/100")
        if quality_a > quality_b:
            print(f"  🏅 本次胜出：策略A")
        elif quality_b > quality_a:
            print(f"  🏅 本次胜出：策略B")
        else:
            print(f"  🏅 本次平局")

        print("="*60)

    print("\n✅ 实战要点总结：")
    print("   1. A/B测试用相同输入对比不同策略的输出")
    print("   2. 可对比字数、耗时、质量等多个维度")
    print("   3. 真实 LangSmith 支持批量自动化对比实验")
    print("   4. 适用于优化提示词和模型参数")


# ============================================================
# 示例4: 性能分析 - 耗时分析
# ============================================================

def demo_performance_analysis():
    """示例4：性能分析 - 耗时分析（分析每一步的耗时和Token消耗）"""
    print("\n" + "="*60)
    print("示例4：性能分析 - 耗时分析")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 性能分析：分析每一步的耗时和Token消耗")
    print("   - 定位性能瓶颈：找出耗时最长的步骤")
    print("   - Token消耗分析：优化成本")
    print("\n📊 应用场景：")
    print("   - 定位性能瓶颈")
    print("   - 优化Token消耗和成本")
    print("   - 评估链的响应时间")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    print("\n【交互式性能分析演示】")
    print("提示：输入问题，系统分析每一步的耗时和Token消耗")
    print("输入 '退出' 结束\n")

    # 模拟 Token 计费
    TOKEN_PRICE = {
        "input": 0.001 / 1000,   # 每千Token 0.001元
        "output": 0.002 / 1000,  # 每千Token 0.002元
    }

    while True:
        question = input("你的问题：").strip()
        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break
        if not question:
            print("请输入有效问题")
            continue

        # 创建运行记录
        run = DebugRun("性能分析链", {"question": question})
        run.start_time = time.time()

        # 步骤1：问题预处理
        step1_start = time.time()
        prompt1 = ChatPromptTemplate.from_messages([
            ("system", "将问题重写为更清晰的形式。简洁输出。"),
            ("human", "{question}")
        ])
        chain1 = prompt1 | model | StrOutputParser()
        try:
            rewritten = chain1.invoke({"question": question})
            step1_ms = (time.time() - step1_start) * 1000
            # 模拟 Token 使用
            step1_tokens = {"input": len(question) * 2, "output": len(rewritten) * 2}
            run.add_step("问题预处理", "llm", {"question": question}, {"rewritten": rewritten}, step1_ms, tokens=step1_tokens)
        except Exception as e:
            step1_ms = (time.time() - step1_start) * 1000
            run.add_step("问题预处理", "llm", {"question": question}, {}, step1_ms, error=str(e))
            rewritten = question

        # 步骤2：知识检索（模拟）
        step2_start = time.time()
        time.sleep(0.15)  # 模拟检索延迟
        context = f"[检索结果] 关于'{rewritten}'的3篇相关文档"
        step2_ms = (time.time() - step2_start) * 1000
        step2_tokens = {"input": len(rewritten) * 2, "output": len(context) * 2}
        run.add_step("知识检索", "retriever", {"query": rewritten}, {"context": context}, step2_ms, tokens=step2_tokens)

        # 步骤3：答案生成
        step3_start = time.time()
        prompt3 = ChatPromptTemplate.from_messages([
            ("system", "根据上下文回答问题，简洁准确。\n上下文：{context}"),
            ("human", "{question}")
        ])
        chain3 = prompt3 | model | StrOutputParser()
        try:
            answer = chain3.invoke({"question": question, "context": context})
            step3_ms = (time.time() - step3_start) * 1000
            step3_tokens = {"input": (len(question) + len(context)) * 2, "output": len(answer) * 2}
            run.add_step("答案生成", "llm", {"question": question, "context": context}, {"answer": answer}, step3_ms, tokens=step3_tokens)
        except Exception as e:
            step3_ms = (time.time() - step3_start) * 1000
            run.add_step("答案生成", "llm", {"question": question, "context": context}, {}, step3_ms, error=str(e))
            answer = ""

        # 步骤4：答案优化
        step4_start = time.time()
        prompt4 = ChatPromptTemplate.from_messages([
            ("system", "优化以下答案，使其更加流畅：\n{answer}"),
            ("human", "请优化")
        ])
        chain4 = prompt4 | model | StrOutputParser()
        try:
            refined = chain4.invoke({"answer": answer})
            step4_ms = (time.time() - step4_start) * 1000
            step4_tokens = {"input": len(answer) * 2, "output": len(refined) * 2}
            run.add_step("答案优化", "llm", {"answer": answer[:100]}, {"refined": refined}, step4_ms, tokens=step4_tokens)
            final_answer = refined
        except Exception as e:
            step4_ms = (time.time() - step4_start) * 1000
            run.add_step("答案优化", "llm", {"answer": answer[:100]}, {}, step4_ms, error=str(e))
            final_answer = answer

        run.end_time = time.time()
        run.status = "success"
        DebugStore.save(run)

        # 显示结果
        print(f"\n🤖 最终回答：{final_answer}")

        # 显示性能分析
        print(f"\n📊 性能分析 [Run ID: {run.id}]：")
        print("="*60)

        # 耗时分析
        total_ms = run.duration_ms
        print(f"\n  ⏱️ 耗时分析（总耗时: {total_ms:.0f}ms）：")
        print(f"  {'─'*55}")
        print(f"  {'步骤':<15} {'耗时(ms)':<12} {'占比':<10} {'可视化'}")
        print(f"  {'─'*55}")
        for step in run.steps:
            pct = step["duration_ms"] / total_ms * 100 if total_ms > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {step['name']:<15} {step['duration_ms']:<12.0f} {pct:<10.0f}% {bar}")

        # Token 消耗分析
        total_input_tokens = sum(s.get("tokens", {}).get("input", 0) for s in run.steps)
        total_output_tokens = sum(s.get("tokens", {}).get("output", 0) for s in run.steps)
        total_tokens = total_input_tokens + total_output_tokens

        print(f"\n  🪙 Token 消耗分析：")
        print(f"  {'─'*55}")
        print(f"  {'步骤':<15} {'输入Token':<12} {'输出Token':<12} {'合计'}")
        print(f"  {'─'*55}")
        for step in run.steps:
            tokens = step.get("tokens", {})
            inp = tokens.get("input", 0)
            out = tokens.get("output", 0)
            print(f"  {step['name']:<15} {inp:<12} {out:<12} {inp+out}")
        print(f"  {'─'*55}")
        print(f"  {'合计':<15} {total_input_tokens:<12} {total_output_tokens:<12} {total_tokens}")

        # 成本估算
        input_cost = total_input_tokens * TOKEN_PRICE["input"]
        output_cost = total_output_tokens * TOKEN_PRICE["output"]
        total_cost = input_cost + output_cost

        print(f"\n  💰 成本估算（模拟价格）：")
        print(f"  {'─'*55}")
        print(f"  输入Token成本：{total_input_tokens} × ¥{TOKEN_PRICE['input']:.6f}/Token = ¥{input_cost:.4f}")
        print(f"  输出Token成本：{total_output_tokens} × ¥{TOKEN_PRICE['output']:.6f}/Token = ¥{output_cost:.4f}")
        print(f"  总成本：¥{total_cost:.4f}")

        # 性能优化建议
        print(f"\n  💡 性能优化建议：")
        bottleneck = max(run.steps, key=lambda s: s["duration_ms"])
        print(f"  1. 瓶颈步骤：{bottleneck['name']}（{bottleneck['duration_ms']:.0f}ms，占比{bottleneck['duration_ms']/total_ms*100:.0f}%）")
        high_token_step = max(run.steps, key=lambda s: sum(s.get("tokens", {}).values()))
        print(f"  2. Token消耗最多：{high_token_step['name']}（{sum(high_token_step.get('tokens', {}).values())} Token）")
        print(f"  3. 建议：考虑优化{bottleneck['name']}步骤以减少延迟")
        print("="*60)

    print("\n✅ 实战要点总结：")
    print("   1. 性能分析可定位耗时最长的步骤")
    print("   2. Token消耗分析可优化成本")
    print("   3. 真实 LangSmith 提供详细的性能分析面板")
    print("   4. 可根据分析结果优化链的结构和参数")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangSmith 调试（Debugging）- 实战案例")
    print("="*60)
    print("\n本示例演示 LangSmith 调试功能的核心概念和使用方法")

    mode = "真实模式" if has_langsmith_key() else "模拟模式"
    print(f"\n当前模式：{mode}")
    if not has_langsmith_key():
        print("提示：配置 LANGSMITH_API_KEY 可连接真实 LangSmith 服务")

    print("\n核心概念：")
    print("  • Run回放: 复现之前的追踪记录")
    print("  • 中间变量: 查看每一步的输入输出")
    print("  • 对比实验: A/B测试不同参数")
    print("  • 性能分析: 耗时和Token消耗分析")

    print("\n应用场景：")
    print("  • 运行回放、中间变量查看、对比实验、性能分析")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 运行回放 - 复现问题")
        print("  2. 中间变量查看 - 调试")
        print("  3. 对比实验 - A/B测试")
        print("  4. 性能分析 - 耗时分析")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_run_replay()
        elif choice == "2":
            demo_intermediate_variables()
        elif choice == "3":
            demo_ab_testing()
        elif choice == "4":
            demo_performance_analysis()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
