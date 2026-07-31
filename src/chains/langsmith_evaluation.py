"""
LangSmith 评估（Evaluation）- 实战交互式案例
==============================================

本示例演示 LangSmith 评估功能的核心概念和使用方法

核心概念：
- Dataset（数据集）：存储测试用例，包含输入和期望输出
- Evaluator（评估器）：自动评估输出质量的工具
- LLM自评（LLM-as-Judge）：用大模型当裁判，自动给输出打分
- 对比评估（Comparative Evaluation）：对比不同版本的输出质量

应用场景：
- 数据集管理：创建和管理测试用例
- 自动评估：使用评估器自动评估输出质量
- LLM裁判：用大模型给输出打分
- 版本对比：对比不同版本的输出质量
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
# 数据集存储（模拟 LangSmith Dataset）
# ============================================================

class Dataset:
    """模拟 LangSmith 的 Dataset"""

    _datasets: dict = {}

    @classmethod
    def create(cls, name: str, description: str = "") -> dict:
        """创建数据集"""
        if name in cls._datasets:
            return {"success": False, "message": f"数据集 '{name}' 已存在"}
        cls._datasets[name] = {
            "name": name,
            "description": description,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "examples": [],
        }
        return {"success": True, "message": f"数据集 '{name}' 创建成功"}

    @classmethod
    def add_example(cls, name: str, question: str, expected_output: str, tags: list = None) -> dict:
        """添加测试用例"""
        if name not in cls._datasets:
            return {"success": False, "message": f"数据集 '{name}' 不存在"}
        example = {
            "id": f"ex-{len(cls._datasets[name]['examples']) + 1:03d}",
            "question": question,
            "expected_output": expected_output,
            "tags": tags or [],
        }
        cls._datasets[name]["examples"].append(example)
        return {"success": True, "message": f"用例 {example['id']} 添加成功", "example": example}

    @classmethod
    def get(cls, name: str) -> Optional[dict]:
        """获取数据集"""
        return cls._datasets.get(name)

    @classmethod
    def list_all(cls) -> dict:
        """列出所有数据集"""
        return cls._datasets

    @classmethod
    def delete(cls, name: str) -> dict:
        """删除数据集"""
        if name in cls._datasets:
            del cls._datasets[name]
            return {"success": True, "message": f"数据集 '{name}' 已删除"}
        return {"success": False, "message": f"数据集 '{name}' 不存在"}


# ============================================================
# 评估结果存储
# ============================================================

class EvalResult:
    """评估结果"""

    def __init__(self, experiment_name: str, dataset_name: str):
        self.id = f"eval-{uuid.uuid4().hex[:8]}"
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.results: list = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_result(self, example_id: str, question: str, expected: str,
                   actual: str, scores: dict, overall_score: float):
        """添加一条评估结果"""
        self.results.append({
            "example_id": example_id,
            "question": question,
            "expected": expected,
            "actual": actual,
            "scores": scores,
            "overall_score": overall_score,
        })

    @property
    def avg_score(self) -> float:
        """平均分"""
        if not self.results:
            return 0
        return sum(r["overall_score"] for r in self.results) / len(self.results)

    @property
    def pass_rate(self) -> float:
        """通过率（得分>=60）"""
        if not self.results:
            return 0
        passed = sum(1 for r in self.results if r["overall_score"] >= 60)
        return passed / len(self.results) * 100


class EvalStore:
    """评估结果存储"""

    _results: list = []

    @classmethod
    def save(cls, result: EvalResult) -> str:
        """保存评估结果"""
        cls._results.append(result)
        return result.id

    @classmethod
    def get_all(cls) -> list:
        """获取所有评估结果"""
        return cls._results


# ============================================================
# 示例1: 数据集管理 - 创建测试集
# ============================================================

def demo_dataset_management():
    """示例1：数据集管理 - 创建测试集（用户输入问答对，自动创建测试集）"""
    print("\n" + "="*60)
    print("示例1：数据集管理 - 创建测试集")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - Dataset: 数据集，存储测试用例")
    print("   - 每个用例包含：输入问题 + 期望输出")
    print("   - 数据集可用于批量评估和回归测试")
    print("\n📊 应用场景：")
    print("   - 创建和管理测试用例")
    print("   - 批量评估模型输出质量")
    print("   - 回归测试验证变更影响")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    print("\n【交互式数据集管理演示】")

    while True:
        print("\n" + "-"*60)
        print("数据集管理菜单：")
        print("  1. 创建数据集")
        print("  2. 添加测试用例")
        print("  3. 查看数据集")
        print("  4. 列出所有数据集")
        print("  5. 删除数据集")
        print("  6. 快速创建示例数据集")
        print("\n  0. 退出")
        print("-"*60)

        choice = input("请选择 (0-6): ").strip()

        if choice == "0":
            print("结束演示")
            break

        elif choice == "1":
            name = input("数据集名称：").strip()
            if not name:
                print("❌ 名称不能为空")
                continue
            desc = input("数据集描述（可选）：").strip()
            result = Dataset.create(name, desc)
            print(f"{'✅' if result['success'] else '❌'} {result['message']}")

        elif choice == "2":
            name = input("目标数据集名称：").strip()
            ds = Dataset.get(name)
            if not ds:
                print(f"❌ 数据集 '{name}' 不存在，请先创建")
                continue
            question = input("问题：").strip()
            if not question:
                print("❌ 问题不能为空")
                continue
            expected = input("期望输出：").strip()
            if not expected:
                print("❌ 期望输出不能为空")
                continue
            tags_str = input("标签（逗号分隔，可选）：").strip()
            tags = [t.strip() for t in tags_str.split(",")] if tags_str else []
            result = Dataset.add_example(name, question, expected, tags)
            print(f"{'✅' if result['success'] else '❌'} {result['message']}")

        elif choice == "3":
            name = input("数据集名称：").strip()
            ds = Dataset.get(name)
            if not ds:
                print(f"❌ 数据集 '{name}' 不存在")
                continue
            print(f"\n📦 数据集：{ds['name']}")
            print(f"   描述：{ds['description'] or '无'}")
            print(f"   创建时间：{ds['created_at']}")
            print(f"   用例数量：{len(ds['examples'])}")
            if ds["examples"]:
                print(f"\n   用例列表：")
                for ex in ds["examples"]:
                    print(f"   [{ex['id']}] Q: {ex['question'][:50]}")
                    print(f"         A: {ex['expected_output'][:50]}")
                    if ex['tags']:
                        print(f"         标签: {', '.join(ex['tags'])}")

        elif choice == "4":
            all_ds = Dataset.list_all()
            if not all_ds:
                print("📭 暂无数据集")
                continue
            print(f"\n📦 数据集列表（共 {len(all_ds)} 个）：")
            for name, ds in all_ds.items():
                print(f"   {name} ({len(ds['examples'])} 用例) - {ds['description'] or '无描述'}")

        elif choice == "5":
            name = input("要删除的数据集名称：").strip()
            confirm = input(f"确认删除 '{name}'？(y/n): ").strip().lower()
            if confirm == 'y':
                result = Dataset.delete(name)
                print(f"{'✅' if result['success'] else '❌'} {result['message']}")
            else:
                print("已取消")

        elif choice == "6":
            # 快速创建示例数据集
            name = "QA测试集"
            Dataset.create(name, "用于评估问答质量的测试数据集")
            examples = [
                ("什么是Python？", "Python是一种高级编程语言，以简洁易读著称"),
                ("什么是机器学习？", "机器学习是人工智能的一个分支，通过数据训练模型"),
                ("什么是API？", "API是应用程序编程接口，用于不同软件间的通信"),
                ("什么是数据库？", "数据库是按照数据结构组织、存储和管理数据的仓库"),
                ("什么是云计算？", "云计算是通过互联网提供计算资源和服务的模式"),
            ]
            for q, a in examples:
                Dataset.add_example(name, q, a)
            print(f"✅ 示例数据集 '{name}' 创建成功（包含 {len(examples)} 个用例）")

        else:
            print("❌ 无效选项")

    print("\n✅ 实战要点总结：")
    print("   1. 数据集是结构化的测试用例集合")
    print("   2. 每个用例包含输入和期望输出")
    print("   3. 真实 LangSmith 支持 CSV/JSON 导入数据集")
    print("   4. 数据集可用于批量评估和回归测试")


# ============================================================
# 示例2: 评估器 - 自动评估
# ============================================================

def demo_evaluator():
    """示例2：评估器 - 自动评估（使用评估器自动评估输出质量）"""
    print("\n" + "="*60)
    print("示例2：评估器 - 自动评估")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - Evaluator: 评估器，自动评估输出质量")
    print("   - 评估维度：准确性、相关性、完整性等")
    print("   - 真实 LangSmith 支持自定义评估器")
    print("\n📊 应用场景：")
    print("   - 自动化评估模型输出质量")
    print("   - 多维度评估（准确性、相关性、完整性）")
    print("   - 批量评估和回归测试")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    # 定义评估器
    def evaluate_accuracy(answer: str, expected: str) -> dict:
        """准确性评估：关键词覆盖率"""
        expected_keywords = [k.strip() for k in expected.replace("，", ",").replace("。", ".").split(",") if k.strip()]
        if not expected_keywords:
            # 简单按字符匹配
            expected_keywords = [expected]
        matched = [k for k in expected_keywords if k in answer]
        score = len(matched) / len(expected_keywords) * 100 if expected_keywords else 0
        return {
            "dimension": "准确性",
            "score": score,
            "detail": f"命中 {len(matched)}/{len(expected_keywords)} 个关键词",
            "matched": matched,
            "missed": [k for k in expected_keywords if k not in answer],
        }

    def evaluate_relevance(answer: str, question: str) -> dict:
        """相关性评估：回答是否切题"""
        # 简单模拟：基于回答长度和问题的比例
        ratio = len(answer) / max(len(question), 1)
        if ratio < 0.5:
            score = 30
            detail = "回答过短，可能不够相关"
        elif ratio > 10:
            score = 60
            detail = "回答过长，可能偏离主题"
        else:
            score = 85
            detail = "回答长度适中，相关性较好"
        return {
            "dimension": "相关性",
            "score": score,
            "detail": detail,
        }

    def evaluate_completeness(answer: str) -> dict:
        """完整性评估：回答是否完整"""
        has_structure = any(m in answer for m in ["1.", "2.", "-", "•", "首先", "其次", "第一"])
        has_conclusion = any(m in answer for m in ["总之", "综上", "因此", "所以", "总结"])
        score = 50
        if has_structure:
            score += 25
        if has_conclusion:
            score += 25
        return {
            "dimension": "完整性",
            "score": min(score, 100),
            "detail": f"结构化: {'有' if has_structure else '无'}, 结论: {'有' if has_conclusion else '无'}",
        }

    print("\n【交互式评估演示】")
    print("提示：输入问题和期望答案，系统自动评估输出质量")
    print("输入 '退出' 结束\n")

    while True:
        question = input("问题：").strip()
        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break
        if not question:
            print("请输入有效问题")
            continue

        expected = input("期望答案（关键词，用逗号分隔）：").strip()
        if not expected:
            print("请输入期望答案")
            continue

        # 调用模型
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个知识问答助手，简洁准确地回答问题"),
            ("human", "{question}")
        ])
        chain = prompt | model | StrOutputParser()
        try:
            answer = chain.invoke({"question": question})
        except Exception as e:
            print(f"❌ 调用失败: {e}")
            continue

        # 运行评估器
        acc_result = evaluate_accuracy(answer, expected)
        rel_result = evaluate_relevance(answer, question)
        comp_result = evaluate_completeness(answer)

        # 综合评分
        overall_score = (acc_result["score"] * 0.5 + rel_result["score"] * 0.3 + comp_result["score"] * 0.2)

        # 显示结果
        print(f"\n🤖 模型回答：{answer}")
        print(f"\n📊 评估结果：")
        print("="*60)
        print(f"  {'维度':<10} {'得分':<10} {'详情'}")
        print(f"  {'─'*55}")
        for result in [acc_result, rel_result, comp_result]:
            print(f"  {result['dimension']:<10} {result['score']:<10.0f} {result['detail']}")
        print(f"  {'─'*55}")
        print(f"  {'综合评分':<10} {overall_score:<10.0f} (准确性50% + 相关性30% + 完整性20%)")

        # 评级
        if overall_score >= 80:
            grade = "优秀 🌟"
        elif overall_score >= 60:
            grade = "良好 ✅"
        elif overall_score >= 40:
            grade = "及格 ⚠️"
        else:
            grade = "不及格 ❌"
        print(f"  {'评级':<10} {grade}")

        # 改进建议
        print(f"\n  💡 改进建议：")
        if acc_result["score"] < 60:
            missed = acc_result.get("missed", [])
            if missed:
                print(f"     - 准确性不足，缺少关键词：{', '.join(missed)}")
        if rel_result["score"] < 60:
            print(f"     - 相关性不足，建议调整提示词使回答更切题")
        if comp_result["score"] < 60:
            print(f"     - 完整性不足，建议要求模型给出结构化回答和总结")

        print("="*60)

    print("\n✅ 实战要点总结：")
    print("   1. 评估器可从多个维度自动评估输出质量")
    print("   2. 常见维度：准确性、相关性、完整性")
    print("   3. 可根据评估结果调整提示词和参数")
    print("   4. 真实 LangSmith 支持自定义评估器和批量评估")


# ============================================================
# 示例3: LLM自评 - 大模型裁判
# ============================================================

def demo_llm_as_judge():
    """示例3：LLM自评 - 大模型裁判（用大模型当裁判，自动给输出打分）"""
    print("\n" + "="*60)
    print("示例3：LLM自评 - 大模型裁判")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - LLM-as-Judge：用大模型当裁判，评估输出质量")
    print("   - 比规则评估更灵活，能理解语义")
    print("   - 真实 LangSmith 支持 LLM 评估器")
    print("\n📊 应用场景：")
    print("   - 语义级别的质量评估")
    print("   - 评估回答的准确性和有用性")
    print("   - 自动化评估流程")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    # LLM 裁判提示词
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的评估裁判，负责评估AI回答的质量。

评估维度：
1. 准确性（0-100分）：回答是否事实正确
2. 相关性（0-100分）：回答是否切题
3. 完整性（0-100分）：回答是否完整
4. 有用性（0-100分）：回答对用户是否有帮助

请严格按照以下JSON格式输出评估结果，不要输出其他内容：
{{"accuracy": 分数, "relevance": 分数, "completeness": 分数, "usefulness": 分数, "overall": 总分, "comment": "简短评语"}}"""),
        ("human", """请评估以下回答的质量：

问题：{question}
期望答案：{expected}
实际回答：{actual}

请输出JSON格式的评估结果：""")
    ])

    print("\n【交互式 LLM 裁判演示】")
    print("提示：输入问题和期望答案，LLM裁判自动给输出打分")
    print("输入 '退出' 结束\n")

    while True:
        question = input("问题：").strip()
        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break
        if not question:
            print("请输入有效问题")
            continue

        expected = input("期望答案（可选，直接回车跳过）：").strip() or "无特定期望"

        # 生成回答
        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个知识问答助手，简洁准确地回答问题"),
            ("human", "{question}")
        ])
        gen_chain = gen_prompt | model | StrOutputParser()
        try:
            answer = gen_chain.invoke({"question": question})
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            continue

        print(f"\n🤖 模型回答：{answer}")

        # LLM 裁判评估
        print(f"\n⚖️ LLM裁判正在评估...")
        judge_chain = judge_prompt | model | StrOutputParser()
        try:
            judge_result_str = judge_chain.invoke({
                "question": question,
                "expected": expected,
                "actual": answer
            })
            # 解析 JSON
            # 尝试从输出中提取 JSON
            import re
            json_match = re.search(r'\{[^}]+\}', judge_result_str.replace('\n', ''))
            if json_match:
                judge_result = json.loads(json_match.group())
            else:
                judge_result = {"overall": 0, "comment": "解析失败"}
        except Exception as e:
            print(f"❌ 评估失败: {e}")
            judge_result = {"overall": 0, "comment": f"评估异常: {e}"}

        # 显示评估结果
        print(f"\n📊 LLM裁判评估结果：")
        print("="*60)
        if "accuracy" in judge_result:
            print(f"  准确性：{judge_result['accuracy']}/100")
        if "relevance" in judge_result:
            print(f"  相关性：{judge_result['relevance']}/100")
        if "completeness" in judge_result:
            print(f"  完整性：{judge_result['completeness']}/100")
        if "usefulness" in judge_result:
            print(f"  有用性：{judge_result['usefulness']}/100")

        overall = judge_result.get("overall", 0)
        comment = judge_result.get("comment", "无评语")

        # 评级
        if isinstance(overall, (int, float)):
            if overall >= 80:
                grade = "优秀 🌟"
            elif overall >= 60:
                grade = "良好 ✅"
            elif overall >= 40:
                grade = "及格 ⚠️"
            else:
                grade = "不及格 ❌"
        else:
            grade = "无法评级"

        print(f"  {'─'*55}")
        print(f"  综合评分：{overall}/100")
        print(f"  评级：{grade}")
        print(f"  评语：{comment}")
        print("="*60)

    print("\n✅ 实战要点总结：")
    print("   1. LLM-as-Judge 用大模型评估输出质量")
    print("   2. 比规则评估更灵活，能理解语义")
    print("   3. 可从多个维度评估（准确性、相关性、完整性、有用性）")
    print("   4. 真实 LangSmith 支持 LLM 评估器集成")


# ============================================================
# 示例4: 对比评估 - 版本对比
# ============================================================

def demo_comparative_evaluation():
    """示例4：对比评估 - 版本对比（对比不同版本的输出质量）"""
    print("\n" + "="*60)
    print("示例4：对比评估 - 版本对比")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 对比评估：对比不同版本的输出质量")
    print("   - 可对比不同提示词、不同模型、不同参数")
    print("   - 真实 LangSmith 支持自动化对比实验")
    print("\n📊 应用场景：")
    print("   - 对比不同版本的提示词效果")
    print("   - 对比不同模型的输出质量")
    print("   - 验证优化是否有效")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    # 定义两个版本的提示词
    version_a = {
        "name": "v1.0 基础版",
        "system_prompt": "你是一个知识问答助手，简洁回答问题。"
    }
    version_b = {
        "name": "v2.0 优化版",
        "system_prompt": "你是一个专业的知识问答助手，请按照以下格式回答：\n1. 先给出直接答案\n2. 再给出简要解释\n3. 如有相关示例请补充"
    }

    print("\n【交互式对比评估演示】")
    print("提示：输入问题，系统用两个版本的提示词对比输出质量")
    print("输入 '退出' 结束\n")

    # 评估函数
    def quick_evaluate(question: str, expected: str, actual: str) -> dict:
        """快速评估"""
        # 准确性
        expected_words = set(expected.replace("，", " ").replace("。", " ").split())
        actual_words = set(actual.replace("，", " ").replace("。", " ").split())
        overlap = expected_words & actual_words
        accuracy = len(overlap) / max(len(expected_words), 1) * 100

        # 完整性
        has_structure = any(m in actual for m in ["1.", "2.", "3.", "-", "•", "首先", "其次"])
        completeness = 70 + 30 if has_structure else 70

        # 有用性
        usefulness = min(100, len(actual) * 2) if len(actual) > 20 else 30

        overall = accuracy * 0.4 + completeness * 0.3 + usefulness * 0.3
        return {
            "accuracy": round(accuracy, 0),
            "completeness": round(completeness, 0),
            "usefulness": round(usefulness, 0),
            "overall": round(overall, 0),
        }

    while True:
        question = input("问题：").strip()
        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break
        if not question:
            print("请输入有效问题")
            continue

        expected = input("期望答案关键词（用逗号分隔）：").strip()
        if not expected:
            expected = question  # 简单回退

        # 版本A
        prompt_a = ChatPromptTemplate.from_messages([
            ("system", version_a["system_prompt"]),
            ("human", "{question}")
        ])
        chain_a = prompt_a | model | StrOutputParser()
        try:
            answer_a = chain_a.invoke({"question": question})
            eval_a = quick_evaluate(question, expected, answer_a)
        except Exception as e:
            answer_a = f"[失败: {e}]"
            eval_a = {"accuracy": 0, "completeness": 0, "usefulness": 0, "overall": 0}

        # 版本B
        prompt_b = ChatPromptTemplate.from_messages([
            ("system", version_b["system_prompt"]),
            ("human", "{question}")
        ])
        chain_b = prompt_b | model | StrOutputParser()
        try:
            answer_b = chain_b.invoke({"question": question})
            eval_b = quick_evaluate(question, expected, answer_b)
        except Exception as e:
            answer_b = f"[失败: {e}]"
            eval_b = {"accuracy": 0, "completeness": 0, "usefulness": 0, "overall": 0}

        # 对比结果
        print(f"\n📊 对比评估结果：")
        print("="*60)
        print(f"  问题：{question}")
        print("="*60)

        print(f"\n  🅰️ {version_a['name']}")
        print(f"  {'─'*55}")
        print(f"  回答：{answer_a[:200]}{'...' if len(answer_a) > 200 else ''}")
        print(f"  评估：准确性={eval_a['accuracy']:.0f} 完整性={eval_a['completeness']:.0f} 有用性={eval_a['usefulness']:.0f}")
        print(f"  综合评分：{eval_a['overall']:.0f}/100")

        print(f"\n  🅱️ {version_b['name']}")
        print(f"  {'─'*55}")
        print(f"  回答：{answer_b[:200]}{'...' if len(answer_b) > 200 else ''}")
        print(f"  评估：准确性={eval_b['accuracy']:.0f} 完整性={eval_b['completeness']:.0f} 有用性={eval_b['usefulness']:.0f}")
        print(f"  综合评分：{eval_b['overall']:.0f}/100")

        # 对比分析
        print(f"\n  📈 版本对比分析：")
        print(f"  {'─'*55}")
        print(f"  {'维度':<10} {'v1.0':<10} {'v2.0':<10} {'差异':<15} {'结论'}")
        print(f"  {'─'*55}")

        for dim in ["accuracy", "completeness", "usefulness", "overall"]:
            dim_names = {"accuracy": "准确性", "completeness": "完整性", "usefulness": "有用性", "overall": "综合"}
            diff = eval_b[dim] - eval_a[dim]
            if diff > 5:
                conclusion = "B更优 ✅"
            elif diff < -5:
                conclusion = "A更优 ✅"
            else:
                conclusion = "持平 ➡️"
            print(f"  {dim_names[dim]:<10} {eval_a[dim]:<10.0f} {eval_b[dim]:<10.0f} {diff:+.0f}{'':<10} {conclusion}")

        # 总体结论
        if eval_b["overall"] > eval_a["overall"] + 5:
            print(f"\n  🏅 总体结论：{version_b['name']} 更优（提升 {eval_b['overall'] - eval_a['overall']:.0f} 分）")
        elif eval_a["overall"] > eval_b["overall"] + 5:
            print(f"\n  🏅 总体结论：{version_a['name']} 更优（提升 {eval_a['overall'] - eval_b['overall']:.0f} 分）")
        else:
            print(f"\n  🏅 总体结论：两个版本表现相当")

        print("="*60)

    print("\n✅ 实战要点总结：")
    print("   1. 对比评估可量化不同版本的输出质量差异")
    print("   2. 从多个维度对比（准确性、完整性、有用性）")
    print("   3. 可验证优化是否有效")
    print("   4. 真实 LangSmith 支持自动化对比实验和统计显著性检验")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangSmith 评估（Evaluation）- 实战案例")
    print("="*60)
    print("\n本示例演示 LangSmith 评估功能的核心概念和使用方法")

    mode = "真实模式" if has_langsmith_key() else "模拟模式"
    print(f"\n当前模式：{mode}")
    if not has_langsmith_key():
        print("提示：配置 LANGSMITH_API_KEY 可连接真实 LangSmith 服务")

    print("\n核心概念：")
    print("  • Dataset: 数据集，存储测试用例")
    print("  • Evaluator: 评估器，自动评估输出质量")
    print("  • LLM自评: 用大模型当裁判")
    print("  • 对比评估: 对比不同版本的效果")

    print("\n应用场景：")
    print("  • 数据集管理、自动评估、LLM裁判、版本对比")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 数据集管理 - 创建测试集")
        print("  2. 评估器 - 自动评估")
        print("  3. LLM自评 - 大模型裁判")
        print("  4. 对比评估 - 版本对比")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_dataset_management()
        elif choice == "2":
            demo_evaluator()
        elif choice == "3":
            demo_llm_as_judge()
        elif choice == "4":
            demo_comparative_evaluation()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
