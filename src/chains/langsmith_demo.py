"""
LangSmith 可观测性 - 实战交互式案例
====================================

本示例演示 LangSmith 可观测性的核心功能（模拟实现）

核心概念：
- 追踪（Tracing）：记录每次 LLM 调用的完整信息
- 评估（Evaluation）：衡量模型输出的质量
- 数据集（Dataset）：管理测试数据以支持评估
- 自定义评估器：根据业务规则定制评估逻辑

应用场景：
- 调试与排错：通过追踪定位问题
- 质量保障：评估模型输出是否达标
- 回归测试：用数据集验证变更影响
- 业务定制：自定义评估器匹配业务需求
"""

import os
import sys
import json
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_loader import get_default_llm


# ============================================================
# 模拟追踪存储（代替真实 LangSmith 后端）
# ============================================================

class TraceStore:
    """模拟 LangSmith 追踪数据存储"""

    _traces = []

    @classmethod
    def record(cls, trace_data: dict):
        """记录一条追踪信息"""
        trace_data["id"] = f"trace-{len(cls._traces) + 1:04d}"
        trace_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cls._traces.append(trace_data)
        return trace_data["id"]

    @classmethod
    def get_all(cls):
        """获取所有追踪记录"""
        return cls._traces

    @classmethod
    def get_by_id(cls, trace_id: str):
        """按 ID 获取追踪记录"""
        for t in cls._traces:
            if t["id"] == trace_id:
                return t
        return None

    @classmethod
    def clear(cls):
        """清空追踪记录"""
        cls._traces = []


# ============================================================
# 模拟数据集存储
# ============================================================

class DatasetStore:
    """模拟 LangSmith 数据集存储"""

    _datasets = {}

    @classmethod
    def create(cls, name: str, description: str = ""):
        """创建数据集"""
        if name in cls._datasets:
            return False, f"数据集 '{name}' 已存在"
        cls._datasets[name] = {
            "name": name,
            "description": description,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "examples": []
        }
        return True, f"数据集 '{name}' 创建成功"

    @classmethod
    def add_example(cls, name: str, example: dict):
        """向数据集添加样本"""
        if name not in cls._datasets:
            return False, f"数据集 '{name}' 不存在"
        example["id"] = f"ex-{len(cls._datasets[name]['examples']) + 1:03d}"
        cls._datasets[name]["examples"].append(example)
        return True, f"样本 {example['id']} 添加成功"

    @classmethod
    def get(cls, name: str):
        """获取数据集"""
        return cls._datasets.get(name)

    @classmethod
    def list_all(cls):
        """列出所有数据集"""
        return cls._datasets

    @classmethod
    def delete(cls, name: str):
        """删除数据集"""
        if name in cls._datasets:
            del cls._datasets[name]
            return True, f"数据集 '{name}' 已删除"
        return False, f"数据集 '{name}' 不存在"


# ============================================================
# 1. 追踪基础
# ============================================================

def demo_tracing_basics():
    """示例1：追踪基础 - 记录每次调用的追踪信息"""
    print("\n" + "="*60)
    print("示例1：追踪基础 - 记录每次调用的追踪信息")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 每次调用自动记录输入、输出、耗时")
    print("   - 追踪信息可用于调试和性能分析")
    print("   - 真实 LangSmith 中会自动记录 Token 用量")

    model = get_default_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个知识问答助手，简洁回答问题"),
        ("human", "{question}")
    ])

    chain = prompt | model | StrOutputParser()

    print("\n【交互式追踪演示】")
    print("提示：输入问题，系统自动记录追踪信息")
    print("输入 '退出' 结束\n")

    while True:
        question = input("你的问题：").strip()

        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not question:
            print("请输入有效问题")
            continue

        # 执行调用并记录追踪
        start_time = time.time()
        try:
            answer = chain.invoke({"question": question})
            elapsed = time.time() - start_time
            status = "success"
        except Exception as e:
            answer = f"[调用失败: {e}]"
            elapsed = time.time() - start_time
            status = "error"

        # 记录追踪信息
        trace_id = TraceStore.record({
            "name": "qa_chain",
            "input": {"question": question},
            "output": answer,
            "duration_ms": round(elapsed * 1000, 2),
            "status": status,
            "metadata": {
                "model": "default_llm",
                "prompt_tokens": len(question) * 2,  # 模拟
                "completion_tokens": len(answer) * 2,  # 模拟
            }
        })

        # 显示回答
        print(f"\n🤖 回答：{answer}")
        print(f"\n📋 追踪信息：")
        print(f"   追踪ID：{trace_id}")
        print(f"   状态：{status}")
        print(f"   耗时：{elapsed:.2f}秒")
        print(f"   输入Token(模拟)：{len(question) * 2}")
        print(f"   输出Token(模拟)：{len(answer) * 2}")
        print("-"*60)

    # 显示追踪汇总
    traces = TraceStore.get_all()
    if traces:
        print(f"\n📊 追踪汇总（共 {len(traces)} 条记录）：")
        print(f"{'ID':<18} {'状态':<10} {'耗时(ms)':<12} {'时间'}")
        print("-"*60)
        for t in traces:
            print(f"{t['id']:<18} {t['status']:<10} {t['duration_ms']:<12} {t['timestamp']}")
    else:
        print("\n暂无追踪记录")

    print("\n✅ 实战要点总结：")
    print("   1. 追踪记录包含输入、输出、耗时、状态")
    print("   2. 可用于调试、性能分析和成本监控")
    print("   3. 真实 LangSmith 提供可视化追踪界面")


# ============================================================
# 2. 评估系统
# ============================================================

def demo_evaluation():
    """示例2：评估系统 - 评估模型输出的质量"""
    print("\n" + "="*60)
    print("示例2：评估系统 - 评估模型输出的质量")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 评估是验证模型输出质量的关键环节")
    print("   - 常见维度：准确性、相关性、完整性")
    print("   - 真实 LangSmith 支持自动化批量评估")

    model = get_default_llm()

    print("\n【交互式评估演示】")
    print("提示：输入问题和期望答案，系统评估模型输出")
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
            ("system", "你是一个知识问答助手，简洁准确地回答"),
            ("human", "{question}")
        ])
        chain = prompt | model | StrOutputParser()

        start_time = time.time()
        try:
            answer = chain.invoke({"question": question})
            elapsed = time.time() - start_time
        except Exception as e:
            answer = f"[调用失败: {e}]"
            elapsed = time.time() - start_time

        # 模拟评估
        expected_keywords = [k.strip() for k in expected.split(",")]
        matched = [k for k in expected_keywords if k in answer]
        accuracy = len(matched) / len(expected_keywords) * 100 if expected_keywords else 0

        # 相关性评估（基于回答长度和问题的比例，模拟）
        relevance = min(100, max(20, len(answer) / max(len(question), 1) * 30))

        # 完整性评估（关键词覆盖率）
        completeness = accuracy

        print(f"\n🤖 模型回答：{answer}")
        print(f"\n📊 评估结果：")
        print(f"   准确性：{accuracy:.0f}% （{len(matched)}/{len(expected_keywords)} 关键词命中）")
        print(f"   相关性：{relevance:.0f}% （模拟评估）")
        print(f"   完整性：{completeness:.0f}% （关键词覆盖率）")
        print(f"   耗时：{elapsed:.2f}秒")
        if matched:
            print(f"   命中关键词：{', '.join(matched)}")
        missed = [k for k in expected_keywords if k not in answer]
        if missed:
            print(f"   未命中关键词：{', '.join(missed)}")
        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 评估需要明确的衡量维度和标准")
    print("   2. 关键词匹配是最基础的评估方法")
    print("   3. 真实 LangSmith 支持 LLM 作为评判者进行评估")


# ============================================================
# 3. 数据集管理
# ============================================================

def demo_dataset_management():
    """示例3：数据集管理 - 创建和管理测试数据集"""
    print("\n" + "="*60)
    print("示例3：数据集管理 - 创建和管理测试数据集")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 数据集用于存储测试用例（输入 + 期望输出）")
    print("   - 支持创建、查看、添加、删除操作")
    print("   - 真实 LangSmith 提供在线数据集管理界面")

    print("\n【交互式数据集管理演示】")
    print("提示：通过菜单管理测试数据集\n")

    while True:
        print("\n" + "-"*60)
        print("数据集管理菜单：")
        print("  1. 创建数据集")
        print("  2. 添加样本")
        print("  3. 查看数据集")
        print("  4. 列出所有数据集")
        print("  5. 删除数据集")
        print("  6. 运行数据集测试")
        print("\n  0. 退出")
        print("-"*60)

        choice = input("请选择 (0-6): ").strip()

        if choice == "0":
            print("结束演示")
            break

        elif choice == "1":
            # 创建数据集
            name = input("数据集名称：").strip()
            if not name:
                print("❌ 名称不能为空")
                continue
            description = input("数据集描述（可选）：").strip()
            ok, msg = DatasetStore.create(name, description)
            print(f"{'✅' if ok else '❌'} {msg}")

        elif choice == "2":
            # 添加样本
            name = input("目标数据集名称：").strip()
            ds = DatasetStore.get(name)
            if not ds:
                print(f"❌ 数据集 '{name}' 不存在，请先创建")
                continue
            inp = input("输入内容：").strip()
            if not inp:
                print("❌ 输入不能为空")
                continue
            expected = input("期望输出：").strip()
            if not expected:
                print("❌ 期望输出不能为空")
                continue
            tags = input("标签（逗号分隔，可选）：").strip()
            example = {"input": inp, "expected_output": expected}
            if tags:
                example["tags"] = [t.strip() for t in tags.split(",")]
            ok, msg = DatasetStore.add_example(name, example)
            print(f"{'✅' if ok else '❌'} {msg}")

        elif choice == "3":
            # 查看数据集
            name = input("数据集名称：").strip()
            ds = DatasetStore.get(name)
            if not ds:
                print(f"❌ 数据集 '{name}' 不存在")
                continue
            print(f"\n📦 数据集：{ds['name']}")
            print(f"   描述：{ds['description'] or '无'}")
            print(f"   创建时间：{ds['created_at']}")
            print(f"   样本数量：{len(ds['examples'])}")
            if ds["examples"]:
                print(f"\n   样本列表：")
                for ex in ds["examples"]:
                    print(f"   [{ex['id']}] 输入: {ex['input']}")
                    print(f"        期望: {ex['expected_output']}")
                    if 'tags' in ex:
                        print(f"        标签: {', '.join(ex['tags'])}")

        elif choice == "4":
            # 列出所有数据集
            all_ds = DatasetStore.list_all()
            if not all_ds:
                print("📭 暂无数据集")
                continue
            print(f"\n📦 数据集列表（共 {len(all_ds)} 个）：")
            for name, ds in all_ds.items():
                print(f"   {name} ({len(ds['examples'])} 样本) - {ds['description'] or '无描述'}")

        elif choice == "5":
            # 删除数据集
            name = input("要删除的数据集名称：").strip()
            confirm = input(f"确认删除 '{name}'？(y/n): ").strip().lower()
            if confirm == 'y':
                ok, msg = DatasetStore.delete(name)
                print(f"{'✅' if ok else '❌'} {msg}")
            else:
                print("已取消")

        elif choice == "6":
            # 运行数据集测试
            name = input("数据集名称：").strip()
            ds = DatasetStore.get(name)
            if not ds:
                print(f"❌ 数据集 '{name}' 不存在")
                continue
            if not ds["examples"]:
                print("❌ 数据集为空，请先添加样本")
                continue

            model = get_default_llm()
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个知识问答助手，简洁准确地回答"),
                ("human", "{input}")
            ])
            chain = prompt | model | StrOutputParser()

            print(f"\n🧪 运行数据集 '{name}' 测试（{len(ds['examples'])} 个样本）...\n")

            total = len(ds["examples"])
            passed = 0
            for i, ex in enumerate(ds["examples"], 1):
                print(f"[{i}/{total}] 测试: {ex['input']}")
                try:
                    result = chain.invoke({"input": ex["input"]})
                    # 简单匹配：期望输出是否在结果中
                    is_match = ex["expected_output"].lower() in result.lower()
                    status = "✅ 通过" if is_match else "❌ 未通过"
                    if is_match:
                        passed += 1
                    print(f"   期望: {ex['expected_output']}")
                    print(f"   实际: {result[:100]}{'...' if len(result) > 100 else ''}")
                    print(f"   {status}")
                except Exception as e:
                    print(f"   ❌ 调用失败: {e}")
                print()

            rate = passed / total * 100 if total > 0 else 0
            print(f"📊 测试结果：{passed}/{total} 通过，通过率 {rate:.0f}%")

        else:
            print("❌ 无效选项")

    print("\n✅ 实战要点总结：")
    print("   1. 数据集是结构化的测试用例集合")
    print("   2. 每个样本包含输入和期望输出")
    print("   3. 数据集可用于回归测试和质量监控")


# ============================================================
# 4. 自定义评估器
# ============================================================

def demo_custom_evaluator():
    """示例4：自定义评估器 - 创建自定义的评估规则"""
    print("\n" + "="*60)
    print("示例4：自定义评估器 - 创建自定义的评估规则")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 内置评估器无法满足所有业务需求")
    print("   - 自定义评估器可匹配特定业务规则")
    print("   - 真实 LangSmith 支持继承 RunEvaluator 类")

    # 预定义的自定义评估器
    evaluators = {
        "1": {
            "name": "长度评估器",
            "description": "评估回答长度是否在合理范围内",
            "evaluate": lambda answer, expected: {
                "length": len(answer),
                "score": 100 if 20 <= len(answer) <= 500 else (70 if 10 <= len(answer) <= 1000 else 30),
                "detail": f"长度 {len(answer)} 字，{'适中' if 20 <= len(answer) <= 500 else '偏短或偏长'}"
            }
        },
        "2": {
            "name": "关键词评估器",
            "description": "评估回答是否包含关键词",
            "evaluate": lambda answer, expected: {
                "keywords_found": [k for k in expected if k in answer],
                "keywords_missed": [k for k in expected if k not in answer],
                "score": len([k for k in expected if k in answer]) / len(expected) * 100 if expected else 0,
                "detail": f"命中 {len([k for k in expected if k in answer])}/{len(expected)} 个关键词"
            }
        },
        "3": {
            "name": "情感评估器",
            "description": "评估回答的情感倾向（简单模拟）",
            "evaluate": lambda answer, expected: {
                "positive_words": len([w for w in ["好", "优秀", "推荐", "不错", "喜欢", "成功"] if w in answer]),
                "negative_words": len([w for w in ["差", "失败", "错误", "问题", "不好", "拒绝"] if w in answer]),
                "score": 80 if len([w for w in ["好", "优秀", "推荐", "不错", "喜欢", "成功"] if w in answer]) > len([w for w in ["差", "失败", "错误", "问题", "不好", "拒绝"] if w in answer]) else 40,
                "detail": "积极倾向" if len([w for w in ["好", "优秀", "推荐", "不错", "喜欢", "成功"] if w in answer]) > len([w for w in ["差", "失败", "错误", "问题", "不好", "拒绝"] if w in answer]) else "消极倾向"
            }
        },
        "4": {
            "name": "格式评估器",
            "description": "评估回答是否符合格式要求",
            "evaluate": lambda answer, expected: {
                "has_list": any(marker in answer for marker in ["1.", "2.", "3.", "-", "•", "第一", "第二"]),
                "has_code": "```" in answer or "def " in answer or "import " in answer,
                "score": 90 if any(marker in answer for marker in ["1.", "2.", "3.", "-", "•", "第一", "第二"]) else 50,
                "detail": "包含列表结构" if any(marker in answer for marker in ["1.", "2.", "3.", "-", "•", "第一", "第二"]) else "缺少结构化格式"
            }
        }
    }

    model = get_default_llm()

    print("\n【交互式自定义评估演示】")
    print("提示：选择评估器，输入问题，系统评估模型输出\n")

    while True:
        # 选择评估器
        print("\n可用评估器：")
        for key, ev in evaluators.items():
            print(f"  {key}. {ev['name']} - {ev['description']}")
        print("\n  0. 退出")

        ev_choice = input("\n选择评估器 (0-4): ").strip()
        if ev_choice == "0":
            print("结束演示")
            break
        if ev_choice not in evaluators:
            print("❌ 无效选择")
            continue

        selected = evaluators[ev_choice]
        print(f"\n已选择：{selected['name']}")

        # 输入问题
        question = input("输入问题：").strip()
        if not question:
            print("请输入有效问题")
            continue

        # 输入评估参考
        if ev_choice == "2":
            ref_input = input("关键词（逗号分隔）：").strip()
            reference = [k.strip() for k in ref_input.split(",") if k.strip()]
        elif ev_choice == "4":
            reference = None
        else:
            reference = input("参考答案（可选，直接回车跳过）：").strip() or ""

        # 调用模型
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个知识问答助手"),
            ("human", "{question}")
        ])
        chain = prompt | model | StrOutputParser()

        try:
            answer = chain.invoke({"question": question})
        except Exception as e:
            answer = f"[调用失败: {e}]"

        # 运行评估器
        result = selected["evaluate"](answer, reference if reference else [])

        # 展示结果
        print(f"\n🤖 模型回答：{answer}")
        print(f"\n📊 {selected['name']} 评估结果：")
        print(f"   评分：{result['score']:.0f}/100")
        print(f"   详情：{result['detail']}")
        for k, v in result.items():
            if k not in ["score", "detail"]:
                print(f"   {k}：{v}")
        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 自定义评估器可以匹配特定业务需求")
    print("   2. 评估器接收回答和参考，返回结构化评分")
    print("   3. 可组合多个评估器进行综合评估")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangSmith 可观测性 - 实战案例")
    print("="*60)
    print("\n本示例演示 LangSmith 可观测性的核心功能（模拟实现）")
    print("\n核心概念：")
    print("  • 追踪（Tracing）：记录 LLM 调用的完整信息")
    print("  • 评估（Evaluation）：衡量模型输出质量")
    print("  • 数据集（Dataset）：管理测试用例")
    print("  • 自定义评估器：匹配业务规则的评估逻辑")
    print("\n应用场景：")
    print("  • 调试排错、质量保障、回归测试、业务定制")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 追踪基础：记录每次调用的追踪信息")
        print("  2. 评估系统：评估模型输出的质量")
        print("  3. 数据集管理：创建和管理测试数据集")
        print("  4. 自定义评估器：创建自定义的评估规则")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_tracing_basics()
        elif choice == "2":
            demo_evaluation()
        elif choice == "3":
            demo_dataset_management()
        elif choice == "4":
            demo_custom_evaluator()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
