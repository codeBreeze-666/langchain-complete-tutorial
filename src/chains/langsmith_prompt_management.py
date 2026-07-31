"""
LangSmith Prompt 管理 - 实战交互式案例
========================================

本示例演示 LangSmith Prompt 管理功能的核心概念和使用方法

核心概念：
- Prompt版本管理：追踪 Prompt 的变化，每次修改自动保存版本
- A/B测试：对比不同 Prompt 的效果，选出最优版本
- 团队协作：多用户共享 Prompt 模板，统一管理
- 回滚机制：回滚到之前的 Prompt 版本，快速恢复

应用场景：
- 版本控制：追踪 Prompt 的变化历史
- A/B测试：对比不同 Prompt 的效果
- 团队协作：多人共享 Prompt 模板
- 回滚机制：回滚到之前的版本
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
# Prompt 版本管理存储（模拟 LangSmith Prompt Hub）
# ============================================================

class PromptVersion:
    """Prompt 版本"""

    def __init__(self, prompt_name: str, template: str, commit_message: str = "",
                 author: str = "anonymous", tags: list = None):
        self.id = f"v-{uuid.uuid4().hex[:8]}"
        self.prompt_name = prompt_name
        self.template = template
        self.commit_message = commit_message
        self.author = author
        self.tags = tags or []
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.is_active = False  # 是否为当前活跃版本

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "prompt_name": self.prompt_name,
            "template": self.template,
            "commit_message": self.commit_message,
            "author": self.author,
            "tags": self.tags,
            "created_at": self.created_at,
            "is_active": self.is_active,
        }


class PromptStore:
    """Prompt 存储管理"""

    _prompts: dict = {}  # {prompt_name: [PromptVersion, ...]}

    @classmethod
    def create(cls, name: str, template: str, commit_message: str = "",
               author: str = "anonymous", tags: list = None) -> PromptVersion:
        """创建 Prompt（第一个版本）"""
        version = PromptVersion(name, template, commit_message or "初始版本", author, tags)
        version.is_active = True
        cls._prompts[name] = [version]
        return version

    @classmethod
    def update(cls, name: str, template: str, commit_message: str = "",
               author: str = "anonymous", tags: list = None) -> Optional[PromptVersion]:
        """更新 Prompt（创建新版本）"""
        if name not in cls._prompts:
            return None
        # 将之前的活跃版本标记为非活跃
        for v in cls._prompts[name]:
            v.is_active = False
        version = PromptVersion(name, template, commit_message or f"更新版本 v{len(cls._prompts[name])+1}", author, tags)
        version.is_active = True
        cls._prompts[name].append(version)
        return version

    @classmethod
    def get_versions(cls, name: str) -> list:
        """获取 Prompt 的所有版本"""
        return cls._prompts.get(name, [])

    @classmethod
    def get_active(cls, name: str) -> Optional[PromptVersion]:
        """获取当前活跃版本"""
        versions = cls._prompts.get(name, [])
        for v in versions:
            if v.is_active:
                return v
        return versions[-1] if versions else None

    @classmethod
    def get_by_version_id(cls, name: str, version_id: str) -> Optional[PromptVersion]:
        """按版本 ID 获取"""
        for v in cls._prompts.get(name, []):
            if v.id == version_id:
                return v
        return None

    @classmethod
    def rollback(cls, name: str, version_id: str) -> Optional[PromptVersion]:
        """回滚到指定版本"""
        target = cls.get_by_version_id(name, version_id)
        if not target:
            return None
        # 将所有版本标记为非活跃
        for v in cls._prompts[name]:
            v.is_active = False
        # 创建回滚版本（基于目标版本的内容）
        rollback_version = PromptVersion(
            name, target.template,
            f"回滚到 {version_id} ({target.commit_message})",
            "system", target.tags
        )
        rollback_version.is_active = True
        cls._prompts[name].append(rollback_version)
        return rollback_version

    @classmethod
    def list_all(cls) -> dict:
        """列出所有 Prompt"""
        return cls._prompts

    @classmethod
    def delete(cls, name: str) -> bool:
        """删除 Prompt"""
        if name in cls._prompts:
            del cls._prompts[name]
            return True
        return False


# ============================================================
# 示例1: Prompt版本管理 - 版本控制
# ============================================================

def demo_prompt_versioning():
    """示例1：Prompt版本管理 - 版本控制（用户输入Prompt，自动保存版本）"""
    print("\n" + "="*60)
    print("示例1：Prompt版本管理 - 版本控制")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - Prompt版本管理：追踪 Prompt 的变化")
    print("   - 每次修改自动保存为新版本")
    print("   - 可查看历史版本和变更记录")
    print("\n📊 应用场景：")
    print("   - 追踪 Prompt 的优化过程")
    print("   - 记录每次修改的原因")
    print("   - 对比不同版本的效果")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    prompt_name = "qa_assistant"

    print("\n【交互式 Prompt 版本管理演示】")
    print("提示：输入系统提示词，系统自动保存版本并测试效果")
    print("输入 '退出' 结束\n")

    while True:
        print(f"\n当前 Prompt: {prompt_name}")
        versions = PromptStore.get_versions(prompt_name)
        if versions:
            active = PromptStore.get_active(prompt_name)
            print(f"当前活跃版本: {active.id} ({active.commit_message})")
            print(f"版本数: {len(versions)}")
        else:
            print("尚未创建，请输入第一个版本")

        print(f"\n操作菜单：")
        print("  1. 创建/更新 Prompt")
        print("  2. 查看版本历史")
        print("  3. 测试当前 Prompt")
        print("  4. 切换到其他 Prompt")
        print("\n  0. 退出")

        choice = input("\n请选择 (0-4): ").strip()

        if choice == "0":
            print("结束演示")
            break

        elif choice == "1":
            print(f"\n当前活跃版本模板：")
            active = PromptStore.get_active(prompt_name)
            if active:
                print(f"  {active.template[:100]}...")
            template = input("\n输入新的系统提示词：").strip()
            if not template:
                print("❌ 提示词不能为空")
                continue
            commit_msg = input("提交说明（可选）：").strip() or f"更新版本"
            author = input("作者（可选，默认anonymous）：").strip() or "anonymous"
            tags_str = input("标签（逗号分隔，可选）：").strip()
            tags = [t.strip() for t in tags_str.split(",")] if tags_str else []

            if not versions:
                version = PromptStore.create(prompt_name, template, commit_msg, author, tags)
            else:
                version = PromptStore.update(prompt_name, template, commit_msg, author, tags)

            print(f"✅ 版本已保存：{version.id} ({version.commit_message})")

        elif choice == "2":
            versions = PromptStore.get_versions(prompt_name)
            if not versions:
                print(f"❌ Prompt '{prompt_name}' 暂无版本")
                continue
            print(f"\n📋 版本历史：")
            print("="*60)
            for v in versions:
                icon = "🟢" if v.is_active else "⚪"
                print(f"  {icon} {v.id} | {v.created_at} | {v.author} | {v.commit_message}")
                print(f"     模板：{v.template[:80]}...")
                if v.tags:
                    print(f"     标签：{', '.join(v.tags)}")
                print()
            print("="*60)

        elif choice == "3":
            active = PromptStore.get_active(prompt_name)
            if not active:
                print(f"❌ Prompt '{prompt_name}' 暂无版本")
                continue
            question = input("输入测试问题：").strip()
            if not question:
                print("❌ 问题不能为空")
                continue
            prompt = ChatPromptTemplate.from_messages([
                ("system", active.template),
                ("human", "{question}")
            ])
            chain = prompt | model | StrOutputParser()
            try:
                answer = chain.invoke({"question": question})
                print(f"\n🤖 回答：{answer}")
            except Exception as e:
                print(f"❌ 调用失败: {e}")

        elif choice == "4":
            new_name = input("输入 Prompt 名称：").strip()
            if new_name:
                prompt_name = new_name
                print(f"✅ 已切换到 Prompt: {prompt_name}")

        else:
            print("❌ 无效选项")

    print("\n✅ 实战要点总结：")
    print("   1. 每次修改 Prompt 自动保存为新版本")
    print("   2. 可查看完整的版本历史和变更记录")
    print("   3. 真实 LangSmith Prompt Hub 提供 Git 式的版本管理")
    print("   4. 可通过 commit message 记录每次修改的原因")


# ============================================================
# 示例2: A/B测试 - 对比测试
# ============================================================

def demo_prompt_ab_testing():
    """示例2：A/B测试 - 对比测试（用户输入不同Prompt，对比效果）"""
    print("\n" + "="*60)
    print("示例2：A/B测试 - 对比测试")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - A/B测试：对比不同 Prompt 的效果")
    print("   - 使用相同的输入，对比不同 Prompt 的输出")
    print("   - 从多个维度评估哪个 Prompt 更好")
    print("\n📊 应用场景：")
    print("   - 对比不同提示词策略")
    print("   - 选择最优的 Prompt 版本")
    print("   - 量化优化效果")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    print("\n【交互式 Prompt A/B 测试演示】")
    print("提示：输入两个不同的系统提示词，对比它们的效果")
    print("输入 '退出' 结束\n")

    # 预设的 A/B 测试模板
    preset_pairs = {
        "1": {
            "name": "简洁 vs 详细",
            "prompt_a": "你是一个助手，用最简洁的语言回答问题，不超过50字。",
            "prompt_b": "你是一个助手，用详细的语言回答问题，包含解释和示例。"
        },
        "2": {
            "name": "专业 vs 通俗",
            "prompt_a": "你是一个专业助手，用专业的术语和严谨的语言回答问题。",
            "prompt_b": "你是一个友好助手，用通俗易懂的语言回答问题，像和朋友聊天一样。"
        },
        "3": {
            "name": "结构化 vs 自由",
            "prompt_a": "你是一个助手，请按照以下格式回答：\n1. 直接答案\n2. 解释说明\n3. 示例",
            "prompt_b": "你是一个助手，自由地回答问题，不要有格式限制。"
        },
    }

    while True:
        print("\n选择测试方式：")
        print("  1. 简洁 vs 详细")
        print("  2. 专业 vs 通俗")
        print("  3. 结构化 vs 自由")
        print("  4. 自定义 Prompt A/B 测试")
        print("\n  0. 退出")

        choice = input("\n请选择 (0-4): ").strip()
        if choice == "0":
            print("结束演示")
            break

        if choice in preset_pairs:
            pair = preset_pairs[choice]
            prompt_a_text = pair["prompt_a"]
            prompt_b_text = pair["prompt_b"]
            print(f"\n📋 已选择：{pair['name']}")
        elif choice == "4":
            print("\n输入 Prompt A（系统提示词）：")
            prompt_a_text = input("A: ").strip()
            if not prompt_a_text:
                print("❌ 不能为空")
                continue
            print("输入 Prompt B（系统提示词）：")
            prompt_b_text = input("B: ").strip()
            if not prompt_b_text:
                print("❌ 不能为空")
                continue
        else:
            print("❌ 无效选项")
            continue

        question = input("\n输入测试问题：").strip()
        if not question:
            print("❌ 问题不能为空")
            continue

        # 执行 Prompt A
        prompt_a = ChatPromptTemplate.from_messages([
            ("system", prompt_a_text),
            ("human", "{question}")
        ])
        chain_a = prompt_a | model | StrOutputParser()
        start_a = time.time()
        try:
            answer_a = chain_a.invoke({"question": question})
            duration_a = (time.time() - start_a) * 1000
        except Exception as e:
            answer_a = f"[失败: {e}]"
            duration_a = 0

        # 执行 Prompt B
        prompt_b = ChatPromptTemplate.from_messages([
            ("system", prompt_b_text),
            ("human", "{question}")
        ])
        chain_b = prompt_b | model | StrOutputParser()
        start_b = time.time()
        try:
            answer_b = chain_b.invoke({"question": question})
            duration_b = (time.time() - start_b) * 1000
        except Exception as e:
            answer_b = f"[失败: {e}]"
            duration_b = 0

        # 对比结果
        print(f"\n📊 Prompt A/B 测试对比：")
        print("="*60)
        print(f"  问题：{question}")
        print("="*60)

        print(f"\n  🅰️ Prompt A")
        print(f"  {'─'*55}")
        print(f"  提示词：{prompt_a_text[:80]}...")
        print(f"  回答：{answer_a[:200]}{'...' if len(answer_a) > 200 else ''}")
        print(f"  字数：{len(answer_a)} | 耗时：{duration_a:.0f}ms")

        print(f"\n  🅱️ Prompt B")
        print(f"  {'─'*55}")
        print(f"  提示词：{prompt_b_text[:80]}...")
        print(f"  回答：{answer_b[:200]}{'...' if len(answer_b) > 200 else ''}")
        print(f"  字数：{len(answer_b)} | 耗时：{duration_b:.0f}ms")

        # 对比分析
        print(f"\n  📈 对比分析：")
        print(f"  {'─'*55}")
        print(f"  字数差异：{len(answer_b) - len(answer_a):+d} 字")
        print(f"  耗时差异：{duration_b - duration_a:+.0f}ms")

        # 简单评估
        has_structure_a = any(m in answer_a for m in ["1.", "2.", "-", "•", "首先"])
        has_structure_b = any(m in answer_b for m in ["1.", "2.", "-", "•", "首先"])
        print(f"  结构化：A={'有' if has_structure_a else '无'} | B={'有' if has_structure_b else '无'}")

        # 人工选择
        print(f"\n  🏆 请选择更优的 Prompt：")
        print(f"  1. Prompt A 更优")
        print(f"  2. Prompt B 更优")
        print(f"  3. 两者相当")
        user_choice = input("  你的选择 (1-3): ").strip()

        if user_choice == "1":
            winner = "Prompt A"
            winning_template = prompt_a_text
        elif user_choice == "2":
            winner = "Prompt B"
            winning_template = prompt_b_text
        else:
            winner = "两者相当"
            winning_template = prompt_a_text

        print(f"\n  ✅ 你的选择：{winner}")

        # 保存胜出的 Prompt
        if winner != "两者相当":
            save = input(f"  是否将 {winner} 保存为优化版本？(y/n): ").strip().lower()
            if save == "y":
                version = PromptStore.create(
                    "ab_test_winner", winning_template,
                    f"A/B测试胜出 - {winner}", "ab_test"
                )
                print(f"  ✅ 已保存：{version.id}")

        print("="*60)

    print("\n✅ 实战要点总结：")
    print("   1. A/B测试用相同输入对比不同 Prompt 的效果")
    print("   2. 从字数、耗时、结构化等维度对比")
    print("   3. 人工选择更优的 Prompt 版本")
    print("   4. 真实 LangSmith 支持自动化 A/B 测试")


# ============================================================
# 示例3: 团队协作 - Prompt共享
# ============================================================

def demo_team_collaboration():
    """示例3：团队协作 - Prompt共享（多用户共享Prompt模板）"""
    print("\n" + "="*60)
    print("示例3：团队协作 - Prompt共享")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 团队协作：多用户共享 Prompt 模板")
    print("   - 统一管理团队的 Prompt 资源")
    print("   - 真实 LangSmith Prompt Hub 支持团队共享")
    print("\n📊 应用场景：")
    print("   - 团队共享 Prompt 模板")
    print("   - 统一管理 Prompt 版本")
    print("   - 避免重复创建 Prompt")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    # 模拟团队成员
    team_members = ["Alice", "Bob", "Charlie"]
    current_user = "Alice"

    # 预创建一些团队共享的 Prompt
    PromptStore.create("客服助手", "你是一个专业的客服助手，耐心回答用户问题，语气友好。", "初始版本", "Alice", ["客服", "基础"])
    PromptStore.create("技术文档助手", "你是一个技术文档助手，用专业的语言解释技术概念，包含代码示例。", "初始版本", "Bob", ["技术", "文档"])
    PromptStore.create("创意写作助手", "你是一个创意写作助手，帮助用户进行创意写作，提供灵感和建议。", "初始版本", "Charlie", ["创意", "写作"])

    print(f"\n当前用户：{current_user}")
    print("\n【交互式团队协作演示】")

    while True:
        print(f"\n{'─'*60}")
        print(f"团队协作菜单（当前用户：{current_user}）：")
        print("  1. 浏览团队 Prompt 库")
        print("  2. 使用团队 Prompt")
        print("  3. 贡献新 Prompt")
        print("  4. 更新团队 Prompt")
        print("  5. 切换用户")
        print("\n  0. 退出")
        print(f"{'─'*60}")

        choice = input("请选择 (0-5): ").strip()

        if choice == "0":
            print("结束演示")
            break

        elif choice == "1":
            # 浏览团队 Prompt 库
            all_prompts = PromptStore.list_all()
            if not all_prompts:
                print("📭 暂无团队 Prompt")
                continue
            print(f"\n📚 团队 Prompt 库（共 {len(all_prompts)} 个）：")
            print("="*60)
            for name, versions in all_prompts.items():
                active = PromptStore.get_active(name)
                if active:
                    print(f"  📝 {name}")
                    print(f"     活跃版本：{active.id} | 作者：{active.author}")
                    print(f"     提交说明：{active.commit_message}")
                    print(f"     模板：{active.template[:60]}...")
                    print(f"     标签：{', '.join(active.tags) if active.tags else '无'}")
                    print(f"     版本数：{len(versions)}")
                    print()

        elif choice == "2":
            # 使用团队 Prompt
            all_prompts = PromptStore.list_all()
            if not all_prompts:
                print("📭 暂无团队 Prompt")
                continue
            print("可用的 Prompt：")
            for name in all_prompts:
                print(f"  - {name}")
            prompt_name = input("\n输入要使用的 Prompt 名称：").strip()
            active = PromptStore.get_active(prompt_name)
            if not active:
                print(f"❌ Prompt '{prompt_name}' 不存在")
                continue
            question = input("输入问题：").strip()
            if not question:
                print("❌ 问题不能为空")
                continue
            prompt = ChatPromptTemplate.from_messages([
                ("system", active.template),
                ("human", "{question}")
            ])
            chain = prompt | model | StrOutputParser()
            try:
                answer = chain.invoke({"question": question})
                print(f"\n🤖 回答（使用 {prompt_name} v{active.id}）：{answer}")
            except Exception as e:
                print(f"❌ 调用失败: {e}")

        elif choice == "3":
            # 贡献新 Prompt
            name = input("Prompt 名称：").strip()
            if not name:
                print("❌ 名称不能为空")
                continue
            template = input("系统提示词模板：").strip()
            if not template:
                print("❌ 模板不能为空")
                continue
            commit_msg = input("提交说明：").strip() or "初始版本"
            tags_str = input("标签（逗号分隔）：").strip()
            tags = [t.strip() for t in tags_str.split(",")] if tags_str else []
            version = PromptStore.create(name, template, commit_msg, current_user, tags)
            print(f"✅ Prompt '{name}' 已创建：{version.id} (作者：{current_user})")

        elif choice == "4":
            # 更新团队 Prompt
            all_prompts = PromptStore.list_all()
            if not all_prompts:
                print("📭 暂无团队 Prompt")
                continue
            print("可更新的 Prompt：")
            for name in all_prompts:
                active = PromptStore.get_active(name)
                print(f"  - {name} (当前版本：{active.id}，作者：{active.author})")
            prompt_name = input("\n输入要更新的 Prompt 名称：").strip()
            versions = PromptStore.get_versions(prompt_name)
            if not versions:
                print(f"❌ Prompt '{prompt_name}' 不存在")
                continue
            active = PromptStore.get_active(prompt_name)
            print(f"当前模板：{active.template}")
            new_template = input("输入新的系统提示词：").strip()
            if not new_template:
                print("❌ 模板不能为空")
                continue
            commit_msg = input("提交说明：").strip() or f"更新版本"
            version = PromptStore.update(prompt_name, new_template, commit_msg, current_user)
            print(f"✅ Prompt '{prompt_name}' 已更新：{version.id} (作者：{current_user})")

        elif choice == "5":
            # 切换用户
            print(f"当前用户：{current_user}")
            print("可选用户：")
            for m in team_members:
                print(f"  - {m}")
            new_user = input("输入用户名：").strip()
            if new_user in team_members:
                current_user = new_user
                print(f"✅ 已切换到用户：{current_user}")
            else:
                print("❌ 用户不存在")

        else:
            print("❌ 无效选项")

    print("\n✅ 实战要点总结：")
    print("   1. 团队协作让多人共享 Prompt 模板")
    print("   2. 每次修改记录作者和提交说明")
    print("   3. 真实 LangSmith Prompt Hub 支持团队共享和权限管理")
    print("   4. 避免重复创建 Prompt，提高团队效率")


# ============================================================
# 示例4: 回滚机制 - 版本回滚
# ============================================================

def demo_prompt_rollback():
    """示例4：回滚机制 - 版本回滚（回滚到之前的Prompt版本）"""
    print("\n" + "="*60)
    print("示例4：回滚机制 - 版本回滚")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 回滚机制：回滚到之前的 Prompt 版本")
    print("   - 当新版本效果不好时，快速恢复")
    print("   - 真实 LangSmith Prompt Hub 支持一键回滚")
    print("\n📊 应用场景：")
    print("   - 新版本效果不好，需要回滚")
    print("   - 误操作修改了 Prompt")
    print("   - 对比不同版本的效果")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    # 预创建一个有多个版本的 Prompt
    prompt_name = "翻译助手"
    PromptStore.create(prompt_name, "你是一个翻译助手，将中文翻译成英文。", "v1.0 基础版", "Alice")
    PromptStore.update(prompt_name, "你是一个专业翻译助手，将中文翻译成英文，保持原文风格和语气。", "v2.0 增加风格保持", "Alice")
    PromptStore.update(prompt_name, "你是一个高级翻译助手，将中文翻译成英文，保持原文风格，并提供翻译注释说明翻译选择。", "v3.0 增加注释（可能过度）", "Bob")

    print(f"\n📋 已预创建 Prompt '{prompt_name}' 的3个版本")
    print("模拟场景：v3.0 版本增加了翻译注释，但可能过度复杂，需要回滚")

    print("\n【交互式回滚演示】")

    while True:
        active = PromptStore.get_active(prompt_name)
        versions = PromptStore.get_versions(prompt_name)

        print(f"\n{'─'*60}")
        print(f"当前活跃版本：{active.id} ({active.commit_message})")
        print(f"模板：{active.template[:80]}...")
        print(f"总版本数：{len(versions)}")

        print(f"\n操作菜单：")
        print("  1. 查看版本历史")
        print("  2. 测试当前版本")
        print("  3. 测试指定版本")
        print("  4. 回滚到指定版本")
        print("  5. 更新当前 Prompt")
        print("\n  0. 退出")
        print(f"{'─'*60}")

        choice = input("请选择 (0-5): ").strip()

        if choice == "0":
            print("结束演示")
            break

        elif choice == "1":
            # 查看版本历史
            print(f"\n📋 版本历史：")
            print("="*60)
            for v in versions:
                icon = "🟢" if v.is_active else "⚪"
                print(f"  {icon} {v.id} | {v.created_at} | {v.author}")
                print(f"     提交说明：{v.commit_message}")
                print(f"     模板：{v.template[:80]}...")
                print()
            print("="*60)

        elif choice == "2":
            # 测试当前版本
            question = input("输入测试文本（中文）：").strip()
            if not question:
                print("❌ 文本不能为空")
                continue
            prompt = ChatPromptTemplate.from_messages([
                ("system", active.template),
                ("human", "{question}")
            ])
            chain = prompt | model | StrOutputParser()
            try:
                answer = chain.invoke({"question": question})
                print(f"\n🤖 翻译结果（{active.id}）：{answer}")
            except Exception as e:
                print(f"❌ 调用失败: {e}")

        elif choice == "3":
            # 测试指定版本
            version_id = input("输入版本 ID：").strip()
            target = PromptStore.get_by_version_id(prompt_name, version_id)
            if not target:
                print(f"❌ 版本 '{version_id}' 不存在")
                continue
            question = input("输入测试文本（中文）：").strip()
            if not question:
                print("❌ 文本不能为空")
                continue
            prompt = ChatPromptTemplate.from_messages([
                ("system", target.template),
                ("human", "{question}")
            ])
            chain = prompt | model | StrOutputParser()
            try:
                answer = chain.invoke({"question": question})
                print(f"\n🤖 翻译结果（{target.id} - {target.commit_message}）：{answer}")
            except Exception as e:
                print(f"❌ 调用失败: {e}")

        elif choice == "4":
            # 回滚到指定版本
            version_id = input("输入要回滚到的版本 ID：").strip()
            target = PromptStore.get_by_version_id(prompt_name, version_id)
            if not target:
                print(f"❌ 版本 '{version_id}' 不存在")
                continue
            confirm = input(f"确认回滚到 {version_id} ({target.commit_message})？(y/n): ").strip().lower()
            if confirm != "y":
                print("已取消")
                continue
            rollback_version = PromptStore.rollback(prompt_name, version_id)
            print(f"✅ 已回滚到版本 {version_id}")
            print(f"   新版本：{rollback_version.id} ({rollback_version.commit_message})")
            print(f"   模板：{rollback_version.template[:80]}...")

        elif choice == "5":
            # 更新当前 Prompt
            new_template = input("输入新的系统提示词：").strip()
            if not new_template:
                print("❌ 模板不能为空")
                continue
            commit_msg = input("提交说明：").strip() or "更新版本"
            version = PromptStore.update(prompt_name, new_template, commit_msg, "current_user")
            print(f"✅ 已更新：{version.id} ({version.commit_message})")

        else:
            print("❌ 无效选项")

    print("\n✅ 实战要点总结：")
    print("   1. 回滚机制可快速恢复到之前的版本")
    print("   2. 回滚会创建新版本，不会丢失历史")
    print("   3. 可先测试指定版本，再决定是否回滚")
    print("   4. 真实 LangSmith Prompt Hub 支持一键回滚")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangSmith Prompt 管理 - 实战案例")
    print("="*60)
    print("\n本示例演示 LangSmith Prompt 管理功能的核心概念和使用方法")

    mode = "真实模式" if has_langsmith_key() else "模拟模式"
    print(f"\n当前模式：{mode}")
    if not has_langsmith_key():
        print("提示：配置 LANGSMITH_API_KEY 可连接真实 LangSmith 服务")

    print("\n核心概念：")
    print("  • Prompt版本管理: 追踪Prompt的变化")
    print("  • A/B测试: 对比不同Prompt的效果")
    print("  • 团队协作: 多人共享Prompt")
    print("  • 回滚机制: 回滚到之前的版本")

    print("\n应用场景：")
    print("  • 版本控制、A/B测试、团队协作、版本回滚")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. Prompt版本管理 - 版本控制")
        print("  2. A/B测试 - 对比测试")
        print("  3. 团队协作 - Prompt共享")
        print("  4. 回滚机制 - 版本回滚")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_prompt_versioning()
        elif choice == "2":
            demo_prompt_ab_testing()
        elif choice == "3":
            demo_team_collaboration()
        elif choice == "4":
            demo_prompt_rollback()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
