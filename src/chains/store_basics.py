"""
LangChain 跨会话存储（Store）实战案例
=====================================

本示例演示四种跨会话存储的典型用法：
- 键值存储：存储和检索键值对数据
- 用户偏好：记住用户的偏好设置
- 共享上下文：多个 Agent 共享的上下文信息
- 持久化存储：将数据保存到文件

实战要点：
- Store 是跨会话的持久化存储，不同于对话记忆（Memory）
- Store 适合存储结构化数据，如配置、偏好、共享状态
- Store 支持命名空间隔离，避免数据冲突
- Store 可以持久化到文件，实现跨会话数据共享
"""

import os
import sys
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_loader import get_default_llm


# ============================================================
# 基础键值存储 - InMemoryStore 的简化实现
# ============================================================

class KeyValueStore:
    """
    键值存储：支持命名空间隔离的键值对存储

    特点：
    - 简单的键值对读写操作
    - 支持命名空间隔离不同业务数据
    - 支持查询、删除、列举操作
    - 可序列化为 JSON 进行持久化
    """

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}  # namespace -> {key -> value}

    def _ensure_namespace(self, namespace: str):
        """确保命名空间存在"""
        if namespace not in self._data:
            self._data[namespace] = {}

    def put(self, namespace: str, key: str, value: Any):
        """
        写入键值对

        Args:
            namespace: 命名空间
            key: 键
            value: 值（支持任意可 JSON 序列化的对象）
        """
        self._ensure_namespace(namespace)
        self._data[namespace][key] = value

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """
        读取键值对

        Args:
            namespace: 命名空间
            key: 键
            default: 默认值

        Returns:
            对应的值，不存在则返回默认值
        """
        return self._data.get(namespace, {}).get(key, default)

    def delete(self, namespace: str, key: str) -> bool:
        """
        删除键值对

        Returns:
            是否成功删除
        """
        if namespace in self._data and key in self._data[namespace]:
            del self._data[namespace][key]
            return True
        return False

    def list_keys(self, namespace: str) -> List[str]:
        """列出命名空间下所有键"""
        return list(self._data.get(namespace, {}).keys())

    def list_namespaces(self) -> List[str]:
        """列出所有命名空间"""
        return list(self._data.keys())

    def get_namespace(self, namespace: str) -> Dict[str, Any]:
        """获取命名空间下所有键值对"""
        return dict(self._data.get(namespace, {}))

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {"namespaces": self._data}

    @classmethod
    def from_dict(cls, data: Dict) -> "KeyValueStore":
        """从字典反序列化"""
        store = cls()
        store._data = data.get("namespaces", {})
        return store


# ============================================================
# 用户偏好存储 - 基于 Store 的偏好管理
# ============================================================

class UserPreferenceStore:
    """
    用户偏好存储：记住用户的偏好设置

    特点：
    - 按用户 ID 隔离偏好数据
    - 支持偏好分类（语言、风格、主题等）
    - LLM 可自动从偏好中提取信息
    - 支持偏好查询与更新
    """

    NAMESPACE = "user_preferences"

    def __init__(self, store: KeyValueStore):
        self.store = store
        self.llm = get_default_llm()

    def set_preference(self, user_id: str, category: str, value: str):
        """
        设置用户偏好

        Args:
            user_id: 用户 ID
            category: 偏好类别（如 language, style, theme）
            value: 偏好值
        """
        key = f"{user_id}:{category}"
        self.store.put(self.NAMESPACE, key, {
            "value": value,
            "updated_at": datetime.now().isoformat(),
        })

    def get_preference(self, user_id: str, category: str, default: str = None) -> Optional[str]:
        """
        获取用户偏好

        Args:
            user_id: 用户 ID
            category: 偏好类别
            default: 默认值

        Returns:
            偏好值，不存在则返回默认值
        """
        key = f"{user_id}:{category}"
        data = self.store.get(self.NAMESPACE, key)
        if data:
            return data.get("value", default)
        return default

    def get_all_preferences(self, user_id: str) -> Dict[str, str]:
        """获取用户所有偏好"""
        result = {}
        for key in self.store.list_keys(self.NAMESPACE):
            if key.startswith(f"{user_id}:"):
                category = key.split(":", 1)[1]
                data = self.store.get(self.NAMESPACE, key)
                if data:
                    result[category] = data.get("value", "")
        return result

    def remove_preference(self, user_id: str, category: str) -> bool:
        """删除用户偏好"""
        key = f"{user_id}:{category}"
        return self.store.delete(self.NAMESPACE, key)

    def build_preference_prompt(self, user_id: str) -> str:
        """构建偏好上下文提示"""
        prefs = self.get_all_preferences(user_id)
        if not prefs:
            return ""
        lines = [f"- {k}: {v}" for k, v in prefs.items()]
        return "用户偏好设置：\n" + "\n".join(lines)

    def extract_preferences_from_text(self, user_id: str, text: str):
        """
        使用 LLM 从用户输入中提取偏好并保存

        Args:
            user_id: 用户 ID
            text: 用户输入的文本
        """
        prompt = ChatPromptTemplate.from_template(
            "从以下用户输入中提取偏好设置。"
            "请以 JSON 格式输出，键为偏好类别，值为偏好内容。"
            "例如：{{\"language\": \"中文\", \"style\": \"简洁\"}}\n\n"
            "如果没有可提取的偏好，输出空 JSON：{{}}\n\n"
            "用户输入：{text}"
        )
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"text": text})

        # 尝试解析 JSON
        try:
            # 去除可能的 markdown 代码块标记
            clean = result.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()

            prefs = json.loads(clean)
            for category, value in prefs.items():
                self.set_preference(user_id, category, str(value))
                print(f"  [已保存偏好] {category}: {value}")
        except json.JSONDecodeError:
            print(f"  [提示] 未能从输入中提取到结构化偏好")


# ============================================================
# 共享上下文存储 - 多 Agent 共享的上下文信息
# ============================================================

class SharedContextStore:
    """
    共享上下文存储：多个 Agent 共享的上下文信息

    特点：
    - 不同 Agent 可以读写同一份上下文
    - 支持上下文频道隔离（如项目频道、任务频道）
    - 支持追加式上下文（如日志、进度）
    - LLM 可基于共享上下文生成更智能的回复
    """

    NAMESPACE = "shared_context"

    def __init__(self, store: KeyValueStore):
        self.store = store
        self.llm = get_default_llm()

    def set_context(self, channel: str, key: str, value: str):
        """
        设置频道中的上下文

        Args:
            channel: 频道名称
            key: 上下文键
            value: 上下文值
        """
        store_key = f"{channel}:{key}"
        self.store.put(self.NAMESPACE, store_key, {
            "value": value,
            "updated_at": datetime.now().isoformat(),
        })

    def get_context(self, channel: str, key: str, default: str = None) -> Optional[str]:
        """获取频道中的上下文"""
        store_key = f"{channel}:{key}"
        data = self.store.get(self.NAMESPACE, store_key)
        if data:
            return data.get("value", default)
        return default

    def append_context(self, channel: str, key: str, entry: str):
        """
        追加式上下文（在已有内容后追加新条目）

        Args:
            channel: 频道名称
            key: 上下文键
            entry: 要追加的条目
        """
        existing = self.get_context(channel, key, "")
        timestamp = datetime.now().strftime("%H:%M:%S")
        new_entry = f"[{timestamp}] {entry}"
        if existing:
            updated = existing + "\n" + new_entry
        else:
            updated = new_entry
        self.set_context(channel, key, updated)

    def get_channel_context(self, channel: str) -> Dict[str, str]:
        """获取频道中所有上下文"""
        result = {}
        for key in self.store.list_keys(self.NAMESPACE):
            if key.startswith(f"{channel}:"):
                context_key = key.split(":", 1)[1]
                data = self.store.get(self.NAMESPACE, key)
                if data:
                    result[context_key] = data.get("value", "")
        return result

    def list_channels(self) -> List[str]:
        """列出所有频道"""
        channels = set()
        for key in self.store.list_keys(self.NAMESPACE):
            channel = key.split(":", 1)[0]
            channels.add(channel)
        return sorted(channels)

    def build_context_prompt(self, channel: str) -> str:
        """构建频道上下文提示"""
        ctx = self.get_channel_context(channel)
        if not ctx:
            return ""
        lines = [f"- {k}: {v}" for k, v in ctx.items()]
        return f"频道 [{channel}] 的共享上下文：\n" + "\n".join(lines)

    def chat_with_context(self, channel: str, user_input: str) -> str:
        """
        基于共享上下文进行对话

        Args:
            channel: 频道名称
            user_input: 用户输入

        Returns:
            AI 回复
        """
        context = self.build_context_prompt(channel)
        if context:
            template = (
                "{context}\n\n"
                "基于以上共享上下文，回答用户的问题：\n"
                "{question}"
            )
        else:
            template = "回答用户的问题：\n{question}"

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": user_input})


# ============================================================
# 持久化存储 - 文件系统持久化
# ============================================================

class PersistentStore:
    """
    持久化存储：将数据保存到文件

    特点：
    - 基于 KeyValueStore 的持久化封装
    - 自动保存到 JSON 文件
    - 启动时自动加载已有数据
    - 支持多文件分片存储
    """

    STORAGE_DIR = "store_data"

    def __init__(self, store: Optional[KeyValueStore] = None):
        self.store = store or KeyValueStore()
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        os.makedirs(self.STORAGE_DIR, exist_ok=True)

    def _get_file_path(self, namespace: str) -> str:
        """获取命名空间对应的文件路径"""
        safe_name = namespace.replace("/", "_").replace("\\", "_")
        return os.path.join(self.STORAGE_DIR, f"{safe_name}.json")

    def save_namespace(self, namespace: str):
        """
        将命名空间数据保存到文件

        Args:
            namespace: 命名空间名称
        """
        data = self.store.get_namespace(namespace)
        file_path = self._get_file_path(namespace)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  [持久化] 命名空间 '{namespace}' 已保存到 {file_path}")

    def load_namespace(self, namespace: str) -> bool:
        """
        从文件加载命名空间数据

        Args:
            namespace: 命名空间名称

        Returns:
            是否成功加载
        """
        file_path = self._get_file_path(namespace)
        if not os.path.exists(file_path):
            print(f"  [持久化] 文件不存在: {file_path}")
            return False

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key, value in data.items():
            self.store.put(namespace, key, value)

        print(f"  [持久化] 命名空间 '{namespace}' 已从文件加载")
        return True

    def save_all(self):
        """保存所有命名空间"""
        for namespace in self.store.list_namespaces():
            self.save_namespace(namespace)

    def list_saved_namespaces(self) -> List[str]:
        """列出所有已持久化的命名空间文件"""
        if not os.path.exists(self.STORAGE_DIR):
            return []
        return [
            f.replace(".json", "")
            for f in os.listdir(self.STORAGE_DIR)
            if f.endswith(".json")
        ]

    def get_file_info(self, namespace: str) -> Optional[Dict]:
        """获取命名空间文件信息"""
        file_path = self._get_file_path(namespace)
        if not os.path.exists(file_path):
            return None
        stat = os.stat(file_path)
        return {
            "file_path": file_path,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }


# ============================================================
# 示例 1：键值存储 - 存储和检索键值对数据
# ============================================================

def demo_key_value_store():
    """
    键值存储示例：存储和检索键值对数据

    实战要点：
    - Store 是最基础的跨会话存储抽象
    - 命名空间隔离不同业务的数据
    - 适合存储配置、缓存、元数据等结构化数据
    - 是构建更复杂存储（偏好、上下文）的基础
    """
    print("\n" + "=" * 60)
    print("  示例 1：键值存储 - 存储和检索键值对数据")
    print("=" * 60)
    print("""
实战要点：
  1. Store 是最基础的跨会话存储抽象
  2. 命名空间隔离不同业务的数据
  3. 适合存储配置、缓存、元数据等结构化数据
  4. 是构建更复杂存储（偏好、上下文）的基础
    """)

    store = KeyValueStore()

    while True:
        print(f"\n{'='*60}")
        print("  键值存储操作菜单")
        print(f"{'='*60}")
        print("  1. 写入键值对")
        print("  2. 读取键值对")
        print("  3. 删除键值对")
        print("  4. 列出命名空间下所有键")
        print("  5. 列出所有命名空间")
        print("  6. 查看命名空间所有数据")
        print("  0. 返回主菜单")
        print(f"{'='*60}")

        choice = input("请选择操作 (0-6): ").strip()

        if choice == "1":
            namespace = input("命名空间: ").strip()
            key = input("键: ").strip()
            value = input("值: ").strip()
            if namespace and key:
                store.put(namespace, key, value)
                print(f"  [已写入] {namespace}/{key} = {value}")
            else:
                print("  [错误] 命名空间和键不能为空")

        elif choice == "2":
            namespace = input("命名空间: ").strip()
            key = input("键: ").strip()
            value = store.get(namespace, key)
            if value is not None:
                print(f"  [读取] {namespace}/{key} = {value}")
            else:
                print(f"  [未找到] {namespace}/{key}")

        elif choice == "3":
            namespace = input("命名空间: ").strip()
            key = input("键: ").strip()
            if store.delete(namespace, key):
                print(f"  [已删除] {namespace}/{key}")
            else:
                print(f"  [未找到] {namespace}/{key}")

        elif choice == "4":
            namespace = input("命名空间: ").strip()
            keys = store.list_keys(namespace)
            if keys:
                print(f"  命名空间 '{namespace}' 下的键:")
                for k in keys:
                    val = store.get(namespace, k)
                    print(f"    - {k}: {val}")
            else:
                print(f"  命名空间 '{namespace}' 为空或不存在")

        elif choice == "5":
            namespaces = store.list_namespaces()
            if namespaces:
                print("  所有命名空间:")
                for ns in namespaces:
                    key_count = len(store.list_keys(ns))
                    print(f"    - {ns} ({key_count} 个键)")
            else:
                print("  暂无命名空间")

        elif choice == "6":
            namespace = input("命名空间: ").strip()
            data = store.get_namespace(namespace)
            if data:
                print(f"  命名空间 '{namespace}' 的所有数据:")
                for k, v in data.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  命名空间 '{namespace}' 为空或不存在")

        elif choice == "0":
            break

        else:
            print("  无效选项，请重新选择")


# ============================================================
# 示例 2：用户偏好 - 记住用户的偏好设置
# ============================================================

def demo_user_preferences():
    """
    用户偏好示例：记住用户的偏好设置

    实战要点：
    - 偏好存储是 Store 的典型应用场景
    - LLM 可自动从对话中提取偏好并保存
    - 偏好作为上下文注入提示词，实现个性化回复
    - 偏好数据可持久化，实现跨会话的个性化体验
    """
    print("\n" + "=" * 60)
    print("  示例 2：用户偏好 - 记住用户的偏好设置")
    print("=" * 60)
    print("""
实战要点：
  1. 偏好存储是 Store 的典型应用场景
  2. LLM 可自动从对话中提取偏好并保存
  3. 偏好作为上下文注入提示词，实现个性化回复
  4. 偏好数据可持久化，实现跨会话的个性化体验
    """)

    store = KeyValueStore()
    pref_store = UserPreferenceStore(store)

    user_id = input("请输入你的用户 ID（直接回车使用默认 'user_001'）: ").strip()
    if not user_id:
        user_id = "user_001"

    print(f"\n  当前用户: {user_id}")

    while True:
        print(f"\n{'='*60}")
        print(f"  用户偏好操作菜单（用户: {user_id}）")
        print(f"{'='*60}")
        print("  1. 手动设置偏好")
        print("  2. 查看所有偏好")
        print("  3. AI 自动提取偏好（输入一段话让 AI 识别偏好）")
        print("  4. 基于偏好进行对话")
        print("  5. 删除偏好")
        print("  0. 返回主菜单")
        print(f"{'='*60}")

        choice = input("请选择操作 (0-5): ").strip()

        if choice == "1":
            category = input("偏好类别（如 language, style, theme）: ").strip()
            value = input("偏好值: ").strip()
            if category and value:
                pref_store.set_preference(user_id, category, value)
                print(f"  [已保存] {category}: {value}")
            else:
                print("  [错误] 类别和值不能为空")

        elif choice == "2":
            prefs = pref_store.get_all_preferences(user_id)
            if prefs:
                print(f"  用户 '{user_id}' 的偏好:")
                for k, v in prefs.items():
                    print(f"    - {k}: {v}")
            else:
                print(f"  用户 '{user_id}' 暂无偏好设置")

        elif choice == "3":
            text = input("请输入一段话（AI 将自动识别偏好）: ").strip()
            if text:
                print("  [AI 正在提取偏好...]")
                pref_store.extract_preferences_from_text(user_id, text)
            else:
                print("  [错误] 输入不能为空")

        elif choice == "4":
            pref_prompt = pref_store.build_preference_prompt(user_id)
            if pref_prompt:
                print(f"  当前偏好上下文:\n  {pref_prompt}")
            else:
                print("  当前无偏好设置，对话将使用默认风格")

            print("\n  输入 'back' 返回菜单")
            while True:
                user_input = input("\n你: ").strip()
                if not user_input:
                    continue
                if user_input.lower() == "back":
                    break

                # 基于偏好构建对话
                pref_context = pref_store.build_preference_prompt(user_id)
                if pref_context:
                    template = (
                        "{preferences}\n\n"
                        "请根据以上用户偏好，回答用户的问题：\n"
                        "{question}"
                    )
                else:
                    template = "回答用户的问题：\n{question}"

                prompt = ChatPromptTemplate.from_template(template)
                llm = get_default_llm()
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({
                    "preferences": pref_context,
                    "question": user_input,
                })
                print(f"AI: {response}")

        elif choice == "5":
            category = input("要删除的偏好类别: ").strip()
            if pref_store.remove_preference(user_id, category):
                print(f"  [已删除] 偏好 '{category}'")
            else:
                print(f"  [未找到] 偏好 '{category}'")

        elif choice == "0":
            break

        else:
            print("  无效选项，请重新选择")


# ============================================================
# 示例 3：共享上下文 - 多 Agent 共享的上下文信息
# ============================================================

def demo_shared_context():
    """
    共享上下文示例：多个 Agent 共享的上下文信息

    实战要点：
    - 共享上下文是 Agent 协作的基础设施
    - 不同 Agent 通过频道读写同一份上下文
    - 追加式上下文适合记录日志、进度等增量信息
    - LLM 可基于共享上下文做出更智能的决策
    """
    print("\n" + "=" * 60)
    print("  示例 3：共享上下文 - 多 Agent 共享的上下文信息")
    print("=" * 60)
    print("""
实战要点：
  1. 共享上下文是 Agent 协作的基础设施
  2. 不同 Agent 通过频道读写同一份上下文
  3. 追加式上下文适合记录日志、进度等增量信息
  4. LLM 可基于共享上下文做出更智能的决策
    """)

    store = KeyValueStore()
    ctx_store = SharedContextStore(store)

    channel = input("请输入频道名称（直接回车使用默认 'project_alpha'）: ").strip()
    if not channel:
        channel = "project_alpha"

    print(f"\n  当前频道: {channel}")

    while True:
        print(f"\n{'='*60}")
        print(f"  共享上下文操作菜单（频道: {channel}）")
        print(f"{'='*60}")
        print("  1. 设置上下文")
        print("  2. 追加上下文（增量记录）")
        print("  3. 查看频道所有上下文")
        print("  4. 基于共享上下文对话")
        print("  5. 切换频道")
        print("  6. 列出所有频道")
        print("  0. 返回主菜单")
        print(f"{'='*60}")

        choice = input("请选择操作 (0-6): ").strip()

        if choice == "1":
            key = input("上下文键: ").strip()
            value = input("上下文值: ").strip()
            if key and value:
                ctx_store.set_context(channel, key, value)
                print(f"  [已设置] {channel}/{key} = {value}")
            else:
                print("  [错误] 键和值不能为空")

        elif choice == "2":
            key = input("上下文键: ").strip()
            entry = input("要追加的内容: ").strip()
            if key and entry:
                ctx_store.append_context(channel, key, entry)
                updated = ctx_store.get_context(channel, key, "")
                print(f"  [已追加] {channel}/{key}:")
                for line in updated.split("\n"):
                    print(f"    {line}")
            else:
                print("  [错误] 键和内容不能为空")

        elif choice == "3":
            ctx = ctx_store.get_channel_context(channel)
            if ctx:
                print(f"  频道 '{channel}' 的所有上下文:")
                for k, v in ctx.items():
                    print(f"    [{k}]")
                    for line in v.split("\n"):
                        print(f"      {line}")
            else:
                print(f"  频道 '{channel}' 暂无上下文")

        elif choice == "4":
            ctx_prompt = ctx_store.build_context_prompt(channel)
            if ctx_prompt:
                print(f"  当前共享上下文:\n  {ctx_prompt}")
            else:
                print("  当前频道无上下文信息")

            print("\n  输入 'back' 返回菜单")
            while True:
                user_input = input("\n你: ").strip()
                if not user_input:
                    continue
                if user_input.lower() == "back":
                    break

                response = ctx_store.chat_with_context(channel, user_input)
                print(f"AI: {response}")

        elif choice == "5":
            new_channel = input("输入新频道名称: ").strip()
            if new_channel:
                channel = new_channel
                print(f"  [已切换] 当前频道: {channel}")
            else:
                print("  [错误] 频道名称不能为空")

        elif choice == "6":
            channels = ctx_store.list_channels()
            if channels:
                print("  所有频道:")
                for ch in channels:
                    ctx = ctx_store.get_channel_context(ch)
                    print(f"    - {ch} ({len(ctx)} 个上下文)")
            else:
                print("  暂无频道")

        elif choice == "0":
            break

        else:
            print("  无效选项，请重新选择")


# ============================================================
# 示例 4：持久化存储 - 将数据保存到文件
# ============================================================

def demo_persistent_store():
    """
    持久化存储示例：将数据保存到文件

    实战要点：
    - 持久化是 Store 从内存走向生产的关键步骤
    - 按命名空间分文件存储，避免单文件过大
    - 启动时自动加载，关闭时自动保存
    - 文件格式使用 JSON，便于查看和调试
    """
    print("\n" + "=" * 60)
    print("  示例 4：持久化存储 - 将数据保存到文件")
    print("=" * 60)
    print("""
实战要点：
  1. 持久化是 Store 从内存走向生产的关键步骤
  2. 按命名空间分文件存储，避免单文件过大
  3. 启动时自动加载，关闭时自动保存
  4. 文件格式使用 JSON，便于查看和调试
    """)

    store = KeyValueStore()
    persistent = PersistentStore(store)

    # 自动加载已有的持久化数据
    saved = persistent.list_saved_namespaces()
    if saved:
        print(f"  发现 {len(saved)} 个已保存的命名空间文件:")
        for ns in saved:
            info = persistent.get_file_info(ns)
            size = info["size"] if info else 0
            print(f"    - {ns} ({size} 字节)")
        load_choice = input("\n  是否加载所有已保存的数据? (y/n, 默认 y): ").strip().lower()
        if load_choice != "n":
            for ns in saved:
                persistent.load_namespace(ns)
            print("  [已加载] 所有已保存的数据")

    while True:
        print(f"\n{'='*60}")
        print("  持久化存储操作菜单")
        print(f"{'='*60}")
        print("  1. 写入数据")
        print("  2. 读取数据")
        print("  3. 保存命名空间到文件")
        print("  4. 从文件加载命名空间")
        print("  5. 保存所有数据到文件")
        print("  6. 查看命名空间数据")
        print("  7. 列出已保存的文件")
        print("  0. 返回主菜单")
        print(f"{'='*60}")

        choice = input("请选择操作 (0-7): ").strip()

        if choice == "1":
            namespace = input("命名空间: ").strip()
            key = input("键: ").strip()
            value = input("值: ").strip()
            if namespace and key:
                store.put(namespace, key, value)
                print(f"  [已写入] {namespace}/{key} = {value}")
            else:
                print("  [错误] 命名空间和键不能为空")

        elif choice == "2":
            namespace = input("命名空间: ").strip()
            key = input("键: ").strip()
            value = store.get(namespace, key)
            if value is not None:
                print(f"  [读取] {namespace}/{key} = {value}")
            else:
                print(f"  [未找到] {namespace}/{key}")

        elif choice == "3":
            namespace = input("要保存的命名空间: ").strip()
            if namespace:
                if store.list_keys(namespace):
                    persistent.save_namespace(namespace)
                else:
                    print(f"  [错误] 命名空间 '{namespace}' 为空或不存在")
            else:
                print("  [错误] 命名空间不能为空")

        elif choice == "4":
            namespace = input("要加载的命名空间: ").strip()
            if namespace:
                persistent.load_namespace(namespace)
            else:
                print("  [错误] 命名空间不能为空")

        elif choice == "5":
            if store.list_namespaces():
                persistent.save_all()
            else:
                print("  [错误] 暂无数据可保存")

        elif choice == "6":
            namespace = input("命名空间: ").strip()
            data = store.get_namespace(namespace)
            if data:
                print(f"  命名空间 '{namespace}' 的数据:")
                for k, v in data.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  命名空间 '{namespace}' 为空或不存在")

        elif choice == "7":
            saved = persistent.list_saved_namespaces()
            if saved:
                print("  已保存的文件:")
                for ns in saved:
                    info = persistent.get_file_info(ns)
                    if info:
                        print(f"    - {ns}")
                        print(f"      路径: {info['file_path']}")
                        print(f"      大小: {info['size']} 字节")
                        print(f"      修改: {info['modified']}")
            else:
                print("  暂无已保存的文件")

        elif choice == "0":
            # 退出前提示保存
            if store.list_namespaces():
                save_choice = input("  是否保存所有数据? (y/n, 默认 n): ").strip().lower()
                if save_choice == "y":
                    persistent.save_all()
            break

        else:
            print("  无效选项，请重新选择")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "=" * 60)
    print("  LangChain 跨会话存储（Store）实战案例")
    print("=" * 60)
    print("\n本示例演示四种跨会话存储的典型用法")
    print("\n存储类型：")
    print("  • 键值存储：存储和检索键值对数据（基础存储）")
    print("  • 用户偏好：记住用户的偏好设置（个性化体验）")
    print("  • 共享上下文：多个 Agent 共享的上下文信息（协作基础）")
    print("  • 持久化存储：将数据保存到文件（生产必备）")

    while True:
        print("\n" + "=" * 60)
        print("请选择要运行的示例：")
        print("=" * 60)
        print("【存储类型】")
        print("  1. 键值存储 - 存储和检索键值对数据")
        print("  2. 用户偏好 - 记住用户的偏好设置")
        print("  3. 共享上下文 - 多个 Agent 共享的上下文信息")
        print("  4. 持久化存储 - 将数据保存到文件")
        print("\n【其他】")
        print("  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_key_value_store()
        elif choice == "2":
            demo_user_preferences()
        elif choice == "3":
            demo_shared_context()
        elif choice == "4":
            demo_persistent_store()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("无效选项，请重新选择")


if __name__ == "__main__":
    main()
