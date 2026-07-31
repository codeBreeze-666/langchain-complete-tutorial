"""
LangChain 对话记忆持久化实战案例
==================================

本示例演示四种对话记忆策略及其持久化方法：
- 对话缓冲：保存完整对话历史
- 对话摘要：将长对话压缩为摘要
- 对话窗口：只保留最近 N 轮对话
- 会话管理：多会话切换与持久化存储

实战要点：
- 不同记忆策略的适用场景
- 记忆的序列化与反序列化
- 多会话隔离与切换
- 文件持久化与恢复
"""

import os
import sys
import json
from typing import List, Dict, Optional
from datetime import datetime
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.utils.llm_loader import get_default_llm


# ============================================================
# 对话缓冲记忆 - 保存完整对话历史
# ============================================================

class ConversationBuffer:
    """
    对话缓冲记忆：保存全部对话历史
    
    特点：
    - 简单直接，保留所有上下文
    - 短对话效果最佳
    - 长对话会超出 Token 限制
    """

    def __init__(self):
        self.messages: List = []
        self.system_prompt: Optional[str] = None

    def set_system_prompt(self, prompt: str):
        """设置系统提示"""
        self.system_prompt = prompt

    def add_user_message(self, content: str):
        """添加用户消息"""
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        """添加 AI 消息"""
        self.messages.append(AIMessage(content=content))

    def get_messages(self) -> List:
        """获取完整消息列表（含系统提示）"""
        result = []
        if self.system_prompt:
            result.append(SystemMessage(content=self.system_prompt))
        result.extend(self.messages)
        return result

    def clear(self):
        """清空对话"""
        self.messages = []

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "system_prompt": self.system_prompt,
            "messages": [
                {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
                for m in self.messages
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationBuffer":
        """从字典反序列化"""
        buffer = cls()
        buffer.system_prompt = data.get("system_prompt")
        for msg in data.get("messages", []):
            if msg["role"] == "human":
                buffer.add_user_message(msg["content"])
            else:
                buffer.add_ai_message(msg["content"])
        return buffer


# ============================================================
# 对话摘要记忆 - 将长对话压缩为摘要
# ============================================================

class ConversationSummary:
    """
    对话摘要记忆：将历史对话压缩为摘要
    
    特点：
    - 节省 Token，适合长对话
    - 需要额外的 LLM 调用来生成摘要
    - 会丢失部分细节信息
    """

    def __init__(self):
        self.llm = get_default_llm()
        self.buffer: List = []  # 当前对话缓冲
        self.summary: str = ""  # 已生成的摘要
        self.system_prompt: Optional[str] = None

    def set_system_prompt(self, prompt: str):
        """设置系统提示"""
        self.system_prompt = prompt

    def add_user_message(self, content: str):
        """添加用户消息"""
        self.buffer.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        """添加 AI 消息"""
        self.buffer.append(AIMessage(content=content))

    def summarize(self) -> str:
        """将缓冲区对话压缩为摘要"""
        if not self.buffer:
            return self.summary

        # 构建摘要提示
        context = f"已有摘要：{self.summary}\n\n" if self.summary else ""
        conversation = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in self.buffer
        )

        prompt_text = (
            f"{context}请将以下对话内容压缩为简洁的摘要，"
            f"保留关键信息、用户偏好和重要事实：\n\n{conversation}"
        )

        messages = [HumanMessage(content=prompt_text)]
        response = self.llm.invoke(messages)

        # 更新摘要并清空缓冲
        self.summary = response.content
        self.buffer = []
        return self.summary

    def get_messages(self) -> List:
        """获取消息列表（摘要 + 当前缓冲）"""
        result = []
        if self.system_prompt:
            result.append(SystemMessage(content=self.system_prompt))
        if self.summary:
            result.append(SystemMessage(content=f"对话历史摘要：{self.summary}"))
        result.extend(self.buffer)
        return result

    def clear(self):
        """清空所有记忆"""
        self.buffer = []
        self.summary = ""

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "system_prompt": self.system_prompt,
            "summary": self.summary,
            "buffer": [
                {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
                for m in self.buffer
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationSummary":
        """从字典反序列化"""
        summary = cls()
        summary.system_prompt = data.get("system_prompt")
        summary.summary = data.get("summary", "")
        for msg in data.get("buffer", []):
            if msg["role"] == "human":
                summary.add_user_message(msg["content"])
            else:
                summary.add_ai_message(msg["content"])
        return summary


# ============================================================
# 对话窗口记忆 - 只保留最近 N 轮对话
# ============================================================

class ConversationWindow:
    """
    对话窗口记忆：只保留最近 N 轮对话
    
    特点：
    - 固定窗口大小，Token 消耗可控
    - 保留最新上下文，适合持续对话
    - 会丢失较早的对话内容
    """

    def __init__(self, window_size: int = 4):
        """
        初始化窗口记忆
        
        Args:
            window_size: 保留的对话轮数（一轮 = 一条用户 + 一条 AI）
        """
        self.window_size = window_size
        self.messages: List = []
        self.system_prompt: Optional[str] = None

    def set_system_prompt(self, prompt: str):
        """设置系统提示"""
        self.system_prompt = prompt

    def add_user_message(self, content: str):
        """添加用户消息"""
        self.messages.append(HumanMessage(content=content))
        self._trim()

    def add_ai_message(self, content: str):
        """添加 AI 消息"""
        self.messages.append(AIMessage(content=content))
        self._trim()

    def _trim(self):
        """裁剪超出窗口的消息"""
        max_messages = self.window_size * 2  # 每轮两条消息
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_messages(self) -> List:
        """获取消息列表"""
        result = []
        if self.system_prompt:
            result.append(SystemMessage(content=self.system_prompt))
        result.extend(self.messages)
        return result

    def clear(self):
        """清空对话"""
        self.messages = []

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "window_size": self.window_size,
            "system_prompt": self.system_prompt,
            "messages": [
                {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
                for m in self.messages
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationWindow":
        """从字典反序列化"""
        window = cls(window_size=data.get("window_size", 4))
        window.system_prompt = data.get("system_prompt")
        for msg in data.get("messages", []):
            if msg["role"] == "human":
                window.add_user_message(msg["content"])
            else:
                window.add_ai_message(msg["content"])
        return window


# ============================================================
# 会话管理器 - 支持多会话切换和持久化
# ============================================================

class SessionManager:
    """
    会话管理器：管理多个独立会话，支持持久化存储
    
    特点：
    - 多会话隔离，互不干扰
    - 支持多种记忆策略
    - 文件持久化与恢复
    - 会话元数据追踪
    """

    STORAGE_DIR = "session_data"

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}  # session_id -> session_data
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        os.makedirs(self.STORAGE_DIR, exist_ok=True)

    def create_session(self, session_id: str, memory_type: str = "buffer", **kwargs) -> str:
        """
        创建新会话
        
        Args:
            session_id: 会话 ID
            memory_type: 记忆类型 (buffer / summary / window)
            **kwargs: 记忆策略参数
            
        Returns:
            会话 ID
        """
        if session_id in self.sessions:
            print(f"会话 '{session_id}' 已存在，将切换到该会话")
            return session_id

        # 创建对应的记忆实例
        if memory_type == "buffer":
            memory = ConversationBuffer()
        elif memory_type == "summary":
            memory = ConversationSummary()
        elif memory_type == "window":
            memory = ConversationWindow(window_size=kwargs.get("window_size", 4))
        else:
            raise ValueError(f"不支持的记忆类型: {memory_type}")

        self.sessions[session_id] = {
            "memory": memory,
            "memory_type": memory_type,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": 0,
        }
        print(f"会话 '{session_id}' 已创建 (记忆类型: {memory_type})")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话"""
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[str]:
        """列出所有会话"""
        return list(self.sessions.keys())

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """获取会话信息"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        return {
            "session_id": session_id,
            "memory_type": session["memory_type"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "message_count": session["message_count"],
        }

    def delete_session(self, session_id: str):
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"会话 '{session_id}' 已删除")
        # 同时删除持久化文件
        file_path = os.path.join(self.STORAGE_DIR, f"{session_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)

    def save_session(self, session_id: str):
        """保存会话到文件"""
        session = self.sessions.get(session_id)
        if not session:
            print(f"会话 '{session_id}' 不存在")
            return

        memory = session["memory"]
        data = {
            "session_id": session_id,
            "memory_type": session["memory_type"],
            "created_at": session["created_at"],
            "updated_at": datetime.now().isoformat(),
            "message_count": session["message_count"],
            "memory_data": memory.to_dict(),
        }

        file_path = os.path.join(self.STORAGE_DIR, f"{session_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"会话 '{session_id}' 已保存到 {file_path}")

    def load_session(self, session_id: str) -> bool:
        """从文件加载会话"""
        file_path = os.path.join(self.STORAGE_DIR, f"{session_id}.json")
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return False

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        memory_type = data["memory_type"]
        if memory_type == "buffer":
            memory = ConversationBuffer.from_dict(data["memory_data"])
        elif memory_type == "summary":
            memory = ConversationSummary.from_dict(data["memory_data"])
        elif memory_type == "window":
            memory = ConversationWindow.from_dict(data["memory_data"])
        else:
            print(f"不支持的记忆类型: {memory_type}")
            return False

        self.sessions[session_id] = {
            "memory": memory,
            "memory_type": memory_type,
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
            "message_count": data["message_count"],
        }
        print(f"会话 '{session_id}' 已从文件加载")
        return True

    def save_all(self):
        """保存所有会话"""
        for session_id in self.sessions:
            self.save_session(session_id)

    def list_saved_sessions(self) -> List[str]:
        """列出所有已持久化的会话文件"""
        if not os.path.exists(self.STORAGE_DIR):
            return []
        return [
            f.replace(".json", "")
            for f in os.listdir(self.STORAGE_DIR)
            if f.endswith(".json")
        ]


# ============================================================
# 交互式对话引擎
# ============================================================

def interactive_chat(memory, llm, session_label: str = "对话"):
    """
    通用交互式对话循环
    
    Args:
        memory: 记忆实例（ConversationBuffer / ConversationSummary / ConversationWindow）
        llm: LLM 实例
        session_label: 会话标签（用于显示）
    """
    print(f"\n{'='*60}")
    print(f"  {session_label}")
    print(f"{'='*60}")
    print("输入消息开始对话，输入 'quit' 退出，输入 'clear' 清空记忆")
    print(f"{'='*60}")

    round_count = 0
    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("退出对话")
            break
        if user_input.lower() == "clear":
            memory.clear()
            round_count = 0
            print("记忆已清空")
            continue

        # 记录用户消息
        memory.add_user_message(user_input)

        # 调用 LLM
        messages = memory.get_messages()
        response = llm.invoke(messages)

        # 记录 AI 回复
        memory.add_ai_message(response.content)

        round_count += 1
        print(f"AI: {response.content}")
        print(f"  [当前消息数: {len(memory.messages)}, 对话轮数: {round_count}]")


# ============================================================
# 示例 1：对话缓冲 - 保存完整对话历史
# ============================================================

def demo_conversation_buffer():
    """
    对话缓冲示例：保存完整的对话历史
    
    实战要点：
    - 最简单的记忆策略，保留所有上下文
    - 适合短对话场景（客服、问答等）
    - 长对话会消耗大量 Token，需注意限制
    - 序列化后可持久化到文件
    """
    print("\n" + "=" * 60)
    print("  示例 1：对话缓冲 - 保存完整对话历史")
    print("=" * 60)
    print("""
实战要点：
  1. 最简单的记忆策略，保留所有上下文
  2. 适合短对话场景（客服、问答等）
  3. 长对话会消耗大量 Token，需注意限制
  4. 序列化后可持久化到文件
    """)

    llm = get_default_llm()
    memory = ConversationBuffer()
    memory.set_system_prompt("你是一个友好的助手，记住用户提到的所有信息。")

    interactive_chat(memory, llm, session_label="对话缓冲模式")


# ============================================================
# 示例 2：对话摘要 - 将长对话压缩为摘要
# ============================================================

def demo_conversation_summary():
    """
    对话摘要示例：将长对话压缩为摘要
    
    实战要点：
    - 额外调用 LLM 生成摘要，有额外开销
    - 适合长对话场景（心理咨询、深度访谈等）
    - 摘要会丢失部分细节，重要信息可能被压缩
    - 可设定触发摘要的阈值（如消息数达到 N 条）
    """
    print("\n" + "=" * 60)
    print("  示例 2：对话摘要 - 将长对话压缩为摘要")
    print("=" * 60)
    print("""
实战要点：
  1. 额外调用 LLM 生成摘要，有额外开销
  2. 适合长对话场景（心理咨询、深度访谈等）
  3. 摘要会丢失部分细节，重要信息可能被压缩
  4. 可设定触发摘要的阈值（如消息数达到 N 条）
    """)

    llm = get_default_llm()
    memory = ConversationSummary()
    memory.set_system_prompt("你是一个耐心的助手，善于总结和记忆对话要点。")

    # 摘要触发阈值
    SUMMARIZE_THRESHOLD = 6  # 缓冲区消息数达到此值时触发摘要

    print(f"\n{'='*60}")
    print("  对话摘要模式")
    print(f"{'='*60}")
    print(f"输入消息开始对话，输入 'quit' 退出，输入 'clear' 清空记忆")
    print(f"输入 'summarize' 手动触发摘要，缓冲区达到 {SUMMARIZE_THRESHOLD} 条自动摘要")
    print(f"{'='*60}")

    round_count = 0
    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("退出对话")
            break
        if user_input.lower() == "clear":
            memory.clear()
            round_count = 0
            print("记忆已清空")
            continue
        if user_input.lower() == "summarize":
            summary = memory.summarize()
            print(f"[摘要结果]: {summary}")
            continue

        # 记录用户消息
        memory.add_user_message(user_input)

        # 调用 LLM
        messages = memory.get_messages()
        response = llm.invoke(messages)

        # 记录 AI 回复
        memory.add_ai_message(response.content)

        round_count += 1
        print(f"AI: {response.content}")

        # 检查是否需要自动摘要
        if len(memory.buffer) >= SUMMARIZE_THRESHOLD:
            print("\n[系统] 缓冲区已满，正在生成摘要...")
            memory.summarize()
            print(f"[系统] 摘要已生成，缓冲区已清空")
            print(f"[系统] 当前摘要: {memory.summary[:100]}...")

        # 显示状态
        status_parts = [f"对话轮数: {round_count}"]
        if memory.summary:
            status_parts.append(f"摘要长度: {len(memory.summary)}")
        status_parts.append(f"缓冲消息数: {len(memory.buffer)}")
        print(f"  [{', '.join(status_parts)}]")


# ============================================================
# 示例 3：对话窗口 - 只保留最近 N 轮对话
# ============================================================

def demo_conversation_window():
    """
    对话窗口示例：只保留最近 N 轮对话
    
    实战要点：
    - Token 消耗可控，窗口大小决定上限
    - 适合持续对话场景（日常闲聊、快速问答等）
    - 早期对话会被丢弃，无法回忆远处信息
    - 窗口大小需根据实际场景调整
    """
    print("\n" + "=" * 60)
    print("  示例 3：对话窗口 - 只保留最近 N 轮对话")
    print("=" * 60)
    print("""
实战要点：
  1. Token 消耗可控，窗口大小决定上限
  2. 适合持续对话场景（日常闲聊、快速问答等）
  3. 早期对话会被丢弃，无法回忆远处信息
  4. 窗口大小需根据实际场景调整
    """)

    # 让用户选择窗口大小
    window_input = input("请输入窗口大小（保留最近几轮对话，默认 3）: ").strip()
    try:
        window_size = int(window_input) if window_input else 3
    except ValueError:
        window_size = 3
        print(f"输入无效，使用默认值: {window_size}")

    llm = get_default_llm()
    memory = ConversationWindow(window_size=window_size)
    memory.set_system_prompt("你是一个风趣的助手，用简洁的方式回答问题。")

    print(f"\n{'='*60}")
    print(f"  对话窗口模式（窗口大小: {window_size} 轮）")
    print(f"{'='*60}")
    print("输入消息开始对话，输入 'quit' 退出，输入 'clear' 清空记忆")
    print(f"{'='*60}")

    round_count = 0
    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("退出对话")
            break
        if user_input.lower() == "clear":
            memory.clear()
            round_count = 0
            print("记忆已清空")
            continue

        # 记录用户消息
        memory.add_user_message(user_input)

        # 调用 LLM
        messages = memory.get_messages()
        response = llm.invoke(messages)

        # 记录 AI 回复
        memory.add_ai_message(response.content)

        round_count += 1
        print(f"AI: {response.content}")
        print(f"  [总轮数: {round_count}, 窗口内消息数: {len(memory.messages)}/{window_size * 2}]")


# ============================================================
# 示例 4：会话管理 - 多会话切换和持久化
# ============================================================

def demo_session_management():
    """
    会话管理示例：支持多会话切换和持久化
    
    实战要点：
    - 多会话隔离，适用于多用户或多功能场景
    - 持久化确保应用重启后会话可恢复
    - 不同会话可使用不同记忆策略
    - 元数据管理有助于会话审计与分析
    """
    print("\n" + "=" * 60)
    print("  示例 4：会话管理 - 多会话切换和持久化")
    print("=" * 60)
    print("""
实战要点：
  1. 多会话隔离，适用于多用户或多功能场景
  2. 持久化确保应用重启后会话可恢复
  3. 不同会话可使用不同记忆策略
  4. 元数据管理有助于会话审计与分析
    """)

    manager = SessionManager()
    llm = get_default_llm()
    current_session_id: Optional[str] = None

    while True:
        print(f"\n{'='*60}")
        print("  会话管理菜单")
        print(f"{'='*60}")
        if current_session_id:
            info = manager.get_session_info(current_session_id)
            print(f"  当前会话: {current_session_id} ({info['memory_type']}, 消息数: {info['message_count']})")
        else:
            print("  当前会话: 无")
        print(f"{'='*60}")
        print("  1. 创建新会话")
        print("  2. 切换会话")
        print("  3. 在当前会话中对话")
        print("  4. 保存当前会话")
        print("  5. 从文件加载会话")
        print("  6. 删除会话")
        print("  7. 列出所有会话")
        print("  8. 列出已保存的会话文件")
        print("  0. 退出")
        print(f"{'='*60}")

        choice = input("请选择操作 (0-8): ").strip()

        if choice == "1":
            # 创建新会话
            sid = input("输入会话 ID: ").strip()
            if not sid:
                print("会话 ID 不能为空")
                continue
            print("选择记忆类型: 1-buffer  2-summary  3-window")
            mem_choice = input("请选择 (1-3, 默认 1): ").strip()
            memory_type_map = {"1": "buffer", "2": "summary", "3": "window"}
            memory_type = memory_type_map.get(mem_choice, "buffer")

            kwargs = {}
            if memory_type == "window":
                ws = input("窗口大小 (默认 3): ").strip()
                try:
                    kwargs["window_size"] = int(ws) if ws else 3
                except ValueError:
                    kwargs["window_size"] = 3

            manager.create_session(sid, memory_type, **kwargs)
            current_session_id = sid

        elif choice == "2":
            # 切换会话
            sessions = manager.list_sessions()
            if not sessions:
                print("没有可用的会话，请先创建")
                continue
            print("可用会话:")
            for s in sessions:
                info = manager.get_session_info(s)
                print(f"  - {s} ({info['memory_type']}, 消息数: {info['message_count']})")
            sid = input("输入要切换的会话 ID: ").strip()
            if sid in manager.sessions:
                current_session_id = sid
                print(f"已切换到会话: {sid}")
            else:
                print(f"会话 '{sid}' 不存在")

        elif choice == "3":
            # 在当前会话中对话
            if not current_session_id:
                print("请先创建或切换到一个会话")
                continue

            session = manager.get_session(current_session_id)
            memory = session["memory"]
            info = manager.get_session_info(current_session_id)

            print(f"\n--- 会话 '{current_session_id}' 对话模式 ---")
            print(f"记忆类型: {info['memory_type']}")
            print("输入 'back' 返回菜单，输入 'clear' 清空记忆")

            while True:
                user_input = input("\n你: ").strip()
                if not user_input:
                    continue
                if user_input.lower() == "back":
                    break
                if user_input.lower() == "clear":
                    memory.clear()
                    session["message_count"] = 0
                    print("记忆已清空")
                    continue

                # 记录用户消息
                memory.add_user_message(user_input)

                # 调用 LLM
                messages = memory.get_messages()
                response = llm.invoke(messages)

                # 记录 AI 回复
                memory.add_ai_message(response.content)

                # 更新会话信息
                session["message_count"] += 1
                session["updated_at"] = datetime.now().isoformat()

                print(f"AI: {response.content}")

        elif choice == "4":
            # 保存当前会话
            if not current_session_id:
                print("请先选择一个会话")
                continue
            manager.save_session(current_session_id)

        elif choice == "5":
            # 从文件加载会话
            saved = manager.list_saved_sessions()
            if not saved:
                print("没有已保存的会话文件")
                continue
            print("已保存的会话:")
            for s in saved:
                print(f"  - {s}")
            sid = input("输入要加载的会话 ID: ").strip()
            if sid in saved:
                if manager.load_session(sid):
                    current_session_id = sid
            else:
                print(f"会话文件 '{sid}' 不存在")

        elif choice == "6":
            # 删除会话
            sid = input("输入要删除的会话 ID: ").strip()
            manager.delete_session(sid)
            if sid == current_session_id:
                current_session_id = None

        elif choice == "7":
            # 列出所有会话
            sessions = manager.list_sessions()
            if not sessions:
                print("没有可用的会话")
            else:
                print("所有会话:")
                for s in sessions:
                    info = manager.get_session_info(s)
                    print(f"  - {s}")
                    print(f"    类型: {info['memory_type']}, 消息数: {info['message_count']}")
                    print(f"    创建: {info['created_at']}")
                    print(f"    更新: {info['updated_at']}")

        elif choice == "8":
            # 列出已保存的会话文件
            saved = manager.list_saved_sessions()
            if not saved:
                print("没有已保存的会话文件")
            else:
                print("已保存的会话文件:")
                for s in saved:
                    file_path = os.path.join(manager.STORAGE_DIR, f"{s}.json")
                    size = os.path.getsize(file_path)
                    print(f"  - {s} ({size} 字节)")

        elif choice == "0":
            # 退出前保存
            if manager.sessions:
                save_choice = input("是否保存所有会话? (y/n): ").strip().lower()
                if save_choice == "y":
                    manager.save_all()
            print("再见！")
            break

        else:
            print("无效选项，请重新选择")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "=" * 60)
    print("  LangChain 对话记忆持久化实战案例")
    print("=" * 60)
    print("\n本示例演示四种对话记忆策略及其持久化方法")
    print("\n记忆策略：")
    print("  • 对话缓冲：保存完整对话历史（简单直接）")
    print("  • 对话摘要：将长对话压缩为摘要（节省 Token）")
    print("  • 对话窗口：只保留最近 N 轮对话（可控消耗）")
    print("  • 会话管理：多会话切换与持久化（多用户场景）")

    while True:
        print("\n" + "=" * 60)
        print("请选择要运行的示例：")
        print("=" * 60)
        print("【记忆策略】")
        print("  1. 对话缓冲 - 保存完整对话历史")
        print("  2. 对话摘要 - 将长对话压缩为摘要")
        print("  3. 对话窗口 - 只保留最近 N 轮对话")
        print("  4. 会话管理 - 多会话切换和持久化")
        print("\n【其他】")
        print("  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_conversation_buffer()
        elif choice == "2":
            demo_conversation_summary()
        elif choice == "3":
            demo_conversation_window()
        elif choice == "4":
            demo_session_management()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("无效选项，请重新选择")


if __name__ == "__main__":
    main()
