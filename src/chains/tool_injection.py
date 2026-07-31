"""
LangChain 工具注入 - 实战交互式案例
====================================

本示例演示 LangChain 中工具注入的核心概念

核心概念：
- InjectedState：工具可以访问 Agent 的运行状态
- InjectedStore：工具可以访问共享存储
- 配置注入：工具可以访问运行时配置
- 上下文感知：工具根据上下文动态调整行为

应用场景：
- 状态感知：工具根据对话历史做出决策
- 数据共享：多个工具之间共享数据
- 配置驱动：工具行为随配置变化
- 上下文适应：工具根据运行环境调整输出
"""

import os
import sys
import json
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.tools import tool, InjectedToolArg
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated
from src.utils.llm_loader import get_default_llm


# ============================================================
# 模拟注入框架（当 InjectedToolArg 不可用时的后备方案）
# ============================================================

class SimulatedAgentState:
    """模拟 Agent 运行状态"""

    def __init__(self):
        self.conversation_history = []
        self.current_step = 0
        self.agent_name = "默认助手"
        self.start_time = datetime.now()
        self.metadata = {}

    def add_message(self, role: str, content: str):
        """添加消息到对话历史"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_summary(self) -> str:
        """获取状态摘要"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return (
            f"Agent 状态摘要:\n"
            f"  名称: {self.agent_name}\n"
            f"  当前步骤: {self.current_step}\n"
            f"  对话轮数: {len(self.conversation_history)}\n"
            f"  运行时长: {elapsed:.1f}s"
        )


class SimulatedStore:
    """模拟共享存储（InjectedStore）"""

    def __init__(self):
        self._data = {}

    def set(self, key: str, value):
        """存储数据"""
        self._data[key] = {
            "value": value,
            "updated_at": datetime.now().isoformat()
        }

    def get(self, key: str, default=None):
        """获取数据"""
        entry = self._data.get(key)
        if entry is None:
            return default
        return entry["value"]

    def delete(self, key: str) -> bool:
        """删除数据"""
        if key in self._data:
            del self._data[key]
            return True
        return False

    def list_keys(self) -> list:
        """列出所有键"""
        return list(self._data.keys())

    def get_all(self) -> dict:
        """获取所有数据"""
        return {
            k: {"value": v["value"], "updated_at": v["updated_at"]}
            for k, v in self._data.items()
        }


class SimulatedConfig:
    """模拟运行时配置"""

    def __init__(self):
        self._config = {
            "language": "zh-CN",
            "max_retries": 3,
            "timeout": 30,
            "debug": False,
            "model": "glm-4.7-flash",
            "temperature": 0.7,
            "region": "cn",
        }

    def get(self, key: str, default=None):
        """获取配置项"""
        return self._config.get(key, default)

    def set(self, key: str, value):
        """设置配置项"""
        self._config[key] = value

    def update(self, config_dict: dict):
        """批量更新配置"""
        self._config.update(config_dict)

    def get_all(self) -> dict:
        """获取所有配置"""
        return dict(self._config)


# 全局实例（模拟注入的依赖）
_global_state = SimulatedAgentState()
_global_store = SimulatedStore()
_global_config = SimulatedConfig()


# ============================================================
# 1. 注入状态 - 工具可以访问 Agent 的运行状态
# ============================================================

def _query_with_state(query: str, state: SimulatedAgentState) -> str:
    """带状态查询（内部实现）

    Args:
        query: 查询内容
        state: 注入的 Agent 状态

    Returns:
        查询结果
    """
    state.current_step += 1
    state.add_message("tool", f"执行查询: {query}")

    history_count = len(state.conversation_history)
    elapsed = (datetime.now() - state.start_time).total_seconds()

    # 模拟根据状态做出不同响应
    if history_count <= 2:
        context_hint = "（首次交互，提供详细引导）"
    elif history_count <= 6:
        context_hint = "（多轮对话，理解更精准）"
    else:
        context_hint = "（深度对话，高度上下文感知）"

    result = (
        f"📋 状态感知查询结果\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"查询内容: {query}\n"
        f"当前步骤: 第 {state.current_step} 步\n"
        f"对话轮数: {history_count} 轮\n"
        f"运行时长: {elapsed:.1f}s\n"
        f"上下文模式: {context_hint}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    state.add_message("tool_result", result)
    return result


def demo_injected_state():
    """示例1：注入状态 - 工具可以访问 Agent 的运行状态"""
    print("\n" + "="*60)
    print("示例1：注入状态 (InjectedState)")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - InjectedState 让工具访问 Agent 运行时状态")
    print("   - 工具可以根据对话历史调整行为")
    print("   - 无需显式传递状态参数，框架自动注入")
    print("   - Annotated[type, InjectedToolArg] 声明注入参数")

    state = SimulatedAgentState()
    state.agent_name = input("\n请输入 Agent 名称（直接回车使用默认）: ").strip() or "智能助手"
    state.add_message("system", f"Agent {state.agent_name} 已启动")

    print(f"\n✅ Agent [{state.agent_name}] 已初始化")
    print("\n【交互式状态感知查询】")
    print("提示：输入任意查询，观察工具如何感知状态变化")
    print("输入 '状态' 查看当前 Agent 状态")
    print("输入 '退出' 结束\n")

    while True:
        user_input = input("你的查询: ").strip()

        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效查询")
            continue

        if user_input == '状态':
            print(f"\n{state.get_summary()}\n")
            continue

        # 模拟用户消息进入状态
        state.add_message("user", user_input)

        # 调用带状态注入的工具
        result = _query_with_state(user_input, state)
        print(f"\n{result}\n")

        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. InjectedState 让工具无需参数即可获取运行状态")
    print("   2. 工具可根据步骤数、对话历史调整响应策略")
    print("   3. LangChain 中通过 Annotated[..., InjectedToolArg] 声明")
    print("   4. 状态注入使工具更具上下文感知能力")


# ============================================================
# 2. 注入存储 - 工具可以访问共享存储
# ============================================================

def _store_operation(operation: str, key: str, value: str = None, store: SimulatedStore = None) -> str:
    """共享存储操作（内部实现）

    Args:
        operation: 操作类型 (get/set/delete/list)
        key: 存储键
        value: 存储值（set 操作时需要）
        store: 注入的共享存储

    Returns:
        操作结果
    """
    if operation == "set":
        if value is None:
            return "❌ set 操作需要提供 value 参数"
        store.set(key, value)
        return f"✅ 已存储: {key} = {value}"

    elif operation == "get":
        result = store.get(key)
        if result is None:
            return f"⚠️ 键 '{key}' 不存在"
        return f"📖 读取: {key} = {result}"

    elif operation == "delete":
        if store.delete(key):
            return f"🗑️ 已删除: {key}"
        else:
            return f"⚠️ 键 '{key}' 不存在，无法删除"

    elif operation == "list":
        keys = store.list_keys()
        if not keys:
            return "📂 存储为空"
        all_data = store.get_all()
        lines = ["📂 共享存储内容:"]
        for k, v in all_data.items():
            lines.append(f"  {k} = {v['value']}  (更新于 {v['updated_at'][:19]})")
        return "\n".join(lines)

    else:
        return f"❌ 未知操作: {operation}（支持: get/set/delete/list）"


def demo_injected_store():
    """示例2：注入存储 - 工具可以访问共享存储"""
    print("\n" + "="*60)
    print("示例2：注入存储 (InjectedStore)")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - InjectedStore 让工具访问持久化的共享存储")
    print("   - 多个工具之间可以通过存储共享数据")
    print("   - 存储由框架管理，工具无需关心生命周期")
    print("   - 适合需要跨工具传递数据的场景")

    store = SimulatedStore()

    # 预设一些初始数据
    store.set("用户偏好.语言", "中文")
    store.set("用户偏好.主题", "深色模式")
    store.set("会话.上次访问", datetime.now().strftime("%Y-%m-%d %H:%M"))

    print("\n✅ 共享存储已初始化（含预设数据）")
    print("\n【交互式存储操作】")
    print("可用操作：")
    print("  set <键> <值>  - 存储数据")
    print("  get <键>        - 读取数据")
    print("  delete <键>     - 删除数据")
    print("  list             - 列出所有数据")
    print("输入 '退出' 结束\n")

    while True:
        user_input = input("存储操作: ").strip()

        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效操作")
            continue

        parts = user_input.split(maxsplit=2)
        operation = parts[0].lower()

        if operation == "set":
            if len(parts) < 3:
                print("用法: set <键> <值>")
                continue
            key, value = parts[1], parts[2]
            result = _store_operation("set", key, value, store=store)
        elif operation == "get":
            if len(parts) < 2:
                print("用法: get <键>")
                continue
            result = _store_operation("get", parts[1], store=store)
        elif operation == "delete":
            if len(parts) < 2:
                print("用法: delete <键>")
                continue
            result = _store_operation("delete", parts[1], store=store)
        elif operation == "list":
            result = _store_operation("list", "", store=store)
        else:
            result = f"❌ 未知操作: {operation}（支持: get/set/delete/list）"

        print(f"\n{result}\n")
        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. InjectedStore 提供跨工具的共享存储空间")
    print("   2. 工具通过注入获取存储引用，无需参数传递")
    print("   3. 适合需要工具间数据传递和持久化的场景")
    print("   4. LangChain 中 Store 通过 RunnableConfig 注入")


# ============================================================
# 3. 配置注入 - 工具可以访问运行时配置
# ============================================================

def _config_aware_greet(name: str, config: SimulatedConfig) -> str:
    """配置感知的问候（内部实现）

    Args:
        name: 用户名
        config: 注入的运行时配置

    Returns:
        问候语
    """
    lang = config.get("language", "zh-CN")
    region = config.get("region", "cn")
    debug = config.get("debug", False)
    temperature = config.get("temperature", 0.7)

    # 根据语言配置调整问候语
    greetings = {
        "zh-CN": f"你好，{name}！",
        "en-US": f"Hello, {name}!",
        "ja-JP": f"こんにちは、{name}さん！",
        "ko-KR": f"안녕하세요, {name}님!",
    }
    greeting = greetings.get(lang, f"Hi, {name}!")

    # 根据温度调整语气
    if temperature > 0.9:
        tone = "热情奔放"
    elif temperature > 0.5:
        tone = "温和友好"
    else:
        tone = "严谨专业"

    # 根据地区调整信息
    region_names = {"cn": "中国", "us": "美国", "jp": "日本", "kr": "韩国"}
    region_name = region_names.get(region, region)

    result = (
        f"🌏 配置感知问候\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{greeting}\n"
        f"当前语言: {lang}\n"
        f"所在地区: {region_name}\n"
        f"语气风格: {tone} (temperature={temperature})\n"
    )

    if debug:
        result += (
            f"\n🐛 调试信息:\n"
            f"  max_retries: {config.get('max_retries')}\n"
            f"  timeout: {config.get('timeout')}s\n"
            f"  model: {config.get('model')}\n"
        )

    result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    return result


def demo_config_injection():
    """示例3：配置注入 - 工具可以访问运行时配置"""
    print("\n" + "="*60)
    print("示例3：配置注入 (Config Injection)")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 通过 RunnableConfig 将配置传递给工具")
    print("   - 工具根据配置动态调整行为")
    print("   - 无需修改工具代码即可改变输出")
    print("   - 适合多环境、多租户场景")

    config = SimulatedConfig()

    print("\n【当前配置】")
    for k, v in config.get_all().items():
        print(f"  {k}: {v}")

    print("\n【交互式配置注入】")
    print("可用命令：")
    print("  greet <名字>          - 根据当前配置生成问候")
    print("  set <配置项> <值>     - 修改配置")
    print("  config                - 查看当前配置")
    print("  preset <预设名>       - 切换预设配置 (cn/us/jp/kr)")
    print("输入 '退出' 结束\n")

    presets = {
        "cn": {"language": "zh-CN", "region": "cn", "temperature": 0.7},
        "us": {"language": "en-US", "region": "us", "temperature": 0.8},
        "jp": {"language": "ja-JP", "region": "jp", "temperature": 0.6},
        "kr": {"language": "ko-KR", "region": "kr", "temperature": 0.65},
    }

    while True:
        user_input = input("命令: ").strip()

        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效命令")
            continue

        parts = user_input.split(maxsplit=2)
        cmd = parts[0].lower()

        if cmd == "greet":
            if len(parts) < 2:
                print("用法: greet <名字>")
                continue
            name = parts[1]
            result = _config_aware_greet(name, config)
            print(f"\n{result}\n")

        elif cmd == "set":
            if len(parts) < 3:
                print("用法: set <配置项> <值>")
                continue
            key, raw_val = parts[1], parts[2]
            # 自动类型转换
            try:
                if raw_val.lower() == 'true':
                    val = True
                elif raw_val.lower() == 'false':
                    val = False
                elif '.' in raw_val:
                    val = float(raw_val)
                else:
                    val = int(raw_val)
            except ValueError:
                val = raw_val
            config.set(key, val)
            print(f"✅ 配置已更新: {key} = {val}")

        elif cmd == "config":
            print("\n【当前配置】")
            for k, v in config.get_all().items():
                print(f"  {k}: {v}")
            print()

        elif cmd == "preset":
            if len(parts) < 2:
                print(f"用法: preset <预设名>（可选: {', '.join(presets.keys())}）")
                continue
            preset_name = parts[1].lower()
            if preset_name not in presets:
                print(f"❌ 未知预设: {preset_name}（可选: {', '.join(presets.keys())}）")
                continue
            config.update(presets[preset_name])
            print(f"✅ 已切换到预设 [{preset_name}]，当前配置:")
            for k, v in config.get_all().items():
                print(f"  {k}: {v}")

        else:
            print(f"❌ 未知命令: {cmd}（支持: greet/set/config/preset）")

        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 配置注入让同一工具在不同配置下行为不同")
    print("   2. RunnableConfig 是 LangChain 配置传递的标准方式")
    print("   3. 适合多语言、多环境、多租户等场景")
    print("   4. 工具通过 config 参数接收注入的配置")


# ============================================================
# 4. 上下文感知工具 - 根据上下文动态调整行为
# ============================================================

def _context_aware_search(query: str, state: SimulatedAgentState,
                          store: SimulatedStore, config: SimulatedConfig) -> str:
    """上下文感知搜索（内部实现）

    综合利用注入的状态、存储和配置，提供上下文感知的结果

    Args:
        query: 搜索查询
        state: 注入的 Agent 状态
        store: 注入的共享存储
        config: 注入的运行时配置

    Returns:
        上下文感知的搜索结果
    """
    state.current_step += 1
    state.add_message("user", query)

    # 从配置获取语言偏好
    lang = config.get("language", "zh-CN")
    debug = config.get("debug", False)

    # 从存储获取用户偏好
    user_theme = store.get("用户偏好.主题", "默认")
    user_lang_pref = store.get("用户偏好.语言", "未设置")

    # 根据对话历史判断上下文深度
    history_count = len(state.conversation_history)
    if history_count <= 2:
        depth = "浅层"
        detail_level = "详细解释"
    elif history_count <= 6:
        depth = "中层"
        detail_level = "适度精简"
    else:
        depth = "深层"
        detail_level = "高度概括"

    # 模拟搜索结果（根据语言返回不同示例）
    mock_results = {
        "zh-CN": [
            f"📘 关于「{query}」的中文资料",
            f"📗 「{query}」实战案例分析",
            f"📙 「{query}」最佳实践指南",
        ],
        "en-US": [
            f"📘 English resources about '{query}'",
            f"📗 Case studies on '{query}'",
            f"📙 Best practices for '{query}'",
        ],
    }
    results = mock_results.get(lang, mock_results["zh-CN"])

    # 组装上下文感知的结果
    output = (
        f"🔍 上下文感知搜索结果\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"查询: {query}\n"
        f"上下文深度: {depth}（对话轮数: {history_count}）\n"
        f"输出风格: {detail_level}\n"
        f"\n搜索结果:"
    )
    for i, r in enumerate(results[:2 if depth == "深层" else 3], 1):
        output += f"\n  {i}. {r}"

    output += f"\n\n用户偏好: 主题={user_theme}, 语言偏好={user_lang_pref}"

    if debug:
        output += (
            f"\n\n🐛 调试信息:"
            f"\n  Agent: {state.agent_name}"
            f"\n  步骤: {state.current_step}"
            f"\n  配置: lang={lang}, temp={config.get('temperature')}"
        )

    output += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 将搜索历史记录到存储
    store.set(f"搜索历史.步骤{state.current_step}", query)
    state.add_message("tool_result", output)

    return output


def demo_context_aware_tool():
    """示例4：上下文感知工具 - 根据上下文动态调整行为"""
    print("\n" + "="*60)
    print("示例4：上下文感知工具 (Context-Aware Tool)")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 综合利用 State/Store/Config 三种注入")
    print("   - 工具根据上下文动态调整输出详略")
    print("   - 搜索结果受用户偏好和语言配置影响")
    print("   - 是工具注入能力的综合实战体现")

    # 初始化三种注入依赖
    state = SimulatedAgentState()
    state.agent_name = "上下文助手"
    store = SimulatedStore()
    config = SimulatedConfig()

    # 预设偏好
    store.set("用户偏好.主题", "深色模式")
    store.set("用户偏好.语言", "中文")

    print("\n✅ 已初始化: Agent状态 + 共享存储 + 运行配置")
    print("\n【交互式上下文感知搜索】")
    print("可用命令：")
    print("  search <查询>           - 执行上下文感知搜索")
    print("  set-store <键> <值>     - 修改用户偏好（存储）")
    print("  set-config <键> <值>    - 修改运行时配置")
    print("  info                    - 查看当前上下文信息")
    print("  debug on/off            - 开关调试模式")
    print("输入 '退出' 结束\n")

    while True:
        user_input = input("命令: ").strip()

        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not user_input:
            print("请输入有效命令")
            continue

        parts = user_input.split(maxsplit=2)
        cmd = parts[0].lower()

        if cmd == "search":
            if len(parts) < 2:
                print("用法: search <查询内容>")
                continue
            query = parts[1]
            result = _context_aware_search(query, state, store, config)
            print(f"\n{result}\n")

        elif cmd == "set-store":
            if len(parts) < 3:
                print("用法: set-store <键> <值>")
                continue
            store.set(parts[1], parts[2])
            print(f"✅ 存储已更新: {parts[1]} = {parts[2]}")

        elif cmd == "set-config":
            if len(parts) < 3:
                print("用法: set-config <键> <值>")
                continue
            key, raw_val = parts[1], parts[2]
            try:
                if raw_val.lower() == 'true':
                    val = True
                elif raw_val.lower() == 'false':
                    val = False
                elif '.' in raw_val:
                    val = float(raw_val)
                else:
                    val = int(raw_val)
            except ValueError:
                val = raw_val
            config.set(key, val)
            print(f"✅ 配置已更新: {key} = {val}")

        elif cmd == "info":
            print("\n【当前上下文信息】")
            print(f"  Agent状态:")
            print(f"    名称: {state.agent_name}")
            print(f"    步骤: {state.current_step}")
            print(f"    对话轮数: {len(state.conversation_history)}")
            print(f"  共享存储:")
            for k in store.list_keys():
                print(f"    {k} = {store.get(k)}")
            print(f"  运行配置:")
            for k, v in config.get_all().items():
                print(f"    {k}: {v}")
            print()

        elif cmd == "debug":
            if len(parts) < 2 or parts[1].lower() not in ('on', 'off'):
                print("用法: debug on/off")
                continue
            config.set("debug", parts[1].lower() == "on")
            print(f"✅ 调试模式: {'开启' if config.get('debug') else '关闭'}")

        else:
            print(f"❌ 未知命令: {cmd}（支持: search/set-store/set-config/info/debug）")

        print("-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 上下文感知工具综合运用多种注入机制")
    print("   2. 工具可根据对话深度自动调整输出详略")
    print("   3. 用户偏好和配置影响工具行为")
    print("   4. 这是构建智能 Agent 工具的核心设计模式")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain 工具注入 - 实战案例")
    print("="*60)
    print("\n本示例演示工具注入的核心概念")
    print("\n核心概念：")
    print("  • InjectedState：工具访问 Agent 运行状态")
    print("  • InjectedStore：工具访问共享存储")
    print("  • 配置注入：工具访问运行时配置")
    print("  • 上下文感知：综合运用多种注入")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 注入状态 - 工具访问 Agent 状态")
        print("  2. 注入存储 - 工具访问共享存储")
        print("  3. 配置注入 - 工具访问运行时配置")
        print("  4. 上下文感知 - 综合运用多种注入")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_injected_state()
        elif choice == "2":
            demo_injected_store()
        elif choice == "3":
            demo_config_injection()
        elif choice == "4":
            demo_context_aware_tool()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
