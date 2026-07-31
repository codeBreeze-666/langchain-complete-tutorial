"""
LangSmith 监控（Monitoring）- 实战交互式案例
==============================================

本示例演示 LangSmith 监控功能的核心概念和使用方法

核心概念：
- 监控（Monitoring）：实时监控 API 调用的延迟、错误率、Token 消耗
- 告警（Alerting）：配置告警规则，异常时自动通知
- 成本追踪（Cost Tracking）：追踪 Token 消耗和成本
- 仪表盘（Dashboard）：数据可视化，展示监控数据

应用场景：
- 生产监控：实时监控 API 调用状态
- 异常告警：错误率或延迟异常时自动通知
- 成本追踪：追踪 Token 消耗和成本
- 性能仪表盘：数据可视化展示
"""

import os
import sys
import json
import time
import uuid
import random
from datetime import datetime, timedelta
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
# 监控数据存储（模拟 LangSmith 后端）
# ============================================================

class MetricPoint:
    """监控指标数据点"""

    def __init__(self, metric_name: str, value: float, tags: dict = None):
        self.timestamp = datetime.now()
        self.metric_name = metric_name
        self.value = value
        self.tags = tags or {}

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "metric_name": self.metric_name,
            "value": self.value,
            "tags": self.tags,
        }


class AlertRule:
    """告警规则"""

    def __init__(self, name: str, metric: str, condition: str, threshold: float,
                 message: str = "", enabled: bool = True):
        self.id = f"alert-{uuid.uuid4().hex[:8]}"
        self.name = name
        self.metric = metric
        self.condition = condition  # "gt", "lt", "gte", "lte"
        self.threshold = threshold
        self.message = message
        self.enabled = enabled
        self.triggered_count = 0
        self.last_triggered: Optional[str] = None

    def check(self, value: float) -> bool:
        """检查是否触发告警"""
        if not self.enabled:
            return False
        if self.condition == "gt" and value > self.threshold:
            return True
        elif self.condition == "lt" and value < self.threshold:
            return True
        elif self.condition == "gte" and value >= self.threshold:
            return True
        elif self.condition == "lte" and value <= self.threshold:
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "metric": self.metric,
            "condition": self.condition,
            "threshold": self.threshold,
            "message": self.message,
            "enabled": self.enabled,
            "triggered_count": self.triggered_count,
            "last_triggered": self.last_triggered,
        }


class MonitoringStore:
    """监控数据存储"""

    _metrics: list = []
    _alerts: list = []
    _alert_history: list = []
    _cost_records: list = []

    # Token 计费（模拟价格）
    TOKEN_PRICING = {
        "input": 0.001 / 1000,    # 每千Token 0.001元
        "output": 0.002 / 1000,   # 每千Token 0.002元
    }

    @classmethod
    def record_metric(cls, metric_name: str, value: float, tags: dict = None):
        """记录指标"""
        point = MetricPoint(metric_name, value, tags)
        cls._metrics.append(point)
        # 检查告警
        for alert in cls._alerts:
            if alert.metric == metric_name and alert.check(value):
                alert.triggered_count += 1
                alert.last_triggered = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cls._alert_history.append({
                    "alert_id": alert.id,
                    "alert_name": alert.name,
                    "metric": metric_name,
                    "value": value,
                    "threshold": alert.threshold,
                    "message": alert.message,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

    @classmethod
    def record_cost(cls, input_tokens: int, output_tokens: int, model: str = "default"):
        """记录成本"""
        input_cost = input_tokens * cls.TOKEN_PRICING["input"]
        output_cost = output_tokens * cls.TOKEN_PRICING["output"]
        cls._cost_records.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "model": model,
        })

    @classmethod
    def add_alert(cls, alert: AlertRule):
        """添加告警规则"""
        cls._alerts.append(alert)

    @classmethod
    def get_metrics(cls, metric_name: str = None, limit: int = 100) -> list:
        """获取指标"""
        if metric_name:
            return [m.to_dict() for m in cls._metrics if m.metric_name == metric_name][-limit:]
        return [m.to_dict() for m in cls._metrics][-limit:]

    @classmethod
    def get_alerts(cls) -> list:
        """获取告警规则"""
        return [a.to_dict() for a in cls._alerts]

    @classmethod
    def get_alert_history(cls) -> list:
        """获取告警历史"""
        return cls._alert_history

    @classmethod
    def get_cost_records(cls) -> list:
        """获取成本记录"""
        return cls._cost_records

    @classmethod
    def get_summary(cls) -> dict:
        """获取监控摘要"""
        latency_metrics = [m for m in cls._metrics if m.metric_name == "latency_ms"]
        error_metrics = [m for m in cls._metrics if m.metric_name == "error"]

        total_calls = len(latency_metrics)
        avg_latency = sum(m.value for m in latency_metrics) / total_calls if total_calls else 0
        error_count = sum(1 for m in error_metrics if m.value == 1)
        error_rate = error_count / total_calls * 100 if total_calls else 0

        total_input_tokens = sum(r["input_tokens"] for r in cls._cost_records)
        total_output_tokens = sum(r["output_tokens"] for r in cls._cost_records)
        total_cost = sum(r["total_cost"] for r in cls._cost_records)

        return {
            "total_calls": total_calls,
            "avg_latency_ms": round(avg_latency, 2),
            "error_rate": round(error_rate, 2),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_cost": round(total_cost, 4),
        }


# ============================================================
# 示例1: 生产监控 - 实时监控
# ============================================================

def demo_production_monitoring():
    """示例1：生产监控 - 实时监控（监控API调用的延迟、错误率、Token消耗）"""
    print("\n" + "="*60)
    print("示例1：生产监控 - 实时监控")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 监控：实时监控 API 调用的状态")
    print("   - 关键指标：延迟、错误率、Token 消耗")
    print("   - 真实 LangSmith 提供 Web 界面实时监控")
    print("\n📊 应用场景：")
    print("   - 实时监控生产环境 API 调用")
    print("   - 检测延迟和错误率异常")
    print("   - 追踪 Token 消耗")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    print("\n【交互式生产监控演示】")
    print("提示：输入问题，系统实时监控 API 调用状态")
    print("输入 '退出' 结束\n")

    call_count = 0

    while True:
        question = input("你的问题：").strip()
        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break
        if not question:
            print("请输入有效问题")
            continue

        call_count += 1

        # 执行调用并监控
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个知识问答助手，简洁准确地回答问题"),
            ("human", "{question}")
        ])
        chain = prompt | model | StrOutputParser()

        start_time = time.time()
        try:
            answer = chain.invoke({"question": question})
            latency_ms = (time.time() - start_time) * 1000
            is_error = False
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            is_error = True
            answer = f"[错误: {e}]"

        # 记录监控指标
        MonitoringStore.record_metric("latency_ms", latency_ms, {"call": call_count})
        MonitoringStore.record_metric("error", 1 if is_error else 0, {"call": call_count})

        # 模拟 Token 消耗
        input_tokens = len(question) * 2 + 20  # 模拟
        output_tokens = len(answer) * 2  # 模拟
        MonitoringStore.record_metric("input_tokens", input_tokens, {"call": call_count})
        MonitoringStore.record_metric("output_tokens", output_tokens, {"call": call_count})
        MonitoringStore.record_cost(input_tokens, output_tokens)

        # 显示结果
        status_icon = "✅" if not is_error else "❌"
        print(f"\n{status_icon} 回答：{answer[:200]}{'...' if len(answer) > 200 else ''}")

        # 显示实时监控数据
        print(f"\n📊 实时监控数据 [调用 #{call_count}]：")
        print(f"   状态：{'成功' if not is_error else '失败'}")
        print(f"   延迟：{latency_ms:.0f}ms")
        print(f"   输入Token：{input_tokens}")
        print(f"   输出Token：{output_tokens}")

        # 显示累计统计
        summary = MonitoringStore.get_summary()
        print(f"\n📈 累计统计：")
        print(f"   总调用：{summary['total_calls']}")
        print(f"   平均延迟：{summary['avg_latency_ms']:.0f}ms")
        print(f"   错误率：{summary['error_rate']:.1f}%")
        print(f"   总Token：{summary['total_tokens']}")
        print(f"   总成本：¥{summary['total_cost']:.4f}")
        print("-"*60)

    # 最终监控报告
    summary = MonitoringStore.get_summary()
    print(f"\n📋 监控报告：")
    print("="*60)
    print(f"  总调用次数：{summary['total_calls']}")
    print(f"  平均延迟：{summary['avg_latency_ms']:.0f}ms")
    print(f"  错误率：{summary['error_rate']:.1f}%")
    print(f"  总Token消耗：{summary['total_tokens']} (输入:{summary['total_input_tokens']} 输出:{summary['total_output_tokens']})")
    print(f"  总成本：¥{summary['total_cost']:.4f}")
    print("="*60)

    print("\n✅ 实战要点总结：")
    print("   1. 实时监控 API 调用的延迟、错误率、Token 消耗")
    print("   2. 关键指标：延迟、错误率、Token 消耗、成本")
    print("   3. 真实 LangSmith 提供 Web 界面实时监控仪表盘")
    print("   4. 可设置告警规则，异常时自动通知")


# ============================================================
# 示例2: 告警配置 - 异常告警
# ============================================================

def demo_alert_configuration():
    """示例2：告警配置 - 异常告警（配置告警规则，异常时自动通知）"""
    print("\n" + "="*60)
    print("示例2：告警配置 - 异常告警")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 告警：配置告警规则，异常时自动通知")
    print("   - 告警条件：延迟 > 阈值、错误率 > 阈值等")
    print("   - 真实 LangSmith 支持邮件/Slack 通知")
    print("\n📊 应用场景：")
    print("   - 延迟异常告警")
    print("   - 错误率异常告警")
    print("   - Token 消耗异常告警")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    # 预配置告警规则
    default_alerts = [
        AlertRule("高延迟告警", "latency_ms", "gt", 5000, "API延迟超过5秒，请检查服务状态"),
        AlertRule("错误率告警", "error", "gte", 1, "API调用失败，请检查错误原因"),
        AlertRule("高Token消耗告警", "output_tokens", "gt", 5000, "单次输出Token超过5000，可能存在异常"),
    ]
    for alert in default_alerts:
        MonitoringStore.add_alert(alert)

    print("\n📋 已预配置告警规则：")
    for alert in default_alerts:
        condition_map = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}
        print(f"   {alert.name}: {alert.metric} {condition_map[alert.condition]} {alert.threshold}")

    print("\n【交互式告警配置演示】")

    while True:
        print(f"\n{'─'*60}")
        print("告警管理菜单：")
        print("  1. 查看告警规则")
        print("  2. 添加告警规则")
        print("  3. 启用/禁用告警")
        print("  4. 查看告警历史")
        print("  5. 模拟触发告警")
        print("\n  0. 退出")
        print(f"{'─'*60}")

        choice = input("请选择 (0-5): ").strip()

        if choice == "0":
            print("结束演示")
            break

        elif choice == "1":
            # 查看告警规则
            alerts = MonitoringStore.get_alerts()
            if not alerts:
                print("📭 暂无告警规则")
                continue
            print(f"\n📋 告警规则列表（共 {len(alerts)} 条）：")
            print("="*60)
            condition_map = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}
            for a in alerts:
                status = "🟢 启用" if a["enabled"] else "🔴 禁用"
                print(f"  {a['id']} | {a['name']} | {status}")
                print(f"     条件：{a['metric']} {condition_map[a['condition']]} {a['threshold']}")
                print(f"     消息：{a['message']}")
                print(f"     触发次数：{a['triggered_count']} | 最后触发：{a['last_triggered'] or '无'}")
                print()
            print("="*60)

        elif choice == "2":
            # 添加告警规则
            name = input("告警名称：").strip()
            if not name:
                print("❌ 名称不能为空")
                continue
            print("可用指标：latency_ms, error, input_tokens, output_tokens")
            metric = input("监控指标：").strip()
            if not metric:
                print("❌ 指标不能为空")
                continue
            print("条件：gt(大于), lt(小于), gte(大于等于), lte(小于等于)")
            condition = input("条件：").strip().lower()
            if condition not in ["gt", "lt", "gte", "lte"]:
                print("❌ 无效条件")
                continue
            threshold = input("阈值：").strip()
            try:
                threshold = float(threshold)
            except ValueError:
                print("❌ 阈值必须是数字")
                continue
            message = input("告警消息：").strip() or f"{name} 触发"
            alert = AlertRule(name, metric, condition, threshold, message)
            MonitoringStore.add_alert(alert)
            print(f"✅ 告警规则已添加：{alert.id}")

        elif choice == "3":
            # 启用/禁用告警
            alerts = MonitoringStore._alerts
            if not alerts:
                print("📭 暂无告警规则")
                continue
            for i, a in enumerate(alerts, 1):
                status = "🟢" if a.enabled else "🔴"
                print(f"  {i}. {status} {a.name}")
            idx = input("选择告警编号：").strip()
            try:
                idx = int(idx) - 1
                if 0 <= idx < len(alerts):
                    alerts[idx].enabled = not alerts[idx].enabled
                    new_status = "启用" if alerts[idx].enabled else "禁用"
                    print(f"✅ 已{new_status}：{alerts[idx].name}")
                else:
                    print("❌ 无效编号")
            except ValueError:
                print("❌ 无效输入")

        elif choice == "4":
            # 查看告警历史
            history = MonitoringStore.get_alert_history()
            if not history:
                print("📭 暂无告警历史")
                continue
            print(f"\n📋 告警历史（共 {len(history)} 条）：")
            print("="*60)
            for h in history[-10:]:  # 只显示最近10条
                print(f"  🚨 {h['alert_name']}")
                print(f"     指标：{h['metric']} = {h['value']} (阈值: {h['threshold']})")
                print(f"     消息：{h['message']}")
                print(f"     时间：{h['timestamp']}")
                print()
            print("="*60)

        elif choice == "5":
            # 模拟触发告警
            print("\n模拟场景：")
            print("  1. 模拟高延迟（8000ms）")
            print("  2. 模拟API错误")
            print("  3. 模拟高Token消耗（6000）")
            print("  4. 模拟正常情况（1000ms）")
            sim_choice = input("请选择 (1-4): ").strip()
            if sim_choice == "1":
                MonitoringStore.record_metric("latency_ms", 8000, {"simulated": True})
                print("✅ 已模拟高延迟（8000ms）")
            elif sim_choice == "2":
                MonitoringStore.record_metric("error", 1, {"simulated": True})
                print("✅ 已模拟API错误")
            elif sim_choice == "3":
                MonitoringStore.record_metric("output_tokens", 6000, {"simulated": True})
                print("✅ 已模拟高Token消耗（6000）")
            elif sim_choice == "4":
                MonitoringStore.record_metric("latency_ms", 1000, {"simulated": True})
                print("✅ 已模拟正常情况（1000ms）")
            else:
                print("❌ 无效选项")
                continue

            # 检查是否有新告警
            history = MonitoringStore.get_alert_history()
            if history:
                latest = history[-1]
                print(f"\n🚨 告警触发！")
                print(f"   告警：{latest['alert_name']}")
                print(f"   指标：{latest['metric']} = {latest['value']} (阈值: {latest['threshold']})")
                print(f"   消息：{latest['message']}")
            else:
                print("\n✅ 未触发告警，指标正常")

        else:
            print("❌ 无效选项")

    print("\n✅ 实战要点总结：")
    print("   1. 告警规则可自动检测异常情况")
    print("   2. 常见告警：高延迟、错误率、高Token消耗")
    print("   3. 真实 LangSmith 支持邮件/Slack/Webhook 通知")
    print("   4. 可根据业务需求自定义告警规则")


# ============================================================
# 示例3: 成本追踪 - Token成本
# ============================================================

def demo_cost_tracking():
    """示例3：成本追踪 - Token成本（追踪Token消耗和成本）"""
    print("\n" + "="*60)
    print("示例3：成本追踪 - Token成本")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 成本追踪：追踪 Token 消耗和成本")
    print("   - Token 计费：输入Token和输出Token价格不同")
    print("   - 真实 LangSmith 提供成本分析仪表盘")
    print("\n📊 应用场景：")
    print("   - 追踪每次调用的 Token 消耗")
    print("   - 分析成本趋势")
    print("   - 优化成本")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    print("\n【交互式成本追踪演示】")
    print("提示：输入问题，系统追踪 Token 消耗和成本")
    print("输入 '退出' 结束\n")

    # 显示计费说明
    print("💰 模拟计费标准：")
    print(f"   输入Token：¥{MonitoringStore.TOKEN_PRICING['input']:.6f}/Token (¥{MonitoringStore.TOKEN_PRICING['input']*1000:.4f}/千Token)")
    print(f"   输出Token：¥{MonitoringStore.TOKEN_PRICING['output']:.6f}/Token (¥{MonitoringStore.TOKEN_PRICING['output']*1000:.4f}/千Token)")
    print()

    while True:
        question = input("你的问题：").strip()
        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break
        if not question:
            print("请输入有效问题")
            continue

        # 执行调用
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个知识问答助手，简洁准确地回答问题"),
            ("human", "{question}")
        ])
        chain = prompt | model | StrOutputParser()

        try:
            answer = chain.invoke({"question": question})
            # 模拟 Token 消耗
            input_tokens = len(question) * 2 + 20
            output_tokens = len(answer) * 2
        except Exception as e:
            answer = f"[错误: {e}]"
            input_tokens = len(question) * 2 + 20
            output_tokens = 0

        # 记录成本
        MonitoringStore.record_cost(input_tokens, output_tokens)
        MonitoringStore.record_metric("latency_ms", 0, {"cost_tracking": True})

        # 计算本次成本
        input_cost = input_tokens * MonitoringStore.TOKEN_PRICING["input"]
        output_cost = output_tokens * MonitoringStore.TOKEN_PRICING["output"]
        total_cost = input_cost + output_cost

        # 显示结果
        print(f"\n🤖 回答：{answer[:200]}{'...' if len(answer) > 200 else ''}")

        # 显示本次成本
        print(f"\n💰 本次调用成本：")
        print(f"   输入Token：{input_tokens} × ¥{MonitoringStore.TOKEN_PRICING['input']:.6f} = ¥{input_cost:.4f}")
        print(f"   输出Token：{output_tokens} × ¥{MonitoringStore.TOKEN_PRICING['output']:.6f} = ¥{output_cost:.4f}")
        print(f"   本次总成本：¥{total_cost:.4f}")

        # 显示累计成本
        records = MonitoringStore.get_cost_records()
        total_input = sum(r["input_tokens"] for r in records)
        total_output = sum(r["output_tokens"] for r in records)
        total_cost_all = sum(r["total_cost"] for r in records)
        avg_cost = total_cost_all / len(records) if records else 0

        print(f"\n📊 累计成本统计：")
        print(f"   调用次数：{len(records)}")
        print(f"   总输入Token：{total_input}")
        print(f"   总输出Token：{total_output}")
        print(f"   总Token：{total_input + total_output}")
        print(f"   总成本：¥{total_cost_all:.4f}")
        print(f"   平均每次成本：¥{avg_cost:.4f}")

        # 成本预估
        if records:
            daily_estimate = avg_cost * 100  # 假设每天100次调用
            monthly_estimate = daily_estimate * 30
            print(f"\n📈 成本预估（假设每天100次调用）：")
            print(f"   日成本：¥{daily_estimate:.4f}")
            print(f"   月成本：¥{monthly_estimate:.4f}")
            print(f"   年成本：¥{monthly_estimate * 12:.4f}")

        print("-"*60)

    # 成本报告
    records = MonitoringStore.get_cost_records()
    if records:
        print(f"\n📋 成本报告：")
        print("="*60)
        print(f"  {'调用#':<8} {'输入Token':<12} {'输出Token':<12} {'成本(¥)':<10} {'时间'}")
        print(f"  {'─'*55}")
        for i, r in enumerate(records, 1):
            print(f"  {i:<8} {r['input_tokens']:<12} {r['output_tokens']:<12} {r['total_cost']:<10.4f} {r['timestamp']}")
        print("="*60)

    print("\n✅ 实战要点总结：")
    print("   1. 成本追踪记录每次调用的 Token 消耗和成本")
    print("   2. 输入Token和输出Token价格不同")
    print("   3. 可预估日/月/年成本")
    print("   4. 真实 LangSmith 提供成本分析仪表盘")


# ============================================================
# 示例4: 性能仪表盘 - 数据可视化
# ============================================================

def demo_performance_dashboard():
    """示例4：性能仪表盘 - 数据可视化（展示监控数据的可视化图表）"""
    print("\n" + "="*60)
    print("示例4：性能仪表盘 - 数据可视化")
    print("="*60)
    print("\n💡 核心概念：")
    print("   - 仪表盘：数据可视化展示监控数据")
    print("   - 真实 LangSmith 提供 Web 界面仪表盘")
    print("   - 终端中用字符图表模拟可视化")
    print("\n📊 应用场景：")
    print("   - 查看延迟趋势")
    print("   - 查看错误率变化")
    print("   - 查看 Token 消耗分布")

    if has_langsmith_key():
        print("\n🔑 检测到 LANGSMITH_API_KEY，可连接真实 LangSmith 服务")
    else:
        print("\n🔓 未检测到 LANGSMITH_API_KEY，使用模拟演示模式")

    model = get_default_llm()

    # 生成模拟数据
    print("\n【步骤1：生成监控数据】")
    print("正在生成模拟监控数据...\n")

    # 模拟过去24小时的调用数据
    for i in range(20):
        latency = random.uniform(500, 5000)
        is_error = random.random() < 0.1  # 10% 错误率
        input_tokens = random.randint(50, 500)
        output_tokens = random.randint(100, 2000)

        MonitoringStore.record_metric("latency_ms", latency)
        MonitoringStore.record_metric("error", 1 if is_error else 0)
        MonitoringStore.record_metric("input_tokens", input_tokens)
        MonitoringStore.record_metric("output_tokens", output_tokens)
        MonitoringStore.record_cost(input_tokens, output_tokens)

    print("✅ 已生成20条模拟监控数据")

    # 交互式仪表盘
    print("\n【步骤2：查看仪表盘】")

    while True:
        print(f"\n{'─'*60}")
        print("性能仪表盘菜单：")
        print("  1. 概览仪表盘")
        print("  2. 延迟趋势图")
        print("  3. 错误率统计")
        print("  4. Token 消耗分布")
        print("  5. 成本分析")
        print("  6. 执行新调用并更新仪表盘")
        print("\n  0. 退出")
        print(f"{'─'*60}")

        choice = input("请选择 (0-6): ").strip()

        if choice == "0":
            print("结束演示")
            break

        elif choice == "1":
            # 概览仪表盘
            summary = MonitoringStore.get_summary()
            latency_metrics = MonitoringStore.get_metrics("latency_ms")
            error_metrics = MonitoringStore.get_metrics("error")

            print(f"\n📊 概览仪表盘：")
            print("="*60)
            print(f"  ┌─────────────────────────────────────────┐")
            print(f"  │  总调用次数: {summary['total_calls']:<28}│")
            print(f"  │  平均延迟:   {summary['avg_latency_ms']:.0f}ms{'':<23}│")
            print(f"  │  错误率:     {summary['error_rate']:.1f}%{'':<25}│")
            print(f"  │  总Token:    {summary['total_tokens']:<28}│")
            print(f"  │  总成本:     ¥{summary['total_cost']:.4f}{'':<24}│")
            print(f"  └─────────────────────────────────────────┘")

            # 健康状态
            if summary['error_rate'] < 5 and summary['avg_latency_ms'] < 3000:
                health = "🟢 健康"
            elif summary['error_rate'] < 10 and summary['avg_latency_ms'] < 5000:
                health = "🟡 警告"
            else:
                health = "🔴 异常"
            print(f"\n  系统状态：{health}")
            print("="*60)

        elif choice == "2":
            # 延迟趋势图
            latency_metrics = MonitoringStore.get_metrics("latency_ms")
            if not latency_metrics:
                print("📭 暂无延迟数据")
                continue

            print(f"\n📊 延迟趋势图：")
            print("="*60)

            # 用字符画柱状图
            values = [m["value"] for m in latency_metrics]
            max_val = max(values) if values else 1
            bar_width = 40

            print(f"  {'调用#':<6} {'延迟(ms)':<10} {'趋势'}")
            print(f"  {'─'*55}")
            for i, m in enumerate(latency_metrics, 1):
                bar_len = int(m["value"] / max_val * bar_width)
                bar = "█" * bar_len + "░" * (bar_width - bar_len)
                # 颜色标记
                if m["value"] < 2000:
                    mark = "🟢"
                elif m["value"] < 5000:
                    mark = "🟡"
                else:
                    mark = "🔴"
                print(f"  {i:<6} {m['value']:<10.0f} {mark} {bar}")

            # 统计
            avg = sum(values) / len(values)
            min_v = min(values)
            max_v = max(values)
            print(f"\n  统计：平均={avg:.0f}ms 最小={min_v:.0f}ms 最大={max_v:.0f}ms")
            print("="*60)

        elif choice == "3":
            # 错误率统计
            error_metrics = MonitoringStore.get_metrics("error")
            latency_metrics = MonitoringStore.get_metrics("latency_ms")
            if not error_metrics:
                print("📭 暂无错误数据")
                continue

            total = len(error_metrics)
            errors = sum(1 for m in error_metrics if m["value"] == 1)
            success = total - errors
            error_rate = errors / total * 100 if total > 0 else 0

            print(f"\n📊 错误率统计：")
            print("="*60)
            print(f"  总调用：{total}")
            print(f"  成功：  {success} ({success/total*100:.1f}%)")
            print(f"  失败：  {errors} ({error_rate:.1f}%)")

            # 饼图模拟
            print(f"\n  成功/失败比例：")
            success_bar = "🟢" * int(success / total * 30) if total > 0 else ""
            error_bar = "🔴" * int(errors / total * 30) if total > 0 else ""
            print(f"  {success_bar}{error_bar}")
            print(f"  🟢 成功 {success/total*100:.1f}%  🔴 失败 {error_rate:.1f}%")

            # 错误分布
            print(f"\n  错误时间线：")
            for i, m in enumerate(error_metrics, 1):
                if m["value"] == 1:
                    print(f"  ❌ 调用#{i} @ {m['timestamp']}")
            print("="*60)

        elif choice == "4":
            # Token 消耗分布
            input_metrics = MonitoringStore.get_metrics("input_tokens")
            output_metrics = MonitoringStore.get_metrics("output_tokens")
            if not input_metrics:
                print("📭 暂无Token数据")
                continue

            print(f"\n📊 Token 消耗分布：")
            print("="*60)

            input_values = [m["value"] for m in input_metrics]
            output_values = [m["value"] for m in output_metrics]

            total_input = sum(input_values)
            total_output = sum(output_values)
            total_tokens = total_input + total_output

            # 柱状图
            max_tokens = max(max(input_values), max(output_values)) if input_values else 1
            bar_width = 30

            print(f"  {'调用#':<6} {'输入Token':<12} {'输出Token':<12} {'分布'}")
            print(f"  {'─'*55}")
            for i in range(len(input_values)):
                inp = input_values[i]
                out = output_values[i]
                inp_bar = "█" * int(inp / max_tokens * bar_width)
                out_bar = "░" * int(out / max_tokens * bar_width)
                print(f"  {i+1:<6} {inp:<12.0f} {out:<12.0f} {inp_bar}{out_bar}")

            # 分布比例
            input_pct = total_input / total_tokens * 100 if total_tokens > 0 else 0
            output_pct = total_output / total_tokens * 100 if total_tokens > 0 else 0

            print(f"\n  Token 分布：")
            input_bar = "🟦" * int(input_pct / 100 * 30)
            output_bar = "🟧" * int(output_pct / 100 * 30)
            print(f"  {input_bar}{output_bar}")
            print(f"  🟦 输入 {input_pct:.1f}%  🟧 输出 {output_pct:.1f}%")
            print("="*60)

        elif choice == "5":
            # 成本分析
            records = MonitoringStore.get_cost_records()
            if not records:
                print("📭 暂无成本数据")
                continue

            print(f"\n📊 成本分析：")
            print("="*60)

            total_cost = sum(r["total_cost"] for r in records)
            total_input = sum(r["input_tokens"] for r in records)
            total_output = sum(r["output_tokens"] for r in records)
            input_cost = sum(r["input_cost"] for r in records)
            output_cost = sum(r["output_cost"] for r in records)

            print(f"  总成本：¥{total_cost:.4f}")
            print(f"  输入成本：¥{input_cost:.4f} ({input_cost/total_cost*100:.1f}%)")
            print(f"  输出成本：¥{output_cost:.4f} ({output_cost/total_cost*100:.1f}%)")

            # 成本分布
            print(f"\n  成本分布：")
            input_pct = input_cost / total_cost * 100 if total_cost > 0 else 0
            output_pct = output_cost / total_cost * 100 if total_cost > 0 else 0
            input_bar = "🟦" * int(input_pct / 100 * 30)
            output_bar = "🟧" * int(output_pct / 100 * 30)
            print(f"  {input_bar}{output_bar}")
            print(f"  🟦 输入成本 {input_pct:.1f}%  🟧 输出成本 {output_pct:.1f}%")

            # 每次调用成本趋势
            print(f"\n  每次调用成本趋势：")
            max_cost = max(r["total_cost"] for r in records) if records else 1
            bar_width = 30
            for i, r in enumerate(records, 1):
                bar_len = int(r["total_cost"] / max_cost * bar_width)
                bar = "█" * bar_len + "░" * (bar_width - bar_len)
                print(f"  {i:<4} ¥{r['total_cost']:.4f} {bar}")

            # 成本优化建议
            print(f"\n  💡 成本优化建议：")
            avg_output = total_output / len(records) if records else 0
            if avg_output > 1000:
                print(f"     - 平均输出Token较高（{avg_output:.0f}），考虑限制输出长度")
            if output_pct > 70:
                print(f"     - 输出成本占比过高（{output_pct:.1f}%），考虑优化输出格式")
            print("="*60)

        elif choice == "6":
            # 执行新调用
            question = input("输入问题：").strip()
            if not question:
                print("❌ 问题不能为空")
                continue
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个知识问答助手，简洁准确地回答问题"),
                ("human", "{question}")
            ])
            chain = prompt | model | StrOutputParser()
            start_time = time.time()
            try:
                answer = chain.invoke({"question": question})
                latency_ms = (time.time() - start_time) * 1000
                is_error = False
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                is_error = True
                answer = f"[错误: {e}]"

            # 记录指标
            MonitoringStore.record_metric("latency_ms", latency_ms)
            MonitoringStore.record_metric("error", 1 if is_error else 0)
            input_tokens = len(question) * 2 + 20
            output_tokens = len(answer) * 2
            MonitoringStore.record_metric("input_tokens", input_tokens)
            MonitoringStore.record_metric("output_tokens", output_tokens)
            MonitoringStore.record_cost(input_tokens, output_tokens)

            print(f"\n✅ 调用完成，数据已更新到仪表盘")
            print(f"   延迟：{latency_ms:.0f}ms | Token：{input_tokens + output_tokens}")

        else:
            print("❌ 无效选项")

    print("\n✅ 实战要点总结：")
    print("   1. 仪表盘可视化展示监控数据")
    print("   2. 包括延迟趋势、错误率、Token 消耗、成本分析")
    print("   3. 终端中用字符图表模拟可视化")
    print("   4. 真实 LangSmith 提供 Web 界面交互式仪表盘")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangSmith 监控（Monitoring）- 实战案例")
    print("="*60)
    print("\n本示例演示 LangSmith 监控功能的核心概念和使用方法")

    mode = "真实模式" if has_langsmith_key() else "模拟模式"
    print(f"\n当前模式：{mode}")
    if not has_langsmith_key():
        print("提示：配置 LANGSMITH_API_KEY 可连接真实 LangSmith 服务")

    print("\n核心概念：")
    print("  • 监控: 实时监控API调用")
    print("  • 告警: 异常时自动通知")
    print("  • 成本追踪: Token消耗和成本")
    print("  • 仪表盘: 数据可视化")

    print("\n应用场景：")
    print("  • 生产监控、异常告警、成本追踪、性能仪表盘")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 生产监控 - 实时监控")
        print("  2. 告警配置 - 异常告警")
        print("  3. 成本追踪 - Token成本")
        print("  4. 性能仪表盘 - 数据可视化")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_production_monitoring()
        elif choice == "2":
            demo_alert_configuration()
        elif choice == "3":
            demo_cost_tracking()
        elif choice == "4":
            demo_performance_dashboard()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
