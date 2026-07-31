"""
AI 数据分析师 / AI Data Analyst
====================================

完整的端到端AI数据分析师项目，整合多种LangChain技术

技术整合 / Tech Integration:
- Chain: 数据分析方案生成、数据清洗建议（LLMChain + PromptTemplate）
- Agent: 趋势预测（Tool Calling Agent + 自定义计算/统计工具）
- Tool: 计算器、统计分析等辅助工具（@tool 装饰器）
- Structured Output: 报表生成（PydanticOutputParser + 结构化报表模型）
- RAG: 数据分析知识库检索（关键词检索 + 上下文注入）
- Memory: 分析历史追踪（对话记忆 + 分析记录）

功能模块 / Features:
1. 数据分析 - 输入数据描述，AI生成分析方案（Chain）
2. 报表生成 - 输入数据，AI生成文字报表（Structured Output）
3. 趋势预测 - 输入历史数据，AI预测趋势（Agent + Tool）
4. 图表建议 - 输入数据类型，AI推荐可视化方式（Tool）
5. 数据清洗 - 输入原始数据描述，AI给出清洗方案（Chain）

应用场景 / Use Cases:
- 企业经营分析：销售数据、财务数据的快速分析
- 市场研究：消费者行为、市场趋势分析
- 运营监控：关键指标追踪、异常检测

Core Concepts:
- Chain: Compose LLM calls for structured analysis workflows
- Agent: Autonomous tool selection for multi-step data reasoning
- Tool: Custom functions for calculations and statistical operations
- Structured Output: Typed report generation with validation
- RAG: Knowledge retrieval for analysis methodology
- Memory: Session persistence for iterative analysis
"""

import os
import sys
import json
import re
import sqlite3
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from src.utils.llm_loader import get_default_llm


# ============================================================
# SQLite 数据库配置
# ============================================================

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "data_analyst.db")


def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库，创建表和示例数据"""
    conn = get_db()
    cursor = conn.cursor()

    # 创建表
    cursor.execute("""CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        data_json TEXT,
        created_at TEXT NOT NULL
    )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER,
        analysis_type TEXT NOT NULL,
        result_json TEXT,
        created_at TEXT NOT NULL
    )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        summary TEXT,
        findings_json TEXT,
        recommendations_json TEXT,
        created_at TEXT NOT NULL
    )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS chart_suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data_type TEXT NOT NULL,
        chart_type TEXT NOT NULL,
        reason TEXT,
        example TEXT
    )""")

    # 插入示例数据（如果表为空）
    cursor.execute("SELECT COUNT(*) FROM datasets")
    if cursor.fetchone()[0] == 0:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sample_datasets = [
            ("销售数据", "月度销售数据，包含营收、订单量、客户数",
             json.dumps([{"month": "1月", "revenue": 580, "orders": 1200, "customers": 800},
                         {"month": "2月", "revenue": 520, "orders": 1100, "customers": 750},
                         {"month": "3月", "revenue": 650, "orders": 1350, "customers": 900}], ensure_ascii=False), now),
            ("用户行为数据", "用户浏览、加购、下单、支付行为漏斗数据",
             json.dumps([{"action": "浏览", "count": 10000},
                         {"action": "加购", "count": 3000},
                         {"action": "下单", "count": 1500},
                         {"action": "支付", "count": 1200}], ensure_ascii=False), now),
            ("产品评分数据", "各产品类目的平均评分和评价数量",
             json.dumps([{"category": "电子产品", "avg_score": 4.2, "reviews": 5600},
                         {"category": "服装", "avg_score": 4.5, "reviews": 8200},
                         {"category": "食品", "avg_score": 4.7, "reviews": 3400}], ensure_ascii=False), now),
        ]
        cursor.executemany(
            "INSERT INTO datasets (name, description, data_json, created_at) VALUES (?, ?, ?, ?)",
            sample_datasets
        )

    # 插入图表建议示例数据
    cursor.execute("SELECT COUNT(*) FROM chart_suggestions")
    if cursor.fetchone()[0] == 0:
        sample_charts = [
            ("时间序列", "折线图", "展示数据随时间的变化趋势，适合连续数据", "plt.plot(dates, values)"),
            ("时间序列", "面积图", "强调累计量和趋势，适合展示总量变化", "plt.fill_between(dates, values)"),
            ("分类对比", "柱状图", "比较不同类别之间的数值差异", "plt.bar(categories, values)"),
            ("分类对比", "条形图", "类别较多时的横向对比，更易读", "plt.barh(categories, values)"),
            ("占比分布", "饼图", "展示各部分占整体的比例（分类<6）", "plt.pie(values, labels=labels)"),
            ("占比分布", "环形图", "分类>6时的占比展示，更美观", "plt.pie(values, wedgeprops=dict(width=0.3))"),
            ("关联关系", "散点图", "展示两个变量之间的相关关系", "plt.scatter(x, y)"),
            ("关联关系", "热力图", "展示相关矩阵或密集数据关系", "sns.heatmap(corr_matrix)"),
            ("分布特征", "直方图", "展示数据的分布形态", "plt.hist(values, bins=20)"),
            ("分布特征", "箱线图", "检测离群值和分布比较", "sns.boxplot(data)"),
            ("地理空间", "地图", "展示地理分布和区域差异", "folium.Map()"),
        ]
        cursor.executemany(
            "INSERT INTO chart_suggestions (data_type, chart_type, reason, example) VALUES (?, ?, ?, ?)",
            sample_charts
        )

    conn.commit()
    conn.close()


# ============================================================
# RAG 检索辅助函数
# ============================================================

def keyword_similarity(query: str, text: str) -> float:
    """基于关键词匹配计算文本相似度"""
    q_lower = query.lower()
    t_lower = text.lower()

    stopwords = {"的", "了", "是", "在", "有", "和", "与", "及", "等", "个",
                 "一", "这", "那", "不", "也", "都", "就", "要", "会", "能",
                 "什么", "怎么", "如何", "哪些", "为什么", "吗", "呢", "吧",
                 "我", "你", "他", "她", "它", "请", "想", "能", "做"}

    keywords = set()
    for word in q_lower.replace("，", " ").replace("。", " ").replace("？", " ") \
                       .replace(",", " ").replace(".", " ").replace("?", " ") \
                       .split():
        word = word.strip()
        if word and word not in stopwords:
            keywords.add(word)
            if len(word) >= 2:
                for i in range(len(word) - 1):
                    keywords.add(word[i:i+2])
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    keywords.add(word[i:i+3])

    if not keywords:
        return 0.0

    hits = sum(1 for kw in keywords if kw in t_lower)
    return hits / len(keywords)


# 分析知识库内容（用于RAG检索，保存在datasets表中description字段）
_ANALYSIS_KNOWLEDGE_CONTENTS = [
    "描述性统计分析：通过均值、中位数、众数、标准差、方差等统计量描述数据的基本特征。"
    "适用于数据探索阶段，快速了解数据的集中趋势和离散程度。箱线图和直方图是常用的可视化方法。",

    "相关性分析：通过皮尔逊相关系数（线性关系）、斯皮尔曼相关系数（单调关系）衡量变量间的关联程度。"
    "相关系数范围 [-1, 1]，绝对值越大约相关。注意：相关不等于因果，需要结合业务逻辑判断。",

    "回归分析：线性回归用于建模自变量与因变量的线性关系，逻辑回归用于分类问题。"
    "评估指标：R²（拟合优度）、MSE/RMSE（预测误差）、F检验（模型显著性）。"
    "多元回归需要注意多重共线性问题，可用 VIF 检测。",

    "时间序列分析：ARIMA 模型用于非平稳时间序列预测，需先做差分使序列平稳。"
    "季节性分解（STL）将序列拆分为趋势、季节和残差成分。"
    "评估指标：MAE（平均绝对误差）、MAPE（平均绝对百分比误差）、RMSE。",

    "聚类分析：K-Means 基于距离划分簇，需要预设 K 值（可用肘部法则确定）。"
    "DBSCAN 基于密度发现任意形状的簇，无需预设 K 值。层次聚类生成树状图。"
    "评估指标：轮廓系数（Silhouette Score）、Calinski-Harabasz 指数。",

    "假设检验：t 检验比较两组均值差异、卡方检验分析分类变量关联性、"
    "ANOVA 比较多组均值差异。p 值 < 0.05 通常认为统计显著。"
    "注意事项：样本量影响检验效力，多重比较需校正（如 Bonferroni）。",

    "数据清洗方法：缺失值处理（删除、均值/中位数填充、插值法）、"
    "异常值检测（3σ 原则、IQR 方法、孤立森林）、重复值处理、数据类型转换。"
    "数据清洗通常占数据分析项目 60-80% 的时间。",

    "A/B 测试分析：随机将用户分为对照组和实验组，比较不同方案的效果。"
    "需要确定：样本量（统计功效）、检验指标、显著性水平（α=0.05）。"
    "常见陷阱：偷看数据、多重比较、样本量不足、辛普森悖论。",

    "数据可视化原则：选择合适的图表类型——比较用柱状图、趋势用折线图、"
    "占比用饼图/环形图、关系用散点图、分布用直方图/箱线图、地理用热力图。"
    "避免 3D 图表、过度装饰、误导性坐标轴。",

    "特征工程：特征选择（过滤法、包装法、嵌入法）、特征构造（交叉特征、多项式特征）、"
    "特征缩放（标准化、归一化）、特征编码（独热编码、标签编码、目标编码）。"
    "好的特征工程比复杂模型更能提升效果。",
]


def retrieve_analysis_knowledge(query: str, top_k: int = 3) -> list[tuple[int, float, str]]:
    """从SQLite数据集中检索相关分析知识"""
    # 先尝试从datasets表检索
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, description FROM datasets WHERE description IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    # 如果datasets中有数据，用它们做检索
    if rows:
        scored = [(row["id"], keyword_similarity(query, row["description"]), row["description"]) for row in rows]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and scored[0][1] > 0.05:
            return scored[:top_k]

    # 否则使用内置知识库
    scored = [(i, keyword_similarity(query, doc), doc) for i, doc in enumerate(_ANALYSIS_KNOWLEDGE_CONTENTS)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ============================================================
# Structured Output 模型（报表生成）
# ============================================================

class DataReport(BaseModel):
    """数据分析报表模型"""
    title: str = Field(description="报表标题")
    summary: str = Field(description="数据概要：对数据整体情况的简要描述")
    key_findings: list[str] = Field(description="关键发现：从数据中发现的3-5个重要结论")
    metrics: list[dict] = Field(description="核心指标：包含指标名称、数值、同比/环比变化的列表")
    risks: list[str] = Field(description="风险提示：数据中反映的潜在风险点")
    recommendations: list[str] = Field(description="行动建议：基于数据的改进建议")
    next_steps: str = Field(description="下一步：建议的后续分析方向")


# ============================================================
# Agent 工具定义（趋势预测 + 图表建议）
# ============================================================

@tool
def calculate_statistics(data_str: str) -> str:
    """计算一组数字的基本统计量，包括均值、中位数、最大值、最小值、标准差

    Args:
        data_str: 逗号分隔的数字字符串，如 "10,20,30,40,50"

    Returns:
        统计计算结果
    """
    try:
        # 解析输入
        numbers = [float(x.strip()) for x in data_str.replace("，", ",").split(",") if x.strip()]

        if not numbers:
            return "错误：未找到有效数字，请输入逗号分隔的数字"

        n = len(numbers)
        mean_val = sum(numbers) / n
        sorted_nums = sorted(numbers)
        median_val = sorted_nums[n // 2] if n % 2 == 1 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2

        if n > 1:
            variance = sum((x - mean_val) ** 2 for x in numbers) / (n - 1)
            std_dev = variance ** 0.5
        else:
            std_dev = 0

        # 计算增长率
        growth_rate = ((numbers[-1] - numbers[0]) / abs(numbers[0]) * 100) if numbers[0] != 0 else 0

        # 简单线性趋势
        if n >= 2:
            x_vals = list(range(n))
            x_mean = sum(x_vals) / n
            numerator = sum((x_vals[i] - x_mean) * (numbers[i] - mean_val) for i in range(n))
            denominator = sum((x - x_mean) ** 2 for x in x_vals)
            slope = numerator / denominator if denominator != 0 else 0
            trend = "上升" if slope > 0 else "下降" if slope < 0 else "平稳"
            next_val = mean_val + slope * n
        else:
            slope = 0
            trend = "数据不足"
            next_val = numbers[0]

        result = (
            f"📊 统计分析结果（共 {n} 个数据点）：\n"
            f"  均值：{mean_val:.2f}\n"
            f"  中位数：{median_val:.2f}\n"
            f"  最大值：{max(numbers):.2f}\n"
            f"  最小值：{min(numbers):.2f}\n"
            f"  标准差：{std_dev:.2f}\n"
            f"  整体增长率：{growth_rate:+.1f}%\n"
            f"  趋势方向：{trend}（斜率={slope:.2f}）\n"
            f"  简单预测下一期：{next_val:.2f}"
        )
        return result

    except ValueError:
        return "错误：请输入有效的数字，用逗号分隔"
    except Exception as e:
        return f"计算错误：{e}"


@tool
def predict_trend(data_str: str, periods: int = 3) -> str:
    """基于历史数据预测未来趋势，使用简单线性外推和移动平均

    Args:
        data_str: 逗号分隔的历史数据，如 "100,120,135,150,168"
        periods: 预测未来几期，默认3期

    Returns:
        趋势预测结果
    """
    try:
        numbers = [float(x.strip()) for x in data_str.replace("，", ",").split(",") if x.strip()]

        if len(numbers) < 3:
            return "数据点不足（至少需要3个），无法进行趋势预测"

        n = len(numbers)

        # 方法1：线性回归外推
        x_vals = list(range(n))
        x_mean = sum(x_vals) / n
        y_mean = sum(numbers) / n

        numerator = sum((x_vals[i] - x_mean) * (numbers[i] - y_mean) for i in range(n))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean

        linear_predictions = [slope * (n + i) + intercept for i in range(1, periods + 1)]

        # 方法2：移动平均（最近3期）
        window = min(3, n)
        ma_base = sum(numbers[-window:]) / window
        ma_growth = (numbers[-1] - numbers[-window]) / window if window > 1 else 0
        ma_predictions = [ma_base + ma_growth * i for i in range(1, periods + 1)]

        # 综合预测（两种方法平均）
        combined = [(l + m) / 2 for l, m in zip(linear_predictions, ma_predictions)]

        result = f"📈 趋势预测结果（基于 {n} 个历史数据点）：\n\n"
        result += "预测方法对比：\n"

        for i in range(periods):
            result += (
                f"  第 +{i+1} 期：\n"
                f"    线性外推：{linear_predictions[i]:.1f}\n"
                f"    移动平均：{ma_predictions[i]:.1f}\n"
                f"    综合预测：{combined[i]:.1f}\n"
            )

        # 趋势判断
        trend_dir = "上升" if slope > 0 else "下降" if slope < 0 else "平稳"
        avg_growth = slope
        result += f"\n趋势判断：{trend_dir}（平均每期变化：{avg_growth:+.1f}）"
        result += "\n\n⚠️ 注意：此预测基于简单统计模型，仅供初步参考。实际业务决策需考虑更多因素。"

        return result

    except ValueError:
        return "错误：请输入有效的数字，用逗号分隔"
    except Exception as e:
        return f"预测错误：{e}"


@tool
def recommend_chart(data_type: str, purpose: str = "展示") -> str:
    """根据数据类型和分析目的推荐合适的图表类型

    Args:
        data_type: 数据类型描述，如 时间序列、分类对比、占比分布、关联关系 等
        purpose: 分析目的，如 比较、趋势、占比、关系、分布

    Returns:
        图表类型推荐及使用说明
    """
    # 从SQLite读取图表建议
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT data_type, chart_type, reason, example FROM chart_suggestions")
    rows = cursor.fetchall()
    conn.close()

    # 构建图表指引字典
    chart_guide = {}
    for row in rows:
        dt = row["data_type"]
        if dt not in chart_guide:
            chart_guide[dt] = {"推荐图表": [], "适用场景": "", "最佳实践": "", "工具建议": ""}
        chart_guide[dt]["推荐图表"].append(row["chart_type"])
        if row["reason"]:
            chart_guide[dt]["适用场景"] = row["reason"]
        if row["example"]:
            chart_guide[dt]["工具建议"] = row["example"]

    # 如果数据库中没有数据，使用内置指引
    if not chart_guide:
        chart_guide = {
            "时间序列": {
                "推荐图表": ["折线图", "面积图", "K线图"],
                "适用场景": "展示数据随时间的变化趋势",
                "最佳实践": "时间轴放X轴，标注关键事件节点，多系列用不同颜色区分",
                "工具建议": "Matplotlib (plt.plot)、Seaborn (sns.lineplot)、Plotly (交互式)",
            },
            "分类对比": {
                "推荐图表": ["柱状图", "条形图", "分组柱状图"],
                "适用场景": "比较不同类别之间的数值差异",
                "最佳实践": "类别数<12用柱状图，>12考虑条形图，排序后更易读",
                "工具建议": "Matplotlib (plt.bar)、Seaborn (sns.barplot)",
            },
            "占比分布": {
                "推荐图表": ["饼图", "环形图", "树状图"],
                "适用场景": "展示各部分占整体的比例",
                "最佳实践": "分类<6用饼图，>6用环形图或树状图，突出重点扇区",
                "工具建议": "Matplotlib (plt.pie)、Plotly (px.pie 交互式)",
            },
            "关联关系": {
                "推荐图表": ["散点图", "气泡图", "热力图"],
                "适用场景": "展示两个或多个变量之间的相关关系",
                "最佳实践": "散点图展示两变量关系，加回归线增强可读性，热力图展示相关矩阵",
                "工具建议": "Matplotlib (plt.scatter)、Seaborn (sns.heatmap)",
            },
            "分布特征": {
                "推荐图表": ["直方图", "箱线图", "小提琴图", "密度图"],
                "适用场景": "展示数据的分布形态和离群值",
                "最佳实践": "箱线图快速检测离群值，小提琴图同时展示分布密度",
                "工具建议": "Seaborn (sns.boxplot/sns.violinplot/sns.histplot)",
            },
            "地理空间": {
                "推荐图表": ["地图", "热力地图", "气泡地图"],
                "适用场景": "展示地理分布和区域差异",
                "最佳实践": "用颜色深浅表示数值，加注标签避免信息过载",
                "工具建议": "Folium、Plotly (px.choropleth)、Pyecharts",
            },
        }

    # 匹配数据类型
    matched_key = None
    for key in chart_guide:
        if key in data_type or data_type in key:
            matched_key = key
            break

    # 如果没有精确匹配，按目的推荐
    if not matched_key:
        purpose_map = {
            "比较": "分类对比",
            "趋势": "时间序列",
            "占比": "占比分布",
            "关系": "关联关系",
            "分布": "分布特征",
        }
        for p_key, c_key in purpose_map.items():
            if p_key in purpose or p_key in data_type:
                matched_key = c_key
                break

    if not matched_key:
        matched_key = "分类对比"  # 默认推荐

    info = chart_guide[matched_key]
    result = (
        f"📊 图表推荐结果：\n\n"
        f"数据类型：{matched_key}\n"
        f"推荐图表：{'、'.join(info['推荐图表'])}\n"
        f"适用场景：{info['适用场景']}\n"
        f"最佳实践：{info.get('最佳实践', '')}\n"
        f"工具建议：{info['工具建议']}\n\n"
        f"💡 其他可选图表类型：{', '.join(k for k in chart_guide if k != matched_key)}"
    )
    return result


@tool
def detect_anomalies(data_str: str, method: str = "3sigma") -> str:
    """检测数据中的异常值

    Args:
        data_str: 逗号分隔的数字字符串
        method: 检测方法，3sigma 或 iqr

    Returns:
        异常值检测结果
    """
    try:
        numbers = [float(x.strip()) for x in data_str.replace("，", ",").split(",") if x.strip()]

        if len(numbers) < 3:
            return "数据点不足（至少需要3个），无法检测异常值"

        n = len(numbers)
        mean_val = sum(numbers) / n
        variance = sum((x - mean_val) ** 2 for x in numbers) / (n - 1)
        std_dev = variance ** 0.5

        if method.lower() == "iqr":
            sorted_nums = sorted(numbers)
            q1_idx = n // 4
            q3_idx = 3 * n // 4
            q1 = sorted_nums[q1_idx]
            q3 = sorted_nums[q3_idx]
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            method_name = "IQR 方法"
        else:
            lower = mean_val - 3 * std_dev
            upper = mean_val + 3 * std_dev
            method_name = "3σ 方法"

        anomalies = [(i, v) for i, v in enumerate(numbers) if v < lower or v > upper]

        result = (
            f"🔍 异常值检测结果（{method_name}）：\n"
            f"  数据量：{n}\n"
            f"  均值：{mean_val:.2f}\n"
            f"  标准差：{std_dev:.2f}\n"
            f"  正常范围：[{lower:.2f}, {upper:.2f}]\n"
            f"  异常值数量：{len(anomalies)}\n"
        )

        if anomalies:
            result += "\n  异常值详情：\n"
            for idx, val in anomalies:
                direction = "偏高" if val > upper else "偏低"
                result += f"    位置 {idx+1}：值={val:.2f}（{direction}）\n"
        else:
            result += "  ✅ 未检测到异常值"

        return result

    except Exception as e:
        return f"检测错误：{e}"


# ============================================================
# Memory - 分析历史追踪器（SQLite持久化）
# ============================================================

class AnalysisTracker:
    """数据分析历史追踪器，使用SQLite持久化"""

    def __init__(self):
        self.chat_history: list = []

    def record(self, action: str, detail: str):
        """记录一次分析活动到SQLite"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO analysis_results (analysis_type, result_json, created_at) VALUES (?, ?, ?)",
            (action, json.dumps({"detail": detail}, ensure_ascii=False), timestamp)
        )
        conn.commit()
        conn.close()

    def add_chat(self, human_msg: str, ai_msg: str):
        """添加对话记录"""
        self.chat_history.append(HumanMessage(content=human_msg))
        self.chat_history.append(AIMessage(content=ai_msg))
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

    def get_summary(self) -> str:
        """获取分析历史摘要"""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM analysis_results")
        total = cursor.fetchone()[0]
        conn.close()

        if total == 0:
            return "暂无分析记录，开始你的数据分析之旅吧！"

        lines = [f"📊 分析历史报告（共 {total} 条记录）"]

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT analysis_type, COUNT(*) as cnt FROM analysis_results GROUP BY analysis_type")
        rows = cursor.fetchall()
        conn.close()

        action_counts = {row["analysis_type"]: row["cnt"] for row in rows}
        for action, count in action_counts.items():
            lines.append(f"  - {action}：{count} 次")

        # 最近记录
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT analysis_type, result_json, created_at FROM analysis_results ORDER BY id DESC LIMIT 5")
        recent = cursor.fetchall()
        conn.close()

        lines.append("\n  最近分析活动：")
        for entry in reversed(recent):
            try:
                detail = json.loads(entry["result_json"]).get("detail", "")[:40] if entry["result_json"] else ""
            except (json.JSONDecodeError, TypeError):
                detail = ""
            lines.append(f"    [{entry['created_at']}] {entry['analysis_type']}：{detail}")

        return "\n".join(lines)


# ============================================================
# 1. 数据分析 - Chain + RAG
# ============================================================

def feature_data_analysis(tracker: AnalysisTracker):
    """功能1：数据分析 - 输入数据描述，AI 生成分析方案（整合 Chain + RAG）"""
    print("\n" + "=" * 60)
    print("  数据分析 - AI 生成分析方案")
    print("=" * 60)
    print("\n💡 技术整合：Chain（LLMChain）+ RAG（知识库检索）")
    print("   先从知识库检索分析方法论，再由 LLM 生成定制化的分析方案")

    # 从SQLite读取数据集数量
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM datasets")
    ds_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM analysis_results")
    kb_count = cursor.fetchone()[0]
    conn.close()
    print(f"\n📚 数据集：{ds_count} 个 | 分析知识库：{len(_ANALYSIS_KNOWLEDGE_CONTENTS)} 条方法论")

    model = get_default_llm()

    prompt = ChatPromptTemplate.from_template(
        "你是一位资深数据分析师。请根据以下信息为用户制定数据分析方案。\n\n"
        "分析方法论参考：\n{methodology}\n\n"
        "数据描述：{data_description}\n"
        "分析目标：{goal}\n\n"
        "请生成完整的分析方案，包括：\n"
        "1. 【数据评估】数据质量、完整性、潜在问题的评估\n"
        "2. 【分析框架】推荐的分析方法和流程\n"
        "3. 【关键指标】需要计算的核心指标\n"
        "4. 【可视化方案】推荐展示结果的图表类型\n"
        "5. 【注意事项】分析中需要特别注意的问题\n"
        "6. 【代码建议】关键步骤的 Python 代码片段"
    )

    chain = prompt | model | StrOutputParser()

    print("\n【交互式数据分析】")
    print("输入数据描述，AI 生成分析方案")
    print("示例：'电商用户行为数据，包含浏览、加购、下单记录，分析用户转化漏斗'")
    print("输入 '退出' 返回主菜单\n")

    while True:
        data_desc = input("数据描述：").strip()

        if data_desc.lower() in ["退出", "exit", "quit", "0"]:
            break

        if not data_desc:
            print("请输入有效描述")
            continue

        goal = input("分析目标（直接回车使用默认）：").strip()
        if not goal:
            goal = "全面分析数据特征和规律"

        try:
            # RAG 检索分析方法论（从SQLite数据集和内置知识库）
            results = retrieve_analysis_knowledge(data_desc + " " + goal, top_k=3)
            methodology = "\n\n".join(
                f"[方法论{i+1}] {doc}" for i, (_, _, doc) in enumerate(results)
            )

            print(f"\n🔍 检索到 {len(results)} 条相关分析方法论")
            print("🤖 AI 正在生成分析方案...\n")
            response = chain.invoke({
                "methodology": methodology,
                "data_description": data_desc,
                "goal": goal,
            })
            print(response)

            # 保存分析结果到SQLite
            tracker.record("数据分析", f"{data_desc[:30]} → {goal[:20]}")

            # 同时保存到analysis_results表
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO analysis_results (analysis_type, result_json, created_at) VALUES (?, ?, ?)",
                ("数据分析", json.dumps({"description": data_desc, "goal": goal, "result": response[:500]}, ensure_ascii=False),
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"❌ 分析失败：{e}")

        print("\n" + "-" * 60)


# ============================================================
# 2. 报表生成 - Structured Output
# ============================================================

def feature_report_generation(tracker: AnalysisTracker):
    """功能2：报表生成 - 输入数据，AI 生成文字报表（整合 Structured Output）"""
    print("\n" + "=" * 60)
    print("  报表生成 - AI 生成文字报表")
    print("=" * 60)
    print("\n💡 技术整合：Structured Output（PydanticOutputParser）")
    print("   使用 Pydantic 模型定义报表结构，确保输出格式规范")
    print("   字段包括：标题、概要、关键发现、核心指标、风险提示、建议")

    model = get_default_llm()
    parser = PydanticOutputParser(pydantic_object=DataReport)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一位专业的数据分析报告撰写专家。请根据提供的数据生成结构化的分析报告。\n\n"
         "{format_instructions}\n\n"
         "注意：核心指标(metrics)中每个dict包含 name、value、change 三个字段。"),
        ("human", "{data_input}")
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | model | parser

    print("\n【交互式报表生成】")
    print("输入数据描述或数据内容，AI 生成结构化报表")
    print("示例：'Q3销售数据：7月营收580万(同比+12%)，8月620万(同比+15%)，9月590万(同比+8%)'")
    print("输入 '退出' 返回主菜单\n")

    while True:
        data_input = input("数据内容：").strip()

        if data_input.lower() in ["退出", "exit", "quit", "0"]:
            break

        if not data_input:
            print("请输入有效数据")
            continue

        try:
            result = chain.invoke({"data_input": data_input})

            print("\n📋 数据分析报表：")
            print("=" * 50)
            print(f"📌 {result.title}")
            print("-" * 50)
            print(f"\n📝 概要：{result.summary}")

            print(f"\n🔍 关键发现：")
            for i, finding in enumerate(result.key_findings, 1):
                print(f"   {i}. {finding}")

            print(f"\n📊 核心指标：")
            for metric in result.metrics:
                name = metric.get("name", "未知")
                value = metric.get("value", "N/A")
                change = metric.get("change", "N/A")
                print(f"   • {name}：{value}（变化：{change}）")

            print(f"\n⚠️ 风险提示：")
            for risk in result.risks:
                print(f"   • {risk}")

            print(f"\n💡 行动建议：")
            for rec in result.recommendations:
                print(f"   • {rec}")

            print(f"\n➡️ 下一步：{result.next_steps}")
            print("=" * 50)

            # 保存报表到SQLite
            conn = get_db()
            cursor = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT INTO reports (title, summary, findings_json, recommendations_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (result.title, result.summary,
                 json.dumps(result.key_findings, ensure_ascii=False),
                 json.dumps(result.recommendations, ensure_ascii=False),
                 now)
            )
            conn.commit()
            conn.close()

            tracker.record("报表生成", result.title)

        except Exception as e:
            print(f"❌ 报表生成失败：{e}")
            print("提示：请确保输入包含具体的数据和指标信息")

        print("\n" + "-" * 60)


# ============================================================
# 3. 趋势预测 - Agent + Tool
# ============================================================

def feature_trend_prediction(tracker: AnalysisTracker):
    """功能3：趋势预测 - Agent 调用工具分析趋势（整合 Agent + Tool）"""
    print("\n" + "=" * 60)
    print("  趋势预测 - AI 预测趋势")
    print("=" * 60)
    print("\n💡 技术整合：Agent（Tool Calling Agent）+ Tool（自定义工具）")
    print("   Agent 可调用的工具：")
    print("   - calculate_statistics：计算基本统计量")
    print("   - predict_trend：趋势预测")
    print("   - detect_anomalies：异常值检测")
    print("   Agent 根据用户输入自动选择和组合工具")

    # 显示SQLite中的历史数据
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name, description FROM datasets")
    datasets = cursor.fetchall()
    conn.close()
    if datasets:
        print(f"\n📊 已有数据集：")
        for ds in datasets:
            print(f"   • {ds['name']}：{ds['description'][:40]}")

    model = get_default_llm()
    tools = [calculate_statistics, predict_trend, detect_anomalies]

    agent = create_react_agent(model, tools, state_modifier="你是一位专业的数据趋势分析师。根据用户提供的历史数据，利用可用工具进行统计分析和趋势预测。你需要：1）先计算基本统计量了解数据特征 2）进行趋势预测 3）检测异常值（如有必要）。最终给出完整的趋势分析结论和建议。")

    print("\n【交互式趋势预测】")
    print("输入历史数据（逗号分隔），AI 自动统计分析并预测趋势")
    print("示例：'最近12个月销售额：100,110,125,118,130,142,138,155,160,172,180,195'")
    print("输入 '退出' 返回主菜单\n")

    while True:
        user_input = input("历史数据或分析需求：").strip()

        if user_input.lower() in ["退出", "exit", "quit", "0"]:
            break

        if not user_input:
            print("请输入有效数据")
            continue

        try:
            result = agent.invoke({"messages": tracker.chat_history + [("user", user_input)]})
            final_message = result["messages"][-1]

            print(f"\n📈 趋势分析结论：\n{final_message.content}")

            # 保存预测结果到SQLite
            tracker.record("趋势预测", user_input[:30])
            tracker.add_chat(user_input, final_message.content[:100])

            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO analysis_results (analysis_type, result_json, created_at) VALUES (?, ?, ?)",
                ("趋势预测", json.dumps({"input": user_input[:100], "result": final_message.content[:500]}, ensure_ascii=False),
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"❌ 预测失败：{e}")

        print("\n" + "-" * 60)


# ============================================================
# 4. 图表建议 - Tool
# ============================================================

def feature_chart_recommendation(tracker: AnalysisTracker):
    """功能4：图表建议 - 推荐数据可视化方式（整合 Tool）"""
    print("\n" + "=" * 60)
    print("  图表建议 - AI 推荐可视化方式")
    print("=" * 60)
    print("\n💡 技术整合：Tool（@tool 装饰器自定义工具）")
    print("   使用 recommend_chart 工具根据数据类型推荐图表")
    print("   图表建议数据从SQLite读取")

    model = get_default_llm()

    print("\n【交互式图表建议】")
    print("输入数据类型或分析目的，AI 推荐最佳可视化方式")
    print("示例：'我想展示月度销售趋势' 或 '占比分布'")
    print("输入 '退出' 返回主菜单\n")

    while True:
        data_type = input("数据类型或目的：").strip()

        if data_type.lower() in ["退出", "exit", "quit", "0"]:
            break

        if not data_type:
            print("请输入有效内容")
            continue

        try:
            # 调用图表推荐工具（从SQLite读取）
            chart_result = recommend_chart.invoke({"data_type": data_type, "purpose": data_type})
            print(f"\n{chart_result}")

            # 额外：LLM 给出代码示例
            print("\n🤖 AI 正在生成代码示例...\n")
            code_prompt = ChatPromptTemplate.from_template(
                "根据以下图表推荐，给出一个 Python 可视化代码示例（使用 matplotlib 或 seaborn）。"
                "代码要简洁可运行，包含模拟数据。\n\n"
                "图表推荐：{chart_info}\n\n"
                "请只输出代码，不要额外解释。"
            )
            code_chain = code_prompt | model | StrOutputParser()
            code_result = code_chain.invoke({"chart_info": chart_result})
            print(f"📝 代码示例：\n{code_result}")

            tracker.record("图表建议", data_type[:20])

        except Exception as e:
            print(f"❌ 建议失败：{e}")

        print("\n" + "-" * 60)


# ============================================================
# 5. 数据清洗 - Chain + RAG
# ============================================================

def feature_data_cleaning(tracker: AnalysisTracker):
    """功能5：数据清洗 - 输入原始数据描述，AI 给出清洗方案（整合 Chain + RAG）"""
    print("\n" + "=" * 60)
    print("  数据清洗 - AI 给出清洗方案")
    print("=" * 60)
    print("\n💡 技术整合：Chain（LLMChain）Cleansing + RAG（知识库检索）")
    print("   检索数据清洗方法论，结合 LLM 生成定制化的清洗方案")

    model = get_default_llm()

    prompt = ChatPromptTemplate.from_template(
        "你是一位数据工程专家，擅长数据清洗和质量治理。请根据以下信息制定数据清洗方案。\n\n"
        "清洗方法论参考：\n{methodology}\n\n"
        "原始数据描述：{raw_data_desc}\n"
        "数据问题：{problems}\n\n"
        "请生成完整的清洗方案，包括：\n"
        "1. 【问题诊断】识别数据中的质量问题\n"
        "2. 【清洗策略】针对每个问题的处理方法\n"
        "3. 【执行步骤】具体的清洗操作步骤（含 Python 代码）\n"
        "4. 【质量验证】清洗后如何验证数据质量\n"
        "5. 【预防措施】如何避免同类问题再次出现"
    )

    chain = prompt | model | StrOutputParser()

    print("\n【交互式数据清洗】")
    print("输入原始数据描述和数据问题，AI 给出清洗方案")
    print("示例：'用户注册表，有15%的缺失值，日期格式不统一，存在重复记录'")
    print("输入 '退出' 返回主菜单\n")

    while True:
        raw_desc = input("原始数据描述：").strip()

        if raw_desc.lower() in ["退出", "exit", "quit", "0"]:
            break

        if not raw_desc:
            print("请输入有效描述")
            continue

        problems = input("已知数据问题（直接回车让AI诊断）：").strip()
        if not problems:
            problems = "需要AI自动诊断"

        try:
            # RAG 检索清洗方法论
            results = retrieve_analysis_knowledge(raw_desc + " 数据清洗", top_k=3)
            methodology = "\n\n".join(
                f"[方法论{i+1}] {doc}" for i, (_, _, doc) in enumerate(results)
            )

            print("🤖 AI 正在生成清洗方案...\n")
            response = chain.invoke({
                "methodology": methodology,
                "raw_data_desc": raw_desc,
                "problems": problems,
            })
            print(response)

            # 保存清洗方案到SQLite
            tracker.record("数据清洗", raw_desc[:30])

            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO analysis_results (analysis_type, result_json, created_at) VALUES (?, ?, ?)",
                ("数据清洗", json.dumps({"description": raw_desc, "problems": problems, "result": response[:500]}, ensure_ascii=False),
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"❌ 方案生成失败：{e}")

        print("\n" + "-" * 60)


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    # 初始化数据库
    init_db()

    print("\n" + "=" * 60)
    print("  AI 数据分析师 - 端到端实战项目")
    print("=" * 60)
    print("\n完整的 AI 数据分析师，整合 Chain / Agent / Tool / Structured Output / RAG / Memory")

    print("\n功能模块：")
    print("  1. 数据分析（输入数据描述，AI 生成分析方案）    [Chain+RAG]")
    print("  2. 报表生成（输入数据，AI 生成文字报表）        [Structured Output]")
    print("  3. 趋势预测（输入历史数据，AI 预测趋势）        [Agent+Tool]")
    print("  4. 图表建议（输入数据类型，AI 推荐可视化方式）   [Tool]")
    print("  5. 数据清洗（输入原始数据描述，AI 给出清洗方案） [Chain+RAG]")

    print("\n应用场景：企业经营分析 / 市场研究 / 运营监控")

    # 创建分析追踪器（整个会话共享）
    tracker = AnalysisTracker()

    while True:
        print("\n" + "=" * 60)
        print("  AI 数据分析师")
        print("=" * 60)
        print("  1. 数据分析（输入数据描述，AI 生成分析方案）")
        print("  2. 报表生成（输入数据，AI 生成文字报表）")
        print("  3. 趋势预测（输入历史数据，AI 预测趋势）")
        print("  4. 图表建议（输入数据类型，AI 推荐可视化方式）")
        print("  5. 数据清洗（输入原始数据描述，AI 给出清洗方案）")
        print("\n  0. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (0-5): ").strip()

        if choice == "1":
            feature_data_analysis(tracker)
        elif choice == "2":
            feature_report_generation(tracker)
        elif choice == "3":
            feature_trend_prediction(tracker)
        elif choice == "4":
            feature_chart_recommendation(tracker)
        elif choice == "5":
            feature_data_cleaning(tracker)
        elif choice == "0":
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM analysis_results")
            has_records = cursor.fetchone()[0] > 0
            conn.close()
            if has_records:
                print("\n" + tracker.get_summary())
            print("\n感谢使用 AI 数据分析师！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
