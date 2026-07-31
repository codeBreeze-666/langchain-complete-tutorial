"""
LangChain 结构化输出 - 实战交互式案例
======================================

本示例演示如何让 AI 输出结构化的数据格式

核心概念：
- Pydantic 模型：定义数据结构
- PydanticOutputParser：自动解析和验证
- 结构化输出：从文本提取特定格式的数据

应用场景：
- 信息提取：从文本中提取结构化信息
- 表单填写：自动填写表单字段
- 数据标注：给数据添加标签和分类
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from src.utils.llm_loader import get_default_llm


# ============================================================
# 数据模型定义
# ============================================================

class PersonInfo(BaseModel):
    """人物信息模型"""
    name: str = Field(description="人物姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")
    skills: list[str] = Field(default_factory=list, description="技能列表")
    email: str = Field(default="", description="邮箱地址")


class ProductInfo(BaseModel):
    """产品信息模型"""
    name: str = Field(description="产品名称")
    price: float = Field(description="价格")
    category: str = Field(description="产品类别")
    stock_status: str = Field(description="库存状态")
    rating: float = Field(default=0.0, description="评分")


class EventInfo(BaseModel):
    """活动信息模型"""
    title: str = Field(description="活动标题")
    date: str = Field(description="活动日期")
    location: str = Field(description="活动地点")
    organizer: str = Field(description="主办方")
    participants: int = Field(default=0, description="参与人数")


# ============================================================
# 1. 人物信息提取
# ============================================================

def demo_person_extraction():
    """示例1：人物信息提取"""
    print("\n" + "="*60)
    print("示例1：人物信息提取")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 定义 Pydantic 模型规定数据结构")
    print("   - PydanticOutputParser 自动生成格式说明")
    print("   - AI 会按照定义的结构输出数据")

    model = get_default_llm()

    # 创建解析器
    parser = PydanticOutputParser(pydantic_object=PersonInfo)

    # 创建提示词（自动包含格式说明）
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个信息提取专家。请从用户提供的文本中提取人物信息。\n\n{format_instructions}"),
        ("human", "{text}")
    ])

    # 部分填充格式说明
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    # 创建链
    chain = prompt | model | parser

    print("\n【交互式人物信息提取】")
    print("提示：输入包含人物信息的文本，AI 会自动提取")
    print("输入 '退出' 结束\n")

    while True:
        text = input("输入文本：").strip()

        if text.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not text:
            print("请输入有效文本")
            continue

        try:
            # 提取结构化信息
            result = chain.invoke({"text": text})

            print("\n【提取结果】")
            print(f"姓名：{result.name}")
            print(f"年龄：{result.age}")
            print(f"职业：{result.occupation}")
            print(f"技能：{', '.join(result.skills) if result.skills else '无'}")
            print(f"邮箱：{result.email if result.email else '无'}")
            print(f"数据类型：{type(result).__name__}")

        except Exception as e:
            print(f"❌ 提取失败：{e}")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. Pydantic 模型定义数据结构")
    print("   2. Field(description='...') 帮助 AI 理解字段含义")
    print("   3. PydanticOutputParser 自动解析和验证")


# ============================================================
# 2. 产品信息提取
# ============================================================

def demo_product_extraction():
    """示例2：产品信息提取"""
    print("\n" + "="*60)
    print("示例2：产品信息提取")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 可以定义多个不同的数据模型")
    print("   - 每个模型对应不同的提取任务")
    print("   - 字段可以有验证规则和默认值")

    model = get_default_llm()

    parser = PydanticOutputParser(pydantic_object=ProductInfo)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个产品信息提取专家。\n\n{format_instructions}"),
        ("human", "{text}")
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | model | parser

    print("\n【交互式产品信息提取】")
    print("提示：输入包含产品信息的文本")
    print("输入 '退出' 结束\n")

    while True:
        text = input("输入文本：").strip()

        if text.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not text:
            print("请输入有效文本")
            continue

        try:
            result = chain.invoke({"text": text})

            print("\n【提取结果】")
            print(f"产品名称：{result.name}")
            print(f"价格：￥{result.price:.2f}")
            print(f"类别：{result.category}")
            print(f"库存状态：{result.stock_status}")
            print(f"评分：{result.rating}/5.0")

        except Exception as e:
            print(f"❌ 提取失败：{e}")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 不同任务使用不同的数据模型")
    print("   2. 可以提取数值、文本、分类等多种类型")
    print("   3. 适合电商、库存管理等场景")


# ============================================================
# 3. 活动信息提取
# ============================================================

def demo_event_extraction():
    """示例3：活动信息提取"""
    print("\n" + "="*60)
    print("示例3：活动信息提取")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 结构化输出适合提取复杂信息")
    print("   - 可以处理日期、地点、组织等信息")
    print("   - 自动验证数据格式")

    model = get_default_llm()

    parser = PydanticOutputParser(pydantic_object=EventInfo)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个活动信息提取专家。\n\n{format_instructions}"),
        ("human", "{text}")
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | model | parser

    print("\n【交互式活动信息提取】")
    print("提示：输入包含活动信息的文本")
    print("输入 '退出' 结束\n")

    while True:
        text = input("输入文本：").strip()

        if text.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not text:
            print("请输入有效文本")
            continue

        try:
            result = chain.invoke({"text": text})

            print("\n【提取结果】")
            print(f"活动标题：{result.title}")
            print(f"活动日期：{result.date}")
            print(f"活动地点：{result.location}")
            print(f"主办方：{result.organizer}")
            print(f"参与人数：{result.participants}")

        except Exception as e:
            print(f"❌ 提取失败：{e}")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 可以提取复杂的多类型字段")
    print("   2. 适合会议、活动、日程管理")
    print("   3. 自动验证字段类型和格式")


# ============================================================
# 4. 批量信息提取
# ============================================================

def demo_batch_extraction():
    """示例4：批量信息提取"""
    print("\n" + "="*60)
    print("示例4：批量信息提取")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 可以一次性提取多条信息")
    print("   - 使用 list[Model] 定义列表类型")
    print("   - 适合批量处理场景")

    model = get_default_llm()

    # 定义列表模型
    from typing import List

    class PersonList(BaseModel):
        """人物列表"""
        persons: List[PersonInfo] = Field(description="人物列表")

    parser = PydanticOutputParser(pydantic_object=PersonList)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个信息提取专家。请提取文本中的所有人物信息。\n\n{format_instructions}"),
        ("human", "{text}")
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | model | parser

    print("\n【交互式批量提取】")
    print("提示：输入包含多个人物信息的文本")
    print("输入 '退出' 结束\n")

    while True:
        text = input("输入文本：").strip()

        if text.lower() in ['退出', 'exit', 'quit']:
            print("结束演示")
            break

        if not text:
            print("请输入有效文本")
            continue

        try:
            result = chain.invoke({"text": text})

            print(f"\n【提取结果】共找到 {len(result.persons)} 个人物：")

            for i, person in enumerate(result.persons, 1):
                print(f"\n{i}. {person.name}")
                print(f"   年龄：{person.age}")
                print(f"   职业：{person.occupation}")
                if person.skills:
                    print(f"   技能：{', '.join(person.skills)}")

        except Exception as e:
            print(f"❌ 提取失败：{e}")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 使用 List[Model] 定义批量提取")
    print("   2. 一次性处理多条信息")
    print("   3. 适合团队、组织等信息提取")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain 结构化输出 - 实战案例")
    print("="*60)
    print("\n本示例演示如何让 AI 输出结构化数据")
    print("\n核心概念：")
    print("  • Pydantic 模型：定义数据结构")
    print("  • PydanticOutputParser：自动解析验证")
    print("  • 结构化输出：从文本提取特定格式")
    print("\n应用场景：")
    print("  • 信息提取、表单填写、数据标注")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 人物信息提取")
        print("  2. 产品信息提取")
        print("  3. 活动信息提取")
        print("  4. 批量信息提取")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_person_extraction()
        elif choice == "2":
            demo_product_extraction()
        elif choice == "3":
            demo_event_extraction()
        elif choice == "4":
            demo_batch_extraction()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()