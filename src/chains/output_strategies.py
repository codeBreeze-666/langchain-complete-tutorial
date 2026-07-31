"""
LangChain 输出策略 - 实战交互式案例
====================================

本示例演示 LangChain 中不同输出解析策略的使用方法

核心概念：
- StrOutputParser：将 LLM 输出转为纯字符串
- JsonOutputParser：将 LLM 输出转为 JSON 对象
- CommaSeparatedListOutputParser：将 LLM 输出转为逗号分隔列表
- 自定义解析器：继承 OutputParser 实现自定义逻辑

应用场景：
- 字符串输出：简单对话、文章生成
- JSON 输出：结构化数据提取、表单填写
- 列表输出：关键词提取、推荐列表
- 自定义解析器：特殊格式需求、复杂转换逻辑
"""

import os
import sys
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.utils.llm_loader import get_default_llm


# ============================================================
# 1. 字符串输出 - 最基础的输出方式
# ============================================================

def demo_str_output():
    """示例1：字符串输出（最基础的输出方式）"""
    print("\n" + "="*60)
    print("示例1：字符串输出（最基础的输出方式）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - StrOutputParser 将 AIMessage 提取为纯字符串")
    print("   - 适用于问答、翻译、写作等纯文本场景")
    print("   - 是最常用、最简单的输出解析器")

    model = get_default_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{role}，请用简洁专业的语言回答问题。"),
        ("human", "{question}")
    ])

    chain = prompt | model | StrOutputParser()

    print("\n【交互式对话 - 角色扮演问答】")
    print("提示：先选择角色，再输入问题")
    print("输入 '退出' 结束\n")

    role = input("选择角色（如：程序员/医生/律师/教师）：").strip() or "助手"

    while True:
        question = input(f"\n向{role}提问：").strip()

        if question.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not question:
            print("请输入有效问题")
            continue

        response = chain.invoke({"role": role, "question": question})
        print(f"\n{role}回答：{response}")

    print("\n✅ 实战要点总结：")
    print("   1. StrOutputParser 直接返回字符串，无需额外处理")
    print("   2. 适合不需要结构化的纯文本场景")
    print("   3. 可以搭配任意 PromptTemplate 使用")


# ============================================================
# 2. JSON 输出 - 使用 JsonOutputParser
# ============================================================

def demo_json_output():
    """示例2：JSON 输出（使用 JsonOutputParser）"""
    print("\n" + "="*60)
    print("示例2：JSON 输出（使用 JsonOutputParser）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - JsonOutputParser 将 LLM 输出解析为字典")
    print("   - 可结合 Pydantic 模型定义期望的 JSON 结构")
    print("   - 自动在 Prompt 中注入格式说明，引导 LLM 输出 JSON")

    model = get_default_llm()

    # 定义期望的输出结构
    class BookAnalysis(BaseModel):
        """书籍分析模型"""
        title: str = Field(description="书名")
        author: str = Field(description="作者")
        genre: str = Field(description="类型/题材")
        rating: float = Field(description="评分（1-10）")
        summary: str = Field(description="一句话总结")
        keywords: list = Field(description="关键词列表")

    parser = JsonOutputParser(pydantic_object=BookAnalysis)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的书评人，请分析用户提到的书籍。"),
        ("human", "{book_name}\n\n{format_instructions}"),
    ])

    chain = prompt | model | parser

    print("\n【交互式书籍分析助手】")
    print("提示：输入书名，AI 将返回结构化的 JSON 分析结果")
    print("输入 '退出' 结束\n")

    while True:
        book_name = input("书名：").strip()

        if book_name.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not book_name:
            print("请输入有效书名")
            continue

        result = chain.invoke({
            "book_name": book_name,
            "format_instructions": parser.get_format_instructions()
        })

        print(f"\n📖 书籍分析结果：")
        print(f"  书名：{result.get('title', 'N/A')}")
        print(f"  作者：{result.get('author', 'N/A')}")
        print(f"  类型：{result.get('genre', 'N/A')}")
        print(f"  评分：{result.get('rating', 'N/A')}/10")
        print(f"  总结：{result.get('summary', 'N/A')}")
        print(f"  关键词：{', '.join(result.get('keywords', []))}")
        print(f"\n  原始 JSON：")
        print(f"  {json.dumps(result, ensure_ascii=False, indent=2)}")

    print("\n✅ 实战要点总结：")
    print("   1. JsonOutputParser 自动生成 format_instructions 注入 Prompt")
    print("   2. 输出为 Python 字典，可直接按 key 访问")
    print("   3. 结合 Pydantic 模型可以约束字段名和类型")


# ============================================================
# 3. 列表输出 - 使用 CommaSeparatedListOutputParser
# ============================================================

def demo_list_output():
    """示例3：列表输出（使用 CommaSeparatedListOutputParser）"""
    print("\n" + "="*60)
    print("示例3：列表输出（使用 CommaSeparatedListOutputParser）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - CommaSeparatedListOutputParser 将输出拆分为列表")
    print("   - 适合关键词提取、推荐列表、分类标签等场景")
    print("   - 输出为 Python list，可直接遍历使用")

    model = get_default_llm()

    parser = CommaSeparatedListOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{domain}专家，请根据用户需求生成相关列表。"),
        ("human", "{request}\n\n{format_instructions}"),
    ])

    chain = prompt | model | parser

    print("\n【交互式列表生成助手】")
    print("提示：选择领域，然后输入需求，AI 将生成逗号分隔的列表")
    print("输入 '退出' 结束\n")

    domain = input("选择领域（如：编程/美食/旅行/电影）：").strip() or "通用"

    while True:
        request = input(f"\n{domain}领域的列表需求：").strip()

        if request.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not request:
            print("请输入有效需求")
            continue

        result = chain.invoke({
            "domain": domain,
            "request": request,
            "format_instructions": parser.get_format_instructions()
        })

        print(f"\n📋 生成列表（共 {len(result)} 项）：")
        for i, item in enumerate(result, 1):
            print(f"  {i}. {item}")

    print("\n✅ 实战要点总结：")
    print("   1. CommaSeparatedListOutputParser 自动按逗号拆分输出")
    print("   2. 返回 Python 列表，便于后续遍历和索引")
    print("   3. 适合数量不固定、格式简单的枚举场景")


# ============================================================
# 4. 自定义解析器 - 创建自定义的输出解析器
# ============================================================

def demo_custom_parser():
    """示例4：自定义解析器（创建自定义的输出解析器）"""
    print("\n" + "="*60)
    print("示例4：自定义解析器（创建自定义的输出解析器）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 继承 OutputParser 基类实现自定义解析逻辑")
    print("   - 需要实现 parse() 和 get_format_instructions() 方法")
    print("   - 适合标准解析器无法满足的特殊格式需求")

    from langchain_core.output_parsers import BaseOutputParser

    class KeyValueOutputParser(BaseOutputParser):
        """
        自定义键值对解析器
        解析格式：key1=value1, key2=value2, ...
        输出为 Python 字典
        """
        def parse(self, text: str) -> dict:
            """将 'key1=val1, key2=val2' 格式解析为字典"""
            result = {}
            # 清理多余空白和换行
            text = text.strip()
            # 尝试按行分割再合并，处理多行输出
            pairs = text.replace("\n", ",").split(",")
            for pair in pairs:
                pair = pair.strip()
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    result[key.strip()] = value.strip()
            return result

        def get_format_instructions(self) -> str:
            """返回格式说明，注入到 Prompt 中引导 LLM 输出"""
            return "请按照 key1=value1, key2=value2 的格式输出，每对用逗号分隔。"

    model = get_default_llm()

    parser = KeyValueOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个数据分析助手，请从用户描述中提取关键属性。"),
        ("human", "{description}\n\n{format_instructions}"),
    ])

    chain = prompt | model | parser

    print("\n【交互式属性提取助手】")
    print("提示：输入产品或物品描述，AI 提取为 key=value 格式")
    print("输入 '退出' 结束\n")

    while True:
        description = input("物品描述：").strip()

        if description.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not description:
            print("请输入有效描述")
            continue

        result = chain.invoke({
            "description": description,
            "format_instructions": parser.get_format_instructions()
        })

        print(f"\n🔑 提取属性：")
        for key, value in result.items():
            print(f"  {key} = {value}")

        print(f"\n  字典类型验证：{type(result).__name__}")

    print("\n✅ 实战要点总结：")
    print("   1. 继承 BaseOutputParser 并实现 parse() 方法")
    print("   2. get_format_instructions() 引导 LLM 输出目标格式")
    print("   3. 自定义解析器可处理任意非标准输出格式")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain 输出策略 - 实战案例")
    print("="*60)
    print("\n本示例演示 LangChain 中不同的输出解析策略")
    print("\n核心概念：")
    print("  • StrOutputParser：字符串输出（最基础）")
    print("  • JsonOutputParser：JSON 输出（结构化数据）")
    print("  • CommaSeparatedListOutputParser：列表输出（枚举项）")
    print("  • 自定义解析器：满足特殊格式需求")
    print("\n应用场景：")
    print("  • 字符串输出：问答、翻译、写作")
    print("  • JSON 输出：数据提取、表单填写")
    print("  • 列表输出：关键词提取、推荐列表")
    print("  • 自定义解析器：特殊格式转换")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 字符串输出（最基础的输出方式）")
        print("  2. JSON 输出（使用 JsonOutputParser）")
        print("  3. 列表输出（使用 CommaSeparatedListOutputParser）")
        print("  4. 自定义解析器（创建自定义的输出解析器）")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_str_output()
        elif choice == "2":
            demo_json_output()
        elif choice == "3":
            demo_list_output()
        elif choice == "4":
            demo_custom_parser()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
