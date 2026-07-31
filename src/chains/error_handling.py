"""
LangChain 错误处理与调试 - 实战交互式案例
==========================================

本示例演示 LangChain 中常见的错误处理策略和调试技巧

核心概念：
- 错误重试（Retry）：API 调用失败时自动重试
- 模型降级（Fallback）：主模型失败时切换备用模型
- 输出验证（Validation）：验证模型输出是否符合预期格式
- 优雅降级（Graceful Degradation）：出错时返回友好的降级结果

应用场景：
- 网络波动导致 API 调用失败
- 主模型服务不可用，需要快速切换
- 模型输出格式不稳定，需要校验和修复
- 生产环境需要保证服务可用性
"""

import os
import sys
import time
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.utils.llm_loader import get_default_llm


# ============================================================
# 1. 错误重试 - API 调用失败时自动重试
# ============================================================

def demo_retry_on_error():
    """示例1：错误重试 - 当 API 调用失败时自动重试"""
    print("\n" + "="*60)
    print("示例1：错误重试（API 调用失败时自动重试）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 网络波动、限流等导致 API 调用偶尔失败")
    print("   - 通过指数退避重试策略提高成功率")
    print("   - 设置最大重试次数避免无限循环")

    # 自定义重试装饰器
    def retry_chain_invoke(chain, inputs, max_retries=3, base_delay=1):
        """
        带重试机制的链调用

        Args:
            chain: LangChain 链对象
            inputs: 输入参数字典
            max_retries: 最大重试次数
            base_delay: 基础延迟秒数（指数退避）

        Returns:
            调用结果或 None（全部失败时）
        """
        for attempt in range(1, max_retries + 1):
            try:
                result = chain.invoke(inputs)
                if attempt > 1:
                    print(f"   ✅ 第 {attempt} 次重试成功！")
                return result
            except Exception as e:
                print(f"   ❌ 第 {attempt} 次调用失败: {type(e).__name__}: {e}")
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    print(f"   ⏳ 等待 {delay} 秒后重试...")
                    time.sleep(delay)
                else:
                    print(f"   🚫 已达最大重试次数 ({max_retries})，放弃调用")
                    return None

    # 创建链
    model = get_default_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的技术顾问，回答要简洁准确"),
        ("human", "{user_question}")
    ])
    chain = prompt | model | StrOutputParser()

    print("\n【交互式问答（带自动重试）】")
    print("提示：输入技术问题，AI 会回答（内置重试机制）")
    print("输入 '退出' 结束\n")

    while True:
        user_question = input("你的问题：").strip()

        if user_question.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not user_question:
            print("请输入有效内容")
            continue

        print(f"\n【调用链（最大重试 3 次，指数退避）】")
        result = retry_chain_invoke(chain, {"user_question": user_question})

        if result:
            print(f"\nAI：{result}")
        else:
            print("\n⚠️ 所有重试均失败，请检查网络或 API 配置")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 指数退避（Exponential Backoff）是最常用的重试策略")
    print("   2. 重试次数不宜过多，一般 3-5 次即可")
    print("   3. 每次重试间隔翻倍，避免短时间内大量请求")
    print("   4. 对瞬时错误（网络超时）重试，对逻辑错误不重试")


# ============================================================
# 2. 模型降级 - 主模型失败时切换备用模型
# ============================================================

def demo_fallback_model():
    """示例2：模型降级 - 主模型失败时切换备用模型"""
    print("\n" + "="*60)
    print("示例2：模型降级（主模型失败时切换备用模型）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 主模型可能因限流、停机等原因不可用")
    print("   - LangChain 提供.with_fallbacks() 实现降级")
    print("   - 降级链自动按顺序尝试备用方案")

    # 主模型
    primary_model = get_default_llm()
    # 备用模型（降低温度，使用相同模型模拟降级）
    from src.utils.llm_loader import LLMLoader
    fallback_model = LLMLoader.create_llm(temperature=0.3)

    # 创建降级链：主模型 -> 备用模型
    primary_chain = ChatPromptTemplate.from_messages([
        ("system", "你是一个高级技术专家，回答要专业深入"),
        ("human", "{user_input}")
    ]) | primary_model | StrOutputParser()

    fallback_chain = ChatPromptTemplate.from_messages([
        ("system", "你是一个通用助手，用简单语言回答"),
        ("human", "{user_input}")
    ]) | fallback_model | StrOutputParser()

    # 组装降级链
    chain_with_fallback = primary_chain.with_fallbacks([fallback_chain])

    # 手动降级演示函数
    def invoke_with_fallback(primary, fallbacks, inputs):
        """
        手动实现降级调用逻辑

        Args:
            primary: 主链
            fallbacks: 备用链列表
            inputs: 输入参数

        Returns:
            调用结果
        """
        # 尝试主链
        try:
            print("   🔵 尝试主模型...")
            result = primary.invoke(inputs)
            print("   ✅ 主模型调用成功")
            return result, "主模型"
        except Exception as e:
            print(f"   ❌ 主模型失败: {type(e).__name__}: {e}")

        # 尝试备用链
        for i, fallback in enumerate(fallbacks, 1):
            try:
                print(f"   🟡 尝试备用模型 #{i}...")
                result = fallback.invoke(inputs)
                print(f"   ✅ 备用模型 #{i} 调用成功")
                return result, f"备用模型 #{i}"
            except Exception as e:
                print(f"   ❌ 备用模型 #{i} 失败: {type(e).__name__}: {e}")

        print("   🚫 所有模型均失败")
        return None, None

    print("\n【交互式问答（带模型降级）】")
    print("提示：输入问题，AI 回答（主模型优先，失败自动降级）")
    print("输入 '退出' 结束\n")

    while True:
        user_input = input("你的问题：").strip()

        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not user_input:
            print("请输入有效内容")
            continue

        print(f"\n【降级调用演示】")
        # 方式一：使用 LangChain 内置 with_fallbacks
        print("\n方式一：LangChain 内置 .with_fallbacks()")
        try:
            result = chain_with_fallback.invoke({"user_input": user_input})
            print(f"AI：{result}")
        except Exception as e:
            print(f"所有模型均失败: {e}")

        # 方式二：手动降级（更灵活的控制）
        print("\n方式二：手动降级调用")
        result, used_model = invoke_with_fallback(
            primary_chain, [fallback_chain], {"user_input": user_input}
        )
        if result:
            print(f"AI（使用 {used_model}）：{result}")
        else:
            print("⚠️ 所有模型均不可用")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. .with_fallbacks() 是 LangChain 原生降级方案")
    print("   2. 可配置多个备用模型，按顺序尝试")
    print("   3. 手动降级可以添加更细粒度的控制（如日志、告警）")
    print("   4. 生产建议：主模型 + 同厂商轻量模型 + 异厂商模型")


# ============================================================
# 3. 输出验证 - 验证模型输出是否符合预期
# ============================================================

# 定义期望的输出结构
class ProductInfo(BaseModel):
    """产品信息结构定义"""
    name: str = Field(description="产品名称")
    category: str = Field(description="产品类别")
    price: float = Field(description="价格（元）")
    description: str = Field(description="简要描述")


class QAResult(BaseModel):
    """问答结果结构定义"""
    question: str = Field(description="用户问题")
    answer: str = Field(description="回答内容")
    confidence: float = Field(description="置信度 0-1")


def demo_output_validation():
    """示例3：输出验证 - 验证模型输出是否符合预期"""
    print("\n" + "="*60)
    print("示例3：输出验证（验证模型输出是否符合预期）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - LLM 输出不稳定，可能格式错误或内容不符")
    print("   - PydanticOutputParser 强制结构化输出")
    print("   - 自定义验证逻辑捕获业务规则违反")

    model = get_default_llm()

    print("\n【交互式产品信息提取（带输出验证）】")
    print("提示：输入产品描述，AI 提取结构化信息并验证")
    print("输入 '退出' 结束\n")

    while True:
        product_desc = input("产品描述：").strip()

        if product_desc.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not product_desc:
            print("请输入有效内容")
            continue

        # --- 方式一：PydanticOutputParser 强制结构化 ---
        print("\n【方式一：PydanticOutputParser 结构化输出】")
        pydantic_parser = PydanticOutputParser(pydantic_object=ProductInfo)

        structured_prompt = ChatPromptTemplate.from_template(
            "从以下描述中提取产品信息：\n\n{product_desc}\n\n"
            "{format_instructions}\n\n"
            "请严格按照格式输出，不要添加额外内容。"
        )

        chain = structured_prompt | model | pydantic_parser

        try:
            result = chain.invoke({
                "product_desc": product_desc,
                "format_instructions": pydantic_parser.get_format_instructions()
            })
            print(f"  ✅ 解析成功！")
            print(f"  产品名称: {result.name}")
            print(f"  产品类别: {result.category}")
            print(f"  价格: ¥{result.price:.2f}")
            print(f"  描述: {result.description}")
        except Exception as e:
            print(f"  ❌ Pydantic 解析失败: {type(e).__name__}")
            print(f"  原因: {e}")

            # 解析失败时，尝试用 StrOutputParser 做降级
            print("\n  🔄 降级为纯文本输出...")
            text_prompt = ChatPromptTemplate.from_template(
                "从以下描述中提取产品信息，用简洁格式列出：\n\n{product_desc}"
            )
            text_chain = text_prompt | model | StrOutputParser()
            try:
                text_result = text_chain.invoke({"product_desc": product_desc})
                print(f"  📝 降级结果:\n{text_result}")
            except Exception as e2:
                print(f"  ❌ 降级也失败: {e2}")

        # --- 方式二：自定义验证逻辑 ---
        print("\n【方式二：自定义验证逻辑】")
        qa_prompt = ChatPromptTemplate.from_template(
            "回答以下问题，给出置信度（0-1之间的数字）：\n\n"
            "问题: {question}\n\n"
            "请用 JSON 格式输出，包含 question、answer、confidence 三个字段。"
        )

        qa_chain = qa_prompt | model | StrOutputParser()

        try:
            raw_output = qa_chain.invoke({"question": product_desc})

            # 尝试解析 JSON
            try:
                # 去除可能的 markdown 代码块标记
                cleaned = raw_output.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                    cleaned = cleaned.strip()

                parsed = json.loads(cleaned)

                # 自定义验证规则
                errors = []
                if "answer" not in parsed or not parsed["answer"]:
                    errors.append("缺少 answer 字段或为空")
                if "confidence" in parsed:
                    conf = parsed["confidence"]
                    if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                        errors.append(f"confidence 值无效: {conf}（应在 0-1 之间）")
                else:
                    errors.append("缺少 confidence 字段")

                if errors:
                    print(f"  ⚠️ 验证未通过：")
                    for err in errors:
                        print(f"     - {err}")
                    print(f"  原始输出: {raw_output[:200]}")
                else:
                    print(f"  ✅ 验证通过！")
                    print(f"  回答: {parsed['answer']}")
                    print(f"  置信度: {parsed['confidence']}")
            except json.JSONDecodeError:
                print(f"  ⚠️ 输出不是合法 JSON，降级为纯文本")
                print(f"  原始输出: {raw_output[:200]}")

        except Exception as e:
            print(f"  ❌ 调用失败: {type(e).__name__}: {e}")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. PydanticOutputParser 强制输出为指定结构")
    print("   2. 解析失败时降级为 StrOutputParser 获取纯文本")
    print("   3. 自定义验证可以检查业务规则（如置信度范围）")
    print("   4. JSON 输出需要清理 markdown 代码块标记")


# ============================================================
# 4. 优雅降级 - 出错时返回友好的降级结果
# ============================================================

def demo_graceful_degradation():
    """示例4：优雅降级 - 出错时返回友好的降级结果"""
    print("\n" + "="*60)
    print("示例4：优雅降级（出错时返回友好的降级结果）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 生产环境不能因为模型失败就完全不可用")
    print("   - 优雅降级：部分功能退化而非整体崩溃")
    print("   - 预设缓存/模板作为最终兜底方案")

    model = get_default_llm()

    # 预设的兜底回答模板
    FALLBACK_TEMPLATES = {
        "翻译": "抱歉，翻译服务暂时不可用，请稍后重试。",
        "总结": "抱歉，总结服务暂时不可用，原始内容如下：\n{content}",
        "问答": "抱歉，AI 问答服务暂时不可用，请联系管理员。",
        "默认": "抱歉，服务暂时不可用，请稍后再试。"
    }

    # 降级包装器
    def graceful_invoke(chain, inputs, service_name="默认", context=None):
        """
        优雅降级调用

        Args:
            chain: LangChain 链
            inputs: 输入参数
            service_name: 服务名称（用于选择兜底模板）
            context: 上下文信息（用于填充模板）

        Returns:
            (result, status): 结果和状态（"success"/"degraded"/"fallback"）
        """
        # 第一层：正常调用
        try:
            result = chain.invoke(inputs)
            if result and len(str(result).strip()) > 0:
                return result, "success"
            raise ValueError("模型返回空结果")
        except Exception as e:
            print(f"   ⚠️ 正常调用失败: {type(e).__name__}")

        # 第二层：简化 prompt 重试
        try:
            print(f"   🔄 尝试简化提示词重试...")
            simplified_prompt = ChatPromptTemplate.from_messages([
                ("system", "请简短回答"),
                ("human", "{user_input}")
            ])
            simple_chain = simplified_prompt | model | StrOutputParser()
            result = simple_chain.invoke({"user_input": inputs.get("user_input", "")})
            if result and len(str(result).strip()) > 0:
                print(f"   ✅ 简化重试成功")
                return result, "degraded"
        except Exception as e:
            print(f"   ⚠️ 简化重试也失败: {type(e).__name__}")

        # 第三层：使用预设兜底模板
        print(f"   📋 使用兜底模板")
        template = FALLBACK_TEMPLATES.get(service_name, FALLBACK_TEMPLATES["默认"])
        if context and "{content}" in template:
            fallback_result = template.format(content=context[:200])
        else:
            fallback_result = template
        return fallback_result, "fallback"

    print("\n【交互式智能问答（带优雅降级）】")
    print("提示：输入问题，AI 回答（三级降级保障）")
    print("  第一级：正常调用")
    print("  第二级：简化 prompt 重试")
    print("  第三级：兜底模板")
    print("输入 '退出' 结束\n")

    # 创建主链
    expert_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个资深技术专家，请给出详细、准确、专业的回答。"),
        ("human", "{user_input}")
    ])
    expert_chain = expert_prompt | model | StrOutputParser()

    while True:
        user_input = input("你的问题：").strip()

        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("结束对话")
            break

        if not user_input:
            print("请输入有效内容")
            continue

        print(f"\n【调用智能问答（三级降级保障）】")
        result, status = graceful_invoke(
            expert_chain,
            {"user_input": user_input},
            service_name="问答",
            context=user_input
        )

        # 根据状态显示不同标识
        status_labels = {
            "success": "🟢 正常",
            "degraded": "🟡 降级",
            "fallback": "🔴 兜底"
        }
        print(f"\n状态: {status_labels.get(status, '未知')}")
        print(f"AI：{result}")

        print("\n" + "-"*60)

    print("\n✅ 实战要点总结：")
    print("   1. 三级降级：正常 -> 简化重试 -> 兜底模板")
    print("   2. 每级降级都要记录日志，便于事后排查")
    print("   3. 兜底模板要友好，不能暴露技术细节")
    print("   4. 降级策略根据业务优先级灵活调整")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain 错误处理与调试 - 实战案例")
    print("="*60)
    print("\n本示例演示 LangChain 中常见的错误处理策略和调试技巧")
    print("\n核心概念：")
    print("  • 错误重试（Retry）：失败时自动重试")
    print("  • 模型降级（Fallback）：主模型失败时切换备用")
    print("  • 输出验证（Validation）：校验模型输出格式")
    print("  • 优雅降级（Graceful Degradation）：友好兜底")
    print("\n应用场景：")
    print("  • 网络波动、API 限流、模型停机")
    print("  • 输出格式不稳定、业务规则校验")
    print("  • 生产环境高可用保障")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 错误重试（API 调用失败时自动重试）")
        print("  2. 模型降级（主模型失败时切换备用模型）")
        print("  3. 输出验证（验证模型输出是否符合预期）")
        print("  4. 优雅降级（出错时返回友好的降级结果）")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_retry_on_error()
        elif choice == "2":
            demo_fallback_model()
        elif choice == "3":
            demo_output_validation()
        elif choice == "4":
            demo_graceful_degradation()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
