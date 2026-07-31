"""
LangChain 文档加载与切分 - 实战交互式案例
==========================================

本示例演示文档加载与切分的核心概念与实战用法，不依赖外部文件，
使用内置的示例文本进行演示。

核心概念：
- 文本切分：将长文本按不同策略拆分为可管理的片段
- 文档加载：从不同来源（文本、模拟文件、JSON）加载文档
- 切片管理：管理文档切片的元数据，实现溯源与分类
- 自定义切分器：按特定规则切分文本，满足定制化需求

应用场景：
- 文本切分：RAG 预处理、长文本摘要前的分块
- 文档加载：知识库构建、多源数据整合
- 切片管理：文档溯源、质量评估、分类过滤
- 自定义切分器：代码切分、Markdown 切分、结构化文本切分
"""

import os
import sys
import json

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_loader import get_default_llm


# ============================================================
# 内置示例文本（用于各示例演示）
# ============================================================

SAMPLE_LONG_TEXT = (
    "人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个重要分支，"
    "旨在研究和开发能够模拟、延伸和扩展人类智能的理论、方法与技术。"
    "人工智能的发展历程可以追溯到 1956 年的达特茅斯会议，"
    "当时 John McCarthy、Marvin Minsky 等学者首次提出了「人工智能」这一概念。\n\n"
    "机器学习是人工智能的核心技术之一，它使计算机能够从数据中自动学习规律和模式。"
    "常见的机器学习算法包括：线性回归、决策树、支持向量机、随机森林和神经网络。"
    "其中，深度学习作为机器学习的一个子领域，通过多层神经网络实现了对复杂数据的深层特征提取。\n\n"
    "自然语言处理（NLP）是人工智能的重要应用方向，"
    "致力于让计算机理解和生成人类语言。"
    "近年来，以 GPT、BERT 为代表的大语言模型（LLM）极大地推动了 NLP 领域的发展。"
    "这些模型通过在海量文本数据上进行预训练，掌握了丰富的语言知识和世界知识。\n\n"
    "LangChain 是一个用于构建大语言模型应用的开源框架。"
    "它提供了文档加载、文本切分、向量存储、检索增强生成（RAG）等核心组件。"
    "其中，文档加载与切分是 RAG 系统的基础环节，"
    "切分质量直接影响检索效果和最终生成答案的质量。\n\n"
    "文档切分的关键参数包括：chunk_size（切片大小）和 chunk_overlap（重叠大小）。"
    "chunk_size 决定每个切片的最大长度，chunk_overlap 决定相邻切片之间的重叠字符数。"
    "合理的参数设置需要在「保留上下文完整性」和「避免冗余信息」之间取得平衡。"
)

SAMPLE_ARTICLE = {
    "title": "深度学习入门指南",
    "author": "AI 研究院",
    "date": "2025-01-15",
    "content": (
        "深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的层次化表示。"
        "与传统的机器学习方法不同，深度学习能够自动从原始数据中提取特征，"
        "无需人工进行特征工程。\n\n"
        "深度学习的核心组件包括：输入层、隐藏层和输出层。"
        "每一层由若干神经元组成，神经元之间通过权重连接。"
        "训练过程通过反向传播算法调整权重，使模型的预测输出逐步逼近真实值。\n\n"
        "常见的深度学习架构有：卷积神经网络（CNN）擅长图像处理，"
        "循环神经网络（RNN）擅长序列数据处理，"
        "Transformer 架构则是当前大语言模型的基础。"
        "GPT、BERT、LLaMA 等模型都基于 Transformer 架构构建。"
    )
}

SAMPLE_CODE = '''def quicksort(arr):
    """快速排序算法"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

class DataProcessor:
    """数据处理器"""
    def __init__(self, source):
        self.source = source
        self.data = []

    def load(self):
        """加载数据"""
        with open(self.source, 'r') as f:
            self.data = f.readlines()
        return self

    def process(self):
        """处理数据"""
        self.data = [line.strip() for line in self.data if line.strip()]
        return self

    def save(self, output_path):
        """保存数据"""
        with open(output_path, 'w') as f:
            f.write('\\n'.join(self.data))
        return self

# 使用示例
if __name__ == "__main__":
    processor = DataProcessor("input.txt")
    processor.load().process().save("output.txt")
'''


# ============================================================
# 1. 文本切分 - 将长文本按不同策略切分
# ============================================================

def demo_text_splitter():
    """示例1：文本切分（将长文本按不同策略切分）"""
    print("\n" + "="*60)
    print("示例1：文本切分（将长文本按不同策略切分）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - RecursiveCharacterTextSplitter：递归切分，按分隔符优先级逐级拆分")
    print("   - CharacterTextSplitter：按单一分隔符切分，简单直接")
    print("   - chunk_size 控制切片大小，chunk_overlap 控制重叠区域")
    print("   - 重叠区域保证上下文连续性，避免语义断裂")

    text = SAMPLE_LONG_TEXT
    print(f"\n📄 原始文本长度：{len(text)} 字符")
    print(f"   前 80 字预览：{text[:80]}...")

    while True:
        print("\n" + "-"*50)
        print("请选择切分策略：")
        print("  1. 递归字符切分（推荐，按段落→换行→句子逐级拆分）")
        print("  2. 单字符切分（按指定分隔符切分）")
        print("  3. 对比两种切分策略")
        print("  4. 自定义参数切分")
        print("\n  0. 返回上级")
        print("-"*50)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "0":
            break

        if choice == "1":
            # 递归字符切分
            chunk_size_input = input("请输入 chunk_size（默认 200，直接回车跳过）：").strip()
            chunk_size = int(chunk_size_input) if chunk_size_input.isdigit() and int(chunk_size_input) > 0 else 200
            overlap_input = input("请输入 chunk_overlap（默认 50，直接回车跳过）：").strip()
            chunk_overlap = int(overlap_input) if overlap_input.isdigit() and int(overlap_input) >= 0 else 50

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "，", " ", ""],
                length_function=len,
            )
            chunks = splitter.split_text(text)

            print(f"\n✅ 递归字符切分结果：")
            print(f"   参数：chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            print(f"   切片数量：{len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"\n   📦 切片 {i+1}（{len(chunk)} 字符）：")
                print(f"   {chunk[:100]}{'...' if len(chunk) > 100 else ''}")

        elif choice == "2":
            # 单字符切分
            separator = input("请输入分隔符（默认换行符 \\n，直接回车跳过）：").strip()
            if not separator:
                separator = "\n"
            elif separator == "\\n":
                separator = "\n"

            chunk_size_input = input("请输入 chunk_size（默认 300，直接回车跳过）：").strip()
            chunk_size = int(chunk_size_input) if chunk_size_input.isdigit() and int(chunk_size_input) > 0 else 300

            splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=0,
                separator=separator,
                length_function=len,
            )
            chunks = splitter.split_text(text)

            print(f"\n✅ 单字符切分结果：")
            print(f"   参数：separator={repr(separator)}, chunk_size={chunk_size}")
            print(f"   切片数量：{len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"\n   📦 切片 {i+1}（{len(chunk)} 字符）：")
                print(f"   {chunk[:100]}{'...' if len(chunk) > 100 else ''}")

        elif choice == "3":
            # 对比两种策略
            chunk_size = 200
            chunk_overlap = 50

            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "，", " ", ""],
                length_function=len,
            )
            char_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=0,
                separator="\n",
                length_function=len,
            )

            recursive_chunks = recursive_splitter.split_text(text)
            char_chunks = char_splitter.split_text(text)

            print(f"\n📊 对比结果（chunk_size={chunk_size}, chunk_overlap={chunk_overlap}）：")
            print(f"\n   递归字符切分：")
            print(f"   - 切片数量：{len(recursive_chunks)}")
            print(f"   - 平均长度：{sum(len(c) for c in recursive_chunks) / len(recursive_chunks):.0f} 字符")
            print(f"   - 长度范围：{min(len(c) for c in recursive_chunks)} ~ {max(len(c) for c in recursive_chunks)} 字符")

            print(f"\n   单字符切分（按换行符）：")
            print(f"   - 切片数量：{len(char_chunks)}")
            print(f"   - 平均长度：{sum(len(c) for c in char_chunks) / len(char_chunks):.0f} 字符")
            print(f"   - 长度范围：{min(len(c) for c in char_chunks)} ~ {max(len(c) for c in char_chunks)} 字符")

            print(f"\n   💡 分析：")
            print(f"   - 递归切分更均匀，因为会按优先级逐级尝试不同分隔符")
            print(f"   - 单字符切分可能产生过短或过长的切片")

        elif choice == "4":
            # 自定义参数
            chunk_size_input = input("请输入 chunk_size（默认 150）：").strip()
            chunk_size = int(chunk_size_input) if chunk_size_input.isdigit() and int(chunk_size_input) > 0 else 150
            overlap_input = input("请输入 chunk_overlap（默认 30）：").strip()
            chunk_overlap = int(overlap_input) if overlap_input.isdigit() and int(overlap_input) >= 0 else 30

            if chunk_overlap >= chunk_size:
                print("⚠️ chunk_overlap 必须小于 chunk_size，已自动调整")
                chunk_overlap = chunk_size // 4

            # 自定义分隔符优先级
            print("\n可选分隔符优先级：")
            print("  1. 段落→换行→句号→逗号→空格（适合中文文章）")
            print("  2. 换行→空格→句号（适合英文文本）")
            print("  3. 句号→换行→逗号（以句子为单位优先）")
            sep_choice = input("请选择（默认 1）：").strip()

            if sep_choice == "2":
                separators = ["\n", " ", ".", ","]
            elif sep_choice == "3":
                separators = ["。", "\n", "，", " ", ""]
            else:
                separators = ["\n\n", "\n", "。", "，", " ", ""]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=len,
            )
            chunks = splitter.split_text(text)

            print(f"\n✅ 自定义参数切分结果：")
            print(f"   参数：chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            print(f"   分隔符优先级：{separators}")
            print(f"   切片数量：{len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"\n   📦 切片 {i+1}（{len(chunk)} 字符）：")
                print(f"   {chunk[:100]}{'...' if len(chunk) > 100 else ''}")

        else:
            print("❌ 无效选项，请重新选择")

    print("\n✅ 实战要点总结：")
    print("   1. RecursiveCharacterTextSplitter 是最常用的切分器，按优先级逐级拆分")
    print("   2. chunk_overlap 保证上下文连续性，一般设为 chunk_size 的 10%~25%")
    print("   3. 选择合适的分隔符优先级对中文文本切分效果至关重要")


# ============================================================
# 2. 文档加载 - 从不同来源加载文档
# ============================================================

def demo_document_loader():
    """示例2：文档加载（从不同来源加载文档）"""
    print("\n" + "="*60)
    print("示例2：文档加载（从不同来源加载文档）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - LangChain 提供 Document 对象统一表示文档（page_content + metadata）")
    print("   - 文档来源可以是文本、文件、API、数据库等")
    print("   - metadata 存储来源信息，便于后续溯源和过滤")
    print("   - 加载后通常需要切分，再送入向量数据库")

    from langchain_core.documents import Document

    while True:
        print("\n" + "-"*50)
        print("请选择文档来源：")
        print("  1. 从纯文本创建文档")
        print("  2. 从模拟文件加载文档")
        print("  3. 从 JSON 数据加载文档")
        print("  4. 用户自由输入文本并加载")
        print("  5. 加载后切分并预览")
        print("\n  0. 返回上级")
        print("-"*50)

        choice = input("\n请输入选项 (0-5): ").strip()

        if choice == "0":
            break

        if choice == "1":
            # 从纯文本创建文档
            doc = Document(
                page_content=SAMPLE_LONG_TEXT,
                metadata={
                    "source": "内置示例文本",
                    "type": "纯文本",
                    "char_count": len(SAMPLE_LONG_TEXT),
                    "topic": "人工智能与文档切分",
                }
            )
            print(f"\n✅ 从纯文本创建文档：")
            print(f"   page_content 长度：{len(doc.page_content)} 字符")
            print(f"   metadata：{json.dumps(doc.metadata, ensure_ascii=False, indent=2)}")
            print(f"   前 80 字预览：{doc.page_content[:80]}...")

        elif choice == "2":
            # 从模拟文件加载文档
            # 模拟不同类型文件的内容
            file_docs = [
                Document(
                    page_content=SAMPLE_ARTICLE["content"],
                    metadata={
                        "source": "深度学习入门指南.md",
                        "type": "Markdown 文件",
                        "author": SAMPLE_ARTICLE["author"],
                        "date": SAMPLE_ARTICLE["date"],
                        "title": SAMPLE_ARTICLE["title"],
                    }
                ),
                Document(
                    page_content=SAMPLE_CODE,
                    metadata={
                        "source": "data_processor.py",
                        "type": "Python 代码文件",
                        "language": "python",
                        "line_count": len(SAMPLE_CODE.split("\n")),
                    }
                ),
            ]

            print(f"\n✅ 从模拟文件加载了 {len(file_docs)} 个文档：")
            for i, doc in enumerate(file_docs):
                print(f"\n   📄 文档 {i+1}：")
                print(f"   来源：{doc.metadata['source']}")
                print(f"   类型：{doc.metadata['type']}")
                print(f"   内容长度：{len(doc.page_content)} 字符")
                preview = doc.page_content[:80].replace("\n", " ")
                print(f"   预览：{preview}...")

        elif choice == "3":
            # 从 JSON 数据加载文档
            json_data = [
                {
                    "title": "Python 基础教程",
                    "author": "张三",
                    "content": "Python 是一种解释型、高级、通用的编程语言。"
                              "Python 的设计哲学强调代码的可读性和简洁性。"
                              "Python 支持多种编程范式，包括面向对象和函数式编程。"
                },
                {
                    "title": "LangChain 实战指南",
                    "author": "李四",
                    "content": "LangChain 是一个用于开发大语言模型应用的开源框架。"
                              "它提供了文档加载、文本切分、向量存储等核心组件。"
                              "通过 Chain 机制可以将多个组件串联起来完成复杂任务。"
                },
                {
                    "title": "RAG 系统设计",
                    "author": "王五",
                    "content": "RAG 系统由检索模块和生成模块组成。"
                              "检索模块负责从知识库中找到与问题相关的文档片段。"
                              "生成模块基于检索到的上下文生成最终答案。"
                },
            ]

            docs = []
            for item in json_data:
                doc = Document(
                    page_content=item["content"],
                    metadata={
                        "source": "json_data",
                        "type": "JSON 记录",
                        "title": item["title"],
                        "author": item["author"],
                    }
                )
                docs.append(doc)

            print(f"\n✅ 从 JSON 数据加载了 {len(docs)} 个文档：")
            for i, doc in enumerate(docs):
                print(f"\n   📄 文档 {i+1}：")
                print(f"   标题：{doc.metadata['title']}")
                print(f"   作者：{doc.metadata['author']}")
                print(f"   内容：{doc.page_content}")

        elif choice == "4":
            # 用户自由输入文本
            print("\n请输入文本内容（可多行，输入空行结束）：")
            lines = []
            while True:
                line = input("  内容：")
                if line.strip() == "":
                    if lines:
                        break
                    print("  请输入至少一行内容")
                    continue
                lines.append(line)

            user_text = "\n".join(lines)
            source_name = input("请输入文档来源名称（默认：用户输入）：").strip()
            if not source_name:
                source_name = "用户输入"

            doc = Document(
                page_content=user_text,
                metadata={
                    "source": source_name,
                    "type": "用户输入",
                    "char_count": len(user_text),
                    "line_count": len(lines),
                }
            )
            print(f"\n✅ 用户输入文档已创建：")
            print(f"   page_content 长度：{len(doc.page_content)} 字符")
            print(f"   metadata：{json.dumps(doc.metadata, ensure_ascii=False, indent=2)}")

        elif choice == "5":
            # 加载后切分并预览
            doc = Document(
                page_content=SAMPLE_LONG_TEXT,
                metadata={
                    "source": "内置示例文本",
                    "type": "纯文本",
                    "topic": "人工智能与文档切分",
                }
            )

            chunk_size_input = input("请输入 chunk_size（默认 200，直接回车跳过）：").strip()
            chunk_size = int(chunk_size_input) if chunk_size_input.isdigit() and int(chunk_size_input) > 0 else 200

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_size // 5,
                separators=["\n\n", "\n", "。", "，", " ", ""],
                length_function=len,
            )
            chunks = splitter.split_documents([doc])

            print(f"\n✅ 文档加载后切分结果：")
            print(f"   原始文档：{len(doc.page_content)} 字符")
            print(f"   切片数量：{len(chunks)}")
            print(f"   chunk_size={chunk_size}, chunk_overlap={chunk_size // 5}")
            for i, chunk in enumerate(chunks):
                print(f"\n   📦 切片 {i+1}（{len(chunk.page_content)} 字符）：")
                print(f"   内容：{chunk.page_content[:80]}{'...' if len(chunk.page_content) > 80 else ''}")
                print(f"   元数据：{json.dumps(chunk.metadata, ensure_ascii=False)}")

        else:
            print("❌ 无效选项，请重新选择")

    print("\n✅ 实战要点总结：")
    print("   1. Document 对象是 LangChain 文档处理的核心数据结构")
    print("   2. metadata 存储来源信息，切分后会自动继承到每个切片")
    print("   3. split_documents 方法比 split_text 更好，因为会保留元数据")
    print("   4. 实际项目中使用 PyPDFLoader、TextLoader 等加载真实文件")


# ============================================================
# 3. 切片管理 - 管理文档切片的元数据
# ============================================================

def demo_chunk_management():
    """示例3：切片管理（管理文档切片的元数据）"""
    print("\n" + "="*60)
    print("示例3：切片管理（管理文档切片的元数据）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 切片元数据是 RAG 系统溯源和过滤的基础")
    print("   - 自定义元数据字段可支持多种业务场景")
    print("   - 元数据过滤可以大幅提升检索精度")
    print("   - 切片质量评估帮助发现和修复切分问题")

    from langchain_core.documents import Document

    # 创建带丰富元数据的文档集合
    docs = [
        Document(
            page_content="Python 是一种高级编程语言，以简洁优雅的语法著称。"
                         "Python 广泛应用于 Web 开发、数据分析、人工智能等领域。",
            metadata={"source": "Python 简介.md", "category": "编程语言", "importance": "high", "author": "张三"}
        ),
        Document(
            page_content="Java 是一种面向对象的编程语言，具有跨平台、安全性高等特点。"
                         "Java 广泛应用于企业级应用开发和 Android 移动开发。",
            metadata={"source": "Java 简介.md", "category": "编程语言", "importance": "medium", "author": "李四"}
        ),
        Document(
            page_content="机器学习是人工智能的核心技术，包括监督学习、无监督学习和强化学习。"
                         "常见的算法有决策树、支持向量机、神经网络等。",
            metadata={"source": "ML 基础.md", "category": "人工智能", "importance": "high", "author": "王五"}
        ),
        Document(
            page_content="深度学习是机器学习的子领域，使用多层神经网络进行特征学习。"
                         "CNN 擅长图像处理，RNN 擅长序列处理，Transformer 是大模型的基础。",
            metadata={"source": "DL 入门.md", "category": "人工智能", "importance": "high", "author": "王五"}
        ),
        Document(
            page_content="RAG 系统通过检索增强生成，减少大模型的幻觉问题。"
                         "核心流程是：检索相关文档 → 注入上下文 → 生成答案。",
            metadata={"source": "RAG 指南.md", "category": "LLM 应用", "importance": "medium", "author": "赵六"}
        ),
        Document(
            page_content="向量数据库专门用于存储和检索向量嵌入，常见的有 Chroma、Milvus、Pinecone。"
                         "它们支持高效的相似度搜索，是 RAG 的关键基础设施。",
            metadata={"source": "向量数据库.md", "category": "LLM 应用", "importance": "medium", "author": "赵六"}
        ),
    ]

    # 切分文档
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["。", "，", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    # 为每个切片添加额外元数据
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    print(f"\n📚 已加载 {len(docs)} 个文档，切分为 {len(chunks)} 个切片")

    while True:
        print("\n" + "-"*50)
        print("请选择管理操作：")
        print("  1. 查看所有切片及元数据")
        print("  2. 按分类过滤切片")
        print("  3. 按重要性过滤切片")
        print("  4. 按作者过滤切片")
        print("  5. 切片质量评估")
        print("  6. 自定义元数据过滤")
        print("\n  0. 返回上级")
        print("-"*50)

        choice = input("\n请输入选项 (0-6): ").strip()

        if choice == "0":
            break

        if choice == "1":
            # 查看所有切片
            print(f"\n📋 所有切片（共 {len(chunks)} 个）：")
            for chunk in chunks:
                print(f"\n   📦 切片 {chunk.metadata['chunk_id']}")
                print(f"   来源：{chunk.metadata['source']}")
                print(f"   分类：{chunk.metadata['category']}")
                print(f"   重要性：{chunk.metadata['importance']}")
                print(f"   作者：{chunk.metadata['author']}")
                print(f"   长度：{chunk.metadata['chunk_size']} 字符")
                preview = chunk.page_content[:60]
                print(f"   内容：{preview}{'...' if len(chunk.page_content) > 60 else ''}")

        elif choice == "2":
            # 按分类过滤
            categories = list(set(chunk.metadata["category"] for chunk in chunks))
            print(f"\n可用的分类：{categories}")
            category = input("请输入要过滤的分类：").strip()
            if not category:
                category = categories[0]

            filtered = [c for c in chunks if c.metadata.get("category") == category]
            if filtered:
                print(f"\n✅ 分类「{category}」下有 {len(filtered)} 个切片：")
                for chunk in filtered:
                    preview = chunk.page_content[:60]
                    print(f"   📦 切片 {chunk.metadata['chunk_id']}：{preview}{'...' if len(chunk.page_content) > 60 else ''}")
            else:
                print(f"\n⚠️ 未找到分类「{category}」的切片")

        elif choice == "3":
            # 按重要性过滤
            importance_levels = ["high", "medium", "low"]
            level = input("请输入重要性级别（high/medium/low，默认 high）：").strip().lower()
            if level not in importance_levels:
                level = "high"

            filtered = [c for c in chunks if c.metadata.get("importance") == level]
            if filtered:
                print(f"\n✅ 重要性「{level}」的切片有 {len(filtered)} 个：")
                for chunk in filtered:
                    preview = chunk.page_content[:60]
                    print(f"   📦 切片 {chunk.metadata['chunk_id']}（{chunk.metadata['category']}）：{preview}{'...' if len(chunk.page_content) > 60 else ''}")
            else:
                print(f"\n⚠️ 未找到重要性为「{level}」的切片")

        elif choice == "4":
            # 按作者过滤
            authors = list(set(chunk.metadata["author"] for chunk in chunks))
            print(f"\n可用的作者：{authors}")
            author = input("请输入要过滤的作者：").strip()
            if not author:
                author = authors[0]

            filtered = [c for c in chunks if c.metadata.get("author") == author]
            if filtered:
                print(f"\n✅ 作者「{author}」的切片有 {len(filtered)} 个：")
                for chunk in filtered:
                    preview = chunk.page_content[:60]
                    print(f"   📦 切片 {chunk.metadata['chunk_id']}（{chunk.metadata['source']}）：{preview}{'...' if len(chunk.page_content) > 60 else ''}")
            else:
                print(f"\n⚠️ 未找到作者「{author}」的切片")

        elif choice == "5":
            # 切片质量评估
            print(f"\n📊 切片质量评估报告：")
            sizes = [c.metadata["chunk_size"] for c in chunks]
            avg_size = sum(sizes) / len(sizes)
            min_size = min(sizes)
            max_size = max(sizes)

            print(f"\n   📏 长度统计：")
            print(f"   - 平均长度：{avg_size:.0f} 字符")
            print(f"   - 最短切片：{min_size} 字符")
            print(f"   - 最长切片：{max_size} 字符")
            print(f"   - 长度标准差：{(sum((s - avg_size) ** 2 for s in sizes) / len(sizes)) ** 0.5:.0f}")

            # 检测过短切片（可能丢失上下文）
            short_threshold = 30
            short_chunks = [c for c in chunks if c.metadata["chunk_size"] < short_threshold]
            if short_chunks:
                print(f"\n   ⚠️ 过短切片（< {short_threshold} 字符）共 {len(short_chunks)} 个：")
                for chunk in short_chunks:
                    print(f"   - 切片 {chunk.metadata['chunk_id']}：{chunk.page_content}")

            # 按来源统计
            source_counts = {}
            for chunk in chunks:
                src = chunk.metadata["source"]
                source_counts[src] = source_counts.get(src, 0) + 1
            print(f"\n   📁 来源分布：")
            for src, count in source_counts.items():
                print(f"   - {src}：{count} 个切片")

            # 按分类统计
            category_counts = {}
            for chunk in chunks:
                cat = chunk.metadata["category"]
                category_counts[cat] = category_counts.get(cat, 0) + 1
            print(f"\n   🏷️ 分类分布：")
            for cat, count in category_counts.items():
                bar = "█" * count + "░" * (max(category_counts.values()) - count)
                print(f"   - {cat}：[{bar}] {count}")

        elif choice == "6":
            # 自定义元数据过滤
            print(f"\n可用的元数据字段：source, category, importance, author, chunk_id, chunk_size")
            field = input("请输入要过滤的字段名：").strip()
            value = input("请输入要匹配的值：").strip()

            if field and value:
                filtered = [c for c in chunks if str(c.metadata.get(field, "")) == value]
                if filtered:
                    print(f"\n✅ 匹配 {field}={value} 的切片有 {len(filtered)} 个：")
                    for chunk in filtered:
                        preview = chunk.page_content[:60]
                        print(f"   📦 切片 {chunk.metadata['chunk_id']}：{preview}{'...' if len(chunk.page_content) > 60 else ''}")
                else:
                    print(f"\n⚠️ 未找到 {field}={value} 的切片")
            else:
                print("❌ 字段名和值不能为空")

        else:
            print("❌ 无效选项，请重新选择")

    print("\n✅ 实战要点总结：")
    print("   1. 元数据是切片管理的基础，支持溯源、过滤和分类")
    print("   2. 切分后自动继承原文档元数据，也可手动添加自定义字段")
    print("   3. 切片质量评估帮助发现过短切片、长度不均等问题")
    print("   4. 元数据过滤在向量数据库中可大幅提升检索精度")


# ============================================================
# 4. 自定义切分器 - 按特定规则切分文本
# ============================================================

def demo_custom_splitter():
    """示例4：自定义切分器（按特定规则切分文本）"""
    print("\n" + "="*60)
    print("示例4：自定义切分器（按特定规则切分文本）")
    print("="*60)
    print("\n💡 实战要点：")
    print("   - 自定义切分器可以按业务规则切分文本")
    print("   - 常见场景：代码切分、Markdown 切分、结构化文本切分")
    print("   - 继承 TextSplitter 基类并实现 split_text 方法即可")
    print("   - 也可以直接使用函数式切分，灵活处理各种文本格式")

    from langchain_core.documents import Document
    from langchain_text_splitters import TextSplitter

    # ---- 自定义切分器：按标题切分 ----
    class HeadingSplitter(TextSplitter):
        """按标题（# 开头）切分 Markdown 文本"""

        def split_text(self, text: str) -> list[str]:
            chunks = []
            current_chunk = ""

            for line in text.split("\n"):
                if line.startswith("#") and current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = line + "\n"
                else:
                    current_chunk += line + "\n"

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            return chunks

    # ---- 自定义切分器：按代码块切分 ----
    class CodeBlockSplitter(TextSplitter):
        """按函数/类定义切分 Python 代码"""

        def split_text(self, text: str) -> list[str]:
            chunks = []
            current_chunk = ""
            indent_level = 0

            for line in text.split("\n"):
                # 检测顶层定义（函数或类）
                stripped = line.lstrip()
                if (stripped.startswith("def ") or stripped.startswith("class ")) and not line.startswith(" ") and not line.startswith("\t"):
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = line + "\n"
                else:
                    current_chunk += line + "\n"

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            return chunks

    # ---- 自定义切分器：按分隔线切分 ----
    class SeparatorSplitter(TextSplitter):
        """按自定义分隔线（如 ---）切分文本"""

        def __init__(self, separator: str = "---", **kwargs):
            super().__init__(**kwargs)
            self.separator = separator

        def split_text(self, text: str) -> list[str]:
            chunks = text.split(self.separator)
            return [c.strip() for c in chunks if c.strip()]

    # 内置示例 Markdown 文本
    sample_markdown = """# 项目概述

本项目是一个基于 LangChain 的文档处理系统，支持多种文档格式的加载和切分。

## 功能特性

- 文档加载：支持 PDF、Markdown、Word 等格式
- 文本切分：提供多种切分策略和自定义切分器
- 向量检索：集成多种向量数据库支持

## 技术架构

系统采用分层架构设计，包括数据层、服务层和应用层。

### 数据层

负责文档的存储和检索，使用向量数据库存储文档嵌入。

### 服务层

提供文档处理的核心逻辑，包括加载、切分和检索。

### 应用层

面向用户的 API 接口，支持 REST 和 SDK 两种调用方式。

# 快速开始

## 安装

使用 pip 安装：pip install langchain

## 配置

需要在 .env 文件中配置 API Key 和模型参数。
"""

    while True:
        print("\n" + "-"*50)
        print("请选择自定义切分器：")
        print("  1. 按标题切分 Markdown（按 # 标题拆分）")
        print("  2. 按代码块切分 Python（按函数/类拆分）")
        print("  3. 按分隔线切分（按 --- 拆分）")
        print("  4. 按用户自定义规则切分")
        print("  5. 组合切分：先按标题切分，再按字符限制二次切分")
        print("\n  0. 返回上级")
        print("-"*50)

        choice = input("\n请输入选项 (0-5): ").strip()

        if choice == "0":
            break

        if choice == "1":
            # 按标题切分 Markdown
            splitter = HeadingSplitter()
            chunks = splitter.split_text(sample_markdown)

            print(f"\n✅ 按标题切分 Markdown 结果：")
            print(f"   切片数量：{len(chunks)}")
            for i, chunk in enumerate(chunks):
                # 提取标题行
                title_line = chunk.split("\n")[0] if chunk else "（无标题）"
                print(f"\n   📦 切片 {i+1}（{len(chunk)} 字符）：")
                print(f"   标题：{title_line}")
                preview = chunk[:80].replace("\n", " ")
                print(f"   预览：{preview}...")

        elif choice == "2":
            # 按代码块切分 Python
            splitter = CodeBlockSplitter()
            chunks = splitter.split_text(SAMPLE_CODE)

            print(f"\n✅ 按代码块切分 Python 结果：")
            print(f"   切片数量：{len(chunks)}")
            for i, chunk in enumerate(chunks):
                # 提取定义行
                first_line = chunk.split("\n")[0] if chunk else ""
                print(f"\n   📦 切片 {i+1}（{len(chunk)} 字符）：")
                print(f"   定义：{first_line}")
                print(f"   内容：")
                for line in chunk.split("\n")[:5]:
                    print(f"     {line}")
                if chunk.count("\n") > 5:
                    print(f"     ...（共 {chunk.count(chr(10)) + 1} 行）")

        elif choice == "3":
            # 按分隔线切分
            # 构造带分隔线的示例文本
            separator_text = (
                "这是第一段内容，讲述 Python 的基本语法。\n"
                "Python 使用缩进来表示代码块，这与大多数语言不同。\n\n"
                "---\n\n"
                "这是第二段内容，讲述 Python 的数据类型。\n"
                "Python 有整数、浮点数、字符串、列表、字典等基本数据类型。\n\n"
                "---\n\n"
                "这是第三段内容，讲述 Python 的函数定义。\n"
                "使用 def 关键字定义函数，支持默认参数和可变参数。"
            )

            print(f"\n📄 带分隔线的示例文本：")
            print(f"   {separator_text[:100]}...")

            sep = input("请输入分隔符（默认 ---，直接回车跳过）：").strip()
            if not sep:
                sep = "---"

            splitter = SeparatorSplitter(separator=sep)
            chunks = splitter.split_text(separator_text)

            print(f"\n✅ 按分隔线「{sep}」切分结果：")
            print(f"   切片数量：{len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"\n   📦 切片 {i+1}（{len(chunk)} 字符）：")
                print(f"   {chunk[:100]}{'...' if len(chunk) > 100 else ''}")

        elif choice == "4":
            # 用户自定义规则切分
            print("\n请输入要切分的文本（可多行，输入空行结束）：")
            print("或输入 '默认' 使用内置示例文本")
            lines = []
            while True:
                line = input("  内容：")
                if line.strip() == "":
                    if lines:
                        break
                    print("  请输入至少一行内容，或输入 '默认'")
                    continue
                if line.strip() == "默认":
                    lines = SAMPLE_LONG_TEXT.split("\n")
                    break
                lines.append(line)

            user_text = "\n".join(lines)

            print("\n请选择切分规则：")
            print("  1. 按固定行数切分")
            print("  2. 按正则表达式切分")
            print("  3. 按指定字符串切分")
            rule_choice = input("请选择 (1-3)：").strip()

            chunks = []
            if rule_choice == "1":
                line_count_input = input("请输入每个切片的行数（默认 3）：").strip()
                line_count = int(line_count_input) if line_count_input.isdigit() and int(line_count_input) > 0 else 3

                all_lines = user_text.split("\n")
                for i in range(0, len(all_lines), line_count):
                    chunk = "\n".join(all_lines[i:i + line_count])
                    if chunk.strip():
                        chunks.append(chunk)

            elif rule_choice == "2":
                import re
                pattern = input("请输入正则表达式（如：(?=^第.+章) 按章节切分）：").strip()
                if not pattern:
                    pattern = r"(?=^第.+章)"
                try:
                    chunks = re.split(pattern, user_text, flags=re.MULTILINE)
                    chunks = [c.strip() for c in chunks if c.strip()]
                except re.error as e:
                    print(f"❌ 正则表达式错误：{e}")
                    continue

            elif rule_choice == "3":
                sep_str = input("请输入分隔字符串（默认：\\n\\n 双换行）：").strip()
                if not sep_str:
                    sep_str = "\n\n"
                elif sep_str == "\\n\\n":
                    sep_str = "\n\n"
                elif sep_str == "\\n":
                    sep_str = "\n"
                chunks = [c.strip() for c in user_text.split(sep_str) if c.strip()]

            else:
                print("❌ 无效选项")
                continue

            print(f"\n✅ 自定义规则切分结果：")
            print(f"   切片数量：{len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"\n   📦 切片 {i+1}（{len(chunk)} 字符）：")
                preview = chunk[:80].replace("\n", " ")
                print(f"   {preview}{'...' if len(chunk) > 80 else ''}")

        elif choice == "5":
            # 组合切分
            heading_splitter = HeadingSplitter()
            first_pass_chunks = heading_splitter.split_text(sample_markdown)

            # 第二次按字符限制切分
            max_size_input = input("请输入二次切分最大长度（默认 200，直接回车跳过）：").strip()
            max_size = int(max_size_input) if max_size_input.isdigit() and int(max_size_input) > 0 else 200

            final_chunks = []
            for chunk in first_pass_chunks:
                if len(chunk) <= max_size:
                    final_chunks.append(chunk)
                else:
                    # 使用 RecursiveCharacterTextSplitter 进行二次切分
                    sub_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=max_size,
                        chunk_overlap=30,
                        separators=["\n", "。", "，", " ", ""],
                        length_function=len,
                    )
                    final_chunks.extend(sub_splitter.split_text(chunk))

            print(f"\n✅ 组合切分结果：")
            print(f"   第一阶段（按标题切分）：{len(first_pass_chunks)} 个切片")
            print(f"   第二阶段（按 {max_size} 字符二次切分）：{len(final_chunks)} 个切片")
            for i, chunk in enumerate(final_chunks):
                first_line = chunk.split("\n")[0] if chunk else ""
                print(f"\n   📦 切片 {i+1}（{len(chunk)} 字符）：")
                print(f"   首行：{first_line}")
                preview = chunk[:80].replace("\n", " ")
                print(f"   预览：{preview}{'...' if len(chunk) > 80 else ''}")

        else:
            print("❌ 无效选项，请重新选择")

    print("\n✅ 实战要点总结：")
    print("   1. 继承 TextSplitter 并实现 split_text 方法即可自定义切分器")
    print("   2. 组合切分是实战常用策略：先按结构切分，再按长度限制二次切分")
    print("   3. 代码切分要按函数/类边界切分，避免切断逻辑单元")
    print("   4. Markdown 切分优先按标题层级切分，保持语义完整性")


# ============================================================
# 主菜单
# ============================================================

def main():
    """主函数 - 交互式菜单"""
    print("\n" + "="*60)
    print("  LangChain 文档加载与切分 - 实战案例")
    print("="*60)
    print("\n本示例演示文档加载与切分的核心概念与实战用法")
    print("使用内置示例文本，无需外部文件")
    print("\n核心概念：")
    print("  • 文本切分：将长文本按不同策略拆分为可管理的片段")
    print("  • 文档加载：从不同来源加载文档并统一表示")
    print("  • 切片管理：管理文档切片的元数据，实现溯源与过滤")
    print("  • 自定义切分器：按特定规则切分文本，满足定制化需求")
    print("\n应用场景：")
    print("  • RAG 预处理、知识库构建、文档溯源、代码切分")

    while True:
        print("\n" + "="*60)
        print("请选择要运行的示例：")
        print("="*60)
        print("  1. 文本切分（将长文本按不同策略切分）")
        print("  2. 文档加载（从不同来源加载文档）")
        print("  3. 切片管理（管理文档切片的元数据）")
        print("  4. 自定义切分器（按特定规则切分文本）")
        print("\n  0. 退出")
        print("="*60)

        choice = input("\n请输入选项 (0-4): ").strip()

        if choice == "1":
            demo_text_splitter()
        elif choice == "2":
            demo_document_loader()
        elif choice == "3":
            demo_chunk_management()
        elif choice == "4":
            demo_custom_splitter()
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
