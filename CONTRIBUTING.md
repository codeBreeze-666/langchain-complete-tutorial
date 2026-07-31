# 贡献指南

感谢你对本项目的关注！欢迎所有形式的贡献。

## 如何贡献

### 报告问题

如果你发现了 Bug 或有功能建议：

1. 在 [GitHub Issues](https://github.com/codeBreeze-666/langchain-complete-tutorial/issues) 中搜索是否已有相关问题
2. 如果没有，创建新的 Issue，包含：
   - 问题的详细描述
   - 复现步骤
   - 期望行为和实际行为
   - 运行环境（Python版本、操作系统等）

### 提交代码

1. Fork 本项目
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'Add some feature'`
4. 推送到分支：`git push origin feature/your-feature`
5. 创建 Pull Request

### 代码规范

- Python 代码遵循 PEP 8 规范
- 所有注释和文档使用中文
- 每个示例必须是交互式的（用户可以输入）
- 新增案例需要包含完整的文档字符串（中英文对照）
- 使用 `from src.utils.llm_loader import get_default_llm` 加载模型

### 新增案例要求

1. 文件顶部包含完整的文档字符串（中英文对照）
2. 所有功能必须是交互式的
3. 包含错误处理
4. 使用统一的模型加载方式
5. 在 `EXAMPLES_INDEX.md` 和对应章节文档中添加说明

## 许可证

提交代码即表示你同意该代码将在 MIT 许可证下发布。
