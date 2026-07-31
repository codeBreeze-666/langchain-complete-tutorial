# 模型配置指南

本文档详细说明如何配置不同的 LLM 提供商。

## 📋 目录

- [快速配置](#快速配置)
- [支持的模型提供商](#支持的模型提供商)
- [详细配置说明](#详细配置说明)
- [常见问题](#常见问题)

---

## 快速配置

### 第一步：复制配置文件

```bash
cp .env.example .env
```

### 第二步：选择模型提供商

编辑 `.env` 文件，设置 `MODEL_PROVIDER`：

```bash
# 可选值: zhipu, openai, deepseek, qwen, ollama
MODEL_PROVIDER=zhipu
```

### 第三步：配置 API Key

根据你选择的提供商，配置对应的 API Key：

```bash
# 如果选择智谱
ZHIPU_API_KEY=your-api-key-here

# 如果选择 OpenAI
# OPENAI_API_KEY=your-api-key-here

# 如果选择 DeepSeek
# DEEPSEEK_API_KEY=your-api-key-here
```

---

## 支持的模型提供商

| 提供商 | 标识符 | 费用 | 特点 |
|--------|--------|------|------|
| 智谱 AI | `zhipu` | 有免费模型 | 推荐学习使用 |
| OpenAI | `openai` | 付费 | 性能最好 |
| DeepSeek | `deepseek` | 部分免费 | 性价比高 |
| 通义千问 | `qwen` | 部分免费 | 中文友好 |
| Ollama | `ollama` | 完全免费 | 本地运行 |

---

## 详细配置说明

### 1. 智谱 AI（推荐）

**优点**：
- ✅ 有免费模型可用（如 glm-4.7-flash）
- ✅ 中文支持好
- ✅ 性能稳定
- ✅ 注册即可使用

**获取 API Key**：
1. 访问 https://open.bigmodel.cn/
2. 注册账号
3. 进入控制台 → API 密钥管理
4. 创建新的 API 密钥

**配置示例**：
```bash
MODEL_PROVIDER=zhipu
ZHIPU_API_KEY=your-api-key-here
ZHIPU_API_BASE=https://open.bigmodel.cn/api/paas/v4
ZHIPU_MODEL_NAME=glm-4.7-flash
```

**可用模型**：
- `glm-4.7-flash`：快速响应，推荐日常使用
- `glm-4`：性能更强，适合复杂任务
- `glm-4.7-flash`：最新快速模型

---

### 2. OpenAI

**优点**：
- ✅ 性能最好
- ✅ 功能最全
- ✅ 国际标准

**缺点**：
- ❌ 需要付费
- ❌ 需要国际支付方式

**获取 API Key**：
1. 访问 https://platform.openai.com/api-keys
2. 注册/登录
3. 创建 API Key

**配置示例**：
```bash
MODEL_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-3.5-turbo
```

**可用模型**：
- `gpt-3.5-turbo`：快速，便宜
- `gpt-4`：强大，昂贵
- `gpt-4-turbo`：平衡选择

---

### 3. DeepSeek

**优点**：
- ✅ 性价比高
- ✅ 有免费模型可用

**获取 API Key**：
1. 访问 https://platform.deepseek.com/
2. 注册账号
3. 创建 API Key

**配置示例**：
```bash
MODEL_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-api-key-here
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL_NAME=deepseek-chat
```

**可用模型**：
- `deepseek-chat`：通用对话模型
- `deepseek-coder`：代码专用

---

### 4. 通义千问（阿里云）

**优点**：
- ✅ 中文优化
- ✅ 有免费模型可用
- ✅ 国内访问快

**获取 API Key**：
1. 访问 https://dashscope.aliyun.com/
2. 注册/登录阿里云
3. 开通灵积模型服务
4. 创建 API Key

**配置示例**：
```bash
MODEL_PROVIDER=qwen
QWEN_API_KEY=your-api-key-here
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL_NAME=qwen-turbo
```

**可用模型**：
- `qwen-turbo`：快速响应
- `qwen-plus`：平衡选择
- `qwen-max`：最强性能

---

### 5. Ollama（本地模型）

**优点**：
- ✅ 完全免费
- ✅ 数据私密
- ✅ 无需联网

**缺点**：
- ❌ 需要本地 GPU
- ❌ 需要下载模型

**安装 Ollama**：
1. 访问 https://ollama.com/
2. 下载并安装
3. 运行 `ollama pull qwen2.5:7b`

**配置示例**：
```bash
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=qwen2.5:7b
```

**推荐模型**：
- `qwen2.5:7b`：中文友好
- `llama3.2:3b`：轻量级
- `codellama:7b`：代码专用

---

## 通用参数配置

### Temperature（温度）

控制输出的随机性：

```bash
TEMPERATURE=0.7
```

**建议值**：
- `0.0-0.3`：事实查询、代码生成
- `0.4-0.7`：日常对话（推荐）
- `0.8-1.5`：创意写作

### Max Tokens（最大令牌数）

限制输出长度：

```bash
MAX_TOKENS=1000
```

### Timeout（超时时间）

请求超时时间（秒）：

```bash
REQUEST_TIMEOUT=60
```

### Max Retries（最大重试次数）

失败后重试次数：

```bash
MAX_RETRIES=3
```

---

## 常见问题

### Q: 如何切换模型提供商？

**A:** 修改 `.env` 文件中的 `MODEL_PROVIDER`：

```bash
MODEL_PROVIDER=openai  # 切换到 OpenAI
```

然后配置对应的 API Key。

### Q: 可以同时配置多个提供商吗？

**A:** 可以！你可以配置所有提供商，通过修改 `MODEL_PROVIDER` 切换：

```bash
# 配置所有提供商
MODEL_PROVIDER=zhipu

ZHIPU_API_KEY=your-zhipu-key
OPENAI_API_KEY=your-openai-key
DEEPSEEK_API_KEY=your-deepseek-key

# 使用时切换
# MODEL_PROVIDER=openai
```

### Q: 推荐使用哪个模型？

**A:** 根据场景：

- **学习/测试**：智谱 AI（有免费模型）
- **生产环境**：OpenAI（性能好）
- **成本敏感**：DeepSeek（性价比高）
- **数据敏感**：Ollama（本地）

### Q: 如何检查配置是否正确？

**A:** 运行测试：

```bash
python src/utils/llm_loader.py
```

如果输出：

```
当前模型配置:
提供商: 智谱 AI
模型: glm-4.7-flash
API 地址: https://open.bigmodel.cn/api/paas/v4
```

说明配置正确。

### Q: API Key 泄露了怎么办？

**A:** 立即在提供商控制台：
1. 删除旧的 API Key
2. 创建新的 API Key
3. 更新 `.env` 文件

**重要**：不要将 `.env` 文件上传到 Git！

### Q: 遇到 429 错误（限流）怎么办？

**A:** 这是正常现象：

1. 等待 1-2 分钟
2. 减少调用频率
3. 添加延迟：`time.sleep(2)`
4. 升级账户获得更多配额

---

## 高级配置

### 使用代理

如果你需要使用代理访问 OpenAI：

```bash
OPENAI_API_BASE=https://your-proxy.com/v1
```

### 自定义模型参数

在代码中可以覆盖环境变量：

```python
from src.utils.llm_loader import LLMLoader

llm = LLMLoader.create_llm(
    model_name="glm-4",
    temperature=0.5,
    provider="zhipu"
)
```

---

## 配置文件示例

完整的 `.env` 文件示例：

```bash
# 当前使用的模型提供商
MODEL_PROVIDER=zhipu

# 智谱配置
ZHIPU_API_KEY=your-zhipu-api-key-here
ZHIPU_API_BASE=https://open.bigmodel.cn/api/paas/v4
ZHIPU_MODEL_NAME=glm-4.7-flash

# OpenAI 配置（备用）
# OPENAI_API_KEY=your-openai-api-key-here
# OPENAI_API_BASE=https://api.openai.com/v1
# OPENAI_MODEL_NAME=gpt-3.5-turbo

# 通用参数
TEMPERATURE=0.7
MAX_TOKENS=1000
REQUEST_TIMEOUT=60
MAX_RETRIES=3

# 日志级别
LOG_LEVEL=INFO
```

---

## 获取帮助

如有问题，请：

1. 查看 [FAQ](FAQ.md)
2. 运行测试：`python src/utils/llm_loader.py`
3. 提交 Issue

---

最后更新：2026-07-27