# Quick Start Guide

> Get started with the LangChain Hands-on Learning Project in 5 minutes

## Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **pip** package manager
- **Git** for cloning the repository
- An API key from a supported model provider (ZhipuAI recommended — free models available)

## Installation Steps

### 1. Clone the repository

```bash
git clone <repo-url>
cd langchain
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This will install LangChain, LangGraph, LangServe, LangSmith SDK, and all required dependencies.

## Model Configuration

### Recommended: ZhipuAI (Free Models)

ZhipuAI offers free models (e.g., glm-4.7-flash), making it ideal for learning.

1. Register at [https://open.bigmodel.cn/](https://open.bigmodel.cn/)
2. Get your API key from the dashboard
3. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
4. Edit `.env` and fill in your API key:
   ```env
   MODEL_PROVIDER=zhipu
   ZHIPU_API_KEY=your-actual-api-key-here
   ```

### Alternative Providers

<details>
<summary>OpenAI</summary>

```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-3.5-turbo
```

Get your key at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

</details>

<details>
<summary>DeepSeek</summary>

```env
MODEL_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-key-here
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL_NAME=deepseek-chat
```

Get your key at [https://platform.deepseek.com/](https://platform.deepseek.com/)

</details>

<details>
<summary>Qwen (Tongyi Qianwen)</summary>

```env
MODEL_PROVIDER=qwen
QWEN_API_KEY=your-key-here
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL_NAME=qwen-turbo
```

Get your key at [https://dashscope.aliyun.com/](https://dashscope.aliyun.com/)

</details>

<details>
<summary>Ollama (Local)</summary>

1. Install Ollama from [https://ollama.com/](https://ollama.com/)
2. Pull a model: `ollama pull qwen2.5:7b`
3. Configure:
   ```env
   MODEL_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL_NAME=qwen2.5:7b
   ```

</details>

## Run Your First Example

```bash
python src/chains/basic_chain.py
```

This will launch the **Chapter 1.1: LCEL Chain** example, which demonstrates:
- LangChain Expression Language (LCEL) chain calls
- Pipe operator (`|`) for composing chains
- Parallel chain execution
- Streaming output

You'll see an interactive menu where you can choose which sub-example to run and provide your own input.

## Run Other Examples

All examples follow the same pattern — just run the Python file directly:

```bash
# Basics
python src/chains/basic_chain.py
python src/chains/model_parameters_chain.py
python src/chains/message_types_demo.py

# Prompts & Output
python src/chains/prompt_basics.py
python src/chains/output_strategies.py
python src/chains/structured_output_fixed.py
python src/chains/streaming_demo.py

# Tools & Agents
python src/chains/tool_basics.py
python src/chains/agent_basics.py
python src/chains/multi_agent.py

# RAG
python src/chains/rag_basics.py
python src/chains/rag_agent.py

# LangGraph
python src/chains/langgraph_basics.py
python src/chains/langgraph_agent.py
python src/chains/langgraph_multi_agent.py

# LangServe
python src/chains/langserve_basics.py

# LangSmith
python src/chains/langsmith_tracing.py
python src/chains/langsmith_evaluation.py

# Real-world Projects
python src/chains/project_learning_assistant.py
python src/chains/project_data_analyst.py
python src/chains/project_code_assistant.py
```

> For the complete list of all 36 examples, see [EXAMPLES_INDEX_EN.md](EXAMPLES_INDEX_EN.md)

## Frequently Asked Questions

### Q: Which Python version should I use?

**A:** Python 3.8+ is required. Python 3.10 or 3.12 is recommended for best compatibility with all dependencies.

### Q: Do I need to pay for an API key?

**A:** No. ZhipuAI offers free models (e.g., glm-4.7-flash) and is the recommended provider for learning. Ollama is also completely free if you run models locally.

### Q: I get `ModuleNotFoundError` when running an example.

**A:** Make sure you've activated the virtual environment and installed dependencies:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Q: I get an authentication error.

**A:** Check that your `.env` file exists and contains the correct API key:
```bash
cat .env  # Verify the file exists and has your key
```
Make sure `MODEL_PROVIDER` matches the key you configured.

### Q: How do I switch model providers?

**A:** Simply change the `MODEL_PROVIDER` value in `.env` and provide the corresponding API key. The LLM loader (`src/utils/llm_loader.py`) handles the rest automatically.

### Q: Some examples fail with my model.

**A:** Not all models support all features (e.g., tool calling, structured output). If an example fails, try switching to a more capable model like `glm-4.7-flash` (ZhipuAI) or `gpt-3.5-turbo` (OpenAI).

### Q: How do I enable LangSmith tracing?

**A:** Set these in your `.env`:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=langchain-demo
```
Get your LangSmith key at [https://smith.langchain.com/](https://smith.langchain.com/)

---

> For the complete examples index, see [EXAMPLES_INDEX_EN.md](EXAMPLES_INDEX_EN.md)
>
> For the project overview, see [README_EN.md](README_EN.md)
>
> For the Chinese version of this guide, see [README.md](README.md)
