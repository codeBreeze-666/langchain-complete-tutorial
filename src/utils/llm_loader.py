"""
LLM 模型加载器
负责创建和配置 LLM 实例
支持多种模型提供商：智谱、OpenAI、DeepSeek、通义千问、Ollama
"""
import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# 加载环境变量
load_dotenv()


class ModelConfig:
    """模型配置类"""
    
    # 支持的模型提供商
    PROVIDERS = {
        "zhipu": {
            "name": "智谱 AI",
            "env_key": "ZHIPU_API_KEY",
            "env_base": "ZHIPU_API_BASE",
            "env_model": "ZHIPU_MODEL_NAME",
            "default_model": "glm-4.7-flash",
        },
        "openai": {
            "name": "OpenAI",
            "env_key": "OPENAI_API_KEY",
            "env_base": "OPENAI_API_BASE",
            "env_model": "OPENAI_MODEL_NAME",
            "default_model": "gpt-3.5-turbo",
        },
        "deepseek": {
            "name": "DeepSeek",
            "env_key": "DEEPSEEK_API_KEY",
            "env_base": "DEEPSEEK_API_BASE",
            "env_model": "DEEPSEEK_MODEL_NAME",
            "default_model": "deepseek-chat",
        },
        "qwen": {
            "name": "通义千问",
            "env_key": "QWEN_API_KEY",
            "env_base": "QWEN_API_BASE",
            "env_model": "QWEN_MODEL_NAME",
            "default_model": "qwen-turbo",
        },
        "ollama": {
            "name": "Ollama (本地)",
            "env_base": "OLLAMA_BASE_URL",
            "env_model": "OLLAMA_MODEL_NAME",
            "default_model": "qwen2.5:7b",
        },
    }
    
    @classmethod
    def get_current_provider(cls) -> str:
        """获取当前配置的模型提供商"""
        provider = os.getenv("MODEL_PROVIDER", "zhipu").lower()
        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"不支持的模型提供商: {provider}\n"
                f"支持的提供商: {list(cls.PROVIDERS.keys())}"
            )
        return provider
    
    @classmethod
    def get_provider_config(cls, provider: Optional[str] = None) -> dict:
        """
        获取模型提供商的配置
        
        Args:
            provider: 提供商名称，如果为 None 则使用环境变量中的配置
            
        Returns:
            配置字典
        """
        if provider is None:
            provider = cls.get_current_provider()
        
        if provider not in cls.PROVIDERS:
            raise ValueError(f"不支持的模型提供商: {provider}")
        
        config = cls.PROVIDERS[provider]
        
        # 获取环境变量
        result = {
            "provider": provider,
            "name": config["name"],
            "model": os.getenv(config.get("env_model", ""), config["default_model"]),
        }
        
        # API Key (Ollama 不需要)
        if "env_key" in config:
            result["api_key"] = os.getenv(config["env_key"])
            if not result["api_key"]:
                raise ValueError(
                    f"请配置 {config['name']} 的 API Key\n"
                    f"环境变量: {config['env_key']}"
                )
        
        # API Base URL
        if "env_base" in config:
            result["base_url"] = os.getenv(config["env_base"])
        
        return result


class LLMLoader:
    """LLM 模型加载器类"""
    
    _instance = None
    
    @classmethod
    def get_llm(
        cls,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        """
        获取 LLM 实例（单例模式）
        
        Args:
            model_name: 模型名称，默认从环境变量读取
            temperature: 温度参数，默认从环境变量读取
            provider: 提供商，默认从环境变量读取
            **kwargs: 其他参数
            
        Returns:
            LLM 实例
        """
        if cls._instance is None:
            cls._instance = cls._create_llm(
                model_name=model_name,
                temperature=temperature,
                provider=provider,
                **kwargs
            )
        
        return cls._instance
    
    @classmethod
    def create_llm(
        cls,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        provider: Optional[str] = None,
        streaming: bool = False,
        **kwargs
    ):
        """
        创建新的 LLM 实例
        
        Args:
            model_name: 模型名称
            temperature: 温度参数
            provider: 提供商
            streaming: 是否启用流式输出
            **kwargs: 其他参数
            
        Returns:
            新的 LLM 实例
        """
        return cls._create_llm(
            model_name=model_name,
            temperature=temperature,
            provider=provider,
            streaming=streaming,
            **kwargs
        )
    
    @staticmethod
    def _create_llm(
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        provider: Optional[str] = None,
        streaming: bool = False,
        **kwargs
    ):
        """内部创建 LLM 的方法"""
        # 获取配置
        config = ModelConfig.get_provider_config(provider)
        
        # 获取参数
        temp = temperature if temperature is not None else float(
            os.getenv("TEMPERATURE", "0.7")
        )
        model = model_name or config["model"]
        
        # 创建 LLM 实例
        if config["provider"] == "ollama":
            # Ollama 本地模型
            return ChatOllama(
                model=model,
                temperature=temp,
                base_url=config.get("base_url", "http://localhost:11434"),
                **kwargs
            )
        else:
            # OpenAI 兼容的 API（智谱、DeepSeek、通义等）
            return ChatOpenAI(
                model=model,
                temperature=temp,
                openai_api_key=config["api_key"],
                openai_api_base=config.get("base_url"),
                streaming=streaming,
                timeout=float(os.getenv("REQUEST_TIMEOUT", "60")),
                max_retries=int(os.getenv("MAX_RETRIES", "3")),
                **kwargs
            )


def get_default_llm(provider: Optional[str] = None):
    """
    获取默认的 LLM 实例
    
    Args:
        provider: 提供商，如果为 None 则使用环境变量配置
    
    Returns:
        LLM 实例
    """
    return LLMLoader.get_llm(provider=provider)


def create_streaming_llm(provider: Optional[str] = None):
    """
    创建流式输出的 LLM 实例
    
    Args:
        provider: 提供商
    
    Returns:
        LLM 实例
    """
    return LLMLoader.create_llm(streaming=True, provider=provider)


def print_model_info():
    """打印当前模型配置信息"""
    try:
        config = ModelConfig.get_provider_config()
        print("\n" + "="*60)
        print("当前模型配置:")
        print("="*60)
        print(f"提供商: {config['name']}")
        print(f"模型: {config['model']}")
        if 'base_url' in config:
            print(f"API 地址: {config['base_url']}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ 配置错误: {e}\n")


if __name__ == "__main__":
    # 测试配置
    print_model_info()
    
    # 测试创建 LLM
    try:
        llm = get_default_llm()
        print(f"✅ 成功创建 LLM 实例")
        print(f"类型: {type(llm)}")
    except Exception as e:
        print(f"❌ 创建失败: {e}")