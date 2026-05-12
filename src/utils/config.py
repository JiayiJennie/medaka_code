import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
    
    # Anthropic (Claude)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME")
    
    # Google (Gemini)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME")
    
    # DashScope (Qwen)
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    DASHSCOPE_MODEL_NAME = os.getenv("DASHSCOPE_MODEL_NAME")

    # OpenRouter (OpenAI-compatible, any routed model id, e.g. anthropic/claude-3.5-sonnet)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER")
    OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE")

    # Together AI (OpenAI-compatible)
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    TOGETHER_MODEL_NAME = os.getenv("TOGETHER_MODEL_NAME")
    TOGETHER_BASE_URL = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")

    # Featherless AI (OpenAI-compatible)
    FEATHERLESS_API_KEY = os.getenv("FEATHERLESS_API_KEY")
    FEATHERLESS_MODEL_NAME = os.getenv("FEATHERLESS_MODEL_NAME")
    FEATHERLESS_BASE_URL = os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")
    
    @classmethod
    def get_available_providers(cls):
        """Returns a list of available LLM providers based on configurations."""
        providers = []
        
        if cls.AZURE_OPENAI_ENDPOINT and cls.AZURE_OPENAI_API_KEY and cls.AZURE_OPENAI_DEPLOYMENT_NAME:
            providers.append("azure_openai")
            
        if cls.OPENAI_API_KEY and cls.OPENAI_MODEL_NAME:
            providers.append("openai")
            
        if cls.ANTHROPIC_API_KEY and cls.ANTHROPIC_MODEL_NAME:
            providers.append("anthropic")
            
        if cls.GOOGLE_API_KEY and cls.GOOGLE_MODEL_NAME:
            providers.append("google")
            
        if cls.DASHSCOPE_API_KEY and cls.DASHSCOPE_MODEL_NAME:
            providers.append("dashscope")

        if cls.OPENROUTER_API_KEY and cls.OPENROUTER_MODEL_NAME:
            providers.append("openrouter")

        if cls.TOGETHER_API_KEY and cls.TOGETHER_MODEL_NAME:
            providers.append("together")

        if cls.FEATHERLESS_API_KEY and cls.FEATHERLESS_MODEL_NAME:
            providers.append("featherless")

        return providers
    
    @classmethod
    def get_default_provider(cls):
        """Returns the default provider to use based on available configurations."""
        providers = cls.get_available_providers()
        if providers:
            return providers[0]
        return None 