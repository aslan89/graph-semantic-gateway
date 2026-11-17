from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Graph Semantic AI Gateway"
    app_env: str = "development"

    # Tez deneylerinde baseline karşılaştırmaları için çalışma modu
    gateway_mode: Literal["hybrid", "vector_only", "passthrough"] = "hybrid"

    # Dış LLM Proxy yapılandırması
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    default_llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()