from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Graph Semantic AI Gateway"
    app_env: str = "development"

    # Tez deneylerinde baseline karşılaştırmaları için çalışma modu
    gateway_mode: Literal["hybrid", "vector_only", "passthrough"] = "hybrid"

    # Katman 2: Özellik Çıkarım Model Ayarları
    embedding_model_name: str = "all-MiniLM-L6-v2"
    spacy_model_name: str = "en_core_web_sm"

    # Katman 3: Neo4j Graf Veritabanı ve Dinamik Eşik Yapılandırması
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    base_similarity_threshold: float = 0.84

    # Katman 4: Redis Önbellek Yapılandırması
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400  # Varsayılan 24 saat

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