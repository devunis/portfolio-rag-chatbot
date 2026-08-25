from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Devunis Portfolio RAG"
    environment: str = "development"
    openai_api_key: str | None = None
    openai_response_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=768, ge=256, le=1536)
    source_index_url: str = "https://devunis.github.io/rag-index.json"
    allowed_origins: str = "https://devunis.github.io,http://localhost:4000,http://127.0.0.1:4000"
    admin_token: str | None = None
    index_on_startup: bool = True
    request_limit_per_minute: int = Field(default=20, ge=1, le=300)
    max_question_length: int = Field(default=500, ge=20, le=2000)
    retrieval_limit: int = Field(default=4, ge=1, le=8)

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip().rstrip("/")
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
