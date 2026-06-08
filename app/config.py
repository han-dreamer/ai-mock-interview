from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7

    # Vision (multi-modal) — falls back to llm_* if not set
    vision_api_key: str = ""
    vision_base_url: str = ""
    vision_model: str = "qwen-vl-plus"

    # Embedding
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"

    # RAG and memory persistence
    chroma_persist_dir: str = "./chroma_data"
    memory_db_path: str = "./memory_data/memory.db"
    memory_store_backend: str = "sqlite"
    memory_vector_backend: str = "chroma"
    upload_dir: str = "./uploads"
    export_dir: str = "./exports"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_allowed_origins: str = "*"
    app_access_token: str = ""

    # Redis runtime enhancement layer. Disabled by default so local tests and
    # demos can run without a Redis server.
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    redis_socket_timeout_seconds: float = 3.0
    rate_limit_start_per_minute: int = 3
    rate_limit_answer_per_minute: int = 20
    rate_limit_resume_per_hour: int = 10
    rate_limit_vision_per_day: int = 5
    redis_lock_ttl_ms: int = 60000
    session_cache_ttl_seconds: int = 86400
    report_cache_ttl_seconds: int = 604800
    ws_presence_ttl_seconds: int = 90

    # LangGraph checkpointer. Keep "memory" for local development and tests;
    # use "postgres" in Docker/production so graph state survives restarts.
    checkpointer_backend: str = "memory"
    postgres_url: str = "postgresql://ai_mock:ai_mock@localhost:5432/ai_mock_interview"

    # Business session metadata persistence. The graph checkpoint alone stores
    # LangGraph state, while this store preserves session metadata such as JD,
    # mode, user_id, resume parse result, and graph_started.
    session_store_backend: str = "memory"
    postgres_pool_min_size: int = 1
    postgres_pool_max_size: int = 5

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ] or ["*"]

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key

    @property
    def effective_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.llm_base_url

    @property
    def effective_vision_api_key(self) -> str:
        return self.vision_api_key or self.llm_api_key

    @property
    def effective_vision_base_url(self) -> str:
        return self.vision_base_url or self.llm_base_url

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def resolve_project_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def memory_db_file(self) -> Path:
        return self.resolve_project_path(self.memory_db_path)

    @property
    def upload_root(self) -> Path:
        return self.resolve_project_path(self.upload_dir)

    @property
    def export_root(self) -> Path:
        return self.resolve_project_path(self.export_dir)


settings = Settings()
