"""Application settings, loaded from environment variables / .env file."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Vector store (produced by notebooks/rag_pipeline.ipynb) ---
    vector_store_dir: str = str(BACKEND_DIR / "data" / "vector_store")
    collection_name: str = "manuals"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 4

    # --- Ollama ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # --- API ---
    allowed_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    log_level: str = "INFO"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
