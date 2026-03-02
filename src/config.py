from pydantic import BaseModel
from functools import lru_cache
from typing import Optional
import os

# Load .env so GEMINI_API_KEY etc. are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class Settings(BaseModel):
    app_name: str = "Privacy-Preserving RAG Demo"
    environment: str = os.getenv("ENVIRONMENT", "local")
    api_keys: dict = {"doctor-key": "doctor", "nurse-key": "nurse", "auditor-key": "auditor"}
    # In a real system, store keys and secrets in a secure vault, not in code.

    encryption_key_env: str = os.getenv("RAG_ENC_KEY", "dev-only-key-please-change")
    vector_dim: int = 384
    top_k: int = 5
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    # Ollama: local LLM, no key or rate limits. e.g. http://localhost:11434
    ollama_base_url: Optional[str] = os.getenv("OLLAMA_BASE_URL")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")


@lru_cache
def get_settings() -> Settings:
    return Settings()

