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

    # Optional: load and ingest a Kaggle healthcare dataset at startup.
    use_kaggle_healthcare_dataset: bool = os.getenv("USE_KAGGLE_HEALTHCARE_DATASET", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    kaggle_healthcare_dataset_id: str = os.getenv("KAGGLE_HEALTHCARE_DATASET_ID", "prasad22/healthcare-dataset")
    kaggle_max_rows: int = int(os.getenv("KAGGLE_MAX_ROWS", "200"))
    # Comma-separated list of column names to include in each RAG doc (leave empty for first 6).
    kaggle_text_columns: Optional[str] = os.getenv("KAGGLE_TEXT_COLUMNS")

    # Optional: use a locally present healthcare dataset CSV for ingestion.
    use_local_healthcare_dataset: bool = os.getenv("USE_LOCAL_HEALTHCARE_DATASET", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    local_healthcare_dataset_csv_path: str = os.getenv("LOCAL_HEALTHCARE_DATASET_CSV_PATH", "healthcare_dataset.csv")
    local_max_rows: int = int(os.getenv("LOCAL_MAX_ROWS", "200"))
    local_text_columns: Optional[str] = os.getenv("LOCAL_TEXT_COLUMNS")


@lru_cache
def get_settings() -> Settings:
    return Settings()

