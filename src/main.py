from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

from .config import get_settings
from .security import rbac
from .security.audit import audit_event
from .rag.pipeline import PrivacyPreservingRAGPipeline


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    metadata: dict


settings = get_settings()
app = FastAPI(title=settings.app_name)
pipeline = PrivacyPreservingRAGPipeline()


@app.on_event("startup")
async def startup_event() -> None:
    """
    Load a tiny demo healthcare corpus that the RAG pipeline can retrieve from.
    """
    # Either load a small static demo corpus, or ingest:
    # 1) Kaggle dataset (optional),
    # 2) local CSV dataset (optional),
    # 3) fallback to the tiny static demo corpus.
    corpus = [
        "In patients with hypertension, lifestyle changes such as reduced salt intake and exercise are first-line.",
        "For diabetic patients, regular HbA1c monitoring is recommended every 3 months.",
        "Adverse drug reactions must be documented in the patient's medical record.",
    ]

    if settings.use_kaggle_healthcare_dataset:
        text_columns = (
            [c.strip() for c in settings.kaggle_text_columns.split(",") if c.strip()]
            if settings.kaggle_text_columns
            else None
        )
        try:
            from .data.kaggle_loader import load_healthcare_docs_from_kaggle

            docs = load_healthcare_docs_from_kaggle(
                dataset_id=settings.kaggle_healthcare_dataset_id,
                max_rows=settings.kaggle_max_rows,
                text_columns=text_columns,
                anonymize=True,
            )
            if docs:
                pipeline.ingest_corpus(docs)
                return
        except Exception as e:
            # Fall back to static corpus if Kaggle download/parsing fails.
            print(f"[startup] Kaggle dataset ingestion failed; falling back to static corpus. Error: {e}")

    # Try local CSV ingestion if present (useful when you already downloaded the dataset).
    if settings.use_local_healthcare_dataset:
        try:
            import os
            from .data.local_csv_loader import load_healthcare_docs_from_local_csv

            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            csv_path = settings.local_healthcare_dataset_csv_path
            resolved_path = csv_path if os.path.isabs(csv_path) else os.path.join(base_dir, csv_path)

            if os.path.exists(resolved_path):
                text_columns = (
                    [c.strip() for c in settings.local_text_columns.split(",") if c.strip()]
                    if settings.local_text_columns
                    else None
                )
                docs = load_healthcare_docs_from_local_csv(
                    csv_path=resolved_path,
                    max_rows=settings.local_max_rows,
                    text_columns=text_columns,
                    anonymize=True,
                )
                if docs:
                    pipeline.ingest_corpus(docs)
                    return
        except Exception as e:
            print(f"[startup] Local CSV ingestion failed; falling back to static corpus. Error: {e}")

    pipeline.ingest_corpus(corpus)


def _get_role_or_401(api_key: Optional[str]) -> str:
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    role = rbac.resolve_role_from_api_key(api_key, settings.api_keys)
    if role is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return role


@app.post("/query", response_model=QueryResponse)
async def query_rag(
    body: QueryRequest,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> QueryResponse:
    """
    Healthcare-style RAG endpoint with:
    - API-key based RBAC
    - Real-time anonymization of PII
    - Prompt sanitization
    - Retrieval + generation
    - Auditing
    """
    role = _get_role_or_401(x_api_key)

    if not rbac.can_access_patient_details(role):
        raise HTTPException(status_code=403, detail="Insufficient role to query patient information")

    result = pipeline.run(user_role=role, query=body.query)
    audit_event(role, "query_served", {"query_length": len(body.query)})

    return QueryResponse(answer=result["answer"], metadata=result["metadata"])

