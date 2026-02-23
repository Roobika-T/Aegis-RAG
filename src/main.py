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
    corpus = [
        "In patients with hypertension, lifestyle changes such as reduced salt intake and exercise are first-line.",
        "For diabetic patients, regular HbA1c monitoring is recommended every 3 months.",
        "Adverse drug reactions must be documented in the patient's medical record.",
    ]
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

