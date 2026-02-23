# Privacy-Preserving RAG Demo

This project is a minimal implementation of the ideas from _“Enhancing Privacy and Security in RAG-based Generative AI Applications”_ (Panda & Mukherjee, 2025), built as a small healthcare-style RAG service.

## Overview

- **RAG core**: `src/rag/vectorstore.py`, `src/rag/pipeline.py`, `src/rag/llm.py`
- **Privacy features (Section 3 & 6.1 of paper)**:
  - Real-time anonymization of PII (simulated NER) – `src/security/anonymization.py`
  - Tokenization vault for sensitive identifiers – `src/security/tokenization.py`
  - Differential-privacy-style noisy analytics – `src/privacy/differential_privacy.py`
- **Security features (Section 4 of paper)**:
  - API-key based RBAC (zero-trust style boundary) – `src/security/rbac.py`
  - Encryption helper to secure sensitive strings at rest – `src/security/encryption.py`
  - Prompt sanitization against basic prompt injection – `src/security/prompt_sanitizer.py`
  - Auditing and logging for governance and compliance – `src/security/audit.py`
- **Case-study alignment**:
  - Healthcare-oriented knowledge snippets and endpoint semantics mirror the healthcare case study described in Section 6.1.

## Running the API

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows use: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn src.main:app --reload
```

Then call the RAG endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -H "x-api-key: doctor-key" \
  -d '{"query": "What is recommended for a patient with hypertension? MRN: ABC-1234"}'
```

The response will:

- Redact and tokenize PII in the query before retrieval and generation.
- Sanitize suspicious prompt content.
- Retrieve relevant healthcare snippets and construct an answer via a placeholder LLM.
- Emit audit logs with minimal, privacy-preserving details.

## Notes

- The LLM call is intentionally implemented as a stub; you can plug in OpenAI/Azure/local models in `src/rag/llm.py`.
- Security primitives such as encryption, DP, and RBAC are simplified for demonstration and are **not** production-grade.

