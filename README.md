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
- Retrieve relevant healthcare snippets and construct an answer via a real LLM (Ollama or Gemini).
- Emit audit logs with minimal, privacy-preserving details.

## LLM: Ollama (local) or Gemini

- **Ollama (recommended to avoid rate limits)**  
  Install [Ollama](https://ollama.com), run a model (e.g. `ollama run llama3.2`), then in `.env` set:
  - `OLLAMA_BASE_URL=http://localhost:11434`
  - `OLLAMA_MODEL=llama3.2` (or `gemma2:2b`, `mistral`, etc.)

- **Gemini**  
  Set `GEMINI_API_KEY=your_key` in `.env` (get a key at [Google AI Studio](https://aistudio.google.com/apikey)). Free tier has rate limits (429); the app retries and then returns a fallback answer.

If both are set, Ollama is used first. If neither is set, responses use a placeholder.

## Optional: Ingest Kaggle Dataset

To replace the tiny static corpus with a larger healthcare corpus from Kaggle at startup, enable the Kaggle ingestion options in `.env`:

- `USE_KAGGLE_HEALTHCARE_DATASET=true`
- `KAGGLE_HEALTHCARE_DATASET_ID=prasad22/healthcare-dataset`
- `KAGGLE_MAX_ROWS=200` (limit docs to keep startup time reasonable)
- `KAGGLE_TEXT_COLUMNS=` (optional comma-separated column names; leave empty to use the first 6 columns)

The loader converts each tabular row into a short text document and anonymizes any detected PII patterns before embedding.

## Notes

- Security primitives such as encryption, DP, and RBAC are simplified for demonstration and are **not** production-grade.
- Security primitives such as encryption, DP, and RBAC are simplified for demonstration and are **not** production-grade.

