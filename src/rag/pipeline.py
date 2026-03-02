from typing import List, Dict, Any, Optional

from ..config import get_settings
from ..security import anonymization, prompt_sanitizer, audit
from ..security.tokenization import InMemoryTokenizer
from .vectorstore import InMemoryVectorStore
from .llm import generate_answer


POISON_MARKERS = ["POISONED DOC:", "MALICIOUS-OUTCOME"]
SENSITIVE_DENYLIST = ["john doe", "abc-1234", "mrn"]


class PrivacyPreservingRAGPipeline:
    """
    High-level orchestration of the privacy-preserving RAG flow inspired by the paper:
    - Real-time anonymization (simulated NER)
    - Tokenization of identifiers
    - Prompt sanitization
    - Retrieval + generation
    - Auditing hooks
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.vector_store = InMemoryVectorStore()
        self.tokenizer = InMemoryTokenizer()

    def ingest_corpus(self, docs: List[str]) -> None:
        self.vector_store.add_texts(docs)

    def run(self, user_role: Optional[str], query: str) -> Dict[str, Any]:
        # Real-time anonymization (Section 6.1 of paper)
        anon_query, pii_mapping = anonymization.anonymize_text(query)

        # Tokenization example: replace any MRN-like tokens (if present) with vault tokens
        tokenized_pii: Dict[str, List[str]] = {}
        for label, values in pii_mapping.items():
            tokenized_pii[label] = [self.tokenizer.tokenize(v) for v in values]

        # Prompt sanitization to defend against basic prompt injection
        safe_prompt, flagged = prompt_sanitizer.sanitize_prompt(anon_query)

        # Retrieval from knowledge base
        retrieved = self.vector_store.similarity_search(safe_prompt, top_k=self.settings.top_k)

        # Drop obviously poisoned documents from the enhanced pipeline
        filtered_chunks: List[str] = []
        for text, _score in retrieved:
            if any(marker.lower() in text.lower() for marker in POISON_MARKERS):
                continue
            filtered_chunks.append(text)
        retrieved_chunks = filtered_chunks or [text for text, _ in retrieved]

        # Call LLM (placeholder or real provider)
        answer = generate_answer(safe_prompt, retrieved_chunks)

        # Output-time anonymization: ensure responses do not echo raw PII
        redacted_answer, _ = anonymization.anonymize_text(answer)

        # Additional deny-list scrub for known sensitive markers
        lowered = redacted_answer.lower()
        for marker in SENSITIVE_DENYLIST:
            if marker in lowered:
                redacted_answer = redacted_answer.replace(marker, "<REDACTED>")

        # Audit trail
        audit.audit_event(
            user_role=user_role,
            action="rag_query",
            details={
                "flagged_prompt": flagged,
                "retrieved_docs": len(retrieved_chunks),
            },
        )

        return {
            "answer": redacted_answer,
            "metadata": {
                "prompt_was_flagged": flagged,
                "retrieved_docs": len(retrieved_chunks),
                "pii_tokenized": tokenized_pii,
            },
        }

